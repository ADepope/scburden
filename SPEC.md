# SPEC — noise-aware coupled tensor–burden factorization

This spec was fixed before any real-data fit. Changes after the first real-data fit go in the
"Deviations" section at the end, with dates.

## 1. Data

**Expression tensor (OneK1K).** The inputs are the per-cell-type moment files from `sc_dosage_J`
(`results/moments/onek1k/*.npz`, as run 2026). They contain $`p`$ (donor × gene pseudobulk share),
$`v`$ (its sampling variance), the number of cells, and the covariates pool, sex and age.

- **Cell types:** the 16 types with ≥ 50 donors at ≥ 20 cells.
- **Donor entries:** an entry $`(d, c)`$ is used only if donor $`d`$ has ≥ 20 cells of type $`c`$. Other
  entries are masked out.
- **Batch removal:** per cell type, the shares are divided by the fitted pool × sex factors $`f`$
  (raking, `normalize_strata`, copied from `scdosage.moments`). This gives relative shares
  $`r = p / f`$ with mean 1 in every stratum, and variance $`v_r = v / f^2`$.
- **Age:** regressed out per $`(c, g)`$, linear.
- **Values:** $`X_{dcg} = r_{dcg} - 1`$ (relative deviation from the stratum mean), with known
  variance $`v_{dcg} = v_{r,dcg}`$.
- **Genes:** genes with an HC-LoF burden column (MAC ≥ 10), matched by symbol through HGNC, and
  expressed in at least one cell type (mean share $`\ge 10^{-5}`$, about 1 count per 100k UMIs).
  Entries $`(c, g)`$ where the gene is not expressed in that cell type (mean share $`< 10^{-6}`$) are
  masked out.

**Burden matrix (UK Biobank).** These are gene-level LoF burden association statistics from
plink2 `--glm`, for 18 quantitative traits. They are computed from individual-level genotypes and
phenotypes by `scripts/burden_splithalf.sh`, which the user runs on the cluster. Everything
downstream (data preparation, the model, the evaluation) reads only the resulting summary
statistics: one effect estimate and standard error per gene and trait. The traits are:

    BASO BMD BMI CHOL CRET DBP EOSI FVC GLU HbA1c HDL HT LDL LYMPH MCH MCV RBC SBP

- **Inputs:** the HC-LoF burden columns, with phenotypes standardized. They are computed separately
  in two random halves of the training individuals (`scripts/burden_splithalf.sh`, run by the user).
- **Values:** $`\hat\beta^{A}_{gt}`$ and $`\mathrm{SE}^{A}_{gt}`$ from half A, and
  $`\hat\beta^{B}_{gt}`$, $`\mathrm{SE}^{B}_{gt}`$ from half B. The full-sample statistics are not
  used, because the halves must be independent.
- **Error correlation:** the trait × trait error correlation $`R`$ is the Pearson correlation of
  $`z_{gt} = \hat\beta_{gt} / \mathrm{SE}_{gt}`$ across genes, using only genes with
  $`\max_t \lvert z_{gt} \rvert < 4`$. This drops clear signals and leaves a negligible truncation bias, because
  fewer than 1% of gene–trait pairs are significant. $`R`$ is estimated per half.

## 2. Model

Indices: donors $`d`$, cell types $`c`$, genes $`g`$, traits $`t`$, programs $`k = 1, \dots, K_{\max}`$
($`K_{\max} = 20`$ for real data).

**Tensor block.** The expression tensor is modelled with a CP decomposition:

```math
X_{dcg} = \sum_{k} U_{dk}\, A_{ck}\, W_{gk} + \varepsilon_{dcg},
\qquad \varepsilon_{dcg} \sim \mathcal{N}\!\left(0,\; v_{dcg} + s^2_{cg}\right)
```

- $`v_{dcg}`$ is the known sampling variance (the "noise-aware" part).
- $`s^2_{cg}`$ is a learned residual between-donor variance, the variation not captured by the $`K`$
  programs.

**Burden block.** The burden effects share the gene loadings $`W`$. For gene $`g`$, with
$`\hat{\boldsymbol\beta}_g`$ its effects across all traits and $`\mathbf{w}_g`$ its row of $`W`$:

```math
\hat{\boldsymbol\beta}_{g} = \Theta^{\top} \mathbf{w}_{g} + \mathbf{e}_{g},
\qquad \mathbf{e}_{g} \sim \mathcal{N}\!\left(\mathbf{0},\; S_g R\, S_g + \mathrm{diag}(\boldsymbol\psi)\right)
```

- $`S_g = \mathrm{diag}(\mathrm{SE}_{g1}, \dots, \mathrm{SE}_{gT})`$ holds the burden standard errors.
- $`R`$ is the cross-trait error correlation (fixed, from §1).
- $`\psi_t`$ is a learned per-trait variance of gene effects outside the programs.
- Missing $`(g, t)`$ entries get $`\mathrm{SE} = 10^{3}`$, which carries no information.

**Shared gene loadings.** $`W \ge 0`$ is shared between the two blocks. A program is a non-negative
gene set. Its sign and cell-type pattern are carried by $`A`$, and its trait effects by $`\Theta`$
(signed).

**Priors (MAP).**

| Parameter | Prior |
|---|---|
| $`U_{dk}`$ | $`\mathcal{N}(0, 1)`$; fixes the scale |
| $`A_{ck}`$ | $`\mathcal{N}(0, 1/\alpha_k)`$ |
| $`W_{gk}`$ | $`\mathrm{Exponential}(\lambda_k)`$ |
| $`\Theta_{kt}`$ | $`\mathrm{Laplace}(0, b)`$; $`b = 1`$ (weak; the scale is set by $`W`$) |
| $`\alpha_k, \lambda_k`$ | $`\mathrm{Gamma}(10^{-3}, 10^{-3})`$ (ARD). Closed-form MAP updates every 100 steps. |

**Number of programs $`K`$.** A program is pruned when
$`\lVert A_{\cdot k} \rVert \, \lVert W_{\cdot k} \rVert < 10^{-3}`$ × the largest such product.
$`K`$ is the number of programs left after pruning.

**Objective.**

```math
\log p(X \mid U, A, W) \;+\; \eta \, \log p(\hat\beta \mid W, \Theta) \;+\; \log p(U, A, W, \Theta)
```

- $`\eta = 1`$ is the plain joint likelihood.
- The expression block has about $`D C / T \approx 900`$ times more entries per gene, so $`\eta > 1`$
  lets the burden statistics shape the programs.
- $`\eta = 0`$ is the expression-only fit. Followed by a regression, it is the two-step baseline (B2).
- (An earlier draft weighted the tensor block instead. That was dropped because down-weighting the
  tensor against fixed priors shrinks the programs.)

**Fitting.** PyTorch, full-batch Adam. $`W`$ is projected onto $`W \ge 0`$ after each step.
Log-variances are optimized jointly. The number of steps is fixed in the code. There are 3 random
restarts, and the best objective is kept. Real data runs on GPU.

**Predictions for held-out genes.** Two routes are reported:

| Route | How the held-out genes are predicted |
|---|---|
| joint | $`\Theta^{\top} \mathbf{w}_g`$, straight from the coupled fit |
| refit (primary) | Keep the coupled fit's donor scores $`U`$, cell-type patterns $`A`$ and $`s^2`$. Re-estimate the gene loadings of all genes from expression alone (same likelihood and prior, burden off). Then fit $`\Theta`$ by weighted ridge on the training genes (penalty by inner CV over training genes), exactly as for the baselines. |

Why refit is primary: in the joint fit, the training genes' loadings are also pulled toward their
own noisy $`\hat\beta`$, but the held-out genes' loadings are not. The pilot simulations showed that
this asymmetry makes joint predictions overconfident, and badly so for $`\eta \ge 10`$. With refit,
training and held-out genes get loadings by the same rule. The burden statistics then act only
through the programs they helped find, which is the claim being tested. Joint $`\Theta`$ remains the
interpretable output for program–trait links.

**Choosing $`\eta`$.** $`\eta`$ is chosen from the grid $`\{1, 3, 10, 30, 100\}`$ (simulation:
$`\{1, 3, 10, 30\}`$):
1. Split the genes into 5 folds (3 in simulation).
2. For each fold, fit using half A only, with the burden rows of the held-out genes masked. Their
   $`W`$ rows are still learned from expression.
3. Predict the held-out genes by each route.
4. For each route separately, pick the $`\eta`$ with the best held-out gain against $`\hat\beta^{A}`$
   (the numerator of $`R^2`$ in §4b):

```math
\mathrm{gain} = \frac{\sum_{g,t} w_{gt} \left( 2\, \hat\beta^{A}_{gt}\, \tilde\beta_{gt} - \tilde\beta_{gt}^{2} \right)}{\sum_{g,t} w_{gt}},
\qquad w_{gt} = 1 / \left(\mathrm{SE}^{A}_{gt}\right)^2
```

where $`\tilde\beta_{gt}`$ is the prediction. $`R^2`$ itself is not used, because its denominator
$`\sum w \left(\hat\beta^2 - \mathrm{SE}^2\right)`$ can be $`\le 0`$ when effects are weak, which flips
the ordering.

Half B is never used for choosing anything. A mild optimism remains, from picking one scalar on
the same genes' half-A data. The baselines' ridge penalties are chosen by nested CV (no reuse at
all).

## 3. Baselines

All baselines produce gene features $`F`$ (genes × $`p`$). $`\hat\beta^{A}`$ is then regressed on $`F`$ per
trait by weighted ridge regression (weights $`1/\mathrm{SE}^2`$), and the prediction for a held-out
gene is $`F_g`$ times the coefficients. The penalty is chosen by an inner 3-fold CV over the training
genes of each outer fold.

| Baseline | Gene features $`F`$ |
|---|---|
| B1 scITD-style | Expression only, unweighted. Each $`(c, g)`$ is standardized across donors. Donor factors = top-$`K`$ left singular vectors of the donor × (cell type · gene) unfolding, varimax-rotated (scITD with full rank in the cell-type and gene modes). Loadings $`L_k = U_{\cdot k}^{\top} X`$ ($`C \times G`$). $`F`$ = the $`K C`$ loadings per gene. $`K`$ is matched to the coupled model's $`K`$. On real data, the scITD R package is also run if R is available (the simplified version is reported either way). |
| B2 two-step noise-aware | The coupled model with the burden block switched off (expression only, same likelihood and priors), then $`F = W`$ |
| B3 random gene sets | $`K`$ non-negative random columns with the same column sparsity as the fitted $`W`$. 20 draws; report the mean. |
| B4 LOEUF | gnomAD v4.0 LOEUF (real data only). For signed effects it predicts 0 on average, so it is evaluated on the unsigned metric (§4b). |
| B0 null | Prediction 0 |

## 4. Metrics

**(a) Simulation** (`src/sctb/simulate.py`). Planted trait-relevant programs, plus "nuisance"
expression programs with no trait effect and larger expression variance. The nuisance programs
are where coupling should help. Scenarios:

| Scenario | Setup |
|---|---|
| S1 main | 4 trait programs + 3 stronger nuisance programs |
| S2 aligned | 4 trait programs, no nuisance |
| S3 null | $`\Theta = 0`$. Checks calibration: there should be no held-out gain. |

Each scenario has 5 replicates. Settings: $`D = 150`$, $`C = 6`$, $`G = 600`$, $`T = 8`$, trait-program
amplitude 0.07, nuisance amplitude 0.35, $`K_{\max} = 10`$, 2000 Adam steps. These are small, so the
benchmark runs on one CPU core. Reported metrics:
- **a1 program recovery:** mean cosine similarity between true trait-program gene loadings and the
  best Hungarian-matched estimated gene vectors. For B1 the gene vector is
  $`\lVert L_k[\cdot, g] \rVert`$.
- **a2 gene effect recovery:** correlation between the predicted and the true (noise-free)
  $`\beta`$ for held-out genes, pooled over traits.
- **a3 held-out half:** noise-corrected $`R^2`$ on half B (see b). For S3, where $`R^2`$ is undefined
  (no true effects), the weighted mean reduction in squared error against predicting 0 is reported
  instead (the gain above, computed on half B; its expectation is $`\le 0`$).
- **a4 link recovery:** AUROC of $`\lvert \hat\Theta \rvert`$ (after matching) for the true non-zero
  program–trait links.
- **a5 cell-type pattern:** $`\lvert \mathrm{correlation} \rvert`$ of matched $`A`$ columns with the true $`A`$.

**(b) Real data, held-out burden variance explained.** For genes in held-out folds (with $`\eta`$
fixed), predictions $`\tilde\beta`$ come from the half A fit and are scored against $`\hat\beta^{B}`$.
The noise-corrected fraction of true effect variance explained is

```math
R^2_B = \frac{\sum_{g} w_{gt} \left( 2\, \hat\beta^{B}_{gt}\, \tilde\beta_{gt} - \tilde\beta_{gt}^{2} \right)}
             {\sum_{g} w_{gt} \left( \left(\hat\beta^{B}_{gt}\right)^{2} - \left(\mathrm{SE}^{B}_{gt}\right)^{2} \right)},
\qquad w_{gt} = 1 / \left(\mathrm{SE}^{B}_{gt}\right)^{2}
```

per trait $`t`$, and pooled by summing over traits as well. It is unbiased in numerator and
denominator because $`E[\hat\beta^B \tilde\beta] = \beta \tilde\beta`$ and
$`E[(\hat\beta^B)^2 - (\mathrm{SE}^B)^2] = \beta^2`$. Uncertainty comes from a bootstrap over genes.
The unsigned version scores the prediction of $`(\hat\beta^B)^2 - (\mathrm{SE}^B)^2`$ by
$`\tilde\beta^2`$ against LOEUF (Spearman correlation).

**(c) Replication of program–trait links.** Fit the model separately on half A and half B (each with
its own $`\eta`$), match the programs by the cosine of $`W`$. Report:
- the correlation of the matched $`\Theta`$;
- the fraction of the links with $`\lvert \Theta \rvert > 0`$ in half A that have the same sign in half B.

**(d) Cell-type interpretation.** For each program: the cell-type profile $`A_{\cdot k}`$, the top
genes of $`W_{\cdot k}`$, and enrichment against reference gene sets (immune pathways, ClinGen
haploinsufficient genes). This part is descriptive. No formal test is planned beyond a
hypergeometric test of the top 100 genes.

**Success criteria (pre-registered).** The coupled model beats B1 and B2 on $`R^2_B`$ (pooled,
bootstrap 95% CI of the difference excludes 0), and its program–trait links replicate across halves
better than B2's.

**Failure is a result.** If the coupled model only matches B2, the conclusion is that the burden
statistics are too sparse to shape the programs. The same pipeline would then serve as a
two-step tool.

## 5. Not done here

- Apart from `scripts/burden_splithalf.sh` (run by the user, through plink2), no code in this
  repository reads individual-level UK Biobank data.
- No binary traits for now.
- No cell-type-specific gene loadings (Tucker core). This is a possible extension if CP is too
  restrictive.

## Deviations

2026-11-01, after pilot simulations and before any real-data fit:
- The coupling weight now sits on the burden block (see Objective).
- $`b`$ for $`\Theta`$ changed from 0.05 to 1. At 0.05 it shrank all links to 0.
- Added the refit prediction route as the primary prediction.
- The $`\eta`$ grid was changed to match the new definition.
- Added the gain metric for the null scenario.
- $`\eta`$ is selected by the half-A gain instead of $`R^2_A`$.
  - The first benchmark run used $`R^2_A`$. In S1 and S2 the denominator was positive in every
    replicate, so the choices are identical.
  - In S3 it was negative in 2 of 5 replicates. S3 was rerun with the gain rule
    (`results/sim/s3_rerun/`).
  - Found from the first benchmark run.
