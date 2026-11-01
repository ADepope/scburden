# SPEC — noise-aware coupled tensor–burden factorization

This spec was fixed before any real-data fit. Changes after the first real-data fit go in the
"Deviations" section at the end, with dates.

## 1. Data

**Expression tensor (OneK1K).** The inputs are the per-cell-type moment files from `sc_dosage_J`
(`results/moments/onek1k/*.npz`, as run 2026). They contain p (donor × gene pseudobulk share),
v (its sampling variance), n_cells, and the covariates pool, sex and age.

- **Cell types:** the 16 types with ≥ 50 donors at ≥ 20 cells.
- **Donor entries:** an entry (d, c) is used only if donor d has ≥ 20 cells of type c. Other
  entries are masked out.
- **Batch removal:** per cell type, the shares are divided by the fitted pool × sex factors (raking,
  `normalize_strata`, copied from `scdosage.moments`). This gives relative shares r with mean 1 in
  every stratum, and v_r = v / factor².
- **Age:** regressed out per (c, g), linear.
- **Values:** X[d,c,g] = r − 1 (relative deviation from the stratum mean), with known variance
  v_r[d,c,g].
- **Genes:** genes with an HC-LoF burden column (MAC ≥ 10), matched by symbol through HGNC, and
  expressed in at least one cell type (mean share ≥ 1e-5, about 1 count per 100k UMIs). Entries
  (c, g) where the gene is not expressed in that cell type (mean share < 1e-6) are masked out.

**Burden matrix (UK Biobank, summary statistics only).** These are plink2 `--glm` results for the
18 quantitative traits:

    BASO BMD BMI CHOL CRET DBP EOSI FVC GLU HbA1c HDL HT LDL LYMPH MCH MCV RBC SBP

- **Inputs:** the HC-LoF burden columns, with phenotypes standardized. They are computed separately
  in two random halves of the training individuals (`scripts/burden_splithalf.sh`, run by the user).
- **Values:** β̂_A[g,t] and SE_A[g,t] from half A, and β̂_B, SE_B from half B. The full-sample
  statistics are not used, because the halves must be independent.
- **R:** the trait × trait error correlation R is the Pearson correlation of z = β̂/SE across genes,
  using only genes with max_t |z| < 4. This drops clear signals and leaves a negligible truncation
  bias, because fewer than 1% of gene–trait pairs are significant. R is estimated per half.

## 2. Model

Indices: donors d, cell types c, genes g, traits t, programs k = 1..K_max (K_max = 20 for real
data).

**Tensor block.** The expression tensor is modelled with a CP decomposition:

    X[d,c,g] = Σ_k U[d,k] A[c,k] W[g,k] + e[d,c,g],
    e ~ N(0, v[d,c,g] + s²[c,g])

- v is the known sampling variance (the "noise-aware" part).
- s²[c,g] is a learned residual between-donor variance, the variation not captured by the K
  programs.

**Burden block.** The burden effects share the gene loadings W:

    β̂[g,:] = W[g,:] Θ + e_g,
    e_g ~ N(0, S_g R S_g + diag(ψ))

- S_g = diag(SE[g,:]) is the burden standard error.
- R is the cross-trait error correlation (fixed, from §1).
- ψ_t is a learned per-trait variance of gene effects outside the programs.
- Missing (g, t) entries get SE = 1e3, which carries no information.

**Shared gene loadings.** W ≥ 0 is shared between the two blocks. A program is a non-negative gene
set. Its sign and cell-type pattern are carried by A, and its trait effects by Θ (signed).

**Priors (MAP).**

| Parameter | Prior |
|---|---|
| U[d,k] | N(0, 1); fixes the scale |
| A[c,k] | N(0, 1/α_k) |
| W[g,k] | Exponential(λ_k) |
| Θ[k,t] | Laplace(0, b); b = 1 (weak; the scale is set by W) |
| α_k, λ_k | Gamma(1e-3, 1e-3) (ARD). Closed-form MAP updates every 100 steps. |

**Number of programs K.** A program is pruned when ‖A_k‖·‖W_k‖ < 1e-3 × the largest such product.
K is the number of programs left after pruning.

**Objective.**

    log p(X | U, A, W) + η · log p(β̂ | W, Θ) + log priors

- η = 1 is the plain joint likelihood.
- The expression block has about D·C/T ≈ 900 times more entries per gene, so η > 1 lets the
  burden statistics shape the programs.
- η = 0 is the expression-only fit. Followed by a regression, it is the two-step baseline (B2).
- (An earlier draft weighted the tensor block instead. That was dropped because down-weighting the
  tensor against fixed priors shrinks the programs.)

**Fitting.** PyTorch, full-batch Adam. W is projected onto W ≥ 0 after each step. Log-variances
are optimized jointly. The number of steps is fixed in the code. There are 3 random restarts, and
the best objective is kept. Real data runs on GPU.

**Predictions for held-out genes.** Two routes are reported:

| Route | How the held-out genes are predicted |
|---|---|
| joint | W_g Θ, straight from the coupled fit |
| refit (primary) | Keep the coupled fit's donor scores U, cell-type patterns A and s². Re-estimate the gene loadings of all genes from expression alone (same likelihood and prior, burden off). Then fit Θ by weighted ridge on the training genes (penalty by inner CV over training genes), exactly as for the baselines. |

Why refit is primary: in the joint fit, the training genes' loadings are also pulled toward their
own noisy β̂, but the held-out genes' loadings are not. The pilot simulations showed that this
asymmetry makes joint predictions overconfident, and badly so for η ≥ 10. With refit, training and
held-out genes get loadings by the same rule. The burden statistics then act only through the
programs they helped find, which is the claim being tested. Joint Θ remains the interpretable
output for program–trait links.

**Choosing η.** η is chosen from the grid {1, 3, 10, 30, 100} (simulation: {1, 3, 10, 30}):
1. Split the genes into 5 folds (3 in simulation).
2. For each fold, fit using half A only, with the burden rows of the held-out genes masked. Their W
   rows are still learned from expression.
3. Predict the held-out genes by each route.
4. For each route separately, pick the η with the best held-out gain against β̂_A, i.e.
   Σ w (2 β̂_A·pred − pred²) / Σ w, the numerator of R². Not R² itself: its denominator
   Σ w (β̂² − SE²) can be ≤ 0 when effects are weak, which flips the ordering.

Half B is never used for choosing anything. A mild optimism remains, from picking one scalar on
the same genes' half-A data. The baselines' ridge penalties are chosen by nested CV (no reuse at
all).

## 3. Baselines

All baselines produce gene features F (genes × p). β̂_A is then regressed on F per trait by weighted
ridge regression (weights 1/SE²), and the prediction for held-out genes is F_g · coefficients. The
penalty is chosen by an inner 3-fold CV over the training genes of each outer fold.

| Baseline | Gene features F |
|---|---|
| B1 scITD-style | Expression only, unweighted. Each (c, g) is standardized across donors. Donor factors = top-K left singular vectors of the donor × (cell type·gene) unfolding, varimax-rotated (scITD with full rank in the cell-type and gene modes). Loadings L_k = U_kᵀ X (C × G). F = the K·C loadings per gene. K is matched to the coupled model's K. On real data, the scITD R package is also run if R is available (the simplified version is reported either way). |
| B2 two-step noise-aware | The coupled model with the burden block switched off (expression only, same likelihood and priors), then F = W |
| B3 random gene sets | K non-negative random columns with the same column sparsity as the fitted W. 20 draws; report the mean. |
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
| S3 null | Θ = 0. Checks calibration: there should be no held-out gain. |

Each scenario has 5 replicates. Settings: D = 150, C = 6, G = 600, T = 8, amp_trait = 0.07,
amp_nuis = 0.35, K_max = 10, 2000 Adam steps. These are small, so the benchmark runs on one CPU
core. Reported metrics:
- **a1 program recovery:** mean cosine similarity between true trait-program gene loadings and the
  best Hungarian-matched estimated gene vectors. For B1 the gene vector is ‖L_k[:, g]‖.
- **a2 gene effect recovery:** correlation between the predicted and the true (noise-free) β for
  held-out genes, pooled over traits.
- **a3 held-out half:** noise-corrected R² on half B (see b). For S3, where R² is undefined
  (no true effects), the weighted mean reduction in squared error against predicting 0 is reported
  instead (gain_B; its expectation is ≤ 0).
- **a4 link recovery:** AUROC of |Θ̂| (after matching) for the true non-zero program–trait links.
- **a5 cell-type pattern:** |correlation| of matched A columns with the true A.

**(b) Real data, held-out burden variance explained.** For genes in held-out folds (with η fixed),
predictions come from the half A fit and are scored against β̂_B. The noise-corrected fraction of
true effect variance explained is

    R²_B = Σ_w (2 β̂_B·pred − pred²) / Σ_w (β̂_B² − SE_B²),   weights w = 1/SE_B²,

per trait and pooled. Uncertainty comes from a bootstrap over genes. The unsigned version scores
the prediction of β̂_B² − SE_B² by Σ_k W_gk²·Θ_kt² against LOEUF (Spearman correlation).

**(c) Replication of program–trait links.** Fit the model separately on half A and half B (each with
its own η), match the programs by the cosine of W. Report:
- the correlation of the matched Θ;
- the fraction of the links with |Θ| > 0 in half A that have the same sign in half B.

**(d) Cell-type interpretation.** For each program: the cell-type profile A_k, the top genes of W_k,
and enrichment against reference gene sets (immune pathways, ClinGen haploinsufficient genes).
This part is descriptive. No formal test is planned beyond a hypergeometric test of the top 100
genes.

**Success criteria (pre-registered).** The coupled model beats B1 and B2 on R²_B (pooled, bootstrap
95% CI of the difference excludes 0), and its program–trait links replicate across halves better
than B2's.

**Failure is a result.** If the coupled model only matches B2, the conclusion is that the burden
statistics are too sparse to shape the programs. The same pipeline would then serve as a
two-step tool.

## 5. Not done here

- No individual-level UK Biobank data is read by this repository. The split-half burden statistics
  come from a script that the user runs.
- No binary traits for now.
- No cell-type-specific gene loadings (Tucker core). This is a possible extension if CP is too
  restrictive.

## Deviations

2026-11-01, after pilot simulations and before any real-data fit:
- The coupling weight now sits on the burden block (see Objective).
- b for Θ changed from 0.05 to 1. At 0.05 it shrank all links to 0.
- Added the refit prediction route as the primary prediction.
- The η grid was changed to match the new definition.
- Added the gain_B metric for the null scenario.
- η is selected by the half-A gain instead of R²_A.
  - The first benchmark run used R²_A. In S1 and S2 the denominator was positive in every
    replicate, so the choices are identical.
  - In S3 it was negative in 2 of 5 replicates. S3 was rerun with the gain rule
    (`results/sim/s3_rerun/`).
  - Found from the first benchmark run.
