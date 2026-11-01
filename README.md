# scburden

Population single-cell atlases show how gene expression varies from one person to the next across
many cell types at once. Genes rarely vary alone. They move together in programs that can span
several cell types, for example an interferon response shared by monocytes and T cells.
Tensor decompositions such as [scITD](https://github.com/kharchenkolab/scITD) find these
multicellular programs from expression alone.

Biobanks give a second view of the same genes. Rare loss-of-function (LoF) burden tests estimate
what happens to a trait when a person carries a broken copy of a gene. These effects are very
sparse and very noisy: fewer than 1% of gene–trait pairs reach significance in UK Biobank.

This project asks whether the two views help each other. Its model learns multicellular gene
programs from a single-cell atlas and from burden statistics at the same time. The two data sets
share the gene loadings, so a program is one set of genes that both co-vary across donors and
cell types and carry related burden effects on traits. The hope is that the burden statistics steer
the decomposition toward programs that matter for traits, even when those programs are only a
small part of the expression variation.

As far as I could find, no existing method couples a donor × cell type × gene tensor to genetic
association statistics in this way. Related work is reviewed in [NOVELTY.md](NOVELTY.md).

## The model

The expression data form a tensor $X_{dcg}$ over donors $d$, cell types $c$ and genes $g$. Each
entry is the pseudobulk expression share of a gene in one donor's cells of one type, relative to
its batch mean. The model is a CP decomposition with $K$ programs:

```math
X_{dcg} = \sum_{k=1}^{K} U_{dk}\, A_{ck}\, W_{gk} + \varepsilon_{dcg},
\qquad \varepsilon_{dcg} \sim \mathcal{N}\!\left(0,\; v_{dcg} + s^2_{cg}\right)
```

| Symbol | Meaning |
|---|---|
| $U_{dk}$ | Donor scores |
| $A_{ck}$ | How strongly program $k$ is used in cell type $c$ |
| $W_{gk} \ge 0$ | Gene loadings, shared with the burden block |
| $v_{dcg}$ | Known sampling variance of the pseudobulk value |
| $s^2_{cg}$ | Learned residual variance between donors |

The burden effects $\hat\beta_{gt}$ of gene $g$ on trait $t$ are modelled with the same gene
loadings:

```math
\hat{\boldsymbol\beta}_{g} = \boldsymbol\Theta^{\top} \mathbf{w}_{g} + \mathbf{e}_{g},
\qquad \mathbf{e}_{g} \sim \mathcal{N}\!\left(\mathbf{0},\; S_g R\, S_g + \mathrm{diag}(\boldsymbol\psi)\right)
```

| Symbol | Meaning |
|---|---|
| $\hat{\boldsymbol\beta}_{g}$ | The gene's burden effects across all traits |
| $\mathbf{w}_{g}$ | Its row of $W$ |
| $\Theta_{kt}$ | Effect of program $k$ on trait $t$ |
| $S_g$ | Diagonal matrix of the burden standard errors |
| $R$ | Cross-trait error correlation |
| $\psi_t$ | Variance of gene effects outside the programs |

The two blocks are fitted together by maximizing

```math
\log p(X \mid U, A, W) \;+\; \eta \,\log p(\hat\beta \mid W, \Theta) \;+\; \log p(U, A, W, \Theta)
```

where $\eta$ is the coupling weight.

What makes the model "noise-aware":
- **Expression noise is known, not fitted.** Each expression entry comes with its sampling
  variance $v_{dcg}$, computed from the cells behind it. Donors with few cells and lowly expressed
  genes therefore count for less.
- **Burden errors are modelled as they are.** Each burden effect comes with its standard error. The
  error correlation $R$, estimated from null genes, accounts for the fact that all traits are
  measured on largely the same people.
- **The number of programs is learned.** Automatic relevance determination priors prune programs
  that are not needed.
- **The coupling weight $\eta$ is cross-validated.** It sets how much the burden statistics may
  shape the programs, and is chosen on held-out genes.

The model is fitted by MAP estimation in PyTorch, on CPU or GPU.

The test is predictive. The program loadings of a held-out gene come from expression alone. Can
they predict that gene's burden effects, measured in an independent half of the biobank? The
model, baselines and metrics were fixed in advance in [SPEC.md](SPEC.md).

## Status

**Simulation: done.** I planted trait-relevant programs that are weak in expression, next to
stronger "nuisance" programs with no trait effect. Coupling then improves the held-out burden
prediction over the same model fitted to expression only:
- noise-corrected $R^2$: 0.39 vs 0.28 (5 of 5 replicates);
- recovery of the true programs: cosine 0.63 vs 0.42.

A scITD-style decomposition reaches 0.13, and random gene sets reach nothing. With no true effects,
coupling creates no spurious signal. Details are in
[results/sim/SUMMARY.md](results/sim/SUMMARY.md). The simulations are small and use the same CP
structure as the model, so they show that the method works as intended, not that it will work on
real data.

**Real data: next.** The atlas is OneK1K: about 1,000 donors and 16 immune cell types, giving a
tensor of 981 × 16 × 6,444 genes after filters. The burden statistics are for 18 quantitative
traits in UK Biobank. The pipeline is written but has not been run yet.

## Repository layout

| Path | Contents |
|---|---|
| `src/sctb/model.py` | The coupled model |
| `src/sctb/cv.py` | Cross-validation over genes; predictions for held-out genes |
| `src/sctb/baselines.py` | scITD-style decomposition, random gene sets, weighted ridge regression |
| `src/sctb/simulate.py` | Simulator: atlas plus two independent halves of burden statistics |
| `src/sctb/metrics.py` | Noise-corrected R², program matching, AUROC |
| `src/sctb/data.py` | Loaders for the OneK1K moments and the plink2 burden tables |
| `scripts/` | Simulation benchmark, split-half burden job, real-data preparation and fit |
| `tests/` | Unit tests (pytest) |

## Running it

You need Python 3.11 with numpy, scipy, pandas and PyTorch. Run the tests and the simulation
benchmark from the repository root:

```bash
python -m pytest tests -q
```

```bash
python scripts/run_sim.py --reps 5
```

```bash
python scripts/summarise_sim.py > results/sim/tables.md
```

The benchmark takes about an hour on a single CPU core.

**Real data** needs access to individual-level UK Biobank data, which is not part of this
repository. The steps are:
1. `scripts/burden_splithalf.sh` computes LoF burden summary statistics in two random halves of
   the cohort (a SLURM job; plink2 does the work).
2. `scripts/prepare_real.py` builds the expression tensor and the burden matrices.
3. `scripts/fit_real.sh` fits the model and the baselines on a GPU.

The paths in these scripts point to my cluster and need adjusting. Only summary statistics leave
step 1.
