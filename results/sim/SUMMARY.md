# Simulation benchmark: summary (2026-11-01)

## Setup

The settings are in SPEC.md §4a and `scripts/run_sim.py`. The benchmark ran on one CPU core, at
about 4 min per replicate.

| Setting | Value |
|---|---|
| Tensor | 150 donors × 6 cell types × 600 genes |
| Burden | 8 traits, two independent burden halves of 150k people each |
| Trait programs | 4, weak in expression (amplitude 0.07) |
| Nuisance programs | 3 in S1 (amplitude 0.35, no trait effect); none in S2 |
| Effects | About 0.7% of gene–trait pairs have \|z\| > 3.3 in one half. This is as sparse as the real LoF burden statistics. |
| Replicates | 5 per scenario |

How the numbers were computed:
- All predictions are for **held-out genes** (3 gene folds) and are scored on the **independent
  half B**.
- $R^2_B$ is noise-corrected.
- η and the ridge penalties were chosen on half A only.

Full tables (mean ± SE over replicates): [tables.md](tables.md). Raw data: `metrics.tsv` (S1, S2)
and `s3_rerun/metrics.tsv` (S3, rerun with the corrected η rule, see SPEC "Deviations").

## Main results

| | S1 main (nuisance programs) | S2 aligned (no nuisance) |
|---|---|---|
| **Held-out $R^2_B$** | | |
| coupled, refit route | **0.388 ± 0.150** | **0.551 ± 0.112** |
| coupled, joint route | 0.296 ± 0.136 | 0.471 ± 0.103 |
| two-step (same model, expression only, then ridge) | 0.278 ± 0.146 | 0.465 ± 0.100 |
| scITD-style + ridge | 0.131 ± 0.090 | 0.180 ± 0.080 |
| random gene sets + ridge | −0.016 | −0.013 |
| **Paired gain, refit − two-step, $R^2_B$** | **+0.110 ± 0.048** (5/5 replicates) | +0.086 ± 0.068 (4/5) |
| **Program recovery** (matched cosine) | | |
| coupled | 0.63 | 0.62 (joint fit: 0.70) |
| two-step | 0.42 | 0.50 |
| scITD-style | 0.26 | 0.39 |
| **Program–trait link AUROC** | | |
| coupled joint fit | 0.92 | 0.81 |
| two-step ridge coefficients | 0.68 | 0.65 |
| **Cell-type pattern correlation** | | |
| coupled | 0.81 | 0.82 |
| two-step | 0.65 | 0.73 |

**S3 null** ($\Theta = 0$, nuisance present). Held-out gain against predicting 0, × 1e4 (expected ≤ 0):

| Method | Gain |
|---|---|
| refit | −0.08 |
| two-step | −0.13 |
| scITD-style | −0.18 |
| random gene sets | −0.14 |

So coupling creates no spurious predictive signal. $R^2_B$ is not meaningful in S3 because its
denominator is about 0.

## Reading

- **Coupling helps most where it should.** The trait-relevant programs are weak in expression and
  hidden behind stronger nuisance programs (S1). There, the burden statistics steer the
  factorization toward them:
  - program recovery goes from 0.42 to 0.63;
  - held-out $R^2$ goes from 0.28 to 0.39.

  When the trait programs dominate expression anyway (S2), the gain is smaller and less consistent.
- **The scITD-style baseline is weak here.** Its cell-type × gene loadings are 6·K features per
  gene, fitted without noise weights. On sparse, noisy burden data that loses to a non-negative,
  noise-aware CP.
- **For prediction, use the refit route.** In the joint fit, the training genes' loadings are also
  fitted to their own noisy $\hat\beta$, which held-out genes do not get. This overstates $\Theta$ for new genes,
  and the problem is severe at η ≥ 10 (pilot runs, and the first S3 run). The joint $\Theta$ is still the
  better estimate of program–trait links (AUROC 0.92).
- **η selection:** CV picked η ≈ 9 (S1) and 5 (S2) for refit, and 3 for joint.

## Limits of this benchmark

- It is small: G = 600, 5 replicates, one simulator. The model and simulator share the CP
  structure, so model misspecification is not tested. A Tucker-generated or cell-type-specific
  simulation would be the next robustness check.
- The real data is larger (981 × 16 × 6,444 after filters), has far more programs, and its burden
  signal is concentrated in a few genes. The simulation gains are no promise for real data.
- The real scITD R package was not run. B1 is a simplified re-implementation.
