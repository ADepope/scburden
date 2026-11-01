# sc_tensor_burden

Multicellular gene programs learned jointly from a population single-cell atlas and rare-variant
burden statistics.

The model is a noise-aware coupled matrix–tensor factorization:
- **Expression tensor:** donors × cell types × genes, from OneK1K pseudobulk shares, with their
  known sampling variance.
- **Burden matrix:** genes × traits of UK Biobank LoF burden effects, with their standard errors
  and cross-trait error correlation.
- **Coupling:** the two blocks share non-negative gene loadings (the programs).

The question is whether the genes' burden effects help find the multicellular programs that matter
for traits, measured by predicting burden effects of held-out genes in an independent half of UK
Biobank.

- [NOVELTY.md](NOVELTY.md): prior work and the novelty verdict.
- [SPEC.md](SPEC.md): model, baselines and metrics, fixed before the real-data fit.
- [results/sim/SUMMARY.md](results/sim/SUMMARY.md): simulation benchmark.

## Layout

| Path | Contents |
|---|---|
| `src/sctb/model.py` | The coupled model (MAP, PyTorch; CPU or GPU) |
| `src/sctb/cv.py` | Gene-fold cross-validation, the joint and refit predictions |
| `src/sctb/baselines.py` | scITD-style decomposition, random gene sets, nested weighted ridge |
| `src/sctb/simulate.py` | Simulator: planted trait programs, nuisance programs, two burden halves |
| `src/sctb/metrics.py` | Noise-corrected held-out R², program matching, AUROC |
| `src/sctb/data.py` | Real-data loaders (OneK1K moments → tensor; plink2 tables → matrices; R; LOEUF) |
| `scripts/run_sim.py` | Simulation benchmark |
| `scripts/burden_splithalf.sh` | Split-half burden statistics (SLURM; **run by the user**) |
| `scripts/prepare_real.py`, `scripts/fit_real.py`, `scripts/fit_real.sh` | Real-data pipeline |
| `tests/` | pytest |

Python environment: `../sc_dosage_J/envs/scdosage` (numpy, scipy, pandas, torch). Always run with
`-I`.

```bash
../sc_dosage_J/envs/scdosage/bin/python -I -m pytest tests -q -p no:cacheprovider
```

```bash
../sc_dosage_J/envs/scdosage/bin/python -I scripts/run_sim.py --reps 5
```

## Real data: steps

1. **Split-half burden statistics.** Run by the user. The script reads individual-level data
   through plink2 only. It writes summary statistics to `data/burden_halves/half{1,2}/` and
   `data/burden_halves/half1.id`.
   ```bash
   sbatch scripts/burden_splithalf.sh
   ```
   This is 18 traits × 2 halves, about 36 plink2 runs. Each run took a few minutes in the
   full-sample version, so the job is set to 3 h.
2. **Build the inputs.** This writes `data/real/tensor.npz` and `data/real/burden.npz`. It uses
   OneK1K moments and summary statistics only, and needs about 4 GB of RAM.
   ```bash
   ../sc_dosage_J/envs/scdosage/bin/python -I scripts/prepare_real.py
   ```
3. **Fit and evaluate.** Uses one A100 GPU, estimated at 2–4 h. The outputs go to `results/real/`.
   ```bash
   sbatch scripts/fit_real.sh
   ```

No step reads individual-level data except step 1, which plink2 runs inside the user's job.
`data/`, `logs/` and `results/real/` are gitignored.
