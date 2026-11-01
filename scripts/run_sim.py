"""Simulation benchmark (SPEC.md §4a): coupled model vs baselines, scenarios S1 main, S2 aligned, S3 null.

Per replicate:
  coupled      eta chosen from ETAS by held-out burden R² on half A (3 gene folds); 'refit' and
               'joint' predictions (see sctb/cv.py), each with its own eta
  two_step     expression-only fit of the same model, then nested weighted ridge on its W (B2)
  scitd        scITD-style SVD + varimax, cell-type × gene loadings as features, ridge (B1)
  random       random gene sets with the two-step W's column sparsity, ridge, mean of 20 draws (B3)
  null         predict 0 (B0)
Metrics: R2_B (noise-corrected R² on the independent half B, held-out genes; undefined in S3),
gainx1e4_B (weighted reduction of squared error vs predicting 0, × 1e4; ≤ 0 expected in S3), r2_true (against the
true effects, held-out genes), prog_cos (a1, matched cosine of trait programs), link_auc (a4),
celltype_cor (a5).
Writes results/sim/metrics.tsv and results/sim/eta.tsv.
Run: python -I scripts/run_sim.py [--reps 5] [--quick]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sctb import baselines, cv, metrics, model  # noqa: E402
from sctb.simulate import simulate  # noqa: E402

SCENARIOS = {
    "S1_main": dict(amp_trait=0.07, K_nuis=3),
    "S2_aligned": dict(amp_trait=0.07, K_nuis=0),
    "S3_null": dict(amp_trait=0.07, K_nuis=3, null=True),
}
ETAS = [1, 3, 10, 30]
K = 10


def program_metrics(truth, W_est, A_est=None, Theta_est=None):
    """a1 matched cosine, a4 link AUROC, a5 cell-type pattern correlation (trait programs only)."""
    idx, cos = metrics.match_programs(truth["W_trait"], W_est)
    out = {"prog_cos": cos.mean()}
    if A_est is not None:
        out["celltype_cor"] = np.mean([abs(np.corrcoef(truth["A_trait"][:, k], A_est[:, j])[0, 1])
                                       if A_est[:, j].std() > 0 else 0 for k, j in enumerate(idx)])
    if Theta_est is not None and truth["Theta"].any():
        # scale Θ by the program's loading norm so effects are comparable across programs
        est = np.abs(Theta_est[idx]) * np.linalg.norm(W_est[:, idx], axis=0)[:, None]
        out["link_auc"] = metrics.auroc(est, truth["Theta"] != 0)
    return out


def score(pred, s):
    return {"R2_B": metrics.r2_heldout(pred, s["B_B"], s["SE_B"])[0],
            "gainx1e4_B": 1e4 * metrics.gain_heldout(pred, s["B_B"], s["SE_B"]),
            "r2_true": metrics.r2_true(pred, s["truth"]["beta"]) if s["truth"]["beta"].any() else np.nan,
            "R2_A": metrics.r2_heldout(pred, s["B_A"], s["SE_A"])[0],
            "gain_A": metrics.gain_heldout(pred, s["B_A"], s["SE_A"])}


def one_replicate(scen, rep, sim_kw, n_steps):
    s = simulate(seed=1000 * list(SCENARIOS).index(scen) + rep, **sim_kw)
    X, V, B, SE, R, truth = s["X"], s["V"], s["B_A"], s["SE_A"], s["R"], s["truth"]
    G = B.shape[0]
    folds = cv.gene_folds(G, 3, seed=rep)
    rows, eta_rows = [], []

    # coupled: every eta, choose by half-A held-out gain (half B untouched). Not by R²_A: its
    # denominator Σ w (β̂² − SE²) can be negative when effects are weak, which flips the ordering.
    best = {}
    for eta in ETAS:
        pj, pr, _ = cv.coupled_cv(X, V, B, SE, R, folds, eta, K=K, n_steps=n_steps, seed=rep)
        for name, p in (("coupled_joint", pj), ("coupled_refit", pr)):
            sc = score(p, s)
            eta_rows.append(dict(scenario=scen, rep=rep, method=name, eta=eta, **sc))
            if name not in best or sc["gain_A"] > best[name][1]["gain_A"]:
                best[name] = (eta, sc)
    for name, (eta, sc) in best.items():
        full = model.fit(X, V, B, SE, R, K=K, eta=eta, n_steps=n_steps, seed=rep)
        rows.append(dict(scenario=scen, rep=rep, method=name, eta=eta, **sc,
                         **program_metrics(truth, full["W"], full["A"], full["Theta"])))

    # B2 two-step: same model, expression only
    f0 = model.fit(X, V, None, K=K, n_steps=n_steps, seed=rep)
    p0 = baselines.ridge_nested(f0["W"], B, SE, folds, seed=rep)
    _, coef0, _ = baselines.ridge_cv(f0["W"], B, SE, folds)
    rows.append(dict(scenario=scen, rep=rep, method="two_step", **score(p0, s),
                     **program_metrics(truth, f0["W"], f0["A"], coef0)))

    # B1 scITD-style, K matched to the two-step fit's active programs
    k1 = max(1, int(f0["active"].sum()))
    F1, L1, _ = baselines.scitd_features(X, k1)
    p1 = baselines.ridge_nested(F1, B, SE, folds, seed=rep)
    rows.append(dict(scenario=scen, rep=rep, method="scitd", **score(p1, s),
                     **program_metrics(truth, baselines.scitd_gene_vectors(L1))))

    # B3 random gene sets with the two-step W's sparsity
    rng = np.random.default_rng(rep)
    act = f0["W"][:, f0["active"]]
    sparsity = (act > 1e-3 * act.max()).mean(0)
    rnd = [score(baselines.ridge_nested(baselines.random_features(G, sparsity, rng), B, SE, folds,
                                        seed=rep), s) for _ in range(20)]
    rows.append(dict(scenario=scen, rep=rep, method="random",
                     **{k: np.mean([r[k] for r in rnd]) for k in rnd[0]}))

    # B0 null
    rows.append(dict(scenario=scen, rep=rep, method="null", **score(np.zeros_like(B), s)))
    return rows, eta_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="1 replicate, small, few steps (smoke test)")
    ap.add_argument("--out", default=str(ROOT / "results" / "sim"))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    a = ap.parse_args()
    torch.set_num_threads(1)
    sim_kw = dict(D=150, C=6, G=600, T=8)
    if a.quick:
        a.reps, a.steps = 1, 300
        sim_kw = dict(D=60, C=4, G=200, T=4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, eta_rows = [], []
    for scen in a.scenarios.split(","):
        kw = SCENARIOS[scen]
        for rep in range(a.reps):
            t0 = time.time()
            r, e = one_replicate(scen, rep, {**sim_kw, **kw}, a.steps)
            rows += r
            eta_rows += e
            pd.DataFrame(rows).to_csv(out / "metrics.tsv", sep="\t", index=False)
            pd.DataFrame(eta_rows).to_csv(out / "eta.tsv", sep="\t", index=False)
            print(f"{scen} rep {rep}: {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["scenario", "method"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
