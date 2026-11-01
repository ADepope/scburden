"""Real-data fit and evaluation (SPEC.md §2–4), from data/real/{tensor,burden}.npz.

1. eta: for each value of --etas, 5-fold gene cross-validation on half A (coupled_cv); the eta with
   the best half-A held-out gain (squared-error reduction vs 0) is kept, separately for the 'refit' and 'joint' predictions.
2. Baselines on the same folds: two-step (expression-only fit + nested ridge), scITD-style, random
   gene sets (20 draws), LOEUF (unsigned metric only), null.
3. Held-out genes scored against the independent half B: noise-corrected R²_B (pooled, per trait),
   bootstrap over genes for the differences coupled − baseline; unsigned Spearman correlation of
   pred² (or −LOEUF) with β̂_B² − SE_B².
4. Full fit on half A at the chosen eta (programs: U, A, W, Θ for interpretation, §4d); with
   --replicate also on half B (own eta grid), for program–trait replication (§4c).
Writes results/real/: eta_cv.tsv, r2_heldout.tsv, r2_bootstrap.tsv, unsigned.tsv, fit_A.npz,
fit_B.npz, replication.tsv. Use a GPU (scripts/fit_real.sh).
Run: python -I scripts/fit_real.py [--device cuda] [--replicate]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sctb import baselines, cv, metrics, model  # noqa: E402


def eta_search(X, V, B, SE, R, folds, etas, K, steps, device, log):
    rows, preds = [], {}
    for eta in etas:
        t0 = time.time()
        pj, pr, _ = cv.coupled_cv(X, V, B, SE, R, folds, eta, K=K, n_steps=steps, device=device)
        for name, p in (("coupled_joint", pj), ("coupled_refit", pr)):
            rows.append(dict(method=name, eta=eta, R2_A=metrics.r2_heldout(p, B, SE)[0],
                             gain_A=metrics.gain_heldout(p, B, SE)))
            preds[name, eta] = p
        log(f"eta {eta}: {time.time() - t0:.0f}s  " + "  ".join(f"{r['method']} {r['R2_A']:.4f}" for r in rows[-2:]))
    df = pd.DataFrame(rows)
    # choose by the gain (numerator of R²): R²'s denominator can be ≤ 0 when effects are weak
    best = df.loc[df.groupby("method").gain_A.idxmax()].set_index("method").eta.to_dict()
    return df, {m: preds[m, e] for m, e in best.items()}, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/real"))
    ap.add_argument("--out", default=str(ROOT / "results/real"))
    ap.add_argument("--etas", default="1,3,10,30,100")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--replicate", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit_real.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t = np.load(Path(a.data) / "tensor.npz", allow_pickle=True)
    b = np.load(Path(a.data) / "burden.npz", allow_pickle=True)
    assert (t["genes"] == b["genes"]).all()
    X, V = t["X"], t["V"]
    BA, SEA, BB, SEB, RA = b["B_A"], b["SE_A"], b["B_B"], b["SE_B"], b["R_A"]
    traits = list(b["traits"])
    G = len(t["genes"])
    etas = [float(e) for e in a.etas.split(",")]
    folds = cv.gene_folds(G, a.folds, seed=0)
    log(f"device {a.device}; tensor {X.shape}; burden {BA.shape}; K_max {a.K}; etas {etas}")

    # 1. coupled model, eta by CV on half A
    eta_df, preds, best = eta_search(X, V, BA, SEA, RA, folds, etas, a.K, a.steps, a.device, log)
    eta_df.to_csv(out / "eta_cv.tsv", sep="\t", index=False)
    log(f"chosen eta: {best}")

    # 2. baselines
    f0 = model.fit(X, V, None, K=a.K, n_steps=a.steps, device=a.device)
    np.savez(out / "fit_expression_only.npz", **{k: v for k, v in f0.items() if isinstance(v, np.ndarray)})
    preds["two_step"] = baselines.ridge_nested(f0["W"], BA, SEA, folds)
    k1 = max(1, int(f0["active"].sum()))
    F1, L1, _ = baselines.scitd_features(X, k1)
    preds["scitd"] = baselines.ridge_nested(F1, BA, SEA, folds)
    rng = np.random.default_rng(0)
    act = f0["W"][:, f0["active"]]
    sparsity = (act > 1e-3 * act.max()).mean(0)
    rnd = [baselines.ridge_nested(baselines.random_features(G, sparsity, rng), BA, SEA, folds) for _ in range(20)]
    preds["null"] = np.zeros_like(BA)

    # 3. scoring on half B (held-out genes, independent people)
    rows = []
    for m, p in preds.items():
        pooled, per = metrics.r2_heldout(p, BB, SEB)
        rows.append(dict(method=m, trait="pooled", R2_B=pooled))
        rows += [dict(method=m, trait=tr, R2_B=r) for tr, r in zip(traits, per)]
    r_rand = [metrics.r2_heldout(p, BB, SEB) for p in rnd]
    rows.append(dict(method="random", trait="pooled", R2_B=np.mean([r[0] for r in r_rand])))
    rows += [dict(method="random", trait=tr, R2_B=np.mean([r[1][i] for r in r_rand])) for i, tr in enumerate(traits)]
    r2 = pd.DataFrame(rows)
    r2.to_csv(out / "r2_heldout.tsv", sep="\t", index=False)
    log(r2[r2.trait == "pooled"].to_string(index=False))

    # bootstrap over genes: coupled − baseline differences
    boot = []
    rng = np.random.default_rng(1)
    for _ in range(a.n_boot):
        i = rng.integers(0, G, G)
        r = {m: metrics.r2_heldout(p[i], BB[i], SEB[i])[0] for m, p in preds.items()}
        boot.append(r)
    boot = pd.DataFrame(boot)
    rows = []
    for m in ("coupled_refit", "coupled_joint"):
        for base in ("two_step", "scitd", "null"):
            d = boot[m] - boot[base]
            rows.append(dict(method=m, vs=base, diff=r2.query("method == @m and trait == 'pooled'").R2_B.item()
                             - r2.query("method == @base and trait == 'pooled'").R2_B.item(),
                             lo=d.quantile(0.025), hi=d.quantile(0.975)))
    pd.DataFrame(rows).to_csv(out / "r2_bootstrap.tsv", sep="\t", index=False)
    log(pd.DataFrame(rows).to_string(index=False))

    # unsigned: does the predicted effect size rank genes by their (noise-corrected) half-B effect?
    target = BB ** 2 - SEB ** 2
    unsigned = {m: p ** 2 for m, p in preds.items() if m != "null"}
    unsigned["loeuf"] = np.repeat(-b["loeuf"][:, None], len(traits), 1)
    rows = []
    for m, s in unsigned.items():
        for j, tr in enumerate(traits):
            ok = ~(np.isnan(s[:, j]) | np.isnan(target[:, j]))
            rows.append(dict(method=m, trait=tr, spearman=spearmanr(s[ok, j], target[ok, j])[0]))
    pd.DataFrame(rows).to_csv(out / "unsigned.tsv", sep="\t", index=False)

    # 4. full fits for interpretation and replication
    fa = model.fit_restarts(X, V, BA, SEA, RA, K=a.K, eta=best["coupled_joint"], n_steps=a.steps, device=a.device)
    np.savez(out / "fit_A.npz", genes=t["genes"], cell_types=t["cell_types"], traits=traits,
             **{k: v for k, v in fa.items() if isinstance(v, np.ndarray)})
    if a.replicate:
        RB = b["R_B"]
        _, _, best_b = eta_search(X, V, BB, SEB, RB, folds, etas, a.K, a.steps, a.device, log)
        fb = model.fit_restarts(X, V, BB, SEB, RB, K=a.K, eta=best_b["coupled_joint"], n_steps=a.steps, device=a.device)
        np.savez(out / "fit_B.npz", genes=t["genes"], cell_types=t["cell_types"], traits=traits,
                 **{k: v for k, v in fb.items() if isinstance(v, np.ndarray)})
        ka, kb = np.flatnonzero(fa["active"]), np.flatnonzero(fb["active"])
        idx, cos = metrics.match_programs(fa["W"][:, ka], fb["W"][:, kb])
        rows = []
        for i, (j, c) in enumerate(zip(idx, cos)):
            if j < 0:
                continue
            ta = fa["Theta"][ka[i]] * np.linalg.norm(fa["W"][:, ka[i]])
            tb = fb["Theta"][kb[j]] * np.linalg.norm(fb["W"][:, kb[j]])
            # W ≥ 0, so the sign of Θ is identified (unlike the signs of U and A)
            big = np.abs(ta) > 0.1 * np.abs(ta).max()
            rows.append(dict(program_A=ka[i], program_B=kb[j], W_cos=c,
                             theta_cor=np.corrcoef(ta, tb)[0, 1],
                             same_sign_frac=np.mean(np.sign(ta[big]) == np.sign(tb[big])) if big.any() else np.nan))
        pd.DataFrame(rows).to_csv(out / "replication.tsv", sep="\t", index=False)
    log("DONE")


if __name__ == "__main__":
    main()
