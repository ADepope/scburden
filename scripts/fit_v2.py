"""Version 2 fit and evaluation (SPEC_v2.md §2–4), from data/real/tensor.npz and data/real/gwas_v2.npz.

Gene scores enter the burden block as B = z (capped, covariates regressed out on the training
chromosomes of each fold) with SE = 1 and R from gwas_v2.npz. Folds = 5 chromosome groups.
1. eta: for each value of --etas, coupled_cv over the chromosome folds; the eta with the best
   held-out gain on the discovery scores is kept, separately for the 'refit' and 'joint' predictions.
2. Baselines on the same folds: B0 covariates only, B1 scITD-style, B2 two-step (expression-only fit
   + nested ridge), B3 random gene sets (20 draws), B5 cell-type specificity (+ nested ridge).
3. Primary score: per trait, Pearson correlation of the out-of-fold predictions with the Biobank
   Japan z of the same genes (covariates regressed out by the same fold rule), mean over the 12
   traits; block bootstrap (~5 Mb gene blocks) for coupled − baseline and for the immune − control
   contrast (criterion 3). Also against raw Biobank Japan z (B0 is scored only there), and the
   within-EUR noise-corrected R² (SE = 1).
4. Full fits at the chosen joint eta: discovery (all genes) and Biobank Japan (same eta, never
   tuned on it); programs matched by W cosine, Θ correlated over the 12 shared traits.
Writes results/v2/: eta_cv.tsv, corr_heldout.tsv, corr_bootstrap.tsv, r2_eur.tsv, preds.npz,
fit_expression_only.npz, fit_disc.npz, fit_rep.npz, replication.tsv. Use a GPU (scripts/fit_v2.sh).
Run: python -I scripts/fit_v2.py [--device cuda]
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
from sctb import baselines, cv, gwas, metrics, model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/real"))
    ap.add_argument("--out", default=str(ROOT / "results/v2"))
    ap.add_argument("--etas", default="1,3,10,30,100")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--max-genes", type=int, default=0, help="smoke test: keep only this many genes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit_v2.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t = np.load(Path(a.data) / "tensor.npz", allow_pickle=True)
    g = np.load(Path(a.data) / "gwas_v2.npz", allow_pickle=True)
    gi = np.arange(len(g["tensor_idx"]))
    if a.max_genes:
        gi = np.sort(np.random.default_rng(0).choice(gi, a.max_genes, replace=False))
    ti = g["tensor_idx"][gi]
    assert (t["genes"][ti] == g["genes"][gi]).all()
    X, V = np.ascontiguousarray(t["X"][:, :, ti]), np.ascontiguousarray(t["V"][:, :, ti])
    Z, COV, Zr, COVr = g["Z"][gi], g["COV"][gi], g["Z_rep"][gi], g["COV_rep"][gi]
    R, Rr, folds = g["R"], g["R_rep"], g["fold"][gi]
    traits, traits_rep, rep_idx, immune = list(g["traits"]), list(g["traits_rep"]), g["rep_idx"], g["immune"]
    imm_rep = immune[rep_idx]
    G, T = Z.shape
    SE, SEr = np.ones_like(Z), np.ones_like(Zr)
    etas = [float(e) for e in a.etas.split(",")]
    log(f"device {a.device}; tensor {X.shape}; gene scores {Z.shape}, replication {Zr.shape}; K_max {a.K}; etas {etas}")

    Zf = gwas.residualise(Z, COV, folds)          # {fold: G × T}
    Zoof = gwas.out_of_fold(Zf, folds)
    Yrep = gwas.out_of_fold(gwas.residualise(Zr, COVr, folds), folds)

    # 1. coupled model, eta by the held-out gain on the discovery scores
    rows, cand = [], {}
    for eta in etas:
        t0 = time.time()
        pj, pr, _ = cv.coupled_cv(X, V, Zf, SE, R, folds, eta, K=a.K, n_steps=a.steps, device=a.device)
        for name, p in (("coupled_joint", pj), ("coupled_refit", pr)):
            rows.append(dict(method=name, eta=eta, gain_EUR=metrics.gain_heldout(p, Zoof, SE),
                             R2_EUR=metrics.r2_heldout(p, Zoof, SE)[0]))
            cand[name, eta] = p
        log(f"eta {eta}: {time.time() - t0:.0f}s  " + "  ".join(f"{r['method']} gain {r['gain_EUR']:.4f}" for r in rows[-2:]))
    eta_df = pd.DataFrame(rows)
    eta_df.to_csv(out / "eta_cv.tsv", sep="\t", index=False)
    best = eta_df.loc[eta_df.groupby("method").gain_EUR.idxmax()].set_index("method").eta.to_dict()
    preds = {m: cand[m, e] for m, e in best.items()}
    log(f"chosen eta: {best}")

    # 2. baselines
    f0 = model.fit(X, V, None, K=a.K, n_steps=a.steps, device=a.device)
    np.savez(out / "fit_expression_only.npz", **{k: v for k, v in f0.items() if isinstance(v, np.ndarray)})
    preds["two_step"] = baselines.ridge_nested(f0["W"], Zf, SE, folds)
    F1, _, _ = baselines.scitd_features(X, max(1, int(f0["active"].sum())))
    preds["scitd"] = baselines.ridge_nested(F1, Zf, SE, folds)
    preds["ct_spec"] = baselines.ridge_nested(gwas.ct_features(g["ct_mean"][gi]), Zf, SE, folds)
    rng = np.random.default_rng(0)
    act = f0["W"][:, f0["active"]]
    sparsity = (act > 1e-3 * act.max()).mean(0)
    rnd = np.stack([baselines.ridge_nested(baselines.random_features(G, sparsity, rng), Zf, SE, folds)
                    for _ in range(a.n_random)])
    b0 = gwas.covariate_pred(Z, COV, folds)
    np.savez(out / "preds.npz", genes=g["genes"][gi], traits=traits, random=rnd, covariates=b0, Z_oof=Zoof,
             Y_rep=Yrep, **preds)

    # 3. scoring: Biobank Japan correlation (primary), raw Biobank Japan z, within-EUR R²
    P = {m: p[:, rep_idx] for m, p in preds.items()}
    rows = []
    for m, p in list(P.items()) + [("random", None), ("covariates", None)]:
        for target, Y in (("rep_resid", Yrep), ("rep_raw", Zr)):
            if m == "covariates" and target == "rep_resid":
                continue
            if m == "random":
                r = np.nanmean(gwas.corr_by_trait(rnd[:, :, rep_idx], Y[None]), 0)
            else:
                r = gwas.corr_by_trait(b0[:, rep_idx] if m == "covariates" else p, Y)
            rows.append(dict(method=m, target=target, trait="mean12", corr=np.nanmean(r)))
            rows.append(dict(method=m, target=target, trait="immune", corr=np.nanmean(r[imm_rep])))
            rows.append(dict(method=m, target=target, trait="control", corr=np.nanmean(r[~imm_rep])))
            rows += [dict(method=m, target=target, trait=tr, corr=c) for tr, c in zip(traits_rep, r)]
    corr = pd.DataFrame(rows)
    corr.to_csv(out / "corr_heldout.tsv", sep="\t", index=False)
    log(corr[corr.trait.isin(["mean12", "immune", "control"])].pivot_table(
        index="method", columns=["target", "trait"], values="corr").round(4).to_string())

    rows = []
    for m, p in list(preds.items()) + [("random", None)]:
        if m == "random":
            rr = [metrics.r2_heldout(q, Zoof, SE) for q in rnd]
            pooled, per = np.mean([x[0] for x in rr]), np.mean([x[1] for x in rr], 0)
        else:
            pooled, per = metrics.r2_heldout(p, Zoof, SE)
        rows.append(dict(method=m, trait="pooled", R2_EUR=pooled))
        rows += [dict(method=m, trait=tr, R2_EUR=r) for tr, r in zip(traits, per)]
    pd.DataFrame(rows).to_csv(out / "r2_eur.tsv", sep="\t", index=False)

    # block bootstrap over ~5 Mb gene blocks
    blocks = gwas.gene_blocks(g["chrom"][gi], g["start"][gi])
    members = [np.flatnonzero(blocks == b) for b in range(blocks.max() + 1)]
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(a.n_boot):
        i = np.concatenate([members[b] for b in rng.integers(0, len(members), len(members))])
        r = {m: gwas.corr_by_trait(p[i], Yrep[i]) for m, p in P.items()}
        r["random"] = np.nanmean(gwas.corr_by_trait(rnd[:, i][:, :, rep_idx], Yrep[i][None]), 0)
        boot.append(r)
    point = {m: gwas.corr_by_trait(p, Yrep) for m, p in P.items()}
    point["random"] = np.nanmean(gwas.corr_by_trait(rnd[:, :, rep_idx], Yrep[None]), 0)
    rows = []
    for m in ("coupled_refit", "coupled_joint"):
        for base in ("two_step", "ct_spec", "scitd", "random"):
            for scope, sel in (("mean12", slice(None)), ("immune", imm_rep), ("control", ~imm_rep),
                               ("immune_minus_control", None)):
                f = ((lambda r: np.nanmean(r[m][imm_rep] - r[base][imm_rep]) - np.nanmean(r[m][~imm_rep] - r[base][~imm_rep]))
                     if sel is None else (lambda r, s=sel: np.nanmean(r[m][s] - r[base][s])))
                d = np.array([f(r) for r in boot])
                rows.append(dict(method=m, vs=base, scope=scope, diff=f(point),
                                 lo=np.quantile(d, 0.025), hi=np.quantile(d, 0.975)))
    bt = pd.DataFrame(rows)
    bt.to_csv(out / "corr_bootstrap.tsv", sep="\t", index=False)
    log(bt.round(4).to_string(index=False))

    # 4. full fits for interpretation and replication (Biobank Japan at the discovery eta)
    eta = best["coupled_joint"]
    all_ = np.zeros(G, int)
    Zall = gwas.residualise(Z, COV, all_)[0]
    Zrall = gwas.residualise(Zr, COVr, all_)[0]
    fd = model.fit_restarts(X, V, Zall, SE, R, K=a.K, eta=eta, n_steps=a.steps, device=a.device)
    fr = model.fit_restarts(X, V, Zrall, SEr, Rr, K=a.K, eta=eta, n_steps=a.steps, device=a.device)
    for name, fit, tr in (("fit_disc", fd, traits), ("fit_rep", fr, traits_rep)):
        np.savez(out / f"{name}.npz", genes=g["genes"][gi], cell_types=t["cell_types"], traits=tr, eta=eta,
                 **{k: v for k, v in fit.items() if isinstance(v, np.ndarray)})
    ka, kb = np.flatnonzero(fd["active"]), np.flatnonzero(fr["active"])
    idx, cos = metrics.match_programs(fd["W"][:, ka], fr["W"][:, kb])
    rows = []
    for i, (j, c) in enumerate(zip(idx, cos)):
        if j < 0:
            continue
        ta = fd["Theta"][ka[i], rep_idx] * np.linalg.norm(fd["W"][:, ka[i]])
        tb = fr["Theta"][kb[j]] * np.linalg.norm(fr["W"][:, kb[j]])
        big = np.abs(ta) > 0.1 * np.abs(ta).max()
        rows.append(dict(program_disc=ka[i], program_rep=kb[j], W_cos=c, theta_cor=np.corrcoef(ta, tb)[0, 1],
                         same_sign_frac=np.mean(np.sign(ta[big]) == np.sign(tb[big])) if big.any() else np.nan))
    pd.DataFrame(rows).to_csv(out / "replication.tsv", sep="\t", index=False)
    log(pd.DataFrame(rows).round(3).to_string(index=False))
    log("DONE")


if __name__ == "__main__":
    main()
