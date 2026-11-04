"""Post-hoc diagnostics of the real-data null result (not pre-registered; see results/real/).

D1 ceiling: how much gene-level burden signal is there, and is it shared between the halves?
   excess_chi2_X = weighted mean of β̂_X² − SE_X² (w = 1/SE_X², i.e. mean z² − 1) per half;
   cross = Σ w β̂_A β̂_B / Σ w, unbiased for the weighted mean β² because the halves share no people;
   top10/top50 = share of cross carried by the genes with the largest max_t |z_A|.
   ceiling R²_B: predict β̂_B by c_t β̂_A, c_t fitted on the training genes of 5 gene folds and scored
   on the held-out genes (the best a gene's own half-A estimate can do); also without the top 20 genes.
D2 features: two-step programs (fit_expression_only W) + nested ridge, as in fit_real.py, scored on
   all genes and on the genes with max_t |z_A| > 4; plus an in-sample ridge on all genes (half A
   fit, half-B score) as an upper bound for what any linear map from W can reach.
D3 LOEUF: the unsigned target β̂_B² − SE_B² is skewed (z² ~ χ²₁, median 0.45), so genes with large SE
   rank low regardless of β; compare Spearman with β̂_B² − SE_B² against Spearman with z_B², and
   the correlation of LOEUF with SE_B.
Reads only data/real/burden.npz and results/real/fit_expression_only.npz (summary statistics).
Writes results/real/diagnostics.tsv and prints it.
Run: python -I scripts/diagnose_real.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sctb import baselines, cv, metrics  # noqa: E402


def wmean(x, w):
    ok = ~np.isnan(x) & (w > 0)
    return (np.where(ok, x, 0) * np.where(ok, w, 0)).sum(0) / np.where(ok, w, 0).sum(0)


def ceiling_pred(BA, BB, SEB, folds):
    """Held-out prediction c_t β̂_A with c_t = Σ w β̂_A β̂_B / Σ w β̂_A² on the training genes."""
    pred = np.full_like(BA, np.nan)
    w = 1 / SEB ** 2
    for f in np.unique(folds):
        tr, test = folds != f, folds == f
        a, b, ww = BA[tr], BB[tr], w[tr]
        ok = ~(np.isnan(a) | np.isnan(b))
        c = (np.where(ok, ww * a * b, 0)).sum(0) / (np.where(ok, ww * a * a, 0)).sum(0)
        pred[test] = c * BA[test]
    return pred


def main():
    b = np.load(ROOT / "data/real/burden.npz", allow_pickle=True)
    f0 = np.load(ROOT / "results/real/fit_expression_only.npz")
    BA, SEA, BB, SEB = b["B_A"], b["SE_A"], b["B_B"], b["SE_B"]
    traits = list(b["traits"])
    G = len(BA)
    folds = cv.gene_folds(G, 5, seed=0)
    zA, zB = BA / SEA, BB / SEB
    topz = np.nanmax(np.abs(zA), 1)
    order = np.argsort(-topz)
    wA, wB = 1 / SEA ** 2, 1 / SEB ** 2
    wAB = 1 / (SEA * SEB)

    rows = []
    # D1
    cross_g = np.where(np.isnan(BA * BB), 0, wAB * BA * BB)
    ceil = ceiling_pred(BA, BB, SEB, folds)
    keep = np.ones(G, bool)
    keep[order[:20]] = False
    _, ceil_t = metrics.r2_heldout(ceil, BB, SEB)
    for j, tr in enumerate(traits + ["pooled"]):
        sl = slice(None) if tr == "pooled" else j
        cg = cross_g[:, sl].reshape(G, -1).sum(1)
        rows.append(dict(
            trait=tr,
            excess_chi2_A=np.nanmean(zA[:, sl] ** 2) - 1,
            excess_chi2_B=np.nanmean(zB[:, sl] ** 2) - 1,
            cross_zAzB=np.nanmean((zA * zB)[:, sl]),
            top10_share=cg[order[:10]].sum() / cg.sum(),
            top50_share=cg[order[:50]].sum() / cg.sum(),
            ceiling_R2B=metrics.r2_heldout(ceil, BB, SEB)[0] if tr == "pooled" else ceil_t[j],
            ceiling_R2B_drop20=(metrics.r2_heldout(ceil[keep], BB[keep], SEB[keep])[0] if tr == "pooled"
                                else metrics.r2_heldout(ceil[keep], BB[keep], SEB[keep])[1][j]),
        ))
    d = pd.DataFrame(rows)

    # D2
    W = f0["W"][:, f0["active"]]
    p2 = baselines.ridge_nested(W, BA, SEA, folds)
    big = topz > 4
    Fs = (W - W.mean(0)) / np.where(W.std(0) > 0, W.std(0), 1)
    _, _, lam = baselines.ridge_cv(Fs, BA, SEA, folds)
    pin = np.full_like(BA, np.nan)
    for t in range(BA.shape[1]):
        ok = ~np.isnan(BA[:, t])
        pin[:, t] = Fs @ baselines._ridge_fit(Fs[ok], BA[ok, t], wA[ok, t], lam[t] * wA[ok, t].mean())
    r_all, r_all_t = metrics.r2_heldout(p2, BB, SEB)
    r_big, r_big_t = metrics.r2_heldout(p2[big], BB[big], SEB[big])
    r_in, r_in_t = metrics.r2_heldout(pin, BB, SEB)
    d["two_step_R2B"] = list(r_all_t) + [r_all]
    d["two_step_R2B_z4genes"] = list(r_big_t) + [r_big]
    d["insample_W_R2B"] = list(r_in_t) + [r_in]

    # D3
    lo = b["loeuf"]
    sp1, sp2 = [], []
    for j in range(len(traits)):
        ok = ~(np.isnan(lo) | np.isnan(BB[:, j]))
        sp1.append(spearmanr(-lo[ok], (BB[ok, j] ** 2 - SEB[ok, j] ** 2))[0])
        sp2.append(spearmanr(-lo[ok], zB[ok, j] ** 2)[0])
    d["loeuf_rho_target"] = sp1 + [np.mean(sp1)]
    d["loeuf_rho_z2"] = sp2 + [np.mean(sp2)]
    ok = ~np.isnan(lo)
    out = ROOT / "results/real/diagnostics.tsv"
    d.to_csv(out, sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(d.round(4).to_string(index=False))
    print(f"\ngenes with max|z_A| > 4: {big.sum()}; spearman(LOEUF, mean SE_B) = "
          f"{spearmanr(lo[ok], np.nanmean(SEB[ok], 1))[0]:.3f}")
    print("top 10 genes by max|z_A| carry", ", ".join(f"{traits[np.nanargmax(np.abs(zA[g]))]}" for g in order[:10]))


if __name__ == "__main__":
    main()
