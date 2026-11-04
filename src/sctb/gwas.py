"""Version 2 helpers (SPEC_v2.md §1, §4): gene-score covariates, cell-type specificity features,
gene blocks for the bootstrap and the correlation score.

residualise      per fold, regress each trait's z on [1, covariates] over the training genes only and
                 subtract the fit from all genes; returns {fold: G × T}
out_of_fold      assemble one G × T matrix taking each gene's row from its own fold's matrix
covariate_pred   baseline B0: held-out prediction of z from the covariates alone
ct_features      baseline B5: log mean expression and specificity (share of the gene's summed mean
                 expression) in each cell type, G × 2C
gene_blocks      contiguous blocks of ~size bp per chromosome
corr_by_trait    Pearson correlation per column over genes with both values; NaN-aware
"""
from __future__ import annotations

import numpy as np


def _ols(C, y):
    return np.linalg.lstsq(C, y, rcond=None)[0]


def _design(COV):
    """COV: G × T × q → G × T × (q + 1) with an intercept."""
    return np.concatenate([np.ones(COV.shape[:2] + (1,)), COV], 2)


def residualise(Z, COV, folds):
    D = _design(COV)
    out = {}
    for f in np.unique(folds):
        tr = folds != f
        R = np.full_like(Z, np.nan, dtype=np.float64)
        for t in range(Z.shape[1]):
            ok = tr & np.isfinite(Z[:, t]) & np.isfinite(D[:, t]).all(1)
            fit = np.isfinite(D[:, t]).all(1)
            R[fit, t] = Z[fit, t] - D[fit, t] @ _ols(D[ok, t], Z[ok, t])
        out[f] = R.astype(np.float32)
    return out


def out_of_fold(by_fold, folds):
    M = np.full_like(next(iter(by_fold.values())), np.nan)
    for f, m in by_fold.items():
        M[folds == f] = m[folds == f]
    return M


def covariate_pred(Z, COV, folds):
    D = _design(COV)
    P = np.full(Z.shape, np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        for t in range(Z.shape[1]):
            ok = tr & np.isfinite(Z[:, t]) & np.isfinite(D[:, t]).all(1)
            P[te, t] = D[te, t] @ _ols(D[ok, t], Z[ok, t])
    return P


def ct_features(M, eps=1e-7):
    spec = M / np.where(M.sum(1, keepdims=True) > 0, M.sum(1, keepdims=True), 1)
    return np.hstack([np.log(M + eps), spec])


def gene_blocks(chrom, start, size=5e6):
    key = np.char.add(np.asarray(chrom).astype(str), np.char.add(":", (np.asarray(start) // size).astype(int).astype(str)))
    return np.unique(key, return_inverse=True)[1]


def corr_by_trait(P, Y):
    """Pearson correlation of P[:, t] and Y[:, t] for each t; P, Y G × T (or n_draws × G × T)."""
    ok = np.isfinite(P) & np.isfinite(Y)
    n = ok.sum(-2)
    p, y = np.where(ok, P, 0), np.where(ok, Y, 0)
    mp, my = p.sum(-2) / n, y.sum(-2) / n
    cov = (p * y).sum(-2) / n - mp * my
    vp = (p * p).sum(-2) / n - mp ** 2
    vy = (y * y).sum(-2) / n - my ** 2
    return cov / np.sqrt(vp * vy)
