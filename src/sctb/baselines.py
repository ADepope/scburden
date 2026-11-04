"""Baselines: gene features from expression only, then regression of the burden effects on them.

scitd_features   scITD-style decomposition (unweighted, standardized, SVD + varimax on the donor
                 mode, full rank in the cell-type and gene modes); features = cell-type × gene
                 loadings of each program
random_features  random non-negative gene sets with a given column sparsity
ridge_cv         per-trait weighted ridge regression of β̂ on gene features, penalty chosen by
                 cross-validation over gene folds; returns held-out predictions
"""
from __future__ import annotations

import numpy as np


def varimax(Phi, gamma=1.0, n_iter=100, tol=1e-6):
    """Varimax rotation of the columns of Phi (p × k); returns (rotated, rotation)."""
    p, k = Phi.shape
    Rot = np.eye(k)
    d = 0
    for _ in range(n_iter):
        L = Phi @ Rot
        u, s, vt = np.linalg.svd(Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag((L ** 2).sum(0))))
        Rot = u @ vt
        d_old, d = d, s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ Rot, Rot


def scitd_features(X, K):
    """scITD-style programs from the D × C × G tensor (NaN = missing, set to the (c,g) mean).

    Each (c, g) is centered and scaled to unit variance across donors (scITD's per-cell-type gene
    scaling), the donor × (C·G) unfolding is decomposed by SVD, the top K donor factors are varimax-
    rotated, and each program's loadings are L_k = U_kᵀ X (C × G).
    Returns features G × (K·C) (loadings of every program in every cell type for each gene),
    the loadings K × C × G and the donor scores D × K.
    """
    D, C, G = X.shape
    Xs = np.where(np.isnan(X), np.nanmean(X, 0, keepdims=True), X)
    Xs = Xs - Xs.mean(0, keepdims=True)
    sd = Xs.std(0, keepdims=True)
    Xs = np.divide(Xs, sd, out=np.zeros_like(Xs), where=sd > 0)
    Y = Xs.reshape(D, C * G)
    u, s, vt = np.linalg.svd(Y, full_matrices=False)
    Ud, _ = varimax(u[:, :K])
    L = (Ud.T @ Y).reshape(K, C, G)
    return np.transpose(L, (2, 0, 1)).reshape(G, K * C), L, Ud


def scitd_gene_vectors(L):
    """Gene-level summary of scITD loadings (K × C × G) for matching to true programs: ‖L_k[:, g]‖."""
    return np.sqrt((L ** 2).sum(1)).T


def random_features(G, sparsity, rng):
    """Random non-negative gene sets: column k has a fraction sparsity[k] of genes, Gamma loadings."""
    F = np.zeros((G, len(sparsity)))
    for k, s in enumerate(sparsity):
        n = max(1, int(round(s * G)))
        F[rng.choice(G, n, replace=False), k] = rng.gamma(2.0, 0.5, n)
    return F


def _ridge_fit(F, y, w, lam):
    Fw = F * w[:, None]
    return np.linalg.solve(F.T @ Fw + lam * np.eye(F.shape[1]), Fw.T @ y)


def ridge_cv(F, B, SE, folds, lambdas=(0.1, 1, 10, 100, 1e3, 1e4, 1e5)):
    """Held-out predictions of β̂ (G × T) from features F by weighted ridge (weights 1/SE²).

    Features are standardized on the training genes; the penalty is chosen per trait by the weighted
    squared error over the held-out folds (pooled). Returns (pred G × T, coefficients K × T fitted
    on all genes with the chosen penalties, chosen lambdas).
    """
    G, T = B.shape
    F = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), 1)
    preds = np.zeros((len(lambdas), G, T))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        for t in range(T):
            ok = tr & ~np.isnan(B[:, t])
            w = 1 / SE[ok, t] ** 2
            for i, lam in enumerate(lambdas):
                preds[i, te, t] = F[te] @ _ridge_fit(F[ok], B[ok, t], w, lam * w.mean())
    err = np.nansum((preds - B[None]) ** 2 / SE[None] ** 2, axis=1)   # lambdas × T
    best = err.argmin(0)
    pred = preds[best, :, np.arange(T)].T
    coef = np.zeros((F.shape[1], T))
    for t in range(T):
        ok = ~np.isnan(B[:, t])
        w = 1 / SE[ok, t] ** 2
        coef[:, t] = _ridge_fit(F[ok], B[ok, t], w, lambdas[best[t]] * w.mean())
    return pred, coef, np.asarray(lambdas)[best]


def ridge_nested(F, B, SE, folds, inner=3, seed=0):
    """Held-out predictions with no leakage: for each outer fold, the ridge penalty is chosen by an
    inner cross-validation over the training genes only, then the model is refitted on all training
    genes and applied to the held-out genes. F may be one G × p matrix (the same features in every
    fold) or a dict {fold: G × p} (features recomputed per fold, as for the coupled model); B may
    likewise be a dict {fold: G × T}."""
    rng = np.random.default_rng(seed)
    Bd = B
    G, T = next(iter(Bd.values())).shape if isinstance(Bd, dict) else Bd.shape
    pred = np.full((G, T), np.nan)
    for f in np.unique(folds):
        Ff = F[f] if isinstance(F, dict) else F
        B = Bd[f] if isinstance(Bd, dict) else Bd
        tr, te = folds != f, folds == f
        mu, sd = Ff[tr].mean(0), Ff[tr].std(0)
        Fs = (Ff - mu) / np.where(sd > 0, sd, 1)
        inner_folds = rng.permutation(np.arange(tr.sum()) % inner)
        _, _, lam = ridge_cv(Fs[tr], B[tr], SE[tr], inner_folds)
        for t in range(T):
            ok = tr & ~np.isnan(B[:, t])
            w = 1 / SE[ok, t] ** 2
            pred[te, t] = Fs[te] @ _ridge_fit(Fs[ok], B[ok, t], w, lam[t] * w.mean())
    return pred
