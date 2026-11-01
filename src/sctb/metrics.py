"""Evaluation metrics (SPEC.md §4)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def r2_heldout(pred, B, SE):
    """Noise-corrected fraction of the true effect variance explained, scored on an independent half.

    E[β̂_B·pred] = β·pred and E[β̂_B² − SE_B²] = β², so with weights w = 1/SE_B²
        R² = Σ w (2 β̂_B pred − pred²) / Σ w (β̂_B² − SE_B²)
    estimates 1 − Σ w (β − pred)² / Σ w β². It can be negative (worse than predicting 0).
    Returns (pooled R², per-trait R²).
    """
    ok = ~(np.isnan(pred) | np.isnan(B) | np.isnan(SE))
    w = np.where(ok, 1 / np.where(ok, SE, 1) ** 2, 0)
    p, b, s = (np.where(ok, x, 0) for x in (pred, B, SE))
    num = (w * (2 * b * p - p ** 2)).sum(0)
    den = (w * (b ** 2 - s ** 2)).sum(0)
    return num.sum() / den.sum(), num / den


def gain_heldout(pred, B, SE):
    """Weighted mean of 2 β̂_B·pred − pred² (w = 1/SE_B²): unbiased for Σ w (β² − (β − pred)²) / Σ w,
    the reduction in squared error against predicting 0. Unlike R² it stays defined when there are
    no true effects (null simulation), where its expectation is ≤ 0."""
    ok = ~(np.isnan(pred) | np.isnan(B) | np.isnan(SE))
    w = np.where(ok, 1 / np.where(ok, SE, 1) ** 2, 0)
    p, b = np.where(ok, pred, 0), np.where(ok, B, 0)
    return (w * (2 * b * p - p ** 2)).sum() / w.sum()


def r2_true(pred, beta, w=None):
    """1 − Σ w (β − pred)² / Σ w β² against known true effects (simulation)."""
    w = np.ones_like(beta) if w is None else w
    return 1 - (w * (beta - pred) ** 2).sum() / (w * beta ** 2).sum()


def cosine_matrix(A, B):
    """Cosine similarity between the columns of A and of B."""
    na = np.linalg.norm(A, axis=0)
    nb = np.linalg.norm(B, axis=0)
    return (A.T @ B) / np.outer(np.where(na > 0, na, 1), np.where(nb > 0, nb, 1))


def match_programs(W_true, W_est):
    """Hungarian matching of true programs to estimated gene vectors by |cosine|.
    Returns (index of the estimated column for each true program, matched |cosine|)."""
    S = np.abs(cosine_matrix(W_true, W_est))
    r, c = linear_sum_assignment(-S)
    idx = np.full(W_true.shape[1], -1)
    idx[r] = c
    return idx, S[r, c]


def auroc(score, label):
    """Area under the ROC curve (ties averaged); NaN if one class is empty."""
    score = np.asarray(score, float).ravel()
    label = np.asarray(label, bool).ravel()
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(score)
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
