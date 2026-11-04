"""Cross-validation over genes for the coupled model (SPEC.md §2, choice of eta; §4 metrics).

For each fold of genes the burden rows of the held-out genes are left out of the fit; their gene
loadings still come from the expression block. Two predictions of the held-out genes' effects:

    joint   W_g Θ from the coupled fit itself
    refit   the coupled fit's donor scores U and cell-type patterns A (the programs, shaped by the
            burden statistics of the training genes) are kept, the gene loadings of ALL genes are
            re-estimated from expression alone, and Θ comes from a weighted ridge regression on the
            training genes (penalty by inner CV). Training and held-out genes then get loadings by
            the same rule, so the prediction is honest; in the joint fit, training genes' loadings
            have also been pulled towards their own (noisy) burden estimates.
"""
from __future__ import annotations

import numpy as np

from . import model
from .baselines import ridge_nested


def gene_folds(G, n_folds, seed=0):
    return np.random.default_rng(seed).permutation(np.arange(G) % n_folds)


def coupled_cv(X, V, B, SE, R, folds, eta, K=10, n_steps=2000, refit_steps=1000, seed=0, device="cpu"):
    """Held-out predictions (joint, refit) for one eta; also returns the per-fold fits.
    B may be one G × T matrix or a dict {fold: G × T} (e.g. covariates regressed out per fold)."""
    G, T = next(iter(B.values())).shape if isinstance(B, dict) else B.shape
    pred_joint = np.full((G, T), np.nan)
    feats, fits = {}, {}
    for f in np.unique(folds):
        tr = folds != f
        fit = model.fit(X, V, B[f] if isinstance(B, dict) else B, SE, R, K=K, eta=eta, burden_genes=tr, n_steps=n_steps,
                        seed=seed, device=device)
        pred_joint[~tr] = model.predict_burden(fit)[~tr]
        re = model.fit(X, V, None, K=K, n_steps=refit_steps, seed=seed, device=device,
                       init={k: fit[k] for k in ("U", "A", "s2")}, freeze=("U", "A", "s2"))
        feats[f] = re["W"]
        fits[f] = fit
    pred_refit = ridge_nested(feats, B, SE, folds, seed=seed)
    return pred_joint, pred_refit, fits
