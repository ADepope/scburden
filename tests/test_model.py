"""Checks of the coupled model, baselines and metrics on small simulated data.

Run: ../sc_dosage_J/envs/scdosage/bin/python -I -m pytest tests -q   (or python -I tests/test_model.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sctb import baselines, cv, metrics, model  # noqa: E402
from sctb.simulate import simulate  # noqa: E402

torch.set_num_threads(1)


def test_burden_loglik_matches_scipy():
    rng = np.random.default_rng(0)
    G, T = 5, 3
    resid = rng.standard_normal((G, T))
    SE = rng.uniform(0.1, 1, (G, T))
    R = np.array([[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]])
    psi = np.array([0.01, 0.02, 0.03])
    gw = np.array([1, 1, 0, 1, 1.0])
    got = model.burden_loglik(*(torch.tensor(x, dtype=torch.float64) for x in (resid, SE, R, psi, gw)))
    want = 0
    for g in range(G):
        cov = np.diag(SE[g]) @ R @ np.diag(SE[g]) + np.diag(psi)
        want += gw[g] * (multivariate_normal(np.zeros(T), cov).logpdf(resid[g]) + T / 2 * np.log(2 * np.pi))
    assert np.isclose(got.item(), want, rtol=1e-8)


def test_r2_heldout_is_noise_corrected():
    # perfect prediction scores ≈ 1 although β̂_B is noisy; predicting 0 scores 0
    rng = np.random.default_rng(1)
    beta = rng.normal(0, 0.1, (20000, 2))
    SE = rng.uniform(0.05, 0.3, beta.shape)
    B = beta + rng.standard_normal(beta.shape) * SE
    assert abs(metrics.r2_heldout(beta, B, SE)[0] - 1) < 0.05
    assert metrics.r2_heldout(np.zeros_like(B), B, SE)[0] == 0
    assert abs(metrics.r2_heldout(0.5 * beta, B, SE)[0] - 0.75) < 0.05


def test_expression_fit_recovers_strong_programs():
    s = simulate(D=100, C=4, G=200, T=4, K_trait=2, K_nuis=1, amp_trait=0.3, amp_nuis=0.3, seed=3)
    f = model.fit(s["X"], s["V"], None, K=6, n_steps=1500, seed=0)
    _, cos = metrics.match_programs(s["truth"]["W"], f["W"])
    assert cos.min() > 0.9, cos
    assert (f["W"] >= 0).all()
    assert f["active"].sum() <= 4  # ARD prunes most of the 6 starting programs (3 true)


def test_coupled_fit_finds_signed_links():
    s = simulate(D=100, C=4, G=300, T=4, K_trait=2, K_nuis=0, amp_trait=0.3, theta_size=(0.2, 0.3),
                 frac_direct=0, log10_f=(-3, -2), seed=4)
    f = model.fit(s["X"], s["V"], s["B_A"], s["SE_A"], s["R"], K=5, eta=1, n_steps=1500, seed=0)
    idx, cos = metrics.match_programs(s["truth"]["W_trait"], f["W"])
    assert cos.min() > 0.9
    true = s["truth"]["Theta"]
    # effect per unit of gene loading, comparable across the two parametrizations
    est = f["Theta"][idx] * np.linalg.norm(f["W"][:, idx], axis=0)[:, None]
    ref = true * np.linalg.norm(s["truth"]["W_trait"], axis=0)[:, None]
    nz = true != 0
    assert (np.sign(est[nz]) == np.sign(ref[nz])).all()
    assert np.corrcoef(est.ravel(), ref.ravel())[0, 1] > 0.9


def test_ridge_nested_noise_features_do_not_help():
    # pure-noise features: held-out gain should not be clearly positive (no leakage)
    rng = np.random.default_rng(5)
    G, T = 600, 3
    SE = rng.uniform(0.05, 0.3, (G, T))
    B = rng.normal(0, 0.05, (G, T)) + rng.standard_normal((G, T)) * SE
    F = rng.standard_normal((G, 10))
    folds = cv.gene_folds(G, 3)
    pred = baselines.ridge_nested(F, B, SE, folds)
    assert metrics.gain_heldout(pred, B, SE) < 1e-4


def test_scitd_features_shapes_and_varimax():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((30, 3, 50))
    X[0, 1, :] = np.nan
    F, L, U = baselines.scitd_features(X, 4)
    assert F.shape == (50, 12) and L.shape == (4, 3, 50) and U.shape == (30, 4)
    assert np.allclose(U.T @ U, np.eye(4), atol=1e-8)  # rotation keeps orthonormal donor factors


def test_freeze_keeps_programs():
    s = simulate(D=60, C=3, G=100, T=2, K_trait=1, K_nuis=1, amp_trait=0.3, seed=7)
    f = model.fit(s["X"], s["V"], None, K=3, n_steps=300)
    g = model.fit(s["X"], s["V"], None, K=3, n_steps=200, init={k: f[k] for k in ("U", "A", "s2")},
                  freeze=("U", "A", "s2"))
    assert np.allclose(f["U"], g["U"]) and np.allclose(f["A"], g["A"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_gwas_helpers():
    from sctb import gwas
    rng = np.random.default_rng(0)
    G, T = 400, 3
    COV = rng.normal(size=(G, T, 2))
    Z = 2 + COV @ np.array([1.0, -0.5]) + rng.normal(size=(G, T))
    folds = np.arange(G) % 4
    Zf = gwas.residualise(Z, COV, folds)
    tr = folds != 0
    # residuals are orthogonal to the design on the training genes of their fold
    D = np.concatenate([np.ones((G, 1)), COV[:, 0]], 1)
    assert np.allclose(D[tr].T @ Zf[0][tr, 0], 0, atol=1e-3)
    oof = gwas.out_of_fold(Zf, folds)
    assert np.allclose(oof[folds == 2], Zf[2][folds == 2])
    P, Y = rng.normal(size=(G, T)), rng.normal(size=(G, T))
    Y[5, 1] = np.nan
    r = gwas.corr_by_trait(P, Y)
    ok = ~np.isnan(Y[:, 1])
    assert np.isclose(r[1], np.corrcoef(P[ok, 1], Y[ok, 1])[0, 1])
    assert gwas.gene_blocks(np.array(["1", "1", "2"]), np.array([1e6, 6e6, 1e6])).tolist() == [0, 1, 2]
