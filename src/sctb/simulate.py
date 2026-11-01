"""Simulated population atlas + two independent halves of burden summary statistics.

Truth:
    trait programs k = 1..K_trait:  gene set (W ≥ 0), cell-type pattern A, donor scores U, and
                                    effects Θ on a few traits each
    nuisance programs:              same, but larger expression variance and no trait effect
                                    (what an expression-only decomposition picks up first)
    direct effects:                 a few genes with trait effects outside any program
Expression: X = Σ_k U A W + residual (s²[c,g]) + sampling noise (V[d,c,g] = κ_g / n_cells[d,c],
i.e. lowly expressed genes and donors with few cells are noisy). Some donor × cell type entries are
missing (too few cells).
Burden: β = W_trait Θ + direct; each half sees β̂ = β + S R S-correlated noise with SE_g from the
gene's LoF carrier frequency (SE ≈ 1 / sqrt(2 N f)).
"""
from __future__ import annotations

import numpy as np


def _programs(rng, G, C, K, frac_genes, n_ct, amp):
    W = np.zeros((G, K))
    A = np.zeros((C, K))
    for k in range(K):
        genes = rng.choice(G, int(frac_genes * G), replace=False)
        W[genes, k] = rng.gamma(2.0, 0.5, len(genes))
        cts = rng.choice(C, rng.integers(n_ct[0], n_ct[1] + 1), replace=False)
        A[cts, k] = amp * rng.choice([-1, 1], len(cts)) * rng.uniform(0.6, 1.4, len(cts))
    return W, A


def simulate(D=200, C=6, G=1000, T=8, K_trait=4, K_nuis=3, amp_trait=0.15, amp_nuis=0.35,
             frac_genes=0.08, theta_size=(0.06, 0.15), links_per_program=(1, 3), null=False,
             frac_direct=0.02, direct_sd=0.05, N_half=150_000, log10_f=(-4.7, -2.5),
             frac_missing=0.05, seed=0):
    rng = np.random.default_rng(seed)
    Wt, At = _programs(rng, G, C, K_trait, frac_genes, (1, 3), amp_trait)
    Wn, An = _programs(rng, G, C, K_nuis, frac_genes, (2, C), amp_nuis)
    W = np.hstack([Wt, Wn])
    A = np.hstack([At, An])
    U = rng.standard_normal((D, K_trait + K_nuis))

    # expression with residual and sampling noise
    M = np.einsum("dk,ck,gk->dcg", U, A, W)
    s2 = (0.08 ** 2) * rng.lognormal(0, 0.5, (C, G))
    kappa = rng.lognormal(np.log(2.0), 1.0, G)                 # per-cell relative noise of each gene
    n_cells = np.round(rng.lognormal(np.log(80), 0.8, (D, C))).clip(20)
    V = kappa[None, None, :] / n_cells[:, :, None]
    X = M + rng.standard_normal(M.shape) * np.sqrt(s2[None]) + rng.standard_normal(M.shape) * np.sqrt(V)
    miss = rng.random((D, C)) < frac_missing
    X[miss] = np.nan
    V[miss] = np.nan

    # trait effects: each trait program acts on a few traits
    Theta = np.zeros((K_trait, T))
    if not null:
        for k in range(K_trait):
            ts = rng.choice(T, rng.integers(links_per_program[0], links_per_program[1] + 1), replace=False)
            Theta[k, ts] = rng.choice([-1, 1], len(ts)) * rng.uniform(*theta_size, len(ts))
    beta = Wt @ Theta
    direct = np.zeros((G, T))
    if not null:
        hit = rng.random((G, T)) < frac_direct
        direct[hit] = rng.normal(0, direct_sd, hit.sum())
    beta = beta + direct

    # cross-trait error correlation: traits in pairs correlated 0.4, otherwise 0.1
    R = np.full((T, T), 0.1)
    for i in range(0, T - 1, 2):
        R[i, i + 1] = R[i + 1, i] = 0.4
    np.fill_diagonal(R, 1.0)
    Lr = np.linalg.cholesky(R)
    f = 10 ** rng.uniform(*log10_f, G)                         # LoF carrier frequency per gene
    se = 1 / np.sqrt(2 * N_half * f)
    SE = np.repeat(se[:, None], T, 1) * rng.uniform(0.95, 1.05, (G, T))  # traits differ a little in N
    halves = []
    for _ in range(2):
        e = (rng.standard_normal((G, T)) @ Lr.T) * SE
        halves.append((beta + e).astype(np.float32))

    return dict(X=X.astype(np.float32), V=V.astype(np.float32), B_A=halves[0], B_B=halves[1],
                SE_A=SE.astype(np.float32), SE_B=SE.astype(np.float32), R=R,
                truth=dict(U=U, A=A, W=W, W_trait=Wt, A_trait=At, Theta=Theta, beta=beta,
                           s2=s2, K_trait=K_trait))
