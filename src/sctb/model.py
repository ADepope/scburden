"""Noise-aware coupled tensor–burden factorization (MAP, PyTorch).

Notation: donors d, cell types c, genes g, traits t, programs k.

Expression block (population single-cell atlas, one value per donor × cell type × gene):
    X[d,c,g] = Σ_k U[d,k] A[c,k] W[g,k] + e,    e ~ N(0, V[d,c,g] + s²[c,g])
V is the known sampling variance of the pseudobulk value (from the cell-level moments), s² a learned
residual between-donor variance per cell type and gene (variation the K programs do not explain).
Entries with NaN in X or V are missing (donor without enough cells, gene not expressed).

Burden block (gene-level LoF burden summary statistics, standardized traits):
    β̂[g,:] = W[g,:] Θ + e_g,    e_g ~ N(0, S_g R S_g + diag(ψ))
S_g = diag(SE[g,:]), R the cross-trait correlation of the estimation errors (people shared between
traits; fixed, estimated from null genes), ψ_t the variance of gene effects outside the programs.
NaN in β̂ or SE = missing (SE set to 1e3, i.e. no information).

The gene loadings W ≥ 0 are shared: a program is a non-negative gene set, its sign and cell-type
pattern sit in A, its effects on traits in Θ (signed).

Priors: U ~ N(0,1); A[:,k] ~ N(0, 1/α_k); W[:,k] ~ Exponential(λ_k); Θ ~ Laplace(0, b_theta);
α_k, λ_k ~ Gamma(1e-3, 1e-3) with closed-form MAP updates (automatic relevance determination: the
precision of an unused program grows and drives it to zero, so K_max only needs to be large enough).

Objective (maximized):  log p(X) + eta · log p(β̂) + log priors.
eta = 1 is the plain joint likelihood. The expression block has far more entries (D·C·G against
G·T), so eta > 1 lets the burden statistics shape the programs; eta is chosen by cross-validation
over held-out genes (see cv.py). Fitting the expression block alone (use_burden=False) gives the
two-step baseline's programs.
"""
from __future__ import annotations

import numpy as np
import torch

A0, B0 = 1e-3, 1e-3  # Gamma prior on the ARD precisions
SE_MISSING = 1e3


def _t(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def burden_loglik(resid, SE, R, psi, gene_w):
    """Σ_g gene_w[g] · log N(resid_g; 0, S_g R S_g + diag(ψ)) (constant dropped).

    resid, SE  G × T;  R  T × T;  psi  T;  gene_w  G (0 = gene left out, e.g. a held-out fold).
    """
    cov = SE[:, :, None] * R[None] * SE[:, None, :] + torch.diag_embed(psi.expand_as(SE))
    L = torch.linalg.cholesky(cov)
    z = torch.linalg.solve_triangular(L, resid[:, :, None], upper=False)[:, :, 0]
    logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    return -0.5 * (gene_w * ((z ** 2).sum(1) + logdet)).sum()


def fit(X, V, B=None, SE=None, R=None, K=10, eta=1.0, burden_genes=None, n_steps=3000, lr=0.03,
        b_theta=1.0, ard_start=500, ard_every=100, seed=0, device="cpu", verbose=False, init=None,
        freeze=()):
    """MAP fit of the coupled model. Returns a dict of numpy arrays (U, A, W, Theta, s2, psi) and
    the final objective.

    X, V          D × C × G arrays (NaN = missing)
    B, SE         G × T burden effects and standard errors (None = expression block only)
    R             T × T error correlation (default identity)
    burden_genes  boolean G, genes whose burden rows enter the fit (default all); the others still
                  get W from the expression block and can be predicted as W Θ
    init, freeze  start from a previous fit's parameters (dict as returned) and keep the named ones
                  fixed, e.g. freeze=("U", "A", "s2") with no burden re-estimates the gene loadings
                  of every gene from expression alone given the programs' donor and cell-type parts
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    V = np.asarray(V, np.float32)
    D, C, G = X.shape
    obs = ~(np.isnan(X) | np.isnan(V))
    mX = _t(obs, device)
    Xt = _t(np.where(obs, X, 0), device)
    Vt = _t(np.where(obs, V, 1), device)
    n_x = obs.sum()

    use_burden = B is not None and eta > 0
    if use_burden:
        B = np.asarray(B, np.float32)
        SE = np.asarray(SE, np.float32)
        T = B.shape[1]
        miss = np.isnan(B) | np.isnan(SE)
        Bt = _t(np.where(miss, 0, B), device)
        SEt = _t(np.where(miss, SE_MISSING, SE), device)
        Rt = _t(np.eye(T) if R is None else R, device)
        gw = np.ones(G) if burden_genes is None else np.asarray(burden_genes, float)
        gwt = _t(gw, device)
    else:
        T = 1 if B is None else np.asarray(B).shape[1]

    # parameters; W starts positive and small, Θ at zero (programs first shaped by expression)
    U = torch.tensor(rng.standard_normal((D, K)), dtype=torch.float32, device=device, requires_grad=True)
    A = torch.tensor(0.1 * rng.standard_normal((C, K)), dtype=torch.float32, device=device, requires_grad=True)
    W = torch.tensor(rng.uniform(0, 0.2, (G, K)), dtype=torch.float32, device=device, requires_grad=True)
    Theta = torch.zeros((K, T), device=device, requires_grad=True)
    # residual variance starts at half the observed variance minus noise, floored
    obs_var = np.nanvar(np.where(obs, X, np.nan), axis=0) - np.nanmean(np.where(obs, V, np.nan), axis=0)
    s2_0 = np.clip(np.nan_to_num(0.5 * obs_var, nan=1e-2), 1e-4, None)
    log_s2 = torch.tensor(np.log(s2_0), dtype=torch.float32, device=device, requires_grad=True)
    log_psi = torch.full((T,), np.log(1e-3), device=device, requires_grad=True)
    alpha = torch.ones(K, device=device)
    lam = torch.ones(K, device=device)
    if init is not None:
        with torch.no_grad():
            for name, p in dict(U=U, A=A, W=W).items():
                if name in init:
                    p.copy_(_t(init[name], device))
            log_s2.copy_(torch.log(_t(init["s2"], device)))
            if use_burden and init["Theta"].shape == tuple(Theta.shape):
                Theta.copy_(_t(init["Theta"], device))
    named = dict(U=U, A=A, W=W, s2=log_s2, Theta=Theta, psi=log_psi)
    params = [p for n, p in named.items()
              if n not in freeze and (use_burden or n not in ("Theta", "psi"))]
    opt = torch.optim.Adam(params, lr=lr)

    def objective():
        M = torch.einsum("dk,ck,gk->dcg", U, A, W)
        var = Vt + torch.exp(log_s2)[None]
        ll_x = -0.5 * (mX * ((Xt - M) ** 2 / var + torch.log(var))).sum()
        lp = -0.5 * (U ** 2).sum() - 0.5 * (alpha * (A ** 2).sum(0)).sum() - (lam * W.sum(0)).sum()
        ll_b = torch.zeros((), device=device)
        if use_burden:
            ll_b = burden_loglik(Bt - W @ Theta, SEt, Rt, torch.exp(log_psi), gwt)
            lp = lp - Theta.abs().sum() / b_theta
        return ll_x + eta * ll_b + lp, ll_x, ll_b

    for step in range(n_steps):
        opt.zero_grad()
        obj, ll_x, ll_b = objective()
        (-obj / n_x).backward()
        opt.step()
        with torch.no_grad():
            W.clamp_(min=0)
            log_s2.clamp_(min=np.log(1e-6))
            log_psi.clamp_(min=np.log(1e-6), max=np.log(1.0))
            # ARD: closed-form MAP of the precisions given the current loadings
            if step >= ard_start and step % ard_every == 0:
                alpha = (A0 - 1 + C / 2) / (B0 + 0.5 * (A ** 2).sum(0))
                lam = (A0 - 1 + G) / (B0 + W.sum(0))
        if verbose and step % 500 == 0:
            print(f"step {step}  obj/n {obj.item() / n_x:.5f}  ll_x {ll_x.item():.1f}  ll_b {ll_b.item():.1f}")

    with torch.no_grad():
        obj, ll_x, ll_b = objective()
    out = {k: v.detach().cpu().numpy() for k, v in
           dict(U=U, A=A, W=W, Theta=Theta, s2=torch.exp(log_s2), psi=torch.exp(log_psi)).items()}
    out.update(objective=obj.item(), ll_x=ll_x.item(), ll_b=ll_b.item())
    out["active"] = active_programs(out)
    if not use_burden:
        out["Theta"][:] = 0
    return out


def active_programs(fit_out, rel=1e-3):
    """Programs kept after ARD: ‖A_k‖·‖W_k‖ above rel × the largest one."""
    size = np.linalg.norm(fit_out["A"], axis=0) * np.linalg.norm(fit_out["W"], axis=0)
    return size > rel * size.max() if size.max() > 0 else np.zeros_like(size, bool)


def fit_restarts(*args, n_restarts=3, seed=0, **kw):
    """Best (highest objective) of n_restarts fits with different seeds."""
    best = None
    for r in range(n_restarts):
        f = fit(*args, seed=seed + 1000 * r, **kw)
        if best is None or f["objective"] > best["objective"]:
            best = f
    return best


def predict_burden(fit_out):
    """Predicted burden effects W Θ (genes × traits)."""
    return fit_out["W"] @ fit_out["Theta"]
