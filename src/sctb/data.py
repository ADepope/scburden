"""Real-data inputs: the OneK1K donor × cell type × gene tensor and the split-half burden statistics.

Expression (SPEC.md §1): per cell type, from the sc_dosage_J moment files (p = pseudobulk share,
v = its sampling variance), donors with ≥ MIN_CELLS cells; shares divided by pool × sex factors
(normalize_strata, raking) to relative shares r with mean 1 per stratum; age regressed out per gene;
X = r − 1 (minus the age fit), V = v / factor². Entries of genes not expressed in a cell type
(mean share < 1e-6) are NaN.

Burden: plink2 --glm tables (gene symbol in ID, BETA, SE), one per trait and half. Summary statistics
only; nothing here touches individual-level data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["BASO", "BMD", "BMI", "CHOL", "CRET", "DBP", "EOSI", "FVC", "GLU", "HbA1c", "HDL", "HT",
          "LDL", "LYMPH", "MCH", "MCV", "RBC", "SBP"]
MIN_CELLS = 20
MIN_DONORS = 50


def normalize_strata(p, v, strata, n_iter=50, tol=1e-6):
    """Copy of scdosage.moments.normalize_strata (sc_dosage_J): divide each donor's shares by a
    product of one fitted factor per stratum column (main-effects multiplicative model, raking).
    Returns relative shares r (mean 1 per stratum), their variance v_r and the stratum dummies."""
    strata = np.asarray(strata)
    if strata.ndim == 1:
        strata = strata[:, None]
    codes, dummies = [], []
    for j in range(strata.shape[1]):
        levels, c = np.unique(strata[:, j], return_inverse=True)
        codes.append(c)
        dummies.append((c[:, None] == np.arange(1, len(levels))[None, :]).astype(float))
    fac = [np.ones((c.max() + 1, p.shape[1])) for c in codes]
    prev = None
    for _ in range(n_iter if len(codes) > 1 else 1):
        for j, c in enumerate(codes):
            other = np.ones_like(p)
            for k, ck in enumerate(codes):
                if k != j:
                    other *= fac[k][ck]
            num = np.zeros_like(fac[j])
            den = np.zeros_like(fac[j])
            np.add.at(num, c, p)
            np.add.at(den, c, other)
            fac[j] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        if prev is not None and np.allclose(fac[-1], prev, rtol=tol, atol=0):
            break
        prev = fac[-1].copy()
    mu = np.ones_like(p)
    for f, c in zip(fac, codes):
        mu *= f[c]
    ok = mu > 0
    r = np.where(ok, p / np.where(ok, mu, 1), 1.0)
    vr = np.where(ok, v / np.where(ok, mu, 1) ** 2, 0.0)
    return r, vr, np.column_stack(dummies)


def build_tensor(moments_dir, genes_keep=None, min_mean_any=1e-5, min_mean_cell=1e-6):
    """D × C × G tensor X and variance V (float32, NaN = missing) from the moment files.

    genes_keep  optional set of gene symbols (e.g. genes with a burden column); genes are further
                limited to those with mean share ≥ min_mean_any in at least one cell type.
    Returns dict(X, V, donors, cell_types, genes, n_cells).
    """
    md = Path(moments_dir)
    man = pd.read_csv(md / "manifest.tsv", sep="\t")
    names = pd.read_csv(md / "genes.tsv", sep="\t")["gene_name"].astype(str).values
    per_ct = {}
    for ct, fn in zip(man.cell_type, man.file):
        z = np.load(md / fn, allow_pickle=True)
        ok = z["n_cells"] >= MIN_CELLS
        if ok.sum() < MIN_DONORS:
            continue
        per_ct[ct] = {k: z[k][ok] for k in ("p", "v", "n_cells", "donor_id", "cov_pool_number", "cov_sex", "cov_age")}
    # gene selection: one column per symbol (first occurrence), expressed somewhere
    first = ~pd.Series(names).duplicated().values
    keep = first if genes_keep is None else first & np.isin(names, list(genes_keep))
    mean_ct = np.vstack([d["p"].mean(0) for d in per_ct.values()])
    keep &= (mean_ct >= min_mean_any).any(0)
    gidx = np.flatnonzero(keep)
    donors = np.unique(np.concatenate([d["donor_id"] for d in per_ct.values()]))
    D, C, G = len(donors), len(per_ct), len(gidx)
    X = np.full((D, C, G), np.nan, np.float32)
    V = np.full((D, C, G), np.nan, np.float32)
    n_cells = np.zeros((D, C))
    for c, (ct, d) in enumerate(per_ct.items()):
        p = d["p"][:, gidx].astype(float)
        v = d["v"][:, gidx].astype(float)
        strata = np.column_stack([d["cov_pool_number"].astype(str), d["cov_sex"].astype(str)])
        r, vr, _ = normalize_strata(p, v, strata)
        # age: linear, regressed out per gene (r has mean 1 within every stratum already)
        age = pd.to_numeric(pd.Series(d["cov_age"]), errors="coerce").values
        age = np.where(np.isnan(age), np.nanmean(age), age)
        Z = np.column_stack([np.ones_like(age), age - age.mean()])
        coef = np.linalg.lstsq(Z, r - 1, rcond=None)[0]
        x = r - 1 - Z[:, 1:] @ coef[1:]
        expressed = p.mean(0) >= min_mean_cell
        x[:, ~expressed] = np.nan
        vr[:, ~expressed] = np.nan
        rows = np.searchsorted(donors, d["donor_id"])
        X[rows, c] = x
        V[rows, c] = vr
        n_cells[rows, c] = d["n_cells"]
    return dict(X=X, V=V, donors=donors, cell_types=np.array(list(per_ct)), genes=names[gidx], n_cells=n_cells)


def load_burden_matrix(pattern, genes=None, traits=TRAITS):
    """G × T matrices of BETA and SE (NaN where missing) from one plink2 table per trait.
    pattern contains {trait}; genes fixes the row order (default: union over traits)."""
    tabs = {}
    for t in traits:
        b = pd.read_csv(pattern.format(trait=t), sep="\t", usecols=["ID", "BETA", "SE", "A1_FREQ", "OBS_CT"])
        tabs[t] = b.dropna(subset=["BETA", "SE"]).drop_duplicates("ID").set_index("ID")
    if genes is None:
        genes = sorted(set().union(*(set(b.index) for b in tabs.values())))
    B = np.column_stack([tabs[t].BETA.reindex(genes).values for t in traits]).astype(np.float32)
    SE = np.column_stack([tabs[t].SE.reindex(genes).values for t in traits]).astype(np.float32)
    mac = np.nanmedian(np.column_stack([(2 * tabs[t].A1_FREQ * tabs[t].OBS_CT).reindex(genes).values
                                        for t in traits]), 1)
    return dict(B=B, SE=SE, genes=np.asarray(genes), traits=np.asarray(traits), mac=mac)


def error_correlation(B, SE, z_max=4.0):
    """Cross-trait correlation of the burden estimation errors: Pearson correlation of z = β̂/SE over
    genes with max_t |z| < z_max (drops genes with clear signals; < 1% of gene–trait pairs are hits,
    so the truncation bias is small). Missing values are dropped pairwise."""
    z = B / SE
    null = np.nanmax(np.abs(z), 1) < z_max
    R = pd.DataFrame(z[null]).corr(min_periods=100).values
    np.fill_diagonal(R, 1.0)
    # nearest positive definite (pairwise deletion can break it)
    w, Q = np.linalg.eigh(R)
    R = Q @ np.diag(np.clip(w, 1e-3, None)) @ Q.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def loeuf(path):
    """gnomAD v4.0 LOEUF (lof.oe_ci.upper) per gene symbol, MANE Select transcripts."""
    g = pd.read_csv(path, sep="\t", usecols=["gene", "mane_select", "lof.oe_ci.upper"])
    g = g[g.mane_select.astype(str).str.lower() == "true"].drop_duplicates("gene")
    return g.set_index("gene")["lof.oe_ci.upper"]
