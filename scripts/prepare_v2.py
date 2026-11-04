"""Gene-score matrix for version 2 (SPEC_v2.md §1-3) from the MAGMA outputs of scripts/magma_genes.sh.

Genes: the tensor genes (data/real/tensor.npz, symbols) mapped to Entrez IDs via Ensembl: symbol ->
Ensembl ID from the OneK1K gene table (first occurrence, as in data.build_tensor) -> Entrez ID from
HGNC. Genes overlapping the MHC (chr6:25-34 Mb, NCBI37.3) and genes without a MAGMA score for any
discovery trait are dropped.
Scores: MAGMA ZSTAT, capped at z = 6.1094 (p = 5e-10, MAGMA's floor for simulated p-values;
SPEC_v2 Deviations). Covariates per gene and trait: log gene length, log NSNPS, log NPARAM; they are
regressed out later, inside each fold. R: correlation of the capped z over genes with max_t z < 2,
per cohort. Folds: 5 chromosome groups, greedy by gene count (largest chromosome first).
ct_mean: mean pseudobulk share of each gene in each tensor cell type (for baseline B5).
Writes data/real/gwas_v2.npz:
  Z, COV (genes x 17 discovery traits [x 3]), Z_rep, COV_rep (genes x 12 Biobank Japan traits [x 3]),
  R, R_rep, traits, traits_rep, rep_idx (column of each replication trait in Z), immune (per
  discovery trait), genes, entrez, chrom, start, stop, tensor_idx (column in tensor.npz), fold,
  ct_mean (genes x cell types).
Public summary statistics only. Run: python -I scripts/prepare_v2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sctb import data  # noqa: E402

BASE = ROOT.parent
G = ROOT / "data/gwas"
MOMENTS = BASE / "sc_dosage_J/results/moments/onek1k"
ONEK1K_GENES = MOMENTS / "genes.tsv"
HGNC = BASE / "sc_dosage_J/data/raw/reference/hgnc_complete_set.txt"
ZCAP = norm.isf(5e-10)
CONTROLS = {"HEIGHT", "LDL", "MCV"}
MHC = ("6", 25e6, 34e6)


def magma(cohort, trait):
    d = pd.read_csv(G / "magma_out" / f"{cohort}_{trait}.genes.out", sep=r"\s+").set_index("GENE")
    cov = np.column_stack([np.log(d.STOP - d.START + 1), np.log(d.NSNPS), np.log(d.NPARAM)])
    return d.ZSTAT.clip(upper=ZCAP), pd.DataFrame(cov, index=d.index)


def null_corr(Z):
    ok = np.isfinite(Z).all(1) & (np.nan_to_num(Z, nan=np.inf).max(1) < 2)
    return np.corrcoef(Z[ok].T), int(ok.sum())


def chrom_folds(chrom, k=5):
    counts = pd.Series(chrom).value_counts()
    load, group = np.zeros(k), {}
    for c, n in counts.items():
        g = int(np.argmin(load))
        group[c], load[g] = g, load[g] + n
    return np.array([group[c] for c in chrom]), load


def main():
    t = np.load(ROOT / "data/real/tensor.npz", allow_pickle=True)
    sym = pd.Series(t["genes"].astype(str))
    ok1 = pd.read_csv(ONEK1K_GENES, sep="\t").drop_duplicates("gene_name").set_index("gene_name").gene_id
    hg = pd.read_csv(HGNC, sep="\t", usecols=["ensembl_gene_id", "entrez_id"], dtype=str).dropna()
    hg = hg.drop_duplicates("ensembl_gene_id", keep=False).set_index("ensembl_gene_id").entrez_id
    entrez = pd.to_numeric(hg.reindex(ok1.reindex(sym).values).values, errors="coerce")
    loc = pd.read_csv(G / "magma/NCBI37.3.gene.loc", sep="\t", header=None,
                      names=["entrez", "chrom", "start", "stop", "strand", "symbol"]).set_index("entrez")
    print(f"tensor genes {len(sym)}; with Ensembl ID {sym.isin(ok1.index).sum()}; with Entrez ID "
          f"{np.isfinite(entrez).sum()}; in NCBI37.3 {pd.Index(entrez).isin(loc.index).sum()}")

    tab = pd.read_csv(ROOT / "scripts/gwas_traits.tsv", sep="\t")
    disc, rep = tab[tab.cohort == "EUR"].trait.tolist(), tab[tab.cohort == "BBJ"].trait.tolist()
    zd, cd = zip(*(magma("EUR", tr) for tr in disc))
    zr, cr = zip(*(magma("BBJ", tr) for tr in rep))
    idx = pd.Index(entrez)
    Z = np.column_stack([z.reindex(idx).values for z in zd])
    keep = np.isfinite(entrez) & idx.isin(loc.index) & np.isfinite(Z).any(1)
    L = loc.reindex(idx)
    mhc = (L.chrom.astype(str).values == MHC[0]) & (L.stop.values > MHC[1]) & (L.start.values < MHC[2])
    print(f"no MAGMA score: {(np.isfinite(entrez) & idx.isin(loc.index) & ~np.isfinite(Z).any(1)).sum()}; "
          f"MHC: {(keep & mhc).sum()}")
    keep &= ~mhc
    ti = np.flatnonzero(keep)
    idx, L = idx[ti], L.iloc[ti]
    stack = lambda zs: np.column_stack([z.reindex(idx).values for z in zs]).astype(np.float32)
    cstack = lambda cs: np.stack([c.reindex(idx).values for c in cs], 1).astype(np.float32)
    Z, Zr, C, Cr = stack(zd), stack(zr), cstack(cd), cstack(cr)
    chrom = L.chrom.astype(str).values
    fold, load = chrom_folds(chrom)
    R, n0 = null_corr(Z)
    Rr, n0r = null_corr(Zr)
    np.savez(ROOT / "data/real/gwas_v2.npz", Z=Z, COV=C, Z_rep=Zr, COV_rep=Cr, R=R, R_rep=Rr,
             traits=np.array(disc), traits_rep=np.array(rep), rep_idx=np.array([disc.index(r) for r in rep]),
             immune=np.array([tr not in CONTROLS for tr in disc]), genes=sym.values[ti],
             entrez=idx.values.astype(int), chrom=chrom, start=L.start.values, stop=L.stop.values,
             tensor_idx=ti, fold=fold, ct_mean=data.celltype_means(MOMENTS, t["cell_types"], sym.values[ti]))

    print(f"genes kept {len(ti)}; discovery missing {np.isnan(Z).mean():.2%}, replication missing "
          f"{np.isnan(Zr).mean():.2%}; capped at {ZCAP:.4f}: {(Z >= ZCAP - 1e-6).sum()} discovery, "
          f"{(Zr >= ZCAP - 1e-6).sum()} replication gene-trait pairs")
    print(f"R from {n0} discovery / {n0r} replication genes with max z < 2; "
          f"largest |off-diagonal| {np.abs(R - np.eye(len(R))).max():.2f} / {np.abs(Rr - np.eye(len(Rr))).max():.2f}")
    for g in range(5):
        print(f"fold {g}: {int(load[g])} genes, chromosomes {sorted(set(chrom[fold == g]), key=lambda c: (len(c), c))}")


if __name__ == "__main__":
    main()
