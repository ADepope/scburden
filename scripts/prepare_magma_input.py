"""SNP p-value file for MAGMA from one GWAS summary-statistics file (SPEC_v2.md §1, gene scores).

Reads row TASK (1-based) of scripts/gwas_traits.tsv and writes data/gwas/magma_in/<COHORT>_<TRAIT>.tsv
with columns SNP, P, N.
- SNP: hm_rsid / rsid / RSID, else variant_id when it is an rsID (SLE rows that failed harmonisation).
  The unharmonised European asthma file mixes rsIDs and chr:pos_alleles IDs: all its SNPs are matched
  to the EUR reference .bim by chromosome, GRCh37 position and the unordered allele pair.
- P: p_value / P; p == 0 is set to 1e-300, missing or out-of-range p is dropped.
- N: the per-SNP column (n, N, cum_eff_sample_size) where there is one, else n_total from the table.
SNPs seen twice keep the smallest p. Public summary statistics only.
Run: python -I scripts/prepare_magma_input.py TASK
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GWAS = ROOT / "data/gwas"
ID_COLS = ["hm_rsid", "rsid", "RSID"]
P_COLS = ["p_value", "P"]
N_COLS = ["n", "N", "cum_eff_sample_size"]


def first(cols, names):
    return next((c for c in names if c in cols), None)


def read(path):
    head = pd.read_csv(path, sep="\t", nrows=0).columns
    pos = ["CHR", "BP", "EA", "NEA"] if "SNP" in head and first(head, ID_COLS) is None else []
    vid = "variant_id" if "variant_id" in head else None
    use = [c for c in [first(head, ID_COLS), vid, first(head, P_COLS), first(head, N_COLS)] + pos if c]
    num = {first(head, P_COLS), first(head, N_COLS)} - {None}
    df = pd.read_csv(path, sep="\t", usecols=use, na_values=["NA", ""],
                     dtype={c: (float if c in num else str) for c in use})
    return df, first(head, ID_COLS), first(head, P_COLS), first(head, N_COLS), bool(pos)


def by_position(df):
    """rsID from the EUR reference .bim, matching chromosome, position and allele pair."""
    bim = pd.read_csv(GWAS / "ref_eur/g1000_eur.bim", sep="\t", header=None, usecols=[0, 1, 3, 4, 5],
                      names=["CHR", "SNP", "BP", "A1", "A2"], dtype=str)
    key = lambda c, p, a, b: c + ":" + p + ":" + np.minimum(a, b) + ":" + np.maximum(a, b)
    bim.index = key(bim.CHR, bim.BP, bim.A1.values.astype(object), bim.A2.values.astype(object))
    bim = bim[~bim.index.duplicated(keep=False)]
    k = key(df.CHR, df.BP, df.EA.values.astype(object), df.NEA.values.astype(object))
    return pd.Series(bim.SNP.reindex(k).values, index=df.index)


def main():
    task = int(sys.argv[1])
    row = pd.read_csv(ROOT / "scripts/gwas_traits.tsv", sep="\t").iloc[task - 1]
    src = GWAS / row.cohort / f"{row.trait}_{row.gcst}" / row.file
    df, idc, pc, nc, pos = read(src)
    n_in = len(df)
    if pos:
        snp = by_position(df)
    else:
        snp = df[idc] if idc else pd.Series(np.nan, index=df.index)
        if "variant_id" in df:
            v = df["variant_id"].where(df["variant_id"].str.startswith("rs", na=False))
            snp = snp.where(snp.str.startswith("rs", na=False), v)
    p = df[pc]
    n = df[nc] if nc else pd.Series(float(row.n_total), index=df.index)
    out = pd.DataFrame({"SNP": snp, "P": p.mask(p == 0, 1e-300), "N": n.round()})
    out = out[out.SNP.str.startswith("rs", na=False) & out.P.between(0, 1) & (out.N > 0)]
    out = out.sort_values("P").drop_duplicates("SNP").sort_index()
    dest = GWAS / "magma_in" / f"{row.cohort}_{row.trait}.tsv"
    dest.parent.mkdir(exist_ok=True)
    out.to_csv(dest, sep="\t", index=False, float_format="%.6g")
    print(f"{row.cohort} {row.trait}: {n_in} rows -> {len(out)} SNPs (id={idc or 'position'}, p={pc}, "
          f"N={nc or row.n_total}) -> {dest}")


if __name__ == "__main__":
    main()
