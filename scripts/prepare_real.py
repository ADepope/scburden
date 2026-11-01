"""Build the real-data inputs (SPEC.md §1) into data/real/:

  tensor.npz   X, V (donors × cell types × genes, NaN = missing), donors, cell_types, genes, n_cells
  burden.npz   B_A, SE_A, B_B, SE_B (genes × traits, half 1 = A, half 2 = B), R_A, R_B, mac, loeuf,
               genes, traits

Genes: HC-LoF burden genes (IDs of the half-1 tables) that are expressed in OneK1K. Needs the
split-half statistics from scripts/burden_splithalf.sh; with --expression-only only tensor.npz is
written (gene list from the existing full-sample tables).
Run: python -I scripts/prepare_real.py [--expression-only]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sctb import data  # noqa: E402

BASE = Path("/nfs/scistore24/robingrp/adepope")
MOMENTS = BASE / "sc_dosage_J/results/moments/onek1k"
HALVES = ROOT / "data/burden_halves/half{h}/{{trait}}_lof.glm.linear"
FULL = str(BASE / "rare_lof_AB/A/runs/{trait}/marginal/{trait}_lof.glm.linear")
GNOMAD = BASE / "sc_dosage_J/data/raw/reference/gnomad.v4.0.constraint_metrics.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expression-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/real"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pat_a, pat_b = str(HALVES).format(h=1), str(HALVES).format(h=2)
    gene_src = FULL if a.expression_only else pat_a
    burden_genes = data.load_burden_matrix(gene_src, traits=["LDL"])["genes"]
    t = data.build_tensor(MOMENTS, genes_keep=set(burden_genes))
    np.savez(out / "tensor.npz", **t)
    obs = ~np.isnan(t["X"])
    print(f"tensor: {t['X'].shape} (donors × cell types × genes), observed {obs.mean():.1%}")
    print("cell types:", list(t["cell_types"]))
    if a.expression_only:
        return

    A = data.load_burden_matrix(pat_a, genes=t["genes"])
    B = data.load_burden_matrix(pat_b, genes=t["genes"])
    lo = data.loeuf(GNOMAD).reindex(t["genes"]).values
    np.savez(out / "burden.npz", B_A=A["B"], SE_A=A["SE"], B_B=B["B"], SE_B=B["SE"],
             R_A=data.error_correlation(A["B"], A["SE"]), R_B=data.error_correlation(B["B"], B["SE"]),
             mac=A["mac"], loeuf=lo, genes=t["genes"], traits=A["traits"])
    z = A["B"] / A["SE"]
    print(f"burden: {A['B'].shape}, half-A |z|>4.5: {np.nanmean(np.abs(z) > 4.5):.4f}, "
          f"LOEUF available for {np.mean(~np.isnan(lo)):.1%} of genes")


if __name__ == "__main__":
    main()
