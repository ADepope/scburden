"""Tables of the simulation benchmark: mean ± SE over replicates per scenario and method, and the
paired difference coupled_refit − two_step. Reads results/sim/metrics.tsv, prints Markdown.
Run: python -I scripts/summarise_sim.py > results/sim/tables.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
f = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/sim/metrics.tsv"
d = pd.read_csv(f, sep="\t", keep_default_na=False, na_values=[""])  # 'null' is a method name
ORDER = ["coupled_refit", "coupled_joint", "two_step", "scitd", "random", "null"]
COLS = ["R2_B", "gainx1e4_B", "r2_true", "prog_cos", "link_auc", "celltype_cor", "eta"]


def fmt(x):
    x = x.dropna()
    if len(x) == 0:
        return "–"
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return f"{x.mean():.3f} ± {se:.3f}" if np.isfinite(se) else f"{x.mean():.3f}"


for scen, ds in d.groupby("scenario"):
    print(f"\n### {scen} ({ds.rep.nunique()} replicates)\n")
    cols = [c for c in COLS if c in ds and ds[c].notna().any()]
    print("| method | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for m in ORDER:
        dm = ds[ds.method == m]
        if len(dm):
            print(f"| {m} | " + " | ".join(fmt(dm[c]) for c in cols) + " |")
    a = ds[ds.method == "coupled_refit"].set_index("rep")
    b = ds[ds.method == "two_step"].set_index("rep")
    for c in ("R2_B", "gainx1e4_B", "prog_cos"):
        diff = (a[c] - b[c]).dropna()
        if len(diff):
            print(f"\npaired coupled_refit − two_step, {c}: {fmt(diff)} (better in {int((diff > 0).sum())}/{len(diff)})")
