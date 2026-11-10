"""Phase 1, step 1.9: fairness audit (recall + FPR by gender, disability, age band) with bootstrap CIs.

Uses the frozen best model at the primary window on the time-aware test set (2014J).
Alerting rule: top `fairness.capacity` share of ALL students by risk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from shapalert.common import CFG, SEED, load_json, out_dir, write_table

fz = load_json(out_dir("frozen") / "frozen_pipeline.json")
T = CFG["primary_window"]
model = fz["best"][f"full|{T}"]["model"]
df = pd.read_csv(out_dir("preds") / f"oulad_time_w{T}_{model}.csv")
cap, n_boot = CFG["fairness"]["capacity"], CFG["fairness"]["n_boot"]
thr = np.sort(df["raw"].values)[::-1][max(1, int(round(cap * len(df)))) - 1]
df["flag"] = (df["raw"] >= thr).astype(int)
rng = np.random.default_rng(SEED)


def rates(d):
    pos, neg = d[d.y == 1], d[d.y == 0]
    rec = pos.flag.mean() if len(pos) else np.nan
    fpr = neg.flag.mean() if len(neg) else np.nan
    return rec, fpr


idx_boot = [rng.integers(0, len(df), len(df)) for _ in range(n_boot)]
rows = []
for gcol in CFG["fairness"]["groups"]:
    levels = [l for l in df[gcol].dropna().unique()]
    ref = df[gcol].value_counts().idxmax()
    boots = {l: [] for l in levels}
    for ib in idx_boot:
        b = df.iloc[ib]
        for l in levels:
            boots[l].append(rates(b[b[gcol] == l]))
    boots = {l: np.array(v) for l, v in boots.items()}
    for l in levels:
        d = df[df[gcol] == l]
        rec, fpr = rates(d)
        small = len(d) < 30 or d.y.sum() < 10
        row = {"attribute": gcol, "group": l, "reference": l == ref, "n": len(d), "n_dropouts": int(d.y.sum()),
               "recall": rec, "fpr": fpr}
        if small:
            row["note"] = "group too small to audit"
        else:
            for j, nm in enumerate(["recall", "fpr"]):
                lo, hi = np.nanquantile(boots[l][:, j], [0.025, 0.975])
                row[f"{nm}_lo"], row[f"{nm}_hi"] = lo, hi
                gap = boots[l][:, j] - boots[ref][:, j]
                row[f"{nm}_gap_vs_ref"] = (rates(d)[j] - rates(df[df[gcol] == ref])[j])
                g_lo, g_hi = np.nanquantile(gap, [0.025, 0.975])
                row[f"{nm}_gap_lo"], row[f"{nm}_gap_hi"] = g_lo, g_hi
                row[f"{nm}_gap_ci_excludes_0"] = bool(g_lo > 0 or g_hi < 0)
        rows.append(row)
tab = pd.DataFrame(rows).round(4)
write_table(tab, "table6_oulad_fairness")
print(f"Fairness audit: model={model}, window={T}, top {int(cap*100)}% alerts")
print(tab[["attribute", "group", "n", "n_dropouts", "recall", "fpr"]].to_string(index=False))
