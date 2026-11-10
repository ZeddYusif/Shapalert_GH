"""Post-hoc sensitivity analysis: do the OULAD fairness gaps persist when demographic attributes are not predictors?

Variants (28 days, time-aware split, top-20% alerting, 1,000 bootstrap resamples, as in scripts/04):
  main            : frozen full-feature HGB (uses gender, age band, disability, IMD, education, prior attempts, credits)
  no_demographics : same frozen HGB hyper-parameters, but gender, age band, disability and IMD band removed
  shared_features : frozen shared-feature HGB (behaviour only; no registration attributes at all)
This is not part of the frozen pipeline; it only retrains with fewer predictors, using the frozen settings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from shapalert import features as F
from shapalert import models as M
from shapalert.common import CFG, SEED, load_json, out_dir, processed_dir, seed_everything, topk_metrics, write_table

seed_everything()
T = CFG["primary_window"]
fz = load_json(out_dir("frozen") / "frozen_pipeline.json")
ts = CFG["oulad"]["time_split"]
n_boot, cap = CFG["fairness"]["n_boot"], CFG["fairness"]["capacity"]
DEMO = ["gender_m", "age_band_ord", "disability_y", "imd_ord"]

df = pd.read_csv(processed_dir() / f"oulad_w{T}.csv")
tr = df[df.code_presentation.isin(ts["train"])]
te = df[df.code_presentation.isin(ts["test"])].reset_index(drop=True)
spec = fz["best"][f"full|{T}"]
name, params = spec["model"], spec["params"]
print("frozen full-feature model family:", name)

scores = {}
scores["main"] = joblib.load(out_dir("models") / f"time_full_w{T}_{name}.joblib").predict_proba(te[F.FULL_FEATURES])[:, 1]
feats_nd = [f for f in F.FULL_FEATURES if f not in DEMO]
m_nd = M.make_gbm(name, params, tr["y"].values).fit(tr[feats_nd], tr["y"].values)
scores["no_demographics"] = m_nd.predict_proba(te[feats_nd])[:, 1]
sname = fz["best"][f"shared|{T}"]["model"]
scores["shared_features"] = joblib.load(out_dir("models") / f"time_shared_w{T}_{sname}.joblib").predict_proba(te[F.SHARED_FEATURES])[:, 1]

rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(te), len(te)) for _ in range(n_boot)]


def rates(d):
    pos, neg = d[d.y == 1], d[d.y == 0]
    return (pos.flag.mean() if len(pos) else np.nan, neg.flag.mean() if len(neg) else np.nan)


perf, rows = [], []
for variant, s in scores.items():
    y = te["y"].values
    tk = topk_metrics(y, s, cap)
    perf.append({"variant": variant, "n_features": {"main": len(F.FULL_FEATURES), "no_demographics": len(feats_nd),
                                                    "shared_features": len(F.SHARED_FEATURES)}[variant],
                 "roc_auc": round(roc_auc_score(y, s), 3), "pr_auc": round(average_precision_score(y, s), 3),
                 "precision@20": round(tk["precision"], 3), "recall@20": round(tk["recall"], 3)})
    d = te[["y", "gender", "disability", "age_band"]].copy()
    thr = np.sort(s)[::-1][max(1, int(round(cap * len(d)))) - 1]
    d["flag"] = (s >= thr).astype(int)
    for attr in CFG["fairness"]["groups"]:
        ref = d[attr].value_counts().idxmax()
        levels = list(d[attr].dropna().unique())
        boots = {l: np.array([rates(d.iloc[ib][d.iloc[ib][attr] == l]) for ib in boot_idx]) for l in levels}
        r_ref = rates(d[d[attr] == ref])
        for l in levels:
            g = d[d[attr] == l]
            rec, fpr = rates(g)
            row = {"variant": variant, "attribute": attr, "group": l, "reference": l == ref, "n": len(g),
                   "n_dropouts": int(g.y.sum()), "recall": round(rec, 3), "fpr": round(fpr, 3)}
            if len(g) >= 30 and g.y.sum() >= 10:
                for j, nm in enumerate(["recall", "fpr"]):
                    gap = boots[l][:, j] - boots[ref][:, j]
                    lo, hi = np.nanquantile(gap, [0.025, 0.975])
                    row[f"{nm}_gap"] = round((rec, fpr)[j] - r_ref[j], 3)
                    row[f"{nm}_gap_lo"], row[f"{nm}_gap_hi"] = round(lo, 3), round(hi, 3)
                    row[f"{nm}_gap_excludes_0"] = bool(lo > 0 or hi < 0)
            rows.append(row)

write_table(pd.DataFrame(perf), "oulad_fairness_sensitivity_performance")
fair = pd.DataFrame(rows)
write_table(fair, "oulad_fairness_sensitivity_gaps")
print(pd.DataFrame(perf).to_string(index=False))
view = fair[~fair.reference & fair.recall_gap.notna()][["variant", "attribute", "group", "recall", "fpr", "recall_gap", "recall_gap_lo",
                                                         "recall_gap_hi", "fpr_gap", "fpr_gap_lo", "fpr_gap_hi"]]
print(view.to_string(index=False))
