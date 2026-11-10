"""Sensitivity analysis: effect of removing students who unregistered before day T (28-day window).

Main pipeline REMOVES them (their outcome is already known). Wu et al. (2026) do not state how they were handled and
their withdrawal rate (26-32%) matches the full dataset, suggesting they were kept. This script shows both versions with a
fixed-setting HistGradientBoosting model (not tuned; trained on 2013B/2013J/2014B, tested on 2014J). Not part of the frozen pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from shapalert import features as F
from shapalert.common import CFG, ROOT, SEED, topk_metrics, write_table

D = ROOT / CFG["paths"]["oulad_dir"]
KEY = ["code_module", "code_presentation", "id_student"]
T, PRE = 28, CFG["pre_start_days"]

info = pd.read_csv(D / "studentInfo.csv")
reg = pd.read_csv(D / "studentRegistration.csv")
b = info.merge(reg, on=KEY)
b["y"] = (b.final_result == "Withdrawn").astype(int)
b["uid"] = np.arange(len(b))
b["course"] = b.code_module + "_" + b.code_presentation
chunks = [c[(c.date < T) & (c.date >= -PRE)] for c in pd.read_csv(
    D / "studentVle.csv", chunksize=2_000_000, dtype={"id_student": "int64", "date": "int16", "sum_click": "int32"})]
ev = pd.concat(chunks).merge(b[KEY + ["uid"]], on=KEY).rename(
    columns={"date": "day", "sum_click": "clicks", "id_site": "res"})
ev["act"] = np.nan
sa = (pd.read_csv(D / "studentAssessment.csv")
      .merge(pd.read_csv(D / "assessments.csv")[["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
      .merge(b[KEY + ["uid"]], on=KEY).rename(columns={"date_submitted": "day"})[["uid", "day", "score"]]
      .dropna(subset=["day"]))


def run(mask, name):
    k = b[mask].copy()
    t = (k.set_index("uid").join(F.behaviour_features(ev[["uid", "day", "clicks", "res", "act"]], k.uid.values, T, PRE))
         .join(F.assessment_features(sa, k.uid.values, T)))
    t = F.add_within_course_z(t, "course", F.Z_MAP)
    tr = t.code_presentation.isin(["2013B", "2013J", "2014B"])
    te = t.code_presentation == "2014J"
    m = HistGradientBoostingClassifier(class_weight="balanced", max_depth=4, max_iter=300, learning_rate=0.05,
                                       random_state=SEED).fit(t.loc[tr, F.SHARED_FEATURES], t.loc[tr, "y"])
    p = m.predict_proba(t.loc[te, F.SHARED_FEATURES])[:, 1]
    y = t.loc[te, "y"].values
    row = {"handling_of_early_unregistrations": name, "n_test": int(te.sum()), "withdrawal_rate_test": round(y.mean(), 3),
           "roc_auc": round(roc_auc_score(y, p), 3), "pr_auc": round(average_precision_score(y, p), 3)}
    for c in CFG["alerts"]["capacities"]:
        tk = topk_metrics(y, p, c)
        row[f"precision@{int(c*100)}"], row[f"recall@{int(c*100)}"] = round(tk["precision"], 3), round(tk["recall"], 3)
    return row


rows = [run(b.date_unregistration.isna() | (b.date_unregistration >= T), "Removed if unregistered before day 28 (main pipeline)"),
        run(b.uid >= 0, "All registrations kept (comparable to Wu et al. 2026)")]
tab = pd.DataFrame(rows)
write_table(tab, "oulad_wu_comparable_sensitivity")
print(tab.to_string(index=False))
