"""Dataset-agnostic behavioural feature builder (same recipe for OULAD and Ghana).

Event table schema (one row per student-day-resource aggregate):
    uid   : int/str unique id of one student-in-one-course (a 'registration')
    day   : int days relative to course start (negative = before start)
    clicks: int
    res   : resource id (optional, NaN where unavailable)
    act   : normalised activity type (optional): forum | quiz | resource | homepage | other

LEAKAGE RULE: only events with day < T are ever used. Assessment records are used
only if submitted/graded before T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Scale-free features that exist in BOTH datasets (used for transfer experiments A, C, D).
SHARED_FEATURES = [
    "clicks_total_z", "active_days_z", "clicks_per_active_day_z", "clicks_last_week_z",
    "click_trend_z", "active_ratio", "gap_frac", "first_click_frac",
    "n_early_assess_z", "early_score_mean", "has_early_assess",
]
# Raw (OULAD 'full' set) behavioural features. Resource/type features exist only in OULAD.
RAW_BEHAVIOUR = ["clicks_total", "active_days", "distinct_resources", "clicks_forum", "clicks_quiz",
                 "clicks_resource", "clicks_homepage", "clicks_last_week", "click_trend",
                 "days_since_last_click", "first_click_day", "n_early_assess", "early_score_mean"]
STATIC = ["gender_m", "age_band_ord", "edu_ord", "disability_y", "num_of_prev_attempts",
          "studied_credits", "imd_ord"]
FULL_FEATURES = RAW_BEHAVIOUR + ["clicks_per_active_day", "active_ratio", "gap_frac"] + STATIC

# Features an advisor can influence (prescriptivity in LEAF). Everything else is fixed.
FIXED_FEATURES = set(STATIC)


def behaviour_features(events: pd.DataFrame, uids, T: int, pre_start: int = 14) -> pd.DataFrame:
    """Aggregate events with -pre_start <= day < T into one row per uid (zero rows kept)."""
    ev = events[(events["day"] < T) & (events["day"] >= -pre_start)]
    assert (ev["day"] < T).all(), "leak: event at/after T"
    g = ev.groupby("uid")
    feat = pd.DataFrame(index=pd.Index(uids, name="uid"))
    feat["clicks_total"] = g["clicks"].sum()
    days = ev[["uid", "day"]].drop_duplicates()
    feat["active_days"] = days.groupby("uid")["day"].size()
    if "res" in ev and ev["res"].notna().any():
        feat["distinct_resources"] = ev.dropna(subset=["res"]).groupby("uid")["res"].nunique()
    last_wk = ev[ev["day"] >= T - 7].groupby("uid")["clicks"].sum()
    first_wk = ev[(ev["day"] >= 0) & (ev["day"] < 7)].groupby("uid")["clicks"].sum()
    feat["clicks_last_week"] = last_wk
    feat["_first_week"] = first_wk
    if "act" in ev and ev["act"].notna().any():
        by_type = ev.pivot_table(index="uid", columns="act", values="clicks", aggfunc="sum")
        for a in ["forum", "quiz", "resource", "homepage"]:
            feat[f"clicks_{a}"] = by_type[a] if a in by_type else np.nan
    first_day = g["day"].min()
    last_day = g["day"].max()
    count_cols = [c for c in feat.columns if c not in ("_first_week",)]
    for c in count_cols:
        if c.startswith("clicks") or c in ("active_days", "distinct_resources"):
            if feat[c].notna().any():
                feat[c] = feat[c].fillna(0)
    feat["_first_week"] = feat["_first_week"].fillna(0)
    feat["click_trend"] = feat["clicks_last_week"] - feat["_first_week"]
    horizon = T + pre_start
    feat["first_click_day"] = first_day.reindex(feat.index).fillna(T)  # never active -> T
    feat["days_since_last_click"] = ((T - 1) - last_day.reindex(feat.index)).fillna(horizon)
    feat["clicks_per_active_day"] = np.where(feat["active_days"] > 0, feat["clicks_total"] / feat["active_days"].clip(lower=1), 0.0)
    feat["active_ratio"] = feat["active_days"] / horizon
    feat["gap_frac"] = feat["days_since_last_click"] / horizon
    feat["first_click_frac"] = (feat["first_click_day"].clip(lower=-pre_start, upper=T) + pre_start) / horizon
    return feat.drop(columns="_first_week")


def assessment_features(subs: pd.DataFrame, uids, T: int) -> pd.DataFrame:
    """subs columns: uid, day (submission/graded day), score (0-100). Only day < T is used."""
    s = subs[subs["day"] < T]
    assert (s["day"] < T).all()
    g = s.groupby("uid")
    out = pd.DataFrame(index=pd.Index(uids, name="uid"))
    out["n_early_assess"] = g.size()
    out["early_score_mean"] = g["score"].mean()
    out["n_early_assess"] = out["n_early_assess"].fillna(0)
    out["has_early_assess"] = (out["n_early_assess"] > 0).astype(int)
    return out  # early_score_mean stays NaN when no submission; models impute / handle NaN


def add_within_course_z(df: pd.DataFrame, group_col: str, cols: dict) -> pd.DataFrame:
    """Add z-scored copies of count features, standardised inside each course (label-free)."""
    df = df.copy()
    for src, dst in cols.items():
        grp = df.groupby(group_col)[src]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0, np.nan)
        df[dst] = ((df[src] - mu) / sd).fillna(0.0)
    return df


Z_MAP = {"clicks_total": "clicks_total_z", "active_days": "active_days_z",
         "clicks_per_active_day": "clicks_per_active_day_z", "clicks_last_week": "clicks_last_week_z",
         "click_trend": "click_trend_z", "n_early_assess": "n_early_assess_z"}
