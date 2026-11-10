"""Leakage and split-integrity tests. Run: python -m pytest tests -q   (after scripts 01 and 05)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from shapalert import features as F
from shapalert.common import CFG, processed_dir


def synthetic_events():
    return pd.DataFrame({"uid": [1, 1, 1, 2], "day": [0, 5, 40, 3], "clicks": [2, 3, 100, 1],
                         "res": [1, 2, 3, 1], "act": ["forum", "quiz", "resource", "forum"]})


def test_events_after_T_are_ignored():
    ev = synthetic_events()
    a = F.behaviour_features(ev, [1, 2], 28)
    b = F.behaviour_features(pd.concat([ev, ev.assign(day=60, clicks=999)]), [1, 2], 28)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[1, "clicks_total"] == 5          # the day-40 click must not count


def test_assessments_after_T_are_ignored():
    s = pd.DataFrame({"uid": [1, 1], "day": [10, 50], "score": [80, 10]})
    f = F.assessment_features(s, [1], 28)
    assert f.loc[1, "n_early_assess"] == 1 and f.loc[1, "early_score_mean"] == 80


def test_zero_activity_students_are_kept():
    f = F.behaviour_features(synthetic_events(), [1, 2, 3], 28)
    assert len(f) == 3 and f.loc[3, "clicks_total"] == 0 and f.loc[3, "days_since_last_click"] >= 28


@pytest.mark.skipif(not (processed_dir() / "oulad_w28.csv").exists(), reason="run scripts/01 first")
def test_oulad_unregistered_before_T_removed():
    D = Path(__file__).resolve().parents[1] / CFG["paths"]["oulad_dir"]
    reg = pd.read_csv(D / "studentRegistration.csv")
    for T in CFG["windows"]:
        tab = pd.read_csv(processed_dir() / f"oulad_w{T}.csv")
        m = tab.merge(reg, on=["code_module", "code_presentation", "id_student"])
        assert not (m["date_unregistration"] < T).any()


@pytest.mark.skipif(not (processed_dir() / "oulad_w28.csv").exists(), reason="run scripts/01 first")
def test_time_split_is_disjoint_and_ordered():
    ts = CFG["oulad"]["time_split"]
    assert not set(ts["train"]) & set(ts["test"]) and not set(ts["val"]) & set(ts["test"])
    assert max(ts["train"]) < min(ts["val"]) < min(ts["test"])


@pytest.mark.skipif(not (processed_dir() / "ghana_w28.csv").exists(), reason="run scripts/05 first")
def test_ghana_features_do_not_contain_label_columns():
    g = pd.read_csv(processed_dir() / "ghana_w28.csv")
    assert not {"n_quizzes_graded", "days_inactive_at_end"} & set(g.columns)
    assert set(F.SHARED_FEATURES) <= set(g.columns)
