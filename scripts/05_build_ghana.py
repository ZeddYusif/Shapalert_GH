"""Phase 2: build the Ghana (KNUST Moodle) table with the SAME recipe as OULAD.

Reads (read-only): knust_source/activity_log.csv, gradebook_grades.csv, quiz_attempts.csv
Writes:           data/processed/ghana_w{T}.csv, outputs/tables/ghana_*.csv, docs/data_dictionary_ghana.csv

LABEL (CONFIRM WITH SUPERVISOR; recorded in docs/label_definition.md):
    dropout = 1 if the student has a graded result for fewer than `n_required_quizzes` of the six module
    quizzes by the end of the export (non-completion).  Variant `noncompletion_and_inactive` also requires
    no LMS activity in the last `inactive_days` days.  The label is built from outcome data only; it is never
    used to build features. All features use events with day < T.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapalert import features as F
from shapalert.common import CFG, ROOT, out_dir, processed_dir, seed_everything, write_table

seed_everything()
D = ROOT / CFG["paths"]["ghana_dir"]
G = CFG["ghana"]
START = pd.Timestamp(G["course_start"], tz="UTC")
PRE = CFG["pre_start_days"]
COURSE = "KNUST_2026"


def day_of(ts: pd.Series) -> pd.Series:
    return ((ts - START).dt.floor("D") / pd.Timedelta(days=1)).astype(int)


def main():
    qc = {}
    act = pd.read_csv(D / "activity_log.csv")
    qc["raw_log_rows"] = len(act)
    act = act.drop_duplicates()
    qc["duplicate_rows_removed"] = qc["raw_log_rows"] - len(act)
    act["t"] = pd.to_datetime(act["activity_time"], utc=True)
    # staff/test accounts: the export is pseudonymised (P#####); flag accounts that are only 'grade'/'ws' system events
    sys_only = act.groupby("participant_id")["activity_type"].agg(lambda s: set(s) <= {"grade"})
    qc["accounts_with_only_system_grade_events"] = int(sys_only.sum())
    students = pd.Index(sorted(act["participant_id"].unique()), name="pseudo_id")
    qc["students_in_log"] = len(students)

    # ---- outcome (label) ------------------------------------------------------------------
    gb = pd.read_csv(D / "gradebook_grades.csv")
    graded = gb.dropna(subset=["finalgrade"]).groupby("participant_id")["grade_item"].nunique()
    n_quiz = graded.reindex(students).fillna(0).astype(int)
    end = act["t"].max()
    last_act = act[act["t"] >= START - pd.Timedelta(days=PRE)].groupby("participant_id")["t"].max().reindex(students)
    gap_end = (end - last_act).dt.days.fillna(9999)
    noncomp = (n_quiz < G["n_required_quizzes"]).astype(int)
    inactive = (gap_end >= G["inactive_days"]).astype(int)
    if G["label_rule"] == "noncompletion":
        y = noncomp
    elif G["label_rule"] == "noncompletion_and_inactive":
        y = (noncomp & inactive).astype(int)
    elif G["label_rule"] == "proposal_inactive21_missed2":
        # Proposal s3.2.2: >=21 consecutive days with no login (ending at the end of the export)
        # AND >=2 consecutive missed graded quizzes (Module n and n+1 both without a graded result).
        lg = act[(act["activity_type"] == "login") & (act["t"] >= START)].groupby("participant_id")["t"].max()
        gap_login = (end - lg.reindex(students)).dt.days.fillna(9999)
        okg = gb.dropna(subset=["finalgrade", "grade_item"]).copy()
        okg["m"] = okg["grade_item"].str.extract(r"(\d)")[0].astype(int)
        done = okg.groupby("participant_id")["m"].apply(set).reindex(students)
        two_missed = done.apply(lambda s_: any((m not in (s_ if isinstance(s_, set) else set())) and
                                               (m + 1 not in (s_ if isinstance(s_, set) else set())) for m in range(1, 6)))
        y = ((gap_login >= G["inactive_days"]) & two_missed.values).astype(int)
    elif G["label_rule"] == "export_at_risk":      # label supplied with the data (sensitivity analysis)
        ex0 = pd.read_csv(ROOT / CFG["paths"]["oulad_dir"] / "knust_moodle_export.csv").set_index("participant_id")
        y = ex0["at_risk"].reindex(students).fillna(1).astype(int)
    else:
        raise ValueError(G["label_rule"])
    lab = pd.DataFrame({"pseudo_id": students, "y": y.values, "n_quizzes_graded": n_quiz.values,
                        "days_inactive_at_end": gap_end.values})
    qc["label_rule"] = G["label_rule"]
    qc["dropout_count"] = int(lab.y.sum())
    qc["dropout_rate"] = float(lab.y.mean())
    qc["WARNING_dropouts_below_40"] = bool(lab.y.sum() < 40)
    # cross-check against the export shipped with the data (label validation only; NOT used as features)
    exp_path = ROOT / CFG["paths"]["oulad_dir"] / "knust_moodle_export.csv"
    if exp_path.exists():
        ex = pd.read_csv(exp_path)[["participant_id", "at_risk"]].rename(columns={"participant_id": "pseudo_id"})
        m = lab.merge(ex, on="pseudo_id")
        qc["agreement_with_export_at_risk"] = float((m.y == m.at_risk).mean())
    print(f"students={len(students):,}  dropouts={lab.y.sum():,} ({lab.y.mean():.3f})")

    # ---- events: one click = one log row ----------------------------------------------------
    ev = act[act["t"] >= START - pd.Timedelta(days=PRE)].copy()
    ev["day"] = day_of(ev["t"])
    ev = ev[ev["day"] < max(CFG["windows"])]
    ev["uid"] = ev["participant_id"]
    ev["clicks"] = 1
    ev["res"] = np.nan
    ev["act"] = np.nan
    events = ev[["uid", "day", "clicks", "res", "act"]]

    # ---- assessments: first finished quiz attempt (known at finish time) ----------------------
    qa = pd.read_csv(D / "quiz_attempts.csv")
    qa = qa[qa["state"] == "finished"].copy()
    qa["t"] = pd.to_datetime(qa["finished"], utc=True)
    qa = qa.sort_values("t").groupby(["participant_id", "quiz_name"], as_index=False).first()
    subs = pd.DataFrame({"uid": qa["participant_id"], "day": day_of(qa["t"]), "score": qa["percent"]})


    for T in CFG["windows"]:
        beh = F.behaviour_features(events, students.values, T, PRE)
        asf = F.assessment_features(subs, students.values, T)
        tab = beh.join(asf)
        tab.index.name = "pseudo_id"
        tab = tab.reset_index().merge(lab[["pseudo_id", "y"]], on="pseudo_id")
        tab["course"] = COURSE
        tab = F.add_within_course_z(tab, "course", F.Z_MAP)
        tab["window"] = T
        tab.to_csv(processed_dir() / f"ghana_w{T}.csv", index=False)
        print(f"T={T}: rows={len(tab):,}, zero-click students={(tab.clicks_total == 0).mean():.3f}")
        qc[f"zero_click_share_w{T}"] = float((tab.clicks_total == 0).mean())

    # ---- quality plots / tables ----------------------------------------------------------------
    wk = events.assign(week=(events.day // 7).clip(lower=-2)).groupby(["uid", "week"])["clicks"].sum().unstack(fill_value=0)
    wk = wk.reindex(students, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    lab_idx = lab.set_index("pseudo_id")["y"]
    for yy, nm in [(0, "completers"), (1, "dropouts")]:
        sub = wk[lab_idx.reindex(wk.index) == yy]
        ax.plot(sub.columns, sub.mean(axis=0), "o-", label=f"{nm} (mean, n={len(sub)})")
    ax.set_xlabel("Week relative to course start")
    ax.set_ylabel("Mean LMS events per student")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir("figures") / "fig_ghana_weekly_clicks.png", dpi=150)
    plt.close(fig)
    qc["students_never_active_in_first_42_days"] = int((wk.sum(axis=1) == 0).sum())
    pd.Series(qc).to_frame("value").to_csv(out_dir("tables") / "ghana_quality_report.csv")

    t1 = pd.DataFrame([{"dataset": COURSE, "students": len(lab), "dropouts": int(lab.y.sum()),
                        "dropout_rate": round(float(lab.y.mean()), 4),
                        "features_shared": len(F.SHARED_FEATURES)}])
    write_table(t1, "table1_ghana_description")

    dd = [("pseudo_id", "Pseudonymised student ID (no names/index numbers)", "text", "LMS export"),
          ("y", "1 = dropout (non-completion, see label_definition.md); outcome only, never a feature", "0/1", "gradebook"),
          ("clicks_total", "LMS log events with day < T", "count", "activity_log"),
          ("active_days", "Distinct days with any event before T", "days", "activity_log"),
          ("clicks_last_week", "Events in days [T-7, T)", "count", "activity_log"),
          ("click_trend", "Last-week events minus first-week events", "count", "activity_log"),
          ("days_since_last_click", "(T-1) minus day of last event; horizon if never active", "days", "activity_log"),
          ("first_click_day", "Day of first event (negative = before start); T if never active", "days", "activity_log"),
          ("clicks_per_active_day", "clicks_total / active_days", "ratio", "derived"),
          ("active_ratio", "active_days / (T + pre_start_days)", "ratio", "derived"),
          ("gap_frac", "days_since_last_click / (T + pre_start_days)", "ratio", "derived"),
          ("first_click_frac", "Scaled first_click_day in [0,1]", "ratio", "derived"),
          ("n_early_assess", "Quizzes finished before T (first attempt)", "count", "quiz_attempts"),
          ("early_score_mean", "Mean first-attempt percent of quizzes finished before T; NaN if none", "%", "quiz_attempts"),
          ("has_early_assess", "n_early_assess > 0", "0/1", "derived"),
          ("*_z", "Within-course z-score of the matching count feature (label-free)", "z", "derived")]
    pd.DataFrame(dd, columns=["column", "meaning", "unit", "source"]).to_csv(ROOT / "docs" / "data_dictionary_ghana.csv", index=False)
    print(pd.Series(qc).to_string())


if __name__ == "__main__":
    main()
