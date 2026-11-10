"""Phase 1, steps 1.1-1.3: OULAD base table + windowed, leak-free features.

Output: data/processed/oulad_w{T}.csv for T in config windows (one row per registration).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from shapalert.common import CFG, ROOT, processed_dir, save_json, seed_everything
from shapalert import features as F

seed_everything()
D = ROOT / CFG["paths"]["oulad_dir"]
PRE = CFG["pre_start_days"]
KEY = ["code_module", "code_presentation", "id_student"]

ACT_MAP = {"forumng": "forum", "oucollaborate": "forum", "quiz": "quiz", "resource": "resource",
           "oucontent": "resource", "subpage": "resource", "url": "resource", "homepage": "homepage"}


def main():
    info = pd.read_csv(D / "studentInfo.csv")
    reg = pd.read_csv(D / "studentRegistration.csv")
    # 1.1 base table
    base = info.merge(reg, on=KEY, how="left", validate="1:1")
    base["y"] = (base["final_result"] == "Withdrawn").astype(int)       # dropout label (Wu et al.)
    base["course"] = base["code_module"] + "_" + base["code_presentation"]
    base["uid"] = np.arange(len(base))
    print(f"base table: {len(base):,} registrations, dropout rate {base.y.mean():.3f}")

    # static features
    age = {"0-35": 0, "35-55": 1, "55<=": 2}
    edu = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
           "HE Qualification": 3, "Post Graduate Qualification": 4}
    base["gender_m"] = (base["gender"] == "M").astype(int)
    base["age_band_ord"] = base["age_band"].map(age)
    base["edu_ord"] = base["highest_education"].map(edu)
    base["disability_y"] = (base["disability"] == "Y").astype(int)
    base["imd_ord"] = base["imd_band"].str.extract(r"^(\d+)")[0].astype(float) / 10  # NaN kept

    # VLE events (450 MB: load only needed columns, aggregate to student-day-site)
    vle = pd.read_csv(D / "vle.csv", usecols=["id_site", "code_module", "code_presentation", "activity_type"])
    vle["act"] = vle["activity_type"].map(ACT_MAP).fillna("other")
    max_T = max(CFG["windows"])
    chunks = []
    for ch in pd.read_csv(D / "studentVle.csv", chunksize=2_000_000,
                          dtype={"id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"}):
        ch = ch[(ch["date"] < max_T) & (ch["date"] >= -PRE)]
        chunks.append(ch)
    ev = pd.concat(chunks, ignore_index=True)
    ev["id_student"] = ev["id_student"].astype("int64")
    ev = ev.merge(base[KEY + ["uid"]].assign(id_student=lambda d: d["id_student"].astype("int64")),
                  on=KEY, how="inner")
    ev = ev.merge(vle[["id_site", "code_module", "code_presentation", "act"]],
                  on=["id_site", "code_module", "code_presentation"], how="left")
    events = ev.rename(columns={"date": "day", "sum_click": "clicks", "id_site": "res"})[
        ["uid", "day", "clicks", "res", "act"]]
    print(f"VLE events kept (day < {max_T}): {len(events):,}")

    # assessments (submitted before T only)
    asm = pd.read_csv(D / "assessments.csv")
    sa = pd.read_csv(D / "studentAssessment.csv").merge(
        asm[["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
    sa = sa.merge(base[KEY + ["uid"]], on=KEY)
    subs = sa.rename(columns={"date_submitted": "day"})[["uid", "day", "score"]].dropna(subset=["day"])

    leak_report = {}
    for T in CFG["windows"]:
        # 1.2 remove students who unregistered before T (outcome already known)
        keep = base[~(base["date_unregistration"] < T)].copy()
        removed = len(base) - len(keep)
        beh = F.behaviour_features(events, keep["uid"].values, T, PRE)
        asf = F.assessment_features(subs, keep["uid"].values, T)
        tab = keep.set_index("uid").join(beh).join(asf)
        tab = F.add_within_course_z(tab, "course", F.Z_MAP)
        tab["window"] = T
        cols = (["course", "code_module", "code_presentation", "id_student", "y", "gender", "disability", "age_band",
                 "final_result", "window"] + [c for c in F.FULL_FEATURES if c in tab.columns] +
                [c for c in F.SHARED_FEATURES if c not in F.FULL_FEATURES])
        tab = tab[cols].reset_index()
        tab.to_csv(processed_dir() / f"oulad_w{T}.csv", index=False)
        # leakage checks recorded for the report
        leak_report[T] = {"rows": len(tab), "removed_unregistered_before_T": int(removed),
                          "dropout_rate": float(tab.y.mean()),
                          "max_event_day_used": int(events[events.day < T].day.max()),
                          "max_assessment_day_used": int(subs[subs.day < T].day.max())}
        print(f"T={T}: rows={len(tab):,} (removed {removed:,} unregistered before T), dropout={tab.y.mean():.3f}")
        assert leak_report[T]["max_event_day_used"] < T
    save_json(leak_report, ROOT / CFG["paths"]["outputs"] / "tables" / "oulad_leakage_checks.json")


if __name__ == "__main__":
    main()
