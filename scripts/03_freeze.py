"""Phase 1, step 1.10: FREEZE. Select models on VALIDATION PR-AUC only, then write the frozen spec.

Phases 2-3 read outputs/frozen/frozen_pipeline.json and must not change it.
Run this BEFORE touching any Ghana modelling script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib
import platform
import subprocess

import pandas as pd

from shapalert import features as F
from shapalert.common import CFG, ROOT, SEED, load_json, out_dir, save_json

t2 = pd.read_csv(out_dir("tables") / "table2_oulad_model_comparison.csv")
params = load_json(out_dir("frozen") / "oulad_tuned_params.json")
gbms = CFG["tuning"]["models"]
frozen = {"seed": SEED, "windows": CFG["windows"], "primary_window": CFG["primary_window"],
          "label": CFG["oulad"]["label"], "pre_start_days": CFG["pre_start_days"],
          "features_full": F.FULL_FEATURES, "features_shared": F.SHARED_FEATURES,
          "selection_rule": "highest validation PR-AUC among tuned GBMs (time-aware split)", "best": {}}
for fs in ["full", "shared"]:
    for T in CFG["windows"]:
        sub = t2[(t2.split == "time") & (t2.featset == fs) & (t2.window == T) & t2.model.isin(gbms)]
        top = sub.sort_values("val_pr_auc", ascending=False).iloc[0]
        frozen["best"][f"{fs}|{T}"] = {"model": top["model"], "val_pr_auc": float(top["val_pr_auc"]),
                                        "params": params[f"time|{fs}|{T}|{top['model']}"]["params"]}
# the single model family carried to Ghana: best shared-feature model at the primary window
prim = frozen["best"][f"shared|{CFG['primary_window']}"]["model"]
frozen["transfer_model_family"] = prim
frozen["transfer_params"] = {str(T): {m: params[f"time|shared|{T}|{m}"]["params"] for m in gbms}
                             for T in CFG["windows"]}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


frozen["code_hashes"] = {str(p.relative_to(ROOT)): sha(p) for p in
                         sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("0[1-3]*.py")) +
                                [ROOT / "config.yaml"])}
try:
    frozen["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                   stderr=subprocess.DEVNULL, text=True).strip()
except Exception:
    frozen["git_commit"] = "no commit yet (see code_hashes)"
frozen["python"] = platform.python_version()
save_json(frozen, out_dir("frozen") / "frozen_pipeline.json")
print("FROZEN. Transfer model family:", prim)
for k, v in frozen["best"].items():
    print(f"  {k:12s} -> {v['model']:9s} val PR-AUC {v['val_pr_auc']:.3f}")
