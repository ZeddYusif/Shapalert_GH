"""Phase 1, steps 1.4-1.8: splits, baselines, tuned gradient boosting, calibration, alerts, abstention.

Tuning uses the VALIDATION set only (PR-AUC). Test data is touched once per model for reporting.
Outputs: outputs/tables/table2_*.csv, table3_*.csv, abstention, smote ablation, best params,
         outputs/preds/*.csv (test predictions), outputs/models/*.joblib, figures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from shapalert import features as F
from shapalert import models as M
from shapalert.common import (CFG, PlattCalibrator, SEED, best_f1_threshold, core_metrics, out_dir,
                              processed_dir, save_json, seed_everything, topk_metrics, write_table)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
seed_everything()
FEATSETS = {"full": F.FULL_FEATURES, "shared": F.SHARED_FEATURES}
BASELINES = ["majority", "logreg", "tree", "rf"]
CAPS = CFG["alerts"]["capacities"]
LO, HI = CFG["alerts"]["abstain_band"]


def load(T):
    return pd.read_csv(processed_dir() / f"oulad_w{T}.csv")


def make_splits(df):
    """Return {split_name: (train_idx, val_idx, test_idx)} as positional indices."""
    ts = CFG["oulad"]["time_split"]
    pres = df["code_presentation"]
    time = (np.where(pres.isin(ts["train"]))[0], np.where(pres.isin(ts["val"]))[0],
            np.where(pres.isin(ts["test"]))[0])
    # group split: whole course presentations held out (Wu et al.). Group list is fixed by seed
    # using the 28-day course list so every window uses the same held-out courses.
    courses = sorted(pd.read_csv(processed_dir() / "oulad_w28.csv")["course"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(courses)
    gs = CFG["oulad"]["group_split"]
    n = len(courses)
    n_te, n_va = max(1, round(gs["test_frac"] * n)), max(1, round(gs["val_frac"] * n))
    te_c, va_c = set(courses[:n_te]), set(courses[n_te:n_te + n_va])
    c = df["course"]
    group = (np.where(~c.isin(te_c | va_c))[0], np.where(c.isin(va_c))[0], np.where(c.isin(te_c))[0])
    return {"time": time, "group": group}, {"group_test_courses": sorted(te_c), "group_val_courses": sorted(va_c)}


def tune(name, Xtr, ytr, Xva, yva):
    def objective(trial):
        params = M.suggest_params(name, trial)
        m = M.make_gbm(name, params, ytr)
        m.fit(Xtr, ytr)
        return average_precision_score(yva, m.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=CFG["tuning"]["n_trials"], show_progress_bar=False)
    best = M.suggest_params(name, optuna.trial.FixedTrial(study.best_params))
    return best, study.best_value


def alert_rows(y, score, meta):
    rows = []
    for cap in CAPS:
        t = topk_metrics(y, score, cap)
        rows.append({**meta, "capacity": cap, **t})
    return rows


def abstention_row(y, cal, meta):
    y = np.asarray(y)
    conf = (cal < LO) | (cal > HI)
    pred = (cal > HI).astype(int)
    hi = cal > HI
    return {**meta, "band": f"{LO}-{HI}", "coverage_confident": float(conf.mean()),
            "share_uncertain": float((~conf).mean()),
            "accuracy_confident": float((pred[conf] == y[conf]).mean()) if conf.any() else np.nan,
            "precision_flagged_high": float(y[hi].mean()) if hi.any() else np.nan,
            "dropout_share_in_uncertain": float(y[~conf].mean()) if (~conf).any() else np.nan,
            "dropouts_in_uncertain": int(y[~conf].sum())}


def main():
    t2, t3, t_abs, best_params, smote_rows = [], [], [], {}, []
    split_info = {}
    curves = {}
    for T in CFG["windows"]:
        df = load(T)
        splits, info = make_splits(df)
        split_info = info
        for sname, (tr, va, te) in splits.items():
            for fs_name, feats in FEATSETS.items():
                if fs_name == "shared" and sname != "time":
                    continue
                X, y = df[feats], df["y"].values
                Xtr, ytr, Xva, yva, Xte, yte = X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], y[te]
                meta = {"split": sname, "featset": fs_name, "window": T}
                print(f"[T={T} split={sname} set={fs_name}] train={len(tr):,} val={len(va):,} test={len(te):,} "
                      f"dropout test={yte.mean():.3f}", flush=True)
                results = {}
                for b in BASELINES:
                    m = M.baseline(b).fit(Xtr, ytr)
                    results[b] = (m, m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], None)
                for g in CFG["tuning"]["models"]:
                    bp, bv = tune(g, Xtr, ytr, Xva, yva)
                    m = M.make_gbm(g, bp, ytr).fit(Xtr, ytr)
                    results[g] = (m, m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], bp)
                    best_params[f"{sname}|{fs_name}|{T}|{g}"] = {"params": bp, "val_pr_auc": bv}
                    joblib.dump(m, out_dir("models") / f"{sname}_{fs_name}_w{T}_{g}.joblib")
                for name, (m, pv, pt, bp) in results.items():
                    if name == "majority":
                        cal_te, thr = pt, 0.5
                    else:
                        cal = PlattCalibrator().fit(pv, yva)
                        cal_te = cal.predict(pt)
                        thr = best_f1_threshold(yva, cal.predict(pv))
                    met = core_metrics(yte, pt, cal_te, thr)
                    met["val_pr_auc"] = float(average_precision_score(yva, pv)) if name != "majority" else np.nan
                    t2.append({**meta, "model": name, **met})
                    if name != "majority":
                        t3 += alert_rows(yte, pt, {**meta, "model": name})
                        if name in CFG["tuning"]["models"] + ["logreg"]:
                            t_abs.append(abstention_row(yte, cal_te, {**meta, "model": name}))
                        if fs_name == "full":
                            pd.DataFrame({"y": yte, "raw": pt, "cal": cal_te,
                                          "gender": df.iloc[te]["gender"].values,
                                          "disability": df.iloc[te]["disability"].values,
                                          "age_band": df.iloc[te]["age_band"].values,
                                          "course": df.iloc[te]["course"].values}).to_csv(
                                out_dir("preds") / f"oulad_{sname}_w{T}_{name}.csv", index=False)
                    if sname == "time" and fs_name == "full" and name in CFG["tuning"]["models"]:
                        curves[(T, name)] = (yte, pt, cal_te)

                # SMOTE ablation (training part only, inside an imblearn pipeline)
                if CFG["smote_ablation"] and sname == "time" and fs_name == "full":
                    from imblearn.over_sampling import SMOTE
                    from imblearn.pipeline import Pipeline as IPipe
                    from sklearn.impute import SimpleImputer
                    g = "lightgbm"
                    bp = best_params[f"{sname}|{fs_name}|{T}|{g}"]["params"]
                    pipe = IPipe([("imp", SimpleImputer(strategy="median")), ("smote", SMOTE(random_state=SEED)),
                                  ("m", M.make_gbm(g, bp, ytr, class_weight=False))]).fit(Xtr, ytr)
                    ps = pipe.predict_proba(Xte)[:, 1]
                    smote_rows.append({"window": T, "variant": "lightgbm + SMOTE (train only)",
                                       "pr_auc": average_precision_score(yte, ps),
                                       **{f"recall@{int(c*100)}%": topk_metrics(yte, ps, c)["recall"] for c in CAPS}})
                    pt = results[g][2]
                    smote_rows.append({"window": T, "variant": "lightgbm + class weight",
                                       "pr_auc": average_precision_score(yte, pt),
                                       **{f"recall@{int(c*100)}%": topk_metrics(yte, pt, c)["recall"] for c in CAPS}})

    t2 = pd.DataFrame(t2)
    write_table(t2.round(4), "table2_oulad_model_comparison")
    write_table(pd.DataFrame(t3).round(4), "table3_oulad_alert_performance")
    write_table(pd.DataFrame(t_abs).round(4), "table3b_oulad_abstention")
    if smote_rows:
        write_table(pd.DataFrame(smote_rows).round(4), "oulad_smote_ablation")
    save_json(best_params, out_dir("frozen") / "oulad_tuned_params.json")
    save_json(split_info, out_dir("frozen") / "oulad_split_info.json")

    # --- figures -------------------------------------------------------------
    best = (t2[(t2.split == "time") & (t2.featset == "full") & (t2.window == CFG["primary_window"]) &
               (t2.model.isin(CFG["tuning"]["models"]))].sort_values("val_pr_auc", ascending=False).iloc[0]["model"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    y, raw, cal = curves[(CFG["primary_window"], best)]
    for ax, s, ttl in [(axes[0], raw, "Uncalibrated"), (axes[1], cal, "Platt-calibrated")]:
        fp, mp = calibration_curve(y, s, n_bins=10, strategy="quantile")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(fp, mp, "o-")
        ax.set_title(f"{ttl} ({best}, {CFG['primary_window']}d, test 2014J)")
        ax.set_xlabel("Predicted risk")
    axes[0].set_ylabel("Observed dropout rate")
    fig.tight_layout()
    fig.savefig(out_dir("figures") / "fig_calibration_oulad.png", dpi=150)
    plt.close(fig)

    t3d = pd.DataFrame(t3)
    sub = t3d[(t3d.split == "time") & (t3d.featset == "full") & (t3d.capacity == 0.2)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for mdl, g in sub.groupby("model"):
        g = g.sort_values("window")
        ax.plot(g.window, g.recall, "o-", label=mdl)
    ax.set_xlabel("Days after course start")
    ax.set_ylabel("Recall at top 20%")
    ax.set_xticks(CFG["windows"])
    ax.set_title("OULAD: recall vs window (test 2014J)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir("figures") / "fig_recall_vs_window_oulad.png", dpi=150)
    plt.close(fig)
    print("Phase 1 modelling complete.")


if __name__ == "__main__":
    main()
