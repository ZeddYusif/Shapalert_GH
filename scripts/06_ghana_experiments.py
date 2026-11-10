"""Phase 3: Ghana experiments A-D, earliness (3.5) and small-sample sensitivity.

READS the frozen Phase 1 spec; never changes it. No tuning on test folds: model settings are fixed
(small capacity, early stopping on an inner split of the TRAINING fold only).

Usage:  python scripts/06_ghana_experiments.py [--small]
   --small  also repeats everything on `ghana.small_draws` random stratified draws of `ghana.small_n` students
            (mimics the 200-350 student scenario in the guide).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from shapalert import features as F
from shapalert import models as M
from shapalert.common import (CFG, PlattCalibrator, SEED, bootstrap_ci, bootstrap_metric_ci, fmt_ci, load_json, logit,
                              out_dir, processed_dir, seed_everything, topk_metrics, write_table)

warnings.filterwarnings("ignore")
seed_everything()
FZ = load_json(out_dir("frozen") / "frozen_pipeline.json")
FAMILY = FZ["transfer_model_family"]
PRIMARY = CFG["primary_window"]
CV = CFG["ghana_cv"]
CAPS = CFG["alerts"]["capacities"]
LMS = F.SHARED_FEATURES
MODELS = ["logreg", FAMILY]


def metrics(y, s) -> dict:
    y = np.asarray(y)
    out = {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s)}
    for c in CAPS:
        t = topk_metrics(y, s, c)
        out[f"recall@{int(c*100)}"] = t["recall"]
        out[f"precision@{int(c*100)}"] = t["precision"]
    t = topk_metrics(y, s, float(y.mean()))   # F1 for the dropout class when flagging as many as the base rate
    p, r = t["precision"], t["recall"]
    out["f1_dropout"] = 2 * p * r / (p + r) if (p + r) else 0.0
    return out


def fit_logreg(Xtr, ytr, w=None):
    m = M.baseline("logreg")
    m.fit(Xtr, ytr, **({} if w is None else {"m__sample_weight": np.asarray(w)}))
    return m


def fit_model(name, Xtr, ytr, Xte, w=None):
    """Return scores for Xte from a small-capacity model trained on (Xtr, ytr)."""
    if name == "logreg":
        return fit_logreg(Xtr, ytr, w).predict_proba(Xte)[:, 1]
    return M.fit_predict_small(name, Xtr, ytr, Xte, sample_weight=w)[1]


def train_oulad_frozen(T, feats):
    """OULAD model(s) with FROZEN Phase 1 hyper-parameters, trained on all OULAD rows (shared features)."""
    o = pd.read_csv(processed_dir() / f"oulad_w{T}.csv")
    X, y = o[feats], o["y"].values
    lr = fit_logreg(X, y)
    params = FZ["transfer_params"][str(T)][FAMILY]
    gbm = M.make_gbm(FAMILY, params, y).fit(X, y)
    return {"logreg": lr, FAMILY: gbm}, o


def cross_fit_brier(y, oof_by_repeat, folds_by_repeat):
    """Brier of Platt-recalibrated OOF scores; the calibrator for fold k is fitted on the other folds' OOF scores."""
    res = []
    for r, s in oof_by_repeat.items():
        cal = np.zeros(len(y))
        for te in folds_by_repeat[r]:
            tr = np.setdiff1d(np.arange(len(y)), te)
            cal[te] = PlattCalibrator().fit(s[tr], y[tr]).predict(s[te])
        res.append(brier_score_loss(y, cal))
    return res


def run_cv(g: pd.DataFrame, feats, T, ou_models, o_df, experiments=("A", "B", "C", "D"), tag=""):
    """Repeated stratified CV. Returns long DataFrame: experiment, model, repeat, metrics..."""
    y = g["y"].values
    X = g[feats].reset_index(drop=True)
    n = len(g)
    oulad_feats = [f for f in feats if f in o_df.columns]
    Xo, yo = o_df[oulad_feats], o_df["y"].values
    # OULAD-trained scores for ALL Ghana rows (labels of Ghana never used) -> experiment A and stacking feature
    s_ou = {m: ou_models[m].predict_proba(g[oulad_feats])[:, 1] for m in MODELS} if oulad_feats else {}
    stack_col = logit(s_ou[FAMILY]) if s_ou else None
    rows = []
    if "A" in experiments:
        for m in MODELS:
            rows.append({"experiment": "A_zero_shot", "model": m, "repeat": 0, **metrics(y, s_ou[m]),
                         "_scores": s_ou[m]})
    cv = RepeatedStratifiedKFold(n_splits=CV["n_splits"], n_repeats=CV["n_repeats"], random_state=SEED)
    keys = [(e, m) for e in ("B", "C", "D") if e in experiments for m in MODELS]
    oof = {k: {r: np.zeros(n) for r in range(CV["n_repeats"])} for k in keys}
    folds = {r: [] for r in range(CV["n_repeats"])}
    fold_pos = []
    for i, (tr, te) in enumerate(cv.split(X, y)):
        r = i // CV["n_splits"]
        folds[r].append(te)
        fold_pos.append(int(y[te].sum()))
        Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
        for m in MODELS:
            if ("B", m) in oof:
                oof[("B", m)][r][te] = fit_model(m, Xtr, ytr, Xte)
            if ("C", m) in oof:   # stacking: OULAD score as an extra feature
                Xs = X.assign(oulad_score=stack_col)
                oof[("C", m)][r][te] = fit_model(m, Xs.iloc[tr], ytr, Xs.iloc[te])
            if ("D", m) in oof:   # pooled OULAD + Ghana (Ghana rows weighted up)
                Xp = pd.concat([Xo[oulad_feats], Xtr[oulad_feats]], ignore_index=True)
                yp = np.concatenate([yo, ytr])
                wp = np.concatenate([np.ones(len(yo)), np.full(len(ytr), CV["ghana_weight"])])
                oof[("D", m)][r][te] = fit_model(m, Xp, yp, Xte[oulad_feats], w=wp)
    names = {"B": "B_local", "C": "C_finetune_stack", "D": "D_pooled"}
    for (e, m), per_rep in oof.items():
        briers = cross_fit_brier(y, per_rep, folds)
        for r, s in per_rep.items():
            rows.append({"experiment": names[e], "model": m, "repeat": r, **metrics(y, s), "brier": briers[r]})
    df = pd.DataFrame(rows)
    df.attrs["test_fold_dropouts"] = (float(np.mean(fold_pos)), int(np.min(fold_pos)))
    df["tag"] = tag
    return df


def summarise(df, metric_cols, y_for_A=None, n_boot=1000):
    """Mean and 95% CI (bootstrap over repeats; experiment A: bootstrap over students)."""
    out = []
    for (e, m), grp in df.groupby(["experiment", "model"]):
        row = {"experiment": e, "model": m}
        for c in metric_cols:
            if c not in grp or grp[c].isna().all():
                continue
            if e.startswith("A") and "_scores" in grp:
                s = grp["_scores"].iloc[0]
                fn = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}.get(c)
                if fn is not None:
                    row[c] = fmt_ci(*bootstrap_metric_ci(y_for_A, s, fn, n_boot))
                    continue
                if c.startswith(("recall@", "precision@")):
                    cap = int(c.split("@")[1]) / 100
                    k = "recall" if c.startswith("recall") else "precision"
                    row[c] = fmt_ci(*bootstrap_metric_ci(y_for_A, s, lambda yy, ss: topk_metrics(yy, ss, cap)[k], n_boot))
                    continue
                row[c] = f"{grp[c].iloc[0]:.3f}"
                continue
            row[c] = fmt_ci(*bootstrap_ci(grp[c].values, n_boot))
        out.append(row)
    return pd.DataFrame(out)


def main(small: bool):
    ghana = {T: pd.read_csv(processed_dir() / f"ghana_w{T}.csv") for T in CFG["windows"]}
    ou = {T: train_oulad_frozen(T, LMS) for T in CFG["windows"]}
    base = ghana[PRIMARY]["y"].mean()
    print(f"Frozen family: {FAMILY}. Ghana n={len(ghana[PRIMARY])}, dropout rate={base:.3f}")
    mcols = ["pr_auc", "roc_auc", "brier", "recall@10", "recall@20", "recall@30", "precision@20", "f1_dropout"]

    # ---- Table 4 (experiments A-D at the primary window) ------------------------------------
    g = ghana[PRIMARY]
    res = run_cv(g, LMS, PRIMARY, ou[PRIMARY][0], ou[PRIMARY][1], tag="full")
    s4 = summarise(res, mcols, y_for_A=g["y"].values)
    maj = pd.DataFrame([{"experiment": "0_majority_baseline", "model": "always 'stays'", "pr_auc": f"{base:.3f}",
                         "roc_auc": "0.500", "recall@20": "0.200 (random)", "accuracy": f"{1-base:.3f}"}])
    t4 = pd.concat([maj, s4], ignore_index=True)
    mean_pos, min_pos = res.attrs["test_fold_dropouts"]
    t4["test_fold_dropouts(mean,min)"] = f"{mean_pos:.1f}, {min_pos}"
    write_table(t4, "table4_ghana_experiments_A_to_D")
    print(t4.to_string(index=False))

    # ---- 3.5 earliness (experiment B at 14/28/42) ----------------------------------------------
    early = []
    for T in CFG["windows"]:
        r = run_cv(ghana[T], LMS, T, ou[T][0], ou[T][1], experiments=("B",), tag=f"w{T}")
        r["window"] = T
        early.append(r)
    er = pd.concat(early)
    rows = []
    for (T, m), grp in er.groupby(["window", "model"]):
        row = {"window": T, "model": m}
        for c in ["pr_auc", "roc_auc", "recall@10", "recall@20", "recall@30"]:
            mu, lo, hi = bootstrap_ci(grp[c].values)
            row[c], row[c + "_lo"], row[c + "_hi"] = mu, lo, hi
        rows.append(row)
    te = pd.DataFrame(rows)
    write_table(te.round(4), "ghana_earliness_B")
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, grp in te.groupby("model"):
        ax.errorbar(grp.window, grp["recall@20"], yerr=[grp["recall@20"] - grp["recall@20_lo"],
                    grp["recall@20_hi"] - grp["recall@20"]], marker="o", capsize=3, label=m)
    ax.axhline(0.6, color="grey", ls="--", lw=1)
    ax.set_xticks(CFG["windows"])
    ax.set_xlabel("Days after course start")
    ax.set_ylabel("Recall at top 20% (Ghana, CV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir("figures") / "fig_recall_vs_window_ghana.png", dpi=150)
    plt.close(fig)
    useful = te[te["recall@20"] >= 0.6].groupby("model")["window"].min()
    print("First window with recall@20% >= 0.60:", useful.to_dict() or "none")

    # ---- small-sample sensitivity -------------------------------------------------------------
    if small:
        n_s = CFG["ghana"]["small_n"]
        parts = []
        for d in range(CFG["ghana"]["small_draws"]):
            sub = g.groupby("y", group_keys=False).apply(
                lambda x: x.sample(frac=n_s / len(g), random_state=SEED + d)).reset_index(drop=True)
            r = run_cv(sub, LMS, PRIMARY, ou[PRIMARY][0], ou[PRIMARY][1], tag=f"draw{d}")
            r["draw"] = d
            r["unit"] = r["draw"] * 100 + r["repeat"]
            parts.append(r)
            print(f"small draw {d}: n={len(sub)}, dropouts={int(sub.y.sum())}")
        rs = pd.concat(parts)
        ts = []
        for (e, m), grp in rs.groupby(["experiment", "model"]):
            row = {"experiment": e, "model": m, "n_students": n_s, "draws": CFG["ghana"]["small_draws"]}
            for c in ["pr_auc", "roc_auc", "recall@20"]:
                row[c] = fmt_ci(*bootstrap_ci(grp[c].values))
            ts.append(row)
        write_table(pd.DataFrame(ts), "table4b_ghana_small_sample_sensitivity")
        print(pd.DataFrame(ts).to_string(index=False))


if __name__ == "__main__":
    main("--small" in sys.argv)
