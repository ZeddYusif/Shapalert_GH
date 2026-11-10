"""Shared helpers: config, paths, metrics, calibration, bootstrap."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = load_config()
# quick smoke-test overrides (never used for reported results): SHAPALERT_TRIALS, SHAPALERT_BOOT
if os.environ.get("SHAPALERT_TRIALS"):
    CFG["tuning"]["n_trials"] = int(os.environ["SHAPALERT_TRIALS"])
if os.environ.get("SHAPALERT_BOOT"):
    CFG["explain"]["n_boot"] = int(os.environ["SHAPALERT_BOOT"])
    CFG["fairness"]["n_boot"] = int(os.environ["SHAPALERT_BOOT"])
    CFG["ghana_cv"]["n_repeats"] = max(2, int(os.environ["SHAPALERT_BOOT"]) // 10)
SEED = CFG["seed"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def p(*parts) -> Path:
    """Path relative to project root; creates parent folders for outputs."""
    path = ROOT.joinpath(*parts)
    return path


def out_dir(sub: str) -> Path:
    d = ROOT / CFG["paths"]["outputs"] / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir() -> Path:
    d = ROOT / CFG["paths"]["processed"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------- metrics
def logit(pr, eps: float = 1e-6):
    pr = np.clip(np.asarray(pr, dtype=float), eps, 1 - eps)
    return np.log(pr / (1 - pr))


class PlattCalibrator:
    """Sigmoid (Platt) calibration fitted on the validation set only."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, raw_p, y):
        self.lr.fit(logit(raw_p).reshape(-1, 1), y)
        return self

    def predict(self, raw_p):
        return self.lr.predict_proba(logit(raw_p).reshape(-1, 1))[:, 1]


def calibration_slope(y, pr) -> float:
    """Slope of logistic regression of outcome on logit(p); 1.0 = ideal."""
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit(pr).reshape(-1, 1), y)
    return float(lr.coef_[0, 0])


def topk_metrics(y, score, frac: float) -> dict:
    """Flag the top `frac` of students by risk; return precision and recall."""
    y = np.asarray(y)
    n = len(y)
    k = max(1, int(round(frac * n)))
    # random (seeded) tie-breaking: many students can share identical scores (e.g. zero-click rows)
    perm = np.random.default_rng(SEED).permutation(n)
    order = np.lexsort((perm, -np.asarray(score, dtype=float)))[:k]
    tp = y[order].sum()
    pos = y.sum()
    return {"k": k, "precision": float(tp / k), "recall": float(tp / pos) if pos else float("nan"),
            "flagged_dropouts": int(tp)}


def best_f1_threshold(y, pr) -> float:
    """Threshold maximising dropout-class F1 (to be chosen on VALIDATION data)."""
    grid = np.unique(np.quantile(pr, np.linspace(0.02, 0.98, 97)))
    scores = [f1_score(y, (pr >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def core_metrics(y, raw_score, cal_prob=None, f1_thr=None) -> dict:
    """PR-AUC, ROC-AUC (rank-based on raw score), Brier + slope (calibrated), F1 (dropout class)."""
    y = np.asarray(y)
    out = {"n": int(len(y)), "n_dropouts": int(y.sum()), "base_rate": float(y.mean())}
    if y.min() == y.max():
        out.update(pr_auc=np.nan, roc_auc=np.nan)
    else:
        out["pr_auc"] = float(average_precision_score(y, raw_score))
        out["roc_auc"] = float(roc_auc_score(y, raw_score))
    if cal_prob is not None:
        out["brier"] = float(brier_score_loss(y, cal_prob))
        out["cal_slope"] = calibration_slope(y, cal_prob)
        if f1_thr is not None:
            out["f1_dropout"] = float(f1_score(y, (np.asarray(cal_prob) >= f1_thr).astype(int), zero_division=0))
    out["brier_majority"] = float(brier_score_loss(y, np.full(len(y), y.mean())))
    return out


def bootstrap_ci(values, n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean of `values`."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
    return (float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def bootstrap_metric_ci(y, score, fn, n_boot: int = 1000, seed: int = SEED):
    """Bootstrap CI of fn(y, score) resampling students."""
    y = np.asarray(y)
    score = np.asarray(score)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], score[idx]))
    vals = np.array(vals)
    return (float(fn(y, score)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def fmt_ci(m, lo, hi, d: int = 3) -> str:
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def write_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    d = out_dir("tables")
    df.to_csv(d / f"{name}.csv", index=index)
    try:
        (d / f"{name}.md").write_text(df.to_markdown(index=index), encoding="utf-8")
    except ImportError:  # tabulate missing
        (d / f"{name}.md").write_text(df.to_string(index=index), encoding="utf-8")
