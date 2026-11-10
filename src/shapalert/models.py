"""Model factory: baselines, gradient boosting, Optuna search spaces, small-data variants."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .common import CFG, SEED

GBM_NAMES = ["hgb", "xgboost", "lightgbm", "catboost"]


def spw(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max(1, (y == 1).sum()))


def baseline(name: str):
    if name == "majority":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler()),
                         ("m", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))])
    if name == "tree":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                                      class_weight="balanced", random_state=SEED))])
    if name == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                      class_weight="balanced_subsample", random_state=SEED))])
    raise ValueError(name)


def default_params(name: str) -> dict:
    return {
        "hgb": dict(learning_rate=0.05, max_depth=4, max_iter=300, min_samples_leaf=20, l2_regularization=1.0),
        "xgboost": dict(learning_rate=0.05, max_depth=4, n_estimators=300, min_child_weight=5, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0),
        "lightgbm": dict(learning_rate=0.05, max_depth=4, n_estimators=300, min_child_samples=20,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
        "catboost": dict(learning_rate=0.05, depth=4, iterations=CFG["tuning"]["catboost_iterations"], l2_leaf_reg=3.0),
    }[name]


def suggest_params(name: str, trial) -> dict:
    """Optuna search space (small, as in the guide)."""
    if name == "hgb":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    max_depth=trial.suggest_int("max_depth", 2, 8),
                    max_iter=trial.suggest_int("max_iter", 100, 400, step=50),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 100),
                    l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True))
    if name == "xgboost":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    max_depth=trial.suggest_int("max_depth", 2, 8),
                    n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10, log=True))
    if name == "lightgbm":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    max_depth=trial.suggest_int("max_depth", 2, 8),
                    n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10, log=True))
    if name == "catboost":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    depth=trial.suggest_int("depth", 2, 8),
                    iterations=CFG["tuning"]["catboost_iterations"],
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True))
    raise ValueError(name)


def make_gbm(name: str, params: dict, y_train=None, class_weight: bool = True):
    """Gradient boosting with class-imbalance handled INSIDE the model (no resampling)."""
    w = spw(y_train) if (class_weight and y_train is not None) else 1.0
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced" if class_weight else None,
                                              **params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=SEED, scale_pos_weight=w, eval_metric="aucpr", n_jobs=4,
                             tree_method="hist", verbosity=0, **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, scale_pos_weight=w, n_jobs=4, verbose=-1, **params)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=SEED, scale_pos_weight=w, verbose=0, thread_count=4,
                                  allow_writing_files=False, **params)
    raise ValueError(name)


def small_data_params(name: str) -> dict:
    """Strongly limited capacity for the Ghana sample (guide step 3.2)."""
    c = CFG["ghana_small_model"]
    d, n, lr, mcs = c["max_depth"], c["n_estimators"], c["learning_rate"], c["min_child_samples"]
    return {
        "hgb": dict(learning_rate=lr, max_depth=d, max_iter=n, min_samples_leaf=mcs, l2_regularization=1.0,
                    early_stopping=True, n_iter_no_change=c["early_stopping_rounds"], validation_fraction=0.15),
        "xgboost": dict(learning_rate=lr, max_depth=d, n_estimators=n, min_child_weight=mcs / 2, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=5.0),
        "lightgbm": dict(learning_rate=lr, max_depth=d, n_estimators=n, min_child_samples=mcs, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0),
        "catboost": dict(learning_rate=lr, depth=d, iterations=n, l2_leaf_reg=10.0),
    }[name]


def fit_predict_small(name: str, Xtr, ytr, Xte, sample_weight=None, seed_val: int = SEED):
    """Fit a small-capacity GBM with early stopping on an inner validation split of the TRAINING fold only."""
    from sklearn.model_selection import train_test_split
    params = small_data_params(name)
    rounds = CFG["ghana_small_model"]["early_stopping_rounds"]
    idx = np.arange(len(Xtr))
    tr_i, va_i = train_test_split(idx, test_size=0.15, stratify=ytr, random_state=seed_val)
    Xa, ya = Xtr.iloc[tr_i], np.asarray(ytr)[tr_i]
    Xv, yv = Xtr.iloc[va_i], np.asarray(ytr)[va_i]
    sw = None if sample_weight is None else np.asarray(sample_weight)
    swa = None if sw is None else sw[tr_i]
    if name == "hgb":
        m = make_gbm("hgb", params, ytr)
        m.fit(Xtr, ytr, sample_weight=sw)
    elif name == "lightgbm":
        import lightgbm as lgb
        m = make_gbm("lightgbm", params, ya)
        m.fit(Xa, ya, sample_weight=swa, eval_set=[(Xv, yv)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(rounds, verbose=False)])
    elif name == "xgboost":
        m = make_gbm("xgboost", {**params, "early_stopping_rounds": rounds}, ya)
        m.fit(Xa, ya, sample_weight=swa, eval_set=[(Xv, yv)], verbose=False)
    elif name == "catboost":
        m = make_gbm("catboost", {**params, "early_stopping_rounds": rounds}, ya)
        m.fit(Xa, ya, sample_weight=swa, eval_set=(Xv, yv))
    else:
        raise ValueError(name)
    return m, m.predict_proba(Xte)[:, 1]
