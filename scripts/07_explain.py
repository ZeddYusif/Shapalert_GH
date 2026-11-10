"""Explain: SHAP (global, local, across windows), OULAD-vs-Ghana comparison, LEAF scores, bootstrap stability.

SHAP is computed on TEST data only, on the frozen model's log-odds output. Platt calibration is a monotone
transform of that output, so feature rankings are unchanged by calibration.
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
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from shapalert import features as F
from shapalert import models as M
from shapalert.common import CFG, SEED, load_json, logit, out_dir, processed_dir, seed_everything, write_table

warnings.filterwarnings("ignore")
seed_everything()
FZ = load_json(out_dir("frozen") / "frozen_pipeline.json")
EX = CFG["explain"]
PRIMARY = CFG["primary_window"]
FIG = out_dir("figures")
rng = np.random.default_rng(SEED)

ACTION = {  # plain-language advisor reading for changeable features
    "gap_frac": "has not logged in recently -> contact the student this week",
    "days_since_last_click": "has not logged in recently -> contact the student this week",
    "clicks_last_week": "little recent activity -> check for access or data-cost problems",
    "click_trend": "activity is falling -> early check-in",
    "active_ratio": "few active days -> offer an activity plan",
    "active_days": "few active days -> offer an activity plan",
    "clicks_total": "low overall engagement -> invite to a study group",
    "early_score_mean": "weak early quiz scores -> point to remedial resources",
    "n_early_assess": "few early quizzes completed -> remind about deadlines",
    "has_early_assess": "no early quiz attempted -> remind about deadlines",
    "first_click_frac": "started late -> check enrolment/onboarding problems",
    "first_click_day": "started late -> check enrolment/onboarding problems",
}


# ------------------------------------------------------------------ SHAP helpers
def shap_matrix(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (n, p) for the dropout class on the log-odds scale."""
    if type(model).__name__ == "HistGradientBoostingClassifier":   # no TreeExplainer support
        f = lambda d: logit(model.predict_proba(pd.DataFrame(d, columns=X.columns))[:, 1])
        bg = X.sample(min(100, len(X)), random_state=SEED)
        return shap.explainers.Permutation(f, bg, max_evals=2 * X.shape[1] + 1, seed=SEED)(X, silent=True).values
    v = shap.TreeExplainer(model).shap_values(X)
    if isinstance(v, list):
        v = v[1]
    v = np.asarray(v)
    return v[:, :, 1] if v.ndim == 3 else v


def base_value(model, X):
    if type(model).__name__ == "HistGradientBoostingClassifier":
        return float(logit(model.predict_proba(X)[:, 1]).mean())
    ev = shap.TreeExplainer(model).expected_value
    return float(np.ravel(ev)[-1])


def importance(sv, cols):
    return pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)


def global_plots(sv, X, tag, title):
    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / f"fig_shap_bar_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / f"fig_shap_beeswarm_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()


def sentence(feat, sv_col, x):
    d = "higher" if sv_col > 0 else "lower"
    return f"{feat}={x:.2f}: pushes risk {d} (SHAP {sv_col:+.2f})"


def local_cases(model, X, y, sv, tag, score):
    thr = np.sort(score)[::-1][int(round(0.2 * len(score))) - 1]
    flag = score >= thr
    groups = {"correctly_flagged_dropout": np.where(flag & (y == 1))[0],
              "missed_dropout": np.where(~flag & (y == 1))[0],
              "false_alarm": np.where(flag & (y == 0))[0]}
    lines = [f"# Local explanations ({tag}) - median-risk case of each type\n"]
    bv = base_value(model, X)
    for name, idx in groups.items():
        if len(idx) == 0:
            continue
        i = idx[np.argsort(score[idx])[len(idx) // 2]]
        exp = shap.Explanation(sv[i], base_values=bv, data=X.iloc[i].values, feature_names=list(X.columns))
        plt.figure()
        shap.plots.waterfall(exp, show=False, max_display=8)
        plt.savefig(FIG / f"fig_waterfall_{tag}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        top = np.argsort(-np.abs(sv[i]))[:3]
        reads = [sentence(X.columns[j], sv[i][j], X.iloc[i, j]) for j in top]
        acts = [ACTION[X.columns[j].replace("_z", "")] for j in top if X.columns[j].replace("_z", "") in ACTION and sv[i][j] > 0]
        acts = acts or [ACTION[X.columns[j].replace("_z", "")] for j in top if X.columns[j].replace("_z", "") in ACTION]
        lines += [f"## {name} (row {i}, risk score {score[i]:.3f}, outcome={'dropout' if y[i] else 'stayed'})",
                  "Top reasons: " + "; ".join(reads),
                  "Advisor reading: " + (acts[0] if acts else "reasons are fixed traits; no direct action, monitor.") +
                  " (This is what the model uses, not proof of cause.)", ""]
    (out_dir("tables") / f"local_explanations_{tag}.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ models
def oulad_split(T):
    ts = CFG["oulad"]["time_split"]
    d = pd.read_csv(processed_dir() / f"oulad_w{T}.csv")
    return d[d.code_presentation.isin(ts["train"])], d[d.code_presentation.isin(ts["test"])]


def oulad_model(T, fs):
    name = FZ["best"][f"{fs}|{T}"]["model"]
    return name, joblib.load(out_dir("models") / f"time_{fs}_w{T}_{name}.joblib")


def ghana_holdout(T, feats):
    g = pd.read_csv(processed_dir() / f"ghana_w{T}.csv")
    tr, te = train_test_split(np.arange(len(g)), test_size=0.3, stratify=g["y"], random_state=SEED)
    fam = FZ["transfer_model_family"]
    Xtr, ytr, Xte = g.iloc[tr][feats], g.iloc[tr]["y"].values, g.iloc[te][feats]
    model, score = M.fit_predict_small(fam, Xtr, ytr, Xte)
    return g.iloc[tr], g.iloc[te], model, score


def ghana_feats(T):
    return F.SHARED_FEATURES


# ------------------------------------------------------------------ LEAF
def leaf_scores(predict_prob, X, sv, thr_prob, sample_idx, k, fixed):
    """Fidelity, local concordance, prescriptivity for the instances in sample_idx (positions in X)."""
    Xv = X.values.astype(float)
    sd = np.nanstd(Xv, axis=0) + 1e-9
    Z = (np.nan_to_num(Xv - np.nanmean(Xv, axis=0))) / sd          # for neighbour search
    cols = list(X.columns)
    t_logit = float(logit(thr_prob))
    fid, fid_nd, conc, presc = [], [], [], []
    r = np.random.default_rng(SEED)
    for i in sample_idx:
        top = np.argsort(-np.abs(sv[i]))[:k]
        # prescriptivity: share of top-3 features an advisor can change
        presc.append(np.mean([cols[j] not in fixed for j in np.argsort(-np.abs(sv[i]))[:3]]))
        # perturbation neighbourhood varying only the explanation's top-k features
        P = np.repeat(Xv[i][None, :], 200, axis=0)
        P[:, top] += r.normal(0, 0.5, (200, len(top))) * sd[top]
        pm = predict_prob(pd.DataFrame(P, columns=cols))
        lm = logit(pm)
        Dz = np.nan_to_num((P[:, top] - Xv[i, top]) / sd[top])      # NaN (missing) features stay unperturbed
        A = np.c_[np.ones(200), Dz]
        w = np.exp(-np.sum(Dz ** 2, axis=1) / (2 * len(top)))
        beta = np.linalg.lstsq(A * np.sqrt(w)[:, None], lm * np.sqrt(w), rcond=None)[0]
        agree = ((A @ beta) >= t_logit) == (lm >= t_logit)
        fid.append(np.average(agree, weights=w))
        if len(np.unique(lm >= t_logit)) > 1:
            fid_nd.append(np.average(agree, weights=w))
        # local concordance: surrogate risk vs model risk for the 20 most similar REAL students
        d = np.sum((Z - Z[i]) ** 2, axis=1)
        nn = np.argsort(d)[1:21]
        An = np.c_[np.ones(len(nn)), np.nan_to_num((Xv[nn][:, top] - Xv[i, top]) / sd[top])]
        sur = 1 / (1 + np.exp(-(An @ beta)))
        conc.append(1 - np.mean(np.abs(sur - predict_prob(X.iloc[nn]))))
    return {"fidelity": np.mean(fid), "fidelity_non_degenerate": np.mean(fid_nd) if fid_nd else np.nan,
            "share_degenerate_neighbourhoods": 1 - len(fid_nd) / len(sample_idx),
            "local_concordance": np.mean(conc), "prescriptivity": np.mean(presc)}


def stability(fit_fn, Xtr, ytr, Xte_small, n_boot):
    """Refit on bootstrap resamples of the TRAINING data; SHAP on a fixed test subsample; compare rankings."""
    imps = []
    r = np.random.default_rng(SEED)
    for b in range(n_boot):
        idx = r.integers(0, len(Xtr), len(Xtr))
        try:
            m = fit_fn(Xtr.iloc[idx].reset_index(drop=True), ytr[idx])
            imps.append(importance(shap_matrix(m, Xte_small), Xte_small.columns).reindex(Xte_small.columns))
        except Exception as e:   # a degenerate resample should not stop the audit
            print("bootstrap fit failed:", e)
    I = pd.DataFrame(imps)
    k = EX["top_k"]
    ranks = I.rank(axis=1, ascending=False)
    pairs = []
    arr = I.values
    for a in range(len(arr)):
        for b in range(a + 1, min(len(arr), a + 6)):    # 5 partners each: 500 pairs for 100 runs
            pairs.append(spearmanr(arr[a], arr[b])[0])
    top10_freq = (ranks <= min(k, I.shape[1])).mean()
    return float(np.nanmean(pairs)), top10_freq, I


def main():
    fam = FZ["transfer_model_family"]
    nb = EX["n_boot"]
    leaf_rows, top10_tables = [], {}

    # ============== OULAD, full features, frozen best model at each window ==============
    imp_win = {}
    for T in CFG["windows"]:
        _, te = oulad_split(T)
        name, model = oulad_model(T, "full")
        X = te[F.FULL_FEATURES].sample(min(EX["shap_oulad_max_rows"] if name != "hgb" else 1500, len(te)),
                                       random_state=SEED)
        y = te.loc[X.index, "y"].values
        sv = shap_matrix(model, X)
        imp_win[T] = importance(sv, X.columns)
        if T == PRIMARY:
            global_plots(sv, X, "oulad_full_w28", f"OULAD {name}, {T} days (test 2014J)")
            score = model.predict_proba(X)[:, 1]
            local_cases(model, X, y, sv, "oulad_w28", score)
            thr = np.sort(score)[::-1][int(round(0.2 * len(score))) - 1]
            sample = rng.choice(len(X), EX["leaf_oulad_sample"], replace=False)
            sc = leaf_scores(lambda d: model.predict_proba(d)[:, 1], X, sv, thr, sample, EX["leaf_top_features"],
                             F.FIXED_FEATURES)
            tr, _ = oulad_split(T)
            params = FZ["best"][f"full|{T}"]["params"]
            stab, freq, _ = stability(lambda Xb, yb: M.make_gbm(name, params, yb).fit(Xb, yb),
                                      tr[F.FULL_FEATURES], tr["y"].values, X.iloc[:300], nb)
            tent = list(freq[freq < EX["stability_tentative_threshold"]].index)
            leaf_rows.append({"dataset": f"OULAD full ({name}, {T}d)", **sc, "stability_spearman": stab,
                              "n_tentative_features": len(tent), "tentative": ", ".join(tent)})
            top10_tables["oulad_full"] = freq
    imp = pd.DataFrame(imp_win)
    imp.index.name = "feature"
    write_table(imp.round(4).reset_index(), "shap_importance_across_windows_oulad")
    ax = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index[:10]].T.plot(marker="o", figsize=(7, 4))
    ax.set_xlabel("Days after course start")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("OULAD: SHAP importance by window")
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / "fig_shap_across_windows_oulad.png", dpi=150)
    plt.close()

    # ============== OULAD shared-feature model (for the comparison with Ghana) ==============
    _, te = oulad_split(PRIMARY)
    name_s, model_s = oulad_model(PRIMARY, "shared")
    Xs = te[F.SHARED_FEATURES].sample(min(3000 if name_s != "hgb" else 1500, len(te)), random_state=SEED)
    imp_o = importance(shap_matrix(model_s, Xs), Xs.columns)

    # ============== Ghana ==============
    lms = ghana_feats(PRIMARY)
    imp_g_win = {}
    for T in CFG["windows"]:
        gtr, gte, gm, gscore = ghana_holdout(T, lms)
        Xg, yg = gte[lms], gte["y"].values
        sv = shap_matrix(gm, Xg)
        imp_g_win[T] = importance(sv, lms)
        if T == PRIMARY:
            global_plots(sv, Xg, "ghana_w28", f"Ghana {fam}, {T} days (hold-out 30%)")
            local_cases(gm, Xg, yg, sv, "ghana_w28", gscore)
            thr = np.sort(gscore)[::-1][int(round(0.2 * len(gscore))) - 1]
            sample = np.arange(len(Xg)) if len(Xg) <= 1500 else rng.choice(len(Xg), 1500, replace=False)
            sc = leaf_scores(lambda d: gm.predict_proba(d)[:, 1], Xg, sv, thr, sample, EX["leaf_top_features"],
                             F.FIXED_FEATURES)
            stab, freq, _ = stability(lambda Xb, yb: M.fit_predict_small(fam, Xb, yb, Xb.iloc[:5])[0],
                                      gtr[lms].reset_index(drop=True), gtr["y"].values, Xg.iloc[:300], nb)
            tent = list(freq[freq < EX["stability_tentative_threshold"]].index)
            leaf_rows.append({"dataset": f"Ghana LMS ({fam}, {T}d)", **sc, "stability_spearman": stab,
                              "n_tentative_features": len(tent), "tentative": ", ".join(tent)})
            top10_tables["ghana_lms"] = freq
            imp_g = imp_g_win[T]
    impg = pd.DataFrame(imp_g_win)
    impg.index.name = "feature"
    write_table(impg.round(4).reset_index(), "shap_importance_across_windows_ghana")

    # ============== 4.4 OULAD vs Ghana ==============
    both = pd.DataFrame({"oulad_mean_abs_shap": imp_o, "ghana_mean_abs_shap": imp_g})
    both["oulad_rank"] = both.oulad_mean_abs_shap.rank(ascending=False)
    both["ghana_rank"] = both.ghana_mean_abs_shap.rank(ascending=False)
    rho, pval = spearmanr(both.oulad_rank, both.ghana_rank)
    both = both.sort_values("oulad_rank")
    write_table(both.round(4).reset_index().rename(columns={"index": "feature"}), "shap_oulad_vs_ghana")
    (out_dir("tables") / "shap_oulad_vs_ghana_spearman.txt").write_text(
        f"Spearman rank correlation of mean|SHAP| across {len(both)} shared features: rho={rho:.3f} (p={pval:.3f})\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=False)
    for ax, col, ttl in [(axes[0], "oulad_mean_abs_shap", f"OULAD ({name_s})"), (axes[1], "ghana_mean_abs_shap", f"Ghana ({fam})")]:
        s = both[col].sort_values().tail(10)
        ax.barh(s.index, s.values)
        ax.set_title(ttl + " - top 10")
        ax.set_xlabel("Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(FIG / "fig_oulad_vs_ghana_top_features.png", dpi=150)
    plt.close()
    print(f"OULAD vs Ghana Spearman rho={rho:.3f}")

    # ============== Table 7 ==============
    t7 = pd.DataFrame(leaf_rows)
    write_table(t7.round(3), "table7_leaf_and_shap_stability")
    for k, v in top10_tables.items():
        write_table(v.rename("share_of_bootstrap_runs_in_top10").round(3).reset_index().rename(columns={"index": "feature"}),
                    f"shap_top10_frequency_{k}")
    print(t7.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
