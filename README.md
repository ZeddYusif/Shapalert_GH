# SHAPAlert-GH - early dropout warning with SHAP explanations (OULAD -> Ghana)

Implements the *SHAPAlert-GH Student methodology guide*: one pipeline run on OULAD (Phase 1), frozen, then
re-used unchanged on the Ghanaian LMS data (Phases 2-3), with SHAP/LEAF explanations (Phase 4).

**Golden rule:** finish and freeze Phase 1 (`03_freeze.py`) before touching any Ghana modelling script.
Do not change the pipeline after seeing Ghana results; report any change as a post-hoc step.

## Run order

```bash
pip install -r requirements.txt
python run_all.py --phase1      # 01 build OULAD -> 02 models -> 03 FREEZE -> 04 fairness
python run_all.py --phase23     # 05 build Ghana -> 06 experiments (+small-sample) -> 07 SHAP/LEAF -> 08 tables
python -m pytest tests -q       # leakage + split-integrity tests
```

| Step | Script | Guide step | Output |
|---|---|---|---|
| 1 | `scripts/01_build_oulad.py` | 1.1-1.3 | `data/processed/oulad_w{14,28,42}.csv` |
| 2 | `scripts/02_oulad_models.py` | 1.4-1.8 | Tables 2, 3, abstention, SMOTE ablation, calibration + recall-vs-window figures |
| 3 | `scripts/03_freeze.py` | 1.10 | `outputs/frozen/frozen_pipeline.json` (features, params, seed, code hashes) |
| 4 | `scripts/04_oulad_fairness.py` | 1.9 | Table 6 (bootstrap CIs, 1000 resamples) |
| 5 | `scripts/05_build_ghana.py` | 2.4-2.6 | `data/processed/ghana_w*.csv`, quality report, data dictionary, Table 1 |
| 6 | `scripts/06_ghana_experiments.py [--small]` | 3.1-3.5 | Table 4 (A-D), earliness, small-sample sensitivity |
| 7 | `scripts/07_explain.py` | 4.1-4.6 | SHAP figures, local explanations, OULAD-vs-Ghana Spearman, Table 7 (LEAF + stability) |
| 8 | `scripts/08_make_tables.py` | - | `outputs/RESULTS.md` index |
| 9 | `scripts/09_wu_comparable_sensitivity.py` | - | Effect of removing early unregistrations (comparison with Wu et al.) |
| 12 | `scripts/12_oulad_fairness_no_demographics.py` | - | Post-hoc fairness check without demographic predictors |
| 10 | `scripts/10_build_report.py` | - | `outputs/SHAPAlert_GH_Manuscript_Draft.docx` (introduction, literature review, methods from `docs/methodology.md` + results from `outputs/tables`) |

Configuration (seed 42, windows, trial counts, label rule, CV settings) is in `config.yaml`.
Raw data (`oulad/`, `knust_source/`) is read-only and git-ignored.

## Design decisions

* **Label.** OULAD: `Withdrawn`. Ghana: proposal rule = >=21 days without login AND >=2 consecutive missed module quizzes (`docs/label_definition.md`; earlier non-completion label kept as a sensitivity set).
* **Leak-free.** Only events with `day < T`; OULAD students who unregistered before `T` are removed; quiz/assessment records only if finished before `T`;
  class imbalance is handled by `scale_pos_weight`/`class_weight` (SMOTE only as an ablation, inside an `imblearn` Pipeline on training data); tuning and Platt calibration use the validation set only.
* **Two OULAD splits.** Group split (whole course presentations held out, Wu et al.) and time-aware split (train 2013, validate 2014B, test 2014J).
* **Transfer features.** Experiments A, C, D need features that exist in both datasets (`SHARED_FEATURES`): activity counts are z-scored *within each course*
  (label-free), the rest are scale-free ratios. Resource-type and static demographic features exist only in OULAD and are used only in the OULAD-only experiments.
* **Frozen transfer model.** `frozen_pipeline.json -> transfer_model_family` with Phase 1 hyper-parameters, retrained on shared features.
  Ghana models use small capacity (depth 3, lr 0.05, min_child_samples 10, early stopping on an inner split of the training fold) and logistic regression is always run alongside.
* **Experiments (Ghana).** Repeated stratified 5-fold x 10 CV; A zero-shot, B local, C OULAD score as stacking feature, D pooled with Ghana rows weighted x5; mean + bootstrap 95% CI;
  the majority baseline is always shown; Brier uses cross-fitted Platt calibration.
* **Explanations.** SHAP (TreeExplainer; permutation explainer for sklearn HGB) on test data. LEAF is implemented here (original code not used):
  fidelity (weighted linear surrogate on the top-5 SHAP features of perturbed neighbours, class agreement), local concordance (1 - MAE of surrogate vs model risk on the 20 nearest real students),
  prescriptivity (share of top-3 features that are changeable), stability (refit on 100 bootstrap training samples; mean pairwise Spearman; features outside the top 10 in >=20% of runs are "tentative").

## Scope and data limits

The study uses OULAD and the KNUST Moodle export only (`activity_log`, `gradebook_grades`, `quiz_attempts`: 4,450 students, one course presentation).
The export has no enrolment table, no demographic fields and no assignment due dates, so fairness is audited on OULAD only and a "missed assessment" is a quiz with no graded result.
Ethics approval and consent are prerequisites for any data use (`docs/ethics_checklist.md`).

## Changes after the freeze (for transparency)

`outputs/frozen/frozen_pipeline.json` records code hashes at the time of freezing. After the freeze, only Ghana-side settings and reporting changed:
the Ghana label rule (non-completion -> proposal rule; the earlier results are kept in `outputs/archive_label_noncompletion/`), removal of the optional
survey branch (archived in `archive_survey/`), and the report builder. The Phase 1 model selection and hyper-parameters were not changed.

## Supervisor one-liners (fill before each meeting)

Label? Window? What did I freeze (`outputs/frozen/frozen_pipeline.json`)? What number would change my conclusion? What could be leaking (`tests/test_leakage.py`)?
