# Materials and methods

## Study design and overview

We developed one early-warning pipeline for course withdrawal and evaluated it in three steps. First, we built and tuned the pipeline on the public Open University Learning Analytics Dataset (OULAD), following the deployment-oriented protocol of Wu et al. [47]: a 28-day early window, group-wise evaluation by course presentation, probability calibration, capacity-based Top-x% alerting, selective prediction (abstention), and a subgroup fairness audit with bootstrap uncertainty. We extended that protocol with a time-aware split, additional gradient boosting models, SHAP explanations and LEAF scoring of the explanations. Second, we froze the pipeline (features, model family, hyper-parameters, random seed and code hashes) before any modelling on Ghanaian data. Third, we applied the frozen pipeline to Moodle records from Kwame Nkrumah University of Science and Technology (KNUST) in four experiments: zero-shot transfer, local training, fine-tuning and pooled training. The pipeline was never modified after Ghanaian results were seen, with one disclosed exception (the Ghana label definition, see *Outcome definition*).

The study addresses three research questions: **RQ1** which gradient boosting model gives the best calibrated early warning on OULAD at 14, 28 and 42 days; **RQ2** how well the OULAD-built pipeline works on Ghanaian LMS data, alone and with fine-tuning; **RQ3** which LMS behaviours drive dropout risk (SHAP), and how stable and actionable those explanations are (LEAF and bootstrap stability) on OULAD and on KNUST data.

## Data sources

### OULAD

OULAD contains 32,593 student registrations (one row per student in one module presentation) across seven modules and four presentations (2013B, 2013J, 2014B, 2014J), with demographic and registration records, daily VLE click counts (`studentVle`), VLE resource metadata (`vle`), and assessment due dates and submissions (`assessments`, `studentAssessment`). We used the seven standard tables as released and confirmed all file and column names against the download.

### KNUST Moodle data

KNUST Moodle exports were supplied by the university as three pseudonymised CSV files: the activity log (950,628 event rows, 4,450 distinct participants, event time, activity type and access source), the gradebook (graded results for six module quizzes) and quiz attempts (per-attempt state, score and timestamps). Participant identifiers are pseudonymous codes (`P00001` style); the files contain no names or index numbers. After removing 2,109 exact duplicate log rows, 4,450 students remained. The export has no enrolment table, no year-of-study field, no forum or assignment records and no demographic fields, so the analysis uses behavioural LMS features only. The log spans several years but the gradebook covers only 2026 (18 January to 12 May); we therefore treated 2026 as a single course presentation with start date 12 January 2026.

## Ethics and data protection

Ghanaian data are handled under the Data Protection Act 2012 (Act 843) [13]. Analysis files contain pseudonymised identifiers only; the key linking identifiers to students is held by the university and not by the analyst. Raw exports are read-only and excluded from version control. Ethical clearance for the study was obtained from the Humanities and Social Sciences Research Ethics Committee (HUSSREC) of KNUST, and the Office of the Registrar approved the release of the LMS data for this research on 4 August 2026. OULAD is public and fully anonymised.

## Outcome definition

*OULAD.* A registration is a dropout (y = 1) if `final_result` is "Withdrawn" and zero otherwise, following Wu et al. Students with "Fail" are not counted as dropouts.

*KNUST.* Dropout (y = 1) follows the proposal rule: no login for at least 21 consecutive days up to the end of the export (20 June 2026) **and** at least two consecutive missed graded assessments (module n and n+1 quizzes both without a graded result). Because the export has no due dates, a "missed assessment" is a module quiz with no graded result, and the 21-day gap is counted at the end of the observation period. Inactivity-based labels have precedent in dropout studies [23]. This yields 790 dropouts among 4,450 students (17.8%). The rule follows the study proposal, which predates the modelling. An initial analysis used non-completion of the six module quizzes (30.6% dropouts) as the label; because the label was changed after seeing its results, that analysis is reported as a sensitivity analysis. The label is built from outcome records only and is never used to construct features.

## Early windows and leakage control

For each window T in {14, 28, 42} days after course start, only events dated before day T were used (events from up to 14 days before the start were retained for both datasets). OULAD registrations that unregistered before day T were removed, because their outcome is already known. Assessment and quiz records were used only if submitted or finished before T. All the above are enforced by assertions in the feature builder and by automated tests (events after T do not change features; unregistered-before-T students are absent; label columns never enter the Ghana feature table). Students with no activity were kept with zero-valued features.

## Feature construction

The same recipe was applied to both datasets (Table M1). *Shared* features exist in both datasets and are scale-free: activity counts are standardised within each course (z-scores computed without labels), and other features are ratios or fractions of the window length. The *full* OULAD set adds features available only in OULAD (VLE resource types, distinct resources, demographics).

**Table M1. Features used in the pipeline.**

| Group | Feature | Definition | Shared |
|---|---|---|---|
| Volume | clicks_total (z) | Events/clicks in the window | yes |
| | active_days (z), active_ratio | Distinct active days; share of the window | yes |
| | clicks_per_active_day (z) | Clicks divided by active days | yes |
| Recency | gap_frac | Days since last click divided by window length | yes |
| | first_click_frac | Day of first click, scaled to [0, 1] | yes |
| Trend | clicks_last_week (z), click_trend (z) | Last-week clicks; last week minus first week | yes |
| Assessment | n_early_assess (z), has_early_assess, early_score_mean | Quizzes/assessments finished before T; their mean score | yes |
| OULAD only | clicks by type (forum, quiz, resource, homepage), distinct_resources | From `vle.activity_type` | no |
| OULAD only | gender, age band, highest education, disability, previous attempts, studied credits, IMD band | Registration records (ordinal/binary coded) | no |

For KNUST, one log row was counted as one click; activity type and resource identifiers were not used for transfer because OULAD and Moodle categories are not equivalent. Missing values (for example scores when no assessment was submitted) were left missing for tree models and median-imputed with a missing indicator for logistic regression.

## Data splits

Two evaluation designs were used on OULAD. In the **group split**, entire course presentations are held out (20% of the 22 presentations for testing and 20% for validation, chosen once with a fixed seed), so the test set is a course never seen in training [47]. In the **time-aware split**, models were trained on the 2013 presentations (2013B, 2013J), tuned and calibrated on 2014B, and tested on 2014J, mimicking prediction for the next semester. Tuning and calibration used the validation set only; the test set was used once per model for reporting.

## Models, class imbalance and tuning

Baselines were a majority-class predictor, logistic regression, a decision tree and a random forest. Gradient boosting models were scikit-learn histogram gradient boosting (HGB) [37], XGBoost [8], LightGBM [24] and CatBoost [41]. Class imbalance was handled inside each model (`class_weight` or `scale_pos_weight`); no resampling was applied before splitting. As an ablation, SMOTE [7] was applied to the training portion only, inside an imbalanced-learn pipeline, with LightGBM. Each boosting model was tuned with Optuna [1] (TPE sampler, 30 trials, seed 42), maximising PR-AUC on the validation set; search spaces covered learning rate, depth, number of trees, minimum leaf size, subsampling and regularisation. Every model was run at 14, 28 and 42 days. The seed was fixed at 42 throughout.

## Calibration and alerting

Predicted probabilities were calibrated with sigmoid (Platt) scaling [38] fitted on the validation set (sigmoid scaling is preferred to isotonic regression when data are scarce [35]), and evaluated by Brier score, calibration slope and calibration plots. To reflect adviser capacity, the top 10%, 20% and 30% of students by risk were flagged, with ties broken at random under a fixed seed, and precision and recall were reported at each level. For selective prediction, calibrated risks between 0.35 and 0.65 were treated as "uncertain, check manually"; we report the share of students abstained on, accuracy on the remainder and the dropout share among abstentions. The dropout-class F1 used a threshold chosen on the validation set.

## Sensitivity analyses

*Early unregistrations.* The main OULAD pipeline removes students who unregistered before day T. Because Wu et al. do not state how these students were handled, and their reported withdrawal rates (26% to 32%) are close to the full-dataset rate (31.2%), we also ran the 28-day time-aware evaluation with all registrations kept, using a fixed-setting HGB model (`scripts/09_wu_comparable_sensitivity.py`). This is a comparison, not a deployable setting, because early unregistrations are already known at prediction time.

*Demographic predictors.* The full OULAD model uses gender, age band, disability and deprivation band as predictors. As a post-hoc check on the fairness audit, we retrained the frozen HGB settings without these four attributes, and also scored the frozen behaviour-only (shared-feature) model, and repeated the audit on the 2014J test set (`scripts/12_oulad_fairness_no_demographics.py`). The frozen pipeline was not changed.

*Class balancing.* A LightGBM model with SMOTE applied to training data only, inside a pipeline, was compared with the class-weighted model.

*Ghana label.* All Ghana experiments were repeated under the earlier non-completion label (fewer than six of six module quizzes graded) and the results are kept alongside the main results.

## Fairness audit (OULAD)

Using the frozen model at 28 days on the time-aware test set, students in the top 20% by risk were flagged and recall and false-positive rate were computed by gender, disability and age band. Uncertainty was quantified with 1,000 bootstrap resamples; group gaps were taken against the largest group. Groups with fewer than 30 students or fewer than 10 dropouts were reported as too small to audit. Fairness was not assessed on KNUST data because the export contains no demographic fields.

## Freezing the pipeline

Before any Ghanaian modelling, the model family for each window was selected by highest validation PR-AUC, and the feature lists, tuned hyper-parameters, seed, code file hashes and software versions were written to a frozen specification (`outputs/frozen/frozen_pipeline.json`). The model family carried to Ghana was the best shared-feature model at 28 days (HGB).

## Ghanaian experiments

Because the shared features must exist in both datasets, transfer experiments used the shared feature set at 28 days; earliness was assessed at all three windows. Models for the Ghanaian data used small capacity (maximum depth 3 for boosting, learning rate 0.05, minimum 10 samples per leaf, up to 300 trees with early stopping on a 15% inner split of the training fold only), and logistic regression was always run alongside. Four experiments were run: **A** zero-shot (OULAD model with frozen hyper-parameters, trained on all OULAD registrations and applied to all KNUST students); **B** local (trained and tested on KNUST only); **C** fine-tuning by stacking (the OULAD model's log-odds as an extra feature for the KNUST model); **D** pooled (OULAD plus KNUST training rows, KNUST rows weighted five times). Experiments B to D used repeated stratified five-fold cross-validation with ten repeats (50 test folds), so each dropout share is preserved. Performance was summarised by PR-AUC (compared with the dropout rate), ROC-AUC, recall and precision at the top 10/20/30%, dropout-class F1 (flagging as many students as the dropout rate) and Brier score (after cross-fitted Platt calibration). Confidence intervals are percentile bootstrap intervals over the repeats (experiment A: over students, 1,000 resamples). The majority-class baseline and the number of dropouts per test fold are always reported, and accuracy is not used as a headline metric. No tuning was done on test folds. Because many institutions have only a few hundred students per course, a sensitivity analysis repeated the experiments on five random stratified draws of 300 students. Earliness was assessed by repeating experiment B at 14, 28 and 42 days.

## Explanation methods

SHAP values [29] were computed on held-out data only: OULAD time-aware test set, and a stratified 30% hold-out of the KNUST data. TreeExplainer [30] was used for XGBoost, LightGBM and CatBoost; for scikit-learn HGB, which TreeExplainer does not support, the permutation explainer was used. SHAP values are on the log-odds scale of the base model; Platt calibration is monotone, so feature rankings are unchanged. We report global importance (mean absolute SHAP, beeswarm), three local waterfall plots (a correctly flagged dropout, a missed dropout, a false alarm, each the median-risk case of its type), importance across the three windows, and the rank agreement (Spearman) between OULAD and KNUST for the shared features.

Explanations were scored with four checks modelled on the LEAF framework of Amparore et al. [4], which Albugami et al. [3] applied to dropout explanations. LEAF was designed for local linear explainers, and we implemented the four checks ourselves for SHAP on a gradient boosting model, so our scores are LEAF-style and not outputs of the original implementation. *Fidelity*: a weighted linear surrogate on the top five SHAP features, fitted on 200 perturbed neighbours of each student, must reproduce the model's alert class (reported with and without neighbourhoods in which all neighbours share one class). *Local concordance*: one minus the mean absolute difference between surrogate and model risk over the 20 most similar real students. *Prescriptivity*: the share of the top three SHAP features that an adviser can change (activity and assessment) rather than fixed traits (demographics). *Stability*: the model was refitted on 100 bootstrap resamples of the training data, SHAP recomputed on a fixed test subsample, and the average Spearman correlation between importance rankings reported; features outside the top ten in more than 20% of resamples were labelled tentative. This is a model-level, bootstrap-refit measure of rank stability and differs from LEAF's reiteration similarity, which measures how consistently an explainer returns the same explanation when it is re-run on the same data point [4], so we call it *bootstrap rank stability*. The checks were applied to 100 random OULAD test students and to all KNUST hold-out students.

## Software and reproducibility

Analyses used Python 3.10.11 with pandas 2.3.3, scikit-learn 1.7.2 [37], XGBoost 3.2.0, LightGBM 4.7.0, CatBoost 1.2.10, SHAP 0.49.1, Optuna 5.0.0, imbalanced-learn 0.14.2 [28], SciPy 1.15.3 and matplotlib 3.10.8 (exact pins in `requirements-lock.txt`). Reporting follows the TRIPOD+AI guidance [10] where applicable; a TRIPOD+AI checklist is provided as Supplementary File S1. All settings are in `config.yaml`, scripts run in a fixed order (`run_all.py`), raw data are read-only, and automated tests cover leakage and split integrity. Code and aggregate outputs can be shared; student-level data cannot.

## Limitations of the design

KNUST data come from one course presentation and one institution, the export lacks forum, assignment, enrolment and demographic information (so fairness cannot be audited on KNUST data and a missed assessment is defined by absent grades), and the dropout label is derived from the same LMS sources as the features, so later windows partly measure the label. SHAP shows what the model uses, not why students leave. These limits should be kept in mind when reading the results.

## References (methods)

1. Wu S, Duan J, Luo M. Calibrated early-warning models with fairness auditing and selective prediction for course withdrawal risk: evidence from OULAD. PLOS ONE. 2026;21(7). doi:10.1371/journal.pone.0352867. [Listing and abstract only were read; check the full text for its exact section headings.]
2. Chow 2024; Al Hashmi et al. 2025; Albugami et al. 2026; Kabathova and Drlik 2021 (as listed in the methodology guide) for SHAP-by-stage, SHAP on gradient boosting, LEAF, and small-data checks.
