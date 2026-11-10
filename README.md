# SHAPAlert-GH

**An explainable gradient boosting early-warning system for e-learning dropout prediction in Ghanaian higher education, using OULAD and KNUST Moodle data.**

SHAPAlert-GH reads a student's learning management system (LMS) activity in the first weeks of a course, produces a calibrated dropout risk, flags the highest-risk students up to the number an adviser can contact, and attaches a SHAP explanation to each alert. The pipeline is built and frozen on the public Open University Learning Analytics Dataset (OULAD), then applied unchanged to pseudonymised Moodle records from Kwame Nkrumah University of Science and Technology (KNUST).

## Research questions

| | Question |
|---|---|
| RQ1 | Which gradient boosting model gives the best calibrated early warning on OULAD at 14, 28 and 42 days? |
| RQ2 | How well does the OULAD-built pipeline work on KNUST data, alone and with fine-tuning? |
| RQ3 | Which LMS behaviours drive dropout risk, and how stable and actionable are the explanations? |

## What the pipeline does

1. **Early windows with leakage control.** Only events before day T (14, 28 or 42) are used. OULAD registrations that unregistered before day T are removed, because their outcome is already known.
2. **Shared, scale-free features.** Activity counts are standardised within each course; other features are ratios or fractions of the window. The same recipe runs on both datasets.
3. **Models.** Majority class, logistic regression, decision tree and random forest baselines; HistGradientBoosting, XGBoost, LightGBM and CatBoost, tuned with Optuna (TPE, 30 trials, seed 42) on validation PR-AUC.
4. **Class imbalance.** Handled inside each model with class weights. No resampling before splitting. SMOTE on training data only is run as an ablation.
5. **Calibration and alerting.** Platt scaling on the validation set; alerts for the top 10%, 20% and 30% by risk; abstention for calibrated risks between 0.35 and 0.65.
6. **Fairness audit (OULAD).** Recall and false-positive rate by gender, disability and age band, with 1,000 bootstrap resamples.
7. **Freeze.** Features, model family, hyper-parameters, seed, code hashes and software versions are written to `outputs/frozen/frozen_pipeline.json` before any Ghanaian modelling.
8. **Transfer experiments on KNUST data.**

   | Experiment | Description |
   |---|---|
   | A. Zero-shot | Frozen OULAD model applied to all KNUST students |
   | B. Local | Trained and tested on KNUST only |
   | C. Fine-tuned | OULAD model's log-odds added as a feature for the KNUST model |
   | D. Pooled | OULAD plus KNUST training rows, KNUST weighted five times |

9. **Explanations.** SHAP values on held-out data (global importance, beeswarm, individual waterfalls), scored with LEAF-style checks (fidelity, local concordance, prescriptivity) and bootstrap rank stability over 100 refits.

## Data

| Dataset | Size | Outcome | Access |
|---|---|---|---|
| OULAD | 32,593 registrations, 7 modules, 4 presentations (2013B to 2014J) | `final_result` = "Withdrawn" | Public: https://analyse.kmi.open.ac.uk/open_dataset |
| KNUST Moodle | 4,450 students, 950,628 log events, one 2026 presentation | No login for 21+ consecutive days and two consecutive missed graded quizzes (790 dropouts, 17.8%) | Not public |

KNUST student-level data cannot be shared. The files are pseudonymised, held under the Data Protection Act 2012 (Act 843), and excluded from version control. Code and aggregate outputs can be shared.

## Requirements

Python 3.10.11. Exact pins are in `requirements-lock.txt`.

| Package | Version |
|---|---|
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| XGBoost | 3.2.0 |
| LightGBM | 4.7.0 |
| CatBoost | 1.2.10 |
| SHAP | 0.49.1 |
| Optuna | 5.0.0 |
| imbalanced-learn | 0.14.2 |
| SciPy | 1.15.3 |
| matplotlib | 3.10.8 |

## Quick start

```bash
# 1. Create an environment and install pinned dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt

# 2. Download OULAD and place the seven CSV tables in the data folder set in config.yaml

# 3. Run the whole pipeline in its fixed order
python run_all.py

# 4. Run the leakage and split-integrity tests
pytest
```

All settings live in `config.yaml`. Raw data are read-only. Without the KNUST files, only the OULAD stages can be reproduced.

<!-- TODO: confirm the data folder paths, the test command and how run_all.py behaves when the KNUST files are absent. -->

## Repository layout

```
config.yaml                 All settings
run_all.py                  Runs the scripts in their fixed order
requirements-lock.txt       Exact package versions
scripts/                    Numbered pipeline steps
  09_wu_comparable_sensitivity.py   28-day evaluation with early unregistrations kept
outputs/
  frozen/frozen_pipeline.json       Frozen specification carried to the KNUST data
tests/                      Leakage and split-integrity tests
```

<!-- TODO: list the remaining scripts and output folders; only the files named in the manuscript are shown here. -->

## Main findings

- **RQ1.** The four gradient boosting models gave very similar calibrated early warnings on OULAD. Discrimination is modest once early withdrawers are removed: ROC-AUC is about 0.64 at 28 days, against 0.79 when they are kept.
- **RQ2.** The OULAD-built pipeline transferred to KNUST with little loss and no retraining. Fine-tuning and pooling added nothing measurable on the full sample. With 300 students, prior OULAD knowledge was at least as useful as local training.
- **RQ3.** Risk was driven by early assessment information on OULAD and by activity volume and consistency on KNUST. KNUST explanations were stable across bootstrap refits; OULAD rankings were less stable.

## Limitations

- Absolute alert quality is limited: recall is about 0.30 at the top 20% at 28 days.
- KNUST data come from one institution and one course presentation.
- The KNUST export has no forum, assignment, enrolment or demographic fields, so fairness could not be audited on KNUST data.
- The KNUST dropout label is derived from the same LMS sources as the features, so later windows partly measure the label.
- SHAP shows what the model uses, not why students leave.

## Ethics

Ethical clearance was obtained from the Humanities and Social Sciences Research Ethics Committee (HUSSREC) of KNUST. The Office of the Registrar approved release of the LMS data on 4 August 2026.

## Reporting

Reporting follows the TRIPOD+AI guidance where applicable. The evaluation protocol extends Wu, Duan and Luo (PLoS ONE, 2026), doi:10.1371/journal.pone.0352867.

## Citation

<!-- TODO: add authors, year, title of the thesis or paper, and DOI once available. -->

## Licence

<!-- TODO: choose a licence for the code. -->

## Contact

<!-- TODO: add a contact name and email. -->
