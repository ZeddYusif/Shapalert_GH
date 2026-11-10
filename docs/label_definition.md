# Label definitions (Phase 0, step 0.2) - fixed once written

| Dataset | Dropout (y = 1) | Not dropout (y = 0) |
|---|---|---|
| OULAD | `final_result == 'Withdrawn'` (follows Wu et al. 2026). `Fail` is NOT dropout; analyse it separately. | Pass, Distinction, Fail |
| Ghana (KNUST Moodle) | **Proposal rule (s3.2.2)**: no login for >=21 consecutive days up to the end of the export (2026-06-20) AND >=2 consecutive missed module quizzes (Module n and n+1 both without a graded result). | Everyone else |

Interpretation choices (my reading of the proposal; confirmed by the researcher): the 21-day gap is counted at the END of the observation
period (a student who paused and returned is not a dropout); "login" means Moodle `login` events; "missed graded assessment"
means a module quiz with no graded result (the export has no due dates). Result: 790 of 4,450 students (17.8%).

Sensitivity labels, switchable in `config.yaml -> ghana.label_rule`:

* `noncompletion` - fewer than 6 of the 6 module quizzes graded (30.6%). This was my first assumption; the first full run used it.
  Those results are kept in `outputs/archive_label_noncompletion/` and the switch to the proposal rule is a documented post-hoc change
  (the proposal rule was specified in the researcher notes before any Ghana modelling; my earlier choice was not).
* `noncompletion_and_inactive`, `export_at_risk` (the `at_risk` flag shipped with the data; ~90% agreement with the proposal rule).

Course start for windows: **2026-01-12** (confirmed) (first week of mass activity). Confirm with the lecturer; set in `config.yaml`.
Windows are 14, 28, 42 days after that date. Never use any information from after the window to build features.
