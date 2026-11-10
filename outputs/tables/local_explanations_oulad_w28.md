# Local explanations (oulad_w28) - median-risk case of each type

## correctly_flagged_dropout (row 1419, risk score 0.706, outcome=dropout)
Top reasons: early_score_mean=11.00: pushes risk higher (SHAP +0.96); distinct_resources=14.00: pushes risk lower (SHAP -0.16); studied_credits=60.00: pushes risk higher (SHAP +0.11)
Advisor reading: weak early quiz scores -> point to remedial resources (This is what the model uses, not proof of cause.)

## missed_dropout (row 1282, risk score 0.417, outcome=dropout)
Top reasons: early_score_mean=93.33: pushes risk lower (SHAP -0.50); days_since_last_click=42.00: pushes risk higher (SHAP +0.19); edu_ord=1.00: pushes risk higher (SHAP +0.13)
Advisor reading: has not logged in recently -> contact the student this week (This is what the model uses, not proof of cause.)

## false_alarm (row 580, risk score 0.715, outcome=stayed)
Top reasons: early_score_mean=1.00: pushes risk higher (SHAP +1.02); distinct_resources=7.00: pushes risk lower (SHAP -0.16); studied_credits=60.00: pushes risk higher (SHAP +0.10)
Advisor reading: weak early quiz scores -> point to remedial resources (This is what the model uses, not proof of cause.)
