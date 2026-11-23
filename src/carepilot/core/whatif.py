"""What-if simulation: re-score a patient after changing factors a person can influence.

This shows how the MODEL's estimate responds, which is an association learned from the
training data, not a causal effect of the intervention. The UI says so.
"""

from __future__ import annotations

from typing import Any

from carepilot.ml.predict import RiskResult, predict_risk

# Measurements lifestyle or treatment can move. Age, sex, family history, pregnancies and
# test findings (ECG, fluoroscopy, thallium) are left out on purpose.
MODIFIABLE: dict[str, tuple[str, ...]] = {
    "diabetes": ("Glucose", "BMI", "BloodPressure"),
    "heart": ("trestbps", "chol", "thalach"),
}


def what_if(disease: str, values: dict[str, Any], changes: dict[str, float]) -> RiskResult:
    allowed = set(MODIFIABLE[disease])
    not_allowed = sorted(set(changes) - allowed)
    if not_allowed:
        raise ValueError(f"not a modifiable factor for {disease}: {', '.join(not_allowed)}")
    return predict_risk(disease, values | changes)
