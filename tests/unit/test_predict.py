"""Prediction tests run against the committed trained models in models/."""

import math

import pytest

from carepilot.ml.datasets import DIABETES
from carepilot.ml.predict import PatientValidationError, predict_risk, validate_patient

HIGH_RISK_DIABETES = {
    "Pregnancies": 2,
    "Glucose": 190,
    "BloodPressure": 85,
    "BMI": 40,
    "DiabetesPedigreeFunction": 1.2,
    "Age": 55,
}
LOW_RISK_DIABETES = {
    "Pregnancies": 0,
    "Glucose": 85,
    "BloodPressure": 66,
    "BMI": 22,
    "DiabetesPedigreeFunction": 0.2,
    "Age": 23,
}
HEART_PATIENT = {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": 280, "fbs": 0, "restecg": 2,
    "thalach": 120, "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 7,
}  # fmt: skip


def test_probability_in_unit_interval_and_factors_cover_all_features():
    r = predict_risk("diabetes", HIGH_RISK_DIABETES)
    assert 0.0 <= r.probability <= 1.0
    assert {f.feature for f in r.factors} == set(DIABETES.feature_order)


def test_high_risk_patient_scores_above_low_risk_patient():
    high = predict_risk("diabetes", HIGH_RISK_DIABETES)
    low = predict_risk("diabetes", LOW_RISK_DIABETES)
    assert high.probability > low.probability
    assert high.flagged and not low.flagged
    assert high.band in {"Elevated", "High"} and low.band == "Low"


def test_glucose_is_top_driver_for_hyperglycemic_patient():
    r = predict_risk("diabetes", HIGH_RISK_DIABETES)
    top = r.top_factors(1, risk_raising_only=True)[0]
    assert top.feature == "Glucose" and top.direction == "raises"


def test_missing_fields_are_imputed_and_reported():
    r = predict_risk("diabetes", {"Glucose": 120, "Age": 40})
    assert "Insulin" in r.imputed and "Glucose" not in r.imputed
    assert not math.isnan(r.probability)


def test_heart_prediction_works_with_categoricals():
    r = predict_risk("heart", HEART_PATIENT)
    assert r.band == "High"


@pytest.mark.parametrize(
    ("patient", "message"),
    [
        ({"Glucose": 0}, "outside plausible range"),  # Pima zero trap must not pass silently
        ({"Glucose": "abc"}, "expected a number"),
        ({"Cholesterol": 200}, "unknown fields"),
    ],
)
def test_invalid_input_rejected_with_reason(patient, message):
    with pytest.raises(PatientValidationError, match=message):
        validate_patient(DIABETES, patient)


def test_invalid_categorical_rejected():
    with pytest.raises(PatientValidationError, match="must be one of"):
        predict_risk("heart", HEART_PATIENT | {"thal": 5})


def test_all_errors_collected_at_once():
    with pytest.raises(PatientValidationError) as exc:
        validate_patient(DIABETES, {"Glucose": 0, "BMI": 500})
    assert len(exc.value.errors) == 2


def test_positive_class_shap_handles_explainer_output_shapes():
    import numpy as np

    from carepilot.ml.predict import positive_class_shap

    two_d = np.array([[1.0, 2.0, 3.0]])
    three_d = np.stack([-two_d, two_d], axis=-1)  # (rows, features, classes)
    as_list = [-two_d, two_d]
    for v in (two_d, three_d, as_list):
        assert positive_class_shap(v).tolist() == [1.0, 2.0, 3.0]
