import pytest

from carepilot.core.whatif import MODIFIABLE, what_if
from carepilot.ml.predict import predict_risk

PATIENT = {"Glucose": 180, "BMI": 38.5, "BloodPressure": 85, "Age": 52}


def test_lowering_glucose_and_bmi_lowers_estimate():
    before = predict_risk("diabetes", PATIENT)
    after = what_if("diabetes", PATIENT, {"Glucose": 110, "BMI": 27})
    assert after.probability < before.probability


def test_no_change_gives_same_estimate():
    assert what_if("diabetes", PATIENT, {}).probability == pytest.approx(
        predict_risk("diabetes", PATIENT).probability
    )


def test_non_modifiable_factor_rejected():
    with pytest.raises(ValueError, match="not a modifiable factor"):
        what_if("diabetes", PATIENT, {"Age": 30})


def test_modifiable_lists_exclude_fixed_traits():
    for fixed in ("Age", "age", "sex", "Pregnancies", "DiabetesPedigreeFunction"):
        assert all(fixed not in f for f in MODIFIABLE.values())
