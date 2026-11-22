import pytest

from carepilot.core.safety import check_red_flags, check_vitals


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("I have crushing chest pain right now", "Possible heart attack"),
        ("pain radiating to my left arm", "Possible heart attack"),
        ("my dad's face is drooping and he has slurred speech", "Possible stroke"),
        ("sudden weakness on one side", "Possible stroke"),
        ("he has fruity breath and keeps vomiting", "Possible diabetic emergency"),
        ("she passed out after taking insulin", "Possible diabetic emergency"),
        ("my BP is 190/125", "Possible hypertensive crisis"),
        ("I want to die", "Crisis support"),
    ],
)
def test_red_flags_detected(text, category):
    assert category in {f.category for f in check_red_flags(text)}


@pytest.mark.parametrize(
    "text",
    [
        "What are the symptoms of diabetes?",
        "What lifestyle changes lower blood pressure?",
        "Is 120/80 a normal blood pressure?",
        "",
    ],
)
def test_ordinary_questions_not_flagged(text):
    assert check_red_flags(text) == []


def test_vitals_hypertensive_crisis():
    assert check_vitals("heart", {"trestbps": 185})
    assert check_vitals("diabetes", {"BloodPressure": 125})
    assert not check_vitals("heart", {"trestbps": 140})
    assert not check_vitals("diabetes", {"BloodPressure": None})
