"""Safety layer: red-flag detection, emergency guidance and the standing disclaimer.

Red flags are checked BEFORE any retrieval or model output is shown. They are simple,
auditable regular expressions on purpose: for emergencies a transparent rule that over-fires
slightly is better than a model that might miss. Patterns follow the warning signs listed in
the indexed NHLBI / NIDDK / WHO documents plus standard stroke (FAST) and crisis guidance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DISCLAIMER = (
    "Demonstration prototype for an AI/ML assessment. Not a medical device and not medical "
    "advice. Risk estimates come from small public research datasets and may not apply to you. "
    "Always consult a qualified health professional."
)

EMERGENCY_NUMBERS = "112 (India / EU) or 911 (US)"


@dataclass(frozen=True)
class RedFlag:
    category: str
    message: str


_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "Possible heart attack",
        r"chest (pain|pressure|tightness|discomfort|heaviness)|pain (spreading|radiating) to (my |the )?(left )?(arm|jaw|neck|back)|crushing (chest )?pain",
        "Chest pain or pressure can be a heart attack. Call emergency services now; do not drive yourself.",
    ),
    (
        "Possible stroke",
        r"face (is )?droop|slurred speech|(sudden|one[- ]sided) (weakness|numbness)|can'?t (move|feel) (one|my (left|right))|sudden (confusion|trouble (speaking|seeing|walking))|worst headache",
        "These can be stroke signs (Face, Arms, Speech, Time). Call emergency services immediately.",
    ),
    (
        "Possible diabetic emergency",
        r"fruity[- ]smelling breath|fruity breath|ketoacidosis|\bdka\b|high ketones|(passed out|passing out|fainted|unconscious|seizure).{0,40}(sugar|glucose|diabet|insulin)|(sugar|glucose).{0,40}(passed out|unconscious|seizure)",
        "This can be a diabetic emergency (very high or very low blood glucose). Seek emergency care now.",
    ),
    (
        "Possible hypertensive crisis",
        r"\b(1[89]\d|2\d\d)\s*/\s*(1[2-9]\d)\b",
        "A blood pressure of 180/120 or higher is a hypertensive crisis. Seek urgent medical care, especially with headache, chest pain or vision changes.",
    ),
    (
        "Crisis support",
        r"kill myself|suicid|want to die|end my life|self[- ]harm",
        "You are not alone. Please contact local emergency services or a crisis line now (India: Tele-MANAS 14416; US: 988).",
    ),
)
_COMPILED = tuple((cat, re.compile(pat, re.IGNORECASE), msg) for cat, pat, msg in _RULES)

HYPERTENSIVE_CRISIS_SYSTOLIC = 180
HYPERTENSIVE_CRISIS_DIASTOLIC = 120


def check_red_flags(text: str) -> list[RedFlag]:
    return [RedFlag(cat, msg) for cat, rx, msg in _COMPILED if rx.search(text or "")]


def check_vitals(disease: str, values: dict[str, float | None]) -> list[RedFlag]:
    """Flag entered measurements that need urgent care regardless of the model's risk score."""
    systolic = values.get("trestbps") if disease == "heart" else None
    diastolic = values.get("BloodPressure") if disease == "diabetes" else None
    crisis = (systolic is not None and systolic >= HYPERTENSIVE_CRISIS_SYSTOLIC) or (
        diastolic is not None and diastolic >= HYPERTENSIVE_CRISIS_DIASTOLIC
    )
    return [RedFlag("Possible hypertensive crisis", _RULES[3][2])] if crisis else []
