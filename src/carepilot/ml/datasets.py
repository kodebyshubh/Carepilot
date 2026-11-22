"""Per-disease dataset specs and cleaning.

Each spec declares which columns are numeric vs categorical, which zeros are physiologically
impossible (so they are missing values in disguise), the binary target, and plausible input
ranges used to validate user input at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from carepilot.config import RAW_DIR


@dataclass(frozen=True)
class FeatureInfo:
    label: str
    unit: str
    min_value: float
    max_value: float
    help: str = ""
    choices: dict[int, str] | None = None  # categorical code -> human label


@dataclass(frozen=True)
class DiseaseSpec:
    name: str
    display_name: str
    csv_name: str
    target: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    zero_as_missing: tuple[str, ...] = ()
    features: dict[str, FeatureInfo] = field(default_factory=dict)

    @property
    def feature_order(self) -> list[str]:
        return [*self.numeric, *self.categorical]


DIABETES = DiseaseSpec(
    name="diabetes",
    display_name="Type 2 diabetes (Pima Indians Diabetes dataset)",
    csv_name="diabetes.csv",
    target="Outcome",
    numeric=("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
             "BMI", "DiabetesPedigreeFunction", "Age"),
    categorical=(),
    # A living person cannot have 0 glucose, 0 diastolic BP, 0 skinfold, 0 insulin or 0 BMI.
    # In this dataset 0 encodes "not measured". Pregnancies = 0 is a real value.
    zero_as_missing=("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"),
    features={
        "Pregnancies": FeatureInfo("Pregnancies", "count", 0, 20),
        "Glucose": FeatureInfo("Plasma glucose (2h OGTT)", "mg/dL", 40, 300,
                               "2-hour oral glucose tolerance test"),
        "BloodPressure": FeatureInfo("Diastolic blood pressure", "mm Hg", 30, 140),
        "SkinThickness": FeatureInfo("Triceps skinfold thickness", "mm", 5, 100),
        "Insulin": FeatureInfo("2-hour serum insulin", "uU/mL", 10, 900),
        "BMI": FeatureInfo("Body mass index", "kg/m2", 12, 70),
        "DiabetesPedigreeFunction": FeatureInfo("Diabetes pedigree function", "score", 0.05, 2.5,
                                                "family-history based genetic risk score"),
        "Age": FeatureInfo("Age", "years", 18, 100),
    },
)

HEART = DiseaseSpec(
    name="heart",
    display_name="Coronary heart disease (UCI Cleveland dataset)",
    csv_name="heart.csv",
    target="num",
    numeric=("age", "trestbps", "chol", "thalach", "oldpeak", "ca"),
    categorical=("sex", "cp", "fbs", "restecg", "exang", "slope", "thal"),
    features={
        "age": FeatureInfo("Age", "years", 18, 100),
        "trestbps": FeatureInfo("Resting systolic blood pressure", "mm Hg", 80, 220),
        "chol": FeatureInfo("Serum cholesterol", "mg/dL", 100, 600),
        "thalach": FeatureInfo("Max heart rate on exercise test", "bpm", 60, 220),
        "oldpeak": FeatureInfo("ST depression on exercise vs rest", "mm", 0, 7),
        "ca": FeatureInfo("Major vessels colored by fluoroscopy", "count", 0, 3),
        "sex": FeatureInfo("Sex", "", 0, 1, choices={0: "Female", 1: "Male"}),
        "cp": FeatureInfo("Chest pain type", "", 1, 4, choices={
            1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}),
        "fbs": FeatureInfo("Fasting blood sugar > 120 mg/dL", "", 0, 1,
                           choices={0: "No", 1: "Yes"}),
        "restecg": FeatureInfo("Resting ECG", "", 0, 2, choices={
            0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}),
        "exang": FeatureInfo("Exercise-induced angina", "", 0, 1, choices={0: "No", 1: "Yes"}),
        "slope": FeatureInfo("Slope of peak exercise ST segment", "", 1, 3, choices={
            1: "Upsloping", 2: "Flat", 3: "Downsloping"}),
        "thal": FeatureInfo("Thallium stress test", "", 3, 7, choices={
            3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}),
    },
)

SPECS: dict[str, DiseaseSpec] = {DIABETES.name: DIABETES, HEART.name: HEART}


def get_spec(disease: str) -> DiseaseSpec:
    try:
        return SPECS[disease]
    except KeyError as exc:
        raise ValueError(f"unknown disease {disease!r}; expected one of {sorted(SPECS)}") from exc


def clean(df: pd.DataFrame, spec: DiseaseSpec) -> pd.DataFrame:
    """Return a cleaned copy: disguised zeros to NaN, binary target, deduplicated rows."""
    out = df.copy()
    for col in spec.zero_as_missing:
        out[col] = out[col].replace(0, np.nan)
    # Heart 'num' is 0 (no disease) to 4 (severity); standard practice is presence vs absence.
    out[spec.target] = (out[spec.target] > 0).astype(int)
    return out.drop_duplicates().reset_index(drop=True)


def load(disease: str) -> tuple[pd.DataFrame, pd.Series]:
    spec = get_spec(disease)
    df = clean(pd.read_csv(RAW_DIR / spec.csv_name), spec)
    return df[spec.feature_order], df[spec.target]
