"""Load a saved model and predict risk for one patient, with per-feature SHAP contributions.

    >>> from carepilot.ml.predict import predict_risk
    >>> r = predict_risk("diabetes", {"Glucose": 165, "BMI": 36.2, "Age": 52})
    >>> r.probability, r.flagged, [f.feature for f in r.top_factors(3)]

Fields left out (or None) are treated as missing and imputed exactly as during training.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from carepilot.config import MODELS_DIR
from carepilot.ml.datasets import DiseaseSpec, get_spec

HIGH_RISK_PROBA = 0.6


class PatientValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Factor:
    feature: str
    label: str
    value: float | None  # None = missing, imputed
    unit: str
    contribution: float  # SHAP value, log-odds (linear) or margin (trees); + raises risk

    @property
    def direction(self) -> str:
        return "raises" if self.contribution > 0 else "lowers"


@dataclass(frozen=True)
class RiskResult:
    disease: str
    model: str
    probability: float
    threshold: float
    factors: tuple[Factor, ...]  # sorted by absolute contribution, largest first
    imputed: tuple[str, ...]

    @property
    def flagged(self) -> bool:
        return self.probability >= self.threshold

    @property
    def band(self) -> str:
        if self.probability >= HIGH_RISK_PROBA:
            return "High"
        return "Elevated" if self.flagged else "Low"

    def top_factors(self, n: int = 3, risk_raising_only: bool = False) -> list[Factor]:
        fs = [f for f in self.factors if f.contribution > 0] if risk_raising_only else self.factors
        return list(fs[:n])


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any
    explainer: Any
    metadata: dict[str, Any]


@lru_cache(maxsize=4)
def load_model(disease: str) -> LoadedModel:
    model_dir = MODELS_DIR / disease
    if not (model_dir / "model.joblib").exists():
        raise FileNotFoundError(
            f"no trained model at {model_dir}; run: python -m carepilot.ml.train --disease {disease}"
        )
    bundle = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    pipeline = bundle["pipeline"]
    clf = pipeline.named_steps["clf"]
    background = bundle["shap_background"]
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, background)
    else:
        explainer = shap.TreeExplainer(clf, background)
    return LoadedModel(pipeline, explainer, metadata)


def validate_patient(spec: DiseaseSpec, patient: dict[str, Any]) -> dict[str, float | None]:
    """Return a clean {feature: value or None}. Collects every problem before raising."""
    errors: list[str] = []
    unknown = sorted(set(patient) - set(spec.feature_order))
    if unknown:
        errors.append(f"unknown fields: {', '.join(unknown)}")
    clean: dict[str, float | None] = {}
    for name in spec.feature_order:
        info = spec.features[name]
        raw = patient.get(name)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            clean[name] = None
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{name}: expected a number, got {raw!r}")
            continue
        if info.choices is not None and int(value) not in info.choices:
            errors.append(f"{name}: must be one of {sorted(info.choices)}")
        elif not info.min_value <= value <= info.max_value:
            errors.append(
                f"{name}: {value:g} outside plausible range "
                f"{info.min_value:g} to {info.max_value:g} {info.unit}".rstrip()
            )
        clean[name] = value
    if errors:
        raise PatientValidationError(errors)
    return clean


def positive_class_shap(values: Any) -> np.ndarray:
    """SHAP values for the positive class of a single row, shape (n_features,).

    Explainers disagree on output shape: linear and XGBoost give (rows, features); sklearn
    forests give (rows, features, classes) or, in older shap, a list per class. Flattening
    blindly would interleave the two classes and scramble the attribution.
    """
    if isinstance(values, list):  # older shap: one array per class
        values = values[-1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[..., -1]
    return arr[0]


def predict_risk(disease: str, patient: dict[str, Any]) -> RiskResult:
    spec = get_spec(disease)
    values = validate_patient(spec, patient)
    model = load_model(disease)
    row = pd.DataFrame(
        [{k: (np.nan if v is None else v) for k, v in values.items()}], columns=spec.feature_order
    )
    proba = float(model.pipeline.predict_proba(row)[0, 1])

    transformed = model.pipeline.named_steps["prep"].transform(row)
    shap_row = positive_class_shap(model.explainer.shap_values(transformed))
    groups: dict[str, list[int]] = model.metadata["feature_groups"]
    factors = [
        Factor(
            feature=name,
            label=spec.features[name].label,
            value=values[name],
            unit=spec.features[name].unit,
            contribution=float(shap_row[idx].sum()),
        )
        for name, idx in groups.items()
    ]
    factors.sort(key=lambda f: abs(f.contribution), reverse=True)
    return RiskResult(
        disease=disease,
        model=model.metadata["selected_model"],
        probability=proba,
        threshold=float(model.metadata["threshold"]),
        factors=tuple(factors),
        imputed=tuple(k for k, v in values.items() if v is None),
    )
