"""Patient dict -> risk -> cited personalized explanation, with real models (extractive LLM)."""

import pytest

from carepilot.core.explain import explain_risk
from carepilot.llm.providers import ExtractiveLLM
from carepilot.ml.predict import predict_risk
from carepilot.rag.retrieve import Retriever

pytestmark = pytest.mark.integration


def test_high_risk_diabetes_patient_gets_cited_explanation():
    result = predict_risk(
        "diabetes", {"Glucose": 185, "BMI": 39, "Age": 50, "DiabetesPedigreeFunction": 1.0}
    )
    ans = explain_risk(result, Retriever.load(), ExtractiveLLM())
    assert not ans.abstained
    assert ans.sources
    assert {s.publisher for s in ans.sources} & {"NIDDK (NIH)", "World Health Organization"}


def test_heart_patient_explanation_draws_on_heart_documents():
    patient = {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": 280, "fbs": 0, "restecg": 2,
        "thalach": 120, "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 7,
    }  # fmt: skip
    ans = explain_risk(predict_risk("heart", patient), Retriever.load(), ExtractiveLLM())
    assert not ans.abstained
    assert any(
        "heart" in s.title.lower() or "blood pressure" in s.title.lower() for s in ans.sources
    )
