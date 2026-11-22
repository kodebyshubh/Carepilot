from carepilot.core.explain import explanation_queries, explanation_question, gather_passages
from carepilot.ml.predict import Factor, RiskResult
from carepilot.rag.chunking import Chunk
from carepilot.rag.retrieve import Hit


def _result(factors, disease="diabetes", p=0.7):
    return RiskResult(disease, "Logistic Regression", p, 0.23, tuple(factors), ())


GLUCOSE = Factor("Glucose", "Plasma glucose", 180, "mg/dL", 1.6)
BMI = Factor("BMI", "Body mass index", 38, "kg/m2", 0.4)
AGE_LOWERS = Factor("Age", "Age", 25, "years", -0.3)


def test_queries_follow_risk_raising_drivers_only():
    qs = explanation_queries(_result([GLUCOSE, BMI, AGE_LOWERS]))
    assert qs[0] == "How does high blood glucose affect the risk of type 2 diabetes?"
    assert any("overweight" in q for q in qs)
    assert not any("older age" in q for q in qs)  # age lowered the risk here
    assert "lifestyle" in qs[-2] and "doctor" in qs[-1]


def test_question_names_patient_values_and_band():
    q = explanation_question(_result([GLUCOSE, BMI]))
    assert "Plasma glucose (180 mg/dL)" in q and "high" in q


def test_low_risk_patient_still_gets_prevention_question():
    q = explanation_question(_result([AGE_LOWERS], p=0.05))
    assert "stay low" in q
    assert len(explanation_queries(_result([AGE_LOWERS], p=0.05))) == 2


class FakeRetriever:
    def search(self, query, k=3, mode="hybrid_rerank"):
        same = Hit(Chunk("d#0", "d", "T", "P", "u", "s", "shared"), 1.0 + len(query) / 100)
        own = Hit(Chunk(f"d#{len(query)}", "d", "T", "P", "u", "s", query), 0.5)
        return [same, own]


def test_gather_dedupes_and_keeps_best_score():
    hits = gather_passages(FakeRetriever(), ["short", "a much longer query"])
    ids = [h.chunk.chunk_id for h in hits]
    assert ids.count("d#0") == 1
    assert hits[0].chunk.chunk_id == "d#0" and hits[0].score > 1.1
