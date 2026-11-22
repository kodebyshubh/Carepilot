"""Bridge from the ML model to the documents: a cited explanation of THIS patient's risk.

1. Take the factors that pushed this patient's estimate up the most (SHAP, from predict_risk).
2. Turn each into a focused retrieval query ("How does high blood glucose affect diabetes
   risk?"), plus one query for lifestyle changes and one for when to see a doctor.
   Several narrow queries retrieve better than one long compound question.
3. Merge the reranked hits (best score per chunk), keep the top passages, and ask the LLM
   for a short explanation that cites them. Same abstention and citation checks as Q&A.
"""

from __future__ import annotations

from carepilot.llm.providers import LLM
from carepilot.ml.predict import Factor, RiskResult
from carepilot.rag.answer import Answer, answer_from_hits
from carepilot.rag.retrieve import Hit, Retriever

HITS_PER_QUERY = 3
MAX_PASSAGES = 6
MAX_DRIVERS = 3

DISEASE_TERM = {"diabetes": "type 2 diabetes", "heart": "coronary heart disease"}

# Model feature -> plain-language topic the documents talk about.
TOPIC = {
    "Glucose": "high blood glucose",
    "BMI": "overweight and obesity",
    "Age": "older age",
    "DiabetesPedigreeFunction": "family history of diabetes",
    "BloodPressure": "high blood pressure",
    "Pregnancies": "pregnancy and gestational diabetes",
    "Insulin": "insulin resistance",
    "SkinThickness": "excess body fat",
    "age": "older age",
    "sex": "sex differences",
    "cp": "chest pain",
    "trestbps": "high blood pressure",
    "chol": "high cholesterol",
    "fbs": "high blood sugar and diabetes",
    "restecg": "heart and blood vessel conditions",
    "thalach": "physical activity and fitness",
    "exang": "chest pain during exercise",
    "oldpeak": "heart and blood vessel conditions",
    "slope": "heart and blood vessel conditions",
    "ca": "narrowed coronary arteries",
    "thal": "heart and blood vessel conditions",
}


def explanation_queries(result: RiskResult) -> list[str]:
    disease = DISEASE_TERM[result.disease]
    drivers = result.top_factors(MAX_DRIVERS, risk_raising_only=True)
    topics = list(dict.fromkeys(TOPIC[f.feature] for f in drivers))  # dedupe, keep order
    queries = [f"How does {t} affect the risk of {disease}?" for t in topics]
    queries.append(f"What lifestyle changes lower the risk of {disease}?")
    queries.append(f"When should someone see a doctor about {disease}?")
    return queries


def gather_passages(retriever: Retriever, queries: list[str]) -> list[Hit]:
    best: dict[str, Hit] = {}
    for q in queries:
        for h in retriever.search(q, k=HITS_PER_QUERY, mode="hybrid_rerank"):
            cur = best.get(h.chunk.chunk_id)
            if cur is None or h.score > cur.score:
                best[h.chunk.chunk_id] = h
    return sorted(best.values(), key=lambda h: h.score, reverse=True)[:MAX_PASSAGES]


def _describe(f: Factor) -> str:
    value = "not provided" if f.value is None else f"{f.value:g} {f.unit}".strip()
    return f"{f.label} ({value})"


def explanation_question(result: RiskResult) -> str:
    disease = DISEASE_TERM[result.disease]
    drivers = result.top_factors(MAX_DRIVERS, risk_raising_only=True)
    if drivers:
        listed = "; ".join(_describe(f) for f in drivers)
        focus = (
            f"The factors that raised this person's estimate most were: {listed}. Explain in "
            f"plain language why these factors matter for {disease}"
        )
    else:
        focus = f"No single factor raised this person's estimate much. Summarize how {disease} risk can stay low"
    return (
        f"A screening model estimated this person's {disease} risk as {result.band.lower()}. "
        f"{focus}, which lifestyle changes the passages recommend, and when the passages say "
        "to see a doctor. Do not restate the estimate as a diagnosis."
    )


def explain_risk(result: RiskResult, retriever: Retriever, llm: LLM) -> Answer:
    hits = gather_passages(retriever, explanation_queries(result))
    return answer_from_hits(explanation_question(result), hits, llm)
