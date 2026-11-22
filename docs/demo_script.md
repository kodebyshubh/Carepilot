# Demo video script (about 2.5 minutes)

Record the screen at 1080p with OBS or the Windows Game Bar (Win + G). Use the Groq provider for fast answers. Speak in the first person.

| Time | Screen | Say |
|---|---|---|
| 0:00 | README top | "This is CarePilot: two risk models, for diabetes and heart disease, plus a RAG system that explains each prediction using cited NIH and WHO documents and refuses when the documents don't cover the question." |
| 0:15 | App, Risk tab, *High-risk example*, Estimate | "A 52-year-old with glucose 180 and BMI 38.5. The model says high risk. The threshold is 23%, not 50%, because I tuned it for at least 85% sensitivity: missing a diabetic costs more than an extra test." |
| 0:35 | SHAP chart | "These are this patient's drivers from SHAP. Glucose dominates, then BMI. Insulin and skinfold weren't measured, so they're imputed the same way as in training, and the app says so." |
| 0:50 | Scroll to explanation + open a source | "Each driver becomes a retrieval query. The answer explains why those factors matter, what lifestyle changes help, and when to see a doctor. Every sentence cites a passage, and here's the NIDDK source." |
| 1:10 | Ask tab: *What are the symptoms of diabetes?* | "Hybrid retrieval, BM25 plus dense with reciprocal rank fusion, then a cross-encoder reranker, then the LLM." |
| 1:25 | Ask: "What is the right dose of metformin?" | "The documents don't cover dosing. The reranker score is below a calibrated threshold, so it abstains without even calling the LLM." |
| 1:40 | Ask: "I have crushing chest pain" | "Red-flag rules run first and show emergency guidance." |
| 1:50 | Report tab | "Logistic Regression won cross-validation on both datasets. XGBoost scored slightly higher on the test set, but the intervals overlap and choosing on the test set would bias the number. Here's the RAG evaluation: 40 questions, a retrieval ablation where the reranker reaches perfect hit-at-1, and an abstention gate that, by a leave-one-out estimate, keeps 97% of answerable questions and rejects 90% of unanswerable ones." |
| 2:20 | Terminal: `uv run pytest` / CI badge | "94 unit and integration tests at 94% coverage, 8 Playwright journeys, CI, and Docker. It's a demonstration prototype; the limitations are in the README." |
