# Demo video script (60 seconds)

Record at 1080p with the Windows Game Bar (Win + G) or OBS. Use the Groq provider so answers come back fast. Before recording, open the app on the Risk tab and pick *High-risk example*.

| Time | Screen | Say |
|---|---|---|
| 0:00 | App, Risk tab, click *Estimate risk* | "CarePilot predicts diabetes and heart-disease risk, then explains it from cited NIH and WHO documents." |
| 0:08 | Risk %, band, SHAP chart | "This patient is high risk, mainly from glucose and BMI. The threshold is set to catch at least 85% of real cases." |
| 0:18 | Scroll to explanation, open one source | "Those exact risk factors become search queries, so the explanation is about this patient, and every sentence cites its source." |
| 0:30 | Ask tab: click *What are the symptoms of diabetes?* | "Questions use hybrid search with a reranker, and every answer shows its sources." |
| 0:38 | Type "What dose of metformin should I take?" → Ask | "If the documents don't cover something, like dosing, it says so instead of guessing." |
| 0:46 | Report tab, scroll the tables | "Everything is measured: model metrics with confidence intervals and a 40-question RAG evaluation. It's tested end to end." |
| 0:56 | Stop | "Code and details are on GitHub." |
