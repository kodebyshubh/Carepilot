"""Offline pipelines run end to end in temporary directories (real artifacts untouched)."""

import json
import sys

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from carepilot.data import download
from carepilot.eval import rag_eval
from carepilot.llm.providers import ExtractiveLLM
from carepilot.ml import evaluate, predict, report, train
from carepilot.rag import ask, index, ingest, retrieve
from carepilot.rag.answer import answer_question

pytestmark = pytest.mark.integration


@pytest.fixture
def fast_training(monkeypatch, tmp_path):
    """Tiny grids, fewer bootstrap resamples, outputs redirected to tmp."""
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(report, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(predict, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(evaluate, "N_BOOTSTRAP", 30)
    full = train.candidates()
    monkeypatch.setattr(train, "candidates", lambda: {k: (est, {}) for k, (est, _) in full.items()})
    predict.load_model.cache_clear()
    yield tmp_path
    predict.load_model.cache_clear()


def test_training_pipeline_writes_artifacts_and_report(fast_training):
    out = train.train_disease("heart")
    meta = out["metadata"]
    assert 0 < meta["threshold"] < 1
    assert set(out["comparison"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    model_dir = fast_training / "models" / "heart"
    for f in ("model.joblib", "metadata.json", "metrics.json", "roc.png", "calibration.png"):
        assert (model_dir / f).exists()
    md = report.write_report().read_text(encoding="utf-8")
    assert "Selected model" in md and "95% bootstrap" in md


def test_tree_model_path_uses_tree_explainer(fast_training, monkeypatch):
    def pick_rf(results):
        return next(r for r in results if r.name == "rf"), "forced for test"

    monkeypatch.setattr(train, "select_model", pick_rf)
    train.train_disease("diabetes")
    loaded = predict.load_model("diabetes")
    assert isinstance(loaded.pipeline.named_steps["clf"], RandomForestClassifier)
    r = predict.predict_risk("diabetes", {"Glucose": 190, "BMI": 41, "Age": 55})
    assert 0 <= r.probability <= 1
    assert r.top_factors(1, risk_raising_only=True)[0].feature == "Glucose"


def test_one_standard_error_rule_prefers_logreg():
    est = object()
    results = [
        train.CvResult("logreg", {}, 0.83, 0.02, est),
        train.CvResult("xgb", {}, 0.835, 0.02, est),  # gap 0.005 < SE 0.009,
    ]
    chosen, why = train.select_model(results)
    assert chosen.name == "logreg" and "one standard error" in why


def test_download_writes_csv_and_manifest(monkeypatch, tmp_path):
    class Resp:
        content = b"6,148,72,35,0,33.6,0.627,50,1\n1,85,66,29,0,26.6,0.351,31,0\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    monkeypatch.setattr(download, "SOURCES", download.SOURCES[:1])
    monkeypatch.setattr(download.requests, "get", lambda url, timeout: Resp())
    download.main()
    assert pd.read_csv(tmp_path / "diabetes.csv").shape == (2, 9)
    assert json.loads((tmp_path / "manifest.json").read_text())[0]["sha256_raw"]


def test_ingest_and_index_build_on_tiny_corpus(monkeypatch, tmp_path):
    html = "<html><body><main><h2>Symptoms</h2><p>" + "Thirst and fatigue are common. " * 30
    html += "</p></main></body></html>"
    src = ingest.SOURCES[0]
    for mod, attr, sub in (
        (ingest, "DOCS_RAW_DIR", "raw"),
        (ingest, "DOCS_CLEAN_DIR", "clean"),
        (index, "DOCS_CLEAN_DIR", "clean"),
        (index, "INDEX_DIR", "index"),
        (retrieve, "INDEX_DIR", "index"),
    ):
        monkeypatch.setattr(mod, attr, tmp_path / sub)
    monkeypatch.setattr(ingest, "SOURCES", (src,))
    monkeypatch.setattr(ingest, "fetch_html", lambda s: html)
    monkeypatch.setattr(sys, "argv", ["ingest"])
    ingest.main()
    assert (tmp_path / "clean" / f"{src.doc_id}.md").exists()

    meta = index.build_index()
    assert meta["n_docs"] == 1 and meta["n_chunks"] >= 1
    hits = retrieve.Retriever.load().search("thirst", k=1, mode="dense")
    assert hits and hits[0].chunk.doc_id == src.doc_id


def test_eval_harness_runs_all_stages_on_small_golden(monkeypatch, tmp_path):
    golden = [json.loads(x) for x in rag_eval.GOLDEN.read_text().splitlines()]
    small = tmp_path / "golden.jsonl"
    small.write_text("\n".join(json.dumps(g) for g in golden[:3] + golden[-3:]))
    monkeypatch.setattr(rag_eval, "GOLDEN", small)
    monkeypatch.setattr(rag_eval, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(rag_eval, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(rag_eval, "RETRIEVAL_JSON", tmp_path / "retrieval.json")
    monkeypatch.setattr(rag_eval, "load_golden", lambda path=small: [
        json.loads(x) for x in path.read_text().splitlines()
    ])  # fmt: skip
    rag_eval.run_retrieval_stage()
    summary = rag_eval.run_e2e_stage("extractive")
    md = rag_eval.write_report().read_text(encoding="utf-8")
    assert "abstain_threshold" in (tmp_path / "rag_config.json").read_text()
    assert summary["abstention_rate_unanswerable"] >= 0
    assert "Retrieval ablation" in md and "Extractive" in md


def test_ask_cli_prints_answer_and_sources(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["ask", "What are the symptoms of diabetes?", "--provider", "extractive"]
    )
    ask.main()
    out = capsys.readouterr().out
    assert "ANSWER" in out and "niddk.nih.gov" in out


def test_real_threshold_file_is_used():
    ans = answer_question(
        "What is the capital of Australia?", retrieve.Retriever.load(), ExtractiveLLM()
    )
    assert ans.abstained and "retrieval gate" in ans.reason
