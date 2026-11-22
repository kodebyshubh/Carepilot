"""Critical user journeys through the Streamlit app (run: uv run pytest -m e2e)."""

import re

import pytest
from playwright.sync_api import expect

from tests.e2e.pages import MODEL_TIMEOUT_MS

pytestmark = pytest.mark.e2e


def test_app_loads_with_disclaimer_and_provider(app):
    expect(app.sidebar.get_by_text("Not a medical device")).to_be_visible()
    expect(app.sidebar.get_by_text(re.compile(r"^Using: "))).to_be_visible()


def test_high_risk_diabetes_patient_gets_risk_and_cited_explanation(app):
    app.choose_preset("High-risk example")
    app.estimate()
    expect(app.risk_metric).to_contain_text(re.compile(r"\d+%"), timeout=MODEL_TIMEOUT_MS)
    expect(app.band("High")).to_be_visible()
    expect(app.main.get_by_text("imputed with training medians")).to_be_visible()
    expect(app.sources_header).to_be_visible(timeout=MODEL_TIMEOUT_MS)
    expect(app.source_items.first).to_contain_text(re.compile(r"NIDDK|World Health"))


def test_typical_adult_is_not_high_risk(app):
    app.choose_preset("Typical adult")
    app.estimate()
    expect(app.risk_metric).to_be_visible(timeout=MODEL_TIMEOUT_MS)
    expect(app.band("High")).to_have_count(0)


def test_heart_high_risk_and_hypertensive_crisis_banner(app):
    app.choose_condition("Heart disease")
    app.choose_preset("High-risk example")
    app.set_number("Resting systolic blood pressure (mm Hg)", 190)
    app.estimate()
    expect(app.band("High")).to_be_visible(timeout=MODEL_TIMEOUT_MS)
    expect(app.emergency_banner.filter(has_text="hypertensive crisis")).to_be_visible()


def test_example_question_answered_with_sources(app):
    app.tab("Ask the documents")
    app.ask_example("What are the symptoms of diabetes?")
    expect(app.sources_header).to_be_visible(timeout=MODEL_TIMEOUT_MS)
    expect(app.source_items.first).to_contain_text("Symptoms")
    app.source_items.first.click()
    expect(app.main.get_by_role("link", name=re.compile(r"niddk\.nih\.gov"))).to_be_visible()


def test_unanswerable_question_states_not_in_documents(app):
    app.tab("Ask the documents")
    app.ask("What is the recommended starting dose of metformin?")
    expect(app.not_found).to_be_visible(timeout=MODEL_TIMEOUT_MS)
    expect(app.sources_header).to_have_count(0)


def test_red_flag_question_shows_emergency_guidance(app):
    app.tab("Ask the documents")
    app.ask("I have crushing chest pain spreading to my left arm")
    expect(app.emergency_banner.filter(has_text="heart attack")).to_be_visible(
        timeout=MODEL_TIMEOUT_MS
    )


def test_report_tab_shows_metrics_and_rag_eval(app):
    app.tab("Model & RAG report")
    expect(app.main.get_by_text(re.compile(r"Logistic Regression selected")).first).to_be_visible()
    expect(app.main.get_by_role("heading", name="RAG evaluation report")).to_be_visible()
    expect(app.main.get_by_test_id("stImage").first).to_be_visible()
