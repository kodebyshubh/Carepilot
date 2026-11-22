"""Real index + real embedding/reranker models (downloaded on first run, CPU)."""

import pytest

from carepilot.llm.providers import ExtractiveLLM
from carepilot.rag.answer import answer_question
from carepilot.rag.retrieve import MODES, Retriever

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def retriever():
    return Retriever.load()


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_the_symptoms_page(retriever, mode):
    hits = retriever.search("What are the symptoms of diabetes?", k=5, mode=mode)
    assert any(h.chunk.doc_id == "niddk_diabetes_symptoms" for h in hits)


def test_exact_term_a1c_retrieved(retriever):
    hits = retriever.search("What A1C goal is recommended?", k=3)
    assert hits[0].chunk.doc_id == "niddk_managing_diabetes"


def test_answer_with_sources_end_to_end(retriever):
    ans = answer_question(
        "What lifestyle changes lower blood pressure?", retriever, ExtractiveLLM()
    )
    assert not ans.abstained and ans.sources
    assert all(s.url.startswith("https://") for s in ans.sources)


def test_off_topic_question_abstains(retriever):
    ans = answer_question("Who won the football world cup in 2018?", retriever, ExtractiveLLM())
    assert ans.abstained
