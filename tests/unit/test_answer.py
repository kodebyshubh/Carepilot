"""answer_question logic with a fake retriever and fake LLMs (no models, no network)."""

from dataclasses import dataclass

import pytest

from carepilot.llm.providers import ExtractiveLLM, LLMError
from carepilot.rag.answer import (
    NOT_FOUND_MESSAGE,
    SENTINEL,
    answer_question,
    extractive_answer,
    flatten_lists,
    validate_citations,
)
from carepilot.rag.chunking import Chunk
from carepilot.rag.retrieve import Hit, rrf, tokenize


def _hit(i: int, text: str, score: float) -> Hit:
    return Hit(Chunk(f"d#{i}", "d", "Doc", "NIH", "https://x", f"S{i}", text), score)


HITS = [
    _hit(0, "Common symptoms include\n\n- increased thirst\n- blurred vision", 6.0),
    _hit(1, "Being active and losing weight can prevent type 2 diabetes.", 2.0),
]


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=5, mode="hybrid_rerank"):
        return self.hits[:k]


@dataclass
class FakeLLM:
    reply: str | Exception
    name: str = "fake"
    calls: int = 0

    def generate(self, system, user):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_retrieval_gate_abstains_without_calling_llm():
    llm = FakeLLM("should not be used")
    ans = answer_question(
        "capital of France?", FakeRetriever([_hit(0, "x", -9.0)]), llm, threshold=-3
    )
    assert ans.abstained and ans.text == NOT_FOUND_MESSAGE and llm.calls == 0


def test_empty_retrieval_abstains():
    ans = answer_question("q", FakeRetriever([]), FakeLLM("x"), threshold=-3)
    assert ans.abstained


def test_generation_gate_sentinel_abstains():
    ans = answer_question("q", FakeRetriever(HITS), FakeLLM(SENTINEL), threshold=-3)
    assert ans.abstained and "generation gate" in ans.reason


def test_answer_keeps_only_cited_sources():
    ans = answer_question("q", FakeRetriever(HITS), FakeLLM("Thirst is common [1]."), threshold=-3)
    assert not ans.abstained
    assert [s.n for s in ans.sources] == [1]


def test_invalid_citations_removed():
    text, cited = validate_citations("Fact one [1, 7]. Fact two [9].", n_passages=2)
    assert text == "Fact one [1]. Fact two."
    assert cited == [1]


def test_uncited_llm_answer_replaced_by_extractive():
    ans = answer_question(
        "symptoms", FakeRetriever(HITS), FakeLLM("Thirst, no citation."), threshold=-3
    )
    assert "extractive fallback" in ans.reason
    assert ans.sources


def test_llm_error_falls_back_to_extractive():
    ans = answer_question("symptoms", FakeRetriever(HITS), FakeLLM(LLMError("down")), threshold=-3)
    assert not ans.abstained and "llm error" in ans.reason


def test_extractive_provider_never_calls_generate():
    ans = answer_question("symptoms thirst", FakeRetriever(HITS), ExtractiveLLM(), threshold=-3)
    assert "[1]" in ans.text


def test_flatten_lists_attaches_bullets_to_intro():
    assert flatten_lists("Symptoms include\n\n- thirst\n- fatigue") == (
        "Symptoms include: thirst; fatigue."
    )


def test_extractive_answer_cites_passages():
    text, cited = extractive_answer("blurred vision symptoms", HITS)
    assert "blurred vision" in text and cited[0] == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [("What is A1C?", ["a1c"]), ("BP 140/90 mmHg", ["bp", "140/90", "mmhg"])],
)
def test_tokenize_keeps_medical_tokens(text, expected):
    assert tokenize(text) == expected


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert fused[0][0] == 1  # ranks 1 and 2
    assert {d for d, _ in fused} == {1, 2, 3, 4}


def test_cited_answer_with_appended_sentinel_is_kept():
    reply = f"Thirst is common [1].\n\n{SENTINEL}"
    ans = answer_question("q", FakeRetriever(HITS), FakeLLM(reply), threshold=-3)
    assert not ans.abstained
    assert SENTINEL not in ans.text and "partial coverage" in ans.reason
