import pytest

from carepilot.eval.rag_eval import (
    calibrate_threshold,
    first_relevant_rank,
    key_fact_coverage,
    leave_one_out,
    load_golden,
    sentence_citation_coverage,
)


def test_threshold_separates_clean_split_at_margin_midpoint():
    c = calibrate_threshold(pos=[2.0, 5.0, 0.0], neg=[-4.0, -10.0])
    assert c["abstain_threshold"] == -2.0
    assert c["answerable_kept"] == 1 and c["unanswerable_rejected"] == 1


def test_threshold_handles_overlap():
    c = calibrate_threshold(pos=[3.0, -5.0, 4.0], neg=[-6.0, -1.0])
    assert 0 < c["balanced_accuracy"] < 1


def test_leave_one_out_is_not_better_than_in_sample():
    rows = [{"answerable": True, "top_score": s} for s in (3, 2, 0.5, -0.2)] + [
        {"answerable": False, "top_score": s} for s in (-0.5, -4, -9)
    ]
    loo = leave_one_out(rows)
    assert loo["loo_answerable_kept"] <= 1 and loo["loo_unanswerable_rejected"] <= 1


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], {"c"}) == 3
    assert first_relevant_rank(["a"], {"z"}) is None


def test_key_fact_coverage_accepts_alternatives():
    facts = [["thirst", "thirsty"], ["urinat"], ["vision"]]
    assert key_fact_coverage("Feeling thirsty and urinating often.", facts) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Thirst is common [1]. Vision may blur [2].", 1.0),
        ("- Thirst is a common symptom. [1]\n- Vision may blur over time. [2]", 1.0),
        ("Thirst is a common symptom [1]. This sentence has no citation.", 0.5),
    ],
)
def test_sentence_citation_coverage(text, expected):
    assert sentence_citation_coverage(text) == expected


def test_golden_set_is_well_formed():
    golden = load_golden()
    assert len({g["id"] for g in golden}) == len(golden) == 40
    for g in golden:
        assert g["answerable"] == bool(g["relevant_docs"]) == bool(g["key_facts"])
