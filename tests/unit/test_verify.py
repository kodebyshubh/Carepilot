from carepilot.core.verify import ClaimCheck, premise_windows, split_claims, supported_rate


def test_split_claims_strips_markers_and_keeps_citations():
    claims = split_claims("Thirst is a common symptom [1, 3]. Blurred vision can occur too [2].")
    assert claims == [
        ("Thirst is a common symptom.", (1, 3)),
        ("Blurred vision can occur too.", (2,)),
    ]


def test_split_claims_handles_bullets_and_skips_intro_lines():
    text = "Most relevant statements from the documents:\n- Thirst is a common symptom. [1]"
    assert split_claims(text) == [("Thirst is a common symptom.", (1,))]


def test_uncited_sentence_has_no_citations():
    assert split_claims("This sentence cites nothing at all.") == [
        ("This sentence cites nothing at all.", ())
    ]


def test_premise_windows_include_whole_passage_sentences_and_pairs():
    passage = "Eat more vegetables daily. Sit less during the day. Walk for thirty minutes."
    windows = premise_windows(passage)
    assert windows[0] == passage
    assert "Sit less during the day." in windows
    assert "Eat more vegetables daily. Sit less during the day." in windows


def test_supported_rate():
    checks = [ClaimCheck("a", (1,), 0.9, True), ClaimCheck("b", (1,), 0.1, False)]
    assert supported_rate(checks) == 0.5
    assert supported_rate([]) is None
