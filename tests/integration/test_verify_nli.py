"""Real NLI model: quotes pass, fabrications fail."""

import pytest

from carepilot.core.verify import verify_claims
from carepilot.rag.answer import Source

pytestmark = pytest.mark.integration

PASSAGE = (
    "Lifestyle changes can help lower high blood pressure and can help anyone with "
    "hypertension. Many who make these changes will still need to take medicine.\n\n"
    "These lifestyle changes can help prevent and lower high blood pressure.\n\nDo:\n\n"
    "- Eat more vegetables and fruits.\n- Sit less."
)
SOURCES = (Source(1, "Hypertension", "Prevention", "WHO", "https://who.int", PASSAGE, 5.0),)


def test_verbatim_claim_supported_even_in_long_list_passage():
    checks = verify_claims(
        "These lifestyle changes can help prevent and lower high blood pressure [1].", SOURCES
    )
    assert checks[0].supported


def test_fabricated_and_contradicting_claims_flagged():
    checks = verify_claims(
        "Drinking green tea daily cures hypertension [1]. "
        "People who change their lifestyle never need medicine [1].",
        SOURCES,
    )
    assert not any(c.supported for c in checks)


def test_uncited_sentence_is_not_supported():
    checks = verify_claims("Eating vegetables helps lower blood pressure.", SOURCES)
    assert checks[0].entailment is None and not checks[0].supported
