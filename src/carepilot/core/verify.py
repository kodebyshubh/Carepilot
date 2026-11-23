"""Claim verification: does each answer sentence actually follow from the passage it cites?

Citations prove the model pointed somewhere, not that the passage says what the sentence
claims. A natural-language-inference (NLI) cross-encoder reads (passage, sentence) and
predicts entailment / neutral / contradiction. A sentence counts as supported when at least
one of its cited passages entails it. Unsupported sentences are flagged in the UI and
counted in the evaluation, they are not silently removed: the reader decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from carepilot.config import NLI_MODEL
from carepilot.rag.answer import Source, flatten_lists

ENTAILMENT_THRESHOLD = 0.5
MIN_SENTENCE_CHARS = 20
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\[)|\n+")


@dataclass(frozen=True)
class ClaimCheck:
    sentence: str  # without citation markers
    cited: tuple[int, ...]
    entailment: float | None  # best entailment probability over cited passages; None = uncited
    supported: bool


@lru_cache(maxsize=1)
def get_nli():  # -> CrossEncoder
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(NLI_MODEL, device="cpu")
    labels = {v.lower(): k for k, v in model.model.config.id2label.items()}
    return model, labels["entailment"]


def split_claims(answer: str) -> list[tuple[str, tuple[int, ...]]]:
    """[(sentence without markers, cited passage numbers)] for substantive sentences."""
    claims = []
    for raw in SENTENCE_SPLIT.split(answer):
        raw = raw.strip().lstrip("- ").strip()
        if len(raw) < MIN_SENTENCE_CHARS or raw.endswith(":"):
            continue
        nums = tuple(
            sorted({int(n) for group in CITATION.findall(raw) for n in re.split(r"\s*,\s*", group)})
        )
        text = re.sub(r"\s+([.,;])", r"\1", CITATION.sub("", raw)).strip()
        claims.append((text, nums))
    return claims


def premise_windows(passage: str, context: str = "") -> list[str]:
    """The whole passage plus every window of 1-2 consecutive sentences.

    NLI models are trained on short premises; given a long, list-heavy passage they can call
    a verbatim quote a contradiction (observed: 0.90 contradiction for a copied sentence).
    Scoring short windows and taking the best keeps the check faithful to what NLI is good at.
    """
    units = [
        u.strip(" -")
        for u in re.split(r"(?<=[.!?])\s+|\n+", flatten_lists(passage))
        if len(u.strip(" -")) >= MIN_SENTENCE_CHARS
    ]
    windows = [
        passage,
        *units,
        *(f"{a} {b}" for a, b in zip(units, units[1:], strict=False)),
        *bullet_sentences(passage, context),
    ]
    return list(dict.fromkeys(windows))


def bullet_sentences(passage: str, context: str = "") -> list[str]:
    """One premise per bullet: "<line introducing the list> <item>".

    NLI judges "Symptoms include: thirst; ...; losing weight unintentionally" as neutral to
    "Unintentional weight loss is a symptom", but entails it from "Symptoms of diabetes
    include losing weight unintentionally". A bullet run with no intro line in the chunk
    (the list started in the previous chunk) uses the section heading as its intro.
    """
    out: list[str] = []
    intro = context
    for line in passage.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            item = line[2:].rstrip(".;")
            out.append(f"{intro.rstrip(':.')} {item}.".strip())
        else:
            intro = line
    return out


def verify_claims(answer: str, sources: tuple[Source, ...]) -> list[ClaimCheck]:
    windows = {
        s.n: premise_windows(s.text, s.section.split(" > ")[-1] if s.section else s.title)
        for s in sources
    }
    claims = split_claims(answer)
    pairs, owners = [], []
    for ci, (text, nums) in enumerate(claims):
        for n in nums:
            for premise in windows.get(n, []):
                pairs.append((premise, text))
                owners.append(ci)
    best: dict[int, float] = {}
    if pairs:
        model, entail_idx = get_nli()
        probs = model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        for ci, p in zip(owners, probs, strict=True):
            best[ci] = max(best.get(ci, 0.0), float(p[entail_idx]))
    return [
        ClaimCheck(
            sentence=text,
            cited=nums,
            entailment=best.get(ci),
            supported=best.get(ci, 0.0) >= ENTAILMENT_THRESHOLD,
        )
        for ci, (text, nums) in enumerate(claims)
    ]


def supported_rate(checks: list[ClaimCheck]) -> float | None:
    return sum(c.supported for c in checks) / len(checks) if checks else None
