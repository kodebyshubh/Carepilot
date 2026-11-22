"""Hybrid retrieval: dense (FAISS) + sparse (BM25), fused with Reciprocal Rank Fusion,
then reranked by a cross-encoder.

Why hybrid: dense embeddings capture paraphrase ("feeling thirsty" ~ "polydipsia") but can
miss exact tokens such as "A1C", "140/90" or drug names; BM25 is the opposite. RRF merges
the two rankings without needing their scores to be on the same scale.

Why rerank: the bi-encoder embeds query and passage separately; a cross-encoder reads them
together and scores relevance much more precisely. It is too slow to run over the whole
corpus, so it only rescores the fused top candidates. Its score is also what the
abstention gate uses ("is anything here actually relevant?").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from carepilot.config import EMBED_QUERY_PREFIX, INDEX_DIR, RERANK_MODEL
from carepilot.rag.chunking import Chunk
from carepilot.rag.index import get_embedder

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")

RRF_K = 60  # standard constant from Cormack et al. (2009); dampens the weight of top ranks
CANDIDATES = 20  # per retriever, before fusion / reranking

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "be",
        "been",
        "can",
        "do",
        "does",
        "what",
        "which",
        "who",
        "how",
        "when",
        "why",
        "should",
        "i",
        "my",
        "me",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "as",
        "at",
        "by",
        "from",
        "if",
        "not",
        "no",
        "so",
        "than",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:/[0-9]+)?", text.lower()) if t not in STOPWORDS]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # score of the final stage (cosine, BM25, RRF or cross-encoder logit)


@lru_cache(maxsize=1)
def get_reranker():  # -> CrossEncoder
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, device="cpu")


class Retriever:
    def __init__(self, chunks: list[Chunk], index: faiss.Index):
        if index.ntotal != len(chunks):
            raise ValueError(f"index has {index.ntotal} vectors but {len(chunks)} chunks")
        self.chunks = chunks
        self.index = index
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])

    @classmethod
    def load(cls) -> Retriever:
        path = INDEX_DIR / "faiss.index"
        if not path.exists():
            raise FileNotFoundError("index missing; run: python -m carepilot.rag.index")
        lines = (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        return cls([Chunk(**json.loads(x)) for x in lines], faiss.read_index(str(path)))

    def dense(self, query: str, k: int = CANDIDATES) -> list[tuple[int, float]]:
        q = get_embedder().encode([EMBED_QUERY_PREFIX + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    def sparse(self, query: str, k: int = CANDIDATES) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def rerank(self, query: str, ids: list[int]) -> list[tuple[int, float]]:
        if not ids:
            return []
        pairs = [(query, self.chunks[i].embed_text) for i in ids]
        scores = get_reranker().predict(pairs, show_progress_bar=False)
        return sorted(zip(ids, map(float, scores), strict=True), key=lambda x: x[1], reverse=True)

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid_rerank") -> list[Hit]:
        if not query.strip():
            return []
        if mode == "dense":
            ranked = self.dense(query)
        elif mode == "bm25":
            ranked = self.sparse(query)
        else:
            dense, sparse = self.dense(query), self.sparse(query)
            ranked = rrf([[i for i, _ in dense], [i for i, _ in sparse]])
            if mode == "hybrid_rerank":
                ranked = self.rerank(query, [i for i, _ in ranked[:CANDIDATES]])
        return [Hit(self.chunks[i], s) for i, s in ranked[:k]]
