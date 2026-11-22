"""Build the retrieval index: chunks -> bge-small embeddings -> FAISS, plus chunk store.

Outputs in index/:
  chunks.jsonl   one Chunk per line (BM25 is rebuilt from these at load time, it is cheap)
  faiss.index    IndexFlatIP over L2-normalized vectors (inner product == cosine)
  index_meta.json  embedding model, dimension, counts, build time

Usage: python -m carepilot.rag.index
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache

import faiss
import numpy as np

from carepilot.config import DOCS_CLEAN_DIR, EMBED_MODEL, INDEX_DIR
from carepilot.rag.chunking import Chunk, chunk_document

log = logging.getLogger(__name__)
EMBED_BATCH = 32


@lru_cache(maxsize=1)
def get_embedder():  # -> SentenceTransformer (import is slow, keep lazy)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed_passages(texts: list[str]) -> np.ndarray:
    vecs = get_embedder().encode(
        texts, batch_size=EMBED_BATCH, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype="float32")


def load_corpus_chunks() -> list[Chunk]:
    manifest = json.loads((DOCS_CLEAN_DIR / "manifest.json").read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for doc in manifest:
        md = (DOCS_CLEAN_DIR / f"{doc['doc_id']}.md").read_text(encoding="utf-8")
        chunks.extend(chunk_document(doc["doc_id"], doc["title"], doc["publisher"], doc["url"], md))
    return chunks


def build_index() -> dict:
    chunks = load_corpus_chunks()
    vectors = embed_passages([c.embed_text for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    lengths = [len(c.text) for c in chunks]
    meta = {
        "embed_model": EMBED_MODEL,
        "dim": int(vectors.shape[1]),
        "n_chunks": len(chunks),
        "n_docs": len({c.doc_id for c in chunks}),
        "chunk_chars_mean": round(float(np.mean(lengths)), 1),
        "chunk_chars_max": int(np.max(lengths)),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (INDEX_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = build_index()
    log.info("indexed %(n_chunks)d chunks from %(n_docs)d docs, dim %(dim)d", meta)


if __name__ == "__main__":
    main()
