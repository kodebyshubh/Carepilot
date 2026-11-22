"""Central paths and runtime settings. Settings come from environment (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOCS_RAW_DIR = DATA_DIR / "docs" / "raw"
DOCS_CLEAN_DIR = DATA_DIR / "docs" / "clean"
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = ROOT / "models"
INDEX_DIR = ROOT / "index"
REPORTS_DIR = ROOT / "reports"

SEED = 42

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge-small recommends this instruction prefix on queries (not on passages) for retrieval.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    groq_api_key: str | None
    groq_model: str
    ollama_host: str
    ollama_model: str


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "auto").strip().lower(),
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
    )
