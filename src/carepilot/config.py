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
    groq_api_keys: tuple[str, ...]  # tried in order; next one used on rate limit / auth error
    groq_model: str
    ollama_host: str
    ollama_model: str


def parse_keys(*raw: str | None) -> tuple[str, ...]:
    """Split comma-separated key lists, strip blanks and quotes, drop duplicates, keep order."""
    keys = (k.strip().strip("\"'") for value in raw if value for k in value.split(","))
    return tuple(dict.fromkeys(k for k in keys if k))


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "auto").strip().lower(),
        groq_api_keys=parse_keys(os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEYS")),
        groq_model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
    )
