"""LLM providers behind one tiny interface, plus automatic selection.

  groq       Llama 3.3 70B via Groq's free API (needs GROQ_API_KEY)
  ollama     a local model through Ollama's HTTP API (no data leaves the machine)
  extractive no LLM at all; the answer is built from retrieved sentences (always works)

`auto` picks the first one that is usable: groq -> ollama -> extractive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import requests

from carepilot.config import Settings, get_settings

log = logging.getLogger(__name__)

TEMPERATURE = 0.1  # near-deterministic: we want faithful restatement, not creativity
MAX_TOKENS = 600
OLLAMA_TIMEOUT_S = 120
OLLAMA_PROBE_TIMEOUT_S = 1.5


class LLMError(RuntimeError):
    """Provider call failed (network, auth, rate limit, bad response)."""


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str: ...


@dataclass
class GroqLLM:
    api_key: str
    model: str
    name: str = "groq"

    def generate(self, system: str, user: str) -> str:
        from groq import Groq, GroqError

        try:
            client = Groq(api_key=self.api_key, timeout=60, max_retries=2)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except GroqError as exc:
            raise LLMError(f"Groq call failed: {type(exc).__name__}") from exc
        return resp.choices[0].message.content or ""

    @property
    def label(self) -> str:
        return f"Groq ({self.model})"


@dataclass
class OllamaLLM:
    host: str
    model: str
    name: str = "ollama"

    def generate(self, system: str, user: str) -> str:
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
                },
                timeout=OLLAMA_TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc

    @property
    def label(self) -> str:
        return f"Ollama local ({self.model})"


@dataclass
class ExtractiveLLM:
    """Marker provider: answer.py builds the answer from retrieved sentences instead."""

    name: str = "extractive"

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - never called
        raise NotImplementedError("extractive mode does not call an LLM")

    @property
    def label(self) -> str:
        return "Extractive (no LLM)"


def ollama_available(host: str, model: str) -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=OLLAMA_PROBE_TIMEOUT_S)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
    except (requests.RequestException, ValueError):
        return False
    return model in names or f"{model}:latest" in names


def get_llm(provider: str | None = None, settings: Settings | None = None) -> LLM:
    s = settings or get_settings()
    choice = (provider or s.llm_provider).lower()
    if choice == "groq" or (choice == "auto" and s.groq_api_key):
        if not s.groq_api_key:
            raise LLMError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        return GroqLLM(s.groq_api_key, s.groq_model)
    if choice == "ollama" or (choice == "auto" and ollama_available(s.ollama_host, s.ollama_model)):
        return OllamaLLM(s.ollama_host, s.ollama_model)
    if choice in ("extractive", "auto"):
        return ExtractiveLLM()
    raise LLMError(f"unknown LLM_PROVIDER {choice!r}; use auto, groq, ollama or extractive")
