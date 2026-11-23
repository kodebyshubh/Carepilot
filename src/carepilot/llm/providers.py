"""LLM providers behind one tiny interface, plus automatic selection.

  groq       Qwen3.8 27B via Groq's free API (needs GROQ_API_KEY)
  ollama     a local model through Ollama's HTTP API (no data leaves the machine)
  extractive no LLM at all; the answer is built from retrieved sentences (always works)

`auto` picks the first one that is usable: groq -> ollama -> extractive.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import requests

from carepilot.config import Settings, get_settings

log = logging.getLogger(__name__)
T = TypeVar("T")

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
    """Groq chat completion with key failover.

    Several keys can be configured (GROQ_API_KEY="k1,k2,..."). Keys are tried in order
    starting from the last one that worked; a rate-limit (429) or rejected key (401/403)
    moves on to the next key. Any other error, or running out of keys, raises LLMError and
    the caller falls back to an extractive answer. Key values are never logged, only their
    position in the list.
    """

    api_keys: tuple[str, ...]
    model: str
    name: str = "groq"
    _active: int = 0  # index of the key that last succeeded

    def _call(self, key: str, system: str, user: str) -> str:
        from groq import Groq

        client = Groq(api_key=key, timeout=60, max_retries=1)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""

    def _open_stream(self, key: str, system: str, user: str) -> Iterator[Any]:
        from groq import Groq

        client = Groq(api_key=key, timeout=60, max_retries=1)
        return client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

    def _with_failover(self, request: Callable[[str], T]) -> T:
        """Run request(key) with each key in turn until one is not rate-limited or rejected."""
        from groq import AuthenticationError, GroqError, PermissionDeniedError, RateLimitError

        if not self.api_keys:
            raise LLMError("no Groq API key configured")
        n = len(self.api_keys)
        failures: list[str] = []
        for step in range(n):
            i = (self._active + step) % n
            try:
                result = request(self.api_keys[i])
            except (RateLimitError, AuthenticationError, PermissionDeniedError) as exc:
                failures.append(f"key #{i + 1}: {type(exc).__name__}")
                log.warning(
                    "Groq key #%d of %d failed (%s), trying next", i + 1, n, type(exc).__name__
                )
                continue
            except GroqError as exc:
                raise LLMError(f"Groq call failed: {type(exc).__name__}") from exc
            self._active = i
            return result
        raise LLMError("all Groq keys failed: " + "; ".join(failures))

    def generate(self, system: str, user: str) -> str:
        return self._with_failover(lambda key: self._call(key, system, user))

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield text deltas. Key failover applies when the request is opened."""
        from groq import GroqError

        events = self._with_failover(lambda key: self._open_stream(key, system, user))
        try:
            for event in events:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta
        except GroqError as exc:
            raise LLMError(f"Groq stream failed: {type(exc).__name__}") from exc

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

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield text deltas from Ollama's newline-delimited JSON stream."""
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
                },
                timeout=OLLAMA_TIMEOUT_S,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        delta = json.loads(line).get("message", {}).get("content", "")
                        if delta:
                            yield delta
        except (requests.RequestException, ValueError) as exc:
            raise LLMError(f"Ollama stream failed: {exc}") from exc

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
    if choice == "groq" or (choice == "auto" and s.groq_api_keys):
        if not s.groq_api_keys:
            raise LLMError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        return GroqLLM(s.groq_api_keys, s.groq_model)
    if choice == "ollama" or (choice == "auto" and ollama_available(s.ollama_host, s.ollama_model)):
        return OllamaLLM(s.ollama_host, s.ollama_model)
    if choice in ("extractive", "auto"):
        return ExtractiveLLM()
    raise LLMError(f"unknown LLM_PROVIDER {choice!r}; use auto, groq, ollama or extractive")
