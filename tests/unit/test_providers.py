import pytest

from carepilot.config import Settings
from carepilot.llm import providers
from carepilot.llm.providers import ExtractiveLLM, GroqLLM, LLMError, OllamaLLM, get_llm


def _settings(provider="auto", key=None):
    return Settings(provider, key, "llama-3.3-70b-versatile", "http://localhost:1", "qwen2.5:3b")


def test_auto_prefers_groq_when_key_present():
    assert isinstance(get_llm(settings=_settings(key="k")), GroqLLM)


def test_auto_uses_ollama_when_no_key_and_ollama_up(monkeypatch):
    monkeypatch.setattr(providers, "ollama_available", lambda h, m: True)
    assert isinstance(get_llm(settings=_settings()), OllamaLLM)


def test_auto_falls_back_to_extractive(monkeypatch):
    monkeypatch.setattr(providers, "ollama_available", lambda h, m: False)
    assert isinstance(get_llm(settings=_settings()), ExtractiveLLM)


def test_explicit_groq_without_key_errors():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        get_llm("groq", settings=_settings())


def test_unknown_provider_errors():
    with pytest.raises(LLMError, match="unknown"):
        get_llm("gpt5", settings=_settings())


def test_ollama_unreachable_reports_unavailable():
    assert providers.ollama_available("http://localhost:1", "x") is False


def test_ollama_generate_wraps_network_errors():
    with pytest.raises(LLMError):
        OllamaLLM("http://localhost:1", "x").generate("s", "u")
