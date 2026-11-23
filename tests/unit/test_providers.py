import pytest

from carepilot.config import Settings
from carepilot.llm import providers
from carepilot.llm.providers import ExtractiveLLM, GroqLLM, LLMError, OllamaLLM, get_llm


def _settings(provider="auto", key=None):
    keys = (key,) if key else ()
    return Settings(provider, keys, "qwen/qwen3.8-27b", "http://localhost:1", "qwen2.5:3b")


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


def _status_error(cls, status):
    import httpx

    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"))
    return cls("error", response=response, body=None)


class ScriptedGroq(GroqLLM):
    """GroqLLM whose network call is replaced by a per-key script of results."""

    def __init__(self, script):
        super().__init__(tuple(script), "m")
        self.script = script
        self.calls = []

    def _call(self, key, system, user):
        self.calls.append(key)
        outcome = self.script[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_rate_limited_key_fails_over_to_next_and_is_remembered():
    from groq import RateLimitError

    llm = ScriptedGroq({"k1": _status_error(RateLimitError, 429), "k2": "ok"})
    assert llm.generate("s", "u") == "ok"
    assert llm.calls == ["k1", "k2"]
    llm.calls.clear()
    llm.generate("s", "u")
    assert llm.calls == ["k2"]  # starts from the key that worked


def test_invalid_key_is_skipped():
    from groq import AuthenticationError

    llm = ScriptedGroq({"bad": _status_error(AuthenticationError, 401), "good": "ok"})
    assert llm.generate("s", "u") == "ok"


def test_all_keys_exhausted_raises_without_leaking_keys():
    from groq import RateLimitError

    llm = ScriptedGroq({"secret1": _status_error(RateLimitError, 429),
                        "secret2": _status_error(RateLimitError, 429)})  # fmt: skip
    with pytest.raises(LLMError, match="all Groq keys failed") as exc:
        llm.generate("s", "u")
    assert "secret" not in str(exc.value)


def test_other_groq_errors_do_not_rotate():
    from groq import InternalServerError

    llm = ScriptedGroq({"k1": _status_error(InternalServerError, 500), "k2": "ok"})
    with pytest.raises(LLMError, match="InternalServerError"):
        llm.generate("s", "u")
    assert llm.calls == ["k1"]
