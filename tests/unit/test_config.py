from carepilot import config


def test_paths_are_under_project_root():
    assert config.MODELS_DIR.parent == config.ROOT
    assert (config.ROOT / "pyproject.toml").exists()


def test_settings_default_provider_is_auto(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert config.get_settings().llm_provider == "auto"


def test_settings_empty_groq_key_is_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert config.get_settings().groq_api_key is None
