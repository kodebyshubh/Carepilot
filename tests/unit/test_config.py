from carepilot import config


def test_paths_are_under_project_root():
    assert config.MODELS_DIR.parent == config.ROOT
    assert (config.ROOT / "pyproject.toml").exists()


def test_settings_default_provider_is_auto(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert config.get_settings().llm_provider == "auto"


def test_settings_empty_groq_key_gives_no_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    assert config.get_settings().groq_api_keys == ()


def test_parse_keys_splits_strips_and_dedupes():
    assert config.parse_keys(" a , b,,'c' ", "b,d", None) == ("a", "b", "c", "d")
