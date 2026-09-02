"""Tests for the models.toml loader.

Before this existed, three docstrings claimed models.toml was the mechanism
for swapping models while nothing ever read the file.
"""

from __future__ import annotations

import pytest

from egoannote.backends.fake import FakeBackend
from egoannote.backends.registry import ModelNotFound, build_backend, load_registry


def _write(tmp_path, body: str):
    p = tmp_path / "models.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_shipped_registry_parses() -> None:
    """The models.toml committed at the repo root must actually be valid TOML
    with the structure the loader expects."""
    reg = load_registry()
    assert "models" in reg
    assert "fake" in reg["models"], "the zero-setup fake backend must be registered"
    flash = reg["models"]["qwen3.8-flash"]
    assert flash["model"] == "qwen/qwen3.8-flash"
    assert flash["price_in_per_mtok"] == 0.15
    assert flash["price_out_per_mtok"] == 0.47
    assert flash["reasoning_enabled"] is False


def test_fake_backend_builds_with_no_key(tmp_path) -> None:
    p = _write(tmp_path, '[models.fake]\nbackend = "fake"\nmax_images = 8\n')
    backend = build_backend("fake", registry_path=p)
    assert isinstance(backend, FakeBackend)


def test_unknown_model_lists_what_is_available(tmp_path) -> None:
    p = _write(tmp_path, '[models.fake]\nbackend = "fake"\n')
    with pytest.raises(ModelNotFound, match="Available: fake"):
        build_backend("nope", registry_path=p)


def test_missing_api_key_is_a_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    p = _write(
        tmp_path,
        '[backend.openai-compat]\nbase_url = "https://example.invalid/v1"\n\n'
        '[models.a]\nbackend = "openai-compat"\nmodel = "vendor/m"\n'
        "max_images = 8\nprice_in_per_mtok = 1.0\nprice_out_per_mtok = 2.0\n",
    )
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_backend("a", registry_path=p)


def test_base_url_falls_back_to_the_backend_default(tmp_path) -> None:
    p = _write(
        tmp_path,
        '[backend.openai-compat]\nbase_url = "https://example.invalid/v1"\n\n'
        '[models.a]\nbackend = "openai-compat"\nmodel = "vendor/m"\n'
        "max_images = 8\nprice_in_per_mtok = 1.0\nprice_out_per_mtok = 2.0\n",
    )
    backend = build_backend("a", registry_path=p, api_key="k")
    assert backend.model_id == "a"
    backend.close()


def test_api_key_in_the_toml_is_rejected_loudly(tmp_path, monkeypatch) -> None:
    """A key in a committed config file is a key in git history. Silently
    ignoring it would leave the user believing it was used; the unknown-key
    check rejects it by name instead."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    p = _write(
        tmp_path,
        '[backend.openai-compat]\nbase_url = "https://example.invalid/v1"\n\n'
        '[models.a]\nbackend = "openai-compat"\nmodel = "vendor/m"\n'
        'api_key = "sk-should-be-ignored"\nmax_images = 8\n'
        "price_in_per_mtok = 1.0\nprice_out_per_mtok = 2.0\n",
    )
    with pytest.raises(ValueError, match="api_key"):
        build_backend("a", registry_path=p)


def test_typo_in_a_price_key_is_rejected_not_defaulted(tmp_path) -> None:
    """THE bug this validation exists for: `entry.get("price_in_per_mtok",
    0.0)` turned a typo into a silent $0.00 price. The spend guard only
    fires when BOTH prices are zero, so one typo'd key sailed past it and
    the whole run was accounted at a fraction of the real rate."""
    p = _write(
        tmp_path,
        '[backend.openai-compat]\nbase_url = "https://example.invalid/v1"\n\n'
        '[models.a]\nbackend = "openai-compat"\nmodel = "vendor/m"\n'
        "max_images = 8\n"
        "price_in_per_mtoken = 1.0\n"  # typo: ...mtoken, not ...mtok
        "price_out_per_mtok = 2.0\n",
    )
    with pytest.raises(ValueError, match="price_in_per_mtoken"):
        build_backend("a", registry_path=p, api_key="k")


def test_openai_compat_entry_without_model_is_rejected(tmp_path) -> None:
    """Used to be a bare KeyError('model') with no file, no model_id, no hint."""
    p = _write(
        tmp_path,
        '[backend.openai-compat]\nbase_url = "https://example.invalid/v1"\n\n'
        '[models.a]\nbackend = "openai-compat"\nmax_images = 8\n',
    )
    with pytest.raises(ValueError, match="no 'model' key"):
        build_backend("a", registry_path=p, api_key="k")


def test_env_var_overrides_the_default_registry_path(tmp_path, monkeypatch) -> None:
    """config.py documents `parents[3]` as a known defect (only resolves for
    an editable install). registry.py had reintroduced it; the env var is the
    only thing that works under a wheel install, since models.toml is not
    bundled in the package."""
    p = _write(tmp_path, '[models.fake]\nbackend = "fake"\n')
    monkeypatch.setenv("EGOANNOTE_MODELS_TOML", str(p))
    import importlib

    from egoannote.backends import registry as reg
    importlib.reload(reg)
    try:
        assert reg.DEFAULT_REGISTRY_PATH == p
        assert isinstance(reg.build_backend("fake"), FakeBackend)
    finally:
        monkeypatch.delenv("EGOANNOTE_MODELS_TOML", raising=False)
        importlib.reload(reg)


def test_missing_registry_file_names_the_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Model registry not found"):
        build_backend("a", registry_path=tmp_path / "absent.toml")
