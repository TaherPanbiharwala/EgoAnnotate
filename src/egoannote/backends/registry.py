"""Load models.toml into ModelConfig objects.

Three docstrings previously advertised models.toml as the swap mechanism
("adding a provider is a models.toml entry, not a new class") while nothing
in the codebase ever parsed the file. This module makes that claim true.

tomllib is stdlib on 3.11+, so this adds no dependency.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

from .fake import FakeBackend
from .openai_compat import ModelConfig, OpenAICompatBackend, SpendTracker

log = logging.getLogger(__name__)

# EGOANNOTE_MODELS_TOML overrides; the repo-relative path is only the
# fallback. config.py documents `parents[3]` as a known v1 defect (it only
# resolves for an editable install from the repo), and review caught this
# module reintroducing the same pattern. pyproject packages only
# src/egoannote, so models.toml is NOT bundled in a wheel — under a wheel
# install the env var is the only thing that works.
_ENV_REGISTRY_PATH = os.environ.get("EGOANNOTE_MODELS_TOML")
DEFAULT_REGISTRY_PATH = (
    Path(_ENV_REGISTRY_PATH)
    if _ENV_REGISTRY_PATH
    else Path(__file__).resolve().parents[3] / "models.toml"
)

# A model entry's contract. Unknown keys are rejected rather than ignored:
# `entry.get("price_in_per_mtok", 0.0)` turns a typo into a silent $0.00
# price, which sails past the spend guard and mis-accounts the whole run.
_REQUIRED_KEYS = {"backend"}
_OPENAI_COMPAT_REQUIRED = {"model"}
_OPTIONAL_KEYS = {
    "base_url",
    "model",
    "max_images",
    "price_in_per_mtok",
    "price_out_per_mtok",
    "provider_order",
    # Declared in models.toml but not yet consumed by any code path. Listed
    # here so they don't trip the unknown-key check, and flagged in
    # models.toml so nobody expects them to do anything.
    "supports_json_schema",
    "image_detail",
    "tokens_per_image_fn",
}
_KNOWN_KEYS = _REQUIRED_KEYS | _OPENAI_COMPAT_REQUIRED | _OPTIONAL_KEYS


class ModelNotFound(KeyError):
    pass


def _validate_entry(model_id: str, entry: dict, path: Path) -> None:
    """Reject unknown/missing keys loudly. A mistyped price key is otherwise
    invisible: it defaults to 0.0, and the spend guard only fires when BOTH
    prices are zero, so one typo silently under-accounts the entire run."""
    unknown = set(entry) - _KNOWN_KEYS
    if unknown:
        raise ValueError(
            f"models.toml entry '{model_id}' ({path}) has unknown key(s): "
            f"{sorted(unknown)}.\n"
            f"  Known keys: {sorted(_KNOWN_KEYS)}\n"
            f"  A mistyped key would otherwise fall back to a default — for "
            f"price_* that means $0.00 and a spend cap that never fires."
        )
    missing = _REQUIRED_KEYS - set(entry)
    if missing:
        raise ValueError(
            f"models.toml entry '{model_id}' ({path}) is missing required "
            f"key(s): {sorted(missing)}"
        )


def load_registry(path: Path | None = None) -> dict:
    path = path or DEFAULT_REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model registry not found at {path}. This file defines which VLMs "
            f"are available; copy models.toml from the repo root, or pass an "
            f"explicit path."
        )
    with path.open("rb") as f:
        return tomllib.load(f)


def build_backend(
    model_id: str,
    *,
    registry_path: Path | None = None,
    api_key: str | None = None,
    max_spend_usd: float | None = None,
    spend_tracker: SpendTracker | None = None,
):
    """Construct a backend for `model_id` as declared in models.toml.

    The API key is read from $OPENROUTER_API_KEY (never from the TOML — a
    key in a committed config file is a key in your git history).
    """
    registry = load_registry(registry_path)
    models = registry.get("models", {})
    if model_id not in models:
        available = ", ".join(sorted(models)) or "(none)"
        raise ModelNotFound(
            f"'{model_id}' is not in the model registry. Available: {available}"
        )

    entry = models[model_id]
    _validate_entry(model_id, entry, registry_path or DEFAULT_REGISTRY_PATH)
    backend_name = entry.get("backend", "openai-compat")

    if backend_name == "fake":
        return FakeBackend(model_id=model_id)

    if backend_name != "openai-compat":
        raise ValueError(
            f"models.toml entry '{model_id}' declares backend={backend_name!r}, "
            f"but only 'openai-compat' and 'fake' are implemented."
        )

    base_url = entry.get("base_url") or registry.get("backend", {}).get(
        "openai-compat", {}
    ).get("base_url")
    if not base_url:
        raise ValueError(
            f"models.toml entry '{model_id}' has no base_url, and no default is "
            f"set under [backend.openai-compat]."
        )

    # OpenRouter routes one model id to different upstream providers, with
    # different quantizations, and can silently re-route mid-run. If model A
    # is served by two providers across a run, the two-model "agreement"
    # measures provider variance, not model disagreement — which is the
    # project's headline metric. Warn rather than hard-fail: a single
    # exploratory run doesn't care, a measurement run does.
    if "openrouter.ai" in base_url and not entry.get("provider_order"):
        log.warning(
            "models.toml entry '%s' targets OpenRouter without provider_order, "
            "so allow_fallbacks stays on and the upstream provider can change "
            "mid-run. Fine for exploration; pin it before any run whose "
            "agreement numbers you intend to report.",
            model_id,
        )

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            f"No API key for '{model_id}'. Set $OPENROUTER_API_KEY, or pass "
            f"api_key= explicitly. (Keys are never read from models.toml — that "
            f"file is committed.)"
        )

    if "model" not in entry:
        raise ValueError(
            f"models.toml entry '{model_id}' has backend='openai-compat' but no "
            f"'model' key. That is the provider-facing model string, e.g. "
            f"model = \"vendor/model-name\"."
        )

    cfg = ModelConfig(
        model_id=model_id,
        backend_name=backend_name,
        base_url=base_url,
        model=entry["model"],
        max_images=entry.get("max_images", 8),
        price_in_per_mtok=float(entry.get("price_in_per_mtok", 0.0)),
        price_out_per_mtok=float(entry.get("price_out_per_mtok", 0.0)),
        provider_order=entry.get("provider_order"),
    )
    return OpenAICompatBackend(
        cfg, key, max_spend_usd=max_spend_usd, spend_tracker=spend_tracker
    )
