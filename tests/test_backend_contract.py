"""Regression tests for bugs found in review of the initial implementation.

Every test here pins a defect that shipped in the first draft. They are
deliberately specific: each one fails if the exact bug returns.
"""

from __future__ import annotations

import json

import httpx
import pytest

from egoannote import config
from egoannote.backends.openai_compat import (
    ModelConfig,
    OpenAICompatBackend,
    PermanentBackendError,
    SpendTracker,
)


def _cfg(**kw) -> ModelConfig:
    base = dict(
        model_id="test-a",
        backend_name="openai-compat",
        base_url="https://example.invalid/api/v1",
        model="vendor/model-a",
        max_images=8,
        price_in_per_mtok=1.0,
        price_out_per_mtok=2.0,
    )
    base.update(kw)
    return ModelConfig(**base)


def _backend(handler, **kw) -> OpenAICompatBackend:
    b = OpenAICompatBackend(_cfg(), "test-key", **kw)
    b._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url=_cfg().base_url
    )
    return b


def _ok_body(text: str = '{"actions":[]}') -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "provider": "TestProvider",
        "model": "vendor/model-a",
    }


# --------------------------------------------------------------------------
# BUG: per-frame labels were never sent, while the prompt told the model each
# image is labeled "Frame N" and asked it to return start_frame/end_frame
# referring to those labels. Every temporal index the model returned was
# therefore unanchored — which is the signal the whole segmentation design
# consumes. This is the single most consequential defect found in review.
# --------------------------------------------------------------------------


def test_each_image_is_preceded_by_its_frame_label() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_body())

    b = _backend(handler)
    b.caption([b"a", b"b", b"c"], "INSTRUCTIONS")

    content = captured["body"]["messages"][0]["content"]
    # Expected shape: instruction, then (label, image) pairs in order.
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "INSTRUCTIONS"

    for i in range(3):
        label_part = content[1 + i * 2]
        image_part = content[2 + i * 2]
        assert label_part["type"] == "text", f"frame {i} has no label part"
        assert label_part["text"] == f"Frame {i}", (
            f"frame {i} label is {label_part['text']!r}; the caption prompt "
            f"tells the model to expect 'Frame {i}'"
        )
        assert image_part["type"] == "image_url", f"frame {i} image not after its label"

    b.close()


def test_frame_labels_are_zero_indexed_matching_the_prompt() -> None:
    """The prompt says N = 0..7. A 1-indexed drift here would silently shift
    every boundary the model reports by one frame (0.75s at 8 frames / 6s)."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_body())

    b = _backend(handler)
    b.caption([b"x"] * config.VLM_WINDOW_FRAMES, "p")
    labels = [
        c["text"]
        for c in captured["body"]["messages"][0]["content"]
        if c["type"] == "text" and c["text"].startswith("Frame ")
    ]
    assert labels[0] == "Frame 0"
    assert labels[-1] == f"Frame {config.VLM_WINDOW_FRAMES - 1}"
    b.close()


# --------------------------------------------------------------------------
# BUG: config declared VLM_MAX_NEW_TOKENS and VLM_TEMPERATURE with a comment
# explaining the truncation headroom, and never sent either. Provider
# defaults applied instead — and default sampling contaminates the two-model
# agreement metric with sampling variance.
# --------------------------------------------------------------------------


def test_max_tokens_and_temperature_are_actually_sent() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_body())

    b = _backend(handler)
    b.caption([b"x"], "p")
    assert captured["body"]["max_tokens"] == config.VLM_MAX_NEW_TOKENS
    assert captured["body"]["temperature"] == config.VLM_TEMPERATURE
    b.close()


# --------------------------------------------------------------------------
# BUG: the spend guard multiplied provider-reported tokens by models.toml
# prices, which ship as 0.0 placeholders. Every call estimated $0.00, the
# running total never rose, and max_spend_usd silently never fired — an
# unlimited budget that looked like a configured one.
# --------------------------------------------------------------------------


def test_zero_priced_model_with_a_spend_cap_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="price"):
        OpenAICompatBackend(
            _cfg(price_in_per_mtok=0.0, price_out_per_mtok=0.0),
            "k",
            max_spend_usd=5.0,
        )


def test_missing_usage_block_does_not_count_as_zero_cost() -> None:
    """A provider that omits `usage` must not silently bill $0 against the cap."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = _ok_body()
        del body["usage"]
        return httpx.Response(200, json=body)

    b = _backend(handler, max_spend_usd=5.0)
    with pytest.raises(PermanentBackendError, match="usage"):
        b.caption([b"x"], "p")
    b.close()


def test_spend_accumulates_and_the_cap_eventually_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
            },
        )

    from egoannote.backends.base import SpendLimitExceeded

    b = _backend(handler, max_spend_usd=2.5)  # 1M in-tokens @ $1/Mtok = $1.00/call
    b.caption([b"x"], "p")
    b.caption([b"x"], "p")
    b.caption([b"x"], "p")  # total now $3.00, over the $2.50 cap
    with pytest.raises(SpendLimitExceeded):
        b.caption([b"x"], "p")
    b.close()


# --------------------------------------------------------------------------
# BUG: a permanently-failing request (bad key, bad model id) was retried,
# burning 1s + 2s of backoff per window for a failure that can never succeed.
# --------------------------------------------------------------------------


def test_auth_failure_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    b = _backend(handler, retries=2)
    with pytest.raises(PermanentBackendError, match="401"):
        b.caption([b"x"], "p")
    assert calls["n"] == 1, "a 401 is permanent — retrying wastes 3s per window"
    b.close()


def test_transient_error_is_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": "upstream busy"})
        return httpx.Response(200, json=_ok_body())

    b = _backend(handler, retries=2)
    resp = b.caption([b"x"], "p")
    assert calls["n"] == 3
    assert resp.provider == "TestProvider"
    b.close()


# --------------------------------------------------------------------------
# BUG: the API key is attached to every request, but nothing checked that the
# endpoint was https. models.toml advertises local http backends, so a
# remote http:// typo would put the key on the wire in cleartext.
# --------------------------------------------------------------------------


def test_remote_http_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="https"):
        OpenAICompatBackend(_cfg(base_url="http://api.example.com/v1"), "k")


def test_localhost_http_is_allowed_for_local_model_servers() -> None:
    b = OpenAICompatBackend(_cfg(base_url="http://localhost:11434/v1"), "k")
    assert b.model_id == "test-a"
    b.close()


# --------------------------------------------------------------------------
# BUG: a model declaring fewer images than a window needs raised per-call,
# producing an error row for every window instead of failing at startup.
# --------------------------------------------------------------------------


def test_model_too_small_for_a_window_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="max_images"):
        OpenAICompatBackend(_cfg(max_images=config.VLM_WINDOW_FRAMES - 1), "k")


def test_placeholder_model_string_is_rejected() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        OpenAICompatBackend(_cfg(model="REPLACE_ME/model-a"), "k")


# --------------------------------------------------------------------------
# SpendTracker: documented as "shared across parallel workers" while using a
# non-atomic read-modify-write.
# --------------------------------------------------------------------------


def test_spend_tracker_is_thread_safe() -> None:
    import threading

    tracker = SpendTracker()

    def worker() -> None:
        for _ in range(1000):
            tracker.add(0.001)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.total_usd == pytest.approx(8 * 1000 * 0.001)
