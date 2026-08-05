"""FakeBackend must satisfy the VLMBackend contract with zero network calls,
so the full pipeline is CI-testable with no API key and no GPU."""

from __future__ import annotations

import json
from pathlib import Path

from egoannote.backends.fake import FakeBackend
from egoannote.parse import parse_caption_v3


def test_fake_backend_returns_valid_schema_by_default() -> None:
    backend = FakeBackend()
    resp = backend.caption([b"fake-jpeg-bytes"], "irrelevant prompt")
    assert resp.cost_usd == 0.0
    parsed = parse_caption_v3(resp.text)
    assert parsed["_schema_ok"] is True


def test_fake_backend_replays_fixture_in_order(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    responses = [
        {"actions": [{"start_frame": 0, "end_frame": 7, "task_step": "first"}]},
        {"actions": [{"start_frame": 0, "end_frame": 7, "task_step": "second"}]},
    ]
    fixture.write_text(
        "\n".join(json.dumps({"response_text": json.dumps(r)}) for r in responses) + "\n"
    )
    backend = FakeBackend(fixture_path=fixture)

    r1 = parse_caption_v3(backend.caption([], "p").text)
    r2 = parse_caption_v3(backend.caption([], "p").text)
    r3 = parse_caption_v3(backend.caption([], "p").text)  # cycles back to first

    assert r1["actions"][0]["task_step"] == "first"
    assert r2["actions"][0]["task_step"] == "second"
    assert r3["actions"][0]["task_step"] == "first"
