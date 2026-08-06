"""Deterministic backend — no network call, no API key, $0. Makes the full
pipeline CI-testable and is what scripts/demo.py runs by default.

Two modes:
  - Fixture-replay: pass `fixture_path` to a JSONL file of REAL captured VLM
    responses (recorded once from a real API call — see examples/README.md
    for how to produce one). This is not a mock; it's the true output,
    replayed. Responses cycle if there are more windows than fixture rows.
  - Synthetic default: with no fixture, returns one plausible action
    spanning the whole window. Good enough for unit tests of the plumbing;
    NOT a substitute for testing the ordered-list prompt itself (that's
    Phase 0.5's job, against a real model).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .base import VLMResponse

_DEFAULT_RESPONSE = json.dumps(
    {
        "actions": [
            {
                "start_frame": 0,
                "end_frame": 7,
                "task_step": "example_action",
                "left_hand": {
                    "verb": "holding", "object": "carrot", "target": None,
                    "contact_type": "grip", "visible": True,
                },
                "right_hand": {
                    "verb": "chopping", "object": "knife", "target": "carrot",
                    "contact_type": "grip", "visible": True,
                },
                "tool_in_use": "knife",
                "coordination": "coordinated",
                "handover_event": False,
                "scene": {
                    "what": "The person chops a carrot on a cutting board.",
                    "how": "Power grip on the knife, palm-down stabilizing grip on the carrot.",
                    "why": "to prepare the carrot for cooking",
                    "location": "kitchen counter",
                },
            }
        ]
    }
)


class FakeBackend:
    name = "fake"

    def __init__(self, model_id: str = "fake", fixture_path: Path | None = None):
        self.model_id = model_id
        self._responses: list[str] = []
        if fixture_path is not None:
            if not fixture_path.exists():
                # Passing None means "I want synthetic". Passing a path that
                # doesn't resolve means "I made a mistake" — silently serving
                # synthetic captions there produces a run full of fake
                # "chops a carrot" data that looks like replayed real output.
                raise FileNotFoundError(
                    f"VLM fixture not found at {fixture_path}. Pass fixture_path=None "
                    f"to use the synthetic default, or see examples/README.md for how "
                    f"to record real responses."
                )
            with fixture_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        self._responses.append(row["response_text"])
        self._call_count = 0
        # caption() may now run from multiple threads at once (see
        # config.CAPTION_MAX_WORKERS) — without this, concurrent calls race
        # on the increment and can duplicate or skip a fixture index.
        self._lock = threading.Lock()

    def caption(self, frames_jpeg: list[bytes], prompt: str) -> VLMResponse:
        with self._lock:
            if self._responses:
                text = self._responses[self._call_count % len(self._responses)]
            else:
                text = _DEFAULT_RESPONSE
            self._call_count += 1
        return VLMResponse(
            text=text,
            latency_ms=1,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            provider="fake",
            model_version="fake-v1",
        )
