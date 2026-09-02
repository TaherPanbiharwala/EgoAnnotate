from __future__ import annotations

from pathlib import Path

import pytest

from egoannote import curated_caption_events as events


def _action(*, object_name: str = "box", task_step: str = "fold_box_flap") -> dict:
    return {
        "start_frame": 0,
        "end_frame": 7,
        "task_step": task_step,
        "action_caption": "The camera wearer folds a box flap.",
        "left_hand": {
            "caption": "The left hand holds the box.",
            "visible": True,
            "verb": "holding",
            "object": object_name,
            "target": None,
            "contact_type": "grip",
        },
        "right_hand": {
            "caption": "The right hand folds the flap.",
            "visible": True,
            "verb": "folding",
            "object": "box flap",
            "target": "box",
            "contact_type": "push",
        },
        "tool_in_use": "none",
        "coordination": "supporting",
        "handover_event": False,
        "scene": {"what": "The wearer folds a box flap."},
    }


def _caption(window_idx: int, action: dict, *, prompt_version: str = "v5") -> dict:
    return {
        "window_idx": window_idx,
        "start_ts_ms": window_idx * 6000,
        "end_ts_ms": (window_idx + 1) * 6000,
        "prompt_version": prompt_version,
        "schema_ok": True,
        "activity": {"caption": "The camera wearer assembles a box."},
        "actions": [action],
    }


def _summary() -> dict:
    return {
        "summary": "The camera wearer assembles a box by folding its flaps.",
        "steps": ["Fold the box flaps."],
        "source_caption_sha256": "sha256:summary-source",
    }


def test_event_timeline_merges_only_exact_structured_state_at_window_seam() -> None:
    timeline = events.compile_event_timeline(
        video_id="GX010057",
        segment_id=2,
        model_id="qwen3.8-flash",
        captions=[_caption(0, _action()), _caption(1, _action())],
        summary=_summary(),
    )

    assert len(timeline["events"]) == 1
    event = timeline["events"][0]
    assert (event["start_ts_ms"], event["end_ts_ms"]) == (0, 12_000)
    assert event["source_window_indices"] == [0, 1]
    assert event["source_actions"] == [
        {"window_idx": 0, "action_idx": 0},
        {"window_idx": 1, "action_idx": 0},
    ]
    assert timeline["merge_policy"]["never_crosses_manual_cuts"] is True


def test_event_timeline_does_not_merge_when_a_hand_object_changes() -> None:
    timeline = events.compile_event_timeline(
        video_id="GX010057",
        segment_id=2,
        model_id="qwen3.8-flash",
        captions=[_caption(0, _action(object_name="box")), _caption(1, _action(object_name="tape"))],
        summary=_summary(),
    )

    assert len(timeline["events"]) == 2
    assert [event["start_ts_ms"] for event in timeline["events"]] == [0, 6000]


def test_event_timeline_rejects_mixed_prompt_versions() -> None:
    with pytest.raises(ValueError, match="mixed prompt versions"):
        events.compile_event_timeline(
            video_id="GX010057",
            segment_id=2,
            model_id="qwen3.8-flash",
            captions=[_caption(0, _action(), prompt_version="v4"), _caption(1, _action())],
            summary=_summary(),
        )


def test_event_timeline_sidecar_requires_a_private_path() -> None:
    with pytest.raises(ValueError, match="must be under private"):
        events.write_event_timeline(
            Path("caption-events.json"),
            {
                "privacy": "private_do_not_ship_or_upload",
            },
        )
