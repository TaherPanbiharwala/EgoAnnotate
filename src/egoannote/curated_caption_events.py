"""Private event timelines derived from dense curated-original caption windows.

This is deliberately *not* ``layers.segment``.  It never proposes motion
boundaries, snaps timestamps, or crosses a manual cut.  It only removes an
artificial six-second seam when two adjacent, frame-grounded VLM actions say
the same thing with the same structured hand state.  Raw window captions
remain stored in SQLite and are always the source of truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import config

_EVENT_SCHEMA_VERSION = 1
_MAX_ADJACENT_GAP_MS = round(1000 / float(config.VLM_FPS))


def _require_private(path: Path, label: str) -> None:
    if not any(part.lower() in {"private", "do-not-ship"} for part in path.parts):
        raise ValueError(f"{label} must be under private or DO-NOT-SHIP")


def event_timeline_path(*, root: Path, video_id: str, model_id: str, segment_id: int) -> Path:
    """Canonical private location for one independently retained segment."""
    path = root / "events" / video_id / model_id / f"segment_{segment_id:04d}.json"
    _require_private(path, "caption event timeline")
    return path


def _source_hash(captions: list[dict[str, Any]]) -> str:
    encoded = json.dumps(captions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"caption {label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"caption {label} must be an integer") from exc


def _hand_signature(action: dict[str, Any], side: str) -> tuple[Any, ...]:
    hand = action.get(f"{side}_hand")
    if not isinstance(hand, dict):
        hand = {}
    # Captions are intentionally excluded: wording can vary at a seam while
    # the visible hand state remains identical.  The structured relation is
    # what makes a merge explainable and conservative.
    return tuple(hand.get(key) for key in ("visible", "verb", "object", "target", "contact_type"))


def _merge_signature(event: dict[str, Any]) -> tuple[Any, ...] | None:
    task_step = event.get("task_step")
    if not isinstance(task_step, str) or not task_step:
        return None
    if not event.get("source_schema_ok"):
        return None
    return (
        task_step,
        _hand_signature(event, "left"),
        _hand_signature(event, "right"),
        event.get("tool_in_use"),
        event.get("coordination"),
        event.get("handover_event"),
    )


def _event_from_action(caption: dict[str, Any], action: dict[str, Any], action_idx: int) -> dict[str, Any]:
    window_start = _as_int(caption.get("start_ts_ms"), "start_ts_ms")
    window_end = _as_int(caption.get("end_ts_ms"), "end_ts_ms")
    start_frame = _as_int(action.get("start_frame"), "action.start_frame")
    end_frame = _as_int(action.get("end_frame"), "action.end_frame")
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("caption action has an invalid frame range")
    sample_ms = 1000 / float(config.VLM_FPS)
    start_ts_ms = max(window_start, min(window_end, window_start + round(start_frame * sample_ms)))
    end_ts_ms = max(start_ts_ms, min(window_end, window_start + round((end_frame + 1) * sample_ms)))
    if end_ts_ms <= start_ts_ms:
        raise ValueError("caption action has an empty timestamp range")
    return {
        "start_ts_ms": start_ts_ms,
        "end_ts_ms": end_ts_ms,
        "task_step": action.get("task_step"),
        "action_caption": action.get("action_caption"),
        "left_hand": action.get("left_hand") or {},
        "right_hand": action.get("right_hand") or {},
        "tool_in_use": action.get("tool_in_use"),
        "coordination": action.get("coordination"),
        "handover_event": action.get("handover_event"),
        "scene": action.get("scene") or {},
        "source_schema_ok": bool(caption.get("schema_ok")),
        "source_window_indices": [_as_int(caption.get("window_idx"), "window_idx")],
        "source_actions": [
            {"window_idx": _as_int(caption.get("window_idx"), "window_idx"), "action_idx": action_idx}
        ],
    }


def _merge_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (item["start_ts_ms"], item["end_ts_ms"])):
        if not output:
            output.append(event)
            continue
        previous = output[-1]
        can_merge = (
            _merge_signature(previous) is not None
            and _merge_signature(previous) == _merge_signature(event)
            and event["start_ts_ms"] <= previous["end_ts_ms"] + _MAX_ADJACENT_GAP_MS
        )
        if not can_merge:
            output.append(event)
            continue
        previous["end_ts_ms"] = max(previous["end_ts_ms"], event["end_ts_ms"])
        previous["source_window_indices"] = sorted(
            set(previous["source_window_indices"]) | set(event["source_window_indices"])
        )
        previous["source_actions"].extend(event["source_actions"])
        # The first action wording is preserved rather than synthesizing a
        # new caption.  That keeps every displayed sentence traceable to a
        # raw VLM observation.
    for index, event in enumerate(output):
        event["event_idx"] = index
    return output


def compile_event_timeline(
    *,
    video_id: str,
    segment_id: int,
    model_id: str,
    captions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Compile one never-cross-cut event timeline from complete raw captions."""
    if segment_id < 0:
        raise ValueError("segment_id must be non-negative")
    if not captions:
        raise ValueError("cannot compile an event timeline without captions")
    if not isinstance(summary.get("summary"), str) or not isinstance(summary.get("steps"), list):
        raise ValueError("caption event timeline requires a validated segment summary")
    source_windows = sorted(captions, key=lambda item: _as_int(item.get("window_idx"), "window_idx"))
    prompt_versions = {str(caption.get("prompt_version")) for caption in source_windows}
    if len(prompt_versions) != 1:
        raise ValueError("caption event timeline refuses mixed prompt versions")
    events: list[dict[str, Any]] = []
    activity_observations: list[dict[str, Any]] = []
    for caption in source_windows:
        actions = caption.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("caption event timeline requires non-empty actions in every window")
        activity = caption.get("activity") or {}
        activity_observations.append(
            {
                "window_idx": _as_int(caption.get("window_idx"), "window_idx"),
                "start_ts_ms": _as_int(caption.get("start_ts_ms"), "start_ts_ms"),
                "end_ts_ms": _as_int(caption.get("end_ts_ms"), "end_ts_ms"),
                "caption": activity.get("caption") if isinstance(activity, dict) else None,
            }
        )
        for action_idx, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError("caption action must be an object")
            events.append(_event_from_action(caption, action, action_idx))
    return {
        "schema_version": _EVENT_SCHEMA_VERSION,
        "artifact_type": "private_curated_caption_event_timeline",
        "privacy": "private_do_not_ship_or_upload",
        "video_id": video_id,
        "segment_id": segment_id,
        "model_id": model_id,
        "source_caption_sha256": _source_hash(source_windows),
        "source_caption_prompt_versions": sorted(prompt_versions),
        "summary": {
            "text": summary["summary"],
            "steps": summary["steps"],
            "source_caption_sha256": summary.get("source_caption_sha256"),
        },
        "activity_observations": activity_observations,
        "events": _merge_events(events),
        "merge_policy": {
            "name": "exact_structured_state_across_adjacent_windows",
            "max_gap_ms": _MAX_ADJACENT_GAP_MS,
            "never_crosses_manual_cuts": True,
            "does_not_move_or_snap_boundaries": True,
        },
    }


def write_event_timeline(path: Path, timeline: dict[str, Any]) -> None:
    """Atomically replace this derived private sidecar from its raw sources."""
    path = path.expanduser().resolve()
    _require_private(path, "caption event timeline")
    if timeline.get("privacy") != "private_do_not_ship_or_upload":
        raise ValueError("caption event timeline has an invalid privacy label")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    partial.replace(path)


def load_event_timeline(path: Path, *, video_id: str, segment_id: int, model_id: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    _require_private(path, "caption event timeline")
    if not path.is_file():
        raise FileNotFoundError(f"caption event timeline not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("schema_version") != _EVENT_SCHEMA_VERSION
        or data.get("artifact_type") != "private_curated_caption_event_timeline"
        or data.get("privacy") != "private_do_not_ship_or_upload"
        or data.get("video_id") != video_id
        or data.get("segment_id") != segment_id
        or data.get("model_id") != model_id
    ):
        raise ValueError("caption event timeline belongs to another video, segment, or model")
    if not isinstance(data.get("events"), list) or not isinstance(data.get("summary"), dict):
        raise ValueError("caption event timeline is incomplete")
    return data
