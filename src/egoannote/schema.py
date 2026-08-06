"""Record shapes for every stage. Dataclasses define the shape; store.py
persists them (SQLite for the annotations table, Parquet for dense
per-frame hand landmarks — kept from v1, which got this split right)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

# ---------------------------------------------------------------------------
# Hand-tracking track (Parquet — dense, ~15 fps, columnar wins). Kept as-is
# from v1; the design held up under review.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HandFrame:
    """One MediaPipe HandLandmarker result for one frame."""

    video_id: str
    frame_idx: int
    timestamp_ms: int
    left_landmarks: list[float] | None  # 63 floats (21 x,y,z) or None
    right_landmarks: list[float] | None
    left_confidence: float | None
    right_confidence: float | None
    hands_present: int  # 0 / 1 / 2
    detector_version: str
    # S6/S7: assignment confidence + a stable track id per hand, so a swap
    # is visible downstream rather than silently corrupting velocity.
    left_track_id: int | None = None
    right_track_id: int | None = None


# Variable-length list (not fixed_size_list) because pyarrow's parquet reader
# converts NULL outer values into empty lists, which then fail a fixed
# size=63 invariant. Kept from v1 — this comment is load-bearing, don't "fix"
# it back to fixed_size_list.
HANDS_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("frame_idx", pa.int64()),
        ("timestamp_ms", pa.int64()),
        ("left_landmarks", pa.list_(pa.float32())),
        ("right_landmarks", pa.list_(pa.float32())),
        ("left_confidence", pa.float32()),
        ("right_confidence", pa.float32()),
        ("hands_present", pa.int8()),
        ("detector_version", pa.string()),
        ("left_track_id", pa.int32()),
        ("right_track_id", pa.int32()),
    ]
)


@dataclass(slots=True)
class HandsMissingGap:
    """A contiguous run of frames in which no hand was detected."""

    video_id: str
    gap_start_frame: int
    gap_end_frame: int
    gap_start_ts_ms: int
    gap_end_ts_ms: int
    duration_ms: int
    n_frames: int


class GapTracker:
    """Incremental run-length encoder for hands_present==0 stretches.

    Extracted from derive_hands_missing (which used to index `records[i-1]`
    into a fully-materialized list) so the SAME gap logic can fold over a
    stream one record at a time — store.write_hands_parquet_streaming needs
    that to derive gaps without ever holding the whole hand track in memory.
    One implementation, two call sites, instead of two copies that could
    silently drift out of agreement.
    """

    def __init__(self) -> None:
        self._video_id: str | None = None
        self._in_gap = False
        self._gap_start_idx = 0
        self._gap_start_ts = 0
        self._prev: HandFrame | None = None
        self.gaps: list[HandsMissingGap] = []

    def feed(self, rec: HandFrame) -> None:
        if self._video_id is None:
            self._video_id = rec.video_id
        if rec.hands_present == 0 and not self._in_gap:
            self._in_gap = True
            self._gap_start_idx = rec.frame_idx
            self._gap_start_ts = rec.timestamp_ms
        elif rec.hands_present != 0 and self._in_gap:
            assert self._prev is not None
            self._close_gap(self._prev)
        self._prev = rec

    def finish(self) -> list[HandsMissingGap]:
        """Call once after the last feed(). Flushes a gap still open at the
        final record (mirrors the old function's tail-gap handling)."""
        if self._in_gap and self._prev is not None:
            self._close_gap(self._prev)
        return self.gaps

    def _close_gap(self, prev: HandFrame) -> None:
        assert self._video_id is not None
        self.gaps.append(
            HandsMissingGap(
                video_id=self._video_id,
                gap_start_frame=self._gap_start_idx,
                gap_end_frame=prev.frame_idx,
                gap_start_ts_ms=self._gap_start_ts,
                gap_end_ts_ms=prev.timestamp_ms,
                duration_ms=prev.timestamp_ms - self._gap_start_ts,
                n_frames=prev.frame_idx - self._gap_start_idx + 1,
            )
        )
        self._in_gap = False


def derive_hands_missing(records: Iterable[HandFrame]) -> list[HandsMissingGap]:
    """Run-length-encode the contiguous frame ranges where hands_present == 0.

    Broadened from `list[HandFrame]` to `Iterable[HandFrame]`: the original
    indexed `records[i - 1]`, which only a Sequence supports. GapTracker
    folds over one record at a time instead, so this now also accepts a
    generator (e.g. hands_layer.run()'s output) directly.
    """
    tracker = GapTracker()
    for rec in records:
        tracker.feed(rec)
    return tracker.finish()


# ---------------------------------------------------------------------------
# Caption track (SQLite `annotations` table, stage="caption")
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WindowCaption:
    """One VLM call's result for one window — may contain MULTIPLE actions
    (plan S1: the ordered-action-list schema). This is the unit stored per
    (video_id, window_idx, "caption", model_id) in the SQLite store."""

    video_id: str
    window_idx: int
    start_ts_ms: int
    end_ts_ms: int
    frame_indices: list[int]
    is_partial: bool
    model_id: str
    prompt_version: str
    prompt_hash: str
    run_id: str
    latency_ms: int
    actions: list[dict[str, Any]]  # see parse.ActionItem for shape
    raw_json: str | None = None
    schema_ok: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    provider: str | None = None  # OpenRouter's realized upstream provider —
    # pin this in models.toml (allow_fallbacks=false) and persist it here, or
    # the agreement metric measures provider variance, not model disagreement
    # (Eng review 2.6).

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WindowCaption:
        return cls(**payload)


# ---------------------------------------------------------------------------
# Segment track (Phase 6 output — SQLite, stage="segment")
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Segment:
    """One proposed/confirmed subtask segment.

    (docs/SCHEMA.md is not written yet — layers/segment.py's module docstring
    is currently the authoritative description of these fields.)
    """

    segment_id: str
    video_id: str
    start_ts_ms: int
    end_ts_ms: int
    boundary_source: str  # "intra_window" | "window_seam"
    boundary_method: str  # "label+motion+visual" | "label+visual" | "label+visual_unsnapped"
    task_step: str | None = None
    left_hand: dict[str, Any] = field(default_factory=dict)
    right_hand: dict[str, Any] = field(default_factory=dict)
    tool_in_use: str | None = None
    coordination: str | None = None
    handover_event: bool | None = None
    scene_what: str | None = None
    scene_how: str | None = None
    scene_why: str | None = None
    scene_location: str | None = None
    model_id: str | None = None
    run_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Segment:
        return cls(**payload)


# ---------------------------------------------------------------------------
# Misc helpers (kept from v1)
# ---------------------------------------------------------------------------


def hash_prompt(prompt_text: str) -> str:
    return "sha256:" + hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def utc_now_isoformat() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
