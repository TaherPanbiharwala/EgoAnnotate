"""Schema round-trip tests, plus the NEW test proving the v1 duplicate-record
bug is structurally impossible here (S9 fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from egoannote.schema import (
    HandFrame,
    WindowCaption,
    derive_hands_missing,
    hash_prompt,
    utc_now_isoformat,
)
from egoannote.store import Store, read_hands_parquet, write_hands_parquet


def _hand_frame(i: int, present: int, **overrides) -> HandFrame:
    base = dict(
        video_id="v",
        frame_idx=i,
        timestamp_ms=i * 66,
        left_landmarks=None,
        right_landmarks=None,
        left_confidence=None,
        right_confidence=None,
        hands_present=present,
        detector_version="test",
        left_track_id=None,
        right_track_id=None,
    )
    base.update(overrides)
    return HandFrame(**base)


def test_hand_frame_parquet_roundtrip(tmp_path: Path) -> None:
    records = [
        _hand_frame(0, 1, left_landmarks=[0.1] * 63, left_confidence=0.92, left_track_id=0),
        _hand_frame(1, 0),
    ]
    path = tmp_path / "hands.parquet"
    n = write_hands_parquet(records, path)
    assert n == 2

    table = read_hands_parquet(path)
    assert table.num_rows == 2
    assert table.column("hands_present").to_pylist() == [1, 0]
    assert table.column("left_landmarks").to_pylist()[0] == pytest.approx([0.1] * 63)
    assert table.column("left_landmarks").to_pylist()[1] is None
    assert table.column("left_track_id").to_pylist()[0] == 0


def test_derive_hands_missing_basic() -> None:
    records = [
        _hand_frame(0, 1), _hand_frame(1, 1),
        _hand_frame(2, 0), _hand_frame(3, 0), _hand_frame(4, 0),  # gap A
        _hand_frame(5, 1),
        _hand_frame(6, 0),                                        # gap B (singleton)
        _hand_frame(7, 2),
    ]
    gaps = derive_hands_missing(records)
    assert len(gaps) == 2
    assert gaps[0].gap_start_frame == 2 and gaps[0].gap_end_frame == 4
    assert gaps[0].n_frames == 3
    assert gaps[1].gap_start_frame == 6 and gaps[1].gap_end_frame == 6


def test_derive_hands_missing_gap_at_end() -> None:
    records = [_hand_frame(0, 1), _hand_frame(1, 0), _hand_frame(2, 0)]
    gaps = derive_hands_missing(records)
    assert len(gaps) == 1
    assert gaps[0].gap_end_frame == 2


def test_hash_prompt_is_deterministic() -> None:
    a = hash_prompt("hello world")
    b = hash_prompt("hello world")
    c = hash_prompt("hello worlds")
    assert a == b
    assert a != c
    assert a.startswith("sha256:")


def _window_caption(window_idx: int, model_id: str, **overrides) -> WindowCaption:
    base = dict(
        video_id="v",
        window_idx=window_idx,
        start_ts_ms=window_idx * 6000,
        end_ts_ms=(window_idx + 1) * 6000,
        frame_indices=list(range(window_idx * 8, window_idx * 8 + 8)),
        is_partial=False,
        model_id=model_id,
        prompt_version="v3",
        prompt_hash="sha256:test",
        run_id=utc_now_isoformat(),
        latency_ms=1000,
        actions=[{"task_step": "test_action"}],
        schema_ok=True,
    )
    base.update(overrides)
    return WindowCaption(**base)


def test_store_write_and_read_roundtrip(tmp_path: Path) -> None:
    store = Store(tmp_path / "run.db")
    rec = _window_caption(0, "model-a")
    store.write(
        video_id=rec.video_id, unit_idx=rec.window_idx, stage="caption",
        model_id=rec.model_id, payload=rec.to_payload(), error=rec.error,
    )
    back = store.read(video_id="v", unit_idx=0, stage="caption", model_id="model-a")
    assert back is not None
    assert back["actions"][0]["task_step"] == "test_action"
    store.close()


def test_store_two_models_coexist(tmp_path: Path) -> None:
    """The primary key includes model_id — two models' results for the
    SAME window must NOT collide, unlike v1 where merge/resume logic keyed
    on window_idx alone (Eng review S7/5.5 finding)."""
    store = Store(tmp_path / "run.db")
    for model_id in ("model-a", "model-b"):
        rec = _window_caption(0, model_id, actions=[{"task_step": f"from_{model_id}"}])
        store.write(
            video_id=rec.video_id, unit_idx=0, stage="caption",
            model_id=model_id, payload=rec.to_payload(), error=None,
        )
    a = store.read(video_id="v", unit_idx=0, stage="caption", model_id="model-a")
    b = store.read(video_id="v", unit_idx=0, stage="caption", model_id="model-b")
    assert a["actions"][0]["task_step"] == "from_model-a"
    assert b["actions"][0]["task_step"] == "from_model-b"
    store.close()


def test_store_retry_overwrites_not_duplicates(tmp_path: Path) -> None:
    """THE v1 bug this design fixes: v1 appended to JSONL and excluded error
    rows from the skip-set, so a retried window left BOTH the error record
    and the success record in the file — nothing deduped them. Here,
    INSERT OR REPLACE means a retry for the same key overwrites in place."""
    store = Store(tmp_path / "run.db")

    # First attempt: fails.
    failed = _window_caption(5, "model-a", actions=[], schema_ok=False, error="TimeoutError: boom")
    store.write(
        video_id="v", unit_idx=5, stage="caption", model_id="model-a",
        payload=failed.to_payload(), error=failed.error,
    )
    assert store.done_set(video_id="v", stage="caption", model_id="model-a") == set()

    # Retry: succeeds. Must OVERWRITE the failed row, not add a second one.
    ok = _window_caption(5, "model-a", actions=[{"task_step": "recovered"}], schema_ok=True)
    store.write(
        video_id="v", unit_idx=5, stage="caption", model_id="model-a",
        payload=ok.to_payload(), error=None,
    )

    rows = list(store.iter_stage(video_id="v", stage="caption", model_id="model-a"))
    assert len(rows) == 1, "retry must overwrite, not duplicate"
    unit_idx, mid, payload, error = rows[0]
    assert error is None
    assert payload["actions"][0]["task_step"] == "recovered"
    assert store.done_set(video_id="v", stage="caption", model_id="model-a") == {5}
    store.close()


def test_store_done_set_excludes_error_rows(tmp_path: Path) -> None:
    store = Store(tmp_path / "run.db")
    ok = _window_caption(0, "model-a")
    bad = _window_caption(1, "model-a", actions=[], schema_ok=False, error="boom")
    store.write(video_id="v", unit_idx=0, stage="caption", model_id="model-a",
                payload=ok.to_payload(), error=None)
    store.write(video_id="v", unit_idx=1, stage="caption", model_id="model-a",
                payload=bad.to_payload(), error=bad.error)
    assert store.done_set(video_id="v", stage="caption", model_id="model-a") == {0}
    store.close()
