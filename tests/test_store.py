"""Schema round-trip tests, plus the NEW test proving the v1 duplicate-record
bug is structurally impossible here (S9 fix)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from egoannote.schema import (
    HandFrame,
    WindowCaption,
    derive_hands_missing,
    hash_prompt,
    utc_now_isoformat,
)
from egoannote.store import (
    Store,
    read_hands_parquet,
    write_hands_parquet,
    write_hands_parquet_streaming,
)


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


def test_store_survives_concurrent_writers(tmp_path: Path) -> None:
    """Store now backs concurrent VLM calls (config.CAPTION_MAX_WORKERS).
    N threads x M writes each must all land — no lost writes, no exceptions
    from the shared sqlite3.Connection, regardless of what libsqlite3 the
    platform happens to ship."""
    store = Store(tmp_path / "run.db")
    n_threads, writes_per_thread = 8, 25
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(writes_per_thread):
                idx = thread_id * writes_per_thread + i
                rec = _window_caption(idx, "model-a")
                store.write(
                    video_id="v", unit_idx=idx, stage="caption",
                    model_id="model-a", payload=rec.to_payload(), error=None,
                )
        except Exception as e:  # captured for the assertion below, not swallowed
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent writes raised: {errors}"
    done = store.done_set(video_id="v", stage="caption", model_id="model-a")
    assert len(done) == n_threads * writes_per_thread, "every write must land, none lost"
    store.close()


def test_iter_stage_across_all_models(tmp_path: Path) -> None:
    """model_id=None reads across models — used by the two-model agreement
    metric. Not exercised by any other test until now."""
    store = Store(tmp_path / "run.db")
    for model_id in ("model-a", "model-b"):
        for idx in (1, 0):
            rec = _window_caption(idx, model_id)
            store.write(video_id="v", unit_idx=idx, stage="caption",
                        model_id=model_id, payload=rec.to_payload(), error=None)
    rows = list(store.iter_stage(video_id="v", stage="caption", model_id=None))
    assert len(rows) == 4
    assert {r[1] for r in rows} == {"model-a", "model-b"}
    store.close()


# --------------------------------------------------------------------------
# Streaming Parquet writer (perf review fix). The old approach —
# list(hands_layer.run(...)) fully materialized, then hands_to_arrow_table,
# then one pq.write_table call — peaked at 613 MB RSS for 1.5h of 1080p
# (81,000 frames), measured on this machine. write_hands_parquet_streaming
# chunks through pq.ParquetWriter and derives no-hands gaps in the SAME
# pass via schema.GapTracker, measured at 108 MB peak — constant regardless
# of video length.
# --------------------------------------------------------------------------


def test_streaming_write_matches_eager_write(tmp_path: Path) -> None:
    """Correctness parity: the new streaming path and the old
    materialize-then-write path must produce identical parquet output."""
    records = [
        _hand_frame(0, 1, left_landmarks=[0.1] * 63, left_confidence=0.92, left_track_id=0),
        _hand_frame(1, 0),
        _hand_frame(2, 2, right_landmarks=[0.2] * 63, right_confidence=0.8, right_track_id=1),
    ]
    eager_path = tmp_path / "eager.parquet"
    write_hands_parquet(records, eager_path)

    stream_path = tmp_path / "stream.parquet"
    n, gaps = write_hands_parquet_streaming(iter(records), stream_path)

    assert n == 3
    eager_table = read_hands_parquet(eager_path)
    stream_table = read_hands_parquet(stream_path)
    assert eager_table.to_pylist() == stream_table.to_pylist()
    assert len(gaps) == 1
    assert gaps[0].gap_start_frame == 1 and gaps[0].gap_end_frame == 1


def test_streaming_write_accepts_a_true_one_shot_generator(tmp_path: Path) -> None:
    """This is the actual call shape in production: hands_layer.run() is a
    generator, not a list. A Sequence-only implementation would reject it."""

    def gen():
        yield _hand_frame(0, 1)
        yield _hand_frame(1, 0)
        yield _hand_frame(2, 1)

    path = tmp_path / "gen.parquet"
    n, gaps = write_hands_parquet_streaming(gen(), path)
    assert n == 3
    assert read_hands_parquet(path).num_rows == 3


def test_streaming_write_chunk_boundary_does_not_split_a_gap(tmp_path: Path) -> None:
    """The gap-tracking state must survive across a pq.ParquetWriter flush.
    A gap spanning frames 3-6 with chunk_size=4 (a flush after frame 3, i.e.
    mid-gap) must still be reported as ONE gap, not torn into two."""
    records = [
        _hand_frame(0, 1), _hand_frame(1, 1), _hand_frame(2, 1),
        _hand_frame(3, 0), _hand_frame(4, 0), _hand_frame(5, 0), _hand_frame(6, 0),  # gap
        _hand_frame(7, 1), _hand_frame(8, 1),
    ]
    path = tmp_path / "boundary.parquet"
    n, gaps = write_hands_parquet_streaming(iter(records), path, chunk_size=4)

    assert n == 9
    assert read_hands_parquet(path).num_rows == 9
    assert len(gaps) == 1, "a gap spanning a chunk-flush boundary must not be split"
    assert gaps[0].gap_start_frame == 3
    assert gaps[0].gap_end_frame == 6
    assert gaps[0].n_frames == 4


def test_streaming_write_many_chunks_reads_back_correctly(tmp_path: Path) -> None:
    """Multiple ParquetWriter.write_table() calls must merge into one
    readable table — not one call per row group silently corrupting."""
    records = [_hand_frame(i, i % 2) for i in range(50)]
    path = tmp_path / "many_chunks.parquet"
    n, gaps = write_hands_parquet_streaming(iter(records), path, chunk_size=7)

    assert n == 50
    table = read_hands_parquet(path)
    assert table.num_rows == 50
    assert table.column("frame_idx").to_pylist() == list(range(50))
    # Every odd frame_idx (i % 2 == 1) is present=1; every even is a gap.
    # 25 singleton gaps: frames 0,2,4,...,48.
    assert len(gaps) == 25
    assert all(g.n_frames == 1 for g in gaps)


def test_streaming_write_empty_input_still_creates_a_valid_file(tmp_path: Path) -> None:
    """No records at all (e.g. probe-only run) must not leave a missing
    file — callers shouldn't need a special case for 'file absent' vs.
    'file has zero rows'."""
    path = tmp_path / "empty.parquet"
    n, gaps = write_hands_parquet_streaming(iter([]), path)
    assert n == 0
    assert gaps == []
    assert path.exists()
    table = read_hands_parquet(path)
    assert table.num_rows == 0
    assert set(table.column_names) == {
        "video_id", "frame_idx", "timestamp_ms", "left_landmarks", "right_landmarks",
        "left_confidence", "right_confidence", "hands_present", "detector_version",
        "left_track_id", "right_track_id",
    }


def test_derive_hands_missing_accepts_a_generator(tmp_path: Path) -> None:
    """Broadened from list[HandFrame] to Iterable[HandFrame] (GapTracker
    refactor) — must actually accept a one-shot generator, not just widen
    the type hint without changing the implementation."""

    def gen():
        yield _hand_frame(0, 1)
        yield _hand_frame(1, 0)
        yield _hand_frame(2, 1)

    gaps = derive_hands_missing(gen())
    assert len(gaps) == 1
    assert gaps[0].gap_start_frame == 1 and gaps[0].gap_end_frame == 1
