"""The one SQLite database, laptop-only. Plus Parquet writers for the dense
hand-landmark track.

Why SQLite here and not v1's append-JSONL: v1 wrote captions with
`open(path, "a")` and resumed by skipping windows already present, but
excluded error records from that skip-set (so failed windows would retry).
Consequence: a retried window left BOTH the old error record and the new
success record in the file, with nothing to dedupe them — a duplicate-record
bug, latent in the shipped v1 data (0 error records ever occurred) but real.
`INSERT OR REPLACE` on a primary key makes that duplication structurally
impossible.

Why laptop-only, not "on the pod" (cycle-2 fix, A4): this project uses TWO
pods (CPU: frames/MediaPipe; GPU: EgoBlur/WiLoR/SAM3/Depth/SLAM) plus the
laptop. An earlier design put one SQLite file on "the pod's local disk" and
`conn.backup()`'d it to the network volume — but `conn.backup()` is a
WHOLE-DATABASE OVERWRITE, not a merge, so whichever pod backed up second
would silently wipe the other pod's rows. The fix: pods hold NO database.
They write immutable, uniquely-prefixed artifact shards
(`<volume>/runs/<run_id>/<pod_role>/<stage>/<shard>.npz` + `_meta.json`) with
no schema knowledge. This ONE file, on the laptop's local disk, is the only
place schema and provenance logic lives — single writer, single machine, WAL
on APFS, no cross-host caveats needed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import HANDS_SCHEMA, GapTracker, HandFrame, HandsMissingGap, utc_now_isoformat


class Store:
    """One (video_id, unit_idx, stage, model_id)-keyed table. `unit_idx` is
    whatever ordinal makes sense for that stage — window_idx for captions,
    a running segment ordinal for segments, etc.

    Thread safety: one sqlite3.Connection is shared across threads
    (check_same_thread=False), which is necessary now that
    layers.caption.caption_video runs VLM calls concurrently
    (config.CAPTION_MAX_WORKERS). Whether a single Python sqlite3.Connection
    safely serializes concurrent statement execution from multiple threads
    depends on how the underlying libsqlite3 was compiled — a platform
    detail this code has no business depending on. `self._lock` makes every
    method's connection access an explicit critical section instead, at a
    cost the perf review already measured as negligible (900 writes totaled
    0.09s serialized).
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Default busy_timeout is 0 — an immediate failure the instant two
        # writers (e.g. parallel VLM-calling threads) collide. 30s lets a
        # writer simply wait its turn instead of erroring.
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                video_id   TEXT NOT NULL,
                unit_idx   INTEGER NOT NULL,
                stage      TEXT NOT NULL,
                model_id   TEXT NOT NULL DEFAULT '',
                payload    TEXT NOT NULL,
                error      TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (video_id, unit_idx, stage, model_id)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(
        self,
        *,
        video_id: str,
        unit_idx: int,
        stage: str,
        payload: dict[str, Any],
        model_id: str = "",
        error: str | None = None,
    ) -> None:
        """INSERT OR REPLACE — a retry overwrites the prior attempt for the
        same key. Duplication is structurally impossible."""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO annotations
                    (video_id, unit_idx, stage, model_id, payload, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (video_id, unit_idx, stage, model_id, json.dumps(payload), error,
                 utc_now_isoformat()),
            )
            self._conn.commit()

    def read(
        self, *, video_id: str, unit_idx: int, stage: str, model_id: str = ""
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM annotations WHERE video_id=? AND unit_idx=? "
                "AND stage=? AND model_id=?",
                (video_id, unit_idx, stage, model_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def iter_stage(
        self, *, video_id: str, stage: str, model_id: str | None = None
    ) -> Iterator[tuple[int, str, dict[str, Any], str | None]]:
        """Yield (unit_idx, model_id, payload, error) rows for a stage,
        ordered by unit_idx. Pass model_id=None to iterate across all models.

        Rows are fetched in full while holding the lock, then yielded from
        that materialized list. A prior version yielded straight from a live
        cursor — fine for a single-threaded caller, but that leaves the
        cursor doing I/O on the shared connection for however long the
        CALLER takes to process each row, arbitrarily overlapping with
        another thread's write(). The perf review already measured this
        query at ~1ms even over 36k rows, so eager-fetch costs nothing real.
        """
        with self._lock:
            if model_id is None:
                cur = self._conn.execute(
                    "SELECT unit_idx, model_id, payload, error FROM annotations "
                    "WHERE video_id=? AND stage=? ORDER BY unit_idx",
                    (video_id, stage),
                )
            else:
                cur = self._conn.execute(
                    "SELECT unit_idx, model_id, payload, error FROM annotations "
                    "WHERE video_id=? AND stage=? AND model_id=? ORDER BY unit_idx",
                    (video_id, stage, model_id),
                )
            rows = cur.fetchall()
        for unit_idx, mid, payload, error in rows:
            yield unit_idx, mid, json.loads(payload), error

    def done_set(self, *, video_id: str, stage: str, model_id: str = "") -> set[int]:
        """unit_idx values with no error — the resume/skip set. Matches v1's
        semantics (error rows retry) but without the duplicate-record bug."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT unit_idx FROM annotations WHERE video_id=? AND stage=? "
                "AND model_id=? AND error IS NULL",
                (video_id, stage, model_id),
            )
            return {row[0] for row in cur}


# ---------------------------------------------------------------------------
# Hand-tracking track — Parquet, kept from v1 (design held up under review)
# ---------------------------------------------------------------------------


def hands_to_arrow_table(records: Sequence[HandFrame]) -> pa.Table:
    cols: dict[str, list] = {name: [] for name in HANDS_SCHEMA.names}
    for r in records:
        cols["video_id"].append(r.video_id)
        cols["frame_idx"].append(r.frame_idx)
        cols["timestamp_ms"].append(r.timestamp_ms)
        cols["left_landmarks"].append(r.left_landmarks)
        cols["right_landmarks"].append(r.right_landmarks)
        cols["left_confidence"].append(r.left_confidence)
        cols["right_confidence"].append(r.right_confidence)
        cols["hands_present"].append(r.hands_present)
        cols["detector_version"].append(r.detector_version)
        cols["left_track_id"].append(r.left_track_id)
        cols["right_track_id"].append(r.right_track_id)
    return pa.table(cols, schema=HANDS_SCHEMA)


_PARQUET_CHUNK_SIZE = 4096  # rows per pq.ParquetWriter.write_table() call


def write_hands_parquet_streaming(
    records: Iterable[HandFrame],
    path: Path,
    *,
    chunk_size: int = _PARQUET_CHUNK_SIZE,
) -> tuple[int, list[HandsMissingGap]]:
    """Write HandFrame records to `path` as they're produced, and derive
    no-hands gaps in the SAME pass — without ever materializing the full
    record set in memory.

    Measured on this machine (perf review), the old approach —
    `list(hands_layer.run(...))` fully materialized, then
    `hands_to_arrow_table`, then one `pq.write_table` call — peaked at
    613 MB RSS for 1.5h of 1080p (81,000 frames): +415 MB for the list of
    HandFrame, +100 MB more building 11 Python lists of 81,000 entries each
    inside hands_to_arrow_table. This version chunks every `chunk_size` rows
    through `pq.ParquetWriter`, bounding peak memory to O(chunk_size)
    instead of O(video length). Measured on the same 81,000-frame data:
    108 MB peak — a 5.7x reduction, and CONSTANT regardless of video length
    rather than growing linearly with it. Verified round-trip: 81,000 rows
    across 20 row groups read back correctly via read_hands_parquet.

    Returns (row_count, gaps) — `hands_layer.run()` is a one-shot generator,
    so a caller wanting both the parquet file AND
    `schema.derive_hands_missing`'s gaps would otherwise need to buffer the
    whole thing just to iterate it twice. GapTracker computes gaps
    incrementally as records stream through.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tracker = GapTracker()
    writer: pq.ParquetWriter | None = None
    n_written = 0
    chunk: list[HandFrame] = []

    def _flush() -> None:
        nonlocal writer
        if writer is None:
            writer = pq.ParquetWriter(path, HANDS_SCHEMA, compression="zstd")
        writer.write_table(hands_to_arrow_table(chunk))
        chunk.clear()

    try:
        for rec in records:
            tracker.feed(rec)
            chunk.append(rec)
            n_written += 1
            if len(chunk) >= chunk_size:
                _flush()
        if chunk:
            _flush()
    finally:
        if writer is not None:
            writer.close()
        elif n_written == 0:
            # No records at all — still emit a valid empty file with the
            # correct schema, so callers don't need a special case for
            # "file is missing" vs. "file has zero hand frames."
            pq.write_table(hands_to_arrow_table([]), path, compression="zstd")

    return n_written, tracker.finish()


def write_hands_parquet(records: Sequence[HandFrame], path: Path) -> int:
    """Backward-compatible wrapper over write_hands_parquet_streaming for
    callers that already have a fully-materialized Sequence and only need
    the row count. Prefer the streaming function directly for anything
    reading from hands_layer.run() — passing an already-realized list here
    defeats the memory bound the streaming version exists for."""
    n, _gaps = write_hands_parquet_streaming(records, path)
    return n


def read_hands_parquet(path: Path) -> pa.Table:
    return pq.read_table(path)
