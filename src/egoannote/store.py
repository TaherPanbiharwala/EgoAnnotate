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
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import HANDS_SCHEMA, HandFrame, utc_now_isoformat


class Store:
    """One (video_id, unit_idx, stage, model_id)-keyed table. `unit_idx` is
    whatever ordinal makes sense for that stage — window_idx for captions,
    a running segment ordinal for segments, etc."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
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
        self._conn.execute(
            """
            INSERT OR REPLACE INTO annotations
                (video_id, unit_idx, stage, model_id, payload, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, unit_idx, stage, model_id, json.dumps(payload), error, utc_now_isoformat()),
        )
        self._conn.commit()

    def read(
        self, *, video_id: str, unit_idx: int, stage: str, model_id: str = ""
    ) -> dict[str, Any] | None:
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
        ordered by unit_idx. Pass model_id=None to iterate across all models."""
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
        for unit_idx, mid, payload, error in cur:
            yield unit_idx, mid, json.loads(payload), error

    def done_set(self, *, video_id: str, stage: str, model_id: str = "") -> set[int]:
        """unit_idx values with no error — the resume/skip set. Matches v1's
        semantics (error rows retry) but without the duplicate-record bug."""
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


def write_hands_parquet(records: Sequence[HandFrame], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = hands_to_arrow_table(records)
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def read_hands_parquet(path: Path) -> pa.Table:
    return pq.read_table(path)
