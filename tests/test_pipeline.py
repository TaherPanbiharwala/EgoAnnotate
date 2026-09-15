from __future__ import annotations

from pathlib import Path

import pytest
from egoannote import pipeline


def test_archive_refuses_redacted_deletion_with_a_stale_hf_receipt(tmp_path: Path) -> None:
    first = tmp_path / "first.blurred.mp4"
    second = tmp_path / "second.blurred.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    pipeline._atomic_json(
        tmp_path / "private" / "run_manifest.json",
        {
            "videos": {
                "first": {"redacted_video": str(first)},
                "second": {"redacted_video": str(second)},
            }
        },
    )
    pipeline._atomic_json(
        tmp_path / "private" / "hf_upload_receipt.json",
        {"video_ids": ["first"]},
    )
    with pytest.raises(RuntimeError, match=r"missing=\['second'\]"):
        pipeline.archive_run(
            tmp_path,
            drive_root="gdrive:dataset",
            delete_local_redacted=True,
        )
