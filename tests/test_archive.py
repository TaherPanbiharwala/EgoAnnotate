from __future__ import annotations

import hashlib
import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import ANY

import pytest

from egoannote import archive


def test_archive_verifies_before_optional_exact_file_delete(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original-private-video")
    md5 = hashlib.md5(source.read_bytes(), usedforsecurity=False).hexdigest()
    calls: list[list[str]] = []

    def fake_rclone(args: list[str]) -> CompletedProcess[str]:
        calls.append(args)
        if args[0] == "lsjson":
            return CompletedProcess(
                args, 0,
                stdout=json.dumps([{"Size": source.stat().st_size, "Hashes": {"MD5": md5}}]),
                stderr="",
            )
        return CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(archive, "_run_rclone", fake_rclone)
    receipt = tmp_path / "receipt.json"
    rows = archive.archive_files(
        [(source, "gdrive:private/original.mp4")],
        receipt,
        delete_local_after_verify=True,
    )
    assert calls[0][0] == "copyto"
    assert calls[1][0] == "lsjson"
    assert not source.exists()
    assert rows[0].local_deleted is True
    assert json.loads(receipt.read_text())["files"][0]["local_deleted"] is True


def test_archive_receipt_tracks_a_partial_local_deletion(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"same-size")
    second.write_bytes(b"same-size")
    md5 = hashlib.md5(b"same-size", usedforsecurity=False).hexdigest()

    def fake_rclone(args: list[str]) -> CompletedProcess[str]:
        if args[0] == "lsjson":
            return CompletedProcess(
                args, 0, stdout=json.dumps([{"Size": 9, "Hashes": {"MD5": md5}}]), stderr=""
            )
        return CompletedProcess(args, 0, stdout="", stderr="")

    original_unlink = Path.unlink

    def fail_second_unlink(path: Path, *args, **kwargs) -> None:
        if path == second:
            raise OSError("disk error")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(archive, "_run_rclone", fake_rclone)
    monkeypatch.setattr(Path, "unlink", fail_second_unlink)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(OSError, match="disk error"):
        archive.archive_files(
            [(first, "gdrive:first.mp4"), (second, "gdrive:second.mp4")],
            receipt,
            delete_local_after_verify=True,
        )
    assert json.loads(receipt.read_text())["files"] == [
        {
            "local_path": str(first.resolve()),
            "remote_path": "gdrive:first.mp4",
            "bytes": 9,
            "md5": md5,
            "verified_at": ANY,
            "local_deleted": True,
        },
        {
            "local_path": str(second.resolve()),
            "remote_path": "gdrive:second.mp4",
            "bytes": 9,
            "md5": md5,
            "verified_at": ANY,
            "local_deleted": False,
        },
    ]
