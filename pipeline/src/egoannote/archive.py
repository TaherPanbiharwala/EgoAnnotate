"""Verified Google Drive archiving through rclone.

The archive stage always copies first, compares the remote size and MD5, and
writes a receipt before local deletion is even considered. ``rclone move`` is
intentionally not used: a transfer interruption must never remove the only
copy of an original video needed later by WiLoR, SAM2, and DepthV3.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class ArchivedFile:
    local_path: str
    remote_path: str
    bytes: int
    md5: str
    verified_at: str
    local_deleted: bool = False


def _md5_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _run_rclone(args: list[str]) -> subprocess.CompletedProcess[str]:
    binary = shutil.which("rclone")
    if binary is None:
        raise RuntimeError("rclone is not installed or is not on PATH")
    return subprocess.run(
        [binary, *args], check=True, text=True, capture_output=True,
    )


def _remote_metadata(remote_path: str) -> tuple[int, str]:
    result = _run_rclone(["lsjson", "--hash", "--files-only", remote_path])
    rows = json.loads(result.stdout)
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one remote file at {remote_path!r}, found {len(rows)}"
        )
    row = rows[0]
    hashes = {str(k).lower(): str(v).lower() for k, v in row.get("Hashes", {}).items()}
    remote_md5 = hashes.get("md5")
    if not remote_md5:
        raise RuntimeError(
            f"remote {remote_path!r} did not report an MD5; refusing to claim verification"
        )
    return int(row["Size"]), remote_md5


def _write_receipt(receipt_path: Path, archived: list[ArchivedFile]) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    tmp.write_text(
        json.dumps({"schema_version": 1, "files": [asdict(x) for x in archived]}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(receipt_path)


def archive_file(local_path: Path, remote_path: str) -> ArchivedFile:
    """Copy one exact file and verify remote size plus MD5."""
    local_path = local_path.resolve()
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    if ":" not in remote_path.split("/", 1)[0]:
        raise ValueError(
            f"remote path must start with an rclone remote, e.g. gdrive:folder; got {remote_path!r}"
        )

    initial_stat = local_path.stat()
    local_size = initial_stat.st_size
    local_md5 = _md5_file(local_path)
    _run_rclone(["copyto", "--checksum", str(local_path), remote_path])
    remote_size, remote_md5 = _remote_metadata(remote_path)
    if remote_size != local_size or remote_md5 != local_md5:
        raise RuntimeError(
            "Drive verification failed for "
            f"{local_path}: local(size={local_size}, md5={local_md5}) != "
            f"remote(size={remote_size}, md5={remote_md5})"
        )
    final_stat = local_path.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != (
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
    ):
        raise RuntimeError(
            f"{local_path} changed while it was being archived; preserving the local file "
            "instead of writing a receipt or deleting it"
        )

    return ArchivedFile(
        local_path=str(local_path),
        remote_path=remote_path,
        bytes=local_size,
        md5=local_md5,
        verified_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def archive_files(
    transfers: list[tuple[Path, str]],
    receipt_path: Path,
    *,
    delete_local_after_verify: bool = False,
) -> list[ArchivedFile]:
    """Archive files, atomically write a receipt, then optionally unlink.

    Local deletion is exact-file-only and opt-in. If one transfer or check
    fails, no receipt is written and no local file is removed.
    """
    archived = [archive_file(local, remote) for local, remote in transfers]
    _write_receipt(receipt_path, archived)

    if delete_local_after_verify:
        for item in archived:
            Path(item.local_path).unlink()
            item.local_deleted = True
            # A process crash or later unlink error must not leave recovery
            # state claiming that an already-deleted local source still exists.
            _write_receipt(receipt_path, archived)
    return archived
