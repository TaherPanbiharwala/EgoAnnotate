from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egoannote import pipeline
from egoannote.pipeline import VideoInput, _load_batch, _resolve_identity
from egoannote.store import Store


def test_existing_truncated_hand_track_cannot_be_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "work" / "vid" / "hands.parquet"
    target.parent.mkdir(parents=True)
    pq.write_table(pa.table({"x": [1]}), target)
    monkeypatch.setattr(
        pipeline,
        "probe",
        lambda _video: type("Info", (), {"fps": 30.0, "n_frames": 10})(),
    )
    with pytest.raises(RuntimeError, match=r"1/10 expected"):
        pipeline._run_hands(VideoInput(tmp_path / "clip.mp4"), "vid", tmp_path)


def test_explicit_video_id_still_loads_and_validates_blur_manifest(tmp_path: Path) -> None:
    blur = tmp_path / "clip.manifest.json"
    blur.write_text(
        json.dumps(
            {
                "clip_id": "clip",
                "source": {"sha256": "a" * 64},
                "status": "PASS_AUTOMATED",
            }
        ),
        encoding="utf-8",
    )
    video_id, loaded, basis = _resolve_identity(
        VideoInput(tmp_path / "clip.blurred.mp4", blur_manifest=blur, video_id="stable-id")
    )
    assert video_id == "stable-id"
    assert loaded is not None and loaded["status"] == "PASS_AUTOMATED"
    assert basis == "explicit"


@pytest.mark.parametrize("video_id", ["../outside", "nested/id", "..", "a\\b"])
def test_video_id_cannot_escape_its_run_paths(tmp_path: Path, video_id: str) -> None:
    with pytest.raises(ValueError, match="video_id must contain"):
        _resolve_identity(VideoInput(tmp_path / "clip.blurred.mp4", video_id=video_id))


def test_batch_paths_are_relative_to_the_manifest_not_the_shell(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "batch"
    manifest_dir.mkdir()
    manifest = manifest_dir / "inputs.jsonl"
    manifest.write_text('{"redacted_video":"videos/clip.blurred.mp4"}\n', encoding="utf-8")
    loaded = _load_batch(manifest)
    assert loaded[0].redacted_video == manifest_dir / "videos" / "clip.blurred.mp4"


def _write_run_manifest(run_dir: Path, captions: dict) -> None:
    pipeline._atomic_json(
        run_dir / "private" / "run_manifest.json",
        {
            "videos": {
                "vid": {
                    "blur_status": "PASS_AUTOMATED",
                    "redacted_video": str(run_dir / "clip.blurred.mp4"),
                    "stages": {"captions": captions},
                }
            }
        },
    )


def test_publish_refuses_a_pilot_caption_run(tmp_path: Path) -> None:
    _write_run_manifest(
        tmp_path,
        {"status": "pilot_complete", "models": ["model-a"], "expected_windows": 1},
    )
    with pytest.raises(RuntimeError, match="pilot or incomplete"):
        pipeline.publish_run(tmp_path, repo_id="owner/dataset", private=True)


def test_public_publish_requires_owner_approved_license_terms(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="--license-file and --license-name"):
        pipeline.publish_run(tmp_path, repo_id="owner/dataset", private=False)


def test_publish_refuses_a_full_run_with_caption_errors(tmp_path: Path) -> None:
    _write_run_manifest(
        tmp_path,
        {"status": "complete", "models": ["model-a"], "expected_windows": 1},
    )
    with Store(tmp_path / "private" / "annotations.db") as store:
        store.write(
            video_id="vid",
            unit_idx=0,
            stage="caption",
            model_id="model-a",
            payload={},
            error="HTTP 500",
        )
    with pytest.raises(RuntimeError, match=r"errored windows=\[0\]"):
        pipeline.publish_run(tmp_path, repo_id="owner/dataset", private=True)


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
