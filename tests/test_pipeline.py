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


def _write_reviewable_redaction(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    video = tmp_path / "GX010057.blurred.mp4"
    video.write_bytes(b"reviewed redacted video")
    manifest = tmp_path / "GX010057.manifest.json"
    source_sha256 = "a" * 64
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "GX010057",
                "status": "NEEDS_REVIEW",
                "source": {"sha256": source_sha256},
                "output": {"sha256": pipeline._sha256_file(video)},
            }
        ),
        encoding="utf-8",
    )
    hand = tmp_path / "private" / "GX010057.hand_suppression.json"
    hand.parent.mkdir()
    hand.write_text(
        json.dumps(
            {
                "review_type": "egoblur_hand_suppression",
                "input": {"clip_id": "GX010057", "source_sha256": source_sha256},
            }
        ),
        encoding="utf-8",
    )
    yunet = tmp_path / "private" / "GX010057.yunet.json"
    yunet.write_text(
        json.dumps(
            {
                "review_type": "post_redaction_yunet",
                "input": {
                    "clip_id": "GX010057",
                    "redacted_sha256": pipeline._sha256_file(video),
                    "egoblur_manifest_sha256": pipeline._sha256_file(manifest),
                },
            }
        ),
        encoding="utf-8",
    )
    return video, manifest, hand, yunet


def test_hash_bound_human_review_allows_annotation_from_needs_review_manifest(
    tmp_path: Path,
) -> None:
    video, manifest, hand, yunet = _write_reviewable_redaction(tmp_path)
    run_dir = tmp_path / "runs" / "gx010057"
    approval = pipeline.create_redaction_review(
        run_dir=run_dir,
        video_id="GX010057",
        redacted_video=video,
        blur_manifest=manifest,
        hand_suppression_report=hand,
        yunet_report=yunet,
        reviewer="Dataset owner",
    )

    item = VideoInput(
        redacted_video=video,
        blur_manifest=manifest,
        redaction_review=approval,
        video_id="GX010057",
    )
    video_id, blur, basis = _resolve_identity(item)
    row = pipeline._register_video(run_dir, item, video_id, blur, basis)

    assert approval == run_dir / "private" / "redaction_reviews" / "GX010057.json"
    assert row["blur_status"] == "NEEDS_REVIEW"
    assert row["redaction_review"]["status"] == "human_approved"
    assert row["redaction_review"]["reviewer"] == "Dataset owner"


def test_hash_bound_human_review_rejects_changed_private_evidence(tmp_path: Path) -> None:
    video, manifest, hand, yunet = _write_reviewable_redaction(tmp_path)
    run_dir = tmp_path / "runs" / "gx010057"
    approval = pipeline.create_redaction_review(
        run_dir=run_dir,
        video_id="GX010057",
        redacted_video=video,
        blur_manifest=manifest,
        hand_suppression_report=hand,
        yunet_report=yunet,
        reviewer="Dataset owner",
    )
    hand.write_text("changed", encoding="utf-8")
    item = VideoInput(
        redacted_video=video,
        blur_manifest=manifest,
        redaction_review=approval,
        video_id="GX010057",
    )
    video_id, blur, basis = _resolve_identity(item)

    with pytest.raises(ValueError, match="hand-suppression report no longer matches"):
        pipeline._register_video(run_dir, item, video_id, blur, basis)


def test_publish_revalidates_human_approval_before_bypassing_needs_review(tmp_path: Path) -> None:
    video, manifest, hand, yunet = _write_reviewable_redaction(tmp_path)
    run_dir = tmp_path / "runs" / "gx010057"
    approval = pipeline.create_redaction_review(
        run_dir=run_dir,
        video_id="GX010057",
        redacted_video=video,
        blur_manifest=manifest,
        hand_suppression_report=hand,
        yunet_report=yunet,
        reviewer="Dataset owner",
    )
    item = VideoInput(
        redacted_video=video,
        blur_manifest=manifest,
        redaction_review=approval,
        video_id="GX010057",
    )
    video_id, blur, basis = _resolve_identity(item)
    pipeline._register_video(run_dir, item, video_id, blur, basis)
    yunet.write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="human redaction review no longer validates"):
        pipeline.publish_run(run_dir, repo_id="owner/dataset", private=True)


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
