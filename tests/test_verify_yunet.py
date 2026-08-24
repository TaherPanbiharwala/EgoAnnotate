import hashlib
import json
from pathlib import Path

import pytest

from egoannote import verify_yunet


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_job(path: Path) -> None:
    path.write_text(
        """
from dataclasses import dataclass

TRACK_IOU_DEFAULT = 0.2

@dataclass
class Detection:
    frame_idx: int
    cls: str
    box: tuple
    score: float

def build_tracks(detections, min_box_px, **kwargs):
    assert min_box_px == 8
    assert kwargs["min_confident_hits"] == 2
    return detections

def tracks_to_fill_map(tracks, width, height, dilate_scale, margin_px):
    assert (width, height, dilate_scale, margin_px) == (8, 8, 1.3, 8)
    return {d.frame_idx: [d.box] for d in tracks}

def check_yunet(cfg, ffmpeg, video, width, height, fill_map, detect_hz, fps, n_frames, *, full_range, record_detections):
    assert cfg.yunet_model.is_file()
    assert (ffmpeg, width, height, detect_hz, fps, n_frames, full_range) == (
        "fake-ffmpeg", 8, 8, 10.0, 20.0, 4, False
    )
    assert fill_map[0] == [(1.0, 1.0, 3.0, 3.0)]
    result = {"n_yunet_uncovered": 0, "yunet_uncovered": [], "yunet_frames": 4}
    if record_detections:
        result["yunet_detections"] = [{"frame_idx": 0, "box": (1, 1, 3, 3), "score": 0.9, "covered": True}]
    return result
""",
        encoding="utf-8",
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    video = tmp_path / "clip.blurred.mp4"
    video.write_bytes(b"redacted video")
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"model")
    job = tmp_path / "job.py"
    _write_job(job)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_dir.joinpath("clip.detections.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_fingerprint": 1,
                        "gen": "2",
                        "lp_present": False,
                        "sweep_threshold": 0.1,
                        "nms_iou": 0.3,
                        "detect_hz": 10.0,
                        "gen2_resize_px": None,
                    }
                ),
                json.dumps(
                    {
                        "frame_idx": 0,
                        "detections": [
                            {"cls": "face", "box": [1, 1, 3, 3], "score": 0.9}
                        ],
                    }
                ),
                json.dumps({"frame_idx": 2, "detections": []}),
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "clip.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip",
                "source": {"width": 8, "height": 8, "fps": 20.0, "n_frames": 4},
                "output": {"sha256": _sha256(video)},
                "egoblur": {
                    "gen": "2",
                    "lp_checked": False,
                    "sweep_threshold": 0.1,
                    "nms_iou": 0.3,
                    "detect_hz": 10.0,
                    "gen2_resize_px": None,
                    "face_threshold": 0.3,
                    "lp_threshold": 0.4,
                    "min_box_px": 8,
                    "dilate_scale": 1.3,
                    "motion_margin_px": 8,
                    "hold_frames": 45,
                    "back_hold_frames": 45,
                    "continue_threshold": 0.0,
                    "min_track_confirmations": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return video, manifest, checkpoint_dir, model, job


def test_verify_yunet_rebuilds_the_matching_fill_map_and_writes_private_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, manifest, checkpoint_dir, model, job = _write_inputs(tmp_path)
    monkeypatch.setattr(verify_yunet, "_ffprobe_color_range", lambda *_args: "tv")
    report = tmp_path / "private" / "review.json"

    result = verify_yunet.verify_yunet(
        redacted_video=video,
        blur_manifest=manifest,
        checkpoint_dir=checkpoint_dir,
        yunet_model=model,
        job_script=job,
        ffmpeg="fake-ffmpeg",
        report=report,
    )

    assert result["review_status"] == "PASS_NO_UNCOVERED_YUNET"
    assert result["reconstruction"]["checkpoint_detections"] == 1
    assert json.loads(report.read_text(encoding="utf-8"))["approval_scope"].startswith("YuNet-only")


def test_verify_yunet_rejects_a_video_from_a_different_egoblur_run(tmp_path: Path) -> None:
    video, manifest, checkpoint_dir, model, job = _write_inputs(tmp_path)
    video.write_bytes(b"wrong redacted video")

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        verify_yunet.verify_yunet(
            redacted_video=video,
            blur_manifest=manifest,
            checkpoint_dir=checkpoint_dir,
            yunet_model=model,
            job_script=job,
            ffmpeg="fake-ffmpeg",
            report=tmp_path / "private" / "review.json",
        )


def test_verify_yunet_rejects_an_incomplete_checkpoint(tmp_path: Path) -> None:
    video, manifest, checkpoint_dir, model, job = _write_inputs(tmp_path)
    checkpoint = checkpoint_dir / "clip.detections.jsonl"
    checkpoint.write_text("\n".join(checkpoint.read_text(encoding="utf-8").splitlines()[:-1]), encoding="utf-8")
    loaded = json.loads(manifest.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        verify_yunet._checkpoint_detections(checkpoint_dir, loaded, verify_yunet._load_job_module(job))


def test_verify_yunet_preview_must_be_private_and_receives_all_detections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, manifest, checkpoint_dir, model, job = _write_inputs(tmp_path)
    monkeypatch.setattr(verify_yunet, "_ffprobe_color_range", lambda *_args: "tv")
    captured = {}
    monkeypatch.setattr(
        verify_yunet,
        "_write_preview_video",
        lambda **kwargs: captured.update(kwargs) or 2,
    )

    with pytest.raises(ValueError, match="private or DO-NOT-SHIP"):
        verify_yunet.verify_yunet(
            redacted_video=video,
            blur_manifest=manifest,
            checkpoint_dir=checkpoint_dir,
            yunet_model=model,
            job_script=job,
            ffmpeg="fake-ffmpeg",
            report=tmp_path / "private" / "review.json",
            preview_video=tmp_path / "preview.mp4",
        )

    private_preview = tmp_path / "private" / "preview.mp4"
    result = verify_yunet.verify_yunet(
        redacted_video=video,
        blur_manifest=manifest,
        checkpoint_dir=checkpoint_dir,
        yunet_model=model,
        job_script=job,
        ffmpeg="fake-ffmpeg",
        report=tmp_path / "private" / "review.json",
        preview_video=private_preview,
    )
    assert captured["detections"][0]["covered"] is True
    assert result["preview"]["frames"] == 2


def test_yunet_candidates_are_temporal_and_only_wearer_overlap_is_lower_priority() -> None:
    detections = [
        {"frame_idx": 0, "box": [0, 0, 10, 10], "score": 0.8, "covered": False},
        {"frame_idx": 2, "box": [1, 0, 11, 10], "score": 0.7, "covered": False},
        {"frame_idx": 2, "box": [40, 40, 50, 50], "score": 0.9, "covered": False},
    ]
    all_limb_only = {"shape": "circle", "kind": "hand", "center": [45, 45], "radius": 20}
    wearer_limb = {"shape": "circle", "kind": "hand", "center": [5, 5], "radius": 20}
    frames = {
        0: {"all": [wearer_limb], "wearer": [wearer_limb]},
        2: {"all": [all_limb_only, wearer_limb], "wearer": [wearer_limb]},
    }

    candidates = verify_yunet.build_candidate_tracks(
        detections, pose_frames=frames, fps=20.0, detect_hz=10.0
    )

    assert len(candidates) == 2
    by_start = {candidate["first_frame_idx"]: candidate for candidate in candidates}
    wearer_candidate = by_start[0]
    assert wearer_candidate["n_observations"] == 2
    assert wearer_candidate["priority"] == "lower_wearer_limb_overlap"
    # The other person's hand is retained in all_limb overlap for diagnosis,
    # but cannot down-rank a candidate under the wearer-only policy.
    other_candidate = next(candidate for candidate in candidates if candidate is not wearer_candidate)
    assert other_candidate["all_limb_overlap_max"] == pytest.approx(1.0)
    assert other_candidate["wearer_limb_overlap_max"] == 0.0
    assert other_candidate["priority"] == "normal"


def test_confirmed_candidate_decisions_become_forced_boxes_only(tmp_path: Path) -> None:
    report = tmp_path / "private" / "review.json"
    report.parent.mkdir()
    report.write_text(
        json.dumps(
            {
                "input": {"clip_id": "clip"},
                "yunet": {
                    "candidates": [
                        {
                            "candidate_id": "yunet-00001",
                            "observations": [{"frame_idx": 3, "box": [1, 2, 3, 4]}],
                        },
                        {
                            "candidate_id": "yunet-00002",
                            "observations": [{"frame_idx": 6, "box": [5, 6, 7, 8]}],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "private" / "decisions.json"
    template = verify_yunet.create_decision_template(report, decisions)
    template["decisions"]["yunet-00001"] = "confirmed_face"
    template["decisions"]["yunet-00002"] = "false_positive"
    decisions.write_text(json.dumps(template), encoding="utf-8")
    forced = tmp_path / "private" / "forced_boxes.json"

    result = verify_yunet.decisions_to_forced_boxes(report, decisions, forced)

    assert result["n_forced_boxes"] == 1
    assert json.loads(forced.read_text(encoding="utf-8")) == {
        "clip": [{"frame_idx": 3, "box": [1, 2, 3, 4]}]
    }
