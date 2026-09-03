from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from egoannote import original_curated, public_release
from egoannote.media.probe import VideoInfo


def _video_info(frames: int) -> VideoInfo:
    return VideoInfo(
        duration_sec=frames / 30,
        width=64,
        height=48,
        fps=30.0,
        n_frames=frames,
        is_vfr=False,
    )


def _write_face_free_timeline(children: Path, video_id: str, frames: int) -> Path:
    child_dir = children / video_id
    child_dir.mkdir(parents=True)
    source = children.parent / "source.mp4"
    source.write_bytes(b"source")
    video = child_dir / f"{video_id}.normalized.mp4"
    video.write_bytes(b"clean-video")
    timeline = child_dir / f"{video_id}.timeline.json"
    timeline.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_type": "private_face_free_normalization",
                "privacy": "private_do_not_ship_or_upload",
                "video_id": video_id,
                "audio": "removed",
                "face_review": {"decision": "manually_reviewed_face_free", "reviewer": "Owner"},
                "normalization": {"target_fps": "30000/1001"},
                "source_video": original_curated._binding(source),
                "output_video": original_curated._binding(video),
                "source": {"fps": 120.0, "n_frames": frames * 4},
                "output": {"fps": 30.0, "n_frames": frames},
                "cut_ranges": [],
                "segments": [
                    {
                        "segment_id": 0,
                        "source_start_frame": 0,
                        "source_end_frame": frames * 4 - 1,
                        "output_start_frame": 0,
                        "output_end_frame": frames - 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return video


def _write_cut_timeline(children: Path, video_id: str) -> Path:
    child_dir = children / video_id
    child_dir.mkdir(parents=True)
    source = children.parent / "source.mp4"
    source.write_bytes(b"source")
    cut_list = children.parent / "cuts.txt"
    cut_list.write_text("2-3\n", encoding="utf-8")
    video = child_dir / f"{video_id}.normalized.mp4"
    video.write_bytes(b"clean-video")
    timeline = child_dir / f"{video_id}.timeline.json"
    timeline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "private_original_frame_curation",
                "privacy": "private_do_not_ship_or_upload",
                "video_id": video_id,
                "audio": "removed",
                "source_video": original_curated._binding(source),
                "cut_list": original_curated._binding(cut_list),
                "output_video": original_curated._binding(video),
                "source": {"fps": 30.0, "n_frames": 6},
                "output": {"fps": 30.0, "n_frames": 4},
                "cut_ranges": [{"start_frame": 2, "end_frame": 3}],
                "segments": [
                    {
                        "segment_id": 0,
                        "source_start_frame": 0,
                        "source_end_frame": 1,
                        "output_start_frame": 0,
                        "output_end_frame": 1,
                    },
                    {
                        "segment_id": 1,
                        "source_start_frame": 4,
                        "source_end_frame": 5,
                        "output_start_frame": 2,
                        "output_end_frame": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return video


def _write_annotation_run(
    run_dir: Path,
    video_id: str,
    video: Path,
    segment_lengths: list[int],
    model_id: str = "qwen3.8-max",
    caption_overrun_ms: int = 0,
) -> None:
    root = run_dir / "private" / "original_curated"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / f"{video_id}.json").write_text(
        json.dumps(
            {
                "workflow": "private_original_trim_mediapipe_vlm",
                "curated_video": original_curated._binding(video),
            }
        ),
        encoding="utf-8",
    )
    for segment_id, length in enumerate(segment_lengths):
        hand_dir = root / "hands" / video_id
        hand_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "video_id": f"{video_id}.segment.{segment_id:04d}",
                "frame_idx": frame_idx,
                "timestamp_ms": round(frame_idx * 1000 / 30),
                "left_landmarks": [0.1] * 63,
                "right_landmarks": [0.2] * 63,
                "left_confidence": 0.9,
                "right_confidence": 0.8,
                "hands_present": 2,
                "detector_version": "mediapipe",
                "left_track_id": 4,
                "right_track_id": 8,
            }
            for frame_idx in range(length)
        ]
        pq.write_table(
            pa.Table.from_pylist(rows),
            hand_dir / f"segment_{segment_id:04d}.parquet",
        )
        event_dir = root / "events" / video_id / model_id
        event_dir.mkdir(parents=True, exist_ok=True)
        end_ms = round(length * 1000 / 30) + caption_overrun_ms
        (event_dir / f"segment_{segment_id:04d}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "private_curated_caption_event_timeline",
                    "privacy": "private_do_not_ship_or_upload",
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "model_id": model_id,
                    "summary": {"text": f"Segment {segment_id} summary.", "steps": ["Do the task."]},
                    "activity_observations": [
                        {
                            "window_idx": 0,
                            "start_ts_ms": 0,
                            "end_ts_ms": end_ms,
                            "caption": "The wearer completes a task.",
                        }
                    ],
                    "events": [
                        {
                            "event_idx": 0,
                            "start_ts_ms": 0,
                            "end_ts_ms": end_ms,
                            "task_step": "complete_task",
                            "action_caption": "The wearer completes a task.",
                            "left_hand": {"visible": True},
                            "right_hand": {"visible": True},
                            "source_schema_ok": True,
                            "source_window_indices": [0],
                            "source_actions": [{"window_idx": 0, "action_idx": 0}],
                        }
                    ],
                    "merge_policy": {"name": "test"},
                }
            ),
            encoding="utf-8",
        )


def _build(
    tmp_path: Path,
    monkeypatch,
    *,
    cut: bool = False,
    caption_overrun_ms: int = 0,
) -> Path:
    children = tmp_path / "private" / "face_free_children"
    video_id = "GX010059"
    video = _write_cut_timeline(children, video_id) if cut else _write_face_free_timeline(
        children, video_id, 4
    )
    segment_lengths = [2, 2] if cut else [4]
    _write_annotation_run(
        tmp_path / "run",
        video_id,
        video,
        segment_lengths,
        caption_overrun_ms=caption_overrun_ms,
    )
    overlays = tmp_path / "annotated-videos"
    overlays.mkdir()
    (overlays / f"{video_id}.hands-qwenmax.mp4").write_bytes(b"overlay-video")

    def fake_probe(path: Path) -> VideoInfo:
        return _video_info(4)

    monkeypatch.setattr(public_release, "probe", fake_probe)
    monkeypatch.setattr(original_curated, "probe", fake_probe)
    monkeypatch.setattr(public_release, "_full_decode_video", lambda *_args, **_kwargs: None)
    output = tmp_path / "public-release"
    public_release.build_public_release(
        children_dir=children,
        annotation_run_dirs=[tmp_path / "run"],
        annotated_video_dirs=[overlays],
        output_dir=output,
        video_ids=[video_id],
        model_id="qwen3.8-max",
        approved_by="Dataset owner",
    )
    return output


def test_public_release_exports_clean_overlay_and_annotations(
    tmp_path: Path, monkeypatch
) -> None:
    output = _build(tmp_path, monkeypatch)

    manifest = json.loads((output / "manifests" / "GX010059.json").read_text(encoding="utf-8"))
    hands = json.loads(
        (output / "annotations" / "GX010059" / "hands.json").read_text(encoding="utf-8")
    )
    captions = json.loads(
        (output / "annotations" / "GX010059" / "captions.json").read_text(encoding="utf-8")
    )

    assert (output / "videos" / "GX010059.mp4").is_file()
    assert (output / "annotated-videos" / "GX010059.mp4").is_file()
    assert (output / "annotations" / "GX010059" / "hands.parquet").is_file()
    assert manifest["privacy"] == "public_release"
    assert manifest["curation"]["curation_method"] == "manually_reviewed_face_free_footage"
    assert manifest["curation"]["annotation_segments"] == [
        {
            "segment_id": 0,
            "start_frame": 0,
            "end_frame": 3,
            "start_ms": 0,
            "end_ms": 133,
            "annotation_state": "fresh",
        }
    ]
    assert manifest["curation"]["discontinuities"] == []
    assert [row["segment_id"] for row in hands["frames"]] == [0, 0, 0, 0]
    assert [row["frame_idx"] for row in hands["frames"]] == [0, 1, 2, 3]
    assert captions["segments"][0]["events"][0]["segment_id"] == 0
    assert "private_do_not_ship_or_upload" not in json.dumps(manifest)
    assert "/Users/" not in json.dumps(manifest)


def test_public_release_marks_cut_boundaries_and_resets_annotation_continuity(
    tmp_path: Path, monkeypatch
) -> None:
    output = _build(tmp_path, monkeypatch, cut=True)

    manifest = json.loads((output / "manifests" / "GX010059.json").read_text(encoding="utf-8"))
    hands = json.loads(
        (output / "annotations" / "GX010059" / "hands.json").read_text(encoding="utf-8")
    )
    captions = json.loads(
        (output / "annotations" / "GX010059" / "captions.json").read_text(encoding="utf-8")
    )

    assert manifest["curation"]["curation_method"] == "manually_removed_face_frames"
    assert manifest["curation"]["discontinuities"] == [
        {
            "after_segment_id": 0,
            "before_segment_id": 1,
            "at_output_frame": 2,
            "at_output_ms": 67,
            "reason": "frames_removed_during_privacy_curation",
            "annotation_state": "MediaPipe tracks and caption windows restart",
        }
    ]
    assert [row["segment_id"] for row in hands["frames"]] == [0, 0, 1, 1]
    assert [row["frame_idx"] for row in hands["frames"]] == [0, 1, 2, 3]
    assert captions["segments"][1]["events"][0]["start_ms"] == 67


def test_public_release_clamps_only_a_trailing_caption_window(
    tmp_path: Path, monkeypatch
) -> None:
    output = _build(tmp_path, monkeypatch, caption_overrun_ms=20)

    captions = json.loads(
        (output / "annotations" / "GX010059" / "captions.json").read_text(encoding="utf-8")
    )
    observation = captions["segments"][0]["activity_observations"][0]
    event = captions["segments"][0]["events"][0]

    assert observation["end_ms"] == 133
    assert event["end_ms"] == 133
    assert observation["end_clamped_to_segment"] is True
    assert event["end_clamped_to_segment"] is True
