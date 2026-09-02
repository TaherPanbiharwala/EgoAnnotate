from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egoannote import original_curated, pipeline
from egoannote.media.probe import VideoInfo


def test_original_curation_ranges_are_zero_based_inclusive_and_segments_are_contiguous(
    tmp_path: Path,
) -> None:
    cut_list = tmp_path / "cuts.txt"
    cut_list.write_text("3-5\nframe_000010.jpg - frame_000011.jpg\n", encoding="utf-8")

    cuts = original_curated.normalize_cut_ranges(
        original_curated.parse_cut_ranges(cut_list), 15
    )
    assert cuts == [original_curated.CutRange(3, 5), original_curated.CutRange(10, 11)]
    assert original_curated.build_segments(15, cuts) == [
        original_curated.TimelineSegment(0, 0, 2, 0, 2),
        original_curated.TimelineSegment(1, 6, 9, 3, 6),
        original_curated.TimelineSegment(2, 12, 14, 7, 9),
    ]


def test_original_curation_requires_private_child_and_manifest() -> None:
    with pytest.raises(ValueError, match="output-video must be under private"):
        original_curated._require_private(Path("/nonprivate-child.mp4"), "--output-video")
    with pytest.raises(ValueError, match="manifest must be under private"):
        original_curated._require_private(Path("/nonprivate-timeline.json"), "--manifest")


def test_face_free_timeline_accepts_one_normalized_output_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "private" / "child.mp4"
    manifest = tmp_path / "private" / "timeline.json"
    source.write_bytes(b"source")
    output.parent.mkdir()
    output.write_bytes(b"child")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_type": "private_face_free_normalization",
                "privacy": "private_do_not_ship_or_upload",
                "video_id": "GX010087",
                "face_review": {"decision": "manually_reviewed_face_free", "reviewer": "Taher"},
                "source_video": original_curated._binding(source),
                "output_video": original_curated._binding(output),
                "source": {"n_frames": 240},
                "output": {"n_frames": 60},
                "segments": [{
                    "segment_id": 0,
                    "source_start_frame": 0,
                    "source_end_frame": 239,
                    "output_start_frame": 0,
                    "output_end_frame": 59,
                }],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        original_curated,
        "probe",
        lambda _path: VideoInfo(2.0, 1920, 1080, 30.0, 60, False),
    )

    loaded = original_curated.load_manifest(manifest, output_video=output)

    assert loaded["artifact_type"] == "private_face_free_normalization"
    assert loaded["segments"][0]["output_end_frame"] == 59


def test_face_free_timeline_can_render_after_source_is_archived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "private" / "child.mp4"
    manifest = tmp_path / "private" / "timeline.json"
    source.write_bytes(b"source")
    output.parent.mkdir()
    output.write_bytes(b"child")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_type": "private_face_free_normalization",
                "privacy": "private_do_not_ship_or_upload",
                "video_id": "GX010087",
                "face_review": {"decision": "manually_reviewed_face_free", "reviewer": "Taher"},
                "source_video": original_curated._binding(source),
                "output_video": original_curated._binding(output),
                "source": {"n_frames": 240},
                "output": {"n_frames": 60},
                "segments": [{
                    "segment_id": 0,
                    "source_start_frame": 0,
                    "source_end_frame": 239,
                    "output_start_frame": 0,
                    "output_end_frame": 59,
                }],
            }
        ),
        encoding="utf-8",
    )
    source.unlink()
    monkeypatch.setattr(
        original_curated,
        "probe",
        lambda _path: VideoInfo(2.0, 1920, 1080, 30.0, 60, False),
    )

    loaded = original_curated.load_manifest(
        manifest,
        output_video=output,
        verify_source=False,
    )

    assert loaded["_source_video"].endswith("source.mp4")
    with pytest.raises(FileNotFoundError, match="source video is missing"):
        original_curated.load_manifest(manifest, output_video=output)


def test_hand_preview_refuses_a_nonprivate_output_before_reading_inputs() -> None:
    with pytest.raises(ValueError, match="output-video must be under private"):
        original_curated.render_hand_preview(
            curated_video=Path("missing.mp4"),
            timeline_manifest=Path("missing.json"),
            run_dir=Path("missing-run"),
            video_id="GX010057",
            output_video=Path("preview.mp4"),
        )


def test_caption_pilot_refuses_a_nonprivate_output_before_reading_inputs() -> None:
    with pytest.raises(ValueError, match="output-video must be under private"):
        original_curated.render_caption_pilot(
            curated_video=Path("missing.mp4"),
            timeline_manifest=Path("missing.json"),
            run_dir=Path("missing-run"),
            video_id="GX010057",
            model_id="qwen3.8-flash",
            segment_id=0,
            window_idx=0,
            output_video=Path("caption-preview.mp4"),
        )


def test_hand_preview_recovers_canonical_hands_after_caption_only_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caption pilot must not require a costly 30 fps MediaPipe rerun."""
    timeline = {
        "video_id": "GX010057",
        "segments": [
            {"segment_id": 0, "source_start_frame": 0, "source_end_frame": 9,
             "output_start_frame": 0, "output_end_frame": 9},
            {"segment_id": 1, "source_start_frame": 20, "source_end_frame": 29,
             "output_start_frame": 10, "output_end_frame": 19},
        ],
    }
    monkeypatch.setattr(original_curated, "load_manifest", lambda *_a, **_k: timeline)
    run_dir = tmp_path / "run"
    root = run_dir / "private" / "original_curated"
    hands_dir = root / "hands" / "GX010057"
    hands_dir.mkdir(parents=True)
    for segment_id in (0, 1):
        pq.write_table(
            pa.table({"frame_idx": [segment_id]}),
            hands_dir / f"segment_{segment_id:04d}.parquet",
        )
    runs_dir = root / "runs"
    runs_dir.mkdir()
    # This mirrors a selected-segment caption-only pilot: it contains no
    # ``hands`` entry and only mentions the pilot segment.
    (runs_dir / "GX010057.json").write_text(
        '{"workflow":"private_original_trim_mediapipe_vlm","segments":[{"segment_id":0}]}',
        encoding="utf-8",
    )

    _video, segments, stored, _summary, _root = original_curated._load_render_context(
        curated_video=tmp_path / "private" / "child.mp4",
        timeline_manifest=tmp_path / "private" / "timeline.json",
        run_dir=run_dir,
        video_id="GX010057",
    )

    assert [segment.segment_id for segment in segments] == [0, 1]
    assert {segment_id: list(rows) for segment_id, rows in stored.items()} == {
        0: [0],
        1: [1],
    }


def test_burned_caption_uses_the_action_active_at_the_current_timestamp() -> None:
    captions = [
        {
            "start_ts_ms": 0,
            "end_ts_ms": 6000,
            "activity": {"caption": "The person assembles a small box."},
            "actions": [
                {
                    "start_frame": 0,
                    "end_frame": 3,
                    "action_caption": "The person holds the box flap.",
                    "left_hand": {"caption": "The left hand steadies the box."},
                    "right_hand": {"caption": "The right hand folds a flap."},
                },
                {
                    "start_frame": 4,
                    "end_frame": 7,
                    "action_caption": "The person presses the flap closed.",
                    "left_hand": {"caption": "The left hand holds the box."},
                    "right_hand": {"caption": "The right hand presses the seam."},
                },
            ],
        }
    ]

    early = [text for text, _color in original_curated._caption_lines(captions, 100)]
    late = [text for text, _color in original_curated._caption_lines(captions, 4000)]

    assert any("folds a flap" in text for text in early)
    assert any("presses the seam" in text for text in late)


def test_curated_caption_pilot_resets_segments_but_shares_one_spend_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curated_video = tmp_path / "curated.mp4"
    timeline_manifest = tmp_path / "timeline.json"
    curated_video.write_bytes(b"video")
    timeline_manifest.write_text("{}", encoding="utf-8")
    timeline = {
        "video_id": "GX010057",
        "segments": [
            {"segment_id": 0, "source_start_frame": 0, "source_end_frame": 9,
             "output_start_frame": 0, "output_end_frame": 9},
            {"segment_id": 1, "source_start_frame": 20, "source_end_frame": 29,
             "output_start_frame": 10, "output_end_frame": 19},
        ],
    }
    monkeypatch.setattr(pipeline.original_curated_layer, "load_manifest", lambda *_a, **_k: timeline)

    def materialize(*, output_video: Path, **_kwargs: object) -> None:
        output_video.write_bytes(b"segment")

    monkeypatch.setattr(pipeline.original_curated_layer, "materialize_segment", materialize)
    backends: list[SimpleNamespace] = []

    def build(*_args: object, spend_tracker: object, **_kwargs: object) -> SimpleNamespace:
        backend = SimpleNamespace(model_id="qwen", spend_tracker=spend_tracker, close=lambda: None)
        backends.append(backend)
        return backend

    monkeypatch.setattr(pipeline, "build_backend", build)
    seen_windows: list[set[int] | None] = []

    def caption(*_args: object, window_indices: set[int] | None = None, **_kwargs: object) -> int:
        seen_windows.append(window_indices)
        return 1

    monkeypatch.setattr(pipeline.caption_layer, "caption_video", caption)

    summary = pipeline.annotate_curated_original(
        run_dir=tmp_path / "run",
        curated_video=curated_video,
        timeline_manifest=timeline_manifest,
        video_id="GX010057",
        model_ids=["qwen"],
        registry_path=None,
        max_spend_per_model=10.0,
        workers=1,
        run_hands=False,
        run_captions=True,
        ffmpeg="ffmpeg",
        caption_windows={0},
    )

    assert len(backends) == 2
    assert backends[0].spend_tracker is backends[1].spend_tracker
    assert seen_windows == [{0}, {0}]
    assert summary["caption_selection"] == {
        "segment_ids": None,
        "window_indices_per_segment": [0],
    }


def test_full_curated_caption_writes_private_event_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curated_video = tmp_path / "curated.mp4"
    timeline_manifest = tmp_path / "timeline.json"
    curated_video.write_bytes(b"video")
    timeline_manifest.write_text("{}", encoding="utf-8")
    timeline = {
        "video_id": "GX010057",
        "segments": [
            {"segment_id": 0, "source_start_frame": 0, "source_end_frame": 7,
             "output_start_frame": 0, "output_end_frame": 7},
        ],
    }
    monkeypatch.setattr(pipeline.original_curated_layer, "load_manifest", lambda *_a, **_k: timeline)
    monkeypatch.setattr(
        pipeline.original_curated_layer,
        "materialize_segment",
        lambda *, output_video, **_kwargs: output_video.write_bytes(b"segment"),
    )
    monkeypatch.setattr(
        pipeline,
        "build_backend",
        lambda *_args, **_kwargs: SimpleNamespace(model_id="qwen", close=lambda: None),
    )

    def caption(_video, scoped_id, _backend, store, *, frames_dir, **_kwargs):
        frame_dir = frames_dir / scoped_id
        frame_dir.mkdir(parents=True)
        (frame_dir / "_COMPLETE.json").write_text('{"frame_count":8}', encoding="utf-8")
        store.write(
            video_id=scoped_id,
            unit_idx=0,
            stage="caption",
            model_id="qwen",
            payload={
                "window_idx": 0,
                "start_ts_ms": 0,
                "end_ts_ms": 6000,
                "prompt_version": "v5",
                "schema_ok": True,
                "activity": {"caption": "The camera wearer folds a box."},
                "actions": [{
                    "start_frame": 0,
                    "end_frame": 7,
                    "task_step": "fold_box_flap",
                    "action_caption": "The camera wearer folds a box flap.",
                    "left_hand": {},
                    "right_hand": {},
                }],
            },
        )
        return 1

    monkeypatch.setattr(pipeline.caption_layer, "caption_video", caption)
    monkeypatch.setattr(
        pipeline.caption_layer,
        "summarize_caption_windows",
        lambda *_args, **_kwargs: (
            {"summary": "The camera wearer folds a box.", "steps": ["Fold the box flap."]},
            True,
        ),
    )

    summary = pipeline.annotate_curated_original(
        run_dir=tmp_path / "run",
        curated_video=curated_video,
        timeline_manifest=timeline_manifest,
        video_id="GX010057",
        model_ids=["qwen"],
        registry_path=None,
        max_spend_per_model=10.0,
        workers=1,
        run_hands=False,
        run_captions=True,
        ffmpeg="ffmpeg",
    )

    event = summary["segments"][0]["captions"]["event_timelines"]["qwen"]
    event_path = Path(event["path"])
    assert event_path.is_file()
    assert "private" in event_path.parts
    assert event["summary_api_called"] is True


def test_curated_hand_confidence_is_run_specific_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curated_video = tmp_path / "curated.mp4"
    timeline_manifest = tmp_path / "timeline.json"
    curated_video.write_bytes(b"video")
    timeline_manifest.write_text("{}", encoding="utf-8")
    timeline = {
        "video_id": "GX010057",
        "segments": [
            {"segment_id": 0, "source_start_frame": 0, "source_end_frame": 9,
             "output_start_frame": 0, "output_end_frame": 9},
        ],
    }
    monkeypatch.setattr(pipeline.original_curated_layer, "load_manifest", lambda *_a, **_k: timeline)
    monkeypatch.setattr(
        pipeline.original_curated_layer,
        "materialize_segment",
        lambda *, output_video, **_kwargs: output_video.write_bytes(b"segment"),
    )
    monkeypatch.setattr(pipeline, "probe", lambda _path: SimpleNamespace())
    received: list[float] = []

    def run_hands(*_args: object, hand_confidence: float, **_kwargs: object):
        received.append(hand_confidence)
        return iter(())

    monkeypatch.setattr(pipeline.hands_layer, "run", run_hands)
    monkeypatch.setattr(
        pipeline,
        "write_hands_parquet_streaming",
        lambda _rows, _path: (0, []),
    )

    summary = pipeline.annotate_curated_original(
        run_dir=tmp_path / "run",
        curated_video=curated_video,
        timeline_manifest=timeline_manifest,
        video_id="GX010057",
        model_ids=[],
        registry_path=None,
        max_spend_per_model=None,
        workers=1,
        run_hands=True,
        run_captions=False,
        ffmpeg="ffmpeg",
        hand_confidence=0.4,
    )

    assert received == [0.4]
    assert summary["segments"][0]["hands"] == {
        "path": str(
            tmp_path / "run" / "private" / "original_curated" / "hands" / "GX010057" / "segment_0000.parquet"
        ),
        "rows": 0,
        "missing_gaps": 0,
        "fps": 30,
        "detection_and_presence_confidence": 0.4,
        "tracking_confidence": 0.5,
    }


def test_face_free_batch_normalizes_and_passes_the_requested_hand_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "batch_upload"
    input_dir.mkdir()
    (input_dir / "GX010088.MP4").write_bytes(b"source-a")
    (input_dir / "GX010087.MP4").write_bytes(b"source-b")
    normalized: list[tuple[str, float]] = []
    annotated: list[tuple[str, float]] = []

    def normalize(*, original_video, video_id, output_video, manifest, **_kwargs):
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_video.write_bytes(b"normalized")
        manifest.write_text("{}", encoding="utf-8")
        normalized.append((video_id, original_video.stat().st_size))
        return {}

    monkeypatch.setattr(pipeline.original_curated_layer, "normalize_face_free_original", normalize)
    monkeypatch.setattr(pipeline.original_curated_layer, "load_manifest", lambda *_a, **_k: {})
    monkeypatch.setattr(pipeline, "_completed_face_free_hands", lambda **_kwargs: False)

    def annotate(*, video_id, hand_confidence, **_kwargs):
        annotated.append((video_id, hand_confidence))
        return {}

    monkeypatch.setattr(pipeline, "annotate_curated_original", annotate)

    result = pipeline.batch_face_free_hands(
        input_dir=input_dir,
        run_dir=tmp_path / "run",
        reviewer="Taher",
        hand_confidence=0.4,
        ffmpeg="ffmpeg",
    )

    assert [video_id for video_id, _size in normalized] == ["GX010087", "GX010088"]
    assert annotated == [("GX010087", 0.4), ("GX010088", 0.4)]
    assert [item["hands"] for item in result["videos"]] == ["created", "created"]
