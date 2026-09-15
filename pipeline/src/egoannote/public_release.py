"""Build an explicitly approved public release from curated annotations.

The original-curated workflow remains private by default. This module is the
narrow, auditable escape hatch for an owner-approved release: it copies only
selected videos, final overlay videos, and re-shaped annotations into a new
public directory. It never uploads a private run directory verbatim.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import pyarrow as pa
import pyarrow.parquet as pq

from . import curated_caption_events, original_curated
from .media.probe import VideoInfo, probe

PUBLIC_RELEASE_SCHEMA_VERSION = 1
PUBLIC_PRIVACY_LABEL = "public_release"
PUBLIC_LICENSE_ID = "cc-by-4.0"


def _is_private_release_path(path: Path) -> bool:
    """Treat project private directories as unsafe, not macOS's /private mount."""
    parts = path.resolve().parts
    for index, part in enumerate(parts):
        lowered = part.lower()
        if lowered == "do-not-ship":
            return True
        if lowered == "private":
            is_macos_runtime_mount = index == 1 and parts[:3] in {("/", "private", "var"), ("/", "private", "tmp")}
            if not is_macos_runtime_mount:
                return True
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if source.stat().st_size != destination.stat().st_size or _sha256(source) != _sha256(destination):
        raise RuntimeError(f"copied release file does not match its source: {source.name}")


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _frame_ms(frame_idx: int, fps: float) -> int:
    return round(frame_idx * 1000 / fps)


def _public_segments(
    segments: list[original_curated.TimelineSegment], *, fps: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public_segments: list[dict[str, Any]] = []
    discontinuities: list[dict[str, Any]] = []
    expected_start = 0
    for index, segment in enumerate(segments):
        if segment.segment_id != index:
            raise ValueError("curation segments must be numbered consecutively from zero")
        if segment.output_start_frame != expected_start:
            raise ValueError("curation output segments must be contiguous")
        expected_start = segment.output_end_frame + 1
        public_segments.append(
            {
                "segment_id": segment.segment_id,
                "start_frame": segment.output_start_frame,
                "end_frame": segment.output_end_frame,
                "start_ms": _frame_ms(segment.output_start_frame, fps),
                "end_ms": _frame_ms(segment.output_end_frame + 1, fps),
                "annotation_state": "fresh" if index == 0 else "reset_after_curation_boundary",
            }
        )
        if index:
            discontinuities.append(
                {
                    "after_segment_id": segments[index - 1].segment_id,
                    "before_segment_id": segment.segment_id,
                    "at_output_frame": segment.output_start_frame,
                    "at_output_ms": _frame_ms(segment.output_start_frame, fps),
                    "reason": "frames_removed_during_privacy_curation",
                    "annotation_state": "MediaPipe tracks and caption windows restart",
                }
            )
    return public_segments, discontinuities


def _validate_release_video(video: Path, *, expected: VideoInfo, label: str) -> VideoInfo:
    actual = probe(video)
    if actual.is_vfr:
        raise ValueError(f"{label} must be constant frame rate")
    if (
        actual.n_frames != expected.n_frames
        or (actual.width, actual.height) != (expected.width, expected.height)
        or abs(actual.fps - expected.fps) > 0.001
    ):
        raise ValueError(f"{label} does not align with the curated clean video")
    return actual


def _full_decode_video(video: Path, *, expected_frames: int, label: str) -> None:
    """Reject a video that ffprobe can describe but cannot fully decode."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {label} for full decode")
    decoded = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded += 1
    finally:
        capture.release()
    if decoded != expected_frames:
        raise RuntimeError(f"{label} decoded {decoded}/{expected_frames} frames")


def _annotation_root(
    *,
    run_dirs: list[Path],
    video_id: str,
    curated_sha256: str,
) -> Path:
    """Choose the first ordered run with complete annotations for this video."""
    for run_dir in run_dirs:
        root = run_dir.resolve() / "private" / "original_curated"
        summary_path = root / "runs" / f"{video_id}.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("workflow") != "private_original_trim_mediapipe_vlm":
            continue
        curated = summary.get("curated_video")
        if not isinstance(curated, dict) or curated.get("sha256") != curated_sha256:
            continue
        return root
    raise FileNotFoundError(
        f"no ordered annotation run binds to the approved curated video for {video_id}"
    )


def _overlay_video(*, overlay_dirs: list[Path], video_id: str) -> Path:
    for directory in overlay_dirs:
        matches = sorted(directory.resolve().glob(f"{video_id}*.mp4"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"ambiguous annotated overlay for {video_id} in {directory}")
    raise FileNotFoundError(f"annotated overlay video not found for {video_id}")


def _public_hand_rows(
    *,
    annotation_root: Path,
    video_id: str,
    segments: list[original_curated.TimelineSegment],
    fps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "frame_idx",
        "left_landmarks",
        "right_landmarks",
        "left_confidence",
        "right_confidence",
        "hands_present",
        "detector_version",
        "left_track_id",
        "right_track_id",
    }
    for segment in segments:
        hand_path = annotation_root / "hands" / video_id / f"segment_{segment.segment_id:04d}.parquet"
        if not hand_path.is_file():
            raise FileNotFoundError(f"hand annotation missing for {video_id} segment {segment.segment_id}")
        segment_rows = pq.read_table(hand_path).to_pylist()
        if len(segment_rows) != segment.n_frames:
            raise ValueError(
                f"hand annotations for {video_id} segment {segment.segment_id} have "
                f"{len(segment_rows)}/{segment.n_frames} rows"
            )
        seen_frames: set[int] = set()
        for row in segment_rows:
            if not required.issubset(row):
                missing = sorted(required - set(row))
                raise ValueError(f"hand annotation is missing fields: {missing}")
            local_frame = _as_int(row["frame_idx"], "hand frame_idx")
            if local_frame < 0 or local_frame >= segment.n_frames or local_frame in seen_frames:
                raise ValueError(f"hand annotation has invalid frame index in segment {segment.segment_id}")
            seen_frames.add(local_frame)
            output_frame = segment.output_start_frame + local_frame
            rows.append(
                {
                    "video_id": video_id,
                    "segment_id": segment.segment_id,
                    "frame_idx": output_frame,
                    "segment_frame_idx": local_frame,
                    "timestamp_ms": _frame_ms(output_frame, fps),
                    "left_landmarks": row["left_landmarks"],
                    "right_landmarks": row["right_landmarks"],
                    "left_confidence": row["left_confidence"],
                    "right_confidence": row["right_confidence"],
                    "hands_present": row["hands_present"],
                    "detector_version": row["detector_version"],
                    "left_track_id": row["left_track_id"],
                    "right_track_id": row["right_track_id"],
                }
            )
        if seen_frames != set(range(segment.n_frames)):
            raise ValueError(f"hand annotations do not cover every frame in segment {segment.segment_id}")
    return rows


def _public_caption_timeline(
    *,
    annotation_root: Path,
    video_id: str,
    model_id: str,
    segments: list[original_curated.TimelineSegment],
    fps: float,
) -> dict[str, Any]:
    public_segments: list[dict[str, Any]] = []
    for segment in segments:
        private = curated_caption_events.load_event_timeline(
            annotation_root / "events" / video_id / model_id / f"segment_{segment.segment_id:04d}.json",
            video_id=video_id,
            segment_id=segment.segment_id,
            model_id=model_id,
        )
        offset_ms = _frame_ms(segment.output_start_frame, fps)
        max_ms = _frame_ms(segment.n_frames, fps)

        def shifted(
            item: dict[str, Any],
            *,
            label: str,
            max_segment_ms: int = max_ms,
            segment_offset_ms: int = offset_ms,
            segment_id: int = segment.segment_id,
        ) -> dict[str, Any]:
            start = _as_int(item.get("start_ts_ms"), f"{label} start_ts_ms")
            end = _as_int(item.get("end_ts_ms"), f"{label} end_ts_ms")
            if start < 0 or end <= start or start >= max_segment_ms:
                raise ValueError(f"{label} has timestamps outside its annotation segment")
            end_was_clamped = end > max_segment_ms
            end = min(end, max_segment_ms)
            copied = {
                key: item.get(key)
                for key in (
                    "window_idx",
                    "task_step",
                    "action_caption",
                    "left_hand",
                    "right_hand",
                    "tool_in_use",
                    "coordination",
                    "handover_event",
                    "scene",
                    "source_schema_ok",
                    "source_window_indices",
                    "source_actions",
                    "event_idx",
                    "caption",
                )
                if key in item
            }
            copied.update(
                {
                    "segment_id": segment_id,
                    "start_ms": segment_offset_ms + start,
                    "end_ms": segment_offset_ms + end,
                }
            )
            if end_was_clamped:
                copied["end_clamped_to_segment"] = True
            return copied

        summary = private.get("summary") or {}
        if not isinstance(summary.get("text"), str) or not isinstance(summary.get("steps"), list):
            raise ValueError(f"caption timeline has no complete summary for segment {segment.segment_id}")
        observations = private.get("activity_observations")
        events = private.get("events")
        if not isinstance(observations, list) or not isinstance(events, list):
            raise ValueError(f"caption timeline is incomplete for segment {segment.segment_id}")
        public_segments.append(
            {
                "segment_id": segment.segment_id,
                "summary": {"text": summary["text"], "steps": summary["steps"]},
                "activity_observations": [
                    shifted(item, label="activity observation") for item in observations
                ],
                "events": [shifted(item, label="caption event") for item in events],
                "merge_policy": private.get("merge_policy"),
            }
        )
    return {
        "schema_version": PUBLIC_RELEASE_SCHEMA_VERSION,
        "artifact_type": "egoannote_public_caption_timeline",
        "privacy": PUBLIC_PRIVACY_LABEL,
        "video_id": video_id,
        "model_id": model_id,
        "segments": public_segments,
    }


def _assert_public_payload(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True).lower()
    forbidden = ("private_do_not_ship_or_upload", "/users/", "do-not-ship")
    if any(token in serialized for token in forbidden):
        raise ValueError("public release payload contains private-only metadata")


def _dataset_card(*, video_count: int) -> str:
    return f"""---
pretty_name: EgoAnnotate face-free egocentric activity annotations
license: {PUBLIC_LICENSE_ID}
tags:
- video
- egocentric
- hand-tracking
- activity-recognition
---

# EgoAnnotate v1

This public release contains {video_count} manually reviewed face-free egocentric
videos with synchronized MediaPipe hand landmarks and dense activity captions.
Each clip includes its clean video, a hand-and-caption overlay video, hand data
as JSON and Parquet, and captions as JSON.

## Curation and continuity

The clean videos are normalized to 29.97 fps with audio and container metadata
removed. Every per-video manifest declares its annotation segments and any
privacy-curation discontinuities. At a discontinuity, MediaPipe track state and
caption windows restart; consumers must not treat annotations across it as one
continuous track. This v1 release has no frame-removal discontinuities: each
clip is a single, manually reviewed face-free segment.

## Files

- videos/: clean, normalized videos.
- annotated-videos/: final videos with hand landmarks and captions burned in.
- annotations/video-id/hands.json: one hand annotation record per output frame.
- annotations/video-id/hands.parquet: the same records in columnar form.
- annotations/video-id/captions.json: summaries, activity captions, atomic actions,
  and structured left/right-hand states.
- manifests/: public media hashes, curation metadata, and annotation counts.

## Limitations

This is a small, owner-curated release. The captions are model-generated and
should be treated as annotations rather than ground truth. The per-video
manifest is the source of truth for curation boundaries and annotation
continuity.
"""


def _license_text() -> str:
    return """EgoAnnotate dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0)

You may share and adapt this dataset for any purpose, including commercially,
as long as you give appropriate credit, provide a link to the license, and
indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode
"""


def build_public_release(
    *,
    children_dir: Path,
    annotation_run_dirs: list[Path],
    annotated_video_dirs: list[Path],
    output_dir: Path,
    video_ids: list[str],
    model_id: str,
    approved_by: str,
    release_version: str = "v1.0",
) -> dict[str, Any]:
    """Create a public, self-contained release folder from explicit video IDs.

    annotation_run_dirs and annotated_video_dirs are ordered. The first
    directory that contains a hash-bound complete artifact wins, allowing an
    operator to prefer a newer threshold-specific run without ambiguity.
    """
    children_dir = children_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    annotation_run_dirs = [path.expanduser().resolve() for path in annotation_run_dirs]
    annotated_video_dirs = [path.expanduser().resolve() for path in annotated_video_dirs]
    if not approved_by.strip() or "\n" in approved_by or "\r" in approved_by:
        raise ValueError("approved_by must be a non-empty single line")
    if not video_ids or len(set(video_ids)) != len(video_ids):
        raise ValueError("video_ids must be a non-empty unique list")
    if output_dir.exists():
        raise FileExistsError(f"public release output already exists: {output_dir}")
    if _is_private_release_path(output_dir):
        raise ValueError("public release output must not be under private or DO-NOT-SHIP")
    if not children_dir.is_dir() or not annotation_run_dirs or not annotated_video_dirs:
        raise ValueError("children, annotation runs, and annotated-video directories are required")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    approval_bindings: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".egoannote-public-release-", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / output_dir.name
        for video_id in video_ids:
            child_dir = children_dir / video_id
            clean_source = child_dir / f"{video_id}.normalized.mp4"
            timeline_path = child_dir / f"{video_id}.timeline.json"
            if not clean_source.is_file() or not timeline_path.is_file():
                raise FileNotFoundError(f"approved child/timeline missing for {video_id}")
            timeline = original_curated.load_manifest(
                timeline_path, output_video=clean_source, verify_source=False
            )
            if timeline.get("video_id") != video_id:
                raise ValueError(f"timeline belongs to another video: {video_id}")
            artifact_type = timeline.get("artifact_type")
            if artifact_type == "private_face_free_normalization":
                if (timeline.get("face_review") or {}).get("decision") != "manually_reviewed_face_free":
                    raise ValueError(f"{video_id} is not a manually reviewed face-free video")
                curation_method = "manually_reviewed_face_free_footage"
            elif artifact_type == "private_original_frame_curation":
                if not timeline.get("cut_ranges"):
                    raise ValueError(f"{video_id} has no recorded privacy-curation frame removals")
                curation_method = "manually_removed_face_frames"
            else:
                raise ValueError(f"{video_id} has an unsupported curation artifact")
            clean_info = probe(clean_source)
            if clean_info.is_vfr:
                raise ValueError(f"{video_id} clean video must be constant frame rate")
            _full_decode_video(
                clean_source, expected_frames=clean_info.n_frames, label=f"{video_id} clean video"
            )
            segments = [original_curated.TimelineSegment(**item) for item in timeline["segments"]]
            public_segments, discontinuities = _public_segments(segments, fps=clean_info.fps)
            if clean_info.n_frames != sum(segment.n_frames for segment in segments):
                raise ValueError(f"{video_id} clean video does not match its curation segments")

            clean_sha256 = _sha256(clean_source)
            annotation_root = _annotation_root(
                run_dirs=annotation_run_dirs,
                video_id=video_id,
                curated_sha256=clean_sha256,
            )
            hands = _public_hand_rows(
                annotation_root=annotation_root,
                video_id=video_id,
                segments=segments,
                fps=clean_info.fps,
            )
            captions = _public_caption_timeline(
                annotation_root=annotation_root,
                video_id=video_id,
                model_id=model_id,
                segments=segments,
                fps=clean_info.fps,
            )
            _assert_public_payload(captions)

            clean_destination = staging / "videos" / f"{video_id}.mp4"
            overlay_source = _overlay_video(overlay_dirs=annotated_video_dirs, video_id=video_id)
            _validate_release_video(overlay_source, expected=clean_info, label=f"{video_id} annotated overlay")
            _full_decode_video(
                overlay_source,
                expected_frames=clean_info.n_frames,
                label=f"{video_id} annotated overlay",
            )
            overlay_destination = staging / "annotated-videos" / f"{video_id}.mp4"
            _copy(clean_source, clean_destination)
            _copy(overlay_source, overlay_destination)

            annotation_dir = staging / "annotations" / video_id
            hands_json = annotation_dir / "hands.json"
            hands_parquet = annotation_dir / "hands.parquet"
            captions_path = annotation_dir / "captions.json"
            _write_json(
                hands_json,
                {
                    "schema_version": PUBLIC_RELEASE_SCHEMA_VERSION,
                    "artifact_type": "egoannote_public_hand_annotations",
                    "privacy": PUBLIC_PRIVACY_LABEL,
                    "video_id": video_id,
                    "frames": hands,
                },
            )
            pq.write_table(pa.Table.from_pylist(hands), hands_parquet, compression="zstd")
            _write_json(captions_path, captions)

            manifest = {
                "schema_version": PUBLIC_RELEASE_SCHEMA_VERSION,
                "artifact_type": "egoannote_public_video_manifest",
                "privacy": PUBLIC_PRIVACY_LABEL,
                "license": PUBLIC_LICENSE_ID,
                "video_id": video_id,
                "curation": {
                    "curation_method": curation_method,
                    "audio": "removed",
                    "source_to_output_normalization": {
                        "source_fps": (timeline.get("source") or {}).get("fps"),
                        "output_fps": clean_info.fps,
                        "target_fps": (timeline.get("normalization") or {}).get("target_fps"),
                    },
                    "annotation_segments": public_segments,
                    "discontinuities": discontinuities,
                },
                "media": {
                    "clean_video": _binding(clean_destination, relative_to=staging),
                    "annotated_video": _binding(overlay_destination, relative_to=staging),
                    "video": asdict(clean_info),
                },
                "annotations": {
                    "hands_json": _binding(hands_json, relative_to=staging),
                    "hands_parquet": _binding(hands_parquet, relative_to=staging),
                    "captions_json": _binding(captions_path, relative_to=staging),
                    "hand_frames": len(hands),
                    "caption_segments": len(captions["segments"]),
                    "caption_events": sum(len(item["events"]) for item in captions["segments"]),
                },
            }
            _assert_public_payload(manifest)
            manifest_path = staging / "manifests" / f"{video_id}.json"
            _write_json(manifest_path, manifest)
            manifests.append(
                {
                    "video_id": video_id,
                    "manifest": _binding(manifest_path, relative_to=staging),
                    "clean_video_sha256": manifest["media"]["clean_video"]["sha256"],
                }
            )
            approval_bindings.append({"video_id": video_id, "clean_video_sha256": clean_sha256})

        _write_json(
            staging / "release-approval.json",
            {
                "schema_version": PUBLIC_RELEASE_SCHEMA_VERSION,
                "artifact_type": "egoannote_public_release_approval",
                "privacy": PUBLIC_PRIVACY_LABEL,
                "approved_by": approved_by.strip(),
                "release_version": release_version,
                "license": PUBLIC_LICENSE_ID,
                "videos": approval_bindings,
            },
        )
        _write_json(
            staging / "release-manifest.json",
            {
                "schema_version": PUBLIC_RELEASE_SCHEMA_VERSION,
                "artifact_type": "egoannote_public_release_manifest",
                "privacy": PUBLIC_PRIVACY_LABEL,
                "release_version": release_version,
                "license": PUBLIC_LICENSE_ID,
                "videos": manifests,
            },
        )
        (staging / "README.md").write_text(_dataset_card(video_count=len(video_ids)), encoding="utf-8")
        (staging / "LICENSE").write_text(_license_text(), encoding="utf-8")
        staging.replace(output_dir)
    return {
        "output_dir": str(output_dir),
        "release_version": release_version,
        "video_ids": video_ids,
        "videos": manifests,
    }
