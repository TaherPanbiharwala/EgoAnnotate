"""Private-only original-video curation and segment-isolated annotation.

This module intentionally has no EgoBlur, YuNet, redaction approval, or
Hugging Face integration.  It is for the explicitly private workflow:

    original -> manually selected frame cuts -> MediaPipe + VLM

Every artifact, including the curated child and its annotations, must live
under ``private`` or ``DO-NOT-SHIP``.  Each retained segment is materialized
independently before annotation, so hand tracking and caption windows cannot
cross an intentional edit.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import pyarrow.parquet as pq

from . import config, curated_caption_events
from .media.probe import probe

_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
_LEFT_COLOR = (70, 220, 70)  # BGR green
_RIGHT_COLOR = (230, 90, 255)  # BGR pink
_FACE_FREE_NORMALIZED_FPS = "30000/1001"

_RANGE_RE = re.compile(
    r"\s*(?:frame_)?(?P<start>\d+)(?:\.(?:jpg|jpeg|png))?"
    r"(?:\s*-\s*(?:frame_)?(?P<end>\d+)(?:\.(?:jpg|jpeg|png))?)?\s*"
)


@dataclass(frozen=True, slots=True)
class CutRange:
    start_frame: int
    end_frame: int


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    segment_id: int
    source_start_frame: int
    source_end_frame: int
    output_start_frame: int
    output_end_frame: int

    @property
    def n_frames(self) -> int:
        return self.output_end_frame - self.output_start_frame + 1


def _is_private(path: Path) -> bool:
    return any(part.lower() in {"private", "do-not-ship"} for part in path.parts)


def _require_private(path: Path, label: str) -> None:
    if not _is_private(path):
        raise ValueError(f"{label} must be under private or DO-NOT-SHIP for this original-video workflow")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _verify_binding(value: Any, label: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"curated-original manifest {label} binding is missing")
    raw, digest, size = value.get("path"), value.get("sha256"), value.get("bytes")
    if not isinstance(raw, str) or not isinstance(digest, str) or not isinstance(size, int):
        raise ValueError(f"curated-original manifest {label} binding is invalid")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"curated-original {label} is missing: {path}")
    if path.stat().st_size != size or _sha256(path) != digest:
        raise ValueError(f"curated-original {label} no longer matches its recorded hash")
    return path


def parse_cut_ranges(path: Path) -> list[CutRange]:
    ranges: list[CutRange] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _RANGE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid cut range on line {line_no}: {raw!r}")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            raise ValueError(f"cut range ends before it starts on line {line_no}")
        ranges.append(CutRange(start, end))
    if not ranges:
        raise ValueError("cut list has no frame ranges")
    return ranges


def normalize_cut_ranges(ranges: list[CutRange], source_frames: int) -> list[CutRange]:
    if source_frames < 1:
        raise ValueError("source video has no frames")
    output: list[CutRange] = []
    for item in sorted(ranges, key=lambda value: (value.start_frame, value.end_frame)):
        if item.start_frame < 0 or item.end_frame >= source_frames:
            raise ValueError(f"cut range {item.start_frame}-{item.end_frame} is out of bounds")
        if output and item.start_frame <= output[-1].end_frame + 1:
            output[-1] = CutRange(output[-1].start_frame, max(output[-1].end_frame, item.end_frame))
        else:
            output.append(item)
    if sum(item.end_frame - item.start_frame + 1 for item in output) >= source_frames:
        raise ValueError("cut list removes every source frame")
    return output


def build_segments(source_frames: int, cuts: list[CutRange]) -> list[TimelineSegment]:
    normalized = normalize_cut_ranges(cuts, source_frames)
    segments: list[TimelineSegment] = []
    source_start = output_start = 0
    for cut in normalized:
        if source_start < cut.start_frame:
            n_frames = cut.start_frame - source_start
            segments.append(
                TimelineSegment(
                    len(segments), source_start, cut.start_frame - 1, output_start,
                    output_start + n_frames - 1,
                )
            )
            output_start += n_frames
        source_start = cut.end_frame + 1
    if source_start < source_frames:
        n_frames = source_frames - source_start
        segments.append(
            TimelineSegment(
                len(segments), source_start, source_frames - 1, output_start,
                output_start + n_frames - 1,
            )
        )
    return segments


def _ffmpeg(ffmpeg: str) -> str:
    if Path(ffmpeg).is_file():
        return str(Path(ffmpeg).resolve())
    binary = shutil.which(ffmpeg)
    if not binary:
        raise RuntimeError(f"ffmpeg not found: {ffmpeg}")
    return binary


def _run(command: list[str], label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed: {result.stderr.strip()}")


def _filter_graph(segments: list[TimelineSegment]) -> str:
    parts: list[str] = []
    labels: list[str] = []
    for segment in segments:
        label = f"v{segment.segment_id}"
        labels.append(f"[{label}]")
        parts.append(
            f"[0:v]trim=start_frame={segment.source_start_frame}:"
            f"end_frame={segment.source_end_frame + 1},setpts=PTS-STARTPTS[{label}]"
        )
    parts.append("[v0]null[outv]" if len(segments) == 1 else f"{''.join(labels)}concat=n={len(segments)}:v=1:a=0[outv]")
    return ";".join(parts)


def trim_original(
    *,
    original_video: Path,
    video_id: str,
    cut_list: Path,
    output_video: Path,
    manifest: Path,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    """Create a private, audio-free curated child from an original video."""
    original_video = original_video.expanduser().resolve()
    output_video = output_video.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if not original_video.is_file():
        raise FileNotFoundError(original_video)
    _require_private(output_video, "--output-video")
    _require_private(manifest, "--manifest")
    if output_video.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite a curated-original output or manifest")
    source = probe(original_video)
    if source.is_vfr:
        raise ValueError("frame-exact cuts require a constant-frame-rate source video")
    cuts = normalize_cut_ranges(parse_cut_ranges(cut_list), source.n_frames)
    segments = build_segments(source.n_frames, cuts)
    expected_frames = sum(segment.n_frames for segment in segments)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    binary = _ffmpeg(ffmpeg)
    with tempfile.TemporaryDirectory(prefix=".curate-original-", dir=output_video.parent) as tmp_dir:
        partial = Path(tmp_dir) / output_video.name
        _run(
            [
                binary, "-n", "-hide_banner", "-loglevel", "error", "-i", str(original_video),
                "-filter_complex", _filter_graph(segments), "-map", "[outv]", "-an", "-map_metadata", "-1",
                "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-fps_mode", "cfr", str(partial),
            ],
            "FFmpeg original-video trim",
        )
        child = probe(partial)
        if child.is_vfr or child.n_frames != expected_frames:
            raise RuntimeError("curated child does not have the expected constant-frame-rate frame count")
        if (child.width, child.height) != (source.width, source.height):
            raise RuntimeError("curated child geometry differs from the source")
        _run([binary, "-v", "error", "-i", str(partial), "-map", "0:v:0", "-f", "null", "-"], "child decode")
        partial.replace(output_video)
    payload = {
        "schema_version": 1,
        "artifact_type": "private_original_frame_curation",
        "privacy": "private_do_not_ship_or_upload",
        "video_id": video_id,
        "audio": "removed",
        "source_video": _binding(original_video),
        "cut_list": _binding(cut_list.expanduser().resolve()),
        "cut_ranges": [asdict(item) for item in cuts],
        "segments": [asdict(item) for item in segments],
        "output_video": _binding(output_video),
        "source": asdict(source),
        "output": asdict(child),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def normalize_face_free_original(
    *,
    original_video: Path,
    video_id: str,
    reviewer: str,
    output_video: Path,
    manifest: Path,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    """Make a private, no-cut 29.97-fps child after a named face-free review.

    The normalisation is not cosmetic.  The hand track is deliberately stored
    at about 30 samples/sec, so a 120-fps input needs a 30-fps child for the
    stored hand row index to remain aligned with every rendered video frame.
    """
    original_video = original_video.expanduser().resolve()
    output_video = output_video.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    reviewer = reviewer.strip()
    if not reviewer or "\n" in reviewer or "\r" in reviewer:
        raise ValueError("--reviewer must be a non-empty single line")
    if not original_video.is_file():
        raise FileNotFoundError(original_video)
    _require_private(output_video, "--output-video")
    _require_private(manifest, "--manifest")
    if output_video.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite a face-free child or manifest")
    source = probe(original_video)
    if source.is_vfr:
        raise ValueError("face-free normalisation requires a constant-frame-rate source video")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    binary = _ffmpeg(ffmpeg)
    with tempfile.TemporaryDirectory(prefix=".face-free-normalize-", dir=output_video.parent) as tmp_dir:
        partial = Path(tmp_dir) / output_video.name
        _run(
            [
                binary, "-n", "-hide_banner", "-loglevel", "error", "-i", str(original_video),
                "-map", "0:v:0", "-an", "-map_metadata", "-1", "-vf", f"fps={_FACE_FREE_NORMALIZED_FPS}",
                "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-fps_mode", "cfr", str(partial),
            ],
            "FFmpeg face-free video normalisation",
        )
        child = probe(partial)
        if child.is_vfr or child.n_frames < 1:
            raise RuntimeError("face-free child is not a usable constant-frame-rate video")
        if (child.width, child.height) != (source.width, source.height):
            raise RuntimeError("face-free child geometry differs from the source")
        _run([binary, "-v", "error", "-i", str(partial), "-map", "0:v:0", "-f", "null", "-"], "child decode")
        partial.replace(output_video)
    segment = TimelineSegment(
        segment_id=0,
        source_start_frame=0,
        source_end_frame=source.n_frames - 1,
        output_start_frame=0,
        output_end_frame=child.n_frames - 1,
    )
    payload = {
        "schema_version": 2,
        "artifact_type": "private_face_free_normalization",
        "privacy": "private_do_not_ship_or_upload",
        "video_id": video_id,
        "audio": "removed",
        "face_review": {
            "decision": "manually_reviewed_face_free",
            "reviewer": reviewer,
            "scope": (
                "Named human confirmation that this source has no visible human faces; "
                "no redaction or frame cuts were applied."
            ),
        },
        "normalization": {
            "target_fps": _FACE_FREE_NORMALIZED_FPS,
            "reason": "align one MediaPipe hand row with one rendered video frame",
        },
        "source_video": _binding(original_video),
        "cut_ranges": [],
        "segments": [asdict(segment)],
        "output_video": _binding(output_video),
        "source": asdict(source),
        "output": asdict(child),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_manifest(
    path: Path,
    *,
    output_video: Path | None = None,
    verify_source: bool = True,
) -> dict[str, Any]:
    """Validate private bindings and canonical output timeline metadata.

    Rendering an already materialized private child does not need the original
    source bytes.  ``verify_source=False`` is therefore limited to that
    read-only rendering path, so an operator may archive the source after the
    child and its hash-bound annotations have been created.  The child output
    binding is always verified.
    """
    path = path.expanduser().resolve()
    _require_private(path, "timeline manifest")
    data = json.loads(path.read_text(encoding="utf-8"))
    artifact_type = data.get("artifact_type")
    schema_version = data.get("schema_version")
    if (schema_version, artifact_type) not in {
        (1, "private_original_frame_curation"),
        (2, "private_face_free_normalization"),
    }:
        raise ValueError("invalid curated-original manifest")
    source_value = data.get("source_video")
    if verify_source:
        source = _verify_binding(source_value, "source video")
    else:
        if not isinstance(source_value, dict) or not isinstance(source_value.get("path"), str):
            raise ValueError("curated-original manifest source video binding is invalid")
        source = Path(source_value["path"]).expanduser().resolve()
    output = _verify_binding(data.get("output_video"), "output video")
    if output_video is not None and output != output_video.expanduser().resolve():
        raise ValueError("timeline manifest belongs to another curated child")
    actual = [TimelineSegment(**item) for item in data.get("segments", [])]
    if schema_version == 1:
        _verify_binding(data.get("cut_list"), "cut list")
        source_frames = int((data.get("source") or {}).get("n_frames", 0))
        cuts = [CutRange(**item) for item in data.get("cut_ranges", [])]
        expected = build_segments(source_frames, cuts)
        if actual != expected:
            raise ValueError("timeline manifest segments do not match its cut ranges")
    else:
        review = data.get("face_review")
        if not isinstance(review, dict) or review.get("decision") != "manually_reviewed_face_free":
            raise ValueError("face-free timeline is missing its named human review decision")
        if len(actual) != 1 or actual[0].segment_id != 0 or actual[0].output_start_frame != 0:
            raise ValueError("face-free timeline must contain exactly one output segment")
        expected = actual
    if probe(output).n_frames != sum(item.n_frames for item in expected):
        raise ValueError("curated child frame count does not match its timeline")
    data["_source_video"] = str(source)
    data["_output_video"] = str(output)
    return data


def _draw_hand(frame, landmarks: list[float] | None, color: tuple[int, int, int]) -> None:
    """Draw one stored 21-point MediaPipe hand without running inference."""
    if not landmarks or len(landmarks) != 63:
        return
    height, width = frame.shape[:2]
    points: list[tuple[int, int]] = []
    for idx in range(21):
        x, y = landmarks[idx * 3], landmarks[idx * 3 + 1]
        points.append((round(x * (width - 1)), round(y * (height - 1))))
    for start, end in _HAND_EDGES:
        cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)


def _load_render_context(
    *,
    curated_video: Path,
    timeline_manifest: Path,
    run_dir: Path,
    video_id: str,
) -> tuple[Path, list[TimelineSegment], dict[int, dict[int, dict[str, Any]]], dict[str, Any], Path]:
    curated_video = curated_video.expanduser().resolve()
    # The retained child, its manifest, and the stored hand/caption artifacts
    # are all hash-bound.  Rendering uses those artifacts only, so it remains
    # valid after the original source has been archived off-device.
    timeline = load_manifest(
        timeline_manifest,
        output_video=curated_video,
        verify_source=False,
    )
    if timeline.get("video_id") != video_id:
        raise ValueError("timeline manifest belongs to a different video_id")
    root = run_dir.expanduser().resolve() / "private" / "original_curated"
    summary_path = root / "runs" / f"{video_id}.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"curated MediaPipe run summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("workflow") != "private_original_trim_mediapipe_vlm":
        raise ValueError("unexpected curated annotation summary")

    segments = [TimelineSegment(**row) for row in timeline["segments"]]
    records_by_segment = {
        int(record["segment_id"]): record
        for record in summary.get("segments", [])
        if isinstance(record, dict) and "segment_id" in record
    }

    # A caption-only pilot may update the run summary after MediaPipe has
    # already completed.  The annotations themselves live at this canonical
    # private path, so recover them from there rather than making the user
    # rerun the expensive 30 fps hand pass.  A recorded path, when available,
    # is still honoured and validated first.
    stored_by_segment: dict[int, dict[int, dict[str, Any]]] = {}
    for segment in segments:
        record = records_by_segment.get(segment.segment_id, {})
        hands = record.get("hands") or {}
        raw_path = hands.get("path") if isinstance(hands, dict) else None
        hand_path = (
            Path(raw_path).expanduser().resolve()
            if isinstance(raw_path, str)
            else root / "hands" / video_id / f"segment_{segment.segment_id:04d}.parquet"
        )
        _require_private(hand_path, "stored hand annotations")
        if not hand_path.is_file():
            raise FileNotFoundError(
                "stored MediaPipe hand annotations missing for "
                f"segment {segment.segment_id}: {hand_path}"
            )
        stored_by_segment[segment.segment_id] = {
            int(row["frame_idx"]): row for row in pq.read_table(hand_path).to_pylist()
        }

    expected_ids = {segment.segment_id for segment in segments}
    if set(stored_by_segment) != expected_ids:
        raise ValueError("stored MediaPipe segments do not match the curated timeline")
    return curated_video, segments, stored_by_segment, summary, root


def _caption_rows_by_segment(
    *, root: Path, segments: list[TimelineSegment], video_id: str, model_id: str
) -> dict[int, list[dict[str, Any]]]:
    """Load only complete, successful VLM windows for every retained segment."""
    db_path = root / "annotations.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"caption database not found: {db_path}")
    output: dict[int, list[dict[str, Any]]] = {}
    with sqlite3.connect(db_path) as conn:
        for segment in segments:
            segment_id = segment.segment_id
            scoped_id = f"{video_id}.segment.{segment_id:04d}"
            rows = conn.execute(
                "SELECT unit_idx, payload, error FROM annotations "
                "WHERE video_id=? AND stage='caption' AND model_id=? ORDER BY unit_idx",
                (scoped_id, model_id),
            ).fetchall()
            if any(error is not None for _idx, _payload, error in rows):
                raise ValueError(f"caption run has failed window(s) in segment {segment_id}")
            frame_marker = root / "caption_frames" / scoped_id / "_COMPLETE.json"
            if not frame_marker.is_file():
                raise FileNotFoundError(f"caption frame cache marker missing: {frame_marker}")
            frame_count = int(json.loads(frame_marker.read_text(encoding="utf-8"))["frame_count"])
            expected_windows = (frame_count + config.VLM_WINDOW_FRAMES - 1) // config.VLM_WINDOW_FRAMES
            by_index = {int(index): json.loads(payload) for index, payload, _error in rows}
            if set(by_index) != set(range(expected_windows)):
                raise ValueError(
                    f"caption run is incomplete for segment {segment_id}: "
                    f"have {len(by_index)}/{expected_windows} successful windows"
                )
            output[segment_id] = [by_index[index] for index in range(expected_windows)]
    return output


def _event_timelines_by_segment(
    *, root: Path, segments: list[TimelineSegment], video_id: str, model_id: str
) -> dict[int, dict[str, Any]]:
    """Load complete derived events, one independently bounded timeline per cut."""
    return {
        segment.segment_id: curated_caption_events.load_event_timeline(
            curated_caption_events.event_timeline_path(
                root=root,
                video_id=video_id,
                model_id=model_id,
                segment_id=segment.segment_id,
            ),
            video_id=video_id,
            segment_id=segment.segment_id,
            model_id=model_id,
        )
        for segment in segments
    }


def _caption_pilot_window(
    *, root: Path, video_id: str, segment_id: int, model_id: str, window_idx: int
) -> dict[str, Any]:
    """Load one successful caption window for a clearly labelled private preview."""
    db_path = root / "annotations.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"caption database not found: {db_path}")
    scoped_id = f"{video_id}.segment.{segment_id:04d}"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload, error FROM annotations "
            "WHERE video_id=? AND stage='caption' AND model_id=? AND unit_idx=?",
            (scoped_id, model_id, window_idx),
        ).fetchone()
    if row is None:
        raise FileNotFoundError(
            f"caption pilot window {window_idx} not found for segment {segment_id} and model {model_id}"
        )
    payload, error = row
    if error is not None:
        raise ValueError(f"caption pilot window {window_idx} failed: {error}")
    if not isinstance(payload, str):
        raise ValueError(f"caption pilot window {window_idx} has no JSON payload")
    return json.loads(payload)


def _caption_lines(captions: list[dict[str, Any]], timestamp_ms: int) -> list[tuple[str, tuple[int, int, int]]]:
    """Choose the activity and relevant atomic hand action at one timestamp."""
    caption = next(
        (
            item
            for item in captions
            if int(item["start_ts_ms"]) <= timestamp_ms < int(item["end_ts_ms"])
        ),
        None,
    )
    if caption is None:
        return [("Activity: no VLM window at this timestamp", (220, 220, 220))]
    activity = caption.get("activity") or {}
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if isinstance(activity.get("caption"), str) and activity["caption"]:
        lines.append((f"Activity: {activity['caption']}", (255, 255, 255)))
    sampled_ms = 1000 / float(config.VLM_FPS)
    action_start = int(caption["start_ts_ms"])
    action = next(
        (
            item
            for item in caption.get("actions", [])
            if action_start + round(int(item["start_frame"]) * sampled_ms) <= timestamp_ms
            < action_start + round((int(item["end_frame"]) + 1) * sampled_ms)
        ),
        None,
    )
    if not isinstance(action, dict):
        return lines or [("Activity: caption has no visible atomic action", (220, 220, 220))]
    if isinstance(action.get("action_caption"), str) and action["action_caption"]:
        lines.append((f"Action: {action['action_caption']}", (255, 255, 255)))
    for label, key, color in (
        ("Left", "left_hand", _LEFT_COLOR),
        ("Right", "right_hand", _RIGHT_COLOR),
    ):
        hand = action.get(key) or {}
        text = hand.get("caption") if isinstance(hand, dict) else None
        if isinstance(text, str) and text:
            lines.append((f"{label}: {text}", color))
    return lines


def _event_lines(timeline: dict[str, Any], timestamp_ms: int) -> list[tuple[str, tuple[int, int, int]]]:
    """Choose a source-grounded event and per-hand captions for the overlay."""
    summary = (timeline.get("summary") or {}).get("text")
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if isinstance(summary, str) and summary:
        lines.append((f"Segment: {summary}", (255, 255, 255)))
    events = timeline.get("events") or []
    active = [
        event
        for event in events
        if isinstance(event, dict)
        and int(event.get("start_ts_ms", 0)) <= timestamp_ms < int(event.get("end_ts_ms", 0))
    ]
    if not active:
        return lines or [("Activity: no caption event at this timestamp", (220, 220, 220))]
    # Raw VLM actions should be ordered and non-overlapping.  If a model did
    # emit an overlap, prefer the more recently started action rather than
    # hide it behind an earlier broad action.
    event = max(active, key=lambda item: (int(item["start_ts_ms"]), int(item["event_idx"])))
    action_caption = event.get("action_caption")
    if isinstance(action_caption, str) and action_caption:
        lines.append((f"Action: {action_caption}", (255, 255, 255)))
    for label, key, color in (("Left", "left_hand", _LEFT_COLOR), ("Right", "right_hand", _RIGHT_COLOR)):
        hand = event.get(key) or {}
        text = hand.get("caption") if isinstance(hand, dict) else None
        if isinstance(text, str) and text:
            lines.append((f"{label}: {text}", color))
    return lines or [("Activity: event has no visible hand caption", (220, 220, 220))]


def _wrap_caption(text: str, width: int, scale: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_caption_panel(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    if not lines:
        return
    height, width = frame.shape[:2]
    scale = max(0.45, min(0.7, width / 2200))
    rendered = [(line, color) for text, color in lines for line in _wrap_caption(text, width - 40, scale)]
    line_height = round(30 * scale / 0.7)
    panel_height = min(height // 2, 18 + line_height * len(rendered))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    y = height - panel_height + line_height
    for text, color in rendered:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_height


def _render_overlay_video(
    *,
    curated_video: Path,
    segments: list[TimelineSegment],
    stored_by_segment: dict[int, dict[int, dict[str, Any]]],
    video_id: str,
    output_video: Path,
    captions_by_segment: dict[int, list[dict[str, Any]]] | None,
    events_by_segment: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_video = output_video.expanduser().resolve()
    _require_private(output_video, "--output-video")
    if output_video.exists():
        raise FileExistsError(f"refusing to overwrite annotated preview: {output_video}")

    info = probe(curated_video)
    capture = cv2.VideoCapture(str(curated_video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode curated video for hand preview: {curated_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hand-preview-", dir=output_video.parent) as tmp_dir:
        partial = Path(tmp_dir) / output_video.name
        writer = cv2.VideoWriter(
            str(partial), cv2.VideoWriter_fourcc(*"mp4v"), float(info.fps), (info.width, info.height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("cannot open MP4 preview encoder")
        frame_idx = 0
        segment_idx = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                while segment_idx + 1 < len(segments) and frame_idx > segments[segment_idx].output_end_frame:
                    segment_idx += 1
                segment = segments[segment_idx]
                relative_idx = frame_idx - segment.output_start_frame
                row = stored_by_segment[segment.segment_id].get(relative_idx)
                if row is not None:
                    _draw_hand(frame, row.get("left_landmarks"), _LEFT_COLOR)
                    _draw_hand(frame, row.get("right_landmarks"), _RIGHT_COLOR)
                if captions_by_segment is not None:
                    timestamp_ms = round(relative_idx * 1000 / float(info.fps))
                    _draw_caption_panel(
                        frame, _caption_lines(captions_by_segment[segment.segment_id], timestamp_ms)
                    )
                elif events_by_segment is not None:
                    timestamp_ms = round(relative_idx * 1000 / float(info.fps))
                    _draw_caption_panel(
                        frame, _event_lines(events_by_segment[segment.segment_id], timestamp_ms)
                    )
                label = f"SEGMENT {segment.segment_id:04d}  |  green: wearer left  pink: wearer right"
                cv2.putText(frame, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
                if frame_idx == segment.output_start_frame and segment.segment_id:
                    cv2.putText(frame, "CUT: TRACK STATE RESET", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                writer.write(frame)
                frame_idx += 1
        finally:
            capture.release()
            writer.release()
        if frame_idx != info.n_frames:
            raise RuntimeError(f"preview decoded {frame_idx}/{info.n_frames} curated frames")
        preview = probe(partial)
        if preview.n_frames != info.n_frames or (preview.width, preview.height) != (info.width, info.height):
            raise RuntimeError("hand preview does not match the curated video geometry or frame count")
        partial.replace(output_video)
    return {
        "video_id": video_id,
        "output_video": _binding(output_video),
        "source_video": _binding(curated_video),
        "n_segments": len(segments),
        "frames": info.n_frames,
        "privacy": "private_do_not_ship_or_upload",
    }


def render_hand_preview(
    *,
    curated_video: Path,
    timeline_manifest: Path,
    run_dir: Path,
    video_id: str,
    output_video: Path,
) -> dict[str, Any]:
    """Render stored per-segment hands only into a private review MP4."""
    _require_private(output_video.expanduser().resolve(), "--output-video")
    curated_video, segments, stored, _summary, _root = _load_render_context(
        curated_video=curated_video,
        timeline_manifest=timeline_manifest,
        run_dir=run_dir,
        video_id=video_id,
    )
    return _render_overlay_video(
        curated_video=curated_video,
        segments=segments,
        stored_by_segment=stored,
        video_id=video_id,
        output_video=output_video,
        captions_by_segment=None,
    )


def render_annotated_video(
    *,
    curated_video: Path,
    timeline_manifest: Path,
    run_dir: Path,
    video_id: str,
    model_id: str,
    output_video: Path,
) -> dict[str, Any]:
    """Burn stored MediaPipe hands and dense VLM captions into one private MP4."""
    _require_private(output_video.expanduser().resolve(), "--output-video")
    curated_video, segments, stored, _summary, root = _load_render_context(
        curated_video=curated_video,
        timeline_manifest=timeline_manifest,
        run_dir=run_dir,
        video_id=video_id,
    )
    events = _event_timelines_by_segment(
        root=root,
        segments=segments,
        video_id=video_id,
        model_id=model_id,
    )
    return _render_overlay_video(
        curated_video=curated_video,
        segments=segments,
        stored_by_segment=stored,
        video_id=video_id,
        output_video=output_video,
        captions_by_segment=None,
        events_by_segment=events,
    )


def render_caption_segment_pilot(
    *,
    curated_video: Path,
    timeline_manifest: Path,
    run_dir: Path,
    video_id: str,
    model_id: str,
    segment_id: int,
    output_video: Path,
) -> dict[str, Any]:
    """Render a complete retained segment with its event timeline and hands.

    This is a review pilot only: it cannot span a manual cut and is labelled
    as such.  It is the appropriate visual check before using the full-video
    renderer, which requires event timelines for every retained segment.
    """
    _require_private(output_video.expanduser().resolve(), "--output-video")
    curated_video, segments, stored, _summary, root = _load_render_context(
        curated_video=curated_video,
        timeline_manifest=timeline_manifest,
        run_dir=run_dir,
        video_id=video_id,
    )
    segment = next((item for item in segments if item.segment_id == segment_id), None)
    if segment is None:
        raise ValueError(f"segment {segment_id} is not in the curated timeline")
    timeline = curated_caption_events.load_event_timeline(
        curated_caption_events.event_timeline_path(
            root=root, video_id=video_id, model_id=model_id, segment_id=segment_id
        ),
        video_id=video_id,
        segment_id=segment_id,
        model_id=model_id,
    )
    output_video = output_video.expanduser().resolve()
    if output_video.exists():
        raise FileExistsError(f"refusing to overwrite caption segment pilot: {output_video}")
    info = probe(curated_video)
    capture = cv2.VideoCapture(str(curated_video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode curated video for caption segment pilot: {curated_video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, segment.output_start_frame)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".caption-segment-pilot-", dir=output_video.parent) as tmp_dir:
        partial = Path(tmp_dir) / output_video.name
        writer = cv2.VideoWriter(
            str(partial), cv2.VideoWriter_fourcc(*"mp4v"), float(info.fps), (info.width, info.height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("cannot open MP4 caption-segment-pilot encoder")
        emitted = 0
        try:
            for relative_idx in range(segment.n_frames):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("caption segment-pilot decode ended before its requested segment")
                hand_row = stored[segment_id].get(relative_idx)
                if hand_row is not None:
                    _draw_hand(frame, hand_row.get("left_landmarks"), _LEFT_COLOR)
                    _draw_hand(frame, hand_row.get("right_landmarks"), _RIGHT_COLOR)
                timestamp_ms = round(relative_idx * 1000 / float(info.fps))
                _draw_caption_panel(frame, _event_lines(timeline, timestamp_ms))
                label = (
                    f"PILOT ONLY | segment {segment_id:04d} event timeline "
                    "| green: wearer left  pink: wearer right"
                )
                cv2.putText(frame, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
                writer.write(frame)
                emitted += 1
        finally:
            capture.release()
            writer.release()
        if emitted != segment.n_frames:
            raise RuntimeError(f"caption segment pilot rendered {emitted}/{segment.n_frames} expected frames")
        preview = probe(partial)
        if preview.n_frames != segment.n_frames or (preview.width, preview.height) != (info.width, info.height):
            raise RuntimeError("caption segment pilot does not match its requested frame range or geometry")
        partial.replace(output_video)
    return {
        "video_id": video_id,
        "segment_id": segment_id,
        "output_video": _binding(output_video),
        "source_video": _binding(curated_video),
        "frames": segment.n_frames,
        "privacy": "private_do_not_ship_or_upload",
    }


def render_caption_pilot(
    *,
    curated_video: Path,
    timeline_manifest: Path,
    run_dir: Path,
    video_id: str,
    model_id: str,
    segment_id: int,
    window_idx: int,
    output_video: Path,
) -> dict[str, Any]:
    """Burn one already-paid VLM window and its stored hands into a short private MP4.

    This intentionally cannot be mistaken for the complete annotated video:
    it renders only one requested window and labels it as a pilot.
    """
    _require_private(output_video.expanduser().resolve(), "--output-video")
    curated_video, segments, stored, _summary, root = _load_render_context(
        curated_video=curated_video,
        timeline_manifest=timeline_manifest,
        run_dir=run_dir,
        video_id=video_id,
    )
    segment = next((item for item in segments if item.segment_id == segment_id), None)
    if segment is None:
        raise ValueError(f"segment {segment_id} is not in the curated timeline")
    if window_idx < 0:
        raise ValueError("window_idx must be non-negative")
    caption = _caption_pilot_window(
        root=root,
        video_id=video_id,
        segment_id=segment_id,
        model_id=model_id,
        window_idx=window_idx,
    )
    start_ms = int(caption["start_ts_ms"])
    end_ms = int(caption["end_ts_ms"])
    if end_ms <= start_ms:
        raise ValueError("caption pilot window has an invalid timestamp range")
    info = probe(curated_video)
    start_in_segment = round(start_ms * float(info.fps) / 1000)
    end_in_segment = min(
        segment.n_frames - 1,
        round(end_ms * float(info.fps) / 1000) - 1,
    )
    if start_in_segment < 0 or start_in_segment > end_in_segment:
        raise ValueError("caption pilot window does not overlap its curated segment")
    output_video = output_video.expanduser().resolve()
    if output_video.exists():
        raise FileExistsError(f"refusing to overwrite caption pilot preview: {output_video}")

    # The caption schema's timestamps are segment-relative.  Rebase the one
    # selected window so the preview begins at zero without changing its
    # atomic action frame indexes.
    preview_caption = dict(caption)
    preview_caption["start_ts_ms"] = 0
    preview_caption["end_ts_ms"] = end_ms - start_ms
    expected_frames = end_in_segment - start_in_segment + 1
    capture = cv2.VideoCapture(str(curated_video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode curated video for caption pilot: {curated_video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, segment.output_start_frame + start_in_segment)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".caption-pilot-", dir=output_video.parent) as tmp_dir:
        partial = Path(tmp_dir) / output_video.name
        writer = cv2.VideoWriter(
            str(partial), cv2.VideoWriter_fourcc(*"mp4v"), float(info.fps), (info.width, info.height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("cannot open MP4 caption-pilot encoder")
        emitted = 0
        try:
            for relative_idx in range(start_in_segment, end_in_segment + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("caption-pilot decode ended before its requested window")
                hand_row = stored[segment_id].get(relative_idx)
                if hand_row is not None:
                    _draw_hand(frame, hand_row.get("left_landmarks"), _LEFT_COLOR)
                    _draw_hand(frame, hand_row.get("right_landmarks"), _RIGHT_COLOR)
                local_ms = round(emitted * 1000 / float(info.fps))
                _draw_caption_panel(frame, _caption_lines([preview_caption], local_ms))
                label = (
                    f"PILOT ONLY | segment {segment_id:04d}, caption window {window_idx:04d} "
                    "| green: wearer left  pink: wearer right"
                )
                cv2.putText(frame, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
                writer.write(frame)
                emitted += 1
        finally:
            capture.release()
            writer.release()
        if emitted != expected_frames:
            raise RuntimeError(f"caption pilot rendered {emitted}/{expected_frames} expected frames")
        preview = probe(partial)
        if preview.n_frames != expected_frames or (preview.width, preview.height) != (info.width, info.height):
            raise RuntimeError("caption pilot does not match its requested frame range or geometry")
        partial.replace(output_video)
    return {
        "video_id": video_id,
        "segment_id": segment_id,
        "window_idx": window_idx,
        "output_video": _binding(output_video),
        "source_video": _binding(curated_video),
        "frames": expected_frames,
        "privacy": "private_do_not_ship_or_upload",
    }


def materialize_segment(
    *,
    curated_video: Path,
    segment: TimelineSegment,
    output_video: Path,
    ffmpeg: str = "ffmpeg",
) -> None:
    """Write one private contiguous segment for an isolated annotation call."""
    binary = _ffmpeg(ffmpeg)
    _run(
        [
            binary, "-n", "-hide_banner", "-loglevel", "error", "-i", str(curated_video), "-filter_complex",
            f"[0:v]trim=start_frame={segment.output_start_frame}:end_frame={segment.output_end_frame + 1},"
            "setpts=PTS-STARTPTS[v]", "-map", "[v]", "-an", "-map_metadata", "-1", "-c:v", "libx264",
            "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", str(output_video),
        ],
        f"materialize segment {segment.segment_id}",
    )
