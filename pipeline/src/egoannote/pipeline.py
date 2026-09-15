"""The active annotation pipeline: curate an original video down to its
face-free segments, then run MediaPipe hands and dense VLM captioning on
what's retained.

Stage 2 segmentation is deliberately absent (see layers/segment.py's history
— removed as an unimplemented stub; the design needs real measurements from
this pipeline's own output before it can be built). The historical
EgoBlur-redacted-video annotate/publish/prepare-public-release commands, and
EgoBlur itself, have moved out of this module — see egoblur/ and
public-release-tools/ at the repo root, both parked but kept working as
public references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import config, curated_caption_events
from . import original_curated as original_curated_layer
from .archive import archive_files
from .backends.openai_compat import SpendTracker
from .backends.registry import build_backend
from .layers import caption as caption_layer
from .layers import hands as hands_layer
from .media.probe import probe
from .store import Store, write_hands_parquet_streaming

log = logging.getLogger("egoannote.pipeline")

_VIDEO_ID_RE = re.compile(r"^[\w.-]+$", flags=re.UNICODE)
_FACE_FREE_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2), encoding="utf-8")
    partial.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_video_id(video_id: str) -> str:
    """Keep a video identifier confined to its run directory and Drive prefix."""
    if (
        not video_id
        or video_id in {".", ".."}
        or not _VIDEO_ID_RE.fullmatch(video_id)
        or "/" in video_id
        or "\\" in video_id
    ):
        raise ValueError(
            "video_id must contain only letters, numbers, underscore, dot, or hyphen and "
            "must not be '.' or '..'"
        )
    return video_id


def _read_private_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "private" / "run_manifest.json"
    if path.exists():
        return _load_json(path)
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": _utc_now(),
        "stage2_segmentation": "paused_not_run",
        "videos": {},
    }


def _save_private_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _atomic_json(run_dir / "private" / "run_manifest.json", manifest)


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def archive_run(
    run_dir: Path,
    *,
    drive_root: str,
    delete_local_originals: bool = False,
    delete_local_redacted: bool = False,
) -> None:
    manifest = _read_private_manifest(run_dir)
    root = drive_root.rstrip("/")
    originals: list[tuple[Path, str]] = []
    redacted: list[tuple[Path, str]] = []
    for video_id, row in manifest["videos"].items():
        if row.get("original_video") and Path(row["original_video"]).is_file():
            source = Path(row["original_video"])
            originals.append((source, f"{root}/private/originals/{video_id}/{source.name}"))
        source = Path(row["redacted_video"])
        if source.is_file():
            redacted.append((source, f"{root}/redacted/{video_id}/{source.name}"))

    receipt_dir = run_dir / "private" / "drive_receipts"
    if originals:
        archive_files(
            originals,
            receipt_dir / "originals.json",
            delete_local_after_verify=delete_local_originals,
        )
    if redacted:
        if delete_local_redacted:
            hf_receipt_path = run_dir / "private" / "hf_upload_receipt.json"
            if not hf_receipt_path.exists():
                raise RuntimeError(
                    "redacted video deletion requires a successful Hugging Face upload receipt"
                )
            uploaded = set(_load_json(hf_receipt_path).get("video_ids", []))
            missing = sorted(set(manifest["videos"]) - uploaded)
            if missing:
                raise RuntimeError(
                    "redacted video deletion requires an upload receipt covering every "
                    f"current video; missing={missing}"
                )
        archive_files(
            redacted,
            receipt_dir / "redacted.json",
            delete_local_after_verify=delete_local_redacted,
        )


def annotate_curated_original(
    *,
    run_dir: Path,
    curated_video: Path,
    timeline_manifest: Path,
    video_id: str,
    model_ids: list[str],
    registry_path: Path | None,
    max_spend_per_model: float | None,
    workers: int,
    run_hands: bool,
    run_captions: bool,
    ffmpeg: str,
    hand_confidence: float = config.MP_MIN_HAND_CONFIDENCE,
    caption_windows: set[int] | None = None,
    caption_segments: set[int] | None = None,
) -> dict[str, Any]:
    """Run the intentionally private original-curated workflow.

    Each retained segment becomes an isolated MediaPipe/VLM input.  Results
    remain below ``run_dir/private/original_curated`` and this function never
    builds a Hugging Face bundle.
    """
    if not run_hands and not run_captions:
        raise ValueError("enable at least one of MediaPipe hands or VLM captions")
    if run_captions and not model_ids:
        raise ValueError("captioning requested but no --model was supplied")
    if not run_captions and (caption_windows is not None or caption_segments is not None):
        raise ValueError("--caption-windows and --caption-segments require captioning")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not 0.0 < hand_confidence <= 1.0:
        raise ValueError("hand_confidence must be in (0, 1]")
    video_id = _validate_video_id(video_id)
    curated_video = curated_video.expanduser().resolve()
    timeline = original_curated_layer.load_manifest(
        timeline_manifest, output_video=curated_video
    )
    if timeline.get("video_id") != video_id:
        raise ValueError("timeline manifest belongs to a different video_id")

    root = run_dir.resolve() / "private" / "original_curated"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "runs" / f"{video_id}.json"
    previous_records: dict[int, dict[str, Any]] = {}
    if summary_path.is_file():
        previous_summary = _load_json(summary_path)
        if (
            previous_summary.get("workflow") != "private_original_trim_mediapipe_vlm"
            or previous_summary.get("video_id") != video_id
        ):
            raise ValueError("existing curated annotation summary belongs to another workflow or video")
        previous_records = {
            int(record["segment_id"]): record
            for record in previous_summary.get("segments", [])
            if isinstance(record, dict) and "segment_id" in record
        }
    models_dir = root / "models"
    hands_dir = root / "hands" / video_id
    frames_dir = root / "caption_frames"
    db_path = root / "annotations.db"
    all_segments = [
        original_curated_layer.TimelineSegment(**row) for row in timeline["segments"]
    ]
    available_segment_ids = {segment.segment_id for segment in all_segments}
    all_records: dict[int, dict[str, Any]] = {}
    for segment in all_segments:
        record = {
            "segment_id": segment.segment_id,
            "output_start_frame": segment.output_start_frame,
            "output_end_frame": segment.output_end_frame,
            "annotation_video_id": f"{video_id}.segment.{segment.segment_id:04d}",
        }
        prior = previous_records.get(segment.segment_id, {})
        if isinstance(prior.get("hands"), dict):
            record["hands"] = prior["hands"]
        if isinstance(prior.get("captions"), dict):
            # Keep derived event-sidecar provenance while a later selected
            # caption run updates another segment.  The raw SQLite captions
            # remain authoritative; this is only a convenient run index.
            record["captions"] = dict(prior["captions"])
        all_records[segment.segment_id] = record
    if caption_segments is not None:
        unknown_segment_ids = caption_segments - available_segment_ids
        if unknown_segment_ids:
            raise ValueError(
                f"requested caption segment(s) {sorted(unknown_segment_ids)} do not exist; "
                f"available IDs are {sorted(available_segment_ids)}"
            )
        segments = [segment for segment in all_segments if segment.segment_id in caption_segments]
    else:
        segments = all_segments
    # One budget must cover the complete curated video. The VLM backend is
    # rebuilt per segment to make every cut a hard temporal reset, so the
    # tracker is explicitly shared across those isolated backend instances.
    spend_trackers = {model_id: SpendTracker() for model_id in model_ids}

    with tempfile.TemporaryDirectory(prefix=f".{video_id}.segments-", dir=root) as temp_dir:
        temporary_root = Path(temp_dir)
        with Store(db_path) as store:
            for segment in segments:
                segment_video = temporary_root / f"segment_{segment.segment_id:04d}.mp4"
                original_curated_layer.materialize_segment(
                    curated_video=curated_video,
                    segment=segment,
                    output_video=segment_video,
                    ffmpeg=ffmpeg,
                )
                scoped_id = f"{video_id}.segment.{segment.segment_id:04d}"
                record = all_records[segment.segment_id]
                if run_hands:
                    hands_path = hands_dir / f"segment_{segment.segment_id:04d}.parquet"
                    if hands_path.exists():
                        raise FileExistsError(
                            f"refusing to overwrite existing private hand annotations: {hands_path}"
                        )
                    segment_info = probe(segment_video)
                    rows, gaps = write_hands_parquet_streaming(
                        hands_layer.run(
                            segment_video,
                            scoped_id,
                            models_dir,
                            info=segment_info,
                            hand_confidence=hand_confidence,
                        ),
                        hands_path,
                    )
                    record["hands"] = {
                        "path": str(hands_path),
                        "rows": rows,
                        "missing_gaps": len(gaps),
                        "fps": config.MP_FPS,
                        "detection_and_presence_confidence": hand_confidence,
                        "tracking_confidence": config.MP_MIN_TRACKING_CONFIDENCE,
                    }
                if run_captions:
                    model_counts: dict[str, int] = {}
                    event_timelines: dict[str, dict[str, Any]] = {}
                    for model_id in model_ids:
                        backend = build_backend(
                            model_id,
                            registry_path=registry_path,
                            max_spend_usd=max_spend_per_model,
                            spend_tracker=spend_trackers[model_id],
                        )
                        try:
                            model_counts[model_id] = caption_layer.caption_video(
                                segment_video,
                                scoped_id,
                                backend,
                                store,
                                run_id=run_dir.name,
                                frames_dir=frames_dir,
                                max_workers=workers,
                                window_indices=caption_windows,
                            )
                            # A selected-window run is intentionally an
                            # incomplete diagnostic and must never create a
                            # segment summary or event timeline.  Full raw
                            # windows only are eligible for consolidation.
                            if caption_windows is None:
                                captions = original_curated_layer._caption_rows_by_segment(
                                    root=root,
                                    segments=[segment],
                                    video_id=video_id,
                                    model_id=model_id,
                                )[segment.segment_id]
                                segment_summary, summary_called = (
                                    caption_layer.summarize_caption_windows(
                                        captions,
                                        backend=backend,
                                        store=store,
                                        video_id=scoped_id,
                                        run_id=run_dir.name,
                                    )
                                )
                                event_timeline = curated_caption_events.compile_event_timeline(
                                    video_id=video_id,
                                    segment_id=segment.segment_id,
                                    model_id=model_id,
                                    captions=captions,
                                    summary=segment_summary,
                                )
                                event_path = curated_caption_events.event_timeline_path(
                                    root=root,
                                    video_id=video_id,
                                    model_id=model_id,
                                    segment_id=segment.segment_id,
                                )
                                curated_caption_events.write_event_timeline(event_path, event_timeline)
                                event_timelines[model_id] = {
                                    "path": str(event_path),
                                    "source_caption_sha256": event_timeline["source_caption_sha256"],
                                    "summary_api_called": summary_called,
                                    "summary_cost_usd": segment_summary.get("cost_usd"),
                                }
                        finally:
                            close = getattr(backend, "close", None)
                            if close is not None:
                                close()
                    caption_record = dict(record.get("captions") or {})
                    previous_counts = dict(caption_record.get("windows_written") or {})
                    previous_counts.update(model_counts)
                    caption_record["windows_written"] = previous_counts
                    if event_timelines:
                        previous_events = dict(caption_record.get("event_timelines") or {})
                        previous_events.update(event_timelines)
                        caption_record["event_timelines"] = previous_events
                    record["captions"] = caption_record

    summary = {
        "schema_version": 1,
        "workflow": "private_original_trim_mediapipe_vlm",
        "privacy": "private_do_not_ship_or_upload",
        "video_id": video_id,
        "curated_video": _file_binding(curated_video, label="curated original video"),
        "timeline_manifest": _file_binding(timeline_manifest, label="timeline manifest"),
        "caption_selection": {
            "segment_ids": sorted(caption_segments) if caption_segments is not None else None,
            "window_indices_per_segment": (
                sorted(caption_windows) if caption_windows is not None else None
            ),
        },
        "segments": [all_records[segment.segment_id] for segment in all_segments],
    }
    _atomic_json(summary_path, summary)
    return summary


def _completed_face_free_hands(*, run_dir: Path, video_id: str, child: Path) -> bool:
    """Return true only for a complete, hash-bound one-segment hand result."""
    root = run_dir.resolve() / "private" / "original_curated"
    summary_path = root / "runs" / f"{video_id}.json"
    if not summary_path.is_file():
        return False
    summary = _load_json(summary_path)
    records = [
        item for item in summary.get("segments", [])
        if isinstance(item, dict) and item.get("segment_id") == 0
    ]
    if len(records) != 1 or not isinstance(records[0].get("hands"), dict):
        return False
    hands = records[0]["hands"]
    raw_path = hands.get("path")
    if not isinstance(raw_path, str):
        return False
    hand_path = Path(raw_path).expanduser().resolve()
    if not hand_path.is_file():
        return False
    expected_rows = probe(child).n_frames
    try:
        actual_rows = pq.ParquetFile(hand_path).metadata.num_rows
    except Exception as exc:
        raise RuntimeError(f"cannot read saved hand annotations for {video_id}: {hand_path}") from exc
    if int(hands.get("rows", -1)) != expected_rows or actual_rows != expected_rows:
        raise RuntimeError(
            f"saved hand annotations for {video_id} are incomplete ({actual_rows}/{expected_rows}); "
            "do not overwrite them automatically"
        )
    return True


def batch_face_free_hands(
    *,
    input_dir: Path,
    run_dir: Path,
    reviewer: str,
    hand_confidence: float,
    ffmpeg: str,
) -> dict[str, Any]:
    """Normalize a reviewed face-free folder and run resumable 0.4-style hands.

    It never invokes a VLM, redaction, publishing, or Drive upload.  Every
    derived child, timeline, model, parquet, and batch receipt is private.
    """
    input_dir = input_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    reviewer = reviewer.strip()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not reviewer or "\n" in reviewer or "\r" in reviewer:
        raise ValueError("--reviewer must be a non-empty single line")
    if not 0.0 < hand_confidence <= 1.0:
        raise ValueError("hand_confidence must be in (0, 1]")
    videos = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in _FACE_FREE_VIDEO_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if not videos:
        raise ValueError(f"no video files found in {input_dir}")
    video_ids = [_validate_video_id(path.stem) for path in videos]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("batch has duplicate video IDs after filename normalization")

    private_root = run_dir / "private"
    children_root = private_root / "face_free_children"
    receipt_path = private_root / "face_free_hands_batch.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "private_face_free_30fps_mediapipe_batch",
        "privacy": "private_do_not_ship_or_upload",
        "input_dir": str(input_dir),
        "reviewer": reviewer,
        "hand_confidence": hand_confidence,
        "videos": [],
    }
    for source, video_id in zip(videos, video_ids, strict=True):
        child_dir = children_root / video_id
        child = child_dir / f"{video_id}.normalized.mp4"
        timeline = child_dir / f"{video_id}.timeline.json"
        if child.exists() != timeline.exists():
            raise RuntimeError(
                f"face-free child state is partial for {video_id}; expected both child and timeline or neither"
            )
        if child.exists():
            original_curated_layer.load_manifest(timeline, output_video=child)
            normalization_status = "reused"
        else:
            original_curated_layer.normalize_face_free_original(
                original_video=source,
                video_id=video_id,
                reviewer=reviewer,
                output_video=child,
                manifest=timeline,
                ffmpeg=ffmpeg,
            )
            normalization_status = "created"

        if _completed_face_free_hands(run_dir=run_dir, video_id=video_id, child=child):
            hands_status = "reused"
        else:
            hand_path = run_dir / "private" / "original_curated" / "hands" / video_id / "segment_0000.parquet"
            if hand_path.exists():
                raise RuntimeError(
                    f"refusing to overwrite unverified saved hands for {video_id}: {hand_path}"
                )
            annotate_curated_original(
                run_dir=run_dir,
                curated_video=child,
                timeline_manifest=timeline,
                video_id=video_id,
                model_ids=[],
                registry_path=None,
                max_spend_per_model=None,
                workers=1,
                run_hands=True,
                run_captions=False,
                ffmpeg=ffmpeg,
                hand_confidence=hand_confidence,
            )
            hands_status = "created"
        result["videos"].append(
            {
                "video_id": video_id,
                "source_video": _file_binding(source, label="face-free source video"),
                "normalized_video": _file_binding(child, label="face-free normalized video"),
                "timeline_manifest": _file_binding(timeline, label="face-free timeline manifest"),
                "normalization": normalization_status,
                "hands": hands_status,
            }
        )
        _atomic_json(receipt_path, result)
    return result


def _parse_windows(value: str | None) -> set[int] | None:
    if value is None:
        return None
    try:
        indices = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "pilot windows must be comma-separated non-negative integers"
        ) from exc
    if not indices or min(indices) < 0:
        raise argparse.ArgumentTypeError("pilot windows must be comma-separated non-negative integers")
    return indices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    curate_original = sub.add_parser(
        "curate-original",
        help="private-only manual frame trimming of an original video; no redaction or publishing stages",
    )
    curate_original.add_argument("--original-video", type=Path, required=True)
    curate_original.add_argument("--video-id", required=True)
    curate_original.add_argument("--cut-list", type=Path, required=True)
    curate_original.add_argument("--output-video", type=Path, required=True)
    curate_original.add_argument("--manifest", type=Path, required=True)
    curate_original.add_argument("--ffmpeg", default="ffmpeg")

    annotate_original = sub.add_parser(
        "annotate-curated-original",
        help="private-only MediaPipe and VLM on independently materialized original-derived segments",
    )
    annotate_original.add_argument("--run-dir", type=Path, required=True)
    annotate_original.add_argument("--curated-video", type=Path, required=True)
    annotate_original.add_argument("--timeline-manifest", type=Path, required=True)
    annotate_original.add_argument("--video-id", required=True)
    annotate_original.add_argument("--model", action="append", default=[])
    annotate_original.add_argument("--models-toml", type=Path)
    annotate_original.add_argument("--max-spend-per-model", type=float, default=10.0)
    annotate_original.add_argument("--workers", type=int, default=1)
    annotate_original.add_argument(
        "--hand-confidence",
        type=float,
        default=config.MP_MIN_HAND_CONFIDENCE,
        help=(
            "MediaPipe detection and presence confidence for this private run "
            f"(default: {config.MP_MIN_HAND_CONFIDENCE})"
        ),
    )
    annotate_original.add_argument("--skip-hands", action="store_true")
    annotate_original.add_argument("--skip-captions", action="store_true")
    annotate_original.add_argument(
        "--caption-windows",
        type=_parse_windows,
        help="optional comma-separated VLM window indexes to run within each selected segment",
    )
    annotate_original.add_argument(
        "--caption-segments",
        type=_parse_windows,
        help="optional comma-separated retained timeline segment IDs to caption",
    )
    annotate_original.add_argument("--ffmpeg", default="ffmpeg")

    face_free_hands = sub.add_parser(
        "batch-face-free-hands",
        help="privately normalize a manually reviewed face-free folder to 30 fps and run MediaPipe hands",
    )
    face_free_hands.add_argument("--input-dir", type=Path, required=True)
    face_free_hands.add_argument("--run-dir", type=Path, required=True)
    face_free_hands.add_argument(
        "--reviewer",
        required=True,
        help="name of the person who manually confirmed every input is face-free",
    )
    face_free_hands.add_argument(
        "--hand-confidence",
        type=float,
        default=config.MP_MIN_HAND_CONFIDENCE,
    )
    face_free_hands.add_argument("--ffmpeg", default="ffmpeg")

    preview_hands = sub.add_parser(
        "preview-curated-hands",
        help="render a private overlay from stored segment-isolated MediaPipe hand results",
    )
    preview_hands.add_argument("--run-dir", type=Path, required=True)
    preview_hands.add_argument("--curated-video", type=Path, required=True)
    preview_hands.add_argument("--timeline-manifest", type=Path, required=True)
    preview_hands.add_argument("--video-id", required=True)
    preview_hands.add_argument("--output-video", type=Path, required=True)

    render_annotations = sub.add_parser(
        "render-curated-annotations",
        help="burn stored MediaPipe hands and dense VLM captions into a private MP4",
    )
    render_annotations.add_argument("--run-dir", type=Path, required=True)
    render_annotations.add_argument("--curated-video", type=Path, required=True)
    render_annotations.add_argument("--timeline-manifest", type=Path, required=True)
    render_annotations.add_argument("--video-id", required=True)
    render_annotations.add_argument("--model", required=True)
    render_annotations.add_argument("--output-video", type=Path, required=True)

    render_caption_pilot = sub.add_parser(
        "preview-curated-caption-pilot",
        help="render one private VLM pilot window with its stored MediaPipe hand overlay",
    )
    render_caption_pilot.add_argument("--run-dir", type=Path, required=True)
    render_caption_pilot.add_argument("--curated-video", type=Path, required=True)
    render_caption_pilot.add_argument("--timeline-manifest", type=Path, required=True)
    render_caption_pilot.add_argument("--video-id", required=True)
    render_caption_pilot.add_argument("--model", required=True)
    render_caption_pilot.add_argument("--segment-id", type=int, required=True)
    render_caption_pilot.add_argument("--window-idx", type=int, required=True)
    render_caption_pilot.add_argument("--output-video", type=Path, required=True)

    render_caption_segment = sub.add_parser(
        "preview-curated-caption-segment",
        help="render one complete private retained segment from its consolidated caption event timeline",
    )
    render_caption_segment.add_argument("--run-dir", type=Path, required=True)
    render_caption_segment.add_argument("--curated-video", type=Path, required=True)
    render_caption_segment.add_argument("--timeline-manifest", type=Path, required=True)
    render_caption_segment.add_argument("--video-id", required=True)
    render_caption_segment.add_argument("--model", required=True)
    render_caption_segment.add_argument("--segment-id", type=int, required=True)
    render_caption_segment.add_argument("--output-video", type=Path, required=True)

    archive = sub.add_parser("archive-drive", help="copy, verify, receipt, optionally delete")
    archive.add_argument("--run-dir", type=Path, required=True)
    archive.add_argument("--drive-root", required=True)
    archive.add_argument("--delete-local-originals", action="store_true")
    archive.add_argument("--delete-local-redacted", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "curate-original":
        result = original_curated_layer.trim_original(
            original_video=args.original_video,
            video_id=_validate_video_id(args.video_id),
            cut_list=args.cut_list,
            output_video=args.output_video,
            manifest=args.manifest,
            ffmpeg=args.ffmpeg,
        )
        print(
            f"complete: {result['video_id']} -> {args.output_video} "
            f"({result['output']['n_frames']} retained frames, {len(result['segments'])} segments)"
        )
        return 0

    if args.command == "annotate-curated-original":
        summary = annotate_curated_original(
            run_dir=args.run_dir,
            curated_video=args.curated_video,
            timeline_manifest=args.timeline_manifest,
            video_id=args.video_id,
            model_ids=args.model,
            registry_path=args.models_toml,
            max_spend_per_model=args.max_spend_per_model,
            workers=args.workers,
            run_hands=not args.skip_hands,
            run_captions=not args.skip_captions,
            ffmpeg=args.ffmpeg,
            hand_confidence=args.hand_confidence,
            caption_windows=args.caption_windows,
            caption_segments=args.caption_segments,
        )
        print(f"complete: {summary['video_id']} -> {args.run_dir}/private/original_curated")
        return 0

    if args.command == "batch-face-free-hands":
        result = batch_face_free_hands(
            input_dir=args.input_dir,
            run_dir=args.run_dir,
            reviewer=args.reviewer,
            hand_confidence=args.hand_confidence,
            ffmpeg=args.ffmpeg,
        )
        print(
            f"complete: {len(result['videos'])} face-free video(s) -> "
            f"{args.run_dir}/private/original_curated/hands"
        )
        return 0

    if args.command == "preview-curated-hands":
        preview = original_curated_layer.render_hand_preview(
            run_dir=args.run_dir,
            curated_video=args.curated_video,
            timeline_manifest=args.timeline_manifest,
            video_id=_validate_video_id(args.video_id),
            output_video=args.output_video,
        )
        print(f"complete: {preview['video_id']} -> {args.output_video}")
        return 0

    if args.command == "render-curated-annotations":
        preview = original_curated_layer.render_annotated_video(
            run_dir=args.run_dir,
            curated_video=args.curated_video,
            timeline_manifest=args.timeline_manifest,
            video_id=_validate_video_id(args.video_id),
            model_id=args.model,
            output_video=args.output_video,
        )
        print(f"complete: {preview['video_id']} -> {args.output_video}")
        return 0

    if args.command == "preview-curated-caption-pilot":
        preview = original_curated_layer.render_caption_pilot(
            run_dir=args.run_dir,
            curated_video=args.curated_video,
            timeline_manifest=args.timeline_manifest,
            video_id=_validate_video_id(args.video_id),
            model_id=args.model,
            segment_id=args.segment_id,
            window_idx=args.window_idx,
            output_video=args.output_video,
        )
        print(f"complete: {preview['video_id']} pilot -> {args.output_video}")
        return 0

    if args.command == "preview-curated-caption-segment":
        preview = original_curated_layer.render_caption_segment_pilot(
            run_dir=args.run_dir,
            curated_video=args.curated_video,
            timeline_manifest=args.timeline_manifest,
            video_id=_validate_video_id(args.video_id),
            model_id=args.model,
            segment_id=args.segment_id,
            output_video=args.output_video,
        )
        print(f"complete: {preview['video_id']} segment pilot -> {args.output_video}")
        return 0

    if args.command == "archive-drive":
        archive_run(
            args.run_dir.resolve(),
            drive_root=args.drive_root,
            delete_local_originals=args.delete_local_originals,
            delete_local_redacted=args.delete_local_redacted,
        )
        return 0

    raise AssertionError(f"unhandled command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
