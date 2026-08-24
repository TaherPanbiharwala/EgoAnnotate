"""Resumable local pipeline for MediaPipe, dense VLM captions, and shipping.

Stage 2 segmentation is deliberately absent. The only video accepted by the
annotation stages is ``redacted_video``; ``original_video`` is registered for
later private perception stages and Drive archiving but is never decoded by
MediaPipe or sent to a VLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import config
from . import pose_prior as pose_prior_layer
from .archive import archive_files
from .backends.registry import build_backend
from .layers import caption as caption_layer
from .layers import hands as hands_layer
from .media.probe import probe
from .pack.huggingface import build_video_bundle, install_public_license, upload_bundle
from .store import Store, write_hands_parquet_streaming
from .verify_yunet import create_decision_template, decisions_to_forced_boxes, verify_yunet

log = logging.getLogger("egoannote.pipeline")

_VIDEO_ID_RE = re.compile(r"^[\w.-]+$", flags=re.UNICODE)


@dataclass(slots=True)
class VideoInput:
    redacted_video: Path
    original_video: Path | None = None
    blur_manifest: Path | None = None
    video_id: str | None = None


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


def _load_batch(path: Path) -> list[VideoInput]:
    inputs: list[VideoInput] = []
    base = path.resolve().parent

    def resolve_from_manifest(value: str | None) -> Path | None:
        if not value:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            inputs.append(
                VideoInput(
                    redacted_video=resolve_from_manifest(row["redacted_video"]),
                    original_video=resolve_from_manifest(row.get("original_video")),
                    blur_manifest=resolve_from_manifest(row.get("blur_manifest")),
                    video_id=row.get("video_id"),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid batch manifest row {line_no}: {exc}") from exc
    if not inputs:
        raise ValueError(f"batch manifest {path} contains no video rows")
    return inputs


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


def _resolve_identity(item: VideoInput) -> tuple[str, dict[str, Any] | None, str]:
    blur = _load_json(item.blur_manifest) if item.blur_manifest else None
    if item.video_id:
        return _validate_video_id(item.video_id), blur, "explicit"
    if blur:
        source = blur.get("source") or {}
        clip_id = blur.get("clip_id")
        source_sha = source.get("sha256")
        if not clip_id or not source_sha:
            raise ValueError(
                f"blur manifest {item.blur_manifest} lacks clip_id or source.sha256"
            )
        return _validate_video_id(f"{clip_id}-{str(source_sha)[:8]}"), blur, "egoblur_manifest"
    if item.original_video and item.original_video.is_file():
        return _validate_video_id(
            config.video_id_from_content(item.original_video, item.original_video.stem)
        ), None, "original_content"

    stem = item.redacted_video.stem
    if stem.endswith(".blurred"):
        stem = stem.removesuffix(".blurred")
    return _validate_video_id(
        config.video_id_from_content(item.redacted_video, stem)
    ), None, "redacted_content"


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


def _register_video(
    run_dir: Path, item: VideoInput, video_id: str, blur: dict[str, Any] | None, basis: str
) -> dict[str, Any]:
    manifest = _read_private_manifest(run_dir)
    redacted = item.redacted_video.resolve()
    original = item.original_video.resolve() if item.original_video else None
    blur_status = (blur or {}).get("status", "UNKNOWN_NO_MANIFEST")
    if blur and not str(blur_status).startswith("PASS"):
        raise ValueError(
            f"EgoBlur manifest status for {video_id} is {blur_status!r}; annotation may "
            "run only from a reviewed PASS redaction"
        )

    row = manifest["videos"].setdefault(video_id, {})
    row.update(
        {
            "video_id": video_id,
            "identity_basis": basis,
            "redacted_video": str(redacted),
            "redacted_sha256": _sha256_file(redacted),
            "original_video": str(original) if original else None,
            "original_sha256": (
                (blur or {}).get("source", {}).get("sha256")
                if blur
                else (_sha256_file(original) if original and original.is_file() else None)
            ),
            "blur_manifest": str(item.blur_manifest.resolve()) if item.blur_manifest else None,
            "blur_status": blur_status,
            "annotation_input": "redacted_only",
            "future_original_stages": ["wilor", "sam2", "depth_v3"],
            "stages": row.get("stages", {}),
        }
    )
    _save_private_manifest(run_dir, manifest)
    return row


def _run_hands(item: VideoInput, video_id: str, run_dir: Path) -> tuple[Path, int, int]:
    work_dir = run_dir / "work" / video_id
    target = work_dir / "hands.parquet"
    info = probe(item.redacted_video)
    stride = max(1, round(info.fps / config.MP_FPS))
    expected_rows = hands_layer._expected_sample_count(info.n_frames, stride)
    if target.exists():
        actual_rows = pq.ParquetFile(target).metadata.num_rows
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"existing hand track {target} contains {actual_rows}/{expected_rows} "
                "expected sampled frames. It is incomplete and must not be resumed; "
                "use a newly regenerated redacted video and a fresh run directory."
            )
        return target, -1, -1
    partial = target.with_suffix(".parquet.partial")
    partial.unlink(missing_ok=True)
    rows, gaps = write_hands_parquet_streaming(
        hands_layer.run(
            item.redacted_video,
            video_id,
            run_dir / "private" / "models",
            info=info,
        ),
        partial,
    )
    partial.replace(target)
    return target, rows, len(gaps)


def _run_captions(
    item: VideoInput,
    video_id: str,
    run_dir: Path,
    *,
    model_ids: list[str],
    registry_path: Path | None,
    max_spend_per_model: float | None,
    workers: int,
    pilot_windows: set[int] | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    frames_dir = run_dir / "private" / "caption_frames"
    with Store(run_dir / "private" / "annotations.db") as store:
        for model_id in model_ids:
            backend = build_backend(
                model_id,
                registry_path=registry_path,
                max_spend_usd=max_spend_per_model,
            )
            try:
                counts[model_id] = caption_layer.caption_video(
                    item.redacted_video,
                    video_id,
                    backend,
                    store,
                    run_id=run_dir.name,
                    frames_dir=frames_dir,
                    max_workers=workers,
                    window_indices=pilot_windows,
                )
            finally:
                close = getattr(backend, "close", None)
                if close is not None:
                    close()
    return counts


def _expected_caption_windows(run_dir: Path, video_id: str) -> int:
    frame_dir = run_dir / "private" / "caption_frames" / video_id
    n_frames = len(list(frame_dir.glob("frame_*.jpg")))
    if n_frames == 0:
        raise RuntimeError(
            f"caption frame cache for {video_id} is missing after captioning; "
            "cannot prove the full caption window count"
        )
    return (n_frames + config.VLM_WINDOW_FRAMES - 1) // config.VLM_WINDOW_FRAMES


def _assert_captions_publishable(run_dir: Path, video_id: str, row: dict[str, Any]) -> None:
    captions = row.get("stages", {}).get("captions") or {}
    if captions.get("status") != "complete":
        raise RuntimeError(
            f"{video_id} captions have status={captions.get('status')!r}; refusing to "
            "publish a pilot or incomplete caption run"
        )
    model_ids = captions.get("models")
    expected_windows = captions.get("expected_windows")
    if not isinstance(model_ids, list) or not model_ids:
        raise RuntimeError(f"{video_id} has no recorded caption model IDs; rerun annotation first")
    if not isinstance(expected_windows, int) or expected_windows < 1:
        raise RuntimeError(
            f"{video_id} has no recorded expected caption-window count; rerun the full "
            "annotation command before publishing"
        )

    expected = set(range(expected_windows))
    with Store(run_dir / "private" / "annotations.db") as store:
        for model_id in model_ids:
            rows = list(store.iter_stage(video_id=video_id, stage="caption", model_id=model_id))
            successful = {window_idx for window_idx, _mid, _payload, error in rows if error is None}
            errored = [window_idx for window_idx, _mid, _payload, error in rows if error is not None]
            missing = sorted(expected - successful)
            if missing or errored:
                raise RuntimeError(
                    f"{video_id}/{model_id} captions are incomplete: missing windows={missing}, "
                    f"errored windows={errored}; resume annotation before publishing"
                )


def annotate_video(
    item: VideoInput,
    *,
    run_dir: Path,
    model_ids: list[str],
    registry_path: Path | None = None,
    max_spend_per_model: float | None = 10.0,
    workers: int = 1,
    pilot_windows: set[int] | None = None,
    run_hands: bool = True,
    run_captions: bool = True,
    prune_caption_frames: bool = False,
) -> str:
    item.redacted_video = item.redacted_video.resolve()
    if not item.redacted_video.is_file():
        raise FileNotFoundError(item.redacted_video)
    if run_captions and not model_ids:
        raise ValueError("captioning requested but no --model was supplied")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    video_id, blur, basis = _resolve_identity(item)
    private_row = _register_video(run_dir, item, video_id, blur, basis)
    hands_path = run_dir / "work" / video_id / "hands.parquet"

    if run_hands:
        hands_path, n_rows, n_gaps = _run_hands(item, video_id, run_dir)
        private_row["stages"]["hands"] = {
            "status": "complete",
            "input": "redacted",
            "path": str(hands_path),
            "rows": n_rows if n_rows >= 0 else "resumed_existing",
            "missing_gaps": n_gaps if n_gaps >= 0 else "resumed_existing",
        }

    if run_captions:
        counts = _run_captions(
            item,
            video_id,
            run_dir,
            model_ids=model_ids,
            registry_path=registry_path,
            max_spend_per_model=max_spend_per_model,
            workers=workers,
            pilot_windows=pilot_windows,
        )
        private_row["stages"]["captions"] = {
            "status": "pilot_complete" if pilot_windows is not None else "complete",
            "input": "redacted",
            "prompt_version": config.CAPTION_PROMPT_VERSION,
            "models": model_ids,
            "windows_written_this_call": counts,
            "pilot_windows": sorted(pilot_windows) if pilot_windows is not None else None,
            "expected_windows": _expected_caption_windows(run_dir, video_id),
        }

    with Store(run_dir / "private" / "annotations.db") as store:
        bundle = build_video_bundle(
            store=store,
            video_id=video_id,
            redacted_video=item.redacted_video,
            hands_parquet=hands_path if hands_path.exists() else None,
            bundle_dir=run_dir / "publish",
            model_ids=set(model_ids) if model_ids else None,
        )
    private_row["stages"]["huggingface_bundle"] = {
        "status": "complete",
        "path": str(run_dir / "publish"),
        "counts": bundle["counts"],
    }

    manifest = _read_private_manifest(run_dir)
    manifest["videos"][video_id] = private_row
    _save_private_manifest(run_dir, manifest)

    if prune_caption_frames:
        if pilot_windows is not None:
            raise ValueError("cannot prune caption frames after a pilot; the full run still needs them")
        cache = run_dir / "private" / "caption_frames" / video_id
        if cache.exists():
            shutil.rmtree(cache)
    return video_id


def publish_run(
    run_dir: Path,
    *,
    repo_id: str,
    private: bool,
    allow_unverified_redaction: bool = False,
    license_file: Path | None = None,
    license_name: str | None = None,
) -> None:
    if not private:
        if license_file is None or license_name is None:
            raise RuntimeError(
                "public release requires --license-file and --license-name; this replaces "
                "the private prerelease terms with owner-approved public-release terms"
            )
        install_public_license(run_dir / "publish", license_file, license_name)
    manifest = _read_private_manifest(run_dir)
    video_manifests: list[dict[str, Any]] = []
    for video_id, row in manifest["videos"].items():
        if not str(row.get("blur_status", "")).startswith("PASS") \
                and not allow_unverified_redaction:
            raise RuntimeError(
                f"{video_id} has blur_status={row.get('blur_status')!r}; refusing to publish "
                "without --allow-unverified-redaction"
            )
        _assert_captions_publishable(run_dir, video_id, row)
        public_manifest = _load_json(run_dir / "publish" / "manifests" / f"{video_id}.json")
        public_manifest["_local_redacted_video"] = row["redacted_video"]
        video_manifests.append(public_manifest)

    upload_bundle(
        bundle_dir=run_dir / "publish",
        repo_id=repo_id,
        video_manifests=video_manifests,
        private=private,
    )
    _atomic_json(
        run_dir / "private" / "hf_upload_receipt.json",
        {"repo_id": repo_id, "uploaded_at": _utc_now(), "video_ids": list(manifest["videos"])},
    )


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

    annotate = sub.add_parser("annotate", help="run MediaPipe and/or dense VLM captioning")
    annotate.add_argument("--run-dir", type=Path, required=True)
    annotate.add_argument("--batch-manifest", type=Path)
    annotate.add_argument("--redacted-video", type=Path)
    annotate.add_argument("--original-video", type=Path)
    annotate.add_argument("--blur-manifest", type=Path)
    annotate.add_argument("--video-id")
    annotate.add_argument("--model", action="append", default=[])
    annotate.add_argument("--models-toml", type=Path)
    annotate.add_argument("--max-spend-per-model", type=float, default=10.0)
    annotate.add_argument("--workers", type=int, default=1)
    annotate.add_argument("--pilot-windows")
    annotate.add_argument("--skip-hands", action="store_true")
    annotate.add_argument("--skip-captions", action="store_true")
    annotate.add_argument("--prune-caption-frames", action="store_true")

    publish = sub.add_parser("publish-hf", help="upload only the privacy-safe bundle")
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--public", action="store_true")
    publish.add_argument("--allow-unverified-redaction", action="store_true")
    publish.add_argument("--license-file", type=Path)
    publish.add_argument("--license-name")

    archive = sub.add_parser("archive-drive", help="copy, verify, receipt, optionally delete")
    archive.add_argument("--run-dir", type=Path, required=True)
    archive.add_argument("--drive-root", required=True)
    archive.add_argument("--delete-local-originals", action="store_true")
    archive.add_argument("--delete-local-redacted", action="store_true")

    yunet = sub.add_parser(
        "verify-yunet",
        help="CPU-only independent face audit on a redacted video; writes a private report",
    )
    yunet.add_argument("--redacted-video", type=Path, required=True)
    yunet.add_argument("--blur-manifest", type=Path, required=True)
    yunet.add_argument("--checkpoint-dir", type=Path, required=True)
    yunet.add_argument("--yunet-model", type=Path, required=True)
    yunet.add_argument("--job-script", type=Path, required=True)
    yunet.add_argument("--ffmpeg", default="ffmpeg")
    yunet.add_argument("--report", type=Path, required=True)
    yunet.add_argument(
        "--preview-video",
        type=Path,
        help="optional redacted-only local overlay; path must include private or DO-NOT-SHIP",
    )
    yunet.add_argument(
        "--pose-prior",
        type=Path,
        help="optional private pre-redaction pose prior; ranks likely wearer-limb false positives",
    )
    yunet.add_argument(
        "--candidate-contact-sheet-dir",
        type=Path,
        help="optional private redacted-only contact sheets, one per temporal YuNet candidate",
    )

    pose = sub.add_parser(
        "pose-prior",
        help="private pre-redaction MediaPipe Pose pass; creates a shadow-only limb prior",
    )
    pose.add_argument("--original-video", type=Path, required=True)
    pose.add_argument("--video-id", required=True)
    pose.add_argument("--output", type=Path, required=True)
    pose.add_argument("--models-dir", type=Path, default=Path("private/models"))
    pose.add_argument(
        "--preview-video",
        type=Path,
        help="optional private original-only pose overlay (amber wearer candidate; blue other pose)",
    )
    pose.add_argument("--detect-hz", type=float, default=pose_prior_layer.POSE_DETECT_HZ)
    pose.add_argument("--num-poses", type=int, default=pose_prior_layer.POSE_NUM_POSES)
    pose.add_argument("--confidence", type=float, default=pose_prior_layer.POSE_MIN_CONFIDENCE)

    decisions = sub.add_parser(
        "init-yunet-decisions",
        help="create a private all-uncertain review-decision template from a YuNet report",
    )
    decisions.add_argument("--report", type=Path, required=True)
    decisions.add_argument("--output", type=Path, required=True)

    forced = sub.add_parser(
        "decisions-to-forced-boxes",
        help="convert only confirmed private YuNet candidate tracks into EgoBlur --forced-boxes JSON",
    )
    forced.add_argument("--report", type=Path, required=True)
    forced.add_argument("--decisions", type=Path, required=True)
    forced.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "annotate":
        if args.batch_manifest:
            if args.redacted_video:
                parser.error("use either --batch-manifest or --redacted-video, not both")
            inputs = _load_batch(args.batch_manifest)
        else:
            if not args.redacted_video:
                parser.error("--redacted-video is required without --batch-manifest")
            inputs = [
                VideoInput(
                    redacted_video=args.redacted_video,
                    original_video=args.original_video,
                    blur_manifest=args.blur_manifest,
                    video_id=args.video_id,
                )
            ]
        pilot = _parse_windows(args.pilot_windows)
        for item in inputs:
            video_id = annotate_video(
                item,
                run_dir=args.run_dir.resolve(),
                model_ids=args.model,
                registry_path=args.models_toml,
                max_spend_per_model=args.max_spend_per_model,
                workers=args.workers,
                pilot_windows=pilot,
                run_hands=not args.skip_hands,
                run_captions=not args.skip_captions,
                prune_caption_frames=args.prune_caption_frames,
            )
            print(f"complete: {video_id}")
        return 0

    if args.command == "publish-hf":
        publish_run(
            args.run_dir.resolve(),
            repo_id=args.repo_id,
            private=not args.public,
            allow_unverified_redaction=args.allow_unverified_redaction,
            license_file=args.license_file,
            license_name=args.license_name,
        )
        return 0

    if args.command == "verify-yunet":
        result = verify_yunet(
            redacted_video=args.redacted_video.resolve(),
            blur_manifest=args.blur_manifest.resolve(),
            checkpoint_dir=args.checkpoint_dir.resolve(),
            yunet_model=args.yunet_model.resolve(),
            job_script=args.job_script.resolve(),
            ffmpeg=args.ffmpeg,
            report=args.report.resolve(),
            preview_video=args.preview_video.resolve() if args.preview_video else None,
            pose_prior_path=args.pose_prior.resolve() if args.pose_prior else None,
            candidate_contact_sheet_dir=(
                args.candidate_contact_sheet_dir.resolve()
                if args.candidate_contact_sheet_dir else None
            ),
        )
        print(f"{result['review_status']}: {args.report}")
        return 0

    if args.command == "pose-prior":
        video_id = _validate_video_id(args.video_id)
        artifact = pose_prior_layer.build_pose_prior(
            original_video=args.original_video,
            video_id=video_id,
            output=args.output,
            models_dir=args.models_dir,
            detect_hz=args.detect_hz,
            num_poses=args.num_poses,
            confidence=args.confidence,
            preview_video=args.preview_video,
        )
        print(f"complete: {artifact['source']['video_id']} -> {args.output}")
        return 0

    if args.command == "init-yunet-decisions":
        template = create_decision_template(args.report.resolve(), args.output.resolve())
        print(f"review template: {len(template['decisions'])} candidate(s) -> {args.output}")
        return 0

    if args.command == "decisions-to-forced-boxes":
        result = decisions_to_forced_boxes(
            args.report.resolve(), args.decisions.resolve(), args.output.resolve()
        )
        print(f"forced boxes: {result['n_forced_boxes']} -> {args.output}")
        return 0

    archive_run(
        args.run_dir.resolve(),
        drive_root=args.drive_root,
        delete_local_originals=args.delete_local_originals,
        delete_local_redacted=args.delete_local_redacted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
