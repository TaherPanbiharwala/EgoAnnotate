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
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import config, curated_caption_events
from . import hand_prior as hand_prior_layer
from . import original_curated as original_curated_layer
from . import pose_prior as pose_prior_layer
from . import public_release as public_release_layer
from .archive import archive_files
from .backends.openai_compat import SpendTracker
from .backends.registry import build_backend
from .layers import caption as caption_layer
from .layers import hands as hands_layer
from .media.probe import probe
from .pack.huggingface import build_video_bundle, install_public_license, upload_bundle
from .store import Store, write_hands_parquet_streaming
from .verify_yunet import create_decision_template, decisions_to_forced_boxes, verify_yunet

log = logging.getLogger("egoannote.pipeline")

_VIDEO_ID_RE = re.compile(r"^[\w.-]+$", flags=re.UNICODE)
_FACE_FREE_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}


@dataclass(slots=True)
class VideoInput:
    redacted_video: Path
    original_video: Path | None = None
    blur_manifest: Path | None = None
    redaction_review: Path | None = None
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
                    redaction_review=resolve_from_manifest(row.get("redaction_review")),
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


def _is_private_path(path: Path) -> bool:
    """Keep human-review evidence out of a public bundle by construction."""
    return any(part.lower() in {"private", "do-not-ship"} for part in path.parts)


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_file_binding(value: Any, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"redaction approval {label} binding is missing or invalid")
    raw_path = value.get("path")
    expected_sha256 = value.get("sha256")
    expected_bytes = value.get("bytes")
    if not isinstance(raw_path, str) or not isinstance(expected_sha256, str) or not isinstance(
        expected_bytes, int
    ):
        raise ValueError(f"redaction approval {label} binding is missing or invalid")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"approved {label} is no longer available: {path}")
    if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha256:
        raise ValueError(
            f"approved {label} no longer matches its recorded SHA-256; create a new human review "
            "approval for the changed artifact"
        )
    return path


def _validated_redaction_review(
    review_path: Path | None,
    *,
    video_id: str,
    redacted_video: Path,
    blur_manifest: Path,
    blur: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate a private human decision bound to all reviewed artifacts.

    A `NEEDS_REVIEW` status never becomes a pass automatically.  This record
    merely preserves the owner's explicit visual decision and makes it stale
    whenever the video, EgoBlur manifest, or either private review report is
    changed or removed.
    """
    if review_path is None:
        return None
    review_path = review_path.expanduser().resolve()
    if not _is_private_path(review_path):
        raise ValueError("--redaction-review must be stored under private or DO-NOT-SHIP")
    if not review_path.is_file():
        raise FileNotFoundError(f"redaction review approval not found: {review_path}")
    approval = _load_json(review_path)
    if approval.get("schema_version") != 1 or approval.get("review_type") != "human_redaction_approval":
        raise ValueError(f"invalid redaction review approval: {review_path}")
    if approval.get("decision") != "approved":
        raise ValueError("redaction review decision is not approved")
    reviewer = approval.get("reviewer")
    reviewed_at = approval.get("reviewed_at")
    if (
        not isinstance(reviewer, str)
        or not reviewer.strip()
        or "\n" in reviewer
        or "\r" in reviewer
        or not isinstance(reviewed_at, str)
        or not reviewed_at
    ):
        raise ValueError("redaction review approval lacks a valid reviewer or timestamp")
    if approval.get("video_id") != video_id:
        raise ValueError(
            f"redaction review video_id={approval.get('video_id')!r} does not match {video_id!r}"
        )

    approved_video = _verify_file_binding(approval.get("redacted_video"), label="redacted video")
    approved_manifest = _verify_file_binding(approval.get("blur_manifest"), label="EgoBlur manifest")
    _verify_file_binding(approval.get("hand_suppression_report"), label="hand-suppression report")
    _verify_file_binding(approval.get("yunet_report"), label="YuNet report")
    if approved_video != redacted_video.resolve() or approved_manifest != blur_manifest.resolve():
        raise ValueError(
            "redaction review approval is bound to different video or EgoBlur manifest paths; "
            "create a new approval for these inputs"
        )
    if _sha256_file(redacted_video) != str((blur.get("output") or {}).get("sha256") or ""):
        raise ValueError("redacted video does not match manifest.output.sha256")

    return {
        "status": "human_approved",
        "approval_path": str(review_path),
        "approval_sha256": _sha256_file(review_path),
        "reviewer": reviewer.strip(),
        "reviewed_at": reviewed_at,
    }


def create_redaction_review(
    *,
    run_dir: Path,
    video_id: str,
    redacted_video: Path,
    blur_manifest: Path,
    hand_suppression_report: Path,
    yunet_report: Path,
    reviewer: str,
) -> Path:
    """Record a private, owner-made visual approval for one exact redaction."""
    video_id = _validate_video_id(video_id)
    reviewer = reviewer.strip()
    if not reviewer or "\n" in reviewer or "\r" in reviewer:
        raise ValueError("--reviewer must be a non-empty single line")
    redacted_video = redacted_video.expanduser().resolve()
    blur_manifest = blur_manifest.expanduser().resolve()
    hand_suppression_report = hand_suppression_report.expanduser().resolve()
    yunet_report = yunet_report.expanduser().resolve()
    blur = _load_json(blur_manifest)
    if str(blur.get("clip_id") or "") != video_id:
        raise ValueError(
            f"EgoBlur manifest clip_id={blur.get('clip_id')!r} does not match video_id={video_id!r}"
        )
    if _sha256_file(redacted_video) != str((blur.get("output") or {}).get("sha256") or ""):
        raise ValueError("redacted video SHA-256 does not match manifest.output.sha256")

    hand = _load_json(hand_suppression_report)
    if hand.get("review_type") != "egoblur_hand_suppression":
        raise ValueError("hand-suppression report has an unexpected review_type")
    if (hand.get("input") or {}).get("clip_id") != video_id:
        raise ValueError("hand-suppression report belongs to a different clip")
    if (hand.get("input") or {}).get("source_sha256") != (blur.get("source") or {}).get("sha256"):
        raise ValueError("hand-suppression report is not bound to this EgoBlur source")

    yunet = _load_json(yunet_report)
    if yunet.get("review_type") != "post_redaction_yunet":
        raise ValueError("YuNet report has an unexpected review_type")
    if (yunet.get("input") or {}).get("clip_id") != video_id:
        raise ValueError("YuNet report belongs to a different clip")
    if (yunet.get("input") or {}).get("redacted_sha256") != _sha256_file(redacted_video):
        raise ValueError("YuNet report is not bound to this redacted video")
    if (yunet.get("input") or {}).get("egoblur_manifest_sha256") != _sha256_file(blur_manifest):
        raise ValueError("YuNet report is not bound to this EgoBlur manifest")

    output = run_dir.resolve() / "private" / "redaction_reviews" / f"{video_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing redaction review approval: {output}")
    _atomic_json(
        output,
        {
            "schema_version": 1,
            "review_type": "human_redaction_approval",
            "privacy": "private_do_not_ship",
            "decision": "approved",
            "reviewer": reviewer,
            "reviewed_at": _utc_now(),
            "video_id": video_id,
            "redacted_video": _file_binding(redacted_video, label="redacted video"),
            "blur_manifest": _file_binding(blur_manifest, label="EgoBlur manifest"),
            "hand_suppression_report": _file_binding(
                hand_suppression_report, label="hand-suppression report"
            ),
            "yunet_report": _file_binding(yunet_report, label="YuNet report"),
            "approval_scope": (
                "Named human visual approval of this exact redacted video and its linked private "
                "EgoBlur, hand-suppression, and YuNet evidence. It is not an automatic detector pass."
            ),
        },
    )
    return output


def _register_video(
    run_dir: Path, item: VideoInput, video_id: str, blur: dict[str, Any] | None, basis: str
) -> dict[str, Any]:
    manifest = _read_private_manifest(run_dir)
    redacted = item.redacted_video.resolve()
    original = item.original_video.resolve() if item.original_video else None
    blur_status = (blur or {}).get("status", "UNKNOWN_NO_MANIFEST")
    if item.redaction_review and not blur:
        raise ValueError("--redaction-review requires --blur-manifest")
    review = (
        _validated_redaction_review(
            item.redaction_review,
            video_id=video_id,
            redacted_video=redacted,
            blur_manifest=item.blur_manifest.resolve(),
            blur=blur,
        )
        if blur and item.redaction_review
        else None
    )
    if blur and not str(blur_status).startswith("PASS") and review is None:
        raise ValueError(
            f"EgoBlur manifest status for {video_id} is {blur_status!r}; annotation may "
            "run only from a reviewed PASS redaction or a valid private --redaction-review"
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
            "redaction_review": review,
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
            "target_fps": config.MP_FPS,
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
        review = row.get("redaction_review")
        review_is_approved = False
        if isinstance(review, dict) and review.get("status") == "human_approved":
            review_path = review.get("approval_path")
            manifest_path = row.get("blur_manifest")
            redacted_path = row.get("redacted_video")
            if not isinstance(review_path, str) or not isinstance(manifest_path, str) or not isinstance(
                redacted_path, str
            ):
                raise RuntimeError(f"{video_id} has an incomplete recorded human redaction review")
            try:
                _validated_redaction_review(
                    Path(review_path),
                    video_id=video_id,
                    redacted_video=Path(redacted_path),
                    blur_manifest=Path(manifest_path),
                    blur=_load_json(Path(manifest_path)),
                )
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{video_id} human redaction review no longer validates: {exc}"
                ) from exc
            review_is_approved = True
        if not str(row.get("blur_status", "")).startswith("PASS") \
                and not review_is_approved and not allow_unverified_redaction:
            raise RuntimeError(
                f"{video_id} has blur_status={row.get('blur_status')!r}; refusing to publish "
                "without a valid human redaction review or --allow-unverified-redaction"
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

    annotate = sub.add_parser("annotate", help="run MediaPipe and/or dense VLM captioning")
    annotate.add_argument("--run-dir", type=Path, required=True)
    annotate.add_argument("--batch-manifest", type=Path)
    annotate.add_argument("--redacted-video", type=Path)
    annotate.add_argument("--original-video", type=Path)
    annotate.add_argument("--blur-manifest", type=Path)
    annotate.add_argument(
        "--redaction-review",
        type=Path,
        help="private hash-bound human approval required for a non-PASS EgoBlur manifest",
    )
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

    public_release = sub.add_parser(
        "prepare-public-release",
        help="build an owner-approved public dataset folder from curated face-free artifacts",
    )
    public_release.add_argument(
        "--children-dir",
        type=Path,
        required=True,
        help="private face-free child directory containing one <video-id> directory per clip",
    )
    public_release.add_argument(
        "--annotation-run-dir",
        type=Path,
        action="append",
        required=True,
        help="ordered private run directory; first hash-bound complete annotation set wins",
    )
    public_release.add_argument(
        "--annotated-video-dir",
        type=Path,
        action="append",
        required=True,
        help="ordered directory containing final hand-and-caption overlay MP4s",
    )
    public_release.add_argument("--output-dir", type=Path, required=True)
    public_release.add_argument("--video-id", action="append", required=True)
    public_release.add_argument("--model", required=True)
    public_release.add_argument("--approved-by", required=True)
    public_release.add_argument("--release-version", default="v1.0")

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
        "--hand-prior",
        type=Path,
        help=(
            "optional private 30-fps pre-redaction Hand Landmarker prior; filters likely "
            "wearer-hand YuNet noise using expanded amber/pink hand regions"
        ),
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

    hand_prior = sub.add_parser(
        "hand-prior",
        help="private pre-redaction MediaPipe Hand Landmarker pass for wearer-hand review",
    )
    hand_prior.add_argument("--original-video", type=Path, required=True)
    hand_prior.add_argument("--video-id", required=True)
    hand_prior.add_argument("--output", type=Path, required=True)
    hand_prior.add_argument("--models-dir", type=Path, default=Path("private/models"))
    hand_prior.add_argument(
        "--preview-video",
        type=Path,
        help="optional private original-only hand overlay (amber stable wearer; magenta provisional; blue other)",
    )
    hand_prior.add_argument("--detect-hz", type=float, default=hand_prior_layer.HAND_DETECT_HZ)
    hand_prior.add_argument(
        "--num-hands",
        type=int,
        default=hand_prior_layer.HAND_NUM_HANDS,
        help=(
            "maximum hands to detect (default: 2 for the pinned active policy; "
            f"use {hand_prior_layer.HAND_REVIEW_NUM_HANDS} for private multi-hand review)"
        ),
    )
    hand_prior.add_argument("--confidence", type=float, default=hand_prior_layer.HAND_MIN_CONFIDENCE)

    approve = sub.add_parser(
        "approve-redaction",
        help="record named private human approval of one reviewed redacted video",
    )
    approve.add_argument("--run-dir", type=Path, required=True)
    approve.add_argument("--video-id", required=True)
    approve.add_argument("--redacted-video", type=Path, required=True)
    approve.add_argument("--blur-manifest", type=Path, required=True)
    approve.add_argument("--hand-suppression-report", type=Path, required=True)
    approve.add_argument("--yunet-report", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)

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

    if args.command == "prepare-public-release":
        release = public_release_layer.build_public_release(
            children_dir=args.children_dir,
            annotation_run_dirs=args.annotation_run_dir,
            annotated_video_dirs=args.annotated_video_dir,
            output_dir=args.output_dir,
            video_ids=[_validate_video_id(video_id) for video_id in args.video_id],
            model_id=args.model,
            approved_by=args.approved_by,
            release_version=args.release_version,
        )
        print(
            f"complete: {len(release['video_ids'])} public release video(s) -> "
            f"{release['output_dir']}"
        )
        return 0

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
                    redaction_review=args.redaction_review,
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

    if args.command == "approve-redaction":
        output = create_redaction_review(
            run_dir=args.run_dir,
            video_id=args.video_id,
            redacted_video=args.redacted_video,
            blur_manifest=args.blur_manifest,
            hand_suppression_report=args.hand_suppression_report,
            yunet_report=args.yunet_report,
            reviewer=args.reviewer,
        )
        print(f"approved: {args.video_id} -> {output}")
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
            hand_prior_path=args.hand_prior.resolve() if args.hand_prior else None,
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

    if args.command == "hand-prior":
        video_id = _validate_video_id(args.video_id)
        artifact = hand_prior_layer.build_hand_prior(
            original_video=args.original_video,
            video_id=video_id,
            output=args.output,
            models_dir=args.models_dir,
            detect_hz=args.detect_hz,
            num_hands=args.num_hands,
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
