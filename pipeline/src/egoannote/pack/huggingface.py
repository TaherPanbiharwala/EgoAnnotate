"""Build and upload the privacy-safe Hugging Face dataset bundle.

SQLite remains resumable local state. Hugging Face receives queryable
Parquet tables, the redacted video, and provenance—never an original video,
original-derived debug image, frame cache, or API response database.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..store import Store

_PRIVATE_LICENSE_NAME = "Egoannote private prerelease terms"
_PRIVATE_LICENSE_TEXT = """EGOANNOTE DATASET — PRIVATE PRERELEASE TERMS

Copyright (c) the dataset creator. All rights reserved.

This private prerelease is supplied only for the repository owner's evaluation
and preparation work. No public permission to copy, redistribute, adapt, or
use the dataset is granted by this file.

Before making a repository public, replace this file with release terms that
are approved by the rights holder and update the dataset card metadata through
the pipeline's --license-file and --license-name options.
"""

WINDOWS_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("model_id", pa.string()),
        ("provider", pa.string()),
        ("window_idx", pa.int32()),
        ("start_ms", pa.int64()),
        ("end_ms", pa.int64()),
        ("activity_caption", pa.string()),
        ("activity_goal", pa.string()),
        ("activity_phase", pa.string()),
        ("activity_progression", pa.string()),
        ("action_count", pa.int16()),
        ("schema_ok", pa.bool_()),
        ("is_partial", pa.bool_()),
        ("prompt_version", pa.string()),
        ("prompt_hash", pa.string()),
        ("run_id", pa.string()),
        ("latency_ms", pa.int64()),
        ("input_tokens", pa.int64()),
        ("output_tokens", pa.int64()),
        ("cost_usd", pa.float64()),
        ("error", pa.string()),
    ]
)


ACTIONS_SCHEMA = pa.schema(
    [
        ("caption_id", pa.string()),
        ("video_id", pa.string()),
        ("model_id", pa.string()),
        ("window_idx", pa.int32()),
        ("action_idx", pa.int16()),
        ("start_ms", pa.int64()),
        ("end_ms", pa.int64()),
        ("start_caption_frame", pa.int64()),
        ("end_caption_frame", pa.int64()),
        ("action_caption", pa.string()),
        ("task_step", pa.string()),
        ("left_hand_caption", pa.string()),
        ("left_verb", pa.string()),
        ("left_object", pa.string()),
        ("left_target", pa.string()),
        ("left_contact", pa.string()),
        ("left_visible", pa.bool_()),
        ("right_hand_caption", pa.string()),
        ("right_verb", pa.string()),
        ("right_object", pa.string()),
        ("right_target", pa.string()),
        ("right_contact", pa.string()),
        ("right_visible", pa.bool_()),
        ("tool_in_use", pa.string()),
        ("coordination", pa.string()),
        ("handover_event", pa.bool_()),
        ("scene_what", pa.string()),
        ("scene_how", pa.string()),
        ("scene_why", pa.string()),
        ("scene_location", pa.string()),
        ("activity_caption", pa.string()),
        ("schema_ok", pa.bool_()),
        ("prompt_version", pa.string()),
        ("prompt_hash", pa.string()),
        ("run_id", pa.string()),
    ]
)


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _write_table(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def _ensure_dataset_card(bundle_dir: Path) -> None:
    """Write the generated card and conservative private-release terms once."""
    card_path = bundle_dir / "README.md"
    if not card_path.exists():
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(
            f"""---
pretty_name: Egoannote dense egocentric activity annotations
license: other
license_name: {_PRIVATE_LICENSE_NAME}
license_link: LICENSE
task_categories:
- video-classification
- visual-question-answering
---

# Egoannote dense activity annotations

This dataset contains privacy-redacted egocentric video with three synchronized
caption tracks: holistic activity captions per time window, temporally grounded
atomic-action captions, and separate left/right-hand action captions. Dense
MediaPipe hand landmarks are stored separately and join by `video_id` and time.

## Files

- `videos/`: EgoBlur-redacted videos only.
- `annotations/*/caption_windows.parquet`: the activity narrative over time.
- `annotations/*/caption_actions.parquet`: atomic and per-hand captions.
- `annotations/*/hands.parquet`: MediaPipe landmarks and track IDs.
- `manifests/`: publish-safe hashes and counts; no original or local paths.

Original videos, extracted caption frames, private manifests, model credentials,
and original-derived debug images are intentionally excluded.
""",
            encoding="utf-8",
        )
    terms_path = bundle_dir / "LICENSE"
    if not terms_path.exists():
        terms_path.write_text(_PRIVATE_LICENSE_TEXT, encoding="utf-8")


def install_public_license(bundle_dir: Path, license_file: Path, license_name: str) -> None:
    """Install owner-approved public-release terms into a generated bundle."""
    license_name = license_name.strip()
    if not license_name or "\n" in license_name or "\r" in license_name:
        raise ValueError("license_name must be a non-empty single line")
    if not license_file.is_file():
        raise FileNotFoundError(license_file)
    license_text = license_file.read_text(encoding="utf-8").strip()
    if not license_text:
        raise ValueError(f"license file {license_file} is empty")

    card_path = bundle_dir / "README.md"
    if not card_path.exists():
        raise RuntimeError(f"dataset card {card_path} is missing; rebuild the bundle first")
    card = card_path.read_text(encoding="utf-8")
    replacement = f"license_name: {json.dumps(license_name)}"
    updated, count = re.subn(r"(?m)^license_name:.*$", replacement, card, count=1)
    if count != 1 or "license: other" not in updated:
        raise RuntimeError(
            f"dataset card {card_path} does not contain the generated custom-license metadata"
        )
    card_path.write_text(updated, encoding="utf-8")
    (bundle_dir / "LICENSE").write_text(license_text + "\n", encoding="utf-8")


def export_captions(
    store: Store,
    *,
    video_id: str,
    out_dir: Path,
    model_ids: set[str] | None = None,
) -> tuple[Path, Path, int, int]:
    """Export one row per window and one row per atomic action."""
    windows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    frame_ms = 1000 * config.VLM_FPS.denominator / config.VLM_FPS.numerator

    for window_idx, model_id, payload, row_error in store.iter_stage(
        video_id=video_id, stage="caption", model_id=None
    ):
        if model_ids is not None and model_id not in model_ids:
            continue
        activity = payload.get("activity") or {}
        window_actions = payload.get("actions") or []
        error = row_error or payload.get("error")
        windows.append(
            {
                "video_id": video_id,
                "model_id": model_id,
                "provider": payload.get("provider"),
                "window_idx": window_idx,
                "start_ms": payload.get("start_ts_ms"),
                "end_ms": payload.get("end_ts_ms"),
                "activity_caption": activity.get("caption"),
                "activity_goal": activity.get("goal"),
                "activity_phase": activity.get("phase"),
                "activity_progression": activity.get("progression"),
                "action_count": len(window_actions),
                "schema_ok": bool(payload.get("schema_ok", False)),
                "is_partial": bool(payload.get("is_partial", False)),
                "prompt_version": payload.get("prompt_version"),
                "prompt_hash": payload.get("prompt_hash"),
                "run_id": payload.get("run_id"),
                "latency_ms": payload.get("latency_ms"),
                "input_tokens": payload.get("input_tokens"),
                "output_tokens": payload.get("output_tokens"),
                "cost_usd": payload.get("cost_usd"),
                "error": error,
            }
        )

        caption_frames = payload.get("frame_indices") or []
        for action_idx, action in enumerate(window_actions):
            left = action.get("left_hand") or {}
            right = action.get("right_hand") or {}
            rel_start = int(action.get("start_frame", 0))
            rel_end = int(action.get("end_frame", rel_start))
            start_ms = round(int(payload["start_ts_ms"]) + rel_start * frame_ms)
            end_ms = min(
                int(payload["end_ts_ms"]),
                round(int(payload["start_ts_ms"]) + (rel_end + 1) * frame_ms),
            )
            start_caption_frame = (
                caption_frames[rel_start] if 0 <= rel_start < len(caption_frames) else None
            )
            end_caption_frame = (
                caption_frames[rel_end] if 0 <= rel_end < len(caption_frames) else None
            )
            actions.append(
                {
                    "caption_id": f"{video_id}:{model_id}:{window_idx}:{action_idx}",
                    "video_id": video_id,
                    "model_id": model_id,
                    "window_idx": window_idx,
                    "action_idx": action_idx,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_caption_frame": start_caption_frame,
                    "end_caption_frame": end_caption_frame,
                    "action_caption": action.get("action_caption"),
                    "task_step": action.get("task_step"),
                    "left_hand_caption": left.get("caption"),
                    "left_verb": left.get("verb"),
                    "left_object": left.get("object"),
                    "left_target": left.get("target"),
                    "left_contact": left.get("contact_type"),
                    "left_visible": left.get("visible"),
                    "right_hand_caption": right.get("caption"),
                    "right_verb": right.get("verb"),
                    "right_object": right.get("object"),
                    "right_target": right.get("target"),
                    "right_contact": right.get("contact_type"),
                    "right_visible": right.get("visible"),
                    "tool_in_use": action.get("tool_in_use"),
                    "coordination": action.get("coordination"),
                    "handover_event": action.get("handover_event"),
                    "scene_what": action.get("scene_what"),
                    "scene_how": action.get("scene_how"),
                    "scene_why": action.get("scene_why"),
                    "scene_location": action.get("scene_location"),
                    "activity_caption": activity.get("caption"),
                    "schema_ok": bool(payload.get("schema_ok", False)),
                    "prompt_version": payload.get("prompt_version"),
                    "prompt_hash": payload.get("prompt_hash"),
                    "run_id": payload.get("run_id"),
                }
            )

    windows_path = out_dir / "caption_windows.parquet"
    actions_path = out_dir / "caption_actions.parquet"
    _write_table(windows, WINDOWS_SCHEMA, windows_path)
    _write_table(actions, ACTIONS_SCHEMA, actions_path)
    return windows_path, actions_path, len(windows), len(actions)


def build_video_bundle(
    *,
    store: Store,
    video_id: str,
    redacted_video: Path,
    hands_parquet: Path | None,
    bundle_dir: Path,
    model_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Create the small local bundle; the video itself is referenced, not copied."""
    _ensure_dataset_card(bundle_dir)
    annotations_dir = bundle_dir / "annotations" / video_id
    windows, actions, n_windows, n_actions = export_captions(
        store, video_id=video_id, out_dir=annotations_dir, model_ids=model_ids
    )
    files = [windows, actions]
    if hands_parquet is not None and hands_parquet.exists():
        hands_target = annotations_dir / "hands.parquet"
        if hands_parquet.resolve() != hands_target.resolve():
            shutil.copy2(hands_parquet, hands_target)
        files.append(hands_target)

    manifest = {
        "schema_version": 1,
        "video_id": video_id,
        "redacted_video": {
            "hub_path": f"videos/{video_id}/{redacted_video.name}",
            "bytes": redacted_video.stat().st_size,
            "sha256": _sha256_file(redacted_video),
        },
        "annotation_files": [
            {
                "path": str(p.relative_to(bundle_dir)),
                "bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            }
            for p in files
        ],
        "counts": {"caption_windows": n_windows, "atomic_actions": n_actions},
        "privacy": {
            "original_video_included": False,
            "caption_frames_included": False,
            "original_debug_media_included": False,
        },
    }
    manifest_path = bundle_dir / "manifests" / f"{video_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Runtime-only transfer source. The underscore key is never written to
    # the publish manifest, so local paths cannot leak into the dataset.
    manifest["_local_redacted_video"] = str(redacted_video.resolve())
    return manifest


def upload_bundle(
    *,
    bundle_dir: Path,
    repo_id: str,
    video_manifests: list[dict[str, Any]],
    private: bool = True,
    token: str | None = None,
) -> None:
    """Upload annotations resumably, then stream each redacted video."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face upload support is not installed; run `uv sync` first"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    # create_repo(..., exist_ok=True) leaves an existing repository's
    # visibility unchanged.  This matters for the intended lifecycle: upload
    # the batch privately, then invoke this command with --public once it is
    # ready to release.
    api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(bundle_dir),
        commit_message="Upload egoannote annotations and manifests",
    )
    for manifest in video_manifests:
        video = manifest["redacted_video"]
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=manifest["_local_redacted_video"],
            path_in_repo=video["hub_path"],
            commit_message=f"Upload redacted video {manifest['video_id']}",
        )
