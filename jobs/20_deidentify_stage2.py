# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "torch==2.5.1",
#   "torchvision==0.20.1",
#   "transformers==4.49.0",
#   "pillow==11.1.0",
#   "tqdm==4.67.1",
#   "hydra-core==1.3.2",
#   "iopath==0.1.10",
# ]
# ///
"""Stage II face de-identification job.

Milestones 1-5 define the fail-closed local contracts, DINO proposal layer,
bounded SAM2 mask-shard layer, verified Stage I-only renderer, private review
artifacts, release gating, and operator command surface. Heavy model imports
remain lazy so deterministic adapters and rendering fixtures can be exercised
without a GPU. Milestone 6 adds verified offline model smoke and real-frame
attestation in this same self-contained PEP-723 job because GPU jobs may not
share an environment; paid clip calibration remains an explicit operator gate.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE2_SCHEMA_VERSION = 1
STAGE2_CODE_VERSION = "milestone-6"
STAGE2_JOB_VERSION = "0.6.0"
DINO_CODE_VERSION = "milestone-2"
SAM_CODE_VERSION = "milestone-6-frame-attested"
RENDER_CODE_VERSION = "milestone-4"

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
DINO_MODEL_REVISION = "e76a695ed7ae1032a61530cce4b4e9b65f4e368b"
DINO_MODEL_WEIGHTS_SHA256 = "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
DINO_PROMPT = "face."
DINO_PROPOSAL_FLOOR = 0.10
DINO_TEXT_THRESHOLD = 0.25
DINO_ANCHOR_SPACING = 20
DINO_TILE_OVERLAP = 0.20
DINO_NMS_IOU = 0.70

SAM_MODEL_ID = "facebook/sam2.1-hiera-large"
SAM_MODEL_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
SAM_MODEL_WEIGHTS_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
SAM_RUNTIME_REPOSITORY = "https://github.com/facebookresearch/sam2.git"
SAM_RUNTIME_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM_MODEL_CONFIG_SHA256 = "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
SAM_FALLBACK_PADDING_SCALE = 0.15
SAM_FALLBACK_MIN_PADDING_PX = 4
SAM_MIN_MASK_PIXELS = 4
SAM_MIN_PROMPT_AREA_RATIO = 0.05
SAM_MAX_PROMPT_AREA_RATIO = 16.0
SAM_MAX_FRAME_AREA_RATIO = 0.65
SAM_ANCHOR_NEIGHBORHOOD_SCALE = 2.0
SAM_MIN_ANCHOR_NEIGHBORHOOD_FRACTION = 0.50
SAM_FRAME_EXTRACTION_VERSION = "ffmpeg-select-jpeg-q2-v1"
SAM_FRAME_ATTESTATION_FILENAME = "frames.manifest.json"

RENDER_DILATION_PIXELS_AT_1080P = 8
RENDER_FILL_YUV = (128, 128, 128)
RENDER_FILL_TOLERANCE = 24
RENDER_MAX_OUTSIDE_MASK_MAE = 12.0

EXPECTED_STAGE1 = {
    "gen": "2",
    "face_threshold": 0.30,
    "continue_threshold": 0.0,
    "redaction": "fill",
    "lp_checked": False,
    "dilate_scale": 1.3,
    "motion_margin_px": 8,
    "hold_frames": 45,
    "back_hold_frames": 45,
}

PROCESSING_STATES = (
    "PENDING",
    "VALIDATED",
    "DINO_COMPLETE",
    "SAM_COMPLETE",
    "RENDER_COMPLETE",
    "PROCESSING_COMPLETE",
    "FAILED",
)
_STATE_ORDER = {name: index for index, name in enumerate(PROCESSING_STATES[:-1])}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Stage2Error(RuntimeError):
    """A fail-closed error with operator-readable recovery information."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        recovery: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.recovery = recovery

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "message": self.message,
            "details": self.details,
            "recovery": self.recovery,
        }


@dataclass(slots=True, frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    bytes: int


@dataclass(slots=True, frozen=True)
class ModelIdentity:
    name: str
    revision: str
    sha256: str


@dataclass(slots=True, frozen=True)
class SamRuntimeIdentity:
    repository: str
    revision: str
    source_tree_sha256: str
    model_config: str
    model_config_sha256: str
    torch_version: str
    cuda_version: str


@dataclass(slots=True, frozen=True)
class VideoFacts:
    coded_width: int
    coded_height: int
    display_width: int
    display_height: int
    fps: float
    n_frames: int
    duration_s: float
    rotation: int
    is_cfr: bool


@dataclass(slots=True, frozen=True)
class StageIInput:
    schema_version: int
    clip_id: str
    source: ArtifactRef
    stage1_video: ArtifactRef
    stage1_manifest: ArtifactRef
    source_video: VideoFacts
    stage1_output_video: VideoFacts
    stage1_status: str
    stage1_audit_reasons: tuple[str, ...]
    egoblur: dict[str, Any]
    warnings: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class Proposal:
    frame_idx: int
    box: tuple[float, float, float, float]
    score: float
    source: str
    label: str = "face"
    proposal_id: str = ""
    origins: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class RawDinoDetection:
    box: tuple[float, float, float, float]
    score: float
    label: str = "face"


@dataclass(slots=True, frozen=True)
class DinoGenerationConfig:
    model: ModelIdentity
    prompt: str = DINO_PROMPT
    proposal_floor: float = DINO_PROPOSAL_FLOOR
    text_threshold: float = DINO_TEXT_THRESHOLD
    anchor_spacing: int = DINO_ANCHOR_SPACING
    tile_rows: int = 2
    tile_cols: int = 2
    tile_overlap: float = DINO_TILE_OVERLAP
    nms_iou: float = DINO_NMS_IOU
    view_batch_size: int = 1
    preprocessing: tuple[tuple[str, Any], ...] = (
        ("color", "RGB"),
        ("resize", "model-default"),
        ("normalization", "model-default"),
    )


@dataclass(slots=True, frozen=True)
class Tile:
    origin: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(slots=True, frozen=True)
class DinoArtifactMeta:
    schema_version: int
    artifact_type: str
    fingerprint: str
    source_sha256: str
    model: ModelIdentity
    prompt: str
    proposal_floor: float
    text_threshold: float
    anchor_spacing: int
    anchor_frames: tuple[int, ...]
    tiling: dict[str, Any]
    nms_iou: float
    preprocessing: dict[str, Any]
    proposals: tuple[Proposal, ...]
    metrics: dict[str, Any]


@dataclass(slots=True, frozen=True)
class DinoGenerationResult:
    artifact: ArtifactRef
    meta: DinoArtifactMeta
    reused_final_artifact: bool
    reused_anchor_count: int
    generated_anchor_count: int


@dataclass(slots=True, frozen=True)
class DinoThresholdSelection:
    threshold: float
    accepted: tuple[Proposal, ...]
    rejected: tuple[Proposal, ...]

    @property
    def metrics(self) -> dict[str, int | float]:
        return {
            "threshold": self.threshold,
            "n_accepted": len(self.accepted),
            "n_rejected": len(self.rejected),
        }


@dataclass(slots=True, frozen=True)
class TemporalWindow:
    index: int
    frame_start: int
    frame_end: int

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start + 1


@dataclass(slots=True, frozen=True)
class SamGenerationConfig:
    model: ModelIdentity
    runtime: SamRuntimeIdentity
    accepted_proposal_threshold: float
    window_size: int
    window_overlap: int
    precision: str = "bfloat16"
    fallback_padding_scale: float = SAM_FALLBACK_PADDING_SCALE
    fallback_min_padding_px: int = SAM_FALLBACK_MIN_PADDING_PX
    min_mask_pixels: int = SAM_MIN_MASK_PIXELS
    min_prompt_area_ratio: float = SAM_MIN_PROMPT_AREA_RATIO
    max_prompt_area_ratio: float = SAM_MAX_PROMPT_AREA_RATIO
    max_frame_area_ratio: float = SAM_MAX_FRAME_AREA_RATIO
    anchor_neighborhood_scale: float = SAM_ANCHOR_NEIGHBORHOOD_SCALE
    min_anchor_neighborhood_fraction: float = SAM_MIN_ANCHOR_NEIGHBORHOOD_FRACTION


@dataclass(slots=True, frozen=True)
class SamPrompt:
    prompt_id: str
    prompt_kind: str
    anchor_frame: int
    box: tuple[float, float, float, float]
    source_id: str


@dataclass(slots=True, frozen=True)
class SamWindowInput:
    source_sha256: str
    frame_start: int
    frame_end: int
    frame_width: int
    frame_height: int
    payload: Any
    frame_payload_sha256: str | None = None
    frame_artifacts: tuple[ArtifactRef, ...] = ()
    attestation_artifact: ArtifactRef | None = None


@dataclass(slots=True, frozen=True)
class RawSamMask:
    frame_idx: int
    prompt_id: str
    crop_box: tuple[int, int, int, int]
    pixels: bytes
    direction: str


@dataclass(slots=True, frozen=True)
class SamPropagationResult:
    masks: tuple[RawSamMask, ...]
    forward_complete: bool
    reverse_complete: bool
    runtime_seconds: float = 0.0
    peak_vram_bytes: int = 0


@dataclass(slots=True, frozen=True)
class PackedMask:
    key: str
    frame_idx: int
    prompt_id: str
    source: str
    anchor_frame: int
    crop_box: tuple[int, int, int, int]
    packed_bits: bytes
    area_pixels: int


@dataclass(slots=True, frozen=True)
class SamMaskShardMeta:
    schema_version: int
    artifact_type: str
    fingerprint: str
    dino_artifact_sha256: str
    dino_fingerprint: str
    model: ModelIdentity
    runtime: SamRuntimeIdentity
    accepted_proposal_threshold: float
    frame_start: int
    frame_end: int
    frame_width: int
    frame_height: int
    window: dict[str, Any]
    precision: str
    prompts: tuple[dict[str, Any], ...]
    masks: tuple[dict[str, Any], ...]
    review_flags: tuple[str, ...]
    metrics: dict[str, Any]


@dataclass(slots=True, frozen=True)
class LoadedSamMaskShard:
    artifact: ArtifactRef
    meta: SamMaskShardMeta
    masks: tuple[PackedMask, ...]


@dataclass(slots=True, frozen=True)
class SamGenerationResult:
    shards: tuple[ArtifactRef, ...]
    metas: tuple[SamMaskShardMeta, ...]
    reused_window_count: int
    generated_window_count: int
    review_flags: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class RenderArtifactMeta:
    schema_version: int
    artifact_type: str
    fingerprint: str
    stage1_video_sha256: str
    mask_set_sha256: str
    output: ArtifactRef
    encoder: dict[str, Any]
    verification: dict[str, Any]


@dataclass(slots=True, frozen=True)
class RenderConfig:
    dilation_pixels_at_1080p: int = RENDER_DILATION_PIXELS_AT_1080P
    fill_yuv: tuple[int, int, int] = RENDER_FILL_YUV
    codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    fill_tolerance: int = RENDER_FILL_TOLERANCE
    max_outside_mask_mae: float = RENDER_MAX_OUTSIDE_MASK_MAE


@dataclass(slots=True, frozen=True)
class RenderGenerationResult:
    artifact: ArtifactRef
    meta: RenderArtifactMeta
    reused: bool


@dataclass(slots=True, frozen=True)
class ManualSeed:
    schema_version: int
    seed_id: str
    clip_id: str
    frame_idx: int
    box: tuple[float, float, float, float]
    reason: str


@dataclass(slots=True, frozen=True)
class FaceEventLabel:
    schema_version: int
    event_id: str
    clip_id: str
    frame_start: int
    frame_end: int
    conservative_box: tuple[float, float, float, float]
    label_kind: str
    visibility: str
    category: str
    stage1_verdict: str
    dino_proposal_verdicts: tuple[dict[str, Any], ...]
    final_mask_coverage: float | None
    reviewer_disposition: str


@dataclass(slots=True, frozen=True)
class ReviewRecord:
    schema_version: int
    review_id: str
    processing_manifest_sha256: str
    output_sha256: str
    review_status: str
    reviewer: str
    reviewed_at: str
    full_clip_reviewed: bool = False
    flagged_intervals_reviewed: bool = False
    notes: str = ""


@dataclass(slots=True, frozen=True)
class ProcessingState:
    schema_version: int
    run_id: str
    clip_id: str
    mode: str
    state: str
    completed_layers: tuple[str, ...] = ()
    reusable_layers: tuple[str, ...] = ()
    last_error: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class ProcessingManifest:
    schema_version: int
    code_version: str
    run_id: str
    clip_id: str
    mode: str
    processing_state: str
    audit_status: str
    review_status: str
    stage1: StageIInput
    dino_artifact: ArtifactRef
    sam_mask_shards: tuple[ArtifactRef, ...]
    render_artifact: ArtifactRef
    manual_seed_artifacts: tuple[ArtifactRef, ...] = ()
    label_artifacts: tuple[ArtifactRef, ...] = ()
    evidence_artifacts: tuple[ArtifactRef, ...] = ()
    review_flag_artifacts: tuple[ArtifactRef, ...] = ()
    review_record: ArtifactRef | None = None


@dataclass(slots=True, frozen=True)
class RunPaths:
    root: Path
    state: Path
    manifest: Path
    dino: Path
    dino_checkpoint: Path
    sam_dir: Path
    frames_dir: Path
    render: Path
    private_root: Path
    reviews_dir: Path
    stop_request: Path


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Stage2Error(
            "INVALID_JSON_VALUE",
            "Artifact data is not finite or JSON-serializable.",
            details={"cause": str(exc)},
        ) from exc
    return encoded.encode("utf-8")


def fingerprint(kind: str, payload: dict[str, Any]) -> str:
    layer_code_version = {
        "dino": DINO_CODE_VERSION,
        "sam": SAM_CODE_VERSION,
        "render": RENDER_CODE_VERSION,
    }.get(kind, STAGE2_CODE_VERSION)
    envelope = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "code_version": layer_code_version,
        "kind": kind,
        "payload": payload,
    }
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def dino_fingerprint(payload: dict[str, Any]) -> str:
    return fingerprint("dino", payload)


def sam_fingerprint(payload: dict[str, Any]) -> str:
    return fingerprint("sam", payload)


def render_fingerprint(payload: dict[str, Any]) -> str:
    return fingerprint("render", payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Stage2Error(
            "FILE_READ_FAILED",
            f"Could not read {path}",
            details={"path": str(path), "cause": str(exc)},
        ) from exc
    return digest.hexdigest()


def sha256_directory_tree(
    root: Path,
    *,
    description: str = "installed SAM2 package",
    unverifiable_code: str = "SAM_RUNTIME_UNVERIFIABLE",
    changed_code: str = "SAM_RUNTIME_CHANGED_DURING_HASH",
) -> str:
    """Hash every file under root by relative path and content, TOCTOU-guarded.

    Generic over which asset directory is being hashed (the installed SAM2
    package, a persistent DINO snapshot, ...); callers customize the error
    codes and wording so a DINO problem doesn't get reported as a SAM one.
    """
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise Stage2Error(
            unverifiable_code,
            f"Could not resolve the {description} directory: {root}",
        ) from exc
    if not root.is_dir():
        raise Stage2Error(
            unverifiable_code,
            f"The {description} path is not a directory: {root}",
        )

    def runtime_files() -> list[Path]:
        found: list[Path] = []
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise Stage2Error(
                    unverifiable_code,
                    f"The {description} contains a symbolic link: {path}",
                )
            if path.is_file():
                found.append(path)
        return sorted(found, key=lambda item: item.relative_to(root).as_posix())

    files = runtime_files()
    if not files:
        raise Stage2Error(
            unverifiable_code,
            f"The {description} contains no hashable files.",
        )
    before_stamps = {path: file_stamp(path) for path in files}
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    if runtime_files() != files or any(file_stamp(path) != before_stamps[path] for path in files):
        raise Stage2Error(
            changed_code,
            f"The {description} changed while its identity was being calculated.",
            recovery="Stop the installer or updater, then retry.",
        )
    return digest.hexdigest()


def _normalized_git_url(value: str) -> str:
    normalized = value.removeprefix("git+").rstrip("/")
    return normalized.removesuffix(".git")


def installed_sam_runtime_revision(package_root: Path) -> str:
    """Read the VCS revision from package metadata, with an editable-clone fallback."""
    try:
        distribution = importlib_metadata.distribution("SAM-2")
    except importlib_metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text:
            try:
                direct_url = json.loads(direct_url_text)
                repository = str(direct_url["url"])
                revision = str(direct_url["vcs_info"]["commit_id"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                if _normalized_git_url(repository) != _normalized_git_url(SAM_RUNTIME_REPOSITORY):
                    raise Stage2Error(
                        "SAM_RUNTIME_REPOSITORY_MISMATCH",
                        "Installed SAM2 runtime came from an unexpected repository.",
                        details={
                            "expected": SAM_RUNTIME_REPOSITORY,
                            "actual": repository,
                        },
                    )
                return revision

    repository_root = package_root.resolve().parent
    if (repository_root / ".git").exists():
        try:
            revision_result = subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            remote_result = subprocess.run(
                ["git", "-C", str(repository_root), "remote", "get-url", "origin"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            revision_result = remote_result = None
        if (
            revision_result is not None
            and remote_result is not None
            and revision_result.returncode == 0
            and remote_result.returncode == 0
        ):
            repository = remote_result.stdout.strip()
            if _normalized_git_url(repository) != _normalized_git_url(SAM_RUNTIME_REPOSITORY):
                raise Stage2Error(
                    "SAM_RUNTIME_REPOSITORY_MISMATCH",
                    "Editable SAM2 runtime came from an unexpected repository.",
                    details={"expected": SAM_RUNTIME_REPOSITORY, "actual": repository},
                )
            return revision_result.stdout.strip()
    raise Stage2Error(
        "SAM_RUNTIME_REVISION_UNVERIFIABLE",
        "Could not prove which SAM2 source revision is installed.",
        recovery=(
            "Install SAM2 from the pinned Git commit using the Stage II setup command, then retry."
        ),
    )


def verified_sam_runtime_identity(
    package_root: Path,
    *,
    revision: str,
    torch_version: str,
    cuda_version: str | None,
) -> SamRuntimeIdentity:
    if revision != SAM_RUNTIME_REVISION:
        raise Stage2Error(
            "SAM_RUNTIME_REVISION_MISMATCH",
            "Installed SAM2 source revision does not match the pinned runtime.",
            details={"expected": SAM_RUNTIME_REVISION, "actual": revision},
        )
    try:
        package_root = package_root.resolve(strict=True)
        config_path = (package_root / SAM_MODEL_CONFIG).resolve(strict=True)
        config_path.relative_to(package_root)
    except (OSError, ValueError) as exc:
        raise Stage2Error(
            "SAM_MODEL_CONFIG_MISSING",
            "Pinned SAM2 model configuration is missing from the installed runtime.",
            details={"config": SAM_MODEL_CONFIG},
        ) from exc
    config_sha256 = sha256_file(config_path)
    if config_sha256 != SAM_MODEL_CONFIG_SHA256:
        raise Stage2Error(
            "SAM_MODEL_CONFIG_HASH_MISMATCH",
            "Installed SAM2 model configuration does not match the pinned configuration.",
            details={
                "config": SAM_MODEL_CONFIG,
                "expected": SAM_MODEL_CONFIG_SHA256,
                "actual": config_sha256,
            },
        )
    if not isinstance(torch_version, str) or not torch_version.strip():
        raise Stage2Error(
            "SAM_RUNTIME_UNVERIFIABLE",
            "Installed Torch runtime did not report a version.",
        )
    source_tree_sha256 = sha256_directory_tree(package_root)
    if sha256_file(config_path) != config_sha256:
        raise Stage2Error(
            "SAM_RUNTIME_CHANGED_DURING_HASH",
            "SAM2 model configuration changed while runtime identity was calculated.",
            recovery="Stop the installer or updater, then retry.",
        )
    return SamRuntimeIdentity(
        repository=SAM_RUNTIME_REPOSITORY,
        revision=revision,
        source_tree_sha256=source_tree_sha256,
        model_config=SAM_MODEL_CONFIG,
        model_config_sha256=config_sha256,
        torch_version=torch_version,
        cuda_version=str(cuda_version or "none"),
    )


def file_stamp(path: Path) -> tuple[int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise Stage2Error(
            "FILE_STAT_FAILED",
            f"Could not inspect {path}",
            details={"path": str(path), "cause": str(exc)},
        ) from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def artifact_ref(path: Path) -> ArtifactRef:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise Stage2Error("NOT_A_FILE", f"Expected a regular file: {path}")
    before = file_stamp(path)
    size = before[2]
    if size <= 0:
        raise Stage2Error("EMPTY_ARTIFACT", f"Artifact is empty: {path}")
    digest = sha256_file(path)
    if file_stamp(path) != before:
        raise Stage2Error(
            "INPUT_CHANGED_DURING_READ",
            f"Artifact changed while it was being hashed: {path}",
            recovery="Stop the writer or transfer, then retry with a stable input file.",
        )
    return ArtifactRef(path=str(path), sha256=digest, bytes=size)


def artifact_set_sha256(artifacts: tuple[ArtifactRef, ...]) -> str:
    if not artifacts:
        raise Stage2Error("EMPTY_ARTIFACT_SET", "Artifact set must not be empty.")
    payload = tuple({"sha256": artifact.sha256, "bytes": artifact.bytes} for artifact in artifacts)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def resolve_input_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise Stage2Error(
            "INPUT_NOT_FOUND",
            f"{label} does not exist or cannot be resolved: {path}",
            details={"path": str(path), "cause": str(exc)},
        ) from exc
    if not resolved.is_file():
        raise Stage2Error("NOT_A_FILE", f"{label} is not a regular file: {resolved}")
    return resolved


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise Stage2Error(
            "ATOMIC_WRITE_FAILED",
            f"Could not atomically write {path}",
            details={"path": str(path), "cause": str(exc)},
        ) from exc


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def write_immutable_json(path: Path, value: Any) -> ArtifactRef:
    expected = canonical_json_bytes(value) + b"\n"
    if path.exists():
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise Stage2Error(
                "ARTIFACT_READ_FAILED",
                f"Could not verify existing artifact {path}",
                details={"cause": str(exc)},
            ) from exc
        if actual != expected:
            raise Stage2Error(
                "IMMUTABLE_ARTIFACT_CONFLICT",
                f"Existing immutable artifact does not match this run: {path}",
                details={
                    "path": str(path),
                    "existing_sha256": hashlib.sha256(actual).hexdigest(),
                    "expected_sha256": hashlib.sha256(expected).hexdigest(),
                },
                recovery="Use a new run ID or explicitly invalidate the affected layer.",
            )
    else:
        atomic_write_bytes(path, expected)
    return artifact_ref(path)


def write_immutable_bytes(path: Path, content: bytes) -> ArtifactRef:
    if not content:
        raise Stage2Error("EMPTY_ARTIFACT", f"Refusing to write an empty artifact: {path}")
    if path.exists():
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise Stage2Error(
                "ARTIFACT_READ_FAILED",
                f"Could not verify existing artifact {path}",
                details={"cause": str(exc)},
            ) from exc
        if actual != content:
            raise Stage2Error(
                "IMMUTABLE_ARTIFACT_CONFLICT",
                f"Existing immutable artifact does not match this run: {path}",
                details={
                    "path": str(path),
                    "existing_sha256": hashlib.sha256(actual).hexdigest(),
                    "expected_sha256": hashlib.sha256(content).hexdigest(),
                },
                recovery="Use a new run ID or explicitly invalidate the affected layer.",
            )
    else:
        atomic_write_bytes(path, content)
    return artifact_ref(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2Error(
            "INVALID_JSON",
            f"Could not read valid JSON from {path}",
            details={"path": str(path), "cause": str(exc)},
        ) from exc
    if not isinstance(value, dict):
        raise Stage2Error("INVALID_JSON_SHAPE", f"Expected a JSON object in {path}")
    return value


def safe_component(value: str, label: str) -> str:
    if value in {".", ".."} or not _SAFE_COMPONENT.fullmatch(value):
        raise Stage2Error(
            "UNSAFE_PATH_COMPONENT",
            f"Unsafe {label}: {value!r}",
            details={label: value},
            recovery=f"Use only letters, numbers, dot, underscore, and dash for {label}.",
        )
    return value


def build_run_paths(work_dir: Path, run_id: str, clip_id: str) -> RunPaths:
    run_id = safe_component(run_id, "run_id")
    clip_id = safe_component(clip_id, "clip_id")
    work_dir = work_dir.expanduser().resolve()
    root = work_dir / "runs" / run_id / clip_id
    for candidate in (work_dir / "runs", work_dir / "runs" / run_id, root):
        if candidate.is_symlink():
            raise Stage2Error(
                "UNSAFE_OUTPUT_LOCATION",
                f"Stage II output path contains a symbolic link: {candidate}",
                recovery="Remove the symbolic link or choose a different work directory.",
            )
    resolved_root = root.resolve(strict=False)
    try:
        resolved_root.relative_to(work_dir)
    except ValueError as exc:
        raise Stage2Error(
            "UNSAFE_OUTPUT_LOCATION",
            "Stage II output path escapes the selected work directory.",
            details={"work_dir": str(work_dir), "output": str(resolved_root)},
        ) from exc
    return RunPaths(
        root=resolved_root,
        state=resolved_root / "state.json",
        manifest=resolved_root / "processing.manifest.json",
        dino=resolved_root / "dino" / "artifact.json",
        dino_checkpoint=resolved_root / "dino" / "checkpoint.jsonl",
        sam_dir=resolved_root / "sam",
        frames_dir=resolved_root / "frames",
        render=resolved_root / "render" / "artifact.json",
        private_root=resolved_root / "DO-NOT-SHIP",
        reviews_dir=resolved_root / "reviews",
        stop_request=resolved_root / "control" / "stop-request.json",
    )


def assert_run_root_safe(paths: RunPaths) -> None:
    for candidate in (paths.root.parents[1], paths.root.parents[0], paths.root):
        if candidate.is_symlink() or candidate.resolve(strict=False) != candidate:
            raise Stage2Error(
                "UNSAFE_OUTPUT_LOCATION",
                f"Stage II run path became symbolic or redirected: {candidate}",
                recovery="Restore the real run directory or choose a new work directory.",
            )


def _parse_rate(value: str | None, *, path: Path, field_name: str) -> float:
    if not value:
        raise Stage2Error(
            "VIDEO_PROBE_INVALID",
            f"ffprobe omitted {field_name} for {path}",
        )
    try:
        numerator, denominator = [*value.split("/"), "1"][:2]
        result = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError) as exc:
        raise Stage2Error(
            "VIDEO_PROBE_INVALID",
            f"ffprobe reported unusable {field_name}={value!r} for {path}",
        ) from exc
    if not math.isfinite(result) or result <= 0:
        raise Stage2Error(
            "VIDEO_PROBE_INVALID",
            f"ffprobe reported unusable {field_name}={value!r} for {path}",
        )
    return result


def probe_video(path: Path) -> VideoFacts:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise Stage2Error(
            "FFPROBE_NOT_FOUND",
            "ffprobe was not found on PATH",
            recovery="Run the Stage II setup command, then retry validation.",
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,nb_frames,duration,rotation",
        "-show_entries",
        "stream_tags=rotate",
        "-show_entries",
        "format=duration",
        "-count_packets",
        "-of",
        "json",
        str(path.resolve(strict=True)),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise Stage2Error(
            "FFPROBE_FAILED",
            f"ffprobe failed for {path}",
            details={"stderr": result.stderr.strip()},
        )
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise Stage2Error(
            "VIDEO_PROBE_INVALID", f"ffprobe found no usable video in {path}"
        ) from exc

    r_fps = _parse_rate(stream.get("r_frame_rate"), path=path, field_name="r_frame_rate")
    avg_fps = _parse_rate(stream.get("avg_frame_rate"), path=path, field_name="avg_frame_rate")
    is_cfr = abs(r_fps - avg_fps) / r_fps <= 0.001
    try:
        rotation = int(stream.get("rotation", stream.get("tags", {}).get("rotate", 0)) or 0)
        coded_width = int(stream["width"])
        coded_height = int(stream["height"])
        frame_value = stream.get("nb_read_packets") or stream.get("nb_frames")
        duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
        n_frames = (
            int(frame_value) if frame_value not in {None, "", "N/A"} else round(duration * r_fps)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error(
            "VIDEO_PROBE_INVALID",
            f"ffprobe reported malformed video facts for {path}",
            details={"stream": stream},
        ) from exc
    if rotation % 180:
        display_width, display_height = coded_height, coded_width
    else:
        display_width, display_height = coded_width, coded_height
    if n_frames <= 0 or duration <= 0:
        raise Stage2Error(
            "VIDEO_PROBE_INVALID",
            f"ffprobe reported no usable frames/duration for {path}",
            details={"n_frames": n_frames, "duration_s": duration},
        )
    return VideoFacts(
        coded_width=coded_width,
        coded_height=coded_height,
        display_width=display_width,
        display_height=display_height,
        fps=r_fps,
        n_frames=n_frames,
        duration_s=duration,
        rotation=rotation,
        is_cfr=is_cfr,
    )


def _expect_equal(actual: Any, expected: Any, field_name: str, problems: list[str]) -> None:
    if isinstance(expected, float):
        equal = isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-9
        )
    else:
        equal = actual == expected
    if not equal:
        problems.append(f"{field_name}: expected {expected!r}, got {actual!r}")


def _expect_video_matches_manifest(
    facts: VideoFacts,
    manifest_source: dict[str, Any],
    label: str,
    problems: list[str],
) -> None:
    if not facts.is_cfr:
        problems.append(f"{label}: video is variable-frame-rate")
    _expect_equal(facts.display_width, manifest_source.get("width"), f"{label}.width", problems)
    _expect_equal(facts.display_height, manifest_source.get("height"), f"{label}.height", problems)
    manifest_fps = manifest_source.get("fps")
    if not isinstance(manifest_fps, (int, float)) or not math.isclose(
        facts.fps, float(manifest_fps), rel_tol=0.001, abs_tol=0.001
    ):
        problems.append(f"{label}.fps: manifest={manifest_fps!r}, probed={facts.fps!r}")
    _expect_equal(facts.n_frames, manifest_source.get("n_frames"), f"{label}.n_frames", problems)


def validate_stage1(
    source_video: Path,
    stage1_video: Path,
    stage1_manifest_path: Path,
    *,
    probe_fn: Callable[[Path], VideoFacts] = probe_video,
) -> StageIInput:
    source_video = resolve_input_file(source_video, "Source video")
    stage1_video = resolve_input_file(stage1_video, "Stage I video")
    stage1_manifest_path = resolve_input_file(stage1_manifest_path, "Stage I manifest")
    if len({source_video, stage1_video, stage1_manifest_path}) != 3:
        raise Stage2Error(
            "INPUT_PATH_COLLISION",
            "Source video, Stage I video, and Stage I manifest must be different files.",
        )

    initial_stamps = {
        source_video: file_stamp(source_video),
        stage1_video: file_stamp(stage1_video),
        stage1_manifest_path: file_stamp(stage1_manifest_path),
    }
    source_ref = artifact_ref(source_video)
    stage1_ref = artifact_ref(stage1_video)
    manifest_ref = artifact_ref(stage1_manifest_path)
    manifest = read_json(stage1_manifest_path)
    problems: list[str] = []
    warnings: list[str] = []

    _expect_equal(manifest.get("schema_version"), 1, "schema_version", problems)
    clip_id = manifest.get("clip_id")
    try:
        safe_component(clip_id, "clip_id")
    except (Stage2Error, TypeError):
        problems.append(f"clip_id is unsafe or missing: {clip_id!r}")
        clip_id = "invalid"

    source = manifest.get("source")
    output = manifest.get("output")
    egoblur = manifest.get("egoblur")
    audit = manifest.get("audit")
    if not isinstance(source, dict):
        problems.append("source must be an object")
        source = {}
    if not isinstance(output, dict):
        problems.append("output must be an object")
        output = {}
    if not isinstance(egoblur, dict):
        problems.append("egoblur must be an object")
        egoblur = {}
    if not isinstance(audit, dict):
        problems.append("audit must be an object")
        audit = {}

    _expect_equal(source.get("sha256"), source_ref.sha256, "source.sha256", problems)
    _expect_equal(source.get("filename"), source_video.name, "source.filename", problems)
    _expect_equal(output.get("sha256"), stage1_ref.sha256, "output.sha256", problems)
    _expect_equal(output.get("bytes"), stage1_ref.bytes, "output.bytes", problems)
    _expect_equal(output.get("path"), stage1_video.name, "output.path", problems)

    source_facts = probe_fn(source_video)
    stage1_facts = probe_fn(stage1_video)
    _expect_video_matches_manifest(source_facts, source, "source", problems)
    manifest_duration = source.get("duration_s")
    duration_tolerance = 1.0 / source_facts.fps
    if not isinstance(manifest_duration, (int, float)) or not math.isclose(
        source_facts.duration_s,
        float(manifest_duration),
        rel_tol=0.0,
        abs_tol=duration_tolerance,
    ):
        problems.append(
            f"source.duration_s: manifest={manifest_duration!r}, probed={source_facts.duration_s!r}"
        )
    if source_facts.rotation != source.get("rotation"):
        problems.append(
            f"source.rotation: manifest={source.get('rotation')!r}, probed={source_facts.rotation!r}"
        )
    if not stage1_facts.is_cfr:
        problems.append("Stage I output is variable-frame-rate")
    _expect_equal(
        stage1_facts.display_width, source_facts.display_width, "Stage I display width", problems
    )
    _expect_equal(
        stage1_facts.display_height,
        source_facts.display_height,
        "Stage I display height",
        problems,
    )
    _expect_equal(stage1_facts.n_frames, source_facts.n_frames, "Stage I frame count", problems)
    if not math.isclose(stage1_facts.fps, source_facts.fps, rel_tol=0.001, abs_tol=0.001):
        problems.append(f"Stage I fps: source={source_facts.fps!r}, output={stage1_facts.fps!r}")
    if not math.isclose(
        stage1_facts.duration_s,
        source_facts.duration_s,
        rel_tol=0.0,
        abs_tol=duration_tolerance,
    ):
        problems.append(
            "Stage I duration: "
            f"source={source_facts.duration_s!r}, output={stage1_facts.duration_s!r}"
        )

    for key, expected in EXPECTED_STAGE1.items():
        _expect_equal(egoblur.get(key), expected, f"egoblur.{key}", problems)

    status = manifest.get("status")
    allowed_statuses = {"PASS_AUTOMATED", "PASS_AUTOMATED_NO_YUNET", "NEEDS_REVIEW"}
    if status not in allowed_statuses:
        problems.append(f"status: expected a technically complete status, got {status!r}")
    _expect_equal(audit.get("status"), status, "audit.status", problems)
    if audit.get("integrity_ran") is not True:
        problems.append("audit.integrity_ran must be true")
    checked = audit.get("fill_integrity_checked")
    if not isinstance(checked, int) or isinstance(checked, bool) or checked < 0:
        problems.append(
            f"audit.fill_integrity_checked must be a non-negative integer, got {checked!r}"
        )
    integrity_frames = audit.get("fill_integrity_frames")
    if integrity_frames != source_facts.n_frames:
        problems.append(
            "audit.fill_integrity_frames must equal the source frame count: "
            f"expected {source_facts.n_frames}, got {integrity_frames!r}"
        )

    reasons = audit.get("status_reasons", [])
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        problems.append("audit.status_reasons must be a list of strings")
        reasons = []
    if checked == 0 and (
        status != "NEEDS_REVIEW"
        or not any("ZERO frames had a FACE redacted" in reason for reason in reasons)
    ):
        problems.append(
            "audit.fill_integrity_checked may be zero only for a NEEDS_REVIEW clip "
            "with EgoBlur's zero-face-redaction review reason"
        )
    if status == "NEEDS_REVIEW":
        warnings.extend(f"Stage I review reason: {reason}" for reason in reasons)
    violations = audit.get("fill_integrity_violations")
    if not isinstance(violations, int) or violations < 0:
        problems.append("audit.fill_integrity_violations must be a non-negative integer")
    elif violations:
        warnings.append(
            f"Stage I reported {violations} legacy fill-integrity threshold findings; "
            "preserved for review rather than treated as proof of exposed pixels."
        )
    if not egoblur.get("face_weights_sha256"):
        warnings.append("Historical Stage I manifest does not prove the EgoBlur face-weight hash.")

    for path, initial_stamp in initial_stamps.items():
        if file_stamp(path) != initial_stamp:
            problems.append(f"input changed while validation was running: {path}")

    if problems:
        raise Stage2Error(
            "STAGE1_VALIDATION_FAILED",
            "Stage I inputs are incompatible with the Stage II contract.",
            details={"problems": problems},
            recovery="Regenerate Stage I with the frozen EgoBlur settings or select matching artifacts.",
        )

    return StageIInput(
        schema_version=STAGE2_SCHEMA_VERSION,
        clip_id=clip_id,
        source=source_ref,
        stage1_video=stage1_ref,
        stage1_manifest=manifest_ref,
        source_video=source_facts,
        stage1_output_video=stage1_facts,
        stage1_status=status,
        stage1_audit_reasons=tuple(reasons),
        egoblur=dict(egoblur),
        warnings=tuple(warnings),
    )


def load_state(path: Path) -> ProcessingState | None:
    if not path.exists():
        return None
    raw = read_json(path)
    try:
        state = ProcessingState(
            schema_version=int(raw["schema_version"]),
            run_id=str(raw["run_id"]),
            clip_id=str(raw["clip_id"]),
            mode=str(raw["mode"]),
            state=str(raw["state"]),
            completed_layers=tuple(raw.get("completed_layers", [])),
            reusable_layers=tuple(raw.get("reusable_layers", [])),
            last_error=raw.get("last_error"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_STATE", f"State file is malformed: {path}") from exc
    if state.schema_version != STAGE2_SCHEMA_VERSION or state.state not in PROCESSING_STATES:
        raise Stage2Error("INVALID_STATE", f"State file is incompatible: {path}")
    _validate_state_record(state, path)
    return state


def _validate_state_record(state: ProcessingState, path: Path) -> None:
    allowed_layers = ("dino", "sam", "render")
    required_layers = {
        "PENDING": (),
        "VALIDATED": (),
        "DINO_COMPLETE": ("dino",),
        "SAM_COMPLETE": ("dino", "sam"),
        "RENDER_COMPLETE": ("dino", "sam", "render"),
        "PROCESSING_COMPLETE": ("dino", "sam", "render"),
    }
    try:
        safe_component(state.run_id, "run_id")
        safe_component(state.clip_id, "clip_id")
    except Stage2Error as exc:
        raise Stage2Error("INVALID_STATE", f"State identity is unsafe: {path}") from exc
    if len(set(state.completed_layers)) != len(state.completed_layers) or any(
        layer not in allowed_layers for layer in state.completed_layers
    ):
        raise Stage2Error("INVALID_STATE", f"State has invalid completed layers: {path}")
    if len(set(state.reusable_layers)) != len(state.reusable_layers) or any(
        layer not in state.completed_layers for layer in state.reusable_layers
    ):
        raise Stage2Error("INVALID_STATE", f"State has invalid reusable layers: {path}")
    required = required_layers.get(state.state)
    if required is not None and not set(required).issubset(state.completed_layers):
        raise Stage2Error(
            "INVALID_STATE",
            f"State {state.state} is missing required completed layers at {path}",
        )


def transition_state(
    path: Path,
    *,
    run_id: str,
    clip_id: str,
    mode: str,
    target: str,
    completed_layers: tuple[str, ...] = (),
    reusable_layers: tuple[str, ...] = (),
    last_error: dict[str, Any] | None = None,
) -> ProcessingState:
    if target not in PROCESSING_STATES:
        raise Stage2Error("INVALID_STATE_TRANSITION", f"Unknown target state {target!r}")
    current = load_state(path)
    if current is not None:
        if (current.run_id, current.clip_id, current.mode) != (run_id, clip_id, mode):
            raise Stage2Error("STATE_IDENTITY_MISMATCH", f"State identity changed at {path}")
        if current.state == "FAILED" and target != "FAILED":
            raise Stage2Error(
                "INVALID_STATE_TRANSITION",
                "A failed state requires explicit recovery before it can advance.",
            )
        if target != "FAILED" and _STATE_ORDER[target] < _STATE_ORDER[current.state]:
            raise Stage2Error(
                "INVALID_STATE_TRANSITION",
                f"State may not move backward from {current.state} to {target}",
            )
    state = ProcessingState(
        schema_version=STAGE2_SCHEMA_VERSION,
        run_id=run_id,
        clip_id=clip_id,
        mode=mode,
        state=target,
        completed_layers=completed_layers,
        reusable_layers=reusable_layers,
        last_error=last_error,
    )
    _validate_state_record(state, path)
    atomic_write_json(path, state)
    return state


def advance_state(
    path: Path,
    *,
    run_id: str,
    clip_id: str,
    mode: str,
    target: str,
    completed_layers: tuple[str, ...] = (),
    reusable_layers: tuple[str, ...] = (),
) -> ProcessingState:
    """Advance a run, or preserve an already-later state during resume.

    `transition_state` remains strict so accidental backward transitions are
    errors. Orchestration uses this helper because a compatible resume is
    expected to walk the same layers again while verifying immutable files.
    """
    if target not in _STATE_ORDER:
        raise Stage2Error("INVALID_STATE_TRANSITION", f"Unknown advancing state {target!r}")
    current = load_state(path)
    if current is not None:
        if (current.run_id, current.clip_id, current.mode) != (run_id, clip_id, mode):
            raise Stage2Error("STATE_IDENTITY_MISMATCH", f"State identity changed at {path}")
        if current.state == "FAILED":
            raise Stage2Error(
                "INVALID_STATE_TRANSITION",
                "A failed state requires explicit recovery before it can advance.",
            )
        if _STATE_ORDER[current.state] >= _STATE_ORDER[target]:
            return current
    return transition_state(
        path,
        run_id=run_id,
        clip_id=clip_id,
        mode=mode,
        target=target,
        completed_layers=completed_layers,
        reusable_layers=reusable_layers,
    )


def validate_dino_config(config: DinoGenerationConfig) -> None:
    problems = []
    if not config.prompt.strip() or not config.prompt.endswith("."):
        problems.append("prompt must be non-empty and end with a period")
    for name, value in (
        ("proposal_floor", config.proposal_floor),
        ("text_threshold", config.text_threshold),
        ("nms_iou", config.nms_iou),
    ):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            problems.append(f"{name} must be finite and in (0, 1], got {value!r}")
    if config.anchor_spacing < 1:
        problems.append("anchor_spacing must be at least 1")
    if config.tile_rows < 1 or config.tile_cols < 1:
        problems.append("tile_rows and tile_cols must be at least 1")
    if not math.isfinite(config.tile_overlap) or not 0.0 <= config.tile_overlap < 1.0:
        problems.append("tile_overlap must be finite and in [0, 1)")
    if config.view_batch_size < 1:
        problems.append("view_batch_size must be at least 1")
    preprocessing_keys = [key for key, _value in config.preprocessing]
    if len(preprocessing_keys) != len(set(preprocessing_keys)):
        problems.append("preprocessing keys must be unique")
    if not re.fullmatch(r"[0-9a-f]{64}", config.model.sha256):
        problems.append("model.sha256 must be a lowercase 64-character SHA-256")
    if problems:
        raise Stage2Error(
            "INVALID_DINO_CONFIG",
            "DINO configuration is invalid.",
            details={"problems": problems},
        )


def anchor_schedule(n_frames: int, spacing: int) -> tuple[int, ...]:
    if n_frames < 1:
        raise Stage2Error("INVALID_ANCHOR_SCHEDULE", "n_frames must be at least 1")
    if spacing < 1:
        raise Stage2Error("INVALID_ANCHOR_SCHEDULE", "anchor spacing must be at least 1")
    anchors = list(range(0, n_frames, spacing))
    if anchors[-1] != n_frames - 1:
        anchors.append(n_frames - 1)
    return tuple(anchors)


def _axis_windows(length: int, count: int, overlap: float) -> tuple[tuple[int, int], ...]:
    if length < count:
        raise Stage2Error(
            "INVALID_TILE_LAYOUT",
            f"Cannot split length {length} into {count} positive tiles.",
        )
    if count == 1:
        return ((0, length),)
    tile_size = min(length, math.ceil(length / (count - (count - 1) * overlap)))
    travel = length - tile_size
    starts = [round(index * travel / (count - 1)) for index in range(count)]
    starts[0] = 0
    starts[-1] = travel
    return tuple((start, start + tile_size) for start in starts)


def tile_layout(
    width: int,
    height: int,
    *,
    rows: int = 2,
    cols: int = 2,
    overlap: float = DINO_TILE_OVERLAP,
) -> tuple[Tile, ...]:
    if width < 1 or height < 1:
        raise Stage2Error("INVALID_TILE_LAYOUT", "Frame dimensions must be positive.")
    if rows < 1 or cols < 1 or not 0.0 <= overlap < 1.0:
        raise Stage2Error("INVALID_TILE_LAYOUT", "Rows, columns, or overlap are invalid.")
    x_windows = _axis_windows(width, cols, overlap)
    y_windows = _axis_windows(height, rows, overlap)
    return tuple(
        Tile(
            origin=f"tile-r{row}-c{col}",
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        for row, (y1, y2) in enumerate(y_windows)
        for col, (x1, x2) in enumerate(x_windows)
    )


def _image_size(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise Stage2Error(
            "INVALID_FRAME",
            "Decoded frame must expose PIL-compatible size=(width, height).",
        )
    width, height = int(size[0]), int(size[1])
    if width < 1 or height < 1:
        raise Stage2Error("INVALID_FRAME", f"Decoded frame has invalid size {size!r}")
    return width, height


def frame_views(
    image: Any, *, rows: int, cols: int, overlap: float
) -> tuple[tuple[Tile, Any], ...]:
    width, height = _image_size(image)
    full = Tile(origin="full-frame", x1=0, y1=0, x2=width, y2=height)
    views: list[tuple[Tile, Any]] = [(full, image)]
    for tile in tile_layout(width, height, rows=rows, cols=cols, overlap=overlap):
        try:
            crop = image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
        except (AttributeError, TypeError, ValueError) as exc:
            raise Stage2Error(
                "INVALID_FRAME",
                "Decoded frame does not support PIL-compatible crop().",
            ) from exc
        if _image_size(crop) != (tile.width, tile.height):
            raise Stage2Error(
                "INVALID_FRAME",
                f"Crop {tile.origin} returned the wrong dimensions.",
            )
        views.append((tile, crop))
    return tuple(views)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _source_class(origins: tuple[str, ...]) -> str:
    has_full = "full-frame" in origins
    has_tile = any(origin.startswith("tile-") for origin in origins)
    if has_full and has_tile:
        return "shared"
    if has_full:
        return "full-frame-only"
    return "tiled-only"


def _proposal_id(proposal: Proposal) -> str:
    payload = {
        "frame_idx": proposal.frame_idx,
        "box": proposal.box,
        "score": proposal.score,
        "label": proposal.label,
        "origins": proposal.origins,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]


def _finalize_proposal(proposal: Proposal) -> Proposal:
    origins = tuple(sorted(set(proposal.origins)))
    if not origins or any(
        origin != "full-frame" and not re.fullmatch(r"tile-r\d+-c\d+", origin) for origin in origins
    ):
        raise Stage2Error(
            "INVALID_DINO_OUTPUT",
            f"DINO proposal has invalid origins: {proposal.origins!r}",
        )
    updated = replace(proposal, source=_source_class(origins), origins=origins, proposal_id="")
    return replace(updated, proposal_id=_proposal_id(updated))


def map_raw_detection(
    raw: RawDinoDetection,
    *,
    frame_idx: int,
    tile: Tile,
    frame_width: int,
    frame_height: int,
    proposal_floor: float,
) -> Proposal | None:
    try:
        score = float(raw.score)
        local = tuple(float(value) for value in raw.box)
    except (TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_DINO_OUTPUT", "DINO returned non-numeric output.") from exc
    if len(local) != 4 or not all(math.isfinite(value) for value in (*local, score)):
        raise Stage2Error("INVALID_DINO_OUTPUT", "DINO returned non-finite box/score values.")
    if not 0.0 <= score <= 1.0:
        raise Stage2Error("INVALID_DINO_OUTPUT", f"DINO returned score outside [0, 1]: {score}")
    if score < proposal_floor:
        return None
    x1 = min(max(local[0], 0.0), float(tile.width))
    y1 = min(max(local[1], 0.0), float(tile.height))
    x2 = min(max(local[2], 0.0), float(tile.width))
    y2 = min(max(local[3], 0.0), float(tile.height))
    if x2 <= x1 or y2 <= y1:
        raise Stage2Error(
            "INVALID_DINO_OUTPUT",
            f"DINO returned a degenerate/outside box {raw.box!r} for {tile.origin}.",
        )
    global_box = (
        min(max(x1 + tile.x1, 0.0), float(frame_width)),
        min(max(y1 + tile.y1, 0.0), float(frame_height)),
        min(max(x2 + tile.x1, 0.0), float(frame_width)),
        min(max(y2 + tile.y1, 0.0), float(frame_height)),
    )
    label = str(raw.label).strip() or "face"
    return _finalize_proposal(
        Proposal(
            frame_idx=frame_idx,
            box=global_box,
            score=score,
            source="",
            label=label,
            origins=(tile.origin,),
        )
    )


def union_nms(proposals: tuple[Proposal, ...], iou_threshold: float) -> tuple[Proposal, ...]:
    if not 0.0 < iou_threshold <= 1.0:
        raise Stage2Error("INVALID_DINO_CONFIG", "NMS IoU must be in (0, 1].")
    ordered = sorted(
        proposals,
        key=lambda proposal: (
            proposal.frame_idx,
            -proposal.score,
            0 if "full-frame" in proposal.origins else 1,
            proposal.box,
            proposal.label,
        ),
    )
    kept: list[Proposal] = []
    for candidate in ordered:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(kept)
                if existing.frame_idx == candidate.frame_idx
                and existing.label == candidate.label
                and box_iou(existing.box, candidate.box) > iou_threshold
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(_finalize_proposal(candidate))
            continue
        existing = kept[duplicate_index]
        kept[duplicate_index] = _finalize_proposal(
            replace(existing, origins=existing.origins + candidate.origins)
        )
    return tuple(
        sorted(
            kept, key=lambda proposal: (proposal.frame_idx, -proposal.score, proposal.proposal_id)
        )
    )


def dino_fingerprint_payload(
    stage1: StageIInput,
    config: DinoGenerationConfig,
    anchors: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "source_sha256": stage1.source.sha256,
        "source_video": _jsonable(stage1.source_video),
        "model": _jsonable(config.model),
        "prompt": config.prompt,
        "proposal_floor": config.proposal_floor,
        "text_threshold": config.text_threshold,
        "anchor_spacing": config.anchor_spacing,
        "anchor_frames": anchors,
        "tiling": {
            "rows": config.tile_rows,
            "cols": config.tile_cols,
            "overlap": config.tile_overlap,
        },
        "nms_iou": config.nms_iou,
        "view_batch_size": config.view_batch_size,
        "preprocessing": dict(config.preprocessing),
    }


def _proposal_from_dict(raw: dict[str, Any]) -> Proposal:
    try:
        proposal = Proposal(
            frame_idx=int(raw["frame_idx"]),
            box=tuple(float(value) for value in raw["box"]),
            score=float(raw["score"]),
            source=str(raw["source"]),
            label=str(raw.get("label", "face")),
            proposal_id=str(raw["proposal_id"]),
            origins=tuple(str(value) for value in raw["origins"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_DINO_ARTIFACT", "A stored DINO proposal is malformed.") from exc
    try:
        canonical = _finalize_proposal(proposal)
    except Stage2Error as exc:
        raise Stage2Error(
            "INVALID_DINO_ARTIFACT",
            f"Stored DINO proposal {proposal.proposal_id!r} has invalid provenance.",
        ) from exc
    if len(proposal.box) != 4 or proposal != canonical:
        raise Stage2Error(
            "INVALID_DINO_ARTIFACT",
            f"Stored DINO proposal {proposal.proposal_id!r} failed canonical validation.",
        )
    return proposal


def _model_identity_from_dict(raw: dict[str, Any]) -> ModelIdentity:
    try:
        return ModelIdentity(
            name=str(raw["name"]), revision=str(raw["revision"]), sha256=str(raw["sha256"])
        )
    except (KeyError, TypeError) as exc:
        raise Stage2Error("INVALID_DINO_ARTIFACT", "Stored model identity is malformed.") from exc


def dino_meta_from_dict(raw: dict[str, Any]) -> DinoArtifactMeta:
    try:
        meta = DinoArtifactMeta(
            schema_version=int(raw["schema_version"]),
            artifact_type=str(raw["artifact_type"]),
            fingerprint=str(raw["fingerprint"]),
            source_sha256=str(raw["source_sha256"]),
            model=_model_identity_from_dict(raw["model"]),
            prompt=str(raw["prompt"]),
            proposal_floor=float(raw["proposal_floor"]),
            text_threshold=float(raw["text_threshold"]),
            anchor_spacing=int(raw["anchor_spacing"]),
            anchor_frames=tuple(int(value) for value in raw["anchor_frames"]),
            tiling=dict(raw["tiling"]),
            nms_iou=float(raw["nms_iou"]),
            preprocessing=dict(raw["preprocessing"]),
            proposals=tuple(_proposal_from_dict(value) for value in raw["proposals"]),
            metrics=dict(raw["metrics"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, Stage2Error):
            raise
        raise Stage2Error("INVALID_DINO_ARTIFACT", "Stored DINO artifact is malformed.") from exc
    if meta.schema_version != STAGE2_SCHEMA_VERSION or meta.artifact_type != "dino_proposals":
        raise Stage2Error("INVALID_DINO_ARTIFACT", "Stored DINO artifact is incompatible.")
    return meta


def validate_dino_meta(
    meta: DinoArtifactMeta,
    *,
    stage1: StageIInput,
    config: DinoGenerationConfig,
    anchors: tuple[int, ...],
    fingerprint_value: str,
) -> None:
    problems: list[str] = []
    expected_tiling = {
        "rows": config.tile_rows,
        "cols": config.tile_cols,
        "overlap": config.tile_overlap,
    }
    expected = {
        "fingerprint": fingerprint_value,
        "source_sha256": stage1.source.sha256,
        "model": config.model,
        "prompt": config.prompt,
        "proposal_floor": config.proposal_floor,
        "text_threshold": config.text_threshold,
        "anchor_spacing": config.anchor_spacing,
        "anchor_frames": anchors,
        "tiling": expected_tiling,
        "nms_iou": config.nms_iou,
        "preprocessing": dict(config.preprocessing),
    }
    for name, expected_value in expected.items():
        if getattr(meta, name) != expected_value:
            problems.append(f"{name}: expected {expected_value!r}, got {getattr(meta, name)!r}")
    seen_ids: set[str] = set()
    source_counts = {"full-frame-only": 0, "tiled-only": 0, "shared": 0}
    for proposal in meta.proposals:
        if proposal.frame_idx not in anchors:
            problems.append(f"proposal {proposal.proposal_id} uses a non-anchor frame")
        if not math.isfinite(proposal.score) or not config.proposal_floor <= proposal.score <= 1.0:
            problems.append(f"proposal {proposal.proposal_id} score is outside stored bounds")
        x1, y1, x2, y2 = proposal.box
        if not (
            all(math.isfinite(value) for value in proposal.box)
            and 0.0 <= x1 < x2 <= stage1.source_video.display_width
            and 0.0 <= y1 < y2 <= stage1.source_video.display_height
        ):
            problems.append(f"proposal {proposal.proposal_id} box is outside the source frame")
        if proposal.proposal_id in seen_ids:
            problems.append(f"duplicate proposal id {proposal.proposal_id}")
        seen_ids.add(proposal.proposal_id)
        if proposal.source not in source_counts:
            problems.append(f"proposal {proposal.proposal_id} has invalid source class")
        else:
            source_counts[proposal.source] += 1
    if meta.metrics.get("n_anchors") != len(anchors):
        problems.append("metrics.n_anchors does not match the anchor schedule")
    if meta.metrics.get("n_proposals") != len(meta.proposals):
        problems.append("metrics.n_proposals does not match the proposal list")
    if meta.metrics.get("source_counts") != source_counts:
        problems.append("metrics.source_counts does not match proposal provenance")
    if problems:
        raise Stage2Error(
            "INVALID_DINO_ARTIFACT",
            "Stored DINO artifact failed provenance validation.",
            details={"problems": problems},
            recovery="Explicitly recompute from the DINO layer.",
        )


def select_dino_proposals(
    meta: DinoArtifactMeta, operating_threshold: float
) -> DinoThresholdSelection:
    if (
        not math.isfinite(operating_threshold)
        or not meta.proposal_floor <= operating_threshold <= 1.0
    ):
        raise Stage2Error(
            "INVALID_DINO_OPERATING_THRESHOLD",
            "DINO operating threshold must be finite and no lower than the stored proposal floor.",
            details={
                "proposal_floor": meta.proposal_floor,
                "operating_threshold": operating_threshold,
            },
            recovery="Recompute DINO with a lower proposal floor before selecting this threshold.",
        )
    accepted = tuple(
        proposal for proposal in meta.proposals if proposal.score >= operating_threshold
    )
    rejected = tuple(
        proposal for proposal in meta.proposals if proposal.score < operating_threshold
    )
    return DinoThresholdSelection(
        threshold=operating_threshold,
        accepted=accepted,
        rejected=rejected,
    )


def _checkpoint_header(fingerprint_value: str) -> dict[str, Any]:
    return {
        "record_type": "header",
        "schema_version": STAGE2_SCHEMA_VERSION,
        "fingerprint": fingerprint_value,
    }


def _checkpoint_row_from_dict(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        if raw["record_type"] != "anchor":
            raise ValueError("not an anchor row")
        row = {
            "record_type": "anchor",
            "frame_idx": int(raw["frame_idx"]),
            "proposals": tuple(_proposal_from_dict(value) for value in raw["proposals"]),
            "metrics": dict(raw["metrics"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, Stage2Error):
            raise
        raise Stage2Error("INVALID_DINO_CHECKPOINT", "DINO checkpoint row is malformed.") from exc
    if any(proposal.frame_idx != row["frame_idx"] for proposal in row["proposals"]):
        raise Stage2Error(
            "INVALID_DINO_CHECKPOINT",
            f"Checkpoint anchor {row['frame_idx']} contains proposals from another frame.",
        )
    return row


def _write_checkpoint_records(path: Path, records: list[dict[str, Any]]) -> None:
    content = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    atomic_write_bytes(path, content)


def load_dino_checkpoint(
    path: Path,
    *,
    fingerprint_value: str,
    anchors: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    header = _checkpoint_header(fingerprint_value)
    if not path.exists():
        _write_checkpoint_records(path, [header])
        return {}
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise Stage2Error("INVALID_DINO_CHECKPOINT", f"Could not read {path}") from exc
    raw_lines = content.splitlines()
    if not raw_lines:
        raise Stage2Error("INVALID_DINO_CHECKPOINT", f"DINO checkpoint is empty: {path}")
    records: list[dict[str, Any]] = []
    repaired_tail = False
    for index, line in enumerate(raw_lines):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(raw_lines) - 1:
                repaired_tail = True
                break
            raise Stage2Error(
                "INVALID_DINO_CHECKPOINT",
                f"DINO checkpoint has corruption before its final row: {path}",
            ) from exc
        if not isinstance(record, dict):
            raise Stage2Error("INVALID_DINO_CHECKPOINT", "Checkpoint record must be an object.")
        records.append(record)
    if not records or records[0] != header:
        actual = records[0] if records else None
        raise Stage2Error(
            "DINO_FINGERPRINT_MISMATCH",
            "DINO checkpoint belongs to different inputs or settings.",
            details={"expected_header": header, "actual_header": actual},
            recovery="Use a new run ID or explicitly recompute from the DINO layer.",
        )
    rows: dict[int, dict[str, Any]] = {}
    allowed_anchors = set(anchors)
    for raw in records[1:]:
        row = _checkpoint_row_from_dict(raw)
        frame_idx = row["frame_idx"]
        if frame_idx not in allowed_anchors:
            raise Stage2Error(
                "INVALID_DINO_CHECKPOINT",
                f"Checkpoint contains unexpected anchor frame {frame_idx}.",
            )
        if frame_idx in rows:
            raise Stage2Error(
                "INVALID_DINO_CHECKPOINT",
                f"Checkpoint contains duplicate anchor frame {frame_idx}.",
            )
        rows[frame_idx] = row
    if repaired_tail or (content and not content.endswith(b"\n")):
        repaired = [header]
        repaired.extend(
            {
                "record_type": "anchor",
                "frame_idx": row["frame_idx"],
                "proposals": row["proposals"],
                "metrics": row["metrics"],
            }
            for row in sorted(rows.values(), key=lambda value: value["frame_idx"])
        )
        _write_checkpoint_records(path, repaired)
    return rows


def append_dino_checkpoint(path: Path, row: dict[str, Any]) -> None:
    encoded = canonical_json_bytes(row) + b"\n"
    try:
        with path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise Stage2Error(
            "DINO_CHECKPOINT_WRITE_FAILED",
            f"Could not persist DINO anchor checkpoint to {path}",
        ) from exc


def _adapter_peak_vram(adapter: Any) -> int:
    getter = getattr(adapter, "peak_vram_bytes", None)
    if not callable(getter):
        return 0
    value = int(getter())
    return max(0, value)


def _adapter_reset_peak_vram(adapter: Any) -> None:
    resetter = getattr(adapter, "reset_peak_vram", None)
    if callable(resetter):
        resetter()


def _run_dino_anchor(
    *,
    frame_idx: int,
    image: Any,
    adapter: Any,
    config: DinoGenerationConfig,
    expected_width: int,
    expected_height: int,
) -> tuple[tuple[Proposal, ...], dict[str, Any]]:
    if _image_size(image) != (expected_width, expected_height):
        raise Stage2Error(
            "FRAME_SIZE_MISMATCH",
            f"Frame {frame_idx} dimensions do not match the validated source video.",
        )
    views = frame_views(
        image, rows=config.tile_rows, cols=config.tile_cols, overlap=config.tile_overlap
    )
    _adapter_reset_peak_vram(adapter)
    started = time.perf_counter()
    mapped: list[Proposal] = []
    below_floor = 0
    model_calls = 0
    for start in range(0, len(views), config.view_batch_size):
        batch = views[start : start + config.view_batch_size]
        try:
            results = adapter.infer_batch(
                [view_image for _tile, view_image in batch],
                prompt=config.prompt,
                box_threshold=config.proposal_floor,
                text_threshold=config.text_threshold,
            )
        except Stage2Error:
            raise
        except Exception as exc:
            raise Stage2Error(
                "DINO_INFERENCE_FAILED",
                f"DINO inference failed at anchor frame {frame_idx}.",
                details={"cause": str(exc)},
            ) from exc
        model_calls += 1
        if not isinstance(results, (list, tuple)) or len(results) != len(batch):
            raise Stage2Error(
                "INVALID_DINO_OUTPUT",
                "DINO adapter returned a different number of results than input views.",
            )
        for (tile, _view_image), detections in zip(batch, results, strict=True):
            if not isinstance(detections, (list, tuple)):
                raise Stage2Error("INVALID_DINO_OUTPUT", "DINO detections must be a sequence.")
            for raw in detections:
                if not isinstance(raw, RawDinoDetection):
                    raise Stage2Error(
                        "INVALID_DINO_OUTPUT", "DINO adapter returned an unknown detection type."
                    )
                proposal = map_raw_detection(
                    raw,
                    frame_idx=frame_idx,
                    tile=tile,
                    frame_width=expected_width,
                    frame_height=expected_height,
                    proposal_floor=config.proposal_floor,
                )
                if proposal is None:
                    below_floor += 1
                else:
                    mapped.append(proposal)
    proposals = union_nms(tuple(mapped), config.nms_iou)
    source_counts = {
        source: sum(1 for proposal in proposals if proposal.source == source)
        for source in ("full-frame-only", "tiled-only", "shared")
    }
    metrics = {
        "n_views": len(views),
        "n_model_calls": model_calls,
        "n_raw_proposals": len(mapped) + below_floor,
        "n_below_floor": below_floor,
        "n_before_nms": len(mapped),
        "n_after_nms": len(proposals),
        "n_nms_suppressed": len(mapped) - len(proposals),
        "source_counts": source_counts,
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_bytes": _adapter_peak_vram(adapter),
    }
    return proposals, metrics


def generate_dino_proposals(
    *,
    stage1: StageIInput,
    paths: RunPaths,
    config: DinoGenerationConfig,
    adapter: Any,
    frame_loader: Callable[[int], Any],
) -> DinoGenerationResult:
    validate_dino_config(config)
    if getattr(adapter, "identity", None) != config.model:
        raise Stage2Error(
            "DINO_MODEL_IDENTITY_MISMATCH",
            "Loaded DINO adapter identity does not match the fingerprinted configuration.",
        )
    anchors = anchor_schedule(stage1.source_video.n_frames, config.anchor_spacing)
    fingerprint_payload = dino_fingerprint_payload(stage1, config, anchors)
    fingerprint_value = dino_fingerprint(fingerprint_payload)

    if paths.dino.exists():
        meta = dino_meta_from_dict(read_json(paths.dino))
        if meta.fingerprint != fingerprint_value:
            raise Stage2Error(
                "DINO_FINGERPRINT_MISMATCH",
                "Existing DINO artifact belongs to different inputs or settings.",
                details={"expected": fingerprint_value, "actual": meta.fingerprint},
                recovery="Use a new run ID or explicitly recompute from the DINO layer.",
            )
        validate_dino_meta(
            meta,
            stage1=stage1,
            config=config,
            anchors=anchors,
            fingerprint_value=fingerprint_value,
        )
        checkpoint_sha256 = meta.metrics.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or not paths.dino_checkpoint.exists():
            raise Stage2Error(
                "INVALID_DINO_ARTIFACT",
                "DINO artifact has no verifiable checkpoint provenance.",
            )
        if sha256_file(paths.dino_checkpoint) != checkpoint_sha256:
            raise Stage2Error(
                "INVALID_DINO_ARTIFACT",
                "DINO checkpoint changed after the final artifact was created.",
                recovery="Explicitly recompute from the DINO layer.",
            )
        checkpoint_rows = load_dino_checkpoint(
            paths.dino_checkpoint,
            fingerprint_value=fingerprint_value,
            anchors=anchors,
        )
        if set(checkpoint_rows) != set(anchors):
            raise Stage2Error(
                "INVALID_DINO_ARTIFACT",
                "DINO checkpoint does not contain every finalized anchor.",
                recovery="Explicitly recompute from the DINO layer.",
            )
        checkpoint_proposals = tuple(
            proposal
            for frame_idx in anchors
            for proposal in checkpoint_rows[frame_idx]["proposals"]
        )
        if checkpoint_proposals != meta.proposals:
            raise Stage2Error(
                "INVALID_DINO_ARTIFACT",
                "DINO proposal artifact does not match its inference checkpoint.",
                recovery="Explicitly recompute from the DINO layer.",
            )
        return DinoGenerationResult(
            artifact=artifact_ref(paths.dino),
            meta=meta,
            reused_final_artifact=True,
            reused_anchor_count=len(anchors),
            generated_anchor_count=0,
        )

    rows = load_dino_checkpoint(
        paths.dino_checkpoint,
        fingerprint_value=fingerprint_value,
        anchors=anchors,
    )
    reused_anchor_count = len(rows)
    for frame_idx in anchors:
        if frame_idx in rows:
            continue
        try:
            image = frame_loader(frame_idx)
        except Stage2Error:
            raise
        except Exception as exc:
            raise Stage2Error(
                "FRAME_DECODE_FAILED",
                f"Could not decode DINO anchor frame {frame_idx}.",
                details={"cause": str(exc)},
            ) from exc
        proposals, metrics = _run_dino_anchor(
            frame_idx=frame_idx,
            image=image,
            adapter=adapter,
            config=config,
            expected_width=stage1.source_video.display_width,
            expected_height=stage1.source_video.display_height,
        )
        row = {
            "record_type": "anchor",
            "frame_idx": frame_idx,
            "proposals": proposals,
            "metrics": metrics,
        }
        append_dino_checkpoint(paths.dino_checkpoint, row)
        rows[frame_idx] = _checkpoint_row_from_dict(_jsonable(row))

    all_proposals = tuple(
        proposal for frame_idx in anchors for proposal in rows[frame_idx]["proposals"]
    )
    anchor_metrics = [rows[frame_idx]["metrics"] for frame_idx in anchors]
    source_counts = {
        source: sum(1 for proposal in all_proposals if proposal.source == source)
        for source in ("full-frame-only", "tiled-only", "shared")
    }
    metrics = {
        "n_anchors": len(anchors),
        "n_proposals": len(all_proposals),
        "n_model_calls": sum(int(value.get("n_model_calls", 0)) for value in anchor_metrics),
        "n_raw_proposals": sum(int(value.get("n_raw_proposals", 0)) for value in anchor_metrics),
        "n_below_floor": sum(int(value.get("n_below_floor", 0)) for value in anchor_metrics),
        "n_nms_suppressed": sum(int(value.get("n_nms_suppressed", 0)) for value in anchor_metrics),
        "source_counts": source_counts,
        "runtime_seconds": sum(
            float(value.get("runtime_seconds", 0.0)) for value in anchor_metrics
        ),
        "peak_vram_bytes": max(
            (int(value.get("peak_vram_bytes", 0)) for value in anchor_metrics), default=0
        ),
        "checkpoint_sha256": sha256_file(paths.dino_checkpoint),
    }
    meta = DinoArtifactMeta(
        schema_version=STAGE2_SCHEMA_VERSION,
        artifact_type="dino_proposals",
        fingerprint=fingerprint_value,
        source_sha256=stage1.source.sha256,
        model=config.model,
        prompt=config.prompt,
        proposal_floor=config.proposal_floor,
        text_threshold=config.text_threshold,
        anchor_spacing=config.anchor_spacing,
        anchor_frames=anchors,
        tiling={
            "rows": config.tile_rows,
            "cols": config.tile_cols,
            "overlap": config.tile_overlap,
        },
        nms_iou=config.nms_iou,
        preprocessing=dict(config.preprocessing),
        proposals=all_proposals,
        metrics=metrics,
    )
    validate_dino_meta(
        meta,
        stage1=stage1,
        config=config,
        anchors=anchors,
        fingerprint_value=fingerprint_value,
    )
    artifact = write_immutable_json(paths.dino, meta)
    return DinoGenerationResult(
        artifact=artifact,
        meta=meta,
        reused_final_artifact=False,
        reused_anchor_count=reused_anchor_count,
        generated_anchor_count=len(anchors) - reused_anchor_count,
    )


class TransformersGroundingDinoAdapter:
    """Lazy official Transformers adapter for a pinned local snapshot.

    Milestone 2 defines this boundary but does not install/download its GPU
    dependencies. The Stage II setup milestone will pin the runtime and
    verify the safetensors file before this adapter is exposed by a real run.
    """

    identity = ModelIdentity(
        name=DINO_MODEL_ID,
        revision=DINO_MODEL_REVISION,
        sha256=DINO_MODEL_WEIGHTS_SHA256,
    )

    def __init__(
        self,
        *,
        device: str = "cuda",
        local_files_only: bool = True,
        model_path: Path | None = None,
    ) -> None:
        verified_weights: ArtifactRef | None = None
        snapshot_sha256: str | None = None
        if model_path is not None:
            supplied = model_path.expanduser()
            if supplied.is_symlink():
                raise Stage2Error(
                    "DINO_SNAPSHOT_UNSAFE",
                    "The persistent DINO snapshot directory may not be a symbolic link.",
                )
            try:
                model_path = supplied.resolve(strict=True)
            except OSError as exc:
                raise Stage2Error(
                    "DINO_SNAPSHOT_MISSING",
                    "The verified persistent DINO snapshot is missing.",
                ) from exc
            if not model_path.is_dir():
                raise Stage2Error(
                    "DINO_SNAPSHOT_MISSING",
                    "The verified persistent DINO snapshot is not a directory.",
                )
            weights = model_path / "model.safetensors"
            if weights.is_symlink():
                raise Stage2Error(
                    "DINO_SNAPSHOT_UNSAFE",
                    "The persistent DINO weights may not be a symbolic link.",
                )
            verified_weights = artifact_ref(weights)
            if verified_weights.sha256 != DINO_MODEL_WEIGHTS_SHA256:
                raise Stage2Error(
                    "DINO_CHECKPOINT_HASH_MISMATCH",
                    "Persistent DINO weights do not match the pinned model.",
                    details={
                        "expected": DINO_MODEL_WEIGHTS_SHA256,
                        "actual": verified_weights.sha256,
                    },
                )
            # model.safetensors is the only file hash-pinned to a known-good
            # value, but from_pretrained also reads config.json,
            # preprocessor_config.json, and tokenizer files from this same
            # directory. Hash the whole snapshot so those files get the same
            # TOCTOU protection the weights already have, instead of loading
            # unverified — mirrors verified_sam_runtime_identity's use of
            # sha256_directory_tree for the installed SAM2 package.
            snapshot_sha256 = sha256_directory_tree(
                model_path,
                description="persistent DINO snapshot",
                unverifiable_code="DINO_SNAPSHOT_UNVERIFIABLE",
                changed_code="DINO_SNAPSHOT_CHANGED",
            )
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise Stage2Error(
                "DINO_DEPENDENCIES_MISSING",
                "Grounding DINO runtime dependencies are not installed.",
                recovery="Run the Stage II setup command before real DINO inference.",
            ) from exc
        self._torch = torch
        self.device = device
        pretrained_source = str(model_path) if model_path is not None else DINO_MODEL_ID
        pretrained_options: dict[str, Any] = {"local_files_only": local_files_only}
        if model_path is None:
            pretrained_options["revision"] = DINO_MODEL_REVISION
        try:
            self.processor = AutoProcessor.from_pretrained(
                pretrained_source,
                **pretrained_options,
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                pretrained_source,
                use_safetensors=True,
                **pretrained_options,
            ).to(device)
            self.model.eval()
        except Exception as exc:
            raise Stage2Error(
                "DINO_MODEL_LOAD_FAILED",
                "Pinned Grounding DINO Base model could not be loaded from the local cache.",
                details={"cause": str(exc), "revision": DINO_MODEL_REVISION},
                recovery="Run the Stage II setup command to verify and prime model assets.",
            ) from exc
        if verified_weights is not None and artifact_ref(Path(verified_weights.path)) != verified_weights:
            raise Stage2Error(
                "DINO_CHECKPOINT_CHANGED",
                "Persistent DINO weights changed while the model was loading.",
            )
        if model_path is not None and sha256_directory_tree(
            model_path,
            description="persistent DINO snapshot",
            unverifiable_code="DINO_SNAPSHOT_UNVERIFIABLE",
            changed_code="DINO_SNAPSHOT_CHANGED",
        ) != snapshot_sha256:
            raise Stage2Error(
                "DINO_SNAPSHOT_CHANGED",
                "Persistent DINO snapshot directory changed while the model was loading.",
            )
        self.snapshot_sha256 = snapshot_sha256

    def infer_batch(
        self,
        images: list[Any],
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> list[tuple[RawDinoDetection, ...]]:
        text = [prompt] * len(images)
        inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = [(image.size[1], image.size[0]) for image in images]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
            text_labels=[[prompt.rstrip(".")]] * len(images),
        )
        converted = []
        for result in results:
            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            labels = result.get("text_labels") or [prompt.rstrip(".")] * len(boxes)
            converted.append(
                tuple(
                    RawDinoDetection(box=tuple(box), score=float(score), label=str(label))
                    for box, score, label in zip(boxes, scores, labels, strict=True)
                )
            )
        return converted

    def reset_peak_vram(self) -> None:
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()

    def peak_vram_bytes(self) -> int:
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            return int(self._torch.cuda.max_memory_allocated())
        return 0


def validate_sam_config(config: SamGenerationConfig) -> None:
    problems: list[str] = []
    if not math.isfinite(config.accepted_proposal_threshold) or not (
        DINO_PROPOSAL_FLOOR <= config.accepted_proposal_threshold <= 1.0
    ):
        problems.append(
            f"accepted_proposal_threshold must be finite and in [{DINO_PROPOSAL_FLOOR}, 1]"
        )
    if config.window_size < 1:
        problems.append("window_size must be at least 1")
    if not 0 <= config.window_overlap < config.window_size:
        problems.append("window_overlap must be in [0, window_size)")
    elif config.window_overlap * 2 > config.window_size:
        problems.append("window_overlap must be at most half of window_size")
    if config.precision not in {"float32", "float16", "bfloat16"}:
        problems.append("precision must be float32, float16, or bfloat16")
    for name, value, lower, upper in (
        ("fallback_padding_scale", config.fallback_padding_scale, 0.0, 2.0),
        ("min_prompt_area_ratio", config.min_prompt_area_ratio, 0.0, 1.0),
        ("min_anchor_neighborhood_fraction", config.min_anchor_neighborhood_fraction, 0.0, 1.0),
    ):
        if not math.isfinite(value) or not lower <= value <= upper:
            problems.append(f"{name} must be finite and in [{lower}, {upper}]")
    if (
        not math.isfinite(config.max_frame_area_ratio)
        or not 0.0 < config.max_frame_area_ratio <= 1.0
    ):
        problems.append("max_frame_area_ratio must be finite and in (0, 1]")
    if not math.isfinite(config.max_prompt_area_ratio) or config.max_prompt_area_ratio < 1.0:
        problems.append("max_prompt_area_ratio must be finite and at least 1")
    if (
        not math.isfinite(config.anchor_neighborhood_scale)
        or config.anchor_neighborhood_scale < 1.0
    ):
        problems.append("anchor_neighborhood_scale must be finite and at least 1")
    if config.fallback_min_padding_px < 0:
        problems.append("fallback_min_padding_px must be non-negative")
    if config.min_mask_pixels < 1:
        problems.append("min_mask_pixels must be at least 1")
    if not re.fullmatch(r"[0-9a-f]{64}", config.model.sha256):
        problems.append("model.sha256 must be a lowercase 64-character SHA-256")
    if not isinstance(config.runtime, SamRuntimeIdentity):
        problems.append("runtime must use the versioned SamRuntimeIdentity schema")
    else:
        for name, value in (
            ("runtime.source_tree_sha256", config.runtime.source_tree_sha256),
            ("runtime.model_config_sha256", config.runtime.model_config_sha256),
        ):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                problems.append(f"{name} must be a lowercase 64-character SHA-256")
        for name, value in (
            ("runtime.repository", config.runtime.repository),
            ("runtime.revision", config.runtime.revision),
            ("runtime.model_config", config.runtime.model_config),
            ("runtime.torch_version", config.runtime.torch_version),
            ("runtime.cuda_version", config.runtime.cuda_version),
        ):
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{name} must be a non-empty string")
    if problems:
        raise Stage2Error(
            "INVALID_SAM_CONFIG",
            "SAM configuration is invalid.",
            details={"problems": problems},
        )


def temporal_windows(
    n_frames: int, window_size: int, window_overlap: int
) -> tuple[TemporalWindow, ...]:
    if (
        n_frames < 1
        or window_size < 1
        or not 0 <= window_overlap < window_size
        or window_overlap * 2 > window_size
    ):
        raise Stage2Error(
            "INVALID_SAM_WINDOW_LAYOUT",
            "Frame count, window size, or overlap is invalid.",
        )
    step = window_size - window_overlap
    windows: list[TemporalWindow] = []
    start = 0
    while start < n_frames:
        end = min(n_frames - 1, start + window_size - 1)
        windows.append(TemporalWindow(len(windows), start, end))
        if end == n_frames - 1:
            break
        start += step
    return tuple(windows)


def _valid_frame_box(box: Any, width: int, height: int) -> bool:
    if (
        not isinstance(box, (tuple, list))
        or len(box) != 4
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in box)
    ):
        return False
    x1, y1, x2, y2 = box
    return (
        all(math.isfinite(value) for value in (x1, y1, x2, y2))
        and 0.0 <= x1 < x2 <= width
        and 0.0 <= y1 < y2 <= height
    )


def padded_box(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    scale: float,
    min_padding_px: int,
) -> tuple[int, int, int, int]:
    if not _valid_frame_box(box, width, height):
        raise Stage2Error("INVALID_SAM_PROMPT", f"Prompt box is outside the frame: {box!r}")
    pad_x = max(float(min_padding_px), (box[2] - box[0]) * scale)
    pad_y = max(float(min_padding_px), (box[3] - box[1]) * scale)
    return (
        max(0, math.floor(box[0] - pad_x)),
        max(0, math.floor(box[1] - pad_y)),
        min(width, math.ceil(box[2] + pad_x)),
        min(height, math.ceil(box[3] + pad_y)),
    )


def validate_manual_seeds(
    seeds: tuple[ManualSeed, ...], stage1: StageIInput
) -> tuple[ManualSeed, ...]:
    seen: set[str] = set()
    validated: list[ManualSeed] = []
    for seed in seeds:
        if not isinstance(seed, ManualSeed):
            raise Stage2Error(
                "INVALID_MANUAL_SEED",
                "Manual seeds must use the versioned ManualSeed schema.",
            )
        problems = []
        if (
            not isinstance(seed.schema_version, int)
            or isinstance(seed.schema_version, bool)
            or seed.schema_version != STAGE2_SCHEMA_VERSION
        ):
            problems.append("schema_version")
        if seed.clip_id != stage1.clip_id:
            problems.append("clip_id")
        if not isinstance(seed.seed_id, str):
            problems.append("seed_id")
        else:
            if not _SAFE_COMPONENT.fullmatch(seed.seed_id) or seed.seed_id in {".", ".."}:
                problems.append("seed_id")
            if seed.seed_id in seen:
                problems.append("duplicate seed_id")
        if (
            not isinstance(seed.frame_idx, int)
            or isinstance(seed.frame_idx, bool)
            or not 0 <= seed.frame_idx < stage1.source_video.n_frames
        ):
            problems.append("frame_idx")
        if not _valid_frame_box(
            seed.box,
            stage1.source_video.display_width,
            stage1.source_video.display_height,
        ):
            problems.append("box")
        if not isinstance(seed.reason, str) or not seed.reason.strip():
            problems.append("reason")
        if problems:
            raise Stage2Error(
                "INVALID_MANUAL_SEED",
                f"Manual seed {seed.seed_id!r} is invalid.",
                details={"problems": problems},
            )
        seen.add(seed.seed_id)
        validated.append(seed)
    return tuple(sorted(validated, key=lambda seed: (seed.frame_idx, seed.seed_id)))


def validate_face_event_labels(
    labels: tuple[FaceEventLabel, ...], stage1: StageIInput
) -> tuple[FaceEventLabel, ...]:
    seen: set[str] = set()
    validated = []
    for label in labels:
        if not isinstance(label, FaceEventLabel):
            raise Stage2Error(
                "INVALID_PRIVATE_LABEL",
                "Private labels must use the versioned FaceEventLabel schema.",
            )
        problems = []
        if (
            not isinstance(label.schema_version, int)
            or isinstance(label.schema_version, bool)
            or label.schema_version != STAGE2_SCHEMA_VERSION
        ):
            problems.append("schema_version")
        if label.clip_id != stage1.clip_id:
            problems.append("clip_id")
        if (
            not isinstance(label.event_id, str)
            or label.event_id in {".", ".."}
            or not _SAFE_COMPONENT.fullmatch(label.event_id)
            or label.event_id in seen
        ):
            problems.append("event_id")
        if (
            not isinstance(label.frame_start, int)
            or isinstance(label.frame_start, bool)
            or not isinstance(label.frame_end, int)
            or isinstance(label.frame_end, bool)
            or not 0 <= label.frame_start <= label.frame_end < stage1.source_video.n_frames
        ):
            problems.append("frame_range")
        if not _valid_frame_box(
            label.conservative_box,
            stage1.source_video.display_width,
            stage1.source_video.display_height,
        ):
            problems.append("conservative_box")
        if label.label_kind not in {"face_event", "negative_example"}:
            problems.append("label_kind")
        if label.visibility not in {"visible", "partial", "occluded", "not_applicable"}:
            problems.append("visibility")
        if not isinstance(label.category, str) or not label.category.strip():
            problems.append("category")
        if label.stage1_verdict not in {
            "covered",
            "partial",
            "missed",
            "not_applicable",
        }:
            problems.append("stage1_verdict")
        if not isinstance(label.dino_proposal_verdicts, tuple):
            problems.append("dino_proposal_verdicts")
        else:
            proposal_ids = set()
            for verdict in label.dino_proposal_verdicts:
                if not isinstance(verdict, dict):
                    problems.append("dino_proposal_verdicts")
                    break
                proposal_id = verdict.get("proposal_id")
                if (
                    not isinstance(proposal_id, str)
                    or not proposal_id
                    or proposal_id in proposal_ids
                    or verdict.get("verdict")
                    not in {"face", "false_positive", "missed", "not_applicable"}
                ):
                    problems.append("dino_proposal_verdicts")
                    break
                proposal_ids.add(proposal_id)
        if label.final_mask_coverage is not None and (
            isinstance(label.final_mask_coverage, bool)
            or not isinstance(label.final_mask_coverage, (int, float))
            or not math.isfinite(label.final_mask_coverage)
            or not 0.0 <= label.final_mask_coverage <= 1.0
        ):
            problems.append("final_mask_coverage")
        if label.reviewer_disposition not in {
            "pending",
            "accepted",
            "rejected",
            "manual_seed_required",
        }:
            problems.append("reviewer_disposition")
        if label.label_kind == "negative_example" and label.stage1_verdict not in {
            "covered",
            "not_applicable",
        }:
            problems.append("negative_example_stage1_verdict")
        if problems:
            raise Stage2Error(
                "INVALID_PRIVATE_LABEL",
                f"Private label {label.event_id!r} is invalid.",
                details={"problems": sorted(set(problems))},
                recovery="Correct the private label file under DO-NOT-SHIP, then retry.",
            )
        seen.add(label.event_id)
        validated.append(label)
    return tuple(
        sorted(validated, key=lambda item: (item.frame_start, item.frame_end, item.event_id))
    )


def _private_path(paths: RunPaths, relative: Path) -> Path:
    assert_run_root_safe(paths)
    if relative.is_absolute() or ".." in relative.parts:
        raise Stage2Error("UNSAFE_PRIVATE_PATH", "Private artifact path is unsafe.")
    if paths.private_root.is_symlink():
        raise Stage2Error("UNSAFE_PRIVATE_PATH", "DO-NOT-SHIP may not be a symbolic link.")
    candidate = paths.private_root / relative
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(paths.private_root.resolve(strict=False))
    except ValueError as exc:
        raise Stage2Error("UNSAFE_PRIVATE_PATH", "Private artifact escaped DO-NOT-SHIP.") from exc
    current = paths.private_root
    for component in relative.parts[:-1]:
        current /= component
        if current.is_symlink():
            raise Stage2Error(
                "UNSAFE_PRIVATE_PATH", "Private artifact path contains a symbolic link."
            )
    return resolved


def _write_private_json(
    paths: RunPaths, *, artifact_type: str, payload: dict[str, Any]
) -> ArtifactRef:
    content = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        **payload,
    }
    digest = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    path = _private_path(paths, Path(artifact_type) / f"{artifact_type}-{digest}.json")
    return write_immutable_json(path, content)


def write_private_labels(
    paths: RunPaths,
    *,
    stage1: StageIInput,
    labels: tuple[FaceEventLabel, ...],
) -> ArtifactRef:
    validated = validate_face_event_labels(labels, stage1)
    return _write_private_json(
        paths,
        artifact_type="labels",
        payload={"clip_id": stage1.clip_id, "labels": validated},
    )


def write_private_manual_seeds(
    paths: RunPaths,
    *,
    stage1: StageIInput,
    seeds: tuple[ManualSeed, ...],
) -> ArtifactRef:
    validated = validate_manual_seeds(seeds, stage1)
    return _write_private_json(
        paths,
        artifact_type="manual-seeds",
        payload={"clip_id": stage1.clip_id, "seeds": validated},
    )


def load_private_labels_file(path: Path, stage1: StageIInput) -> tuple[FaceEventLabel, ...]:
    raw = read_json(resolve_input_file(path, "private labels file"))
    if (
        raw.get("schema_version") != STAGE2_SCHEMA_VERSION
        or raw.get("artifact_type") != "labels"
        or raw.get("clip_id") != stage1.clip_id
        or not isinstance(raw.get("labels"), list)
    ):
        raise Stage2Error(
            "INVALID_PRIVATE_LABEL",
            "Private labels file envelope is invalid.",
            recovery="Use the versioned labels envelope for this exact clip.",
        )
    labels = []
    try:
        for value in raw["labels"]:
            labels.append(
                FaceEventLabel(
                    schema_version=value["schema_version"],
                    event_id=value["event_id"],
                    clip_id=value["clip_id"],
                    frame_start=value["frame_start"],
                    frame_end=value["frame_end"],
                    conservative_box=tuple(value["conservative_box"]),
                    label_kind=value["label_kind"],
                    visibility=value["visibility"],
                    category=value["category"],
                    stage1_verdict=value["stage1_verdict"],
                    dino_proposal_verdicts=tuple(value["dino_proposal_verdicts"]),
                    final_mask_coverage=value.get("final_mask_coverage"),
                    reviewer_disposition=value["reviewer_disposition"],
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_PRIVATE_LABEL", "Private labels are malformed.") from exc
    return validate_face_event_labels(tuple(labels), stage1)


def load_private_manual_seeds_file(path: Path, stage1: StageIInput) -> tuple[ManualSeed, ...]:
    raw = read_json(resolve_input_file(path, "private manual seeds file"))
    if (
        raw.get("schema_version") != STAGE2_SCHEMA_VERSION
        or raw.get("artifact_type") != "manual-seeds"
        or raw.get("clip_id") != stage1.clip_id
        or not isinstance(raw.get("seeds"), list)
    ):
        raise Stage2Error(
            "INVALID_MANUAL_SEED",
            "Private manual-seed file envelope is invalid.",
            recovery="Use the versioned manual-seeds envelope for this exact clip.",
        )
    try:
        seeds = tuple(
            ManualSeed(
                schema_version=value["schema_version"],
                seed_id=value["seed_id"],
                clip_id=value["clip_id"],
                frame_idx=value["frame_idx"],
                box=tuple(value["box"]),
                reason=value["reason"],
            )
            for value in raw["seeds"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_MANUAL_SEED", "Private manual seeds are malformed.") from exc
    return validate_manual_seeds(seeds, stage1)


def write_private_review_flags(
    paths: RunPaths,
    *,
    clip_id: str,
    sam_shards: tuple[ArtifactRef, ...],
) -> ArtifactRef:
    safe_component(clip_id, "clip_id")
    flags = []
    for reference in sam_shards:
        loaded = load_sam_mask_shard(Path(reference.path), expected_artifact=reference)
        flags.extend(loaded.meta.review_flags)
    return _write_private_json(
        paths,
        artifact_type="review-flags",
        payload={
            "clip_id": clip_id,
            "sam_mask_set_sha256": artifact_set_sha256(sam_shards),
            "flags": tuple(sorted(set(flags))),
        },
    )


def register_private_evidence(
    paths: RunPaths,
    *,
    clip_id: str,
    evidence_id: str,
    evidence_kind: str,
    source: Path,
    frame_start: int,
    frame_end: int,
) -> tuple[ArtifactRef, ArtifactRef]:
    safe_component(clip_id, "clip_id")
    safe_component(evidence_id, "evidence_id")
    if not evidence_kind.strip() or not 0 <= frame_start <= frame_end:
        raise Stage2Error("INVALID_PRIVATE_EVIDENCE", "Evidence metadata is invalid.")
    source = resolve_input_file(source, "private evidence")
    source_ref = artifact_ref(source)
    suffix = (
        source.suffix.lower()
        if re.fullmatch(r"\.[a-z0-9]{1,10}", source.suffix.lower())
        else ".bin"
    )
    evidence_path = _private_path(
        paths,
        Path("evidence") / f"{evidence_id}-{source_ref.sha256}{suffix}",
    )
    try:
        content = source.read_bytes()
    except OSError as exc:
        raise Stage2Error(
            "PRIVATE_EVIDENCE_READ_FAILED", "Could not read private evidence."
        ) from exc
    evidence_ref = write_immutable_bytes(evidence_path, content)
    metadata_ref = _write_private_json(
        paths,
        artifact_type="evidence-records",
        payload={
            "clip_id": clip_id,
            "evidence_id": evidence_id,
            "evidence_kind": evidence_kind,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "evidence": evidence_ref,
        },
    )
    return evidence_ref, metadata_ref


def sam_prompts_for_window(
    window: TemporalWindow,
    proposals: tuple[Proposal, ...],
    manual_seeds: tuple[ManualSeed, ...],
) -> tuple[SamPrompt, ...]:
    prompts = [
        SamPrompt(
            prompt_id=f"dino-{proposal.proposal_id}",
            prompt_kind="dino",
            anchor_frame=proposal.frame_idx,
            box=proposal.box,
            source_id=proposal.proposal_id,
        )
        for proposal in proposals
        if window.frame_start <= proposal.frame_idx <= window.frame_end
    ]
    prompts.extend(
        SamPrompt(
            prompt_id=f"manual-{seed.seed_id}",
            prompt_kind="manual",
            anchor_frame=seed.frame_idx,
            box=seed.box,
            source_id=seed.seed_id,
        )
        for seed in manual_seeds
        if window.frame_start <= seed.frame_idx <= window.frame_end
    )
    prompts.sort(key=lambda prompt: (prompt.anchor_frame, prompt.prompt_kind, prompt.source_id))
    if len({prompt.prompt_id for prompt in prompts}) != len(prompts):
        raise Stage2Error("INVALID_SAM_PROMPT", "SAM prompt IDs are not unique within a window.")
    return tuple(prompts)


def _sam_prompt_records(prompts: tuple[SamPrompt, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {**_jsonable(prompt), "local_object_id": index + 1} for index, prompt in enumerate(prompts)
    )


def sam_window_fingerprint_payload(
    *,
    dino_artifact: ArtifactRef,
    dino_meta: DinoArtifactMeta,
    config: SamGenerationConfig,
    window: TemporalWindow,
    prompts: tuple[SamPrompt, ...],
    frame_payload_sha256: str | None = None,
    frame_payload_status: str = "metadata-only",
) -> dict[str, Any]:
    return {
        "dino_artifact_sha256": dino_artifact.sha256,
        "dino_fingerprint": dino_meta.fingerprint,
        "model": _jsonable(config.model),
        "runtime": _jsonable(config.runtime),
        "accepted_proposal_threshold": config.accepted_proposal_threshold,
        "prompt_conversion": "global-xyxy-box-to-local-object",
        "window": _jsonable(window),
        "window_layout": {
            "size": config.window_size,
            "overlap": config.window_overlap,
        },
        "frame_payload": {
            "sha256": frame_payload_sha256,
            "status": frame_payload_status,
        },
        "precision": config.precision,
        "fallback": {
            "padding_scale": config.fallback_padding_scale,
            "min_padding_px": config.fallback_min_padding_px,
        },
        "validation": {
            "min_mask_pixels": config.min_mask_pixels,
            "min_prompt_area_ratio": config.min_prompt_area_ratio,
            "max_prompt_area_ratio": config.max_prompt_area_ratio,
            "max_frame_area_ratio": config.max_frame_area_ratio,
            "anchor_neighborhood_scale": config.anchor_neighborhood_scale,
            "min_anchor_neighborhood_fraction": config.min_anchor_neighborhood_fraction,
        },
        "prompts": _sam_prompt_records(prompts),
    }


def _sam_frame_names(window: TemporalWindow) -> tuple[str, ...]:
    return tuple(f"{local_index:06d}.jpg" for local_index in range(window.n_frames))


def _inspect_sam_jpeg_frames(
    frame_dir: Path,
    *,
    window: TemporalWindow,
    width: int,
    height: int,
    include_attestation: bool,
) -> tuple[dict[str, Any], ...]:
    if frame_dir.is_symlink() or not frame_dir.is_dir():
        raise Stage2Error(
            "INVALID_FRAME_PAYLOAD",
            "Extracted SAM frame payload must be a real directory.",
        )
    expected_names = _sam_frame_names(window)
    allowed_names = set(expected_names)
    if include_attestation:
        allowed_names.add(SAM_FRAME_ATTESTATION_FILENAME)
    entries = tuple(frame_dir.iterdir())
    if any(path.is_symlink() for path in entries):
        raise Stage2Error(
            "INVALID_FRAME_PAYLOAD",
            "Extracted SAM frame payload may not contain symbolic links.",
        )
    actual_names = {path.name for path in entries}
    if actual_names != allowed_names:
        raise Stage2Error(
            "INVALID_FRAME_PAYLOAD",
            "Extracted SAM frame names or count do not match the requested window.",
            details={
                "expected": sorted(allowed_names),
                "actual": sorted(actual_names),
            },
        )
    try:
        from PIL import Image
    except ImportError as exc:
        raise Stage2Error(
            "PILLOW_NOT_FOUND",
            "Pillow is required to verify extracted SAM frames.",
        ) from exc
    records = []
    for local_index, name in enumerate(expected_names):
        path = frame_dir / name
        if not path.is_file() or path.is_symlink():
            raise Stage2Error("INVALID_FRAME_PAYLOAD", f"Extracted SAM frame is unsafe: {path}")
        before = artifact_ref(path)
        try:
            with Image.open(path) as image:
                image_size = image.size
                image_format = image.format
                image.verify()
        except Exception as exc:
            raise Stage2Error(
                "INVALID_FRAME_PAYLOAD",
                f"Extracted SAM frame is not a valid JPEG: {path}",
                details={"cause": str(exc)},
            ) from exc
        if image_format != "JPEG" or image_size != (width, height):
            raise Stage2Error(
                "INVALID_FRAME_PAYLOAD",
                "Extracted SAM frame dimensions or format are wrong.",
                details={
                    "path": str(path),
                    "expected_size": (width, height),
                    "actual_size": image_size,
                    "format": image_format,
                },
            )
        if artifact_ref(path) != before:
            raise Stage2Error(
                "FRAME_PAYLOAD_CHANGED",
                f"Extracted SAM frame changed during validation: {path}",
            )
        records.append(
            {
                "local_index": local_index,
                "global_index": window.frame_start + local_index,
                "name": name,
                "sha256": before.sha256,
                "bytes": before.bytes,
            }
        )
    return tuple(records)


def _frame_payload_identity(
    *,
    source_sha256: str,
    window: TemporalWindow,
    width: int,
    height: int,
    frames: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "artifact_type": "sam_frame_payload",
        "extraction_version": SAM_FRAME_EXTRACTION_VERSION,
        "source_sha256": source_sha256,
        "frame_start": window.frame_start,
        "frame_end": window.frame_end,
        "frame_width": width,
        "frame_height": height,
        "frames": _jsonable(frames),
    }


def load_attested_sam_window(
    frame_dir: Path,
    *,
    source_sha256: str,
    window: TemporalWindow,
    width: int,
    height: int,
) -> SamWindowInput:
    supplied = frame_dir.expanduser()
    if supplied.is_symlink():
        raise Stage2Error(
            "INVALID_FRAME_PAYLOAD", "SAM frame payload directory may not be a symlink."
        )
    try:
        frame_dir = supplied.resolve(strict=True)
    except OSError as exc:
        raise Stage2Error(
            "INVALID_FRAME_PAYLOAD", "SAM frame payload directory is missing."
        ) from exc
    records = _inspect_sam_jpeg_frames(
        frame_dir,
        window=window,
        width=width,
        height=height,
        include_attestation=True,
    )
    attestation_path = frame_dir / SAM_FRAME_ATTESTATION_FILENAME
    attestation_ref = artifact_ref(attestation_path)
    raw = read_json(attestation_path)
    identity = _frame_payload_identity(
        source_sha256=source_sha256,
        window=window,
        width=width,
        height=height,
        frames=records,
    )
    expected_fields = {*identity, "frame_payload_sha256"}
    if set(raw) != expected_fields or any(raw.get(key) != value for key, value in identity.items()):
        raise Stage2Error(
            "INVALID_FRAME_ATTESTATION",
            "Extracted SAM frame attestation does not match the source window.",
        )
    payload_sha256 = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    if raw.get("frame_payload_sha256") != payload_sha256:
        raise Stage2Error(
            "INVALID_FRAME_ATTESTATION",
            "Extracted SAM frame payload digest is invalid.",
        )
    frame_artifacts = tuple(artifact_ref(frame_dir / record["name"]) for record in records)
    if artifact_ref(attestation_path) != attestation_ref:
        raise Stage2Error(
            "FRAME_PAYLOAD_CHANGED",
            "Extracted SAM frame attestation changed during validation.",
        )
    return SamWindowInput(
        source_sha256=source_sha256,
        frame_start=window.frame_start,
        frame_end=window.frame_end,
        frame_width=width,
        frame_height=height,
        payload=frame_dir,
        frame_payload_sha256=payload_sha256,
        frame_artifacts=frame_artifacts,
        attestation_artifact=attestation_ref,
    )


def extract_attested_sam_window(
    *,
    stage1: StageIInput,
    paths: RunPaths,
    window: TemporalWindow,
) -> SamWindowInput:
    """Atomically extract and attest the exact JPEG sequence SAM2 will load."""
    assert_run_root_safe(paths)
    if not 0 <= window.frame_start <= window.frame_end < stage1.source_video.n_frames:
        raise Stage2Error("INVALID_FRAME_WINDOW", "SAM extraction window is out of bounds.")
    source_path = Path(stage1.source.path)
    before_source = file_stamp(source_path)
    frame_parent = paths.frames_dir / f"source-{stage1.source.sha256[:16]}"
    target = frame_parent / f"window-{window.frame_start:06d}-{window.frame_end:06d}"
    for candidate in (paths.frames_dir, frame_parent, target):
        if candidate.is_symlink():
            raise Stage2Error(
                "UNSAFE_FRAME_PAYLOAD_PATH",
                f"SAM frame payload path contains a symlink: {candidate}",
            )
    if target.exists():
        return load_attested_sam_window(
            target,
            source_sha256=stage1.source.sha256,
            window=window,
            width=stage1.source_video.display_width,
            height=stage1.source_video.display_height,
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise Stage2Error(
            "FFMPEG_NOT_FOUND",
            "ffmpeg is required to extract source-bound SAM windows.",
            recovery="Run the Stage II persistent setup command, then retry.",
        )
    frame_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=frame_parent, prefix=f".{target.name}.partial-"))
    promoted = False
    try:
        output_pattern = temporary / "%06d.jpg"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vf",
            f"select=between(n\\,{window.frame_start}\\,{window.frame_end})",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(window.n_frames),
            "-q:v",
            "2",
            "-start_number",
            "0",
            "-an",
            "-sn",
            "-dn",
            "-y",
            str(output_pattern),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise Stage2Error(
                "FRAME_EXTRACTION_FAILED",
                "ffmpeg could not extract the requested SAM frame window.",
                details={"stderr": result.stderr[-4000:], "window": _jsonable(window)},
            )
        records = _inspect_sam_jpeg_frames(
            temporary,
            window=window,
            width=stage1.source_video.display_width,
            height=stage1.source_video.display_height,
            include_attestation=False,
        )
        for record in records:
            frame_fd = os.open(temporary / record["name"], os.O_RDONLY)
            try:
                os.fsync(frame_fd)
            finally:
                os.close(frame_fd)
        identity = _frame_payload_identity(
            source_sha256=stage1.source.sha256,
            window=window,
            width=stage1.source_video.display_width,
            height=stage1.source_video.display_height,
            frames=records,
        )
        payload_sha256 = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        atomic_write_json(
            temporary / SAM_FRAME_ATTESTATION_FILENAME,
            {**identity, "frame_payload_sha256": payload_sha256},
        )
        if file_stamp(source_path) != before_source:
            raise Stage2Error(
                "SOURCE_VIDEO_CHANGED", "Original video changed during SAM frame extraction."
            )
        try:
            os.replace(temporary, target)
        except OSError as exc:
            raise Stage2Error(
                "FRAME_PAYLOAD_PROMOTION_FAILED",
                "Verified SAM frame payload could not be promoted atomically.",
                details={"cause": str(exc)},
            ) from exc
        promoted = True
        directory_fd = os.open(frame_parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)
    return load_attested_sam_window(
        target,
        source_sha256=stage1.source.sha256,
        window=window,
        width=stage1.source_video.display_width,
        height=stage1.source_video.display_height,
    )


def sam_shard_path(paths: RunPaths, window: TemporalWindow, fingerprint_value: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint_value):
        raise Stage2Error("INVALID_SAM_FINGERPRINT", "SAM fingerprint is not a SHA-256.")
    return paths.sam_dir / (
        f"window-{window.frame_start:06d}-{window.frame_end:06d}-{fingerprint_value}.npz"
    )


def validate_sam_window_input(
    value: Any, *, stage1: StageIInput, window: TemporalWindow
) -> SamWindowInput:
    if not isinstance(value, SamWindowInput):
        raise Stage2Error(
            "INVALID_SAM_WINDOW_INPUT",
            "SAM window loader must return source-bound window metadata.",
        )
    expected = (
        stage1.source.sha256,
        window.frame_start,
        window.frame_end,
        stage1.source_video.display_width,
        stage1.source_video.display_height,
    )
    actual = (
        value.source_sha256,
        value.frame_start,
        value.frame_end,
        value.frame_width,
        value.frame_height,
    )
    if actual != expected:
        raise Stage2Error(
            "INVALID_SAM_WINDOW_INPUT",
            "SAM window does not match the validated original source and frame range.",
            details={"expected": expected, "actual": actual},
        )
    if isinstance(value.payload, (str, Path)):
        verified = load_attested_sam_window(
            Path(value.payload),
            source_sha256=stage1.source.sha256,
            window=window,
            width=stage1.source_video.display_width,
            height=stage1.source_video.display_height,
        )
        if (
            value.frame_payload_sha256 != verified.frame_payload_sha256
            or value.frame_artifacts != verified.frame_artifacts
            or value.attestation_artifact != verified.attestation_artifact
        ):
            raise Stage2Error(
                "INVALID_FRAME_ATTESTATION",
                "SAM window metadata does not bind the exact extracted frame payload.",
            )
    elif (
        value.frame_payload_sha256 is not None
        or value.frame_artifacts
        or value.attestation_artifact is not None
    ):
        raise Stage2Error(
            "INVALID_FRAME_ATTESTATION",
            "Non-directory SAM input may not claim extracted-frame attestation.",
        )
    return value


def _pack_bits(values: bytes) -> bytes:
    packed = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value not in (0, 1):
            raise Stage2Error("INVALID_SAM_MASK", "Mask pixels must be binary 0/1 bytes.")
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return bytes(packed)


def _validate_packed_bits(packed: bytes, size: int) -> None:
    if size < 0 or len(packed) != (size + 7) // 8:
        raise Stage2Error("INVALID_SAM_SHARD", "Packed mask length is inconsistent.")
    if size % 8 and packed and packed[-1] & ((1 << (8 - size % 8)) - 1):
        raise Stage2Error("INVALID_SAM_SHARD", "Packed mask has nonzero padding bits.")


def _unpack_bits(packed: bytes, size: int) -> bytes:
    _validate_packed_bits(packed, size)
    return bytes((packed[index // 8] >> (7 - index % 8)) & 1 for index in range(size))


def _packed_bit(packed: bytes, index: int) -> int:
    return (packed[index // 8] >> (7 - index % 8)) & 1


def _packed_ones(size: int) -> bytes:
    if size < 0:
        raise Stage2Error("INVALID_SAM_SHARD", "Packed mask size cannot be negative.")
    complete, remaining = divmod(size, 8)
    tail = bytes([0xFF << (8 - remaining) & 0xFF]) if remaining else b""
    return b"\xff" * complete + tail


def _packed_mask_key(
    *,
    frame_idx: int,
    prompt_id: str,
    source: str,
    anchor_frame: int,
    crop_box: tuple[int, int, int, int],
    packed_bits: bytes,
) -> str:
    payload = {
        "frame_idx": frame_idx,
        "prompt_id": prompt_id,
        "source": source,
        "anchor_frame": anchor_frame,
        "crop_box": crop_box,
        "packed_sha256": hashlib.sha256(packed_bits).hexdigest(),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]


def rectangle_packed_mask(
    *,
    frame_idx: int,
    prompt: SamPrompt,
    crop_box: tuple[int, int, int, int],
    source: str,
) -> PackedMask:
    x1, y1, x2, y2 = crop_box
    area = (x2 - x1) * (y2 - y1)
    packed = _packed_ones(area)
    return PackedMask(
        key=_packed_mask_key(
            frame_idx=frame_idx,
            prompt_id=prompt.prompt_id,
            source=source,
            anchor_frame=prompt.anchor_frame,
            crop_box=crop_box,
            packed_bits=packed,
        ),
        frame_idx=frame_idx,
        prompt_id=prompt.prompt_id,
        source=source,
        anchor_frame=prompt.anchor_frame,
        crop_box=crop_box,
        packed_bits=packed,
        area_pixels=area,
    )


def _tighten_raw_mask(
    raw: RawSamMask, *, width: int, height: int
) -> tuple[tuple[int, int, int, int], bytes, int]:
    try:
        x1, y1, x2, y2 = (int(value) for value in raw.crop_box)
    except (TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_SAM_MASK", "SAM mask crop is malformed.") from exc
    if (x1, y1, x2, y2) != raw.crop_box or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise Stage2Error("INVALID_SAM_MASK", "SAM mask crop is outside the frame.")
    crop_width = x2 - x1
    crop_height = y2 - y1
    if not isinstance(raw.pixels, bytes) or len(raw.pixels) != crop_width * crop_height:
        raise Stage2Error("INVALID_SAM_MASK", "SAM mask pixel count is inconsistent.")
    nonzero_count = 0
    local_x1 = crop_width
    local_y1 = crop_height
    local_x2 = 0
    local_y2 = 0
    for index, value in enumerate(raw.pixels):
        if value not in (0, 1):
            raise Stage2Error("INVALID_SAM_MASK", "SAM mask pixels are not binary.")
        if value:
            x = index % crop_width
            y = index // crop_width
            nonzero_count += 1
            local_x1 = min(local_x1, x)
            local_y1 = min(local_y1, y)
            local_x2 = max(local_x2, x + 1)
            local_y2 = max(local_y2, y + 1)
    if nonzero_count == 0:
        return (x1, y1, x1, y1), b"", 0
    tightened = bytearray()
    for row in range(local_y1, local_y2):
        start = row * crop_width + local_x1
        tightened.extend(raw.pixels[start : start + (local_x2 - local_x1)])
    return (
        (x1 + local_x1, y1 + local_y1, x1 + local_x2, y1 + local_y2),
        bytes(tightened),
        nonzero_count,
    )


def _expanded_neighborhood(
    box: tuple[float, float, float, float], width: int, height: int, scale: float
) -> tuple[int, int, int, int]:
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    extra_x = box_width * (scale - 1.0) / 2.0
    extra_y = box_height * (scale - 1.0) / 2.0
    return (
        max(0, math.floor(box[0] - extra_x)),
        max(0, math.floor(box[1] - extra_y)),
        min(width, math.ceil(box[2] + extra_x)),
        min(height, math.ceil(box[3] + extra_y)),
    )


def assess_raw_sam_mask(
    raw: RawSamMask,
    *,
    prompt: SamPrompt,
    config: SamGenerationConfig,
    width: int,
    height: int,
) -> tuple[PackedMask | None, str | None]:
    if raw.direction not in {"anchor", "forward", "reverse"}:
        return None, "SAM_MASK_DIRECTION_INVALID"
    try:
        crop_box, pixels, area = _tighten_raw_mask(raw, width=width, height=height)
    except Stage2Error:
        return None, "SAM_MASK_MALFORMED"
    if area == 0:
        return None, "SAM_MASK_EMPTY"
    prompt_area = (prompt.box[2] - prompt.box[0]) * (prompt.box[3] - prompt.box[1])
    minimum = max(config.min_mask_pixels, math.ceil(prompt_area * config.min_prompt_area_ratio))
    if area < minimum:
        return None, "SAM_MASK_UNDERSIZED"
    if area > prompt_area * config.max_prompt_area_ratio:
        return None, "SAM_MASK_EXPLODED"
    if area >= width * height * config.max_frame_area_ratio:
        return None, "SAM_MASK_NEAR_FULL_FRAME"
    if raw.frame_idx == prompt.anchor_frame:
        neighborhood = _expanded_neighborhood(
            prompt.box, width, height, config.anchor_neighborhood_scale
        )
        crop_width = crop_box[2] - crop_box[0]
        inside = 0
        for index, value in enumerate(pixels):
            if not value:
                continue
            x = crop_box[0] + index % crop_width
            y = crop_box[1] + index // crop_width
            if neighborhood[0] <= x < neighborhood[2] and neighborhood[1] <= y < neighborhood[3]:
                inside += 1
        if inside / area < config.min_anchor_neighborhood_fraction:
            return None, "SAM_MASK_OFF_PROMPT"
    packed = _pack_bits(pixels)
    source = f"sam2-{raw.direction}"
    return (
        PackedMask(
            key=_packed_mask_key(
                frame_idx=raw.frame_idx,
                prompt_id=prompt.prompt_id,
                source=source,
                anchor_frame=prompt.anchor_frame,
                crop_box=crop_box,
                packed_bits=packed,
            ),
            frame_idx=raw.frame_idx,
            prompt_id=prompt.prompt_id,
            source=source,
            anchor_frame=prompt.anchor_frame,
            crop_box=crop_box,
            packed_bits=packed,
            area_pixels=area,
        ),
        None,
    )


def reassess_packed_sam_mask(
    mask: PackedMask,
    *,
    prompt: SamPrompt,
    config: SamGenerationConfig,
    width: int,
    height: int,
) -> str | None:
    direction = mask.source.removeprefix("sam2-")
    if direction not in {"anchor", "forward", "reverse"}:
        return "SAM_MASK_DIRECTION_INVALID"
    x1, y1, x2, y2 = mask.crop_box
    crop_width = x2 - x1
    crop_height = y2 - y1
    size = crop_width * crop_height
    try:
        _validate_packed_bits(mask.packed_bits, size)
    except Stage2Error:
        return "SAM_MASK_MALFORMED"
    if mask.area_pixels < 1:
        return "SAM_MASK_EMPTY"
    prompt_area = (prompt.box[2] - prompt.box[0]) * (prompt.box[3] - prompt.box[1])
    minimum = max(config.min_mask_pixels, math.ceil(prompt_area * config.min_prompt_area_ratio))
    if mask.area_pixels < minimum:
        return "SAM_MASK_UNDERSIZED"
    if mask.area_pixels > prompt_area * config.max_prompt_area_ratio:
        return "SAM_MASK_EXPLODED"
    if mask.area_pixels >= width * height * config.max_frame_area_ratio:
        return "SAM_MASK_NEAR_FULL_FRAME"
    top = any(_packed_bit(mask.packed_bits, index) for index in range(crop_width))
    bottom_start = (crop_height - 1) * crop_width
    bottom = any(_packed_bit(mask.packed_bits, bottom_start + index) for index in range(crop_width))
    left = any(_packed_bit(mask.packed_bits, row * crop_width) for row in range(crop_height))
    right = any(
        _packed_bit(mask.packed_bits, row * crop_width + crop_width - 1)
        for row in range(crop_height)
    )
    if not (top and bottom and left and right):
        return "SAM_MASK_MALFORMED"
    if mask.frame_idx == prompt.anchor_frame:
        neighborhood = _expanded_neighborhood(
            prompt.box, width, height, config.anchor_neighborhood_scale
        )
        overlap_x1 = max(x1, neighborhood[0])
        overlap_y1 = max(y1, neighborhood[1])
        overlap_x2 = min(x2, neighborhood[2])
        overlap_y2 = min(y2, neighborhood[3])
        inside = 0
        for y in range(overlap_y1, overlap_y2):
            row_start = (y - y1) * crop_width
            for x in range(overlap_x1, overlap_x2):
                inside += _packed_bit(mask.packed_bits, row_start + x - x1)
        if inside / mask.area_pixels < config.min_anchor_neighborhood_fraction:
            return "SAM_MASK_OFF_PROMPT"
    return None


def _mask_record(mask: PackedMask) -> dict[str, Any]:
    return {
        "key": mask.key,
        "frame_idx": mask.frame_idx,
        "prompt_id": mask.prompt_id,
        "source": mask.source,
        "anchor_frame": mask.anchor_frame,
        "crop_box": mask.crop_box,
        "shape": (
            mask.crop_box[3] - mask.crop_box[1],
            mask.crop_box[2] - mask.crop_box[0],
        ),
        "area_pixels": mask.area_pixels,
        "packed_bytes": len(mask.packed_bits),
        "packed_sha256": hashlib.sha256(mask.packed_bits).hexdigest(),
    }


def _npy_uint8_bytes(values: bytes) -> bytes:
    header = repr(
        {
            "descr": "|u1",
            "fortran_order": False,
            "shape": (len(values),),
        }
    ).encode("latin1")
    prefix_size = 6 + 2 + 2
    padding = (16 - ((prefix_size + len(header) + 1) % 16)) % 16
    header += b" " * padding + b"\n"
    if len(header) > 65535:
        raise Stage2Error("INVALID_SAM_SHARD", "NumPy header is unexpectedly large.")
    return b"\x93NUMPY" + b"\x01\x00" + struct.pack("<H", len(header)) + header + values


def _read_npy_uint8(content: bytes) -> bytes:
    if len(content) < 10 or content[:8] != b"\x93NUMPY\x01\x00":
        raise Stage2Error("INVALID_SAM_SHARD", "Shard member is not NumPy v1.0 data.")
    header_length = struct.unpack("<H", content[8:10])[0]
    header_end = 10 + header_length
    if header_end > len(content):
        raise Stage2Error("INVALID_SAM_SHARD", "NumPy shard member has a torn header.")
    try:
        header = ast.literal_eval(content[10:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", "NumPy shard header is malformed.") from exc
    if not isinstance(header, dict) or header.get("descr") != "|u1":
        raise Stage2Error("INVALID_SAM_SHARD", "Shard arrays must use uint8 data.")
    if header.get("fortran_order") is not False:
        raise Stage2Error("INVALID_SAM_SHARD", "Shard arrays must be C-contiguous.")
    shape = header.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 1 or not isinstance(shape[0], int):
        raise Stage2Error("INVALID_SAM_SHARD", "Shard arrays must be one-dimensional.")
    values = content[header_end:]
    if shape[0] < 0 or len(values) != shape[0]:
        raise Stage2Error("INVALID_SAM_SHARD", "NumPy shard shape does not match its bytes.")
    return values


def encode_sam_mask_shard(meta: SamMaskShardMeta, masks: tuple[PackedMask, ...]) -> bytes:
    by_key = {mask.key: mask for mask in masks}
    if len(by_key) != len(masks):
        raise Stage2Error("INVALID_SAM_SHARD", "Mask keys must be unique within a shard.")
    if tuple(record["key"] for record in meta.masks) != tuple(sorted(by_key)):
        raise Stage2Error("INVALID_SAM_SHARD", "Mask metadata and arrays are out of order.")
    buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(
            buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            members = [("metadata.npy", _npy_uint8_bytes(canonical_json_bytes(meta)))]
            members.extend(
                (f"{key}.npy", _npy_uint8_bytes(by_key[key].packed_bits)) for key in sorted(by_key)
            )
            for name, content in members:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise Stage2Error("SAM_SHARD_ENCODE_FAILED", "Could not encode SAM mask shard.") from exc
    return buffer.getvalue()


def _sam_model_identity_from_dict(raw: dict[str, Any]) -> ModelIdentity:
    try:
        return ModelIdentity(
            name=str(raw["name"]), revision=str(raw["revision"]), sha256=str(raw["sha256"])
        )
    except (KeyError, TypeError) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", "Stored SAM model identity is malformed.") from exc


def _sam_runtime_identity_from_dict(raw: dict[str, Any]) -> SamRuntimeIdentity:
    try:
        return SamRuntimeIdentity(
            repository=str(raw["repository"]),
            revision=str(raw["revision"]),
            source_tree_sha256=str(raw["source_tree_sha256"]),
            model_config=str(raw["model_config"]),
            model_config_sha256=str(raw["model_config_sha256"]),
            torch_version=str(raw["torch_version"]),
            cuda_version=str(raw["cuda_version"]),
        )
    except (KeyError, TypeError) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", "Stored SAM runtime identity is malformed.") from exc


def _sam_mask_record_from_dict(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            "key": str(raw["key"]),
            "frame_idx": int(raw["frame_idx"]),
            "prompt_id": str(raw["prompt_id"]),
            "source": str(raw["source"]),
            "anchor_frame": int(raw["anchor_frame"]),
            "crop_box": tuple(int(value) for value in raw["crop_box"]),
            "shape": tuple(int(value) for value in raw["shape"]),
            "area_pixels": int(raw["area_pixels"]),
            "packed_bytes": int(raw["packed_bytes"]),
            "packed_sha256": str(raw["packed_sha256"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", "Stored mask record is malformed.") from exc


def sam_meta_from_dict(raw: dict[str, Any]) -> SamMaskShardMeta:
    try:
        meta = SamMaskShardMeta(
            schema_version=int(raw["schema_version"]),
            artifact_type=str(raw["artifact_type"]),
            fingerprint=str(raw["fingerprint"]),
            dino_artifact_sha256=str(raw["dino_artifact_sha256"]),
            dino_fingerprint=str(raw["dino_fingerprint"]),
            model=_sam_model_identity_from_dict(raw["model"]),
            runtime=_sam_runtime_identity_from_dict(raw["runtime"]),
            accepted_proposal_threshold=float(raw["accepted_proposal_threshold"]),
            frame_start=int(raw["frame_start"]),
            frame_end=int(raw["frame_end"]),
            frame_width=int(raw["frame_width"]),
            frame_height=int(raw["frame_height"]),
            window=dict(raw["window"]),
            precision=str(raw["precision"]),
            prompts=tuple(dict(value) for value in raw["prompts"]),
            masks=tuple(_sam_mask_record_from_dict(value) for value in raw["masks"]),
            review_flags=tuple(str(value) for value in raw["review_flags"]),
            metrics=dict(raw["metrics"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", "Stored SAM shard metadata is malformed.") from exc
    if meta.schema_version != STAGE2_SCHEMA_VERSION or meta.artifact_type != "sam_mask_shard":
        raise Stage2Error("INVALID_SAM_SHARD", "Stored SAM shard is incompatible.")
    return meta


def _packed_mask_from_record(record: dict[str, Any], packed_bits: bytes) -> PackedMask:
    try:
        crop_box = tuple(int(value) for value in record["crop_box"])
        mask = PackedMask(
            key=str(record["key"]),
            frame_idx=int(record["frame_idx"]),
            prompt_id=str(record["prompt_id"]),
            source=str(record["source"]),
            anchor_frame=int(record["anchor_frame"]),
            crop_box=crop_box,
            packed_bits=packed_bits,
            area_pixels=int(record["area_pixels"]),
        )
        shape = tuple(int(value) for value in record["shape"])
        packed_bytes = int(record["packed_bytes"])
        packed_sha256 = str(record["packed_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", "Stored mask record is malformed.") from exc
    if len(crop_box) != 4 or shape != (crop_box[3] - crop_box[1], crop_box[2] - crop_box[0]):
        raise Stage2Error("INVALID_SAM_SHARD", "Stored mask crop and shape disagree.")
    size = shape[0] * shape[1]
    _validate_packed_bits(packed_bits, size)
    if (
        packed_bytes != len(packed_bits)
        or packed_sha256 != hashlib.sha256(packed_bits).hexdigest()
        or mask.area_pixels != sum(value.bit_count() for value in packed_bits)
        or mask.area_pixels < 1
    ):
        raise Stage2Error("INVALID_SAM_SHARD", "Stored mask integrity check failed.")
    expected_key = _packed_mask_key(
        frame_idx=mask.frame_idx,
        prompt_id=mask.prompt_id,
        source=mask.source,
        anchor_frame=mask.anchor_frame,
        crop_box=mask.crop_box,
        packed_bits=mask.packed_bits,
    )
    if mask.key != expected_key:
        raise Stage2Error("INVALID_SAM_SHARD", "Stored mask key is not canonical.")
    return mask


def load_sam_mask_shard(
    path: Path, *, expected_artifact: ArtifactRef | None = None
) -> LoadedSamMaskShard:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("not a regular file")
        before = file_stamp(resolved)
        content = resolved.read_bytes()
        after = file_stamp(resolved)
        if before != after:
            raise Stage2Error(
                "INPUT_CHANGED_DURING_READ", "SAM shard changed while it was being read."
            )
        actual_ref = ArtifactRef(
            path=str(resolved),
            sha256=hashlib.sha256(content).hexdigest(),
            bytes=len(content),
        )
        if not content:
            raise Stage2Error("INVALID_SAM_SHARD", "SAM shard is empty.")
        if expected_artifact is not None and actual_ref != expected_artifact:
            raise Stage2Error(
                "INVALID_SAM_SHARD", "SAM shard hash or size changed after recording."
            )
        with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "metadata.npy" not in names:
                raise Stage2Error(
                    "INVALID_SAM_SHARD", "SAM shard members are incomplete or duplicate."
                )
            metadata_bytes = _read_npy_uint8(archive.read("metadata.npy"))
            try:
                raw_meta = json.loads(metadata_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Stage2Error(
                    "INVALID_SAM_SHARD", "SAM shard metadata is not valid JSON."
                ) from exc
            if not isinstance(raw_meta, dict):
                raise Stage2Error("INVALID_SAM_SHARD", "SAM shard metadata must be an object.")
            meta = sam_meta_from_dict(raw_meta)
            expected_names = {"metadata.npy"}
            masks = []
            for record in meta.masks:
                key = str(record.get("key", ""))
                member = f"{key}.npy"
                expected_names.add(member)
                if member not in names:
                    raise Stage2Error("INVALID_SAM_SHARD", f"SAM shard is missing {member}.")
                packed = _read_npy_uint8(archive.read(member))
                masks.append(_packed_mask_from_record(record, packed))
            if set(names) != expected_names:
                raise Stage2Error("INVALID_SAM_SHARD", "SAM shard contains unrecorded members.")
    except Stage2Error:
        raise
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", f"Could not read SAM shard {path}.") from exc
    mask_tuple = tuple(masks)
    if encode_sam_mask_shard(meta, mask_tuple) != content:
        raise Stage2Error("INVALID_SAM_SHARD", "SAM shard is not canonically encoded.")
    return LoadedSamMaskShard(artifact=actual_ref, meta=meta, masks=mask_tuple)


def _sam_review_flag(
    code: str,
    window: TemporalWindow,
    *,
    prompt_id: str | None = None,
    frame_idx: int | None = None,
    detail: str | None = None,
) -> str:
    parts = [code, f"window={window.index}"]
    if prompt_id is not None:
        parts.append(f"prompt={prompt_id}")
    if frame_idx is not None:
        parts.append(f"frame={frame_idx}")
    if detail is not None:
        parts.append(detail)
    return "|".join(parts)


def _compact_frame_ranges(frames: list[int]) -> str:
    if not frames:
        return ""
    ranges: list[str] = []
    start = previous = frames[0]
    for frame_idx in frames[1:]:
        if frame_idx == previous + 1:
            previous = frame_idx
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = frame_idx
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def build_sam_mask_shard(
    *,
    dino_artifact: ArtifactRef,
    dino_meta: DinoArtifactMeta,
    config: SamGenerationConfig,
    window: TemporalWindow,
    prompts: tuple[SamPrompt, ...],
    fingerprint_value: str,
    propagation: SamPropagationResult | None,
    propagation_failure: str | None,
    width: int,
    height: int,
) -> tuple[SamMaskShardMeta, tuple[PackedMask, ...]]:
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    review_flags: set[str] = set()
    masks_by_key: dict[str, PackedMask] = {}
    valid_prompt_frames: set[tuple[str, int]] = set()
    rejected_count = 0
    if propagation_failure is not None:
        review_flags.add(_sam_review_flag(propagation_failure, window))
    if propagation is not None:
        raw_masks: tuple[Any, ...] | list[Any]
        if not isinstance(propagation.masks, (tuple, list)):
            raw_masks = ()
            review_flags.add(_sam_review_flag("SAM_OUTPUT_MALFORMED", window))
        else:
            raw_masks = propagation.masks
        if not propagation.forward_complete:
            review_flags.add(_sam_review_flag("SAM_FORWARD_INTERRUPTED", window))
        if not propagation.reverse_complete:
            review_flags.add(_sam_review_flag("SAM_REVERSE_INTERRUPTED", window))
        for raw in raw_masks:
            if not isinstance(raw, RawSamMask):
                rejected_count += 1
                review_flags.add(_sam_review_flag("SAM_MASK_MALFORMED", window))
                continue
            prompt = prompt_by_id.get(raw.prompt_id)
            if prompt is None:
                rejected_count += 1
                review_flags.add(
                    _sam_review_flag(
                        "SAM_UNKNOWN_OBJECT",
                        window,
                        prompt_id=raw.prompt_id,
                        frame_idx=raw.frame_idx,
                    )
                )
                continue
            if not window.frame_start <= raw.frame_idx <= window.frame_end:
                rejected_count += 1
                review_flags.add(
                    _sam_review_flag(
                        "SAM_MASK_OUTSIDE_WINDOW",
                        window,
                        prompt_id=raw.prompt_id,
                        frame_idx=raw.frame_idx,
                    )
                )
                continue
            packed, rejection = assess_raw_sam_mask(
                raw,
                prompt=prompt,
                config=config,
                width=width,
                height=height,
            )
            if packed is None:
                rejected_count += 1
                review_flags.add(
                    _sam_review_flag(
                        rejection or "SAM_MASK_REJECTED",
                        window,
                        prompt_id=raw.prompt_id,
                        frame_idx=raw.frame_idx,
                    )
                )
                continue
            masks_by_key.setdefault(packed.key, packed)
            valid_prompt_frames.add((packed.prompt_id, packed.frame_idx))

    fallback_count = 0
    for prompt in prompts:
        fallback_crop = padded_box(
            prompt.box,
            width=width,
            height=height,
            scale=config.fallback_padding_scale,
            min_padding_px=config.fallback_min_padding_px,
        )
        fallback_source = "dino-fallback" if prompt.prompt_kind == "dino" else "manual-fallback"
        anchor_fallback = rectangle_packed_mask(
            frame_idx=prompt.anchor_frame,
            prompt=prompt,
            crop_box=fallback_crop,
            source=fallback_source,
        )
        masks_by_key.setdefault(anchor_fallback.key, anchor_fallback)
        missing_frames = []
        for frame_idx in range(window.frame_start, window.frame_end + 1):
            if (prompt.prompt_id, frame_idx) in valid_prompt_frames:
                continue
            missing_frames.append(frame_idx)
            fallback = rectangle_packed_mask(
                frame_idx=frame_idx,
                prompt=prompt,
                crop_box=fallback_crop,
                source=fallback_source,
            )
            if fallback.key not in masks_by_key:
                masks_by_key[fallback.key] = fallback
                fallback_count += 1
        if missing_frames:
            review_flags.add(
                _sam_review_flag(
                    "SAM_FALLBACK_USED",
                    window,
                    prompt_id=prompt.prompt_id,
                    detail=(
                        f"ranges={_compact_frame_ranges(missing_frames)}|count={len(missing_frames)}"
                    ),
                )
            )

    masks = tuple(
        sorted(
            masks_by_key.values(),
            key=lambda mask: mask.key,
        )
    )
    records = tuple(_mask_record(mask) for mask in masks)
    try:
        runtime_seconds = float(propagation.runtime_seconds) if propagation is not None else 0.0
        peak_vram_bytes = int(propagation.peak_vram_bytes) if propagation is not None else 0
    except (TypeError, ValueError):
        runtime_seconds = 0.0
        peak_vram_bytes = 0
        review_flags.add(_sam_review_flag("SAM_METRICS_MALFORMED", window))
    if not math.isfinite(runtime_seconds) or runtime_seconds < 0 or peak_vram_bytes < 0:
        runtime_seconds = 0.0
        peak_vram_bytes = 0
        review_flags.add(_sam_review_flag("SAM_METRICS_MALFORMED", window))
    metrics = {
        "n_prompts": len(prompts),
        "n_valid_sam_masks": sum(1 for mask in masks if mask.source.startswith("sam2-")),
        "n_rejected_sam_masks": rejected_count,
        "n_fallback_masks": sum(1 for mask in masks if mask.source.endswith("fallback")),
        "n_missing_frame_fallbacks": fallback_count,
        "n_manual_fallback_masks": sum(1 for mask in masks if mask.source == "manual-fallback"),
        "n_masks": len(masks),
        "mask_area_pixels_unioned_later": sum(mask.area_pixels for mask in masks),
        "runtime_seconds": runtime_seconds,
        "peak_vram_bytes": peak_vram_bytes,
    }
    meta = SamMaskShardMeta(
        schema_version=STAGE2_SCHEMA_VERSION,
        artifact_type="sam_mask_shard",
        fingerprint=fingerprint_value,
        dino_artifact_sha256=dino_artifact.sha256,
        dino_fingerprint=dino_meta.fingerprint,
        model=config.model,
        runtime=config.runtime,
        accepted_proposal_threshold=config.accepted_proposal_threshold,
        frame_start=window.frame_start,
        frame_end=window.frame_end,
        frame_width=width,
        frame_height=height,
        window={
            "index": window.index,
            "size": config.window_size,
            "overlap": config.window_overlap,
        },
        precision=config.precision,
        prompts=_sam_prompt_records(prompts),
        masks=records,
        review_flags=tuple(sorted(review_flags)),
        metrics=metrics,
    )
    return meta, masks


def validate_sam_mask_shard(
    loaded: LoadedSamMaskShard,
    *,
    dino_artifact: ArtifactRef,
    dino_meta: DinoArtifactMeta,
    config: SamGenerationConfig,
    window: TemporalWindow,
    prompts: tuple[SamPrompt, ...],
    fingerprint_value: str,
    width: int,
    height: int,
) -> None:
    meta = loaded.meta
    expected = {
        "fingerprint": fingerprint_value,
        "dino_artifact_sha256": dino_artifact.sha256,
        "dino_fingerprint": dino_meta.fingerprint,
        "model": config.model,
        "runtime": config.runtime,
        "accepted_proposal_threshold": config.accepted_proposal_threshold,
        "frame_start": window.frame_start,
        "frame_end": window.frame_end,
        "frame_width": width,
        "frame_height": height,
        "window": {
            "index": window.index,
            "size": config.window_size,
            "overlap": config.window_overlap,
        },
        "precision": config.precision,
        "prompts": _sam_prompt_records(prompts),
    }
    problems: list[str] = []
    for field_name, expected_value in expected.items():
        if getattr(meta, field_name) != expected_value:
            problems.append(
                f"{field_name}: expected {expected_value!r}, got {getattr(meta, field_name)!r}"
            )
    records = tuple(_mask_record(mask) for mask in loaded.masks)
    if meta.masks != records:
        problems.append("mask records do not match packed arrays")
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    prompt_frames: set[tuple[str, int]] = set()
    sam_prompt_frames: set[tuple[str, int]] = set()
    anchor_fallbacks: set[str] = set()
    valid_sources = {
        "sam2-anchor",
        "sam2-forward",
        "sam2-reverse",
        "dino-fallback",
        "manual-fallback",
    }
    for mask in loaded.masks:
        prompt = prompt_by_id.get(mask.prompt_id)
        if prompt is None:
            problems.append(f"mask {mask.key} references unknown prompt {mask.prompt_id}")
            continue
        if not window.frame_start <= mask.frame_idx <= window.frame_end:
            problems.append(f"mask {mask.key} is outside its window")
        if mask.anchor_frame != prompt.anchor_frame:
            problems.append(f"mask {mask.key} has the wrong anchor frame")
        if mask.source not in valid_sources:
            problems.append(f"mask {mask.key} has invalid source {mask.source}")
        x1, y1, x2, y2 = mask.crop_box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            problems.append(f"mask {mask.key} crop is outside the frame")
            continue
        mask_size = (x2 - x1) * (y2 - y1)
        if mask.source.endswith("fallback"):
            expected_crop = padded_box(
                prompt.box,
                width=width,
                height=height,
                scale=config.fallback_padding_scale,
                min_padding_px=config.fallback_min_padding_px,
            )
            if (
                mask.crop_box != expected_crop
                or mask.area_pixels != mask_size
                or mask.packed_bits != _packed_ones(mask_size)
            ):
                problems.append(f"fallback mask {mask.key} is not the full padded prompt box")
        elif mask.source.startswith("sam2-"):
            rejection = reassess_packed_sam_mask(
                mask,
                prompt=prompt,
                config=config,
                width=width,
                height=height,
            )
            if rejection is not None:
                problems.append(f"stored SAM mask {mask.key} no longer passes validation")
            else:
                sam_prompt_frames.add((mask.prompt_id, mask.frame_idx))
        prompt_frames.add((mask.prompt_id, mask.frame_idx))
        expected_fallback = "dino-fallback" if prompt.prompt_kind == "dino" else "manual-fallback"
        if mask.frame_idx == prompt.anchor_frame and mask.source == expected_fallback:
            anchor_fallbacks.add(prompt.prompt_id)
    for prompt in prompts:
        if prompt.prompt_id not in anchor_fallbacks:
            problems.append(f"prompt {prompt.prompt_id} has no immutable anchor fallback")
        for frame_idx in range(window.frame_start, window.frame_end + 1):
            if (prompt.prompt_id, frame_idx) not in prompt_frames:
                problems.append(
                    f"prompt {prompt.prompt_id} has no SAM or fallback mask at frame {frame_idx}"
                )
                break
    required_fallback_flags: set[str] = set()
    for prompt in prompts:
        missing_frames = [
            frame_idx
            for frame_idx in range(window.frame_start, window.frame_end + 1)
            if (prompt.prompt_id, frame_idx) not in sam_prompt_frames
        ]
        if missing_frames:
            required_fallback_flags.add(
                _sam_review_flag(
                    "SAM_FALLBACK_USED",
                    window,
                    prompt_id=prompt.prompt_id,
                    detail=(
                        f"ranges={_compact_frame_ranges(missing_frames)}|"
                        f"count={len(missing_frames)}"
                    ),
                )
            )
    if (
        not isinstance(meta.review_flags, tuple)
        or tuple(sorted(set(meta.review_flags))) != meta.review_flags
    ):
        problems.append("review_flags must be unique strings in canonical order")
    missing_required_flags = required_fallback_flags.difference(meta.review_flags)
    if missing_required_flags:
        problems.append(
            "review_flags omit required fallback review items: "
            + ", ".join(sorted(missing_required_flags))
        )
    expected_metrics = {
        "n_prompts": len(prompts),
        "n_valid_sam_masks": sum(1 for mask in loaded.masks if mask.source.startswith("sam2-")),
        "n_fallback_masks": sum(1 for mask in loaded.masks if mask.source.endswith("fallback")),
        "n_masks": len(loaded.masks),
        "mask_area_pixels_unioned_later": sum(mask.area_pixels for mask in loaded.masks),
        "n_manual_fallback_masks": sum(
            1 for mask in loaded.masks if mask.source == "manual-fallback"
        ),
    }
    for name, expected_value in expected_metrics.items():
        if meta.metrics.get(name) != expected_value:
            problems.append(f"metrics.{name} does not match shard contents")
    for name in ("n_rejected_sam_masks", "n_missing_frame_fallbacks", "peak_vram_bytes"):
        value = meta.metrics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            problems.append(f"metrics.{name} must be a non-negative integer")
    runtime = meta.metrics.get("runtime_seconds")
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or not math.isfinite(runtime)
        or runtime < 0
    ):
        problems.append("metrics.runtime_seconds must be finite and non-negative")
    if problems:
        raise Stage2Error(
            "INVALID_SAM_SHARD",
            "SAM shard failed provenance or privacy validation.",
            details={"problems": problems},
            recovery="Explicitly recompute the affected SAM window.",
        )


def validate_sam_shard_set(
    metas: tuple[SamMaskShardMeta, ...],
    *,
    n_frames: int,
    window_size: int,
    window_overlap: int,
) -> None:
    expected = temporal_windows(n_frames, window_size, window_overlap)
    actual = tuple((meta.window.get("index"), meta.frame_start, meta.frame_end) for meta in metas)
    expected_ranges = tuple(
        (window.index, window.frame_start, window.frame_end) for window in expected
    )
    if actual != expected_ranges:
        raise Stage2Error(
            "INCOMPLETE_SAM_SHARD_SET",
            "SAM shard windows do not exactly cover the clip.",
            details={"expected": expected_ranges, "actual": actual},
        )


def generate_sam_mask_shards(
    *,
    stage1: StageIInput,
    paths: RunPaths,
    dino_artifact: ArtifactRef,
    dino_meta: DinoArtifactMeta,
    config: SamGenerationConfig,
    adapter: Any,
    window_loader: Callable[[TemporalWindow], Any],
    manual_seeds: tuple[ManualSeed, ...] = (),
) -> SamGenerationResult:
    validate_sam_config(config)
    dino_path = Path(dino_artifact.path)
    before_dino_read = file_stamp(dino_path)
    stored_dino_meta = dino_meta_from_dict(read_json(dino_path))
    after_dino_read = file_stamp(dino_path)
    actual_dino_ref = artifact_ref(dino_path)
    if before_dino_read != after_dino_read:
        raise Stage2Error(
            "DINO_ARTIFACT_CHANGED",
            "DINO artifact changed while SAM was validating it.",
        )
    if actual_dino_ref != dino_artifact:
        raise Stage2Error(
            "DINO_ARTIFACT_CHANGED",
            "DINO artifact changed before SAM processing.",
        )
    if stored_dino_meta != dino_meta:
        raise Stage2Error(
            "INVALID_DINO_ARTIFACT",
            "Provided DINO metadata does not match its immutable artifact.",
        )
    if dino_meta.source_sha256 != stage1.source.sha256:
        raise Stage2Error(
            "DINO_SOURCE_MISMATCH",
            "DINO proposals were generated from a different original video.",
            details={
                "expected_source_sha256": stage1.source.sha256,
                "dino_source_sha256": dino_meta.source_sha256,
            },
            recovery="Select or recompute the DINO artifact for this exact original video.",
        )
    if getattr(adapter, "identity", None) != config.model:
        raise Stage2Error(
            "SAM_MODEL_IDENTITY_MISMATCH",
            "Loaded SAM adapter identity does not match the fingerprinted configuration.",
        )
    if getattr(adapter, "runtime_identity", None) != config.runtime:
        raise Stage2Error(
            "SAM_RUNTIME_IDENTITY_MISMATCH",
            "Loaded SAM runtime does not match the fingerprinted configuration.",
            recovery="Rebuild the SAM configuration from the verified loaded adapter.",
        )
    selection = select_dino_proposals(dino_meta, config.accepted_proposal_threshold)
    seeds = validate_manual_seeds(manual_seeds, stage1)
    windows = temporal_windows(
        stage1.source_video.n_frames, config.window_size, config.window_overlap
    )
    shard_refs: list[ArtifactRef] = []
    metas: list[SamMaskShardMeta] = []
    reused = 0
    generated = 0
    for window in windows:
        prompts = sam_prompts_for_window(window, selection.accepted, seeds)
        propagation: SamPropagationResult | None = None
        propagation_failure: str | None = None
        window_input: SamWindowInput | None = None
        if prompts:
            try:
                window_input = validate_sam_window_input(
                    window_loader(window), stage1=stage1, window=window
                )
            except Stage2Error as exc:
                propagation_failure = exc.code
            except Exception:
                propagation_failure = "SAM_WINDOW_LOAD_FAILED"
        else:
            propagation = SamPropagationResult(
                masks=(), forward_complete=True, reverse_complete=True
            )
        frame_payload_status = (
            f"error:{propagation_failure}"
            if propagation_failure is not None
            else "verified"
            if window_input is not None and window_input.frame_payload_sha256 is not None
            else "metadata-only"
            if window_input is not None
            else "not-required"
        )
        fingerprint_payload = sam_window_fingerprint_payload(
            dino_artifact=dino_artifact,
            dino_meta=dino_meta,
            config=config,
            window=window,
            prompts=prompts,
            frame_payload_sha256=(
                window_input.frame_payload_sha256 if window_input is not None else None
            ),
            frame_payload_status=frame_payload_status,
        )
        fingerprint_value = sam_fingerprint(fingerprint_payload)
        shard_path = sam_shard_path(paths, window, fingerprint_value)
        if shard_path.exists():
            loaded = load_sam_mask_shard(shard_path)
            validate_sam_mask_shard(
                loaded,
                dino_artifact=dino_artifact,
                dino_meta=dino_meta,
                config=config,
                window=window,
                prompts=prompts,
                fingerprint_value=fingerprint_value,
                width=stage1.source_video.display_width,
                height=stage1.source_video.display_height,
            )
            shard_refs.append(loaded.artifact)
            metas.append(loaded.meta)
            reused += 1
            continue

        if prompts and window_input is not None:
            try:
                candidate = adapter.propagate_window(
                    window_input=window_input.payload,
                    window=window,
                    prompts=prompts,
                    width=stage1.source_video.display_width,
                    height=stage1.source_video.display_height,
                    precision=config.precision,
                )
                if not isinstance(candidate, SamPropagationResult):
                    raise TypeError("adapter returned an unknown result type")
                propagation = candidate
            except Exception:
                propagation_failure = "SAM_INFERENCE_FAILED"
        meta, masks = build_sam_mask_shard(
            dino_artifact=dino_artifact,
            dino_meta=dino_meta,
            config=config,
            window=window,
            prompts=prompts,
            fingerprint_value=fingerprint_value,
            propagation=propagation,
            propagation_failure=propagation_failure,
            width=stage1.source_video.display_width,
            height=stage1.source_video.display_height,
        )
        encoded = encode_sam_mask_shard(meta, masks)
        shard_ref = write_immutable_bytes(shard_path, encoded)
        loaded = load_sam_mask_shard(shard_path, expected_artifact=shard_ref)
        validate_sam_mask_shard(
            loaded,
            dino_artifact=dino_artifact,
            dino_meta=dino_meta,
            config=config,
            window=window,
            prompts=prompts,
            fingerprint_value=fingerprint_value,
            width=stage1.source_video.display_width,
            height=stage1.source_video.display_height,
        )
        shard_refs.append(shard_ref)
        metas.append(loaded.meta)
        generated += 1
    meta_tuple = tuple(metas)
    validate_sam_shard_set(
        meta_tuple,
        n_frames=stage1.source_video.n_frames,
        window_size=config.window_size,
        window_overlap=config.window_overlap,
    )
    return SamGenerationResult(
        shards=tuple(shard_refs),
        metas=meta_tuple,
        reused_window_count=reused,
        generated_window_count=generated,
        review_flags=tuple(sorted({flag for meta in meta_tuple for flag in meta.review_flags})),
    )


def union_sam_masks_for_frame(
    shards: tuple[LoadedSamMaskShard, ...],
    *,
    frame_idx: int,
    width: int,
    height: int,
) -> bytes:
    relevant = tuple(
        shard for shard in shards if shard.meta.frame_start <= frame_idx <= shard.meta.frame_end
    )
    if not relevant:
        raise Stage2Error("SAM_SHARD_COVERAGE_GAP", f"No SAM shard covers frame {frame_idx}.")
    if (
        width < 1
        or height < 1
        or any(
            shard.meta.frame_width != width or shard.meta.frame_height != height
            for shard in relevant
        )
    ):
        raise Stage2Error(
            "SAM_FRAME_SIZE_MISMATCH",
            "SAM shard frame dimensions do not match the requested render frame.",
        )
    output = bytearray(width * height)
    for shard in relevant:
        for mask in shard.masks:
            if mask.frame_idx != frame_idx:
                continue
            x1, y1, x2, y2 = mask.crop_box
            crop_width = x2 - x1
            crop_height = y2 - y1
            pixels = _unpack_bits(mask.packed_bits, crop_width * crop_height)
            for row in range(crop_height):
                source_start = row * crop_width
                destination_start = (y1 + row) * width + x1
                for column, value in enumerate(pixels[source_start : source_start + crop_width]):
                    if value:
                        output[destination_start + column] = 1
    return bytes(output)


def scaled_render_dilation(height: int, pixels_at_1080p: int) -> int:
    if height < 1 or pixels_at_1080p < 0:
        raise Stage2Error(
            "INVALID_RENDER_CONFIG",
            "Render height must be positive and dilation must be non-negative.",
        )
    if pixels_at_1080p == 0:
        return 0
    return max(1, round(pixels_at_1080p * height / 1080))


def dilate_binary_mask(mask: bytes, *, width: int, height: int, radius: int) -> bytes:
    """Dilate a dense 0/1 mask with a square kernel using an integral image."""
    if width < 1 or height < 1 or radius < 0 or len(mask) != width * height:
        raise Stage2Error("INVALID_RENDER_MASK", "Render mask dimensions are invalid.")
    try:
        import numpy as np
    except ImportError as exc:
        raise Stage2Error(
            "NUMPY_NOT_FOUND",
            "NumPy is required by the Stage II renderer.",
            recovery="Run the Stage II setup command, then retry rendering.",
        ) from exc
    source = np.frombuffer(mask, dtype=np.uint8).reshape(height, width)
    if np.any(source > 1):
        raise Stage2Error("INVALID_RENDER_MASK", "Render masks must contain only 0 or 1.")
    if radius == 0 or not source.any():
        return source.tobytes()
    padded = np.pad(source, radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    diameter = 2 * radius + 1
    counts = (
        integral[diameter:, diameter:]
        - integral[:-diameter, diameter:]
        - integral[diameter:, :-diameter]
        + integral[:-diameter, :-diameter]
    )
    return (counts > 0).astype(np.uint8).tobytes()


def apply_mask_to_yuv420p(
    frame: bytes,
    mask: bytes,
    *,
    width: int,
    height: int,
    fill_yuv: tuple[int, int, int],
) -> bytes:
    if width % 2 or height % 2:
        raise Stage2Error(
            "UNSUPPORTED_RENDER_DIMENSIONS",
            "The yuv420p renderer requires even display dimensions.",
            details={"width": width, "height": height},
        )
    expected_size = width * height * 3 // 2
    if len(frame) != expected_size or len(mask) != width * height:
        raise Stage2Error("INVALID_RENDER_FRAME", "Render frame or mask size is invalid.")
    if len(fill_yuv) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
        for value in fill_yuv
    ):
        raise Stage2Error("INVALID_RENDER_CONFIG", "YUV fill values must be bytes.")
    try:
        import numpy as np
    except ImportError as exc:
        raise Stage2Error("NUMPY_NOT_FOUND", "NumPy is required by the renderer.") from exc
    output = np.frombuffer(frame, dtype=np.uint8).copy()
    luma_size = width * height
    chroma_size = luma_size // 4
    luma_mask = np.frombuffer(mask, dtype=np.uint8).reshape(height, width).astype(bool)
    if np.any(np.frombuffer(mask, dtype=np.uint8) > 1):
        raise Stage2Error("INVALID_RENDER_MASK", "Render masks must contain only 0 or 1.")
    output[:luma_size].reshape(height, width)[luma_mask] = fill_yuv[0]
    chroma_mask = luma_mask.reshape(height // 2, 2, width // 2, 2).any(axis=(1, 3))
    output[luma_size : luma_size + chroma_size].reshape(height // 2, width // 2)[chroma_mask] = (
        fill_yuv[1]
    )
    output[luma_size + chroma_size :].reshape(height // 2, width // 2)[chroma_mask] = fill_yuv[2]
    return output.tobytes()


def _validate_render_config(config: RenderConfig) -> None:
    problems = []
    if (
        isinstance(config.dilation_pixels_at_1080p, bool)
        or not isinstance(config.dilation_pixels_at_1080p, int)
        or not 0 <= config.dilation_pixels_at_1080p <= 256
    ):
        problems.append("dilation_pixels_at_1080p must be an integer from 0 to 256")
    if config.pixel_format != "yuv420p":
        problems.append("pixel_format must be yuv420p")
    if not config.codec or not config.preset:
        problems.append("codec and preset must be non-empty")
    if isinstance(config.crf, bool) or not isinstance(config.crf, int) or not 0 <= config.crf <= 51:
        problems.append("crf must be an integer from 0 to 51")
    if len(config.fill_yuv) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
        for value in config.fill_yuv
    ):
        problems.append("fill_yuv must contain three byte values")
    if (
        isinstance(config.fill_tolerance, bool)
        or not isinstance(config.fill_tolerance, int)
        or not 0 <= config.fill_tolerance <= 127
    ):
        problems.append("fill_tolerance must be an integer from 0 to 127")
    if not math.isfinite(config.max_outside_mask_mae) or config.max_outside_mask_mae < 0:
        problems.append("max_outside_mask_mae must be finite and non-negative")
    if problems:
        raise Stage2Error(
            "INVALID_RENDER_CONFIG",
            "Stage II render configuration is invalid.",
            details={"problems": problems},
        )


def render_fingerprint_payload(
    *,
    stage1: StageIInput,
    sam_shards: tuple[ArtifactRef, ...],
    config: RenderConfig,
) -> dict[str, Any]:
    return {
        "stage1_video_sha256": stage1.stage1_video.sha256,
        "stage1_video_bytes": stage1.stage1_video.bytes,
        "mask_set_sha256": artifact_set_sha256(sam_shards),
        "mask_shards": tuple(
            {"sha256": shard.sha256, "bytes": shard.bytes} for shard in sam_shards
        ),
        "display_size": (
            stage1.stage1_output_video.display_width,
            stage1.stage1_output_video.display_height,
        ),
        "fps": stage1.stage1_output_video.fps,
        "n_frames": stage1.stage1_output_video.n_frames,
        "dilation_pixels_at_1080p": config.dilation_pixels_at_1080p,
        "fill_yuv": config.fill_yuv,
        "encoder": {
            "codec": config.codec,
            "crf": config.crf,
            "preset": config.preset,
            "pixel_format": config.pixel_format,
        },
        "verification_policy": {
            "fill_tolerance": config.fill_tolerance,
            "max_outside_mask_mae": config.max_outside_mask_mae,
        },
    }


def _load_render_shard_metas(
    sam_shards: tuple[ArtifactRef, ...], *, stage1: StageIInput
) -> tuple[SamMaskShardMeta, ...]:
    if not sam_shards:
        raise Stage2Error("EMPTY_ARTIFACT_SET", "Rendering requires SAM mask shards.")
    metas = []
    for reference in sam_shards:
        loaded = load_sam_mask_shard(Path(reference.path), expected_artifact=reference)
        meta = loaded.meta
        if (
            meta.frame_width != stage1.stage1_output_video.display_width
            or meta.frame_height != stage1.stage1_output_video.display_height
        ):
            raise Stage2Error(
                "SAM_FRAME_SIZE_MISMATCH",
                "SAM shard dimensions do not match the Stage I render source.",
            )
        metas.append(meta)
    meta_tuple = tuple(metas)
    first = meta_tuple[0]
    consistency = (
        first.schema_version,
        first.artifact_type,
        first.dino_artifact_sha256,
        first.dino_fingerprint,
        first.model,
        first.runtime,
        first.accepted_proposal_threshold,
        first.precision,
    )
    if first.schema_version != STAGE2_SCHEMA_VERSION or first.artifact_type != "sam_mask_shard":
        raise Stage2Error("INVALID_SAM_SHARD", "SAM shard schema or artifact type is invalid.")
    if any(
        (
            meta.schema_version,
            meta.artifact_type,
            meta.dino_artifact_sha256,
            meta.dino_fingerprint,
            meta.model,
            meta.runtime,
            meta.accepted_proposal_threshold,
            meta.precision,
        )
        != consistency
        for meta in meta_tuple
    ):
        raise Stage2Error(
            "INCONSISTENT_SAM_SHARD_SET",
            "SAM shards do not share one model, runtime, DINO artifact, and threshold.",
        )
    try:
        window_size = int(first.window["size"])
        window_overlap = int(first.window["overlap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", "SAM window layout is malformed.") from exc
    try:
        layout_disagrees = any(
            int(meta.window.get("size", -1)) != window_size
            or int(meta.window.get("overlap", -1)) != window_overlap
            for meta in meta_tuple
        )
    except (TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_SAM_SHARD", "SAM window layout is malformed.") from exc
    if layout_disagrees:
        raise Stage2Error("INCOMPLETE_SAM_SHARD_SET", "SAM window layouts disagree.")
    validate_sam_shard_set(
        meta_tuple,
        n_frames=stage1.stage1_output_video.n_frames,
        window_size=window_size,
        window_overlap=window_overlap,
    )
    return meta_tuple


class _RenderShardReader:
    def __init__(
        self,
        references: tuple[ArtifactRef, ...],
        metas: tuple[SamMaskShardMeta, ...],
    ) -> None:
        self.references = references
        self.metas = metas
        self.loaded: dict[int, LoadedSamMaskShard] = {}

    def for_frame(self, frame_idx: int) -> tuple[LoadedSamMaskShard, ...]:
        relevant = tuple(
            index
            for index, meta in enumerate(self.metas)
            if meta.frame_start <= frame_idx <= meta.frame_end
        )
        if not relevant:
            raise Stage2Error("SAM_SHARD_COVERAGE_GAP", f"No SAM shard covers frame {frame_idx}.")
        for index in tuple(self.loaded):
            if index not in relevant:
                del self.loaded[index]
        for index in relevant:
            if index not in self.loaded:
                self.loaded[index] = load_sam_mask_shard(
                    Path(self.references[index].path),
                    expected_artifact=self.references[index],
                )
        return tuple(self.loaded[index] for index in relevant)


def _read_exact_frame(stream: Any, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _ffmpeg_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise Stage2Error(
            "FFMPEG_NOT_FOUND",
            "ffmpeg was not found on PATH.",
            recovery="Run the Stage II setup command, then retry rendering.",
        )
    return ffmpeg


def _rawvideo_decoder(path: Path, *, pixel_format: str, stderr: Any) -> subprocess.Popen:
    command = [
        _ffmpeg_path(),
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "pipe:1",
    ]
    try:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
    except OSError as exc:
        raise Stage2Error("FFMPEG_START_FAILED", "Could not start the video decoder.") from exc


def _rawvideo_encoder(
    path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    config: RenderConfig,
    color_facts: dict[str, str],
    stderr: Any,
) -> subprocess.Popen:
    command = [
        _ffmpeg_path(),
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        config.pixel_format,
        "-s:v",
        f"{width}x{height}",
        "-r",
        format(fps, ".12g"),
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-c:v",
        config.codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-pix_fmt",
        config.pixel_format,
    ]
    color_options = {
        "color_range": "-color_range",
        "color_space": "-colorspace",
        "color_transfer": "-color_trc",
        "color_primaries": "-color_primaries",
        "chroma_location": "-chroma_sample_location",
    }
    for field_name, option in color_options.items():
        value = color_facts.get(field_name)
        if value and value not in {"unknown", "reserved", "unspecified"}:
            command.extend((option, value))
    command.extend(
        [
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ]
    )
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr)
    except OSError as exc:
        raise Stage2Error("FFMPEG_START_FAILED", "Could not start the video encoder.") from exc


def _stderr_text(handle: Any) -> str:
    handle.flush()
    handle.seek(0)
    return handle.read().decode("utf-8", errors="replace").strip()[-4000:]


def _render_frames(
    *,
    stage1_path: Path,
    output_path: Path,
    stage1: StageIInput,
    sam_shards: tuple[ArtifactRef, ...],
    metas: tuple[SamMaskShardMeta, ...],
    config: RenderConfig,
) -> dict[str, Any]:
    width = stage1.stage1_output_video.display_width
    height = stage1.stage1_output_video.display_height
    if width % 2 or height % 2:
        raise Stage2Error(
            "UNSUPPORTED_RENDER_DIMENSIONS",
            "The yuv420p renderer requires even display dimensions.",
        )
    frame_size = width * height * 3 // 2
    radius = scaled_render_dilation(height, config.dilation_pixels_at_1080p)
    reader = _RenderShardReader(sam_shards, metas)
    source_media = _probe_media_contract(stage1_path)
    source_video_streams = [
        stream for stream in source_media["streams"] if stream.get("codec_type") == "video"
    ]
    if len(source_video_streams) != 1:
        raise Stage2Error("RENDER_SOURCE_INVALID", "Stage I must contain exactly one video stream.")
    color_fields = (
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "chroma_location",
    )
    color_facts = {
        name: str(source_video_streams[0][name])
        for name in color_fields
        if source_video_streams[0].get(name)
    }
    masked_pixels = 0
    started = time.monotonic()
    with tempfile.TemporaryFile() as decoder_stderr, tempfile.TemporaryFile() as encoder_stderr:
        decoder = _rawvideo_decoder(
            stage1_path, pixel_format=config.pixel_format, stderr=decoder_stderr
        )
        encoder = _rawvideo_encoder(
            output_path,
            width=width,
            height=height,
            fps=stage1.stage1_output_video.fps,
            config=config,
            color_facts=color_facts,
            stderr=encoder_stderr,
        )
        assert decoder.stdout is not None and encoder.stdin is not None
        try:
            for frame_idx in range(stage1.stage1_output_video.n_frames):
                frame = _read_exact_frame(decoder.stdout, frame_size)
                if len(frame) != frame_size:
                    raise Stage2Error(
                        "RENDER_SOURCE_TRUNCATED",
                        "Stage I decode ended before the recorded frame count.",
                        details={"frame_idx": frame_idx, "bytes": len(frame)},
                    )
                active = reader.for_frame(frame_idx)
                mask = union_sam_masks_for_frame(
                    active, frame_idx=frame_idx, width=width, height=height
                )
                mask = dilate_binary_mask(mask, width=width, height=height, radius=radius)
                masked_pixels += mask.count(1)
                rendered = apply_mask_to_yuv420p(
                    frame,
                    mask,
                    width=width,
                    height=height,
                    fill_yuv=config.fill_yuv,
                )
                try:
                    encoder.stdin.write(rendered)
                except BrokenPipeError as exc:
                    raise Stage2Error(
                        "RENDER_ENCODE_FAILED", "The Stage II encoder stopped early."
                    ) from exc
            if decoder.stdout.read(1):
                raise Stage2Error(
                    "RENDER_SOURCE_FRAME_COUNT_MISMATCH",
                    "Stage I decoded more frames than its validated frame count.",
                )
        except BaseException:
            decoder.kill()
            encoder.kill()
            decoder.wait()
            encoder.wait()
            raise
        finally:
            decoder.stdout.close()
            with contextlib.suppress(BrokenPipeError):
                encoder.stdin.close()
        decoder_code = decoder.wait()
        encoder_code = encoder.wait()
        if decoder_code != 0:
            raise Stage2Error(
                "RENDER_DECODE_FAILED",
                "ffmpeg could not decode the complete Stage I video.",
                details={"stderr": _stderr_text(decoder_stderr)},
            )
        if encoder_code != 0:
            raise Stage2Error(
                "RENDER_ENCODE_FAILED",
                "ffmpeg could not encode the complete Stage II video.",
                details={"stderr": _stderr_text(encoder_stderr)},
            )
    return {
        "runtime_seconds": time.monotonic() - started,
        "n_frames": stage1.stage1_output_video.n_frames,
        "dilation_pixels": radius,
        "added_mask_pixels": masked_pixels,
        "added_mask_area_fraction": masked_pixels
        / (width * height * stage1.stage1_output_video.n_frames),
        "hand_tool_overlap": {"available": False, "overlap_pixels": None},
        "source_color_facts": color_facts,
    }


def _probe_media_contract(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise Stage2Error("FFPROBE_NOT_FOUND", "ffprobe was not found on PATH.")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise Stage2Error(
            "RENDER_MEDIA_PROBE_FAILED",
            "ffprobe could not inspect the rendered video.",
            details={"stderr": result.stderr.strip()},
        )
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise Stage2Error(
            "RENDER_MEDIA_PROBE_FAILED", "Rendered media facts are malformed."
        ) from exc
    if not isinstance(streams, list):
        raise Stage2Error("RENDER_MEDIA_PROBE_FAILED", "Rendered streams are malformed.")
    return payload


def _mask_arrays(mask: bytes, *, width: int, height: int) -> tuple[Any, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise Stage2Error("NUMPY_NOT_FOUND", "NumPy is required by verification.") from exc
    luma = np.frombuffer(mask, dtype=np.uint8).reshape(height, width).astype(bool)
    chroma = luma.reshape(height // 2, 2, width // 2, 2).any(axis=(1, 3))
    return luma, chroma


def verify_rendered_video(
    *,
    stage1: StageIInput,
    output_path: Path,
    sam_shards: tuple[ArtifactRef, ...],
    metas: tuple[SamMaskShardMeta, ...],
    config: RenderConfig,
) -> dict[str, Any]:
    """Decode both videos and fail closed unless media and redaction checks pass."""
    output_path = resolve_input_file(output_path, "temporary Stage II output")
    output_facts = probe_video(output_path)
    expected = stage1.stage1_output_video
    problems = []
    if (output_facts.display_width, output_facts.display_height) != (
        expected.display_width,
        expected.display_height,
    ):
        problems.append("display dimensions changed")
    if output_facts.n_frames != expected.n_frames:
        problems.append(f"frame count changed from {expected.n_frames} to {output_facts.n_frames}")
    if not output_facts.is_cfr or not math.isclose(
        output_facts.fps, expected.fps, rel_tol=0.0, abs_tol=1e-6
    ):
        problems.append("frame rate or constant-frame-rate contract changed")
    if output_facts.rotation != 0:
        problems.append("rendered video retains rotation metadata")
    media = _probe_media_contract(output_path)
    streams = media["streams"]
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        problems.append("rendered output must contain exactly one video stream")
    elif streams[0].get("pix_fmt") != config.pixel_format:
        problems.append(
            f"pixel format is {streams[0].get('pix_fmt')!r}, expected {config.pixel_format!r}"
        )
    source_media = _probe_media_contract(Path(stage1.stage1_video.path))
    source_video_streams = [
        stream for stream in source_media["streams"] if stream.get("codec_type") == "video"
    ]
    color_fields = (
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "chroma_location",
    )
    source_color_facts = {
        name: str(source_video_streams[0][name])
        for name in color_fields
        if len(source_video_streams) == 1 and source_video_streams[0].get(name)
    }
    output_color_facts = {
        name: str(streams[0][name])
        for name in color_fields
        if len(streams) == 1 and streams[0].get(name)
    }
    for name, value in source_color_facts.items():
        if (
            value not in {"unknown", "reserved", "unspecified"}
            and output_color_facts.get(name) != value
        ):
            problems.append(
                f"color fact {name} changed from {value!r} to {output_color_facts.get(name)!r}"
            )
    forbidden_tag_names = {
        "title",
        "artist",
        "album",
        "comment",
        "description",
        "creation_time",
        "location",
        "location-eng",
        "make",
        "model",
    }
    tags = dict(media.get("format", {}).get("tags", {}) or {})
    for stream in streams:
        tags.update(stream.get("tags", {}) or {})
    retained_forbidden_tags = sorted(
        forbidden_tag_names.intersection(str(name).casefold() for name in tags)
    )
    if retained_forbidden_tags:
        problems.append(f"forbidden metadata remains: {retained_forbidden_tags}")
    if problems:
        raise Stage2Error(
            "RENDER_MEDIA_VERIFICATION_FAILED",
            "Rendered media contract verification failed.",
            details={"problems": problems},
        )

    width = expected.display_width
    height = expected.display_height
    frame_size = width * height * 3 // 2
    luma_size = width * height
    chroma_size = luma_size // 4
    radius = scaled_render_dilation(height, config.dilation_pixels_at_1080p)
    reader = _RenderShardReader(sam_shards, metas)
    fill_checked = 0
    fill_violations = 0
    outside_abs_sum = 0
    outside_squared_sum = 0
    outside_samples = 0
    try:
        import numpy as np
    except ImportError as exc:
        raise Stage2Error("NUMPY_NOT_FOUND", "NumPy is required by verification.") from exc
    with tempfile.TemporaryFile() as source_stderr, tempfile.TemporaryFile() as output_stderr:
        source_decoder = _rawvideo_decoder(
            Path(stage1.stage1_video.path),
            pixel_format=config.pixel_format,
            stderr=source_stderr,
        )
        output_decoder = _rawvideo_decoder(
            output_path, pixel_format=config.pixel_format, stderr=output_stderr
        )
        assert source_decoder.stdout is not None and output_decoder.stdout is not None
        for frame_idx in range(expected.n_frames):
            source_frame = _read_exact_frame(source_decoder.stdout, frame_size)
            output_frame = _read_exact_frame(output_decoder.stdout, frame_size)
            if len(source_frame) != frame_size or len(output_frame) != frame_size:
                source_decoder.kill()
                output_decoder.kill()
                raise Stage2Error(
                    "RENDER_DECODE_VERIFICATION_FAILED",
                    "A verification decode ended before the expected frame count.",
                    details={"frame_idx": frame_idx},
                )
            mask = union_sam_masks_for_frame(
                reader.for_frame(frame_idx),
                frame_idx=frame_idx,
                width=width,
                height=height,
            )
            mask = dilate_binary_mask(mask, width=width, height=height, radius=radius)
            luma_mask, chroma_mask = _mask_arrays(mask, width=width, height=height)
            source_values = np.frombuffer(source_frame, dtype=np.uint8)
            output_values = np.frombuffer(output_frame, dtype=np.uint8)
            output_luma = output_values[:luma_size].reshape(height, width)
            source_luma = source_values[:luma_size].reshape(height, width)
            if luma_mask.any():
                deviations = np.abs(output_luma[luma_mask].astype(np.int16) - config.fill_yuv[0])
                fill_checked += int(deviations.size)
                fill_violations += int((deviations > config.fill_tolerance).sum())
            for offset, fill_value in (
                (luma_size, config.fill_yuv[1]),
                (luma_size + chroma_size, config.fill_yuv[2]),
            ):
                plane = output_values[offset : offset + chroma_size].reshape(
                    height // 2, width // 2
                )
                if chroma_mask.any():
                    deviations = np.abs(plane[chroma_mask].astype(np.int16) - fill_value)
                    fill_checked += int(deviations.size)
                    fill_violations += int((deviations > config.fill_tolerance).sum())
            outside = ~luma_mask
            if outside.any():
                difference = output_luma[outside].astype(np.int16) - source_luma[outside].astype(
                    np.int16
                )
                absolute = np.abs(difference).astype(np.int64)
                outside_abs_sum += int(absolute.sum())
                outside_squared_sum += int((difference.astype(np.int64) ** 2).sum())
                outside_samples += int(difference.size)
        source_extra = source_decoder.stdout.read(1)
        output_extra = output_decoder.stdout.read(1)
        source_decoder.stdout.close()
        output_decoder.stdout.close()
        source_code = source_decoder.wait()
        output_code = output_decoder.wait()
        if source_code != 0 or output_code != 0 or source_extra or output_extra:
            raise Stage2Error(
                "RENDER_DECODE_VERIFICATION_FAILED",
                "Verification decoders did not consume exactly the expected frames.",
                details={
                    "source_stderr": _stderr_text(source_stderr),
                    "output_stderr": _stderr_text(output_stderr),
                },
            )
    outside_mae = outside_abs_sum / outside_samples if outside_samples else 0.0
    outside_mse = outside_squared_sum / outside_samples if outside_samples else 0.0
    outside_psnr = None if outside_mse == 0 else 10 * math.log10(255**2 / outside_mse)
    if fill_violations:
        raise Stage2Error(
            "RENDER_FILL_INTEGRITY_FAILED",
            "Encoded Stage II fill pixels exceeded the allowed deviation.",
            details={
                "checked": fill_checked,
                "violations": fill_violations,
                "tolerance": config.fill_tolerance,
            },
        )
    if outside_mae > config.max_outside_mask_mae:
        raise Stage2Error(
            "RENDER_QUALITY_FAILED",
            "Outside-mask quality loss exceeded the configured ceiling.",
            details={
                "outside_mask_mae": outside_mae,
                "ceiling": config.max_outside_mask_mae,
            },
        )
    return {
        "passed": True,
        "publishable": False,
        "frame_count": output_facts.n_frames,
        "fps": output_facts.fps,
        "display_width": output_facts.display_width,
        "display_height": output_facts.display_height,
        "cfr": output_facts.is_cfr,
        "stream_types": tuple(stream.get("codec_type") for stream in streams),
        "pixel_format": streams[0].get("pix_fmt"),
        "source_color_facts": source_color_facts,
        "output_color_facts": output_color_facts,
        "forbidden_metadata_tags": retained_forbidden_tags,
        "fill_integrity_checked": fill_checked,
        "fill_integrity_violations": fill_violations,
        "fill_tolerance": config.fill_tolerance,
        "outside_mask_luma_mae": outside_mae,
        "outside_mask_luma_psnr_db": outside_psnr,
        "outside_mask_samples": outside_samples,
        "input_artifact_hashes_verified": True,
        "complete_shard_set_verified": True,
    }


def render_meta_from_dict(raw: dict[str, Any]) -> RenderArtifactMeta:
    try:
        output = raw["output"]
        return RenderArtifactMeta(
            schema_version=int(raw["schema_version"]),
            artifact_type=str(raw["artifact_type"]),
            fingerprint=str(raw["fingerprint"]),
            stage1_video_sha256=str(raw["stage1_video_sha256"]),
            mask_set_sha256=str(raw["mask_set_sha256"]),
            output=ArtifactRef(
                path=str(output["path"]),
                sha256=str(output["sha256"]),
                bytes=int(output["bytes"]),
            ),
            encoder=dict(raw["encoder"]),
            verification=dict(raw["verification"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_RENDER_ARTIFACT", "Render artifact is malformed.") from exc


def _promote_render_output(temporary: Path, final: Path) -> ArtifactRef:
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary_ref = artifact_ref(temporary)
    if final.exists():
        existing = artifact_ref(final)
        if (existing.sha256, existing.bytes) != (
            temporary_ref.sha256,
            temporary_ref.bytes,
        ):
            raise Stage2Error(
                "IMMUTABLE_ARTIFACT_CONFLICT",
                "A different immutable render output already exists.",
                details={"path": str(final)},
            )
        temporary.unlink()
        return existing
    try:
        os.replace(temporary, final)
        directory_fd = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise Stage2Error(
            "RENDER_PROMOTION_FAILED",
            "Verified output could not be promoted atomically.",
            details={"cause": str(exc)},
        ) from exc
    return artifact_ref(final)


def render_stage2_video(
    *,
    stage1: StageIInput,
    paths: RunPaths,
    sam_shards: tuple[ArtifactRef, ...],
    config: RenderConfig | None = None,
) -> RenderGenerationResult:
    """Render from Stage I only, verify the temporary output, then promote it."""
    if config is None:
        config = RenderConfig()
    _validate_render_config(config)
    stage1_path = Path(stage1.stage1_video.path)
    current_stage1_ref = artifact_ref(stage1_path)
    if current_stage1_ref != stage1.stage1_video:
        raise Stage2Error(
            "STAGE1_VIDEO_CHANGED",
            "The Stage I render source changed after validation.",
        )
    metas = _load_render_shard_metas(sam_shards, stage1=stage1)
    payload = render_fingerprint_payload(stage1=stage1, sam_shards=sam_shards, config=config)
    fingerprint_value = render_fingerprint(payload)
    mask_set_sha256 = artifact_set_sha256(sam_shards)
    final_output = paths.render.parent / f"stage2-{fingerprint_value[:16]}.mp4"
    if paths.render.exists():
        meta = render_meta_from_dict(read_json(paths.render))
        expected_encoder = {
            "codec": config.codec,
            "crf": config.crf,
            "preset": config.preset,
            "pixel_format": config.pixel_format,
            "audio": "stripped",
            "subtitles": "stripped",
            "data_streams": "stripped",
            "metadata": "stripped",
            "pixel_source": "stage1_video_only",
            "fill_yuv": list(config.fill_yuv),
            "dilation_pixels_at_1080p": config.dilation_pixels_at_1080p,
        }
        expected_fields = (
            meta.schema_version == STAGE2_SCHEMA_VERSION,
            meta.artifact_type == "render",
            meta.fingerprint == fingerprint_value,
            meta.stage1_video_sha256 == stage1.stage1_video.sha256,
            meta.mask_set_sha256 == mask_set_sha256,
            meta.verification.get("passed") is True,
            meta.encoder == expected_encoder,
            Path(meta.output.path) == final_output.resolve(strict=False),
        )
        if not all(expected_fields):
            raise Stage2Error(
                "STALE_RENDER_ARTIFACT",
                "Existing render evidence does not match the requested render fingerprint.",
                recovery="Explicitly invalidate the render layer, then rerun from frame zero.",
            )
        if artifact_ref(Path(meta.output.path)) != meta.output:
            raise Stage2Error(
                "INVALID_RENDER_ARTIFACT", "Recorded render output hash or size changed."
            )
        verify_rendered_video(
            stage1=stage1,
            output_path=Path(meta.output.path),
            sam_shards=sam_shards,
            metas=metas,
            config=config,
        )
        if artifact_ref(Path(meta.output.path)) != meta.output:
            raise Stage2Error(
                "INVALID_RENDER_ARTIFACT", "Recorded output changed during verification."
            )
        if artifact_ref(stage1_path) != stage1.stage1_video:
            raise Stage2Error(
                "STAGE1_VIDEO_CHANGED", "Stage I changed during render reuse verification."
            )
        return RenderGenerationResult(artifact=artifact_ref(paths.render), meta=meta, reused=True)

    paths.render.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        dir=paths.render.parent,
        prefix=f".{final_output.name}.",
        suffix=".tmp.mp4",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    temporary.unlink(missing_ok=True)
    try:
        metrics = _render_frames(
            stage1_path=stage1_path,
            output_path=temporary,
            stage1=stage1,
            sam_shards=sam_shards,
            metas=metas,
            config=config,
        )
        rendered_temporary_ref = artifact_ref(temporary)
        if artifact_ref(stage1_path) != stage1.stage1_video:
            raise Stage2Error(
                "STAGE1_VIDEO_CHANGED", "Stage I changed while rendering was in progress."
            )
        verification = verify_rendered_video(
            stage1=stage1,
            output_path=temporary,
            sam_shards=sam_shards,
            metas=metas,
            config=config,
        )
        if artifact_ref(temporary) != rendered_temporary_ref:
            raise Stage2Error(
                "RENDER_OUTPUT_CHANGED",
                "Temporary output changed while technical verification was running.",
            )
        if artifact_ref(stage1_path) != stage1.stage1_video:
            raise Stage2Error(
                "STAGE1_VIDEO_CHANGED", "Stage I changed during technical verification."
            )
        output_ref = _promote_render_output(temporary, final_output)
        if (output_ref.sha256, output_ref.bytes) != (
            rendered_temporary_ref.sha256,
            rendered_temporary_ref.bytes,
        ):
            raise Stage2Error(
                "RENDER_OUTPUT_CHANGED", "Promoted output does not match verified bytes."
            )
    finally:
        temporary.unlink(missing_ok=True)
    verification.update(metrics)
    verification["output_bytes"] = output_ref.bytes
    encoder = {
        "codec": config.codec,
        "crf": config.crf,
        "preset": config.preset,
        "pixel_format": config.pixel_format,
        "audio": "stripped",
        "subtitles": "stripped",
        "data_streams": "stripped",
        "metadata": "stripped",
        "pixel_source": "stage1_video_only",
        "fill_yuv": config.fill_yuv,
        "dilation_pixels_at_1080p": config.dilation_pixels_at_1080p,
    }
    meta = RenderArtifactMeta(
        schema_version=STAGE2_SCHEMA_VERSION,
        artifact_type="render",
        fingerprint=fingerprint_value,
        stage1_video_sha256=stage1.stage1_video.sha256,
        mask_set_sha256=mask_set_sha256,
        output=output_ref,
        encoder=encoder,
        verification=verification,
    )
    artifact = write_immutable_json(paths.render, meta)
    if artifact_ref(final_output) != output_ref:
        raise Stage2Error(
            "RENDER_OUTPUT_CHANGED", "Promoted output changed while evidence was written."
        )
    return RenderGenerationResult(artifact=artifact, meta=meta, reused=False)


def finalize_verified_processing(
    *,
    stage1: StageIInput,
    paths: RunPaths,
    run_id: str,
    dino_artifact: ArtifactRef,
    sam_shards: tuple[ArtifactRef, ...],
    render_result: RenderGenerationResult,
    mode: str = "production",
    manual_seed_artifacts: tuple[ArtifactRef, ...] = (),
    label_artifacts: tuple[ArtifactRef, ...] = (),
    evidence_artifacts: tuple[ArtifactRef, ...] = (),
    review_flag_artifacts: tuple[ArtifactRef, ...] = (),
) -> ProcessingManifest:
    """Advance to processing complete only from immutable verified evidence."""
    if mode == "fake":
        raise Stage2Error(
            "INVALID_PROCESSING_MODE",
            "Real render evidence cannot be finalized with fake processing status.",
        )
    current = load_state(paths.state)
    if current is None or current.state not in {
        "SAM_COMPLETE",
        "RENDER_COMPLETE",
        "PROCESSING_COMPLETE",
    }:
        raise Stage2Error(
            "RENDER_PREREQUISITE_MISSING",
            "A run must have verified DINO and SAM layers before render finalization.",
        )
    if (current.run_id, current.clip_id, current.mode) != (run_id, stage1.clip_id, mode):
        raise Stage2Error("STATE_IDENTITY_MISMATCH", "Render finalization identity changed.")
    dino_actual = artifact_ref(Path(dino_artifact.path))
    render_actual = artifact_ref(Path(render_result.artifact.path))
    if dino_actual != dino_artifact or render_actual != render_result.artifact:
        raise Stage2Error(
            "PROCESSING_ARTIFACT_CHANGED",
            "DINO or render evidence changed before processing finalization.",
        )
    stored_render = render_meta_from_dict(read_json(Path(render_result.artifact.path)))
    if (
        canonical_json_bytes(stored_render) != canonical_json_bytes(render_result.meta)
        or stored_render.verification.get("passed") is not True
    ):
        raise Stage2Error(
            "INVALID_RENDER_ARTIFACT",
            "Only matching, technically verified render evidence may be finalized.",
        )
    if artifact_ref(Path(stored_render.output.path)) != stored_render.output:
        raise Stage2Error("INVALID_RENDER_ARTIFACT", "Rendered output changed before finalization.")
    metas = _load_render_shard_metas(sam_shards, stage1=stage1)
    if metas[0].dino_artifact_sha256 != dino_artifact.sha256:
        raise Stage2Error("DINO_SAM_MISMATCH", "SAM shards do not bind the selected DINO artifact.")
    if stored_render.mask_set_sha256 != artifact_set_sha256(sam_shards):
        raise Stage2Error(
            "SAM_RENDER_MISMATCH", "Render evidence does not bind the selected SAM shards."
        )
    private_groups = {
        "manual_seed_artifacts": manual_seed_artifacts,
        "label_artifacts": label_artifacts,
        "evidence_artifacts": evidence_artifacts,
        "review_flag_artifacts": review_flag_artifacts,
    }
    private_root = paths.private_root.resolve(strict=False)
    for group_name, references in private_groups.items():
        for reference in references:
            reference_path = Path(reference.path)
            if reference_path.is_symlink() or artifact_ref(reference_path) != reference:
                raise Stage2Error(
                    "PRIVATE_ARTIFACT_CHANGED", f"{group_name} contains a changed artifact."
                )
            try:
                reference_path.resolve(strict=True).relative_to(private_root)
            except (OSError, ValueError) as exc:
                raise Stage2Error(
                    "PRIVATE_ARTIFACT_OUTSIDE_DO_NOT_SHIP",
                    f"{group_name} must remain under DO-NOT-SHIP.",
                ) from exc
    advance_state(
        paths.state,
        run_id=run_id,
        clip_id=stage1.clip_id,
        mode=mode,
        target="RENDER_COMPLETE",
        completed_layers=("dino", "sam", "render"),
        reusable_layers=("dino", "sam", "render"),
    )
    review_flags = tuple(sorted({flag for meta in metas for flag in meta.review_flags}))
    audit_status = (
        "NEEDS_REVIEW"
        if stage1.stage1_status == "NEEDS_REVIEW" or review_flags
        else "PASS_AUTOMATED_TECHNICAL"
    )
    manifest = ProcessingManifest(
        schema_version=STAGE2_SCHEMA_VERSION,
        code_version=STAGE2_CODE_VERSION,
        run_id=run_id,
        clip_id=stage1.clip_id,
        mode=mode,
        processing_state="PROCESSING_COMPLETE",
        audit_status=audit_status,
        review_status="PENDING",
        stage1=stage1,
        dino_artifact=dino_artifact,
        sam_mask_shards=sam_shards,
        render_artifact=render_result.artifact,
        manual_seed_artifacts=manual_seed_artifacts,
        label_artifacts=label_artifacts,
        evidence_artifacts=evidence_artifacts,
        review_flag_artifacts=review_flag_artifacts,
    )
    write_immutable_json(paths.manifest, manifest)
    advance_state(
        paths.state,
        run_id=run_id,
        clip_id=stage1.clip_id,
        mode=mode,
        target="PROCESSING_COMPLETE",
        completed_layers=("dino", "sam", "render"),
        reusable_layers=("dino", "sam", "render"),
    )
    return manifest


def _artifact_ref_from_raw(raw: Any, label: str) -> ArtifactRef:
    try:
        reference = ArtifactRef(
            path=str(raw["path"]), sha256=str(raw["sha256"]), bytes=int(raw["bytes"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error(
            "INVALID_PROCESSING_MANIFEST", f"{label} artifact reference is malformed."
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{64}", reference.sha256) or reference.bytes < 1:
        raise Stage2Error("INVALID_PROCESSING_MANIFEST", f"{label} artifact reference is invalid.")
    return reference


def _manifest_private_artifacts(paths: RunPaths, raw: dict[str, Any]) -> tuple[ArtifactRef, ...]:
    references = []
    private_root = paths.private_root.resolve(strict=False)
    for field_name in (
        "manual_seed_artifacts",
        "label_artifacts",
        "evidence_artifacts",
        "review_flag_artifacts",
    ):
        values = raw.get(field_name, [])
        if not isinstance(values, list):
            raise Stage2Error("INVALID_PROCESSING_MANIFEST", f"{field_name} must be a list.")
        for index, value in enumerate(values):
            reference = _artifact_ref_from_raw(value, f"{field_name}[{index}]")
            path = Path(reference.path)
            try:
                path.resolve(strict=True).relative_to(private_root)
            except (OSError, ValueError) as exc:
                raise Stage2Error(
                    "PRIVATE_ARTIFACT_OUTSIDE_DO_NOT_SHIP",
                    f"{field_name} escaped DO-NOT-SHIP.",
                ) from exc
            if path.is_symlink() or artifact_ref(path) != reference:
                raise Stage2Error(
                    "PRIVATE_ARTIFACT_CHANGED",
                    f"{field_name} contains a changed private artifact.",
                )
            references.append(reference)
    if len({(reference.path, reference.sha256) for reference in references}) != len(references):
        raise Stage2Error(
            "INVALID_PROCESSING_MANIFEST", "Private artifact references must be unique."
        )
    return tuple(references)


def current_processing_evidence(
    paths: RunPaths,
) -> tuple[dict[str, Any], ArtifactRef, RenderArtifactMeta]:
    if not paths.manifest.exists():
        raise Stage2Error(
            "PROCESSING_MANIFEST_MISSING",
            "No completed processing manifest exists for this run.",
            recovery="Resume processing and complete technical verification first.",
        )
    raw = read_json(paths.manifest)
    manifest_ref = artifact_ref(paths.manifest)
    if (
        raw.get("schema_version") != STAGE2_SCHEMA_VERSION
        or raw.get("processing_state") != "PROCESSING_COMPLETE"
        or raw.get("mode") == "fake"
    ):
        raise Stage2Error(
            "PROCESSING_NOT_REVIEWABLE",
            "Only complete, non-fake processing evidence may be reviewed.",
            recovery="Run the production pipeline through technical verification.",
        )
    render_ref = _artifact_ref_from_raw(raw.get("render_artifact"), "render")
    if artifact_ref(Path(render_ref.path)) != render_ref:
        raise Stage2Error(
            "PROCESSING_ARTIFACT_CHANGED", "Render evidence changed after processing."
        )
    render_meta = render_meta_from_dict(read_json(Path(render_ref.path)))
    if render_meta.verification.get("passed") is not True:
        raise Stage2Error(
            "PROCESSING_NOT_REVIEWABLE", "Render technical verification did not pass."
        )
    if artifact_ref(Path(render_meta.output.path)) != render_meta.output:
        raise Stage2Error(
            "PROCESSING_ARTIFACT_CHANGED", "Rendered output changed after processing."
        )
    _manifest_private_artifacts(paths, raw)
    return raw, manifest_ref, render_meta


def _validated_reviewed_at(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Stage2Error(
            "INVALID_REVIEW_TIME",
            "reviewed_at must be an explicit UTC timestamp ending in Z.",
            recovery="Pass --reviewed-at in ISO-8601 UTC form, for example 2026-08-20T12:00:00Z.",
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Stage2Error("INVALID_REVIEW_TIME", "reviewed_at is not valid ISO-8601.") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise Stage2Error("INVALID_REVIEW_TIME", "reviewed_at must be UTC.")
    return value


def _private_label_acceptance_problems(raw_manifest: dict[str, Any]) -> list[str]:
    problems = []
    for artifact_index, raw_reference in enumerate(raw_manifest.get("label_artifacts", [])):
        reference = _artifact_ref_from_raw(raw_reference, f"label_artifacts[{artifact_index}]")
        envelope = read_json(Path(reference.path))
        if envelope.get("artifact_type") != "labels" or not isinstance(
            envelope.get("labels"), list
        ):
            problems.append(f"label artifact {artifact_index} is malformed")
            continue
        for value in envelope["labels"]:
            if not isinstance(value, dict):
                problems.append(f"label artifact {artifact_index} contains a malformed label")
                continue
            event_id = value.get("event_id", "unknown")
            if value.get("reviewer_disposition") != "accepted":
                problems.append(f"label {event_id} is not accepted")
            coverage = value.get("final_mask_coverage")
            if (
                value.get("label_kind") == "face_event"
                and value.get("visibility")
                in {
                    "visible",
                    "partial",
                }
                and (
                    isinstance(coverage, bool)
                    or not isinstance(coverage, (int, float))
                    or not math.isfinite(coverage)
                    or coverage < 0.95
                )
            ):
                problems.append(f"visible label {event_id} has final coverage below 0.95")
    return problems


def review_record_from_dict(raw: dict[str, Any]) -> ReviewRecord:
    expected_fields = {
        "schema_version",
        "review_id",
        "processing_manifest_sha256",
        "output_sha256",
        "review_status",
        "reviewer",
        "reviewed_at",
        "full_clip_reviewed",
        "flagged_intervals_reviewed",
        "notes",
    }
    try:
        record = ReviewRecord(
            schema_version=int(raw["schema_version"]),
            review_id=str(raw["review_id"]),
            processing_manifest_sha256=str(raw["processing_manifest_sha256"]),
            output_sha256=str(raw["output_sha256"]),
            review_status=str(raw["review_status"]),
            reviewer=str(raw["reviewer"]),
            reviewed_at=str(raw["reviewed_at"]),
            full_clip_reviewed=raw.get("full_clip_reviewed") is True,
            flagged_intervals_reviewed=raw.get("flagged_intervals_reviewed") is True,
            notes=str(raw.get("notes", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2Error("INVALID_REVIEW_RECORD", "Review record is malformed.") from exc
    problems = []
    if set(raw) != expected_fields:
        problems.append("review_record_fields")
    if (
        not isinstance(raw.get("schema_version"), int)
        or isinstance(raw.get("schema_version"), bool)
        or record.schema_version != STAGE2_SCHEMA_VERSION
    ):
        problems.append("schema_version")
    for field_name in (
        "review_id",
        "processing_manifest_sha256",
        "output_sha256",
        "review_status",
        "reviewer",
        "reviewed_at",
        "notes",
    ):
        if not isinstance(raw.get(field_name, ""), str):
            problems.append(field_name)
    for field_name in ("full_clip_reviewed", "flagged_intervals_reviewed"):
        if not isinstance(raw.get(field_name), bool):
            problems.append(field_name)
    if not _SAFE_COMPONENT.fullmatch(record.review_id) or record.review_id in {".", ".."}:
        problems.append("review_id")
    if not re.fullmatch(r"[0-9a-f]{64}", record.processing_manifest_sha256):
        problems.append("processing_manifest_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", record.output_sha256):
        problems.append("output_sha256")
    if record.review_status not in {"ACCEPTED", "REJECTED"}:
        problems.append("review_status")
    if not record.reviewer.strip():
        problems.append("reviewer")
    try:
        _validated_reviewed_at(record.reviewed_at)
    except Stage2Error:
        problems.append("reviewed_at")
    if record.review_status == "ACCEPTED" and not (
        record.full_clip_reviewed and record.flagged_intervals_reviewed
    ):
        problems.append("accepted_review_completeness")
    identity = {
        "schema_version": record.schema_version,
        "processing_manifest_sha256": record.processing_manifest_sha256,
        "output_sha256": record.output_sha256,
        "review_status": record.review_status,
        "reviewer": record.reviewer,
        "reviewed_at": record.reviewed_at,
        "full_clip_reviewed": record.full_clip_reviewed,
        "flagged_intervals_reviewed": record.flagged_intervals_reviewed,
        "notes": record.notes,
    }
    expected_review_id = f"review-{hashlib.sha256(canonical_json_bytes(identity)).hexdigest()}"
    if record.review_id != expected_review_id:
        problems.append("review_id_content_binding")
    if problems:
        raise Stage2Error(
            "INVALID_REVIEW_RECORD",
            "Review record failed validation.",
            details={"problems": problems},
        )
    return record


def create_review_record(
    paths: RunPaths,
    *,
    reviewer: str,
    reviewed_at: str,
    review_status: str,
    full_clip_reviewed: bool,
    flagged_intervals_reviewed: bool,
    notes: str = "",
) -> tuple[ArtifactRef, ReviewRecord]:
    assert_run_root_safe(paths)
    raw, manifest_ref, render_meta = current_processing_evidence(paths)
    bound_private = {reference.sha256 for reference in _manifest_private_artifacts(paths, raw)}
    if _private_artifact_hashes(paths).difference(bound_private):
        raise Stage2Error(
            "CORRECTION_REQUIRES_REPROCESSING",
            "Private corrections or evidence changed after this output was processed.",
            recovery="Resume with --recompute-from sam or render as appropriate, then review the new output.",
        )
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise Stage2Error("INVALID_REVIEWER", "Reviewer must be a non-empty identity.")
    reviewed_at = _validated_reviewed_at(reviewed_at)
    review_status = review_status.upper()
    if review_status not in {"ACCEPTED", "REJECTED"}:
        raise Stage2Error("INVALID_REVIEW_STATUS", "Review decision must be ACCEPTED or REJECTED.")
    if review_status == "ACCEPTED" and not (full_clip_reviewed and flagged_intervals_reviewed):
        raise Stage2Error(
            "INCOMPLETE_HUMAN_REVIEW",
            "Acceptance requires full-clip and flagged-interval review attestations.",
            recovery="Watch the full clip and every flagged interval, then rerun review.",
        )
    if review_status == "ACCEPTED":
        label_problems = _private_label_acceptance_problems(raw)
        if label_problems:
            raise Stage2Error(
                "LABEL_ACCEPTANCE_INCOMPLETE",
                "Private answer-sheet labels are not ready for output acceptance.",
                details={"problems": label_problems},
                recovery="Resolve every label, rerender corrections, and record final coverage before review.",
            )
    identity = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "processing_manifest_sha256": manifest_ref.sha256,
        "output_sha256": render_meta.output.sha256,
        "review_status": review_status,
        "reviewer": reviewer.strip(),
        "reviewed_at": reviewed_at,
        "full_clip_reviewed": full_clip_reviewed,
        "flagged_intervals_reviewed": flagged_intervals_reviewed,
        "notes": notes,
    }
    review_id = f"review-{hashlib.sha256(canonical_json_bytes(identity)).hexdigest()}"
    record = ReviewRecord(review_id=review_id, **identity)
    review_record_from_dict(_jsonable(record))
    if paths.reviews_dir.is_symlink():
        raise Stage2Error("UNSAFE_REVIEW_PATH", "Reviews directory may not be a symlink.")
    path = paths.reviews_dir / f"{review_id}.json"
    return write_immutable_json(path, record), record


def load_review_records(paths: RunPaths) -> tuple[tuple[ArtifactRef, ReviewRecord], ...]:
    assert_run_root_safe(paths)
    if not paths.reviews_dir.exists():
        return ()
    if paths.reviews_dir.is_symlink() or not paths.reviews_dir.is_dir():
        raise Stage2Error("UNSAFE_REVIEW_PATH", "Reviews path is not a safe directory.")
    records = []
    for path in sorted(paths.reviews_dir.glob("review-*.json")):
        if path.is_symlink():
            raise Stage2Error("UNSAFE_REVIEW_PATH", "Review record may not be a symlink.")
        records.append((artifact_ref(path), review_record_from_dict(read_json(path))))
    return tuple(records)


def effective_review_status(paths: RunPaths) -> tuple[str, ArtifactRef | None]:
    try:
        raw, manifest_ref, render_meta = current_processing_evidence(paths)
    except Stage2Error as exc:
        if exc.code in {"PROCESSING_MANIFEST_MISSING", "PROCESSING_NOT_REVIEWABLE"}:
            return "PENDING", None
        raise
    bound_private = {reference.sha256 for reference in _manifest_private_artifacts(paths, raw)}
    if _private_artifact_hashes(paths).difference(bound_private):
        return "PENDING", None
    matching = [
        (reference, record)
        for reference, record in load_review_records(paths)
        if record.processing_manifest_sha256 == manifest_ref.sha256
        and record.output_sha256 == render_meta.output.sha256
    ]
    if not matching:
        return "PENDING", None
    reference, record = max(
        matching,
        key=lambda item: (
            datetime.fromisoformat(item[1].reviewed_at[:-1] + "+00:00"),
            item[1].review_id,
        ),
    )
    return record.review_status, reference


def _private_artifact_hashes(paths: RunPaths) -> set[str]:
    if not paths.private_root.exists():
        return set()
    if paths.private_root.is_symlink() or not paths.private_root.is_dir():
        raise Stage2Error("UNSAFE_PRIVATE_PATH", "DO-NOT-SHIP is not a safe directory.")
    hashes = set()
    for path in paths.private_root.rglob("*"):
        if path.is_symlink():
            raise Stage2Error("UNSAFE_PRIVATE_PATH", "DO-NOT-SHIP contains a symlink.")
        if path.is_file():
            hashes.add(artifact_ref(path).sha256)
    return hashes


def release_check(paths: RunPaths, *, release_root: Path) -> dict[str, Any]:
    assert_run_root_safe(paths)
    raw, manifest_ref, render_meta = current_processing_evidence(paths)
    review_status, review_ref = effective_review_status(paths)
    problems = []
    if review_status != "ACCEPTED" or review_ref is None:
        problems.append("current processing output has no matching accepted human review")
    supplied_release_root = release_root.expanduser()
    if supplied_release_root.is_symlink():
        raise Stage2Error("UNSAFE_RELEASE_ROOT", "Release root may not be a symlink.")
    try:
        release_root = supplied_release_root.resolve(strict=True)
    except OSError as exc:
        raise Stage2Error(
            "RELEASE_ROOT_MISSING",
            "Release root does not exist.",
            recovery="Create the release staging directory and copy only public artifacts into it.",
        ) from exc
    if not release_root.is_dir() or release_root.is_symlink():
        raise Stage2Error("UNSAFE_RELEASE_ROOT", "Release root must be a real directory.")
    private_hashes = _private_artifact_hashes(paths)
    bound_private_hashes = {
        reference.sha256 for reference in _manifest_private_artifacts(paths, raw)
    }
    release_hashes = set()
    leaked_private = []
    leaked_internal = []
    forbidden_directories = []
    internal_hashes = {manifest_ref.sha256, artifact_ref(paths.state).sha256}
    internal_hashes.update(reference.sha256 for reference, _record in load_review_records(paths))
    render_reference = _artifact_ref_from_raw(raw.get("render_artifact"), "render")
    internal_hashes.add(render_reference.sha256)
    for path in release_root.rglob("*"):
        if path.is_symlink():
            problems.append(f"release package contains symlink: {path.relative_to(release_root)}")
            continue
        if path.is_dir() and path.name == "DO-NOT-SHIP":
            forbidden_directories.append(str(path.relative_to(release_root)))
        elif path.is_file():
            digest = artifact_ref(path).sha256
            release_hashes.add(digest)
            if digest in private_hashes:
                leaked_private.append(str(path.relative_to(release_root)))
            if digest in internal_hashes:
                leaked_internal.append(str(path.relative_to(release_root)))
    if forbidden_directories:
        problems.append(f"DO-NOT-SHIP directories are present: {forbidden_directories}")
    if leaked_private:
        problems.append(f"private artifact content is present: {leaked_private}")
    if leaked_internal:
        problems.append(f"internal processing/review artifacts are present: {leaked_internal}")
    unbound_private = sorted(private_hashes.difference(bound_private_hashes))
    if unbound_private:
        problems.append(
            f"private corrections/evidence are not bound to current processing: {unbound_private}"
        )
    if render_meta.output.sha256 not in release_hashes:
        problems.append("the accepted rendered output is absent from the release package")
    if _private_artifact_hashes(paths) != private_hashes:
        problems.append("DO-NOT-SHIP changed while release checks were running")
    if artifact_ref(Path(render_meta.output.path)) != render_meta.output:
        problems.append("accepted rendered output changed while release checks were running")
    if review_ref is not None and artifact_ref(Path(review_ref.path)) != review_ref:
        problems.append("human review record changed while release checks were running")
    if raw.get("review_status") not in {"PENDING", None}:
        problems.append("processing manifest improperly embeds a human review decision")
    if problems:
        raise Stage2Error(
            "RELEASE_CHECK_FAILED",
            "Release package is not safe to publish.",
            details={"problems": problems, "release_root": str(release_root)},
            recovery="Remove private/stale artifacts, stage the exact accepted output, and rerun release-check.",
        )
    return {
        "release_ready": True,
        "processing_manifest_sha256": manifest_ref.sha256,
        "output_sha256": render_meta.output.sha256,
        "review_record": review_ref,
        "release_root": str(release_root),
        "release_file_count": sum(1 for path in release_root.rglob("*") if path.is_file()),
        "private_artifact_count": len(private_hashes),
    }


def request_stop(paths: RunPaths) -> dict[str, Any]:
    assert_run_root_safe(paths)
    state = load_state(paths.state)
    if state is None:
        raise Stage2Error("RUN_NOT_FOUND", "Cannot stop a run that does not exist.")
    if state.state == "PROCESSING_COMPLETE":
        return {"stop_requested": False, "reason": "already-complete", "state": state}
    marker = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "artifact_type": "stop_request",
        "run_id": state.run_id,
        "clip_id": state.clip_id,
        "requested": True,
    }
    atomic_write_json(paths.stop_request, marker)
    return {"stop_requested": True, "state": state, "marker": str(paths.stop_request)}


def ensure_run_not_stopped(paths: RunPaths) -> None:
    assert_run_root_safe(paths)
    if not paths.stop_request.exists():
        return
    if paths.stop_request.is_symlink():
        raise Stage2Error("UNSAFE_CONTROL_PATH", "Stop request may not be a symlink.")
    marker = read_json(paths.stop_request)
    state = load_state(paths.state)
    if (
        state is None
        or marker.get("schema_version") != STAGE2_SCHEMA_VERSION
        or marker.get("artifact_type") != "stop_request"
        or marker.get("run_id") != state.run_id
        or marker.get("clip_id") != state.clip_id
        or marker.get("requested") is not True
    ):
        raise Stage2Error("INVALID_STOP_REQUEST", "Stop request marker is malformed.")
    raise Stage2Error(
        "STOP_REQUESTED",
        "Processing stopped cooperatively before starting the next layer.",
        details={"reusable_layers": state.reusable_layers},
        recovery=(
            "Resume explicitly with: bash scripts/runpod_stage2.sh resume "
            f"--work-dir {shlex.quote(str(paths.root.parents[2]))} "
            f"--run-id {shlex.quote(state.run_id)} --clip-id {shlex.quote(state.clip_id)}"
        ),
    )


def clear_stop_request(paths: RunPaths) -> bool:
    assert_run_root_safe(paths)
    if not paths.stop_request.exists():
        return False
    if paths.stop_request.is_symlink():
        raise Stage2Error("UNSAFE_CONTROL_PATH", "Stop request may not be a symlink.")
    try:
        paths.stop_request.unlink()
    except OSError as exc:
        raise Stage2Error("STOP_CLEAR_FAILED", "Could not clear the stop request.") from exc
    return True


def run_status(paths: RunPaths) -> dict[str, Any]:
    assert_run_root_safe(paths)
    state = load_state(paths.state)
    if state is None:
        raise Stage2Error("RUN_NOT_FOUND", f"No Stage II run state exists at {paths.state}")
    audit_status = None
    embedded_review_status = None
    manifest_ref = None
    if paths.manifest.exists():
        raw = read_json(paths.manifest)
        audit_status = raw.get("audit_status")
        embedded_review_status = raw.get("review_status")
        manifest_ref = artifact_ref(paths.manifest)
    if embedded_review_status == "NOT_REVIEWABLE_FAKE":
        review_status, review_ref = embedded_review_status, None
    else:
        review_status, review_ref = effective_review_status(paths)
    return {
        "processing_state": state.state,
        "audit_status": audit_status,
        "review_status": review_status,
        "completed_layers": state.completed_layers,
        "reusable_layers": state.reusable_layers,
        "last_error": state.last_error,
        "stop_requested": paths.stop_request.exists(),
        "processing_manifest": manifest_ref,
        "review_record": review_ref,
        "artifact_locations": {
            "run_root": str(paths.root),
            "dino": str(paths.dino),
            "sam": str(paths.sam_dir),
            "frames": str(paths.frames_dir),
            "render": str(paths.render),
            "private": str(paths.private_root),
            "reviews": str(paths.reviews_dir),
        },
    }


class MetaSam2VideoAdapter:
    """Lazy boundary around Meta's official SAM2.1 video predictor.

    The checkpoint and pinned runtime must already exist locally. This adapter
    verifies their hashes and source revision before model loading. A window
    input is a directory of sequential JPEG frames whose local index zero
    corresponds to ``window.frame_start``.
    """

    identity = ModelIdentity(
        name=SAM_MODEL_ID,
        revision=SAM_MODEL_REVISION,
        sha256=SAM_MODEL_WEIGHTS_SHA256,
    )

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str = "cuda",
    ) -> None:
        checkpoint_path = resolve_input_file(checkpoint_path, "SAM2.1 checkpoint")
        if sha256_file(checkpoint_path) != SAM_MODEL_WEIGHTS_SHA256:
            raise Stage2Error(
                "SAM_CHECKPOINT_HASH_MISMATCH",
                "SAM2.1 checkpoint hash does not match the pinned model.",
            )
        try:
            import sam2
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise Stage2Error(
                "SAM_DEPENDENCIES_MISSING",
                "SAM2 runtime dependencies are not installed.",
                recovery="Run the Stage II setup command before real SAM inference.",
            ) from exc
        self._torch = torch
        self.device = device
        package_file = getattr(sam2, "__file__", None)
        if not isinstance(package_file, str) or not package_file:
            raise Stage2Error(
                "SAM_RUNTIME_UNVERIFIABLE",
                "Installed SAM2 package did not report its source location.",
            )
        package_root = Path(package_file).resolve().parent
        revision = installed_sam_runtime_revision(package_root)
        self.runtime_identity = verified_sam_runtime_identity(
            package_root,
            revision=revision,
            torch_version=str(getattr(torch, "__version__", "")),
            cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
        )
        try:
            self.predictor = build_sam2_video_predictor(
                SAM_MODEL_CONFIG,
                str(checkpoint_path),
                device=device,
                mode="eval",
            )
        except Exception as exc:
            raise Stage2Error(
                "SAM_MODEL_LOAD_FAILED",
                "Pinned SAM2.1 Hiera Large model could not be loaded.",
                details={"cause": str(exc)},
                recovery="Run the Stage II setup model-load check.",
            ) from exc

    def _raw_masks_from_output(
        self,
        *,
        frame_idx: int,
        object_ids: Any,
        logits: Any,
        id_to_prompt: dict[int, SamPrompt],
        direction: str,
        width: int,
        height: int,
    ) -> tuple[RawSamMask, ...]:
        masks = []
        for position, object_id in enumerate(object_ids):
            prompt = id_to_prompt.get(int(object_id))
            if prompt is None:
                continue
            binary = (logits[position, 0] > 0).to(dtype=self._torch.uint8).cpu()
            if tuple(binary.shape) != (height, width):
                raise Stage2Error("INVALID_SAM_OUTPUT", "SAM returned the wrong mask dimensions.")
            nonzero = binary.nonzero(as_tuple=False)
            if nonzero.numel() == 0:
                crop_box = (0, 0, 1, 1)
                pixels = b"\x00"
            else:
                y1 = int(nonzero[:, 0].min().item())
                y2 = int(nonzero[:, 0].max().item()) + 1
                x1 = int(nonzero[:, 1].min().item())
                x2 = int(nonzero[:, 1].max().item()) + 1
                crop_box = (x1, y1, x2, y2)
                pixels = bytes(binary[y1:y2, x1:x2].contiguous().view(-1).tolist())
            masks.append(
                RawSamMask(
                    frame_idx=frame_idx,
                    prompt_id=prompt.prompt_id,
                    crop_box=crop_box,
                    pixels=pixels,
                    direction=direction,
                )
            )
        return tuple(masks)

    def propagate_window(
        self,
        *,
        window_input: Any,
        window: TemporalWindow,
        prompts: tuple[SamPrompt, ...],
        width: int,
        height: int,
        precision: str,
    ) -> SamPropagationResult:
        if not prompts:
            return SamPropagationResult(masks=(), forward_complete=True, reverse_complete=True)
        frame_dir = Path(window_input).resolve(strict=True)
        if not frame_dir.is_dir():
            raise Stage2Error("INVALID_SAM_WINDOW", "SAM window input must be a frame directory.")
        dtype = {
            "float32": self._torch.float32,
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
        }[precision]
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
            autocast = self._torch.autocast("cuda", dtype=dtype)
        else:
            autocast = contextlib.nullcontext()
        started = time.perf_counter()
        masks: list[RawSamMask] = []
        forward_seen: set[int] = set()
        reverse_seen: set[int] = set()
        state = None
        try:
            with self._torch.inference_mode(), autocast:
                state = self.predictor.init_state(video_path=str(frame_dir))
                id_to_prompt = {index + 1: prompt for index, prompt in enumerate(prompts)}
                for object_id, prompt in id_to_prompt.items():
                    self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=prompt.anchor_frame - window.frame_start,
                        obj_id=object_id,
                        box=prompt.box,
                    )
                earliest = min(prompt.anchor_frame for prompt in prompts) - window.frame_start
                latest = max(prompt.anchor_frame for prompt in prompts) - window.frame_start
                for local_frame, object_ids, logits in self.predictor.propagate_in_video(
                    state,
                    start_frame_idx=earliest,
                    max_frame_num_to_track=window.n_frames - earliest,
                    reverse=False,
                ):
                    global_frame = window.frame_start + int(local_frame)
                    forward_seen.add(global_frame)
                    masks.extend(
                        self._raw_masks_from_output(
                            frame_idx=global_frame,
                            object_ids=object_ids,
                            logits=logits,
                            id_to_prompt=id_to_prompt,
                            direction="forward",
                            width=width,
                            height=height,
                        )
                    )
                if latest > 0:
                    for local_frame, object_ids, logits in self.predictor.propagate_in_video(
                        state,
                        start_frame_idx=latest,
                        max_frame_num_to_track=latest + 1,
                        reverse=True,
                    ):
                        global_frame = window.frame_start + int(local_frame)
                        reverse_seen.add(global_frame)
                        masks.extend(
                            self._raw_masks_from_output(
                                frame_idx=global_frame,
                                object_ids=object_ids,
                                logits=logits,
                                id_to_prompt=id_to_prompt,
                                direction="reverse",
                                width=width,
                                height=height,
                            )
                        )
        finally:
            if state is not None:
                reset = getattr(self.predictor, "reset_state", None)
                if callable(reset):
                    reset(state)
        peak_vram = 0
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            peak_vram = int(self._torch.cuda.max_memory_allocated())
        expected_forward = set(range(window.frame_start + earliest, window.frame_end + 1))
        expected_reverse = (
            set()
            if latest == 0
            else set(range(window.frame_start, window.frame_start + latest + 1))
        )
        return SamPropagationResult(
            masks=tuple(masks),
            forward_complete=expected_forward.issubset(forward_seen),
            reverse_complete=expected_reverse.issubset(reverse_seen),
            runtime_seconds=time.perf_counter() - started,
            peak_vram_bytes=peak_vram,
        )


@dataclass(slots=True, frozen=True)
class FakeImage:
    width: int
    height: int
    crop_origin: tuple[int, int] = (0, 0)

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def crop(self, box: tuple[int, int, int, int]) -> FakeImage:
        x1, y1, x2, y2 = box
        return FakeImage(
            width=x2 - x1,
            height=y2 - y1,
            crop_origin=(self.crop_origin[0] + x1, self.crop_origin[1] + y1),
        )


class FakeDinoAdapter:
    identity = ModelIdentity(name="fake-grounding-dino", revision="milestone-2", sha256="f" * 64)

    def __init__(self) -> None:
        self.calls = 0

    def infer_batch(
        self,
        images: list[Any],
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> list[tuple[RawDinoDetection, ...]]:
        del prompt, box_threshold, text_threshold
        self.calls += 1
        results = []
        for image in images:
            width, height = _image_size(image)
            results.append(
                (
                    RawDinoDetection(
                        box=(width * 0.4, height * 0.4, width * 0.6, height * 0.6),
                        score=0.42,
                    ),
                )
            )
        return results

    def reset_peak_vram(self) -> None:
        return None

    def peak_vram_bytes(self) -> int:
        return 0


class FakeSamAdapter:
    identity = ModelIdentity(name="fake-sam2", revision="milestone-3", sha256="a" * 64)
    runtime_identity = SamRuntimeIdentity(
        repository="fake://sam2-runtime",
        revision="milestone-3",
        source_tree_sha256="b" * 64,
        model_config="fake-sam2-config",
        model_config_sha256="c" * 64,
        torch_version="fake-torch",
        cuda_version="none",
    )

    def __init__(self) -> None:
        self.calls = 0
        self.windows: list[TemporalWindow] = []

    def propagate_window(
        self,
        *,
        window_input: Any,
        window: TemporalWindow,
        prompts: tuple[SamPrompt, ...],
        width: int,
        height: int,
        precision: str,
    ) -> SamPropagationResult:
        del window_input, precision
        self.calls += 1
        self.windows.append(window)
        masks = []
        for prompt in prompts:
            x1 = max(0, math.floor(prompt.box[0]))
            y1 = max(0, math.floor(prompt.box[1]))
            x2 = min(width, math.ceil(prompt.box[2]))
            y2 = min(height, math.ceil(prompt.box[3]))
            pixels = b"\x01" * ((x2 - x1) * (y2 - y1))
            for frame_idx in range(window.frame_start, window.frame_end + 1):
                direction = (
                    "anchor"
                    if frame_idx == prompt.anchor_frame
                    else "forward"
                    if frame_idx > prompt.anchor_frame
                    else "reverse"
                )
                masks.append(
                    RawSamMask(
                        frame_idx=frame_idx,
                        prompt_id=prompt.prompt_id,
                        crop_box=(x1, y1, x2, y2),
                        pixels=pixels,
                        direction=direction,
                    )
                )
        return SamPropagationResult(
            masks=tuple(masks),
            forward_complete=True,
            reverse_complete=True,
            runtime_seconds=0.0,
            peak_vram_bytes=0,
        )


def run_fake_pipeline(
    *,
    source_video: Path,
    stage1_video: Path,
    stage1_manifest: Path,
    work_dir: Path,
    run_id: str,
    probe_fn: Callable[[Path], VideoFacts] = probe_video,
) -> ProcessingManifest:
    validated = validate_stage1(source_video, stage1_video, stage1_manifest, probe_fn=probe_fn)
    paths = build_run_paths(work_dir, run_id, validated.clip_id)
    ensure_run_not_stopped(paths)
    resolved_inputs = {
        Path(validated.source.path),
        Path(validated.stage1_video.path),
        Path(validated.stage1_manifest.path),
    }
    if any(paths.root == item or paths.root in item.parents for item in resolved_inputs):
        raise Stage2Error(
            "UNSAFE_OUTPUT_LOCATION",
            "A Stage II run directory may not contain any input artifact.",
        )

    current = load_state(paths.state)
    if current is None:
        transition_state(
            paths.state,
            run_id=run_id,
            clip_id=validated.clip_id,
            mode="fake",
            target="PENDING",
        )
    advance_state(
        paths.state,
        run_id=run_id,
        clip_id=validated.clip_id,
        mode="fake",
        target="VALIDATED",
    )

    dino = FakeDinoAdapter()
    dino_config = DinoGenerationConfig(
        model=dino.identity,
        preprocessing=(
            ("color", "synthetic-RGB"),
            ("resize", "none"),
            ("normalization", "none"),
            ("frame_decoder", "fake-image"),
        ),
    )
    dino_result = generate_dino_proposals(
        stage1=validated,
        paths=paths,
        config=dino_config,
        adapter=dino,
        frame_loader=lambda _frame_idx: FakeImage(
            validated.source_video.display_width,
            validated.source_video.display_height,
        ),
    )
    dino_ref = dino_result.artifact
    ensure_run_not_stopped(paths)
    advance_state(
        paths.state,
        run_id=run_id,
        clip_id=validated.clip_id,
        mode="fake",
        target="DINO_COMPLETE",
        completed_layers=("dino",),
        reusable_layers=("dino",),
    )

    sam = FakeSamAdapter()
    sam_window_size = min(10, validated.source_video.n_frames)
    sam_result = generate_sam_mask_shards(
        stage1=validated,
        paths=paths,
        dino_artifact=dino_ref,
        dino_meta=dino_result.meta,
        config=SamGenerationConfig(
            model=sam.identity,
            runtime=sam.runtime_identity,
            accepted_proposal_threshold=0.20,
            window_size=sam_window_size,
            window_overlap=min(2, sam_window_size // 2),
            precision="float32",
        ),
        adapter=sam,
        window_loader=lambda window: SamWindowInput(
            source_sha256=validated.source.sha256,
            frame_start=window.frame_start,
            frame_end=window.frame_end,
            frame_width=validated.source_video.display_width,
            frame_height=validated.source_video.display_height,
            payload=None,
        ),
    )
    sam_refs = sam_result.shards
    ensure_run_not_stopped(paths)
    mask_set_sha256 = artifact_set_sha256(sam_refs)
    advance_state(
        paths.state,
        run_id=run_id,
        clip_id=validated.clip_id,
        mode="fake",
        target="SAM_COMPLETE",
        completed_layers=("dino", "sam"),
        reusable_layers=("dino", "sam"),
    )

    render_payload = {
        "stage1_video_sha256": validated.stage1_video.sha256,
        "mask_set_sha256": mask_set_sha256,
        "dilation": {"pixels_at_1080p": 8},
        "encoder": {"mode": "fake-no-video"},
    }
    fake_output = write_immutable_json(
        paths.render.with_name("fake-output.json"),
        {
            "schema_version": STAGE2_SCHEMA_VERSION,
            "mode": "fake",
            "clip_id": validated.clip_id,
            "stage1_video_sha256": validated.stage1_video.sha256,
            "mask_set_sha256": mask_set_sha256,
        },
    )
    render_meta = RenderArtifactMeta(
        schema_version=STAGE2_SCHEMA_VERSION,
        artifact_type="render",
        fingerprint=render_fingerprint(render_payload),
        stage1_video_sha256=validated.stage1_video.sha256,
        mask_set_sha256=mask_set_sha256,
        output=fake_output,
        encoder={"mode": "fake-no-video"},
        verification={"mode": "fake", "passed": True, "publishable": False},
    )
    render_ref = write_immutable_json(paths.render, render_meta)
    ensure_run_not_stopped(paths)
    advance_state(
        paths.state,
        run_id=run_id,
        clip_id=validated.clip_id,
        mode="fake",
        target="RENDER_COMPLETE",
        completed_layers=("dino", "sam", "render"),
        reusable_layers=("dino", "sam", "render"),
    )

    manifest = ProcessingManifest(
        schema_version=STAGE2_SCHEMA_VERSION,
        code_version=STAGE2_CODE_VERSION,
        run_id=run_id,
        clip_id=validated.clip_id,
        mode="fake",
        processing_state="PROCESSING_COMPLETE",
        audit_status="NOT_RUN_FAKE",
        review_status="NOT_REVIEWABLE_FAKE",
        stage1=validated,
        dino_artifact=dino_ref,
        sam_mask_shards=sam_refs,
        render_artifact=render_ref,
    )
    write_immutable_json(paths.manifest, manifest)
    advance_state(
        paths.state,
        run_id=run_id,
        clip_id=validated.clip_id,
        mode="fake",
        target="PROCESSING_COMPLETE",
        completed_layers=("dino", "sam", "render"),
        reusable_layers=("dino", "sam", "render"),
    )
    return manifest


def _remove_layer_path(paths: RunPaths, candidate: Path) -> None:
    assert_run_root_safe(paths)
    if candidate.is_symlink():
        raise Stage2Error(
            "UNSAFE_INVALIDATION_PATH",
            f"Refusing to follow symlink during invalidation: {candidate}",
        )
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(paths.root)
    except ValueError as exc:
        raise Stage2Error("UNSAFE_INVALIDATION_PATH", "Invalidation escaped the run root.") from exc
    if not candidate.exists() and not candidate.is_symlink():
        return
    try:
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()
    except OSError as exc:
        raise Stage2Error(
            "INVALIDATION_FAILED",
            f"Could not invalidate {candidate}",
            details={"cause": str(exc)},
        ) from exc


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _archive_invalidated_reviews(paths: RunPaths, *, layer: str) -> None:
    """Preserve prior review records before invalidate_from deletes reviews_dir.

    Recompute is designed to invalidate every earlier human review record
    (see docs/stage2-operator.md), but the earlier implementation deleted
    them outright with no trace, letting a rejected review be silently
    discarded and replaced by a fresh acceptance of byte-identical
    recomputed content. Append-only history keeps the invalidation itself
    (not just its cause) auditable.
    """
    records = load_review_records(paths)
    if not records:
        return
    history_path = paths.root / "reviews-history.jsonl"
    if history_path.is_symlink():
        raise Stage2Error("UNSAFE_REVIEW_PATH", "Review history log may not be a symlink.")
    existing = b""
    if history_path.exists():
        try:
            existing = history_path.read_bytes()
        except OSError as exc:
            raise Stage2Error(
                "ARTIFACT_READ_FAILED",
                f"Could not read existing review history log {history_path}",
                details={"cause": str(exc)},
            ) from exc
    invalidated_at = _utc_now_z()
    appended = b"".join(
        canonical_json_bytes(
            {
                **_jsonable(record),
                "review_record_sha256": reference.sha256,
                "invalidated_from_layer": layer,
                "invalidated_at": invalidated_at,
            }
        )
        + b"\n"
        for reference, record in records
    )
    atomic_write_bytes(history_path, existing + appended)


def invalidate_from(paths: RunPaths, *, layer: str) -> ProcessingState:
    state = load_state(paths.state)
    if state is None:
        raise Stage2Error("RUN_NOT_FOUND", "Cannot invalidate a run that does not exist.")
    if layer not in {"dino", "sam", "render"}:
        raise Stage2Error("INVALID_RECOMPUTE_LAYER", "Layer must be dino, sam, or render.")
    prerequisite_met = {
        "dino": state.state not in {"PENDING", "FAILED"} or bool(state.reusable_layers),
        "sam": "dino" in state.completed_layers,
        "render": "sam" in state.completed_layers,
    }[layer]
    if not prerequisite_met:
        raise Stage2Error(
            "RECOMPUTE_PREREQUISITE_MISSING",
            f"Cannot recompute from {layer} before its upstream contract is complete.",
        )
    _archive_invalidated_reviews(paths, layer=layer)
    targets = {
        "dino": (paths.dino.parent, paths.sam_dir, paths.frames_dir, paths.render.parent),
        "sam": (paths.sam_dir, paths.frames_dir, paths.render.parent),
        "render": (paths.render.parent,),
    }
    for target in targets[layer]:
        _remove_layer_path(paths, target)
    for target in (paths.manifest, paths.reviews_dir):
        _remove_layer_path(paths, target)
    completed = {
        "dino": (),
        "sam": ("dino",),
        "render": ("dino", "sam"),
    }[layer]
    target_state = {
        "dino": "VALIDATED",
        "sam": "DINO_COMPLETE",
        "render": "SAM_COMPLETE",
    }[layer]
    reset = ProcessingState(
        schema_version=STAGE2_SCHEMA_VERSION,
        run_id=state.run_id,
        clip_id=state.clip_id,
        mode=state.mode,
        state=target_state,
        completed_layers=completed,
        reusable_layers=completed,
    )
    _validate_state_record(reset, paths.state)
    atomic_write_json(paths.state, reset)
    clear_stop_request(paths)
    return reset


def prepare_resume(paths: RunPaths) -> ProcessingState:
    state = load_state(paths.state)
    if state is None:
        raise Stage2Error("RUN_NOT_FOUND", "Cannot resume a run that does not exist.")
    clear_stop_request(paths)
    if state.state != "FAILED":
        return state
    reusable = tuple(layer for layer in ("dino", "sam", "render") if layer in state.reusable_layers)
    if "render" in reusable:
        target = "RENDER_COMPLETE"
        completed = ("dino", "sam", "render")
    elif "sam" in reusable:
        target = "SAM_COMPLETE"
        completed = ("dino", "sam")
    elif "dino" in reusable:
        target = "DINO_COMPLETE"
        completed = ("dino",)
    else:
        target = "PENDING"
        completed = ()
    resumed = ProcessingState(
        schema_version=STAGE2_SCHEMA_VERSION,
        run_id=state.run_id,
        clip_id=state.clip_id,
        mode=state.mode,
        state=target,
        completed_layers=completed,
        reusable_layers=tuple(layer for layer in completed if layer in reusable),
    )
    _validate_state_record(resumed, paths.state)
    atomic_write_json(paths.state, resumed)
    return resumed


def stage2_asset_paths(workspace_root: Path) -> dict[str, Path]:
    root = workspace_root.expanduser().resolve(strict=False)
    return {
        "workspace": root,
        "uv_cache": root / ".uv-cache",
        "hf_home": root / ".cache" / "huggingface",
        "torch_home": root / ".cache" / "torch",
        "dino_weights": root / "models" / "stage2" / "dino" / "model.safetensors",
        "sam_checkpoint": root / "models" / "stage2" / "sam2" / "sam2.1_hiera_large.pt",
        "sam_runtime": root / "models" / "stage2" / "sam2-runtime",
        "asset_manifest": root / "models" / "stage2" / "assets.manifest.json",
        "work_dir": root / "stage2",
    }


def doctor_stage2(workspace_root: Path) -> dict[str, Any]:
    paths = stage2_asset_paths(workspace_root)
    tools = {name: shutil.which(name) for name in ("uv", "ffmpeg", "ffprobe", "git")}
    problems = [f"{name} not found on PATH" for name, path in tools.items() if path is None]
    assets = {}
    for name, expected in (
        ("dino_weights", DINO_MODEL_WEIGHTS_SHA256),
        ("sam_checkpoint", SAM_MODEL_WEIGHTS_SHA256),
    ):
        path = paths[name]
        if path.exists() and path.is_file() and not path.is_symlink():
            actual = sha256_file(path)
            assets[name] = {"path": str(path), "sha256": actual, "verified": actual == expected}
            if actual != expected:
                problems.append(f"{name} hash mismatch")
        else:
            assets[name] = {"path": str(path), "verified": False, "missing": True}
    runtime = paths["sam_runtime"]
    runtime_revision = None
    runtime_remote = None
    runtime_clean = False
    config_sha256 = None

    def git_probe(*arguments: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                [str(tools["git"]), "-C", str(runtime), *arguments],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    if (
        tools.get("git")
        and runtime.is_dir()
        and not runtime.is_symlink()
        and (runtime / ".git").exists()
    ):
        result = git_probe("rev-parse", "HEAD")
        if result is not None and result.returncode == 0:
            runtime_revision = result.stdout.strip()
        remote_result = git_probe("remote", "get-url", "origin")
        status_result = git_probe("status", "--porcelain", "--untracked-files=no")
        runtime_remote = (
            remote_result.stdout.strip()
            if remote_result is not None and remote_result.returncode == 0
            else None
        )
        runtime_clean = (
            status_result is not None
            and status_result.returncode == 0
            and not status_result.stdout.strip()
        )
        config_path = runtime / "sam2" / SAM_MODEL_CONFIG
        if config_path.is_file() and not config_path.is_symlink():
            config_sha256 = sha256_file(config_path)
        if runtime_revision != SAM_RUNTIME_REVISION:
            problems.append("SAM2 runtime revision mismatch")
        if _normalized_git_url(runtime_remote or "") != _normalized_git_url(SAM_RUNTIME_REPOSITORY):
            problems.append("SAM2 runtime repository mismatch")
        if not runtime_clean:
            problems.append("SAM2 runtime has tracked modifications")
        if config_sha256 != SAM_MODEL_CONFIG_SHA256:
            problems.append("SAM2 model configuration hash mismatch")
    assets["sam_runtime"] = {
        "path": str(runtime),
        "revision": runtime_revision,
        "repository": runtime_remote,
        "clean": runtime_clean,
        "model_config_sha256": config_sha256,
        "verified": (
            runtime_revision == SAM_RUNTIME_REVISION
            and _normalized_git_url(runtime_remote or "")
            == _normalized_git_url(SAM_RUNTIME_REPOSITORY)
            and runtime_clean
            and config_sha256 == SAM_MODEL_CONFIG_SHA256
        ),
    }
    tools_ready = all(tools.values())
    assets_ready = all(item.get("verified") is True for item in assets.values())
    return {
        "status": (
            "READY_FOR_MILESTONE_6_GPU_SMOKE" if tools_ready and assets_ready else "SETUP_REQUIRED"
        ),
        "tools": tools,
        "persistent_paths": {name: str(path) for name, path in paths.items()},
        "assets": assets,
        "problems": problems,
        "gpu_execution_enabled": False,
        "next_command": "bash scripts/runpod_setup_stage2.sh",
    }


def run_offline_gpu_smoke(
    workspace_root: Path,
    *,
    doctor_fn: Callable[[Path], dict[str, Any]] = doctor_stage2,
    torch_module: Any | None = None,
    dino_factory: Callable[[], Any] | None = None,
    sam_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load and exercise DINO then SAM2 sequentially with networking disabled."""
    doctor = doctor_fn(workspace_root)
    if doctor.get("status") != "READY_FOR_MILESTONE_6_GPU_SMOKE":
        raise Stage2Error(
            "STAGE2_SETUP_INCOMPLETE",
            "Pinned Stage II assets are not ready for the real-GPU smoke test.",
            details={"problems": doctor.get("problems", [])},
            recovery="Run bash scripts/runpod_setup_stage2.sh, then retry doctor --load-models.",
        )
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise Stage2Error(
                "TORCH_NOT_FOUND", "Torch is required for the real-GPU smoke test."
            ) from exc
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise Stage2Error(
            "CUDA_UNAVAILABLE",
            "A CUDA device is required for the Milestone 6 model smoke test.",
            recovery="Start a CUDA-capable RunPod and rerun the persistent Stage II setup.",
        )
    device_index = 0
    properties = cuda.get_device_properties(device_index)
    capability = cuda.get_device_capability(device_index)
    sam_precision = "bfloat16" if int(capability[0]) >= 8 else "float16"
    device = {
        "name": cuda.get_device_name(device_index),
        "capability": tuple(int(value) for value in capability),
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(getattr(torch_module, "__version__", "")),
        "cuda_version": str(getattr(getattr(torch_module, "version", None), "cuda", "")),
    }
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise Stage2Error(
            "PILLOW_NOT_FOUND", "Pillow is required for the real-GPU smoke test."
        ) from exc

    paths = stage2_asset_paths(workspace_root)
    dino_factory = dino_factory or (
        lambda: TransformersGroundingDinoAdapter(
            device="cuda",
            model_path=paths["dino_weights"].parent,
        )
    )
    sam_factory = sam_factory or (lambda **kwargs: MetaSam2VideoAdapter(**kwargs))
    work_dir = paths["work_dir"] / "model-smoke"
    work_dir.mkdir(parents=True, exist_ok=True)
    dino_started = time.perf_counter()
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()
    dino = dino_factory()
    synthetic = Image.new("RGB", (128, 96), (127, 127, 127))
    dino_results = dino.infer_batch(
        [synthetic],
        prompt=DINO_PROMPT,
        box_threshold=DINO_PROPOSAL_FLOOR,
        text_threshold=DINO_TEXT_THRESHOLD,
    )
    cuda.synchronize()
    dino_metrics = {
        "model": _jsonable(getattr(dino, "identity", TransformersGroundingDinoAdapter.identity)),
        "loaded": True,
        "inference_completed": True,
        "detections": len(dino_results[0]),
        "runtime_seconds": time.perf_counter() - dino_started,
        "peak_vram_bytes": int(cuda.max_memory_allocated()),
    }
    del dino, dino_results, synthetic
    gc.collect()
    cuda.empty_cache()
    cuda.synchronize()
    dino_residual_vram = int(cuda.memory_allocated())

    sam_started = time.perf_counter()
    cuda.reset_peak_memory_stats()
    sam = sam_factory(checkpoint_path=paths["sam_checkpoint"], device="cuda")
    sam_runtime = getattr(sam, "runtime_identity", None)
    window = TemporalWindow(index=0, frame_start=0, frame_end=4)
    prompt = SamPrompt(
        prompt_id="gpu-smoke-box",
        prompt_kind="manual",
        anchor_frame=2,
        box=(40.0, 24.0, 88.0, 72.0),
        source_id="synthetic-smoke",
    )
    with tempfile.TemporaryDirectory(dir=work_dir, prefix="frames-") as temporary:
        frame_dir = Path(temporary)
        for frame_idx in range(window.n_frames):
            frame = Image.new("RGB", (128, 96), (96, 96, 96))
            draw = ImageDraw.Draw(frame)
            offset = frame_idx - 2
            draw.ellipse((48 + offset, 28, 80 + offset, 68), fill=(180, 180, 180))
            frame.save(frame_dir / f"{frame_idx:06d}.jpg", format="JPEG", quality=95)
        propagation = sam.propagate_window(
            window_input=frame_dir,
            window=window,
            prompts=(prompt,),
            width=128,
            height=96,
            precision=sam_precision,
        )
    cuda.synchronize()
    if not propagation.forward_complete or not propagation.reverse_complete:
        raise Stage2Error(
            "SAM_GPU_SMOKE_INCOMPLETE",
            "SAM2 did not complete both forward and reverse synthetic propagation.",
        )
    sam_metrics = {
        "model": _jsonable(getattr(sam, "identity", MetaSam2VideoAdapter.identity)),
        "runtime": _jsonable(sam_runtime),
        "loaded": True,
        "anchor_mask_completed": any(mask.frame_idx == 2 for mask in propagation.masks),
        "forward_complete": propagation.forward_complete,
        "reverse_complete": propagation.reverse_complete,
        "mask_count": len(propagation.masks),
        "precision": sam_precision,
        "runtime_seconds": time.perf_counter() - sam_started,
        "peak_vram_bytes": int(cuda.max_memory_allocated()),
    }
    if not sam_metrics["anchor_mask_completed"]:
        raise Stage2Error("SAM_GPU_SMOKE_INCOMPLETE", "SAM2 did not return the anchor-frame mask.")
    del sam, propagation
    gc.collect()
    cuda.empty_cache()
    cuda.synchronize()
    return {
        "status": "PASS_OFFLINE_GPU_SMOKE",
        "job_version": STAGE2_JOB_VERSION,
        "code_version": STAGE2_CODE_VERSION,
        "networking_disabled": True,
        "models_loaded_sequentially": True,
        "verified_assets": doctor.get("assets", {}),
        "device": device,
        "dino": dino_metrics,
        "dino_residual_vram_before_sam_bytes": dino_residual_vram,
        "sam": sam_metrics,
    }


def operator_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "dry_run": True,
        "command": args.command,
        "writes": [],
        "gpu_execution_enabled": False,
    }
    if args.command in {"pilot", "sweep", "run"}:
        payload.update(
            {
                "planned_layers": ("validate", "dino", "sam", "render", "technical-verify"),
                "pixel_source": "stage1-video-only",
                "next_boundary": "Milestone 6 real-GPU smoke test",
                "private_inputs": {
                    "manual_seeds": str(args.manual_seeds) if args.manual_seeds else None,
                    "labels": str(args.labels) if args.labels else None,
                },
            }
        )
    elif args.command == "review":
        payload["planned_review_status"] = args.decision
    elif args.command == "release-check":
        payload["planned_checks"] = (
            "processing",
            "human-review",
            "output-hash",
            "private-content",
        )
    elif args.command == "stop":
        payload["planned_marker"] = "control/stop-request.json"
    elif args.command == "resume":
        payload["recompute_from"] = args.recompute_from
    return payload


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--stage1-video", type=Path, required=True)
    parser.add_argument("--stage1-manifest", type=Path, required=True)


def _add_run_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clip-id", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=STAGE2_JOB_VERSION)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate intent and report planned writes only"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate Stage I inputs only")
    _add_input_arguments(validate)

    fake_run = subcommands.add_parser(
        "fake-run", help="exercise contracts and resume behavior without GPU models"
    )
    _add_input_arguments(fake_run)
    fake_run.add_argument("--work-dir", type=Path, required=True)
    fake_run.add_argument("--run-id", required=True)

    doctor = subcommands.add_parser("doctor", help="check persistent setup and pinned assets")
    doctor.add_argument("--workspace-root", type=Path, default=Path("/workspace"))
    doctor.add_argument(
        "--load-models",
        action="store_true",
        help="run the offline sequential DINO/SAM CUDA smoke test",
    )

    smoke = subcommands.add_parser(
        "smoke", help="run the deterministic local workflow without GPU models"
    )
    _add_input_arguments(smoke)
    smoke.add_argument("--work-dir", type=Path, required=True)
    smoke.add_argument("--run-id", required=True)

    pilot = subcommands.add_parser("pilot", help="plan the real-GPU pilot boundary")
    pilot.add_argument("clip_id")
    pilot.add_argument("--work-dir", type=Path, default=Path("/workspace/stage2"))
    pilot.add_argument("--manual-seeds", type=Path)
    pilot.add_argument("--labels", type=Path)

    sweep = subcommands.add_parser("sweep", help="plan the fixed DINO threshold sweep")
    sweep.add_argument("clip_id")
    sweep.add_argument("--work-dir", type=Path, default=Path("/workspace/stage2"))
    sweep.add_argument("--thresholds", nargs="+", type=float, default=(0.15, 0.20, 0.25, 0.30))
    sweep.add_argument("--manual-seeds", type=Path)
    sweep.add_argument("--labels", type=Path)

    run = subcommands.add_parser("run", help="run Stage II or its deterministic fake path")
    _add_input_arguments(run)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--fake", action="store_true")
    run.add_argument("--manual-seeds", type=Path)
    run.add_argument("--labels", type=Path)

    status = subcommands.add_parser("status", help="read local processing state")
    _add_run_identity_arguments(status)

    resume = subcommands.add_parser("resume", help="prepare a compatible interrupted run")
    _add_run_identity_arguments(resume)
    resume.add_argument("--recompute-from", choices=("dino", "sam", "render"))

    stop = subcommands.add_parser("stop", help="request a cooperative stop")
    _add_run_identity_arguments(stop)

    review = subcommands.add_parser("review", help="write an immutable human review record")
    _add_run_identity_arguments(review)
    review.add_argument("--decision", choices=("ACCEPTED", "REJECTED"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reviewed-at", required=True)
    review.add_argument("--full-clip-reviewed", action="store_true")
    review.add_argument("--flagged-intervals-reviewed", action="store_true")
    review.add_argument("--notes", default="")

    release = subcommands.add_parser(
        "release-check", help="fail closed unless a release package is safe"
    )
    _add_run_identity_arguments(release)
    release.add_argument("--release-root", type=Path, required=True)
    return parser.parse_args(argv)


def _paths_from_args(args: argparse.Namespace) -> RunPaths:
    return build_run_paths(args.work_dir, args.run_id, args.clip_id)


def _deferred_gpu_error(command: str) -> Stage2Error:
    return Stage2Error(
        "REAL_GPU_EXECUTION_DEFERRED",
        f"{command} is defined but paid clip execution remains locked until the smoke-slice gate.",
        details={
            "completed_boundary": "Milestone 6 offline GPU smoke preparation",
            "reusable_layers": (),
        },
        recovery=(
            "Run 'bash scripts/runpod_setup_stage2.sh' on the CUDA pod, preserve its "
            "gpu-smoke.json evidence, then use '--dry-run' to inspect the 30-60 second pilot."
        ),
    )


def _enrich_operator_error(args: argparse.Namespace, exc: Stage2Error) -> None:
    if not all(hasattr(args, name) for name in ("work_dir", "run_id")):
        return
    if not hasattr(args, "clip_id"):
        try:
            run_component = safe_component(args.run_id, "run_id")
        except Stage2Error:
            return
        run_group = args.work_dir.expanduser().resolve() / "runs" / run_component
        exc.details.setdefault(
            "artifact_locations",
            {
                "run_group": str(run_group),
                "clip_directory": "created after Stage I validation",
                "logs": str(run_group / "<validated-clip-id>" / "logs"),
            },
        )
        exc.details.setdefault("reusable_layers", ())
        if exc.recovery is None and all(
            hasattr(args, name) for name in ("source_video", "stage1_video", "stage1_manifest")
        ):
            exc.recovery = (
                "Retry validation with: bash scripts/runpod_stage2.sh validate "
                f"--source-video {shlex.quote(str(args.source_video))} "
                f"--stage1-video {shlex.quote(str(args.stage1_video))} "
                f"--stage1-manifest {shlex.quote(str(args.stage1_manifest))}"
            )
        return
    try:
        paths = _paths_from_args(args)
        state = load_state(paths.state)
    except Stage2Error:
        return
    exc.details.setdefault(
        "artifact_locations",
        {
            "run_root": str(paths.root),
            "state": str(paths.state),
            "manifest": str(paths.manifest),
            "logs": str(paths.root / "logs"),
        },
    )
    exc.details.setdefault("reusable_layers", state.reusable_layers if state else ())
    if exc.recovery is None:
        exc.recovery = (
            "Inspect the run with: bash scripts/runpod_stage2.sh status "
            f"--work-dir {shlex.quote(str(args.work_dir))} "
            f"--run-id {shlex.quote(str(args.run_id))} "
            f"--clip-id {shlex.quote(str(args.clip_id))}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.dry_run:
            result: Any = operator_dry_run(args)
        elif args.command == "validate":
            result: Any = validate_stage1(
                args.source_video, args.stage1_video, args.stage1_manifest
            )
        elif args.command in {"fake-run", "smoke"}:
            result = run_fake_pipeline(
                source_video=args.source_video,
                stage1_video=args.stage1_video,
                stage1_manifest=args.stage1_manifest,
                work_dir=args.work_dir,
                run_id=args.run_id,
            )
        elif args.command == "doctor":
            result = (
                run_offline_gpu_smoke(args.workspace_root)
                if args.load_models
                else doctor_stage2(args.workspace_root)
            )
        elif args.command == "run":
            if not args.fake:
                raise _deferred_gpu_error("run")
            if args.manual_seeds or args.labels:
                raise Stage2Error(
                    "PRIVATE_INPUT_NOT_REVIEWABLE_FAKE",
                    "Fake runs cannot consume private correction or answer-sheet artifacts.",
                    recovery="Use fake inputs only for plumbing; apply private inputs during the Milestone 6 production pilot.",
                )
            result = run_fake_pipeline(
                source_video=args.source_video,
                stage1_video=args.stage1_video,
                stage1_manifest=args.stage1_manifest,
                work_dir=args.work_dir,
                run_id=args.run_id,
            )
        elif args.command in {"pilot", "sweep"}:
            raise _deferred_gpu_error(args.command)
        elif args.command == "status":
            result = run_status(_paths_from_args(args))
        elif args.command == "stop":
            result = request_stop(_paths_from_args(args))
        elif args.command == "resume":
            paths = _paths_from_args(args)
            resumed = (
                invalidate_from(paths, layer=args.recompute_from)
                if args.recompute_from
                else prepare_resume(paths)
            )
            result = {
                "resume_ready": True,
                "state": resumed,
                "status": run_status(paths),
                "gpu_execution_enabled": False,
                "next_boundary": "Milestone 6 real-GPU smoke test",
            }
        elif args.command == "review":
            reference, record = create_review_record(
                _paths_from_args(args),
                reviewer=args.reviewer,
                reviewed_at=args.reviewed_at,
                review_status=args.decision,
                full_clip_reviewed=args.full_clip_reviewed,
                flagged_intervals_reviewed=args.flagged_intervals_reviewed,
                notes=args.notes,
            )
            result = {"review_record": reference, "review": record}
        elif args.command == "release-check":
            result = release_check(_paths_from_args(args), release_root=args.release_root)
        else:
            raise Stage2Error("UNKNOWN_COMMAND", f"Unsupported operator command: {args.command}")
        payload = _jsonable(result)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Stage2Error as exc:
        _enrich_operator_error(args, exc)
        if args.json:
            print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        else:
            print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
            if exc.details:
                print(json.dumps(exc.details, indent=2, sort_keys=True), file=sys.stderr)
            if exc.recovery:
                print(f"recovery: {exc.recovery}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
