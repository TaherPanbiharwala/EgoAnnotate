# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = []
# ///
"""Stage II face de-identification job.

Milestone 1 intentionally contains no real Grounding DINO or SAM2 imports.
It defines and tests the local contracts, validates EgoBlur artifacts, writes
crash-safe state, and offers deterministic fake adapters for exercising the
pipeline without a GPU. Later milestones extend this same self-contained
PEP-723 job because GPU jobs may not share a Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

STAGE2_SCHEMA_VERSION = 1
STAGE2_CODE_VERSION = "milestone-2"
STAGE2_JOB_VERSION = "0.2.0"

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
DINO_MODEL_REVISION = "e76a695ed7ae1032a61530cce4b4e9b65f4e368b"
DINO_MODEL_WEIGHTS_SHA256 = "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
DINO_PROMPT = "face."
DINO_PROPOSAL_FLOOR = 0.10
DINO_TEXT_THRESHOLD = 0.25
DINO_ANCHOR_SPACING = 20
DINO_TILE_OVERLAP = 0.20
DINO_NMS_IOU = 0.70

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
class SamMaskShardMeta:
    schema_version: int
    artifact_type: str
    fingerprint: str
    dino_artifact_sha256: str
    model: ModelIdentity
    frame_start: int
    frame_end: int
    window: dict[str, Any]
    masks: tuple[dict[str, Any], ...]
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
    review_record: ArtifactRef | None = None


@dataclass(slots=True, frozen=True)
class RunPaths:
    root: Path
    state: Path
    manifest: Path
    dino: Path
    dino_checkpoint: Path
    sam_shard: Path
    render: Path


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
    envelope = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "code_version": STAGE2_CODE_VERSION,
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
    return RunPaths(
        root=root,
        state=root / "state.json",
        manifest=root / "processing.manifest.json",
        dino=root / "dino" / "artifact.json",
        dino_checkpoint=root / "dino" / "checkpoint.jsonl",
        sam_shard=root / "sam" / "window-000000.json",
        render=root / "render" / "artifact.json",
    )


def _parse_rate(value: str | None, *, path: Path, field_name: str) -> float:
    if not value:
        raise Stage2Error(
            "VIDEO_PROBE_INVALID",
            f"ffprobe omitted {field_name} for {path}",
        )
    try:
        numerator, denominator = (value.split("/") + ["1"])[:2]
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
        raise Stage2Error("VIDEO_PROBE_INVALID", f"ffprobe found no usable video in {path}") from exc

    r_fps = _parse_rate(stream.get("r_frame_rate"), path=path, field_name="r_frame_rate")
    avg_fps = _parse_rate(
        stream.get("avg_frame_rate"), path=path, field_name="avg_frame_rate"
    )
    is_cfr = abs(r_fps - avg_fps) / r_fps <= 0.001
    try:
        rotation = int(
            stream.get("rotation", stream.get("tags", {}).get("rotate", 0)) or 0
        )
        coded_width = int(stream["width"])
        coded_height = int(stream["height"])
        frame_value = stream.get("nb_read_packets") or stream.get("nb_frames")
        duration = float(
            stream.get("duration") or data.get("format", {}).get("duration") or 0.0
        )
        n_frames = (
            int(frame_value)
            if frame_value not in {None, "", "N/A"}
            else round(duration * r_fps)
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
            "source.duration_s: "
            f"manifest={manifest_duration!r}, probed={source_facts.duration_s!r}"
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
        problems.append(
            f"Stage I fps: source={source_facts.fps!r}, output={stage1_facts.fps!r}"
        )
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
    if not isinstance(checked, int) or checked <= 0:
        problems.append(f"audit.fill_integrity_checked must be positive, got {checked!r}")
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
        warnings.append(
            "Historical Stage I manifest does not prove the EgoBlur face-weight hash."
        )

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
        origin != "full-frame" and not re.fullmatch(r"tile-r\d+-c\d+", origin)
        for origin in origins
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
        sorted(kept, key=lambda proposal: (proposal.frame_idx, -proposal.score, proposal.proposal_id))
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
            problems.append(
                f"{name}: expected {expected_value!r}, got {getattr(meta, name)!r}"
            )
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
    if not math.isfinite(operating_threshold) or not meta.proposal_floor <= operating_threshold <= 1.0:
        raise Stage2Error(
            "INVALID_DINO_OPERATING_THRESHOLD",
            "DINO operating threshold must be finite and no lower than the stored proposal floor.",
            details={
                "proposal_floor": meta.proposal_floor,
                "operating_threshold": operating_threshold,
            },
            recovery="Recompute DINO with a lower proposal floor before selecting this threshold.",
        )
    accepted = tuple(proposal for proposal in meta.proposals if proposal.score >= operating_threshold)
    rejected = tuple(proposal for proposal in meta.proposals if proposal.score < operating_threshold)
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
        proposal
        for frame_idx in anchors
        for proposal in rows[frame_idx]["proposals"]
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
        "n_raw_proposals": sum(
            int(value.get("n_raw_proposals", 0)) for value in anchor_metrics
        ),
        "n_below_floor": sum(int(value.get("n_below_floor", 0)) for value in anchor_metrics),
        "n_nms_suppressed": sum(
            int(value.get("n_nms_suppressed", 0)) for value in anchor_metrics
        ),
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
    """Lazy official Transformers adapter; model files must already be cached.

    Milestone 2 defines this boundary but does not install/download its GPU
    dependencies. The Stage II setup milestone will pin the runtime and
    verify the safetensors file before this adapter is exposed by a real run.
    """

    identity = ModelIdentity(
        name=DINO_MODEL_ID,
        revision=DINO_MODEL_REVISION,
        sha256=DINO_MODEL_WEIGHTS_SHA256,
    )

    def __init__(self, *, device: str = "cuda", local_files_only: bool = True) -> None:
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
        try:
            self.processor = AutoProcessor.from_pretrained(
                DINO_MODEL_ID,
                revision=DINO_MODEL_REVISION,
                local_files_only=local_files_only,
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_MODEL_ID,
                revision=DINO_MODEL_REVISION,
                local_files_only=local_files_only,
                use_safetensors=True,
            ).to(device)
            self.model.eval()
        except Exception as exc:
            raise Stage2Error(
                "DINO_MODEL_LOAD_FAILED",
                "Pinned Grounding DINO Base model could not be loaded from the local cache.",
                details={"cause": str(exc), "revision": DINO_MODEL_REVISION},
                recovery="Run the Stage II setup command to verify and prime model assets.",
            ) from exc

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
    identity = ModelIdentity(
        name="fake-grounding-dino", revision="milestone-2", sha256="f" * 64
    )

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

    def detect(self, frame_idx: int, width: int, height: int) -> tuple[Proposal, ...]:
        size = max(8.0, min(width, height) * 0.1)
        x1 = float((frame_idx * 7) % max(1, int(width - size)))
        y1 = float((frame_idx * 5) % max(1, int(height - size)))
        return (
            Proposal(
                frame_idx=frame_idx,
                box=(x1, y1, x1 + size, y1 + size),
                score=0.42,
                source="fake-full-frame",
            ),
        )


class FakeSamAdapter:
    identity = ModelIdentity(name="fake-sam2", revision="milestone-1", sha256="a" * 64)

    def propagate(
        self, proposals: tuple[Proposal, ...], frame_start: int, frame_end: int
    ) -> tuple[dict[str, Any], ...]:
        masks = []
        for proposal in proposals:
            for frame_idx in range(frame_start, frame_end + 1):
                masks.append(
                    {
                        "frame_idx": frame_idx,
                        "box": list(proposal.box),
                        "source": "fake-sam2",
                        "anchor_frame": proposal.frame_idx,
                    }
                )
        return tuple(masks)


def run_fake_pipeline(
    *,
    source_video: Path,
    stage1_video: Path,
    stage1_manifest: Path,
    work_dir: Path,
    run_id: str,
    probe_fn: Callable[[Path], VideoFacts] = probe_video,
) -> ProcessingManifest:
    validated = validate_stage1(
        source_video, stage1_video, stage1_manifest, probe_fn=probe_fn
    )
    paths = build_run_paths(work_dir, run_id, validated.clip_id)
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
    proposals = dino_result.meta.proposals
    fake_frame_end = min(validated.source_video.n_frames - 1, 9)
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
    sam_payload = {
        "dino_artifact_sha256": dino_ref.sha256,
        "model": _jsonable(sam.identity),
        "accepted_proposal_threshold": 0.20,
        "window": {
            "start": 0,
            "end": min(validated.source_video.n_frames - 1, 9),
            "note": "bounded fake smoke window",
        },
        "manual_seeds_sha256": None,
    }
    sam_meta = SamMaskShardMeta(
        schema_version=STAGE2_SCHEMA_VERSION,
        artifact_type="sam_mask_shard",
        fingerprint=sam_fingerprint(sam_payload),
        dino_artifact_sha256=dino_ref.sha256,
        model=sam.identity,
        frame_start=0,
        frame_end=fake_frame_end,
        window={"mode": "fake", "overlap": 0},
        masks=sam.propagate(proposals, 0, fake_frame_end),
        review_flags=(),
    )
    sam_ref = write_immutable_json(paths.sam_shard, sam_meta)
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
        "mask_set_sha256": sam_ref.sha256,
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
            "mask_set_sha256": sam_ref.sha256,
        },
    )
    render_meta = RenderArtifactMeta(
        schema_version=STAGE2_SCHEMA_VERSION,
        artifact_type="render",
        fingerprint=render_fingerprint(render_payload),
        stage1_video_sha256=validated.stage1_video.sha256,
        mask_set_sha256=sam_ref.sha256,
        output=fake_output,
        encoder={"mode": "fake-no-video"},
        verification={"mode": "fake", "passed": True, "publishable": False},
    )
    render_ref = write_immutable_json(paths.render, render_meta)
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
        sam_mask_shards=(sam_ref,),
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


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--stage1-video", type=Path, required=True)
    parser.add_argument("--stage1-manifest", type=Path, required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=STAGE2_JOB_VERSION)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate Stage I inputs only")
    _add_input_arguments(validate)

    fake_run = subcommands.add_parser(
        "fake-run", help="exercise contracts and resume behavior without GPU models"
    )
    _add_input_arguments(fake_run)
    fake_run.add_argument("--work-dir", type=Path, required=True)
    fake_run.add_argument("--run-id", required=True)

    status = subcommands.add_parser("status", help="read local processing state")
    status.add_argument("--work-dir", type=Path, required=True)
    status.add_argument("--run-id", required=True)
    status.add_argument("--clip-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate":
            result: Any = validate_stage1(
                args.source_video, args.stage1_video, args.stage1_manifest
            )
        elif args.command == "fake-run":
            result = run_fake_pipeline(
                source_video=args.source_video,
                stage1_video=args.stage1_video,
                stage1_manifest=args.stage1_manifest,
                work_dir=args.work_dir,
                run_id=args.run_id,
            )
        else:
            paths = build_run_paths(args.work_dir, args.run_id, args.clip_id)
            result = load_state(paths.state)
            if result is None:
                raise Stage2Error(
                    "RUN_NOT_FOUND",
                    f"No Stage II run state exists at {paths.state}",
                )
        payload = _jsonable(result)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Stage2Error as exc:
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
