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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STAGE2_SCHEMA_VERSION = 1
STAGE2_CODE_VERSION = "milestone-1"
STAGE2_JOB_VERSION = "0.1.0"

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


@dataclass(slots=True, frozen=True)
class DinoArtifactMeta:
    schema_version: int
    artifact_type: str
    fingerprint: str
    source_sha256: str
    model: ModelIdentity
    prompt: str
    proposal_floor: float
    preprocessing: dict[str, Any]
    proposals: tuple[Proposal, ...]


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
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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


class FakeDinoAdapter:
    identity = ModelIdentity(
        name="fake-grounding-dino", revision="milestone-1", sha256="f" * 64
    )

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
    fake_frame_end = min(validated.source_video.n_frames - 1, 9)
    anchor_frames = tuple(sorted({0, fake_frame_end // 2, fake_frame_end}))
    proposals = tuple(
        proposal
        for frame_idx in anchor_frames
        for proposal in dino.detect(
            frame_idx,
            validated.source_video.display_width,
            validated.source_video.display_height,
        )
    )
    dino_payload = {
        "source_sha256": validated.source.sha256,
        "model": _jsonable(dino.identity),
        "prompt": "face.",
        "proposal_floor": 0.10,
        "preprocessing": {"mode": "fake", "tiling": "full-frame-only"},
        "anchor_frames": anchor_frames,
    }
    dino_meta = DinoArtifactMeta(
        schema_version=STAGE2_SCHEMA_VERSION,
        artifact_type="dino_proposals",
        fingerprint=dino_fingerprint(dino_payload),
        source_sha256=validated.source.sha256,
        model=dino.identity,
        prompt="face.",
        proposal_floor=0.10,
        preprocessing={"mode": "fake", "tiling": "full-frame-only"},
        proposals=proposals,
    )
    dino_ref = write_immutable_json(paths.dino, dino_meta)
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
