# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "av>=14,<17",
#   "huggingface-hub>=0.34,<1",
#   "numpy>=1.26,<3",
#   "opencv-python-headless>=4.10,<5",
#   "pillow>=11,<12",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# ///
"""Private one-clip GPU comparison: Depth Anything 3 Metric Large vs MoGe-3.

This is an *experiment*, not an EgoAnnotate annotation layer.  It accepts one
already-redacted, continuous input child and writes only to a directory whose
path contains ``private``.  An explicit, user-authorized
``--allow-private-unredacted-input`` exception exists for a one-off private
source-footage experiment; it is non-publishable and is recorded distinctly in
every manifest and report.  There is intentionally no upload, Drive write,
Hugging Face write, reconstruction, SLAM, segmentation, or VLM code here.

Why the workers are separate environments
------------------------------------------
DA3's pinned upstream package declares ``numpy<2`` while MoGe-3's declares
``numpy>=2``.  A single environment would either fail resolution or silently
override one model's declared dependency constraint.  The PEP 723 script is
therefore a small private orchestrator.  It invokes this exact file once in a
DA3 environment and once in a MoGe environment, each via ``uv run --script
--with`` and an immutable Git revision.  The two workers exchange only
lossless private ``.npz`` arrays and JSON state under the run directory.

Completion semantics are deliberately strict.  A run is ``complete`` only
after every selected source frame has a readable array for every requested
model, all preview videos fully decode to the expected frame count, and the
Markdown report has been written.  Per-frame arrays are atomically renamed and
both worker and run manifests are atomically replaced and fsynced.  Re-running
the same fingerprint resumes verified per-frame results; a changed input or
inference setting refuses to mix outputs under the same run ID.

Depth semantics
---------------
DA3 Metric Large exposes canonical depth.  Its upstream FAQ specifies
``metres = focal_px * net_output / 300``.  Supplied calibrated focal length is
used for that conversion.  Without it the script uses DA3's inferred focal
when available (otherwise the resized frame width) only to make an explicitly
labelled *exploratory* metre estimate; it is never claimed as a measurement.
MoGe produces metric estimates directly, but these remain estimates without
ground-truth validation.  If supplied GoPro FOV/calibration exists, MoGe is
conditioned on it and additionally run once without it to retain its
independently predicted intrinsics for comparison.

Run from a Linux CUDA RunPod volume, for example:

    /workspace/bin/uv run jobs/30_depth_compare_da3_moge3.py --help

The no-GPU ``--dry-run`` performs the privacy attestation, hash, ffprobe and
full-decode preflight, creates a private planned manifest, and exits before
model downloads or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
JOB_NAME = "experiment-1-da3-moge3"

# Immutable upstream sources resolved on 2026-09-18.  Do not replace these
# with branch names: a rerun needs the same code even after upstream moves.
DA3_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
DA3_MODEL_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
DA3_CODE_REPOSITORY = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
MOGE_MODEL_ID = "Ruicheng/moge-3-vitl"
MOGE_MODEL_REVISION = "184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be"
MOGE_CODE_REPOSITORY = "https://github.com/microsoft/MoGe.git"
MOGE_CODE_REVISION = "74fbce054ebed49800de42d0ad0e83495065719a"

DEFAULT_METRIC_MIN_M = 0.20
DEFAULT_METRIC_MAX_M = 8.00
DEFAULT_KEYFRAME_COUNT = 6
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ExperimentError(RuntimeError):
    """A fail-closed condition that leaves a private, inspectable manifest."""


@dataclass(frozen=True)
class Calibration:
    """Only camera facts supplied by the operator; no lens facts are invented."""

    source_path: str
    source_sha256: str
    projection: str
    fx: float | None
    fy: float | None
    cx: float | None
    cy: float | None
    fov_x_deg: float | None
    image_width: int | None
    image_height: int | None

    @property
    def has_full_intrinsics(self) -> bool:
        return None not in (self.fx, self.fy, self.cx, self.cy)

    @property
    def focal_px(self) -> float | None:
        values = [value for value in (self.fx, self.fy) if value is not None]
        return sum(values) / len(values) if values else None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "projection": self.projection,
            "fx_px": self.fx,
            "fy_px": self.fy,
            "cx_px": self.cx,
            "cy_px": self.cy,
            "fov_x_deg": self.fov_x_deg,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "has_full_intrinsics": self.has_full_intrinsics,
        }


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fsync_directory(path: Path) -> None:
    """Persist a rename on filesystems where directory fsync is supported."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_save_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Write a losslessly-compressed NumPy archive without a visible partial."""

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.partial.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def ensure_private_path(path: Path, label: str) -> None:
    if not any("private" in component.lower() for component in path.resolve().parts):
        raise ExperimentError(
            f"{label} must be on a private path (one path component must contain 'private'): {path}"
        )


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ExperimentError(
            "--run-id must use only letters, digits, '.', '_' and '-', start with an alphanumeric, "
            "and be at most 128 characters"
        )
    return run_id


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"{label} must be a JSON object: {path}")
    return value


def require_input_attestation(attestation_path: Path, input_sha256: str,
                              allow_private_unredacted_input: bool) -> dict[str, Any]:
    """Bind the input to a truthful privacy status; never infer redaction from pixels."""

    data = load_json_object(attestation_path, "input attestation")
    acceptable_statuses = {"redacted_public_safe", "redacted", "face_free_public_safe"}
    private_exception_status = "private_unredacted_user_authorized"
    privacy_status = data.get("privacy_status")
    if privacy_status == private_exception_status:
        if not allow_private_unredacted_input:
            raise ExperimentError(
                "input is explicitly marked private unredacted source footage. Refusing to process it "
                "without --allow-private-unredacted-input."
            )
        if data.get("publication_permitted") is not False:
            raise ExperimentError(
                "private-unredacted attestation must set publication_permitted: false; this exception "
                "can never be represented as public-safe input."
            )
        reason = data.get("private_exception_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ExperimentError(
                "private-unredacted attestation requires a non-empty private_exception_reason."
            )
    elif privacy_status in acceptable_statuses:
        if allow_private_unredacted_input:
            raise ExperimentError(
                "--allow-private-unredacted-input is only valid when privacy_status is "
                "private_unredacted_user_authorized. Remove the flag for a redacted input."
            )
    else:
        raise ExperimentError(
            "input is not attested as already-redacted public-safe footage. Refusing to process "
            "possible source footage; request a redacted continuous child, or use the explicit "
            "private-unredacted exception with a truthful attestation."
        )
    if data.get("continuous_child") is not True:
        raise ExperimentError(
            "input attestation must set continuous_child: true. A privacy cut is a hard sequence "
            "boundary and cannot be treated as one depth sequence."
        )
    if data.get("contains_privacy_cuts") is not False:
        raise ExperimentError(
            "input attestation must set contains_privacy_cuts: false. Split at every privacy cut first."
        )
    declared_hash = data.get("input_sha256")
    if declared_hash is not None and declared_hash != input_sha256:
        raise ExperimentError(
            "input SHA-256 does not match the redacted-input attestation; refusing to bind results "
            "to an unverifiable clip."
        )
    projection = data.get("projection")
    if projection == "fisheye":
        raise ExperimentError(
            "attestation identifies a fisheye input. Supply an approved redacted dewarped/rectilinear "
            "child before metric comparison; do not invent a lens correction."
        )
    return data


def load_calibration(path: Path | None, width: int, height: int) -> Calibration | None:
    if path is None:
        return None
    data = load_json_object(path, "camera calibration")
    projection = data.get("projection")
    if projection not in {"rectilinear", "dewarped_rectilinear"}:
        raise ExperimentError(
            "camera calibration must state projection as 'rectilinear' or 'dewarped_rectilinear'; "
            "do not pass unapproved fisheye calibration to a pinhole depth comparison."
        )

    def optional_float(name: str) -> float | None:
        value = data.get(name)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ExperimentError(f"camera calibration {name!r} must be numeric") from exc
        if not math.isfinite(number) or (number <= 0 and name not in {"cx", "cy"}):
            raise ExperimentError(f"camera calibration {name!r} is invalid: {number!r}")
        return number

    fx, fy, cx, cy = (optional_float(name) for name in ("fx", "fy", "cx", "cy"))
    fov_x_deg = optional_float("fov_x_deg")
    if fov_x_deg is not None and not 1.0 < fov_x_deg < 179.0:
        raise ExperimentError("camera calibration fov_x_deg must be between 1 and 179 degrees")
    if fx is None and fy is None and fov_x_deg is None:
        raise ExperimentError("camera calibration needs fx/fy and/or fov_x_deg; no focal fact was supplied")

    calibration_width = data.get("image_width")
    calibration_height = data.get("image_height")
    if calibration_width is not None and int(calibration_width) != width:
        raise ExperimentError(
            f"calibration image_width={calibration_width} does not match input width={width}; "
            "rescale/approve calibration explicitly rather than guessing."
        )
    if calibration_height is not None and int(calibration_height) != height:
        raise ExperimentError(
            f"calibration image_height={calibration_height} does not match input height={height}; "
            "rescale/approve calibration explicitly rather than guessing."
        )
    if fov_x_deg is None and fx is not None:
        fov_x_deg = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    return Calibration(
        source_path=str(path.resolve()),
        source_sha256=sha256_file(path),
        projection=projection,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fov_x_deg=fov_x_deg,
        image_width=int(calibration_width) if calibration_width is not None else None,
        image_height=int(calibration_height) if calibration_height is not None else None,
    )


def fraction_from_text(value: str | None, label: str) -> Fraction:
    if not value or value == "0/0":
        raise ExperimentError(f"ffprobe did not provide a valid {label}")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ExperimentError(f"invalid {label} from ffprobe: {value!r}") from exc
    if result <= 0:
        raise ExperimentError(f"invalid non-positive {label} from ffprobe: {value!r}")
    return result


def run_ffprobe(input_path: Path, ffprobe_bin: str) -> dict[str, Any]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=index,codec_name,codec_type,width,height,avg_frame_rate,r_frame_rate,time_base,"
        "duration,nb_frames,nb_read_frames,pix_fmt:format=duration,size,format_name",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(output.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"ffprobe preflight failed for {input_path}: {exc}") from exc
    video_streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise ExperimentError(f"expected exactly one video stream, found {len(video_streams)}")
    stream = video_streams[0]
    count_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        frame_count = int(count_raw)
    except (TypeError, ValueError) as exc:
        raise ExperimentError("ffprobe did not return a usable frame count; refusing an unverifiable input") from exc
    if frame_count <= 0:
        raise ExperimentError(f"ffprobe returned non-positive frame count: {frame_count}")
    if not stream.get("width") or not stream.get("height"):
        raise ExperimentError("ffprobe did not return video resolution")
    return {
        "command": command,
        "raw": payload,
        "video_stream": stream,
        "frame_count": frame_count,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "time_base": stream.get("time_base"),
    }


def decode_source_clock(input_path: Path, expected_frame_count: int) -> list[dict[str, Any]]:
    """Fully decode once; a partial decode is a hard failure, never an empty pass."""

    try:
        import av
    except ImportError as exc:  # pragma: no cover - PEP environment guarantee
        raise ExperimentError("PyAV import failed; PEP 723 environment is incomplete") from exc
    try:
        container = av.open(str(input_path))
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None or stream.time_base is None:
            raise ExperimentError("input has no decodable video stream/time base")
        time_base = Fraction(stream.time_base)
        records: list[dict[str, Any]] = []
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ExperimentError(f"decoded frame {index} has no PTS; cannot preserve source clock")
            records.append(
                {
                    "frame_index": index,
                    "pts": int(frame.pts),
                    "time_base": f"{time_base.numerator}/{time_base.denominator}",
                    "timestamp_seconds": float(Fraction(int(frame.pts)) * time_base),
                }
            )
    except ExperimentError:
        raise
    except Exception as exc:
        raise ExperimentError(f"full input decode failed: {exc}") from exc
    finally:
        try:
            container.close()
        except UnboundLocalError:
            pass
    if len(records) != expected_frame_count:
        raise ExperimentError(
            f"full input decode yielded {len(records)} frames but ffprobe reported {expected_frame_count}; "
            "refusing partial input."
        )
    return records


def select_source_frames(clock: list[dict[str, Any]], requested_fps: float | None) -> list[dict[str, Any]]:
    if not clock:
        raise ExperimentError("cannot select from an empty source clock")
    if requested_fps is None:
        return list(clock)
    if not math.isfinite(requested_fps) or requested_fps <= 0:
        raise ExperimentError("--fps must be a positive finite number")
    interval = Fraction(str(requested_fps)).limit_denominator(1_000_000)
    interval = 1 / interval
    selected: list[dict[str, Any]] = []
    due = Fraction(clock[0]["pts"]) * Fraction(clock[0]["time_base"])
    for record in clock:
        timestamp = Fraction(record["pts"]) * Fraction(record["time_base"])
        if timestamp >= due:
            selected.append(record)
            due += interval
    if not selected:
        raise ExperimentError("source FPS selection unexpectedly produced zero frames")
    return selected


def parse_resize(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)[xX](\d+)", value)
    if not match:
        raise ExperimentError("--resize must have the form WIDTHxHEIGHT, for example 1280x720")
    width, height = (int(item) for item in match.groups())
    if width < 32 or height < 32:
        raise ExperimentError("--resize dimensions must each be at least 32")
    return width, height


def model_input_size(source_width: int, source_height: int, resize: tuple[int, int] | None,
                     max_resolution: int | None) -> tuple[int, int, str]:
    if resize is not None and max_resolution is not None:
        raise ExperimentError("--resize and --max-resolution are mutually exclusive")
    if resize is not None:
        return resize[0], resize[1], "explicit_resize"
    if max_resolution is None:
        return source_width, source_height, "source_resolution"
    if max_resolution < 32:
        raise ExperimentError("--max-resolution must be at least 32")
    longest = max(source_width, source_height)
    if longest <= max_resolution:
        return source_width, source_height, "source_resolution"
    scale = max_resolution / longest
    width = max(32, round(source_width * scale))
    height = max(32, round(source_height * scale))
    return width, height, "aspect_preserving_max_resolution"


def choose_keyframes(records: list[dict[str, Any]], count: int) -> set[int]:
    if count < 0:
        raise ExperimentError("--keyframe-count cannot be negative")
    if count == 0 or not records:
        return set()
    count = min(count, len(records))
    if count == 1:
        return {records[len(records) // 2]["frame_index"]}
    return {
        records[round(position * (len(records) - 1) / (count - 1))]["frame_index"]
        for position in range(count)
    }


def model_specs(models: str) -> list[str]:
    if models == "both":
        return ["da3", "moge3"]
    if models in {"da3", "moge3"}:
        return [models]
    raise ExperimentError(f"unsupported --models value: {models}")


def dependencies_for_worker(model_name: str) -> str:
    if model_name == "da3":
        return f"depth-anything-3 @ git+{DA3_CODE_REPOSITORY}@{DA3_CODE_REVISION}"
    if model_name == "moge3":
        return f"moge @ git+{MOGE_CODE_REPOSITORY}@{MOGE_CODE_REVISION}"
    raise ExperimentError(f"unknown worker model {model_name!r}")


def source_model_metadata(model_name: str) -> dict[str, Any]:
    if model_name == "da3":
        return {
            "name": "Depth Anything 3 Metric Large",
            "model_id": DA3_MODEL_ID,
            "model_revision": DA3_MODEL_REVISION,
            "license": "Apache-2.0",
            "code_repository": DA3_CODE_REPOSITORY,
            "code_revision": DA3_CODE_REVISION,
            "worker_dependency": dependencies_for_worker("da3"),
        }
    if model_name == "moge3":
        return {
            "name": "MoGe-3 ViT-L",
            "model_id": MOGE_MODEL_ID,
            "model_revision": MOGE_MODEL_REVISION,
            "license": "MIT",
            "code_repository": MOGE_CODE_REPOSITORY,
            "code_revision": MOGE_CODE_REVISION,
            "worker_dependency": dependencies_for_worker("moge3"),
        }
    raise ExperimentError(f"unknown model {model_name!r}")


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def compact_input_metadata(probe: dict[str, Any], input_path: Path, input_sha256: str,
                           attestation_path: Path, attestation: dict[str, Any],
                           clock: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "local_path": str(input_path.resolve()),
        "sha256": input_sha256,
        "size_bytes": input_path.stat().st_size,
        "ffprobe": probe,
        "frame_count": probe["frame_count"],
        "decoded_frame_count": len(clock),
        "frame_rate": {
            "avg_frame_rate": probe["avg_frame_rate"],
            "r_frame_rate": probe["r_frame_rate"],
            "time_base": probe["time_base"],
        },
        "resolution": {"width": probe["width"], "height": probe["height"]},
        "input_attestation": {
            "path": str(attestation_path.resolve()),
            "sha256": sha256_file(attestation_path),
            "privacy_status": attestation.get("privacy_status"),
            "continuous_child": attestation.get("continuous_child"),
            "contains_privacy_cuts": attestation.get("contains_privacy_cuts"),
            "projection": attestation.get("projection"),
            "publication_permitted": attestation.get("publication_permitted"),
            "private_unredacted_exception": (
                attestation.get("privacy_status") == "private_unredacted_user_authorized"
            ),
        },
    }


def build_run_fingerprint(args: argparse.Namespace, input_sha256: str, attestation_path: Path,
                          attestation: dict[str, Any], clock: list[dict[str, Any]],
                          selected: list[dict[str, Any]], calibration: Calibration | None,
                          input_size: tuple[int, int, str]) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "job": JOB_NAME,
        "job_script_sha256": sha256_file(script_path),
        "input_sha256": input_sha256,
        "input_attestation_sha256": sha256_file(attestation_path),
        "input_privacy_status": attestation.get("privacy_status"),
        "allow_private_unredacted_input": args.allow_private_unredacted_input,
        "source_clock_sha256": fingerprint(clock),
        "selected_clock_sha256": fingerprint(selected),
        "requested_models": model_specs(args.models),
        "requested_fps": args.fps,
        "resize": args.resize,
        "max_resolution": args.max_resolution,
        "model_input_size": {"width": input_size[0], "height": input_size[1], "mode": input_size[2]},
        "requested_moge_precision": args.precision,
        "da3_precision_policy": "upstream_automatic_bf16_if_supported_else_fp16",
        "moge_num_tokens": args.moge_num_tokens,
        "moge_resolution_level": args.moge_resolution_level,
        "moge_refine_steps": args.moge_refine_steps,
        "metric_preview_range_m": [args.metric_min_m, args.metric_max_m],
        "keyframe_count": args.keyframe_count,
        "export_point_clouds": args.export_point_clouds,
        "calibration": calibration.to_manifest() if calibration else None,
        "models": {model: source_model_metadata(model) for model in model_specs(args.models)},
    }


def initial_run_manifest(run_id: str, run_dir: Path, input_metadata: dict[str, Any],
                         calibration: Calibration | None, clock: list[dict[str, Any]],
                         selected: list[dict[str, Any]], run_fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "private_depth_model_comparison",
        "status": "preflight_complete",
        "run_id": run_id,
        "run_directory": str(run_dir.resolve()),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "privacy": {
            "input_required": (
                "explicit user-authorized private unredacted source-footage exception"
                if input_metadata["input_attestation"]["private_unredacted_exception"]
                else "already-redacted public-safe continuous child"
            ),
            "input_privacy_status": input_metadata["input_attestation"]["privacy_status"],
            "private_unredacted_exception": input_metadata["input_attestation"]["private_unredacted_exception"],
            "publication_permitted": input_metadata["input_attestation"]["publication_permitted"],
            "privacy_cuts_are_hard_boundaries": True,
            "uploads_implemented": False,
            "scope": "private one-clip depth-only experiment",
        },
        "input": input_metadata,
        "camera_calibration": calibration.to_manifest() if calibration else None,
        "source_clock": clock,
        "selected_source_clock": selected,
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_sha256": fingerprint(run_fingerprint),
        "models": {name: {"status": "pending", **source_model_metadata(name)} for name in run_fingerprint["requested_models"]},
        "previews": {"status": "pending"},
        "report": {"status": "pending"},
        "failures": [],
    }


def load_or_create_run(manifest_path: Path, initial: dict[str, Any]) -> dict[str, Any]:
    if not manifest_path.exists():
        atomic_write_json(manifest_path, initial)
        return initial
    existing = load_json_object(manifest_path, "existing run manifest")
    existing_fingerprint = existing.get("run_fingerprint_sha256")
    expected_fingerprint = initial["run_fingerprint_sha256"]
    if existing_fingerprint != expected_fingerprint:
        raise ExperimentError(
            "existing run ID has a different input/configuration fingerprint. Refusing to mix "
            "results; choose a new --run-id."
        )
    return existing


def update_run_manifest(manifest_path: Path, manifest: dict[str, Any], **changes: Any) -> None:
    manifest.update(changes)
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def build_worker_request(run_dir: Path, model_name: str, input_path: Path,
                         selected_clock: list[dict[str, Any]], run_fingerprint_sha256: str,
                         calibration: Calibration | None, input_size: tuple[int, int, str],
                         args: argparse.Namespace, keyframes: set[int]) -> Path:
    inference_settings: dict[str, Any]
    if model_name == "da3":
        inference_settings = {
            "api": "DepthAnything3.inference",
            "process_res": max(input_size[0], input_size[1]),
            "process_res_method": "upper_bound_resize",
            "export_format": "mini_npz (no upstream export written)",
            "intrinsics_policy": "pass K only when fx, fy, cx, cy are all supplied",
            "metric_conversion": "depth_m_estimate = canonical_depth * focal_px / 300",
            "confidence": "upstream prediction.conf, stored float32 without probability calibration claim",
            "precision_policy": "Depth Anything 3 upstream automatically selects bf16 when supported, else fp16; --precision controls MoGe only",
        }
    else:
        inference_settings = {
            "api": "MoGeModel.infer",
            "force_projection": True,
            "apply_mask": True,
            "num_tokens": args.moge_num_tokens,
            "resolution_level": args.moge_resolution_level,
            "refine_steps": args.moge_refine_steps,
            "use_fp16": args.precision == "fp16",
            "intrinsics_policy": "always save unconditioned predicted intrinsics; use supplied fov_x_deg when available for output geometry",
        }
    request = {
        "schema_version": SCHEMA_VERSION,
        "worker_model": model_name,
        "run_dir": str(run_dir.resolve()),
        "input_path": str(input_path.resolve()),
        "selected_source_clock": selected_clock,
        "run_fingerprint_sha256": run_fingerprint_sha256,
        "model": source_model_metadata(model_name),
        "calibration": calibration.to_manifest() if calibration else None,
        "model_input_size": {"width": input_size[0], "height": input_size[1], "mode": input_size[2]},
        "moge_precision": args.precision,
        "moge_num_tokens": args.moge_num_tokens,
        "moge_resolution_level": args.moge_resolution_level,
        "moge_refine_steps": args.moge_refine_steps,
        "keyframe_indices": sorted(keyframes),
        "export_point_clouds": args.export_point_clouds,
        "inference_settings": inference_settings,
    }
    path = run_dir / "worker-requests" / f"{model_name}.json"
    atomic_write_json(path, request)
    return path


def run_worker_subprocess(worker_request: Path, model_name: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise ExperimentError("uv is not on PATH; run /workspace/egoblur/runpod_setup.sh first")
    command = [
        uv,
        "run",
        "--isolated",
        "--script",
        str(Path(__file__).resolve()),
        "--with",
        dependencies_for_worker(model_name),
        "--",
        "--worker",
        model_name,
        "--worker-request",
        str(worker_request),
    ]
    completed = subprocess.run(command, text=True)
    if completed.returncode != 0:
        raise ExperimentError(f"{model_name} worker exited {completed.returncode}; inspect its private worker manifest")


def worker_state_path(run_dir: Path, model_name: str) -> Path:
    return run_dir / "workers" / model_name / "worker_manifest.json"


def validate_worker_result(path: Path, model_name: str) -> dict[str, Any]:
    """Verify an existing result is nonempty and has the required lossless fields."""

    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            required = {"frame_index", "pts", "depth_m_estimate"}
            if model_name == "da3":
                required |= {"confidence"}
            elif model_name == "moge3":
                required |= {"valid_mask", "intrinsics_used_normalized", "intrinsics_predicted_normalized"}
            missing = required - keys
            if missing:
                raise ExperimentError(f"{path} is missing required arrays: {sorted(missing)}")
            depth = archive["depth_m_estimate"]
            if depth.dtype != np.float32 or depth.ndim != 2 or depth.size == 0:
                raise ExperimentError(f"{path} has invalid depth_m_estimate dtype/shape")
            if not np.isfinite(depth).any():
                raise ExperimentError(f"{path} has no finite depth values")
            return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    except ExperimentError:
        raise
    except Exception as exc:
        raise ExperimentError(f"cannot validate existing result {path}: {exc}") from exc


def resize_bgr_for_model(frame_bgr: Any, width: int, height: int) -> Any:
    import cv2

    if frame_bgr.shape[1] == width and frame_bgr.shape[0] == height:
        return frame_bgr
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


def upsample_to_source(array: Any, source_width: int, source_height: int, interpolation: int) -> Any:
    import cv2

    if array.shape[:2] == (source_height, source_width):
        return array
    return cv2.resize(array, (source_width, source_height), interpolation=interpolation)


def calibration_from_manifest(value: dict[str, Any] | None) -> Calibration | None:
    if value is None:
        return None
    return Calibration(
        source_path=str(value["source_path"]),
        source_sha256=str(value["source_sha256"]),
        projection=str(value["projection"]),
        fx=value.get("fx_px"),
        fy=value.get("fy_px"),
        cx=value.get("cx_px"),
        cy=value.get("cy_px"),
        fov_x_deg=value.get("fov_x_deg"),
        image_width=value.get("image_width"),
        image_height=value.get("image_height"),
    )


def intrinsics_for_resized_input(calibration: Calibration | None, source_width: int, source_height: int,
                                 model_width: int, model_height: int) -> Any | None:
    """Return K only when all four entries were supplied, never assumed."""

    if calibration is None or not calibration.has_full_intrinsics:
        return None
    import numpy as np

    sx = model_width / source_width
    sy = model_height / source_height
    return np.array(
        [[calibration.fx * sx, 0.0, calibration.cx * sx], [0.0, calibration.fy * sy, calibration.cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def focal_for_depth_map(calibration: Calibration | None, source_width: int, source_height: int,
                        depth_width: int, depth_height: int) -> float | None:
    """Scale supplied focal facts to DA3's actual canonical-depth grid.

    DA3's own input processor can make a final patch-size resize after this job
    passes it an image.  The documented focal conversion must use that final
    grid, not the nominal preprocessor input dimensions.
    """

    if calibration is None:
        return None
    values: list[float] = []
    if calibration.fx is not None:
        values.append(calibration.fx * depth_width / source_width)
    if calibration.fy is not None:
        values.append(calibration.fy * depth_height / source_height)
    return sum(values) / len(values) if values else None


def resolve_da3_model() -> tuple[Any, dict[str, Any]]:
    import torch
    from depth_anything_3.api import DepthAnything3
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(DA3_MODEL_ID, revision=DA3_MODEL_REVISION)
    if info.sha != DA3_MODEL_REVISION:
        raise ExperimentError(
            f"DA3 resolved {info.sha}, not pinned revision {DA3_MODEL_REVISION}; refusing a moving checkpoint"
        )
    snapshot = snapshot_download(repo_id=DA3_MODEL_ID, revision=DA3_MODEL_REVISION)
    model = DepthAnything3.from_pretrained(snapshot).to(torch.device("cuda")).eval()
    return model, {
        "huggingface_model_id": DA3_MODEL_ID,
        "requested_revision": DA3_MODEL_REVISION,
        "resolved_revision": info.sha,
        "local_snapshot": snapshot,
        "model_file_sha256": sha256_file(Path(snapshot) / "model.safetensors"),
    }


def resolve_moge_model() -> tuple[Any, dict[str, Any]]:
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from moge.model.v3 import MoGeModel

    info = HfApi().model_info(MOGE_MODEL_ID, revision=MOGE_MODEL_REVISION)
    if info.sha != MOGE_MODEL_REVISION:
        raise ExperimentError(
            f"MoGe resolved {info.sha}, not pinned revision {MOGE_MODEL_REVISION}; refusing a moving checkpoint"
        )
    checkpoint = hf_hub_download(repo_id=MOGE_MODEL_ID, filename="model.pt", revision=MOGE_MODEL_REVISION)
    model = MoGeModel.from_pretrained(checkpoint).to(torch.device("cuda")).eval()
    return model, {
        "huggingface_model_id": MOGE_MODEL_ID,
        "requested_revision": MOGE_MODEL_REVISION,
        "resolved_revision": info.sha,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": sha256_file(Path(checkpoint)),
    }


def tensor_to_numpy(value: Any) -> Any:
    return value.detach().float().cpu().numpy()


def infer_da3(model: Any, frame_bgr: Any, source_width: int, source_height: int,
              calibration: Calibration | None) -> tuple[dict[str, Any], dict[str, Any]]:
    import cv2
    import numpy as np
    import torch

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    camera_k = intrinsics_for_resized_input(calibration, source_width, source_height, width, height)
    kwargs: dict[str, Any] = {
        "process_res": max(width, height),
        "process_res_method": "upper_bound_resize",
        "export_format": "mini_npz",
    }
    if camera_k is not None:
        kwargs["intrinsics"] = camera_k[None, ...]
    torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = model.inference([rgb], **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    canonical_depth = np.asarray(prediction.depth[0], dtype=np.float32)
    depth_height, depth_width = canonical_depth.shape
    confidence = np.asarray(prediction.conf[0], dtype=np.float32)
    raw_depth = upsample_to_source(canonical_depth, source_width, source_height, cv2.INTER_LINEAR).astype(np.float32)
    confidence = upsample_to_source(confidence, source_width, source_height, cv2.INTER_LINEAR).astype(np.float32)
    predicted_intrinsics = getattr(prediction, "intrinsics", None)
    predicted_intrinsics_array = (
        np.asarray(predicted_intrinsics[0], dtype=np.float32)
        if predicted_intrinsics is not None and len(predicted_intrinsics) > 0
        else np.full((3, 3), np.nan, dtype=np.float32)
    )
    focal_px = focal_for_depth_map(calibration, source_width, source_height, depth_width, depth_height)
    if focal_px is not None:
        metric_status = "calibration_conditioned_metres_estimate"
        focal_source = "supplied_calibration"
    else:
        candidate = float((predicted_intrinsics_array[0, 0] + predicted_intrinsics_array[1, 1]) / 2.0)
        if not math.isfinite(candidate) or candidate <= 0:
            candidate = float(depth_width)
            focal_source = "canonical_depth_grid_width_fallback"
        else:
            focal_source = "da3_predicted_intrinsics"
        focal_px = candidate
        metric_status = "exploratory_metres_estimate_no_calibration"
    depth_m = (raw_depth * np.float32(focal_px / 300.0)).astype(np.float32)
    if not np.isfinite(depth_m).any() or not np.any(depth_m > 0):
        raise ExperimentError("DA3 emitted no positive finite depth values")
    arrays = {
        "depth_m_estimate": depth_m,
        "depth_canonical": raw_depth,
        "confidence": confidence,
        "intrinsics_predicted": predicted_intrinsics_array,
        "intrinsics_supplied": camera_k if camera_k is not None else np.full((3, 3), np.nan, dtype=np.float32),
        "focal_px_used": np.asarray([focal_px], dtype=np.float32),
    }
    record = {
        "latency_ms": elapsed_ms,
        "metric_status": metric_status,
        "focal_px_used": focal_px,
        "focal_source": focal_source,
        "model_input_resolution": {"width": width, "height": height},
        "canonical_depth_grid_resolution": {"width": depth_width, "height": depth_height},
        "stored_map_resolution": {"width": source_width, "height": source_height},
        "effective_precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
    }
    return arrays, record


def infer_moge(model: Any, frame_bgr: Any, source_width: int, source_height: int,
               calibration: Calibration | None, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    import cv2
    import numpy as np
    import torch

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    image = torch.from_numpy(rgb).to(device="cuda", dtype=torch.float32).permute(2, 0, 1) / 255.0
    common: dict[str, Any] = {
        "num_tokens": request.get("moge_num_tokens"),
        "resolution_level": request["moge_resolution_level"],
        "refine_steps": request["moge_refine_steps"],
        "use_fp16": request["moge_precision"] == "fp16",
        "apply_mask": True,
    }
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Save truly model-predicted intrinsics before conditioning with GoPro FOV.
    predicted = model.infer(image, fov_x=None, **common)
    torch.cuda.synchronize()
    prediction_latency_ms = (time.perf_counter() - started) * 1000.0
    if calibration is not None and calibration.fov_x_deg is not None:
        torch.cuda.synchronize()
        conditioned_started = time.perf_counter()
        output = model.infer(image, fov_x=calibration.fov_x_deg, **common)
        torch.cuda.synchronize()
        conditioned_latency_ms = (time.perf_counter() - conditioned_started) * 1000.0
        intrinsics_source = "supplied_calibration_fov"
        metric_status = "calibration_conditioned_metres_estimate"
    else:
        output = predicted
        conditioned_latency_ms = 0.0
        intrinsics_source = "moge_predicted"
        metric_status = "exploratory_metres_estimate_no_calibration"

    def output_array(mapping: dict[str, Any], key: str) -> Any:
        if key not in mapping:
            raise ExperimentError(f"MoGe output did not contain required {key!r}")
        return tensor_to_numpy(mapping[key])

    depth = output_array(output, "depth").astype(np.float32)
    valid_mask = output_array(output, "mask").astype(bool)
    points = output_array(output, "points").astype(np.float32)
    normal = output_array(output, "normal").astype(np.float32)
    used_intrinsics = output_array(output, "intrinsics").astype(np.float32)
    predicted_intrinsics = output_array(predicted, "intrinsics").astype(np.float32)
    depth = upsample_to_source(depth, source_width, source_height, cv2.INTER_LINEAR).astype(np.float32)
    valid_mask = upsample_to_source(valid_mask.astype(np.uint8), source_width, source_height, cv2.INTER_NEAREST).astype(bool)
    depth[~valid_mask] = np.nan
    if not np.isfinite(depth).any() or not np.any(depth[valid_mask] > 0):
        raise ExperimentError("MoGe emitted no positive finite valid depth values")
    arrays = {
        "depth_m_estimate": depth,
        "valid_mask": valid_mask,
        "intrinsics_predicted_normalized": predicted_intrinsics,
        "intrinsics_used_normalized": used_intrinsics,
    }
    geometry = {
        "points_camera_m_estimate": points,
        "normal_camera": normal,
        "valid_mask": output_array(output, "mask").astype(bool),
        "model_input_width": width,
        "model_input_height": height,
    }
    record = {
        "latency_ms": prediction_latency_ms + conditioned_latency_ms,
        "unconditioned_prediction_latency_ms": prediction_latency_ms,
        "calibration_conditioned_latency_ms": conditioned_latency_ms,
        "intrinsics_source": intrinsics_source,
        "metric_status": metric_status,
        "model_input_resolution": {"width": width, "height": height},
        "stored_map_resolution": {"width": source_width, "height": source_height},
    }
    return arrays, record, geometry


def write_point_cloud(path: Path, points: Any, normals: Any, mask: Any) -> dict[str, Any]:
    """Optional camera-space PLY: explicitly a model estimate, never reconstruction truth."""

    import numpy as np

    valid = mask & np.isfinite(points).all(axis=-1) & np.isfinite(normals).all(axis=-1)
    points_flat = points[valid]
    normals_flat = normals[valid]
    if not len(points_flat):
        raise ExperimentError("selected MoGe keyframe has no valid points for PLY export")
    max_points = 200_000
    stride = max(1, math.ceil(len(points_flat) / max_points))
    points_flat = points_flat[::stride]
    normals_flat = normals_flat[::stride]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("w", encoding="ascii") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write("comment Private model estimate from MoGe-3; not ground-truth reconstruction.\n")
            handle.write("comment OpenCV camera coordinates: x right, y down, z forward; metres estimated.\n")
            handle.write(f"element vertex {len(points_flat)}\n")
            for axis in ("x", "y", "z", "nx", "ny", "nz"):
                handle.write(f"property float {axis}\n")
            handle.write("end_header\n")
            for point, normal in zip(points_flat, normals_flat, strict=True):
                handle.write(" ".join(f"{float(value):.7g}" for value in (*point, *normal)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        fsync_directory(path.parent)
    finally:
        if partial.exists():
            partial.unlink(missing_ok=True)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "point_count": len(points_flat)}


def load_worker_state(path: Path, initial: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        atomic_write_json(path, initial)
        return initial
    state = load_json_object(path, "worker manifest")
    if state.get("worker_fingerprint_sha256") != initial["worker_fingerprint_sha256"]:
        raise ExperimentError("worker manifest fingerprint mismatch; choose a new run ID")
    return state


def worker_results_are_complete(run_dir: Path, model_name: str, selected: list[dict[str, Any]],
                                run_fingerprint_sha256: str) -> bool:
    """A completed marker counts only when every atomic archive still validates."""

    path = worker_state_path(run_dir, model_name)
    if not path.exists():
        return False
    try:
        state = load_json_object(path, f"{model_name} worker manifest")
        if state.get("status") != "complete":
            return False
        if state.get("worker_fingerprint", {}).get("run_fingerprint_sha256") != run_fingerprint_sha256:
            return False
        expected = {record["frame_index"] for record in selected}
        completed = {int(index) for index in state.get("completed_frames", {})}
        if expected != completed:
            return False
        for frame_index in expected:
            output = run_dir / "arrays" / model_name / f"frame_{frame_index:08d}.npz"
            verified = validate_worker_result(output, model_name)
            if verified["sha256"] != state["completed_frames"][str(frame_index)].get("sha256"):
                return False
        return True
    except ExperimentError:
        return False


def expected_preview_names(models: list[str]) -> set[str]:
    """Name every preview required for a terminal run, including comparison views."""

    names = {f"{model_name}_{kind}" for model_name in models for kind in ("metric", "structural")}
    if set(models) == {"da3", "moge3"}:
        names |= {"comparison_metric", "comparison_structural"}
    return names


def completed_run_is_valid(run_dir: Path, manifest: dict[str, Any], selected: list[dict[str, Any]],
                           requested_models: list[str]) -> bool:
    """Revalidate every terminal artifact before a rerun trusts ``status=complete``.

    Numeric archives alone are insufficient: a lost/corrupt preview or report must
    downgrade the run rather than making a partial result look complete.
    """

    if manifest.get("status") != "complete":
        return False
    try:
        if not all(
            worker_results_are_complete(run_dir, model_name, selected, manifest["run_fingerprint_sha256"])
            for model_name in requested_models
        ):
            return False
        previews = manifest.get("previews", {})
        artifacts = previews.get("artifacts")
        if previews.get("status") != "complete" or not isinstance(artifacts, dict):
            return False
        if set(artifacts) != expected_preview_names(requested_models):
            return False
        for name in sorted(artifacts):
            info = artifacts[name]
            if not isinstance(info, dict) or not isinstance(info.get("path"), str):
                return False
            path = Path(info["path"])
            if not path.is_file() or not path.resolve().is_relative_to(run_dir.resolve()):
                return False
            if info.get("sha256") != sha256_file(path):
                return False
            decoded = verify_preview_decode(path, selected)
            if decoded["sha256"] != info["sha256"]:
                return False
        report = manifest.get("report", {})
        if report.get("status") != "complete" or not isinstance(report.get("path"), str):
            return False
        report_path = Path(report["path"])
        if not report_path.is_file() or not report_path.resolve().is_relative_to(run_dir.resolve()):
            return False
        if report.get("sha256") != sha256_file(report_path):
            return False
        return True
    except (ExperimentError, OSError, ValueError):
        return False


def mark_worker_failure(state_path: Path, state: dict[str, Any], exc: BaseException) -> None:
    state["status"] = "failed"
    state["failure"] = {"at": utc_now(), "error": str(exc), "traceback": traceback.format_exc(limit=12)}
    state["updated_at"] = utc_now()
    atomic_write_json(state_path, state)


def worker_process(request: dict[str, Any]) -> None:
    """Run exactly one model in its isolated dependency environment."""

    import av
    import numpy as np
    import torch

    model_name = request["worker_model"]
    run_dir = Path(request["run_dir"])
    input_path = Path(request["input_path"])
    state_path = worker_state_path(run_dir, model_name)
    model_dir = run_dir / "arrays" / model_name
    geometry_dir = run_dir / "keyframes" / model_name
    source_clock = request["selected_source_clock"]
    calibration = calibration_from_manifest(request.get("calibration"))
    worker_fingerprint = {
        "run_fingerprint_sha256": request["run_fingerprint_sha256"],
        "model": request["model"],
        "model_input_size": request["model_input_size"],
        "moge_precision": request["moge_precision"],
        "moge_num_tokens": request["moge_num_tokens"],
        "moge_resolution_level": request["moge_resolution_level"],
        "moge_refine_steps": request["moge_refine_steps"],
        "calibration": request.get("calibration"),
        "inference_settings": request["inference_settings"],
    }
    initial_state = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "private_depth_worker",
        "worker_model": model_name,
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "worker_fingerprint": worker_fingerprint,
        "worker_fingerprint_sha256": fingerprint(worker_fingerprint),
        "expected_frame_indices": [record["frame_index"] for record in source_clock],
        "completed_frames": {},
        "model_provenance": source_model_metadata(model_name),
        "inference_settings": request["inference_settings"],
        "failures": [],
    }
    state = load_worker_state(state_path, initial_state)
    state["status"] = "running"
    state["updated_at"] = utc_now()
    atomic_write_json(state_path, state)
    try:
        if not torch.cuda.is_available():
            raise ExperimentError("CUDA is unavailable; refusing a CPU fallback for this RunPod GPU experiment")
        if model_name == "da3":
            model, model_provenance = resolve_da3_model()
        elif model_name == "moge3":
            model, model_provenance = resolve_moge_model()
        else:
            raise ExperimentError(f"unknown worker model: {model_name}")
        state["model_provenance"].update(model_provenance)
        state["environment"] = {
            "python": sys.version,
            "packages": package_versions(["torch", "torchvision", "numpy", "av", "opencv-python-headless", "depth-anything-3", "moge"]),
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_device_count": torch.cuda.device_count(),
        }
        atomic_write_json(state_path, state)

        expected_by_index = {record["frame_index"]: record for record in source_clock}
        source_width = None
        source_height = None
        container = av.open(str(input_path))
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ExperimentError("worker cannot find input video stream")
        try:
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index not in expected_by_index:
                    continue
                record = expected_by_index[frame_index]
                if frame.pts is None or int(frame.pts) != record["pts"]:
                    raise ExperimentError(
                        f"worker source clock changed at frame {frame_index}; expected PTS {record['pts']}, got {frame.pts}"
                    )
                frame_bgr = frame.to_ndarray(format="bgr24")
                source_height, source_width = frame_bgr.shape[:2]
                result_path = model_dir / f"frame_{frame_index:08d}.npz"
                existing = state["completed_frames"].get(str(frame_index))
                if existing is not None and result_path.exists():
                    try:
                        verified = validate_worker_result(result_path, model_name)
                    except ExperimentError:
                        state["completed_frames"].pop(str(frame_index), None)
                    else:
                        if verified["sha256"] == existing.get("sha256"):
                            continue
                        state["completed_frames"].pop(str(frame_index), None)

                model_width = request["model_input_size"]["width"]
                model_height = request["model_input_size"]["height"]
                model_input = resize_bgr_for_model(frame_bgr, model_width, model_height)
                torch.cuda.reset_peak_memory_stats()
                if model_name == "da3":
                    arrays, details = infer_da3(model, model_input, source_width, source_height, calibration)
                    geometry = None
                else:
                    arrays, details, geometry = infer_moge(model, model_input, source_width, source_height, calibration, request)
                arrays.update(
                    {
                        "frame_index": np.asarray([frame_index], dtype=np.int64),
                        "pts": np.asarray([record["pts"]], dtype=np.int64),
                        "time_base_num": np.asarray([Fraction(record["time_base"]).numerator], dtype=np.int64),
                        "time_base_den": np.asarray([Fraction(record["time_base"]).denominator], dtype=np.int64),
                    }
                )
                atomic_save_npz(result_path, arrays)
                verified = validate_worker_result(result_path, model_name)
                details.update(
                    {
                        "sha256": verified["sha256"],
                        "size_bytes": verified["size_bytes"],
                        "pts": record["pts"],
                        "time_base": record["time_base"],
                        "timestamp_seconds": record["timestamp_seconds"],
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
                if geometry is not None and frame_index in set(request["keyframe_indices"]):
                    geometry_path = geometry_dir / f"frame_{frame_index:08d}.npz"
                    atomic_save_npz(
                        geometry_path,
                        {
                            "points_camera_m_estimate": geometry["points_camera_m_estimate"],
                            "normal_camera": geometry["normal_camera"],
                            "valid_mask": geometry["valid_mask"],
                            "frame_index": np.asarray([frame_index], dtype=np.int64),
                            "pts": np.asarray([record["pts"]], dtype=np.int64),
                        },
                    )
                    geometry_record: dict[str, Any] = {
                        "npz_path": str(geometry_path),
                        "npz_sha256": sha256_file(geometry_path),
                        "npz_size_bytes": geometry_path.stat().st_size,
                        "coordinate_system": "OpenCV camera: x right, y down, z forward; estimated metres",
                    }
                    if request["export_point_clouds"]:
                        geometry_record["ply"] = write_point_cloud(
                            geometry_dir / f"frame_{frame_index:08d}.estimated.ply",
                            geometry["points_camera_m_estimate"],
                            geometry["normal_camera"],
                            geometry["valid_mask"],
                        )
                    details["keyframe_geometry"] = geometry_record
                state["completed_frames"][str(frame_index)] = details
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
        finally:
            container.close()

        expected_indices = {record["frame_index"] for record in source_clock}
        complete_indices = {int(item) for item in state["completed_frames"]}
        if complete_indices != expected_indices:
            missing = sorted(expected_indices - complete_indices)
            raise ExperimentError(f"worker ended with missing frames: {missing[:20]}")
        # Re-open every final archive before making a completion claim.
        for frame_index in sorted(expected_indices):
            validate_worker_result(model_dir / f"frame_{frame_index:08d}.npz", model_name)
        state["status"] = "complete"
        state["completed_at"] = utc_now()
        state["updated_at"] = utc_now()
        atomic_write_json(state_path, state)
    except BaseException as exc:
        mark_worker_failure(state_path, state, exc)
        raise


def load_depth_archive(path: Path, model_name: str) -> dict[str, Any]:
    import numpy as np

    validate_worker_result(path, model_name)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def colourize_metric_depth(depth: Any, valid: Any, minimum_m: float, maximum_m: float,
                           exploratory: bool) -> Any:
    import cv2
    import numpy as np

    normalized = np.clip((depth - minimum_m) / (maximum_m - minimum_m), 0.0, 1.0)
    image = cv2.applyColorMap((255.0 * (1.0 - normalized)).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    label = f"METRIC SCALE {minimum_m:.2f}-{maximum_m:.2f} m"
    if exploratory:
        label += " | EXPLORATORY: NO CALIBRATION"
    cv2.putText(image, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return image


def colourize_structural_depth(depth: Any, valid: Any) -> Any:
    import cv2
    import numpy as np

    finite = depth[valid & np.isfinite(depth)]
    if finite.size < 4:
        image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        lower, upper = np.percentile(finite, [2.0, 98.0])
        upper = max(float(upper), float(lower) + 1e-6)
        normalized = np.clip((depth - lower) / (upper - lower), 0.0, 1.0)
        image = cv2.applyColorMap((255.0 * (1.0 - normalized)).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[~valid] = 0
    label = "STRUCTURAL ONLY | PER-FRAME NORMALIZED | NOT A METRIC COMPARISON"
    cv2.putText(image, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (20, 20, 20), 1, cv2.LINE_AA)
    return image


def annotate_panel(image: Any, title: str, subtitle: str | None = None) -> Any:
    import cv2

    panel = image.copy()
    header_height = 62 if subtitle else 38
    cv2.rectangle(panel, (0, 0), (panel.shape[1], header_height), (0, 0, 0), thickness=-1)
    cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def quality_overlay(da3_confidence: Any, moge_valid: Any) -> Any:
    import cv2
    import numpy as np

    finite = da3_confidence[np.isfinite(da3_confidence)]
    if finite.size >= 4:
        low, high = np.percentile(finite, [2.0, 98.0])
        high = max(float(high), float(low) + 1e-6)
        values = np.clip((da3_confidence - low) / (high - low), 0.0, 1.0)
    else:
        values = np.zeros_like(da3_confidence, dtype=np.float32)
    confidence = cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    invalid = ~moge_valid.astype(bool)
    confidence[invalid] = (0, 0, 255)
    cv2.putText(confidence, "DA3 model confidence | RED = MoGe invalid", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(confidence, "DA3 model confidence | RED = MoGe invalid", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (20, 20, 20), 1, cv2.LINE_AA)
    return confidence


def tile_panels(panels: list[Any], max_width: int) -> Any:
    import cv2

    if len(panels) != 4:
        raise ExperimentError("comparison tile needs exactly four panels")
    source_height, source_width = panels[0].shape[:2]
    panel_width = min(source_width, max(160, max_width // 2))
    panel_height = max(2, round(source_height * panel_width / source_width))
    if panel_height % 2:
        panel_height -= 1
    if panel_width % 2:
        panel_width -= 1
    resized = [cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA) for panel in panels]
    return cv2.vconcat([cv2.hconcat(resized[:2]), cv2.hconcat(resized[2:])])


def open_preview_writer(path: Path, width: int, height: int, rate: Fraction, time_base: Fraction) -> tuple[Any, Any]:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = time_base
    stream.options = {"crf": "18", "preset": "medium", "movflags": "+faststart"}
    return container, stream


def encode_preview_frame(container: Any, stream: Any, bgr: Any, pts: int, time_base: Fraction) -> None:
    import av

    frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
    frame.pts = pts
    frame.time_base = time_base
    for packet in stream.encode(frame):
        container.mux(packet)


def close_preview_writer(container: Any, stream: Any) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def verify_preview_decode(path: Path, expected_records: list[dict[str, Any]]) -> dict[str, Any]:
    import av

    try:
        container = av.open(str(path))
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ExperimentError(f"preview {path} has no video stream")
        decoded = list(container.decode(stream))
        if len(decoded) != len(expected_records):
            raise ExperimentError(
                f"preview {path.name} decodes {len(decoded)} frames, expected {len(expected_records)}"
            )
        if any(frame.pts is None for frame in decoded):
            raise ExperimentError(f"preview {path.name} contains frame(s) without PTS")
        for output_frame, source_record in zip(decoded, expected_records, strict=True):
            output_time = Fraction(int(output_frame.pts)) * Fraction(stream.time_base)
            source_time = Fraction(int(source_record["pts"])) * Fraction(source_record["time_base"])
            if output_time != source_time:
                raise ExperimentError(
                    f"preview {path.name} timestamp mismatch: output {output_time}, source {source_time} "
                    f"at source frame {source_record['frame_index']}"
                )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "decoded_frame_count": len(decoded),
            "stream_time_base": str(stream.time_base),
            "status": "complete_decode_verified",
        }
    except ExperimentError:
        raise
    except Exception as exc:
        raise ExperimentError(f"preview decode check failed for {path}: {exc}") from exc
    finally:
        try:
            container.close()
        except UnboundLocalError:
            pass


def render_previews(run_dir: Path, input_path: Path, selected: list[dict[str, Any]],
                    probe: dict[str, Any], models: list[str], metric_min_m: float,
                    metric_max_m: float, preview_max_width: int) -> dict[str, Any]:
    """Render visual artifacts only after every requested numeric map is complete."""

    import av
    import numpy as np

    if not 0 < metric_min_m < metric_max_m:
        raise ExperimentError("metric preview range must satisfy 0 < --metric-min-m < --metric-max-m")
    if preview_max_width < 320:
        raise ExperimentError("--preview-max-width must be at least 320")
    selected_by_index = {record["frame_index"]: record for record in selected}
    time_base = Fraction(probe["time_base"])
    rate = fraction_from_text(probe["avg_frame_rate"], "avg_frame_rate")
    exploratory_by_model: dict[str, bool] = {}
    for model_name in models:
        worker = load_json_object(worker_state_path(run_dir, model_name), f"{model_name} worker manifest")
        frame_details = list(worker.get("completed_frames", {}).values())
        exploratory_by_model[model_name] = any(
            detail.get("metric_status") == "exploratory_metres_estimate_no_calibration"
            for detail in frame_details
        )
    previews_dir = run_dir / "previews"
    targets: dict[str, Path] = {}
    for model_name in models:
        targets[f"{model_name}_metric"] = previews_dir / f"{model_name}_metric_depth.partial.mp4"
        targets[f"{model_name}_structural"] = previews_dir / f"{model_name}_structural_only.partial.mp4"
    if set(models) == {"da3", "moge3"}:
        targets["comparison_metric"] = previews_dir / "comparison_metric.partial.mp4"
        targets["comparison_structural"] = previews_dir / "comparison_structural_only.partial.mp4"
    writers: dict[str, tuple[Any, Any]] = {}
    final_paths = {name: path.with_name(path.name.replace(".partial", "")) for name, path in targets.items()}
    try:
        input_container = av.open(str(input_path))
        stream = next((item for item in input_container.streams if item.type == "video"), None)
        if stream is None:
            raise ExperimentError("cannot render previews: input video stream is absent")
        for frame_index, frame in enumerate(input_container.decode(stream)):
            if frame_index not in selected_by_index:
                continue
            record = selected_by_index[frame_index]
            if frame.pts is None or int(frame.pts) != record["pts"]:
                raise ExperimentError(f"preview decode clock mismatch at source frame {frame_index}")
            rgb_bgr = frame.to_ndarray(format="bgr24")
            model_data = {
                model_name: load_depth_archive(run_dir / "arrays" / model_name / f"frame_{frame_index:08d}.npz", model_name)
                for model_name in models
            }
            panels_metric: dict[str, Any] = {}
            panels_structural: dict[str, Any] = {}
            for model_name, arrays in model_data.items():
                depth = arrays["depth_m_estimate"]
                valid = np.isfinite(depth)
                if model_name == "moge3":
                    valid &= arrays["valid_mask"].astype(bool)
                panels_metric[model_name] = colourize_metric_depth(
                    depth, valid, metric_min_m, metric_max_m, exploratory_by_model[model_name]
                )
                panels_structural[model_name] = colourize_structural_depth(depth, valid)
            if not writers:
                for model_name in models:
                    writers[f"{model_name}_metric"] = open_preview_writer(
                        targets[f"{model_name}_metric"], rgb_bgr.shape[1], rgb_bgr.shape[0], rate, time_base
                    )
                    writers[f"{model_name}_structural"] = open_preview_writer(
                        targets[f"{model_name}_structural"], rgb_bgr.shape[1], rgb_bgr.shape[0], rate, time_base
                    )
                if "comparison_metric" in targets:
                    comparison_shape = tile_panels([rgb_bgr, rgb_bgr, rgb_bgr, rgb_bgr], preview_max_width)
                    writers["comparison_metric"] = open_preview_writer(
                        targets["comparison_metric"], comparison_shape.shape[1], comparison_shape.shape[0], rate, time_base
                    )
                    writers["comparison_structural"] = open_preview_writer(
                        targets["comparison_structural"], comparison_shape.shape[1], comparison_shape.shape[0], rate, time_base
                    )
            for model_name in models:
                encode_preview_frame(*writers[f"{model_name}_metric"], panels_metric[model_name], record["pts"], time_base)
                encode_preview_frame(*writers[f"{model_name}_structural"], panels_structural[model_name], record["pts"], time_base)
            if "comparison_metric" in writers:
                da3 = model_data["da3"]
                moge = model_data["moge3"]
                quality = quality_overlay(da3["confidence"], moge["valid_mask"])
                metric_tile = tile_panels(
                    [
                        annotate_panel(rgb_bgr, "Redacted RGB", "Source PTS preserved"),
                        annotate_panel(
                            panels_metric["da3"],
                            "DA3 Metric Large",
                            f"Fixed scale {metric_min_m:.2f}-{metric_max_m:.2f} m"
                            + (" | exploratory, no calibration" if exploratory_by_model["da3"] else ""),
                        ),
                        annotate_panel(
                            panels_metric["moge3"],
                            "MoGe-3 ViT-L",
                            f"Fixed scale {metric_min_m:.2f}-{metric_max_m:.2f} m"
                            + (" | exploratory, no calibration" if exploratory_by_model["moge3"] else ""),
                        ),
                        annotate_panel(quality, "Quality / confidence", "DA3 confidence; red = MoGe invalid"),
                    ],
                    preview_max_width,
                )
                structural_tile = tile_panels(
                    [
                        annotate_panel(rgb_bgr, "Redacted RGB", "Source PTS preserved"),
                        annotate_panel(
                            panels_structural["da3"],
                            "DA3 structural only",
                            "Per-frame normalized; not metric comparison",
                        ),
                        annotate_panel(
                            panels_structural["moge3"],
                            "MoGe-3 structural only",
                            "Per-frame normalized; not metric comparison",
                        ),
                        annotate_panel(quality, "Quality / confidence", "DA3 confidence; red = MoGe invalid"),
                    ],
                    preview_max_width,
                )
                encode_preview_frame(*writers["comparison_metric"], metric_tile, record["pts"], time_base)
                encode_preview_frame(*writers["comparison_structural"], structural_tile, record["pts"], time_base)
        input_container.close()
        for container, stream_writer in writers.values():
            close_preview_writer(container, stream_writer)
        writers.clear()
        for partial, final in zip(targets.values(), final_paths.values(), strict=True):
            os.replace(partial, final)
            fsync_directory(final.parent)
        return {name: verify_preview_decode(path, selected) for name, path in final_paths.items()}
    except BaseException:
        for container, stream_writer in writers.values():
            try:
                close_preview_writer(container, stream_writer)
            except Exception:
                container.close()
        raise


def byte_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], amount: float) -> float | None:
    import numpy as np

    return float(np.percentile(values, amount)) if values else None


def compute_temporal_flicker(run_dir: Path, model_name: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    values: list[float] = []
    previous: Any | None = None
    for record in selected:
        archive = load_depth_archive(run_dir / "arrays" / model_name / f"frame_{record['frame_index']:08d}.npz", model_name)
        depth = archive["depth_m_estimate"]
        valid = np.isfinite(depth)
        if model_name == "moge3":
            valid &= archive["valid_mask"].astype(bool)
        if previous is not None:
            previous_depth, previous_valid = previous
            overlap = valid & previous_valid
            if overlap.any():
                values.append(float(np.median(np.abs(depth[overlap] - previous_depth[overlap]))))
        previous = depth, valid
    return {
        "definition": "median absolute depth delta over same-pixel valid intersections on adjacent selected frames; motion and viewpoint change are not separated from flicker",
        "pair_count": len(values),
        "median_m": percentile(values, 50.0),
        "p95_m": percentile(values, 95.0),
    }


def model_report_summary(run_dir: Path, model_name: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    state = load_json_object(worker_state_path(run_dir, model_name), f"{model_name} worker manifest")
    if state.get("status") != "complete":
        raise ExperimentError(f"cannot report {model_name}: worker status is {state.get('status')!r}")
    frames = list(state.get("completed_frames", {}).values())
    if len(frames) != len(selected):
        raise ExperimentError(f"cannot report {model_name}: completion count does not match selected clock")
    latencies = [float(frame["latency_ms"]) for frame in frames]
    peak_allocated = [int(frame["cuda_peak_allocated_bytes"]) for frame in frames]
    peak_reserved = [int(frame["cuda_peak_reserved_bytes"]) for frame in frames]
    output_bytes = sum(int(frame["size_bytes"]) for frame in frames)
    return {
        "worker_manifest": str(worker_state_path(run_dir, model_name)),
        "model_provenance": state["model_provenance"],
        "environment": state.get("environment"),
        "per_frame_latency_ms": {"median": percentile(latencies, 50.0), "p95": percentile(latencies, 95.0), "max": max(latencies) if latencies else None},
        "cuda_peak_memory_bytes": {"max_allocated": max(peak_allocated) if peak_allocated else None, "max_reserved": max(peak_reserved) if peak_reserved else None},
        "numeric_output_size_bytes": output_bytes,
        "temporal_flicker_proxy": compute_temporal_flicker(run_dir, model_name, selected),
    }


def representative_review_rows(selected: list[dict[str, Any]], keyframes: set[int]) -> list[str]:
    rows = []
    for record in selected:
        if record["frame_index"] in keyframes:
            rows.append(
                f"| {record['frame_index']} | {record['timestamp_seconds']:.3f} | Pending human review | "
                "hands/fingertips; hand-object separation; contact-adjacent motion; glossy cans; thin tools; "
                "shelves/clutter; occlusion; edge bleeding; motion blur |"
            )
    return rows or ["| — | — | No keyframes requested | Request keyframes before qualitative review |"]


def write_report(run_dir: Path, manifest: dict[str, Any], previews: dict[str, Any],
                 keyframes: set[int]) -> Path:
    selected = manifest["selected_source_clock"]
    models = list(manifest["models"])
    summaries = {model: model_report_summary(run_dir, model, selected) for model in models}
    output_size = byte_size(run_dir)
    calibration = manifest.get("camera_calibration")
    metric_note = (
        "A supplied calibration/FOV was used where the adapter supports it. Outputs remain model estimates; this pilot has no ground-truth depth."
        if calibration
        else "No calibrated intrinsics/FOV was supplied. Both models' metre-valued outputs are exploratory estimates, not measurements; this pilot has no ground-truth depth."
    )
    private_unredacted_exception = manifest["privacy"].get("private_unredacted_exception", False)
    scope_line = (
        "- Scope: one private unredacted source-footage exception, explicitly user-authorized; "
        "this run is non-publishable and has no uploads."
        if private_unredacted_exception
        else "- Scope: one private, already-redacted, continuous child; no SLAM, reconstruction, "
        "segmentation, production annotations, VLM calls, or uploads."
    )
    lines = [
        "# Experiment 1 — DA3 Metric Large vs MoGe-3 ViT-L",
        "",
        "## Scope and result status",
        "",
        f"- Status: **{manifest['status']}**",
        scope_line,
        f"- Source frames: {len(manifest['source_clock'])}; inferred frames: {len(selected)}.",
        f"- Input SHA-256: `{manifest['input']['sha256']}`",
        f"- Input resolution / rate: {manifest['input']['resolution']['width']}x{manifest['input']['resolution']['height']} at `{manifest['input']['frame_rate']['avg_frame_rate']}` fps.",
        f"- Metric comparison colour scale: {manifest['run_fingerprint']['metric_preview_range_m'][0]:.2f}-{manifest['run_fingerprint']['metric_preview_range_m'][1]:.2f} m, fixed for both models.",
        f"- {metric_note}",
        f"- Recorded failed attempt(s) before completion: {len(manifest.get('failures', []))}.",
        "- Decode requirement: full source decode matched ffprobe before inference; each preview MP4 was fully decoded after encode.",
        "",
        "## Provenance and licenses",
        "",
        "| Model | Checkpoint | Immutable revision | License | Pinned source revision |",
        "| --- | --- | --- | --- | --- |",
    ]
    if private_unredacted_exception:
        lines.append(
            "- Privacy restriction: input status is `private_unredacted_user_authorized`; do not publish, "
            "share, or upload this input or any derived artifact."
        )
    for model in models:
        provenance = summaries[model]["model_provenance"]
        lines.append(
            f"| {provenance['name']} | `{provenance['huggingface_model_id']}` | `{provenance['resolved_revision']}` | {provenance['license']} | `{provenance['code_revision']}` |"
        )
    lines.extend(["", "## Runtime, memory, and outputs", ""])
    for model in models:
        summary = summaries[model]
        latency = summary["per_frame_latency_ms"]
        memory = summary["cuda_peak_memory_bytes"]
        lines.extend(
            [
                f"### {summary['model_provenance']['name']}",
                "",
                f"- Per-frame latency: median {latency['median']:.1f} ms; p95 {latency['p95']:.1f} ms; max {latency['max']:.1f} ms.",
                f"- Practical peak GPU memory: allocated {memory['max_allocated'] / (1024**3):.2f} GiB; reserved {memory['max_reserved'] / (1024**3):.2f} GiB.",
                f"- Lossless numeric output: {summary['numeric_output_size_bytes'] / (1024**2):.1f} MiB.",
                f"- Temporal-flicker proxy: {summary['temporal_flicker_proxy']['median_m']!r} m median / {summary['temporal_flicker_proxy']['p95_m']!r} m p95 over {summary['temporal_flicker_proxy']['pair_count']} adjacent selected-frame pairs. This proxy includes true camera/object motion.",
            ]
        )
    lines.extend(
        [
            "",
            f"- Total private run size: {output_size / (1024**2):.1f} MiB.",
            "- Every preview MP4 was fully decoded after encoding; results are listed in the machine-readable manifest.",
            "",
            "## Representative-frame qualitative assessment",
            "",
            "The job deliberately does **not** make semantic claims about hands or objects from depth arrays. The rows below are a required human assessment record, preselected evenly across the clip. Use the metric comparison first; use the structurally normalised comparison only to inspect shape, never to compare scale.",
            "",
            "| Source frame | Source time (s) | Assessment status | Review for |",
            "| ---: | ---: | --- | --- |",
            *representative_review_rows(selected, keyframes),
            "",
            "Required review criteria: hands and fingertips; hand-object separation; contact-adjacent motion; glossy paint cans; thin tools; shelves and clutter; occlusion; edge bleeding; motion blur; and temporal flicker. Record which model is more plausible and any failure frame in this private report before treating the pilot as evidence.",
            "",
            "## Metric-scale limitation",
            "",
            "There is no ground-truth depth in this pilot. Visual plausibility, agreement between models, or a stable-looking preview must not be converted into an accuracy claim. DA3's conversion uses `focal_px * canonical_depth / 300`; MoGe reports a model-estimated metric point/depth map. Calibration reduces an input ambiguity but is not a validation target.",
            "",
            "## Preview artifacts",
            "",
        ]
    )
    for name, info in previews.items():
        lines.append(f"- `{name}`: `{info['path']}` — {info['decoded_frame_count']} frames, decode verified.")
    report_path = run_dir / "REPORT.md"
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def run_orchestrator(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    attestation_path = args.input_attestation.resolve()
    ensure_private_path(input_path, "--input")
    ensure_private_path(output_dir, "--output-dir")
    validate_run_id(args.run_id)
    if not input_path.is_file():
        raise ExperimentError(f"input video does not exist: {input_path}")
    if not attestation_path.is_file():
        raise ExperimentError(f"input attestation does not exist: {attestation_path}")
    if args.metric_min_m <= 0 or args.metric_max_m <= args.metric_min_m:
        raise ExperimentError("metric scale requires 0 < --metric-min-m < --metric-max-m")
    if args.moge_resolution_level < 0 or args.moge_resolution_level > 9:
        raise ExperimentError("--moge-resolution-level must be in [0, 9]")
    if args.moge_refine_steps < 0:
        raise ExperimentError("--moge-refine-steps must be non-negative")
    if args.moge_num_tokens is not None and args.moge_num_tokens <= 0:
        raise ExperimentError("--moge-num-tokens must be positive")

    run_dir = output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_sha256 = sha256_file(input_path)
    attestation = require_input_attestation(
        attestation_path, input_sha256, args.allow_private_unredacted_input
    )
    probe = run_ffprobe(input_path, args.ffprobe_bin)
    # This durable preflight record exists before the expensive full decode or a model download.
    # It is the binding record for a Drive transfer even when a corrupt clip fails later.
    transfer_manifest_path = run_dir / "transfer_preflight_manifest.json"
    transfer_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "private_depth_experiment_input_preflight",
        "status": "ffprobe_complete_pending_full_decode",
        "recorded_at": utc_now(),
        "input": {
            "local_path": str(input_path),
            "sha256": input_sha256,
            "size_bytes": input_path.stat().st_size,
            "ffprobe": probe,
            "frame_count": probe["frame_count"],
            "frame_rate": {
                "avg_frame_rate": probe["avg_frame_rate"],
                "r_frame_rate": probe["r_frame_rate"],
                "time_base": probe["time_base"],
            },
            "resolution": {"width": probe["width"], "height": probe["height"]},
            "input_attestation_path": str(attestation_path),
            "input_attestation_sha256": sha256_file(attestation_path),
            "input_privacy_status": attestation.get("privacy_status"),
        },
    }
    atomic_write_json(transfer_manifest_path, transfer_manifest)
    try:
        clock = decode_source_clock(input_path, probe["frame_count"])
    except BaseException as exc:
        transfer_manifest.update({"status": "failed_incomplete_or_unreadable_decode", "failure": str(exc), "updated_at": utc_now()})
        atomic_write_json(transfer_manifest_path, transfer_manifest)
        raise
    transfer_manifest.update({"status": "complete_decode_verified", "decoded_frame_count": len(clock), "updated_at": utc_now()})
    atomic_write_json(transfer_manifest_path, transfer_manifest)
    calibration = load_calibration(args.camera_calibration.resolve() if args.camera_calibration else None, probe["width"], probe["height"])
    resize = parse_resize(args.resize)
    input_size = model_input_size(probe["width"], probe["height"], resize, args.max_resolution)
    selected = select_source_frames(clock, args.fps)
    keyframes = choose_keyframes(selected, args.keyframe_count)
    input_metadata = compact_input_metadata(probe, input_path, input_sha256, attestation_path, attestation, clock)
    run_fingerprint = build_run_fingerprint(
        args, input_sha256, attestation_path, attestation, clock, selected, calibration, input_size
    )
    manifest_path = run_dir / "experiment_manifest.json"
    manifest = load_or_create_run(
        manifest_path,
        initial_run_manifest(args.run_id, run_dir, input_metadata, calibration, clock, selected, run_fingerprint),
    )
    requested_models = model_specs(args.models)
    if manifest.get("status") == "complete":
        if completed_run_is_valid(run_dir, manifest, selected, requested_models):
            print(f"experiment already complete and verified: {manifest_path}")
            return 0
        raise ExperimentError(
            "run manifest claims complete but one or more numeric, preview, or report artifacts no longer validate; "
            "use a new run ID rather than overwriting completed evidence."
        )
    update_run_manifest(manifest_path, manifest, status="preflight_complete")
    if args.dry_run:
        update_run_manifest(
            manifest_path,
            manifest,
            status="dry_run_complete",
            dry_run={"completed_at": utc_now(), "models_not_downloaded": True, "selected_frame_count": len(selected)},
        )
        print(f"dry run complete: {manifest_path}")
        return 0

    try:
        for model_name in requested_models:
            request_path = build_worker_request(
                run_dir,
                model_name,
                input_path,
                selected,
                manifest["run_fingerprint_sha256"],
                calibration,
                input_size,
                args,
                keyframes,
            )
            update_run_manifest(
                manifest_path,
                manifest,
                status="running",
                models={
                    **manifest["models"],
                    model_name: {**manifest["models"][model_name], "status": "running", "worker_request": str(request_path)},
                },
            )
            if not worker_results_are_complete(
                run_dir, model_name, selected, manifest["run_fingerprint_sha256"]
            ):
                run_worker_subprocess(request_path, model_name)
            worker_state = load_json_object(worker_state_path(run_dir, model_name), f"{model_name} worker manifest")
            if worker_state.get("status") != "complete":
                raise ExperimentError(f"{model_name} worker did not make a completion claim")
            manifest["models"][model_name].update(
                {"status": "complete", "worker_manifest": str(worker_state_path(run_dir, model_name))}
            )
            update_run_manifest(manifest_path, manifest)

        previews = render_previews(
            run_dir,
            input_path,
            selected,
            probe,
            requested_models,
            args.metric_min_m,
            args.metric_max_m,
            args.preview_max_width,
        )
        manifest["previews"] = {"status": "complete", "artifacts": previews}
        update_run_manifest(manifest_path, manifest)
        # The report describes the final intended terminal state.  If its atomic
        # write fails, the exception handler below persists ``failed`` instead.
        manifest["status"] = "complete"
        report_path = write_report(run_dir, manifest, previews, keyframes)
        manifest["report"] = {"status": "complete", "path": str(report_path), "sha256": sha256_file(report_path)}
        manifest["completed_at"] = utc_now()
        update_run_manifest(manifest_path, manifest)
        print(f"experiment complete: {manifest_path}")
        return 0
    except BaseException as exc:
        failures = list(manifest.get("failures", []))
        failures.append({"at": utc_now(), "error": str(exc)})
        update_run_manifest(manifest_path, manifest, status="failed", failures=failures)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="private local path to one continuous input clip")
    parser.add_argument(
        "--input-attestation", "--redacted-input-attestation", dest="input_attestation", type=Path,
        help="private JSON binding input hash, continuity, projection, and truthful privacy status",
    )
    parser.add_argument(
        "--allow-private-unredacted-input", action="store_true",
        help=("allow only privacy_status=private_unredacted_user_authorized with publication_permitted=false; "
              "records a non-publishable private exception"),
    )
    parser.add_argument("--output-dir", type=Path, help="private parent directory for experiment run IDs")
    parser.add_argument("--run-id", help="private experiment run ID")
    parser.add_argument("--models", choices=["both", "da3", "moge3"], default="both", help="default: both")
    parser.add_argument("--camera-calibration", type=Path, help="optional approved private rectilinear/dewarped calibration JSON")
    parser.add_argument("--fps", type=float, help="explicit lower inference FPS; default processes every source frame")
    parser.add_argument("--resize", help="explicit model input WIDTHxHEIGHT; maps are upsampled and documented at source alignment")
    parser.add_argument("--max-resolution", type=int, help="aspect-preserving maximum model-input long edge")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--moge-num-tokens", type=int, help="optional MoGe token budget; overrides resolution level")
    parser.add_argument("--moge-resolution-level", type=int, default=9, help="MoGe level 0..9; default 9")
    parser.add_argument("--moge-refine-steps", type=int, default=3, help="MoGe sparse refinement steps; default 3")
    parser.add_argument("--metric-min-m", type=float, default=DEFAULT_METRIC_MIN_M)
    parser.add_argument("--metric-max-m", type=float, default=DEFAULT_METRIC_MAX_M)
    parser.add_argument("--keyframe-count", type=int, default=DEFAULT_KEYFRAME_COUNT)
    parser.add_argument("--export-point-clouds", action="store_true", help="write optional selected-keyframe estimated camera-space PLYs")
    parser.add_argument("--preview-max-width", type=int, default=1920, help="maximum width of 2x2 comparison preview")
    parser.add_argument("--ffprobe-bin", default=os.environ.get("FFPROBE_BIN", "ffprobe"))
    parser.add_argument("--dry-run", action="store_true", help="preflight/hash/decode only; do not download models or infer")
    parser.add_argument("--worker", choices=["da3", "moge3"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker is not None:
            if args.worker_request is None:
                raise ExperimentError("worker mode requires --worker-request")
            request = load_json_object(args.worker_request, "worker request")
            if request.get("worker_model") != args.worker:
                raise ExperimentError("worker request model does not match --worker")
            worker_process(request)
            return 0
        missing = [name for name in ("input", "input_attestation", "output_dir", "run_id") if getattr(args, name) is None]
        if missing:
            raise ExperimentError(f"required arguments missing: {', '.join('--' + item.replace('_', '-') for item in missing)}")
        return run_orchestrator(args)
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
