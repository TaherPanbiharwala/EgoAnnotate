"""Private, pre-redaction MediaPipe Pose prior for redaction review.

This module reads an original video solely to identify limbs that commonly
trigger face-detector false positives.  It is deliberately a *soft* prior:
missing pose information means ``unknown``, never ``safe``.  The resulting
artifact therefore lives in a private/DO-NOT-SHIP directory and is consumed
only to rank review candidates; it must never define where a face detector is
allowed to look.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from .media.probe import VideoInfo, probe

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "pre_redaction_pose_prior"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
POSE_MODEL_FILENAME = "pose_landmarker_full.task"
POSE_MODEL_MIN_BYTES = 1_000_000
POSE_NUM_POSES = 4
POSE_MIN_CONFIDENCE = 0.5
POSE_DETECT_HZ = 10.0
LIMB_OVERLAP_GRID = 24
LIMB_OVERLAP_LOWER_PRIORITY = 0.25
WEARER_EDGE_MARGIN = 0.12
WEARER_LOWER_BAND = 0.66
POSE_PREVIEW_WEARER_BGR = (0, 191, 255)  # amber
POSE_PREVIEW_OTHER_BGR = (255, 180, 0)  # blue

# Only limb joints are persisted. Face/torso landmarks are neither needed nor
# retained: the artifact is a negative prior over hand/arm/leg anatomy, not a
# person-reidentification representation.
_LIMB_LANDMARK_INDICES = frozenset(range(11, 33))
_ARM_CHAINS = ((11, 13, 15), (12, 14, 16))
_LEG_CHAINS = ((23, 25, 27), (24, 26, 28))
_HAND_TIPS = ((15, 17, 19, 21), (16, 18, 20, 22))
_FOOT_TIPS = ((27, 29, 31), (28, 30, 32))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is a canonical SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def require_private_path(path: Path, *, option: str) -> None:
    """Reject a privacy-sensitive artifact path outside a deliberate boundary."""
    parts = path.resolve().parts
    # macOS has a filesystem root named /private. That alone is not a caller's
    # deliberate private artifact boundary, hence parts[2:].
    has_private_dir = any(part == "private" for part in parts[2:])
    if not has_private_dir and "DO-NOT-SHIP" not in parts:
        raise ValueError(
            f"{option} must be inside a directory named private or DO-NOT-SHIP; "
            "it contains data derived from an original video and must never be published"
        )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_model(models_dir: Path) -> Path:
    """Return the pinned full PoseLandmarker task, downloading atomically."""
    target = models_dir / POSE_MODEL_FILENAME
    if target.is_file():
        if target.stat().st_size < POSE_MODEL_MIN_BYTES:
            raise RuntimeError(
                f"PoseLandmarker model {target} is only {target.stat().st_size} bytes; "
                "remove it and retry rather than running an incomplete model"
            )
        return target

    models_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=models_dir, suffix=".partial")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(POSE_MODEL_URL, timeout=120) as response:
            shutil.copyfileobj(response, stream)
        if temporary.stat().st_size < POSE_MODEL_MIN_BYTES:
            raise RuntimeError(
                f"PoseLandmarker download is only {temporary.stat().st_size} bytes; "
                "the transfer was truncated"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _build_detector(model: Path, *, num_poses: int, confidence: float):
    base_options = mp.tasks.BaseOptions
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=confidence,
        min_pose_presence_confidence=confidence,
        min_tracking_confidence=confidence,
        output_segmentation_masks=False,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _landmark_row(landmark: Any) -> list[float | None]:
    """Normalized x/y plus quality fields; z is not used by the prior."""
    return [
        float(landmark.x),
        float(landmark.y),
        (None if getattr(landmark, "visibility", None) is None else float(landmark.visibility)),
        (None if getattr(landmark, "presence", None) is None else float(landmark.presence)),
    ]


def _usable(point: list[float | None], confidence: float) -> bool:
    if len(point) != 4 or point[0] is None or point[1] is None:
        return False
    visibility, presence = point[2], point[3]
    # A missing quality value is not assumed reliable. This only removes a
    # negative-prior region; it must never create a false "no limb" claim.
    return visibility is not None and presence is not None and visibility >= confidence and presence >= confidence


def _xy(point: list[float | None], width: int, height: int) -> tuple[float, float]:
    return float(point[0]) * width, float(point[1]) * height


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _capsule(a: tuple[float, float], b: tuple[float, float], radius: float, kind: str) -> dict[str, Any]:
    return {"shape": "capsule", "kind": kind, "a": list(a), "b": list(b), "radius": radius}


def _circle(center: tuple[float, float], radius: float, kind: str) -> dict[str, Any]:
    return {"shape": "circle", "kind": kind, "center": list(center), "radius": radius}


def limb_regions(
    landmarks: dict[str, list[float | None]], *, width: int, height: int,
    confidence: float = POSE_MIN_CONFIDENCE,
) -> list[dict[str, Any]]:
    """Turn reliable limb-only landmarks into conservative capsules/circles.

    There is intentionally no face or torso region in this representation.
    The output can identify a likely arm/hand/leg false positive, but never
    establishes an area that is safe to skip during face detection.
    """
    regions: list[dict[str, Any]] = []

    def point(index: int) -> tuple[float, float] | None:
        raw = landmarks.get(str(index))
        return _xy(raw, width, height) if raw is not None and _usable(raw, confidence) else None

    for chains, kind in ((_ARM_CHAINS, "arm"), (_LEG_CHAINS, "leg")):
        for chain in chains:
            for first, second in pairwise(chain):
                a, b = point(first), point(second)
                if a is None or b is None:
                    continue
                radius = max(12.0, min(_distance(a, b) * 0.20, max(width, height) * 0.08))
                regions.append(_capsule(a, b, radius, kind))

    for chain in _HAND_TIPS:
        wrist = point(chain[0])
        tips = [candidate for index in chain[1:] if (candidate := point(index)) is not None]
        if wrist is None:
            continue
        radius = max([12.0, *(_distance(wrist, candidate) * 1.25 for candidate in tips)])
        regions.append(_circle(wrist, min(radius, max(width, height) * 0.08), "hand"))

    for chain in _FOOT_TIPS:
        ankle = point(chain[0])
        tips = [candidate for index in chain[1:] if (candidate := point(index)) is not None]
        if ankle is None:
            continue
        radius = max([12.0, *(_distance(ankle, candidate) * 1.10 for candidate in tips)])
        regions.append(_circle(ankle, min(radius, max(width, height) * 0.08), "foot"))

    return regions


def _point_in_region(x: float, y: float, region: dict[str, Any]) -> bool:
    radius = float(region["radius"])
    if region["shape"] == "circle":
        cx, cy = (float(value) for value in region["center"])
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    if region["shape"] != "capsule":
        raise ValueError(f"unsupported limb region shape: {region['shape']!r}")
    ax, ay = (float(value) for value in region["a"])
    bx, by = (float(value) for value in region["b"])
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-9:
        return (x - ax) ** 2 + (y - ay) **2 <= radius**2
    fraction = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_squared))
    px, py = ax + fraction * dx, ay + fraction * dy
    return (x - px) **2 + (y - py) **2 <= radius**2


def limb_overlap_fraction(
    box: Iterable[float], regions: Iterable[dict[str, Any]], *, grid: int = LIMB_OVERLAP_GRID
) -> float | None:
    """Approximate the fraction of a candidate face box inside any limb mask.

    ``None`` means no reliable pose regions existed at that frame. It is not
    equivalent to 0.0: callers must preserve that distinction when ranking
    review candidates.
    """
    region_list = list(regions)
    if not region_list:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    if grid < 1:
        raise ValueError("overlap grid must be >= 1")
    inside = 0
    for x_index in range(grid):
        x = x1 + (x_index + 0.5) / grid * (x2 - x1)
        for y_index in range(grid):
            y = y1 + (y_index + 0.5) / grid * (y2 - y1)
            if any(_point_in_region(x, y, region) for region in region_list):
                inside += 1
    return inside / (grid * grid)


def _expected_indices(info: VideoInfo, detect_hz: float) -> set[int]:
    if detect_hz <= 0:
        raise ValueError("detect_hz must be > 0")
    stride = max(1, round(info.fps / detect_hz))
    return set(range(0, info.n_frames, stride))


def _pose_rows(result: Any, *, width: int, height: int, confidence: float) -> list[dict[str, Any]]:
    poses: list[dict[str, Any]] = []
    for pose in result.pose_landmarks:
        if len(pose) < 33:
            raise RuntimeError(f"PoseLandmarker returned {len(pose)} landmarks, expected 33")
        landmarks = {str(index): _landmark_row(pose[index]) for index in _LIMB_LANDMARK_INDICES}
        poses.append(
            {
                "landmarks": landmarks,
                "limb_regions": limb_regions(
                    landmarks, width=width, height=height, confidence=confidence
                ),
                "wearer_candidate": _is_camera_near_wearer_candidate(
                    landmarks, confidence=confidence
                ),
            }
        )
    return poses


def _is_camera_near_wearer_candidate(
    landmarks: dict[str, list[float | None]], *, confidence: float
) -> bool:
    """Flag a pose that may belong to the camera wearer, conservatively.

    This deliberately describes a *candidate*, not identity. A pose must have
    at least two reliable limb landmarks against the frame edge or lower band;
    the raw all-person regions are retained privately so the pilot can measure
    whether this heuristic is useful before it ever affects a production
    policy.
    """
    camera_near = 0
    for point in landmarks.values():
        if not _usable(point, confidence):
            continue
        x, y = float(point[0]), float(point[1])
        if x <= WEARER_EDGE_MARGIN or x >= 1.0 - WEARER_EDGE_MARGIN or y >= WEARER_LOWER_BAND:
            camera_near += 1
    return camera_near >= 2


def _draw_pose_region(frame_bgr: Any, region: dict[str, Any], color: tuple[int, int, int]) -> None:
    """Draw one limb-only mask on a private pose-preview frame."""
    thickness = 2
    radius = max(1, round(float(region["radius"])))
    if region["shape"] == "circle":
        center = tuple(round(float(value)) for value in region["center"])
        cv2.circle(frame_bgr, center, radius, color, thickness, cv2.LINE_AA)
        return
    a = tuple(round(float(value)) for value in region["a"])
    b = tuple(round(float(value)) for value in region["b"])
    cv2.line(frame_bgr, a, b, color, thickness, cv2.LINE_AA)
    cv2.circle(frame_bgr, a, radius, color, thickness, cv2.LINE_AA)
    cv2.circle(frame_bgr, b, radius, color, thickness, cv2.LINE_AA)


def draw_pose_preview(
    frame_bgr: Any,
    poses: Iterable[dict[str, Any]],
    *,
    frame_idx: int,
    sampled_this_frame: bool,
) -> None:
    """Overlay limb masks on an original-only, private review preview.

    Amber is a camera-near wearer *candidate*, never a verified identity.
    Blue is another detected pose. The latest 10 Hz pose is held until the
    next sample so the video makes the temporal prior understandable.
    """
    pose_rows = list(poses)
    wearer_count = 0
    for pose in pose_rows:
        wearer = bool(pose.get("wearer_candidate"))
        wearer_count += wearer
        color = POSE_PREVIEW_WEARER_BGR if wearer else POSE_PREVIEW_OTHER_BGR
        for region in pose.get("limb_regions", []):
            _draw_pose_region(frame_bgr, region, color)
    sample_label = "sample" if sampled_this_frame else "held"
    label = (
        f"Pose {sample_label} | amber: wearer candidate ({wearer_count}) "
        f"| blue: other pose | frame {frame_idx}"
    )
    cv2.rectangle(frame_bgr, (0, 0), (min(frame_bgr.shape[1], 820), 32), (0, 0, 0), -1)
    cv2.putText(frame_bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)


def build_pose_prior(
    *,
    original_video: Path,
    video_id: str,
    output: Path,
    models_dir: Path,
    detect_hz: float = POSE_DETECT_HZ,
    num_poses: int = POSE_NUM_POSES,
    confidence: float = POSE_MIN_CONFIDENCE,
    preview_video: Path | None = None,
) -> dict[str, Any]:
    """Create a complete private limb-prior artifact from an original video."""
    original_video = original_video.resolve()
    output = output.resolve()
    models_dir = models_dir.resolve()
    preview_video = preview_video.resolve() if preview_video is not None else None
    require_private_path(output, option="--output")
    require_private_path(models_dir, option="--models-dir")
    if preview_video is not None:
        require_private_path(preview_video, option="--preview-video")
    if not original_video.is_file():
        raise FileNotFoundError(f"original video not found: {original_video}")
    if not video_id:
        raise ValueError("video_id must not be empty")
    if num_poses < 1:
        raise ValueError("num_poses must be >= 1")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing pose prior: {output}")
    if preview_video is not None and preview_video.exists():
        raise FileExistsError(f"refusing to overwrite existing pose preview: {preview_video}")
    if preview_video is not None and preview_video == original_video:
        raise ValueError("--preview-video must not overwrite --original-video")
    if preview_video is not None and preview_video == output:
        raise ValueError("--preview-video must be different from --output")

    info = probe(original_video)
    expected = _expected_indices(info, detect_hz)
    model = ensure_model(models_dir)
    detector = _build_detector(model, num_poses=num_poses, confidence=confidence)
    capture = cv2.VideoCapture(str(original_video))
    if not capture.isOpened():
        detector.close()
        raise RuntimeError(f"OpenCV could not open original video {original_video}")

    frames: list[dict[str, Any]] = []
    source_idx = 0
    completed = False
    latest_poses: list[dict[str, Any]] = []
    preview_partial: Path | None = None
    preview_writer: Any | None = None
    if preview_video is not None:
        preview_video.parent.mkdir(parents=True, exist_ok=True)
        preview_partial = preview_video.with_name(
            f"{preview_video.stem}.partial{preview_video.suffix}"
        )
        preview_partial.unlink(missing_ok=True)
        preview_writer = cv2.VideoWriter(
            str(preview_partial),
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps,
            (info.width, info.height),
        )
        if not preview_writer.isOpened():
            capture.release()
            detector.close()
            raise RuntimeError(f"OpenCV could not create pose preview: {preview_video}")
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_bgr.shape[1] != info.width or frame_bgr.shape[0] != info.height:
                raise RuntimeError(
                    f"pose decode dimensions changed to {frame_bgr.shape[1]}x{frame_bgr.shape[0]}; "
                    f"expected {info.width}x{info.height}"
                )
            if source_idx in expected:
                timestamp_ms = round(source_idx / info.fps * 1000)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = detector.detect_for_video(image, timestamp_ms)
                poses = _pose_rows(result, width=info.width, height=info.height, confidence=confidence)
                latest_poses = poses
                frames.append(
                    {
                        "frame_idx": source_idx,
                        "timestamp_ms": timestamp_ms,
                        "poses": poses,
                        "limb_regions": [
                            region for pose in poses for region in pose["limb_regions"]
                        ],
                        "wearer_limb_regions": [
                            region
                            for pose in poses
                            if pose["wearer_candidate"]
                            for region in pose["limb_regions"]
                        ],
                    }
                )
            if preview_writer is not None:
                preview_frame = frame_bgr.copy()
                draw_pose_preview(
                    preview_frame,
                    latest_poses,
                    frame_idx=source_idx,
                    sampled_this_frame=source_idx in expected,
                )
                preview_writer.write(preview_frame)
            source_idx += 1
        if source_idx != info.n_frames:
            raise RuntimeError(
                f"pose decode of {original_video} yielded {source_idx} frames, "
                f"expected {info.n_frames}; refusing a partial pose prior"
            )
        emitted = {int(frame["frame_idx"]) for frame in frames}
        if emitted != expected:
            raise RuntimeError(
                f"pose sampling emitted {len(emitted)}/{len(expected)} frames; "
                f"missing={sorted(expected - emitted)[:8]}"
            )
        completed = True
    finally:
        if preview_writer is not None:
            preview_writer.release()
        detector.close()
        capture.release()
        if preview_partial is not None and not completed:
            preview_partial.unlink(missing_ok=True)

    if preview_video is not None:
        assert preview_partial is not None
        if not preview_partial.is_file() or preview_partial.stat().st_size == 0:
            raise RuntimeError(f"pose preview was not written: {preview_video}")
        preview_partial.replace(preview_video)

    stride = max(1, round(info.fps / detect_hz))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "privacy": "private_original_derived_do_not_ship",
        "source": {
            "video_id": video_id,
            "sha256": sha256_file(original_video),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "n_frames": info.n_frames,
        },
        "sampling": {
            "detect_hz": detect_hz,
            "stride": stride,
            "n_sampled_frames": len(frames),
        },
        "model": {
            "name": "mediapipe_pose_landmarker_full",
            "url": POSE_MODEL_URL,
            "sha256": sha256_file(model),
        },
        "configuration": {
            "num_poses": num_poses,
            "min_confidence": confidence,
            "limb_overlap_grid": LIMB_OVERLAP_GRID,
            "limb_overlap_lower_priority": LIMB_OVERLAP_LOWER_PRIORITY,
            "wearer_edge_margin": WEARER_EDGE_MARGIN,
            "wearer_lower_band": WEARER_LOWER_BAND,
        },
        "frames": frames,
    }
    if preview_video is not None:
        artifact["preview"] = {
            "path": preview_video.name,
            "sha256": sha256_file(preview_video),
            "fps": info.fps,
            "n_frames": info.n_frames,
            "privacy": "private_original_derived_do_not_ship",
        }
    _atomic_json(output, artifact)
    return artifact


def load_pose_prior(
    path: Path,
    *,
    source_sha256: str,
    width: int,
    height: int,
    fps: float,
    n_frames: int,
    detect_hz: float,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    """Validate a private prior against a particular source timeline.

    The return value preserves absent pose data as empty region lists. Empty
    means unknown; it is never a claim that an area is safe.
    """
    path = path.resolve()
    require_private_path(path, option="--pose-prior")
    if not path.is_file():
        raise FileNotFoundError(f"pose prior not found: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pose prior JSON {path}: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION \
            or artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"{path} is not a supported pose-prior artifact")
    source = artifact.get("source")
    sampling = artifact.get("sampling")
    model = artifact.get("model")
    if not isinstance(source, dict) or not isinstance(sampling, dict) or not isinstance(model, dict):
        raise ValueError(f"{path} lacks source/sampling/model provenance")
    expected_source = {
        "sha256": source_sha256,
        "width": width,
        "height": height,
        "fps": fps,
        "n_frames": n_frames,
    }
    mismatches = {
        name: {"expected": expected, "artifact": source.get(name)}
        for name, expected in expected_source.items()
        if source.get(name) != expected
    }
    if sampling.get("detect_hz") != detect_hz:
        mismatches["detect_hz"] = {"expected": detect_hz, "artifact": sampling.get("detect_hz")}
    model_sha = model.get("sha256")
    if not _is_sha256(model_sha):
        mismatches["model.sha256"] = {"expected": "64 hex characters", "artifact": model_sha}
    if mismatches:
        raise ValueError(f"pose prior provenance does not match this clip: {mismatches}")

    stride = max(1, round(fps / detect_hz))
    expected_indices = set(range(0, n_frames, stride))
    frames = artifact.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{path} frames must be a list")
    by_frame: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError(f"{path} has a non-object frame entry")
        try:
            frame_idx = int(row["frame_idx"])
            regions = row["limb_regions"]
            wearer_regions = row["wearer_limb_regions"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} has an invalid frame entry") from exc
        if frame_idx not in expected_indices or frame_idx in by_frame:
            raise ValueError(f"{path} has unexpected or duplicate sampled frame {frame_idx}")
        if not isinstance(regions, list) or not isinstance(wearer_regions, list):
            raise ValueError(f"{path} frame {frame_idx} limb regions must be lists")
        for region in [*regions, *wearer_regions]:
            _validate_region(region)
        by_frame[frame_idx] = {"all": regions, "wearer": wearer_regions}
    if set(by_frame) != expected_indices:
        raise ValueError(
            f"pose prior is incomplete: sampled_frames={len(by_frame)}/{len(expected_indices)}, "
            f"missing={sorted(expected_indices - set(by_frame))[:8]}"
        )
    return by_frame


def _validate_region(region: Any) -> None:
    if not isinstance(region, dict) or region.get("shape") not in {"circle", "capsule"}:
        raise ValueError(f"invalid limb region: {region!r}")
    try:
        radius = float(region["radius"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid limb region radius: {region!r}") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"invalid limb region radius: {region!r}")
    point_names = ("center",) if region["shape"] == "circle" else ("a", "b")
    for name in point_names:
        point = region.get(name)
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"invalid limb region point {name}: {region!r}")
        try:
            if not all(math.isfinite(float(value)) for value in point):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid limb region point {name}: {region!r}") from exc
