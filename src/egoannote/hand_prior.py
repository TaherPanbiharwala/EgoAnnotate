"""Private, pre-redaction MediaPipe Hand Landmarker prior.

This is deliberately separate from :mod:`layers.hands`: that stage consumes a
*redacted* video for the public annotation dataset, while this module reads an
original only to prevent a known EgoBlur false positive (the camera wearer's
own hand) from being amplified into a long fill track.  It never supplies a
face-search ROI.  Missing or ambiguous hand evidence is unknown, never safe.
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
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from .media.probe import VideoInfo, probe
from .pose_prior import require_private_path

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "pre_redaction_hand_prior"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_FILENAME = "hand_landmarker.task"
HAND_MODEL_MIN_BYTES = 5_000_000
HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
HAND_NUM_HANDS = 2
HAND_MIN_CONFIDENCE = 0.5
HAND_DETECT_HZ = 10.0

# A hand has to be both large enough to plausibly belong to the wearer and
# camera-near.  The geometry is intentionally a candidate heuristic, not an
# identity assertion.  Track and face-overlap gates below make suppression far
# stricter still.
WEARER_EDGE_MARGIN = 0.15
WEARER_LOWER_BAND = 0.70
WEARER_MIN_RADIUS_FRAC = 0.025
TRACK_MAX_DISTANCE = 0.28
TRACK_MAX_GAP_SAMPLES = 1
MIN_STABLE_SAMPLES = 5

# The GPU job validates this complete mapping before an artifact can change
# redaction behaviour. A matching schema alone is not enough provenance.
ACTIVE_SUPPRESSION_CONFIGURATION = {
    "num_hands": HAND_NUM_HANDS,
    "min_confidence": HAND_MIN_CONFIDENCE,
    "wearer_edge_margin": WEARER_EDGE_MARGIN,
    "wearer_lower_band": WEARER_LOWER_BAND,
    "wearer_min_radius_frac": WEARER_MIN_RADIUS_FRAC,
    "track_max_distance": TRACK_MAX_DISTANCE,
    "track_max_gap_samples": TRACK_MAX_GAP_SAMPLES,
    "min_stable_samples": MIN_STABLE_SAMPLES,
}

HAND_PREVIEW_STABLE_BGR = (0, 191, 255)  # amber
HAND_PREVIEW_PROVISIONAL_BGR = (255, 0, 255)  # magenta
HAND_PREVIEW_OTHER_BGR = (255, 180, 0)  # blue


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
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


def _verify_model(path: Path) -> None:
    if path.stat().st_size < HAND_MODEL_MIN_BYTES:
        raise RuntimeError(
            f"HandLandmarker model {path} is only {path.stat().st_size} bytes; "
            "remove it and retry rather than running an incomplete model"
        )
    actual = sha256_file(path)
    if actual != HAND_MODEL_SHA256:
        raise RuntimeError(
            f"HandLandmarker model sha256 mismatch: expected {HAND_MODEL_SHA256}, "
            f"got {actual}. Remove {path} and retry so no unexpected model bytes "
            "can influence active suppression."
        )


def ensure_model(models_dir: Path) -> Path:
    """Return the pinned Hand Landmarker task, downloading atomically."""
    target = models_dir / HAND_MODEL_FILENAME
    if target.is_file():
        _verify_model(target)
        return target

    models_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=models_dir, suffix=".partial")
    temporary = Path(temporary_name)
    try:
        with (
            os.fdopen(fd, "wb") as stream,
            urllib.request.urlopen(HAND_MODEL_URL, timeout=120) as response,
        ):
            shutil.copyfileobj(response, stream)
        _verify_model(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _build_detector(model: Path, *, num_hands: int, confidence: float):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def _expected_indices(info: VideoInfo, detect_hz: float) -> set[int]:
    if detect_hz <= 0:
        raise ValueError("detect_hz must be > 0")
    stride = max(1, round(info.fps / detect_hz))
    return set(range(0, info.n_frames, stride))


def _center(points: Iterable[tuple[float, float]]) -> tuple[float, float]:
    point_list = list(points)
    return (
        sum(point[0] for point in point_list) / len(point_list),
        sum(point[1] for point in point_list) / len(point_list),
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_region(
    landmarks: Iterable[object],
    *,
    width: int,
    height: int,
    score: float,
    handedness: str | None = None,
) -> dict[str, Any]:
    """Convert all 21 2D hand landmarks into a tight, auditable circle.

    A circle deliberately errs slightly *inside* the full landmark envelope
    rather than creating a broad arm/torso mask.  EgoBlur only acts when each
    raw face box is almost wholly inside this region, so a nearby real face
    cannot be withheld merely because a hand is adjacent to it.
    """
    raw = list(landmarks)
    if len(raw) != 21:
        raise RuntimeError(f"HandLandmarker returned {len(raw)} landmarks, expected 21")
    normalized = [(float(point.x), float(point.y)) for point in raw]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in normalized):
        raise RuntimeError("HandLandmarker returned non-finite landmark coordinates")
    points = [(x * width, y * height) for x, y in normalized]
    center = _center(points)
    radius = max(8.0, max(_distance(center, point) for point in points) * 1.04)
    xs, ys = zip(*normalized, strict=True)
    edge = min(xs) <= WEARER_EDGE_MARGIN or max(xs) >= 1.0 - WEARER_EDGE_MARGIN
    lower = max(ys) >= WEARER_LOWER_BAND
    large = radius >= max(width, height) * WEARER_MIN_RADIUS_FRAC
    # Either a large lower hand (common while working) or a large hand entering
    # from a side edge can be a wearer candidate.  An ordinary, distant
    # bystander's hand is normally neither large nor camera-near.
    wearer_candidate = large and (edge or lower)
    return {
        "shape": "circle",
        "kind": "hand",
        "center": list(center),
        "radius": radius,
        "landmarks": [[x, y] for x, y in normalized],
        "score": float(score),
        "handedness": handedness,
        "wearer_candidate": wearer_candidate,
    }


def _extract_hands(result: Any, *, width: int, height: int) -> list[dict[str, Any]]:
    hands: list[dict[str, Any]] = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness, strict=True):
        top = handedness[0] if handedness else None
        hands.append(
            hand_region(
                landmarks,
                width=width,
                height=height,
                score=float(top.score) if top is not None else 0.0,
                handedness=str(top.category_name) if top is not None else None,
            )
        )
    return hands


def _track_hands(
    frames: list[dict[str, Any]],
    *,
    stride: int,
    width: int,
    height: int,
) -> None:
    """Assign short-lived physical hand tracks, then mark stable candidates.

    Handedness labels are intentionally ignored for identity: egocentric views
    regularly label both hands alike.  Nearest-centre continuity at the shared
    10 Hz EgoBlur cadence is the only link, and a missed sample may bridge one
    cadence interval but no more.
    """
    active: dict[int, tuple[int, tuple[float, float]]] = {}
    max_distance_px = math.hypot(width, height) * TRACK_MAX_DISTANCE
    next_track_id = 0
    observations_by_track: dict[int, list[tuple[int, bool]]] = {}
    for row in frames:
        frame_idx = int(row["frame_idx"])
        used_tracks: set[int] = set()
        for hand in row["hands"]:
            center = tuple(float(value) for value in hand["center"])
            possible = [
                (_distance(center, previous_center), track_id)
                for track_id, (last_frame, previous_center) in active.items()
                if track_id not in used_tracks
                and frame_idx - last_frame <= stride * (TRACK_MAX_GAP_SAMPLES + 1)
                and _distance(center, previous_center) <= max_distance_px
            ]
            if possible:
                _distance_value, track_id = min(possible)
            else:
                track_id = next_track_id
                next_track_id += 1
            used_tracks.add(track_id)
            active[track_id] = (frame_idx, center)
            hand["track_id"] = track_id
            hand["stable_wearer_candidate"] = False
            observations_by_track.setdefault(track_id, []).append(
                (frame_idx, bool(hand["wearer_candidate"]))
            )

    stable: set[tuple[int, int]] = set()
    for track_id, observations in observations_by_track.items():
        run: list[int] = []
        last_frame: int | None = None
        for frame_idx, wearer_candidate in observations:
            if last_frame is not None and frame_idx - last_frame > stride * (
                TRACK_MAX_GAP_SAMPLES + 1
            ):
                if len(run) >= MIN_STABLE_SAMPLES:
                    stable.update((track_id, value) for value in run)
                run = []
            if wearer_candidate:
                run.append(frame_idx)
            else:
                if len(run) >= MIN_STABLE_SAMPLES:
                    stable.update((track_id, value) for value in run)
                run = []
            last_frame = frame_idx
        if len(run) >= MIN_STABLE_SAMPLES:
            stable.update((track_id, value) for value in run)
    for row in frames:
        for hand in row["hands"]:
            hand["stable_wearer_candidate"] = (
                int(hand["track_id"]),
                int(row["frame_idx"]),
            ) in stable


def _draw_hand(frame_bgr: Any, hand: dict[str, Any]) -> None:
    stable = bool(hand.get("stable_wearer_candidate"))
    candidate = bool(hand.get("wearer_candidate"))
    color = (
        HAND_PREVIEW_STABLE_BGR
        if stable
        else HAND_PREVIEW_PROVISIONAL_BGR
        if candidate
        else HAND_PREVIEW_OTHER_BGR
    )
    points = [
        (round(float(x) * frame_bgr.shape[1]), round(float(y) * frame_bgr.shape[0]))
        for x, y in hand["landmarks"]
    ]
    for first, second in (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
    ):
        cv2.line(frame_bgr, points[first], points[second], color, 2, cv2.LINE_AA)
    center = tuple(round(float(value)) for value in hand["center"])
    cv2.circle(frame_bgr, center, max(1, round(float(hand["radius"]))), color, 2, cv2.LINE_AA)
    label = f"hand {hand['track_id']} {'stable wearer' if stable else 'candidate' if candidate else 'other'}"
    cv2.putText(
        frame_bgr,
        label,
        (center[0] + 4, max(16, center[1] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _write_preview(
    original_video: Path,
    preview_video: Path,
    info: VideoInfo,
    frames: list[dict[str, Any]],
) -> None:
    by_frame = {int(row["frame_idx"]): row["hands"] for row in frames}
    capture = cv2.VideoCapture(str(original_video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open original video {original_video}")
    preview_video.parent.mkdir(parents=True, exist_ok=True)
    partial = preview_video.with_name(f"{preview_video.stem}.partial{preview_video.suffix}")
    partial.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(partial), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create hand preview: {preview_video}")
    latest_hands: list[dict[str, Any]] = []
    source_idx = 0
    completed = False
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if source_idx in by_frame:
                latest_hands = by_frame[source_idx]
            preview_frame = frame_bgr.copy()
            for hand in latest_hands:
                _draw_hand(preview_frame, hand)
            sampled = "sample" if source_idx in by_frame else "held"
            label = f"Hand {sampled} | amber: stable wearer | magenta: provisional | blue: other"
            cv2.rectangle(preview_frame, (0, 0), (min(info.width, 790), 32), (0, 0, 0), -1)
            cv2.putText(
                preview_frame,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(preview_frame)
            source_idx += 1
        if source_idx != info.n_frames:
            raise RuntimeError(
                f"hand preview decode of {original_video} yielded {source_idx} frames, "
                f"expected {info.n_frames}"
            )
        completed = True
    finally:
        writer.release()
        capture.release()
        if not completed:
            partial.unlink(missing_ok=True)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"hand preview was not written: {preview_video}")
    partial.replace(preview_video)


def build_hand_prior(
    *,
    original_video: Path,
    video_id: str,
    output: Path,
    models_dir: Path,
    detect_hz: float = HAND_DETECT_HZ,
    num_hands: int = HAND_NUM_HANDS,
    confidence: float = HAND_MIN_CONFIDENCE,
    preview_video: Path | None = None,
) -> dict[str, Any]:
    """Create a complete, original-only private Hand Landmarker artifact."""
    original_video, output, models_dir = (
        original_video.resolve(),
        output.resolve(),
        models_dir.resolve(),
    )
    preview_video = preview_video.resolve() if preview_video is not None else None
    require_private_path(output, option="--output")
    require_private_path(models_dir, option="--models-dir")
    if preview_video is not None:
        require_private_path(preview_video, option="--preview-video")
    if not original_video.is_file():
        raise FileNotFoundError(f"original video not found: {original_video}")
    if not video_id:
        raise ValueError("video_id must not be empty")
    if num_hands < 1:
        raise ValueError("num_hands must be >= 1")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing hand prior: {output}")
    if preview_video is not None and preview_video.exists():
        raise FileExistsError(f"refusing to overwrite existing hand preview: {preview_video}")

    info = probe(original_video)
    expected = _expected_indices(info, detect_hz)
    model = ensure_model(models_dir)
    detector = _build_detector(model, num_hands=num_hands, confidence=confidence)
    capture = cv2.VideoCapture(str(original_video))
    if not capture.isOpened():
        detector.close()
        raise RuntimeError(f"OpenCV could not open original video {original_video}")
    frames: list[dict[str, Any]] = []
    source_idx = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_bgr.shape[1] != info.width or frame_bgr.shape[0] != info.height:
                raise RuntimeError(
                    f"hand decode dimensions changed to {frame_bgr.shape[1]}x{frame_bgr.shape[0]}; "
                    f"expected {info.width}x{info.height}"
                )
            if source_idx in expected:
                timestamp_ms = round(source_idx / info.fps * 1000)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms
                )
                frames.append(
                    {
                        "frame_idx": source_idx,
                        "timestamp_ms": timestamp_ms,
                        "hands": _extract_hands(result, width=info.width, height=info.height),
                    }
                )
            source_idx += 1
        if source_idx != info.n_frames:
            raise RuntimeError(
                f"hand decode of {original_video} yielded {source_idx} frames, "
                f"expected {info.n_frames}; refusing a partial hand prior"
            )
    finally:
        detector.close()
        capture.release()
    emitted = {int(frame["frame_idx"]) for frame in frames}
    if emitted != expected:
        raise RuntimeError(
            f"hand sampling emitted {len(emitted)}/{len(expected)} frames; "
            f"missing={sorted(expected - emitted)[:8]}"
        )
    stride = max(1, round(info.fps / detect_hz))
    _track_hands(frames, stride=stride, width=info.width, height=info.height)
    if preview_video is not None:
        _write_preview(original_video, preview_video, info, frames)

    artifact: dict[str, Any] = {
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
        "sampling": {"detect_hz": detect_hz, "stride": stride, "n_sampled_frames": len(frames)},
        "model": {
            "name": "mediapipe_hand_landmarker",
            "url": HAND_MODEL_URL,
            "sha256": sha256_file(model),
        },
        "configuration": {
            **ACTIVE_SUPPRESSION_CONFIGURATION,
            "num_hands": num_hands,
            "min_confidence": confidence,
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


def _validate_hand(region: Any) -> None:
    if (
        not isinstance(region, dict)
        or region.get("shape") != "circle"
        or region.get("kind") != "hand"
    ):
        raise ValueError(f"invalid hand-prior region: {region!r}")
    try:
        radius = float(region["radius"])
        track_id = int(region["track_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid hand-prior radius or track id: {region!r}") from exc
    if not math.isfinite(radius) or radius <= 0 or track_id < 0:
        raise ValueError(f"invalid hand-prior radius or track id: {region!r}")
    center, landmarks = region.get("center"), region.get("landmarks")
    if (
        not isinstance(center, list)
        or len(center) != 2
        or not isinstance(landmarks, list)
        or len(landmarks) != 21
    ):
        raise ValueError(f"invalid hand-prior geometry: {region!r}")
    try:
        values = [*center, *(value for point in landmarks for value in point)]
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid hand-prior geometry: {region!r}") from exc
    if not all(isinstance(point, list) and len(point) == 2 for point in landmarks):
        raise ValueError(f"invalid hand-prior landmarks: {region!r}")
    if not isinstance(region.get("wearer_candidate"), bool) or not isinstance(
        region.get("stable_wearer_candidate"), bool
    ):
        raise ValueError(f"invalid hand-prior wearer state: {region!r}")
    if region["stable_wearer_candidate"] and not region["wearer_candidate"]:
        raise ValueError(
            "invalid hand-prior wearer state: a stable wearer hand must also be a wearer candidate"
        )


def load_hand_prior(
    path: Path,
    *,
    source_sha256: str,
    width: int,
    height: int,
    fps: float,
    n_frames: int,
    detect_hz: float,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    """Validate a private Hand Landmarker prior against one source timeline."""
    path = path.resolve()
    require_private_path(path, option="--hand-prior")
    if not path.is_file():
        raise FileNotFoundError(f"hand prior not found: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid hand prior JSON {path}: {exc}") from exc
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("artifact_type") != ARTIFACT_TYPE
    ):
        raise ValueError(f"{path} is not a supported hand-prior artifact")
    source, sampling, model, configuration = (
        artifact.get("source"),
        artifact.get("sampling"),
        artifact.get("model"),
        artifact.get("configuration"),
    )
    if (
        not isinstance(source, dict)
        or not isinstance(sampling, dict)
        or not isinstance(model, dict)
        or not isinstance(configuration, dict)
    ):
        raise ValueError(f"{path} lacks hand-prior source/sampling/model provenance")
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
    expected_model = {
        "name": "mediapipe_hand_landmarker",
        "url": HAND_MODEL_URL,
        "sha256": HAND_MODEL_SHA256,
    }
    if model != expected_model:
        mismatches["model"] = {"expected": expected_model, "artifact": model}
    if configuration != ACTIVE_SUPPRESSION_CONFIGURATION:
        mismatches["configuration"] = {
            "expected": ACTIVE_SUPPRESSION_CONFIGURATION,
            "artifact": configuration,
        }
    if mismatches:
        raise ValueError(f"hand-prior provenance does not match this clip: {mismatches}")
    stride = max(1, round(fps / detect_hz))
    expected_indices = set(range(0, n_frames, stride))
    rows = artifact.get("frames")
    if not isinstance(rows, list):
        raise ValueError(f"{path} frames must be a list")
    by_frame: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} has a non-object frame entry")
        try:
            frame_idx, hands = int(row["frame_idx"]), row["hands"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} has an invalid frame entry") from exc
        if (
            frame_idx not in expected_indices
            or frame_idx in by_frame
            or not isinstance(hands, list)
        ):
            raise ValueError(f"{path} has an unexpected, duplicate, or invalid frame {frame_idx}")
        for hand in hands:
            _validate_hand(hand)
        by_frame[frame_idx] = {
            "all": hands,
            "wearer": [hand for hand in hands if hand["wearer_candidate"]],
            "stable_wearer": [hand for hand in hands if hand["stable_wearer_candidate"]],
        }
    if set(by_frame) != expected_indices:
        raise ValueError(
            f"hand prior is incomplete: sampled_frames={len(by_frame)}/{len(expected_indices)}, "
            f"missing={sorted(expected_indices - set(by_frame))[:8]}"
        )
    return by_frame
