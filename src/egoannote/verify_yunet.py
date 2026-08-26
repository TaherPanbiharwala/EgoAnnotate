"""CPU-only, post-redaction YuNet verification.

This verifier deliberately never opens the original video.  It rebuilds
EgoBlur's fill map from the private detector checkpoint, detects faces in the
*redacted* video with an independent YuNet model, and reports only faces that
are not covered by that fill map.  Its JSON report is private review material;
it is not a Hugging Face artifact and is not an approval of the complete
redaction audit by itself.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from . import hand_prior, pose_prior

CANDIDATE_TRACK_IOU = 0.20
CANDIDATE_MAX_GAP_SAMPLES = 2
CANDIDATE_LOWER_PRIORITY_OVERLAP = pose_prior.LIMB_OVERLAP_LOWER_PRIORITY
CANDIDATE_CONTACT_SHEET_SAMPLES = 6
# YuNet frequently draws a loose face box around fingers, palms, and wrists.
# This filter is intentionally more permissive than EgoBlur's raw-track
# policy: both amber and pink wearer-hand evidence can suppress an uncovered
# YuNet review observation, and the private circle is enlarged only here.
YUNET_HAND_RADIUS_SCALE = 1.5
YUNET_HAND_MIN_OVERLAP = 0.10
_DECISIONS = frozenset({"confirmed_face", "false_positive", "uncertain"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _expanded_hand_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return private hand circles enlarged for the noisy YuNet audit only."""
    expanded: list[dict[str, Any]] = []
    for region in regions:
        if region.get("shape") != "circle" or region.get("kind") != "hand":
            raise ValueError(f"unsupported Hand Landmarker region in YuNet filter: {region!r}")
        enlarged = dict(region)
        enlarged["radius"] = float(region["radius"]) * YUNET_HAND_RADIUS_SCALE
        expanded.append(enlarged)
    return expanded


def filter_yunet_wearer_hand_noise(
    detections: list[dict[str, Any]],
    *,
    hand_frames: dict[int, dict[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop likely wearer-hand false positives from the YuNet review queue.

    The raw detector result is retained in the private report, including why
    it was filtered.  Covered detections are left alone because they were
    never candidate residual faces.  Amber gets first precedence; pink is
    deliberately accepted too, so YuNet can stay useful for non-hand edge
    cases instead of overwhelming review with limb noise.
    """
    actionable: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for detection in detections:
        if bool(detection["covered"]):
            actionable.append(detection)
            continue
        frame_idx = int(detection["frame_idx"])
        frame = hand_frames.get(frame_idx)
        if frame is None:
            raise ValueError(f"hand prior lacks YuNet sampled frame {frame_idx}")
        box = tuple(float(value) for value in detection["box"])
        amber_overlap = pose_prior.limb_overlap_fraction(
            box, _expanded_hand_regions(frame["stable_wearer"])
        )
        pink_overlap = pose_prior.limb_overlap_fraction(
            box, _expanded_hand_regions(frame["provisional_wearer"])
        )
        if amber_overlap is not None and amber_overlap >= YUNET_HAND_MIN_OVERLAP:
            suppressed.append(
                {
                    **detection,
                    "hand_state": "amber",
                    "expanded_hand_overlap": amber_overlap,
                }
            )
            continue
        if pink_overlap is not None and pink_overlap >= YUNET_HAND_MIN_OVERLAP:
            suppressed.append(
                {
                    **detection,
                    "hand_state": "pink",
                    "expanded_hand_overlap": pink_overlap,
                }
            )
            continue
        actionable.append(detection)
    return actionable, suppressed


def _load_job_module(job_script: Path) -> ModuleType:
    """Load the redaction helpers without starting the GPU redaction job."""
    if not job_script.is_file():
        raise FileNotFoundError(f"EgoBlur job script not found: {job_script}")
    spec = importlib.util.spec_from_file_location("egoannote_egoblur_helpers", job_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load EgoBlur job script {job_script}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules during
    # class creation. Register before execution, exactly as normal import
    # machinery does, or a valid job script can fail while defining ClipInfo.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    required = ("Detection", "build_tracks", "tracks_to_fill_map", "check_yunet")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"{job_script} does not expose the required EgoBlur helpers: {', '.join(missing)}"
        )
    return module


def _checkpoint_detections(
    checkpoint_dir: Path, manifest: dict[str, Any], job: ModuleType
) -> list[Any]:
    """Load a complete, configuration-compatible detector checkpoint.

    A checkpoint is the only private input needed to reconstruct the fill
    geometry.  Validating every expected sampled frame prevents a truncated
    checkpoint from turning into a misleading "no uncovered faces" result.
    """
    try:
        clip_id = str(manifest["clip_id"])
        source = manifest["source"]
        egoblur = manifest["egoblur"]
        n_frames = int(source["n_frames"])
        fps = float(source["fps"])
        detect_hz = float(egoblur["detect_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest lacks clip_id/source/egoblur timing metadata") from exc
    if n_frames <= 0 or fps <= 0 or detect_hz <= 0:
        raise ValueError("manifest has unusable frame-count, fps, or detect_hz")

    path = checkpoint_dir / f"{clip_id}.detections.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"detector checkpoint not found: {path}. Download the complete private "
            "checkpoints directory alongside the manifest."
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"checkpoint {path} is empty")
    try:
        fingerprint = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"checkpoint {path} has an invalid fingerprint line") from exc
    expected_fingerprint = {
        "_fingerprint": 1,
        "gen": egoblur.get("gen"),
        "lp_present": bool(egoblur.get("lp_checked")),
        "sweep_threshold": egoblur.get("sweep_threshold"),
        "nms_iou": egoblur.get("nms_iou"),
        "detect_hz": egoblur.get("detect_hz"),
        "gen2_resize_px": egoblur.get("gen2_resize_px"),
    }
    mismatches = {
        key: {"manifest": value, "checkpoint": fingerprint.get(key)}
        for key, value in expected_fingerprint.items()
        if fingerprint.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "checkpoint fingerprint does not match the manifest; refusing to reconstruct "
            f"a different run's fill map: {mismatches}"
        )

    stride = max(1, round(fps / detect_hz))
    expected_frames = set(range(0, n_frames, stride))
    seen: set[int] = set()
    detections: list[Any] = []
    for line_no, line in enumerate(lines[1:], 2):
        try:
            row = json.loads(line)
            frame_idx = int(row["frame_idx"])
            raw_detections = row["detections"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint row {line_no} in {path}") from exc
        if frame_idx not in expected_frames:
            raise ValueError(f"checkpoint row {line_no} has unexpected frame_idx={frame_idx}")
        if frame_idx in seen:
            raise ValueError(f"checkpoint {path} has duplicate frame_idx={frame_idx}")
        if not isinstance(raw_detections, list):
            raise ValueError(f"checkpoint row {line_no} detections must be a list")
        seen.add(frame_idx)
        for raw in raw_detections:
            try:
                cls = str(raw["cls"])
                box = tuple(float(value) for value in raw["box"])
                score = float(raw["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid detection in checkpoint row {line_no}") from exc
            if cls not in {"face", "lp"} or len(box) != 4:
                raise ValueError(f"invalid detection in checkpoint row {line_no}: {raw!r}")
            detections.append(job.Detection(frame_idx=frame_idx, cls=cls, box=box, score=score))
    if seen != expected_frames:
        missing = sorted(expected_frames - seen)
        extra = sorted(seen - expected_frames)
        raise ValueError(
            f"checkpoint is incomplete: sampled_frames={len(seen)}/{len(expected_frames)}, "
            f"missing={missing[:8]}, unexpected={extra[:8]}"
        )
    return detections


def _rebuild_fill_map(manifest: dict[str, Any], detections: list[Any], job: ModuleType) -> dict:
    """Replay the exact post-detection tracking configuration from the manifest."""
    try:
        source = manifest["source"]
        egoblur = manifest["egoblur"]
        width, height = int(source["width"]), int(source["height"])
        fps = float(source["fps"])
        operating = {
            "face": float(egoblur["face_threshold"]),
            "lp": float(egoblur["lp_threshold"]),
        }
        detect_hz = float(egoblur["detect_hz"])
        min_box_px = int(egoblur["min_box_px"])
        dilate_scale = float(egoblur["dilate_scale"])
        motion_margin_px = int(egoblur["motion_margin_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest lacks required EgoBlur tracking settings") from exc

    hold_frames = int(egoblur.get("hold_frames", 0))
    back_hold_frames = int(egoblur.get("back_hold_frames", hold_frames))
    if hold_frames <= 0 or back_hold_frames <= 0:
        raise ValueError("manifest has unresolved/non-positive hold-frame settings")
    confident = [det for det in detections if det.score > operating[det.cls]]
    continue_threshold = float(egoblur.get("continue_threshold") or 0.0)
    if continue_threshold:
        track_input = detections
        start_threshold = operating
        continuation_threshold = dict.fromkeys(operating, continue_threshold)
    else:
        track_input = confident
        start_threshold = None
        continuation_threshold = None
    low_absorbed: list[Any] = []
    unconfirmed: list[Any] = []
    tracks = job.build_tracks(
        track_input,
        min_box_px,
        iou_thresh=job.TRACK_IOU_DEFAULT,
        hold_frames=hold_frames,
        back_hold_frames=back_hold_frames,
        start_thresh=start_threshold,
        cont_thresh=continuation_threshold,
        low_absorbed=low_absorbed,
        stride=max(1, round(fps / detect_hz)),
        min_confident_hits=int(egoblur.get("min_track_confirmations", 1)),
        unconfirmed=unconfirmed,
    )
    return job.tracks_to_fill_map(tracks, width, height, dilate_scale, motion_margin_px)


def _ffprobe_color_range(ffmpeg: str, video: Path) -> str:
    sibling = Path(ffmpeg).with_name("ffprobe")
    ffprobe = str(sibling) if sibling.is_file() else shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required beside --ffmpeg or on PATH")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_range",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe failed on {video}: {result.stderr.strip()}")
    try:
        return str(json.loads(result.stdout)["streams"][0].get("color_range") or "tv")
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ffprobe returned no usable video stream for {video}") from exc


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2), encoding="utf-8")
    partial.replace(path)


def _require_private_path(path: Path, option: str) -> None:
    pose_prior.require_private_path(path, option=option)


def _require_private_preview_path(path: Path) -> None:
    _require_private_path(path, "--preview-video")


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _candidate_observations(
    detections: list[dict[str, Any]],
    *,
    pose_frames: dict[int, dict[str, list[dict[str, Any]]]] | None,
) -> list[dict[str, Any]]:
    """Keep every uncovered YuNet hit and attach a soft wearer-limb score."""
    observations: list[dict[str, Any]] = []
    for detection in detections:
        if bool(detection["covered"]):
            continue
        box = tuple(float(value) for value in detection["box"])
        frame_idx = int(detection["frame_idx"])
        row = {
            "frame_idx": frame_idx,
            "box": list(box),
            "score": float(detection["score"]),
        }
        if pose_frames is None:
            row["all_limb_overlap"] = None
            row["wearer_limb_overlap"] = None
        else:
            frame = pose_frames.get(frame_idx)
            if frame is None:
                raise ValueError(f"pose prior lacks YuNet sampled frame {frame_idx}")
            row["all_limb_overlap"] = pose_prior.limb_overlap_fraction(box, frame["all"])
            row["wearer_limb_overlap"] = pose_prior.limb_overlap_fraction(box, frame["wearer"])
        observations.append(row)
    return sorted(observations, key=lambda row: (int(row["frame_idx"]), -float(row["score"])))


def build_candidate_tracks(
    detections: list[dict[str, Any]],
    *,
    pose_frames: dict[int, dict[str, list[dict[str, Any]]]] | None,
    fps: float,
    detect_hz: float,
) -> list[dict[str, Any]]:
    """Associate uncovered YuNet hits into review units, never discard them.

    Association happens only to reduce human review repetition. The raw
    `n_yunet_uncovered` count remains untouched, and a pose-overlapping track
    is merely rendered later in the lower-priority amber queue.
    """
    if fps <= 0 or detect_hz <= 0:
        raise ValueError("fps and detect_hz must be positive for candidate tracking")
    stride = max(1, round(fps / detect_hz))
    max_gap = stride * CANDIDATE_MAX_GAP_SAMPLES
    observations = _candidate_observations(detections, pose_frames=pose_frames)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in observations:
        by_frame.setdefault(int(row["frame_idx"]), []).append(row)
    active: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    for frame_idx, frame_rows in sorted(by_frame.items()):
        active = [track for track in active if frame_idx - int(track["last_frame_idx"]) <= max_gap]
        used: set[int] = set()
        for row in frame_rows:
            box = tuple(float(value) for value in row["box"])
            best_index, best_iou = -1, 0.0
            for index, track in enumerate(active):
                if index in used:
                    continue
                overlap = _iou(box, tuple(track["last_box"]))
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_index >= 0 and best_iou >= CANDIDATE_TRACK_IOU:
                track = active[best_index]
                used.add(best_index)
            else:
                track = {"last_frame_idx": frame_idx, "last_box": list(box), "observations": []}
                active.append(track)
                tracks.append(track)
                used.add(len(active) - 1)
            track["observations"].append(row)
            track["last_frame_idx"] = frame_idx
            track["last_box"] = list(box)

    candidates: list[dict[str, Any]] = []
    for index, track in enumerate(tracks, 1):
        observations = track["observations"]
        scores = [float(row["score"]) for row in observations]
        wearer = [
            float(row["wearer_limb_overlap"])
            for row in observations
            if row["wearer_limb_overlap"] is not None
        ]
        all_limb = [
            float(row["all_limb_overlap"])
            for row in observations
            if row["all_limb_overlap"] is not None
        ]
        wearer_hits = sum(value >= CANDIDATE_LOWER_PRIORITY_OVERLAP for value in wearer)
        priority = (
            "lower_wearer_limb_overlap"
            if wearer and wearer_hits * 2 >= len(wearer)
            else "normal"
        )
        first = int(observations[0]["frame_idx"])
        last = int(observations[-1]["frame_idx"])
        candidates.append(
            {
                "candidate_id": f"yunet-{index:05d}",
                "priority": priority,
                "first_frame_idx": first,
                "last_frame_idx": last,
                "first_timestamp_s": first / fps,
                "last_timestamp_s": last / fps,
                "duration_s": (last - first) / fps,
                "n_observations": len(observations),
                "score_max": max(scores),
                "score_median": statistics.median(scores),
                "all_limb_overlap_max": max(all_limb) if all_limb else None,
                "all_limb_overlap_mean": sum(all_limb) / len(all_limb) if all_limb else None,
                "wearer_limb_overlap_max": max(wearer) if wearer else None,
                "wearer_limb_overlap_mean": sum(wearer) / len(wearer) if wearer else None,
                "wearer_limb_overlap_observations": len(wearer),
                "wearer_limb_overlap_hits": wearer_hits,
                "observations": observations,
            }
        )
    # The reviewer sees non-wearer candidates first, while IDs/timestamps
    # remain stable and every lower-priority candidate remains present.
    return sorted(candidates, key=lambda row: (row["priority"] != "normal", row["first_frame_idx"]))


def _candidate_color(candidate: dict[str, Any]) -> tuple[int, int, int]:
    return (0, 165, 255) if candidate["priority"] != "normal" else (0, 0, 255)


def _write_preview_video(
    *,
    redacted_video: Path,
    output: Path,
    detections: list[dict[str, Any]],
    width: int,
    height: int,
    source_fps: float,
    n_source_frames: int,
    detect_hz: float,
) -> int:
    """Write a sampled, redacted-only overlay for local visual review.

    Green boxes were already covered by EgoBlur's fill map; red boxes are
    independent YuNet hits outside it.  The preview has no audio, is encoded
    from the redacted input only, and must stay in a private/DO-NOT-SHIP tree.
    """
    import cv2

    _require_private_preview_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
    temporary.unlink(missing_ok=True)
    stride = max(1, round(source_fps / detect_hz))
    expected_indices = set(range(0, n_source_frames, stride))
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        frame_idx = int(detection["frame_idx"])
        if frame_idx not in expected_indices:
            raise ValueError(f"YuNet returned an unexpected sampled frame {frame_idx}")
        by_frame.setdefault(frame_idx, []).append(detection)

    capture = cv2.VideoCapture(str(redacted_video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open redacted video {redacted_video}")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps / stride,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"OpenCV could not create review preview {temporary}; mp4v support is required"
        )
    source_idx = 0
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError(
                    f"preview decode dimensions changed to {frame.shape[1]}x{frame.shape[0]}; "
                    f"expected {width}x{height}"
                )
            if source_idx in expected_indices:
                for detection in by_frame.get(source_idx, []):
                    x1, y1, x2, y2 = (round(float(value)) for value in detection["box"])
                    covered = bool(detection["covered"])
                    color = (0, 180, 0) if covered else (0, 0, 255)  # BGR green/red
                    label = f"YuNet {'covered' if covered else 'UNCOVERED'} {detection['score']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                writer.write(frame)
                written += 1
            source_idx += 1
        if source_idx != n_source_frames:
            raise RuntimeError(
                f"preview decode yielded {source_idx} frames, expected {n_source_frames}; "
                "refusing to write a partial review video"
            )
        if written != len(expected_indices):
            raise RuntimeError(
                f"preview wrote {written}/{len(expected_indices)} expected sampled frames"
            )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        writer.release()
    temporary.replace(output)
    return written


def _selected_candidate_observations(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    observations = candidate["observations"]
    if len(observations) <= CANDIDATE_CONTACT_SHEET_SAMPLES:
        return observations
    indices = {
        round(index * (len(observations) - 1) / (CANDIDATE_CONTACT_SHEET_SAMPLES - 1))
        for index in range(CANDIDATE_CONTACT_SHEET_SAMPLES)
    }
    return [observations[index] for index in sorted(indices)]


def _candidate_tile(frame, candidate: dict[str, Any], observation: dict[str, Any]):
    """Return a fixed-size contextual crop from the already-redacted video."""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = (float(value) for value in observation["box"])
    height, width = frame.shape[:2]
    side = max(x2 - x1, y2 - y1, 80.0)
    padding = side * 0.75
    left, top = max(0, round(x1 - padding)), max(0, round(y1 - padding))
    right, bottom = min(width, round(x2 + padding)), min(height, round(y2 + padding))
    if right <= left or bottom <= top:
        raise RuntimeError(f"candidate {candidate['candidate_id']} crop is empty")
    crop = frame[top:bottom, left:right].copy()
    color = _candidate_color(candidate)
    cv2.rectangle(crop, (round(x1 - left), round(y1 - top)), (round(x2 - left), round(y2 - top)), color, 2)
    label = (
        f"{candidate['candidate_id']} {candidate['priority']} "
        f"t={float(observation['frame_idx']):.0f} score={float(observation['score']):.2f}"
    )
    cv2.putText(crop, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    target_w, target_h = 320, 180
    scale = min(target_w / crop.shape[1], target_h / crop.shape[0])
    resized = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)))
    tile = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - resized.shape[0]) // 2
    x_offset = (target_w - resized.shape[1]) // 2
    tile[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return tile


def _write_candidate_contact_sheets(
    *,
    redacted_video: Path,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    width: int,
    height: int,
    n_source_frames: int,
) -> int:
    """Write one compact redacted-only contact sheet per candidate track."""
    import cv2
    import numpy as np

    _require_private_path(output_dir, "--candidate-contact-sheet-dir")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate contact sheets: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.partial-", dir=output_dir.parent))
    selected: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    expected_tiles: dict[str, int] = {}
    for candidate in candidates:
        picks = _selected_candidate_observations(candidate)
        expected_tiles[candidate["candidate_id"]] = len(picks)
        for observation in picks:
            selected.setdefault(int(observation["frame_idx"]), []).append((candidate, observation))
    tiles: dict[str, list[Any]] = {candidate["candidate_id"]: [] for candidate in candidates}
    capture = cv2.VideoCapture(str(redacted_video))
    if not capture.isOpened():
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError(f"OpenCV could not open redacted video {redacted_video}")
    source_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError(
                    f"candidate review decode dimensions changed to {frame.shape[1]}x{frame.shape[0]}; "
                    f"expected {width}x{height}"
                )
            for candidate, observation in selected.get(source_idx, []):
                tiles[candidate["candidate_id"]].append(_candidate_tile(frame, candidate, observation))
            source_idx += 1
        if source_idx != n_source_frames:
            raise RuntimeError(
                f"candidate review decode yielded {source_idx} frames, expected {n_source_frames}; "
                "refusing to write partial contact sheets"
            )
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            candidate_tiles = tiles[candidate_id]
            if len(candidate_tiles) != expected_tiles[candidate_id]:
                raise RuntimeError(
                    f"candidate contact sheet {candidate_id} has {len(candidate_tiles)}/"
                    f"{expected_tiles[candidate_id]} selected frames"
                )
            columns = min(3, len(candidate_tiles))
            rows = (len(candidate_tiles) + columns - 1) // columns
            canvas = np.zeros((rows * 180, columns * 320, 3), dtype=np.uint8)
            for index, tile in enumerate(candidate_tiles):
                row, column = divmod(index, columns)
                canvas[row * 180:(row + 1) * 180, column * 320:(column + 1) * 320] = tile
            target = temporary / f"{candidate_id}.jpg"
            if not cv2.imwrite(str(target), canvas):
                raise RuntimeError(f"could not write candidate contact sheet {target}")
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        capture.release()
    return len(candidates)


def create_decision_template(report: Path, output: Path) -> dict[str, Any]:
    """Create a private, hash-bound disposition file for every candidate."""
    _require_private_path(report, "--report")
    _require_private_path(output, "--output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review decisions: {output}")
    data = _load_json(report)
    candidates = _report_candidates(data)
    template = {
        "schema_version": 1,
        "review_type": "yunet_candidate_decisions",
        "review_report_sha256": _sha256_file(report),
        "clip_id": data.get("input", {}).get("clip_id"),
        "decisions": {candidate["candidate_id"]: "uncertain" for candidate in candidates},
    }
    _atomic_json(output, template)
    return template


def decisions_to_forced_boxes(report: Path, decisions: Path, output: Path) -> dict[str, Any]:
    """Convert only human-confirmed candidate tracks into EgoBlur's input map."""
    for path, option in ((report, "--report"), (decisions, "--decisions"), (output, "--output")):
        _require_private_path(path, option)
    if not decisions.is_file():
        raise FileNotFoundError(f"review decisions not found: {decisions}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite forced boxes: {output}")
    report_data = _load_json(report)
    decision_data = _load_json(decisions)
    if decision_data.get("review_type") != "yunet_candidate_decisions":
        raise ValueError(f"{decisions} is not a YuNet candidate decision file")
    if decision_data.get("review_report_sha256") != _sha256_file(report):
        raise ValueError("review decisions do not match this exact YuNet report")
    candidates = _report_candidates(report_data)
    expected_ids = {candidate["candidate_id"] for candidate in candidates}
    raw_decisions = decision_data.get("decisions")
    if not isinstance(raw_decisions, dict) or set(raw_decisions) != expected_ids:
        raise ValueError("review decisions must contain one disposition for every candidate ID")
    invalid = {candidate_id: value for candidate_id, value in raw_decisions.items() if value not in _DECISIONS}
    if invalid:
        raise ValueError(f"invalid candidate disposition(s): {invalid}")
    clip_id = str(report_data.get("input", {}).get("clip_id") or "")
    if not clip_id:
        raise ValueError("YuNet report lacks input.clip_id")
    boxes = [
        {"frame_idx": int(observation["frame_idx"]), "box": observation["box"]}
        for candidate in candidates
        if raw_decisions[candidate["candidate_id"]] == "confirmed_face"
        for observation in candidate["observations"]
    ]
    if not boxes:
        raise ValueError("no confirmed_face candidates; no corrective EgoBlur run should be started")
    _atomic_json(output, {clip_id: boxes})
    return {
        "clip_id": clip_id,
        "n_forced_boxes": len(boxes),
        "n_confirmed_candidates": sum(
            value == "confirmed_face" for value in raw_decisions.values()
        ),
        "n_uncertain_candidates": sum(value == "uncertain" for value in raw_decisions.values()),
    }


def _report_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    yunet = data.get("yunet")
    candidates = yunet.get("candidates") if isinstance(yunet, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("YuNet report lacks full temporal candidates; rerun verify-yunet")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
            raise ValueError("YuNet report contains an invalid candidate")
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen or not isinstance(candidate.get("observations"), list):
            raise ValueError("YuNet report contains duplicate or incomplete candidates")
        seen.add(candidate_id)
    return candidates


def verify_yunet(
    *,
    redacted_video: Path,
    blur_manifest: Path,
    checkpoint_dir: Path,
    yunet_model: Path,
    job_script: Path,
    ffmpeg: str,
    report: Path,
    preview_video: Path | None = None,
    pose_prior_path: Path | None = None,
    hand_prior_path: Path | None = None,
    candidate_contact_sheet_dir: Path | None = None,
) -> dict[str, Any]:
    """Run and persist a fail-closed, post-redaction YuNet review report."""
    for name, path in (("redacted video", redacted_video), ("manifest", blur_manifest),
                       ("YuNet model", yunet_model)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    if report.resolve() in {redacted_video.resolve(), blur_manifest.resolve()}:
        raise ValueError("--report must not overwrite the redacted video or EgoBlur manifest")
    _require_private_path(report, "--report")
    if preview_video is not None:
        _require_private_preview_path(preview_video)
        if preview_video.resolve() in {redacted_video.resolve(), blur_manifest.resolve()}:
            raise ValueError("--preview-video must not overwrite the redacted video or EgoBlur manifest")
    if candidate_contact_sheet_dir is not None:
        _require_private_path(candidate_contact_sheet_dir, "--candidate-contact-sheet-dir")

    manifest = _load_json(blur_manifest)
    expected_video_hash = str((manifest.get("output") or {}).get("sha256") or "")
    actual_video_hash = _sha256_file(redacted_video)
    if len(expected_video_hash) != 64 or actual_video_hash != expected_video_hash:
        raise ValueError(
            "redacted video SHA-256 does not match manifest.output.sha256; refusing to "
            "compare a possibly different video to this run's fill map"
        )

    job = _load_job_module(job_script)
    detections = _checkpoint_detections(checkpoint_dir, manifest, job)
    fill_map = _rebuild_fill_map(manifest, detections, job)
    source = manifest["source"]
    pose_frames = None
    pose_artifact_sha256 = None
    if pose_prior_path is not None:
        pose_frames = pose_prior.load_pose_prior(
            pose_prior_path,
            source_sha256=str(source.get("sha256") or ""),
            width=int(source["width"]),
            height=int(source["height"]),
            fps=float(source["fps"]),
            n_frames=int(source["n_frames"]),
            detect_hz=float(manifest["egoblur"]["detect_hz"]),
        )
        pose_artifact_sha256 = _sha256_file(pose_prior_path)
    hand_frames = None
    hand_artifact_sha256 = None
    if hand_prior_path is not None:
        hand_frames = hand_prior.load_hand_prior(
            hand_prior_path,
            source_sha256=str(source.get("sha256") or ""),
            width=int(source["width"]),
            height=int(source["height"]),
            fps=float(source["fps"]),
            n_frames=int(source["n_frames"]),
            detect_hz=float(manifest["egoblur"]["detect_hz"]),
        )
        hand_artifact_sha256 = _sha256_file(hand_prior_path)
    config = SimpleNamespace(
        yunet_model=yunet_model,
        nms_iou=float(manifest["egoblur"]["nms_iou"]),
        output_dir=report.parent,
    )
    full_range = _ffprobe_color_range(ffmpeg, redacted_video) == "pc"
    result = job.check_yunet(
        config,
        ffmpeg,
        redacted_video,
        int(source["width"]),
        int(source["height"]),
        fill_map,
        float(manifest["egoblur"]["detect_hz"]),
        float(source["fps"]),
        int(source["n_frames"]),
        full_range=full_range,
        # Candidate grouping is private and needs the full untruncated set,
        # even when the optional full-overlay diagnostic is not requested.
        record_detections=True,
    )
    all_detections = result.pop("yunet_detections", None)
    if not isinstance(all_detections, list):
        raise RuntimeError("YuNet helper did not return detections for candidate review")
    candidate_detections = all_detections
    hand_suppressed: list[dict[str, Any]] = []
    if hand_frames is not None:
        candidate_detections, hand_suppressed = filter_yunet_wearer_hand_noise(
            all_detections, hand_frames=hand_frames
        )
    candidates = build_candidate_tracks(
        candidate_detections,
        pose_frames=pose_frames,
        fps=float(source["fps"]),
        detect_hz=float(manifest["egoblur"]["detect_hz"]),
    )
    result["candidates"] = candidates
    raw_uncovered = int(result.get("n_yunet_uncovered", 0))
    result["n_yunet_actionable_uncovered"] = sum(
        not bool(row["covered"]) for row in candidate_detections
    )
    result["n_yunet_hand_suppressed"] = len(hand_suppressed)
    result["yunet_hand_suppressed"] = hand_suppressed[:200]
    result["yunet_hand_suppressed_truncated"] = max(0, len(hand_suppressed) - 200)
    preview_frames = None
    if preview_video is not None:
        preview_frames = _write_preview_video(
            redacted_video=redacted_video,
            output=preview_video,
            detections=all_detections,
            width=int(source["width"]),
            height=int(source["height"]),
            source_fps=float(source["fps"]),
            n_source_frames=int(source["n_frames"]),
            detect_hz=float(manifest["egoblur"]["detect_hz"]),
        )
    contact_sheets = None
    if candidate_contact_sheet_dir is not None:
        contact_sheets = _write_candidate_contact_sheets(
            redacted_video=redacted_video,
            output_dir=candidate_contact_sheet_dir,
            candidates=candidates,
            width=int(source["width"]),
            height=int(source["height"]),
            n_source_frames=int(source["n_frames"]),
        )
    actionable_uncovered = int(result["n_yunet_actionable_uncovered"])
    report_data = {
        "schema_version": 1,
        "review_type": "post_redaction_yunet",
        "review_status": (
            "NEEDS_REVIEW"
            if actionable_uncovered > 0
            else "PASS_NO_UNCOVERED_YUNET"
            if raw_uncovered == 0
            else "PASS_NO_ACTIONABLE_UNCOVERED_YUNET"
        ),
        "approval_scope": (
            "YuNet-only independent post-redaction check. This report does not approve "
            "the complete EgoBlur audit, replace human review, or permit publication by itself."
        ),
        "input": {
            "clip_id": manifest.get("clip_id"),
            "redacted_sha256": actual_video_hash,
            "egoblur_manifest_sha256": _sha256_file(blur_manifest),
            "checkpoint": str((checkpoint_dir / f"{manifest['clip_id']}.detections.jsonl").name),
            "job_script_sha256": _sha256_file(job_script),
            "yunet_model_sha256": _sha256_file(yunet_model),
        },
        "reconstruction": {
            "fill_frames": len(fill_map),
            "checkpoint_detections": len(detections),
            "detect_hz": manifest["egoblur"]["detect_hz"],
            "full_range": full_range,
        },
        "pose_prior": (
            None
            if pose_prior_path is None
            else {
                "path": pose_prior_path.name,
                "sha256": pose_artifact_sha256,
                "priority_policy": "wearer_limb_overlap_only",
            }
        ),
        "hand_prior": (
            None
            if hand_prior_path is None
            else {
                "path": hand_prior_path.name,
                "sha256": hand_artifact_sha256,
                "filter_policy": {
                    "amber_and_pink_suppress_review_noise": True,
                    "radius_scale": YUNET_HAND_RADIUS_SCALE,
                    "min_box_overlap": YUNET_HAND_MIN_OVERLAP,
                    "raw_uncovered_detections": raw_uncovered,
                    "actionable_uncovered_detections": actionable_uncovered,
                },
            }
        ),
        "yunet": result,
    }
    if preview_video is not None:
        report_data["preview"] = {
            "path": preview_video.name,
            "fps": float(manifest["egoblur"]["detect_hz"]),
            "frames": preview_frames,
            "warning": "Local private review video only; DO NOT SHIP or upload.",
        }
    if candidate_contact_sheet_dir is not None:
        report_data["candidate_contact_sheets"] = {
            "path": candidate_contact_sheet_dir.name,
            "count": contact_sheets,
            "warning": "Redacted-only local review material; DO NOT SHIP or upload.",
        }
    _atomic_json(report, report_data)
    return report_data
