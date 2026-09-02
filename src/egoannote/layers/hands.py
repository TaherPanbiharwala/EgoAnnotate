"""MediaPipe HandLandmarker runner — local CPU, free tier, 30 fps default.

FIX (plan S6/S7): v1's `_pick_hand` iterated detections and returned the
FIRST one whose top handedness label matched what was asked for. In
egocentric footage MediaPipe frequently labels BOTH detected hands "Right"
(the classifier is trained mostly on exocentric/frontal views) — so
`_pick_hand(result, "Left")` returned None even when a second hand was
genuinely visible, and `hands_present` under-counted. Measured on the real
v1 data: 757 slot on/off transitions in 285.8s (2.65/s) and 26 frames with
the exact simultaneous-swap signature (a wrist "teleporting" 122% of frame
width in one 67ms frame) — direct evidence of this failure mode.

This version assigns BOTH detections jointly, using temporal continuity
(nearest wrist position to the previous frame's left/right slots) as the
primary signal and the handedness label only as a tie-breaker when there is
no prior frame to anchor to. A physical-plausibility bound
(config.MAX_PLAUSIBLE_DELTA_PER_FRAME) is applied on top: a slot whose
assigned wrist moved further than a hand physically can in one frame is
given a fresh track_id, so the discontinuity is visible downstream rather
than being recorded as smooth motion.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import cv2
import mediapipe as mp

from .. import config
from ..media.probe import VideoInfo, probe
from ..schema import HandFrame

log = logging.getLogger(__name__)

_DETECTOR_VERSION = f"mediapipe-{mp.__version__}-hand_landmarker.task"

Landmarks = list[float]  # 63 floats: 21 x (x, y, z)


class HandDecodeError(RuntimeError):
    """The decoder did not yield every frame advertised by ffprobe."""


def _expected_sample_count(source_frames: int, stride: int) -> int:
    """Number of indices in ``range(0, source_frames, stride)``."""
    return (source_frames + stride - 1) // stride


def _require_complete_sampling(*, emitted: int, expected: int, video: Path) -> None:
    """Reject a partial decode instead of publishing a truncated hand track.

    OpenCV returns ``False`` both at ordinary EOF and after a decoder error.
    Comparing its emitted sample count to ffprobe's source-frame count keeps
    those states distinct. A partial parquet is deliberately left for
    diagnostics, but the caller must not mark the stage complete.
    """
    if emitted != expected:
        raise HandDecodeError(
            f"decoded only {emitted}/{expected} expected sampled frames from {video}. "
            "The video is truncated or corrupt; regenerate/restore the redacted "
            "artifact before running MediaPipe."
        )


def _ensure_model(models_dir: Path) -> Path:
    """Download the HandLandmarker model, atomically.

    Two failure modes the naive version has:
      1. `target.open("wb")` truncates the destination BEFORE the body is
         read, so an interrupted transfer leaves a partial .task on disk —
         and the `if target.exists()` guard then returns that corrupt file
         forever on every subsequent run.
      2. No integrity check at all: whatever bytes arrive get handed to
         MediaPipe's native TFLite parser.

    Download to a temp file in the same directory, verify, then os.replace()
    (atomic on the same filesystem). A partial download leaves the cache
    untouched.
    """
    def verify(path: Path) -> None:
        size = path.stat().st_size
        if size < config.MP_MODEL_MIN_BYTES:
            raise RuntimeError(
                f"HandLandmarker model {path} is {size} bytes, below the "
                f"{config.MP_MODEL_MIN_BYTES} byte floor. Remove it and retry."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config.MP_MODEL_SHA256:
            raise RuntimeError(
                f"Model checksum mismatch.\n"
                f"  expected {config.MP_MODEL_SHA256}\n"
                f"  got      {digest}\n"
                f"Remove {path} and retry. Unexpected model bytes must not be used."
            )

    target = models_dir / config.MP_MODEL_FILENAME
    if target.exists() and target.stat().st_size > 0:
        verify(target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading MediaPipe HandLandmarker model -> %s", target)

    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".partial")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(
            config.MP_MODEL_URL, timeout=120
        ) as resp:
            shutil.copyfileobj(resp, f)

        verify(tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _flatten_landmarks(landmarks) -> Landmarks:
    out: Landmarks = []
    for lm in landmarks:
        out.extend((float(lm.x), float(lm.y), float(lm.z)))
    return out


def _wrist_xy(landmarks: Landmarks) -> tuple[float, float]:
    """Landmark 0 is the wrist; each landmark is 3 floats (x, y, z)."""
    return landmarks[0], landmarks[1]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class _Detection:
    __slots__ = ("label", "landmarks", "score", "wrist")

    def __init__(self, landmarks: Landmarks, score: float, label: str):
        self.landmarks = landmarks
        self.score = score
        self.label = label
        self.wrist = _wrist_xy(landmarks)


def _extract_detections(result) -> list[_Detection]:
    out: list[_Detection] = []
    # strict=True: these two lists are parallel by MediaPipe's contract. If a
    # version ever returns them mismatched, zip() would silently truncate to
    # the shorter one and drop a hand — the same class of silent hand-loss
    # this module exists to fix. Fail loudly instead.
    for landmarks, handedness in zip(
        result.hand_landmarks, result.handedness, strict=True
    ):
        top = handedness[0]
        out.append(_Detection(_flatten_landmarks(landmarks), float(top.score), top.category_name))
    return out


class HandAssigner:
    """Stateful left/right assignment across frames of ONE video. Not
    thread-safe; construct one per video."""

    def __init__(self) -> None:
        self._prev_left_wrist: tuple[float, float] | None = None
        self._prev_right_wrist: tuple[float, float] | None = None
        self._left_track_id: int | None = None
        self._right_track_id: int | None = None
        self._next_track_id = 0

    def _new_track_id(self) -> int:
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    def assign(
        self, detections: list[_Detection]
    ) -> tuple[_Detection | None, _Detection | None, int | None, int | None]:
        """Returns (left, right, left_track_id, right_track_id)."""
        if not detections:
            self._prev_left_wrist = None
            self._prev_right_wrist = None
            self._left_track_id = None
            self._right_track_id = None
            return None, None, None, None

        if len(detections) == 1:
            d = detections[0]
            return self._assign_one(d)

        # len == 2: try continuity-based assignment first, then fall back to
        # handedness labels only if there's no prior frame to anchor to.
        d0, d1 = detections[0], detections[1]

        if self._prev_left_wrist is not None or self._prev_right_wrist is not None:
            left, right = self._assign_by_continuity(d0, d1)
        else:
            left, right = self._assign_by_label(d0, d1)

        # Physical-plausibility gate. _assign_by_continuity picks the CHEAPER
        # of the two pairings, but cheaper is not the same as plausible — if
        # both pairings require an impossible jump (one hand left frame and a
        # false positive appeared elsewhere), the winner is still wrong. A
        # wrist cannot cross an eighth of the frame in one 33ms step, so a
        # slot whose displacement exceeds the bound is treated as a DIFFERENT
        # hand: it gets a fresh track_id. That makes the discontinuity visible
        # downstream instead of silently corrupting the velocity signal the
        # segmentation layer reads.
        left_broke = self._exceeds_plausible(left, self._prev_left_wrist)
        right_broke = self._exceeds_plausible(right, self._prev_right_wrist)
        if left_broke:
            self._left_track_id = None
        if right_broke:
            self._right_track_id = None

        left_tid = self._track_id_for_slot("left", left is not None)
        right_tid = self._track_id_for_slot("right", right is not None)
        self._prev_left_wrist = left.wrist if left else None
        self._prev_right_wrist = right.wrist if right else None
        return left, right, left_tid, right_tid

    @staticmethod
    def _exceeds_plausible(
        det: _Detection | None, prev: tuple[float, float] | None
    ) -> bool:
        if det is None or prev is None:
            return False
        return _dist(det.wrist, prev) > config.MAX_PLAUSIBLE_DELTA_PER_FRAME

    def _assign_one(
        self, d: _Detection
    ) -> tuple[_Detection | None, _Detection | None, int | None, int | None]:
        """Single detection: assign by continuity if we have a prior slot to
        compare to, else by its own handedness label."""
        dist_to_left = _dist(d.wrist, self._prev_left_wrist) if self._prev_left_wrist else None
        dist_to_right = _dist(d.wrist, self._prev_right_wrist) if self._prev_right_wrist else None

        if dist_to_left is not None and dist_to_right is not None:
            goes_left = dist_to_left <= dist_to_right
        elif dist_to_left is not None:
            goes_left = dist_to_left <= config.MAX_PLAUSIBLE_DELTA_PER_FRAME
        elif dist_to_right is not None:
            goes_left = not (dist_to_right <= config.MAX_PLAUSIBLE_DELTA_PER_FRAME)
        else:
            goes_left = d.label == "Left"

        if goes_left:
            tid = self._track_id_for_slot("left", True)
            self._track_id_for_slot("right", False)
            self._prev_left_wrist, self._prev_right_wrist = d.wrist, None
            return d, None, tid, None
        else:
            tid = self._track_id_for_slot("right", True)
            self._track_id_for_slot("left", False)
            self._prev_left_wrist, self._prev_right_wrist = None, d.wrist
            return None, d, None, tid

    def _assign_by_continuity(
        self, d0: _Detection, d1: _Detection
    ) -> tuple[_Detection | None, _Detection | None]:
        """Pick whichever of the two possible (left, right) pairings has the
        smaller total wrist displacement from the previous frame; this is
        the physical-plausibility cross-check that catches the swap bug
        (S6/S7): if labels disagree with continuity, continuity wins,
        because a handedness swap is common but a hand teleporting across
        the frame in one 67ms step is not."""
        prev_l = self._prev_left_wrist
        prev_r = self._prev_right_wrist

        def cost(a: _Detection, b: _Detection) -> float:
            # a -> left slot, b -> right slot
            c = 0.0
            if prev_l is not None:
                c += _dist(a.wrist, prev_l)
            if prev_r is not None:
                c += _dist(b.wrist, prev_r)
            return c

        cost_ab = cost(d0, d1)  # d0->left, d1->right
        cost_ba = cost(d1, d0)  # d1->left, d0->right

        if cost_ab <= cost_ba:
            return d0, d1
        return d1, d0

    def _assign_by_label(
        self, d0: _Detection, d1: _Detection
    ) -> tuple[_Detection | None, _Detection | None]:
        """No prior frame to anchor to: fall back to handedness labels. If
        both share the same label (the exact v1 bug case), break the tie by
        x-position: smaller x goes to the left slot."""
        if d0.label == "Left" and d1.label == "Right":
            return d0, d1
        if d0.label == "Right" and d1.label == "Left":
            return d1, d0
        # Same label on both — no reliable signal from handedness at all.
        # Order by x-coordinate: smaller x -> left slot. This is a heuristic,
        # not a guarantee, but it is at least DETERMINISTIC and doesn't
        # silently drop a hand the way v1's first-match-wins logic did.
        if d0.wrist[0] <= d1.wrist[0]:
            return d0, d1
        return d1, d0

    def _track_id_for_slot(self, slot: str, present: bool) -> int | None:
        attr = f"_{slot}_track_id"
        current = getattr(self, attr)
        if not present:
            setattr(self, attr, None)
            return None
        if current is None:
            new_id = self._new_track_id()
            setattr(self, attr, new_id)
            return new_id
        return current


def _build_detector(
    models_dir: Path,
    *,
    detection_confidence: float = config.MP_MIN_HAND_CONFIDENCE,
    presence_confidence: float = config.MP_MIN_PRESENCE_CONFIDENCE,
) -> object:
    model_path = _ensure_model(models_dir)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)


def run(
    video: Path,
    video_id: str,
    models_dir: Path,
    *,
    info: VideoInfo | None = None,
    fps: int = config.MP_FPS,
    hand_confidence: float = config.MP_MIN_HAND_CONFIDENCE,
) -> Iterator[HandFrame]:
    """Yield HandFrame records for every sampled frame of `video`.

    `info` MUST come from media.probe.probe() on the same file. The hand
    track and the caption track both derive timestamps in milliseconds, and
    an F1@±0.5s boundary claim is only meaningful if both derive them from
    the SAME frame rate. An earlier version read `cv2.CAP_PROP_FPS` here and
    silently fell back to 30.0 when OpenCV returned 0 — a second, unrelated
    clock, which is exactly the bug probe.py's docstring says is fixed.
    Passing None re-probes rather than guessing.
    """
    if not 0.0 < hand_confidence <= 1.0:
        raise ValueError("hand_confidence must be in (0, 1]")
    if info is None:
        info = probe(video)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")

    source_fps = info.fps
    if source_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"ffprobe reported fps={source_fps} for {video}. Every timestamp in "
            f"the hand track derives from this, so guessing a default would put "
            f"the hand and caption tracks on different clocks."
        )
    stride = max(1, round(source_fps / fps))
    log.info(
        "hands: %s -- source_fps=%.2f stride=%d target_fps~=%.2f",
        video_id, source_fps, stride, source_fps / stride,
    )

    detector = _build_detector(
        models_dir,
        detection_confidence=hand_confidence,
        presence_confidence=hand_confidence,
    )
    assigner = HandAssigner()
    try:
        emitted_idx = 0
        source_idx = 0
        t0 = time.time()
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if source_idx % stride != 0:
                source_idx += 1
                continue

            timestamp_ms = round(source_idx / source_fps * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            detections = _extract_detections(result)
            left, right, left_tid, right_tid = assigner.assign(detections)
            hands_present = int(left is not None) + int(right is not None)

            yield HandFrame(
                video_id=video_id,
                frame_idx=emitted_idx,
                timestamp_ms=timestamp_ms,
                left_landmarks=left.landmarks if left else None,
                right_landmarks=right.landmarks if right else None,
                left_confidence=left.score if left else None,
                right_confidence=right.score if right else None,
                hands_present=hands_present,
                detector_version=_DETECTOR_VERSION,
                left_track_id=left_tid,
                right_track_id=right_tid,
            )
            emitted_idx += 1
            source_idx += 1

        elapsed = time.time() - t0
        log.info(
            "hands: %s -- %d frames in %.1fs (%.1f fps)",
            video_id, emitted_idx, elapsed, emitted_idx / max(elapsed, 1e-9),
        )
        _require_complete_sampling(
            emitted=emitted_idx,
            expected=_expected_sample_count(info.n_frames, stride),
            video=video,
        )
    finally:
        detector.close()
        cap.release()
