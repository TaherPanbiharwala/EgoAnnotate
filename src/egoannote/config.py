"""Pipeline knobs in one place. Import from here, don't hard-code.

v1 computed REPO_ROOT via `Path(__file__).resolve().parents[3]`, which only
resolves correctly for an editable install from the repo (DX review, v1
carry-forward note). This version resolves the data root from an env var
with a sane default, so a `pip install egoannote` or a moved checkout still
works.
"""

from __future__ import annotations

import hashlib
import os
from fractions import Fraction
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Where runtime data lives. Override with EGOANNOTE_DATA_DIR for a moved
# checkout or a non-default disk. Defaults to ./runs relative to cwd, which
# matches "no CLI, run scripts from the repo root" (see Locked decisions).
DATA_DIR: Final[Path] = Path(os.environ.get("EGOANNOTE_DATA_DIR", "./runs")).resolve()

PROMPTS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "prompts"
MODELS_DIR: Final[Path] = DATA_DIR / "models"
REPORTS_DIR: Final[Path] = DATA_DIR / "reports"

# ---------------------------------------------------------------------------
# VLM caption track
# ---------------------------------------------------------------------------

# Cycle-2 amendment: 6s windows at 6s stride (no overlap) — see plan S1.
# Overlap existed to triangulate boundaries via the interval argument; S1's
# ordered-list prompt reports sub-window boundaries directly, so overlap is
# no longer required. This HALVES window count vs v1's design.
#
# 8 frames / 6s = 4/3 fps, NOT an integer. Pass this as an exact rational to
# ffmpeg (`-vf fps=4/3`), never as a float — floats drift over a long video
# (Eng review A1-11). VLM_FPS is a Fraction for this reason.
VLM_WINDOW_SECONDS: Final[int] = 6
VLM_WINDOW_FRAMES: Final[int] = 8
VLM_FPS: Final[Fraction] = Fraction(VLM_WINDOW_FRAMES, VLM_WINDOW_SECONDS)  # 4/3
VLM_FPS_STR: Final[str] = f"{VLM_FPS.numerator}/{VLM_FPS.denominator}"  # ffmpeg -vf fps=

# Provisional (Phase 0.5 gate). If ordered-list temporal placement is
# unreliable, fall back to 3s stride with the fused-signal design (see plan
# "C1 RESOLUTION"), NOT the plain interval argument — S3 showed that only
# covers the true boundary 50% of the time.
VLM_STRIDE_SECONDS: Final[int] = VLM_WINDOW_SECONDS  # == window ⇒ no overlap

# ---------------------------------------------------------------------------
# MediaPipe hand-tracking track — local CPU, free tier
# ---------------------------------------------------------------------------

MP_FPS: Final[int] = 15
MP_MIN_HAND_CONFIDENCE: Final[float] = 0.5
MP_MIN_PRESENCE_CONFIDENCE: Final[float] = 0.5
MP_MIN_TRACKING_CONFIDENCE: Final[float] = 0.5
MP_NUM_HANDS: Final[int] = 2
MP_MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MP_MODEL_FILENAME: Final[str] = "hand_landmarker.task"
# Integrity checks for the downloaded model. The size floor catches a
# truncated transfer (the observed good file is ~7.8 MB). MP_MODEL_SHA256 is
# empty until a known-good copy is verified and its digest pinned here —
# once set, a mismatch hard-fails instead of feeding unexpected bytes to
# MediaPipe's native TFLite parser.
MP_MODEL_MIN_BYTES: Final[int] = 5_000_000
MP_MODEL_SHA256: Final[str] = ""

# S6/S7: assignment-swap guard. A frame-to-frame wrist jump larger than this
# (in normalized image units, i.e. fraction of frame width/height) is treated
# as a detection artifact, not real motion — measured swaps in v1 data hit
# 18.33 units/s at 15 fps ≈ 1.22 units/frame, i.e. "the wrist teleported
# across the frame." 0.25/frame is a generous plausibility bound, not a tight
# one; tune against the human-labeled sample in Phase 6.5.
MAX_PLAUSIBLE_DELTA_PER_FRAME: Final[float] = 0.25

# ---------------------------------------------------------------------------
# Caption generation knobs
# ---------------------------------------------------------------------------

CAPTION_PROMPT_VERSION: Final[str] = "v3"
CAPTION_PROMPT_FILE: Final[Path] = PROMPTS_DIR / f"caption_{CAPTION_PROMPT_VERSION}.txt"

# S1's ordered-list-of-actions schema is longer than v1's single v2.5 object;
# v1 sized 1024 for one object and hit mid-string truncation at 480 before
# that. Raised here; re-verify empirically in Phase 0.5/4 (Eng review A1-6).
VLM_MAX_NEW_TOKENS: Final[int] = 2048
VLM_TEMPERATURE: Final[float] = 0.2

# Provisional pending the Phase 4 3-arm ablation (768 / 1280 / 1920). v1's
# actual default was 768px — NOT 1920 — so 1280 is a cost INCREASE, not a
# saving; see plan S10 correction. Do not treat this constant as settled.
VLM_FRAME_RESIZE_LONG_EDGE: Final[int] = 1280

VLM_RETRIES: Final[int] = 2

# Every VLM call is network-bound — measured per-call latency is 5-15s of
# mostly wait, not compute (perf review). Captioning ~900 windows serially
# measured out to 1.25-3.75 hours of wall clock for zero correctness
# benefit: the store (SQLite, 900 writes measured at 0.09s total), the HTTP
# client, and the spend tracker are all concurrency-safe. At 8 concurrent
# calls the same run is roughly 15-30 minutes. Each in-flight call holds
# ~11 MB (raw + base64 JPEGs + the serialized request body), so 8 workers
# costs ~90 MB — trivial on an 8 GB machine. Pass max_workers=1 to
# layers.caption.caption_video for strictly serial, deterministic execution
# (what the test suite uses for exact-call-count assertions).
CAPTION_MAX_WORKERS: Final[int] = 8

# ---------------------------------------------------------------------------
# Segmentation (Phase 6 - see plan "S1-S5" + "C1 RESOLUTION")
# ---------------------------------------------------------------------------

SEG_MIN_SEG_SECONDS: Final[float] = 1.5
SEG_MAX_SEG_SECONDS: Final[float] = 20.0
# S3: NOT asserted — set from the measured p90 of |proposed - human| on the
# boundary sample in Phase 6.5. This is a placeholder until that pilot runs.
SEG_SNAP_RADIUS_SECONDS_PROVISIONAL: Final[float] = 0.75
SEG_PRESENCE_DEBOUNCE_FRAMES: Final[int] = 8  # ≥8 consecutive frames, per S5

SEG_WEIGHT_LABEL: Final[float] = 0.5
SEG_WEIGHT_MOTION: Final[float] = 0.35
SEG_WEIGHT_VISUAL: Final[float] = 0.15
# Degraded (no hands in frame) reweighting — see plan S1's degradation path.
SEG_WEIGHT_LABEL_DEGRADED: Final[float] = 0.7
SEG_WEIGHT_VISUAL_DEGRADED: Final[float] = 0.3


def video_id_from_content(path: Path, stem: str) -> str:
    """video_id = stem + content hash prefix.

    S9 fix: GoPro filenames repeat across cards/sessions (GX010001.MP4 is one
    of the most common camera filenames on earth). Bare `path.stem` silently
    merges unrelated recordings under one video_id. Hash a stable prefix of
    the file to disambiguate; hard-error upstream on a stem collision with a
    differing hash rather than silently overwriting.

    Raises on an empty file: every zero-byte file hashes identically, so two
    failed downloads would silently share one video_id and one set of
    annotations.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        # First 8 MiB is enough to disambiguate distinct files cheaply; two
        # different recordings sharing 8 MiB of identical header+content is
        # not a realistic collision for GoPro MP4s.
        chunk = f.read(8 * 1024 * 1024)
    if not chunk:
        raise ValueError(
            f"{path} is empty (0 bytes). Every empty file hashes identically, so "
            f"this would silently share a video_id with any other empty file — "
            f"the download or copy probably failed."
        )
    h.update(chunk)
    return f"{stem}-{h.hexdigest()[:8]}"
