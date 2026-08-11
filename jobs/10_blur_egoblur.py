# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "egoblur==2.0.1",
#   "opencv-python-headless>=4.10,<5",
#   "numpy>=1.24,<3",
#   "tqdm>=4.64,<5",
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
"""GPU anonymization job: EgoBlur face + license-plate redaction, one burn-in
re-encode. Runs on an ephemeral RunPod pod. Self-contained per PEP 723 — see
jobs/_contract.py's module docstring for why this can't import src/.

Pipeline: preflight -> discover clips -> detect (10 Hz, tracked +
interpolated) -> redact + encode (one pass, yuv420p raw pipes) -> verify ->
ship manifest + artifacts -> stop the pod. Nothing here ever opens a
database; the laptop owns the only one.

EXIT CODE IS A REAL SIGNAL, NOT JUST "DIDN'T CRASH": exit 0 means every
clip reached PASS_AUTOMATED (or PASS_AUTOMATED_NO_YUNET). Exit 1 means the
run completed but at least one clip needs a human look (NEEDS_REVIEW —
see build_audit()) or failed outright (see run_manifest.json's
failed_clip_ids). Only an uncaught exception during setup (preflight,
clip discovery, detector construction, the budget probe) leaves the pod
running instead of shutting down cleanly — a single bad clip does not,
by design (see main()'s per-clip try/except).

A restart with the same --output-dir SKIPS clips that already have a
manifest.json (pass --force-reprocess to redo them) — this is what makes
resuming a partially-completed batch after a crash cheap instead of
silently re-burning GPU cost on already-shipped clips.

WHAT IS AND ISN'T VERIFIED, READ BEFORE RUNNING FOR REAL:
  - The Gen2 detector glue (Gen2Detector below) is built from a byte-level
    read of the egoblur==2.0.1 wheel's gen2/script/predictor.py — the
    constructor signature, ClassID values, and the NMS+threshold formula in
    Gen2Detector.detect_batch() are direct transcriptions of that source,
    not guesses. What is NOT independently verified: (1) whether calling
    .inference() directly (batched, bypassing the library's own .run()/
    pre_process()) produces numerically correct detections on a real GPU
    — .run() is the maintainers' own tested path but only accepts one
    image at a time, .inference() is the documented multi-image entry
    point but the demo script never actually calls it, so this
    combination has no known-good reference run; (2) whether a
    single-class-trained model's pred_classes actually uses the same 0/1
    FACE/LICENSE_PLATE encoding the class filter assumes (logged at DEBUG
    on every call — see Gen2Detector.detect_batch); (3) whether the raw
    uint8 tensor built by _bgr_batch_to_cuda_tensor() is what the scripted
    model's input layer actually expects when pre_process() is bypassed.
    Use --probe-only first and eyeball probe_frames/*.jpg (written by
    _write_probe_frames()) before trusting a real run — a wrong dtype or
    class mapping most likely shows up there as garbage or empty boxes,
    not a crash.
  - The Gen1 fallback (Gen1Detector) has no upstream class to import at
    all — gen1/script/ ships only a demo script (Apache-2.0), so
    Gen1Detector.detect_batch() is that script's ~30 lines, kept close to
    verbatim (torch.jit.load -> get_image_tensor -> forward -> NMS ->
    threshold) rather than reworked, specifically so any future diff
    against the upstream file stays legible. Gen1 cannot batch (confirmed:
    GitHub facebookresearch/EgoBlur#6, #14 — a 4-D batch dim throws a
    RuntimeError from inside the traced graph) and is reported at ~2 fps
    per image (#14 cites a cudaStreamSync stall) with no official
    benchmark — the probe phase measures your actual hardware rather than
    trusting that number.
  - Weights: both generations' code is Apache-2.0 (verified: the "EgoBlur
    Model License Agreement" page is verbatim, unmodified Apache-2.0). The
    .jit weight files themselves are gated behind an email-capture form at
    projectaria.com that multiple users report as broken (GitHub #32,
    still open at time of writing). Gen1 weights are also mirrored,
    ungated, on HuggingFace at revision projectaria/EgoBlur@9c0b319 (NOT
    the repo's current HEAD, which has removed them) as
    ego_blur_face.zip / ego_blur_lp.zip. No ungated Gen2 mirror is known.
    This script does not download weights for you — point --face-weights*
    at files you already have.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# NOTE ON jobs/_contract.py
#
# Other job scripts vendor _contract.py's ShardMeta/write_shard by copy,
# because a PEP-723 script cannot import from src/. This job does NOT: it
# produces a redacted video plus a manifest, not arrays, so there is nothing
# for the .npz shard format to carry. An earlier version pasted the contract
# in anyway — 61 lines nothing ever called, while three comments and the
# module docstring claimed shards were being shipped. Dead code that lies
# about the output contract is worse than no code, so it is gone; add it back
# only alongside a real write_shard() call.
# =============================================================================

log = logging.getLogger("blur")

DETECT_HZ_DEFAULT = 10.0
FACE_THRESHOLD_DEFAULT = 0.30  # NOT EgoBlur's 0.674 default — see module
# docstring reasoning: EgoBlur's published recall on truncated faces (the
# dominant case at the edge of an egocentric frame) is 0.430. A false
# positive here costs a gray rectangle; a false negative costs a person's
# identity. That asymmetry is why this is lower than the paper's operating
# point, not a mistake.
LP_THRESHOLD_DEFAULT = 0.40
SWEEP_THRESHOLD_DEFAULT = 0.10  # candidate-miss band floor for the audit —
# see check_low_threshold_sweep().
NMS_IOU_DEFAULT = 0.30

# cv2.FaceDetectorYN.detect returns [n, 15]; col 14 is the face score.
# Cols 4-5 are the RIGHT EYE x,y — reading those as a score (an earlier bug
# here) silently disables the confidence gate, since a pixel coordinate is
# essentially always >= any sane threshold.
YUNET_SCORE_COL = 14
YUNET_SCORE_MIN = 0.5

# Track association IoU. Deliberately looser than --nms-iou: NMS decides
# "are these the same detection in one frame", this decides "is this the
# same face one detection-frame later", after it has moved.
TRACK_IOU_DEFAULT = 0.2

# Cap on per-check items written into the manifest/audit. The full counts
# are always reported; this only bounds the embedded sample.
AUDIT_MAX_ITEMS = 200

# Default frames per detector call — override with --detect-batch. Used by
# BOTH the real detection pass and the probe; the probe previously batched
# all --probe-frames at once, which OOM'd the GPU before the run began. 8 at
# native 1080p fit comfortably on a 32GB 5090; a smaller/shared GPU needs a
# lower value, and a CUDA OOM here is a hard crash mid-batch, not a clean
# refusal — pass --detect-batch conservatively rather than finding out.
DETECT_BATCH = 8

# Cost-model assumptions for the budget gate. These are ASSUMPTIONS, not
# measurements: nothing in this job times encode or verify throughput, so
# the estimate is a floor. Named here rather than buried as literals so the
# gate's optimism is visible and adjustable.
ASSUMED_ENCODE_FPS = 40.0   # libx264 -preset slow at 1080p on a pod's vCPUs
ASSUMED_DECODE_FPS = 380.0  # CPU decode of the encoded output
ASSUMED_YUNET_MS = 35.0     # per sampled frame, CPU

# How much of a detected face must be inside the union of that frame's fill
# boxes before it counts as redacted. Deliberately near-1: this gates a
# privacy claim, so "mostly covered" is a miss, not a pass.
COVERAGE_MIN_FRAC = 0.98

# How far a redacted pixel may drift from FILL_VALUE and still count as
# redacted. Not zero, because the output is lossily encoded: a flat block
# still moves by a couple of levels through H.264. Measured on a real
# CRF-18 encode, the eroded interior of a filled box lands at |dev| <= 3
# with std ~0.1, so these are tight-but-real, not arbitrary.
FILL_MAX_DEVIATION = 3
FILL_MAX_STD = 2

# Erosion before checking, in each plane's own samples. These are measured,
# not guessed: on a real CRF-18 encode of a filled 100x100 box, deviation
# from FILL_VALUE against erosion depth came out as
#   chroma samples eroded: 0 -> 4, 1 -> 4, 2 -> 4, 3 -> 2, 4 -> 2
# so ringing from the box's sharp edge reaches ~3 chroma samples (6 luma
# px) inward. The original 2-luma-px erosion is only ONE chroma sample, so
# every correctly-redacted large box was reported as a violation — 30/30
# frames — which would have made every clip NEEDS_REVIEW and trained the
# reviewer to ignore the gate entirely.
LUMA_ERODE_PX = 2
CHROMA_ERODE_SAMPLES = 3

FILL_VALUE = 128  # mid-gray. Exact value matters: check_fill_integrity()
# treats deviation from this as a hard invariant, not a heuristic.


# =============================================================================
# CLI / config
# =============================================================================


@dataclass(slots=True)
class Config:
    input_dir: Path
    output_dir: Path
    run_id: str
    gen: str  # "1" | "2" | "auto"
    device: str
    face_weights_gen2: Path | None
    lp_weights_gen2: Path | None
    face_weights_gen1: Path | None
    lp_weights_gen1: Path | None
    face_weights_sha256: str | None
    lp_weights_sha256: str | None
    face_threshold: float
    lp_threshold: float
    sweep_threshold: float
    nms_iou: float
    detect_hz: float
    redaction: str  # "fill" | "blur"
    dilate_scale: float
    motion_margin_px: int
    hold_frames: int
    min_box_px: int
    encode_preset: str
    encode_crf: int
    budget_usd: float
    gpu_rate_usd_per_hr: float
    force: bool
    probe_only: bool
    probe_frames: int
    yunet_model: Path | None
    forced_boxes: Path | None
    watchdog_hours: float
    no_watchdog: bool
    skip_shutdown: bool
    force_reprocess: bool
    probe_frames_dir: Path
    back_hold_frames: int
    gen2_resize_px: int | None
    detect_batch: int


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--gen", choices=["1", "2", "auto"], default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--face-weights-gen2", type=Path)
    p.add_argument("--lp-weights-gen2", type=Path)
    p.add_argument("--face-weights-gen1", type=Path)
    p.add_argument("--lp-weights-gen1", type=Path)
    p.add_argument("--face-weights-sha256")
    p.add_argument("--lp-weights-sha256")
    p.add_argument("--face-threshold", type=float, default=FACE_THRESHOLD_DEFAULT)
    p.add_argument("--lp-threshold", type=float, default=LP_THRESHOLD_DEFAULT)
    p.add_argument("--sweep-threshold", type=float, default=SWEEP_THRESHOLD_DEFAULT)
    p.add_argument("--nms-iou", type=float, default=NMS_IOU_DEFAULT)
    p.add_argument("--detect-hz", type=float, default=DETECT_HZ_DEFAULT)
    p.add_argument("--redaction", choices=["fill", "blur"], default="fill")
    p.add_argument("--dilate-scale", type=float, default=1.3)
    p.add_argument("--motion-margin-px", type=int, default=8)
    p.add_argument("--hold-frames", type=int, default=0, help="0 = auto (1s of frames)")
    p.add_argument("--back-hold-frames", type=int, default=0,
                    help="frames to cover BEFORE a track's first detection. 0 = auto "
                         "(same as --hold-frames). Was hardcoded to one detection "
                         "stride, i.e. a tenth of the forward hold: a face walking "
                         "into view shipped unredacted from the moment it appeared "
                         "until the detector first caught it, typically 5-25 frames.")
    p.add_argument("--min-box-px", type=int, default=8)
    p.add_argument("--encode-preset", default="slow")
    p.add_argument("--encode-crf", type=int, default=18)
    p.add_argument("--budget-usd", type=float, default=3.00)
    p.add_argument("--gpu-rate-usd-per-hr", type=float, default=0.16)
    p.add_argument("--force", action="store_true", help="skip the budget gate")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--probe-frames", type=int, default=600)
    p.add_argument("--yunet-model", type=Path, help="cv2 FaceDetectorYN onnx path; omit to skip")
    p.add_argument("--forced-boxes", type=Path, help='JSON mapping clip_id -> [{"frame_idx": int, "box": [x1,y1,x2,y2]}], listing faces a human confirmed were missed. Requires --force-reprocess to re-run an already-manifested clip.')
    p.add_argument("--watchdog-hours", type=float, default=8.0)
    p.add_argument("--no-watchdog", action="store_true")
    p.add_argument("--skip-shutdown", action="store_true", help="for local/laptop testing")
    p.add_argument("--force-reprocess", action="store_true",
                    help="redo clips that already have a manifest.json from a prior run "
                         "(default: skip them — a restart after a crash on clip N must not "
                         "silently re-burn GPU/CPU cost and re-clobber clips 1..N-1)")
    p.add_argument("--gen2-resize-px", type=int, default=None,
                    help="Gen2 only: resize the short AND long side to this many "
                         "pixels before inference. Omit for native resolution (the "
                         "default). Upstream uses 1200; native keeps small distant "
                         "faces at full size but runs the model at a scale it was "
                         "never evaluated at. A/B the two with --probe-only before "
                         "trusting either operating point.")
    p.add_argument("--detect-batch", type=int, default=DETECT_BATCH,
                    help=f"frames per detector call, both the real pass and the "
                         f"probe (default {DETECT_BATCH}, sized for a 32GB 5090 at "
                         f"native 1080p resolution). Lower this on a smaller GPU — "
                         f"an OOM here is a CUDA error mid-batch, not a clean "
                         f"refusal, so guess conservatively rather than finding out "
                         f"the hard way.")
    p.add_argument("--probe-frames-dir", type=Path, default=None,
                    help="where --probe-only writes annotated sample frames. These are "
                         "UNREDACTED originals, so this deliberately defaults OUTSIDE "
                         "--output-dir (a sibling '<output-dir>-probe-DO-NOT-SHIP') and "
                         "must never be synced or published.")
    a = p.parse_args(argv)

    if a.dilate_scale < 1.0:
        p.error(
            f"--dilate-scale {a.dilate_scale} < 1.0 would SHRINK redaction boxes below "
            f"the raw detected box instead of padding them — almost certainly a typo "
            f"for a value like 1.3."
        )
    if a.motion_margin_px < 0:
        p.error(f"--motion-margin-px {a.motion_margin_px} is negative — would shrink boxes.")
    if a.detect_hz <= 0:
        p.error(f"--detect-hz {a.detect_hz} must be > 0 (it divides fps to pick a stride).")
    if a.hold_frames < 0:
        p.error(f"--hold-frames {a.hold_frames} is negative — disables track association.")
    if a.back_hold_frames < 0:
        p.error(f"--back-hold-frames {a.back_hold_frames} is negative.")
    if a.detect_batch < 1:
        p.error(f"--detect-batch {a.detect_batch} must be >= 1.")
    if a.gen2_resize_px is not None and a.gen2_resize_px < 64:
        p.error(f"--gen2-resize-px {a.gen2_resize_px} is implausibly small.")
    if a.min_box_px < 0:
        p.error(f"--min-box-px {a.min_box_px} is negative.")
    for name, val in (("--face-threshold", a.face_threshold),
                       ("--lp-threshold", a.lp_threshold),
                       ("--sweep-threshold", a.sweep_threshold)):
        if not 0.0 < val <= 1.0:
            p.error(f"{name} {val} must be in (0, 1].")
    if a.sweep_threshold >= min(a.face_threshold, a.lp_threshold):
        # The sweep inspects [sweep, operating). If the floor is not strictly
        # below the operating threshold the band is empty and the audit's
        # candidate-miss check silently reports 0 forever — the exact bug
        # this whole detect-low/filter-later design exists to prevent.
        p.error(
            f"--sweep-threshold {a.sweep_threshold} must be strictly below both "
            f"--face-threshold {a.face_threshold} and --lp-threshold {a.lp_threshold}; "
            f"otherwise the audit's low-confidence band is empty and the "
            f"candidate-miss check can never fire."
        )
    od = a.output_dir.expanduser().resolve()
    if a.probe_frames_dir is None:
        # resolve() first: with --output-dir "." or ".." the unresolved name
        # is empty, and the "sibling" landed INSIDE the publish tree —
        # putting unredacted originals back exactly where this defends.
        a.probe_frames_dir = od.parent / f"{od.name}-probe-DO-NOT-SHIP"
    pfd = a.probe_frames_dir.expanduser().resolve()
    if pfd == od or od in pfd.parents:
        p.error(
            f"--probe-frames-dir {pfd} is inside --output-dir {od}. Probe frames'"
            f" are UNREDACTED originals and must never sit in the tree that gets"
            f" synced off the pod and published."
        )
    a.probe_frames_dir = pfd

    return Config(
        input_dir=a.input_dir,
        output_dir=a.output_dir,
        run_id=a.run_id,
        gen=a.gen,
        device=a.device,
        face_weights_gen2=a.face_weights_gen2,
        lp_weights_gen2=a.lp_weights_gen2,
        face_weights_gen1=a.face_weights_gen1,
        lp_weights_gen1=a.lp_weights_gen1,
        face_weights_sha256=a.face_weights_sha256,
        lp_weights_sha256=a.lp_weights_sha256,
        face_threshold=a.face_threshold,
        lp_threshold=a.lp_threshold,
        sweep_threshold=a.sweep_threshold,
        nms_iou=a.nms_iou,
        detect_hz=a.detect_hz,
        redaction=a.redaction,
        dilate_scale=a.dilate_scale,
        motion_margin_px=a.motion_margin_px,
        hold_frames=a.hold_frames,
        back_hold_frames=a.back_hold_frames,
        min_box_px=a.min_box_px,
        encode_preset=a.encode_preset,
        encode_crf=a.encode_crf,
        budget_usd=a.budget_usd,
        gpu_rate_usd_per_hr=a.gpu_rate_usd_per_hr,
        force=a.force,
        probe_only=a.probe_only,
        probe_frames=a.probe_frames,
        yunet_model=a.yunet_model,
        forced_boxes=a.forced_boxes,
        watchdog_hours=a.watchdog_hours,
        no_watchdog=a.no_watchdog,
        skip_shutdown=a.skip_shutdown,
        force_reprocess=a.force_reprocess,
        probe_frames_dir=a.probe_frames_dir,
        gen2_resize_px=a.gen2_resize_px,
        detect_batch=a.detect_batch,
    )


# =============================================================================
# Phase 0 — preflight
# =============================================================================


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def arm_watchdog(hours: float, output_dir: Path) -> None:
    """A wall-clock backstop independent of whether the job itself succeeds,
    fails, or hangs — Risk 7 in the plan measures a forgotten L4 at
    $9.36/day; this bounds the damage of ANY failure mode, not just the
    ones the code anticipates. Its own output is logged, not discarded —
    an unmonitored watchdog (missing runpodctl, a stale pod_id, an auth
    failure) fails exactly the way it exists to prevent: silently."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        log.warning(
            "RUNPOD_POD_ID not set (not on a RunPod pod, or the env var is "
            "missing) — watchdog NOT armed. Set --no-watchdog explicitly if "
            "that's intentional, so this isn't a silent gap."
        )
        return
    if not shutil.which("runpodctl"):
        raise RuntimeError(
            "runpodctl not found on PATH but RUNPOD_POD_ID is set — the "
            "watchdog and the clean-exit shutdown both depend on it. "
            "Refusing to start a job whose only cost backstop can't fire."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "watchdog.log"
    seconds = int(hours * 3600)
    with log_path.open("w") as logf:
        subprocess.Popen(
            ["bash", "-c", f"sleep {seconds}; runpodctl pod stop {pod_id}"],
            start_new_session=True,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    log.info("watchdog armed: pod stops in %.1fh regardless of job state (log: %s)",
              hours, log_path)


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found on PATH — check the pod image.")
    return path


def preflight(cfg: Config) -> dict:
    """Fail loudly here, not 40 minutes into a paid GPU run."""
    report: dict[str, Any] = {}

    ffmpeg = _bin("ffmpeg")
    ffprobe = _bin("ffprobe")
    report["ffmpeg"] = ffmpeg
    report["ffprobe"] = ffprobe

    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    ).stdout
    if "libx264" not in encoders:
        raise RuntimeError("ffmpeg build has no libx264 encoder — wrong image?")

    # Every decode in this job passes -fps_mode, which needs ffmpeg >= 5.0.
    # Ubuntu 22.04 still ships 4.4, which rejects it outright. ffprobe keeps
    # working (its options are ancient and stable), so clip discovery passes
    # and the FIRST decode yields zero bytes instead of an error the caller
    # can see. That surfaced as "probe found zero detections" — a statement
    # about the footage, from a binary that never decoded a single frame.
    # Checking the option itself, not a parsed version string, because that
    # is the thing that actually has to work.
    opt_check = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "nullsrc=s=16x16:d=0.1", "-fps_mode", "passthrough",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if opt_check.returncode != 0:
        raise RuntimeError(
            f"{ffmpeg} rejected -fps_mode, which every decode in this job "
            f"uses (needs ffmpeg >= 5.0; 4.x fails exactly this way). Every "
            f"decode would silently produce zero frames. ffmpeg said: "
            f"{opt_check.stderr.strip()}"
        )

    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch not importable — PEP 723 env resolution failed") from e

    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "cfg.device='cuda' but torch.cuda.is_available() is False. "
            "Do not silently fall back to CPU — Gen1/Gen2 on CPU is a "
            "different (much slower) cost regime the budget gate hasn't "
            "priced. Fix the pod or pass --device cpu explicitly."
        )
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)

    weight_pairs = [
        (cfg.face_weights_gen2, cfg.face_weights_sha256, "face_weights_gen2"),
        (cfg.lp_weights_gen2, cfg.lp_weights_sha256, "lp_weights_gen2"),
        (cfg.face_weights_gen1, None, "face_weights_gen1"),
        (cfg.lp_weights_gen1, None, "lp_weights_gen1"),
    ]
    for wpath, wsha, label in weight_pairs:
        if wpath is None:
            continue
        if not wpath.exists():
            raise RuntimeError(f"{label} does not exist: {wpath}")
        if wsha:
            actual = sha256_file(wpath)
            if actual != wsha:
                raise RuntimeError(
                    f"{label} sha256 mismatch: expected {wsha}, got {actual}. "
                    f"Refusing to run inference on unexpected model bytes."
                )
            report[f"{label}_sha256_verified"] = True

    if cfg.forced_boxes is not None and not cfg.forced_boxes.exists():
        # Checked here, before any GPU time: these are faces a human already
        # confirmed were missed, so silently running without them republishes
        # a known face.
        raise RuntimeError(
            f"--forced-boxes {cfg.forced_boxes} does not exist. Refusing to "
            f"start a remediation run that would silently skip them."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(cfg.output_dir)
    report["free_gb"] = du.free / 1e9
    if du.free < 5 * 1e9:
        raise RuntimeError(f"only {du.free/1e9:.1f} GB free at {cfg.output_dir} — too tight")

    log.info("preflight OK: %s", json.dumps(report, default=str))
    return report


# =============================================================================
# Phase 1 — clip discovery + integrity gates
# =============================================================================


@dataclass(slots=True)
class ClipInfo:
    path: Path
    clip_id: str
    clip_group: str
    chapter_index: int
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    rotation: int
    sha256: str
    # Source color tags, propagated verbatim into the encode. Hardcoding
    # bt709/tv here and only on the output side rewrote every pixel; see
    # open_encoder's docstring.
    color_primaries: str = "bt709"
    color_trc: str = "bt709"
    colorspace: str = "bt709"
    color_range: str = "tv"


def ffprobe_json(ffprobe: str, path: Path) -> dict:
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,"
        "duration,rotation,codec_name,pix_fmt,"
        "color_primaries,color_transfer,color_space,color_range",
        "-show_entries", "stream_tags=rotate",
        "-of", "json",
        "-count_packets",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _parse_rate(rate: str) -> float:
    num, _, den = rate.partition("/")
    den = den or "1"
    return float(num) / float(den) if float(den) else 0.0


def assert_cfr(probe: dict, path: Path) -> None:
    """GoPro records CFR. Anything else means the file was already touched
    by something upstream, and every t = frame_idx / fps computation this
    job and everything downstream relies on would silently be wrong for
    that clip. Refuse rather than paper over it with -fps_mode cfr, which
    would fabricate timestamps instead of surfacing the problem."""
    stream = probe["streams"][0]
    r = _parse_rate(stream["r_frame_rate"])
    avg = _parse_rate(stream["avg_frame_rate"])
    if r <= 0:
        raise RuntimeError(f"{path}: ffprobe reports r_frame_rate={r}")
    if abs(r - avg) / r > 0.001:
        raise RuntimeError(
            f"{path}: r_frame_rate={r:.4f} avg_frame_rate={avg:.4f} — not CFR. "
            f"Marking NEEDS_MANUAL rather than guessing a fps."
        )


def parse_gopro_chapter(path: Path) -> tuple[str, int]:
    """GXjjcccc.MP4: jj = chapter (01, 02, ...), cccc = clip group. Chapters
    of one recording must not be silently concatenated (that's a second
    re-encode) but DO need a shared clip_group so the laptop can order
    them. Falls back to (stem, 0) for non-GoPro filenames."""
    stem = path.stem
    if len(stem) == 8 and stem[:2].upper() == "GX" and stem[2:].isdigit():
        chapter = int(stem[2:4])
        group = stem[4:8]
        return group, chapter
    return stem, 0


def discover_clips(input_dir: Path, ffprobe: str) -> list[ClipInfo]:
    paths = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in (".mp4", ".mov") and p.is_file()
    )
    if not paths:
        raise RuntimeError(f"no video files under {input_dir}")

    names = [p.name for p in paths]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise RuntimeError(
            f"duplicate filenames across subdirectories: {sorted(dupes)}. "
            f"The rclone dedupe gate lives upstream of this script (Drive "
            f"permits same-name files in one folder and silently drops "
            f"one on copy) — this refusal is the backstop, not the fix."
        )

    clips = []
    for path in paths:
        probe = ffprobe_json(ffprobe, path)
        assert_cfr(probe, path)
        stream = probe["streams"][0]
        rotation = int(stream.get("rotation", stream.get("tags", {}).get("rotate", 0)) or 0)
        w, h = int(stream["width"]), int(stream["height"])
        if rotation in (90, -90, 270, -270):
            w, h = h, w  # ffmpeg -autorotate (default on) swaps display dims
        group, chapter = parse_gopro_chapter(path)
        clips.append(ClipInfo(
            path=path,
            clip_id=path.stem,
            clip_group=group,
            chapter_index=chapter,
            width=w,
            height=h,
            fps=_parse_rate(stream["r_frame_rate"]),
            n_frames=int(stream["nb_read_packets"]),
            duration_s=float(stream.get("duration", 0.0)),
            rotation=rotation,
            sha256=sha256_file(path),
            color_primaries=stream.get("color_primaries") or "bt709",
            color_trc=stream.get("color_transfer") or "bt709",
            colorspace=stream.get("color_space") or "bt709",
            color_range=stream.get("color_range") or "tv",
        ))
    log.info("discovered %d clip(s)", len(clips))
    return clips


# =============================================================================
# Detector abstraction — ONE call shape, TWO real, verified APIs behind it.
# =============================================================================


@dataclass(slots=True)
class Detection:
    frame_idx: int
    cls: str  # "face" | "lp"
    box: tuple[float, float, float, float]  # x1,y1,x2,y2, ORIGINAL pixel space
    score: float


def _bgr_batch_to_cuda_tensor(frames_bgr: list, device: str):
    """CxHxW uint8, BGR, unnormalized. The tensor-BUILDING recipe (transpose
    to CHW, no normalization, no dtype cast) is verified to match
    get_image_tensor() in both gen1 and gen2's own demo scripts — that part
    is not a guess. What is NOT verified: whether Gen2's .inference(),
    called directly and bypassing pre_process()/transform_image() (done
    here specifically to sidestep transform_image()'s reported unwanted
    RGB2BGR conversion), accepts this same uint8 tensor as-is, or whether
    the scripted model's true input layer expects something pre_process()
    would otherwise have produced (e.g. a float/normalized tensor).
    Stacked here into BxCxHxW for Gen2's real batched .inference() call.
    This is exactly the assumption --probe-only's probe_frames/*.jpg
    output exists to let a human catch before a real run: a wrong dtype
    here would likely surface as garbage or empty detections, not a
    crash."""
    import numpy as np
    import torch

    arr = np.stack([np.transpose(f, (2, 0, 1)) for f in frames_bgr])  # B,C,H,W
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


class Gen2Detector:
    """gen2.script.predictor.EgoblurDetector, called at the low-level
    .inference() entry point (true multi-image batching) rather than the
    demo's .run() (confirmed single-image-only: it accepts rank-3 and
    auto-unsqueezes to a batch of 1). Constructor signature, ClassID
    values, and the NMS+threshold formula below are transcribed from a
    direct read of gen2/script/predictor.py in the egoblur==2.0.1 wheel —
    see the module docstring for exactly what is and isn't independently
    verified about this path."""

    def __init__(self, weights_path: Path, cls: str, device: str, score_threshold: float,
                 nms_iou: float, resize_px: int | None = None):
        from gen2.script.predictor import ClassID, EgoblurDetector

        self.cls = cls
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.device = device
        class_id = ClassID.FACE if cls == "face" else ClassID.LICENSE_PLATE
        # resize_px=None (the default) runs at NATIVE resolution: small and
        # distant bystander faces keep full linear size instead of losing
        # 37.5% to the 1200px short side that both upstream entry points
        # use. That is a deliberate recall bias, not an oversight, and no
        # rescale-back step is needed since input and inference resolution
        # are then identical.
        #
        # It is also UNMEASURED. Running 1080x1920 instead of 675x1200 puts
        # every object 1.6x larger in linear size than the scales the model
        # was evaluated at, which is exactly the kind of shift that moves a
        # score distribution — and FACE_THRESHOLD_DEFAULT is calibrated
        # against nothing but judgement. --gen2-resize-px exists so the two
        # can be A/B'd on the same frames with --probe-only, cheaply,
        # instead of arguing about it. Compare n_face_above_threshold and
        # max_face_score between the two runs before trusting either.
        self._detector = EgoblurDetector(
            model_path=str(weights_path),
            device=device,
            detection_class=class_id,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou,
            resize_aug=(None if resize_px is None
                        else {"min_size_test": resize_px, "max_size_test": resize_px}),
        )
        self._class_value = int(class_id.value)

    def detect_batch(self, frames_bgr: list, frame_idxs: list[int]) -> list[Detection]:
        import torch
        import torchvision
        from gen2.script.detectron2.export.torchscript_patch import patch_instances
        from gen2.script.detectron2.utils.utils import convert_scripted_instances
        from gen2.script.predictor import PATCH_INSTANCES_FIELDS

        batch = _bgr_batch_to_cuda_tensor(frames_bgr, self.device)
        out: list[Detection] = []
        with torch.no_grad(), patch_instances(fields=PATCH_INSTANCES_FIELDS):
            preds = self._detector.inference(batch)
            for frame_idx, scripted_inst in zip(frame_idxs, preds, strict=True):
                inst = convert_scripted_instances(scripted_inst)
                boxes = inst.pred_boxes.tensor
                scores = inst.scores
                if boxes.numel() == 0:
                    continue
                if inst.has("pred_classes"):
                    n_before = boxes.shape[0]
                    mask = inst.pred_classes == self._class_value
                    boxes, scores = boxes[mask], scores[mask]
                    # UNVERIFIED ASSUMPTION (see module docstring): this
                    # filter is transcribed verbatim from predictor.py's
                    # _post_process(), but whether a single-class-trained
                    # model's pred_classes actually uses the SAME 0/1
                    # FACE/LICENSE_PLATE encoding this detector instance
                    # assumes was never independently confirmed against a
                    # running model. If it's wrong, this filter zeroes out
                    # every detection for that class, silently. Logged at
                    # DEBUG on every call, not just probe, so a real run
                    # left with default logging can still be diagnosed
                    # after the fact from its own log.
                    log.debug("%s: class filter kept %d/%d raw detections",
                              self.cls, boxes.shape[0], n_before)
                if boxes.numel() == 0:
                    continue
                # NMS + threshold, verbatim from predictor.py's
                # _post_process() — .inference() itself applies neither.
                keep = torchvision.ops.nms(boxes, scores, self.nms_iou)
                boxes, scores = boxes[keep], scores[keep]
                boxes_np = boxes.cpu().numpy()
                scores_np = scores.cpu().numpy()
                for box, score in zip(boxes_np, scores_np, strict=True):
                    if score <= self.score_threshold:
                        continue
                    out.append(Detection(
                        frame_idx=frame_idx, cls=self.cls,
                        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        score=float(score),
                    ))
        return out


class Gen1Detector:
    """Gen1 ships no importable detector class — gen1/script/ contains only
    a demo script (facebookresearch/EgoBlur, Apache-2.0), re-exported as a
    console-script main(), nothing else. This is that script's
    get_image_tensor + get_detections logic (demo_ego_blur_gen1.py, lines
    ~246-286), reproduced close to verbatim rather than reworked, so any
    future diff against the upstream file stays legible. Single image
    only — batching a 4-D tensor throws inside the traced graph itself
    (facebookresearch/EgoBlur#6, #14), not merely a limitation of this
    demo code, so detect_batch() loops internally."""

    def __init__(self, weights_path: Path, cls: str, device: str, score_threshold: float,
                 nms_iou: float):
        import torch

        self.cls = cls
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.device = device
        model = torch.jit.load(str(weights_path), map_location="cpu")
        model.to(device)
        model.eval()
        self._model = model

    def detect_batch(self, frames_bgr: list, frame_idxs: list[int]) -> list[Detection]:
        import numpy as np
        import torch
        import torchvision

        out: list[Detection] = []
        with torch.no_grad():
            for frame_idx, bgr in zip(frame_idxs, frames_bgr, strict=True):
                transposed = np.transpose(bgr, (2, 0, 1))
                image_tensor = torch.from_numpy(np.ascontiguousarray(transposed)).to(self.device)
                detections = self._model(image_tensor)
                boxes, _, scores, _ = detections  # model returns (boxes, labels, scores, dims)
                if boxes.numel() == 0:
                    continue
                keep = torchvision.ops.nms(boxes, scores, self.nms_iou)
                boxes, scores = boxes[keep], scores[keep]
                boxes_np = boxes.cpu().numpy()
                scores_np = scores.cpu().numpy()
                for box, score in zip(boxes_np, scores_np, strict=True):
                    if score <= self.score_threshold:
                        continue
                    out.append(Detection(
                        frame_idx=frame_idx, cls=self.cls,
                        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        score=float(score),
                    ))
        return out


def build_detectors(cfg: Config) -> tuple[str, Any, Any]:
    """Resolves gen='auto' at startup only — never mid-run. Switching
    detectors partway through a clip would mean two different recall
    profiles inside one manifest, which defeats the point of recording
    egoblur.gen in provenance at all."""
    # FACE weights are mandatory; LICENSE-PLATE weights are optional.
    # Indoor egocentric footage contains no plates, and running the LP
    # detector there doubles GPU cost to find nothing. Skipping it is a
    # legitimate configuration — but it is NOT free: the manifest and the
    # audit both record that plates were never checked, so a clip processed
    # without LP weights can never be mistaken later for one that was
    # cleared of plates. Never infer "no plates present" from "no plate
    # detector ran".
    have_gen2 = cfg.face_weights_gen2 is not None
    have_gen1 = cfg.face_weights_gen1 is not None

    if cfg.gen == "2":
        use_gen2 = True
        if not have_gen2:
            raise RuntimeError("--gen 2 requires --face-weights-gen2")
    elif cfg.gen == "1":
        use_gen2 = False
        if not have_gen1:
            raise RuntimeError("--gen 1 requires --face-weights-gen1")
    else:  # auto
        if have_gen2:
            use_gen2 = True
        elif have_gen1:
            use_gen2 = False
        else:
            raise RuntimeError(
                "--gen auto found no face weights. Pass --face-weights-gen2 "
                "(preferred) or --face-weights-gen1."
            )

    # Detectors run at the SWEEP floor, not the operating threshold. This is
    # load-bearing, not a tuning choice: constructing them at the operating
    # threshold makes check_low_threshold_sweep() structurally dead, because
    # it looks for scores in [sweep, operating) and the detector has already
    # dropped every one of them. The operating threshold is applied later,
    # in process_clip(), when choosing which detections feed redaction — so
    # what actually gets grayed out is unchanged, but the audit can finally
    # see the low-confidence band it exists to inspect.
    if use_gen2:
        log.info("using Gen2 detector (batched), detecting down to %.2f", cfg.sweep_threshold)
        face = Gen2Detector(cfg.face_weights_gen2, "face", cfg.device,
                             cfg.sweep_threshold, cfg.nms_iou, cfg.gen2_resize_px)
        lp = None
        if cfg.lp_weights_gen2 is not None:
            lp = Gen2Detector(cfg.lp_weights_gen2, "lp", cfg.device,
                               cfg.sweep_threshold, cfg.nms_iou, cfg.gen2_resize_px)
        else:
            log.warning("no --lp-weights-gen2: LICENSE PLATES WILL NOT BE "
                         "DETECTED OR REDACTED in this run. Recorded in the "
                         "manifest as lp_checked=false.")
        return "2", face, lp

    log.info("using Gen1 detector (single-image, no batching — see module docstring), "
              "detecting down to %.2f", cfg.sweep_threshold)
    face = Gen1Detector(cfg.face_weights_gen1, "face", cfg.device,
                         cfg.sweep_threshold, cfg.nms_iou)
    lp = None
    if cfg.lp_weights_gen1 is not None:
        lp = Gen1Detector(cfg.lp_weights_gen1, "lp", cfg.device,
                           cfg.sweep_threshold, cfg.nms_iou)
    else:
        log.warning("no --lp-weights-gen1: LICENSE PLATES WILL NOT BE "
                     "DETECTED OR REDACTED in this run. Recorded in the "
                     "manifest as lp_checked=false.")
    return "1", face, lp


# =============================================================================
# Raw yuv420p pipe I/O — shared by detection (decode only), redact+encode
# (decode+encode), and verify (decode only).
# =============================================================================


def frame_byte_size(w: int, h: int) -> int:
    if w % 2 or h % 2:
        raise RuntimeError(f"odd dimensions {w}x{h} — yuv420p needs even W and H")
    return w * h + 2 * (w // 2) * (h // 2)


def open_decoder(ffmpeg: str, path: Path, stderr_log: Path) -> subprocess.Popen:
    """CPU decode, not -hwaccel cuda: decode-only measured at ~380 fps on
    CPU vs. detector throughput an order of magnitude lower — keeping the
    GPU at 100% for inference is worth more than a faster decode.
    -fps_mode passthrough (not the default cfr) so a frame is never
    duplicated or dropped to hit a target rate, which is what the
    frames_read == nb_read_packets assertion at EOF depends on.

    stderr goes straight to a file, never subprocess.PIPE: this pipeline
    exists specifically to handle real-world corrupted/truncated GoPro
    footage, and ffmpeg keeps logging at -loglevel error even for a
    survivable decode error (a bad NAL, a missing reference frame). An
    undrained stderr PIPE fills its OS buffer (~64KB), ffmpeg blocks on
    write() and stops producing stdout too, and read_frames()'s stdout.read()
    then hangs forever with no error — a real deadlock, not a hypothetical
    one, on exactly the input this job is built to tolerate."""
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(path), "-map", "0:v:0",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
    ]
    with stderr_log.open("wb") as errf:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)


def read_frames(proc: subprocess.Popen, w: int, h: int) -> Iterator[tuple]:
    import numpy as np

    frame_size = frame_byte_size(w, h)
    cw, ch = w // 2, h // 2
    y_size = w * h
    u_size = cw * ch
    stdout = proc.stdout
    assert stdout is not None
    while True:
        buf = stdout.read(frame_size)
        if len(buf) == 0:
            break
        if len(buf) < frame_size:
            raise RuntimeError(f"short read: got {len(buf)} of {frame_size} bytes — truncated?")
        arr = np.frombuffer(buf, dtype=np.uint8)
        y = arr[:y_size].reshape(h, w).copy()
        u = arr[y_size:y_size + u_size].reshape(ch, cw).copy()
        v = arr[y_size + u_size:].reshape(ch, cw).copy()
        yield y, u, v


def open_encoder(ffmpeg: str, w: int, h: int, fps: float, out_path: Path,
                  preset: str, crf: int, stderr_log: Path, *,
                  color_primaries: str = "bt709", color_trc: str = "bt709",
                  colorspace: str = "bt709", color_range: str = "tv") -> subprocess.Popen:
    """Raw yuv420p in, never bgr24: a BGR round trip costs real quality on
    EVERY pixel (measured 45.5 dB PSNR), not just the redacted ones, and
    this file becomes the single source of truth for hand tracking, VLM
    captioning, and the public release. -an -sn -dn strips
    audio/subtitles/data unconditionally: voices are biometric and GPMF
    timed metadata can carry GPS, both defeating the point of this job.

    COLOR TAGS MUST BE SET ON THE INPUT AS WELL AS THE OUTPUT. Raw pipes
    carry no color metadata, so ffmpeg treats an output-only
    `-color_range tv` as a CONVERSION request from assumed-full to limited
    range — it does not just tag, it rewrites every pixel. Measured on a
    plain re-encode with no redaction at all: mean |Y| drift 13.8, max 33,
    75.6% of pixels altered, versus 0.09 / 5 / 0.0% with the flags on both
    sides. That is a systematic degradation of the published master, and
    it was introduced by adding these flags to prevent washed-out playback
    — which is the same defect, worse. Declaring the input identical makes
    the conversion an identity and leaves the tag doing only its job.

    stderr to a file, same reasoning as open_decoder — the failure path in
    redact_and_encode() reads this file back on a nonzero exit instead of
    proc.stderr, so nothing is lost."""
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    color = [
        "-color_primaries", color_primaries, "-color_trc", color_trc,
        "-colorspace", colorspace, "-color_range", color_range,
    ]
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", f"{fps}",
        *color,          # INPUT side: describes the bytes we are piping in
        "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        *color,          # OUTPUT side: same values, so the convert is a no-op
        "-an", "-sn", "-dn", "-map_metadata", "-1", "-movflags", "+faststart",
        str(out_path),
    ]
    with stderr_log.open("wb") as errf:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)


# =============================================================================
# Phase 2 — probe & budget gate
# =============================================================================


def probe_and_budget(cfg: Config, clips: list[ClipInfo], face_det, lp_det,
                      ffmpeg: str) -> dict:
    """Measures real ms/frame on real frames from the actual footage and
    extrapolates dollars BEFORE committing to the full run. No published
    Gen2 benchmark exists for any GPU this job can rent, and Gen1's ~2 fps
    figure is one user's anecdote — this is the safety valve for both
    being wrong, not a formality.

    Also writes cfg.output_dir/probe_frames/*.jpg with drawn detection
    boxes — the module docstring's "eyeball this before trusting a real
    run" line refers to these. Given the Gen2 .inference() bypass path
    (see module docstring: constructor/ClassID/NMS verified against the
    wheel source, but never executed against a real GPU) this is the
    cheapest available check that the detector is finding plausible faces
    at all, not silently returning garbage from a dtype/shape mismatch."""
    # Sample round-robin across EVERY clip, not just the largest one. The
    # probe's conclusions ("is the detector alive", "what will this cost")
    # are claims about the whole batch, so drawing them from one clip is
    # unsound — and if that clip happens to be faceless, a batch-wide abort
    # fires on evidence from a single file.
    per_clip = max(1, cfg.probe_frames // len(clips))
    sampled_bgr: list = []
    sampled_idx: list[int] = []
    sampled_clip: list[str] = []
    stride = max(1, round(clips[0].fps / cfg.detect_hz))
    for clip in clips:
        stride = max(1, round(clip.fps / cfg.detect_hz))
        want = min(per_clip, max(1, clip.n_frames // stride))
        # Spread the samples across the WHOLE clip. Sampling at the detection
        # stride and stopping the moment `want` was reached only ever looked
        # at the first want*stride frames — 60 seconds of a 4.5-minute clip —
        # so "the probe found no faces" was a claim about the opening minute,
        # not about the footage. A pipe can't seek, so this decodes the whole
        # file either way; only the conversion of kept frames costs anything,
        # and decode is cheap next to detection.
        span = max(stride, clip.n_frames // max(1, want))
        proc = open_decoder(ffmpeg, clip.path,
                            cfg.output_dir / "logs" / f"probe_{clip.clip_id}.stderr.log")
        got = 0
        idx = 0
        for y, u, v in read_frames(proc, clip.width, clip.height):
            if idx % span == 0 and got < want:
                sampled_bgr.append(yuv_to_bgr(y, u, v, clip.color_range == "pc"))
                sampled_idx.append(idx)
                sampled_clip.append(clip.clip_id)
                got += 1
            idx += 1
        if proc.stdout is not None:
            proc.stdout.close()
        proc.terminate()
        proc.wait()
        if len(sampled_bgr) >= cfg.probe_frames:
            break

    if not sampled_bgr:
        # Fail CLOSED, for the same reason check_fill_integrity() and
        # check_yunet() do. A decoder that dies produces zero frames, the
        # detector is never called, and every count below is 0 — which came
        # back worded as "probe found ZERO detections", indistinguishable
        # from footage that genuinely has no faces. That is not theoretical:
        # ffmpeg 4.4 rejected -fps_mode, emitted nothing at all, and this
        # function reported it as a finding about the video.
        raise RuntimeError(
            f"probe sampled ZERO frames from {len(clips)} clip(s) — the "
            f"decoder produced no output at all, so nothing was detected on "
            f"and nothing here is a statement about your footage. See "
            f"{cfg.output_dir / 'logs'}/probe_*.stderr.log for what ffmpeg "
            f"actually said."
        )

    # Run the probe through the SAME batch size production uses (now
    # --detect-batch, default 8). Passing all --probe-frames (default 600)
    # as one batch meant ~3.7 GB of BGR frames, doubled by np.stack's
    # contiguous copy, then a 600-image GPU batch — 75x the production
    # batch — which OOMs any rentable card. The failure path
    # deliberately leaves the pod up for debugging, so that OOM burned the
    # full watchdog window for zero output.
    t0 = time.monotonic()
    face_dets: list[Detection] = []
    lp_dets: list[Detection] = []
    for i in range(0, len(sampled_bgr), cfg.detect_batch):
        chunk = sampled_bgr[i:i + cfg.detect_batch]
        # POSITIONS into sampled_bgr, not clip-local frame numbers. Every
        # clip has a frame 0, so clip-local indices collide across clips —
        # which merged different clips' detections under one key and drew
        # one clip's boxes onto another clip's image in the probe dump.
        # _write_probe_frames maps positions back to (clip_id, frame_idx).
        chunk_pos = list(range(i, i + len(chunk)))
        face_dets.extend(face_det.detect_batch(chunk, chunk_pos))
        if lp_det is not None:
            lp_dets.extend(lp_det.detect_batch(chunk, chunk_pos))
    detect_s = time.monotonic() - t0
    ms_per_frame = 1000 * detect_s / max(1, len(sampled_bgr))

    # Probe frames are crops of the ORIGINAL, UNREDACTED footage — the one
    # place in this job that writes identifiable faces to disk. They must
    # never land in output_dir, which is the tree that gets synced off the
    # pod and published, and they are only worth writing when a human is
    # actually sitting there to look at them (--probe-only).
    if cfg.probe_only:
        _write_probe_frames(cfg.probe_frames_dir, sampled_bgr, sampled_idx,
                             sampled_clip, face_dets + lp_dets,
                             {"face": cfg.face_threshold, "lp": cfg.lp_threshold})
        log.warning(
            "wrote %d UNREDACTED probe frames to %s — original faces, outside "
            "the output dir on purpose. Do NOT sync or publish this directory; "
            "delete it when you are done eyeballing the detections.",
            min(len(sampled_bgr), 20), cfg.probe_frames_dir,
        )

    total_detect_frames = sum(math.ceil(c.n_frames / stride) for c in clips)
    total_frames = sum(c.n_frames for c in clips)
    detect_hours = (total_detect_frames * ms_per_frame / 1000) / 3600
    # Encode and verify are NOT measured here — this is an assumed rate, and
    # a real `-preset slow` 1080p encode runs well under it. Verify adds two
    # more full decodes plus a YuNet pass per clip. Treat the estimate as a
    # floor, which is why the watchdog check below exists as a second gate.
    est_encode_hours = total_frames / (ASSUMED_ENCODE_FPS * 3600)
    est_verify_hours = (2 * total_frames / (ASSUMED_DECODE_FPS * 3600)
                         + total_detect_frames * ASSUMED_YUNET_MS / 1000 / 3600)
    est_hours = detect_hours + est_encode_hours + est_verify_hours + 0.5
    est_usd = est_hours * cfg.gpu_rate_usd_per_hr

    # Raw counts are at the SWEEP FLOOR (0.10 by default), so most of them
    # are deliberately-collected noise that never gets redacted — reading
    # "424 face detections" as "424 faces" is wrong in the alarming
    # direction, and reading a big license-plate count on indoor footage as
    # a bug is wrong in the other (printed text on packaging scores ~0.2).
    # The above-threshold counts are the ones that answer "would this run
    # actually redact anything", so report both, side by side, always.
    n_face_hot = sum(1 for d in face_dets if d.score > cfg.face_threshold)
    n_lp_hot = sum(1 for d in lp_dets if d.score > cfg.lp_threshold)

    report = {
        "sampled_frames": len(sampled_bgr),
        "n_face_detections": len(face_dets),
        "n_lp_detections": len(lp_dets),
        "n_face_above_threshold": n_face_hot,
        "n_lp_above_threshold": n_lp_hot,
        "max_face_score": max((d.score for d in face_dets), default=0.0),
        "max_lp_score": max((d.score for d in lp_dets), default=0.0),
        "face_threshold": cfg.face_threshold,
        "lp_threshold": cfg.lp_threshold,
        "sweep_threshold": cfg.sweep_threshold,
        "ms_per_frame_detect_both_models": ms_per_frame,
        "stride": stride,
        "total_detect_frames": total_detect_frames,
        "total_frames": total_frames,
        "estimated_detect_hours": detect_hours,
        "estimated_encode_hours": est_encode_hours,
        "estimated_verify_hours": est_verify_hours,
        "estimated_hours": est_hours,
        "estimated_usd": est_usd,
        "budget_usd": cfg.budget_usd,
        "watchdog_hours": None if cfg.no_watchdog else cfg.watchdog_hours,
    }
    log.info("probe: %s", json.dumps(report, indent=2))
    if len(face_dets) == 0 and len(lp_dets) == 0:
        # A WARNING, not an abort. Zero detections has two very different
        # causes and the probe cannot tell them apart: a broken detector, or
        # footage that genuinely contains no faces and no plates. Egocentric
        # indoor footage is frequently the latter, so aborting the batch here
        # would refuse to process perfectly valid material. The per-clip
        # zero-coverage canary in build_audit() is the right place for this:
        # it forces a human look at the specific clips that redacted nothing,
        # instead of blocking everything up front on one ambiguous signal.
        log.warning(
            "probe found ZERO detections across %d frames sampled from %d "
            "clip(s). If this footage should contain faces, STOP and check "
            "the detector with --probe-only before spending GPU time — the "
            "Gen2 .inference() path is unverified (see module docstring). If "
            "the footage genuinely has no faces, this is expected; every "
            "clip that redacts nothing will still come back NEEDS_REVIEW so "
            "you confirm it by eye.",
            len(sampled_bgr), len(clips),
        )

    if not cfg.force and est_usd > cfg.budget_usd:
        raise RuntimeError(
            f"estimated ${est_usd:.2f} exceeds --budget-usd ${cfg.budget_usd:.2f}. "
            f"Pass --force to override, or lower cost (raise --detect-hz "
            f"stride implicitly via a lower rate, or reduce clip count)."
        )
    # The budget gate alone is not enough: at Gen1's documented ~2 fps the
    # estimate can come in UNDER the dollar cap while still running past the
    # watchdog, which then stops the pod mid-run — money spent, no complete
    # output. Both numbers are right here, so check them together.
    if not cfg.no_watchdog and not cfg.force and est_hours > cfg.watchdog_hours:
        raise RuntimeError(
            f"estimated {est_hours:.1f}h exceeds --watchdog-hours "
            f"{cfg.watchdog_hours:.1f}; the watchdog would stop the pod "
            f"mid-run and you would pay for a partial result. Raise "
            f"--watchdog-hours, lower --detect-hz, split the clip set, or "
            f"pass --force."
        )
    return report


def _write_probe_frames(out_dir: Path, frames_bgr: list, frame_idxs: list[int],
                         clip_ids: list[str], detections: list[Detection],
                         operating: dict, max_images: int = 20) -> None:
    """Annotated sample frames for a human to eyeball.

    WHICH frames get written is the entire value of this function, and
    taking the first N in order — what this used to do — is the worst
    available choice. Samples are collected in order, so the reviewer got
    frames 0, 3, 6 ... 57: twenty near-identical images of the first two
    seconds of the first clip. That cannot answer "does the detector find
    faces in this footage", which is the only question the probe exists to
    answer. So the budget is split: half on the highest-scoring detections
    ("is it right when it fires"), half on an even spread across the whole
    sample ("what did it walk past").

    Detections carry a POSITION into frames_bgr in .frame_idx, not a
    clip-local frame number — every clip has a frame 0, so clip-local
    indices collided across clips and drew one clip's boxes on another's
    image. clip_ids/frame_idxs map positions back for the filename.

    Boxes above the operating threshold are drawn thick and starred;
    everything else is hairline. The detectors run at the sweep floor
    (0.10), so most of what they return is deliberately sub-threshold and
    never redacted — drawing it all identically makes a healthy probe look
    alarming and buries the handful of boxes that would actually be burned
    into the video.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        return
    by_pos: dict[int, list[Detection]] = {}
    for d in detections:
        by_pos.setdefault(d.frame_idx, []).append(d)

    # Highest-scoring first — a detection that would actually be redacted
    # outranks one that wouldn't, so a real face always beats sweep noise
    # even when the noise is numerically close.
    ranked = sorted(
        by_pos,
        key=lambda p: -max((d.score + (1.0 if d.score > operating.get(d.cls, 1.0) else 0.0))
                            for d in by_pos[p]),
    )
    chosen: list[int] = ranked[:max_images // 2]

    step = max(1, len(frames_bgr) // max(1, max_images - len(chosen)))
    for p in range(0, len(frames_bgr), step):
        if len(chosen) >= max_images:
            break
        if p not in chosen:
            chosen.append(p)

    for pos in sorted(set(chosen))[:max_images]:
        img = frames_bgr[pos].copy()
        for d in by_pos.get(pos, []):
            x1, y1, x2, y2 = (int(v) for v in d.box)
            hot = d.score > operating.get(d.cls, 1.0)
            color = (0, 255, 0) if d.cls == "face" else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if hot else 1)
            cv2.putText(img, f"{d.cls} {d.score:.2f}{'*REDACTED' if hot else ''}",
                        (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2 if hot else 1)
        cv2.imwrite(
            str(out_dir / f"{clip_ids[pos]}_frame_{frame_idxs[pos]:08d}.jpg"), img)


def yuv_to_bgr(y, u, v, full_range: bool):
    """YUV planes -> BGR for the DETECTORS. Never for the encoder.

    cv2.COLOR_YUV2BGR_I420 hardcodes the BT.601 LIMITED-range inverse,
    clip(1.164 * (Y - 16)). GoPro writes yuvj420p / color_range=pc — FULL
    range — and the raw yuv420p pipe preserves that faithfully (verified:
    a lossless pc clip round-trips [0,8,16,64,128,180,235,247,255]
    unchanged). Handing those samples to COLOR_YUV2BGR_I420 destroys both
    ends of the scale: the same probe comes out [0,0,0,56,130,191,255,255,
    255]. Everything below Y=16 collapses to black and everything above
    Y=235 to white, with a 16.4% contrast stretch in between.

    That is a detection-input defect, not an output one — the encoder is
    fed the untouched planes and the published video is unaffected. It
    still matters more than it looks: EgoBlur, YuNet and the --probe-only
    JPEGs all convert through here, so the "independent second detector"
    is handed the identically corrupted image and its misses become
    CORRELATED with EgoBlur's. The audit's power comes from those two
    failing differently.

    Pre-compressing to limited range exactly inverts what OpenCV is about
    to do, and measurably round-trips to the identity.

    DO NOT apply this to the planes headed for open_encoder. Those must
    stay in their native range: converting there is the pixel-rewriting
    bug fixed earlier, and FILL_VALUE=128 is defined in plane space.
    """
    import cv2
    import numpy as np

    if full_range:
        y = (16.0 + y.astype(np.float32) * (219.0 / 255.0)).round().astype(np.uint8)
        u = (16.0 + u.astype(np.float32) * (224.0 / 255.0)).round().astype(np.uint8)
        v = (16.0 + v.astype(np.float32) * (224.0 / 255.0)).round().astype(np.uint8)
    return cv2.cvtColor(_pack_yuv420p(y, u, v), cv2.COLOR_YUV2BGR_I420)


def _pack_yuv420p(y, u, v):
    """cv2.cvtColor(..., COLOR_YUV2BGR_I420) wants one interleaved I420
    buffer, not three separate planes — repack for the conversion only;
    the y/u/v arrays used for redaction elsewhere stay separate."""
    import numpy as np
    h, w = y.shape
    return np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(
        h * 3 // 2, w
    )


# =============================================================================
# Phase 3 — detection pass (checkpointed via per-clip JSONL, resumable)
# =============================================================================


def checkpoint_fingerprint(cfg: Config, gen: str, lp_present: bool) -> dict:
    """Everything that changes what a checkpointed detection row MEANS.

    A resume that reuses rows produced under a different configuration is
    not a resume, it is a lie: detection_pass skips every frame already in
    done_frames, so a rerun with a different detector calls the model ZERO
    times, inherits the previous run's boxes, and write_manifest then
    records the NEW configuration over them. Two confirmed fail-open cases:
    adding --lp-weights-gen2 on a second pass yields a manifest claiming
    lp_checked=true for a clip no plate detector ever saw, and switching
    --gen ships one generation's boxes under the other's provenance.

    Weights are identified by name+size+mtime rather than sha256 — hashing
    420 MB twice per clip to guard a resume is not worth it, and a swapped
    file changes all three.
    """
    def wid(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            st = p.stat()
            return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            return str(p)

    face = cfg.face_weights_gen2 if gen == "2" else cfg.face_weights_gen1
    lp = cfg.lp_weights_gen2 if gen == "2" else cfg.lp_weights_gen1
    return {
        "_fingerprint": 1,
        "gen": gen,
        "face_weights": wid(face),
        "lp_weights": wid(lp) if lp_present else None,
        "lp_present": lp_present,
        "sweep_threshold": cfg.sweep_threshold,
        "nms_iou": cfg.nms_iou,
        "detect_hz": cfg.detect_hz,
        "gen2_resize_px": cfg.gen2_resize_px,
    }


def detection_pass(cfg: Config, clip: ClipInfo, face_det, lp_det, ffmpeg: str,
                    checkpoint_dir: Path, gen: str = "2") -> list[Detection]:
    """Checkpoint format: ONE JSONL line per ATTEMPTED sampled frame —
    `{"frame_idx": i, "detections": [...]}`, empty list included — not one
    line per detection. This is deliberate, fixing two real bugs a
    per-detection log has: (1) a crowded frame producing multiple lines
    means a crash between them lets resume see the frame as 'done' from
    just the first line, permanently skipping re-detection of whatever
    didn't get written — a silent missed-face risk, not just a data-loss
    one; (2) a frame with zero detections never appears in a per-detection
    log at all, so resume can't tell 'attempted, found nothing' from
    'never attempted' and always redundantly redoes it. One line per
    frame, written+flushed+fsynced as the last step of handling that
    frame, makes each frame's checkpoint atomic in practice: either the
    whole frame's line lands, or (on a crash) it doesn't, and either way
    resume's done-set is exactly correct."""
    stride = max(1, round(clip.fps / cfg.detect_hz))
    done_frames: set[int] = set()
    detections: list[Detection] = []

    # Resume checkpoint is the per-clip JSONL below. It is this stage's
    # own progress format, not an output contract — the manifest and the
    # redacted video are what downstream consumes.
    jsonl_path = checkpoint_dir / f"{clip.clip_id}.detections.jsonl"
    fingerprint = checkpoint_fingerprint(cfg, gen, lp_det is not None)

    if jsonl_path.exists():
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        # Line 0 is the config fingerprint. Anything else — a mismatch, a
        # checkpoint from before fingerprints existed, an empty file — means
        # these rows cannot be trusted to describe this configuration, and
        # reusing them would skip inference entirely while the manifest
        # claimed the new config. Discard and re-detect; that costs GPU
        # time, which is the cheap half of this trade.
        head = None
        if lines:
            try:
                head = json.loads(lines[0])
            except json.JSONDecodeError:
                head = None
        if not (isinstance(head, dict) and head.get("_fingerprint")):
            log.warning("%s: checkpoint has no config fingerprint — discarding "
                         "and re-detecting", clip.clip_id)
            jsonl_path.unlink()
            lines = []
        elif head != fingerprint:
            differing = sorted(
                k for k in set(head) | set(fingerprint)
                if head.get(k) != fingerprint.get(k))
            log.warning(
                "%s: checkpoint was produced under a DIFFERENT configuration "
                "(%s changed) — discarding it. Reusing it would skip inference "
                "entirely and record the new config over the old boxes.",
                clip.clip_id, ", ".join(differing))
            jsonl_path.unlink()
            lines = []
        else:
            lines = lines[1:]

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    # Only the LAST line can legitimately be a crash-mid-
                    # write truncation in an append-only log; anything
                    # earlier being malformed means real corruption, and
                    # should still raise rather than be silently dropped.
                    log.warning(
                        "%s: discarding truncated trailing line in %s "
                        "(crash mid-write) — that frame will be re-detected",
                        clip.clip_id, jsonl_path,
                    )
                    continue
                raise
            done_frames.add(row["frame_idx"])
            for d in row["detections"]:
                detections.append(Detection(
                    frame_idx=row["frame_idx"], cls=d["cls"],
                    box=tuple(d["box"]), score=d["score"],
                ))
        log.info("%s: resuming, %d detection-frames already attempted", clip.clip_id, len(done_frames))

    if not jsonl_path.exists():
        # Fingerprint FIRST, fsynced, before a single detection row exists —
        # otherwise a crash between the first flush() and the header leaves
        # rows with no provenance, which the resume path above (correctly)
        # throws away.
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(fingerprint) + "\n")
            f.flush()
            os.fsync(f.fileno())

    stderr_log = cfg.output_dir / "logs" / f"{clip.clip_id}.detect_decode.stderr.log"
    proc = open_decoder(ffmpeg, clip.path, stderr_log)
    batch_bgr, batch_idx = [], []
    frames_read = 0

    def flush():
        nonlocal detections
        if not batch_bgr:
            return
        new_dets = face_det.detect_batch(batch_bgr, batch_idx)
        if lp_det is not None:
            new_dets = new_dets + lp_det.detect_batch(batch_bgr, batch_idx)
        detections.extend(new_dets)
        by_frame: dict[int, list[Detection]] = {i: [] for i in batch_idx}
        for d in new_dets:
            by_frame[d.frame_idx].append(d)
        with jsonl_path.open("a", encoding="utf-8") as f:
            for i in batch_idx:
                row = {
                    "frame_idx": i,
                    "detections": [
                        {"cls": d.cls, "box": list(d.box), "score": d.score}
                        for d in by_frame[i]
                    ],
                }
                f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        batch_bgr.clear()
        batch_idx.clear()

    idx = 0
    for y, u, v in read_frames(proc, clip.width, clip.height):
        if idx % stride == 0 and idx not in done_frames:
            batch_bgr.append(yuv_to_bgr(y, u, v, clip.color_range == "pc"))
            batch_idx.append(idx)
            if len(batch_bgr) >= cfg.detect_batch:
                flush()
        idx += 1
        frames_read += 1
    flush()
    proc.wait()

    if frames_read != clip.n_frames:
        raise RuntimeError(
            f"{clip.clip_id}: decoded {frames_read} frames, ffprobe reported "
            f"nb_read_packets={clip.n_frames}. Refusing to trust this clip's "
            f"frame indexing — a mismatch here is exactly the failure mode "
            f"that produces boxes on the wrong frames."
        )
    log.info("%s: detection pass done, %d detections over %d frames",
              clip.clip_id, len(detections), frames_read)
    return detections


# =============================================================================
# Phase 4 — track building + interpolation
# =============================================================================


@dataclass(slots=True)
class Track:
    track_id: int
    cls: str
    # frame_idx -> (box, source) where source in {"det", "interp", "hold"}
    frames: dict = field(default_factory=dict)
    # Highest DETECTION frame index in `frames`. Kept as a field rather than
    # recomputed with max(tr.frames): interpolation grows that dict by one
    # entry per video frame, so scanning it once per detection frame is
    # quadratic in how long the subject stays visible.
    last_frame: int = -1


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _covered_fraction(box: tuple, fill_boxes: list[tuple]) -> float:
    """What fraction of `box` is covered by the UNION of `fill_boxes`.

    IoU is the wrong question for "did we redact this face", and it fails
    in the dangerous direction. A face at (0,0,100,100) whose right half
    only is filled by (50,0,150,100) scores IoU 0.33 — over any sane
    threshold — so half an unredacted face reads as covered. The
    symmetric error also bites: a small face fully inside a large dilated
    box scores IoU 0.007 and reads as uncovered, hard-failing the clip
    for nothing.

    Union, not per-box: a face straddling two adjacent fill rectangles is
    fully covered but overlaps neither one enough on its own.

    Exact via coordinate compression rather than sampling — overlapping
    rectangles can't just be summed. Cell count is O(n^2) in the number
    of fill boxes that touch this one, and n is a handful per frame.
    """
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 1.0  # degenerate box encloses no pixels, so nothing can leak

    clipped = []
    for bx1, by1, bx2, by2 in fill_boxes:
        cx1, cy1 = max(x1, bx1), max(y1, by1)
        cx2, cy2 = min(x2, bx2), min(y2, by2)
        if cx2 > cx1 and cy2 > cy1:
            clipped.append((cx1, cy1, cx2, cy2))
    if not clipped:
        return 0.0

    xs = sorted({x1, x2}.union(c[0] for c in clipped).union(c[2] for c in clipped))
    ys = sorted({y1, y2}.union(c[1] for c in clipped).union(c[3] for c in clipped))
    covered = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            mx = (xs[i] + xs[i + 1]) / 2
            my = (ys[j] + ys[j + 1]) / 2
            for bx1, by1, bx2, by2 in clipped:
                if bx1 <= mx <= bx2 and by1 <= my <= by2:
                    covered += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
                    break
    return covered / area


def _box_is_covered(box: tuple, fill_boxes: list[tuple]) -> bool:
    return _covered_fraction(box, fill_boxes) >= COVERAGE_MIN_FRAC


def _apply_forced_boxes(path: Path, clip: ClipInfo, fill_map: dict, *,
                         hold_frames: int, back_hold_frames: int,
                         dilate_scale: float, motion_margin_px: int) -> int:
    """Merge human-supplied "you missed this face" boxes into fill_map.

    Every failure here is loud, because this is the remediation path: the
    boxes are by definition faces a human already confirmed the detector
    missed, so a silently-skipped file republishes a known face. Previously
    a typo'd path just failed an `.exists()` test with no warning, and a
    clip_id that didn't match any key logged "applied 0 forced boxes" —
    which reads like success.

    Boxes are clamped to the frame here, once, rather than trusted by the
    three consumers downstream; an unclamped negative coordinate reaching
    check_fill_integrity()'s slice produced an empty array and a ValueError
    from np.max, which main() then recorded as a FAILED clip. A reviewer
    covering a face at the frame edge naturally draws a box that starts
    slightly off-screen.
    """
    if not path.exists():
        raise RuntimeError(
            f"--forced-boxes {path} does not exist. These are faces a human "
            f"confirmed were missed — refusing to run as though they were applied."
        )
    forced = json.loads(path.read_text(encoding="utf-8"))
    if clip.clip_id not in forced:
        log.warning(
            "%s: --forced-boxes %s has no entry for this clip (keys: %s). If you "
            "meant to cover a face here, the key must match the clip_id exactly.",
            clip.clip_id, path, sorted(forced)[:10],
        )
        return 0

    forced_dets: list[Detection] = []
    for entry in forced[clip.clip_id]:
        frame_idx = int(entry["frame_idx"])
        if not 0 <= frame_idx < clip.n_frames:
            # Loud, not a warning: an out-of-range index used to be a silent
            # no-op that still counted toward "applied N forced boxes", so a
            # typo'd frame number reported success and republished the face.
            raise RuntimeError(
                f"--forced-boxes frame_idx {frame_idx} is outside "
                f"{clip.clip_id}'s range 0..{clip.n_frames - 1}."
            )
        x1, y1, x2, y2 = (float(v) for v in entry["box"])
        cx1, cy1 = max(0.0, min(x1, x2)), max(0.0, min(y1, y2))
        cx2 = min(float(clip.width), max(x1, x2))
        cy2 = min(float(clip.height), max(y1, y2))
        if cx2 <= cx1 or cy2 <= cy1:
            log.warning("%s: forced box %s on frame %s is empty after clamping — skipped",
                         clip.clip_id, entry["box"], frame_idx)
            continue
        forced_dets.append(Detection(frame_idx=frame_idx, cls="face",
                                      box=(cx1, cy1, cx2, cy2), score=1.0))

    if not forced_dets:
        raise RuntimeError(
            f"--forced-boxes listed {clip.clip_id} but no usable box survived "
            f"clamping to {clip.width}x{clip.height}. Refusing to report a "
            f"remediation run that redacted nothing new."
        )

    # Forced boxes go through the SAME temporal machinery as detections.
    # Writing each one to its single listed frame was the critical hole: the
    # review page hands out one entry per flagged frame, and flagged frames
    # sit on the detection stride grid — so the reviewer covered frames
    # 90, 93, 96... and every frame between them still shipped the face.
    # Both checks that could have noticed sample that same stride grid, so
    # the clip then flipped NEEDS_REVIEW -> PASS_AUTOMATED with the face
    # visible on two thirds of its frames. Interpolation joins consecutive
    # forced boxes; the holds cover the ends.
    tracks = build_tracks(forced_dets, min_box_px=0, iou_thresh=TRACK_IOU_DEFAULT,
                           hold_frames=hold_frames, back_hold_frames=back_hold_frames)
    forced_map = tracks_to_fill_map(tracks, clip.width, clip.height,
                                     dilate_scale, motion_margin_px)
    for frame_idx, boxes in forced_map.items():
        fill_map.setdefault(frame_idx, []).extend(boxes)

    log.info("%s: %d forced box(es) expanded to %d covered frame(s)",
              clip.clip_id, len(forced_dets), len(forced_map))
    return len(forced_dets)


def build_tracks(detections: list[Detection], min_box_px: int, iou_thresh: float,
                  hold_frames: int, back_hold_frames: int = 0,
                  dropped_small: list | None = None) -> list[Track]:
    """Greedy nearest-IoU association across consecutive DETECTION frames
    (not every video frame — detections are already sparse at detect_hz),
    then linear interpolation across the gap between consecutive detection
    frames of one track, then a hold for `hold_frames` video-frames past
    the last real detection and `back_hold_frames` before the first. This —
    not re-detection — is where most of the actual recall against missed
    frames comes from: a face seen at detection-frame k-1 and k+1 stays
    covered at every frame in between even when the detector missed the
    frame(s) that would have covered them directly.

    back_hold_frames defaults (in process_clip) to the SAME value as
    hold_frames. It was one detection stride, on the reasoning that it only
    had to bridge the sampling gap; measured against real footage, a face
    entering frame goes uncovered from the moment it appears until the
    detector first scores it above threshold, which was 5-25 frames in 8 of
    13 inspected misses. Nothing downstream catches that: fill_integrity
    only inspects frames the map already claims, and YuNet samples the same
    stride grid."""
    if dropped_small is None:
        dropped_small = []
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        w = d.box[2] - d.box[0]
        h = d.box[3] - d.box[1]
        if w < min_box_px or h < min_box_px:
            # Dropped here means never redacted AND never audited: the sweep
            # only inspects the sub-operating-threshold band, and integrity
            # only inspects boxes already in the fill_map. A confident
            # detection of a small or edge-truncated face would vanish
            # silently, so the caller is handed the count to gate on.
            dropped_small.append(d)
            continue
        by_frame.setdefault(d.frame_idx, []).append(d)

    frame_order = sorted(by_frame)
    tracks: list[Track] = []
    next_id = 0
    # keyed by (cls, box) isn't enough for multi-face frames; keep a small
    # per-class list of "active" tracks and match by best IoU each step.
    active: dict[str, list[Track]] = {"face": [], "lp": []}

    for frame_idx in frame_order:
        dets = by_frame[frame_idx]
        by_cls: dict[str, list[Detection]] = {}
        for d in dets:
            by_cls.setdefault(d.cls, []).append(d)

        for cls in ("face", "lp"):
            cur_dets = by_cls.get(cls, [])

            # Evict tracks unmatched for more than hold_frames BEFORE
            # matching this frame. Without this, a track that lost its
            # subject stays "active" forever and can silently reattach to
            # an entirely unrelated face/plate arbitrarily far in the
            # future purely because the two happen to overlap in screen
            # space (very plausible with a semi-fixed egocentric framing
            # — a doorway, a desk — reused by different people) —
            # _interpolate() would then draw a straight-line blend across
            # the whole gap between two different people. Reusing
            # hold_frames as the max gap keeps this consistent with the
            # forward-hold semantics below: a track's identity is trusted
            # for exactly as long as its coverage is trusted.
            #
            # max_gap is at LEAST one detection stride. Using hold_frames
            # alone silently breaks tracking whenever stride > hold_frames
            # (e.g. --detect-hz 0.5 on 30fps gives stride 60 vs the auto
            # hold of 30): every track is evicted before it can ever match
            # its second detection, so interpolation never runs and the
            # frames between two sightings of the SAME stationary face go
            # unredacted — with no warning and a PASS status.
            #
            # tr.last_frame, not max(tr.frames): _interpolate() grows that
            # dict by `stride` entries per detection frame, so re-scanning
            # it here and again below is quadratic in track lifetime — for
            # one subject visible through a 1.5h clip that measured out to
            # ~8.7e9 key visits, minutes of pure Python with the GPU idle.
            max_gap = max(hold_frames, back_hold_frames)
            active[cls] = [tr for tr in active[cls]
                           if frame_idx - tr.last_frame <= max_gap]

            used = set()
            for tr in active[cls]:
                last_frame = tr.last_frame
                last_box = tr.frames[last_frame][0]
                best_iou, best_j = 0.0, -1
                for j, d in enumerate(cur_dets):
                    if j in used:
                        continue
                    iou = _iou(last_box, d.box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thresh:
                    d = cur_dets[best_j]
                    used.add(best_j)
                    if frame_idx > last_frame + 1:
                        _interpolate(tr, last_frame, last_box, frame_idx, d.box)
                    tr.frames[frame_idx] = (d.box, "det")
                    tr.last_frame = frame_idx
                # An unmatched track stays active until max_gap evicts it
                # above; the previous rebuild-into-still_active was a no-op
                # (both branches appended) and is gone.

            for j, d in enumerate(cur_dets):
                if j in used:
                    continue
                tr = Track(track_id=next_id, cls=cls, last_frame=frame_idx)
                next_id += 1
                tr.frames[frame_idx] = (d.box, "det")
                tracks.append(tr)
                active[cls].append(tr)

    # Hold FORWARD past the last detection, and BACKWARD before the first.
    # The backward hold is not symmetry for its own sake: a face entering
    # frame is visible — and published unredacted — from the moment it
    # appears until the detector first scores it above threshold. That used
    # to be modeled as one detection stride (the sampling gap only); measured
    # against real footage the actual gap was 5-26 frames, because detection
    # LATENCY (small/angled/partly-occluded face, not yet confident) matters
    # far more than sampling gap. Nothing downstream can catch a miss here:
    # fill_integrity only inspects frames the fill_map already claims, and
    # YuNet samples on the same stride, so it looks at the very frame that
    # IS covered.
    for tr in tracks:
        last_frame = max(tr.frames)
        last_box, _ = tr.frames[last_frame]
        for f in range(last_frame + 1, last_frame + 1 + hold_frames):
            tr.frames.setdefault(f, (last_box, "hold"))

        first_frame = min(tr.frames)
        first_box, _ = tr.frames[first_frame]
        for f in range(max(0, first_frame - back_hold_frames), first_frame):
            tr.frames.setdefault(f, (first_box, "hold"))

    return tracks


def _interpolate(tr: Track, f0: int, box0: tuple, f1: int, box1: tuple) -> None:
    span = f1 - f0
    for f in range(f0 + 1, f1):
        t = (f - f0) / span
        box = tuple(box0[k] + t * (box1[k] - box0[k]) for k in range(4))
        tr.frames[f] = (box, "interp")


def dilate_box(box: tuple, scale: float, margin_px: int, w: int, h: int) -> tuple:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale + 2 * margin_px, (y2 - y1) * scale + 2 * margin_px
    nx1, ny1 = cx - bw / 2, cy - bh / 2
    nx2, ny2 = cx + bw / 2, cy + bh / 2
    return (max(0.0, nx1), max(0.0, ny1), min(float(w), nx2), min(float(h), ny2))


def tracks_to_fill_map(tracks: list[Track], w: int, h: int, dilate_scale: float,
                        margin_px: int) -> dict:
    fill_map: dict[int, list[tuple]] = {}
    for tr in tracks:
        for frame_idx, (box, _source) in tr.frames.items():
            dbox = dilate_box(box, dilate_scale, margin_px, w, h)
            fill_map.setdefault(frame_idx, []).append(dbox)
    return fill_map


# =============================================================================
# Phase 5 — redact + encode (one burn-in re-encode)
# =============================================================================


def redact_frame_inplace(y, u, v, boxes: list[tuple], mode: str) -> bool:
    """Returns whether anything was actually redacted on this frame (for
    counting). mode='fill' sets an exact mid-gray — see FILL_VALUE and
    check_fill_integrity(). mode='blur' is a best-effort alternative kept
    for completeness; fill is the recommended default (re-identification
    risk — Revelio reports 95.9% re-ID at a blur kernel 32% of face
    width — and it turns verification into an exact invariant instead of
    a variance heuristic)."""
    h, w = y.shape
    touched = False
    for x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        touched = True
        if mode == "fill":
            y[iy1:iy2, ix1:ix2] = FILL_VALUE
            # Chroma is half-resolution: the chroma columns/rows that
            # fully COVER luma range [a, b) are [a//2, ceil(b/2)), i.e.
            # floor on the start, CEILING on the end. Using floor on both
            # ends (iy2//2 / ix2//2) under-covers by one chroma unit
            # whenever ix2/iy2 is odd — leaves a ~2-luma-pixel strip at
            # the box's right/bottom edge with original (non-gray) hue
            # under gray luma. (ix2+1)//2 is integer ceiling division.
            cy1, cx1 = iy1 // 2, ix1 // 2
            cy2, cx2 = max(cy1 + 1, (iy2 + 1) // 2), max(cx1 + 1, (ix2 + 1) // 2)
            u[cy1:cy2, cx1:cx2] = FILL_VALUE
            v[cy1:cy2, cx1:cx2] = FILL_VALUE
        else:
            import cv2
            region = y[iy1:iy2, ix1:ix2]
            k = max(3, (min(region.shape) // 2) | 1)  # odd kernel
            y[iy1:iy2, ix1:ix2] = cv2.GaussianBlur(region, (k, k), 0)
            # Chroma must be blurred too. Blurring luma alone leaves the
            # face's original hue and saturation intact at full chroma
            # resolution under a soft-looking grey — which is exactly the
            # leak check_fill_integrity()'s docstring describes, and blur
            # mode has no integrity check to catch it.
            cy1, cx1 = iy1 // 2, ix1 // 2
            cy2, cx2 = max(cy1 + 1, (iy2 + 1) // 2), max(cx1 + 1, (ix2 + 1) // 2)
            for plane in (u, v):
                creg = plane[cy1:cy2, cx1:cx2]
                ck = max(3, (min(creg.shape) // 2) | 1)
                plane[cy1:cy2, cx1:cx2] = cv2.GaussianBlur(creg, (ck, ck), 0)
    return touched


def redact_and_encode(cfg: Config, clip: ClipInfo, fill_map: dict, ffmpeg: str,
                       out_path: Path) -> dict:
    dec_log = cfg.output_dir / "logs" / f"{clip.clip_id}.redact_decode.stderr.log"
    enc_log = cfg.output_dir / "logs" / f"{clip.clip_id}.encode.stderr.log"
    dec = open_decoder(ffmpeg, clip.path, dec_log)
    enc = open_encoder(ffmpeg, clip.width, clip.height, clip.fps, out_path,
                        cfg.encode_preset, cfg.encode_crf, enc_log,
                        color_primaries=clip.color_primaries, color_trc=clip.color_trc,
                        colorspace=clip.colorspace, color_range=clip.color_range)
    assert enc.stdin is not None
    n_frames_with_fill = 0
    max_area_frac = 0.0
    frame_area = clip.width * clip.height
    frames_written = 0
    try:
        for idx, (y, u, v) in enumerate(read_frames(dec, clip.width, clip.height)):
            boxes = fill_map.get(idx, [])
            if boxes:
                touched = redact_frame_inplace(y, u, v, boxes, cfg.redaction)
                if touched:
                    n_frames_with_fill += 1
                    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
                    max_area_frac = max(max_area_frac, area / frame_area)
            enc.stdin.write(y.tobytes())
            enc.stdin.write(u.tobytes())
            enc.stdin.write(v.tobytes())
            frames_written += 1
    finally:
        # If the loop above was abandoned mid-iteration (any exception —
        # a dead encoder raising BrokenPipeError, a bug in
        # redact_frame_inplace), `dec` is still decoding and trying to
        # write more frames to a stdout pipe nobody is reading anymore.
        # terminate() BEFORE wait() unblocks that write so wait() can't
        # hang forever; harmless to call on a decoder that already hit
        # EOF normally (it's a zombie by then, terminate() is a no-op).
        try:
            enc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        dec.terminate()
        if dec.stdout is not None:
            dec.stdout.close()
        dec.wait()
        enc.wait()

    if enc.returncode != 0:
        stderr = enc_log.read_text(errors="replace") if enc_log.exists() else ""
        raise RuntimeError(f"{clip.clip_id}: encode failed (see {enc_log}): {stderr}")
    if frames_written != clip.n_frames:
        raise RuntimeError(
            f"{clip.clip_id}: wrote {frames_written} frames, expected "
            f"{clip.n_frames} — do not ship a mismatched output."
        )
    return {
        "n_frames_with_fill": n_frames_with_fill,
        "frames_with_fill_frac": n_frames_with_fill / max(1, frames_written),
        "max_fill_area_frac": max_area_frac,
    }


# =============================================================================
# Phase 6 — verify
# =============================================================================


def check_fill_integrity(ffmpeg: str, out_path: Path, w: int, h: int,
                          fill_map: dict, mode: str, stderr_log: Path,
                          n_frames: int) -> dict:
    """Only meaningful for mode='fill': every redacted region is a
    constant value, so this is an EXACT invariant, not a heuristic
    threshold on variance-of-Laplacian. Erode 2px to skip DCT ringing at
    box edges. Catches frame-index desync directly: if boxes are on the
    wrong frame, gray will not be where fill_map says it is.

    Checks ALL THREE planes (Y, U, V), not just luma — a box can have
    perfectly gray luma while still carrying its original hue/saturation
    in chroma if only Y got redacted (redact_frame_inplace() sets all
    three, but this check exists specifically to catch cases where the
    two drift, including a future refactor that touches one and not the
    other). A box that erodes to empty (small boxes) falls back to being
    checked UNERODED rather than silently exempted — counted separately
    as fill_integrity_unverifiable so a gap in coverage is visible in the
    audit instead of invisible."""
    if mode != "fill":
        return {"integrity_skipped": f"mode={mode} has no exact-invariant check"}
    import numpy as np

    proc = open_decoder(ffmpeg, out_path, stderr_log)
    violations = 0
    checked = 0
    unverifiable = 0
    chroma_unverifiable = 0
    frames_seen = 0
    for idx, (y, u, v) in enumerate(read_frames(proc, w, h)):
        frames_seen = idx + 1
        boxes = fill_map.get(idx)
        if not boxes:
            continue
        for x1, y1, x2, y2 in boxes:
            # Clamp exactly as redact_frame_inplace does. fill_map boxes from
            # dilate_box() are already in range, but --forced-boxes entries
            # reach here raw; a negative coord makes numpy wrap to a high
            # index, the slice comes back empty, and np.max raises
            # "zero-size array to reduction operation" — recorded as a
            # FAILED clip on the human remediation path.
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w, int(x2)), min(h, int(y2))
            eix1, eiy1 = ix1 + LUMA_ERODE_PX, iy1 + LUMA_ERODE_PX
            eix2, eiy2 = ix2 - LUMA_ERODE_PX, iy2 - LUMA_ERODE_PX
            if eix2 <= eix1 or eiy2 <= eiy1:
                eix1, eiy1, eix2, eiy2 = ix1, iy1, ix2, iy2  # too small to erode
            if eix2 <= eix1 or eiy2 <= eiy1:
                unverifiable += 1  # genuinely zero-area box; nothing to check
                continue
            y_region = y[eiy1:eiy2, eix1:eix2].astype(np.int16)

            # Chroma is eroded in its OWN samples, by a measured amount (see
            # CHROMA_ERODE_SAMPLES). Deriving the chroma window from the
            # luma-eroded box instead only buys one chroma sample of margin,
            # which is a third of the ringing depth — that is what made every
            # correctly-redacted large box report a violation. A box too
            # small to erode this far has no trustworthy chroma interior at
            # all, so its chroma is left unchecked and counted, never faked.
            cx1 = ix1 // 2 + CHROMA_ERODE_SAMPLES
            cy1 = iy1 // 2 + CHROMA_ERODE_SAMPLES
            cx2 = ix2 // 2 - CHROMA_ERODE_SAMPLES
            cy2 = iy2 // 2 - CHROMA_ERODE_SAMPLES
            if cx2 > cx1 and cy2 > cy1:
                u_region = u[cy1:cy2, cx1:cx2].astype(np.int16)
                v_region = v[cy1:cy2, cx1:cx2].astype(np.int16)
            else:
                u_region = v_region = None
                chroma_unverifiable += 1
            checked += 1
            planes_ok = all(
                np.max(np.abs(region - FILL_VALUE)) <= FILL_MAX_DEVIATION
                and np.std(region) <= FILL_MAX_STD
                for region in (y_region, u_region, v_region)
                if region is not None and region.size
            )
            if not planes_ok:
                violations += 1
    proc.wait()
    # Fail CLOSED. A decode that dies or truncates yields zero frames, the
    # loop body never runs, and violations stays 0 — which reads exactly
    # like "verified clean". detection_pass() already guards this same risk
    # with a frames_read assertion; the two checks that actually gate the
    # privacy claim were the two without it.
    if proc.returncode != 0:
        raise RuntimeError(
            f"integrity decode of {out_path} exited {proc.returncode} "
            f"(see {stderr_log}) — refusing to report it as verified."
        )
    if frames_seen != n_frames:
        raise RuntimeError(
            f"integrity decode of {out_path} yielded {frames_seen} frames, "
            f"expected {n_frames}. A short decode reads as zero violations, "
            f"so this cannot be treated as a clean verification."
        )
    return {
        "fill_integrity_checked": checked,
        "fill_integrity_violations": violations,
        "fill_integrity_unverifiable": unverifiable,
        "fill_integrity_chroma_unverifiable": chroma_unverifiable,
        "fill_integrity_frames": frames_seen,
    }


def check_low_threshold_sweep(detections: list[Detection], fill_map: dict,
                               sweep_threshold: float, operating_thresholds: dict) -> dict:
    """The check with real statistical power against false negatives — it
    uses ORIGINAL pixels. Everything scored in [sweep_threshold,
    operating_threshold) that isn't already covered by a fill box is a
    candidate miss for human review."""
    candidates = []
    for d in detections:
        op_thresh = operating_thresholds.get(d.cls, 0.5)
        if not (sweep_threshold <= d.score < op_thresh):
            continue
        boxes = fill_map.get(d.frame_idx, [])
        covered = _box_is_covered(d.box, boxes)
        if not covered:
            candidates.append(d)
    # Rank WITHIN a class by how close the score came to that class's own
    # operating threshold, then round-robin across classes into the cap.
    #
    # A single absolute-score sort across both classes is wrong in the
    # privacy direction and silently so: the face band is [sweep, 0.30) and
    # the plate band is [sweep, 0.40), so EVERY plate candidate scoring 0.30
    # or better outranks EVERY face candidate that can possibly exist. On
    # egocentric footage where the plate model fires on printed packaging
    # text, that filled all 200 slots with cardboard and handed the reviewer
    # not one face to look at — while n_candidate_misses honestly reported
    # thousands. Round-robin guarantees faces half the sample whenever any
    # exist, and costs plates nothing when they don't.
    by_cls: dict[str, list[Detection]] = {}
    for d in candidates:
        by_cls.setdefault(d.cls, []).append(d)
    for cls, pool in by_cls.items():
        op = operating_thresholds.get(cls, 0.5) or 1.0
        pool.sort(key=lambda d: -(d.score / op))

    ranked: list[Detection] = []
    pools = [by_cls.get("face", []), by_cls.get("lp", [])]
    depth = 0
    while len(ranked) < AUDIT_MAX_ITEMS and any(len(p) > depth for p in pools):
        for pool in pools:
            if depth < len(pool) and len(ranked) < AUDIT_MAX_ITEMS:
                ranked.append(pool[depth])
        depth += 1

    return {
        "n_candidate_misses": len(candidates),
        "n_candidate_misses_face": len(by_cls.get("face", [])),
        "n_candidate_misses_lp": len(by_cls.get("lp", [])),
        "n_frames_with_candidate_miss": len({d.frame_idx for d in candidates}),
        # The embedded list is capped, the COUNT above is not. The review
        # page reads the list; without an explicit truncation marker it
        # would show 200 and call it the total, hiding the rest from the
        # only human who looks.
        "candidates": [dataclasses.asdict(d) for d in ranked],
        "candidates_truncated": max(0, len(candidates) - AUDIT_MAX_ITEMS),
    }


def check_yunet(cfg: Config, ffmpeg: str, out_path: Path, w: int, h: int,
                 fill_map: dict, detect_hz: float, fps: float, n_frames: int,
                 full_range: bool = False) -> dict:
    """Second, independent detector — different training distribution
    (WIDER FACE, not egocentric) and inductive biases from EgoBlur, so it
    catches a different slice of misses. MIT-licensed, ships inside
    opencv-python's bindings — but the ONNX weights are a separate
    download this script does not fetch; pass --yunet-model or this check
    is honestly skipped rather than silently faked."""
    if cfg.yunet_model is None:
        log.warning(
            "%s: --yunet-model not provided — the independent second-"
            "detector check is SKIPPED. This is the only check with power "
            "against faces EgoBlur itself is systematically blind to; "
            "running without it means the audit's only real signal against "
            "missed faces is the low-threshold sweep alone.",
            out_path.stem,
        )
        return {"yunet_skipped": "no --yunet-model provided"}
    import cv2

    # score_threshold MUST be passed. OpenCV defaults it to 0.9, so the
    # YUNET_SCORE_MIN filter below never gated anything and the check ran
    # far stricter than intended — silently missing exactly the
    # low-confidence second-opinion hits it exists to surface.
    detector = cv2.FaceDetectorYN.create(
        str(cfg.yunet_model), "", (w, h), YUNET_SCORE_MIN, cfg.nms_iou)
    if abs(detector.getScoreThreshold() - YUNET_SCORE_MIN) > 1e-6:
        raise RuntimeError(
            f"YuNet score threshold is {detector.getScoreThreshold()}, expected "
            f"{YUNET_SCORE_MIN} — refusing to run a check whose sensitivity is "
            f"not what the audit reports."
        )
    stride = max(1, round(fps / detect_hz))
    stderr_log = cfg.output_dir / "logs" / f"{out_path.stem}.yunet_decode.stderr.log"
    proc = open_decoder(ffmpeg, out_path, stderr_log)
    uncovered = []
    frames_seen = 0
    for idx, (y, u, v) in enumerate(read_frames(proc, w, h)):
        frames_seen = idx + 1
        if idx % stride != 0:
            continue
        bgr = yuv_to_bgr(y, u, v, full_range)
        _n, faces = detector.detect(bgr)
        if faces is None:
            continue
        boxes_here = fill_map.get(idx, [])
        for f in faces:
            # cv2.FaceDetectorYN.detect returns [n, 15]: 0-1 bbox xy, 2-3 wh,
            # 4-5 RIGHT EYE xy, 6-7 left eye, 8-9 nose, 10-13 mouth corners,
            # 14 face score. Reading f[4] as the score (an earlier bug here)
            # compares an eye's pixel x against 0.5 — always true — so the
            # confidence gate never fired and a pixel coordinate was recorded
            # and displayed as "score" all the way into the review page.
            score = float(f[YUNET_SCORE_COL])
            if score < YUNET_SCORE_MIN:
                continue
            box = (float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3]))
            if not _box_is_covered(box, boxes_here):
                uncovered.append({"frame_idx": idx, "box": box, "score": score})
    proc.wait()
    # Fail closed, same reasoning as check_fill_integrity: a dead decode
    # produces zero uncovered faces, which is indistinguishable from a
    # genuinely clean pass.
    if proc.returncode != 0:
        raise RuntimeError(
            f"YuNet decode of {out_path} exited {proc.returncode} "
            f"(see {stderr_log}) — refusing to report it as verified."
        )
    if frames_seen != n_frames:
        raise RuntimeError(
            f"YuNet decode of {out_path} yielded {frames_seen} frames, "
            f"expected {n_frames} — a short decode reads as zero findings."
        )
    return {
        "n_yunet_uncovered": len(uncovered),
        "yunet_uncovered": uncovered[:AUDIT_MAX_ITEMS],
        "yunet_truncated": max(0, len(uncovered) - AUDIT_MAX_ITEMS),
        "yunet_frames": frames_seen,
    }


def build_audit(clip: ClipInfo, fill_stats: dict, integrity: dict, sweep: dict,
                 yunet: dict, gen: str, n_dropped_small: int = 0,
                 lp_checked: bool = True, n_face_fill_frames: int | None = None) -> dict:
    """status/hard_fail gate on EVERY check with actual power against a
    missed face, not just fill_integrity. fill_integrity only proves boxes
    that already exist are correctly gray — it is structurally incapable
    of catching a face the detector never boxed at all. The low-threshold
    sweep and YuNet are the two checks that use signal OTHER than 'did the
    fill match the fill_map', so a nonzero count from either is a hard
    fail here too, by design biased toward over-flagging (the sweep WILL
    have false positives) rather than ever silently reporting PASS on an
    unreviewed candidate miss.

    A check that did not RUN is never treated as a check that passed. The
    integrity check is skipped entirely under --redaction blur, and YuNet
    is skipped without --yunet-model; both previously contributed 0 to
    hard_fail via .get(..., 0), so a blur-mode run with no YuNet model
    could report PASS having verified literally nothing.

    The zero-coverage canary is the other half. Every term above is a
    count, and every count is 0 when the detector returns NOTHING — so a
    run where detection silently produced no boxes at all (a wrong dtype,
    a wrong class mapping, a dead model) redacted nothing, verified
    nothing, and reported PASS. `max_fill_area_frac` guarded against too
    MUCH fill; nothing guarded against none."""
    yunet_ran = "yunet_skipped" not in yunet
    integrity_ran = "integrity_skipped" not in integrity

    reasons = []
    if integrity.get("fill_integrity_violations", 0) > 0:
        reasons.append("fill_integrity_violations > 0")
    if sweep.get("n_candidate_misses", 0) > 0:
        reasons.append("low-threshold sweep found uncovered candidates")
    if yunet_ran and yunet.get("n_yunet_uncovered", 0) > 0:
        reasons.append("YuNet found uncovered faces")
    if not integrity_ran:
        reasons.append(f"integrity check did not run ({integrity.get('integrity_skipped')})")
    # Class-aware when the caller supplies it. n_frames_with_fill counts
    # frames where ANY box landed, so on footage where the plate model fires
    # on printed packaging a single lp box silenced this canary for the whole
    # clip — while the face model could have returned nothing at all.
    n_redacted = (fill_stats.get("n_frames_with_fill", 0) if n_face_fill_frames is None
                  else n_face_fill_frames)
    if n_redacted == 0 and clip.n_frames > 0:
        # Still always a human look — "we redacted nothing" must never
        # auto-pass. But say WHY it is suspicious, because on footage that
        # genuinely contains no faces this will fire on most clips, and a
        # reviewer who sees the same undifferentiated alarm every time stops
        # reading it. An independent detector agreeing there is nothing here
        # is real corroborating evidence and is worth saying out loud.
        if yunet_ran and yunet.get("n_yunet_uncovered", 0) == 0:
            reasons.append(
                "ZERO frames had a FACE redacted — but YuNet independently "
                "found no faces either, so this is consistent with genuinely "
                "faceless footage. Confirm by eye (the timelapse is enough).")
        else:
            reasons.append(
                "ZERO frames had a FACE redacted and nothing corroborates "
                "that — detection may have failed silently. Check before "
                "shipping.")
    if integrity_ran and integrity.get("fill_integrity_checked", 0) == 0 \
            and fill_stats.get("n_frames_with_fill", 0) > 0:
        reasons.append("integrity verified zero boxes despite a non-empty fill_map")
    if n_dropped_small > 0:
        # Above the operating threshold but below --min-box-px: the detector
        # was confident and we redacted nothing, and no other check can see
        # it. Small/edge-truncated faces are the dominant miss case in
        # egocentric footage, so this is a human decision, not a silent drop.
        reasons.append(
            f"{n_dropped_small} confident detection(s) discarded by --min-box-px "
            f"and never redacted or audited")

    status = "NEEDS_REVIEW" if reasons else "PASS_AUTOMATED"
    if status == "PASS_AUTOMATED" and not yunet_ran:
        status = "PASS_AUTOMATED_NO_YUNET"  # a real but weaker claim — say so
    return {
        "clip_id": clip.clip_id,
        "status": status,
        "status_reasons": reasons,
        # Explicit, not inferred from a shared "skipped" key: integrity and
        # yunet each had one, for unrelated reasons, and they collided when
        # flattened into this dict.
        "yunet_ran": yunet_ran,
        "integrity_ran": integrity_ran,
        "n_dropped_small": n_dropped_small,
        # Frames with a FACE box specifically, as distinct from
        # n_frames_with_fill which counts any class. None when the caller
        # did not compute it (the canary then falls back to the any-class
        # count, which is weaker — see above).
        "n_face_fill_frames": n_face_fill_frames,
        # False means no plate detector ran at all. Recorded so a face-only
        # run is never later read as "this clip has no license plates".
        "lp_checked": lp_checked,
        "gen": gen,
        "note": (
            "Re-running the detector on the redacted output cannot find a "
            "face the detector missed in pass 1 — those pixels are "
            "unchanged, so the same detector at the same threshold misses "
            "them again by construction. fill_integrity_violations gates "
            "status but is NOT sufficient alone: it can only prove that "
            "boxes which already exist are grey. The checks with real power "
            "against a face that was never boxed are the low-threshold "
            "sweep on the ORIGINAL pixels and the independent YuNet pass. A "
            "check that did not run counts as a failure, not a pass, and a "
            "clip where nothing at all was redacted is always NEEDS_REVIEW."
        ),
        **fill_stats,
        **integrity,
        **sweep,
        **yunet,
    }


def write_audit_summary(audit: dict, clip: ClipInfo, path: Path) -> None:
    # status_reasons FIRST, before the metrics. build_audit has gate reasons
    # that no printed metric backs (n_dropped_small, "integrity verified zero
    # boxes despite a non-empty fill_map"), so a NEEDS_REVIEW clip could show
    # an all-clean metric block and give the reviewer nothing to act on —
    # which teaches them the status line is noise.
    reasons = audit.get("status_reasons") or []
    lines = [
        f"# {clip.clip_id} — {audit['status']}",
        "",
    ]
    if reasons:
        lines += ["## Why this needs a look", ""]
        lines += [f"{i}. {r}" for i, r in enumerate(reasons, 1)]
        lines += [""]
    lines += [
        f"frames {clip.n_frames}  fps {clip.fps:.2f}  gen {audit['gen']}",
        f"frames_with_fill: {audit.get('n_frames_with_fill', 0)} "
        f"({audit.get('frames_with_fill_frac', 0) * 100:.1f}%)",
        f"fill_integrity_violations: {audit.get('fill_integrity_violations', 'n/a')}  (hard gate — desync/leak proof, must be 0)",
        f"fill_integrity_unverifiable: {audit.get('fill_integrity_unverifiable', 'n/a')}  (boxes too small to check even unerroded)",
        f"candidate_misses [sweep_threshold, operating): {audit.get('n_candidate_misses', 'n/a')}  (hard gate — review these)",
        f"yunet_uncovered: {audit.get('n_yunet_uncovered', 'n/a') if audit.get('yunet_ran') else 'SKIPPED — see note'}  (hard gate when run — review these first)",
        f"max_fill_area_frac: {audit.get('max_fill_area_frac', 0):.4f}  (runaway false-positive canary)",
        "",
        audit["note"],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Phase 7 — ship & shutdown
# =============================================================================


def write_manifest(clip: ClipInfo, gen: str, cfg: Config, out_path: Path,
                    audit: dict, timing: dict, manifest_path: Path, *,
                    hold_frames: int, back_hold_frames: int) -> None:
    """hold_frames/back_hold_frames are RESOLVED values (see resolve_holds),
    not cfg.hold_frames/cfg.back_hold_frames directly. Both raw fields carry
    a 0 = "auto" sentinel — cfg.back_hold_frames in particular defaults to
    0 on every run that doesn't pass --back-hold-frames explicitly, which is
    the common case since it exists to override the auto-symmetric default,
    not to be set every time. Recording the raw cfg value here would print
    back_hold_frames: 0 into every manifest whose track building actually
    used a real, non-zero backward hold — silently contradicting the exact
    parameter this field exists to make legible after the fact."""
    import torch

    manifest = {
        "schema_version": 1,
        "run_id": cfg.run_id,
        "clip_id": clip.clip_id,
        "clip_group": clip.clip_group,
        "chapter_index": clip.chapter_index,
        "source": {
            "filename": clip.path.name,
            "sha256": clip.sha256,
            "width": clip.width, "height": clip.height, "fps": clip.fps,
            "n_frames": clip.n_frames, "duration_s": clip.duration_s,
            "rotation": clip.rotation,
        },
        "egoblur": {
            "gen": gen,
            "face_threshold": cfg.face_threshold, "lp_threshold": cfg.lp_threshold,
            "sweep_threshold": cfg.sweep_threshold, "nms_iou": cfg.nms_iou,
            "detect_hz": cfg.detect_hz, "redaction": cfg.redaction,
            "lp_checked": cfg.lp_weights_gen2 is not None or cfg.lp_weights_gen1 is not None,
            "dilate_scale": cfg.dilate_scale, "motion_margin_px": cfg.motion_margin_px,
            "hold_frames": hold_frames, "back_hold_frames": back_hold_frames,
            "min_box_px": cfg.min_box_px,
            # None = native resolution. Recorded because it changes the scale
            # the model runs at, and therefore what a score MEANS — two runs
            # at different values are not comparable.
            "gen2_resize_px": cfg.gen2_resize_px,
            # Batching only affects throughput, not results — NOT in the
            # checkpoint fingerprint on purpose, so resuming across a GPU
            # swap (this pod's whole situation right now) still works.
            "detect_batch": cfg.detect_batch,
        },
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "python": sys.version,
        },
        "output": {
            "path": out_path.name,
            "sha256": sha256_file(out_path),
            "bytes": out_path.stat().st_size,
        },
        "audit": audit,
        "timing": timing,
        "status": audit["status"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def shutdown_pod() -> None:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        log.warning("RUNPOD_POD_ID not set — skipping shutdown (not on a pod?)")
        return
    log.info("clean exit — stopping pod %s (stop, not terminate: logs survive)", pod_id)
    subprocess.run(["runpodctl", "pod", "stop", pod_id], check=False)


# =============================================================================
# main
# =============================================================================


def resolve_holds(cfg: Config, fps: float) -> tuple[int, int]:
    """(forward, backward) hold in frames.

    A function rather than two lines inside process_clip so the defaulting
    can be tested without standing up a decoder, a GPU and a video. The
    backward value was hardcoded to one detection stride — a tenth of the
    forward hold — which left a face unredacted from the moment it entered
    frame until the detector first scored it. On real footage that gap was
    5-26 frames against a backward hold of 3, and nothing downstream could
    see it: fill_integrity only inspects frames the map already claims, and
    YuNet samples the same stride grid, so it looks at the very frame that
    IS covered.
    """
    forward = cfg.hold_frames or round(fps)
    backward = cfg.back_hold_frames or forward
    return forward, backward


def process_clip(cfg: Config, clip: ClipInfo, gen: str, face_det, lp_det,
                  ffmpeg: str, checkpoint_dir: Path) -> dict:
    manifest_path = cfg.output_dir / f"{clip.clip_id}.manifest.json"
    if manifest_path.exists() and not cfg.force_reprocess:
        # A restart after clip N crashed must not silently re-decode +
        # re-encode + re-verify clips 1..N-1 that already shipped —
        # detection_pass()'s own jsonl checkpoint is cheap to redo, but
        # redact_and_encode()/check_fill_integrity()/check_yunet() have no
        # such checkpoint and would otherwise unconditionally overwrite an
        # already-audited, possibly already-delivered output file.
        try:
            existing = json.loads(manifest_path.read_text())
            cached_audit = existing["audit"]
        except (json.JSONDecodeError, KeyError) as e:
            # A manifest torn by a kill mid-write would otherwise make this
            # clip permanently FAILED: the file still exists, so every
            # subsequent run re-parses it and re-raises, even though the
            # .blurred.mp4 beside it may be perfectly good. Reprocess
            # instead of getting stuck.
            log.warning("%s: manifest exists but is unreadable (%s) — reprocessing",
                         clip.clip_id, e)
        else:
            log.info("%s: manifest already exists (status=%s) — skipping, pass "
                      "--force-reprocess to redo", clip.clip_id, existing.get("status"))
            return cached_audit

    t0 = time.monotonic()
    log.info("=== %s ===", clip.clip_id)

    detections = detection_pass(cfg, clip, face_det, lp_det, ffmpeg,
                                 checkpoint_dir, gen)
    t_detect = time.monotonic()

    # Detectors ran down to cfg.sweep_threshold (see build_detectors). Only
    # the above-operating-threshold subset drives redaction; the full list —
    # including the low-confidence band — goes to the audit, which is the
    # whole point of detecting below the operating threshold.
    operating = {"face": cfg.face_threshold, "lp": cfg.lp_threshold}
    redact_dets = [d for d in detections if d.score > operating[d.cls]]
    log.info("%s: %d detections total, %d above operating threshold (redacted), "
              "%d in the audit sweep band", clip.clip_id, len(detections),
              len(redact_dets), len(detections) - len(redact_dets))

    hold_frames, back_hold = resolve_holds(cfg, clip.fps)
    stride = max(1, round(clip.fps / cfg.detect_hz))
    dropped_small: list[Detection] = []
    tracks = build_tracks(redact_dets, cfg.min_box_px, iou_thresh=TRACK_IOU_DEFAULT,
                           hold_frames=hold_frames, back_hold_frames=back_hold,
                           dropped_small=dropped_small)
    fill_map = tracks_to_fill_map(tracks, clip.width, clip.height,
                                   cfg.dilate_scale, cfg.motion_margin_px)
    # Count FACE-covered frames separately before the tracks go away.
    # tracks_to_fill_map() discards Track.cls, and the zero-redaction canary
    # in build_audit() keys off n_frames_with_fill, which counts frames where
    # ANY box was burned in. On footage like this — where the plate model
    # fires on printed packaging text — a single above-threshold lp box
    # anywhere in 8134 frames silences the only structural defence against
    # "the FACE model returned nothing at all" for the entire clip.
    n_face_fill_frames = len({f for tr in tracks if tr.cls == "face" for f in tr.frames})
    del tracks  # ~260 MB of per-frame entries; nothing below reads it again

    if cfg.forced_boxes is not None:
        n_forced = _apply_forced_boxes(
            cfg.forced_boxes, clip, fill_map,
            hold_frames=hold_frames, back_hold_frames=back_hold,
            dilate_scale=cfg.dilate_scale, motion_margin_px=cfg.motion_margin_px)
        log.info("%s: applied %d forced boxes from human review", clip.clip_id, n_forced)

    out_path = cfg.output_dir / f"{clip.clip_id}.blurred.mp4"
    fill_stats = redact_and_encode(cfg, clip, fill_map, ffmpeg, out_path)
    t_encode = time.monotonic()

    integrity_log = cfg.output_dir / "logs" / f"{clip.clip_id}.integrity_decode.stderr.log"
    integrity = check_fill_integrity(ffmpeg, out_path, clip.width, clip.height,
                                      fill_map, cfg.redaction, integrity_log,
                                      clip.n_frames)
    sweep = check_low_threshold_sweep(
        detections, fill_map, cfg.sweep_threshold, operating,
    )
    # The output was encoded with the source's own colour_range, so a pc
    # source yields a pc output — YuNet needs the same range handling the
    # primary detector gets, or the "independent" check runs on a crushed
    # image and its misses correlate with EgoBlur's instead of being
    # independent evidence.
    yunet = check_yunet(cfg, ffmpeg, out_path, clip.width, clip.height,
                         fill_map, cfg.detect_hz, clip.fps, clip.n_frames,
                         full_range=clip.color_range == "pc")
    t_verify = time.monotonic()

    audit = build_audit(clip, fill_stats, integrity, sweep, yunet, gen,
                         n_dropped_small=len(dropped_small),
                         lp_checked=lp_det is not None,
                         n_face_fill_frames=n_face_fill_frames)
    write_audit_summary(audit, clip, cfg.output_dir / f"{clip.clip_id}.audit_summary.md")

    timing = {
        "detect_s": t_detect - t0,
        "encode_s": t_encode - t_detect,
        "verify_s": t_verify - t_encode,
        "total_s": t_verify - t0,
    }
    write_manifest(clip, gen, cfg, out_path, audit,
                    timing, cfg.output_dir / f"{clip.clip_id}.manifest.json",
                    hold_frames=hold_frames, back_hold_frames=back_hold)

    log.info("%s: %s (%.1fs)", clip.clip_id, audit["status"], timing["total_s"])
    return audit


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_args(argv)

    if not cfg.no_watchdog:
        arm_watchdog(cfg.watchdog_hours, cfg.output_dir)

    try:
        preflight(cfg)
        ffmpeg, ffprobe = _bin("ffmpeg"), _bin("ffprobe")
        clips = discover_clips(cfg.input_dir, ffprobe)
        gen, face_det, lp_det = build_detectors(cfg)

        budget = probe_and_budget(cfg, clips, face_det, lp_det, ffmpeg)
        if cfg.probe_only:
            print(json.dumps(budget, indent=2))
            return 0

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = cfg.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # One bad clip (corrupt file, a transient CUDA error) must not
        # abort clips that haven't been attempted yet — that would leave
        # the pod up (per the except below) burning cost for no reason
        # when the actual problem is scoped to a single input file.
        audits = []
        failed_clip_ids = []
        for c in clips:
            try:
                audits.append(process_clip(cfg, c, gen, face_det, lp_det, ffmpeg, checkpoint_dir))
            except Exception:
                log.exception("%s: failed — continuing with remaining clips", c.clip_id)
                failed_clip_ids.append(c.clip_id)
                audits.append({"clip_id": c.clip_id, "status": "FAILED"})
                # redact_and_encode() writes the .blurred.mp4 into the publish
                # directory BEFORE verification runs, so a clip that fails
                # check_fill_integrity or check_yunet leaves a file that is
                # indistinguishable by name from a verified one — with no
                # manifest beside it to say otherwise. Rename it to something
                # no publish glob will match, rather than delete it: the file
                # is the evidence for whatever went wrong.
                orphan = cfg.output_dir / f"{c.clip_id}.blurred.mp4"
                if orphan.exists():
                    quarantined = orphan.with_suffix(".mp4.FAILED-DO-NOT-SHIP")
                    orphan.replace(quarantined)
                    log.warning("%s: quarantined unverified output as %s",
                                 c.clip_id, quarantined.name)

        pass_statuses = {"PASS_AUTOMATED", "PASS_AUTOMATED_NO_YUNET"}
        n_pass = sum(1 for a in audits if a["status"] in pass_statuses)
        n_needs_review = sum(1 for a in audits if a["status"] == "NEEDS_REVIEW")
        run_manifest = {
            "run_id": cfg.run_id, "gen": gen, "n_clips": len(clips),
            "clip_ids": [c.clip_id for c in clips],
            "n_pass": n_pass,
            "n_needs_review": n_needs_review,
            "n_failed": len(failed_clip_ids),
            "failed_clip_ids": failed_clip_ids,
        }
        (cfg.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2), encoding="utf-8"
        )
        log.info("run complete: %s", json.dumps(run_manifest))

    except Exception:
        log.exception("job failed — pod left running (not stopped) so logs survive for debugging")
        raise
    else:
        if not cfg.skip_shutdown:
            shutdown_pod()

    # Exit code reflects the run's actual outcome, not just "no uncaught
    # exception" — a caller (or a human glancing at $?) must not be able
    # to read exit 0 as an unconditional green light when clips came back
    # NEEDS_REVIEW or FAILED. See build_audit()'s docstring for why
    # fill_integrity alone was never a sufficient gate for this.
    return 0 if (n_needs_review == 0 and not failed_clip_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
