"""Frame extraction via system ffmpeg.

We shell out to `ffmpeg` instead of pulling in `ffmpeg-python` or `decord`.
ffmpeg is already a system dep on every dev machine; the wrappers add risk
without value for the I/O patterns we need. (Kept from v1 — the reasoning
held up under review.)
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .probe import FfmpegNotFoundError


def _ffmpeg_bin() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH.\n"
            "  macOS    brew install ffmpeg\n"
            "  Ubuntu   sudo apt install ffmpeg\n"
            "  conda    conda install -c conda-forge ffmpeg"
        )
    return binary


def extract_frames(
    video: Path,
    fps: str,
    out_dir: Path,
    *,
    long_edge: int | None = None,
    pattern: str = "frame_%06d.jpg",
    quality: int = 3,
) -> list[Path]:
    """Extract frames at the given fps into out_dir/pattern.

    Args:
        video: source video file.
        fps: target frames-per-second as an ffmpeg `-vf fps=` expression —
            pass a rational string like "4/3", not a float. See config.py
            VLM_FPS_STR: floats drift over a long video (Eng review A1-11).
        out_dir: directory to write frames into. Created if missing.
        long_edge: if set, resize so the longer edge is this many pixels.
        pattern: ffmpeg output filename pattern; %06d is the 1-indexed frame number.
        quality: ffmpeg `-q:v` for mjpeg (2 = highest quality, 31 = worst).

    Returns the sorted list of generated frame paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vf_parts = [f"fps={fps}"]
    if long_edge:
        vf_parts.append(
            f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
        )
    vf = ",".join(vf_parts)

    cmd = [
        _ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", vf,
        "-q:v", str(quality),
        str(out_dir / pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video}: {result.stderr.strip()}")

    return sorted(out_dir.glob("frame_*.jpg"))


@dataclass(slots=True, frozen=True)
class Window:
    window_idx: int
    frame_indices: list[int]
    is_partial: bool  # True if this window has fewer frames than the target
    # window count — the trailing window of a video whose duration isn't an
    # exact multiple of window length. S7: must be marked, not silently
    # weighted the same as a full window (it carries less evidence).


def iter_window_indices(n_frames: int, window: int) -> Iterator[Window]:
    """Yield Window records for non-overlapping windows over n_frames.

    Stride == window (see config.VLM_STRIDE_SECONDS): overlap is not used
    post-S1, since the ordered-list prompt reports sub-window boundaries
    directly rather than relying on window-to-window triangulation.
    """
    for window_idx, start in enumerate(range(0, n_frames, window)):
        indices = list(range(start, min(start + window, n_frames)))
        yield Window(
            window_idx=window_idx,
            frame_indices=indices,
            is_partial=len(indices) < window,
        )
