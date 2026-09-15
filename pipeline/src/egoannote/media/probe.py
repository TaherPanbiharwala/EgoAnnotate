"""ffprobe wrapper. One source of truth for a video's real fps/duration/dims.

Eng review "clock-agreement" finding: v1 derived the caption track's clock
from `window_idx * window * 1000 // fps` and the hands track's clock from
`source_idx / cap.get(CAP_PROP_FPS)` independently — two clocks assumed
equal, never asserted. Both tracks must call `probe()` once and derive
timestamps from the SAME VideoInfo, or an F1@±0.5s claim is unfounded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FfmpegNotFoundError(RuntimeError):
    pass


class FfprobeError(RuntimeError):
    pass


def _ffprobe_bin() -> str:
    binary = shutil.which("ffprobe")
    if not binary:
        raise FfmpegNotFoundError(
            "ffprobe not found on PATH (ships with ffmpeg).\n"
            "  macOS    brew install ffmpeg\n"
            "  Ubuntu   sudo apt install ffmpeg\n"
            "  conda    conda install -c conda-forge ffmpeg"
        )
    return binary


@dataclass(slots=True, frozen=True)
class VideoInfo:
    duration_sec: float
    width: int
    height: int
    fps: float
    n_frames: int  # source frame count, not after sampling
    is_vfr: bool  # r_frame_rate != avg_frame_rate beyond tolerance


def probe(video: Path) -> VideoInfo:
    """Probe a video file with ffprobe: duration, dims, fps, frame count, VFR flag."""
    cmd = [
        _ffprobe_bin(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        # Absolute path, not str(video): a relative filename beginning with
        # "-" is parsed by ffprobe as an option, not an input file (verified:
        # a file named "-help.mp4" yields "Missing argument for option").
        # An absolute path can never start with "-". No shell is involved
        # (argument list, shell=False), so this is option smuggling, not
        # command injection — but camera/download filenames are attacker-
        # influenced and this costs nothing to close.
        str(video.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FfprobeError(f"ffprobe failed on {video}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise FfprobeError(
            f"ffprobe reports 0 video streams for {video} — "
            f"likely a truncated download or an HTML error page, not a video file."
        )
    stream = data["streams"][0]
    format_info = data.get("format", {})

    def _parse_rate(s: str) -> float:
        # ffprobe emits "N/A" for avg_frame_rate on some streams, and "0/0"
        # for others. Both would otherwise escape as a bare ValueError or a
        # silent 0.0 — and fps=0 propagates into every timestamp downstream.
        try:
            num, denom = (s.split("/") + ["1"])[:2]
            rate = float(num) / max(float(denom), 1e-9)
        except (ValueError, TypeError) as e:
            raise FfprobeError(
                f"ffprobe reported an unusable frame rate {s!r} for {video}. "
                f"Every timestamp derives from this, so it cannot be defaulted."
            ) from e
        if rate <= 0:
            raise FfprobeError(
                f"ffprobe reported frame rate {s!r} (= {rate}) for {video}. "
                f"Every timestamp derives from this, so it cannot be defaulted."
            )
        return rate

    avg_fps = _parse_rate(stream["avg_frame_rate"])
    r_fps = _parse_rate(stream.get("r_frame_rate", stream["avg_frame_rate"]))
    is_vfr = abs(avg_fps - r_fps) > 0.05

    raw_nb = stream.get("nb_frames", "N/A")
    duration = float(stream.get("duration") or format_info.get("duration", 0.0))
    n_frames = int(raw_nb) if raw_nb not in ("N/A", "", None) else round(duration * avg_fps)

    return VideoInfo(
        duration_sec=duration,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=avg_fps,
        n_frames=n_frames,
        is_vfr=is_vfr,
    )
