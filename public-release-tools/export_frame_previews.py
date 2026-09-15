#!/usr/bin/env python3
"""Export separate, web-friendly preview MP4s from exact video-frame ranges.

The input is a CSV file with one clip per row.  ``start_frame`` and
``end_frame`` are zero-based and inclusive, so ``30,89`` exports exactly 60
frames.  Relative input paths are resolved relative to the CSV file.  Relative
output names are placed in ``--output-dir``.

Example CSV::

    input,start_frame,end_frame,output
    public-release/egoannote-v1/videos/GX010059.mp4,0,239,GX010059-intro.mp4
    public-release/egoannote-v1/annotated-videos/GX010059.mp4,600,899,GX010059-annotations.mp4

Run from any directory::

    python3 scripts/export_frame_previews.py preview_clips.csv

The script re-encodes each range as H.264/AAC so that the result plays
reliably in Substack and on the web. Video-frame selection is frame-exact.
Use ``--mute`` for a silent preview; it is also the safe choice for a
variable-frame-rate camera file.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

REQUIRED_COLUMNS = ("input", "start_frame", "end_frame")


class PreviewError(RuntimeError):
    """A clear, user-facing problem that prevents safe preview export."""


@dataclass(frozen=True)
class SourceInfo:
    fps: Fraction
    frame_count: int
    has_audio: bool
    is_constant_frame_rate: bool


@dataclass(frozen=True)
class Clip:
    row_number: int
    source: Path
    start_frame: int
    end_frame: int
    output: Path
    source_info: SourceInfo

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


def _binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise PreviewError(
            f"{name} is not on PATH. Install FFmpeg first "
            "(macOS: brew install ffmpeg)."
        )
    return binary


def _run(command: list[str], *, description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise PreviewError(f"{description} failed:\n{detail}")
    return result


def _probe_json(command: list[str], *, description: str) -> dict[str, object]:
    result = _run(command, description=description)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PreviewError(f"{description} returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PreviewError(f"{description} returned an unexpected response.")
    return data


def _single_stream(data: dict[str, object], *, description: str) -> dict[str, object] | None:
    streams = data.get("streams")
    if not isinstance(streams, list):
        raise PreviewError(f"{description} returned no stream list.")
    if not streams:
        return None
    stream = streams[0]
    if not isinstance(stream, dict):
        raise PreviewError(f"{description} returned malformed stream metadata.")
    return stream


def _parse_positive_rate(value: object, *, source: Path) -> Fraction:
    if not isinstance(value, str) or value in {"", "0/0", "N/A"}:
        raise PreviewError(f"Could not determine a usable frame rate for {source}.")
    try:
        fps = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise PreviewError(f"Could not parse frame rate {value!r} for {source}.") from exc
    if fps <= 0:
        raise PreviewError(f"Frame rate must be positive for {source}; got {value!r}.")
    return fps


def _parse_frame_count(value: object, *, source: Path) -> int | None:
    if not isinstance(value, str) or value in {"", "N/A"}:
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise PreviewError(f"Could not parse frame count {value!r} for {source}.") from exc
    if count <= 0:
        raise PreviewError(f"Frame count must be positive for {source}; got {value!r}.")
    return count


def _decoded_frame_count(ffprobe: str, source: Path) -> int:
    data = _probe_json(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(source),
        ],
        description=f"Counting frames in {source}",
    )
    stream = _single_stream(data, description=f"Counting frames in {source}")
    if stream is None:
        raise PreviewError(f"No video stream found in {source}.")
    count = _parse_frame_count(stream.get("nb_read_frames"), source=source)
    if count is None:
        raise PreviewError(f"Could not count frames in {source}.")
    return count


def probe_source(ffprobe: str, source: Path) -> SourceInfo:
    data = _probe_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(source),
        ],
        description=f"Inspecting {source}",
    )
    stream = _single_stream(data, description=f"Inspecting {source}")
    if stream is None:
        raise PreviewError(f"No video stream found in {source}.")

    fps = _parse_positive_rate(stream.get("avg_frame_rate"), source=source)
    real_rate = _parse_positive_rate(stream.get("r_frame_rate"), source=source)
    frame_count = _parse_frame_count(stream.get("nb_frames"), source=source)
    if frame_count is None:
        frame_count = _decoded_frame_count(ffprobe, source)

    audio_data = _probe_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(source),
        ],
        description=f"Inspecting audio in {source}",
    )
    has_audio = _single_stream(audio_data, description=f"Inspecting audio in {source}") is not None
    return SourceInfo(
        fps=fps,
        frame_count=frame_count,
        has_audio=has_audio,
        is_constant_frame_rate=fps == real_rate,
    )


def _frame_number(value: str | None, *, column: str, row_number: int) -> int:
    if value is None or not value.strip():
        raise PreviewError(f"CSV row {row_number}: {column} is required.")
    try:
        number = int(value)
    except ValueError as exc:
        raise PreviewError(
            f"CSV row {row_number}: {column} must be a whole number, got {value!r}."
        ) from exc
    if number < 0:
        raise PreviewError(f"CSV row {row_number}: {column} must be zero or greater.")
    return number


def _relative_to(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def read_clips(manifest: Path, output_dir: Path, ffprobe: str) -> list[Clip]:
    try:
        file = manifest.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise PreviewError(f"Could not read CSV file {manifest}: {exc}") from exc

    with file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise PreviewError("The CSV file is empty; add a header row and at least one clip.")
        reader.fieldnames = [header.strip() if header else "" for header in reader.fieldnames]
        headers = set(reader.fieldnames)
        missing = set(REQUIRED_COLUMNS) - headers
        if missing:
            joined = ", ".join(sorted(missing))
            raise PreviewError(f"CSV is missing required column(s): {joined}.")

        clips: list[Clip] = []
        cached_sources: dict[Path, SourceInfo] = {}
        used_outputs: set[Path] = set()
        for row_number, row in enumerate(reader, start=2):
            input_value = (row.get("input") or "").strip()
            if not input_value or input_value.startswith("#"):
                continue
            source = _relative_to(manifest.parent, input_value)
            if not source.is_file():
                raise PreviewError(f"CSV row {row_number}: input video does not exist: {source}")

            start_frame = _frame_number(
                row.get("start_frame"), column="start_frame", row_number=row_number
            )
            end_frame = _frame_number(row.get("end_frame"), column="end_frame", row_number=row_number)
            if end_frame < start_frame:
                raise PreviewError(
                    f"CSV row {row_number}: end_frame must be at least start_frame."
                )

            source_info = cached_sources.get(source)
            if source_info is None:
                source_info = probe_source(ffprobe, source)
                cached_sources[source] = source_info
            if end_frame >= source_info.frame_count:
                raise PreviewError(
                    f"CSV row {row_number}: end_frame {end_frame} is outside {source.name}, "
                    f"which has frames 0 through {source_info.frame_count - 1}."
                )

            output_value = (row.get("output") or "").strip()
            if output_value:
                output = _relative_to(output_dir, output_value)
            else:
                output = output_dir / f"{source.stem}_frames_{start_frame}-{end_frame}.mp4"
            if output.suffix.lower() != ".mp4":
                raise PreviewError(f"CSV row {row_number}: output must end in .mp4: {output}")
            output = output.resolve()
            if output == source.resolve():
                raise PreviewError(f"CSV row {row_number}: output cannot overwrite its input video.")
            if output in used_outputs:
                raise PreviewError(f"CSV row {row_number}: duplicate output path: {output}")
            used_outputs.add(output)
            clips.append(
                Clip(
                    row_number=row_number,
                    source=source.resolve(),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    output=output,
                    source_info=source_info,
                )
            )

    if not clips:
        raise PreviewError("The CSV has no clip rows. Add a row below the header.")
    return clips


def _seconds(value: Fraction) -> str:
    """Render a non-scientific decimal that FFmpeg accepts as seconds."""
    with localcontext() as context:
        context.prec = 30
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal, "f")


def export_clip(ffmpeg: str, clip: Clip, *, overwrite: bool, mute: bool) -> None:
    if clip.output.exists() and not overwrite:
        raise PreviewError(
            f"Refusing to overwrite existing file: {clip.output}\n"
            "Pass --overwrite only if replacing it is intentional."
        )

    clip.output.parent.mkdir(parents=True, exist_ok=True)
    if clip.source_info.has_audio and not mute and not clip.source_info.is_constant_frame_rate:
        raise PreviewError(
            f"{clip.source} has a variable frame rate. Use --mute for an exact "
            "frame-number preview, or first convert the source to a constant frame rate."
        )
    start_seconds = Fraction(clip.start_frame, 1) / clip.source_info.fps
    end_seconds = Fraction(clip.end_frame + 1, 1) / clip.source_info.fps
    video_filter = (
        f"select='between(n\\,{clip.start_frame}\\,{clip.end_frame})',"
        "setpts=PTS-STARTPTS"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(clip.source),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
    ]
    if clip.source_info.has_audio and not mute:
        command.extend(
            [
                "-map",
                "0:a:0",
                "-af",
                f"atrim=start={_seconds(start_seconds)}:end={_seconds(end_seconds)},"
                "asetpts=PTS-STARTPTS",
            ]
        )
    command.extend(
        [
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if clip.source_info.has_audio and not mute:
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    command.append(str(clip.output))
    _run(command, description=f"Exporting CSV row {clip.row_number} ({clip.output.name})")


def verify_export(ffprobe: str, clip: Clip) -> None:
    actual_count = _decoded_frame_count(ffprobe, clip.output)
    if actual_count != clip.frame_count:
        raise PreviewError(
            f"{clip.output} decoded to {actual_count} frames; expected {clip.frame_count}. "
            "The file was kept for inspection and was not marked as a valid preview."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export separate H.264 preview MP4s from exact, inclusive frame ranges."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="CSV with input,start_frame,end_frame and optional output columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for relative output names (default: <CSV directory>/preview-videos).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output MP4s that already exist.",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Omit audio from every exported preview.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ffmpeg = _binary("ffmpeg")
        ffprobe = _binary("ffprobe")
        manifest = args.csv_file.expanduser().resolve()
        if not manifest.is_file():
            raise PreviewError(f"CSV file does not exist: {manifest}")
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else manifest.parent / "preview-videos"
        )
        clips = read_clips(manifest, output_dir, ffprobe)
        print(f"Validated {len(clips)} preview clip(s).")
        for clip in clips:
            print(
                f"Exporting {clip.output.name}: {clip.source.name} "
                f"frames {clip.start_frame}-{clip.end_frame} ({clip.frame_count} frames)"
            )
            export_clip(ffmpeg, clip, overwrite=args.overwrite, mute=args.mute)
            verify_export(ffprobe, clip)
        print("Done. Every output passed an exact decoded-frame-count check.")
        return 0
    except PreviewError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
