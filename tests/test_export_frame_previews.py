"""Integration coverage for the standalone frame-accurate preview exporter."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "public-release-tools" / "export_frame_previews.py"
SYNTHETIC_VIDEO = ROOT / "examples" / "synthetic_smoke_test.mp4"


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="The standalone preview exporter requires system FFmpeg.",
)
def test_exports_the_requested_inclusive_frame_range(tmp_path: Path) -> None:
    manifest = tmp_path / "clips.csv"
    manifest.write_text(
        "input,start_frame,end_frame,output\n"
        f"{SYNTHETIC_VIDEO},30,89,short-preview.mp4\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "previews"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(manifest), "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = output_dir / "short-preview.mp4"
    assert output.is_file()
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(probe.stdout)["streams"][0]["nb_read_frames"] == "60"
