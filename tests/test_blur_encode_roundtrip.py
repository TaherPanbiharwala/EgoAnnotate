"""End-to-end redact -> encode -> verify against a REAL H.264 encode.

Everything else in the blur test suite mocks the decoder, and that hid a
bug no unit test could have caught: check_fill_integrity's chroma window
was derived from the luma-eroded box, which left only ONE chroma sample of
margin against the ringing a lossy encoder produces at the box edge.
Measured on a real CRF-18 encode, ringing reaches ~3 chroma samples, so
every correctly-redacted large box was reported as a violation — 30 of 30
frames. Every clip would have come back NEEDS_REVIEW, which trains a
reviewer to ignore the one gate that matters.

These tests are the antidote: no mocks anywhere. They run the actual
redact_and_encode -> check_fill_integrity path over a real encoded file.

ffmpeg comes from imageio-ffmpeg (a self-contained static binary) rather
than the system, so a broken or absent system ffmpeg cannot silently skip
the only unmocked coverage in the suite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("imageio_ffmpeg", reason="needs imageio-ffmpeg for a real encoder")
import imageio_ffmpeg

W, H, N = 320, 240, 24


@pytest.fixture(scope="module")
def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture(scope="module")
def source(ffmpeg, tmp_path_factory) -> Path:
    """A textured clip. A flat source would survive encoding trivially and
    prove nothing about ringing."""
    d = tmp_path_factory.mktemp("enc")
    src = d / "src.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=30:duration=0.8",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    return src


def _clip(blur_job, src: Path):
    return blur_job.ClipInfo(path=src, clip_id="t", clip_group="t", chapter_index=0,
                             width=W, height=H, fps=30.0, n_frames=N, duration_s=0.8,
                             rotation=0, sha256="0" * 64)


def _cfg(blur_job, tmp_path):
    return blur_job.parse_args(["--input-dir", str(tmp_path), "--output-dir",
                                 str(tmp_path), "--run-id", "t"])


@pytest.mark.parametrize("label,box", [
    ("large", (50.0, 50.0, 150.0, 150.0)),
    ("mid", (100.0, 100.0, 140.0, 140.0)),
    ("small", (150.0, 100.0, 174.0, 124.0)),
])
def test_correctly_redacted_boxes_survive_a_real_encode(
        blur_job, ffmpeg, source, tmp_path, label, box):
    """The regression that mocks could not catch. Every one of these boxes IS
    correctly redacted; the check must not call them violations."""
    clip = _clip(blur_job, source)
    cfg = _cfg(blur_job, tmp_path)
    fill_map = {i: [box] for i in range(N)}
    out = tmp_path / f"{label}.mp4"

    blur_job.redact_and_encode(cfg, clip, fill_map, ffmpeg, out)
    r = blur_job.check_fill_integrity(ffmpeg, out, W, H, fill_map, "fill",
                                       tmp_path / "e.log", N)
    assert r["fill_integrity_violations"] == 0, (
        f"{label} box: correctly-redacted region reported as a violation "
        f"(encoder ringing leaking into the checked window)")
    assert r["fill_integrity_checked"] == N


def test_an_unredacted_region_is_still_caught_after_encoding(
        blur_job, ffmpeg, source, tmp_path):
    """The other direction: loosening tolerances until real encodes pass must
    not blind the check to an actual miss."""
    clip = _clip(blur_job, source)
    cfg = _cfg(blur_job, tmp_path)
    out = tmp_path / "clean.mp4"
    blur_job.redact_and_encode(cfg, clip, {}, ffmpeg, out)  # redact nothing

    claimed = {i: [(50.0, 50.0, 150.0, 150.0)] for i in range(N)}
    r = blur_job.check_fill_integrity(ffmpeg, out, W, H, claimed, "fill",
                                       tmp_path / "e.log", N)
    assert r["fill_integrity_violations"] == N


def test_frame_index_desync_is_caught_after_encoding(
        blur_job, ffmpeg, source, tmp_path):
    """Boxes burned in on the right frames, then verified against a shifted
    map — the desync canary, end to end."""
    clip = _clip(blur_job, source)
    cfg = _cfg(blur_job, tmp_path)
    box = (50.0, 50.0, 150.0, 150.0)
    out = tmp_path / "shift.mp4"
    blur_job.redact_and_encode(cfg, clip, {0: [box], 1: [box]}, ffmpeg, out)

    shifted = {i: [box] for i in range(5, 10)}  # nothing was filled there
    r = blur_job.check_fill_integrity(ffmpeg, out, W, H, shifted, "fill",
                                       tmp_path / "e.log", N)
    assert r["fill_integrity_violations"] == 5


def test_encode_preserves_frame_count_and_pixels_outside_boxes(
        blur_job, ffmpeg, source, tmp_path):
    """yuv420p raw pipes both ways: everything outside a redaction box should
    come through the re-encode essentially untouched, and no frame may be
    dropped or duplicated."""
    import numpy as np

    clip = _clip(blur_job, source)
    cfg = _cfg(blur_job, tmp_path)
    out = tmp_path / "rt.mp4"
    # No redaction at all. Burning in a large flat box changes x264's bit
    # allocation for the whole frame, which perturbs unrelated pixels by a
    # legitimate few levels — that would muddy what this test is actually
    # pinning, which is that the decode->encode round trip itself preserves
    # the image.
    stats = blur_job.redact_and_encode(cfg, clip, {}, ffmpeg, out)
    assert stats["n_frames_with_fill"] == 0

    def first_frame(path):
        proc = blur_job.open_decoder(ffmpeg, path, tmp_path / "d.log")
        y, _u, _v = next(blur_job.read_frames(proc, W, H))
        proc.stdout.close()
        proc.terminate()
        proc.wait()
        return y.astype(np.int16)

    src_y, out_y = first_frame(source), first_frame(out)
    d = np.abs(src_y - out_y)

    # Whole-frame MEAN, and the fraction of pixels moved at all — these are
    # the two numbers that actually separated the bug from a healthy
    # re-encode when it was measured:
    #     output-only -color_range tv : mean 13.86, 75.9% of pixels moved >3
    #     colour flags on both sides  : mean  0.09,  0.0% of pixels moved >3
    # Max is deliberately not asserted: a second CRF-18 generation moves a
    # handful of isolated high-contrast edge pixels by tens of levels, which
    # is normal and says nothing about a systematic range conversion.
    moved = float((d > 3).mean())
    assert d.mean() < 1.0 and moved < 0.05, (
        f"re-encode systematically altered the image (mean {d.mean():.2f}, "
        f"{moved * 100:.1f}% of pixels moved >3). The colour flags must be set "
        f"on the INPUT side too — otherwise ffmpeg treats -color_range as a "
        f"conversion request on the raw pipe and rewrites every pixel.")
