"""Tests for video_id derivation and the parse/config constant coupling."""

from __future__ import annotations

import pytest

from egoannote import config
from egoannote.config import video_id_from_content


def test_same_file_yields_the_same_id(tmp_path) -> None:
    """video_id is a primary-key column every resume decision keys on. If it
    were unstable for the same file, every resume would silently become a
    full re-run at full API cost, with no error to notice."""
    p = tmp_path / "GX010001.MP4"
    p.write_bytes(b"recording-one" * 1000)
    assert video_id_from_content(p, "GX010001") == video_id_from_content(p, "GX010001")


def test_repeated_gopro_stems_do_not_collide(tmp_path) -> None:
    """GX010001.MP4 is one of the most common camera filenames in existence.
    Two cards with the same stem must not merge into one video_id."""
    a, b = tmp_path / "card_a", tmp_path / "card_b"
    a.mkdir()
    b.mkdir()
    (a / "GX010001.MP4").write_bytes(b"recording-one" * 1000)
    (b / "GX010001.MP4").write_bytes(b"recording-two" * 1000)
    assert video_id_from_content(a / "GX010001.MP4", "GX010001") != video_id_from_content(
        b / "GX010001.MP4", "GX010001"
    )


def test_id_keeps_the_stem_for_human_legibility(tmp_path) -> None:
    p = tmp_path / "GX010059.MP4"
    p.write_bytes(b"data" * 100)
    assert video_id_from_content(p, "GX010059").startswith("GX010059-")


def test_empty_file_raises_rather_than_silently_sharing_an_id(tmp_path) -> None:
    """BUG guard: every zero-byte file hashes identically, so two failed
    downloads would silently share one video_id and one set of annotations."""
    p = tmp_path / "truncated.MP4"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        video_id_from_content(p, "truncated")


def test_vlm_fps_is_an_exact_rational_not_a_float() -> None:
    """8 frames / 6s = 1.333... A float here drifts over a long video and
    puts the caption track on a different clock than the hand track."""
    from fractions import Fraction

    assert isinstance(config.VLM_FPS, Fraction)
    assert config.VLM_FPS == Fraction(4, 3)
    assert config.VLM_FPS_STR == "4/3", "ffmpeg needs the exact rational form"


def test_parser_frame_bound_tracks_the_configured_window_size() -> None:
    """BUG guard: _MAX_FRAME_IDX was hardcoded to 7. Changing the window size
    would have left the parser silently clamping to a stale bound."""
    from egoannote.parse import _MAX_FRAME_IDX

    assert _MAX_FRAME_IDX == config.VLM_WINDOW_FRAMES - 1


def test_no_overlap_stride_matches_window() -> None:
    assert config.VLM_STRIDE_SECONDS == config.VLM_WINDOW_SECONDS
