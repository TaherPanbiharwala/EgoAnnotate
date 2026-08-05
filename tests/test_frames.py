"""Tests for windowing. The is_partial marker (S7 fix) matters: a trailing
short window carries less evidence and must not be weighted the same as a
full window in downstream segmentation scoring."""

from __future__ import annotations

from egoannote.media.frames import iter_window_indices


def test_non_overlapping_windows_cover_all_frames() -> None:
    windows = list(iter_window_indices(20, 8))
    assert [w.window_idx for w in windows] == [0, 1, 2]
    assert windows[0].frame_indices == list(range(8))
    assert windows[1].frame_indices == list(range(8, 16))
    assert windows[2].frame_indices == list(range(16, 20))


def test_trailing_short_window_marked_partial() -> None:
    windows = list(iter_window_indices(20, 8))
    assert windows[0].is_partial is False
    assert windows[1].is_partial is False
    assert windows[2].is_partial is True  # only 4 frames, not 8


def test_exact_multiple_has_no_partial_window() -> None:
    windows = list(iter_window_indices(16, 8))
    assert len(windows) == 2
    assert all(not w.is_partial for w in windows)


def test_zero_frames_yields_no_windows() -> None:
    assert list(iter_window_indices(0, 8)) == []
