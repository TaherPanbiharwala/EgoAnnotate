from pathlib import Path

import pytest

from egoannote.layers.hands import (
    HandDecodeError,
    _expected_sample_count,
    _require_complete_sampling,
)


def test_expected_sample_count_matches_stride_selection() -> None:
    assert _expected_sample_count(8_134, 2) == 4_067
    assert _expected_sample_count(8_134, 3) == 2_712


def test_partial_decode_is_not_mistaken_for_clean_eof(tmp_path: Path) -> None:
    with pytest.raises(HandDecodeError, match=r"14/4067"):
        _require_complete_sampling(emitted=14, expected=4_067, video=tmp_path / "clip.mp4")
