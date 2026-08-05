"""The prompt template must match the frames actually sent.

Review finding: the prompt hardcoded "8 consecutive frames", "N = 0..7", and
"start_frame=0, end_frame=7" while `iter_window_indices` emits a short
trailing window for any video whose frame count isn't an exact multiple of
the window size. The model was told to answer for 8 images when 3 were sent,
and the parser clamped to the global bound regardless — so the last window of
most videos produced actions indexing frames that were never sent.
"""

from __future__ import annotations

import re

import pytest

from egoannote import config
from egoannote.layers.caption import _load_prompt, _render_prompt


def test_template_has_no_leftover_placeholders_after_render() -> None:
    template, _ = _load_prompt()
    rendered = _render_prompt(template, config.VLM_WINDOW_FRAMES)
    assert "{n_frames}" not in rendered
    assert "{max_frame_idx}" not in rendered


def test_full_window_renders_correct_bounds() -> None:
    template, _ = _load_prompt()
    rendered = _render_prompt(template, 8)
    assert "Below are 8 consecutive frames" in rendered
    assert "N = 0..7" in rendered
    assert "INTEGERS 0-7" in rendered
    assert "start_frame=0, end_frame=7." in rendered


def test_partial_window_renders_its_own_smaller_bounds() -> None:
    template, _ = _load_prompt()
    rendered = _render_prompt(template, 3)
    assert "Below are 3 consecutive frames" in rendered
    assert "N = 0..2" in rendered
    assert "INTEGERS 0-2" in rendered
    assert "start_frame=0, end_frame=2." in rendered
    # The full-window bound must not survive anywhere.
    assert "0..7" not in rendered
    assert "end_frame=7." not in rendered


def test_json_example_braces_survive_rendering() -> None:
    """The prompt body contains a literal JSON example. str.format() would
    choke on those braces — which is why _render_prompt uses replace()."""
    template, _ = _load_prompt()
    rendered = _render_prompt(template, 8)
    assert '"actions": [' in rendered
    assert '"left_hand": {' in rendered
    assert '"contact_type": "grip|push|pull' in rendered


def test_zero_frames_is_rejected() -> None:
    template, _ = _load_prompt()
    with pytest.raises(ValueError):
        _render_prompt(template, 0)


def test_prompt_hash_is_of_the_template_not_the_render() -> None:
    """Every window in a run must share one prompt_hash, or the field is
    useless for grouping a run's records."""
    _, hash_a = _load_prompt()
    _, hash_b = _load_prompt()
    assert hash_a == hash_b
    assert hash_a.startswith("sha256:")


def test_no_hardcoded_frame_bound_remains_in_the_template() -> None:
    """Guard against someone re-hardcoding a bound in the prompt later."""
    template, _ = _load_prompt()
    body = template.replace("{max_frame_idx}", "PLACEHOLDER")
    # "0..7" / "0-7" / "end_frame=7" must only ever appear via the placeholder.
    assert not re.search(r"0\.\.7\b", body)
    assert not re.search(r"INTEGERS 0-7\b", body)
    assert "end_frame=7." not in body
