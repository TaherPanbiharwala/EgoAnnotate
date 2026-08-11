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


def test_window_duration_is_templated_not_hardcoded():
    """The prompt used to hardcode "6 seconds" while templating only the
    frame counts, so a 3-frame trailing window was told it covered 6s when
    it covers 2.25s — the exact bug _render_prompt exists to fix, one line
    further down the same file."""
    from egoannote.layers.caption import _load_prompt, _render_prompt
    template, _ = _load_prompt()
    assert "{window_seconds}" in template, "duration placeholder went missing"
    assert "6 seconds" not in template, "duration is hardcoded again"

    full = _render_prompt(template, 8)
    partial = _render_prompt(template, 3)
    assert "whole 6 seconds" in full
    assert "whole 2.25 seconds" in partial, "partial window told the wrong duration"


def test_no_placeholder_survives_rendering():
    import re
    from egoannote.layers.caption import _load_prompt, _render_prompt
    template, _ = _load_prompt()
    for n in (1, 3, 8):
        left = re.findall(r"\{(n_frames|max_frame_idx|window_seconds)\}",
                          _render_prompt(template, n))
        assert not left, f"n_frames={n} left {left} unsubstituted"


def test_prompt_does_not_bias_the_model_toward_a_single_action():
    """The whole justification for non-overlapping windows is that the model
    reports multiple actions WITHIN a window. The prompt used to say "most
    clips will have ONE action" — its most directive sentence telling the
    model to do the opposite, with nothing downstream able to tell a genuine
    single action from a refusal to split."""
    from egoannote.layers.caption import _load_prompt
    template, _ = _load_prompt()
    assert "most clips will have" not in template
    assert "Do not force multiple actions" not in template


def test_prompt_does_not_ask_for_cross_window_consistency():
    """Each window is a stateless call with only its own frames — the model
    cannot see the previous window's label, so asking it to reuse one was
    literally unsatisfiable, at all 44 window seams per clip."""
    from egoannote.layers.caption import _load_prompt
    template, _ = _load_prompt()
    assert "identical label again" not in template
    assert "continues into the next" not in template
