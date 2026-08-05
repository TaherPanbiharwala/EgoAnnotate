"""Tests for the v3 ordered-action-list parser.

Covers everything v1's parser tests covered (closed-vocab enforcement,
task_step slugification, visible/verb contradiction coercion, length caps,
the fence-strip -> json.loads -> json_repair -> regex ladder) PLUS the new
schema_ok distinction (plan S4/A1-4): valid JSON that doesn't match the
expected shape must NOT be treated as usable, which is exactly the failure
v1's `_parse_ok` (valid-JSON-only) missed.
"""

from __future__ import annotations

import json

from egoannote.parse import _strip_code_fence, parse_caption_v3


def _good_action(**overrides) -> dict:
    base = {
        "start_frame": 0,
        "end_frame": 3,
        "task_step": "pluck_leaves",
        "left_hand": {
            "verb": "pinching", "object": "green leaf", "target": "blue bowl",
            "contact_type": "pinch", "visible": True,
        },
        "right_hand": {
            "verb": "holding", "object": "leafy stem", "target": None,
            "contact_type": "grip", "visible": True,
        },
        "tool_in_use": "none",
        "coordination": "supporting",
        "handover_event": False,
        "scene": {
            "what": "Person plucks leaves from a stem and drops them into a blue bowl.",
            "how": "Right hand pinch-grips the stem; left hand pinches each leaf.",
            "why": "To detach leaves from the stem into the collection bowl.",
            "location": "kitchen counter near sink",
        },
    }
    base.update(overrides)
    return base


def _response(*actions: dict) -> str:
    return json.dumps({"actions": list(actions)})


def test_parses_clean_single_action_happy_path() -> None:
    out = parse_caption_v3(_response(_good_action()))
    assert out["_schema_ok"] is True
    assert len(out["actions"]) == 1
    a = out["actions"][0]
    assert a["scene_what"].startswith("Person plucks leaves")
    assert a["left_hand"]["verb"] == "pinching"
    assert a["left_hand"]["target"] == "blue bowl"
    assert a["right_hand"]["contact_type"] == "grip"
    assert a["task_step"] == "pluck_leaves"
    assert a["handover_event"] is False
    assert a["coordination"] == "supporting"


def test_parses_multiple_actions_in_order() -> None:
    a1 = _good_action(start_frame=0, end_frame=3, task_step="rinse_carrot")
    a2 = _good_action(start_frame=4, end_frame=7, task_step="chop_carrot")
    out = parse_caption_v3(_response(a1, a2))
    assert out["_schema_ok"] is True
    assert len(out["actions"]) == 2
    assert out["actions"][0]["task_step"] == "rinse_carrot"
    assert out["actions"][1]["task_step"] == "chop_carrot"
    assert out["actions"][0]["start_frame"] == 0
    assert out["actions"][1]["end_frame"] == 7


def test_strips_markdown_code_fences() -> None:
    fenced = "```json\n" + _response(_good_action()) + "\n```"
    out = parse_caption_v3(fenced)
    assert out["_schema_ok"] is True
    assert out["actions"][0]["scene_what"].startswith("Person plucks leaves")


def test_bare_list_without_wrapper_still_validates() -> None:
    """Model sometimes skips the {"actions": [...]} wrapper and returns a
    bare JSON array. This must still validate correctly, not be treated as
    schema-invalid — but ALSO must not be confused with the failure mode
    below (empty list)."""
    bare = json.dumps([_good_action()])
    out = parse_caption_v3(bare)
    assert out["_schema_ok"] is True
    assert len(out["actions"]) == 1


def test_empty_actions_list_is_schema_ok_false() -> None:
    """CRITICAL (plan S4/A1-4): {"actions": []} is valid JSON AND matches
    the pydantic shape, but carries zero usable information. v1's
    _parse_ok (JSON validity only) would have called this "ok" — that is
    the exact silent-poisoning failure mode S4 exists to prevent. Must be
    schema_ok=False so s_label skips it rather than treating it as
    'confirmed no actions here.'"""
    out = parse_caption_v3(json.dumps({"actions": []}))
    assert out["_schema_ok"] is False


def test_missing_required_field_is_schema_ok_false() -> None:
    """A dict with the right top-level key but action items missing
    required fields (start_frame/end_frame) must fail schema validation,
    not silently coerce to defaults."""
    broken = json.dumps({"actions": [{"task_step": "no_frame_range"}]})
    out = parse_caption_v3(broken)
    assert out["_schema_ok"] is False


def test_unknown_contact_type_coerced_to_none() -> None:
    a = _good_action()
    a["left_hand"]["contact_type"] = "smushing"
    out = parse_caption_v3(_response(a))
    assert out["actions"][0]["left_hand"]["contact_type"] is None


def test_unknown_coordination_coerced_to_none() -> None:
    a = _good_action()
    a["coordination"] = "synchronized_ballet"
    out = parse_caption_v3(_response(a))
    assert out["actions"][0]["coordination"] is None


def test_task_step_with_spaces_gets_slugified() -> None:
    a = _good_action(task_step="rinse the carrot")
    out = parse_caption_v3(_response(a))
    assert out["actions"][0]["task_step"] == "rinse_the_carrot"


def test_task_step_too_long_truncated() -> None:
    a = _good_action(task_step="a_very_long_step_name_that_overflows_the_budget")
    out = parse_caption_v3(_response(a))
    ts = out["actions"][0]["task_step"]
    assert ts is not None
    assert len(ts) <= 24


def test_visible_true_with_null_verb_coerced_to_false() -> None:
    a = _good_action()
    a["left_hand"]["verb"] = None
    a["left_hand"]["visible"] = True
    out = parse_caption_v3(_response(a))
    assert out["actions"][0]["left_hand"]["visible"] is False
    assert out["actions"][0]["left_hand"]["verb"] is None


def test_visible_false_nulls_verb_object_fields() -> None:
    a = _good_action()
    a["right_hand"]["visible"] = False
    out = parse_caption_v3(_response(a))
    rh = out["actions"][0]["right_hand"]
    assert rh["visible"] is False
    assert rh["verb"] is None
    assert rh["object"] is None
    assert rh["target"] is None
    assert rh["contact_type"] is None


# --------------------------------------------------------------------------
# Frame-range validation. These replace an earlier test that asserted the
# indices were CLAMPED into range — which review showed was the bug, not the
# behavior: clamping made three distinct failures indistinguishable from
# clean output while still reporting schema_ok=True.
# --------------------------------------------------------------------------


def test_out_of_range_frame_index_is_not_schema_ok() -> None:
    """`end_frame: 15` in an 8-frame window used to clamp to 7 and report
    schema_ok=True — silently truncating the action's real span."""
    out = parse_caption_v3(_response(_good_action(start_frame=0, end_frame=15)))
    assert out["_schema_ok"] is False


def test_negative_frame_index_is_not_schema_ok() -> None:
    out = parse_caption_v3(_response(_good_action(start_frame=-2, end_frame=4)))
    assert out["_schema_ok"] is False


def test_inverted_frame_range_is_not_schema_ok() -> None:
    """start > end would produce a negative-duration segment downstream.
    Previously preserved as-is with schema_ok=True."""
    out = parse_caption_v3(_response(_good_action(start_frame=7, end_frame=0)))
    assert out["_schema_ok"] is False


def test_one_indexed_output_is_rejected_not_silently_shifted() -> None:
    """The dangerous case. Models commonly return 1-indexed lists. Clamping
    turned `1..8` into `1..7` — a plausible-looking range shifted by one
    frame (0.75s at 4/3 fps) on EVERY action, marked trustworthy. It must be
    rejected so the degradation path fires instead."""
    out = parse_caption_v3(_response(_good_action(start_frame=1, end_frame=8)))
    assert out["_schema_ok"] is False


def test_partial_window_rejects_indices_past_frames_actually_sent() -> None:
    """A trailing window may hold 3 frames. An action claiming frame 7 refers
    to an image the model was never shown."""
    out = parse_caption_v3(_response(_good_action(start_frame=0, end_frame=7)), n_frames=3)
    assert out["_schema_ok"] is False

    ok = parse_caption_v3(_response(_good_action(start_frame=0, end_frame=2)), n_frames=3)
    assert ok["_schema_ok"] is True


def test_valid_range_at_the_boundary_is_accepted() -> None:
    """The guard must not be so strict it rejects legitimate full-span
    actions — that would send every window down the degradation path."""
    out = parse_caption_v3(_response(_good_action(start_frame=0, end_frame=7)))
    assert out["_schema_ok"] is True
    assert out["actions"][0]["start_frame"] == 0
    assert out["actions"][0]["end_frame"] == 7


def test_single_frame_action_is_valid() -> None:
    """start == end is a legitimate 1-frame action (the prompt explicitly
    asks for short actions to be listed separately)."""
    out = parse_caption_v3(_response(_good_action(start_frame=3, end_frame=3)))
    assert out["_schema_ok"] is True


def test_malformed_json_uses_json_repair_fallback() -> None:
    """Trailing-comma JSON should be repaired and parsed against the real
    schema (not just recovered as flat text, per v1's design)."""
    malformed = (
        '{"actions": [{"start_frame": 0, "end_frame": 3, "task_step": "x",'
        ' "left_hand": {"verb": null, "object": null, "target": null,'
        ' "contact_type": null, "visible": false},'
        ' "right_hand": {"verb": null, "object": null, "target": null,'
        ' "contact_type": null, "visible": false},'
        ' "tool_in_use": "none", "coordination": "none", "handover_event": false,'
        ' "scene": {"what": "test", "how": "x", "why": "y", "location": "z"},}]}'
    )
    out = parse_caption_v3(malformed)
    assert out["_schema_ok"] is True  # json_repair recovered valid, schema-matching JSON
    assert out["actions"][0]["scene_what"] == "test"
    assert out["actions"][0]["task_step"] == "x"


def test_completely_broken_falls_through_to_single_degraded_action() -> None:
    """Last-tier regex grab still returns a usable (degraded) record,
    never raises, and is correctly marked schema_ok=False so it's excluded
    from s_label."""
    out = parse_caption_v3("this is not JSON at all garbage garbage garbage")
    assert out["_schema_ok"] is False
    assert len(out["actions"]) == 1
    a = out["actions"][0]
    assert a["scene_what"] is None
    assert a["left_hand"]["verb"] is None
    assert "garbage" in out["raw_json"]


def test_length_caps_enforced() -> None:
    a = _good_action()
    a["scene"]["what"] = "x" * 800
    out = parse_caption_v3(_response(a))
    assert len(out["actions"][0]["scene_what"]) <= 400


def test_fence_stripping_is_linear_not_quadratic() -> None:
    """BUG guard: the closing fence was stripped with an unanchored regex
    (`re.sub(r"\\s*```\\s*$", ...)`), which backtracks quadratically over a
    trailing whitespace run. Measured before the fix: 32k spaces took 1.6s,
    extrapolating to minutes at 1 MB — a hostile or malfunctioning model
    response would hang the caption worker on that window.

    200k spaces must complete essentially instantly. The generous 2s bound
    is a hang detector, not a benchmark; the pre-fix code took ~60s here.
    """
    import time

    payload = "```json\n" + '{"actions":[]}' + " " * 200_000 + "x"
    t0 = time.monotonic()
    _strip_code_fence(payload)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"fence stripping took {elapsed:.1f}s — the regex regressed"


def test_huge_whitespace_response_still_parses_without_hanging() -> None:
    payload = "```json\n" + '{"actions":[]}' + " " * 100_000
    out = parse_caption_v3(payload)
    # Empty actions list => not schema_ok (see the S4 fix), but it must
    # RETURN rather than hang.
    assert out["_schema_ok"] is False
