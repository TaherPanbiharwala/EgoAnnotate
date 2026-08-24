"""Parse the VLM's ordered-action-list response into validated records.

Legacy V3 schema (see prompts/caption_v3.txt): one window returns a LIST of actions,
each shaped like v1's single v2.5 object, plus start_frame/end_frame giving
its position within the window's 8 frames:

    {"actions": [
        {"start_frame": 0, "end_frame": 3, "task_step": "rinse_carrot",
         "left_hand": {...}, "right_hand": {...}, "tool_in_use": "...",
         "coordination": "...", "handover_event": false,
         "scene": {"what": "...", "how": "...", "why": "...", "location": "..."}},
        ...
    ]}

Unifying every action item under one shape (rather than a bare
{action, start_frac} pair) means the SAME comparison function computes
s_label whether two actions are adjacent within one window's list or
adjacent across a window seam (last action of window N vs. first of N+1) —
resolving the "two different formulas" problem the plan's Eng review found
in the split-schema alternative.

Robustness ladder (kept from v1, which held up under review):
    1. Strip ```json``` fences.
    2. json.loads() — happy path.
    3. json_repair fallback — fixes trailing commas, unquoted keys, etc.
    4. Regex extraction — best-effort single degraded action for the window.

CRITICAL FIX (plan S4 / A1-4): v1's `_parse_ok` meant "valid JSON," not
"matches the expected shape." Under this variable-length schema that's not
enough — a bare JSON array or a `{"actions": []}` wrapper both parse as
valid JSON while carrying no usable action. We validate against the actual
pydantic schema and report `_schema_ok`, not `_parse_ok`. Callers (the
segmentation layer) MUST gate `s_label` on `_schema_ok`, not on JSON
validity — see config.py and the plan's C1/S4 sections.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from . import config

log = logging.getLogger(__name__)


# Closed vocabularies — values outside these get coerced to None. The parser
# doesn't fight the model; it records the deviation cleanly (kept from v1).
CONTACT_TYPE_VOCAB = {
    "grip", "push", "pull", "tap", "slide", "twist", "pinch", "reach", "release", "none",
}
COORDINATION_VOCAB = {"independent", "supporting", "coordinated", "none"}
ACTIVITY_PHASE_VOCAB = {
    "starting", "preparing", "executing", "checking", "transitioning", "finishing", "idle",
}
PROGRESSION_VOCAB = {"starts", "continues", "changes", "completes", "unclear"}

TASK_STEP_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")

LEN_CAP = {
    "scene.what": 400,
    "scene.how": 400,
    "scene.why": 240,
    "scene.location": 60,
    "verb": 40,
    "object": 80,
    "target": 80,
    "tool_in_use": 24,
    "activity.caption": 500,
    "activity.goal": 240,
    "action_caption": 400,
    "hand.caption": 240,
}

# A frame index outside [0, VLM_WINDOW_FRAMES-1] is a schema violation, not
# just an out-of-range value. Derived, not hardcoded, so changing the window
# size can't leave the parser clamping to a stale bound.
_MAX_FRAME_IDX = config.VLM_WINDOW_FRAMES - 1


class HandBlock(BaseModel):
    verb: str | None = None
    object: str | None = None
    target: str | None = None
    contact_type: str | None = None
    visible: bool | None = None


class SceneBlock(BaseModel):
    what: str | None = None
    how: str | None = None
    why: str | None = None
    location: str | None = None


class ActionItem(BaseModel):
    start_frame: int
    end_frame: int
    task_step: str | None = None
    left_hand: HandBlock = HandBlock()
    right_hand: HandBlock = HandBlock()
    tool_in_use: str | None = None
    coordination: str | None = None
    handover_event: bool | None = None
    scene: SceneBlock = SceneBlock()


class ActionsResponse(BaseModel):
    actions: list[ActionItem]


class DenseHandBlock(HandBlock):
    """The structured hand relation plus a natural-language hand track."""

    caption: str | None = None


class ActivityBlock(BaseModel):
    """One holistic caption for the current window, independent of hand slots."""

    caption: str = Field(min_length=1)
    goal: str | None = None
    phase: str | None = None
    progression: str | None = None


class DenseActionItem(ActionItem):
    action_caption: str = Field(min_length=1)
    left_hand: DenseHandBlock = DenseHandBlock()
    right_hand: DenseHandBlock = DenseHandBlock()


class DenseCaptionResponse(BaseModel):
    activity: ActivityBlock
    actions: list[DenseActionItem]


def _strip_code_fence(raw: str) -> str:
    """Remove ```json ... ``` fencing.

    The closing fence is stripped with removesuffix, NOT a regex. The obvious
    pattern (`re.sub(r"\\s*```\\s*$", "", s)`) is unanchored at the start, so
    the engine retries at every offset while `\\s*` backtracks across the whole
    whitespace run — quadratic. Measured: 8k trailing spaces = 0.1s, 32k = 1.6s,
    extrapolating to minutes at 1 MB. A model returning a fence plus a large
    run of whitespace would hang the caption worker on that window.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?[ \t]*\n?", "", s)
        s = s.strip().removesuffix("```").strip()
    return s


def _try_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _try_json_repair(s: str) -> Any:
    try:
        from json_repair import repair_json
    except ImportError:
        return None
    try:
        return json.loads(repair_json(s))
    except Exception:
        return None


def _regex_grab_single_action(raw: str, n_frames: int = config.VLM_WINDOW_FRAMES) -> dict:
    """Last-resort: pull whatever we can into ONE degraded action spanning
    the whole window. Marked _schema_ok=False downstream — S4's rule means
    this action's fields must NOT contribute to s_label."""

    def grab(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', raw)
        return m.group(1) if m else None

    return {
        "start_frame": 0,
        "end_frame": max(0, n_frames - 1),
        "task_step": grab("task_step"),
        "left_hand": {"verb": None, "object": None, "target": None,
                      "contact_type": None, "visible": None},
        "right_hand": {"verb": None, "object": None, "target": None,
                       "contact_type": None, "visible": None},
        "tool_in_use": grab("tool_in_use"),
        "coordination": grab("coordination"),
        "handover_event": None,
        "scene": {"what": grab("what"), "how": grab("how"),
                  "why": grab("why"), "location": grab("location")},
    }


def _clamp_str(value: Any, cap: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value[:cap] if value else None


def _normalize_vocab(value: Any, vocab: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in vocab else None


def _slugify_task_step(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if not v:
        return None
    v = re.sub(r"[^a-z0-9]+", "_", v).strip("_")[:24]
    if not v or not v[0].isalpha():
        return None
    return v if TASK_STEP_RE.match(v) else None


def _clamp_hand(hand: HandBlock) -> dict:
    """Validate + clamp a per-hand block, applying the visible/verb
    contradiction rules kept from v1 (these held up under review)."""
    visible = hand.visible if isinstance(hand.visible, bool) else None
    verb = _clamp_str(hand.verb, LEN_CAP["verb"])
    obj = _clamp_str(hand.object, LEN_CAP["object"])
    target = _clamp_str(hand.target, LEN_CAP["target"])
    contact = _normalize_vocab(hand.contact_type, CONTACT_TYPE_VOCAB)
    caption = _clamp_str(getattr(hand, "caption", None), LEN_CAP["hand.caption"])

    if visible is True and verb is None:
        visible, verb, obj, target, contact, caption = False, None, None, None, None, None
    if visible is False:
        verb, obj, target, contact, caption = None, None, None, None, None

    return {"verb": verb, "object": obj, "target": target,
            "contact_type": contact, "visible": visible, "caption": caption}


def _frame_range_ok(item: ActionItem, n_frames: int) -> bool:
    """Is this action's frame range actually usable?

    CRITICAL (found in review): this used to be a silent clamp,
    `max(0, min(_MAX_FRAME_IDX, x))`, while the module comment claimed an
    out-of-range index was "a schema violation, not just an out-of-range
    value." Clamping made three real failure modes indistinguishable from
    clean output, all reported as schema_ok=True:

      * `end_frame: 15` -> clamped to 7. Silently truncated.
      * `start_frame: 7, end_frame: 0` -> preserved inverted, producing a
        negative-duration segment downstream.
      * `start_frame: 1, end_frame: 8` -> became 1..7. This is the dangerous
        one: 1-indexed output is common LLM behavior, and clamping turned it
        into a plausible-looking range shifted by one frame (0.75s at 4/3
        fps) on EVERY action, marked trustworthy.

    `schema_ok` is what gates whether an action contributes to the boundary
    signal, so a lie here is worse than a parse failure.

    Bound is the ACTUAL window length, not the global window size: trailing
    windows are short (see media.frames.Window.is_partial), and an action
    claiming frame 7 of a 3-frame window refers to an image never sent.
    """
    max_idx = n_frames - 1
    if not (0 <= item.start_frame <= max_idx):
        return False
    if not (0 <= item.end_frame <= max_idx):
        return False
    return item.start_frame <= item.end_frame


def _clamp_action(item: ActionItem) -> dict:
    return {
        "start_frame": item.start_frame,
        "end_frame": item.end_frame,
        "action_caption": _clamp_str(
            getattr(item, "action_caption", None), LEN_CAP["action_caption"]
        ),
        "task_step": _slugify_task_step(item.task_step),
        "left_hand": _clamp_hand(item.left_hand),
        "right_hand": _clamp_hand(item.right_hand),
        "tool_in_use": _clamp_str(item.tool_in_use, LEN_CAP["tool_in_use"]),
        "coordination": _normalize_vocab(item.coordination, COORDINATION_VOCAB),
        "handover_event": item.handover_event if isinstance(item.handover_event, bool) else None,
        "scene_what": _clamp_str(item.scene.what, LEN_CAP["scene.what"]),
        "scene_how": _clamp_str(item.scene.how, LEN_CAP["scene.how"]),
        "scene_why": _clamp_str(item.scene.why, LEN_CAP["scene.why"]),
        "scene_location": _clamp_str(item.scene.location, LEN_CAP["scene.location"]),
    }


def _clamp_activity(activity: ActivityBlock) -> dict:
    return {
        "caption": _clamp_str(activity.caption, LEN_CAP["activity.caption"]),
        "goal": _clamp_str(activity.goal, LEN_CAP["activity.goal"]),
        "phase": _normalize_vocab(activity.phase, ACTIVITY_PHASE_VOCAB),
        "progression": _normalize_vocab(activity.progression, PROGRESSION_VOCAB),
    }


def _dense_hand_ok(hand: DenseHandBlock) -> bool:
    """Check the dense-hand fields *after* their output normalization.

    Pydantic accepts whitespace-only strings and the closed vocabularies are
    deliberately normalized after validation.  The schema-validity flag must
    describe the stored record, not the pre-normalized response: otherwise a
    model can return ``visible=true`` plus ``caption=\"   \"`` and the stored
    caption becomes ``None`` while ``schema_ok`` incorrectly remains true.
    """
    visible = hand.visible
    if not isinstance(visible, bool):
        return False
    if not visible:
        return True
    return bool(
        _clamp_str(hand.caption, LEN_CAP["hand.caption"])
        and _clamp_str(hand.verb, LEN_CAP["verb"])
    )


def parse_caption_v3(raw: str, n_frames: int = config.VLM_WINDOW_FRAMES) -> dict:
    """Parse a VLM ordered-action-list response.

    Args:
        raw: the model's response text.
        n_frames: how many frames were ACTUALLY sent in this window. Trailing
            windows are short (media.frames.Window.is_partial), and an action
            indexing frame 7 of a 3-frame window refers to an image the model
            never saw. Defaults to a full window.

    Returns:
        {
          "actions": [ {..action dict..}, ... ],
          "raw_json": raw,
          "_schema_ok": bool,   # response matched the schema AND every frame
                                # range is in-bounds and correctly ordered
        }

    `_schema_ok=False` means every action in this window is a degradation
    fallback and must be excluded from s_label comparisons (config/S4 rule) —
    the caller emits NaN for any comparison touching one, not 0.
    """
    s = _strip_code_fence(raw)

    obj = _try_json_loads(s)
    if obj is None:
        obj = _try_json_repair(s)

    schema_ok = False
    actions: list[dict] = []

    if obj is not None:
        # Tolerate a bare list (model skips the {"actions": [...]} wrapper)
        # by wrapping it before validation — still schema-checked either way.
        if isinstance(obj, list):
            obj = {"actions": obj}
        try:
            validated = ActionsResponse.model_validate(obj)
            if validated.actions:
                # Shape is valid; now check the VALUES. A response can match
                # the schema perfectly and still be unusable (see
                # _frame_range_ok). Both must hold for schema_ok.
                ranges_ok = all(_frame_range_ok(a, n_frames) for a in validated.actions)
                if not ranges_ok:
                    log.warning(
                        "parse: frame ranges out of bounds or inverted for a "
                        "%d-frame window (got %s) — marking schema_ok=False so "
                        "these actions do not contribute to boundary detection",
                        n_frames,
                        [(a.start_frame, a.end_frame) for a in validated.actions],
                    )
                actions = [_clamp_action(a) for a in validated.actions]
                schema_ok = ranges_ok
            # An empty actions list is valid JSON, valid schema, and useless —
            # explicitly NOT schema_ok, matching the plan's A1-4 fix.
        except ValidationError:
            pass

    if not actions:
        log.warning(
            "parse: schema validation failed, falling back to single degraded "
            "action (first 200 chars: %r)", s[:200],
        )
        actions = [
            _clamp_action(ActionItem.model_validate(_regex_grab_single_action(s, n_frames)))
        ]

    return {"actions": actions, "raw_json": raw, "_schema_ok": schema_ok}


def parse_caption_v4(raw: str, n_frames: int = config.VLM_WINDOW_FRAMES) -> dict:
    """Parse the dense multi-track caption schema used by ``caption_v4``.

    V4 deliberately keeps three parallel representations in one atomic VLM
    response: a holistic window activity caption, ordered atomic actions, and
    a natural-language caption for each visible hand. The v3 parser remains
    available for old stored responses; new runs are strict about the v4
    fields so a legacy-looking response cannot silently masquerade as dense.
    """
    s = _strip_code_fence(raw)
    obj = _try_json_loads(s)
    if obj is None:
        obj = _try_json_repair(s)

    if obj is not None:
        try:
            validated = DenseCaptionResponse.model_validate(obj)
            if validated.actions:
                ranges_ok = all(_frame_range_ok(a, n_frames) for a in validated.actions)
                activity = _clamp_activity(validated.activity)
                actions = [_clamp_action(a) for a in validated.actions]
                dense_fields_ok = (
                    bool(activity["caption"])
                    and activity["phase"] is not None
                    and activity["progression"] is not None
                    and all(
                        bool(action["action_caption"])
                        and _dense_hand_ok(source.left_hand)
                        and _dense_hand_ok(source.right_hand)
                        for action, source in zip(actions, validated.actions, strict=True)
                    )
                )
                schema_ok = ranges_ok and dense_fields_ok
                if not schema_ok:
                    log.warning(
                        "parse v4: invalid frame range or incomplete dense fields "
                        "for a %d-frame window", n_frames,
                    )
                return {
                    "activity": activity,
                    "actions": actions,
                    "raw_json": raw,
                    "_schema_ok": schema_ok,
                }
        except ValidationError:
            pass

    log.warning(
        "parse v4: dense schema validation failed; preserving a degraded v3 "
        "action so the raw response remains inspectable (first 200 chars: %r)",
        s[:200],
    )
    legacy = parse_caption_v3(raw, n_frames=n_frames)
    return {
        "activity": {"caption": None, "goal": None, "phase": None, "progression": None},
        "actions": legacy["actions"],
        "raw_json": raw,
        "_schema_ok": False,
    }
