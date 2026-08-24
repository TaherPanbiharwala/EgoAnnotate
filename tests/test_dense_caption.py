from __future__ import annotations

import json

from egoannote.parse import parse_caption_v4


def _dense_response() -> dict:
    return {
        "activity": {
            "caption": "The person is opening a jar on the counter.",
            "goal": "Open the jar for the next step.",
            "phase": "executing",
            "progression": "continues",
        },
        "actions": [
            {
                "start_frame": 0,
                "end_frame": 7,
                "action_caption": "The person twists the lid while holding the jar.",
                "task_step": "open_jar",
                "left_hand": {
                    "caption": "The left hand grips and steadies the glass jar.",
                    "verb": "holding",
                    "object": "glass jar",
                    "target": None,
                    "contact_type": "grip",
                    "visible": True,
                },
                "right_hand": {
                    "caption": "The right hand twists the metal lid.",
                    "verb": "twisting",
                    "object": "metal lid",
                    "target": None,
                    "contact_type": "twist",
                    "visible": True,
                },
                "tool_in_use": "none",
                "coordination": "coordinated",
                "handover_event": False,
                "scene": {
                    "what": "The person opens a jar.",
                    "how": "Both hands apply opposing force.",
                    "why": "To remove the lid.",
                    "location": "kitchen counter",
                },
            }
        ],
    }


def test_v4_keeps_holistic_atomic_and_per_hand_tracks() -> None:
    parsed = parse_caption_v4(json.dumps(_dense_response()))
    assert parsed["_schema_ok"] is True
    assert parsed["activity"]["caption"].startswith("The person")
    assert parsed["actions"][0]["action_caption"].startswith("The person")
    assert parsed["actions"][0]["left_hand"]["caption"].startswith("The left hand")
    assert parsed["actions"][0]["right_hand"]["caption"].startswith("The right hand")


def test_v4_rejects_a_visible_hand_without_its_caption() -> None:
    response = _dense_response()
    response["actions"][0]["left_hand"]["caption"] = None
    parsed = parse_caption_v4(json.dumps(response))
    assert parsed["_schema_ok"] is False


def test_v4_rejects_whitespace_only_dense_fields_after_normalization() -> None:
    response = _dense_response()
    response["activity"]["phase"] = "not_a_phase"
    response["actions"][0]["right_hand"]["caption"] = "   "
    parsed = parse_caption_v4(json.dumps(response))
    assert parsed["_schema_ok"] is False
    assert parsed["actions"][0]["right_hand"]["caption"] is None


def test_v4_marks_legacy_v3_shape_as_degraded() -> None:
    parsed = parse_caption_v4('{"actions":[{"start_frame":0,"end_frame":7}]}')
    assert parsed["_schema_ok"] is False
    assert parsed["activity"]["caption"] is None
