"""Characterization tests for the S6/S7 hand-assignment fix.

v1's `_pick_hand` returned the FIRST detection whose top handedness label
matched what was asked for — so when MediaPipe labeled BOTH detected hands
"Right" (common in egocentric footage), `_pick_hand(result, "Left")`
silently returned None even though a second hand was genuinely visible.

`HandAssigner` instead assigns both detections jointly: continuity with the
previous frame's slot positions is the primary signal, handedness labels
only break ties when there's no prior frame to anchor to.
"""

from __future__ import annotations

from egoannote import config
from egoannote.layers.hands import HandAssigner, _Detection


def _det(x: float, y: float, label: str, score: float = 0.9) -> _Detection:
    """A minimal detection: 21 landmarks all at (x, y, 0), so wrist (landmark
    0) sits at (x, y)."""
    landmarks = [x, y, 0.0] * 21
    return _Detection(landmarks, score, label)


def test_single_detection_labeled_left_goes_left() -> None:
    assigner = HandAssigner()
    left, right, l_tid, r_tid = assigner.assign([_det(0.3, 0.5, "Left")])
    assert left is not None and right is None
    assert l_tid is not None and r_tid is None


def test_two_detections_different_labels_assigned_by_label() -> None:
    assigner = HandAssigner()
    left, right, _, _ = assigner.assign([_det(0.3, 0.5, "Left"), _det(0.7, 0.5, "Right")])
    assert left is not None and right is not None
    assert left.label == "Left"
    assert right.label == "Right"


def test_v1_bug_scenario_both_labeled_right_still_yields_two_hands() -> None:
    """THE bug this fix exists for: both detections say "Right". v1 would
    return None for the left slot here (hands_present would under-count to
    1). This assigner must still recognize two distinct hands are present —
    it may not know which is anatomically left/right, but it must not drop
    one."""
    assigner = HandAssigner()
    left, right, l_tid, r_tid = assigner.assign(
        [_det(0.2, 0.5, "Right"), _det(0.8, 0.5, "Right")]
    )
    assert left is not None
    assert right is not None
    assert l_tid is not None
    assert r_tid is not None


def test_no_detections_clears_state() -> None:
    assigner = HandAssigner()
    assigner.assign([_det(0.3, 0.5, "Left")])
    left, right, l_tid, r_tid = assigner.assign([])
    assert left is None and right is None
    assert l_tid is None and r_tid is None


def test_continuity_prefers_nearest_slot_over_stale_label() -> None:
    """Once a previous frame has anchored left/right slots by position, a
    label swap on a LATER frame (common — MediaPipe's handedness classifier
    flips its guess frame to frame even for the same physical hand) must
    not cause the two hands to swap slots. This is the core of the S6/S7
    fix: continuity outranks the label once we have prior positions."""
    assigner = HandAssigner()
    # Frame 1: establishes left @ x=0.2, right @ x=0.8, both correctly labeled.
    assigner.assign([_det(0.2, 0.5, "Left"), _det(0.8, 0.5, "Right")])

    # Frame 2: hands barely moved, but MediaPipe's labels FLIPPED (both now
    # say "Left") — a real failure mode. Continuity must still keep the
    # physically-left hand (still near x=0.2) in the left slot.
    left, right, _, _ = assigner.assign([_det(0.22, 0.5, "Left"), _det(0.78, 0.5, "Left")])
    assert left.wrist[0] < right.wrist[0], "continuity must not let a mislabel swap the slots"


def test_track_id_increments_on_new_occupancy_episode() -> None:
    assigner = HandAssigner()
    _, _, l_tid_1, _ = assigner.assign([_det(0.3, 0.5, "Left")])
    # Hand leaves frame.
    assigner.assign([])
    # Hand reappears — should get a NEW track id, since occupancy was
    # interrupted (this is the "slot churn is visible downstream" property
    # from S6/S7, not full cross-frame re-identification).
    _, _, l_tid_2, _ = assigner.assign([_det(0.3, 0.5, "Left")])
    assert l_tid_1 is not None and l_tid_2 is not None
    assert l_tid_1 != l_tid_2


def test_track_id_stable_across_continuous_presence() -> None:
    assigner = HandAssigner()
    _, _, l_tid_1, _ = assigner.assign([_det(0.3, 0.5, "Left")])
    _, _, l_tid_2, _ = assigner.assign([_det(0.31, 0.5, "Left")])
    assert l_tid_1 == l_tid_2


# --------------------------------------------------------------------------
# BUG found in review: the module docstring claimed a physical-plausibility
# check guarded the two-detection path, but MAX_PLAUSIBLE_DELTA_PER_FRAME was
# only ever applied in the single-detection branch. _assign_by_continuity
# picks the CHEAPER of two pairings — and cheaper is not plausible. A wrist
# "teleporting" across the frame was recorded as smooth continuous motion,
# silently corrupting the velocity signal the segmentation layer reads.
# --------------------------------------------------------------------------


def test_implausible_jump_in_two_detection_path_starts_a_new_track() -> None:
    assigner = HandAssigner()
    # Frame 1: hands well separated at x=0.2 and x=0.8.
    _, _, _, r_tid_1 = assigner.assign([_det(0.2, 0.5, "Left"), _det(0.8, 0.5, "Right")])

    # Frame 2: the right hand has left frame and a false positive appeared
    # next to the left hand. The cheapest pairing still requires a ~0.55-unit
    # jump for the right slot — far beyond what a hand can do in one frame.
    _, right, _, r_tid_2 = assigner.assign(
        [_det(0.20, 0.5, "Left"), _det(0.25, 0.5, "Right")]
    )

    moved = abs(right.wrist[0] - 0.8)
    assert moved > config.MAX_PLAUSIBLE_DELTA_PER_FRAME
    assert r_tid_2 != r_tid_1, (
        "an implausible wrist jump must surface as a new track_id, not be "
        "recorded as the same hand moving smoothly"
    )


def test_plausible_motion_keeps_its_track_id() -> None:
    """The guard must not fire on ordinary hand motion — otherwise track ids
    churn constantly and the signal is just as useless."""
    assigner = HandAssigner()
    _, _, l1, r1 = assigner.assign([_det(0.2, 0.5, "Left"), _det(0.8, 0.5, "Right")])
    _, _, l2, r2 = assigner.assign([_det(0.24, 0.5, "Left"), _det(0.76, 0.5, "Right")])
    assert (l1, r1) == (l2, r2), "normal motion must not reset tracks"
