"""Tests for jobs/10_blur_egoblur.py.

Every test here pins a bug that was actually found and fixed, not a
hypothetical. The job runs unattended on a paid GPU against footage of real
people, so the failure mode that matters is not "it crashed" — it is "it
reported PASS while a face shipped unredacted". Most of these assert on that
directly.

Deliberately NOT covered, because a mock would only pin the mock:
Gen2Detector/Gen1Detector.detect_batch (need real weights and a GPU; the
module docstring lists the specific unverified assumptions), probe_and_budget's
TIMING (its whole job there is measuring real hardware), and
arm_watchdog/shutdown_pod (need runpodctl and a live pod).

probe_and_budget's frame SELECTION is covered, and was not always: this file
used to exempt the whole function on the "measures real hardware" argument,
which quietly extended an honest exemption for the timing math to the pure
logic sitting next to it. Four real bugs lived in that gap until the job was
run against actual footage — see the probe section at the bottom.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BASE_ARGS = [
    "--input-dir", "/tmp/in",
    "--output-dir", "/tmp/out",
    "--checkpoint-dir", "/tmp/private/checkpoints",
    "--run-id", "r",
]


def _clip(blur_job, **over):
    kw = dict(path=Path("x.mp4"), clip_id="x", clip_group="x", chapter_index=0,
              width=64, height=64, fps=30.0, n_frames=10, duration_s=1.0,
              rotation=0, sha256="0" * 64)
    kw.update(over)
    return blur_job.ClipInfo(**kw)


def _planes(w=64, h=64, fill=0):
    return (np.full((h, w), fill, np.uint8),
            np.full((h // 2, w // 2), fill, np.uint8),
            np.full((h // 2, w // 2), fill, np.uint8))


class _Proc:
    """Stand-in for the ffmpeg decoder Popen."""

    def __init__(self, returncode=0):
        self.returncode = returncode

    def wait(self):
        return self.returncode


# ---------------------------------------------------------------------------
# Coverage metric. Was `_iou(...) > 0.1`, which classified a face with half
# its pixels OUTSIDE the fill box as covered (IoU 0.333) — a false negative
# in the privacy direction, reported as PASS.
# ---------------------------------------------------------------------------


def test_half_covered_face_is_not_considered_covered(blur_job):
    face = (0.0, 0.0, 100.0, 100.0)
    right_half_only = (50.0, 0.0, 150.0, 100.0)
    # The old metric said "covered"; spell that out so the regression is legible.
    assert blur_job._iou(face, right_half_only) > 0.1
    assert blur_job._covered_fraction(face, [right_half_only]) == pytest.approx(0.5)
    assert not blur_job._box_is_covered(face, [right_half_only])


def test_small_face_fully_inside_a_large_box_is_covered(blur_job):
    """The symmetric error: IoU 0.0068 read as uncovered and hard-failed the
    clip for nothing."""
    small_face = (100.0, 100.0, 112.0, 112.0)
    big_fill = (0.0, 0.0, 146.0, 146.0)
    assert blur_job._iou(small_face, big_fill) < 0.1
    assert blur_job._box_is_covered(small_face, [big_fill])


def test_face_straddling_two_adjacent_boxes_is_covered(blur_job):
    """Coverage is judged against the UNION. A face across two boxes overlaps
    neither one enough on its own."""
    face = (0.0, 0.0, 100.0, 100.0)
    left = (-5.0, -5.0, 50.0, 105.0)
    right = (50.0, -5.0, 105.0, 105.0)
    assert blur_job._box_is_covered(face, [left, right])


def test_overlapping_boxes_are_not_double_counted(blur_job):
    """Union area, not summed area — otherwise two overlapping half-covers
    would add to >1.0 and read as fully covered."""
    face = (0.0, 0.0, 100.0, 100.0)
    a = (0.0, 0.0, 60.0, 100.0)
    b = (10.0, 0.0, 60.0, 100.0)  # entirely inside a's span
    assert blur_job._covered_fraction(face, [a, b]) == pytest.approx(0.6)


def test_uncovered_face_with_no_boxes(blur_job):
    assert blur_job._covered_fraction((0.0, 0.0, 10.0, 10.0), []) == 0.0


# ---------------------------------------------------------------------------
# The low-threshold sweep. Was structurally dead: detectors filtered at the
# operating threshold, so the [sweep, operating) band was unreachable and
# n_candidate_misses was permanently 0 — while gating status.
# ---------------------------------------------------------------------------


def test_sub_threshold_uncovered_detection_is_flagged(blur_job):
    sub = [blur_job.Detection(frame_idx=0, cls="face", box=(0.0, 0.0, 50.0, 50.0),
                               score=0.15)]
    out = blur_job.check_low_threshold_sweep(sub, {}, 0.10, {"face": 0.30, "lp": 0.40})
    assert out["n_candidate_misses"] == 1


def test_sub_threshold_detection_already_covered_is_not_flagged(blur_job):
    sub = [blur_job.Detection(frame_idx=0, cls="face", box=(10.0, 10.0, 50.0, 50.0),
                               score=0.15)]
    out = blur_job.check_low_threshold_sweep(
        sub, {0: [(0.0, 0.0, 64.0, 64.0)]}, 0.10, {"face": 0.30, "lp": 0.40})
    assert out["n_candidate_misses"] == 0


def test_above_threshold_detections_are_not_sweep_candidates(blur_job):
    """Those are redacted normally; only the low-confidence band is a
    'maybe we missed this' signal."""
    dets = [blur_job.Detection(frame_idx=0, cls="face", box=(0.0, 0.0, 50.0, 50.0),
                                score=0.9)]
    out = blur_job.check_low_threshold_sweep(dets, {}, 0.10, {"face": 0.30})
    assert out["n_candidate_misses"] == 0


def test_sweep_threshold_must_be_below_operating_thresholds(blur_job):
    """The wiring guard. If the floor is not strictly below the operating
    threshold the band is empty again and the whole check silently dies."""
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--sweep-threshold", "0.9",
                             "--face-threshold", "0.3"])


def test_pose_shadow_reports_wearer_overlap_without_mutating_detection_or_tracks(blur_job, tmp_path):
    """The pre-redaction prior is review-only: it may never change fill input."""
    clip = _clip(blur_job, sha256="a" * 64)
    detection = blur_job.Detection(0, "face", (0.0, 0.0, 10.0, 10.0), 0.9)
    track = blur_job.Track(track_id=7, cls="face", last_frame=0)
    track.frames[0] = (detection.box, "det")
    before_detections = copy.deepcopy([detection])
    before_frames = copy.deepcopy(track.frames)
    cfg = SimpleNamespace(detect_hz=10.0, face_threshold=0.3, min_track_confirmations=2)
    prior_path = tmp_path / "private" / "clip.pose_prior.json"
    prior_path.parent.mkdir()
    prior_path.write_text("{}", encoding="utf-8")
    report = blur_job.build_pose_shadow_report(
        clip,
        cfg,
        prior_path,
        "b" * 64,
        {0: {"all": [], "wearer": []}},
        [detection],
        [track],
    )

    assert [detection] == before_detections
    assert track.frames == before_frames
    assert report["shadow_only"] is True
    assert report["detection_samples"][0]["wearer_limb_overlap"] is None


def test_pink_hand_overlap_reports_fill_amplification_without_changing_tracks(blur_job, tmp_path):
    """The private report retains pink evidence independently of an active run."""
    clip = _clip(blur_job, sha256="a" * 64)
    hand = {
        "shape": "circle", "kind": "hand", "center": [5.0, 50.0], "radius": 20.0,
        "track_id": 12,
    }
    detections = [
        blur_job.Detection(frame_idx, "face", (0.0, 45.0, 10.0, 55.0), 0.9)
        for frame_idx in (0, 3, 6)
    ]
    track = blur_job.Track(track_id=7, cls="face", last_frame=8)
    for detection in detections:
        track.frames[detection.frame_idx] = (detection.box, "det")
    track.frames[4] = (detections[0].box, "interp")
    track.frames[8] = (detections[0].box, "hold")
    regions = {
        frame_idx: {
            "all": [hand], "wearer": [hand], "provisional_wearer": [hand],
            "stable_wearer": [],
        }
        for frame_idx in (0, 3, 6)
    }
    cfg = SimpleNamespace(detect_hz=10.0, face_threshold=0.3, min_track_confirmations=2)
    prior_path = tmp_path / "private" / "clip.hand_prior.json"
    prior_path.parent.mkdir()
    prior_path.write_text("{}", encoding="utf-8")
    before_detections = copy.deepcopy(detections)
    before_frames = copy.deepcopy(track.frames)

    report = blur_job.build_hand_suppression_report(
        clip, cfg, prior_path, "b" * 64, regions, detections, [track], []
    )

    assert detections == before_detections
    assert track.frames == before_frames
    assert report["summary"]["n_pink_amplification_candidates"] == 1
    assert report["summary"]["n_pink_amplification_extended_fill_frames"] == 2
    candidate = report["pink_amplification_candidates"][0]
    assert candidate["provisional_hand_track_id"] == 12
    assert candidate["n_interpolated_fill_frames"] == 1
    assert candidate["n_hold_fill_frames"] == 1
    assert candidate["suggested_action"] == "review_only_no_pixel_change"
    assert report["detection_samples"][0]["provisional_wearer_hand_overlap"] == 1.0


def test_pink_generated_fill_demotion_retains_raw_detections(blur_job):
    """Pink evidence may trim amplification, never a detector-backed face hit."""
    hand = {
        "shape": "circle", "kind": "hand", "center": [5.0, 50.0], "radius": 20.0,
        "track_id": 12,
    }
    track = blur_job.Track(track_id=7, cls="face", last_frame=8)
    box = (0.0, 45.0, 10.0, 55.0)
    track.frames = {
        0: (box, "det"),
        1: (box, "interp"),
        4: (box, "interp"),
        8: (box, "det"),
        9: (box, "hold"),
        10: (box, "det"),
        12: (box, "hold"),
    }

    demoted = blur_job.demote_provisional_hand_generated_fills(
        [track], {0: [hand], 8: [hand], 10: [hand]}, context_frames=1
    )

    assert set(track.frames) == {0, 1, 8, 9, 10}
    assert track.frames[0][1] == "det"
    assert track.frames[8][1] == "det"
    assert len(demoted) == 1
    assert demoted[0]["provisional_hand_track_id"] == 12
    assert demoted[0]["n_generated_fill_frames_removed"] == 2
    assert demoted[0]["n_interpolated_fill_frames_removed"] == 1
    assert demoted[0]["n_hold_fill_frames_removed"] == 1
    assert demoted[0]["policy"] == "raw_detections_retained_generated_fills_capped"


def test_amber_and_pink_have_intentionally_nearby_raw_suppression_gates(blur_job, monkeypatch):
    """Pink can suppress raw tracks, but needs only slightly stronger overlap."""
    hand = {
        "shape": "circle", "kind": "hand", "center": [5.0, 50.0], "radius": 20.0,
        "track_id": 12,
    }
    monkeypatch.setattr(blur_job, "pose_overlap_fraction", lambda *_args: 0.91)
    box = (0.0, 45.0, 10.0, 55.0)

    amber = blur_job.Track(track_id=1, cls="face", last_frame=9)
    for frame_idx in (0, 3, 6, 9):
        amber.frames[frame_idx] = (box, "det")
    kept, suppressed = blur_job.suppress_stable_wearer_hand_tracks(
        [amber], {frame_idx: [hand] for frame_idx in (0, 3, 6, 9)}
    )
    assert kept == []
    assert suppressed[0]["suppression_state"] == "amber"

    pink = blur_job.Track(track_id=2, cls="face", last_frame=3)
    pink.frames = {0: (box, "det"), 3: (box, "det"), 6: (box, "det")}
    kept, suppressed = blur_job.suppress_provisional_wearer_hand_tracks(
        [pink], {0: [hand], 3: [hand], 6: [hand]}
    )
    assert kept == [pink]
    assert suppressed == []

    monkeypatch.setattr(blur_job, "pose_overlap_fraction", lambda *_args: 0.93)
    kept, suppressed = blur_job.suppress_provisional_wearer_hand_tracks(
        [pink], {0: [hand], 3: [hand], 6: [hand]}
    )
    assert kept == []
    assert suppressed[0]["suppression_state"] == "pink"


def test_prior_flags_must_be_private_and_paired(blur_job, capsys):
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--pose-prior", "/tmp/private/prior.json"])
    with pytest.raises(SystemExit):
        blur_job.parse_args(
            [
                *BASE_ARGS,
                "--pose-prior", "/tmp/prior.json",
                "--pose-shadow-report", "/tmp/private/report.json",
            ]
        )
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--hand-prior", "/tmp/private/prior.json"])
    with pytest.raises(SystemExit):
        blur_job.parse_args(
            [
                *BASE_ARGS,
                "--hand-prior", "/tmp/prior.json",
                "--hand-suppression-report", "/tmp/private/report.json",
            ]
        )
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--hand-suppress-wearer-hands"])
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--pink-demote-generated-fills"])
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--pink-suppress-wearer-hands"])
    with pytest.raises(SystemExit):
        blur_job.parse_args(
            [
                *BASE_ARGS,
                "--detect-hz", "30",
                "--hand-prior", "/tmp/private/prior.json",
                "--hand-suppression-report", "/tmp/private/report.json",
                "--hand-suppress-wearer-hands",
            ]
        )
    assert "requires --detect-hz 10.0" in capsys.readouterr().err


def test_active_hand_suppression_accepts_a_same_hand_majority(blur_job):
    hand = {
        "shape": "circle",
        "kind": "hand",
        "center": [5, 50],
        "radius": 20,
        "track_id": 12,
    }
    stable = {frame_idx: [hand] for frame_idx in (0, 3, 6, 9)}

    all_hand = blur_job.Track(track_id=1, cls="face", last_frame=9)
    for frame_idx in (0, 3, 6, 9):
        all_hand.frames[frame_idx] = ((0.0, 45.0, 10.0, 55.0), "det")
    mixed = blur_job.Track(track_id=2, cls="face", last_frame=9)
    for frame_idx, box in (
        (0, (0.0, 45.0, 10.0, 55.0)),
        (3, (0.0, 45.0, 10.0, 55.0)),
        (6, (35.0, 0.0, 55.0, 20.0)),  # cannot be proven to be the hand
        (9, (0.0, 45.0, 10.0, 55.0)),
    ):
        mixed.frames[frame_idx] = (box, "det")

    kept, suppressed = blur_job.suppress_stable_wearer_hand_tracks(
        [all_hand, mixed], stable
    )

    assert kept == []
    assert [entry["track_id"] for entry in suppressed] == [1, 2]
    assert suppressed[1]["n_overlap_hits"] == 3
    assert suppressed[1]["required_overlap_hits"] == 3


def test_active_hand_suppression_rejects_an_unstable_or_mixed_track(blur_job):
    hand = {
        "shape": "circle",
        "kind": "hand",
        "center": [5, 50],
        "radius": 20,
        "track_id": 12,
    }
    track = blur_job.Track(track_id=1, cls="face", last_frame=9)
    for frame_idx in (0, 3, 6, 9):
        track.frames[frame_idx] = ((0.0, 45.0, 10.0, 55.0), "det")
    kept, suppressed = blur_job.suppress_stable_wearer_hand_tracks(
        [track], {0: [hand], 3: [hand]}
    )
    assert kept == [track]
    assert suppressed == []


def test_active_hand_suppression_requires_one_hand_track_for_a_majority(blur_job):
    first = {"shape": "circle", "kind": "hand", "center": [5, 50], "radius": 20, "track_id": 1}
    second = {"shape": "circle", "kind": "hand", "center": [5, 50], "radius": 20, "track_id": 2}
    track = blur_job.Track(track_id=1, cls="face", last_frame=9)
    for frame_idx in (0, 3, 6, 9):
        track.frames[frame_idx] = ((0.0, 45.0, 10.0, 55.0), "det")
    kept, suppressed = blur_job.suppress_stable_wearer_hand_tracks(
        [track], {0: [first], 3: [first], 6: [second], 9: [second]}
    )
    assert kept == [track]
    assert suppressed == []


def test_hand_prior_loader_rejects_weakened_or_stale_active_evidence(blur_job, tmp_path):
    clip = _clip(blur_job, sha256="a" * 64)
    path = tmp_path / "private" / "x.hand_prior.json"
    report = tmp_path / "private" / "x.hand_suppression.json"
    hand = {
        "shape": "circle",
        "kind": "hand",
        "center": [5.0, 50.0],
        "radius": 20.0,
        "landmarks": [[0.1, 0.8]] * 21,
        "track_id": 1,
        "wearer_candidate": True,
        "stable_wearer_candidate": True,
    }
    rows = [{"frame_idx": frame_idx, "hands": [hand]} for frame_idx in range(10)]
    artifact = {
        "schema_version": 1,
        "artifact_type": "pre_redaction_hand_prior",
        "source": {
            "video_id": "x",
            "sha256": "a" * 64,
            "width": 64,
            "height": 64,
            "fps": 30.0,
            "n_frames": 10,
        },
        "sampling": {"detect_hz": 30.0},
        "model": {
            "name": blur_job.HAND_PRIOR_MODEL_NAME,
            "url": blur_job.HAND_PRIOR_MODEL_URL,
            "sha256": blur_job.HAND_PRIOR_MODEL_SHA256,
        },
        "configuration": dict(blur_job.HAND_ACTIVE_CONFIGURATION),
        "frames": rows,
    }
    path.parent.mkdir()
    path.write_text(json.dumps(artifact), encoding="utf-8")
    cfg = blur_job.parse_args(
        [
            *BASE_ARGS,
            "--hand-prior", str(path),
            "--hand-suppression-report", str(report),
            "--hand-suppress-wearer-hands",
        ]
    )
    loaded = blur_job.load_hand_prior_for_clip(cfg, clip)
    assert loaded is not None
    assert set(loaded[2]) == {0, 3, 6, 9}

    artifact["configuration"]["wearer_edge_margin"] = 0.5
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        blur_job.load_hand_prior_for_clip(cfg, clip)

    artifact["configuration"] = dict(blur_job.HAND_ACTIVE_CONFIGURATION)
    artifact["frames"][0]["hands"][0]["wearer_candidate"] = False
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="stable wearer hand"):
        blur_job.load_hand_prior_for_clip(cfg, clip)


def test_batch_refuses_a_single_hand_suppression_report(blur_job, monkeypatch):
    """Every active-suppression decision needs retained private evidence."""
    clips = [_clip(blur_job), _clip(blur_job, clip_id="y")]
    monkeypatch.setattr(blur_job, "preflight", lambda cfg: {})
    monkeypatch.setattr(blur_job, "_bin", lambda name: name)
    monkeypatch.setattr(blur_job, "discover_clips", lambda *_args: clips)

    with pytest.raises(RuntimeError, match="hand-suppression-report must be a directory"):
        blur_job.main(
            [
                *BASE_ARGS,
                "--no-watchdog",
                "--hand-prior", "/tmp/private/prior.json",
                "--hand-suppression-report", "/tmp/private/report.json",
            ]
        )


def test_checkpoint_dir_must_be_private_and_outside_publishable_output(blur_job):
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--checkpoint-dir", "/tmp/checkpoints"])
    with pytest.raises(SystemExit):
        blur_job.parse_args(
            [
                "--input-dir", "/tmp/in",
                "--output-dir", "/tmp/private/out",
                "--checkpoint-dir", "/tmp/private/out/checkpoints",
                "--run-id", "r",
            ]
        )


# ---------------------------------------------------------------------------
# The audit gate. Every term is a COUNT, and all counts are zero when the
# pipeline produced nothing — so total failure read as a clean pass.
# ---------------------------------------------------------------------------


def _audit(blur_job, *, fill=None, integrity=None, sweep=None, yunet=None,
           hand_suppressed=0, pink_suppressed=0, pink_demoted=0):
    return blur_job.build_audit(
        _clip(blur_job),
        fill if fill is not None else {"n_frames_with_fill": 10},
        integrity if integrity is not None else {"fill_integrity_violations": 0,
                                                  "fill_integrity_checked": 5},
        sweep if sweep is not None else {"n_candidate_misses": 0},
        yunet if yunet is not None else {"n_yunet_uncovered": 0},
        "2",
        n_hand_suppressed_tracks=hand_suppressed,
        n_pink_suppressed_tracks=pink_suppressed,
        n_pink_demoted_tracks=pink_demoted,
        n_pink_generated_fill_frames_removed=pink_demoted * 2,
    )


def test_clean_run_passes(blur_job):
    assert _audit(blur_job)["status"] == "PASS_AUTOMATED"


def test_skipped_yunet_is_a_visibly_weaker_pass(blur_job):
    a = _audit(blur_job, yunet={"yunet_skipped": "no --yunet-model provided"})
    assert a["status"] == "PASS_AUTOMATED_NO_YUNET"
    assert a["yunet_ran"] is False


def test_active_hand_suppression_always_requires_review(blur_job):
    audit = _audit(blur_job, hand_suppressed=1)
    assert audit["status"] == "NEEDS_REVIEW"
    assert "amber Hand Landmarker suppression" in " ".join(audit["status_reasons"])


def test_pink_raw_hand_suppression_always_requires_review(blur_job):
    audit = _audit(blur_job, pink_suppressed=1)
    assert audit["status"] == "NEEDS_REVIEW"
    assert "pink Hand Landmarker suppression" in " ".join(audit["status_reasons"])


def test_pink_generated_fill_demotion_always_requires_review(blur_job):
    audit = _audit(blur_job, pink_demoted=1)
    assert audit["status"] == "NEEDS_REVIEW"
    assert "pink Hand Landmarker evidence" in " ".join(audit["status_reasons"])


def test_zero_redacted_frames_never_passes(blur_job):
    """The canary. A run where detection silently produced nothing redacted
    nothing, verified nothing, and used to report PASS with exit 0."""
    a = _audit(blur_job, fill={"n_frames_with_fill": 0})
    assert a["status"] == "NEEDS_REVIEW"
    assert any("ZERO frames" in r for r in a["status_reasons"])


def test_skipped_integrity_check_never_passes(blur_job):
    """--redaction blur skips the integrity check entirely; .get(..., 0) then
    contributed 0 violations and the clip passed having verified nothing."""
    a = _audit(blur_job, integrity={"integrity_skipped": "mode=blur"})
    assert a["status"] == "NEEDS_REVIEW"
    assert a["integrity_ran"] is False


@pytest.mark.parametrize("kwargs", [
    {"integrity": {"fill_integrity_violations": 1, "fill_integrity_checked": 5}},
    {"sweep": {"n_candidate_misses": 1}},
    {"yunet": {"n_yunet_uncovered": 1}},
])
def test_any_positive_finding_forces_review(blur_job, kwargs):
    assert _audit(blur_job, **kwargs)["status"] == "NEEDS_REVIEW"


def test_integrity_that_checked_nothing_despite_a_fill_map_fails(blur_job):
    a = _audit(blur_job, fill={"n_frames_with_fill": 10},
               integrity={"fill_integrity_violations": 0, "fill_integrity_checked": 0})
    assert a["status"] == "NEEDS_REVIEW"


def test_skip_reasons_from_both_checks_do_not_collide(blur_job):
    """integrity and yunet each used to return a bare 'skipped' key, which
    silently overwrote one another when flattened into the audit dict."""
    a = _audit(blur_job, integrity={"integrity_skipped": "mode=blur"},
               yunet={"yunet_skipped": "no model"})
    assert a["integrity_ran"] is False and a["yunet_ran"] is False
    assert a["integrity_skipped"] == "mode=blur"
    assert a["yunet_skipped"] == "no model"


# ---------------------------------------------------------------------------
# Redaction pixel math.
# ---------------------------------------------------------------------------


def test_fill_grays_all_three_planes(blur_job):
    y, u, v = _planes()
    assert blur_job.redact_frame_inplace(y, u, v, [(10.0, 10.0, 40.0, 40.0)], "fill")
    assert np.all(y[10:40, 10:40] == blur_job.FILL_VALUE)
    assert np.all(u[5:20, 5:20] == blur_job.FILL_VALUE)
    assert np.all(v[5:20, 5:20] == blur_job.FILL_VALUE)
    assert y[9, 9] == 0 and y[40, 40] == 0  # nothing outside the box


def test_fill_chroma_covers_odd_box_edges(blur_job):
    """Chroma is half-resolution; the covering range is floor-start,
    CEIL-end. Flooring both ends left a ~2px strip of original hue under
    gray luma whenever the box edge was odd."""
    y, u, v = _planes(w=16, h=16)
    blur_job.redact_frame_inplace(y, u, v, [(3.0, 3.0, 9.0, 9.0)], "fill")
    assert np.all(u[1:5, 1:5] == blur_job.FILL_VALUE)
    assert np.all(v[1:5, 1:5] == blur_job.FILL_VALUE)


def test_blur_mode_redacts_chroma_not_just_luma(blur_job):
    """Blurring Y alone left the face's original hue and saturation intact at
    full chroma resolution — and blur mode has no integrity check to catch
    it. Chroma must be non-uniform here: blurring a flat patch is a no-op, so
    a constant test region would pass even against the broken version."""
    y = np.full((64, 64), 200, np.uint8)
    grad = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    u, v = grad.copy(), (255 - grad).astype(np.uint8)
    u0, v0 = u.copy(), v.copy()
    assert blur_job.redact_frame_inplace(y, u, v, [(10.0, 10.0, 50.0, 50.0)], "blur")
    assert not np.array_equal(u[5:25, 5:25], u0[5:25, 5:25])
    assert not np.array_equal(v[5:25, 5:25], v0[5:25, 5:25])
    assert np.array_equal(u[0:4, 0:4], u0[0:4, 0:4])  # untouched outside


@pytest.mark.parametrize("box", [
    (-50.0, -50.0, -10.0, -10.0),  # entirely off the top-left
    (100.0, 100.0, 200.0, 200.0),  # entirely off the bottom-right
    (30.0, 30.0, 30.0, 30.0),      # zero area
    (40.0, 40.0, 10.0, 10.0),      # inverted
])
def test_degenerate_boxes_redact_nothing(blur_job, box):
    y, u, v = _planes()
    assert blur_job.redact_frame_inplace(y, u, v, [box], "fill") is False
    assert not y.any() and not u.any() and not v.any()


# ---------------------------------------------------------------------------
# Tracking. Interpolation and holds are where most coverage against a missed
# detection frame actually comes from.
# ---------------------------------------------------------------------------


def test_interpolation_fills_the_gap_between_two_detections(blur_job):
    dets = [
        blur_job.Detection(frame_idx=0, cls="face", box=(0.0, 0.0, 20.0, 20.0), score=0.9),
        blur_job.Detection(frame_idx=3, cls="face", box=(6.0, 0.0, 26.0, 20.0), score=0.9),
    ]
    tracks = blur_job.build_tracks(dets, min_box_px=8, iou_thresh=0.2, hold_frames=30)
    assert len(tracks) == 1
    frames = tracks[0].frames
    assert frames[1][1] == "interp" and frames[2][1] == "interp"
    assert frames[1][0][0] == pytest.approx(2.0)  # linear in x
    assert frames[2][0][0] == pytest.approx(4.0)


def test_track_holds_backward_before_its_first_detection(blur_job):
    """Detection samples every `stride` frames, so a face walking into shot
    between samples shipped unredacted until its first detection."""
    dets = [blur_job.Detection(frame_idx=30, cls="face", box=(10.0, 10.0, 50.0, 50.0),
                                score=0.9)]
    tracks = blur_job.build_tracks(dets, 8, 0.2, hold_frames=30, back_hold_frames=3)
    frames = tracks[0].frames
    assert 60 in frames, "forward hold"
    assert 29 in frames and 27 in frames, "backward hold"


def test_backward_hold_clamps_at_frame_zero(blur_job):
    dets = [blur_job.Detection(frame_idx=1, cls="face", box=(0.0, 0.0, 20.0, 20.0),
                                score=0.9)]
    tracks = blur_job.build_tracks(dets, 8, 0.2, hold_frames=30, back_hold_frames=30)
    assert min(tracks[0].frames) >= 0


def test_stride_larger_than_hold_still_associates(blur_job):
    """hold_frames was overloaded as the association max-gap, so any config
    with stride > hold_frames evicted every track before its second
    detection: no interpolation, and the whole gap unredacted."""
    dets = [
        blur_job.Detection(frame_idx=0, cls="face", box=(0.0, 0.0, 20.0, 20.0), score=0.9),
        blur_job.Detection(frame_idx=60, cls="face", box=(0.0, 0.0, 20.0, 20.0), score=0.9),
    ]
    tracks = blur_job.build_tracks(dets, 8, 0.2, hold_frames=30, back_hold_frames=60)
    assert len(tracks) == 1
    covered = set(tracks[0].frames)
    assert not [f for f in range(61) if f not in covered]


def test_a_long_dead_track_does_not_hijack_an_unrelated_face(blur_job):
    """Without eviction a track stays active forever and reattaches to any
    later box that happens to overlap — interpolating a straight line across
    minutes, between two different people."""
    dets = [
        blur_job.Detection(frame_idx=0, cls="face", box=(0.0, 0.0, 20.0, 20.0), score=0.9),
        blur_job.Detection(frame_idx=500, cls="face", box=(0.0, 0.0, 20.0, 20.0), score=0.9),
    ]
    tracks = blur_job.build_tracks(dets, 8, 0.2, hold_frames=30, back_hold_frames=3)
    assert len(tracks) == 2


def test_face_and_plate_tracks_never_cross_associate(blur_job):
    dets = [
        blur_job.Detection(frame_idx=0, cls="face", box=(0.0, 0.0, 20.0, 20.0), score=0.9),
        blur_job.Detection(frame_idx=3, cls="lp", box=(0.0, 0.0, 20.0, 20.0), score=0.9),
    ]
    tracks = blur_job.build_tracks(dets, 8, 0.2, hold_frames=30)
    assert len(tracks) == 2
    assert {t.cls for t in tracks} == {"face", "lp"}


def test_min_box_px_drops_tiny_detections(blur_job):
    """A dropped detection is never redacted, so this threshold is a privacy
    decision — pin it so a change is deliberate."""
    d = blur_job.Detection(frame_idx=0, cls="face", box=(0.0, 0.0, 7.0, 7.0), score=0.99)
    assert blur_job.build_tracks([d], min_box_px=8, iou_thresh=0.2, hold_frames=30) == []


def test_dilate_box_pads_and_clamps(blur_job):
    assert blur_job.dilate_box((100, 100, 200, 200), 1.3, 8, 1000, 1000) == \
        (77.0, 77.0, 223.0, 223.0)
    # At the frame edge the box is clamped, never pushed out of bounds.
    x1, y1, x2, y2 = blur_job.dilate_box((980, 0, 1000, 20), 1.3, 8, 1000, 1000)
    assert x1 >= 0 and y1 >= 0 and x2 <= 1000 and y2 <= 1000


def test_tracks_to_fill_map_dilates_every_held_frame(blur_job):
    d = blur_job.Detection(frame_idx=0, cls="face", box=(100.0, 100.0, 200.0, 200.0),
                            score=0.9)
    tracks = blur_job.build_tracks([d], 8, 0.2, hold_frames=2)
    fm = blur_job.tracks_to_fill_map(tracks, 1000, 1000, 1.3, 8)
    assert sorted(fm) == [0, 1, 2]
    assert fm[0] == [(77.0, 77.0, 223.0, 223.0)]


# ---------------------------------------------------------------------------
# check_fill_integrity — the one check that runs over 100% of frames.
# ---------------------------------------------------------------------------


def _patch_decoder(monkeypatch, blur_job, frames, returncode=0):
    monkeypatch.setattr(blur_job, "open_decoder", lambda *a, **k: _Proc(returncode))
    monkeypatch.setattr(blur_job, "read_frames", lambda p, w, h: iter(frames))


def test_integrity_passes_on_correctly_filled_frames(blur_job, monkeypatch, tmp_path):
    y, u, v = _planes(fill=blur_job.FILL_VALUE)
    _patch_decoder(monkeypatch, blur_job, [(y, u, v)])
    out = blur_job.check_fill_integrity("ffmpeg", tmp_path / "o.mp4", 64, 64,
                                         {0: [(10.0, 10.0, 40.0, 40.0)]}, "fill",
                                         tmp_path / "e.log", 1)
    assert out["fill_integrity_violations"] == 0
    assert out["fill_integrity_checked"] == 1


def test_integrity_catches_an_unredacted_region(blur_job, monkeypatch, tmp_path):
    y, u, v = _planes(fill=blur_job.FILL_VALUE)
    y[10:40, 10:40] = 30  # face pixels the encoder never grayed
    _patch_decoder(monkeypatch, blur_job, [(y, u, v)])
    out = blur_job.check_fill_integrity("ffmpeg", tmp_path / "o.mp4", 64, 64,
                                         {0: [(10.0, 10.0, 40.0, 40.0)]}, "fill",
                                         tmp_path / "e.log", 1)
    assert out["fill_integrity_violations"] == 1


def test_integrity_catches_chroma_left_unredacted(blur_job, monkeypatch, tmp_path):
    """Luma gray, chroma original — the exact leak the check exists for."""
    y, u, v = _planes(fill=blur_job.FILL_VALUE)
    u[5:20, 5:20] = 20
    _patch_decoder(monkeypatch, blur_job, [(y, u, v)])
    out = blur_job.check_fill_integrity("ffmpeg", tmp_path / "o.mp4", 64, 64,
                                         {0: [(10.0, 10.0, 40.0, 40.0)]}, "fill",
                                         tmp_path / "e.log", 1)
    assert out["fill_integrity_violations"] == 1


def test_integrity_catches_frame_index_desync(blur_job, monkeypatch, tmp_path):
    """Boxes applied to the wrong frames: the output is clean where the
    fill_map claims gray."""
    clean = _planes(fill=0)
    _patch_decoder(monkeypatch, blur_job, [clean])
    out = blur_job.check_fill_integrity("ffmpeg", tmp_path / "o.mp4", 64, 64,
                                         {0: [(10.0, 10.0, 40.0, 40.0)]}, "fill",
                                         tmp_path / "e.log", 1)
    assert out["fill_integrity_violations"] == 1


def test_integrity_survives_a_forced_box_off_the_frame_edge(blur_job, monkeypatch, tmp_path):
    """--forced-boxes entries reach this slice raw. A negative coordinate made
    numpy wrap, the slice came back empty, and np.max raised — recording the
    clip as FAILED on the human remediation path."""
    y, u, v = _planes(fill=blur_job.FILL_VALUE)
    _patch_decoder(monkeypatch, blur_job, [(y, u, v)])
    out = blur_job.check_fill_integrity("ffmpeg", tmp_path / "o.mp4", 64, 64,
                                         {0: [(-10.0, -10.0, 40.0, 40.0)]}, "fill",
                                         tmp_path / "e.log", 1)
    assert out["fill_integrity_violations"] == 0


def test_integrity_fails_closed_on_a_short_decode(blur_job, monkeypatch, tmp_path):
    """A dead decode yields zero frames, the loop never runs, and zero
    violations used to read as 'verified clean'."""
    y, u, v = _planes(fill=blur_job.FILL_VALUE)
    _patch_decoder(monkeypatch, blur_job, [(y, u, v)])
    with pytest.raises(RuntimeError, match="expected 99"):
        blur_job.check_fill_integrity("ffmpeg", tmp_path / "o.mp4", 64, 64,
                                       {0: [(10.0, 10.0, 40.0, 40.0)]}, "fill",
                                       tmp_path / "e.log", 99)


def test_integrity_fails_closed_on_a_nonzero_decoder_exit(blur_job, monkeypatch, tmp_path):
    y, u, v = _planes(fill=blur_job.FILL_VALUE)
    _patch_decoder(monkeypatch, blur_job, [(y, u, v)], returncode=1)
    with pytest.raises(RuntimeError, match="exited 1"):
        blur_job.check_fill_integrity("ffmpeg", tmp_path / "o.mp4", 64, 64,
                                       {0: [(10.0, 10.0, 40.0, 40.0)]}, "fill",
                                       tmp_path / "e.log", 1)


def test_integrity_skipped_in_blur_mode_is_labelled_not_zeroed(blur_job, tmp_path):
    out = blur_job.check_fill_integrity("ffmpeg", tmp_path / "o.mp4", 64, 64,
                                         {}, "blur", tmp_path / "e.log", 1)
    assert "integrity_skipped" in out
    assert "fill_integrity_violations" not in out


# ---------------------------------------------------------------------------
# Forced boxes — the human remediation path. Every failure here must be loud.
# ---------------------------------------------------------------------------


def _forced(blur_job, tmp_path, payload, clip=None, fill_map=None, **kw):
    import json
    p = tmp_path / "f.json"
    p.write_text(json.dumps(payload))
    opts = dict(hold_frames=30, back_hold_frames=3, dilate_scale=1.3, motion_margin_px=8)
    opts.update(kw)
    return blur_job._apply_forced_boxes(
        p, clip or _clip(blur_job, n_frames=200, width=640, height=480),
        fill_map if fill_map is not None else {}, **opts)


def test_missing_forced_boxes_file_raises(blur_job, tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        blur_job._apply_forced_boxes(
            tmp_path / "nope.json", _clip(blur_job), {},
            hold_frames=30, back_hold_frames=3, dilate_scale=1.3, motion_margin_px=8)


def test_forced_boxes_with_no_matching_clip_id_warns_and_applies_nothing(
        blur_job, tmp_path, caplog):
    fill_map = {}
    n = _forced(blur_job, tmp_path,
                {"OTHER": [{"frame_idx": 5, "box": [0, 0, 10, 10]}]},
                fill_map=fill_map)
    assert n == 0 and fill_map == {}
    assert any("no entry for this clip" in r.message for r in caplog.records)


def test_forced_boxes_that_all_clamp_away_raise(blur_job, tmp_path):
    with pytest.raises(RuntimeError, match="no usable box"):
        _forced(blur_job, tmp_path,
                {"x": [{"frame_idx": 0, "box": [-50, -50, -10, -10]}]})


def test_forced_box_frame_index_out_of_range_raises(blur_job, tmp_path):
    """A typo'd frame number used to be a silent no-op that still counted
    toward 'applied N forced boxes' — reporting success while republishing
    the face."""
    with pytest.raises(RuntimeError, match="outside"):
        _forced(blur_job, tmp_path,
                {"x": [{"frame_idx": 9999, "box": [10, 10, 40, 40]}]})


def test_forced_box_covers_frames_around_the_one_listed(blur_job, tmp_path):
    """THE critical fix. A forced box written only to its listed frame left
    the face visible on every frame between the reviewer's entries — and
    both audit checks sample that same stride grid, so the clip then flipped
    NEEDS_REVIEW -> PASS_AUTOMATED with the face still shipping."""
    fill_map = {}
    _forced(blur_job, tmp_path,
            {"x": [{"frame_idx": 90, "box": [100, 100, 140, 140]}]},
            fill_map=fill_map, hold_frames=30, back_hold_frames=3)
    assert 90 in fill_map
    assert 91 in fill_map and 100 in fill_map, "forward hold"
    assert 88 in fill_map, "backward hold"


def test_forced_boxes_interpolate_between_consecutive_entries(blur_job, tmp_path):
    """The documented remediation loop emits one entry per flagged frame, and
    flagged frames sit on the detection stride. Every frame in between must
    be covered."""
    fill_map = {}
    entries = [{"frame_idx": f, "box": [100, 100, 140, 140]} for f in range(90, 121, 3)]
    _forced(blur_job, tmp_path, {"x": entries}, fill_map=fill_map,
            hold_frames=30, back_hold_frames=3)
    missing = [f for f in range(90, 121) if f not in fill_map]
    assert not missing, f"frames still unredacted between forced boxes: {missing}"


def test_forced_boxes_are_clamped_to_the_frame(blur_job, tmp_path):
    fill_map = {}
    _forced(blur_job, tmp_path,
            {"x": [{"frame_idx": 0, "box": [-10, -10, 40, 40]}]},
            clip=_clip(blur_job, n_frames=200), fill_map=fill_map)
    x1, y1, x2, y2 = fill_map[0][0]
    assert x1 >= 0 and y1 >= 0, "negative coords must not reach the redactor"
    assert x2 <= 64 and y2 <= 64, "must stay inside the frame"


# ---------------------------------------------------------------------------
# CLI guards. Each of these silently corrupted a run before it was rejected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags,why", [
    (["--dilate-scale", "0.5"], "shrinks boxes below the detected face"),
    (["--motion-margin-px", "-1"], "negative padding"),
    (["--hold-frames", "-5"], "disables all track association"),
    (["--detect-hz", "0"], "ZeroDivisionError deep in the run"),
    (["--detect-hz", "-10"], "negative stride"),
    (["--face-threshold", "1.5"], "detects nothing, redacts nothing"),
    (["--face-threshold", "0"], "out of range"),
    (["--min-box-px", "-1"], "negative"),
])
def test_parse_args_rejects_unsafe_values(blur_job, flags, why):
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, *flags])


def test_probe_frames_default_outside_the_output_dir(blur_job):
    """Probe frames are UNREDACTED originals. They must never default into
    the tree that gets synced off the pod and published."""
    cfg = blur_job.parse_args(BASE_ARGS)
    assert cfg.probe_frames_dir != cfg.output_dir
    assert cfg.output_dir not in cfg.probe_frames_dir.parents
    assert "DO-NOT-SHIP" in cfg.probe_frames_dir.name


# ---------------------------------------------------------------------------
# Ingest guards.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem,expected", [
    ("GX010042", ("0042", 1)),
    ("GX020042", ("0042", 2)),
    ("GOPR0042", ("GOPR0042", 0)),
    ("clip", ("clip", 0)),
])
def test_parse_gopro_chapter(blur_job, stem, expected):
    assert blur_job.parse_gopro_chapter(Path(f"{stem}.MP4")) == expected


def test_frame_byte_size_rejects_odd_dimensions(blur_job):
    assert blur_job.frame_byte_size(1920, 1080) == 1920 * 1080 * 3 // 2
    with pytest.raises(RuntimeError):
        blur_job.frame_byte_size(1921, 1080)


@pytest.mark.parametrize("stream", [
    {"r_frame_rate": "30/1", "avg_frame_rate": "24/1"},   # genuinely VFR
    {"r_frame_rate": "0/1", "avg_frame_rate": "0/1"},     # unusable
])
def test_assert_cfr_rejects_non_cfr(blur_job, stream):
    """Every timestamp downstream derives from one fps; a VFR clip would make
    frame indexing silently wrong rather than loudly broken."""
    with pytest.raises(RuntimeError):
        blur_job.assert_cfr({"streams": [stream]}, Path("x.mp4"))


def test_assert_cfr_accepts_ntsc_rates(blur_job):
    blur_job.assert_cfr(
        {"streams": [{"r_frame_rate": "30000/1001", "avg_frame_rate": "30000/1001"}]},
        Path("x.mp4"))


# ---------------------------------------------------------------------------
# check_yunet — the independent second opinion. Previously had NO tests, and
# a mutation reverting its score threshold survived the whole suite.
# ---------------------------------------------------------------------------


def _yunet_row(x, y, w, h, score, right_eye_x=523.0):
    """cv2.FaceDetectorYN.detect -> [n, 15]: 0-1 bbox xy, 2-3 wh, 4-5 RIGHT
    EYE xy, 6-7 left eye, 8-9 nose, 10-13 mouth, 14 face score."""
    return np.array([x, y, w, h, right_eye_x, 118.0, 540.0, 118.0, 530.0,
                     130.0, 520.0, 140.0, 545.0, 140.0, score], np.float32)


def _patch_yunet(monkeypatch, blur_job, rows, frames, returncode=0, capture=None):
    import cv2

    class _Det:
        def __init__(self, thresh):
            self._t = thresh

        def getScoreThreshold(self):
            return self._t

        def detect(self, bgr):
            return len(rows), (np.array(rows) if rows else None)

    def _create(model, cfg_, size, score_threshold=0.9, nms=0.3, *a, **k):
        if capture is not None:
            capture["score_threshold"] = score_threshold
        return _Det(score_threshold)

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", staticmethod(_create))
    monkeypatch.setattr(blur_job, "open_decoder", lambda *a, **k: _Proc(returncode))
    monkeypatch.setattr(blur_job, "read_frames", lambda p, w, h: iter(frames))


def _yunet_cfg(blur_job, tmp_path):
    return blur_job.parse_args(["--input-dir", str(tmp_path), "--output-dir",
                                 str(tmp_path), "--run-id", "t",
                                 "--yunet-model", str(tmp_path / "y.onnx")])


def test_yunet_is_created_with_our_score_threshold(blur_job, monkeypatch, tmp_path):
    """OpenCV defaults score_threshold to 0.9. Leaving it unset made
    YUNET_SCORE_MIN dead code and ran the check far stricter than the audit
    claimed — silently dropping the low-confidence second opinions it exists
    to surface."""
    cap = {}
    _patch_yunet(monkeypatch, blur_job, [], [_planes()], capture=cap)
    blur_job.check_yunet(_yunet_cfg(blur_job, tmp_path), "ffmpeg",
                          tmp_path / "o.mp4", 64, 64, {}, 10.0, 30.0, 1)
    assert cap["score_threshold"] == blur_job.YUNET_SCORE_MIN


def test_yunet_reads_the_score_from_column_14_not_column_4(blur_job, monkeypatch, tmp_path):
    """f[4] is the right-eye x in PIXELS, so reading it as the score compared
    ~523 against 0.5 — the gate never fired and a coordinate was recorded and
    displayed as a confidence all the way into the review page."""
    row = _yunet_row(10, 10, 20, 20, score=0.92, right_eye_x=523.0)
    _patch_yunet(monkeypatch, blur_job, [row], [_planes()])
    out = blur_job.check_yunet(_yunet_cfg(blur_job, tmp_path), "ffmpeg",
                                tmp_path / "o.mp4", 64, 64, {}, 10.0, 30.0, 1)
    assert out["n_yunet_uncovered"] == 1
    assert out["yunet_uncovered"][0]["score"] == pytest.approx(0.92)


def test_yunet_covered_face_is_not_flagged(blur_job, monkeypatch, tmp_path):
    row = _yunet_row(10, 10, 20, 20, score=0.92)
    _patch_yunet(monkeypatch, blur_job, [row], [_planes()])
    out = blur_job.check_yunet(_yunet_cfg(blur_job, tmp_path), "ffmpeg",
                                tmp_path / "o.mp4", 64, 64,
                                {0: [(0.0, 0.0, 64.0, 64.0)]}, 10.0, 30.0, 1)
    assert out["n_yunet_uncovered"] == 0


def test_yunet_can_return_private_detection_geometry_for_a_local_preview(
    blur_job, monkeypatch, tmp_path
):
    row = _yunet_row(10, 10, 20, 20, score=0.92)
    _patch_yunet(monkeypatch, blur_job, [row], [_planes()])
    out = blur_job.check_yunet(
        _yunet_cfg(blur_job, tmp_path), "ffmpeg", tmp_path / "o.mp4", 64, 64,
        {0: [(0.0, 0.0, 64.0, 64.0)]}, 10.0, 30.0, 1, record_detections=True,
    )
    detection = out["yunet_detections"][0]
    assert detection["frame_idx"] == 0
    assert detection["box"] == (10.0, 10.0, 30.0, 30.0)
    assert detection["score"] == pytest.approx(0.92)
    assert detection["covered"] is True


def test_yunet_fails_closed_on_a_short_decode(blur_job, monkeypatch, tmp_path):
    _patch_yunet(monkeypatch, blur_job, [], [_planes()])
    with pytest.raises(RuntimeError, match="expected 99"):
        blur_job.check_yunet(_yunet_cfg(blur_job, tmp_path), "ffmpeg",
                              tmp_path / "o.mp4", 64, 64, {}, 10.0, 30.0, 99)


def test_yunet_fails_closed_on_a_nonzero_decoder_exit(blur_job, monkeypatch, tmp_path):
    _patch_yunet(monkeypatch, blur_job, [], [_planes()], returncode=1)
    with pytest.raises(RuntimeError, match="exited 1"):
        blur_job.check_yunet(_yunet_cfg(blur_job, tmp_path), "ffmpeg",
                              tmp_path / "o.mp4", 64, 64, {}, 10.0, 30.0, 1)


def test_yunet_without_a_model_is_labelled_skipped_not_clean(blur_job, tmp_path):
    cfg = blur_job.parse_args(["--input-dir", str(tmp_path), "--output-dir",
                                str(tmp_path), "--run-id", "t"])
    out = blur_job.check_yunet(cfg, "ffmpeg", tmp_path / "o.mp4", 64, 64,
                                {}, 10.0, 30.0, 1)
    assert "yunet_skipped" in out
    assert "n_yunet_uncovered" not in out


def test_face_only_run_is_allowed_but_records_that_plates_were_not_checked(blur_job):
    """Indoor egocentric footage has no plates, and running the LP detector
    there doubles GPU cost to find nothing — so face-only is a legitimate
    configuration. It must never be silently indistinguishable from a run
    that actually cleared plates."""
    a = blur_job.build_audit(
        _clip(blur_job), {"n_frames_with_fill": 10},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"n_yunet_uncovered": 0}, "2",
        lp_checked=False)
    assert a["lp_checked"] is False
    assert a["status"] == "PASS_AUTOMATED"  # face-only is valid, not a failure


def test_gen2_requires_only_face_weights(blur_job, tmp_path):
    """LP weights optional; face weights are not."""
    face = tmp_path / "f.jit"
    face.write_bytes(b"x")
    cfg = blur_job.parse_args([*BASE_ARGS, "--gen", "2",
                                "--face-weights-gen2", str(face)])
    assert cfg.face_weights_gen2 == face and cfg.lp_weights_gen2 is None


def test_zero_coverage_reason_distinguishes_corroborated_from_unexplained(blur_job):
    """Most of this project's footage is faceless kitchen video, so the
    zero-coverage flag fires constantly. If every case reads identically the
    reviewer stops reading it — so an independent detector agreeing there is
    nothing here must be stated, while an uncorroborated zero stays alarming."""
    corroborated = blur_job.build_audit(
        _clip(blur_job), {"n_frames_with_fill": 0},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 0},
        {"n_candidate_misses": 0}, {"n_yunet_uncovered": 0}, "2")
    unexplained = blur_job.build_audit(
        _clip(blur_job), {"n_frames_with_fill": 0},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 0},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2")

    # Both still require a human — "we redacted nothing" never auto-passes.
    assert corroborated["status"] == "NEEDS_REVIEW"
    assert unexplained["status"] == "NEEDS_REVIEW"
    assert any("YuNet independently found no faces" in r
               for r in corroborated["status_reasons"])
    assert any("nothing corroborates that" in r
               for r in unexplained["status_reasons"])


# ---------------------------------------------------------------------------
# Probe frame selection. This had NO coverage at all, and four real bugs
# lived in it — found only by running it against real footage on a pod:
#   1. _write_probe_frames took [:max_images] from a list built in order, so
#      the 20 images a human reviews were frames 0,3,6..57 — the first TWO
#      SECONDS of the first clip.
#   2. probe_and_budget broke out of the decode loop the moment it had
#      enough samples, so it only ever looked at the first 60 seconds of a
#      4.5-minute clip. "The probe found no faces" was a claim about the
#      opening minute.
#   3. Detections were keyed by clip-local frame index, which collides
#      across clips (every clip has a frame 0) — one clip's boxes were drawn
#      onto another clip's image, and same-named files overwrote each other.
#   4. Zero sampled frames was reported as "found zero detections" rather
#      than raised, so a dead decoder read as a fact about the footage.
# ---------------------------------------------------------------------------


class _ProbeProc:
    """probe_and_budget touches .stdout and .terminate(); _Proc does not."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = None

    def terminate(self):
        pass

    def wait(self):
        return self.returncode


class _FakeDetector:
    """Emits a detection at chosen POSITIONS in the sampled list."""

    def __init__(self, blur_job, cls, scores_by_pos=None):
        self._blur_job = blur_job
        self.cls = cls
        self.scores_by_pos = scores_by_pos or {}
        self.seen = []

    def detect_batch(self, frames_bgr, frame_idxs):
        self.seen.extend(frame_idxs)
        return [
            self._blur_job.Detection(frame_idx=p, cls=self.cls,
                                     box=(1.0, 1.0, 20.0, 20.0),
                                     score=self.scores_by_pos[p])
            for p in frame_idxs if p in self.scores_by_pos
        ]


def _probe_setup(blur_job, monkeypatch, tmp_path, n_frames=100, probe_frames=20):
    cfg = blur_job.parse_args(
        ["--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
         "--run-id", "r", "--probe-only", "--probe-frames", str(probe_frames)])
    clip = _clip(blur_job, n_frames=n_frames)
    frames = [_planes() for _ in range(n_frames)]
    monkeypatch.setattr(blur_job, "open_decoder", lambda *a, **k: _ProbeProc())
    monkeypatch.setattr(blur_job, "read_frames", lambda p, w, h: iter(frames))
    return cfg, clip


def test_probe_samples_span_the_whole_clip_not_just_the_opening(
        blur_job, monkeypatch, tmp_path):
    """Bug 2. With 100 frames and 20 wanted, the old code sampled at the
    detection stride (3) and broke at 20 — reaching frame 57 and never
    looking at the back two thirds of the clip."""
    cfg, clip = _probe_setup(blur_job, monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(blur_job, "_write_probe_frames",
                        lambda *a, **k: seen.update(idxs=list(a[2])))
    blur_job.probe_and_budget(cfg, [clip], _FakeDetector(blur_job, "face"), None, "ffmpeg")
    assert max(seen["idxs"]) > 0.8 * clip.n_frames, (
        f"probe only reached frame {max(seen['idxs'])} of {clip.n_frames} — "
        f"it is drawing conclusions from the opening of the clip only")


def test_probe_fails_closed_when_the_decoder_yields_nothing(
        blur_job, monkeypatch, tmp_path):
    """Bug 4. ffmpeg 4.4 rejected -fps_mode and emitted zero bytes; this
    function reported that as 'probe found ZERO detections'."""
    cfg, clip = _probe_setup(blur_job, monkeypatch, tmp_path)
    monkeypatch.setattr(blur_job, "read_frames", lambda p, w, h: iter([]))
    with pytest.raises(RuntimeError, match="ZERO frames"):
        blur_job.probe_and_budget(cfg, [clip], _FakeDetector(blur_job, "face"),
                                   None, "ffmpeg")


def test_probe_writes_the_highest_scoring_frames_not_the_first_ones(
        blur_job, tmp_path):
    """Bug 1. The one real detection sits at the END of the sample; taking
    the first N images would miss it entirely."""
    frames = [np.full((64, 64, 3), i % 256, np.uint8) for i in range(40)]
    idxs = list(range(0, 400, 10))
    clip_ids = ["c"] * 40
    hot = blur_job.Detection(frame_idx=39, cls="face", box=(1.0, 1.0, 20.0, 20.0),
                              score=0.95)
    noise = [blur_job.Detection(frame_idx=i, cls="lp", box=(1.0, 1.0, 9.0, 9.0),
                                 score=0.12) for i in range(5)]
    out = tmp_path / "probe"
    blur_job._write_probe_frames(out, frames, idxs, clip_ids, [hot, *noise],
                                  {"face": 0.30, "lp": 0.40})
    written = sorted(p.name for p in out.glob("*.jpg"))
    assert any("00000390" in n for n in written), (
        f"the only above-threshold detection (frame 390) was not written; "
        f"got {written}")


def test_probe_frames_from_two_clips_do_not_overwrite_each_other(
        blur_job, tmp_path):
    """Bug 3. Both clips have a frame 0; the filename was frame_{idx}.jpg."""
    frames = [np.zeros((64, 64, 3), np.uint8) for _ in range(2)]
    out = tmp_path / "probe"
    blur_job._write_probe_frames(out, frames, [0, 0], ["clipA", "clipB"], [],
                                  {"face": 0.30, "lp": 0.40})
    assert len(list(out.glob("*.jpg"))) == 2, (
        "two clips' frame 0 collapsed onto one filename")


def test_probe_reports_above_threshold_counts_separately(
        blur_job, monkeypatch, tmp_path):
    """Raw counts are at the 0.10 sweep floor and are mostly deliberate
    noise; the number that answers 'would this redact anything' is the
    above-threshold one, and it must be reported in its own right."""
    cfg, clip = _probe_setup(blur_job, monkeypatch, tmp_path)
    # One real face, three pieces of sub-threshold sweep noise.
    det = _FakeDetector(blur_job, "face", {0: 0.92, 1: 0.12, 2: 0.15, 3: 0.11})
    monkeypatch.setattr(blur_job, "_write_probe_frames", lambda *a, **k: None)
    rep = blur_job.probe_and_budget(cfg, [clip], det, None, "ffmpeg")
    assert rep["n_face_detections"] == 4
    assert rep["n_face_above_threshold"] == 1
    assert rep["max_face_score"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# Full-range (yuvj420p / color_range=pc) handling. GoPro writes pc, and
# cv2.COLOR_YUV2BGR_I420 hardcodes the LIMITED-range inverse — so every
# detector was handed an image with both ends of the luma scale clipped
# flat. Measured on a real pc clip: [0,8,16,64,128,180,235,247,255] arrived
# at the detector as [0,0,0,56,130,191,255,255,255].
# ---------------------------------------------------------------------------

_RAMP = [0, 8, 16, 64, 128, 180, 235, 247, 255]


def _ramp_planes(vals=_RAMP, w=72, h=8):
    y = np.zeros((h, w), np.uint8)
    for i, v in enumerate(vals):
        y[:, i * 8:(i + 1) * 8] = v
    c = np.full((h // 2, w // 2), 128, np.uint8)
    return y, c, c


def test_full_range_luma_survives_the_bgr_conversion(blur_job):
    """The blocking regression. Without the pre-compression, Y=8 and Y=16
    both land on 0 and Y=235/247/255 all land on 255 — the detector cannot
    distinguish a face in shadow from black."""
    y, u, v = _ramp_planes()
    bgr = blur_job.yuv_to_bgr(y, u, v, True)
    got = [int(bgr[4, i * 8 + 4, 1]) for i in range(len(_RAMP))]
    assert got == _RAMP, f"full-range luma was altered: {_RAMP} -> {got}"


def test_full_range_conversion_does_not_clip_either_end(blur_job):
    y, u, v = _ramp_planes()
    bgr = blur_job.yuv_to_bgr(y, u, v, True)
    got = [int(bgr[4, i * 8 + 4, 1]) for i in range(len(_RAMP))]
    assert len(set(got)) == len(_RAMP), (
        f"distinct luma levels collapsed onto each other: {got}")


def test_limited_range_input_is_left_alone(blur_job):
    """tv-range footage must NOT be double-compressed."""
    y, u, v = _ramp_planes()
    plain = blur_job.yuv_to_bgr(y, u, v, False)
    import cv2
    expect = cv2.cvtColor(blur_job._pack_yuv420p(y, u, v), cv2.COLOR_YUV2BGR_I420)
    assert np.array_equal(plain, expect)


# ---------------------------------------------------------------------------
# Zero-redaction canary was class-blind: n_frames_with_fill counts ANY box,
# and tracks_to_fill_map discards Track.cls, so one above-threshold licence
# plate silenced the only structural defence against "the FACE model
# returned nothing at all" for a whole clip.
# ---------------------------------------------------------------------------


def test_a_licence_plate_fill_does_not_silence_the_zero_face_canary(blur_job):
    clip = _clip(blur_job, n_frames=8134)
    # Thousands of frames were filled — every one of them a licence plate.
    fill_stats = {"n_frames_with_fill": 5000, "frames_with_fill_frac": 0.6,
                  "max_fill_area_frac": 0.01}
    audit = blur_job.build_audit(
        clip, fill_stats, {"fill_integrity_violations": 0, "fill_integrity_checked": 10},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_face_fill_frames=0)
    assert audit["status"] == "NEEDS_REVIEW"
    assert any("ZERO frames had a FACE redacted" in r for r in audit["status_reasons"]), \
        f"canary stayed silent; reasons were {audit['status_reasons']}"


def test_face_fills_do_not_trip_the_zero_face_canary(blur_job):
    clip = _clip(blur_job, n_frames=100)
    fill_stats = {"n_frames_with_fill": 40, "frames_with_fill_frac": 0.4,
                  "max_fill_area_frac": 0.01}
    audit = blur_job.build_audit(
        clip, fill_stats, {"fill_integrity_violations": 0, "fill_integrity_checked": 40},
        {"n_candidate_misses": 0}, {"n_yunet_uncovered": 0}, "2",
        n_face_fill_frames=40)
    assert not any("ZERO frames" in r for r in audit["status_reasons"])


# ---------------------------------------------------------------------------
# Sweep candidate ranking. Face band is [sweep, 0.30), plate band is
# [sweep, 0.40) — a single absolute-score sort meant every plate >= 0.30
# outranked every face that could exist, so 200 slots filled with cardboard.
# ---------------------------------------------------------------------------


def test_plate_noise_cannot_evict_every_face_from_the_audit_sample(blur_job):
    d = blur_job.Detection
    # 400 plates all scoring above the entire face band, 5 real face candidates.
    plates = [d(frame_idx=i, cls="lp", box=(0.0, 0.0, 9.0, 9.0), score=0.39)
              for i in range(400)]
    faces = [d(frame_idx=9000 + i, cls="face", box=(0.0, 0.0, 9.0, 9.0), score=0.29)
             for i in range(5)]
    out = blur_job.check_low_threshold_sweep(
        plates + faces, {}, 0.10, {"face": 0.30, "lp": 0.40})
    shown = {c["cls"] for c in out["candidates"]}
    assert "face" in shown, (
        "every face candidate was evicted from the sample by plate noise")
    assert out["n_candidate_misses_face"] == 5
    assert out["n_candidate_misses_lp"] == 400


# ---------------------------------------------------------------------------
# Checkpoint config fingerprint. A resume that reuses rows from a different
# detector config performs ZERO inference and records the new config over
# the old boxes.
# ---------------------------------------------------------------------------


def _ckpt_cfg(blur_job, tmp_path, **over):
    args = ["--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
            "--run-id", "r"]
    for k, v in over.items():
        args += [k, str(v)]
    return blur_job.parse_args(args)


class _CountingDetector:
    def __init__(self, blur_job, cls="face"):
        self._b, self.cls, self.calls = blur_job, cls, 0

    def detect_batch(self, frames, idxs):
        self.calls += len(idxs)
        return []


def _run_detection(blur_job, monkeypatch, tmp_path, cfg, det, lp, gen, n=12):
    clip = _clip(blur_job, n_frames=n, fps=30.0)
    monkeypatch.setattr(blur_job, "open_decoder", lambda *a, **k: _ProbeProc())
    monkeypatch.setattr(blur_job, "read_frames",
                        lambda p, w, h: iter([_planes() for _ in range(n)]))
    ck = tmp_path / "ck"
    ck.mkdir(exist_ok=True)
    return blur_job.detection_pass(cfg, clip, det, lp, "ffmpeg", ck, gen), ck


def test_checkpoint_from_a_different_config_is_discarded_not_reused(
        blur_job, monkeypatch, tmp_path):
    """The confirmed fail-open: run 1 face-only, run 2 adds the plate
    detector. Reusing run 1's rows means the plate model never executes
    while the manifest reports lp_checked=true."""
    cfg = _ckpt_cfg(blur_job, tmp_path)
    d1 = _CountingDetector(blur_job)
    _run_detection(blur_job, monkeypatch, tmp_path, cfg, d1, None, "2")
    assert d1.calls > 0

    # Same clip, same everything — except an LP detector now exists.
    d2, lp2 = _CountingDetector(blur_job), _CountingDetector(blur_job, "lp")
    _run_detection(blur_job, monkeypatch, tmp_path, cfg, d2, lp2, "2")
    assert d2.calls > 0, (
        "detector was never called — stale face-only rows were reused for a "
        "run that claims plates were checked")
    assert lp2.calls > 0


def test_checkpoint_with_a_matching_config_still_resumes(
        blur_job, monkeypatch, tmp_path):
    """The fingerprint must not defeat resumption, which is the whole point
    of the checkpoint."""
    cfg = _ckpt_cfg(blur_job, tmp_path)
    d1 = _CountingDetector(blur_job)
    _run_detection(blur_job, monkeypatch, tmp_path, cfg, d1, None, "2")
    d2 = _CountingDetector(blur_job)
    _run_detection(blur_job, monkeypatch, tmp_path, cfg, d2, None, "2")
    assert d2.calls == 0, "identical config should have resumed, not re-detected"


def test_switching_gen_discards_the_checkpoint(blur_job, monkeypatch, tmp_path):
    cfg = _ckpt_cfg(blur_job, tmp_path)
    d1 = _CountingDetector(blur_job)
    _run_detection(blur_job, monkeypatch, tmp_path, cfg, d1, None, "2")
    d2 = _CountingDetector(blur_job)
    _run_detection(blur_job, monkeypatch, tmp_path, cfg, d2, None, "1")
    assert d2.calls > 0, "gen 1 run inherited gen 2 boxes under gen 1 provenance"


# ---------------------------------------------------------------------------
# audit_summary.md must state WHY a clip needs review. Two of build_audit's
# gate reasons are backed by no printed metric.
# ---------------------------------------------------------------------------


def test_audit_summary_states_the_reason_for_needs_review(blur_job, tmp_path):
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5, "max_fill_area_frac": 0.1},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_dropped_small=7, n_face_fill_frames=5)
    p = tmp_path / "s.md"
    blur_job.write_audit_summary(audit, clip, p)
    text = p.read_text()
    assert "NEEDS_REVIEW" in text
    assert "min-box-px" in text, (
        "the only stated cause of NEEDS_REVIEW is missing from the summary a "
        f"human actually reads:\n{text}")


def test_build_audit_reports_det_low_frames(blur_job):
    """det_low_frames used to not exist at all: n_low_absorbed reached the
    manifest but nothing said WHICH frames it counted."""
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5, "max_fill_area_frac": 0.1},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_face_fill_frames=5, n_low_absorbed=2, det_low_frames=[9, 3, 3])
    # Caller passes raw appends (process_clip's own list can repeat a
    # frame_idx across tracks) -- build_audit stores what it's given as-is;
    # sorting/deduping is process_clip's job at the call site, not this
    # function's, so this pins that build_audit itself does not silently
    # do it a second time and mask a caller-side regression.
    assert audit["det_low_frames"] == [9, 3, 3]


def test_build_audit_det_low_frames_defaults_empty(blur_job):
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5, "max_fill_area_frac": 0.1},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_face_fill_frames=5)
    assert audit["det_low_frames"] == []


def test_build_audit_caps_det_low_frames_like_its_siblings(blur_job):
    """Same convention as candidates_truncated/yunet_truncated: cap the
    stored list at AUDIT_MAX_ITEMS and count what got cut, rather than
    storing an unbounded list -- a long clip under heavy hysteresis use
    could otherwise put thousands of entries in every manifest."""
    clip = _clip(blur_job)
    frames = list(range(blur_job.AUDIT_MAX_ITEMS + 5))
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5, "max_fill_area_frac": 0.1},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_face_fill_frames=5, n_low_absorbed=len(frames), det_low_frames=frames)
    assert len(audit["det_low_frames"]) == blur_job.AUDIT_MAX_ITEMS
    assert audit["det_low_frames_truncated"] == 5


def test_audit_summary_prints_low_absorbed_count_and_frames(blur_job, tmp_path):
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5, "max_fill_area_frac": 0.1},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_face_fill_frames=5, n_low_absorbed=2, det_low_frames=[3, 9])
    p = tmp_path / "s.md"
    blur_job.write_audit_summary(audit, clip, p)
    text = p.read_text()
    assert "n_low_absorbed: 2" in text, (
        f"n_low_absorbed reaches the JSON manifest but not the markdown a "
        f"human actually reads:\n{text}")
    assert "3, 9" in text, (
        f"the count is printed but not WHICH frames it refers to:\n{text}")


def test_audit_summary_caps_the_printed_frame_list(blur_job, tmp_path):
    """A clip with dozens of absorptions must not turn the summary into an
    unreadable dump -- the full list still reaches the JSON manifest."""
    clip = _clip(blur_job)
    frames = list(range(20))
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5, "max_fill_area_frac": 0.1},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_face_fill_frames=5, n_low_absorbed=20, det_low_frames=frames)
    assert audit["det_low_frames"] == frames, "the JSON field must stay uncapped"
    p = tmp_path / "s.md"
    blur_job.write_audit_summary(audit, clip, p)
    text = p.read_text()
    assert "+12 more" in text, f"expected the markdown excerpt capped at 8:\n{text}"


def test_audit_summary_does_not_guess_why_nothing_was_absorbed(blur_job, tmp_path):
    """write_audit_summary only ever sees the audit dict -- never
    cfg.continue_threshold -- so it cannot tell 'hysteresis was off' apart
    from 'hysteresis was on and genuinely absorbed nothing on this clip'.
    Regression: it used to print '0 when --continue-threshold is off' as a
    hardcoded, unverified guess, which is simply false on a clip run WITH
    hysteresis that happened to absorb nothing."""
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5, "max_fill_area_frac": 0.1},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_face_fill_frames=5)  # n_low_absorbed/det_low_frames both default empty
    p = tmp_path / "s.md"
    blur_job.write_audit_summary(audit, clip, p)
    text = p.read_text()
    assert "n_low_absorbed: 0" in text
    assert "continue-threshold" not in text, (
        f"asserted a specific cause for zero absorptions that this function "
        f"has no way to actually verify:\n{text}")


def test_audit_summary_labels_absorptions_separately_from_distinct_frames(
        blur_job, tmp_path):
    """n_low_absorbed counts EVENTS; det_low_frames is DEDUPLICATED frames.
    Two different tracks absorbing on the same frame_idx makes these
    legitimately differ (2 events, 1 distinct frame) -- the summary must
    label them as two different measures, not print a bare frame count next
    to an absorption count as though they're the same number."""
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5, "max_fill_area_frac": 0.1},
        {"fill_integrity_violations": 0, "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_face_fill_frames=5, n_low_absorbed=2, det_low_frames=[5])
    p = tmp_path / "s.md"
    blur_job.write_audit_summary(audit, clip, p)
    text = p.read_text()
    assert "n_low_absorbed: 2" in text
    assert "1 distinct frame" in text, (
        f"a reviewer seeing 'n_low_absorbed: 2' next to a single frame number "
        f"needs the mismatch explained, not left to look like a bug:\n{text}")


def test_changing_gen2_resize_px_discards_the_checkpoint(blur_job, monkeypatch, tmp_path):
    """Inference scale changes what a score means, so cached rows from the
    other scale are not reusable — same reasoning as gen and weights."""
    cfg_a = _ckpt_cfg(blur_job, tmp_path)
    d1 = _CountingDetector(blur_job)
    _run_detection(blur_job, monkeypatch, tmp_path, cfg_a, d1, None, "2")
    cfg_b = _ckpt_cfg(blur_job, tmp_path, **{"--gen2-resize-px": 1200})
    d2 = _CountingDetector(blur_job)
    _run_detection(blur_job, monkeypatch, tmp_path, cfg_b, d2, None, "2")
    assert d2.calls > 0, "boxes from a different inference scale were reused"


def test_gen2_resize_px_defaults_to_native(blur_job):
    assert blur_job.parse_args(BASE_ARGS).gen2_resize_px is None


def test_absurd_gen2_resize_px_is_rejected(blur_job):
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--gen2-resize-px", "8"])


# ---------------------------------------------------------------------------
# --detect-batch. Added when the 5090 this job was tuned against became
# unavailable (RunPod Community Cloud released it back to the host pool on
# pod stop) and a smaller/shared GPU was the alternative — the hardcoded
# batch of 8 at native 1080p was sized for 32GB and had no way to turn down.
# ---------------------------------------------------------------------------


def test_detect_batch_defaults_to_the_module_constant(blur_job):
    assert blur_job.parse_args(BASE_ARGS).detect_batch == blur_job.DETECT_BATCH


def test_detect_batch_is_overridable(blur_job):
    cfg = blur_job.parse_args([*BASE_ARGS, "--detect-batch", "2"])
    assert cfg.detect_batch == 2


def test_detect_batch_below_one_is_rejected(blur_job):
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--detect-batch", "0"])


def test_probe_actually_honours_detect_batch_not_the_module_constant(
        blur_job, monkeypatch, tmp_path):
    """The real regression risk with a new config knob: probe_and_budget or
    detection_pass reading the module constant instead of cfg.detect_batch,
    which would make the flag a no-op that LOOKS wired up."""
    cfg = blur_job.parse_args(
        ["--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
         "--run-id", "r", "--probe-only", "--probe-frames", "9",
         "--detect-batch", "3"])
    clip = _clip(blur_job, n_frames=90, fps=30.0)
    monkeypatch.setattr(blur_job, "open_decoder", lambda *a, **k: _ProbeProc())
    monkeypatch.setattr(blur_job, "read_frames",
                        lambda p, w, h: iter([_planes() for _ in range(90)]))
    sizes = []
    det = _FakeDetector(blur_job, "face")
    real_detect = det.detect_batch

    def spy(frames, idxs):
        sizes.append(len(idxs))
        return real_detect(frames, idxs)
    det.detect_batch = spy
    monkeypatch.setattr(blur_job, "_write_probe_frames", lambda *a, **k: None)
    blur_job.probe_and_budget(cfg, [clip], det, None, "ffmpeg")
    assert sizes and max(sizes) == 3, (
        f"probe batched at {sizes}, not --detect-batch=3 — the flag is a no-op")


# ---------------------------------------------------------------------------
# Leading-edge coverage. back_hold_frames was hardcoded to one detection
# stride — a tenth of the forward hold — so a face was unredacted from the
# moment it entered frame until the detector first scored it. Measured on
# real footage: 8 of 13 inspected misses were exactly this, with gaps of
# 5-26 frames against a backward hold of 3.
# ---------------------------------------------------------------------------


def test_backward_hold_covers_a_face_entering_frame(blur_job):
    """The real pattern: nothing, then a confident detection at frame 4614.
    Frames 4585-4613 shipped the face."""
    d = blur_job.Detection(frame_idx=4614, cls="face",
                           box=(100.0, 100.0, 180.0, 190.0), score=0.40)
    stride, hold = 3, 30

    old = blur_job.build_tracks([d], 8, blur_job.TRACK_IOU_DEFAULT, hold,
                                 back_hold_frames=stride)
    new = blur_job.build_tracks([d], 8, blur_job.TRACK_IOU_DEFAULT, hold,
                                 back_hold_frames=hold)
    first_old, first_new = min(old[0].frames), min(new[0].frames)
    assert first_old == 4614 - stride
    assert first_new == 4614 - hold, "backward hold did not widen"
    assert first_new < first_old, (
        "a symmetric hold must reach further back than one stride")


def test_back_hold_defaults_to_symmetric_with_forward_hold(blur_job):
    """The DEFAULTING, not just the flag. An earlier version of this test
    asserted `(cfg.back_hold_frames or hold) == hold`, which is a tautology
    and passed happily when process_clip still used stride."""
    cfg = blur_job.parse_args([*BASE_ARGS])
    fwd, back = blur_job.resolve_holds(cfg, 29.97)
    assert fwd == 30
    assert back == fwd, (
        f"backward hold {back} != forward {fwd}; an asymmetric default leaves "
        f"a face entering frame unredacted until first detection")


def test_back_hold_is_not_one_detection_stride(blur_job):
    """The specific regression: stride at 29.97fps / 10Hz is 3 frames, a
    tenth of the forward hold. Measured misses needed 5-26."""
    cfg = blur_job.parse_args([*BASE_ARGS])
    _fwd, back = blur_job.resolve_holds(cfg, 29.97)
    assert back > 3, "backward hold fell back to one stride"


def test_explicit_back_hold_overrides_the_symmetric_default(blur_job):
    cfg = blur_job.parse_args([*BASE_ARGS, "--back-hold-frames", "45"])
    fwd, back = blur_job.resolve_holds(cfg, 29.97)
    assert (fwd, back) == (30, 45)


def test_back_hold_frames_is_explicitly_settable(blur_job):
    cfg = blur_job.parse_args([*BASE_ARGS, "--back-hold-frames", "45"])
    assert cfg.back_hold_frames == 45


def test_negative_back_hold_is_rejected(blur_job):
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--back-hold-frames", "-1"])


# ---------------------------------------------------------------------------
# write_manifest must record the RESOLVED hold values, not the raw cfg
# fields. cfg.back_hold_frames defaults to 0 ("auto") on every run that
# doesn't pass --back-hold-frames explicitly -- the common case, since the
# flag exists to OVERRIDE the auto-symmetric default. Recording the raw
# field printed back_hold_frames: 0 into the manifest of every run whose
# track building actually used a real, non-zero backward hold.
# ---------------------------------------------------------------------------


def _stub_torch(monkeypatch):
    """write_manifest imports torch to record its version/cuda flag — a real
    GPU dependency this test suite deliberately doesn't install (see
    conftest.py). These tests are about hold-frame recording, not torch, so
    stub the two attributes actually read rather than skip the coverage."""
    import sys
    import types
    fake = types.ModuleType("torch")
    fake.__version__ = "0.0.0-test"
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_manifest_records_resolved_hold_not_raw_cfg_default(blur_job, tmp_path, monkeypatch):
    _stub_torch(monkeypatch)
    cfg = blur_job.parse_args([*BASE_ARGS, "--hold-frames", "45"])
    assert cfg.back_hold_frames == 0, "still the auto sentinel, as this run leaves it"
    fwd, back = blur_job.resolve_holds(cfg, 29.97)
    assert (fwd, back) == (45, 45), "sanity: resolution itself must be symmetric here"

    path = tmp_path / "m.json"
    out_video = tmp_path / "out.mp4"
    out_video.write_bytes(b"not a real video, just needs to exist for sha256/stat")
    blur_job.write_manifest(_clip(blur_job), "2", cfg, out_video,
                            {"status": "PASS_AUTOMATED"}, {}, path,
                            hold_frames=fwd, back_hold_frames=back)
    import json
    eb = json.loads(path.read_text())["egoblur"]
    assert eb["back_hold_frames"] == 45, (
        f"manifest says back_hold_frames={eb['back_hold_frames']!r}, but 45 frames "
        f"were actually used to build every track's leading edge")
    assert eb["hold_frames"] == 45


def test_manifest_hold_fields_match_an_explicit_back_hold_override(blur_job, tmp_path, monkeypatch):
    _stub_torch(monkeypatch)
    cfg = blur_job.parse_args([*BASE_ARGS, "--hold-frames", "30",
                               "--back-hold-frames", "90"])
    fwd, back = blur_job.resolve_holds(cfg, 29.97)
    assert (fwd, back) == (30, 90)
    path = tmp_path / "m.json"
    out_video = tmp_path / "out.mp4"
    out_video.write_bytes(b"not a real video, just needs to exist for sha256/stat")
    blur_job.write_manifest(_clip(blur_job), "2", cfg, out_video,
                            {"status": "PASS_AUTOMATED"}, {}, path,
                            hold_frames=fwd, back_hold_frames=back)
    import json
    eb = json.loads(path.read_text())["egoblur"]
    assert (eb["hold_frames"], eb["back_hold_frames"]) == (30, 90)


# ---------------------------------------------------------------------------
# HYSTERESIS (--continue-threshold). A detection needs the operating
# threshold to SEED a track but only the lower continue threshold to EXTEND
# one -- ByteTrack's BYTE association, Canny's double-threshold edges.
#
# The whole safety argument rests on one property: a low-confidence blob can
# never START a track, so face-shaped noise on the wearer's hands (which has
# no confirmed track to continue) is still discarded. Measured on real
# footage: flat 0.20 seeded 321 EXTRA tracks and ruined the clip; hysteresis
# at 0.15 seeded ZERO.
# ---------------------------------------------------------------------------

_HYST_START = {"face": 0.30, "lp": 0.40}
_HYST_CONT = {"face": 0.15, "lp": 0.15}


def _d(blur_job, frame, score, box=(100.0, 100.0, 200.0, 200.0), cls="face"):
    return blur_job.Detection(frame_idx=frame, cls=cls, box=box, score=score)


def _tracks(blur_job, dets, **kw):
    """hold_frames must exceed the detection stride or max_gap evicts every
    track before its second detection -- that is what caught the missing
    stride floor in max_gap."""
    kw.setdefault("start_thresh", _HYST_START)
    kw.setdefault("cont_thresh", _HYST_CONT)
    return blur_job.build_tracks(dets, 8, blur_job.TRACK_IOU_DEFAULT,
                                 hold_frames=30, back_hold_frames=30,
                                 stride=3, **kw)


def test_low_confidence_detection_cannot_seed_a_track(blur_job):
    """THE load-bearing property. A 0.18 blob on a hand, alone in the world,
    must produce nothing at all."""
    tracks = _tracks(blur_job, [_d(blur_job, 0, 0.18)])
    assert tracks == [], "low-confidence detection seeded a track on its own"


def test_low_confidence_extends_a_confirmed_track(blur_job):
    """A real face that turns away and drops to 0.18 must stay covered."""
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.18)]
    tracks = _tracks(blur_job, dets)
    assert len(tracks) == 1
    assert tracks[0].frames[3][1] == "det_low", (
        "the low box did not extend the confirmed track as a real detection "
        f"(got source {tracks[0].frames[3][1]!r} -- a plain 'hold' means it "
        f"was NOT absorbed)")
    assert tracks[0].last_frame == 3


def test_hysteresis_absorbs_are_counted(blur_job):
    absorbed = []
    _tracks(blur_job, [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.18)],
            low_absorbed=absorbed)
    assert len(absorbed) == 1 and absorbed[0].score == pytest.approx(0.18)


def test_low_box_must_overlap_to_be_absorbed(blur_job):
    """The mechanism that makes hand-noise safe: a low box somewhere ELSE in
    the frame has nothing to continue, so it is dropped rather than becoming
    its own grey rectangle."""
    # Overlap must be SMALL BUT NONZERO. A box with zero overlap is not a
    # test of the threshold at all: best_iou starts at 0.0 with a strict >,
    # so best_j is never set and the box is rejected no matter what
    # iou_thresh is. This one overlaps at IoU ~0.005, under the 0.2 gate.
    dets = [_d(blur_job, 0, 0.90, box=(100.0, 100.0, 200.0, 200.0)),
            _d(blur_job, 3, 0.18, box=(190.0, 190.0, 290.0, 290.0))]
    assert 0.0 < blur_job._iou(dets[0].box, dets[1].box) < blur_job.TRACK_IOU_DEFAULT
    absorbed = []
    tracks = _tracks(blur_job, dets, low_absorbed=absorbed)
    assert len(tracks) == 1, "the non-overlapping low box created a second track"
    # Presence in .frames is NOT the test -- the forward hold fills those
    # frames regardless. What matters is that nothing was ABSORBED.
    assert absorbed == [], "a non-overlapping low box was absorbed"
    assert not any(src == "det_low" for _b, src in tracks[0].frames.values())


def test_below_continue_threshold_is_ignored_entirely(blur_job):
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.11)]  # 0.11 < cont 0.15
    absorbed = []
    tracks = _tracks(blur_job, dets, low_absorbed=absorbed)
    assert absorbed == []
    assert not any(src == "det_low" for _b, src in tracks[0].frames.values())


def test_low_run_is_capped_so_a_track_cannot_drift_indefinitely(blur_job):
    """An unbounded chain of weak detections could walk a track off its
    subject, and the hold would then extend the error past the chain."""
    dets = [_d(blur_job, 0, 0.90)] + [_d(blur_job, f, 0.18)
                                       for f in (3, 6, 9, 12, 15, 18, 21)]
    tracks = _tracks(blur_job, dets, max_low_run=4)
    det_frames = [f for f, (_b, src) in tracks[0].frames.items() if src == "det_low"]
    assert len(det_frames) == 4, (
        f"expected the low chain capped at 4, got {len(det_frames)}: {det_frames}")


def test_a_confident_detection_resets_the_low_run(blur_job):
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.18), _d(blur_job, 6, 0.18),
            _d(blur_job, 9, 0.95),  # resets
            _d(blur_job, 12, 0.18), _d(blur_job, 15, 0.18), _d(blur_job, 18, 0.18)]
    absorbed = []
    tracks = _tracks(blur_job, dets, max_low_run=4, low_absorbed=absorbed)
    # NOT `18 in frames` -- the forward hold fills it either way. The reset
    # is observable only in how many low boxes were actually absorbed: 5
    # with the reset (3,6 then 12,15,18), 4 without (the run caps at 18).
    assert len(absorbed) == 5, (
        f"expected 5 absorbed with the low-run reset, got {len(absorbed)}")
    assert tracks[0].frames[18][1] == "det_low"


def test_stride_is_a_floor_on_the_track_eviction_gap(blur_job):
    """Regression: max_gap was max(hold, back_hold) and back_hold used to BE
    the stride, so the floor existed by accident. Making the backward hold
    symmetric removed it silently -- with --detect-hz 0.5 on 30fps (stride
    60) against a 45-frame hold, every track is evicted before matching its
    SECOND detection, interpolation never runs, and the frames between two
    sightings of one stationary face ship unredacted under a PASS status.

    Only observable when hold < stride, which is why the earlier version of
    this test (hold 30, stride 3) could not see it."""
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 10, 0.90)]
    joined = blur_job.build_tracks(dets, 8, blur_job.TRACK_IOU_DEFAULT,
                                   hold_frames=2, back_hold_frames=2, stride=10)
    assert len(joined) == 1, (
        "the two detections did not associate: max_gap lost its stride floor")
    assert joined[0].frames[5][1] == "interp", "gap was not interpolated"

    # Same inputs with no stride floor: evicted, two separate tracks, and the
    # frames between them go uncovered.
    split = blur_job.build_tracks(dets, 8, blur_job.TRACK_IOU_DEFAULT,
                                  hold_frames=2, back_hold_frames=2, stride=1)
    assert len(split) == 2, "sanity: without the floor these must NOT associate"


def test_hysteresis_off_by_default_reproduces_old_behaviour(blur_job):
    """Default must be a no-op: --continue-threshold 0 means single-threshold."""
    cfg = blur_job.parse_args([*BASE_ARGS])
    assert cfg.continue_threshold == 0.0
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.18)]
    plain = blur_job.build_tracks(dets, 8, blur_job.TRACK_IOU_DEFAULT, 30, 30,
                                   stride=3)
    assert len(plain) == 1
    # NOT `3 in frames` -- the forward hold fills frames 1..30 regardless, and
    # "det_low" would satisfy it too. Only the SOURCE TAG separates the three
    # cases (associated normally / absorbed by hysteresis / merely held).
    # Mutation-verified: the membership assertion passed even when hysteresis
    # was forced ON by default, i.e. it tested nothing.
    assert plain[0].frames[3][1] == "det", (
        f"with no thresholds supplied the 0.18 detection must associate as a "
        f"PLAIN detection; got {plain[0].frames[3][1]!r} -- hysteresis is not off")
    assert plain[0].last_frame == 3


def test_min_box_px_drop_only_counts_seedable_detections(blur_job):
    """Under hysteresis the caller passes the whole low band. Counting sub-
    threshold noise as 'confident detection discarded by --min-box-px' would
    push every clip to NEEDS_REVIEW with a reason that is untrue of it."""
    tiny = (100.0, 100.0, 103.0, 103.0)   # 3px, below min_box_px=8
    dropped = []
    _tracks(blur_job, [_d(blur_job, 0, 0.90, box=tiny),
                       _d(blur_job, 3, 0.18, box=tiny)], dropped_small=dropped)
    assert len(dropped) == 1, f"expected only the confident one, got {len(dropped)}"
    assert dropped[0].score == pytest.approx(0.90)


# --- CLI guards -------------------------------------------------------------

def test_continue_threshold_above_operating_is_rejected(blur_job):
    """Hysteresis LOWERS the bar to continue; above it would raise it."""
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--continue-threshold", "0.35",
                             "--face-threshold", "0.30"])


def test_continue_threshold_below_sweep_floor_is_rejected(blur_job):
    """Nothing was ever scored that low, so nothing could be absorbed."""
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--continue-threshold", "0.05",
                             "--sweep-threshold", "0.10"])


def test_valid_continue_threshold_is_accepted(blur_job):
    cfg = blur_job.parse_args([*BASE_ARGS, "--continue-threshold", "0.15"])
    assert cfg.continue_threshold == pytest.approx(0.15)


def test_continue_threshold_is_not_in_the_checkpoint_fingerprint(blur_job):
    """Like the operating thresholds, it is applied post-hoc to stored
    detections — changing it must REUSE the checkpoint, not re-run 25
    minutes of GPU."""
    base = blur_job.parse_args([*BASE_ARGS])
    hyst = blur_job.parse_args([*BASE_ARGS, "--continue-threshold", "0.15"])
    assert (blur_job.checkpoint_fingerprint(base, "2", False)
            == blur_job.checkpoint_fingerprint(hyst, "2", False))


# ---------------------------------------------------------------------------
# Hysteresis must never REDUCE coverage. The first version let an absorbed
# low box REPLACE the track's position: a 0.18 blob offset 40px from a 100px
# face still associated (IoU 0.220 > TRACK_IOU_DEFAULT 0.20), dragged the
# track off the real face, and the forward hold perpetuated the wrong
# position. Measured: the real face went from 11 uncovered frames to 40 while
# n_candidate_misses fell 1 -> 0, because the blob was now "covered" by
# itself. NEEDS_REVIEW -> PASS with the face MORE exposed.
# ---------------------------------------------------------------------------

_REAL_FACE = (100.0, 100.0, 200.0, 200.0)
_WEAK_BLOB = (140.0, 140.0, 240.0, 240.0)   # IoU 0.220 against _REAL_FACE


def _coverage_and_gate(blur_job, hysteresis: bool):
    dets = [blur_job.Detection(0, "face", _REAL_FACE, 0.90),
            blur_job.Detection(3, "face", _REAL_FACE, 0.90),
            blur_job.Detection(6, "face", _WEAK_BLOB, 0.18)]
    kw = (dict(start_thresh={"face": 0.30, "lp": 0.40},
               cont_thresh={"face": 0.15, "lp": 0.15}) if hysteresis else {})
    inp = dets if hysteresis else [d for d in dets if d.score > 0.30]
    tracks = blur_job.build_tracks(inp, 8, blur_job.TRACK_IOU_DEFAULT,
                                    30, 30, stride=3, **kw)
    fm = blur_job.tracks_to_fill_map(tracks, 640, 640, 1.3, 8)
    uncovered = sum(
        1 for f in range(45)
        if blur_job._covered_fraction(_REAL_FACE, fm.get(f, []))
        < blur_job.COVERAGE_MIN_FRAC)
    sweep = blur_job.check_low_threshold_sweep(
        dets, fm, 0.10, {"face": 0.30, "lp": 0.40})
    return uncovered, sweep["n_candidate_misses"]


def test_hysteresis_never_reduces_coverage_of_a_confirmed_face(blur_job):
    assert blur_job._iou(_REAL_FACE, _WEAK_BLOB) > blur_job.TRACK_IOU_DEFAULT, (
        "fixture invalid: the blob must ASSOCIATE under the confident gate, "
        "otherwise this cannot exercise the regression at all")
    off, _ = _coverage_and_gate(blur_job, hysteresis=False)
    on, _ = _coverage_and_gate(blur_job, hysteresis=True)
    assert on <= off, (
        f"hysteresis left the real face uncovered on {on} frames vs {off} "
        f"without it -- an absorbed low box displaced the track")


def test_hysteresis_does_not_silence_the_sweep_gate(blur_job):
    _, off_gate = _coverage_and_gate(blur_job, hysteresis=False)
    _, on_gate = _coverage_and_gate(blur_job, hysteresis=True)
    assert on_gate >= off_gate, (
        f"candidate_misses fell {off_gate} -> {on_gate}: hysteresis silenced "
        f"the one gate that would have flagged the displaced coverage")


def test_a_low_detection_needs_stronger_overlap_than_a_confident_one(blur_job):
    """Weak evidence must clear a higher spatial bar. IoU 0.220 is enough for
    a confident detection and must NOT be enough for a 0.18 one."""
    assert blur_job.LOW_ASSOC_IOU_DEFAULT > blur_job.TRACK_IOU_DEFAULT
    dets = [blur_job.Detection(0, "face", _REAL_FACE, 0.90),
            blur_job.Detection(3, "face", _WEAK_BLOB, 0.18)]
    absorbed = []
    _tracks(blur_job, dets, low_absorbed=absorbed)
    assert absorbed == [], (
        "a 0.18 detection at IoU 0.220 was absorbed; low associations must "
        "clear LOW_ASSOC_IOU_DEFAULT, not TRACK_IOU_DEFAULT")


def test_an_absorbed_low_box_is_unioned_not_substituted(blur_job):
    """Coverage must be monotone: absorbing can add area, never move it."""
    near = (110.0, 110.0, 210.0, 210.0)   # IoU ~0.68, clears the low gate
    dets = [blur_job.Detection(0, "face", _REAL_FACE, 0.90),
            blur_job.Detection(3, "face", near, 0.18)]
    tracks = _tracks(blur_job, dets)
    box = tracks[0].frames[3][0]
    assert tracks[0].frames[3][1] == "det_low", "fixture did not absorb"
    assert blur_job._covered_fraction(_REAL_FACE, [box]) >= 0.999, (
        f"the absorbed box {box} does not still cover the confirmed face "
        f"{_REAL_FACE} -- it replaced rather than unioned")


# --- the wiring itself (mutation-verified as previously untested) -----------

def test_hysteresis_wiring_hands_the_tracker_the_low_band(blur_job):
    """Making this return redact_dets makes the whole feature inert while the
    manifest still records continue_threshold. Verified by mutation that no
    other test caught it."""
    cfg = blur_job.parse_args([*BASE_ARGS, "--continue-threshold", "0.15"])
    operating = {"face": cfg.face_threshold, "lp": cfg.lp_threshold}
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.18)]
    redact = [d for d in dets if d.score > operating[d.cls]]
    ti, st, ct = blur_job.resolve_hysteresis(cfg, dets, redact, operating)
    assert len(ti) == 2, "low band filtered out before tracking -- feature inert"
    assert st == operating
    assert ct == {"face": 0.15, "lp": 0.15}


def test_default_wiring_passes_only_above_threshold_detections(blur_job):
    cfg = blur_job.parse_args([*BASE_ARGS])
    operating = {"face": cfg.face_threshold, "lp": cfg.lp_threshold}
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.18)]
    redact = [d for d in dets if d.score > operating[d.cls]]
    ti, st, ct = blur_job.resolve_hysteresis(cfg, dets, redact, operating)
    assert ti == redact and st is None and ct is None


def test_low_run_budget_is_frames_since_confident_not_absorption_count(blur_job):
    """The exact regression the review caught: an earlier version counted
    ABSORPTION EVENTS, so 4 weak detections 30 frames apart (real drift far
    outside any reasonable bound) were treated identically to 4 weak
    detections 3 frames apart -- both "4 events", so both allowed. With
    stride=3 and max_low_run=4 the real budget is 12 frames since the last
    CONFIDENT sighting: a detection 30 frames later must be rejected
    outright, not merely the fifth one in a row."""
    dets = [_d(blur_job, 0, 0.90)] + [_d(blur_job, f, 0.18) for f in (30, 60, 90, 120)]
    absorbed = []
    tracks = _tracks(blur_job, dets, max_low_run=4, low_absorbed=absorbed)
    assert absorbed == [], (
        f"a detection 30 frames past the last confident sighting was "
        f"absorbed even though the budget (max_low_run*stride = 4*3 = 12) "
        f"was exceeded on the very first weak detection: {absorbed}")
    assert tracks[0].last_frame == 0, "the track should never have re-matched"


def test_low_run_budget_scales_with_stride(blur_job):
    """The complementary case: the SAME 4-events-in-a-row pattern from the
    original (pre-fix) test must still be allowed when the spacing genuinely
    fits the frame budget -- this fix must not become simply stricter, it
    must become CORRECT regardless of --detect-hz."""
    dets = [_d(blur_job, 0, 0.90)] + [_d(blur_job, f, 0.18) for f in (10, 20, 30, 40)]
    absorbed = []
    blur_job.build_tracks(
        dets, 8, blur_job.TRACK_IOU_DEFAULT, hold_frames=60, back_hold_frames=60,
        stride=10, max_low_run=4, low_absorbed=absorbed,
        start_thresh=_HYST_START, cont_thresh=_HYST_CONT)
    assert len(absorbed) == 4, (
        f"expected all 4 absorbed (budget = 4*stride=10 = 40, spacing fits "
        f"exactly), got {len(absorbed)}")


# ---------------------------------------------------------------------------
# max_fill_area_frac was computed and printed as a "runaway false-positive
# canary" but build_audit never read it -- a genuinely runaway frame (a
# detection bug producing a box spanning most of the frame) could still
# report PASS_AUTOMATED with the number sitting there, printed, ignored.
# ---------------------------------------------------------------------------


def test_runaway_fill_area_forces_review(blur_job):
    a = _audit(blur_job, fill={"n_frames_with_fill": 10, "max_fill_area_frac": 0.73})
    assert a["status"] == "NEEDS_REVIEW"
    assert any("max_fill_area_frac" in r for r in a["status_reasons"]), (
        f"a 73% redacted frame produced no reason: {a['status_reasons']}")


def test_normal_fill_area_does_not_trip_the_ceiling(blur_job):
    """A legitimate close-up face, generously dilated, must not false-alarm."""
    a = _audit(blur_job, fill={"n_frames_with_fill": 10, "max_fill_area_frac": 0.22})
    assert a["status"] == "PASS_AUTOMATED"


def test_fill_area_ceiling_is_a_boundary_not_a_typo(blur_job):
    at_ceiling = _audit(blur_job, fill={"n_frames_with_fill": 10,
                                        "max_fill_area_frac": blur_job.MAX_FILL_AREA_FRAC_CEILING})
    assert at_ceiling["status"] == "PASS_AUTOMATED", "exactly at the ceiling should not fire"
    just_over = _audit(blur_job, fill={"n_frames_with_fill": 10,
                                       "max_fill_area_frac": blur_job.MAX_FILL_AREA_FRAC_CEILING + 0.001})
    assert just_over["status"] == "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# --min-track-confirmations. Measured on real footage (test-run-3, GX010057):
# 335 of 392 face tracks had exactly one confident detection, and those
# tracks alone accounted for 77.7% of redacted area -- one false positive on
# a hand or shadow gets amplified by the hold window into ~3s of gray box
# identically to a real, persistently-seen face. A hand-checked sample of
# the worst single-detection tracks was 0-for-7 real persistent faces.
# Default 1 = off, byte-identical to every test above this section.
# ---------------------------------------------------------------------------


def test_min_confident_hits_default_reproduces_old_behaviour(blur_job):
    """A single detection must still get the full hold when the gate isn't
    explicitly raised -- every test above this section depends on that."""
    d = blur_job.Detection(frame_idx=0, cls="face", box=(100.0, 100.0, 200.0, 200.0),
                            score=0.9)
    tracks = blur_job.build_tracks([d], 8, 0.2, hold_frames=5, back_hold_frames=5)
    assert len(tracks) == 1
    assert sorted(tracks[0].frames) == list(range(0, 6)), (
        "a lone detection lost its hold with the gate at its default (off)")


def test_single_hit_track_is_dropped_entirely_when_confirmation_required(blur_job):
    """The whole point: with min_confident_hits=2, an isolated blip
    contributes NOTHING -- not a shortened hold, zero fill -- because the
    amplification this gate exists to stop happens entirely in the hold
    step. Dropping it before that step is what removes it."""
    d = blur_job.Detection(frame_idx=0, cls="face", box=(100.0, 100.0, 200.0, 200.0),
                            score=0.9)
    tracks = blur_job.build_tracks([d], 8, 0.2, hold_frames=45, back_hold_frames=45,
                                    min_confident_hits=2)
    assert tracks == [], f"an isolated single detection produced fill: {tracks}"


def test_two_hit_track_is_completely_unaffected_by_the_gate(blur_job):
    """Tracks that clear the bar get the exact same hold as with the gate
    off -- this can only ever REMOVE coverage from single-hit tracks, never
    touch a persistent one."""
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.90)]
    off = blur_job.build_tracks(dets, 8, blur_job.TRACK_IOU_DEFAULT,
                                 hold_frames=10, back_hold_frames=10, stride=3)
    on = blur_job.build_tracks(dets, 8, blur_job.TRACK_IOU_DEFAULT,
                                hold_frames=10, back_hold_frames=10, stride=3,
                                min_confident_hits=2)
    assert len(off) == 1 and len(on) == 1
    assert off[0].frames == on[0].frames, (
        "a 2-detection track's coverage changed even though it clears the "
        "confirmation bar either way")


def test_unconfirmed_out_parameter_collects_the_rejected_track(blur_job):
    """Mirrors dropped_small/low_absorbed: a confident detection that gets
    suppressed must never vanish without a trace."""
    d = blur_job.Detection(frame_idx=0, cls="face", box=(100.0, 100.0, 200.0, 200.0),
                            score=0.9)
    unconfirmed = []
    tracks = blur_job.build_tracks([d], 8, 0.2, hold_frames=45, back_hold_frames=45,
                                    min_confident_hits=2, unconfirmed=unconfirmed)
    assert tracks == []
    assert len(unconfirmed) == 1
    assert unconfirmed[0].frames[0][0] == d.box, (
        "the rejected track's own detection box should still be recoverable "
        "from the unconfirmed list for debugging")


def test_det_low_counts_toward_confirmation(blur_job):
    """A det_low absorption is real (if weak) persistence evidence -- it can
    only ever attach to an already-active track (never seed one), so it
    demonstrates the same kind of repeated sighting a second "det" would."""
    dets = [_d(blur_job, 0, 0.90), _d(blur_job, 3, 0.18)]  # seeds, then hysteresis-absorbs
    tracks = _tracks(blur_job, dets, min_confident_hits=2)
    assert len(tracks) == 1, (
        "one det + one det_low should clear a confirmation bar of 2, since "
        "det_low is real persistence evidence, not just interpolation/hold")
    assert tracks[0].frames[3][1] == "det_low", "fixture did not absorb"


def test_min_track_confirmations_defaults_to_one(blur_job):
    assert blur_job.parse_args(BASE_ARGS).min_track_confirmations == 1


def test_min_track_confirmations_is_overridable(blur_job):
    cfg = blur_job.parse_args([*BASE_ARGS, "--min-track-confirmations", "2"])
    assert cfg.min_track_confirmations == 2


def test_min_track_confirmations_below_one_is_rejected(blur_job):
    with pytest.raises(SystemExit):
        blur_job.parse_args([*BASE_ARGS, "--min-track-confirmations", "0"])


def test_min_track_confirmations_is_not_in_the_checkpoint_fingerprint(blur_job):
    """Like the operating thresholds, it's applied post-hoc to stored
    detections (it's a property of how tracks get BUILT from them, not of
    what the detector scored) -- changing it must REUSE the checkpoint, not
    re-run 25 minutes of GPU."""
    base = blur_job.parse_args([*BASE_ARGS])
    raised = blur_job.parse_args([*BASE_ARGS, "--min-track-confirmations", "2"])
    assert (blur_job.checkpoint_fingerprint(base, "2", False)
            == blur_job.checkpoint_fingerprint(raised, "2", False))


def test_build_audit_reports_n_unconfirmed_tracks(blur_job):
    a = _audit(blur_job)
    assert a["n_unconfirmed_tracks"] == 0, "must default to 0, not be absent"
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5}, {"fill_integrity_violations": 0},
        {"n_candidate_misses": 0}, {"n_yunet_uncovered": 0}, "2",
        n_unconfirmed_tracks=7)
    assert audit["n_unconfirmed_tracks"] == 7


def test_n_unconfirmed_tracks_does_not_force_needs_review(blur_job):
    """Deliberate design choice, not an oversight: suppressing isolated-blip
    noise is this gate's INTENDED behaviour whenever it's enabled, not a
    symptom something went wrong. Gating status on it would flood
    NEEDS_REVIEW on essentially every clip the gate is turned on for, and
    defeat the point of having it -- unlike n_dropped_small, which pins a
    genuinely rare edge case worth a human's attention every time."""
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5}, {"fill_integrity_violations": 0,
                                          "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"n_yunet_uncovered": 0}, "2",
        n_unconfirmed_tracks=50)
    assert audit["status"] == "PASS_AUTOMATED", (
        f"n_unconfirmed_tracks alone must never gate status: {audit['status_reasons']}")


def test_audit_summary_prints_n_unconfirmed_tracks(blur_job, tmp_path):
    clip = _clip(blur_job)
    audit = blur_job.build_audit(
        clip, {"n_frames_with_fill": 5}, {"fill_integrity_violations": 0,
                                          "fill_integrity_checked": 5},
        {"n_candidate_misses": 0}, {"yunet_skipped": "no model"}, "2",
        n_unconfirmed_tracks=42)
    p = tmp_path / "s.md"
    blur_job.write_audit_summary(audit, clip, p)
    text = p.read_text()
    assert "n_unconfirmed_tracks: 42" in text, (
        f"reaches the JSON manifest but not the markdown a human reads:\n{text}")
