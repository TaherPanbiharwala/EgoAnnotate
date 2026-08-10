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

from pathlib import Path

import numpy as np
import pytest

BASE_ARGS = ["--input-dir", "/tmp/in", "--output-dir", "/tmp/out", "--run-id", "r"]


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


# ---------------------------------------------------------------------------
# The audit gate. Every term is a COUNT, and all counts are zero when the
# pipeline produced nothing — so total failure read as a clean pass.
# ---------------------------------------------------------------------------


def _audit(blur_job, *, fill=None, integrity=None, sweep=None, yunet=None):
    return blur_job.build_audit(
        _clip(blur_job),
        fill if fill is not None else {"n_frames_with_fill": 10},
        integrity if integrity is not None else {"fill_integrity_violations": 0,
                                                  "fill_integrity_checked": 5},
        sweep if sweep is not None else {"n_candidate_misses": 0},
        yunet if yunet is not None else {"n_yunet_uncovered": 0},
        "2",
    )


def test_clean_run_passes(blur_job):
    assert _audit(blur_job)["status"] == "PASS_AUTOMATED"


def test_skipped_yunet_is_a_visibly_weaker_pass(blur_job):
    a = _audit(blur_job, yunet={"yunet_skipped": "no --yunet-model provided"})
    assert a["status"] == "PASS_AUTOMATED_NO_YUNET"
    assert a["yunet_ran"] is False


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
