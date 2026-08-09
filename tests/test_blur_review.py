"""Tests for scripts/22_blur_review.py.

This script produces the single artifact a human looks at before deciding
whether a clip is safe to publish. The failure that matters is not a crash —
it is the page quietly showing fewer flagged regions than exist, or claiming
there are none. Most of these assert on that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


def _manifest(**over):
    m = {
        "schema_version": 1,
        "run_id": "t",
        "clip_id": "GX010042",
        "source": {"filename": "GX010042.MP4", "sha256": "x", "width": 640,
                    "height": 360, "fps": 30.0, "n_frames": 180,
                    "duration_s": 6.0, "rotation": 0},
        "output": {"path": "GX010042.blurred.mp4", "sha256": "y", "bytes": 1},
        "status": "NEEDS_REVIEW",
        "audit": {"clip_id": "GX010042", "status": "NEEDS_REVIEW", "yunet_ran": True,
                   "gen": "1", "n_frames_with_fill": 12, "frames_with_fill_frac": 0.07,
                   "fill_integrity_checked": 12, "fill_integrity_violations": 0,
                   "n_candidate_misses": 0, "candidates": [],
                   "n_yunet_uncovered": 0, "yunet_uncovered": [], "note": "n"},
    }
    m.update(over)
    return m


# ---------------------------------------------------------------------------
# collect_flags — merges two sources with deliberately different meanings.
# ---------------------------------------------------------------------------


def test_yunet_flags_sort_before_sweep_flags(blur_review):
    """A second, independent model seeing a face the pipeline did not redact
    is stronger evidence than EgoBlur's own sub-threshold guess, even when
    the sweep's number is numerically larger."""
    audit = {
        "candidates": [{"frame_idx": 1, "box": [0, 0, 10, 10], "score": 0.99,
                        "cls": "face"}],
        "yunet_uncovered": [{"frame_idx": 2, "box": [0, 0, 10, 10], "score": 0.10}],
    }
    flags = blur_review.collect_flags(audit)
    assert [f["origin"] for f in flags] == ["yunet", "sweep"]


def test_flags_sort_by_descending_score_within_origin(blur_review):
    audit = {"yunet_uncovered": [
        {"frame_idx": 1, "box": [0, 0, 1, 1], "score": 0.2},
        {"frame_idx": 2, "box": [0, 0, 1, 1], "score": 0.8},
    ]}
    assert [f["score"] for f in blur_review.collect_flags(audit)] == [0.8, 0.2]


def test_collect_flags_on_an_empty_audit(blur_review):
    assert blur_review.collect_flags({}) == []
    assert blur_review.collect_flags({"candidates": [], "yunet_uncovered": []}) == []


def test_a_malformed_entry_does_not_kill_the_whole_review(blur_review):
    """One bad row used to raise KeyError and take down review of a clip that
    had real findings in it."""
    audit = {"yunet_uncovered": [{"frame_idx": 2, "box": [0, 0, 10, 10]}]}
    flags = blur_review.collect_flags(audit)
    assert flags[0]["score"] == 0.0


def test_sweep_entries_default_to_face_class(blur_review):
    audit = {"candidates": [{"frame_idx": 1, "box": [0, 0, 10, 10], "score": 0.2}]}
    assert blur_review.collect_flags(audit)[0]["cls"] == "face"


# ---------------------------------------------------------------------------
# find_manifests
# ---------------------------------------------------------------------------


def test_find_manifests_over_a_directory_and_a_file(blur_review, tmp_path):
    (tmp_path / "a.manifest.json").write_text("{}")
    (tmp_path / "b.manifest.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in blur_review.find_manifests(tmp_path)] == \
        ["a.manifest.json", "b.manifest.json"]
    one = tmp_path / "a.manifest.json"
    assert blur_review.find_manifests(one) == [one]


def test_find_manifests_exits_on_missing_or_empty(blur_review, tmp_path):
    with pytest.raises(SystemExit):
        blur_review.find_manifests(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        blur_review.find_manifests(empty)


# ---------------------------------------------------------------------------
# write_html — the page must never overstate how much was reviewed.
# ---------------------------------------------------------------------------


def test_page_never_claims_clean_when_extractions_failed(blur_review, tmp_path):
    """The worst possible bug in this tool: every extraction fails, `tiles` is
    empty, and the page renders 'No flagged regions' — the exact opposite of
    the truth, on the artifact used to decide whether a face ships."""
    body = blur_review.write_html("c", _manifest(), [], tmp_path, None,
                                   n_flagged=40, n_failed=40).read_text()
    assert "No flagged regions" not in body
    assert "40 flagged region(s), but none could be rendered" in body
    assert "nothing here has been reviewed" in body


def test_page_reports_shown_versus_flagged(blur_review, tmp_path):
    tiles = [{"origin": "yunet", "frame_idx": 1, "box": [0, 0, 10, 10], "score": 0.9,
              "cls": "face", "tile": "tiles/a.jpg", "context": "frames/f.jpg"}]
    body = blur_review.write_html("c", _manifest(), tiles, tmp_path, None,
                                   n_flagged=5, n_failed=4).read_text()
    assert "1 shown of 5 flagged" in body
    assert "4 flagged region(s) could NOT be extracted" in body


def test_page_warns_when_limit_truncated_the_list(blur_review, tmp_path):
    body = blur_review.write_html("c", _manifest(), [], tmp_path, None,
                                   n_flagged=300, n_truncated=100).read_text()
    assert "truncated by --limit" in body


def test_genuinely_empty_audit_says_so_without_overclaiming(blur_review, tmp_path):
    body = blur_review.write_html("c", _manifest(), [], tmp_path, None,
                                   n_flagged=0).read_text()
    assert "No flagged regions" in body
    # Still must not imply the clip is proven clean.
    assert "can still contain a face" in body


def test_manifest_values_are_escaped_into_the_page(blur_review, tmp_path):
    """The manifest travels back from a rented third-party pod. Numeric-looking
    fields were interpolated raw, so a string value rendered as live markup and
    could hide flagged tiles or overwrite the visible status."""
    m = _manifest()
    m["source"]["width"] = "<script>alert(1)</script>"
    m["source"]["n_frames"] = "<img src=x onerror=y>"
    body = blur_review.write_html("c", m, [], tmp_path, None, n_flagged=0).read_text()
    assert "<script>alert(1)</script>" not in body
    assert "<img src=x onerror=y>" not in body
    assert "&lt;script&gt;" in body


def test_page_survives_a_manifest_with_no_audit(blur_review, tmp_path):
    body = blur_review.write_html("c", {"status": "NEEDS_REVIEW"}, [], tmp_path,
                                   None, n_flagged=0).read_text()
    assert "NEEDS_REVIEW" in body


# ---------------------------------------------------------------------------
# crop_and_annotate
# ---------------------------------------------------------------------------


def _write_frame(tmp_path, w=640, h=480):
    import cv2
    frame = np.zeros((h, w, 3), np.uint8)
    frame[100:140, 100:140] = 200
    frame[300:340, 400:440] = 90
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), frame)
    return p


def test_two_flags_on_one_frame_get_distinct_tiles(blur_review, tmp_path):
    """Tile names keyed on frame+origin+2dp-score collided, so imwrite
    overwrote the first crop and the page showed one region twice — the
    second flagged region was never put in front of the reviewer."""
    frame_png = _write_frame(tmp_path)
    flags = [
        {"origin": "yunet", "frame_idx": 5, "box": [100, 100, 140, 140],
         "score": 0.9123, "cls": "face"},
        {"origin": "yunet", "frame_idx": 5, "box": [400, 300, 440, 340],
         "score": 0.9145, "cls": "face"},
    ]
    tiles = blur_review.crop_and_annotate(frame_png, flags, tmp_path, 5, 640, 480)
    assert len(tiles) == 2
    assert tiles[0]["tile"] != tiles[1]["tile"]
    assert len(list((tmp_path / "tiles").glob("*.jpg"))) == 2


def test_unreadable_frame_yields_no_tiles_rather_than_raising(blur_review, tmp_path):
    bad = tmp_path / "not-an-image.png"
    bad.write_text("nope")
    flags = [{"origin": "sweep", "frame_idx": 1, "box": [0, 0, 10, 10],
              "score": 0.2, "cls": "face"}]
    assert blur_review.crop_and_annotate(bad, flags, tmp_path, 1, 640, 480) == []


# ---------------------------------------------------------------------------
# review_clip input validation. The manifest is untrusted input.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clip_id", ["../../etc/evil", "/abs/evil", "a/b", ""])
def test_unsafe_clip_id_is_refused(blur_review, tmp_path, clip_id):
    """clip_id becomes a path component that this tool mkdirs, writes into,
    and (for the timelapse temp dir) rmtree's. An absolute value replaces the
    root outright when joined."""
    m = _manifest(clip_id=clip_id)
    p = tmp_path / "c.manifest.json"
    p.write_text(json.dumps(m))
    out = blur_review.review_clip(p, "ffmpeg", tmp_path / "review", False, 200)
    assert out["skipped"] == "unsafe clip_id"


@pytest.mark.parametrize("path", ["../escape.mp4", "/abs/x.mp4", "sub/dir.mp4", ""])
def test_unsafe_output_path_is_refused(blur_review, tmp_path, path):
    m = _manifest()
    m["output"]["path"] = path
    p = tmp_path / "c.manifest.json"
    p.write_text(json.dumps(m))
    out = blur_review.review_clip(p, "ffmpeg", tmp_path / "review", False, 200)
    assert out["skipped"] == "unsafe output path"


def test_missing_video_is_skipped_not_crashed(blur_review, tmp_path):
    p = tmp_path / "c.manifest.json"
    p.write_text(json.dumps(_manifest()))
    out = blur_review.review_clip(p, "ffmpeg", tmp_path / "review", False, 200)
    assert out["skipped"] == "missing video"


def test_unusable_geometry_is_skipped(blur_review, tmp_path):
    m = _manifest()
    m["source"]["fps"] = 0
    (tmp_path / "GX010042.blurred.mp4").write_bytes(b"x")
    p = tmp_path / "c.manifest.json"
    p.write_text(json.dumps(m))
    out = blur_review.review_clip(p, "ffmpeg", tmp_path / "review", False, 200)
    assert out["skipped"] == "bad geometry"
