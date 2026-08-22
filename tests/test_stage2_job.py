"""Milestone 1 contract tests for the Stage II de-identification job."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


def _facts(stage2_job, **over):
    values = dict(
        coded_width=1920,
        coded_height=1080,
        display_width=1920,
        display_height=1080,
        fps=30.0,
        n_frames=90,
        duration_s=3.0,
        rotation=0,
        is_cfr=True,
    )
    values.update(over)
    return stage2_job.VideoFacts(**values)


def _inputs(stage2_job, tmp_path, *, status="NEEDS_REVIEW", violations=12):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "GX010057.MP4"
    stage1 = tmp_path / "GX010057.blurred.mp4"
    manifest_path = tmp_path / "GX010057.manifest.json"
    source.write_bytes(b"original-video-fixture")
    stage1.write_bytes(b"stage-one-video-fixture")
    reasons = ["fill_integrity_violations > 0"] if status == "NEEDS_REVIEW" else []
    manifest = {
        "schema_version": 1,
        "run_id": "test-run-3",
        "clip_id": "GX010057",
        "source": {
            "filename": source.name,
            "sha256": stage2_job.sha256_file(source),
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "n_frames": 90,
            "duration_s": 3.0,
            "rotation": 0,
        },
        "egoblur": {
            "gen": "2",
            "face_threshold": 0.30,
            "lp_threshold": 0.40,
            "sweep_threshold": 0.10,
            "nms_iou": 0.30,
            "continue_threshold": 0.0,
            "detect_hz": 10.0,
            "redaction": "fill",
            "lp_checked": False,
            "dilate_scale": 1.3,
            "motion_margin_px": 8,
            "hold_frames": 45,
            "back_hold_frames": 45,
            "min_box_px": 8,
            "gen2_resize_px": None,
            "detect_batch": 8,
            "min_track_confirmations": 2,
        },
        "output": {
            "path": stage1.name,
            "sha256": stage2_job.sha256_file(stage1),
            "bytes": stage1.stat().st_size,
        },
        "audit": {
            "status": status,
            "status_reasons": reasons,
            "integrity_ran": True,
            "fill_integrity_checked": 100,
            "fill_integrity_violations": violations,
            "fill_integrity_frames": 90,
        },
        "status": status,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, stage1, manifest_path, manifest


def _probe(stage2_job, source, stage1, *, source_facts=None, stage1_facts=None):
    source_facts = source_facts or _facts(stage2_job)
    stage1_facts = stage1_facts or _facts(stage2_job)

    def probe(path):
        return source_facts if path.resolve() == source.resolve() else stage1_facts

    return probe


def _validate(stage2_job, tmp_path, **kwargs):
    source, stage1, manifest_path, manifest = _inputs(stage2_job, tmp_path, **kwargs)
    result = stage2_job.validate_stage1(
        source,
        stage1,
        manifest_path,
        probe_fn=_probe(stage2_job, source, stage1),
    )
    return result, source, stage1, manifest_path, manifest


def test_valid_stage1_is_hashed_and_preserves_review_findings(stage2_job, tmp_path):
    result, source, stage1, manifest_path, _ = _validate(stage2_job, tmp_path)
    assert result.clip_id == "GX010057"
    assert result.source.sha256 == stage2_job.sha256_file(source)
    assert result.stage1_video.sha256 == stage2_job.sha256_file(stage1)
    assert result.stage1_manifest.sha256 == stage2_job.sha256_file(manifest_path)
    assert result.stage1_status == "NEEDS_REVIEW"
    assert result.stage1_audit_reasons == ("fill_integrity_violations > 0",)
    assert any("12 legacy fill-integrity" in warning for warning in result.warnings)
    assert any("face-weight hash" in warning for warning in result.warnings)


def test_nonzero_legacy_integrity_findings_do_not_reject_proven_stage1(stage2_job, tmp_path):
    result, *_ = _validate(stage2_job, tmp_path, violations=23_216)
    assert result.stage1_status == "NEEDS_REVIEW"
    assert any("23216" in warning for warning in result.warnings)


def test_zero_redaction_needs_review_clip_is_valid_stage2_input(stage2_job, tmp_path):
    source, stage1, manifest_path, manifest = _inputs(
        stage2_job, tmp_path, status="NEEDS_REVIEW", violations=0
    )
    zero_reason = (
        "ZERO frames had a FACE redacted and nothing corroborates that — "
        "detection may have failed silently. Check before shipping."
    )
    manifest["audit"]["fill_integrity_checked"] = 0
    manifest["audit"]["status_reasons"] = [zero_reason]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = stage2_job.validate_stage1(
        source,
        stage1,
        manifest_path,
        probe_fn=_probe(stage2_job, source, stage1),
    )
    assert result.stage1_status == "NEEDS_REVIEW"
    assert result.stage1_audit_reasons == (zero_reason,)


def test_zero_integrity_checks_without_zero_redaction_reason_fail(stage2_job, tmp_path):
    source, stage1, manifest_path, manifest = _inputs(stage2_job, tmp_path)
    manifest["audit"]["fill_integrity_checked"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(
            source,
            stage1,
            manifest_path,
            probe_fn=_probe(stage2_job, source, stage1),
        )
    assert caught.value.code == "STAGE1_VALIDATION_FAILED"
    assert any("may be zero only" in item for item in caught.value.details["problems"])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("gen", "1"),
        ("face_threshold", 0.2),
        ("continue_threshold", 0.15),
        ("redaction", "blur"),
        ("lp_checked", True),
        ("dilate_scale", 1.0),
        ("motion_margin_px", 0),
        ("hold_frames", 44),
        ("back_hold_frames", 44),
        ("min_track_confirmations", 1),
    ],
)
def test_stage1_frozen_settings_are_strict(stage2_job, tmp_path, field, bad_value):
    source, stage1, manifest_path, manifest = _inputs(stage2_job, tmp_path)
    manifest["egoblur"][field] = bad_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(
            source,
            stage1,
            manifest_path,
            probe_fn=_probe(stage2_job, source, stage1),
        )
    assert caught.value.code == "STAGE1_VALIDATION_FAILED"
    assert any(field in problem for problem in caught.value.details["problems"])


def test_stage1_hash_mismatch_fails_closed(stage2_job, tmp_path):
    source, stage1, manifest_path, manifest = _inputs(stage2_job, tmp_path)
    manifest["output"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(
            source,
            stage1,
            manifest_path,
            probe_fn=_probe(stage2_job, source, stage1),
        )
    assert caught.value.code == "STAGE1_VALIDATION_FAILED"
    assert any("output.sha256" in problem for problem in caught.value.details["problems"])


def test_missing_input_returns_a_structured_error(stage2_job, tmp_path):
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(
            tmp_path / "missing.mp4",
            tmp_path / "also-missing.mp4",
            tmp_path / "missing.json",
        )
    assert caught.value.code == "INPUT_NOT_FOUND"
    assert "Source video" in caught.value.message


def test_input_change_during_probe_fails_closed(stage2_job, tmp_path):
    source, stage1, manifest_path, _ = _inputs(stage2_job, tmp_path)
    calls = 0

    def changing_probe(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            stage1.write_bytes(b"changed-during-validation")
        return _facts(stage2_job)

    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(source, stage1, manifest_path, probe_fn=changing_probe)
    assert caught.value.code == "STAGE1_VALIDATION_FAILED"
    assert any("input changed" in item for item in caught.value.details["problems"])


@pytest.mark.parametrize(
    "stage1_facts",
    [
        {"n_frames": 89},
        {"fps": 29.0},
        {"display_width": 1280},
        {"is_cfr": False},
    ],
)
def test_stage1_video_facts_must_match_source(stage2_job, tmp_path, stage1_facts):
    source, stage1, manifest_path, _ = _inputs(stage2_job, tmp_path)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(
            source,
            stage1,
            manifest_path,
            probe_fn=_probe(
                stage2_job,
                source,
                stage1,
                stage1_facts=_facts(stage2_job, **stage1_facts),
            ),
        )
    assert caught.value.code == "STAGE1_VALIDATION_FAILED"


def test_duration_mismatch_fails_closed(stage2_job, tmp_path):
    source, stage1, manifest_path, _ = _inputs(stage2_job, tmp_path)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(
            source,
            stage1,
            manifest_path,
            probe_fn=_probe(
                stage2_job,
                source,
                stage1,
                stage1_facts=_facts(stage2_job, duration_s=2.0),
            ),
        )
    assert any("Stage I duration" in item for item in caught.value.details["problems"])


def test_malformed_ffprobe_numbers_return_a_structured_error(stage2_job, tmp_path, monkeypatch):
    video = tmp_path / "bad.mp4"
    video.write_bytes(b"not-a-real-video")
    monkeypatch.setattr(stage2_job.shutil, "which", lambda _name: "/fake/ffprobe")
    monkeypatch.setattr(
        stage2_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": "wrong",
                            "height": 1080,
                            "r_frame_rate": "30/1",
                            "avg_frame_rate": "30/1",
                            "nb_read_packets": "90",
                            "duration": "3.0",
                        }
                    ]
                }
            ),
            stderr="",
        ),
    )
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.probe_video(video)
    assert caught.value.code == "VIDEO_PROBE_INVALID"


def test_integrity_must_have_run_across_the_complete_clip(stage2_job, tmp_path):
    source, stage1, manifest_path, manifest = _inputs(stage2_job, tmp_path)
    manifest["audit"]["integrity_ran"] = False
    manifest["audit"]["fill_integrity_frames"] = 89
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(
            source,
            stage1,
            manifest_path,
            probe_fn=_probe(stage2_job, source, stage1),
        )
    assert any("integrity_ran" in item for item in caught.value.details["problems"])
    assert any("fill_integrity_frames" in item for item in caught.value.details["problems"])


@pytest.mark.parametrize("value", ["../escape", "/absolute", "a/b", "", ".", ".."])
def test_run_and_clip_ids_cannot_escape_the_work_directory(stage2_job, tmp_path, value):
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.build_run_paths(tmp_path, value, "clip")
    assert caught.value.code == "UNSAFE_PATH_COMPONENT"


def test_run_paths_reject_preexisting_symlink_escape(stage2_job, tmp_path):
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (work / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.build_run_paths(work, "run", "clip")
    assert caught.value.code == "UNSAFE_OUTPUT_LOCATION"


def test_layered_fingerprints_are_stable_and_scoped(stage2_job):
    dino = {"source": "a", "prompt": "face.", "threshold": 0.10}
    first = stage2_job.dino_fingerprint(dino)
    assert first == stage2_job.dino_fingerprint(dict(reversed(list(dino.items()))))
    assert first != stage2_job.dino_fingerprint({**dino, "prompt": "head."})
    sam = stage2_job.sam_fingerprint({"dino": first, "threshold": 0.20})
    render_a = stage2_job.render_fingerprint({"sam": sam, "crf": 18})
    render_b = stage2_job.render_fingerprint({"sam": sam, "crf": 20})
    assert render_a != render_b
    assert sam == stage2_job.sam_fingerprint({"dino": first, "threshold": 0.20})


def test_immutable_artifact_reuses_identical_content_and_rejects_change(stage2_job, tmp_path):
    path = tmp_path / "artifact.json"
    first = stage2_job.write_immutable_json(path, {"value": 1})
    second = stage2_job.write_immutable_json(path, {"value": 1})
    assert first == second
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.write_immutable_json(path, {"value": 2})
    assert caught.value.code == "IMMUTABLE_ARTIFACT_CONFLICT"
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_atomic_replace_keeps_previous_file(stage2_job, tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_bytes(b"old-state\n")

    def fail_replace(_source, _destination):
        raise OSError("simulated interrupted promotion")

    monkeypatch.setattr(stage2_job.os, "replace", fail_replace)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.atomic_write_json(path, {"state": "new"})
    assert caught.value.code == "ATOMIC_WRITE_FAILED"
    assert path.read_bytes() == b"old-state\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_malformed_rotation_is_a_validation_error_not_a_crash(stage2_job, tmp_path):
    source, stage1, manifest_path, manifest = _inputs(stage2_job, tmp_path)
    manifest["source"]["rotation"] = "not-a-number"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_stage1(
            source,
            stage1,
            manifest_path,
            probe_fn=_probe(stage2_job, source, stage1),
        )
    assert caught.value.code == "STAGE1_VALIDATION_FAILED"
    assert any("source.rotation" in item for item in caught.value.details["problems"])


def test_state_is_atomic_and_cannot_move_backward(stage2_job, tmp_path):
    path = tmp_path / "state.json"
    stage2_job.transition_state(
        path, run_id="run", clip_id="clip", mode="fake", target="PENDING"
    )
    stage2_job.transition_state(
        path, run_id="run", clip_id="clip", mode="fake", target="VALIDATED"
    )
    assert stage2_job.load_state(path).state == "VALIDATED"
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.transition_state(
            path, run_id="run", clip_id="clip", mode="fake", target="PENDING"
        )
    assert caught.value.code == "INVALID_STATE_TRANSITION"


def test_resume_cannot_reuse_another_runs_state(stage2_job, tmp_path):
    path = tmp_path / "state.json"
    stage2_job.transition_state(
        path,
        run_id="first-run",
        clip_id="clip",
        mode="fake",
        target="PROCESSING_COMPLETE",
        completed_layers=("dino", "sam", "render"),
        reusable_layers=("dino", "sam", "render"),
    )
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.advance_state(
            path,
            run_id="second-run",
            clip_id="clip",
            mode="fake",
            target="VALIDATED",
        )
    assert caught.value.code == "STATE_IDENTITY_MISMATCH"


def test_corrupt_state_is_never_treated_as_resumable(stage2_job, tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{torn", encoding="utf-8")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.load_state(path)
    assert caught.value.code == "INVALID_JSON"


def test_complete_state_without_completed_artifacts_is_rejected(stage2_job, tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "clip_id": "clip",
                "mode": "fake",
                "state": "PROCESSING_COMPLETE",
                "completed_layers": [],
                "reusable_layers": [],
                "last_error": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.load_state(path)
    assert caught.value.code == "INVALID_STATE"


def test_fake_pipeline_is_deterministic_resumable_and_not_review_accepted(stage2_job, tmp_path):
    source, stage1, manifest_path, _ = _inputs(stage2_job, tmp_path / "inputs")
    probe = _probe(stage2_job, source, stage1)
    work = tmp_path / "work"
    first = stage2_job.run_fake_pipeline(
        source_video=source,
        stage1_video=stage1,
        stage1_manifest=manifest_path,
        work_dir=work,
        run_id="milestone-1",
        probe_fn=probe,
    )
    paths = stage2_job.build_run_paths(work, "milestone-1", "GX010057")
    sam_paths = tuple(Path(ref.path) for ref in first.sam_mask_shards)
    before = {
        path: path.read_bytes() for path in (paths.dino, *sam_paths, paths.render)
    }
    second = stage2_job.run_fake_pipeline(
        source_video=source,
        stage1_video=stage1,
        stage1_manifest=manifest_path,
        work_dir=work,
        run_id="milestone-1",
        probe_fn=probe,
    )
    after = {path: path.read_bytes() for path in before}
    assert first == second
    assert before == after
    assert stage2_job.load_state(paths.state).state == "PROCESSING_COMPLETE"
    assert first.processing_state == "PROCESSING_COMPLETE"
    assert first.review_status == "NOT_REVIEWABLE_FAKE"
    assert first.audit_status == "NOT_RUN_FAKE"
    assert json.loads(paths.manifest.read_text())["mode"] == "fake"


def test_fake_run_cli_accepts_real_three_frame_fixture(stage2_job, tmp_path, capsys):
    ffmpeg = stage2_job.shutil.which("ffmpeg")
    if ffmpeg is None or stage2_job.shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for the real fixture acceptance test")
    source, stage1, manifest_path, manifest = _inputs(stage2_job, tmp_path / "inputs")
    completed = stage2_job.subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=100x80:r=30:d=0.1",
            "-frames:v",
            "3",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    stage1.write_bytes(source.read_bytes())
    facts = stage2_job.probe_video(source)
    manifest["source"].update(
        {
            "sha256": stage2_job.sha256_file(source),
            "width": facts.display_width,
            "height": facts.display_height,
            "fps": facts.fps,
            "n_frames": facts.n_frames,
            "duration_s": facts.duration_s,
            "rotation": facts.rotation,
        }
    )
    manifest["output"].update(
        {
            "sha256": stage2_job.sha256_file(stage1),
            "bytes": stage1.stat().st_size,
        }
    )
    manifest["audit"].update(
        {
            "fill_integrity_checked": 0,
            "fill_integrity_violations": 0,
            "fill_integrity_frames": facts.n_frames,
            "status_reasons": [
                "ZERO frames had a FACE redacted and nothing corroborates that — "
                "detection may have failed silently. Check before shipping."
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    work = tmp_path / "work"
    exit_code = stage2_job.main(
        [
            "--json",
            "fake-run",
            "--source-video",
            str(source),
            "--stage1-video",
            str(stage1),
            "--stage1-manifest",
            str(manifest_path),
            "--work-dir",
            str(work),
            "--run-id",
            "three-frame-fixture",
        ]
    )
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["processing_state"] == "PROCESSING_COMPLETE"
    assert output["review_status"] == "NOT_REVIEWABLE_FAKE"


def test_schema_records_keep_review_separate_from_processing(stage2_job):
    review = stage2_job.ReviewRecord(
        schema_version=1,
        review_id="review-1",
        processing_manifest_sha256="1" * 64,
        output_sha256="2" * 64,
        review_status="ACCEPTED",
        reviewer="human",
        reviewed_at="2026-08-20T00:00:00Z",
    )
    changed_output = replace(review, output_sha256="3" * 64)
    assert review.processing_manifest_sha256 == changed_output.processing_manifest_sha256
    assert review.output_sha256 != changed_output.output_sha256


def test_label_schema_represents_faces_and_negative_examples(stage2_job):
    face = stage2_job.FaceEventLabel(
        schema_version=1,
        event_id="face-1",
        clip_id="GX010057",
        frame_start=10,
        frame_end=20,
        conservative_box=(1.0, 2.0, 30.0, 40.0),
        label_kind="face_event",
        visibility="partial",
        category="profile",
        stage1_verdict="missed",
        dino_proposal_verdicts=({"proposal_id": "p1", "verdict": "face"},),
        final_mask_coverage=None,
        reviewer_disposition="pending",
    )
    negative = replace(
        face,
        event_id="negative-1",
        label_kind="negative_example",
        category="hand",
        stage1_verdict="covered",
    )
    assert face.label_kind == "face_event"
    assert negative.label_kind == "negative_example"
