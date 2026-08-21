from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest


def _stage1(stage2_job, tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("source.mp4", "stage1.mp4", "stage1.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append(stage2_job.artifact_ref(path))
    facts = stage2_job.VideoFacts(
        coded_width=100,
        coded_height=80,
        display_width=100,
        display_height=80,
        fps=30.0,
        n_frames=20,
        duration_s=2 / 3,
        rotation=0,
        is_cfr=True,
    )
    return stage2_job.StageIInput(
        schema_version=1,
        clip_id="clip",
        source=files[0],
        stage1_video=files[1],
        stage1_manifest=files[2],
        source_video=facts,
        stage1_output_video=facts,
        stage1_status="PASS_AUTOMATED",
        stage1_audit_reasons=(),
        egoblur={},
        warnings=(),
    )


def _label(stage2_job, *, kind="face_event", event_id="event-1"):
    return stage2_job.FaceEventLabel(
        schema_version=1,
        event_id=event_id,
        clip_id="clip",
        frame_start=2,
        frame_end=4,
        conservative_box=(10.0, 10.0, 20.0, 20.0),
        label_kind=kind,
        visibility="visible" if kind == "face_event" else "not_applicable",
        category="profile" if kind == "face_event" else "hand",
        stage1_verdict="missed" if kind == "face_event" else "not_applicable",
        dino_proposal_verdicts=(
            {
                "proposal_id": "proposal-1",
                "verdict": "face" if kind == "face_event" else "false_positive",
            },
        ),
        final_mask_coverage=None,
        reviewer_disposition="pending",
    )


def _completed_run(stage2_job, tmp_path: Path):
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", "clip")
    output = paths.render.parent / "stage2.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"verified stage2 video")
    output_ref = stage2_job.artifact_ref(output)
    render_meta = stage2_job.RenderArtifactMeta(
        schema_version=1,
        artifact_type="render",
        fingerprint="1" * 64,
        stage1_video_sha256="2" * 64,
        mask_set_sha256="3" * 64,
        output=output_ref,
        encoder={"pixel_source": "stage1_video_only"},
        verification={"passed": True, "publishable": False},
    )
    render_ref = stage2_job.write_immutable_json(paths.render, render_meta)
    dino_ref = stage2_job.ArtifactRef(path="/private/dino/artifact.json", sha256="4" * 64, bytes=10)
    sam_refs = [
        stage2_job.ArtifactRef(path="/private/sam/window-0.npz", sha256="5" * 64, bytes=10),
    ]
    manifest = {
        "schema_version": 1,
        "code_version": stage2_job.STAGE2_CODE_VERSION,
        "run_id": "run",
        "clip_id": "clip",
        "mode": "production",
        "processing_state": "PROCESSING_COMPLETE",
        "audit_status": "PASS_AUTOMATED_TECHNICAL",
        "review_status": "PENDING",
        "dino_artifact": dino_ref,
        "sam_mask_shards": sam_refs,
        "render_artifact": render_ref,
    }
    stage2_job.write_immutable_json(paths.manifest, manifest)
    stage2_job.transition_state(
        paths.state,
        run_id="run",
        clip_id="clip",
        mode="production",
        target="PROCESSING_COMPLETE",
        completed_layers=("dino", "sam", "render"),
        reusable_layers=("dino", "sam", "render"),
    )
    return paths, output_ref


def test_private_labels_seeds_and_evidence_stay_under_do_not_ship(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, tmp_path)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    labels = (
        _label(stage2_job),
        _label(stage2_job, kind="negative_example", event_id="negative-1"),
    )
    label_ref = stage2_job.write_private_labels(paths, stage1=stage1, labels=labels)
    seed = stage2_job.ManualSeed(
        schema_version=1,
        seed_id="miss-1",
        clip_id="clip",
        frame_idx=3,
        box=(10.0, 10.0, 20.0, 20.0),
        reason="human-confirmed miss",
    )
    seed_ref = stage2_job.write_private_manual_seeds(paths, stage1=stage1, seeds=(seed,))
    assert stage2_job.load_private_labels_file(
        Path(label_ref.path), stage1
    ) == stage2_job.validate_face_event_labels(labels, stage1)
    assert stage2_job.load_private_manual_seeds_file(Path(seed_ref.path), stage1) == (seed,)
    evidence = tmp_path / "face-crop.jpg"
    evidence.write_bytes(b"private pixels")
    evidence_ref, evidence_meta = stage2_job.register_private_evidence(
        paths,
        clip_id="clip",
        evidence_id="face-1-frame-3",
        evidence_kind="face_crop",
        source=evidence,
        frame_start=3,
        frame_end=3,
    )
    for reference in (label_ref, seed_ref, evidence_ref, evidence_meta):
        assert paths.private_root in Path(reference.path).parents
        assert "DO-NOT-SHIP" in Path(reference.path).parts


def test_register_private_evidence_hashes_the_bytes_it_actually_writes(
    stage2_job, tmp_path, monkeypatch
):
    stage1 = _stage1(stage2_job, tmp_path)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    evidence = tmp_path / "face-crop.jpg"
    evidence.write_bytes(b"original private pixels")

    # Simulate the source file being rewritten by another process in the
    # narrow window between a hash pass and the read that actually gets
    # persisted - the exact TOCTOU race this closes. artifact_ref is no
    # longer called on the source at all after the fix, so this side
    # effect never fires on the fixed code path.
    original_artifact_ref = stage2_job.artifact_ref

    def racing_artifact_ref(path, *args, **kwargs):
        result = original_artifact_ref(path, *args, **kwargs)
        if Path(path) == evidence:
            evidence.write_bytes(b"tampered content written mid-race")
        return result

    monkeypatch.setattr(stage2_job, "artifact_ref", racing_artifact_ref)

    evidence_ref, _metadata_ref = stage2_job.register_private_evidence(
        paths,
        clip_id="clip",
        evidence_id="face-1-frame-3",
        evidence_kind="face_crop",
        source=evidence,
        frame_start=3,
        frame_end=3,
    )
    written_content = Path(evidence_ref.path).read_bytes()
    actual_hash = stage2_job.hashlib.sha256(written_content).hexdigest()
    assert actual_hash in Path(evidence_ref.path).name
    assert evidence_ref.sha256 == actual_hash


def test_invalid_or_duplicate_private_labels_fail_closed(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, tmp_path)
    duplicate = (_label(stage2_job), _label(stage2_job))
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_face_event_labels(duplicate, stage1)
    assert caught.value.code == "INVALID_PRIVATE_LABEL"
    out_of_bounds = replace(_label(stage2_job), frame_end=20)
    with pytest.raises(stage2_job.Stage2Error):
        stage2_job.validate_face_event_labels((out_of_bounds,), stage1)


def test_human_acceptance_requires_complete_attestation_and_exact_hashes(stage2_job, tmp_path):
    paths, output_ref = _completed_run(stage2_job, tmp_path)
    with pytest.raises(stage2_job.Stage2Error) as incomplete:
        stage2_job.create_review_record(
            paths,
            reviewer="human",
            reviewed_at="2026-08-20T12:00:00Z",
            review_status="ACCEPTED",
            full_clip_reviewed=True,
            flagged_intervals_reviewed=False,
        )
    assert incomplete.value.code == "INCOMPLETE_HUMAN_REVIEW"
    reference, record = stage2_job.create_review_record(
        paths,
        reviewer="human",
        reviewed_at="2026-08-20T12:00:00Z",
        review_status="ACCEPTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )
    assert reference.sha256 == stage2_job.artifact_ref(Path(reference.path)).sha256
    assert record.output_sha256 == output_ref.sha256
    assert stage2_job.effective_review_status(paths) == ("ACCEPTED", reference)
    status = stage2_job.run_status(paths)
    assert status["processing_state"] == "PROCESSING_COMPLETE"
    assert status["audit_status"] == "PASS_AUTOMATED_TECHNICAL"
    assert status["review_status"] == "ACCEPTED"

    raw = stage2_job.read_json(paths.manifest)
    raw["correction"] = "manual seed changed"
    stage2_job.atomic_write_json(paths.manifest, raw)
    assert stage2_job.effective_review_status(paths) == ("PENDING", None)


def test_release_check_flags_dino_and_sam_artifacts_as_internal(stage2_job, tmp_path):
    paths, output_ref = _completed_run(stage2_job, tmp_path)
    paths.dino.parent.mkdir(parents=True, exist_ok=True)
    paths.dino.write_bytes(b"dino proposals derived from the unredacted original")
    dino_ref = stage2_job.artifact_ref(paths.dino)
    paths.sam_dir.mkdir(parents=True, exist_ok=True)
    sam_shard = paths.sam_dir / "window-0.npz"
    sam_shard.write_bytes(b"sam mask shard derived from the unredacted original")
    sam_ref = stage2_job.artifact_ref(sam_shard)
    manifest = stage2_job.read_json(paths.manifest)
    manifest["dino_artifact"] = stage2_job._jsonable(dino_ref)
    manifest["sam_mask_shards"] = [stage2_job._jsonable(sam_ref)]
    stage2_job.atomic_write_json(paths.manifest, manifest)
    stage2_job.create_review_record(
        paths,
        reviewer="human",
        reviewed_at="2026-08-20T12:00:00Z",
        review_status="ACCEPTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )

    release = tmp_path / "release"
    release.mkdir()
    shutil.copyfile(output_ref.path, release / "accepted.mp4")
    shutil.copyfile(paths.dino, release / "innocent-name.json")
    with pytest.raises(stage2_job.Stage2Error) as leaked_dino:
        stage2_job.release_check(paths, release_root=release)
    assert "internal processing/review artifacts are present" in " ".join(
        leaked_dino.value.details["problems"]
    )
    (release / "innocent-name.json").unlink()

    shutil.copyfile(sam_shard, release / "also-innocent.npz")
    with pytest.raises(stage2_job.Stage2Error) as leaked_sam:
        stage2_job.release_check(paths, release_root=release)
    assert "internal processing/review artifacts are present" in " ".join(
        leaked_sam.value.details["problems"]
    )
    (release / "also-innocent.npz").unlink()

    result = stage2_job.release_check(paths, release_root=release)
    assert result["release_ready"] is True


def test_release_check_rejects_private_content_even_when_renamed(stage2_job, tmp_path):
    """Private bytes and post-review corrections both fail closed."""
    paths, output_ref = _completed_run(stage2_job, tmp_path)
    stage1 = _stage1(stage2_job, tmp_path / "private-inputs")
    accepted_label = replace(
        _label(stage2_job), final_mask_coverage=1.0, reviewer_disposition="accepted"
    )
    labels = stage2_job.write_private_labels(paths, stage1=stage1, labels=(accepted_label,))
    manifest = stage2_job.read_json(paths.manifest)
    manifest["label_artifacts"] = [stage2_job._jsonable(labels)]
    stage2_job.atomic_write_json(paths.manifest, manifest)
    stage2_job.create_review_record(
        paths,
        reviewer="human",
        reviewed_at="2026-08-20T12:00:00Z",
        review_status="ACCEPTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )
    release = tmp_path / "release"
    release.mkdir()
    shutil.copyfile(output_ref.path, release / "accepted.mp4")
    shutil.copyfile(labels.path, release / "innocent-name.json")
    with pytest.raises(stage2_job.Stage2Error) as leaked:
        stage2_job.release_check(paths, release_root=release)
    assert leaked.value.code == "RELEASE_CHECK_FAILED"
    assert "private artifact content" in " ".join(leaked.value.details["problems"])
    (release / "innocent-name.json").unlink()
    result = stage2_job.release_check(paths, release_root=release)
    assert result["release_ready"] is True
    correction = stage2_job.ManualSeed(
        schema_version=1,
        seed_id="late-correction",
        clip_id="clip",
        frame_idx=5,
        box=(10.0, 10.0, 20.0, 20.0),
        reason="miss found after acceptance",
    )
    stage2_job.write_private_manual_seeds(paths, stage1=stage1, seeds=(correction,))
    assert stage2_job.effective_review_status(paths) == ("PENDING", None)
    with pytest.raises(stage2_job.Stage2Error) as invalidated:
        stage2_job.release_check(paths, release_root=release)
    assert "not bound to current processing" in " ".join(invalidated.value.details["problems"])


def test_effective_review_status_orders_by_parsed_time_not_string(stage2_job, tmp_path):
    paths, _output_ref = _completed_run(stage2_job, tmp_path)
    stage2_job.create_review_record(
        paths,
        reviewer="human",
        reviewed_at="2026-08-20T12:00:00Z",
        review_status="REJECTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )
    # Chronologically later (500ms after the REJECTED record above) despite sorting
    # lexicographically *before* it as a raw string, since "." < "Z" in ASCII.
    stage2_job.create_review_record(
        paths,
        reviewer="human",
        reviewed_at="2026-08-20T12:00:00.500000Z",
        review_status="ACCEPTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )
    status, _reference = stage2_job.effective_review_status(paths)
    assert status == "ACCEPTED"


def test_canonical_review_edit_cannot_forge_acceptance(stage2_job, tmp_path):
    paths, _output_ref = _completed_run(stage2_job, tmp_path)
    reference, _record = stage2_job.create_review_record(
        paths,
        reviewer="human",
        reviewed_at="2026-08-20T12:00:00Z",
        review_status="REJECTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )
    raw = stage2_job.read_json(Path(reference.path))
    raw["review_status"] = "ACCEPTED"
    stage2_job.atomic_write_json(Path(reference.path), raw)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.effective_review_status(paths)
    assert caught.value.code == "INVALID_REVIEW_RECORD"
    assert "review_id_content_binding" in caught.value.details["problems"]


def test_review_records_reject_unknown_fields_and_non_boolean_attestations(stage2_job, tmp_path):
    paths, _output_ref = _completed_run(stage2_job, tmp_path)
    reference, _record = stage2_job.create_review_record(
        paths,
        reviewer="human",
        reviewed_at="2026-08-20T12:00:00Z",
        review_status="REJECTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )
    review_path = Path(reference.path)
    original = stage2_job.read_json(review_path)
    with_extra = {**original, "publication_approved": True}
    stage2_job.atomic_write_json(review_path, with_extra)
    with pytest.raises(stage2_job.Stage2Error) as unknown:
        stage2_job.effective_review_status(paths)
    assert "review_record_fields" in unknown.value.details["problems"]

    malformed = {**original, "full_clip_reviewed": 1}
    stage2_job.atomic_write_json(review_path, malformed)
    with pytest.raises(stage2_job.Stage2Error) as non_boolean:
        stage2_job.effective_review_status(paths)
    assert "full_clip_reviewed" in non_boolean.value.details["problems"]


def test_pending_or_undercovered_labels_block_acceptance(stage2_job, tmp_path):
    paths, _output_ref = _completed_run(stage2_job, tmp_path)
    stage1 = _stage1(stage2_job, tmp_path / "private-inputs")
    label_ref = stage2_job.write_private_labels(paths, stage1=stage1, labels=(_label(stage2_job),))
    manifest = stage2_job.read_json(paths.manifest)
    manifest["label_artifacts"] = [stage2_job._jsonable(label_ref)]
    stage2_job.atomic_write_json(paths.manifest, manifest)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.create_review_record(
            paths,
            reviewer="human",
            reviewed_at="2026-08-20T12:00:00Z",
            review_status="ACCEPTED",
            full_clip_reviewed=True,
            flagged_intervals_reviewed=True,
        )
    assert caught.value.code == "LABEL_ACCEPTANCE_INCOMPLETE"


def test_stop_resume_and_explicit_render_invalidation(stage2_job, tmp_path):
    paths, _output = _completed_run(stage2_job, tmp_path)
    assert stage2_job.request_stop(paths)["stop_requested"] is False
    state = stage2_job.invalidate_from(paths, layer="render")
    assert state.state == "SAM_COMPLETE"
    assert not paths.render.exists()
    assert not paths.manifest.exists()
    assert stage2_job.request_stop(paths)["stop_requested"] is True
    assert paths.stop_request.exists()
    with pytest.raises(stage2_job.Stage2Error) as stopped:
        stage2_job.ensure_run_not_stopped(paths)
    assert stopped.value.code == "STOP_REQUESTED"
    resumed = stage2_job.prepare_resume(paths)
    assert resumed.state == "SAM_COMPLETE"
    assert not paths.stop_request.exists()


def test_recompute_archives_prior_review_records_instead_of_erasing_them(stage2_job, tmp_path):
    paths, _output = _completed_run(stage2_job, tmp_path)
    rejected_reference, rejected_record = stage2_job.create_review_record(
        paths,
        reviewer="human",
        reviewed_at="2026-08-20T12:00:00Z",
        review_status="REJECTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )
    history_path = paths.root / "reviews-history.jsonl"
    assert not history_path.exists()

    stage2_job.invalidate_from(paths, layer="render")

    assert not paths.reviews_dir.exists()
    assert history_path.exists()
    first_batch = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(first_batch) == 1
    archived = first_batch[0]
    assert archived["review_id"] == rejected_record.review_id
    assert archived["review_status"] == "REJECTED"
    assert archived["review_record_sha256"] == rejected_reference.sha256
    assert archived["invalidated_from_layer"] == "render"
    assert archived["invalidated_at"].endswith("Z")

    # A recompute with nothing in reviews_dir must not append empty/duplicate entries.
    stage2_job.invalidate_from(paths, layer="render")
    assert [json.loads(line) for line in history_path.read_text().splitlines()] == first_batch

    # A second real review followed by another recompute must append, not overwrite,
    # so the log stays a complete history rather than only remembering the latest.
    paths_after = stage2_job.build_run_paths(tmp_path / "work", "run", "clip")
    output = paths_after.render.parent / "stage2.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"second verified stage2 video")
    output_ref = stage2_job.artifact_ref(output)
    render_meta = stage2_job.RenderArtifactMeta(
        schema_version=1,
        artifact_type="render",
        fingerprint="1" * 64,
        stage1_video_sha256="2" * 64,
        mask_set_sha256="3" * 64,
        output=output_ref,
        encoder={"pixel_source": "stage1_video_only"},
        verification={"passed": True, "publishable": False},
    )
    render_ref = stage2_job.write_immutable_json(paths_after.render, render_meta)
    manifest = {
        "schema_version": 1,
        "code_version": stage2_job.STAGE2_CODE_VERSION,
        "run_id": "run",
        "clip_id": "clip",
        "mode": "production",
        "processing_state": "PROCESSING_COMPLETE",
        "audit_status": "PASS_AUTOMATED_TECHNICAL",
        "review_status": "PENDING",
        "render_artifact": render_ref,
    }
    stage2_job.write_immutable_json(paths_after.manifest, manifest)
    stage2_job.transition_state(
        paths_after.state,
        run_id="run",
        clip_id="clip",
        mode="production",
        target="PROCESSING_COMPLETE",
        completed_layers=("dino", "sam", "render"),
        reusable_layers=("dino", "sam", "render"),
    )
    _accepted_reference, accepted_record = stage2_job.create_review_record(
        paths_after,
        reviewer="human",
        reviewed_at="2026-08-21T09:00:00Z",
        review_status="ACCEPTED",
        full_clip_reviewed=True,
        flagged_intervals_reviewed=True,
    )
    stage2_job.invalidate_from(paths_after, layer="render")
    second_batch = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(second_batch) == 2
    assert second_batch[0]["review_id"] == rejected_record.review_id
    assert second_batch[1]["review_id"] == accepted_record.review_id
    assert second_batch[1]["review_status"] == "ACCEPTED"


def test_recompute_refuses_symlinked_layer_directory(stage2_job, tmp_path):
    paths, _output = _completed_run(stage2_job, tmp_path)
    paths.private_root.mkdir()
    marker = paths.private_root / "keep.txt"
    marker.write_text("must survive")
    shutil.rmtree(paths.render.parent)
    paths.render.parent.symlink_to(paths.private_root, target_is_directory=True)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.invalidate_from(paths, layer="render")
    assert caught.value.code == "UNSAFE_INVALIDATION_PATH"
    assert marker.read_text() == "must survive"


def test_operator_command_surface_and_dry_run(stage2_job, capsys, tmp_path):
    commands = {
        "doctor",
        "smoke",
        "pilot",
        "sweep",
        "run",
        "status",
        "resume",
        "stop",
        "review",
        "release-check",
    }
    help_text = stage2_job.parse_args
    assert callable(help_text)
    exit_code = stage2_job.main(
        ["--json", "--dry-run", "pilot", "GX010057", "--work-dir", str(tmp_path)]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["gpu_execution_enabled"] is False
    parser_source = Path(stage2_job.__file__).read_text()
    assert all(f'"{command}"' in parser_source for command in commands)


def test_real_gpu_commands_fail_with_actionable_boundary(stage2_job, capsys, tmp_path):
    exit_code = stage2_job.main(["--json", "pilot", "GX010057", "--work-dir", str(tmp_path)])
    assert exit_code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error_code"] == "REAL_GPU_EXECUTION_DEFERRED"
    assert "runpod_setup_stage2.sh" in error["recovery"]


def test_doctor_reports_persistent_paths_without_mutating(stage2_job, tmp_path):
    workspace = tmp_path / "workspace"
    result = stage2_job.doctor_stage2(workspace)
    assert result["status"] == "SETUP_REQUIRED"
    assert result["gpu_execution_enabled"] is False
    assert result["persistent_paths"]["uv_cache"].startswith(str(workspace))
    assert not workspace.exists()


def test_stage2_setup_dry_run_is_idempotent_and_non_mutating(stage2_job, tmp_path):
    repo = Path(stage2_job.__file__).resolve().parents[1]
    script = repo / "scripts" / "runpod_setup_stage2.sh"
    workspace = tmp_path / "persistent-workspace"
    assert os.access(script, os.X_OK)
    for _attempt in range(2):
        result = subprocess.run(
            ["bash", str(script), "--dry-run", "--workspace-root", str(workspace)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "no writes" in result.stdout
        assert stage2_job.DINO_MODEL_REVISION in result.stdout
        assert stage2_job.SAM_RUNTIME_REVISION in result.stdout
        assert "gpu-smoke.json" in result.stdout
    assert not workspace.exists()


def test_stage2_setup_rejects_missing_symlinked_and_root_workspace(stage2_job, tmp_path):
    repo = Path(stage2_job.__file__).resolve().parents[1]
    script = repo / "scripts" / "runpod_setup_stage2.sh"
    real_workspace = tmp_path / "real-workspace"
    real_workspace.mkdir()
    linked_workspace = tmp_path / "linked-workspace"
    linked_workspace.symlink_to(real_workspace, target_is_directory=True)
    root_alias = tmp_path.joinpath(*(Path("..") for _part in tmp_path.parts))
    invocations = (
        ["bash", str(script), "--workspace-root"],
        [
            "bash",
            str(script),
            "--dry-run",
            "--workspace-root",
            str(linked_workspace),
        ],
        ["bash", str(script), "--dry-run", "--workspace-root", str(root_alias)],
    )
    for command in invocations:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 2
        assert "FATAL" in result.stderr


def test_gpu_environment_and_setup_assets_are_pinned_together(stage2_job):
    repo = Path(stage2_job.__file__).resolve().parents[1]
    job_text = Path(stage2_job.__file__).read_text()
    setup_text = (repo / "scripts" / "runpod_setup_stage2.sh").read_text()
    for dependency in (
        '"torch==2.5.1"',
        '"torchvision==0.20.1"',
        '"transformers==4.49.0"',
        '"hydra-core==1.3.2"',
        '"iopath==0.1.10"',
    ):
        assert dependency in job_text
    for identity in (
        stage2_job.DINO_MODEL_REVISION,
        stage2_job.DINO_MODEL_WEIGHTS_SHA256,
        stage2_job.SAM_RUNTIME_REVISION,
        stage2_job.SAM_MODEL_WEIGHTS_SHA256,
        stage2_job.SAM_MODEL_CONFIG_SHA256,
    ):
        assert identity in setup_text
    assert "local_dir=sys.argv[2]" in setup_text
