from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest


def test_scaled_dilation_and_binary_mask(stage2_job):
    mask = bytearray(9 * 7)
    mask[3 * 9 + 4] = 1
    dilated = stage2_job.dilate_binary_mask(bytes(mask), width=9, height=7, radius=2)
    assert sum(dilated) == 25
    assert dilated[1 * 9 + 2] == 1
    assert dilated[5 * 9 + 6] == 1
    assert stage2_job.scaled_render_dilation(1080, 8) == 8
    assert stage2_job.scaled_render_dilation(540, 8) == 4


def test_yuv_fill_changes_only_masked_luma_and_covered_chroma(stage2_job):
    width, height = 8, 6
    luma_size = width * height
    chroma_size = luma_size // 4
    frame = bytes([20] * luma_size + [30] * chroma_size + [40] * chroma_size)
    mask = bytearray(luma_size)
    mask[2 * width + 3] = 1
    output = stage2_job.apply_mask_to_yuv420p(
        frame,
        bytes(mask),
        width=width,
        height=height,
        fill_yuv=(128, 129, 130),
    )
    assert output[2 * width + 3] == 128
    assert output[2 * width + 2] == 20
    chroma_index = (2 // 2) * (width // 2) + (3 // 2)
    assert output[luma_size + chroma_index] == 129
    assert output[luma_size + chroma_size + chroma_index] == 130
    assert output[luma_size] == 30


def _make_video(stage2_job, tmp_path: Path):
    ffmpeg = stage2_job.shutil.which("ffmpeg")
    ffprobe = stage2_job.shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for render round-trip tests")
    video = tmp_path / "stage1.mp4"
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-metadata",
            "title=private source title",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        pytest.skip(f"test ffmpeg lacks required encoders: {result.stderr}")
    facts = stage2_job.probe_video(video)
    video_ref = stage2_job.artifact_ref(video)
    dummy = tmp_path / "manifest.json"
    dummy.write_text("{}")
    stage1 = stage2_job.StageIInput(
        schema_version=1,
        clip_id="clip",
        source=video_ref,
        stage1_video=video_ref,
        stage1_manifest=stage2_job.artifact_ref(dummy),
        source_video=facts,
        stage1_output_video=facts,
        stage1_status="PASS_AUTOMATED",
        stage1_audit_reasons=(),
        egoblur={},
        warnings=(),
    )
    return stage1


def _make_shard(stage2_job, tmp_path: Path, stage1, *, dino_sha256="2" * 64):
    facts = stage1.stage1_output_video
    prompt = stage2_job.SamPrompt(
        prompt_id="face-1",
        prompt_kind="dino",
        anchor_frame=0,
        box=(20.0, 12.0, 36.0, 28.0),
        source_id="proposal-1",
    )
    masks = tuple(
        sorted(
            (
                stage2_job.rectangle_packed_mask(
                    frame_idx=frame_idx,
                    prompt=prompt,
                    crop_box=(20, 12, 36, 28),
                    source="dino-fallback",
                )
                for frame_idx in range(facts.n_frames)
            ),
            key=lambda mask: mask.key,
        )
    )
    identity = stage2_job.FakeSamAdapter.identity
    runtime = stage2_job.FakeSamAdapter.runtime_identity
    meta = stage2_job.SamMaskShardMeta(
        schema_version=1,
        artifact_type="sam_mask_shard",
        fingerprint="1" * 64,
        dino_artifact_sha256=dino_sha256,
        dino_fingerprint="3" * 64,
        model=identity,
        runtime=runtime,
        accepted_proposal_threshold=0.2,
        frame_start=0,
        frame_end=facts.n_frames - 1,
        frame_width=facts.display_width,
        frame_height=facts.display_height,
        window={"index": 0, "size": facts.n_frames, "overlap": 0},
        precision="float32",
        prompts=(),
        masks=tuple(stage2_job._mask_record(mask) for mask in masks),
        review_flags=(),
        metrics={},
    )
    path = tmp_path / "sam" / "window.npz"
    return stage2_job.write_immutable_bytes(path, stage2_job.encode_sam_mask_shard(meta, masks))


def test_render_fingerprint_binds_every_result_affecting_setting(stage2_job, tmp_path):
    stage1 = _make_video(stage2_job, tmp_path)
    shard = _make_shard(stage2_job, tmp_path, stage1)
    original = stage2_job.render_fingerprint_payload(
        stage1=stage1,
        sam_shards=(shard,),
        config=stage2_job.RenderConfig(),
    )
    first = stage2_job.render_fingerprint(original)
    for change in (
        {"dilation_pixels_at_1080p": 9},
        {"fill_yuv": (120, 128, 128)},
        {"crf": 20},
        {"preset": "fast"},
        {"fill_tolerance": 30},
        {"max_outside_mask_mae": 10.0},
    ):
        changed = stage2_job.render_fingerprint_payload(
            stage1=stage1,
            sam_shards=(shard,),
            config=replace(stage2_job.RenderConfig(), **change),
        )
        assert stage2_job.render_fingerprint(changed) != first


def test_real_render_is_verified_promoted_and_reusable(stage2_job, tmp_path):
    stage1 = _make_video(stage2_job, tmp_path)
    shard = _make_shard(stage2_job, tmp_path, stage1)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    result = stage2_job.render_stage2_video(
        stage1=stage1,
        paths=paths,
        sam_shards=(shard,),
        config=stage2_job.RenderConfig(dilation_pixels_at_1080p=0),
    )
    assert result.reused is False
    assert result.meta.verification["passed"] is True
    assert result.meta.verification["fill_integrity_violations"] == 0
    assert result.meta.verification["stream_types"] == ("video",)
    assert result.meta.verification["forbidden_metadata_tags"] == []
    assert result.meta.encoder["pixel_source"] == "stage1_video_only"
    assert Path(result.meta.output.path).exists()
    second = stage2_job.render_stage2_video(
        stage1=stage1,
        paths=paths,
        sam_shards=(shard,),
        config=stage2_job.RenderConfig(dilation_pixels_at_1080p=0),
    )
    assert second.reused is True
    assert second.artifact == result.artifact


def test_render_stage2_video_honors_a_stop_request(stage2_job, tmp_path):
    stage1 = _make_video(stage2_job, tmp_path)
    shard = _make_shard(stage2_job, tmp_path, stage1)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    stage2_job.transition_state(
        paths.state,
        run_id="run",
        clip_id=stage1.clip_id,
        mode="production",
        target="SAM_COMPLETE",
        completed_layers=("dino", "sam"),
        reusable_layers=("dino", "sam"),
    )
    stage2_job.request_stop(paths)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.render_stage2_video(
            stage1=stage1,
            paths=paths,
            sam_shards=(shard,),
            config=stage2_job.RenderConfig(dilation_pixels_at_1080p=0),
        )
    assert caught.value.code == "STOP_REQUESTED"
    assert not paths.render.exists()


def test_failed_verification_never_promotes_output(stage2_job, tmp_path, monkeypatch):
    stage1 = _make_video(stage2_job, tmp_path)
    shard = _make_shard(stage2_job, tmp_path, stage1)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)

    def reject(**_kwargs):
        raise stage2_job.Stage2Error("SIMULATED_VERIFY_FAILURE", "failed")

    promoted = False

    def forbidden_promotion(*_args):
        nonlocal promoted
        promoted = True
        raise AssertionError("unverified output must never reach promotion")

    monkeypatch.setattr(stage2_job, "verify_rendered_video", reject)
    monkeypatch.setattr(stage2_job, "_promote_render_output", forbidden_promotion)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.render_stage2_video(
            stage1=stage1,
            paths=paths,
            sam_shards=(shard,),
            config=stage2_job.RenderConfig(dilation_pixels_at_1080p=0),
        )
    assert caught.value.code == "SIMULATED_VERIFY_FAILURE"
    assert promoted is False
    assert not paths.render.exists()
    assert not list(paths.render.parent.glob("stage2-*.mp4"))
    assert not list(paths.render.parent.glob("*.tmp.mp4"))


def test_only_verified_render_can_advance_processing_complete(stage2_job, tmp_path):
    stage1 = _make_video(stage2_job, tmp_path)
    dino_path = tmp_path / "dino.json"
    dino_path.write_text('{"artifact_type":"dino"}')
    dino = stage2_job.artifact_ref(dino_path)
    shard = _make_shard(stage2_job, tmp_path, stage1, dino_sha256=dino.sha256)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    stage2_job.transition_state(
        paths.state,
        run_id="run",
        clip_id=stage1.clip_id,
        mode="production",
        target="SAM_COMPLETE",
        completed_layers=("dino", "sam"),
        reusable_layers=("dino", "sam"),
    )
    render = stage2_job.render_stage2_video(
        stage1=stage1,
        paths=paths,
        sam_shards=(shard,),
        config=stage2_job.RenderConfig(dilation_pixels_at_1080p=0),
    )
    manifest = stage2_job.finalize_verified_processing(
        stage1=stage1,
        paths=paths,
        run_id="run",
        dino_artifact=dino,
        sam_shards=(shard,),
        render_result=render,
    )
    assert manifest.processing_state == "PROCESSING_COMPLETE"
    assert manifest.review_status == "PENDING"
    assert manifest.audit_status == "PASS_AUTOMATED_TECHNICAL"
    state = stage2_job.load_state(paths.state)
    assert state.state == "PROCESSING_COMPLETE"


def test_finalize_carries_forward_a_skipped_stage1_yunet_check(stage2_job, tmp_path):
    stage1 = replace(_make_video(stage2_job, tmp_path), stage1_status="PASS_AUTOMATED_NO_YUNET")
    dino_path = tmp_path / "dino.json"
    dino_path.write_text('{"artifact_type":"dino"}')
    dino = stage2_job.artifact_ref(dino_path)
    shard = _make_shard(stage2_job, tmp_path, stage1, dino_sha256=dino.sha256)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    stage2_job.transition_state(
        paths.state,
        run_id="run",
        clip_id=stage1.clip_id,
        mode="production",
        target="SAM_COMPLETE",
        completed_layers=("dino", "sam"),
        reusable_layers=("dino", "sam"),
    )
    render = stage2_job.render_stage2_video(
        stage1=stage1,
        paths=paths,
        sam_shards=(shard,),
        config=stage2_job.RenderConfig(dilation_pixels_at_1080p=0),
    )
    manifest = stage2_job.finalize_verified_processing(
        stage1=stage1,
        paths=paths,
        run_id="run",
        dino_artifact=dino,
        sam_shards=(shard,),
        render_result=render,
    )
    # Must not read the same as a clip whose Stage I redaction got the full
    # independent check (test_only_verified_render_can_advance_processing_complete
    # asserts plain PASS_AUTOMATED_TECHNICAL for that case).
    assert manifest.audit_status == "PASS_AUTOMATED_TECHNICAL_NO_YUNET"


def test_renderer_uses_stage1_and_never_decodes_original(stage2_job, tmp_path, monkeypatch):
    stage1 = _make_video(stage2_job, tmp_path)
    original = tmp_path / "private-original.bin"
    original.write_bytes(b"not the Stage I video")
    stage1 = replace(stage1, source=stage2_job.artifact_ref(original))
    shard = _make_shard(stage2_job, tmp_path, stage1)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    decoded_paths = []
    real_decoder = stage2_job._rawvideo_decoder

    def recording_decoder(path, **kwargs):
        decoded_paths.append(path.resolve())
        return real_decoder(path, **kwargs)

    monkeypatch.setattr(stage2_job, "_rawvideo_decoder", recording_decoder)
    stage2_job.render_stage2_video(
        stage1=stage1,
        paths=paths,
        sam_shards=(shard,),
        config=stage2_job.RenderConfig(dilation_pixels_at_1080p=0),
    )
    assert original.resolve() not in decoded_paths
    assert Path(stage1.stage1_video.path).resolve() in decoded_paths


def test_stale_config_and_changed_output_cannot_be_reused(stage2_job, tmp_path):
    stage1 = _make_video(stage2_job, tmp_path)
    shard = _make_shard(stage2_job, tmp_path, stage1)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    config = stage2_job.RenderConfig(dilation_pixels_at_1080p=0)
    first = stage2_job.render_stage2_video(
        stage1=stage1, paths=paths, sam_shards=(shard,), config=config
    )
    with pytest.raises(stage2_job.Stage2Error) as stale:
        stage2_job.render_stage2_video(
            stage1=stage1,
            paths=paths,
            sam_shards=(shard,),
            config=replace(config, crf=20),
        )
    assert stale.value.code == "STALE_RENDER_ARTIFACT"
    Path(first.meta.output.path).write_bytes(Path(first.meta.output.path).read_bytes() + b"x")
    with pytest.raises(stage2_job.Stage2Error) as changed:
        stage2_job.render_stage2_video(
            stage1=stage1, paths=paths, sam_shards=(shard,), config=config
        )
    assert changed.value.code == "INVALID_RENDER_ARTIFACT"
