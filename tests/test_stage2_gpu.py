"""Milestone 6 local guards for real-GPU frame payloads and calibration."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_source_video(stage2_job, tmp_path: Path):
    ffmpeg = stage2_job.shutil.which("ffmpeg")
    ffprobe = stage2_job.shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for frame-attestation tests")
    source = tmp_path / "source.mp4"
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        pytest.skip(f"test ffmpeg lacks libx264: {result.stderr}")
    manifest = tmp_path / "stage1.json"
    manifest.write_text("{}")
    facts = stage2_job.probe_video(source)
    return stage2_job.StageIInput(
        schema_version=1,
        clip_id="clip",
        source=stage2_job.artifact_ref(source),
        stage1_video=stage2_job.artifact_ref(source),
        stage1_manifest=stage2_job.artifact_ref(manifest),
        source_video=facts,
        stage1_output_video=facts,
        stage1_status="PASS_AUTOMATED",
        stage1_audit_reasons=(),
        egoblur={},
        warnings=(),
    )


def test_exact_sam_window_is_atomically_extracted_attested_and_reused(stage2_job, tmp_path):
    stage1 = _make_source_video(stage2_job, tmp_path)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    window = stage2_job.TemporalWindow(index=0, frame_start=2, frame_end=5)

    first = stage2_job.extract_attested_sam_window(stage1=stage1, paths=paths, window=window)
    frame_dir = Path(first.payload)
    assert sorted(path.name for path in frame_dir.iterdir()) == [
        "000000.jpg",
        "000001.jpg",
        "000002.jpg",
        "000003.jpg",
        stage2_job.SAM_FRAME_ATTESTATION_FILENAME,
    ]
    assert len(first.frame_artifacts) == window.n_frames
    assert re.fullmatch(r"[0-9a-f]{64}", first.frame_payload_sha256 or "")
    raw = stage2_job.read_json(Path(first.attestation_artifact.path))
    assert [record["global_index"] for record in raw["frames"]] == [2, 3, 4, 5]
    assert raw["source_sha256"] == stage1.source.sha256
    assert stage2_job.validate_sam_window_input(first, stage1=stage1, window=window) == first

    second = stage2_job.extract_attested_sam_window(stage1=stage1, paths=paths, window=window)
    assert second == first


def test_extract_attested_sam_window_cache_hit_does_not_need_the_source_video(
    stage2_job, tmp_path
):
    stage1 = _make_source_video(stage2_job, tmp_path)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    window = stage2_job.TemporalWindow(index=0, frame_start=1, frame_end=3)

    first = stage2_job.extract_attested_sam_window(stage1=stage1, paths=paths, window=window)

    # Simulate the source video being archived off the pod once every shard
    # that needs it has already been cached.
    Path(stage1.source.path).unlink()

    reused = stage2_job.extract_attested_sam_window(stage1=stage1, paths=paths, window=window)
    assert reused == first


def test_real_sam_window_loader_matches_generate_sam_mask_shards_contract(stage2_job, tmp_path):
    """extract_attested_sam_window can't be window_loader=... directly (keyword-only args
    vs generate_sam_mask_shards calling window_loader(window) positionally) - real_sam_window_loader
    is the binding that closes the gap."""
    stage1 = _make_source_video(stage2_job, tmp_path)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    window = stage2_job.TemporalWindow(index=0, frame_start=1, frame_end=3)

    with pytest.raises(TypeError):
        stage2_job.extract_attested_sam_window(window)

    loader = stage2_job.real_sam_window_loader(stage1=stage1, paths=paths)
    direct = stage2_job.extract_attested_sam_window(stage1=stage1, paths=paths, window=window)
    assert loader(window) == direct


def test_real_dino_frame_loader_matches_the_exact_requested_anchor_frame(stage2_job, tmp_path):
    stage1 = _make_source_video(stage2_job, tmp_path)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    loader = stage2_job.real_dino_frame_loader(stage1=stage1, paths=paths)

    from PIL import Image

    frame_idx = 2
    window = stage2_job.TemporalWindow(index=0, frame_start=frame_idx, frame_end=frame_idx)
    direct = stage2_job.extract_attested_sam_window(stage1=stage1, paths=paths, window=window)
    with Image.open(Path(direct.payload) / "000000.jpg") as expected:
        expected.load()
        expected_bytes = expected.tobytes()
        expected_size = expected.size

    image = loader(frame_idx)
    assert image.size == expected_size == (
        stage1.source_video.display_width,
        stage1.source_video.display_height,
    )
    assert image.tobytes() == expected_bytes
    # The actual contract generate_dino_proposals' tiling code needs
    # (frame_views/_image_size) - a PIL-compatible crop(), not just a size.
    cropped = image.crop((0, 0, 8, 8))
    assert cropped.size == (8, 8)


def test_real_dino_frame_loader_cache_hit_does_not_need_the_source_video(stage2_job, tmp_path):
    stage1 = _make_source_video(stage2_job, tmp_path)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    loader = stage2_job.real_dino_frame_loader(stage1=stage1, paths=paths)

    first = loader(2)
    first_bytes = first.tobytes()

    # Simulate the source video being archived off the pod once every anchor
    # frame that needs it has already been cached - same guarantee
    # extract_attested_sam_window itself already provides for SAM windows.
    Path(stage1.source.path).unlink()

    reused = loader(2)
    assert reused.tobytes() == first_bytes


def test_changed_or_extra_sam_frame_payload_fails_closed(stage2_job, tmp_path):
    stage1 = _make_source_video(stage2_job, tmp_path)
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    window = stage2_job.TemporalWindow(index=0, frame_start=1, frame_end=3)
    extracted = stage2_job.extract_attested_sam_window(stage1=stage1, paths=paths, window=window)
    frame_dir = Path(extracted.payload)
    (frame_dir / "000001.jpg").write_bytes(b"tampered")
    with pytest.raises(stage2_job.Stage2Error) as changed:
        stage2_job.validate_sam_window_input(extracted, stage1=stage1, window=window)
    assert changed.value.code == "INVALID_FRAME_PAYLOAD"

    (frame_dir / "unexpected.jpg").write_bytes(b"extra")
    with pytest.raises(stage2_job.Stage2Error) as extra:
        stage2_job.load_attested_sam_window(
            frame_dir,
            source_sha256=stage1.source.sha256,
            window=window,
            width=stage1.source_video.display_width,
            height=stage1.source_video.display_height,
        )
    assert extra.value.code == "INVALID_FRAME_PAYLOAD"


def test_frame_payload_digest_changes_sam_fingerprint(stage2_job, tmp_path):
    dino_path = tmp_path / "dino.json"
    dino_path.write_text("{}")
    dino_ref = stage2_job.artifact_ref(dino_path)
    dino_meta = stage2_job.DinoArtifactMeta(
        schema_version=1,
        artifact_type="dino_proposals",
        fingerprint="1" * 64,
        source_sha256="2" * 64,
        model=stage2_job.TransformersGroundingDinoAdapter.identity,
        prompt="face.",
        proposal_floor=0.10,
        text_threshold=0.25,
        anchor_spacing=20,
        anchor_frames=(0,),
        tiling={"rows": 2, "cols": 2, "overlap": 0.2},
        nms_iou=0.7,
        preprocessing={},
        proposals=(),
        metrics={},
    )
    config = stage2_job.SamGenerationConfig(
        model=stage2_job.FakeSamAdapter.identity,
        runtime=stage2_job.FakeSamAdapter.runtime_identity,
        accepted_proposal_threshold=0.2,
        window_size=4,
        window_overlap=1,
    )
    window = stage2_job.TemporalWindow(index=0, frame_start=0, frame_end=3)
    first = stage2_job.sam_fingerprint(
        stage2_job.sam_window_fingerprint_payload(
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=config,
            window=window,
            prompts=(),
            frame_payload_sha256="3" * 64,
            frame_payload_status="verified",
        )
    )
    second = stage2_job.sam_fingerprint(
        stage2_job.sam_window_fingerprint_payload(
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=config,
            window=window,
            prompts=(),
            frame_payload_sha256="4" * 64,
            frame_payload_status="verified",
        )
    )
    assert first != second


def test_sam_recompute_removes_attested_frame_payloads(stage2_job, tmp_path):
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", "clip")
    stage2_job.transition_state(
        paths.state,
        run_id="run",
        clip_id="clip",
        mode="production",
        target="SAM_COMPLETE",
        completed_layers=("dino", "sam"),
        reusable_layers=("dino", "sam"),
    )
    marker = paths.frames_dir / "source" / "window" / "frame.jpg"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"private source frame")
    state = stage2_job.invalidate_from(paths, layer="sam")
    assert state.state == "DINO_COMPLETE"
    assert not paths.frames_dir.exists()


class _FakeCuda:
    def __init__(self, *, capability=(8, 0)):
        self.capability = capability
        self.events = []

    def is_available(self):
        return True

    def get_device_properties(self, _index):
        return SimpleNamespace(total_memory=24 * 1024**3)

    def get_device_capability(self, _index):
        return self.capability

    def get_device_name(self, _index):
        return "Fake CUDA GPU"

    def empty_cache(self):
        self.events.append("empty-cache")

    def reset_peak_memory_stats(self):
        self.events.append("reset-peak")

    def synchronize(self):
        self.events.append("synchronize")

    def max_memory_allocated(self):
        return 2 * 1024**3

    def memory_allocated(self):
        return 128 * 1024**2


class _FakeDino:
    def __init__(self, events):
        self.events = events

    def infer_batch(self, images, *, prompt, box_threshold, text_threshold):
        assert len(images) == 1
        assert prompt
        assert box_threshold > 0
        assert text_threshold > 0
        self.events.append("dino-infer")
        return [()]


class _FakeSam:
    def __init__(self, stage2_job, events):
        self.stage2_job = stage2_job
        self.events = events

    def propagate_window(self, *, window_input, window, prompts, width, height, precision):
        assert sorted(path.name for path in Path(window_input).glob("*.jpg")) == [
            f"{index:06d}.jpg" for index in range(5)
        ]
        assert window.frame_start == 0
        assert len(prompts) == 1
        assert (width, height) == (128, 96)
        self.events.append(f"sam-infer:{precision}")
        return self.stage2_job.SamPropagationResult(
            masks=(
                self.stage2_job.RawSamMask(
                    frame_idx=2,
                    prompt_id=prompts[0].prompt_id,
                    crop_box=(40, 24, 88, 72),
                    pixels=b"\x01" * (48 * 48),
                    direction="anchor",
                ),
            ),
            forward_complete=True,
            reverse_complete=True,
        )


def _make_real_stage1_inputs(stage2_job, tmp_path: Path):
    """A real, ffmpeg-encoded, validate_stage1-passing Stage I input triple.

    Distinct from _make_source_video (which builds a StageIInput directly,
    bypassing validate_stage1's manifest checks) - run_real_pipeline needs
    real files and a real manifest, since it calls validate_stage1 itself
    and its real DINO/SAM loaders actually ffmpeg-extract frames from them.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the real-pipeline orchestration test")
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.mp4"
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        pytest.skip(f"test ffmpeg lacks libx264: {result.stderr}")
    # validate_stage1 only requires three distinct PATHS, not distinct
    # content - a plain copy is enough and far simpler than re-encoding.
    stage1_video = tmp_path / "stage1.mp4"
    shutil.copyfile(source, stage1_video)

    facts = stage2_job.probe_video(source)
    source_ref = stage2_job.artifact_ref(source)
    stage1_ref = stage2_job.artifact_ref(stage1_video)
    manifest_path = tmp_path / "stage1.manifest.json"
    manifest = {
        "schema_version": 1,
        "clip_id": "clip",
        "source": {
            "sha256": source_ref.sha256,
            "filename": source.name,
            "width": facts.display_width,
            "height": facts.display_height,
            "fps": facts.fps,
            "n_frames": facts.n_frames,
            "duration_s": facts.duration_s,
            "rotation": facts.rotation,
        },
        "output": {
            "path": stage1_video.name,
            "sha256": stage1_ref.sha256,
            "bytes": stage1_ref.bytes,
        },
        "egoblur": dict(stage2_job.EXPECTED_STAGE1),
        "audit": {
            "status": "PASS_AUTOMATED",
            "status_reasons": [],
            "integrity_ran": True,
            "fill_integrity_checked": facts.n_frames,
            "fill_integrity_violations": 0,
            "fill_integrity_frames": facts.n_frames,
        },
        "status": "PASS_AUTOMATED",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, stage1_video, manifest_path


def test_run_real_pipeline_sequences_layers_with_fake_adapters_and_real_render(
    stage2_job, tmp_path
):
    """The orchestration itself - sequencing, state transitions, free-before-
    next-load, real extraction/render - is fully testable with only the two
    model inference calls faked. Only real weight loading and actual
    detection/segmentation quality require the pod."""
    source, stage1_video, manifest_path = _make_real_stage1_inputs(stage2_job, tmp_path / "inputs")
    work_dir = tmp_path / "work"
    cuda = _FakeCuda(capability=(8, 0))

    manifest = stage2_job.run_real_pipeline(
        source_video=source,
        stage1_video=stage1_video,
        stage1_manifest=manifest_path,
        work_dir=work_dir,
        run_id="pilot-test",
        workspace_root=tmp_path / "workspace",
        window_size=4,
        window_overlap=1,
        expected_clip_id="clip",
        torch_module=SimpleNamespace(cuda=cuda),
        dino_factory=stage2_job.FakeDinoAdapter,
        sam_factory=stage2_job.FakeSamAdapter,
    )

    assert manifest.processing_state == "PROCESSING_COMPLETE"
    assert manifest.mode == "production"
    assert manifest.review_status == "PENDING"
    paths = stage2_job.build_run_paths(work_dir, "pilot-test", "clip")
    assert stage2_job.load_state(paths.state).state == "PROCESSING_COMPLETE"
    assert Path(manifest.render_artifact.path).exists()
    # DINO must be freed before SAM loads (never co-resident): two separate
    # empty-cache/synchronize rounds, not one shared round.
    assert cuda.events.count("empty-cache") == 2
    assert cuda.events.count("synchronize") == 2


def test_run_real_pipeline_rejects_a_clip_id_mismatch(stage2_job, tmp_path):
    source, stage1_video, manifest_path = _make_real_stage1_inputs(stage2_job, tmp_path / "inputs")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.run_real_pipeline(
            source_video=source,
            stage1_video=stage1_video,
            stage1_manifest=manifest_path,
            work_dir=tmp_path / "work",
            run_id="pilot-test",
            workspace_root=tmp_path / "workspace",
            window_size=4,
            window_overlap=1,
            expected_clip_id="not-the-real-clip-id",
            torch_module=SimpleNamespace(cuda=_FakeCuda()),
            dino_factory=stage2_job.FakeDinoAdapter,
            sam_factory=stage2_job.FakeSamAdapter,
        )
    assert caught.value.code == "CLIP_ID_MISMATCH"


@pytest.mark.parametrize(
    ("capability", "expected_precision"),
    [((8, 0), "bfloat16"), ((7, 5), "float16")],
)
def test_offline_gpu_smoke_loads_models_sequentially_and_records_metrics(
    stage2_job, tmp_path, capability, expected_precision, monkeypatch
):
    events = []
    cuda = _FakeCuda(capability=capability)
    torch_module = SimpleNamespace(
        __version__="2.5.1",
        version=SimpleNamespace(cuda="12.4"),
        cuda=cuda,
    )

    def dino_factory():
        events.append("dino-load")
        return _FakeDino(events)

    def sam_factory(**kwargs):
        assert kwargs == {
            "checkpoint_path": stage2_job.stage2_asset_paths(tmp_path)["sam_checkpoint"],
            "device": "cuda",
        }
        events.append("sam-load")
        return _FakeSam(stage2_job, events)

    # setenv (not delenv(raising=False)) so monkeypatch actually registers an
    # undo entry for these two: delenv on an already-absent key is a no-op
    # that leaves nothing to restore, so run_offline_gpu_smoke's own direct
    # os.environ[...] = "1" assignment below would otherwise leak past this
    # test into every later test in the same pytest session.
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    result = stage2_job.run_offline_gpu_smoke(
        tmp_path,
        doctor_fn=lambda _root: {"status": "READY_FOR_MILESTONE_6_GPU_SMOKE"},
        torch_module=torch_module,
        dino_factory=dino_factory,
        sam_factory=sam_factory,
    )

    assert result["status"] == "PASS_OFFLINE_GPU_SMOKE"
    assert result["models_loaded_sequentially"] is True
    assert result["dino"]["inference_completed"] is True
    assert result["sam"]["precision"] == expected_precision
    assert result["sam"]["anchor_mask_completed"] is True
    assert result["dino_residual_vram_before_sam_bytes"] == 128 * 1024**2
    assert events == [
        "dino-load",
        "dino-infer",
        "sam-load",
        f"sam-infer:{expected_precision}",
    ]
    assert stage2_job.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert stage2_job.os.environ["HF_HUB_OFFLINE"] == "1"


def test_monkeypatch_setenv_survives_a_raw_os_environ_overwrite_on_undo(stage2_job):
    """Guards the fix above: monkeypatch.setenv (not delenv(raising=False))
    is required before calling run_offline_gpu_smoke, because delenv on an
    already-absent key registers nothing to undo. Without this, the direct
    os.environ[...] = "1" assignment run_offline_gpu_smoke performs itself
    would leak past this test's teardown into every later test in the same
    pytest session."""
    name = "EGOANNOTE_STAGE2_TEST_ONLY_ENV_LEAK_PROBE"
    assert name not in stage2_job.os.environ
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv(name, "0")
        # Simulate run_offline_gpu_smoke's own direct assignment, which
        # bypasses monkeypatch entirely - this is exactly what a naive
        # delenv(raising=False) fails to track.
        stage2_job.os.environ[name] = "1"
        assert stage2_job.os.environ[name] == "1"
    finally:
        mp.undo()
    assert name not in stage2_job.os.environ
