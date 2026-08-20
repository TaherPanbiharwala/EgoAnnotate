"""Milestone 3 tests for bounded SAM2 propagation and immutable mask shards."""

from __future__ import annotations

import builtins
import contextlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _stage1(stage2_job, *, n_frames=20, width=100, height=80):
    facts = stage2_job.VideoFacts(
        coded_width=width,
        coded_height=height,
        display_width=width,
        display_height=height,
        fps=30.0,
        n_frames=n_frames,
        duration_s=n_frames / 30.0,
        rotation=0,
        is_cfr=True,
    )
    return stage2_job.StageIInput(
        schema_version=1,
        clip_id="clip",
        source=stage2_job.ArtifactRef("/original.mp4", "1" * 64, 100),
        stage1_video=stage2_job.ArtifactRef("/blurred.mp4", "2" * 64, 90),
        stage1_manifest=stage2_job.ArtifactRef("/stage1.json", "3" * 64, 80),
        source_video=facts,
        stage1_output_video=facts,
        stage1_status="NEEDS_REVIEW",
        stage1_audit_reasons=(),
        egoblur={},
        warnings=(),
    )


def _proposal(stage2_job, *, frame_idx, box=(40.0, 30.0, 60.0, 50.0), score=0.5):
    return stage2_job._finalize_proposal(
        stage2_job.Proposal(
            frame_idx=frame_idx,
            box=box,
            score=score,
            source="full-frame-only",
            label="face",
            origins=("full-frame",),
        )
    )


def _dino_artifact(stage2_job, tmp_path, stage1, proposals=()):
    source_counts = {
        source: sum(1 for proposal in proposals if proposal.source == source)
        for source in ("full-frame-only", "tiled-only", "shared")
    }
    anchors = tuple(sorted({0, stage1.source_video.n_frames - 1, *(p.frame_idx for p in proposals)}))
    meta = stage2_job.DinoArtifactMeta(
        schema_version=1,
        artifact_type="dino_proposals",
        fingerprint="d" * 64,
        source_sha256=stage1.source.sha256,
        model=stage2_job.ModelIdentity("fake-dino", "one", "4" * 64),
        prompt="face.",
        proposal_floor=0.10,
        text_threshold=0.25,
        anchor_spacing=20,
        anchor_frames=anchors,
        tiling={"rows": 2, "cols": 2, "overlap": 0.2},
        nms_iou=0.7,
        preprocessing={"color": "RGB"},
        proposals=tuple(proposals),
        metrics={
            "n_anchors": len(anchors),
            "n_proposals": len(proposals),
            "source_counts": source_counts,
        },
    )
    path = tmp_path / "dino-artifact.json"
    ref = stage2_job.write_immutable_json(path, meta)
    return ref, meta


def _config(stage2_job, adapter, **changes):
    values = dict(
        model=adapter.identity,
        runtime=adapter.runtime_identity,
        accepted_proposal_threshold=0.20,
        window_size=10,
        window_overlap=2,
        precision="float32",
    )
    values.update(changes)
    return stage2_job.SamGenerationConfig(**values)


def _window_input(stage2_job, stage1, window, payload=None):
    return stage2_job.SamWindowInput(
        source_sha256=stage1.source.sha256,
        frame_start=window.frame_start,
        frame_end=window.frame_end,
        frame_width=stage1.source_video.display_width,
        frame_height=stage1.source_video.display_height,
        payload=payload,
    )


def _generate(
    stage2_job,
    tmp_path,
    *,
    stage1=None,
    proposals=(),
    adapter=None,
    config=None,
    manual_seeds=(),
):
    stage1 = stage1 or _stage1(stage2_job)
    adapter = adapter or stage2_job.FakeSamAdapter()
    config = config or _config(stage2_job, adapter)
    dino_ref, dino_meta = _dino_artifact(
        stage2_job, tmp_path / "inputs", stage1, proposals
    )
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    result = stage2_job.generate_sam_mask_shards(
        stage1=stage1,
        paths=paths,
        dino_artifact=dino_ref,
        dino_meta=dino_meta,
        config=config,
        adapter=adapter,
        window_loader=lambda window: _window_input(stage2_job, stage1, window),
        manual_seeds=manual_seeds,
    )
    return result, paths, dino_ref, dino_meta, adapter, config


@pytest.mark.parametrize(
    ("n_frames", "size", "overlap", "expected"),
    [
        (1, 10, 2, [(0, 0)]),
        (20, 10, 2, [(0, 9), (8, 17), (16, 19)]),
        (21, 10, 0, [(0, 9), (10, 19), (20, 20)]),
    ],
)
def test_temporal_windows_cover_edges_with_exact_overlap(
    stage2_job, n_frames, size, overlap, expected
):
    windows = stage2_job.temporal_windows(n_frames, size, overlap)
    assert [(window.frame_start, window.frame_end) for window in windows] == expected
    assert [window.index for window in windows] == list(range(len(windows)))
    assert max(
        sum(window.frame_start <= frame_idx <= window.frame_end for window in windows)
        for frame_idx in range(n_frames)
    ) <= 2


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (10, 6)])
def test_invalid_window_layout_fails_closed(stage2_job, size, overlap):
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.temporal_windows(20, size, overlap)
    assert caught.value.code == "INVALID_SAM_WINDOW_LAYOUT"


def test_padded_fallback_box_is_conservative_and_clamped(stage2_job):
    assert stage2_job.padded_box(
        (1.0, 2.0, 11.0, 12.0),
        width=100,
        height=80,
        scale=0.20,
        min_padding_px=4,
    ) == (0, 0, 15, 16)


def test_fake_adapter_propagates_forward_and_reverse_and_keeps_anchor_fallback(
    stage2_job, tmp_path
):
    proposal = _proposal(stage2_job, frame_idx=5)
    result, *_ = _generate(
        stage2_job,
        tmp_path,
        stage1=_stage1(stage2_job, n_frames=10),
        proposals=(proposal,),
    )
    loaded = stage2_job.load_sam_mask_shard(Path(result.shards[0].path))
    sources = {(mask.frame_idx, mask.source) for mask in loaded.masks}
    assert (4, "sam2-reverse") in sources
    assert (5, "sam2-anchor") in sources
    assert (6, "sam2-forward") in sources
    assert (5, "dino-fallback") in sources
    assert [prompt["local_object_id"] for prompt in loaded.meta.prompts] == [1]
    assert loaded.meta.runtime == stage2_job.FakeSamAdapter.runtime_identity
    assert result.review_flags == ()


def test_complete_shards_reuse_without_model_or_frame_loader_calls(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    proposal = _proposal(stage2_job, frame_idx=5)
    first_adapter = stage2_job.FakeSamAdapter()
    first, paths, dino_ref, dino_meta, _, config = _generate(
        stage2_job,
        tmp_path,
        stage1=stage1,
        proposals=(proposal,),
        adapter=first_adapter,
    )
    assert first.generated_window_count == 3
    assert first_adapter.calls == 1
    second_adapter = stage2_job.FakeSamAdapter()
    loader_called = False

    def forbidden_loader(_window):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("reused shards must not decode frames")

    second = stage2_job.generate_sam_mask_shards(
        stage1=stage1,
        paths=paths,
        dino_artifact=dino_ref,
        dino_meta=dino_meta,
        config=config,
        adapter=second_adapter,
        window_loader=forbidden_loader,
    )
    assert second.reused_window_count == 3
    assert second.generated_window_count == 0
    assert second.shards == first.shards
    assert second_adapter.calls == 0
    assert loader_called is False


def test_no_prompt_windows_are_persisted_without_running_sam(stage2_job, tmp_path):
    adapter = stage2_job.FakeSamAdapter()
    result, *_ = _generate(stage2_job, tmp_path, adapter=adapter)
    assert len(result.shards) == 3
    assert adapter.calls == 0
    loaded = tuple(
        stage2_job.load_sam_mask_shard(Path(ref.path)) for ref in result.shards
    )
    assert stage2_job.union_sam_masks_for_frame(
        loaded, frame_idx=12, width=100, height=80
    ) == bytes(100 * 80)


def test_wrong_window_source_is_never_sent_to_sam(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, n_frames=10)
    proposal = _proposal(stage2_job, frame_idx=5)
    dino_ref, dino_meta = _dino_artifact(
        stage2_job, tmp_path / "inputs", stage1, (proposal,)
    )
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    adapter = stage2_job.FakeSamAdapter()
    result = stage2_job.generate_sam_mask_shards(
        stage1=stage1,
        paths=paths,
        dino_artifact=dino_ref,
        dino_meta=dino_meta,
        config=_config(stage2_job, adapter),
        adapter=adapter,
        window_loader=lambda window: replace(
            _window_input(stage2_job, stage1, window), source_sha256="0" * 64
        ),
    )
    assert adapter.calls == 0
    assert any(
        flag.startswith("INVALID_SAM_WINDOW_INPUT") for flag in result.review_flags
    )
    loaded = stage2_job.load_sam_mask_shard(Path(result.shards[0].path))
    assert {mask.frame_idx for mask in loaded.masks} == set(range(10))


def test_dino_artifact_from_another_source_is_rejected_before_sam(
    stage2_job, tmp_path
):
    stage1 = _stage1(stage2_job, n_frames=10)
    proposal = _proposal(stage2_job, frame_idx=5)
    dino_ref, dino_meta = _dino_artifact(
        stage2_job, tmp_path / "inputs", stage1, (proposal,)
    )
    mismatched_stage1 = replace(
        stage1,
        source=replace(stage1.source, sha256="9" * 64),
    )
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    adapter = stage2_job.FakeSamAdapter()
    loader_called = False

    def forbidden_loader(_window):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("source mismatch must fail before frame loading")

    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_sam_mask_shards(
            stage1=mismatched_stage1,
            paths=paths,
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=_config(stage2_job, adapter),
            adapter=adapter,
            window_loader=forbidden_loader,
        )
    assert caught.value.code == "DINO_SOURCE_MISMATCH"
    assert adapter.calls == 0
    assert loader_called is False


@pytest.mark.parametrize(
    ("raw", "changes", "reason"),
    [
        ({"crop_box": (40, 30, 60, 50), "pixels": bytes(400)}, {}, "SAM_MASK_EMPTY"),
        (
            {"crop_box": (40, 30, 41, 31), "pixels": b"\x01"},
            {},
            "SAM_MASK_UNDERSIZED",
        ),
        (
            {"crop_box": (0, 0, 10, 10), "pixels": b"\x01" * 100},
            {},
            "SAM_MASK_OFF_PROMPT",
        ),
        (
            {"crop_box": (10, 10, 70, 70), "pixels": b"\x01" * 3600},
            {"max_prompt_area_ratio": 4.0, "max_frame_area_ratio": 0.99},
            "SAM_MASK_EXPLODED",
        ),
        (
            {"crop_box": (0, 0, 100, 80), "pixels": b"\x01" * 8000},
            {"max_prompt_area_ratio": 1000.0},
            "SAM_MASK_NEAR_FULL_FRAME",
        ),
    ],
)
def test_pathological_sam_masks_are_rejected(stage2_job, raw, changes, reason):
    adapter = stage2_job.FakeSamAdapter()
    config = replace(_config(stage2_job, adapter), **changes)
    prompt = stage2_job.SamPrompt(
        "dino-p", "dino", 5, (40.0, 30.0, 60.0, 50.0), "p"
    )
    packed, rejection = stage2_job.assess_raw_sam_mask(
        stage2_job.RawSamMask(
            frame_idx=5,
            prompt_id=prompt.prompt_id,
            direction="anchor",
            **raw,
        ),
        prompt=prompt,
        config=config,
        width=100,
        height=80,
    )
    assert packed is None
    assert rejection == reason


class _FailingAdapter:
    def __init__(self, stage2_job):
        self.identity = stage2_job.FakeSamAdapter.identity
        self.runtime_identity = stage2_job.FakeSamAdapter.runtime_identity
        self.calls = 0

    def propagate_window(self, **_kwargs):
        self.calls += 1
        raise RuntimeError("simulated SAM failure")


class _InterruptedAdapter:
    def __init__(self, stage2_job):
        self.job = stage2_job
        self.identity = stage2_job.FakeSamAdapter.identity
        self.runtime_identity = stage2_job.FakeSamAdapter.runtime_identity

    def propagate_window(self, **_kwargs):
        return self.job.SamPropagationResult(
            masks=(), forward_complete=False, reverse_complete=False
        )


class _SparseAdapter:
    def __init__(self, stage2_job, *, off_prompt=False):
        self.job = stage2_job
        self.identity = stage2_job.FakeSamAdapter.identity
        self.runtime_identity = stage2_job.FakeSamAdapter.runtime_identity
        self.off_prompt = off_prompt

    def propagate_window(self, *, window, prompts, **_kwargs):
        prompt = prompts[0]
        if self.off_prompt:
            masks = (
                self.job.RawSamMask(
                    frame_idx=prompt.anchor_frame,
                    prompt_id=prompt.prompt_id,
                    crop_box=(0, 0, 10, 10),
                    pixels=b"\x01" * 100,
                    direction="anchor",
                ),
            )
        else:
            masks = tuple(
                self.job.RawSamMask(
                    frame_idx=frame_idx,
                    prompt_id=prompt.prompt_id,
                    crop_box=(40, 30, 60, 50),
                    pixels=b"\x01" * 400,
                    direction="reverse" if frame_idx < prompt.anchor_frame else "forward",
                )
                for frame_idx in range(window.frame_start, window.frame_end + 1)
                if frame_idx != 7
            )
        return self.job.SamPropagationResult(
            masks=masks, forward_complete=True, reverse_complete=True
        )


def test_sam_failure_cannot_remove_dino_fallback(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, n_frames=10)
    proposal = _proposal(stage2_job, frame_idx=5)
    adapter = _FailingAdapter(stage2_job)
    result, *_ = _generate(
        stage2_job,
        tmp_path,
        stage1=stage1,
        proposals=(proposal,),
        adapter=adapter,
        config=_config(stage2_job, adapter),
    )
    loaded = stage2_job.load_sam_mask_shard(Path(result.shards[0].path))
    assert adapter.calls == 1
    assert any(flag.startswith("SAM_INFERENCE_FAILED") for flag in result.review_flags)
    assert any(flag.startswith("SAM_FALLBACK_USED") for flag in result.review_flags)
    assert {mask.frame_idx for mask in loaded.masks} == set(range(10))
    frame = stage2_job.union_sam_masks_for_frame(
        (loaded,), frame_idx=9, width=100, height=80
    )
    assert sum(frame) > (proposal.box[2] - proposal.box[0]) * (
        proposal.box[3] - proposal.box[1]
    )


def test_reused_fallback_shard_cannot_erase_required_review_flag(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, n_frames=10)
    proposal = _proposal(stage2_job, frame_idx=5)
    adapter = _FailingAdapter(stage2_job)
    result, _paths, _dino_ref, _dino_meta, _adapter, config = _generate(
        stage2_job,
        tmp_path,
        stage1=stage1,
        proposals=(proposal,),
        adapter=adapter,
        config=_config(stage2_job, adapter),
    )
    path = Path(result.shards[0].path)
    loaded = stage2_job.load_sam_mask_shard(path)
    path.write_bytes(
        stage2_job.encode_sam_mask_shard(
            replace(loaded.meta, review_flags=()), loaded.masks
        )
    )
    with pytest.raises(stage2_job.Stage2Error) as caught:
        _generate(
            stage2_job,
            tmp_path,
            stage1=stage1,
            proposals=(proposal,),
            adapter=stage2_job.FakeSamAdapter(),
            config=config,
        )
    assert caught.value.code == "INVALID_SAM_SHARD"
    assert any(
        "required fallback" in problem
        for problem in caught.value.details["problems"]
    )


def test_interrupted_forward_and_reverse_are_flagged_with_fallback(stage2_job, tmp_path):
    adapter = _InterruptedAdapter(stage2_job)
    proposal = _proposal(stage2_job, frame_idx=5)
    result, *_ = _generate(
        stage2_job,
        tmp_path,
        stage1=_stage1(stage2_job, n_frames=10),
        proposals=(proposal,),
        adapter=adapter,
        config=_config(stage2_job, adapter),
    )
    assert any(flag.startswith("SAM_FORWARD_INTERRUPTED") for flag in result.review_flags)
    assert any(flag.startswith("SAM_REVERSE_INTERRUPTED") for flag in result.review_flags)
    loaded = stage2_job.load_sam_mask_shard(Path(result.shards[0].path))
    assert all(mask.source == "dino-fallback" for mask in loaded.masks)


def test_pathological_mask_creates_flag_and_preserves_fallback(stage2_job, tmp_path):
    adapter = _SparseAdapter(stage2_job, off_prompt=True)
    proposal = _proposal(stage2_job, frame_idx=5)
    result, *_ = _generate(
        stage2_job,
        tmp_path,
        stage1=_stage1(stage2_job, n_frames=10),
        proposals=(proposal,),
        adapter=adapter,
        config=_config(stage2_job, adapter),
    )
    assert any(flag.startswith("SAM_MASK_OFF_PROMPT") for flag in result.review_flags)
    loaded = stage2_job.load_sam_mask_shard(Path(result.shards[0].path))
    assert {mask.frame_idx for mask in loaded.masks} == set(range(10))
    assert all(mask.source == "dino-fallback" for mask in loaded.masks)


def test_single_occluded_frame_uses_local_fallback_without_losing_other_sam_masks(
    stage2_job, tmp_path
):
    adapter = _SparseAdapter(stage2_job)
    proposal = _proposal(stage2_job, frame_idx=5)
    result, *_ = _generate(
        stage2_job,
        tmp_path,
        stage1=_stage1(stage2_job, n_frames=10),
        proposals=(proposal,),
        adapter=adapter,
        config=_config(stage2_job, adapter),
    )
    loaded = stage2_job.load_sam_mask_shard(Path(result.shards[0].path))
    frame_seven = [mask for mask in loaded.masks if mask.frame_idx == 7]
    assert [mask.source for mask in frame_seven] == ["dino-fallback"]
    assert any(mask.source.startswith("sam2-") for mask in loaded.masks if mask.frame_idx == 6)
    assert any(flag.startswith("SAM_FALLBACK_USED") for flag in result.review_flags)


def test_manual_seed_change_invalidates_only_intersecting_window(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    first, paths, dino_ref, dino_meta, _, config = _generate(
        stage2_job, tmp_path, stage1=stage1
    )
    seed = stage2_job.ManualSeed(
        schema_version=1,
        seed_id="miss-1",
        clip_id="clip",
        frame_idx=5,
        box=(10.0, 10.0, 20.0, 20.0),
        reason="human-confirmed miss",
    )
    adapter = stage2_job.FakeSamAdapter()
    second = stage2_job.generate_sam_mask_shards(
        stage1=stage1,
        paths=paths,
        dino_artifact=dino_ref,
        dino_meta=dino_meta,
        config=config,
        adapter=adapter,
        window_loader=lambda window: _window_input(stage2_job, stage1, window),
        manual_seeds=(seed,),
    )
    assert second.generated_window_count == 1
    assert second.reused_window_count == 2
    assert adapter.calls == 1
    assert second.shards[0] != first.shards[0]
    assert second.shards[1:] == first.shards[1:]


def test_threshold_change_reuses_dino_but_invalidates_sam_layer(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    proposals = (
        _proposal(stage2_job, frame_idx=5, score=0.22),
        _proposal(stage2_job, frame_idx=15, box=(10.0, 10.0, 20.0, 20.0), score=0.28),
    )
    first, paths, dino_ref, dino_meta, _, config = _generate(
        stage2_job, tmp_path, stage1=stage1, proposals=proposals
    )
    adapter = stage2_job.FakeSamAdapter()
    changed = replace(config, accepted_proposal_threshold=0.25)
    second = stage2_job.generate_sam_mask_shards(
        stage1=stage1,
        paths=paths,
        dino_artifact=dino_ref,
        dino_meta=dino_meta,
        config=changed,
        adapter=adapter,
        window_loader=lambda window: _window_input(stage2_job, stage1, window),
    )
    assert second.generated_window_count == 3
    assert second.reused_window_count == 0
    assert dino_ref == stage2_job.artifact_ref(Path(dino_ref.path))
    assert {ref.path for ref in first.shards}.isdisjoint(ref.path for ref in second.shards)


def test_overlapping_shards_union_masks_without_global_identity(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    proposals = (
        _proposal(stage2_job, frame_idx=5, box=(10.0, 10.0, 20.0, 20.0)),
        _proposal(stage2_job, frame_idx=12, box=(70.0, 50.0, 80.0, 60.0)),
    )
    result, *_ = _generate(stage2_job, tmp_path, stage1=stage1, proposals=proposals)
    loaded = tuple(
        stage2_job.load_sam_mask_shard(Path(ref.path)) for ref in result.shards
    )
    frame = stage2_job.union_sam_masks_for_frame(
        loaded, frame_idx=8, width=100, height=80
    )
    assert frame[15 * 100 + 15] == 1
    assert frame[55 * 100 + 75] == 1


def test_shard_is_deterministic_compressed_numpy_archive(stage2_job, tmp_path):
    proposal = _proposal(stage2_job, frame_idx=5)
    result, *_ = _generate(
        stage2_job,
        tmp_path,
        stage1=_stage1(stage2_job, n_frames=10),
        proposals=(proposal,),
    )
    path = Path(result.shards[0].path)
    before = path.read_bytes()
    loaded = stage2_job.load_sam_mask_shard(path)
    assert stage2_job.encode_sam_mask_shard(loaded.meta, loaded.masks) == before
    with np.load(path, allow_pickle=False) as archive:
        assert "metadata" in archive.files
        assert all(archive[name].dtype == np.uint8 for name in archive.files)


def test_noncanonical_or_tampered_shard_is_rejected(stage2_job, tmp_path):
    proposal = _proposal(stage2_job, frame_idx=5)
    result, *_ = _generate(
        stage2_job,
        tmp_path,
        stage1=_stage1(stage2_job, n_frames=10),
        proposals=(proposal,),
    )
    path = Path(result.shards[0].path)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.load_sam_mask_shard(path)
    assert caught.value.code == "INVALID_SAM_SHARD"


def test_real_adapter_first_frame_prompt_does_not_require_reverse_output(
    stage2_job, tmp_path
):
    class Predictor:
        def __init__(self):
            self.reverse_calls = 0

        def init_state(self, **_kwargs):
            return object()

        def add_new_points_or_box(self, **_kwargs):
            return None

        def propagate_in_video(self, _state, *, reverse, **_kwargs):
            if reverse:
                self.reverse_calls += 1
                return
            for frame_idx in range(3):
                yield frame_idx, (), ()

        def reset_state(self, _state):
            return None

    adapter = object.__new__(stage2_job.MetaSam2VideoAdapter)
    predictor = Predictor()
    adapter.predictor = predictor
    adapter.device = "cpu"
    adapter._torch = SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        inference_mode=contextlib.nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    result = adapter.propagate_window(
        window_input=tmp_path,
        window=stage2_job.TemporalWindow(index=0, frame_start=0, frame_end=2),
        prompts=(
            stage2_job.SamPrompt(
                prompt_id="prompt",
                prompt_kind="dino",
                anchor_frame=0,
                box=(10.0, 10.0, 20.0, 20.0),
                source_id="proposal",
            ),
        ),
        width=100,
        height=80,
        precision="float32",
    )
    assert result.forward_complete is True
    assert result.reverse_complete is True
    assert predictor.reverse_calls == 0


def test_shard_set_rejects_missing_window(stage2_job, tmp_path):
    result, *_ = _generate(stage2_job, tmp_path)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_sam_shard_set(
            result.metas[:-1], n_frames=20, window_size=10, window_overlap=2
        )
    assert caught.value.code == "INCOMPLETE_SAM_SHARD_SET"


def test_frame_union_fails_when_no_shard_covers_frame(stage2_job):
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.union_sam_masks_for_frame((), frame_idx=0, width=100, height=80)
    assert caught.value.code == "SAM_SHARD_COVERAGE_GAP"


def test_frame_union_rejects_wrong_dimensions(stage2_job, tmp_path):
    result, *_ = _generate(stage2_job, tmp_path)
    loaded = tuple(
        stage2_job.load_sam_mask_shard(Path(ref.path)) for ref in result.shards
    )
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.union_sam_masks_for_frame(
            loaded, frame_idx=0, width=99, height=80
        )
    assert caught.value.code == "SAM_FRAME_SIZE_MISMATCH"


@pytest.mark.parametrize(
    "change",
    [
        {"accepted_proposal_threshold": 0.25},
        {"window_size": 12},
        {"window_overlap": 3},
        {"precision": "bfloat16"},
        {"fallback_padding_scale": 0.20},
        {"fallback_min_padding_px": 8},
        {"min_mask_pixels": 8},
        {"min_prompt_area_ratio": 0.10},
        {"max_prompt_area_ratio": 8.0},
        {"max_frame_area_ratio": 0.60},
        {"anchor_neighborhood_scale": 1.5},
        {"min_anchor_neighborhood_fraction": 0.25},
    ],
)
def test_every_sam_meaning_change_changes_window_fingerprint(stage2_job, tmp_path, change):
    stage1 = _stage1(stage2_job)
    proposal = _proposal(stage2_job, frame_idx=5)
    dino_ref, dino_meta = _dino_artifact(stage2_job, tmp_path, stage1, (proposal,))
    adapter = stage2_job.FakeSamAdapter()
    config = _config(stage2_job, adapter)
    window = stage2_job.temporal_windows(20, 10, 2)[0]
    prompts = stage2_job.sam_prompts_for_window(window, (proposal,), ())
    original = stage2_job.sam_fingerprint(
        stage2_job.sam_window_fingerprint_payload(
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=config,
            window=window,
            prompts=prompts,
        )
    )
    changed = replace(config, **change)
    changed_value = stage2_job.sam_fingerprint(
        stage2_job.sam_window_fingerprint_payload(
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=changed,
            window=window,
            prompts=prompts,
        )
    )
    assert changed_value != original


@pytest.mark.parametrize(
    "model_change",
    [
        {"name": "different-sam"},
        {"revision": "different-revision"},
        {"sha256": "9" * 64},
    ],
)
def test_every_sam_model_identity_field_changes_fingerprint(
    stage2_job, tmp_path, model_change
):
    stage1 = _stage1(stage2_job)
    proposal = _proposal(stage2_job, frame_idx=5)
    dino_ref, dino_meta = _dino_artifact(stage2_job, tmp_path, stage1, (proposal,))
    adapter = stage2_job.FakeSamAdapter()
    config = _config(stage2_job, adapter)
    window = stage2_job.temporal_windows(20, 10, 2)[0]
    prompts = stage2_job.sam_prompts_for_window(window, (proposal,), ())
    original = stage2_job.sam_fingerprint(
        stage2_job.sam_window_fingerprint_payload(
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=config,
            window=window,
            prompts=prompts,
        )
    )
    changed = replace(config, model=replace(config.model, **model_change))
    assert original != stage2_job.sam_fingerprint(
        stage2_job.sam_window_fingerprint_payload(
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=changed,
            window=window,
            prompts=prompts,
        )
    )


def test_sam_runtime_identity_changes_window_fingerprint(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    proposal = _proposal(stage2_job, frame_idx=5)
    dino_ref, dino_meta = _dino_artifact(stage2_job, tmp_path, stage1, (proposal,))
    adapter = stage2_job.FakeSamAdapter()
    config = _config(stage2_job, adapter)
    window = stage2_job.temporal_windows(20, 10, 2)[0]
    prompts = stage2_job.sam_prompts_for_window(window, (proposal,), ())
    original = stage2_job.sam_fingerprint(
        stage2_job.sam_window_fingerprint_payload(
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=config,
            window=window,
            prompts=prompts,
        )
    )
    changed = replace(
        config,
        runtime=replace(config.runtime, source_tree_sha256="9" * 64),
    )
    assert original != stage2_job.sam_fingerprint(
        stage2_job.sam_window_fingerprint_payload(
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=changed,
            window=window,
            prompts=prompts,
        )
    )


def test_wrong_sam_runtime_identity_fails_before_window_loading(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, n_frames=10)
    proposal = _proposal(stage2_job, frame_idx=5)
    dino_ref, dino_meta = _dino_artifact(
        stage2_job, tmp_path / "inputs", stage1, (proposal,)
    )
    paths = stage2_job.build_run_paths(tmp_path / "work", "run", stage1.clip_id)
    adapter = stage2_job.FakeSamAdapter()
    config = replace(
        _config(stage2_job, adapter),
        runtime=replace(adapter.runtime_identity, revision="different-revision"),
    )
    loader_called = False

    def forbidden_loader(_window):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("runtime mismatch must fail before frame loading")

    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_sam_mask_shards(
            stage1=stage1,
            paths=paths,
            dino_artifact=dino_ref,
            dino_meta=dino_meta,
            config=config,
            adapter=adapter,
            window_loader=forbidden_loader,
        )
    assert caught.value.code == "SAM_RUNTIME_IDENTITY_MISMATCH"
    assert adapter.calls == 0
    assert loader_called is False


def test_sam_code_changes_do_not_invalidate_dino_layer(stage2_job, monkeypatch):
    assert stage2_job.DINO_CODE_VERSION == "milestone-2"
    dino_payload = {"source": "same", "prompt": "face."}
    sam_payload = {"dino": "same", "window": [0, 9]}
    dino_before = stage2_job.dino_fingerprint(dino_payload)
    sam_before = stage2_job.sam_fingerprint(sam_payload)
    monkeypatch.setattr(stage2_job, "SAM_CODE_VERSION", "milestone-3-mutated")
    assert stage2_job.dino_fingerprint(dino_payload) == dino_before
    assert stage2_job.sam_fingerprint(sam_payload) != sam_before


def test_manual_seed_validation_is_clip_and_bounds_safe(stage2_job):
    stage1 = _stage1(stage2_job)
    seed = stage2_job.ManualSeed(
        schema_version=1,
        seed_id="seed",
        clip_id="wrong",
        frame_idx=999,
        box=(-1.0, 0.0, 5.0, 5.0),
        reason="",
    )
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_manual_seeds((seed,), stage1)
    assert caught.value.code == "INVALID_MANUAL_SEED"


def test_malformed_manual_seed_types_return_structured_error(stage2_job):
    stage1 = _stage1(stage2_job)
    seed = stage2_job.ManualSeed(
        schema_version=1,
        seed_id="seed",
        clip_id="clip",
        frame_idx="not-a-frame",
        box=("not-a-number", 0.0, 5.0, 5.0),
        reason="human-confirmed miss",
    )
    valid = replace(
        seed,
        seed_id="valid-seed",
        frame_idx=1,
        box=(0.0, 0.0, 5.0, 5.0),
    )
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.validate_manual_seeds((valid, seed), stage1)
    assert caught.value.code == "INVALID_MANUAL_SEED"


def test_official_sam_identity_is_fully_pinned(stage2_job):
    identity = stage2_job.MetaSam2VideoAdapter.identity
    assert identity.name == "facebook/sam2.1-hiera-large"
    assert identity.revision == "665f8e2ad61cf5f53d65644ff27c8ee525124610"
    assert identity.sha256 == "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
    assert stage2_job.SAM_RUNTIME_REVISION == "2b90b9f5ceec907a1c18123530e92e794ad901a4"
    assert stage2_job.SAM_MODEL_CONFIG == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert (
        stage2_job.SAM_MODEL_CONFIG_SHA256
        == "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
    )


def test_verified_sam_runtime_hashes_config_code_and_execution_stack(
    stage2_job, tmp_path, monkeypatch
):
    package_root = tmp_path / "sam2"
    config = package_root / stage2_job.SAM_MODEL_CONFIG
    config.parent.mkdir(parents=True)
    config.write_bytes(b"pinned-test-config\n")
    (package_root / "predictor.py").write_bytes(b"runtime-code\n")
    monkeypatch.setattr(
        stage2_job,
        "SAM_MODEL_CONFIG_SHA256",
        stage2_job.sha256_file(config),
    )
    identity = stage2_job.verified_sam_runtime_identity(
        package_root,
        revision=stage2_job.SAM_RUNTIME_REVISION,
        torch_version="2.5.1+cu124",
        cuda_version="12.4",
    )
    assert identity.revision == stage2_job.SAM_RUNTIME_REVISION
    assert identity.model_config_sha256 == stage2_job.sha256_file(config)
    assert identity.source_tree_sha256 == stage2_job.sha256_directory_tree(package_root)
    assert identity.torch_version == "2.5.1+cu124"
    assert identity.cuda_version == "12.4"


def test_verified_sam_runtime_rejects_revision_and_config_drift(
    stage2_job, tmp_path
):
    package_root = tmp_path / "sam2"
    config = package_root / stage2_job.SAM_MODEL_CONFIG
    config.parent.mkdir(parents=True)
    config.write_bytes(b"changed-config\n")
    with pytest.raises(stage2_job.Stage2Error) as revision_error:
        stage2_job.verified_sam_runtime_identity(
            package_root,
            revision="wrong-revision",
            torch_version="2.5.1",
            cuda_version="12.4",
        )
    assert revision_error.value.code == "SAM_RUNTIME_REVISION_MISMATCH"
    with pytest.raises(stage2_job.Stage2Error) as config_error:
        stage2_job.verified_sam_runtime_identity(
            package_root,
            revision=stage2_job.SAM_RUNTIME_REVISION,
            torch_version="2.5.1",
            cuda_version="12.4",
        )
    assert config_error.value.code == "SAM_MODEL_CONFIG_HASH_MISMATCH"


def test_installed_sam_revision_uses_vcs_metadata_and_checks_repository(
    stage2_job, tmp_path, monkeypatch
):
    package_root = tmp_path / "sam2"
    package_root.mkdir()

    class Distribution:
        def __init__(self, repository):
            self.repository = repository

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": self.repository,
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": stage2_job.SAM_RUNTIME_REVISION,
                    },
                }
            )

    monkeypatch.setattr(
        stage2_job.importlib_metadata,
        "distribution",
        lambda _name: Distribution(stage2_job.SAM_RUNTIME_REPOSITORY),
    )
    assert (
        stage2_job.installed_sam_runtime_revision(package_root)
        == stage2_job.SAM_RUNTIME_REVISION
    )
    monkeypatch.setattr(
        stage2_job.importlib_metadata,
        "distribution",
        lambda _name: Distribution("https://example.com/not-meta/sam2.git"),
    )
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.installed_sam_runtime_revision(package_root)
    assert caught.value.code == "SAM_RUNTIME_REPOSITORY_MISMATCH"


def test_real_adapter_rejects_wrong_checkpoint_before_import(stage2_job, tmp_path):
    checkpoint = tmp_path / "sam.pt"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.MetaSam2VideoAdapter(checkpoint_path=checkpoint)
    assert caught.value.code == "SAM_CHECKPOINT_HASH_MISMATCH"


def test_real_adapter_dependency_failure_is_actionable_and_lazy(
    stage2_job, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "sam.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        stage2_job,
        "sha256_file",
        lambda _path: stage2_job.SAM_MODEL_WEIGHTS_SHA256,
    )
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("sam2"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.MetaSam2VideoAdapter(checkpoint_path=checkpoint)
    assert caught.value.code == "SAM_DEPENDENCIES_MISSING"
    assert "setup" in caught.value.recovery.lower()
