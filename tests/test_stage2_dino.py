"""Milestone 2 tests for DINO proposal generation and reuse."""

from __future__ import annotations

import builtins
from dataclasses import replace
from pathlib import Path

import pytest


def _stage1(stage2_job, *, n_frames=21, width=100, height=80):
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
    source = stage2_job.ArtifactRef(path="/private/source.mp4", sha256="1" * 64, bytes=100)
    stage1_video = stage2_job.ArtifactRef(
        path="/private/stage1.mp4", sha256="2" * 64, bytes=100
    )
    manifest = stage2_job.ArtifactRef(
        path="/private/stage1.manifest.json", sha256="3" * 64, bytes=100
    )
    return stage2_job.StageIInput(
        schema_version=1,
        clip_id="GX010057",
        source=source,
        stage1_video=stage1_video,
        stage1_manifest=manifest,
        source_video=facts,
        stage1_output_video=facts,
        stage1_status="NEEDS_REVIEW",
        stage1_audit_reasons=("review",),
        egoblur={},
        warnings=(),
    )


class GlobalFaceAdapter:
    """Returns one fixed global face through whichever crop contains it."""

    def __init__(self, stage2_job, *, fail_on_call=None, malformed=None):
        self.stage2_job = stage2_job
        self.identity = stage2_job.ModelIdentity(
            name="scripted-dino", revision="test", sha256="d" * 64
        )
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.malformed = malformed

    def infer_batch(self, images, *, prompt, box_threshold, text_threshold):
        assert prompt.endswith(".")
        assert box_threshold == pytest.approx(0.10)
        assert text_threshold == pytest.approx(0.25)
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        if self.malformed == "wrong-result-count":
            return []
        results = []
        for image in images:
            if self.malformed == "nan":
                results.append(
                    (
                        self.stage2_job.RawDinoDetection(
                            box=(1.0, 1.0, 5.0, 5.0), score=float("nan")
                        ),
                    )
                )
                continue
            gx1, gy1, gx2, gy2 = (45.0, 35.0, 55.0, 45.0)
            ox, oy = image.crop_origin
            lx1, ly1, lx2, ly2 = gx1 - ox, gy1 - oy, gx2 - ox, gy2 - oy
            if lx2 <= 0 or ly2 <= 0 or lx1 >= image.width or ly1 >= image.height:
                results.append(())
                continue
            results.append(
                (
                    self.stage2_job.RawDinoDetection(
                        box=(lx1, ly1, lx2, ly2), score=0.42, label="face"
                    ),
                )
            )
        return results

    def reset_peak_vram(self):
        return None

    def peak_vram_bytes(self):
        return 1234


def _config(stage2_job, adapter, **changes):
    base = stage2_job.DinoGenerationConfig(
        model=adapter.identity,
        preprocessing=(
            ("color", "synthetic-RGB"),
            ("resize", "none"),
            ("normalization", "none"),
            ("frame_decoder", "fake-image"),
        ),
    )
    return replace(base, **changes)


def _generate(stage2_job, tmp_path, stage1, adapter, config):
    paths = stage2_job.build_run_paths(tmp_path, "dino-test", stage1.clip_id)
    result = stage2_job.generate_dino_proposals(
        stage1=stage1,
        paths=paths,
        config=config,
        adapter=adapter,
        frame_loader=lambda _frame_idx: stage2_job.FakeImage(
            stage1.source_video.display_width,
            stage1.source_video.display_height,
        ),
    )
    return result, paths


@pytest.mark.parametrize(
    ("n_frames", "spacing", "expected"),
    [
        (1, 20, (0,)),
        (40, 20, (0, 20, 39)),
        (41, 20, (0, 20, 40)),
        (5, 10, (0, 4)),
    ],
)
def test_anchor_schedule_always_includes_both_clip_edges(stage2_job, n_frames, spacing, expected):
    assert stage2_job.anchor_schedule(n_frames, spacing) == expected


@pytest.mark.parametrize(("n_frames", "spacing"), [(0, 20), (10, 0), (-1, 5)])
def test_invalid_anchor_schedule_fails(stage2_job, n_frames, spacing):
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.anchor_schedule(n_frames, spacing)
    assert caught.value.code == "INVALID_ANCHOR_SCHEDULE"


def test_two_by_two_tiles_cover_frame_and_overlap(stage2_job):
    tiles = stage2_job.tile_layout(100, 80, rows=2, cols=2, overlap=0.20)
    assert len(tiles) == 4
    by_origin = {tile.origin: tile for tile in tiles}
    assert by_origin["tile-r0-c0"] == stage2_job.Tile("tile-r0-c0", 0, 0, 56, 45)
    assert by_origin["tile-r0-c1"] == stage2_job.Tile("tile-r0-c1", 44, 0, 100, 45)
    assert by_origin["tile-r1-c0"] == stage2_job.Tile("tile-r1-c0", 0, 35, 56, 80)
    assert by_origin["tile-r1-c1"] == stage2_job.Tile("tile-r1-c1", 44, 35, 100, 80)
    assert by_origin["tile-r0-c0"].x2 - by_origin["tile-r0-c1"].x1 == 12
    assert by_origin["tile-r0-c0"].y2 - by_origin["tile-r1-c0"].y1 == 10


def test_frame_views_include_full_frame_then_four_tiles(stage2_job):
    image = stage2_job.FakeImage(100, 80)
    views = stage2_job.frame_views(image, rows=2, cols=2, overlap=0.20)
    assert [tile.origin for tile, _image in views] == [
        "full-frame",
        "tile-r0-c0",
        "tile-r0-c1",
        "tile-r1-c0",
        "tile-r1-c1",
    ]
    assert [view.crop_origin for _tile, view in views[1:]] == [
        (0, 0),
        (44, 0),
        (0, 35),
        (44, 35),
    ]


def test_tiled_coordinates_map_back_to_original_frame(stage2_job):
    tile = stage2_job.Tile("tile-r1-c1", 44, 35, 100, 80)
    proposal = stage2_job.map_raw_detection(
        stage2_job.RawDinoDetection(box=(1.0, 2.0, 11.0, 12.0), score=0.5),
        frame_idx=20,
        tile=tile,
        frame_width=100,
        frame_height=80,
        proposal_floor=0.10,
    )
    assert proposal.box == (45.0, 37.0, 55.0, 47.0)
    assert proposal.origins == ("tile-r1-c1",)
    assert proposal.source == "tiled-only"


def _proposal(stage2_job, *, box, score, origin, frame_idx=0, label="face"):
    tile = stage2_job.Tile(origin, 0, 0, 100, 100)
    return stage2_job.map_raw_detection(
        stage2_job.RawDinoDetection(box=box, score=score, label=label),
        frame_idx=frame_idx,
        tile=tile,
        frame_width=100,
        frame_height=100,
        proposal_floor=0.10,
    )


def test_nms_unions_full_and_tiled_provenance(stage2_job):
    full = _proposal(
        stage2_job, box=(10, 10, 30, 30), score=0.8, origin="full-frame"
    )
    tiled = _proposal(
        stage2_job, box=(10, 10, 30, 30), score=0.7, origin="tile-r0-c0"
    )
    result = stage2_job.union_nms((tiled, full), 0.70)
    assert len(result) == 1
    assert result[0].box == full.box
    assert result[0].score == 0.8
    assert result[0].source == "shared"
    assert result[0].origins == ("full-frame", "tile-r0-c0")


def test_nms_is_deterministic_and_preserves_close_distinct_faces(stage2_job):
    left = _proposal(stage2_job, box=(10, 10, 30, 30), score=0.8, origin="full-frame")
    right = _proposal(stage2_job, box=(20, 10, 40, 30), score=0.8, origin="full-frame")
    forward = stage2_job.union_nms((left, right), 0.70)
    reverse = stage2_job.union_nms((right, left), 0.70)
    assert forward == reverse
    assert len(forward) == 2


def test_nms_never_merges_across_frames_or_labels(stage2_job):
    first = _proposal(stage2_job, box=(10, 10, 30, 30), score=0.8, origin="full-frame")
    next_frame = _proposal(
        stage2_job,
        box=(10, 10, 30, 30),
        score=0.8,
        origin="full-frame",
        frame_idx=1,
    )
    other_label = _proposal(
        stage2_job,
        box=(10, 10, 30, 30),
        score=0.8,
        origin="full-frame",
        label="head",
    )
    assert len(stage2_job.union_nms((first, next_frame, other_label), 0.70)) == 3


def test_below_floor_is_dropped_but_invalid_output_fails(stage2_job):
    tile = stage2_job.Tile("full-frame", 0, 0, 100, 100)
    assert stage2_job.map_raw_detection(
        stage2_job.RawDinoDetection(box=(1, 1, 5, 5), score=0.09),
        frame_idx=0,
        tile=tile,
        frame_width=100,
        frame_height=100,
        proposal_floor=0.10,
    ) is None
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.map_raw_detection(
            stage2_job.RawDinoDetection(box=(1, 1, 5, 5), score=float("nan")),
            frame_idx=0,
            tile=tile,
            frame_width=100,
            frame_height=100,
            proposal_floor=0.10,
        )
    assert caught.value.code == "INVALID_DINO_OUTPUT"


def test_generation_runs_full_and_tiled_views_and_records_metrics(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, adapter, view_batch_size=2)
    result, paths = _generate(stage2_job, tmp_path, stage1, adapter, config)
    assert result.generated_anchor_count == 2
    assert result.reused_anchor_count == 0
    assert result.meta.anchor_frames == (0, 20)
    assert adapter.calls == 6  # 5 views per anchor, batched 2+2+1
    assert len(result.meta.proposals) == 2
    assert {proposal.source for proposal in result.meta.proposals} == {"shared"}
    assert result.meta.metrics["source_counts"] == {
        "full-frame-only": 0,
        "tiled-only": 0,
        "shared": 2,
    }
    assert result.meta.metrics["n_nms_suppressed"] == 8
    assert result.meta.metrics["peak_vram_bytes"] == 1234
    assert paths.dino.exists() and paths.dino_checkpoint.exists()


def test_complete_artifact_reuse_makes_zero_model_calls(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    first_adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, first_adapter, view_batch_size=5)
    first, paths = _generate(stage2_job, tmp_path, stage1, first_adapter, config)
    second_adapter = GlobalFaceAdapter(stage2_job, fail_on_call=1)
    second = stage2_job.generate_dino_proposals(
        stage1=stage1,
        paths=paths,
        config=config,
        adapter=second_adapter,
        frame_loader=lambda _index: (_ for _ in ()).throw(AssertionError("decoded on reuse")),
    )
    assert second.reused_final_artifact is True
    assert second.generated_anchor_count == 0
    assert second.reused_anchor_count == len(first.meta.anchor_frames)
    assert second_adapter.calls == 0
    assert second.artifact == first.artifact


def test_interrupted_generation_resumes_completed_anchors(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, n_frames=25)
    interrupted = GlobalFaceAdapter(stage2_job, fail_on_call=2)
    config = _config(stage2_job, interrupted, view_batch_size=5)
    paths = stage2_job.build_run_paths(tmp_path, "dino-test", stage1.clip_id)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=config,
            adapter=interrupted,
            frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
        )
    assert caught.value.code == "DINO_INFERENCE_FAILED"
    assert not paths.dino.exists()

    resumed_adapter = GlobalFaceAdapter(stage2_job)
    resumed = stage2_job.generate_dino_proposals(
        stage1=stage1,
        paths=paths,
        config=config,
        adapter=resumed_adapter,
        frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
    )
    assert resumed.reused_anchor_count == 1
    assert resumed.generated_anchor_count == 2
    assert resumed_adapter.calls == 2
    assert resumed.meta.anchor_frames == (0, 20, 24)


def test_generate_dino_proposals_honors_a_stop_request_mid_run(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, n_frames=25)  # anchors (0, 20, 24)
    adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, adapter)
    paths = stage2_job.build_run_paths(tmp_path, "dino-test", stage1.clip_id)
    stage2_job.transition_state(
        paths.state, run_id="dino-test", clip_id=stage1.clip_id, mode="production", target="VALIDATED"
    )
    loaded_frames = []

    def frame_loader(frame_idx):
        loaded_frames.append(frame_idx)
        if len(loaded_frames) == 2:
            stage2_job.request_stop(paths)
        return stage2_job.FakeImage(100, 80)

    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_dino_proposals(
            stage1=stage1, paths=paths, config=config, adapter=adapter, frame_loader=frame_loader
        )
    assert caught.value.code == "STOP_REQUESTED"
    # Stopped once the marker appeared, before starting the 3rd anchor -
    # not after silently running every remaining anchor to completion.
    assert loaded_frames == [0, 20]


def test_torn_final_checkpoint_row_is_repaired_on_resume(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, n_frames=25)
    interrupted = GlobalFaceAdapter(stage2_job, fail_on_call=2)
    config = _config(stage2_job, interrupted, view_batch_size=5)
    paths = stage2_job.build_run_paths(tmp_path, "dino-test", stage1.clip_id)
    with pytest.raises(stage2_job.Stage2Error):
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=config,
            adapter=interrupted,
            frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
        )
    with paths.dino_checkpoint.open("ab") as handle:
        handle.write(b'{"record_type":"anchor"')

    resumed_adapter = GlobalFaceAdapter(stage2_job)
    resumed = stage2_job.generate_dino_proposals(
        stage1=stage1,
        paths=paths,
        config=config,
        adapter=resumed_adapter,
        frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
    )
    assert resumed.reused_anchor_count == 1
    assert paths.dino_checkpoint.read_bytes().endswith(b"\n")
    assert b'{"record_type":"anchor"\n' not in paths.dino_checkpoint.read_bytes()


def test_checkpoint_fingerprint_mismatch_never_reuses_rows(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job, n_frames=25)
    interrupted = GlobalFaceAdapter(stage2_job, fail_on_call=2)
    config = _config(stage2_job, interrupted, view_batch_size=5)
    paths = stage2_job.build_run_paths(tmp_path, "dino-test", stage1.clip_id)
    with pytest.raises(stage2_job.Stage2Error):
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=config,
            adapter=interrupted,
            frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
        )
    changed = replace(config, prompt="human face.")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=changed,
            adapter=GlobalFaceAdapter(stage2_job),
            frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
        )
    assert caught.value.code == "DINO_FINGERPRINT_MISMATCH"


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "human face."},
        {"proposal_floor": 0.11},
        {"text_threshold": 0.3},
        {"anchor_spacing": 10},
        {"tile_rows": 3},
        {"tile_cols": 3},
        {"tile_overlap": 0.1},
        {"nms_iou": 0.5},
        {"view_batch_size": 2},
        {"preprocessing": (("color", "BGR"),)},
    ],
)
def test_every_dino_meaning_change_changes_fingerprint(stage2_job, change):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, adapter)
    anchors = stage2_job.anchor_schedule(stage1.source_video.n_frames, config.anchor_spacing)
    original = stage2_job.dino_fingerprint(
        stage2_job.dino_fingerprint_payload(stage1, config, anchors)
    )
    changed = replace(config, **change)
    changed_anchors = stage2_job.anchor_schedule(
        stage1.source_video.n_frames, changed.anchor_spacing
    )
    changed_value = stage2_job.dino_fingerprint(
        stage2_job.dino_fingerprint_payload(stage1, changed, changed_anchors)
    )
    assert original != changed_value


def test_source_and_model_identity_change_dino_fingerprint(stage2_job):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, adapter)
    anchors = stage2_job.anchor_schedule(stage1.source_video.n_frames, config.anchor_spacing)
    original = stage2_job.dino_fingerprint(
        stage2_job.dino_fingerprint_payload(stage1, config, anchors)
    )
    changed_source = replace(
        stage1, source=replace(stage1.source, sha256="9" * 64)
    )
    changed_model = replace(
        config, model=replace(config.model, revision="different-revision")
    )
    assert original != stage2_job.dino_fingerprint(
        stage2_job.dino_fingerprint_payload(changed_source, config, anchors)
    )
    assert original != stage2_job.dino_fingerprint(
        stage2_job.dino_fingerprint_payload(stage1, changed_model, anchors)
    )


@pytest.mark.parametrize(
    "model_change",
    [
        {"name": "different-model"},
        {"revision": "different-revision"},
        {"sha256": "9" * 64},
    ],
)
def test_every_dino_model_identity_field_changes_fingerprint(stage2_job, model_change):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, adapter)
    anchors = stage2_job.anchor_schedule(stage1.source_video.n_frames, config.anchor_spacing)
    original = stage2_job.dino_fingerprint(
        stage2_job.dino_fingerprint_payload(stage1, config, anchors)
    )
    changed = replace(config, model=replace(config.model, **model_change))
    assert original != stage2_job.dino_fingerprint(
        stage2_job.dino_fingerprint_payload(stage1, changed, anchors)
    )


def test_operating_threshold_reuses_proposals_and_reports_accept_reject(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    result, paths = _generate(stage2_job, tmp_path, stage1, adapter, _config(stage2_job, adapter))
    artifact_before = paths.dino.read_bytes()
    lower = stage2_job.select_dino_proposals(result.meta, 0.20)
    higher = stage2_job.select_dino_proposals(result.meta, 0.50)
    assert lower.metrics == {"threshold": 0.20, "n_accepted": 2, "n_rejected": 0}
    assert higher.metrics == {"threshold": 0.50, "n_accepted": 0, "n_rejected": 2}
    assert lower.accepted == result.meta.proposals
    assert paths.dino.read_bytes() == artifact_before


def test_threshold_below_stored_floor_requires_dino_recompute(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    result, _paths = _generate(stage2_job, tmp_path, stage1, adapter, _config(stage2_job, adapter))
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.select_dino_proposals(result.meta, 0.09)
    assert caught.value.code == "INVALID_DINO_OPERATING_THRESHOLD"


@pytest.mark.parametrize("malformed", ["wrong-result-count", "nan"])
def test_malformed_adapter_output_fails_closed(stage2_job, tmp_path, malformed):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job, malformed=malformed)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        _generate(stage2_job, tmp_path, stage1, adapter, _config(stage2_job, adapter))
    assert caught.value.code == "INVALID_DINO_OUTPUT"


def test_wrong_model_identity_fails_before_frame_decode(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    wrong_model = replace(adapter.identity, revision="wrong")
    paths = stage2_job.build_run_paths(tmp_path, "dino-test", stage1.clip_id)
    decoded = False

    def frame_loader(_index):
        nonlocal decoded
        decoded = True
        return stage2_job.FakeImage(100, 80)

    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=_config(stage2_job, adapter, model=wrong_model),
            adapter=adapter,
            frame_loader=frame_loader,
        )
    assert caught.value.code == "DINO_MODEL_IDENTITY_MISMATCH"
    assert decoded is False


def test_decoded_frame_size_mismatch_fails_closed(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    paths = stage2_job.build_run_paths(tmp_path, "dino-test", stage1.clip_id)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=_config(stage2_job, adapter),
            adapter=adapter,
            frame_loader=lambda _index: stage2_job.FakeImage(99, 80),
        )
    assert caught.value.code == "FRAME_SIZE_MISMATCH"


def test_final_artifact_tampering_is_not_reused(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, adapter)
    result, paths = _generate(stage2_job, tmp_path, stage1, adapter, config)
    raw = stage2_job.read_json(paths.dino)
    raw["metrics"]["n_proposals"] = 999
    stage2_job.atomic_write_json(paths.dino, raw)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=config,
            adapter=adapter,
            frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
        )
    assert result.meta.metrics["n_proposals"] != 999
    assert caught.value.code == "INVALID_DINO_ARTIFACT"


def test_canonical_proposal_edit_that_diverges_from_checkpoint_is_not_reused(
    stage2_job, tmp_path
):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, adapter)
    result, paths = _generate(stage2_job, tmp_path, stage1, adapter, config)
    original = result.meta.proposals[0]
    changed = stage2_job._finalize_proposal(
        replace(original, box=(original.box[0] + 0.5, *original.box[1:]))
    )
    raw = stage2_job.read_json(paths.dino)
    raw["proposals"][0] = stage2_job._jsonable(changed)
    stage2_job.atomic_write_json(paths.dino, raw)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=config,
            adapter=adapter,
            frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
        )
    assert caught.value.code == "INVALID_DINO_ARTIFACT"
    assert "checkpoint" in caught.value.message.lower()


def test_checkpoint_tampering_after_finalization_is_not_reused(stage2_job, tmp_path):
    stage1 = _stage1(stage2_job)
    adapter = GlobalFaceAdapter(stage2_job)
    config = _config(stage2_job, adapter)
    _result, paths = _generate(stage2_job, tmp_path, stage1, adapter, config)
    with paths.dino_checkpoint.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.generate_dino_proposals(
            stage1=stage1,
            paths=paths,
            config=config,
            adapter=adapter,
            frame_loader=lambda _index: stage2_job.FakeImage(100, 80),
        )
    assert caught.value.code == "INVALID_DINO_ARTIFACT"


def test_official_model_identity_is_fully_pinned(stage2_job):
    identity = stage2_job.TransformersGroundingDinoAdapter.identity
    assert identity.name == "IDEA-Research/grounding-dino-base"
    assert identity.revision == "e76a695ed7ae1032a61530cce4b4e9b65f4e368b"
    assert identity.sha256 == "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"


def test_real_adapter_dependency_failure_is_actionable_and_lazy(stage2_job, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing GPU runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.TransformersGroundingDinoAdapter()
    assert caught.value.code == "DINO_DEPENDENCIES_MISSING"
    assert "setup" in caught.value.recovery.lower()


def test_real_adapter_rejects_unverified_persistent_snapshot_before_import(
    stage2_job, tmp_path
):
    snapshot = tmp_path / "dino"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"not the pinned checkpoint")
    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.TransformersGroundingDinoAdapter(model_path=snapshot)
    assert caught.value.code == "DINO_CHECKPOINT_HASH_MISMATCH"


def test_directory_tree_hash_covers_every_file_not_just_the_pinned_one(stage2_job, tmp_path):
    snapshot = tmp_path / "dino"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "config.json").write_bytes(b'{"architecture": "grounding-dino"}')
    before = stage2_job.sha256_directory_tree(snapshot)

    # A file that TransformersGroundingDinoAdapter never hash-pins on its own
    # (only model.safetensors gets a known-good sha256) must still change
    # the snapshot's overall identity when it changes, or it isn't actually
    # covered by any integrity check.
    (snapshot / "config.json").write_bytes(b'{"architecture": "tampered"}')
    after = stage2_job.sha256_directory_tree(snapshot)
    assert before != after

    # And reverting it must reproduce the original hash exactly (deterministic).
    (snapshot / "config.json").write_bytes(b'{"architecture": "grounding-dino"}')
    assert stage2_job.sha256_directory_tree(snapshot) == before


def test_directory_tree_hash_uses_caller_specific_error_codes(stage2_job, tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(stage2_job.Stage2Error) as default_caught:
        stage2_job.sha256_directory_tree(missing)
    assert default_caught.value.code == "SAM_RUNTIME_UNVERIFIABLE"

    with pytest.raises(stage2_job.Stage2Error) as dino_caught:
        stage2_job.sha256_directory_tree(
            missing,
            description="persistent DINO snapshot",
            unverifiable_code="DINO_SNAPSHOT_UNVERIFIABLE",
            changed_code="DINO_SNAPSHOT_CHANGED",
        )
    assert dino_caught.value.code == "DINO_SNAPSHOT_UNVERIFIABLE"
    assert "DINO" in dino_caught.value.message
    assert "SAM2" not in dino_caught.value.message


def test_directory_tree_hash_allows_a_symlink_contained_within_the_tree(stage2_job, tmp_path):
    # Matches the real facebookresearch/sam2 layout: a legacy top-level
    # config file is a symlink into configs/ within the SAME installed tree
    # -- e.g. sam2_hiera_t.yaml -> configs/sam2/sam2_hiera_t.yaml. The git
    # revision/remote/clean-worktree checks already prove this exact tree,
    # symlink included, is what the pinned upstream commit contains, so this
    # must not be rejected as unverifiable.
    package = tmp_path / "sam2"
    (package / "configs" / "sam2").mkdir(parents=True)
    real_config = package / "configs" / "sam2" / "sam2_hiera_t.yaml"
    real_config.write_bytes(b"model: hiera_t")
    legacy_alias = package / "sam2_hiera_t.yaml"
    legacy_alias.symlink_to(Path("configs/sam2/sam2_hiera_t.yaml"))

    before = stage2_job.sha256_directory_tree(package)

    # The symlink's target content must actually be covered by the hash,
    # not silently skipped once let through.
    real_config.write_bytes(b"model: hiera_t_tampered")
    after = stage2_job.sha256_directory_tree(package)
    assert before != after


def test_directory_tree_hash_rejects_a_symlink_escaping_the_tree(stage2_job, tmp_path):
    package = tmp_path / "sam2"
    package.mkdir()
    (package / "config.yaml").write_bytes(b"model: hiera_l")
    outside = tmp_path / "outside-secret.yaml"
    outside.write_bytes(b"attacker controlled")
    (package / "escaped.yaml").symlink_to(outside)

    with pytest.raises(stage2_job.Stage2Error) as caught:
        stage2_job.sha256_directory_tree(package)
    assert caught.value.code == "SAM_RUNTIME_UNVERIFIABLE"
    assert "outside its own tree" in caught.value.message
