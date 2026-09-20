"""Contract tests for the private Depth Anything 3 Metric Large pilot job.

These tests deliberately import the PEP 723 job without a GPU or either model
package.  Heavy imports are worker-local so provenance, privacy, clock and
resume rules remain testable on a development machine.
"""

from __future__ import annotations

import json

import pytest


def _clock(job):
    return [
        {
            "frame_index": index,
            "pts": index * 1001,
            "time_base": "1/30000",
            "timestamp_seconds": index * 1001 / 30000,
        }
        for index in range(10)
    ]


def test_da3_model_and_immutable_revisions_are_pinned(depth_compare_job):
    assert depth_compare_job.DA3_MODEL_ID == "depth-anything/DA3METRIC-LARGE"
    assert len(depth_compare_job.DA3_MODEL_REVISION) == 40
    assert "@" + depth_compare_job.DA3_CODE_REVISION in depth_compare_job.dependencies_for_worker("da3")
    assert "addict>=2.4,<3" in depth_compare_job.worker_dependencies("da3")
    assert depth_compare_job.model_specs() == ["da3"]
    with pytest.raises(depth_compare_job.ExperimentError, match="unknown worker model"):
        depth_compare_job.dependencies_for_worker("another-model")


def test_da3_absent_confidence_is_an_explicit_unavailable_sentinel(depth_compare_job):
    import numpy as np

    confidence, status, available = depth_compare_job.da3_confidence_for_storage(None, (2, 3), 6, 4)

    assert confidence.dtype == np.float32
    assert confidence.shape == (4, 6)
    assert not np.isfinite(confidence).any()
    assert status == "upstream_not_emitted_all_nan_sentinel"
    assert available is False


def test_lower_fps_selection_uses_source_pts_and_keeps_first_frame(depth_compare_job):
    selected = depth_compare_job.select_source_frames(_clock(depth_compare_job), 10.0)
    assert selected[0]["frame_index"] == 0
    assert [item["pts"] for item in selected] == [0, 3003, 6006, 9009]
    assert all(item["time_base"] == "1/30000" for item in selected)


def test_worker_command_keeps_uv_options_before_the_script_operand(depth_compare_job, tmp_path):
    request = tmp_path / "private-worker-request.json"
    command = depth_compare_job.build_worker_command("/workspace/bin/uv", request, "da3")

    script_index = command.index(str(depth_compare_job.Path(depth_compare_job.__file__).resolve()))
    with_indexes = [index for index, value in enumerate(command) if value == "--with"]
    worker_index = command.index("--worker")
    assert with_indexes and max(with_indexes) < script_index < worker_index
    assert "--" not in command
    assert [command[index + 1] for index in with_indexes] == depth_compare_job.worker_dependencies("da3")
    assert command[worker_index:] == ["--worker", "da3", "--worker-request", str(request)]


def test_input_attestation_rejects_fisheye_or_noncontinuous_source(depth_compare_job, tmp_path):
    attestation = tmp_path / "input.json"
    attestation.write_text(
        json.dumps(
            {
                "privacy_status": "redacted_public_safe",
                "continuous_child": True,
                "contains_privacy_cuts": False,
                "input_sha256": "a" * 64,
                "projection": "fisheye",
            }
        )
    )
    with pytest.raises(depth_compare_job.ExperimentError, match="fisheye"):
        depth_compare_job.require_input_attestation(attestation, "a" * 64, False)

    attestation.write_text(
        json.dumps(
            {
                "privacy_status": "redacted_public_safe",
                "continuous_child": False,
                "contains_privacy_cuts": False,
                "input_sha256": "a" * 64,
                "projection": "rectilinear",
            }
        )
    )
    with pytest.raises(depth_compare_job.ExperimentError, match="continuous_child"):
        depth_compare_job.require_input_attestation(attestation, "a" * 64, False)


def test_unredacted_private_exception_requires_flag_hash_and_nonpublication(depth_compare_job, tmp_path):
    attestation = tmp_path / "input.json"
    attestation.write_text(
        json.dumps(
            {
                "privacy_status": "private_unredacted_user_authorized",
                "continuous_child": True,
                "contains_privacy_cuts": False,
                "input_sha256": "a" * 64,
                "projection": "rectilinear",
                "publication_permitted": False,
                "private_exception_reason": "One-off private experiment explicitly authorized by the user.",
            }
        )
    )
    with pytest.raises(depth_compare_job.ExperimentError, match="allow-private-unredacted-input"):
        depth_compare_job.require_input_attestation(attestation, "a" * 64, False)

    accepted = depth_compare_job.require_input_attestation(attestation, "a" * 64, True)
    assert accepted["privacy_status"] == "private_unredacted_user_authorized"

    attestation.write_text(
        json.dumps({**accepted, "publication_permitted": True})
    )
    with pytest.raises(depth_compare_job.ExperimentError, match="publication_permitted"):
        depth_compare_job.require_input_attestation(attestation, "a" * 64, True)


def test_calibration_never_fills_in_missing_principal_point(depth_compare_job, tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps({
            "approval_status": depth_compare_job.CALIBRATION_APPROVAL_STATUS,
            "camera_setup_fingerprint_sha256": "a" * 64,
            "projection": "dewarped_rectilinear", "fx": 1000.0, "fov_x_deg": 90.0,
        })
    )
    calibration = depth_compare_job.load_calibration(calibration_path, 1920, 1080)
    assert calibration is not None
    assert calibration.focal_px == 1000.0
    assert not calibration.has_full_intrinsics
    assert depth_compare_job.intrinsics_for_resized_input(calibration, 1920, 1080, 960, 540) is None


def test_da3_rejects_raw_or_wrong_group_calibration(depth_compare_job, tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"projection": "rectilinear", "fx": 1000.0}), encoding="utf-8")
    with pytest.raises(depth_compare_job.ExperimentError, match="approved COLMAP"):
        depth_compare_job.load_calibration(calibration_path, 1920, 1080)

    calibration_path.write_text(json.dumps({
        "approval_status": depth_compare_job.CALIBRATION_APPROVAL_STATUS,
        "camera_setup_fingerprint_sha256": "a" * 64,
        "projection": "rectilinear", "fx": 1000.0,
    }), encoding="utf-8")
    calibration = depth_compare_job.load_calibration(calibration_path, 1920, 1080)
    assert calibration is not None
    with pytest.raises(depth_compare_job.ExperimentError, match="does not match"):
        depth_compare_job.require_calibration_setup_binding(calibration, {"camera_setup_fingerprint_sha256": "b" * 64})


def test_existing_run_refuses_a_configuration_fingerprint_change(depth_compare_job, tmp_path):
    path = tmp_path / "experiment_manifest.json"
    initial = {"run_fingerprint_sha256": "a", "status": "preflight_complete"}
    depth_compare_job.load_or_create_run(path, initial)
    with pytest.raises(depth_compare_job.ExperimentError, match="different input/configuration fingerprint"):
        depth_compare_job.load_or_create_run(path, {"run_fingerprint_sha256": "b"})


def test_model_input_size_preserves_aspect_for_max_resolution(depth_compare_job):
    assert depth_compare_job.model_input_size(1920, 1080, None, 960) == (
        960,
        540,
        "aspect_preserving_max_resolution",
    )
    with pytest.raises(depth_compare_job.ExperimentError, match="mutually exclusive"):
        depth_compare_job.model_input_size(1920, 1080, (1280, 720), 960)


def test_da3_focal_uses_the_actual_canonical_depth_grid(depth_compare_job):
    calibration = depth_compare_job.Calibration(
        source_path="/private/calibration.json",
        source_sha256="a" * 64,
        condition_type=depth_compare_job.CONDITION_TYPE_APPROVED_CALIBRATION,
        camera_segment=None,
        approval_status=depth_compare_job.CALIBRATION_APPROVAL_STATUS,
        camera_setup_fingerprint_sha256="b" * 64,
        projection="dewarped_rectilinear",
        fx=1200.0,
        fy=1200.0,
        cx=960.0,
        cy=540.0,
        fov_x_deg=77.3,
        image_width=1920,
        image_height=1080,
    )

    focal = depth_compare_job.focal_for_depth_map(calibration, 1920, 1080, 1918, 1078)

    assert focal == pytest.approx((1200.0 * 1918 / 1920 + 1200.0 * 1078 / 1080) / 2)


def test_simple_camera_segments_condition_linear_groups_but_never_wide_stabilized_group(depth_compare_job):
    linear = depth_compare_job.metadata_conditioning_for_segment("1", 1920, 1080)
    wide = depth_compare_job.metadata_conditioning_for_segment("3", 1920, 1080)

    assert linear.condition_type == depth_compare_job.CONDITION_TYPE_METADATA_FOV
    assert linear.camera_segment == "1"
    assert linear.has_full_intrinsics
    assert linear.fx == pytest.approx(914.7, abs=1.0)
    assert depth_compare_job.metric_conditioning_status(linear)[0] == "metadata_fov_conditioned_exploratory_metres_estimate"
    # A segment is a simple explicit operator choice, not an approved
    # calibration that requires a private setup-fingerprint binding.
    depth_compare_job.require_calibration_setup_binding(
        linear, {"camera_setup_fingerprint_sha256": "unrelated-private-value"}
    )
    assert wide.condition_type == depth_compare_job.CONDITION_TYPE_METADATA_NO_PINHOLE
    assert not wide.has_full_intrinsics
    assert depth_compare_job.metric_conditioning_status(wide)[0] == "exploratory_metres_estimate_no_calibration"


def test_complete_run_needs_preview_and_report_evidence(depth_compare_job, monkeypatch, tmp_path):
    run_dir = tmp_path / "private-run"
    previews_dir = run_dir / "previews"
    previews_dir.mkdir(parents=True)
    preview_paths = {}
    for name in depth_compare_job.expected_preview_names(["da3"]):
        path = previews_dir / f"{name}.mp4"
        path.write_bytes(b"preview")
        preview_paths[name] = {
            "path": str(path),
            "sha256": depth_compare_job.sha256_file(path),
        }

    monkeypatch.setattr(depth_compare_job, "worker_results_are_complete", lambda *args: True)
    monkeypatch.setattr(
        depth_compare_job,
        "verify_preview_decode",
        lambda path, records: {"sha256": depth_compare_job.sha256_file(path)},
    )
    manifest = {
        "status": "complete",
        "run_fingerprint_sha256": "fingerprint",
        "previews": {"status": "complete", "artifacts": preview_paths},
        "report": {"status": "complete", "path": str(run_dir / "REPORT.md"), "sha256": "missing"},
    }

    assert not depth_compare_job.completed_run_is_valid(run_dir, manifest, _clock(depth_compare_job), ["da3"])


def test_da3_request_has_only_da3_inference_settings(depth_compare_job, tmp_path):
    request_path = depth_compare_job.build_worker_request(
        tmp_path / "private-run",
        "da3",
        tmp_path / "private-input.mp4",
        [],
        "fingerprint",
        None,
        (1920, 1080, "source_resolution"),
    )
    request = json.loads(request_path.read_text())

    assert "automatic" in request["inference_settings"]["precision_policy"]
    assert "keyframe_indices" not in request
    assert set(request) == {
        "schema_version", "worker_model", "run_dir", "input_path",
        "selected_source_clock", "run_fingerprint_sha256", "model",
        "calibration", "model_input_size", "inference_settings",
    }


def test_parser_does_not_offer_a_competing_model_or_comparison_options(depth_compare_job):
    with pytest.raises(SystemExit):
        depth_compare_job.build_parser().parse_args(["--models", "both"])
    with pytest.raises(SystemExit):
        depth_compare_job.build_parser().parse_args(["--export-point-clouds"])


def test_report_uses_terminal_complete_status(depth_compare_job, monkeypatch, tmp_path):
    summary = {
        "model_provenance": {
            "name": "Depth Anything 3 Metric Large",
            "huggingface_model_id": "depth-anything/DA3METRIC-LARGE",
            "resolved_revision": "a" * 40,
            "license": "Apache-2.0",
            "code_revision": "b" * 40,
        },
        "per_frame_latency_ms": {"median": 1.0, "p95": 2.0, "max": 3.0},
        "cuda_peak_memory_bytes": {"max_allocated": 1, "max_reserved": 2},
        "numeric_output_size_bytes": 3,
        "temporal_flicker_proxy": {"median_m": 0.1, "p95_m": 0.2, "pair_count": 1},
        "confidence": {
            "statuses": ["upstream_not_emitted_all_nan_sentinel"],
            "available_frame_count": 0,
            "unavailable_frame_count": 1,
        },
    }
    monkeypatch.setattr(depth_compare_job, "model_report_summary", lambda *args: summary)
    manifest = {
        "status": "complete",
        "selected_source_clock": _clock(depth_compare_job)[:1],
        "source_clock": _clock(depth_compare_job),
        "models": {"da3": {}},
        "input": {
            "sha256": "c" * 64,
            "resolution": {"width": 1920, "height": 1080},
            "frame_rate": {"avg_frame_rate": "30000/1001"},
        },
        "run_fingerprint": {"metric_preview_range_m": [0.2, 8.0]},
        "camera_calibration": depth_compare_job.metadata_conditioning_for_segment("1", 1920, 1080).to_manifest(),
        "privacy": {"private_unredacted_exception": False},
    }

    report_path = depth_compare_job.write_report(tmp_path, manifest, {}, set())

    report = report_path.read_text()
    assert "- Status: **complete**" in report
    assert "Recorded GoPro FOV from the selected camera segment" in report
