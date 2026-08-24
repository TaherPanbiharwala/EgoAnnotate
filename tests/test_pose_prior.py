import json
from pathlib import Path

import numpy as np
import pytest

from egoannote import pose_prior


def _artifact(
    path: Path,
    *,
    source_sha256: str = "a" * 64,
    width: int = 64,
    detect_hz: float = 10.0,
    model_sha256: str = "b" * 64,
    complete: bool = True,
) -> None:
    frames = [
        {"frame_idx": 0, "limb_regions": [], "wearer_limb_regions": []},
        {"frame_idx": 2, "limb_regions": [], "wearer_limb_regions": []},
    ]
    if not complete:
        frames.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": pose_prior.SCHEMA_VERSION,
                "artifact_type": pose_prior.ARTIFACT_TYPE,
                "source": {
                    "sha256": source_sha256,
                    "width": width,
                    "height": 32,
                    "fps": 20.0,
                    "n_frames": 4,
                },
                "sampling": {"detect_hz": detect_hz},
                "model": {"sha256": model_sha256},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def test_limb_masks_do_not_require_or_use_face_landmarks() -> None:
    landmarks = {
        "11": [0.1, 0.2, 0.9, 0.9],
        "13": [0.3, 0.3, 0.9, 0.9],
        "15": [0.5, 0.4, 0.9, 0.9],
        "17": [0.55, 0.42, 0.9, 0.9],
        "19": [0.52, 0.45, 0.9, 0.9],
        "21": [0.48, 0.45, 0.9, 0.9],
    }
    regions = pose_prior.limb_regions(landmarks, width=100, height=100)

    assert regions
    assert {region["kind"] for region in regions} <= {"arm", "hand", "leg", "foot"}
    assert all("face" not in region["kind"] and "torso" not in region["kind"] for region in regions)


def test_overlap_preserves_unknown_when_no_reliable_wearer_regions() -> None:
    assert pose_prior.limb_overlap_fraction((0, 0, 10, 10), []) is None
    circle = {"shape": "circle", "kind": "hand", "center": [5, 5], "radius": 20}
    assert pose_prior.limb_overlap_fraction((0, 0, 10, 10), [circle]) == pytest.approx(1.0)


def test_pose_preview_draws_limb_only_colours_and_status_label() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    poses = [
        {
            "wearer_candidate": True,
            "limb_regions": [
                {"shape": "circle", "kind": "hand", "center": [30, 40], "radius": 10}
            ],
        },
        {
            "wearer_candidate": False,
            "limb_regions": [
                {"shape": "circle", "kind": "hand", "center": [80, 40], "radius": 10}
            ],
        },
    ]

    pose_prior.draw_pose_preview(frame, poses, frame_idx=12, sampled_this_frame=True)

    assert np.any(frame == pose_prior.POSE_PREVIEW_WEARER_BGR)
    assert np.any(frame == pose_prior.POSE_PREVIEW_OTHER_BGR)
    assert frame[:32].any()  # status banner makes sample-versus-held state visible


def test_pose_prior_loader_rejects_stale_or_incomplete_private_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "private" / "clip.pose_prior.json"
    _artifact(path)
    loaded = pose_prior.load_pose_prior(
        path,
        source_sha256="a" * 64,
        width=64,
        height=32,
        fps=20.0,
        n_frames=4,
        detect_hz=10.0,
    )
    assert set(loaded) == {0, 2}
    for kwargs in (
        {"source_sha256": "c" * 64},
        {"width": 128},
        {"detect_hz": 5.0},
    ):
        expected = {
            "source_sha256": "a" * 64,
            "width": 64,
            "height": 32,
            "fps": 20.0,
            "n_frames": 4,
            "detect_hz": 10.0,
        }
        expected.update(kwargs)
        with pytest.raises(ValueError, match="provenance does not match"):
            pose_prior.load_pose_prior(path, **expected)

    _artifact(path, model_sha256="not-a-sha256")
    with pytest.raises(ValueError, match="provenance does not match"):
        pose_prior.load_pose_prior(
            path,
            source_sha256="a" * 64,
            width=64,
            height=32,
            fps=20.0,
            n_frames=4,
            detect_hz=10.0,
        )

    _artifact(path, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        pose_prior.load_pose_prior(
            path,
            source_sha256="a" * 64,
            width=64,
            height=32,
            fps=20.0,
            n_frames=4,
            detect_hz=10.0,
        )


def test_pose_prior_paths_must_have_a_deliberate_private_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private or DO-NOT-SHIP"):
        pose_prior.require_private_path(tmp_path / "pose.json", option="--output")
