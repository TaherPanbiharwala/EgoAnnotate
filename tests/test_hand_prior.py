import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from egoannote import hand_prior


def _landmarks(x: float, y: float) -> list[SimpleNamespace]:
    # A compact, close hand near the lower-left edge.
    offsets = (
        (0, 0),
        (0.03, -0.02),
        (0.06, -0.03),
        (0.08, -0.02),
        (0.10, 0),
        (0.02, -0.05),
        (0.04, -0.08),
        (0.06, -0.10),
        (0.08, -0.11),
        (0, -0.06),
        (0.01, -0.10),
        (0.02, -0.13),
        (0.03, -0.15),
        (-0.02, -0.05),
        (-0.03, -0.09),
        (-0.04, -0.12),
        (-0.05, -0.14),
        (-0.04, -0.03),
        (-0.07, -0.06),
        (-0.09, -0.08),
        (-0.11, -0.09),
    )
    return [SimpleNamespace(x=x + dx, y=y + dy) for dx, dy in offsets]


def _hand(x: float = 0.10, y: float = 0.88) -> dict:
    return hand_prior.hand_region(_landmarks(x, y), width=100, height=100, score=0.9)


def test_hand_region_is_hand_only_and_requires_camera_near_scale() -> None:
    wearer_hand = _hand()
    distant_hand = hand_prior.hand_region(_landmarks(0.50, 0.45), width=100, height=100, score=0.9)

    assert wearer_hand["shape"] == "circle"
    assert wearer_hand["kind"] == "hand"
    assert len(wearer_hand["landmarks"]) == 21
    assert wearer_hand["wearer_candidate"] is True
    assert distant_hand["wearer_candidate"] is False


def test_stable_wearer_status_requires_temporal_hand_continuity() -> None:
    rows = [
        {"frame_idx": frame_idx, "hands": [_hand(0.10 + frame_idx / 10_000, 0.88)]}
        for frame_idx in (0, 2, 4, 6, 8)
    ]
    hand_prior._track_hands(rows, stride=2, width=100, height=100)

    assert {row["hands"][0]["track_id"] for row in rows} == {0}
    assert all(row["hands"][0]["stable_wearer_candidate"] for row in rows)

    brief = [{"frame_idx": frame_idx, "hands": [_hand()]} for frame_idx in (0, 2, 4, 6)]
    hand_prior._track_hands(brief, stride=2, width=100, height=100)
    assert not any(row["hands"][0]["stable_wearer_candidate"] for row in brief)

    interrupted = [
        {"frame_idx": frame_idx, "hands": [_hand()]} for frame_idx in (0, 2, 4, 6, 8, 10, 12)
    ]
    interrupted[3]["hands"][0]["wearer_candidate"] = False
    hand_prior._track_hands(interrupted, stride=2, width=100, height=100)
    assert not any(row["hands"][0]["stable_wearer_candidate"] for row in interrupted)


def _artifact(path: Path, *, source_sha256: str = "a" * 64, complete: bool = True) -> None:
    rows = [
        {"frame_idx": 0, "hands": [_hand()]},
        {"frame_idx": 2, "hands": [_hand()]},
    ]
    hand_prior._track_hands(rows, stride=2, width=100, height=100)
    if not complete:
        rows.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": hand_prior.SCHEMA_VERSION,
                "artifact_type": hand_prior.ARTIFACT_TYPE,
                "source": {
                    "sha256": source_sha256,
                    "width": 100,
                    "height": 100,
                    "fps": 20.0,
                    "n_frames": 4,
                },
                "sampling": {"detect_hz": 10.0},
                "model": {
                    "name": "mediapipe_hand_landmarker",
                    "url": hand_prior.HAND_MODEL_URL,
                    "sha256": hand_prior.HAND_MODEL_SHA256,
                },
                "configuration": dict(hand_prior.ACTIVE_SUPPRESSION_CONFIGURATION),
                "frames": rows,
            }
        ),
        encoding="utf-8",
    )


def test_hand_prior_loader_rejects_stale_or_incomplete_private_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "private" / "clip.hand_prior.json"
    _artifact(path)
    loaded = hand_prior.load_hand_prior(
        path,
        source_sha256="a" * 64,
        width=100,
        height=100,
        fps=20.0,
        n_frames=4,
        detect_hz=10.0,
    )
    assert set(loaded) == {0, 2}
    assert loaded[0]["stable_wearer"] == []  # two samples are not enough to act.

    with pytest.raises(ValueError, match="provenance does not match"):
        hand_prior.load_hand_prior(
            path,
            source_sha256="c" * 64,
            width=100,
            height=100,
            fps=20.0,
            n_frames=4,
            detect_hz=10.0,
        )
    _artifact(path, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        hand_prior.load_hand_prior(
            path,
            source_sha256="a" * 64,
            width=100,
            height=100,
            fps=20.0,
            n_frames=4,
            detect_hz=10.0,
        )


def test_hand_prior_paths_must_have_a_deliberate_private_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private or DO-NOT-SHIP"):
        hand_prior.require_private_path(tmp_path / "hand.json", option="--output")


def test_cached_hand_model_must_match_the_pinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "private" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / hand_prior.HAND_MODEL_FILENAME).write_bytes(b"unexpected model bytes")
    monkeypatch.setattr(hand_prior, "HAND_MODEL_MIN_BYTES", 1)

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        hand_prior.ensure_model(model_dir)
