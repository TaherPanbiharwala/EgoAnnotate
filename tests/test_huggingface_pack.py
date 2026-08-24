from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub
import pyarrow.parquet as pq

from egoannote.pack.huggingface import (
    build_video_bundle,
    export_captions,
    install_public_license,
    upload_bundle,
)
from egoannote.schema import WindowCaption
from egoannote.store import Store


def _write_dense_window(store: Store) -> None:
    rec = WindowCaption(
        video_id="vid",
        window_idx=0,
        start_ts_ms=0,
        end_ts_ms=6000,
        frame_indices=list(range(8)),
        is_partial=False,
        model_id="model-a",
        prompt_version="v4",
        prompt_hash="sha256:test",
        run_id="run-a",
        latency_ms=1,
        activity={
            "caption": "The person opens a jar.",
            "goal": "Remove the lid.",
            "phase": "executing",
            "progression": "continues",
        },
        actions=[
            {
                "start_frame": 0,
                "end_frame": 3,
                "action_caption": "The person twists the lid.",
                "task_step": "open_jar",
                "left_hand": {
                    "caption": "The left hand holds the jar.",
                    "verb": "holding",
                    "object": "jar",
                    "target": None,
                    "contact_type": "grip",
                    "visible": True,
                },
                "right_hand": {
                    "caption": "The right hand twists the lid.",
                    "verb": "twisting",
                    "object": "lid",
                    "target": None,
                    "contact_type": "twist",
                    "visible": True,
                },
                "tool_in_use": "none",
                "coordination": "coordinated",
                "handover_event": False,
                "scene_what": "The person opens a jar.",
                "scene_how": "Both hands apply force.",
                "scene_why": "To remove the lid.",
                "scene_location": "kitchen counter",
            }
        ],
        schema_ok=True,
    )
    store.write(
        video_id="vid",
        unit_idx=0,
        stage="caption",
        model_id="model-a",
        payload=rec.to_payload(),
    )


def test_export_produces_parallel_window_and_action_tables(tmp_path: Path) -> None:
    with Store(tmp_path / "annotations.db") as store:
        _write_dense_window(store)
        windows, actions, n_windows, n_actions = export_captions(
            store, video_id="vid", out_dir=tmp_path / "out"
        )
    assert (n_windows, n_actions) == (1, 1)
    window_row = pq.read_table(windows).to_pylist()[0]
    action_row = pq.read_table(actions).to_pylist()[0]
    assert window_row["activity_caption"] == "The person opens a jar."
    assert action_row["left_hand_caption"] == "The left hand holds the jar."
    assert action_row["right_hand_caption"] == "The right hand twists the lid."
    assert action_row["start_ms"] == 0
    assert action_row["end_ms"] == 3000


def test_public_manifest_never_contains_original_or_local_paths(tmp_path: Path) -> None:
    video = tmp_path / "clip.blurred.mp4"
    video.write_bytes(b"redacted-video")
    bundle_dir = tmp_path / "publish"
    with Store(tmp_path / "annotations.db") as store:
        _write_dense_window(store)
        runtime_manifest = build_video_bundle(
            store=store,
            video_id="vid",
            redacted_video=video,
            hands_parquet=None,
            bundle_dir=bundle_dir,
        )
    public_text = (bundle_dir / "manifests" / "vid.json").read_text(encoding="utf-8")
    public = json.loads(public_text)
    assert "local_source" not in public_text
    assert "original" not in public["redacted_video"]
    assert public["privacy"]["original_video_included"] is False
    assert runtime_manifest["_local_redacted_video"] == str(video.resolve())
    card = (bundle_dir / "README.md").read_text(encoding="utf-8")
    assert "license: other" in card
    assert (bundle_dir / "LICENSE").read_text(encoding="utf-8").startswith(
        "EGOANNOTE DATASET — PRIVATE PRERELEASE TERMS"
    )


def test_public_license_replaces_private_terms_and_card_name(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "publish"
    video = tmp_path / "clip.blurred.mp4"
    video.write_bytes(b"redacted-video")
    with Store(tmp_path / "annotations.db") as store:
        _write_dense_window(store)
        build_video_bundle(
            store=store,
            video_id="vid",
            redacted_video=video,
            hands_parquet=None,
            bundle_dir=bundle_dir,
        )
    approved = tmp_path / "terms.txt"
    approved.write_text("Approved terms", encoding="utf-8")
    install_public_license(bundle_dir, approved, "Egoannote Data Use Terms v1.0")
    assert (bundle_dir / "LICENSE").read_text(encoding="utf-8") == "Approved terms\n"
    assert 'license_name: "Egoannote Data Use Terms v1.0"' in (
        bundle_dir / "README.md"
    ).read_text(encoding="utf-8")


def test_upload_updates_visibility_for_an_existing_dataset(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("init", {"token": token}))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def update_repo_settings(self, **kwargs):
            calls.append(("update_repo_settings", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    upload_bundle(
        bundle_dir=tmp_path,
        repo_id="owner/dataset",
        video_manifests=[],
        private=False,
    )
    assert ("update_repo_settings", {
        "repo_id": "owner/dataset", "repo_type": "dataset", "private": False,
    }) in calls
