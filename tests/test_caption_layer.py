"""Tests for the caption orchestration layer.

The resume logic here is the whole reason the storage layer was rewritten,
and it had NO test — test_store.py covered the Store primitive but nothing
proved caption_video actually consults it. Deleting the skip check would
have left every test green while silently re-billing every completed window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from egoannote.backends.base import SpendLimitExceeded, VLMResponse
from egoannote.layers import caption as capmod
from egoannote.media.probe import VideoInfo
from egoannote.store import Store


class _Backend:
    """Deterministic backend that can be told which windows to fail on."""

    name = "test"
    model_id = "test-model"

    def __init__(self, fail_on: set[int] | None = None, raises: Exception | None = None):
        self.fail_on = fail_on or set()
        self.raises = raises
        self.calls = 0
        self.seen: list[int] = []

    def caption(self, frames_jpeg, prompt) -> VLMResponse:
        idx = self.calls
        self.calls += 1
        self.seen.append(idx)
        if self.raises is not None:
            raise self.raises
        if idx in self.fail_on:
            raise RuntimeError("simulated per-window failure")
        return VLMResponse(
            text='{"actions":[{"start_frame":0,"end_frame":7,"task_step":"x"}]}',
            latency_ms=1,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )


@pytest.fixture
def stub_media(monkeypatch, tmp_path):
    """Replace ffprobe/ffmpeg so these tests need neither binary nor a video."""
    monkeypatch.setattr(
        capmod, "probe",
        lambda v: VideoInfo(
            duration_sec=18.0, width=640, height=360, fps=30.0, n_frames=540, is_vfr=False
        ),
    )

    def _extract(video, fps, out_dir, long_edge=None):
        paths = []
        for i in range(24):  # 24 frames / 8 per window = exactly 3 windows
            p = Path(out_dir) / f"frame_{i:06d}.jpg"
            p.write_bytes(b"jpeg")
            paths.append(p)
        return sorted(paths)

    monkeypatch.setattr(capmod, "extract_frames", _extract)


def test_all_windows_captioned_on_a_clean_run(stub_media, tmp_path):
    store = Store(tmp_path / "run.db")
    backend = _Backend()
    n = capmod.caption_video(Path("fake.mp4"), "v", backend, store, frames_dir=tmp_path / "frames")
    assert n == 3
    assert backend.calls == 3
    assert store.done_set(video_id="v", stage="caption", model_id="test-model") == {0, 1, 2}
    store.close()


def test_resume_skips_completed_and_retries_only_failures(stub_media, tmp_path):
    """The core regression guard: a second run must re-call the backend ONLY
    for the window that failed, and must overwrite rather than duplicate."""
    store = Store(tmp_path / "run.db")

    first = _Backend(fail_on={1})
    capmod.caption_video(Path("fake.mp4"), "v", first, store, frames_dir=tmp_path / "frames")
    assert first.calls == 3
    assert store.done_set(video_id="v", stage="caption", model_id="test-model") == {0, 2}

    second = _Backend()
    n = capmod.caption_video(Path("fake.mp4"), "v", second, store, frames_dir=tmp_path / "frames")
    assert second.calls == 1, "resume must re-call the backend only for the failed window"
    assert n == 1

    rows = list(store.iter_stage(video_id="v", stage="caption", model_id="test-model"))
    assert len(rows) == 3, "retry must overwrite the error row, never append a second"
    assert all(err is None for *_, err in rows)
    store.close()


def test_per_window_failure_is_persisted_and_run_continues(stub_media, tmp_path):
    store = Store(tmp_path / "run.db")
    backend = _Backend(fail_on={0})
    n = capmod.caption_video(Path("fake.mp4"), "v", backend, store, frames_dir=tmp_path / "frames")
    assert n == 3, "one bad window must not abort the whole run"

    errors = {idx: err for idx, _, _, err in
              store.iter_stage(video_id="v", stage="caption", model_id="test-model")}
    assert errors[0] is not None and "RuntimeError" in errors[0]
    assert errors[1] is None and errors[2] is None
    store.close()


def test_spend_limit_aborts_the_run_rather_than_error_rowing_every_window(
    stub_media, tmp_path
):
    """BUG: `except Exception` swallowed SpendLimitExceeded (a RuntimeError
    subclass), so a blown budget produced an error row per window and the run
    reported success. The operator had to read the error column to find out."""
    store = Store(tmp_path / "run.db")
    backend = _Backend(raises=SpendLimitExceeded("cap reached"))
    with pytest.raises(SpendLimitExceeded):
        capmod.caption_video(Path("fake.mp4"), "v", backend, store, frames_dir=tmp_path / "frames")
    assert backend.calls == 1, "budget exhaustion must stop the run immediately"
    store.close()


def test_zero_extracted_frames_is_not_a_silent_success(monkeypatch, tmp_path):
    """BUG: a truncated video yielded 0 frames -> 0 windows -> a clean
    'success' indistinguishable from 'already done', with no error row for a
    resume to ever retry. The video silently vanished from the dataset."""
    monkeypatch.setattr(
        capmod, "probe",
        lambda v: VideoInfo(0.5, 640, 360, 30.0, 15, False),
    )
    monkeypatch.setattr(capmod, "extract_frames", lambda *a, **k: [])

    store = Store(tmp_path / "run.db")
    with pytest.raises(RuntimeError, match="0 frames"):
        capmod.caption_video(Path("truncated.mp4"), "v", _Backend(), store, frames_dir=tmp_path / "frames")
    store.close()


def test_resume_reuses_cached_frames_instead_of_re_extracting(monkeypatch, tmp_path):
    """Measured before this fix: extracting 1.5h of 1080p cost ~14.5 min and
    up to 2.8 GB, into a TemporaryDirectory that was deleted on exit. A
    resume with 899/900 windows already done paid that entire cost again to
    make one API call. Frames now persist and are reused."""
    monkeypatch.setattr(
        capmod, "probe",
        lambda v: VideoInfo(18.0, 640, 360, 30.0, 540, False),
    )
    extractions = {"n": 0}

    def _extract(video, fps, out_dir, long_edge=None):
        extractions["n"] += 1
        paths = []
        for i in range(24):
            p = Path(out_dir) / f"frame_{i:06d}.jpg"
            p.write_bytes(b"jpeg")
            paths.append(p)
        return sorted(paths)

    monkeypatch.setattr(capmod, "extract_frames", _extract)

    frames = tmp_path / "frames"
    store = Store(tmp_path / "run.db")

    capmod.caption_video(Path("fake.mp4"), "v", _Backend(), store, frames_dir=frames)
    assert extractions["n"] == 1

    # Second pass: everything is already done, so no API calls — and crucially
    # no second extraction.
    second = _Backend()
    capmod.caption_video(Path("fake.mp4"), "v", second, store, frames_dir=frames)
    assert extractions["n"] == 1, "resume must not re-extract frames"
    assert second.calls == 0
    store.close()


def test_frame_cache_is_isolated_per_video_id(monkeypatch, tmp_path):
    """Two videos sharing one cache root must not read each other's frames."""
    monkeypatch.setattr(
        capmod, "probe",
        lambda v: VideoInfo(18.0, 640, 360, 30.0, 540, False),
    )
    seen_dirs = []

    def _extract(video, fps, out_dir, long_edge=None):
        seen_dirs.append(Path(out_dir))
        paths = []
        for i in range(8):
            p = Path(out_dir) / f"frame_{i:06d}.jpg"
            p.write_bytes(b"jpeg")
            paths.append(p)
        return sorted(paths)

    monkeypatch.setattr(capmod, "extract_frames", _extract)

    frames = tmp_path / "frames"
    store = Store(tmp_path / "run.db")
    capmod.caption_video(Path("a.mp4"), "vid-a", _Backend(), store, frames_dir=frames)
    capmod.caption_video(Path("b.mp4"), "vid-b", _Backend(), store, frames_dir=frames)

    assert len(seen_dirs) == 2, "distinct video_ids must each extract"
    assert seen_dirs[0] != seen_dirs[1]
    assert seen_dirs[0].name == "vid-a"
    assert seen_dirs[1].name == "vid-b"
    store.close()


def test_two_models_do_not_collide_on_the_same_window(stub_media, tmp_path):
    """The agreement metric depends on both models' results coexisting."""
    store = Store(tmp_path / "run.db")

    class _B(_Backend):
        def __init__(self, mid):
            super().__init__()
            self.model_id = mid

    capmod.caption_video(Path("fake.mp4"), "v", _B("model-a"), store, frames_dir=tmp_path / "frames")
    capmod.caption_video(Path("fake.mp4"), "v", _B("model-b"), store, frames_dir=tmp_path / "frames")

    assert store.done_set(video_id="v", stage="caption", model_id="model-a") == {0, 1, 2}
    assert store.done_set(video_id="v", stage="caption", model_id="model-b") == {0, 1, 2}
    all_rows = list(store.iter_stage(video_id="v", stage="caption", model_id=None))
    assert len(all_rows) == 6
    store.close()
