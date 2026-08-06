#!/usr/bin/env python3
"""Zero-setup demo. No API key, no GPU, no accounts, no Drive access.

    uv run scripts/demo.py

Runs the real pipeline (probe -> frame extraction -> MediaPipe hands ->
VLM captioning) against a bundled synthetic clip, using the deterministic
FakeBackend in place of a real API call. Reports what ran and what's still
pending (segmentation and preview rendering — see the module docstrings in
src/egoannote/layers/segment.py and pack/preview.py for why).

This is intentionally honest about scope: it is a plumbing smoke test, not
proof the annotation pipeline produces good captions or correct segments —
that requires real footage (Phase 0, currently blocked on Drive access) and
the Phase 0.5 prompt-accuracy gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from egoannote import config
from egoannote.backends.fake import FakeBackend
from egoannote.layers import caption as caption_layer
from egoannote.layers import hands as hands_layer
from egoannote.media.probe import probe
from egoannote.store import Store, write_hands_parquet_streaming


def _display(p: Path) -> str:
    """Repo-relative when it is inside the repo, absolute otherwise —
    EGOANNOTE_DATA_DIR can point anywhere, so relative_to() would raise."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    video = REPO_ROOT / "examples" / "synthetic_smoke_test.mp4"
    if not video.exists():
        print(f"error: demo clip not found at {video}")
        print("  This should be committed to the repo — if it's missing, something")
        print("  is wrong with your checkout, not with your setup.")
        return 2

    # Honor EGOANNOTE_DATA_DIR like the rest of the pipeline rather than
    # hardcoding a path (which is what left `config` imported-but-unused).
    run_dir = config.DATA_DIR / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    video_id = "synthetic_smoke_test"

    print(f"== probing {video.name} ==")
    info = probe(video)
    print(f"  {info.width}x{info.height} @ {info.fps:.2f}fps, {info.duration_sec:.1f}s"
          f"{' [VARIABLE FRAME RATE]' if info.is_vfr else ''}")

    print("== hands (MediaPipe, local CPU) ==")
    # Pass the SAME VideoInfo the caption track will use — both tracks must
    # derive timestamps from one clock or boundary metrics are meaningless.
    #
    # The hand track is streamed straight into the parquet writer rather
    # than materialized with list(...) first. hands_layer.run() is a
    # one-shot generator; write_hands_parquet_streaming consumes it once and
    # derives the no-hands gaps in the same pass (schema.GapTracker), so
    # peak memory stays flat regardless of video length instead of holding
    # every HandFrame (and then every Arrow column list) in memory at once.
    # A small counting wrapper keeps this script's own printed stats without
    # re-materializing anything.
    counts = {"total": 0, "present": 0}

    def _counted(frames):
        for rec in frames:
            counts["total"] += 1
            if rec.hands_present > 0:
                counts["present"] += 1
            yield rec

    hands_path = run_dir / "hands.parquet"
    _n_hand_frames, gaps = write_hands_parquet_streaming(
        _counted(hands_layer.run(video, video_id, config.MODELS_DIR, info=info)),
        hands_path,
    )
    print(f"  {counts['total']} frames sampled, {counts['present']} with a hand present"
          f" (0 expected — this clip is a synthetic test pattern, not real footage)")
    print(f"  wrote {_display(hands_path)} ({len(gaps)} no-hands gaps)")

    print("== captions (FakeBackend — no network, no key, $0) ==")
    fixture = REPO_ROOT / "examples" / "vlm_fixture.jsonl"
    backend = FakeBackend(fixture_path=fixture if fixture.exists() else None)
    if not fixture.exists():
        print("  no recorded fixture found — using the synthetic default response")
        print("  (see examples/README.md for how to record a real one)")
    store = Store(run_dir / "annotations.db")
    n = caption_layer.caption_video(video, video_id, backend, store)
    print(f"  wrote {n} window(s) to {_display(run_dir / 'annotations.db')}")

    print()
    print("== NOT yet run (see task list / plan for status) ==")
    print("  - segmentation (layers/segment.py) — pending real footage + Phase 0.5 gate")
    print("  - preview rendering (pack/preview.py) — not yet implemented")
    print()
    print("Demo complete. Real annotation quality cannot be judged from this run —")
    print("it only proves the plumbing runs end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
