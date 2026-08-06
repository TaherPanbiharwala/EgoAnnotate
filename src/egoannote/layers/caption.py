"""Windowing + VLM calls. Ties together media.frames, backends.base, and
parse.parse_caption_v3, writing WindowCaption records to the Store.

Resume semantics: Store.done_set() returns unit_idx values with no error, so
a re-run skips completed windows and retries failed ones — the retry
OVERWRITES the prior row (Store.write uses INSERT OR REPLACE), so there is
no possibility of the duplicate-record bug v1 had with append-mode JSONL.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import config
from ..backends.base import PermanentBackendError, SpendLimitExceeded, VLMBackend
from ..media.frames import Window, extract_frames, iter_window_indices
from ..media.probe import VideoInfo, probe
from ..parse import parse_caption_v3
from ..schema import WindowCaption, hash_prompt, utc_now_isoformat
from ..store import Store

log = logging.getLogger(__name__)


def _load_prompt() -> tuple[str, str]:
    """Load the prompt template and hash it.

    The hash is of the TEMPLATE, not the rendered text, so every window in a
    run shares one prompt_hash regardless of window length — that is what
    makes prompt_hash usable for grouping a run's records.
    """
    try:
        text = config.CAPTION_PROMPT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Caption prompt '{config.CAPTION_PROMPT_VERSION}' not found at "
            f"{config.CAPTION_PROMPT_FILE}.\n"
            f"  If you installed egoannote as a package, prompts/ is not bundled "
            f"yet — run from a repo checkout instead.\n"
            f"  If you're in a checkout, the file is missing or was renamed."
        ) from e
    return text, hash_prompt(text)


def _render_prompt(template: str, n_frames: int) -> str:
    """Fill the frame-count placeholders for THIS window.

    Not str.format() — the prompt body contains a literal JSON example full
    of braces, which format() would try to interpret.

    Why per-window and not once per run: trailing windows are short
    (media.frames.Window.is_partial). Sending the 8-frame prompt with 3
    images tells the model to answer `end_frame=7` for images 3-7 that were
    never sent, and those indices then define segment boundaries. Found in
    review; the mismatch hits every video whose frame count isn't an exact
    multiple of the window size.
    """
    if n_frames < 1:
        raise ValueError(f"cannot render a caption prompt for {n_frames} frames")
    return (
        template
        .replace("{n_frames}", str(n_frames))
        .replace("{max_frame_idx}", str(n_frames - 1))
    )


class _RunAborted(Exception):
    """Internal only — never escapes caption_video. Raised by
    _run_window_with_abort when a window is skipped because a prior
    control-flow exception already tripped the abort flag."""


def _caption_one_window(
    window: Window,
    frame_paths: list[Path],
    backend: VLMBackend,
    prompt_text: str,
    prompt_hash: str,
    video_id: str,
    run_id: str,
) -> WindowCaption:
    """Do the VLM call + parse for one window. No store access — safe to
    call from any thread; the caller does all writes. Control-flow
    exceptions (SpendLimitExceeded, PermanentBackendError, KeyboardInterrupt)
    propagate; anything else is captured into an error record instead of
    raising, so one bad window doesn't abort the run.
    """
    start_ts_ms = round(window.frame_indices[0] * 1000 / config.VLM_FPS)
    end_ts_ms = round((window.frame_indices[-1] + 1) * 1000 / config.VLM_FPS)

    try:
        jpegs = [frame_paths[i].read_bytes() for i in window.frame_indices]
        # Render for THIS window's real length, and parse against the same
        # number, so a partial window can never yield an action indexing a
        # frame that was not sent.
        n_in_window = len(jpegs)
        resp = backend.caption(jpegs, _render_prompt(prompt_text, n_in_window))
        parsed = parse_caption_v3(resp.text, n_frames=n_in_window)

        return WindowCaption(
            video_id=video_id,
            window_idx=window.window_idx,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
            frame_indices=window.frame_indices,
            is_partial=window.is_partial,
            model_id=backend.model_id,
            prompt_version=config.CAPTION_PROMPT_VERSION,
            prompt_hash=prompt_hash,
            run_id=run_id,
            latency_ms=resp.latency_ms,
            actions=parsed["actions"],
            raw_json=parsed["raw_json"],
            schema_ok=parsed["_schema_ok"],
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            provider=resp.provider,
        )
    except (SpendLimitExceeded, PermanentBackendError, KeyboardInterrupt):
        # Control-flow signals, not per-window data problems. Writing an
        # error row for every remaining window and returning normally would
        # make a blown budget or a bad API key look like "the run finished"
        # — the operator would have to read the error column to discover why
        # every window is empty.
        raise
    except Exception as e:
        log.exception("caption_video: window %d failed", window.window_idx)
        return WindowCaption(
            video_id=video_id,
            window_idx=window.window_idx,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
            frame_indices=window.frame_indices,
            is_partial=window.is_partial,
            model_id=backend.model_id,
            prompt_version=config.CAPTION_PROMPT_VERSION,
            prompt_hash=prompt_hash,
            run_id=run_id,
            latency_ms=0,
            actions=[],
            error=f"{type(e).__name__}: {e}",
        )


def _run_window_with_abort(
    window: Window,
    frame_paths: list[Path],
    backend: VLMBackend,
    prompt_text: str,
    prompt_hash: str,
    video_id: str,
    run_id: str,
    abort_event: threading.Event,
) -> WindowCaption:
    """Wraps _caption_one_window with a self-checked abort flag.

    Why this exists on top of the executor's own .cancel(): .cancel() only
    stops a future that HASN'T STARTED, and only if the main thread gets to
    call it before a worker claims that future. With max_workers=1 and a
    backend that fails instantly (a bad API key returns 401 in
    microseconds — a realistic case, not a test artifact), the single
    worker thread can pull the NEXT queued task and start running it before
    the main thread's as_completed loop even gets scheduled to react to the
    first failure. Verified empirically: relying on cancel() alone let a
    second call through under exactly this condition.

    The fix: the thread that JUST failed sets abort_event itself, in its own
    except clause, before re-raising — not the main thread, and not on a
    delay. For max_workers=1 that is the same thread that will next pull the
    following queued item, so the check-and-skip is sequential within one
    thread with no cross-thread race: `abort_event.is_set()` is guaranteed
    to already be True by the time that thread looks at it again. For
    max_workers>1 this doesn't eliminate the (documented, intentional)
    bounded overrun from OTHER threads already in flight — it only adds a
    second, tighter backstop on top of .cancel() for whatever hasn't been
    claimed yet.
    """
    if abort_event.is_set():
        raise _RunAborted()
    try:
        return _caption_one_window(
            window, frame_paths, backend, prompt_text, prompt_hash, video_id, run_id,
        )
    except (SpendLimitExceeded, PermanentBackendError, KeyboardInterrupt):
        abort_event.set()
        raise


def caption_video(
    video: Path,
    video_id: str,
    backend: VLMBackend,
    store: Store,
    *,
    run_id: str | None = None,
    frames_dir: Path | None = None,
    max_workers: int = config.CAPTION_MAX_WORKERS,
) -> int:
    """Caption every window of `video` with `backend`, writing to `store`
    under stage="caption", model_id=backend.model_id. Returns count written
    (including error rows).

    Args:
        frames_dir: root for the extracted-frame cache. Frames land in
            `frames_dir / video_id`. Defaults to `config.DATA_DIR / "frames"`.
            Pass an explicit path to isolate concurrent runs or tests — the
            cache is keyed on video_id alone, so two callers sharing a root
            and a video_id share frames.
        max_workers: how many VLM calls run concurrently (see
            config.CAPTION_MAX_WORKERS for the measured rationale). Windows
            complete out of order under concurrency, but each is still
            written to the store as soon as it finishes, so a resume after
            an abort skips exactly the windows already done — same
            guarantee as serial execution, just not the same completion
            order. Pass 1 for strictly serial, deterministic-order
            execution.

    Concurrency and control-flow exceptions: once any window raises
    SpendLimitExceeded / PermanentBackendError / KeyboardInterrupt, no NEW
    windows are started — pending futures are cancelled — but up to
    `max_workers` windows already in flight are allowed to finish (each one
    independently re-checks the backend's shared, thread-safe spend/permanent
    state and fails the same way with no HTTP call if the condition still
    holds). This is a deliberately bounded overrun, not the exactly-one-call
    guarantee serial execution gives — call with max_workers=1 if you need
    that guarantee (the test suite does, for exact call-count assertions).
    """
    prompt_text, prompt_hash = _load_prompt()
    run_id = run_id or utc_now_isoformat()
    info: VideoInfo = probe(video)
    if info.is_vfr:
        log.warning(
            "caption_video: %s is variable frame rate (may cost ~1 frame of "
            "boundary precision in downstream segmentation)", video_id,
        )

    done = store.done_set(video_id=video_id, stage="caption", model_id=backend.model_id)
    n_written = 0

    # Frames are cached on disk rather than extracted into a
    # TemporaryDirectory. Measured on this machine, eager-extract-to-tempdir
    # cost ~14.5 min and 1.35-2.8 GB for 1.5h of 1080p BEFORE the first
    # caption, and because TemporaryDirectory deletes on exit, a resume with
    # 899 of 900 windows done paid the whole cost again to make one API call.
    #
    # Not a full chunked-extraction rewrite: that changes the ffmpeg
    # invocation per window and needs its own correctness work on seek
    # accuracy. Caching removes the repeat cost, which is the part that bites
    # during iteration.
    cache_root = frames_dir if frames_dir is not None else config.DATA_DIR / "frames"
    window_cache = cache_root / video_id
    window_cache.mkdir(parents=True, exist_ok=True)
    cached = sorted(window_cache.glob("frame_*.jpg"))

    if cached:
        log.info(
            "caption_video: reusing %d cached frames in %s (delete that "
            "directory to force re-extraction)", len(cached), window_cache,
        )
        frame_paths = cached
    else:
        frame_paths = extract_frames(
            video,
            fps=config.VLM_FPS_STR,
            out_dir=window_cache,
            long_edge=config.VLM_FRAME_RESIZE_LONG_EDGE,
        )

    n_frames = len(frame_paths)
    if n_frames == 0:
        # Without this, a truncated or unsupported-codec video produces a
        # clean "0 windows written" that is indistinguishable from "already
        # fully captioned" — and since no error row is stored, no resume
        # ever retries it. The video just silently vanishes from the dataset.
        raise RuntimeError(
            f"ffmpeg extracted 0 frames from {video} (probed as "
            f"{info.duration_sec:.1f}s @ {info.fps:.2f}fps). The file is "
            f"likely truncated or uses a codec ffmpeg can't decode."
        )

    todo = [
        w for w in iter_window_indices(n_frames, config.VLM_WINDOW_FRAMES)
        if w.window_idx not in done
    ]

    # ThreadPoolExecutor(max_workers=1) is not a special case here: with one
    # worker thread, tasks run strictly in submission order (a single thread
    # pulling a FIFO queue can't do otherwise), so it gives the exact same
    # ordering as a hand-written serial loop, with negligible pool overhead.
    # The abort_event (see _run_window_with_abort) is what makes the
    # exact-one-extra-call guarantee hold even for an instantly-failing
    # backend — .cancel() alone races and loses that guarantee.
    abort_event = threading.Event()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_window_with_abort, window, frame_paths, backend,
                prompt_text, prompt_hash, video_id, run_id, abort_event,
            ): window
            for window in todo
        }
        control_exc: BaseException | None = None

        for future in as_completed(futures):
            try:
                rec = future.result()
            except (CancelledError, _RunAborted):
                # Never ran — cancelled before a worker claimed it, or
                # skipped via abort_event because a sibling window's
                # control-flow exception fired first.
                continue
            except (SpendLimitExceeded, PermanentBackendError, KeyboardInterrupt) as e:
                if control_exc is None:
                    control_exc = e
                    window = futures[future]
                    log.warning(
                        "caption_video: %s aborting after window %d — %s: %s. "
                        "Windows already in flight may still complete; no new "
                        "ones will start.",
                        video_id, window.window_idx, type(e).__name__, e,
                    )
                    for f in futures:
                        f.cancel()  # no-op for already-running/finished futures
                continue

            store.write(
                video_id=video_id,
                unit_idx=rec.window_idx,
                stage="caption",
                model_id=backend.model_id,
                payload=rec.to_payload(),
                error=rec.error,
            )
            n_written += 1

        if control_exc is not None:
            raise control_exc

    return n_written
