"""Windowing + VLM calls. Ties together media.frames, backends.base, and
parse.parse_caption_v4, writing WindowCaption records to the Store.

Resume semantics: Store.done_set() returns unit_idx values with no error, so
a re-run skips completed windows and retries failed ones — the retry
OVERWRITES the prior row (Store.write uses INSERT OR REPLACE), so there is
no possibility of the duplicate-record bug v1 had with append-mode JSONL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import config
from ..backends.base import PermanentBackendError, SpendLimitExceeded, VLMBackend
from ..media.frames import Window, extract_frames, iter_window_indices
from ..media.probe import VideoInfo, probe
from ..parse import parse_caption_v4
from ..schema import WindowCaption, hash_prompt, utc_now_isoformat
from ..store import Store

log = logging.getLogger(__name__)


# This is intentionally a separate, private derived record.  It never
# replaces the frame-grounded window responses, which remain the source of
# truth for every caption event and for later error review.
_SEGMENT_SUMMARY_STAGE = "caption_segment_summary"
_SEGMENT_SUMMARY_VERSION = "v1"
_SEGMENT_SUMMARY_MAX_SOURCE_CHARS = 100_000


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

    The DURATION needs the same treatment and originally didn't get it: the
    prompt hardcoded "6 seconds" while only the frame counts were templated,
    so a 3-frame trailing window was told it covered 6 seconds when it
    covers 2.25. Exactly the bug this function was written to fix, one line
    further down the same file.
    """
    if n_frames < 1:
        raise ValueError(f"cannot render a caption prompt for {n_frames} frames")
    window_seconds = n_frames / float(config.VLM_FPS)
    # "6" not "6.0" for the common whole-second case — the prompt reads as
    # instructions to a model, not as a float dump.
    seconds_str = (f"{window_seconds:g}" if window_seconds % 1 else
                   str(int(window_seconds)))
    return (
        template
        .replace("{n_frames}", str(n_frames))
        .replace("{max_frame_idx}", str(n_frames - 1))
        .replace("{window_seconds}", seconds_str)
    )


def _summary_observations(captions: list[dict]) -> dict:
    """Make a bounded, deterministic text-only evidence packet for a summary call.

    A segment summary must be grounded in the saved visual-window observations;
    it must not be a new, untraceable interpretation of the original video.
    Deliberately omit verbose scene prose here: task steps, atomic captions,
    hands, and timing are the material needed to summarize the procedure.
    """
    windows: list[dict] = []
    for caption in sorted(captions, key=lambda item: int(item["window_idx"])):
        actions: list[dict] = []
        for action in caption.get("actions", []):
            if not isinstance(action, dict):
                continue
            actions.append(
                {
                    "start_frame": action.get("start_frame"),
                    "end_frame": action.get("end_frame"),
                    "task_step": action.get("task_step"),
                    "action_caption": action.get("action_caption"),
                    "left_hand": action.get("left_hand"),
                    "right_hand": action.get("right_hand"),
                    "tool_in_use": action.get("tool_in_use"),
                    "coordination": action.get("coordination"),
                    "handover_event": action.get("handover_event"),
                }
            )
        windows.append(
            {
                "window_idx": caption.get("window_idx"),
                "start_ts_ms": caption.get("start_ts_ms"),
                "end_ts_ms": caption.get("end_ts_ms"),
                "activity": (caption.get("activity") or {}).get("caption"),
                "actions": actions,
            }
        )
    return {"windows": windows}


def _summary_source_hash(observations: dict) -> str:
    encoded = json.dumps(observations, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _segment_summary_prompt(observations: dict) -> str:
    evidence = json.dumps(observations, ensure_ascii=False, separators=(",", ":"))
    if len(evidence) > _SEGMENT_SUMMARY_MAX_SOURCE_CHARS:
        raise ValueError(
            "caption segment is too long for one source-grounded summary call; "
            "split the curated original at a natural cut before captioning"
        )
    return (
        "You are consolidating already-saved, frame-grounded observations from a "
        "single uninterrupted first-person video segment. Do not add facts that are "
        "not in these observations. Do not mention a window number, frame number, "
        "or missing information. Describe the camera wearer, not an unnamed person.\n\n"
        "Return one JSON object only, with no markdown:\n"
        '{"summary":"one to three concise sentences",'
        '"steps":["short ordered visible procedure step", "..."]}\n\n'
        "The summary describes the overall visible activity. steps are ordered, "
        "deduplicated procedure-level steps supported by the observations; do not "
        "turn a continuing action at a six-second boundary into a new step.\n\n"
        f"OBSERVATIONS={evidence}"
    )


def _parse_segment_summary(raw: str) -> dict[str, object]:
    """Validate the deliberately small summary response instead of trusting text."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.strip().removesuffix("```").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PermanentBackendError("caption segment summary was not valid JSON") from exc
    if not isinstance(value, dict):
        raise PermanentBackendError("caption segment summary must be a JSON object")
    summary = value.get("summary")
    steps = value.get("steps")
    if not isinstance(summary, str) or not summary.strip() or len(summary.strip()) > 1_200:
        raise PermanentBackendError("caption segment summary has no usable summary text")
    if not isinstance(steps, list) or not steps or len(steps) > 32:
        raise PermanentBackendError("caption segment summary has no usable ordered steps")
    normalized_steps = [step.strip() for step in steps if isinstance(step, str) and step.strip()]
    if len(normalized_steps) != len(steps) or any(len(step) > 300 for step in normalized_steps):
        raise PermanentBackendError("caption segment summary contains invalid ordered steps")
    return {"summary": summary.strip(), "steps": normalized_steps}


def summarize_caption_windows(
    captions: list[dict],
    *,
    backend: VLMBackend,
    store: Store,
    video_id: str,
    run_id: str,
) -> tuple[dict, bool]:
    """Create or reuse one source-bound, text-only summary for a full segment.

    The bool is true only when an API call was made.  The cache key is the
    deterministic hash of all complete raw captions, so a changed raw caption
    cannot silently inherit an old summary.
    """
    if not captions:
        raise ValueError("cannot summarize an empty caption segment")
    observations = _summary_observations(captions)
    source_sha256 = _summary_source_hash(observations)
    cached = store.read(
        video_id=video_id,
        unit_idx=0,
        stage=_SEGMENT_SUMMARY_STAGE,
        model_id=backend.model_id,
    )
    if (
        isinstance(cached, dict)
        and cached.get("summary_version") == _SEGMENT_SUMMARY_VERSION
        and cached.get("source_caption_sha256") == source_sha256
        and isinstance(cached.get("summary"), str)
        and isinstance(cached.get("steps"), list)
    ):
        return cached, False

    response = backend.caption([], _segment_summary_prompt(observations))
    parsed = _parse_segment_summary(response.text)
    payload = {
        "summary_version": _SEGMENT_SUMMARY_VERSION,
        "source_caption_sha256": source_sha256,
        "source_window_count": len(captions),
        "model_id": backend.model_id,
        "run_id": run_id,
        "summary": parsed["summary"],
        "steps": parsed["steps"],
        "latency_ms": response.latency_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "provider": response.provider,
    }
    store.write(
        video_id=video_id,
        unit_idx=0,
        stage=_SEGMENT_SUMMARY_STAGE,
        model_id=backend.model_id,
        payload=payload,
    )
    return payload, True


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
        parsed = parse_caption_v4(resp.text, n_frames=n_in_window)

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
            activity=parsed["activity"],
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
    window_indices: set[int] | None = None,
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
        window_indices: optional exact subset for a pilot. A later call with
            ``None`` resumes the same store and fills every remaining window.

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

    # A resume is safe only when it resumes the same prompt contract.  The
    # primary key intentionally has no prompt column (a retry must overwrite
    # its old row), so inspect successful saved rows before treating them as
    # done.  Otherwise a pilot made with an earlier prompt could silently be
    # mixed into a V5 event timeline.
    for _index, _model, payload, error in store.iter_stage(
        video_id=video_id, stage="caption", model_id=backend.model_id
    ):
        if error is None and payload.get("prompt_version") != config.CAPTION_PROMPT_VERSION:
            raise ValueError(
                f"caption resume for {video_id} mixes prompt versions "
                f"{payload.get('prompt_version')!r} and {config.CAPTION_PROMPT_VERSION!r}; "
                "use a fresh --run-dir for the new prompt"
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
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = sorted(window_cache.glob("frame_*.jpg"))
    marker = window_cache / "_COMPLETE.json"
    source_stat = video.stat() if video.exists() else None
    source_size = source_stat.st_size if source_stat else None
    source_mtime_ns = source_stat.st_mtime_ns if source_stat else None
    cache_complete = False
    if marker.exists():
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            cache_complete = (
                metadata.get("frame_count") == len(cached)
                and metadata.get("source_size") == source_size
                and metadata.get("source_mtime_ns") == source_mtime_ns
            )
        except (json.JSONDecodeError, OSError):
            cache_complete = False

    if cached and cache_complete:
        log.info(
            "caption_video: reusing %d cached frames in %s (delete that "
            "directory to force re-extraction)", len(cached), window_cache,
        )
        frame_paths = cached
    else:
        # An interrupted eager extraction can leave a non-empty directory.
        # Non-empty is not proof of completeness: using it would silently
        # truncate the video to however many JPEGs landed before the crash.
        # Extract to a sibling and atomically swap it in only after a marker
        # recording the complete count and source identity has been written.
        partial_cache = cache_root / f".{video_id}.partial"
        if partial_cache.exists():
            shutil.rmtree(partial_cache)
        partial_cache.mkdir(parents=True)
        extracted = extract_frames(
            video,
            fps=config.VLM_FPS_STR,
            out_dir=partial_cache,
            long_edge=config.VLM_FRAME_RESIZE_LONG_EDGE,
        )
        if not extracted:
            shutil.rmtree(partial_cache)
            frame_paths = []
        else:
            (partial_cache / "_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "frame_count": len(extracted),
                        "source_size": source_size,
                        "source_mtime_ns": source_mtime_ns,
                    }
                ),
                encoding="utf-8",
            )
            if window_cache.exists():
                shutil.rmtree(window_cache)
            partial_cache.replace(window_cache)
            frame_paths = sorted(window_cache.glob("frame_*.jpg"))

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

    all_windows = list(iter_window_indices(n_frames, config.VLM_WINDOW_FRAMES))
    if window_indices is not None:
        available = {w.window_idx for w in all_windows}
        missing = window_indices - available
        if missing:
            raise ValueError(
                f"requested caption window(s) {sorted(missing)} do not exist; "
                f"available range is 0..{max(available)}"
            )
    todo = [
        w for w in all_windows
        if w.window_idx not in done
        and (window_indices is None or w.window_idx in window_indices)
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
