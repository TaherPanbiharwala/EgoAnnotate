"""Windowing + VLM calls. Ties together media.frames, backends.base, and
parse.parse_caption_v3, writing WindowCaption records to the Store.

Resume semantics: Store.done_set() returns unit_idx values with no error, so
a re-run skips completed windows and retries failed ones — the retry
OVERWRITES the prior row (Store.write uses INSERT OR REPLACE), so there is
no possibility of the duplicate-record bug v1 had with append-mode JSONL.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import config
from ..backends.base import PermanentBackendError, SpendLimitExceeded, VLMBackend
from ..media.frames import extract_frames, iter_window_indices
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


def caption_video(
    video: Path,
    video_id: str,
    backend: VLMBackend,
    store: Store,
    *,
    run_id: str | None = None,
    frames_dir: Path | None = None,
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

    for window in iter_window_indices(n_frames, config.VLM_WINDOW_FRAMES):
        if window.window_idx in done:
            continue

        start_ts_ms = round(window.frame_indices[0] * 1000 / config.VLM_FPS)
        end_ts_ms = round((window.frame_indices[-1] + 1) * 1000 / config.VLM_FPS)

        try:
            jpegs = [frame_paths[i].read_bytes() for i in window.frame_indices]
            # Render for THIS window's real length, and parse against the
            # same number, so a partial window can never yield an action
            # indexing a frame that was not sent.
            n_in_window = len(jpegs)
            resp = backend.caption(jpegs, _render_prompt(prompt_text, n_in_window))
            parsed = parse_caption_v3(resp.text, n_frames=n_in_window)

            rec = WindowCaption(
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
            # Control-flow signals, not per-window data problems. Writing
            # an error row for every remaining window and returning
            # normally would make a blown budget or a bad API key look
            # like "the run finished" — the operator would have to read
            # the error column to discover why every window is empty.
            raise
        except Exception as e:
            log.exception("caption_video: window %d failed", window.window_idx)
            rec = WindowCaption(
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

        store.write(
            video_id=video_id,
            unit_idx=window.window_idx,
            stage="caption",
            model_id=backend.model_id,
            payload=rec.to_payload(),
            error=rec.error,
        )
        n_written += 1

    return n_written
