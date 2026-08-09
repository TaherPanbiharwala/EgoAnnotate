#!/usr/bin/env python3
"""Turn a blur job's flagged coordinates into something a human can actually look at.

    uv run scripts/22_blur_review.py runs/blur/GX010042.manifest.json
    uv run scripts/22_blur_review.py runs/blur/            # every manifest in a dir
    uv run scripts/22_blur_review.py runs/blur/ --timelapse

jobs/10_blur_egoblur.py marks a clip NEEDS_REVIEW when the low-threshold
sweep or the independent YuNet pass finds a region that may contain a face
nothing redacted. It reports those as raw `[x1,y1,x2,y2]` coordinates at a
frame index — which is unreviewable by hand. This extracts each flagged
region as an image and builds one scrollable HTML page.

CROPS COME FROM THE BLURRED OUTPUT, NEVER THE ORIGINAL. That is not a
limitation, it is the point: the question being reviewed is "does the file
I am about to publish have a visible face at this spot?" If the detector
missed a face, those pixels are unchanged in the output, so the crop shows
it — which is the finding. And since every pixel here is one that already
ships in the released video, reviewing it adds no exposure that publishing
would not.

Deliberately NOT in scope: recording your approve/reject verdict. That
belongs with the laptop-side ingest that owns the SQLite database (the
pipeline's single source of truth); this script only ever reads. Nothing
here writes to the blur job's output files either — review artifacts go to
a separate review/ directory.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Context around a flagged box, as a multiple of its size. A bare box crop
# of a 40px face is unidentifiable — you need the surrounding shoulders and
# background to judge "is that a person or a doorknob".
CROP_CONTEXT_SCALE = 3.0
CROP_MIN_PX = 160  # upscale tiny crops so they're viewable without zooming
TIMELAPSE_SAMPLE_FPS = 1
TIMELAPSE_PLAY_FPS = 15
TIMELAPSE_WIDTH = 960


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"error: {name} not found on PATH (macOS: brew install ffmpeg)")
    return path


def find_manifests(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        found = sorted(target.glob("*.manifest.json"))
        if not found:
            sys.exit(f"error: no *.manifest.json under {target}")
        return found
    sys.exit(f"error: {target} does not exist")


def collect_flags(audit: dict) -> list[dict]:
    """Merge both flag sources into one list sorted by descending score.

    They have deliberately different meanings and different shapes, so the
    origin is kept: a YuNet hit is a *different model* independently seeing
    a face the pipeline did not redact, which is stronger evidence than
    EgoBlur's own sub-threshold guess. Review those first.
    """
    flags = []
    # .get with defaults on score, not [...]: one malformed entry must not
    # kill the whole review run for a clip that has real findings in it.
    for c in audit.get("candidates", []):
        flags.append({
            "origin": "sweep",
            "frame_idx": int(c["frame_idx"]),
            "box": [float(v) for v in c["box"]],
            "score": float(c.get("score") or 0.0),
            "cls": c.get("cls", "face"),
        })
    for y in audit.get("yunet_uncovered", []):
        flags.append({
            "origin": "yunet",
            "frame_idx": int(y["frame_idx"]),
            "box": [float(v) for v in y["box"]],
            "score": float(y.get("score") or 0.0),
            "cls": "face",
        })
    # YuNet first at equal score — an independent detector agreeing that
    # something is a face outranks EgoBlur's own low-confidence guess.
    flags.sort(key=lambda f: (f["origin"] != "yunet", -f["score"]))
    return flags


def extract_frame(ffmpeg: str, video: Path, frame_idx: int, fps: float, out: Path) -> bool:
    """One frame as PNG. `-ss` before `-i` so ffmpeg seeks to the nearest
    keyframe and decodes forward instead of decoding the whole file — the
    difference between seconds and minutes on a late frame. Landing a frame
    early or late would not change a review verdict (the face is still
    there either way), so keyframe-relative accuracy is fine here."""
    timestamp = frame_idx / fps
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{timestamp:.6f}", "-i", str(video),
        "-frames:v", "1", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists():
        print(f"  warn: could not extract frame {frame_idx}: {result.stderr.strip()}")
        return False
    return True


def crop_and_annotate(frame_png: Path, flags_here: list[dict], out_dir: Path,
                      frame_idx: int, width: int, height: int) -> list[dict]:
    import cv2

    img = cv2.imread(str(frame_png))
    if img is None:
        print(f"  warn: could not read extracted frame {frame_png}")
        return []

    tiles = []
    context = img.copy()
    for i, flag in enumerate(flags_here):
        x1, y1, x2, y2 = (float(v) for v in flag["box"])
        colour = (0, 0, 255) if flag["origin"] == "yunet" else (0, 165, 255)
        cv2.rectangle(context, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        cv2.putText(context, f"{flag['origin']} {flag['score']:.2f}",
                    (int(x1), max(12, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w = max((x2 - x1) * CROP_CONTEXT_SCALE, CROP_MIN_PX) / 2
        half_h = max((y2 - y1) * CROP_CONTEXT_SCALE, CROP_MIN_PX) / 2
        cx1 = max(0, int(cx - half_w))
        cy1 = max(0, int(cy - half_h))
        cx2 = min(width, int(cx + half_w))
        cy2 = min(height, int(cy + half_h))
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        tile = img[cy1:cy2, cx1:cx2].copy()
        # Redraw the box in tile-local coordinates so it's obvious which
        # thing in the crop is the flagged region.
        cv2.rectangle(tile, (int(x1) - cx1, int(y1) - cy1),
                      (int(x2) - cx1, int(y2) - cy1), colour, 2)
        if tile.shape[1] < 320:
            scale = 320 / tile.shape[1]
            tile = cv2.resize(tile, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)

        # The per-flag index is what makes this unique. Keying on
        # frame+origin+2dp-score collided whenever one frame had two flags of
        # the same origin with similar scores: imwrite overwrote the first
        # tile, both entries pointed at the same file, and the page showed
        # one crop twice — so a flagged, possibly unredacted region was
        # never actually put in front of the reviewer.
        name = f"f{frame_idx:08d}_{i:03d}_{flag['origin']}_{flag['score']:.2f}.jpg"
        tile_path = out_dir / "tiles" / name
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tile_path), tile)
        tiles.append({**flag, "tile": f"tiles/{name}"})

    context_path = out_dir / "frames" / f"frame_{frame_idx:08d}.jpg"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(context_path), context)
    for t in tiles:
        t["context"] = f"frames/frame_{frame_idx:08d}.jpg"
    return tiles


def build_timelapse(ffmpeg: str, video: Path, out_path: Path) -> bool:
    """Complete visual coverage, fast. The flagged crops only show what a
    detector already suspected; this is the only artifact that puts every
    part of the clip in front of you, which is the one way to catch a face
    BOTH detectors missed. 1.5h at 1 sample/sec played at 15fps is ~6 min.

    Done in two steps — sample to JPEGs, then encode the image sequence —
    rather than one filtergraph. Single-command variants using setpts to
    compress time were all measurably wrong here: `-r` alone duplicates
    frames to fill a CFR grid (output ran the full source length, no
    speedup at all), and both `setpts=N/RATE/TB` and `setpts=PTS/RATE`
    produced either garbage timestamps or the wrong frame count depending
    on the filter timebase. Encoding an image sequence with `-framerate`
    has no timebase ambiguity: N images in, N frames out, duration N/rate,
    verified exactly.
    """
    tmp = out_path.parent / ".tl_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        sample = subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(video),
            "-vf", f"fps={TIMELAPSE_SAMPLE_FPS},scale={TIMELAPSE_WIDTH}:-2",
            "-q:v", "4",
            str(tmp / "%06d.jpg"),
        ], capture_output=True, text=True, check=False)
        if sample.returncode != 0:
            print(f"  warn: timelapse sampling failed: {sample.stderr.strip()}")
            return False
        if not any(tmp.iterdir()):
            print("  warn: timelapse sampling produced no frames")
            return False

        encode = subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-framerate", str(TIMELAPSE_PLAY_FPS),
            "-i", str(tmp / "%06d.jpg"),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an",
            str(out_path),
        ], capture_output=True, text=True, check=False)
        if encode.returncode != 0:
            print(f"  warn: timelapse encode failed: {encode.stderr.strip()}")
            return False
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_html(clip_id: str, manifest: dict, tiles: list[dict], out_dir: Path,
               timelapse: str | None, n_flagged: int = 0, n_failed: int = 0,
               n_truncated: int = 0) -> Path:
    """n_flagged/n_failed/n_truncated are what make this page honest. The
    tile list only contains regions that were successfully extracted and
    cropped; rendering a count of THOSE, and saying "no flagged regions"
    when the list is empty, told the reviewer the opposite of the truth on
    a clip whose extractions all failed."""
    audit = manifest.get("audit", {})
    src = manifest.get("source", {})
    fps = src.get("fps", 0) or 1

    rows = []
    for t in tiles:
        secs = t["frame_idx"] / fps
        stamp = f"{int(secs // 60):02d}:{secs % 60:05.2f}"
        origin_label = ("YuNet — independent detector saw a face here"
                        if t["origin"] == "yunet"
                        else "EgoBlur sub-threshold guess")
        rows.append(f"""
        <div class="card {html.escape(t['origin'])}">
          <img src="{html.escape(t['tile'])}" loading="lazy">
          <div class="meta">
            <strong>{html.escape(origin_label)}</strong>
            <div>score {t['score']:.3f} &middot; frame {t['frame_idx']} &middot; t={stamp}</div>
            <div>box {[round(v) for v in t['box']]}</div>
            <a href="{html.escape(t['context'])}" target="_blank">full frame in context &rarr;</a>
          </div>
        </div>""")

    timelapse_block = ""
    if timelapse:
        timelapse_block = f"""
    <h2>Full-clip timelapse</h2>
    <p>Every flagged crop below only shows what a detector already suspected.
       This is the only view covering the <em>whole</em> clip — the one way to
       catch a face both detectors missed entirely.</p>
    <video src="{html.escape(timelapse)}" controls style="max-width:100%"></video>"""

    warnings = []
    if n_failed:
        warnings.append(
            f"<strong>{n_failed} flagged region(s) could NOT be extracted</strong> and are "
            f"missing from this page. They were flagged — treat them as unreviewed.")
    if n_truncated:
        warnings.append(
            f"<strong>{n_truncated} flagged region(s) were truncated by --limit</strong> and "
            f"are not shown. Re-run with --limit 0 to see all of them.")
    warn_block = ""
    if warnings:
        warn_block = ('<div class="warn"><p>' + "</p><p>".join(warnings) + "</p></div>")

    if rows:
        body = "".join(rows)
    elif n_flagged:
        body = ("<div class='warn'><p><strong>This clip has "
                f"{n_flagged} flagged region(s), but none could be rendered.</strong> "
                "Do not read this as 'clean' — nothing here has been reviewed. "
                "Check the warnings above and the terminal output.</p></div>")
    else:
        body = ("<p class='ok'>No flagged regions — the audit reported none for this "
                "clip. Note that a clip can still contain a face neither detector "
                "found; see the note at the bottom.</p>")

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Blur review — {html.escape(clip_id)}</title>
<style>
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
  .status {{ display:inline-block; padding:.25rem .6rem; border-radius:4px;
             font-weight:600; }}
  .NEEDS_REVIEW {{ background:#fde2e1; color:#8a1c14; }}
  .PASS_AUTOMATED, .PASS_AUTOMATED_NO_YUNET {{ background:#e0f2e0; color:#1c5c1c; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
           gap:1rem; margin-top:1rem; }}
  .card {{ border:1px solid #ddd; border-radius:6px; overflow:hidden; }}
  .card.yunet {{ border-color:#d33; border-width:2px; }}
  .card img {{ width:100%; display:block; background:#111; }}
  .meta {{ padding:.6rem; font-size:12px; }}
  .meta strong {{ display:block; margin-bottom:.25rem; }}
  code, .mono {{ font-family: ui-monospace, monospace; }}
  .ok {{ color:#1c5c1c; }}
  .warn {{ background:#fff4e5; border:2px solid #d98324; border-radius:6px;
           padding:.75rem 1rem; margin:1rem 0; color:#7a4a06; }}
  dl {{ display:grid; grid-template-columns:max-content 1fr; gap:.2rem 1rem; }}
  dt {{ font-weight:600; }}
</style>
<h1>Blur review — {html.escape(clip_id)}</h1>
<p class="status {html.escape(manifest.get('status',''))}">{html.escape(manifest.get('status',''))}</p>
<dl>
  <dt>source</dt><dd class="mono">{html.escape(str(src.get('filename','?')))}</dd>
  <dt>resolution</dt><dd>{html.escape(str(src.get('width','?')))}&times;{html.escape(str(src.get('height','?')))} @ {fps:.2f}fps</dd>
  <dt>frames</dt><dd>{html.escape(str(src.get('n_frames','?')))}</dd>
  <dt>redacted frames</dt><dd>{html.escape(str(audit.get('n_frames_with_fill','?')))}
      ({(audit.get('frames_with_fill_frac') or 0)*100:.1f}%)</dd>
  <dt>fill integrity</dt><dd>{html.escape(str(audit.get('fill_integrity_violations','n/a')))} violation(s)
      &middot; {html.escape(str(audit.get('fill_integrity_checked','n/a')))} checked</dd>
  <dt>YuNet</dt><dd>{'ran' if audit.get('yunet_ran') else 'SKIPPED — weakest possible review'}</dd>
</dl>
{warn_block}
{timelapse_block}
<h2>Flagged regions — {len(tiles)} shown of {n_flagged} flagged</h2>
<p>Cropped from the <strong>blurred output</strong> — what actually ships. If you
   can see a face in one of these, the redaction missed it.
   Red border = an independent detector (YuNet) saw a face here; those are the
   strongest signals, and they sort first.</p>
<div class="grid">{body}</div>
<p style="margin-top:2rem;color:#666">{html.escape(audit.get('note',''))}</p>
"""
    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path


def review_clip(manifest_path: Path, ffmpeg: str, out_root: Path,
                want_timelapse: bool, limit: int) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip_id = str(manifest.get("clip_id", manifest_path.stem))
    audit = manifest.get("audit", {})
    src = manifest.get("source", {})
    fps = src.get("fps") or 0
    width = src.get("width") or 0
    height = src.get("height") or 0

    print(f"== {clip_id} — {manifest.get('status','?')} ==")

    # clip_id and output.path come from a manifest that travelled back from a
    # rented third-party pod, and both are used to build paths this script
    # writes to (and, for the timelapse temp dir, rmtree's). An absolute or
    # ../-containing value silently escapes out_root — pathlib drops the base
    # entirely on an absolute join.
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", clip_id):
        print(f"  skip: refusing unsafe clip_id {clip_id!r} (not [A-Za-z0-9._-]{{1,64}})")
        return {"clip_id": clip_id, "skipped": "unsafe clip_id"}
    out_name = manifest.get("output", {}).get("path", "")
    if not out_name or "/" in out_name or "\\" in out_name or out_name.startswith("."):
        print(f"  skip: refusing unsafe output.path {out_name!r} (must be a bare filename)")
        return {"clip_id": clip_id, "skipped": "unsafe output path"}

    video = manifest_path.parent / out_name
    if not video.is_file():
        print(f"  skip: blurred output not found at {video}")
        print("        (sync the job's output files down from the pod first)")
        return {"clip_id": clip_id, "skipped": "missing video"}
    if fps <= 0 or width <= 0 or height <= 0:
        print(f"  skip: manifest has unusable source geometry (fps={fps} {width}x{height})")
        return {"clip_id": clip_id, "skipped": "bad geometry"}

    flags = collect_flags(audit)
    n_flagged = len(flags)
    n_truncated = 0
    if limit and n_flagged > limit:
        n_truncated = n_flagged - limit
        print(f"  {n_flagged} flagged regions, showing the top {limit} by score "
              f"({n_truncated} not shown — use --limit 0 for all)")
        flags = flags[:limit]

    out_dir = out_root / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grouped by frame so each frame is decoded once, not once per flag —
    # but that means tiles come out in FRAME order, so the priority order
    # from collect_flags() has to be reapplied before rendering (below).
    by_frame: dict[int, list[dict]] = {}
    for f in flags:
        by_frame.setdefault(f["frame_idx"], []).append(f)

    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    tiles: list[dict] = []
    n_failed = 0
    for n, (frame_idx, flags_here) in enumerate(sorted(by_frame.items()), 1):
        print(f"  [{n}/{len(by_frame)}] frame {frame_idx}", end="\r", flush=True)
        frame_png = tmp_dir / f"{frame_idx}.png"
        if not extract_frame(ffmpeg, video, frame_idx, fps, frame_png):
            n_failed += len(flags_here)
            continue
        got = crop_and_annotate(frame_png, flags_here, out_dir,
                                frame_idx, width, height)
        # A frame that decoded but whose crops all fell through (unreadable
        # image, degenerate box) is still an unreviewed flagged region.
        n_failed += len(flags_here) - len(got)
        tiles.extend(got)
        frame_png.unlink(missing_ok=True)
    if by_frame:
        print()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if n_failed:
        print(f"  WARNING: {n_failed} flagged region(s) could not be rendered — "
              f"they are NOT on the page and remain unreviewed")

    # Restore priority order (YuNet first, then descending score) — see
    # by_frame above. Without this the page reads in frame order, burying
    # the strongest evidence wherever it happens to fall in the timeline.
    tiles.sort(key=lambda t: (t["origin"] != "yunet", -t["score"]))

    timelapse_name = None
    if want_timelapse:
        print("  building timelapse (decodes the whole clip, be patient)")
        if build_timelapse(ffmpeg, video, out_dir / "timelapse.mp4"):
            timelapse_name = "timelapse.mp4"

    index = write_html(clip_id, manifest, tiles, out_dir, timelapse_name,
                        n_flagged=n_flagged, n_failed=n_failed,
                        n_truncated=n_truncated)
    print(f"  {len(tiles)} of {n_flagged} flagged region(s) rendered -> {index}")
    return {"clip_id": clip_id, "n_tiles": len(tiles), "n_flagged": n_flagged,
            "n_failed": n_failed, "index": str(index)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("target", type=Path,
                   help="a *.manifest.json, or a directory containing them")
    p.add_argument("--out", type=Path, default=None,
                   help="review output root (default: <target dir>/review)")
    p.add_argument("--timelapse", action="store_true",
                   help="also build a whole-clip timelapse — slower (full decode), "
                        "but the only view that covers frames no detector flagged")
    p.add_argument("--limit", type=int, default=200,
                   help="max flagged regions per clip, highest score first (0 = all)")
    p.add_argument("--all", action="store_true",
                   help="review every clip, not just NEEDS_REVIEW ones")
    a = p.parse_args()

    ffmpeg = _bin("ffmpeg")
    manifests = find_manifests(a.target)
    root = a.target if a.target.is_dir() else a.target.parent
    out_root = a.out or (root / "review")

    results = []
    for m in manifests:
        try:
            status = json.loads(m.read_text(encoding="utf-8")).get("status", "")
        except json.JSONDecodeError:
            print(f"skip: {m} is not valid JSON")
            continue
        if not a.all and status.startswith("PASS"):
            print(f"== {m.stem} — {status}, skipping (pass --all to review anyway) ==")
            continue
        results.append(review_clip(m, ffmpeg, out_root, a.timelapse, a.limit))

    reviewed = [r for r in results if "index" in r]
    print()
    if not reviewed:
        print("Nothing to review.")
        return 0
    print(f"Open these in a browser ({len(reviewed)} clip(s)):")
    for r in reviewed:
        print(f"  {r['index']}")
    print()
    print("A visible face in any tile means the redaction missed it. To cover it:")
    print("  1. write {\"<clip_id>\": [{\"frame_idx\": N, \"box\": [x1,y1,x2,y2]}]} to forced.json")
    print("  2. re-run the blur job with BOTH --forced-boxes forced.json --force-reprocess")
    print("     (without --force-reprocess the clip is skipped: it already has a manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
