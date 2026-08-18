# egoannote — session handover

Written 2026-08-11, end of a long tuning session on `jobs/10_blur_egoblur.py`
(EgoBlur face/plate redaction), updated same day after two more fixes
landed. Repo HEAD at handoff: **`d0cbe10`**, tree clean, 272 tests passing.
Read this top to bottom before doing anything — the "do this first" item
below is a genuine unknown, not busywork.

## The project, in one paragraph

Solo portfolio project: salvaging a shelved egocentric-video-annotation
startup into a public, forkable pipeline. Plan is to publish ~1.5h of the
user's own GoPro footage (a family paint/hardware shop) as an annotated
HuggingFace dataset, then write an honest post-mortem. Pipeline order:
**EgoBlur (GPU, redacts faces/plates)** → MediaPipe hands (CPU/local) → VLM
captioning (API) → segmentation (not yet implemented, deliberately). This
session was entirely about getting EgoBlur right on the first real clip
before scaling to the other 15.

## Do this first — genuinely unknown, not a formality

**Confirm `test-run-3`'s fill-integrity result.** Early in the session,
`test-run-1` showed `fill_integrity_violations: 14096` (65% of checked
boxes). The user watched the actual output video and it looked correctly
redacted, so we agreed it was almost certainly the *check* being too
strict against real encoder ringing — not a real leak — and moved on to
threshold tuning. **We never circled back and confirmed this on the
current best-known settings.** The `out2/GX010057.manifest.json` sitting
locally in `~/Downloads` is stale (it's `test-run-2`'s data — threshold
0.2, the bad hand-blurring run). Nobody has read `test-run-3`'s actual
`audit_summary.md`.

```bash
cat /workspace/out2/GX010057.audit_summary.md
```

If `fill_integrity_violations` is still elevated at these settings, **that
is the real blocker**, ahead of everything else in this file. If it's
clean (or `check_fill_integrity` was already legitimately fixed by an
earlier session and this is a non-issue), say so and move on.

## Current best-known-good settings

Arrived at empirically via `sweep_params.py` against the clip's own
detection checkpoint (see "How we got here" below) — **not a guess**:

```bash
uv run jobs/10_blur_egoblur.py --input-dir /workspace/in --output-dir /workspace/out2 \
  --run-id test-run-3 --gen 2 \
  --face-weights-gen2 /workspace/weights/ego_blur_face_gen2.jit \
  --face-threshold 0.30 --hold-frames 45 \
  --gpu-rate-usd-per-hr <your actual $/hr> --skip-shutdown
```

Notes on why each piece is there:
- **No `--lp-weights-gen2`.** The plate model fires on printed cardboard
  text ("THIS SIDE UP") on this footage; dropping it removed most of the
  audit noise and made `fill_map` coverage checks mean what they say.
  `lp_checked: false` will be recorded honestly in the manifest.
- **`--hold-frames 45`** — `--dilate-scale` / `--motion-margin-px` are left
  at their defaults (1.3 / 8px). A `--face-threshold 0.20` experiment
  (`test-run-2`) was tried and was a real mistake: it caught a few more
  faces but also buried the wearer's own hands under grey boxes for most
  of the clip (screenshot-confirmed, see git log for `0.20` context).
  `sweep_params.py` proved 0.30 + the backward-hold fix alone gives
  **35/35 coverage** of every timestamp the user manually flagged, with
  hand-leakage statistically unchanged from the very first conservative
  run. Do not lower the threshold again without re-running that sweep.

## Two real, small bugs — both fixed now (commit `d0cbe10`)

Both flagged in the `/review` pass, both fixed and mutation-tested before
this handoff, kept here so the reasoning is still visible:

1. **`max_fill_area_frac` was a dead canary.** `redact_and_encode` computed
   it, `write_audit_summary` printed it labelled "runaway false-positive
   canary" — but `build_audit` never read it, so nothing could actually
   gate on it; a flooded frame could still report `PASS_AUTOMATED`.
   **Fixed:** `build_audit` now flags `NEEDS_REVIEW` above
   `MAX_FILL_AREA_FRAC_CEILING = 0.5` — deliberately generous so a
   legitimate close-up face doesn't false-alarm.

2. **The hysteresis drift bound (`max_low_run`) counted absorption
   events, not video frames** — so its real-time budget silently depended
   on `--detect-hz`. At the tested default stride this happened to look
   right, hiding the bug; at a coarser stride (or just detections that
   don't land on every sampled frame) the same "4 events" could span far
   more real drift than intended — reproduced directly: weak detections
   30 frames apart were all absorbed under the old gate, none are now.
   **Fixed:** `Track` now tracks `last_confident_frame`; the gate is
   `frame_idx - tr.last_confident_frame > max_low_run * stride`, which
   means the same thing regardless of `--detect-hz`. This was still
   open when the message above ("hysteresis buys ~12 frames...") was
   written — it's item 1 in the numbered list further down in this same
   conversation, now resolved.

## Still open — the resumed-batch config gap

Not fixed this session; the batch is currently small enough (16 clips, one
sitting) that it's a real but lower-urgency risk than the two above were:

**A resumed batch can silently mix redaction configs across clips.**
`process_clip`'s manifest-skip (`if manifest_path.exists() and not
cfg.force_reprocess`) only checks *whether* a manifest exists, never
*what config produced it* — unlike `detection_pass`'s own
`checkpoint_fingerprint()`, which does exactly this correctly for the
detection step. If the 16-clip batch gets interrupted (pod disconnects
have happened repeatedly this session — see "RunPod gotchas" below) and
resumed with a changed flag, already-finished clips keep their old
settings with **zero warning**, and `run_manifest.json` doesn't even
record which thresholds were used. Fix: extend something like
`checkpoint_fingerprint()` to cover redaction-relevant config
(thresholds, dilate/margin, holds) and check it at the manifest-skip site
too, or at minimum log the mismatch loudly. Doesn't touch detection, so
fixing it doesn't cost any GPU re-run.

## Dormant code — inactive unless `--continue-threshold` is passed

Two-threshold hysteresis (`build_tracks`'s `start_thresh`/`cont_thresh`,
extend-an-existing-track-on-weak-evidence) was built, found to contain a
**real privacy regression** (an absorbed low-confidence box could
*replace* a track's position instead of adding to it, making a confirmed
face's coverage *worse* while silencing the sweep gate that would have
flagged it — see commit `f17faf0` for the full repro), and fixed. It is
**off by default** (`--continue-threshold` defaults to `0`) and the
recommended batch command above does not use it. Its drift bound
(`max_low_run` counting absorption events instead of real video frames —
originally listed here) was fixed alongside the two bugs above, in the
same commit (`d0cbe10`). These are the remaining real findings from that
review pass, all currently unreachable, left as-is:

- ~~One `--continue-threshold` value applies to both face and plate
  classes, even though only the face model's behaviour was measured.~~
  **Resolved by context, not by code.** `--lp-weights-gen2` is dropped
  for the whole batch (see "Current best-known-good settings" above), so
  `lp_det` is `None` and `detection_pass` never produces a single
  `cls == "lp"` detection — confirmed directly: `lp_det.detect_batch(...)`
  only runs inside `if lp_det is not None`. `resolve_hysteresis` still
  builds a `cont_thresh` dict with an `"lp"` key, but nothing ever reads
  it, so a shared threshold has no second class left to mishandle. This
  is a fact about how the job is run right now, not a code change — the
  risk is live again immediately if plate redaction is ever re-enabled
  for different footage.
- Greedy single-stage IoU association can let a low-confidence detection
  outbid a high-confidence one for the same track (real ByteTrack
  matches high-score detections first; this doesn't).
- The `det_low` source tag never reaches any human-readable output —
  `tracks_to_fill_map` discards it.
- `n_low_absorbed` reaches the JSON manifest but not
  `write_audit_summary`'s markdown.

**Do not turn hysteresis on for the batch without addressing the three
still-open items above.** If a future session wants it, treat that list
as the acceptance criteria, not just the union-not-replace fix already
shipped.

## MediaPipe as a second detector — investigated, concluded, code removed

The user proposed MediaPipe Face Landmarker as an independent second
check (filling the role `--yunet-model` was designed for but never had
real weights sourced for). Built a standalone comparison tool
(`compare_mediapipe_egoblur.py`, lived in a scratchpad, never committed —
not currently in `~/Downloads` either), ran it against the full clip
(2712 sampled frames, ~2 min on CPU), and got a decisive, verified
negative result: **MediaPipe Face Landmarker run on whole 1920x1080
frames is functionally blind to faces at the size they appear in this
footage.** Proved this directly — a face EgoBlur scored at 1.00
confidence (a real, frontal, well-lit, unoccluded face) produced zero
MediaPipe detections at any confidence threshold down to 0.1 on the full
frame, but was found instantly the moment the exact same pixels were
cropped and upscaled 3x. This is a genuine BlazeFace-family scale
limitation (built for near-field/selfie use), not a config or threshold
issue. All 4 of the frames where MediaPipe found something EgoBlur
didn't were MediaPipe's *own* false positives on hands/tools — it never
once caught a real face EgoBlur had missed.

**Conclusion: do not wire MediaPipe in as-is.** It would add false
confidence ("independently corroborated!") on a detector that's
structurally unable to see the faces in question. The real fix — tiling
the frame into overlapping crops before detection — is a legitimate
engineering path if this is revisited, but is real added complexity, not
a quick win. Deprioritized in favor of shipping the batch with the
threshold-tuned single-detector pipeline.

`.mp_models/` (the downloaded model file) and a stray `forced_boxes.json`
(unrelated leftover from an earlier `--forced-boxes` analysis attempt,
accidentally committed by an over-broad `git add -A`) were both removed
this session — see commits `8b1b2fa`. `jobs/10_blur_egoblur.py` itself was
never touched by any of this; zero MediaPipe references in it.

## Accepted residual risk — not fixable today, know about it

- **No independent second-opinion detector is actually running.**
  YuNet weights were never sourced (gated download, never resolved);
  MediaPipe just proved non-viable naively (above). The low-threshold
  sweep on EgoBlur's *own* detections is the only signal against a
  genuinely missed face — every clip that redacts nothing will need a
  human look (`build_audit`'s zero-coverage canary), but a face EgoBlur
  is confidently wrong about in a *different* way has no independent
  check. State this honestly in the eventual dataset card.
- **The PEP 723 header pins torch to a CUDA-only index**
  (`download.pytorch.org/whl/cu128`), so `jobs/10_blur_egoblur.py` cannot
  resolve its dependencies on the user's local macOS machine at all.
  Irrelevant while running on the RunPod GPU pod; a real portability bug
  for anyone forking this repo on a Mac.

## How we got here (condensed timeline, for context on *why*)

1. `test-run-1` (defaults: threshold 0.30, hold "auto"=30, LP on) — ran
   clean, `NEEDS_REVIEW` as expected (no YuNet), but `fill_integrity_violations`
   very high. User watched the video, spotted faces still visible at 35
   specific timestamps, mostly at the *leading edge* of a face entering
   frame.
2. Root-caused: `build_tracks`' backward hold was hardcoded to one
   detection stride (~0.1s) while the forward hold was a full second — a
   10x asymmetry with no real justification. Fixed (`0ab77da`,
   `dcad7e1`) to be symmetric by default, with `resolve_holds()` and a
   `--back-hold-frames` override.
3. Tried lowering `--face-threshold` to 0.20 to mop up the remaining
   gaps (`test-run-2`). This was a mistake — massively over-redacted the
   wearer's own hands. User caught it by watching the output, not by any
   metric.
4. Built `sweep_params.py` to test parameter combinations against
   already-paid-for detections (zero GPU cost) instead of guessing.
   Result: `--hold-frames 45` alone (threshold back at 0.30) gives
   **35/35** coverage of every user-flagged timestamp, with area/hand-leak
   stats matching the very first conservative run. This became
   `test-run-3` and the recommended settings above.
5. Explored two-threshold hysteresis and MediaPipe-as-second-detector as
   further improvements. Both investigated properly, both had real
   findings, neither is currently wired into the recommended run (see
   sections above for why).

## RunPod gotchas already fixed — don't re-discover these

All fixed in `scripts/runpod_setup.sh`, but worth knowing *why* it looks
the way it does, since the pattern (container disk vs `/workspace`
volume) will bite again if a future job script installs something new
into `/usr/bin` or `/root`:

- **Container disk is wiped on every pod `stop`; `/workspace` (the
  volume) survives.** `uv`, `ffmpeg`, `ffprobe`, and `rclone` all now
  install to `/workspace/bin`, never via `apt-get` (which lands in
  `/usr/bin` — this exact mistake cost 3 separate run attempts before it
  was fixed).
- **`rclone`'s config also used to live off-volume**
  (`/root/.config/rclone/rclone.conf`) — now pinned to
  `/workspace/.rclone.conf` via `$RCLONE_CONFIG`.
- **After any pod restart, or any new terminal connecting to an already-
  running pod**, run:
  ```bash
  cd /workspace/egoannote && git pull && bash scripts/runpod_setup.sh
  source /workspace/env.sh
  ```
  The `source` step matters — a script cannot export into its parent
  shell, and this has caused `uv: command not found` more than once.
- **Long jobs must run detached**, or a browser/terminal disconnect kills
  them mid-run (happened twice this session, though the detection
  checkpoint saved almost all the lost work both times). Pattern used
  successfully:
  ```bash
  cd /workspace/egoannote && setsid nohup env PATH="/workspace/bin:$PATH" \
    UV_CACHE_DIR=/workspace/.uv-cache /workspace/bin/uv run jobs/10_blur_egoblur.py \
    <args> > /workspace/run.log 2>&1 < /dev/null &
  tail -f /workspace/run.log   # Ctrl-C only stops watching, not the job
  ```
- **The 8-hour watchdog (`arm_watchdog`) almost certainly can't actually
  fire** — `runpodctl` on this pod was never configured with an API key.
  Don't rely on it; stop the pod yourself when done.
- **Community Cloud can fail to resume** ("not enough free GPUs") after a
  stop. If it happens: try Resume a few times first (frees up often),
  then consider a Network Volume (survives independent of any specific
  pod) if it keeps happening.
- **The detection checkpoint (`checkpoints/<clip>.detections.jsonl`) is
  fsynced per batch and genuinely resumable.** Re-running the identical
  `uv run` command after any crash resumes near-instantly rather than
  re-paying the ~25 min GPU cost — confirm this by watching for
  `resuming, N detection-frames already attempted` in the log. Changing
  `--face-threshold` / `--hold-frames` / `--dilate-scale` /
  `--motion-margin-px` does **not** invalidate the checkpoint (by
  design — see `checkpoint_fingerprint()`), so re-tuning those is
  always cheap. Changing `--gen`, weights, `--sweep-threshold`,
  `--nms-iou`, `--detect-hz`, or adding/removing `--lp-weights-gen2`
  **does** invalidate it correctly.

## Diagnostic scripts built this session

All are standalone, read-only tools operating on a run's own
`*.manifest.json` + `checkpoints/*.jsonl` — **zero GPU cost**, safe to run
locally with the pod stopped. None are committed to the repo (by design —
throwaway analysis tools, not pipeline code). Confirmed still present in
`~/Downloads` at handoff: `analyze_timestamps.py`, `sweep_params.py`.
Others (`pick_threshold.py`, `inspect_frame.py`, `diag_integrity.py`,
`annotate_local.py`) were generated and sent during the session but their
presence on disk now is **unconfirmed** — regenerate from this
conversation's history if needed, or ask for them again; they're small.

- **`analyze_timestamps.py`** — given a list of MM:SS timestamps where a
  human saw an uncovered face, classifies each as `DETECTED_FILTERED`
  (fixable by threshold), `NEVER_DETECTED` (needs a forced box), or
  `COVERED` (box present but possibly too small). Can auto-generate a
  `--forced-boxes` JSON from the detector's own sub-threshold
  coordinates. Handles ambiguous timestamp entry (e.g. `1.4` could be
  1:40 or 1:04) by checking both readings rather than guessing.
- **`sweep_params.py`** — the tool that actually produced the current
  settings. Tests named parameter combinations against a run's existing
  checkpoint, reporting per-combo: how many of a given timestamp list end
  up fully covered, and — critically — what fraction of the redacted
  area lands in the bottom third of frame (a proxy for "is this
  blurring the wearer's own hands").
- **`diag_integrity.py`** *(not confirmed present)* — rebuilds a run's
  exact `fill_map` from its checkpoint and measures real pixel deviation
  inside claimed-filled boxes against `FILL_VALUE`, bucketed by box size,
  to distinguish "the check is too strict" from "there's a real leak."
  **This is the tool to reach for on the "do this first" item above if
  the plain audit-summary read isn't conclusive enough.**
- **`inspect_frame.py`** *(not confirmed present)* — point it at a
  specific timestamp, get every detection in a window around it with
  scores, to answer "was this face detected-and-filtered or never seen
  at all."
- **`pick_threshold.py`** *(not confirmed present)* — shows what
  `--face-threshold` value would keep vs. drop each stored detection,
  bucketed by box size/position, without re-running detection.
- **`annotate_local.py`** *(not confirmed present)* — Mac-local runner
  for MediaPipe hands + VLM captioning on a real clip (mirrors
  `scripts/demo.py` but against real footage instead of the synthetic
  test clip). Used once to smoke-test the plumbing on the blurred output.

## What's downstream and genuinely not started

Once the batch of 16 clips clears EgoBlur:

- **MediaPipe hand tracking** — code exists (`src/egoannote/layers/hands.py`),
  proven working (Metal-accelerated, fast, free, runs on the Mac), but
  never run against real *blurred* output — only the synthetic smoke
  test and one plumbing check via `annotate_local.py`.
- **VLM captioning** — code exists, but `models.toml` still ships
  placeholder model IDs and `$0.00` prices; needs two real models from
  *different labs* (for the agreement-metric design) filled in before
  any real captioning run. The prompt (`prompts/caption_v3.txt`) was
  independently reviewed and fixed this session for an unrelated bug
  (see `ffac1f9`) — that fix is done and not blocked on anything here.
- **Segmentation** (`src/egoannote/layers/segment.py`) — deliberately
  unimplemented; its own docstring explains it's blocked on real
  measurements from hands + captions that don't exist yet.
- **`verify/` and `pack/`** — both empty files. Nobody has started the
  code that would actually assemble and validate a HuggingFace dataset
  from the layer outputs.

## Immediate next action, concretely

1. Resume/start the pod, `git pull`, `bash scripts/runpod_setup.sh`,
   `source /workspace/env.sh`.
2. Read `test-run-3`'s real `audit_summary.md` (see "Do this first"). The
   dead-canary and drift-bound bugs are already fixed (`d0cbe10`) —
   nothing to do there, this step is purely about the actual fill-integrity
   number.
3. If clean: optionally address the remaining resumed-batch config-drift
   gap (see "Still open" above) — lower urgency, doesn't block a single
   uninterrupted 16-clip run — then run the batch with the settings block
   above, one clip's `--input-dir` pointed at all 16 source files via
   `rclone` (already configured this session,
   `RCLONE_CONFIG=/workspace/.rclone.conf`, remote name `gdrive`,
   folder `nbt-videos`).
4. If not clean: use `diag_integrity.py` (regenerate if needed) to
   determine whether it's a real leak or an over-strict check, the same
   way the `test-run-1` investigation started but never finished.
