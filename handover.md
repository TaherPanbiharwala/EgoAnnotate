# egoannote — session handover

## Current checkpoint and next-chat focus (2026-08-25)

`master` is at `6a8828d` (`fix: harden active wearer hand suppression`), and
the working tree was clean when this note was updated. The current pod has
successfully completed the private Hand Landmarker pre-pass for `GX010057`:

- artifact: `/workspace/private/hand-prior/GX010057.hand_prior.json`;
- private original-only preview:
  `/workspace/private/hand-preview/GX010057.hands.mp4`.

The preview has **not** yet been visually reviewed. Before running active
hand suppression, make a fresh **four-hand diagnostic** prior/preview (the
existing artifact requested only two hands, so it cannot show a third hand).
The new preview labels `H#` short-continuity anchors, MediaPipe's `MP:
Left/Right` hint and score, plus `entry` and `now` image zones; a `3+ hands`
badge is review evidence only. A long-gap re-entry deliberately receives a
new `H#` and is `unknown`, never automatically classified as wearer or
outsider. This four-hand artifact is diagnostic-only and must not be passed
to active suppression, which remains pinned to a separately generated,
reviewed two-hand artifact. Inspect amber/magenta/blue labels before that
second pass; if amber follows another person, do not run active suppression.
If it is reliable, run the documented `gx010057-active-hand-v1` pilot,
inspect the separate private suppression report, and manually review its
redacted output. No private artifact, preview, checkpoint, or report may be
uploaded to Hugging Face.

The user's requested focus for the next chat is to **review and edit the
dense VLM captioning design/code**, rather than run a paid caption batch.
Start with `docs/MEDIAPIPE_VLM_PIPELINE.md`, `prompts/caption_v4.txt`,
`src/egoannote/layers/caption.py`, `src/egoannote/parse.py`, and
`src/egoannote/schema.py`. Preserve these non-negotiable output tracks for
every six-second redacted-video window:

1. one holistic `activity.caption` describing the full activity as it unfolds;
2. temporally bounded atomic `actions[]`; and
3. one dense caption plus structured state for each visible anatomical hand
   (`left_hand` and `right_hand`).

Captioning must remain redacted-video-only. It has not yet been run against
real verified footage: `models.toml` still needs two real, cross-lab provider
entries, pinned providers/prices, and `OPENROUTER_API_KEY`. Do not change
Stage 2, WiLoR, SAM2, or DepthV3 in that captioning-focused chat.

## Current redaction-review work (2026-08-25)

Stage 2 and YuNet are paused for the current run. The active protocol is
EgoBlur plus private MediaPipe diagnostics; YuNet is preserved in the source
but must not be run for this pilot.

- `egoannote-run pose-prior` reads an original video only, downloads the
  versioned MediaPipe Pose Landmarker Full task if needed, and writes a private
  artifact with the source/model hashes, complete sampled-frame coverage, and
  limb-only landmarks/masks. It does not retain face/torso landmarks. Its
  optional `--preview-video` draws private original-only limb overlays: amber
  for a wearer candidate and blue for other detected poses, held between the
  10 Hz samples.
- The pose artifact identifies **camera-near wearer candidates**, not people
  with certainty. `--pose-prior` and `--pose-shadow-report` are permanently
  shadow-only. Do not use Pose amber/blue assignment to suppress redaction:
  on `GX010057` it generated candidate hands but zero stable active evidence,
  and visually swapped between the wearer and another person.
- `egoannote-run hand-prior` is the behavior-changing prior. It reads the
  original privately at 10 Hz using MediaPipe Hand Landmarker, stores all 21
  2D landmarks plus a short continuity track, and makes an original-only
  preview: amber = stable wearer candidate, magenta = provisional, blue =
  other hand. `--hand-suppress-wearer-hands` is pinned to 10 Hz and can
  withhold only a face track
  whose four or more raw boxes are each >=98% inside the *same*, stable,
  large/lower-or-side wearer hand region. It never changes full-frame
  detection, thresholding, or treatment of other people. Any nonzero count
  forces `NEEDS_REVIEW` and must be reviewed in the private hand report. For
  batches, use a private `--hand-suppression-report` directory so every clip
  retains its own suppression evidence.
  Magenta/provisional overlap always produces a private
  `pink_amplification_candidate` when one same provisional hand accounts for
  every raw face hit in a track that later receives interpolation/hold fills.
  The opt-in `--pink-demote-generated-fills` is a narrowly bounded active
  exception: after two raw hits with >=90% overlap to the same pink hand, it
  keeps all raw detections but caps only generated interpolation/hold context
  (12 frames by default). It requires the pinned two-hand artifact, reports
  every removal privately, and any nonzero demotion forces `NEEDS_REVIEW`.
- EgoBlur's raw detector checkpoints are original-derived and now require a
  private `--checkpoint-dir`; never keep them beside a publishable redacted
  output directory.
- `verify-yunet` still scans the entire redacted video. It now groups all
  uncovered hits into temporal candidates and produces redacted-only contact
  sheets. Only repeated overlap with a likely wearer limb is amber/lower
  priority. Other people's limbs stay normal priority; no candidate is
  dropped or removed from `NEEDS_REVIEW`.
- `init-yunet-decisions` creates a hash-bound private decision template.
  `decisions-to-forced-boxes` accepts only `confirmed_face` entries and emits
  the existing EgoBlur `--forced-boxes` JSON. `uncertain` is not a privacy
  clearance.

Do not run a batch until the Hand Landmarker preview and the first active-hand
pilot are visually reviewed on `GX010057`, with its ordinary `min_track_confirmations=2`
run retained for comparison. The nonzero-suppression result must remain
`NEEDS_REVIEW` until a person checks the private report/video.

Written 2026-08-11, substantially rewritten in a later session once the
"do this first" item below was actually resolved (with evidence, not a
visual spot-check) and two of the three remaining hysteresis blockers got
fixed. This historical section predates the checkpoint above. The current
repository state is `master` at **`6a8828d`** with the active-hand hardening
committed; the latest local validation was **342 passing tests**.

## Latest pipeline attempt — blocked on a corrupt redacted artifact (2026-08-24)

The first real MediaPipe/VLM pilot was started from `master` using
`GX010057.blurred.mp4`; Stage 2 was not invoked. The result is **not
usable**: ffprobe advertises 8,134 frames / 271.4 seconds, but a full
FFmpeg decode reports a 99.5% H.264 decode-error rate. MediaPipe emitted
only 14 of the expected 4,067 samples at the configured ~15 FPS. The
private pilot manifest records `hands.status = failed_incomplete_decode`;
do not publish or reuse its `hands.parquet`.

The pipeline now rejects this failure mode in two places: the MediaPipe
generator compares its emitted sample count with ffprobe's expected count,
and a pre-existing `hands.parquet` is checked before resume. A partial
decode can no longer be marked complete or silently reused. Focused tests
for both guards pass.

**Resume path:** restore/locate the original `GX010057`, regenerate a new
EgoBlur derivative with the documented `test-run-3` settings and preserve
its passing manifest, then run a fresh MediaPipe/VLM pilot directory. The
original is not presently in this repository, and the old derivative must
not be repaired or used as an annotation input. Captioning additionally
needs two real, cross-lab VLM entries (including verified prices/provider
pins) in `models.toml` and `OPENROUTER_API_KEY`; only placeholders exist
today. MediaPipe and VLM continue to read redacted video only. WiLoR,
SAM2, and DepthV3 remain future original-only private stages.

## The project, in one paragraph

Solo portfolio project: salvaging a shelved egocentric-video-annotation
startup into a public, forkable pipeline. Plan is to publish ~1.5h of the
user's own GoPro footage (a family paint/hardware shop) as an annotated
HuggingFace dataset, then write an honest post-mortem. Pipeline order:
**EgoBlur (GPU, redacts faces/plates)** → MediaPipe hands (CPU/local) → VLM
captioning (API) → segmentation (not yet implemented, deliberately).

## "Do this first" — RESOLVED, with real evidence

The previous handoff's biggest open risk: `test-run-1` showed
`fill_integrity_violations: 14096` (~65% of checked boxes), accepted as
"probably encoder ringing, not a real leak" purely because the user
watched the output video and it looked fine — never independently
confirmed. That confirmation has now actually happened.

**`test-run-3`'s real numbers:** `checked=36731`, `violations=23216` →
**63.2%** — essentially flat vs. `test-run-1`'s ~65%, despite 69% more
boxes being checked overall (the newer, longer, now-symmetric hold window
draws more boxes). Rate holding steady while volume grew is itself
evidence the check is catching something systemic, not something the
newer settings introduced.

That comparison alone wasn't proof, so a purpose-built diagnostic
(`diag_integrity.py` — see "Diagnostic scripts" below) re-decoded the
*already-encoded* output and measured, for every violating box, **how far**
past the exact-pixel threshold it actually was, instead of just
pass/fail:

| How far past the limit | Count | % of violations |
|---|---|---|
| Barely over (1–5) | 22,432 | **96.6%** |
| Moderate (6–20) | 784 | 3.4% |
| Severe (21–60) | **0** | 0% |
| Extreme (60+) | **0** | 0% |

Zero boxes fall into "severe" or "extreme" — a real leaked face would blow
past a deviation of 60 easily (skin tone is 50–100+ gray levels from
mid-gray `FILL_VALUE=128`). The worst offender in the entire clip
deviates by only 22, with `max_std` under 1.0 (limit 2) — meaning the
"violating" patch is still nearly perfectly *flat*, just slightly the
wrong shade of gray. A real leak would show high internal variance
(texture, an edge, a color change); this shows none. Several of the worst
offenders share the *exact same box coordinates* across nearby frames
(e.g. `(367, 124, 532, 266)` repeats at frames 6001/6002/6036/6048, all
within ~1.7s) — consistent with one held/static box sitting over the same
patch of background hitting the same mild compression artifact, not a
moving/evolving leak.

**Conclusion: `check_fill_integrity`'s exact-pixel tolerance is too
strict against ordinary H.264 encoder quantization, not catching a real
privacy leak.** Consider this item closed. Re-verify with
`diag_integrity.py` (already delivered to `/workspace/diag_integrity.py`
on the pod) if this is ever revisited on different footage or settings.

## Current best-known-good settings

Unchanged since the last handoff — arrived at empirically via
`sweep_params.py`, not a guess:

```bash
uv run jobs/10_blur_egoblur.py --input-dir /workspace/in --output-dir /workspace/out2 \
  --run-id test-run-3 --gen 2 \
  --face-weights-gen2 /workspace/weights/ego_blur_face_gen2.jit \
  --face-threshold 0.30 --hold-frames 45 \
  --gpu-rate-usd-per-hr <your actual $/hr> --skip-shutdown
```

No `--lp-weights-gen2` (plate model false-fires on cardboard packaging on
this footage; `lp_checked: false` recorded honestly). `--dilate-scale` /
`--motion-margin-px` left at defaults. Do not lower `--face-threshold`
below 0.30 without re-running `sweep_params.py` — 0.20 was tried
(`test-run-2`) and buried the wearer's own hands under grey boxes.

## Two real, small bugs — fixed (commit `d0cbe10`)

1. **`max_fill_area_frac` was a dead canary** — computed and printed but
   never gated on. **Fixed:** `build_audit` now flags `NEEDS_REVIEW` above
   `MAX_FILL_AREA_FRAC_CEILING = 0.5`.
2. **The hysteresis drift bound counted absorption events, not video
   frames**, so its real-time budget silently depended on `--detect-hz`.
   **Fixed:** `Track.last_confident_frame` + a frame-based gate.

## Still open — the resumed-batch config gap

Not touched this session; still the one real, if lower-urgency, risk
before an unattended batch run:

**A resumed batch can silently mix redaction configs across clips.**
`process_clip`'s manifest-skip (`if manifest_path.exists() and not
cfg.force_reprocess`) only checks *whether* a manifest exists, never
*what config produced it* — unlike `detection_pass`'s own
`checkpoint_fingerprint()`. If the 16-clip batch gets interrupted (pod
disconnects have happened repeatedly) and resumed with a changed flag,
already-finished clips keep their old settings with **zero warning**.
Fix: extend `checkpoint_fingerprint()`-style tracking to cover
redaction-relevant config and check it at the manifest-skip site too, or
at minimum log the mismatch loudly. Doesn't touch detection, so fixing it
doesn't cost any GPU re-run.

## Hysteresis — 3 of 4 blockers now resolved, 1 left

Two-threshold hysteresis (`build_tracks`'s `start_thresh`/`cont_thresh`)
is **off by default** (`--continue-threshold` defaults to `0`) and the
recommended batch command above does not use it. Original acceptance
criteria before it's safe to turn on, and their current status:

1. ~~One `--continue-threshold` value applies to both face and plate
   classes, even though only the face model's behaviour was measured.~~
   **Resolved by context, not by code.** `--lp-weights-gen2` is dropped
   for the whole batch, so `lp_det` is `None` and no `cls == "lp"`
   detection is ever produced for a shared threshold to mishandle. This
   is a fact about current usage, not a code fix — the risk returns
   immediately if plate redaction is ever re-enabled for different
   footage.
2. ~~The `det_low` source tag never reached any human-readable output —
   `tracks_to_fill_map` discarded it.~~ **Fixed, and fixed *properly* on
   the second try.** The first attempt threaded a new
   `low_source_frames` out-parameter through `tracks_to_fill_map`'s hot
   loop — an adversarial code review then found this was unnecessary
   complexity: `low_absorbed` (populated during `build_tracks`, one call
   earlier, never deleted) already carries a `Detection` per absorption
   with the identical `frame_idx`. The actual fix, with zero changes to
   `tracks_to_fill_map`: `det_low_frames = sorted({d.frame_idx for d in
   low_absorbed})` at the `build_audit` call site in `process_clip`.
   `tracks_to_fill_map` is back to its original, untouched form.
3. ~~`n_low_absorbed` reached the JSON manifest but not
   `write_audit_summary`'s markdown.~~ **Fixed** — now printed, plus
   which frames it refers to (capped excerpt, full list in the JSON).
   Two real bugs surfaced by the same adversarial review and fixed along
   the way: (a) the empty-case fallback text used to hard-code `"0 when
   --continue-threshold is off"`, which `write_audit_summary` has no way
   to actually verify (it never sees `cfg.continue_threshold`) — a clip
   run *with* hysteresis on that happened to absorb nothing would have
   printed a false claim; now it just states the fact with no guessed
   cause. (b) `n_low_absorbed` (an event count) and `det_low_frames` (a
   *deduplicated* frame list) can legitimately diverge — two different
   tracks absorbing on the same `frame_idx` means 2 events, 1 distinct
   frame — so the line now reads `n_low_absorbed: 2  (...; 1 distinct
   frame(s): 5)` instead of silently pairing numbers that might not
   match. `det_low_frames` is also now capped at `AUDIT_MAX_ITEMS` (200)
   with a paired `det_low_frames_truncated` field, matching this file's
   existing `candidates_truncated`/`yunet_truncated` convention instead
   of inventing a third, ad hoc truncation scheme.
4. **Still open:** greedy single-stage IoU association can let a
   low-confidence detection outbid a high-confidence one for the same
   track (real ByteTrack matches high-score detections first; this
   doesn't). Not touched — this is a real behavioral fix to the matching
   logic, not a visibility fix, and a bigger, separate piece of work.

**Don't turn `--continue-threshold` on for the real batch until item 4 is
addressed too.** It doesn't block the batch itself, which doesn't use
hysteresis at all.

## MediaPipe as a second detector — investigated, concluded, code removed

MediaPipe Face Landmarker was investigated as an independent second check
and found genuinely non-viable: functionally blind to faces at the size
they appear in this footage (a face EgoBlur scored 1.00 confidence
produced zero MediaPipe detections on the full frame, found instantly
once cropped/upscaled — a BlazeFace-family scale limitation, not a config
issue). All 4 frames where MediaPipe found something EgoBlur didn't were
MediaPipe's own false positives on hands/tools. **Do not wire it in
as-is.** `jobs/10_blur_egoblur.py` has zero MediaPipe references.

## Accepted residual risk — not fixable today, know about it

- **No independent second-opinion detector is actually running.** YuNet
  weights were never sourced; MediaPipe proved non-viable (above). The
  low-threshold sweep on EgoBlur's *own* detections is the only signal
  against a genuinely missed face. State this honestly in the eventual
  dataset card.
- **The PEP 723 header pins torch to a CUDA-only index**, so the job
  cannot resolve its dependencies on macOS at all. Irrelevant on the
  RunPod pod; a real portability bug for anyone forking this on a Mac.

## SSH access — now properly set up (was a real time-sink this session)

The web terminal's paste buffer silently truncates large pastes (a
250-line script landed as ~20 lines with no error until it was run) —
this is what actually forced setting up real SSH instead of working
around it.

**Which SSH mode matters:** RunPod offers two. **Basic SSH**
(`ssh <id>@ssh.runpod.io`) does **not** support SCP/SFTP — useless for
getting files onto the pod, which was the actual problem. **Full SSH over
exposed TCP** (`ssh root@<ip> -p <port>`) does support SCP and is what
actually unblocked file transfer. Requires: `22` listed under the pod's
**Expose TCP Ports**, and `sshd` actually running in the container
(`pgrep -a sshd`; start with `service ssh start` or
`mkdir -p /run/sshd && /usr/sbin/sshd` if not).

**Made durable across pods, not just this one:** `scripts/runpod_setup.sh`
now keeps the real `authorized_keys` on `/workspace/.ssh/` (the volume)
and symlinks `/root/.ssh/authorized_keys` to it — mirroring the exact
same pattern the script already used for the rclone config (container
disk wiped on every stop, volume survives). Populate
`/workspace/.ssh/authorized_keys` once, and every future pod that mounts
this same volume has working SSH automatically, no per-pod key-pasting.
It also rescues any key RunPod's own account-level injection may have
already written to container disk before replacing the file with the
symlink, so it won't clobber a working setup. **Caveat the user
themselves flagged and was right about:** this only helps for pods that
reattach *this specific* volume — for a genuinely different pod/volume,
also register the key once at **Console → User Settings → SSH Keys**
(RunPod's own account-level mechanism, works on any pod using a
standard template, no volume dependency at all).

This mechanism is currently **uncommitted** — see below.

## How we got here (condensed timeline, for context on *why*)

1. `test-run-1` — high `fill_integrity_violations`, faces visible at 35
   flagged timestamps, mostly at the *leading edge* of a face entering
   frame.
2. Root-caused: `build_tracks`' backward hold was 10x shorter than the
   forward hold. Fixed (`0ab77da`, `dcad7e1`) to be symmetric by default.
3. Tried `--face-threshold 0.20` (`test-run-2`) — a mistake, over-redacted
   the wearer's own hands.
4. Built `sweep_params.py`; `--hold-frames 45` alone (threshold back at
   0.30) gave **35/35** coverage. Became `test-run-3`.
5. Explored hysteresis and MediaPipe-as-second-detector. Hysteresis kept,
   off by default; MediaPipe rejected.
6. `test-run-3`'s fill-integrity number finally confirmed with real
   severity evidence (`diag_integrity.py`), not just a visual spot-check
   — see "Do this first" above.
7. Built `det_low`/`n_low_absorbed` visibility fixes for hysteresis, ran
   an adversarial code review against them, found 2 real correctness bugs
   plus 2 design issues in the fix itself, fixed all 4 (mutation-tested).

## RunPod gotchas already fixed — don't re-discover these

- **Container disk is wiped on every pod `stop`; `/workspace` (the
  volume) survives.** `uv`, `ffmpeg`, `ffprobe`, `rclone`, and now SSH's
  `authorized_keys` all live on `/workspace`, never installed/written via
  a path that lands on container disk.
- **`rclone`'s config** is pinned to `/workspace/.rclone.conf` via
  `$RCLONE_CONFIG`.
- **After any pod restart, or any new terminal connecting to an already-
  running pod**, run:
  ```bash
  cd /workspace/egoannote && git pull && bash scripts/runpod_setup.sh
  source /workspace/env.sh
  ```
  The `source` step matters — a script cannot export into its parent
  shell.
- **Long jobs must run detached**, or a browser/terminal disconnect kills
  them mid-run:
  ```bash
  cd /workspace/egoannote && setsid nohup env PATH="/workspace/bin:$PATH" \
    UV_CACHE_DIR=/workspace/.uv-cache /workspace/bin/uv run jobs/10_blur_egoblur.py \
    <args> > /workspace/run.log 2>&1 < /dev/null &
  tail -f /workspace/run.log   # Ctrl-C only stops watching, not the job
  ```
- **The 8-hour watchdog (`arm_watchdog`) almost certainly can't fire** —
  `runpodctl` was never configured with an API key. Stop the pod
  yourself when done.
- **Community Cloud can fail to resume** ("not enough free GPUs"). Try
  Resume a few times first; a Network Volume survives independent of any
  specific pod if it keeps happening.
- **The detection checkpoint is genuinely resumable and fsynced per
  batch.** `--face-threshold` / `--hold-frames` / `--dilate-scale` /
  `--motion-margin-px` do **not** invalidate it; `--gen`, weights,
  `--sweep-threshold`, `--nms-iou`, `--detect-hz`, or adding/removing
  `--lp-weights-gen2` **do**.
- **The web terminal silently truncates large pastes** with no error
  until you try to run the result. Use real SSH (see above) for anything
  longer than a couple of lines.

## Diagnostic scripts

All standalone, read-only tools operating on a run's own
`*.manifest.json` + `checkpoints/*.jsonl` — **zero GPU cost**. None are
committed to the repo (by design — throwaway analysis tools, not
pipeline code).

- **`diag_integrity.py`** — **confirmed present, at
  `/workspace/diag_integrity.py` on the pod** (delivered via `scp` once
  SSH was working; the web terminal paste kept truncating it). Rebuilds a
  run's *exact* `fill_map` by replaying `build_tracks`/
  `tracks_to_fill_map` from the real detection checkpoint against the
  config recorded in the run's own manifest, re-decodes the
  already-encoded output, and reports per-violation severity instead of
  a bare pass/fail count. Validated by sanity check: its rebuilt
  `checked`/`violations` matched `test-run-3`'s manifest exactly
  (36731/23216) before its severity numbers were trusted. Usage:
  ```bash
  uv run diag_integrity.py \
    --manifest /workspace/out2/GX010057.manifest.json \
    --checkpoint-dir /workspace/out2/checkpoints \
    --job-script /workspace/egoannote/jobs/10_blur_egoblur.py \
    --ffmpeg /workspace/bin/ffmpeg
  ```
  Known limitation: doesn't replay `--forced-boxes` (fine for
  `test-run-3`, which didn't use it — check before reusing on a
  different run's manifest).
- **`sweep_params.py`** and **`analyze_timestamps.py`** — presence
  unconfirmed as of this rewrite; regenerate from conversation history
  if needed, or ask again — they're small. (`sweep_params.py` is the
  tool that actually produced `test-run-3`'s settings: tests named
  parameter combinations against a run's existing checkpoint, reporting
  coverage of a timestamp list plus bottom-third-of-frame area as a
  hand-leakage proxy. `analyze_timestamps.py` classifies human-flagged
  MM:SS timestamps as `DETECTED_FILTERED`/`NEVER_DETECTED`/`COVERED`.)
- **`pick_threshold.py`**, **`inspect_frame.py`**, **`annotate_local.py`**
  — presence unconfirmed; see prior session notes if needed.

## What's uncommitted right now

Sitting in the working tree, tested (279 tests passing) and
mutation-tested, but **not yet committed**:

- `jobs/10_blur_egoblur.py` — the `det_low`/`n_low_absorbed` visibility
  fixes described above (final, adversarial-review-hardened version).
- `tests/test_blur_job.py` — matching new/updated tests.
- `scripts/runpod_setup.sh` — the durable SSH `authorized_keys` symlink
  mechanism described above.

Run `git status`/`git diff` before assuming this handoff describes what's
actually on `HEAD` — it doesn't yet.

## What's downstream and genuinely not started

Once the batch of 16 clips clears EgoBlur:

- **MediaPipe hand tracking** — code exists, proven working, never run
  against real *blurred* output.
- **VLM captioning** — code exists; `models.toml` still ships placeholder
  model IDs/prices, needs two real models from *different labs* before
  any real run.
- **Segmentation** — deliberately unimplemented, blocked on real
  measurements from hands + captions that don't exist yet.
- **`verify/` and `pack/`** — both empty files. Dataset assembly/
  validation not started.

## Deferred — hand-tracking occlusion, revisit when hands actually gets wired up

Raised while planning ahead (WiLoR + MediaPipe for hand detection, and
separately DepthV3/SAM2 for depth), **not yet built, explicitly parked
for later**:

- **`src/egoannote/layers/hands.py`'s `run()` is fully generic today** —
  no redaction/fill_map awareness, no occlusion detection at all. Feed it
  a frame where a gray box covers part of a hand and it silently returns
  whatever MediaPipe/WiLoR guess, no warning. Nothing currently
  distinguishes "no hand, correctly" from "hand was there, occluded by
  our own redaction."
- **A real tension worth resolving, not just noting:** the README/
  AGENTS.md's stated design runs hand tracking on the *redacted* video
  specifically (defense in depth — this stage never touches original
  pixels at all). That default is only safe if redaction essentially
  never legitimately covers a hand — which this session's own
  measurements directly disproved for the *pre*-fix pipeline (77.7% of
  redacted area was noise, much of it landing on hands) and
  significantly improved but did **not** prove is now zero. Decide
  explicitly whether "hands on redacted video" stays the default, or
  switches to original — see the next point for why original is likely
  right for DepthV3/SAM2 at least.
- **DepthV3 and SAM2** (if used for general segmentation, separate from
  Stage II's DINO/SAM2 de-identification use): should almost certainly
  run on the **original** video — both need real pixel signal for
  quality (a flat gray box gives a depth model zero signal and produces
  a hallucinated value or an edge artifact exactly where the wearer's
  hands/the person they're interacting with are), and both produce
  derived, non-identity-revealing outputs (depth arrays, mask arrays —
  not pixels), so this doesn't reopen the privacy question as long as no
  intermediate visual artifact (debug overlay, cached crop) derived from
  the original ever reaches a published directory. Same `DO-NOT-SHIP`-
  style boundary Stage II already has, applied to whatever pipeline
  produces these.
- **Two ways to actually detect a hand frame corrupted by redaction,
  neither built yet:**
  1. Cheap: watch MediaPipe/WiLoR's own per-frame confidence. An
     unexplained drop or gap — not a smooth "hand left frame" pattern —
     is suspect.
  2. Precise: rebuild the exact `fill_map` for the clip from its
     detection checkpoint + manifest (same technique as
     `diag_integrity.py`/`diag_hand_noise.py` this session — zero GPU
     cost) and directly check spatial overlap between the hand-detection
     region and any redacted box for that frame. Overlap → mark that
     frame's hand data explicitly unreliable rather than trusting it.
     Same "a check that can't verify something is never a pass"
     principle the rest of this pipeline is built on.

## Immediate next action, concretely

1. Decide whether to commit the uncommitted work above (recommended —
   it's tested and mutation-tested, just never landed).
2. Optionally address the resumed-batch config-drift gap (see "Still
   open" above) — lower urgency, doesn't block a single uninterrupted
   16-clip run.
3. Restore/locate the original `GX010057`, regenerate its redacted video
   with the settings block above, retain its passing manifest, and verify a
   complete decode before MediaPipe/VLM. The existing
   `GX010057.blurred.mp4` is corrupt and must not be annotated or shipped.
4. Configure the two real VLM models, then run the five-window pilot on the
   newly generated redacted video. Keep Stage 2 paused; WiLoR/SAM2/DepthV3
   remain later, private original-only stages.
5. After that pilot passes, run the 16-clip EgoBlur batch with the settings
   block above, one clip's `--input-dir` pointed at all 16 source files via
   `rclone` (remote name `gdrive`, folder `nbt-videos`, config at
   `/workspace/.rclone.conf`).
