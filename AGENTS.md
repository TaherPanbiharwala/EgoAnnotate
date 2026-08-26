# egoannote — agent briefing

This file exists so any coding agent (Codex, Claude, etc.) can pick up this
repo cold. Written 2026-08-11, substantially updated in a later session
after the EgoBlur fill-integrity question got resolved with real evidence
and two more hysteresis visibility bugs got fixed. Repo HEAD at time of
writing: **`5cca081`** — but the working tree has real, tested,
**uncommitted** changes on top (the `det_low`/`n_low_absorbed` fixes below,
plus an SSH durability fix in `scripts/runpod_setup.sh`); run `git status`
before trusting this describes `HEAD` exactly. 279 tests passing
(`uv run --extra test pytest tests/ -q`), including the uncommitted work.

If you're an agent starting a fresh session here, read this whole file
before touching code — several hard-won lessons below aren't visible from
reading the source alone, and re-learning them costs real GPU money.

## What this project is

A solo developer is turning a shelved egocentric-video-annotation startup
into a public, forkable annotation pipeline — a portfolio piece, not a
company. The plan: publish ~1.5 hours of the developer's own GoPro
footage (worn while working in a family paint/hardware shop) as an
annotated dataset on HuggingFace, with faces automatically redacted for
privacy, then write an honest post-mortem about the whole project.

Pipeline, in order:

```
EgoBlur (GPU pod)  →  MediaPipe hands (local CPU)  →  VLM captioning (API)
   redacts faces         tracks hand landmarks         describes actions
   and plates            on the REDACTED video          per 6s window
                                                              ↓
                                          segmentation (NOT implemented yet)
                                          turns captions into action segments
                                                              ↓
                                          verify/pack (NOT implemented yet)
                                          assembles + validates HF dataset
```

**Only the EgoBlur stage needs a GPU.** Everything else runs free, locally,
on the developer's Mac (MediaPipe is CPU/Metal; captioning is an HTTP API
call). This matters: don't assume the whole pipeline needs to run on a
rented pod.

## Repo layout

```
jobs/10_blur_egoblur.py     the ONE GPU job. Self-contained PEP 723 script
                             (see below for why). ~2800 lines, heavily
                             tested and heavily reviewed — read its module
                             docstring before changing anything in it.
jobs/_contract.py           the shard-metadata contract other future GPU
                             jobs (hand-pose, depth, SLAM) will vendor by
                             copy — NOT imported (see PEP 723 section).
scripts/22_blur_review.py   builds an HTML gallery of flagged detections
                             from a run's manifest, for human review before
                             publishing.
scripts/runpod_setup.sh     idempotent RunPod environment setup. Re-run
                             after every pod restart (see "RunPod" below).
scripts/demo.py             zero-setup plumbing smoke test — no GPU, no
                             API key, runs the whole non-GPU pipeline
                             against a bundled synthetic clip.
src/egoannote/
  layers/hands.py           MediaPipe hand tracking. Built, tested, proven
                             working — NOT yet run against real blurred
                             output, only synthetic test data.
  layers/caption.py         VLM captioning via any OpenAI-compatible API.
                             Built, tested — models.toml still has
                             placeholder model IDs, not yet run for real.
  layers/segment.py         Turns captions into action segments.
                             DELIBERATELY UNIMPLEMENTED — its own
                             docstring explains it's blocked on real
                             measurements (boundary error, velocity-minima
                             density) that don't exist until hands +
                             captions have run on real footage. Don't
                             implement this until that data exists.
  backends/                 VLM backend abstraction (OpenAI-compatible +
                             a deterministic FakeBackend for tests/demo).
  media/probe.py            ffprobe wrapper, single source of truth for
                             fps/duration (both hand and caption tracks
                             must derive timestamps from this one clock).
  store.py                  SQLite + parquet persistence. Currently only
                             wired for the hands layer.
  schema.py                 Dataclasses: HandFrame, WindowCaption, Segment.
verify/, pack/               EMPTY FILES. Nobody has started dataset
                             assembly/validation. Real work, not stubs to
                             delete.
models.toml                 VLM model registry. Ships with placeholder
                             entries; needs 2 real models from DIFFERENT
                             labs (see file's own comments on why) before
                             any real captioning run.
tests/                       pytest, 279 tests. Run before AND after any
                             change: `uv run --extra test pytest tests/ -q`
handover.md                  a PREVIOUS session's own continuation notes.
                             Tracked in git (not gitignored) — read it,
                             it's usually more current/detailed than this
                             file on whatever the last session actually did.
```

## Why the GPU job is a single self-contained script

`jobs/10_blur_egoblur.py` starts with a PEP 723 header (`# /// script ...`)
declaring its own dependencies (`egoblur`, `torch`+CUDA, `opencv`, etc.)
and runs via `uv run jobs/10_blur_egoblur.py <args>` — no install step, no
shared virtualenv. This is deliberate: it runs on an ephemeral rented GPU
pod with no persistent environment, and future GPU jobs (hand-mesh
estimation, depth, SLAM — see `jobs/_contract.py`'s comment for the
planned stage names) will each need **mutually incompatible** torch/CUDA
versions. One shared environment can't satisfy all of them at once; one
script per job, each fully self-contained, can. `jobs/_contract.py`
(the shard-writing metadata format other GPU jobs will use) is meant to
be **vendored by copy** into each new job script, not imported — a real
import would break the isolation that's the whole point.

**Known bug in this pattern, not yet fixed:** the PEP 723 header pins
`torch`/`torchvision` to a CUDA-only package index
(`download.pytorch.org/whl/cu128`), so the script cannot resolve its
dependencies on macOS at all. Irrelevant while running on the Linux GPU
pod (the only place this job actually runs); a real portability bug if
anyone tries to run it locally on a Mac.

## Design principles this codebase already learned the hard way

These recur throughout `jobs/10_blur_egoblur.py` and its ~270 tests.
Violating them silently reintroduces bugs that were already found, fixed,
and pinned with a regression test:

- **Fail closed, always.** Every verification check (`check_fill_integrity`,
  `check_yunet`) raises on decode failure or frame-count mismatch rather
  than returning "0 problems found." A silent decode failure must never
  be indistinguishable from "verified clean." This project already shipped
  and caught one real instance of this exact bug.
- **Detect low, filter later.** Detectors run at a low `sweep_threshold`
  floor (0.10); the higher operating threshold (`--face-threshold`) is
  applied *afterward*, when deciding what to actually redact. This is
  what makes the audit's "did we miss anything near the threshold"
  check possible at all — building detectors at the operating threshold
  directly makes that check structurally dead (this exact bug shipped
  once and was fixed).
- **A check that didn't run is never a pass.** Skipping `--yunet-model`
  degrades the status to `PASS_AUTOMATED_NO_YUNET`, a visibly weaker
  claim — never silently `PASS_AUTOMATED`.
- **Provenance travels with the artifact.** Every parameter that changes
  what a score or a pixel *means* (detector generation, weight file
  identity, resize scale, color range, redaction thresholds) gets
  recorded in the manifest — because "1.5GB became 644MB, why?" or "is
  this clip's `back_hold_frames` really what built these tracks?" must be
  answerable from the manifest alone, months later, without this
  conversation's context. (There is a currently-open bug where this
  guarantee is *not yet complete* for a resumed/interrupted batch — see
  "Known open work" below.)
- **Measure before you build.** `layers/segment.py` is unimplemented
  specifically because its tunable constants need real measurements from
  data that doesn't exist yet. Don't pre-guess them.
- **Mutation-test every fix.** The pattern used throughout: revert the
  fix, confirm the new test fails, restore, confirm it passes. A test
  that still passes against a reverted fix is worse than no test — it's
  been caught doing exactly that at least four separate times in this
  codebase's history (grep test file docstrings for "vacuous" if curious
  why this is stated so bluntly).

## Current status, stage by stage

| Stage | Status |
|---|---|
| EgoBlur redaction | Heavily built, heavily tested, heavily reviewed. One real clip (`GX010057`) has been run three times while tuning parameters. **The fill-integrity question that used to gate scaling is resolved** — see "Immediate priority" below. The pre-redaction PoseLandmarker prior is permanently shadow-only. The experimental `--hand-suppress-wearer-hands` option can withhold only a repeated face track whose every raw box is nearly fully inside the same stable, close camera-wearer hand. Active behavior is pinned to 10 Hz, two requested hands, and the approved Hand Landmarker SHA-256, and rejects stale policy settings. It never becomes a detector ROI, never reduces detection on other people, records private evidence, and a nonzero count forces `NEEDS_REVIEW`. Amber keeps its four-hit requirement but now accepts >=95% overlap with one stable hand; pink retains raw detector hits but caps only generated interpolation/hold context after two raw hits at >=90% overlap. A nonzero amber suppression or pink demotion forces `NEEDS_REVIEW`; both require the pinned two-hand artifact, never the diagnostic four-hand artifact. A distinct four-hand diagnostic prior/preview makes multi-person scenes reviewable: `H#` is only a short-continuity anchor, while MediaPipe Left/Right, entry edge, and current screen side are private hints—not identity or suppression evidence; `3+ hands` triggers review only. The existing two-hand private prior for `GX010057` was generated successfully on 2026-08-25; regenerate the new four-hand diagnostic preview before active review. A batch must use a private report directory so every clip retains its suppression evidence. |
| MediaPipe hands | Code complete and unit-tested. A first real-redacted pilot on 2026-08-24 was correctly rejected: `GX010057.blurred.mp4` has a 99.5% decode-error rate, yielding only 14 of 4,067 expected samples. The pipeline now fails partial decodes and refuses to resume a truncated hand Parquet. Keep this redacted-video annotation stage distinct from the new private pre-redaction Hand Landmarker prior; it still needs a verified redacted input and a fresh run directory. |
| VLM captioning | The next-chat focus is to review and edit this code/prompt, not execute a paid batch. The intended dense representation is a six-second holistic `activity.caption`, bounded atomic `actions[]`, and a caption plus structured state for each visible anatomical left/right hand. Review `prompts/caption_v4.txt`, `layers/caption.py`, parsing, and schema together; captions must use redacted video only. Real execution remains blocked until `models.toml` has two real cross-lab models with provider/price pins and `OPENROUTER_API_KEY`, plus a verified redacted pilot input. |
| Segmentation | Not started. Deliberately — blocked on measurements from the two stages above. |
| verify/pack (dataset assembly) | Not started. Empty files. |

## Immediate priority — the EgoBlur redaction work

Read this section before doing anything with `jobs/10_blur_egoblur.py`.

**The fill-integrity question is RESOLVED, with actual evidence — not
just a visual spot-check.** An early run (`test-run-1`, default settings)
showed `fill_integrity_violations: 14096` (~65% of checked boxes). The
working theory was "the check is too strict against H.264 encoder
ringing, not a real leak" — accepted at the time purely because the
developer watched the output video and it looked fine.

That got properly confirmed in a later session. `test-run-3`'s real
numbers: `checked=36731`, `violations=23216` → **63.2%**, essentially
flat vs. `test-run-1` despite 69% more boxes being checked overall. More
importantly, a purpose-built diagnostic (`diag_integrity.py` — rebuilds
the exact `fill_map` from the real checkpoint, re-decodes the
already-encoded output, and measures actual pixel deviation instead of a
bare pass/fail) found **96.6% of violations are within 5 gray levels of
the allowed limit, and 0% are "severe" or "extreme"** — a real leaked
face would blow past the limit by 50-100+ gray levels (skin tone vs.
mid-gray `FILL_VALUE=128`); this doesn't. **Conclusion: the check is
too strict against ordinary encoder quantization, confirmed, not
assumed.** Don't re-litigate this unless settings change materially; if
you do need to re-check, `diag_integrity.py` is on the pod at
`/workspace/diag_integrity.py` — not committed to this repo, by design
(see `handover.md`'s "Diagnostic scripts" section: these are throwaway
analysis tools, not pipeline code, kept off the pod's `/workspace` or
regenerated on request rather than checked in).

**Current best-known-good settings** (arrived at empirically, not
guessed — a parameter-sweep tool was built specifically to test
combinations against already-computed detections at zero extra GPU
cost):

```bash
uv run jobs/10_blur_egoblur.py --input-dir /workspace/in --output-dir /workspace/out2 \
  --run-id test-run-3 --gen 2 \
  --face-weights-gen2 /workspace/weights/ego_blur_face_gen2.jit \
  --face-threshold 0.30 --hold-frames 45 \
  --gpu-rate-usd-per-hr <actual $/hr> --skip-shutdown
```

No `--lp-weights-gen2` (the license-plate model false-positives heavily
on printed cardboard text in this footage — dropped entirely, recorded
honestly as `lp_checked: false`). `--dilate-scale`/`--motion-margin-px`
left at defaults. A `--face-threshold 0.20` experiment was tried and
**was a real mistake** — it caught a few more faces but buried the
wearer's own hands under grey redaction boxes for most of the clip. Do
not lower the threshold again without re-testing against real footage
first (visually, not just by the numbers — this exact regression looked
fine in aggregate stats and only showed up when a human watched the
video).

## Known open work (real, scoped, not busywork)

**Fixed since this file was first written:** the `max_fill_area_frac`
dead-canary gate and hysteresis's drift bound (commit `d0cbe10`); plus,
**currently uncommitted but tested and mutation-tested**, two of the
three remaining hysteresis visibility gaps (see item 2 below) and a
durable SSH `authorized_keys` mechanism in `scripts/runpod_setup.sh`
(see "Operational notes for RunPod"). What's left:

1. **A resumed multi-clip batch can silently mix redaction configs.**
   `process_clip`'s manifest-skip only checks whether a manifest file
   exists, never what config produced it — unlike `detection_pass`'s own
   `checkpoint_fingerprint()`, which does this correctly for the
   detection step. If a batch run is interrupted and resumed with a
   changed flag, already-finished clips silently keep old settings with
   no warning, and `run_manifest.json` doesn't even record which
   thresholds were used. Fix: extend fingerprinting to cover
   redaction-relevant config and check it at the manifest-skip site.
   Not touched this session.
2. **Two-threshold hysteresis exists in the code but is off by default**
   (`--continue-threshold 0`). Of its original 4 acceptance criteria
   before it's safe to turn on, 3 are now resolved:
   - ~~One threshold value shared between face and plate classes, only
     face's behaviour measured~~ — moot for how this project actually
     runs: `--lp-weights-gen2` is dropped project-wide, so no `cls ==
     "lp"` detection is ever produced. A config fact, not a code fix.
   - ~~The `det_low` source tag never reached human-readable output~~ —
     fixed properly on the second try. The first attempt threaded a new
     out-parameter through `tracks_to_fill_map`'s hot loop; an
     adversarial code review found this unnecessary — `low_absorbed`
     (populated during `build_tracks`, never deleted) already carries
     the same frame indices. Final fix derives `det_low_frames` from
     `low_absorbed` directly in `process_clip`, with zero changes to
     `tracks_to_fill_map`.
   - ~~`n_low_absorbed` reached the JSON manifest but not the
     markdown~~ — fixed, plus two real bugs the same adversarial review
     caught in the fix itself: an unverifiable hardcoded claim ("0 when
     `--continue-threshold` is off" — `write_audit_summary` never
     actually sees `cfg.continue_threshold`) and a count/list mismatch
     (`n_low_absorbed` counts absorption *events*, `det_low_frames` is a
     *deduplicated* frame list — two tracks absorbing on the same frame
     makes these legitimately differ, now labeled explicitly instead of
     looking like a bug). `det_low_frames` is also now capped at
     `AUDIT_MAX_ITEMS` with a paired `det_low_frames_truncated` field,
     matching this file's own `candidates_truncated`/`yunet_truncated`
     convention instead of inventing a new one.
   - **Still open:** greedy single-stage IoU association can let a
     low-confidence detection outbid a high-confidence one for the same
     track (real ByteTrack matches high-score detections first; this
     doesn't). A real behavioral fix to the matching logic, not a
     visibility fix — bigger, separate work. Don't enable
     `--continue-threshold` for a real batch until this is addressed.
3. **YuNet is now an independent, post-redaction review signal, not an
   approval mechanism.** The standalone `egoannote-run verify-yunet`
   command validates the redacted-video hash and full detector checkpoint,
   rebuilds EgoBlur's fill map, then runs YuNet only on the redacted video.
   It emits private temporal candidates and never changes the redaction.
   The Pose prior may make likely wearer-limb candidates lower review
   priority, but it may never discard them or make YuNet scan an ROI. A human
   marks each candidate; only `confirmed_face` tracks become `--forced-boxes`
   for a corrective re-run. MediaPipe Face Landmarker remains non-viable as a
   naive whole-frame independent detector at this footage's face scales.

## Operational notes for RunPod

- **The pod's container disk is wiped on every `stop`; only
  `/workspace` (the mounted volume) survives.** `scripts/runpod_setup.sh`
  installs `uv`, `ffmpeg`, `ffprobe`, and `rclone` to `/workspace/bin`
  specifically because installing via `apt-get` (which lands in
  `/usr/bin`) silently evaporates on the next restart — this cost
  several wasted run attempts before being fixed. If you add any new
  tool dependency to the pod, install it to the volume, not the
  container disk.
- After any pod restart or new terminal:
  ```bash
  cd /workspace/egoannote && git pull && bash scripts/runpod_setup.sh
  source /workspace/env.sh
  ```
- **Long-running jobs must be detached**, or a disconnected
  browser/terminal kills them mid-run:
  ```bash
  cd /workspace/egoannote && setsid nohup env PATH="/workspace/bin:$PATH" \
    UV_CACHE_DIR=/workspace/.uv-cache /workspace/bin/uv run jobs/10_blur_egoblur.py \
    <args> > /workspace/run.log 2>&1 < /dev/null &
  tail -f /workspace/run.log
  ```
- **The detection checkpoint is genuinely resumable and fsynced per
  batch.** Re-running the identical command after a crash resumes near-
  instantly instead of re-paying ~25 minutes of GPU time — look for
  `resuming, N detection-frames already attempted` in the log.
  `--face-threshold`, `--hold-frames`, `--dilate-scale`, and
  `--motion-margin-px` do **not** invalidate the checkpoint by design
  (they're applied post-hoc); `--gen`, weight files,
  `--sweep-threshold`, `--nms-iou`, `--detect-hz`, and adding/removing
  `--lp-weights-gen2` correctly do invalidate it.
- The 8-hour job watchdog (`arm_watchdog`) likely cannot actually fire —
  `runpodctl` on the working pod was never authenticated with an API
  key. Don't rely on it as a real cost backstop; stop the pod manually.
- **The web terminal's paste buffer silently truncates large pastes** —
  a 250-line script landed as ~20 lines with no error until it was run.
  Use real SSH for anything longer than a couple of lines. RunPod offers
  two SSH modes: **Basic SSH** (`ssh <id>@ssh.runpod.io`) does **not**
  support SCP/SFTP, so it can't actually solve the paste problem; **Full
  SSH over exposed TCP** (`ssh root@<ip> -p <port>`) does. Needs `22`
  listed under the pod's Expose TCP Ports, and `sshd` actually running
  (`pgrep -a sshd`; `service ssh start` or `mkdir -p /run/sshd &&
  /usr/sbin/sshd` if not). `scripts/runpod_setup.sh` now keeps the real
  `authorized_keys` on `/workspace/.ssh/` and symlinks
  `/root/.ssh/authorized_keys` to it — same wiped-container-disk pattern
  as everything else above — so populating
  `/workspace/.ssh/authorized_keys` once makes SSH work automatically on
  every future pod that mounts this same volume. For a genuinely
  different pod/volume, also register the key at Console → User Settings
  → SSH Keys (RunPod's own account-level mechanism, no volume
  dependency).

## How to work in this repo

```bash
uv sync                                  # install deps (non-GPU deps only)
uv run --extra test pytest tests/ -q     # run the full test suite
uv run scripts/demo.py                   # zero-setup plumbing smoke test
```

The GPU job (`jobs/10_blur_egoblur.py`) is tested WITHOUT a GPU by loading
it via `importlib` (its filename starts with a digit, so it can't be a
normal import — see `tests/conftest.py`'s `_load()` helper) and testing
its pure logic (tracking, coverage math, audit gating) directly, with the
decoder/detector mocked. One test file
(`tests/test_blur_encode_roundtrip.py`) deliberately uses a REAL ffmpeg
encode (via the `imageio-ffmpeg` static binary) rather than a mock — two
real bugs shipped and were only caught once mocked-decoder tests were
replaced with this real encode/decode round trip. Keep that file's
tests real; don't mock them for convenience.

Before committing: run the full test suite, and if you change anything
in `jobs/10_blur_egoblur.py`'s tracking, redaction, or audit logic,
mutation-test your own fix (revert it, confirm the new test fails,
restore) before considering it done — this project has a real, repeated
history of tests that looked correct but passed against a reverted fix.
