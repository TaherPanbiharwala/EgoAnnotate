# egoannote — agent briefing

This file exists so any coding agent (Codex, Claude, etc.) can pick up this
repo cold. Written 2026-08-11 and updated on the Stage II feature branch on
2026-08-20. The branch includes clean `master` baseline **`9acfe07`** plus
the reviewed Stage II plan and Milestones 1-5 implementation. The Milestone 5
implementation baseline is committed at **`dba85a9`**. The complete suite
was verified at **446 passing tests** with
`uv run --extra test pytest tests/ -q`. Run `git status`, then read
`handover.md` and `STAGE2_DEIDENTIFICATION_PLAN.md` before changing either
GPU stage.

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
EgoBlur (GPU pod)  →  Stage II DINO/SAM2 (same GPU pod)  →  MediaPipe hands (local CPU)  →  VLM captioning (API)
   first-pass boxes      recovers difficult face misses      tracks hand landmarks         describes actions
                                                              on the REDACTED video          per 6s window
                                                              ↓
                                          segmentation (NOT implemented yet)
                                          turns captions into action segments
                                                              ↓
                                          verify/pack (NOT implemented yet)
                                          assembles + validates HF dataset
```

**EgoBlur and the eventual real DINO/SAM2 Stage II need a GPU.** They run
sequentially on the same pod and persistent `/workspace` volume, but in
independent PEP 723 environments because their Torch/CUDA stacks are not
assumed compatible. Stage II Milestones 1-5 are locally testable with fake
adapters and real local ffmpeg fixtures; the real model adapters are lazy and
cache-only. Milestone 5 provides the persistent setup and operator paths, but
downloads, CUDA/model-load verification, and real inference begin in Milestone 6.
MediaPipe remains local CPU/Metal and captioning remains an HTTP API call.

## Repo layout

```
jobs/10_blur_egoblur.py     first GPU job. Self-contained PEP 723 script
                             (see below for why). ~2800 lines, heavily
                             tested and heavily reviewed — read its module
                             docstring before changing anything in it.
jobs/20_deidentify_stage2.py
                             Stage II self-contained PEP 723 job. Milestones
                             1-5 contain schemas, strict Stage I validation,
                             layered fingerprints, atomic state, immutable
                             artifacts, full/tiled DINO orchestration, anchor
                             checkpoints, threshold reuse, bounded forward/
                             reverse SAM propagation, fail-closed DINO/manual
                             fallback, compressed per-window mask shards, and
                             deterministic fake adapters. SAM shards bind the
                             verified config, runtime source tree, Torch, and
                             CUDA identities, plus Stage I-only YUV rendering,
                             technical verification, atomic output promotion,
                             guarded processing finalization, private labels,
                             immutable review/release gates, stop/resume, and
                             the complete operator command surface. Real model
                             adapters are lazy/cache-only until Milestone 6.
jobs/_contract.py           the shard-metadata contract other future GPU
                             jobs (hand-pose, depth, SLAM) will vendor by
                             copy — NOT imported (see PEP 723 section).
scripts/22_blur_review.py   builds an HTML gallery of flagged detections
                             from a run's manifest, for human review before
                             publishing.
scripts/runpod_setup.sh     idempotent RunPod environment setup. Re-run
                             after every pod restart (see "RunPod" below).
scripts/runpod_setup_stage2.sh
                             persistent, hash-verifying Stage II RunPod setup.
scripts/runpod_stage2.sh    thin wrapper for the Stage II operator commands.
docs/stage2-operator.md     setup, command, correction, review, and release guide.
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
tests/                       pytest suite. Run the complete suite before AND after any
                             change: `uv run --extra test pytest tests/ -q`
handover.md                  the latest continuation notes for the Stage II
                             worktree. Tracked in git (not gitignored); it
                             contains exact commits, review fixes, remaining
                             concerns, and the next-milestone entry point.
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
| EgoBlur redaction | Heavily built, heavily tested, heavily reviewed. One real clip (`GX010057`) has been run three times while tuning parameters. **The fill-integrity question that used to gate scaling is resolved** — see "Immediate priority" below. One known, lower-urgency gap remains (resumed-batch config drift, see "Known open work") before an unattended 16-clip run. |
| Stage II DINO/SAM2 | Milestones 1-5 complete on `feature/stage2-deidentification`: contracts, DINO proposals/reuse, bounded SAM propagation/fallback shards, verified Stage I-only rendering, private review/release gates, and the persistent operator workflow. Official model adapters remain pinned and lazy. Continue with the Milestone 6 real-GPU smoke test and `GX010057` calibration in `STAGE2_DEIDENTIFICATION_PLAN.md`. |
| MediaPipe hands | Code complete, unit-tested, proven working (Metal-accelerated, fast). Never run against real *redacted* output — only a synthetic test clip. |
| VLM captioning | Code complete, unit-tested. `models.toml` still has placeholder model IDs / $0.00 prices — cannot run for real until filled in with two real models from different labs. |
| Segmentation | Not started. Deliberately — blocked on measurements from the two stages above. |
| verify/pack (dataset assembly) | Not started. Empty files. |

## Stage II continuation point

Milestones 1-5 and their review fixes are committed separately on
`feature/stage2-deidentification`:

- implementation: `03762c5`, `0c8bc87`, `5905aee`;
- review hardening: `a31e65c`, `ce08d73`, `c5609ca`;
- documentation refresh: `c7b4005`;
- verified Stage I-only rendering: `f0754f1`;
- private review, release gates, and operator workflow: `dba85a9`.

The review's P1 provenance defects are resolved. Reused DINO proposals must
match their finalized checkpoint rows; SAM fallback review flags cannot be
canonically removed; and every SAM shard fingerprint binds the pinned SAM2
revision, configuration file and hash, installed source-tree hash, Torch
version, and CUDA version. Milestone 4 adds the verified renderer. Milestone 5
adds private content-addressed labels/evidence, immutable hash-bound human
reviews, fail-closed release checks, explicit stop/resume/recompute, the full
operator command surface, and verified persistent setup. The Stage II job is
now version `0.5.0` with code version `milestone-5`.

Continue with **Milestone 6 — real-GPU smoke test and full `GX010057`
calibration** in `STAGE2_DEIDENTIFICATION_PLAN.md`. Run the persistent setup on
the pod, verify offline CUDA/model loading, attest extracted frame payloads,
select the safe SAM window size, and only then begin the full threshold sweep.

Review concerns intentionally carried into later milestones:

- before real SAM2 execution, verify the extracted frame directory's actual
  count, ordering, and content identity rather than trusting only loader-supplied
  metadata;
- profile the real adapter's full-resolution GPU-to-CPU mask transfers and
  Python-list conversion during the Milestone 6 GPU pilot;
- describe the conservative fallback accurately as a held padded DINO box, not
  learned tracking or true interpolation.

These are planned real-adapter/performance tasks, not current failing tests.
The full handoff, exact runtime pin, validation evidence, and suggested first
prompt for a new session are in `handover.md`.

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
`/workspace/diag_integrity.py`. It is deliberately not committed to this
repo because it was a one-off analysis tool, not pipeline code. Regenerate it
from the historical session if the pod volume no longer contains it.

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
dead-canary gate and hysteresis's drift bound (commit `d0cbe10`), two of the
three remaining hysteresis visibility gaps (see item 2 below), and the durable
SSH `authorized_keys` mechanism in `scripts/runpod_setup.sh` (see
"Operational notes for RunPod"). Those fixes are part of the merged `master`
baseline used by this worktree. What's left:

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
3. **No independent second-opinion face detector is actually running.**
   `--yunet-model` was designed for this but real weights were never
   sourced. MediaPipe Face Landmarker was investigated as an
   alternative and found genuinely non-viable as a naive whole-frame
   check — it's structurally blind to faces at the size they appear in
   this footage (a face EgoBlur scored at 1.00 confidence, real and
   frontal and well-lit, produced zero MediaPipe detections at any
   threshold on the full frame, but was found instantly once cropped —
   a known BlazeFace-family scale limitation, not a config problem). The
   low-confidence sweep on EgoBlur's *own* detections is currently the
   only signal against a genuinely missed face. State this honestly if
   it makes it into a dataset card; don't silently claim independent
   verification that isn't happening.

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
