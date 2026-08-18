# egoannote — agent briefing

This file exists so any coding agent (Codex, Claude, etc.) can pick up this
repo cold. Written 2026-08-11, last updated same day after two more fixes
landed. Repo HEAD at time of writing: **`d0cbe10`**, tree clean, 272 tests
passing (`uv run --extra test pytest tests/ -q`).

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
tests/                       pytest, 267 tests. Run before AND after any
                             change: `uv run --extra test pytest tests/ -q`
handover.md                  a PREVIOUS session's own continuation notes
                             (gitignored, may or may not still exist on
                             disk — don't assume it's current).
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
| EgoBlur redaction | Heavily built, heavily tested, heavily reviewed. One real clip (`GX010057`) has been run three times while tuning parameters. **Not yet confirmed safe to scale to the other 15 clips** — see "Immediate priority" below. |
| MediaPipe hands | Code complete, unit-tested, proven working (Metal-accelerated, fast). Never run against real *redacted* output — only a synthetic test clip. |
| VLM captioning | Code complete, unit-tested. `models.toml` still has placeholder model IDs / $0.00 prices — cannot run for real until filled in with two real models from different labs. |
| Segmentation | Not started. Deliberately — blocked on measurements from the two stages above. |
| verify/pack (dataset assembly) | Not started. Empty files. |

## Immediate priority — the EgoBlur redaction work

This is what the previous session spent most of its time on. Read this
section before doing anything with `jobs/10_blur_egoblur.py`.

**The one thing to check first, genuinely unresolved:** an early run
(`test-run-1`, default settings) showed `fill_integrity_violations: 14096`
— 65% of checked redaction boxes failing an exact-pixel-value check. The
developer watched the actual output video and it looked correctly
redacted, so the working theory was that the *check* is too strict
against real H.264 encoder ringing at box edges, not a real privacy leak
— but this was **never confirmed on the current best-known settings**
(`test-run-3`, see below). If you have access to the pod, the very first
thing to do is:

```bash
cat /workspace/out2/GX010057.audit_summary.md
```

and read the actual `fill_integrity_violations` number under the current
settings before trusting anything is ready to scale.

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

**Fixed since this file was first written** (commit `d0cbe10`): the
`max_fill_area_frac` dead-canary gate, and hysteresis's drift bound
(it counted absorption events instead of real video frames — now budgets
`max_low_run * stride` frames since the track's last confident detection).
Both mutation-tested. What's left:

1. **A resumed multi-clip batch can silently mix redaction configs.**
   `process_clip`'s manifest-skip only checks whether a manifest file
   exists, never what config produced it — unlike `detection_pass`'s own
   `checkpoint_fingerprint()`, which does this correctly for the
   detection step. If a batch run is interrupted and resumed with a
   changed flag, already-finished clips silently keep old settings with
   no warning, and `run_manifest.json` doesn't even record which
   thresholds were used. Fix: extend fingerprinting to cover
   redaction-relevant config and check it at the manifest-skip site.
2. **Two-threshold hysteresis exists in the code but is off by default**
   (`--continue-threshold 0`) and has known unresolved issues if ever
   turned on: one threshold value applies to both face and plate classes,
   greedy association can let a low-confidence detection outbid a
   high-confidence one for the same track, and its audit fields
   (`n_low_absorbed`, the `det_low` source tag) don't fully reach
   human-readable output. Two real privacy regressions in this feature
   have already been found and fixed (an absorbed low-confidence
   detection could make a confirmed face's coverage *worse* while
   silencing the audit check that would have caught it; its drift bound
   didn't actually bound drift — see `git log --oneline | grep -iE
   "hysteresis|drift bound"`) — but the items above were left open.
   Don't enable `--continue-threshold` for a real batch without
   addressing them.
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
