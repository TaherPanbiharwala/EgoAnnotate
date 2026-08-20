# egoannote Stage II session handover

Updated 2026-08-20 for the next coding session.

## Start here

- Worktree: `/Users/taherpanbiharwala/Desktop/Annotated_Data/egoannote-stage2`
- Branch: `feature/stage2-deidentification`
- Baseline HEAD before Milestone 4: `7cd5b9a`
- Base merged from `master`: `9acfe07`
- Milestone 4 implementation, tests, and documentation follow that baseline.
- Nothing from this branch has been pushed or merged into `master` by this
  session.

In a new chat, set the workspace to the worktree above, then read these files
in order:

1. `AGENTS.md`
2. `STAGE2_DEIDENTIFICATION_PLAN.md`
3. this `handover.md`
4. the module docstring and command parser in `jobs/20_deidentify_stage2.py`

Run these checks before changing code:

```bash
git status --short
git log --oneline -10
uv run --extra test pytest tests/ -q
```

## What is being built

Stage II is a privacy-recovery layer after EgoBlur:

```text
original video -> EgoBlur Stage I video -> DINO face proposals
               -> SAM2 temporal masks -> Stage II renderer -> human review
```

EgoBlur remains the first pass. Stage II can add redaction for faces EgoBlur
missed, but it cannot restore pixels hidden by an EgoBlur false positive.

DINO proposes difficult face locations at low thresholds. SAM2 turns accepted
boxes or manual seeds into masks and propagates them through bounded temporal
windows. If SAM2 fails or produces a pathological mask, the padded DINO box
remains as the fail-closed fallback and the interval is flagged for review.

## Decisions already locked

- Keep EgoBlur as Stage I and DINO plus SAM2 as Stage II.
- Run both GPU stages sequentially on one RunPod pod and persistent
  `/workspace` network volume.
- Keep their Torch/CUDA dependencies isolated in separate self-contained
  PEP 723 jobs.
- Stage II uses the original video for model inference but renders by adding
  masks to the already-redacted Stage I video. It never reconstructs Stage I
  pixels.
- A DINO anchor frame always retains a padded-box fallback, even when a SAM2
  mask is rejected.
- Use bounded overlapping SAM2 windows rather than global identity tracking.
- Persist immutable, compressed, bounding-box-cropped mask shards rather than
  a clip-wide full-resolution mask map.
- Use layered fingerprints so a DINO threshold experiment can reuse detector
  proposals without rerunning SAM or EgoBlur unnecessarily.
- `processing complete` and `review accepted` are separate immutable states.
  Automatic success is never publication approval.
- Rendering decodes Stage I as its only pixel source, uses constant YUV fill,
  verifies the encoded temporary artifact, and promotes it only after every
  technical check passes.
- The full `GX010057` clip will be privately labeled and calibrated before a
  broader rollout. Labels and review evidence stay under `DO-NOT-SHIP`.

## Completed milestones

The original milestone implementations are separate commits:

- `03762c5` — Milestone 1 contracts, Stage I validation, state, and local job
  skeleton.
- `0c8bc87` — Milestone 2 DINO proposal generation, tiling, checkpoints,
  fingerprints, resume, and threshold reuse.
- `5905aee` — Milestone 3 SAM2 bounded propagation, fallback safety, manual
  seeds, and immutable compressed mask shards.

The complete review then produced separate milestone-specific fix commits:

- `a31e65c` — Milestone 1 validation hardening, including safe output paths,
  valid zero-redaction Stage I input, malformed manual-seed handling, a real
  CLI/ffprobe fixture, and short-clip fake-run support.
- `ce08d73` — Milestone 2 reuse now verifies that the final DINO proposal list
  exactly matches the finalized checkpoint rows.
- `c5609ca` — Milestone 3 shard/runtime provenance, source binding,
  non-erasable fallback review flags, and correct reverse propagation at local
  frame zero.
- `c7b4005` — documentation refreshed to match Milestones 1-3.

Milestone 4 adds deterministic Stage I-only YUV rendering, scaled safety
dilation, bounded shard loading, source color-fact preservation, stripped
streams/metadata, decoded fill and outside-mask verification, atomic promotion,
safe reuse, and guarded transition to `PROCESSING_COMPLETE`. The job is now
version `0.4.0` with render code version `milestone-4`.

## P1 review problems resolved

The high-priority provenance problems found during review are fixed:

1. A canonically edited DINO final artifact can no longer be reused merely by
   recomputing its internal IDs. Its proposals must match the checkpoint rows.
2. A canonically edited SAM shard can no longer remove required fallback flags
   and appear clean.
3. SAM shard fingerprints now bind the exact SAM2 runtime, configuration,
   installed source tree, Torch version, and CUDA version. Changing the runtime
   cannot silently reuse an old shard.

The production SAM2 identity is pinned to:

- repository: `https://github.com/facebookresearch/sam2`
- revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- config: `configs/sam2.1/sam2.1_hiera_l.yaml`
- config SHA-256:
  `1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107`

Those Milestone 3 artifacts were produced by job version `0.3.1`. The current
Milestone 4 job reports version `0.4.0` and code version `milestone-4`.

## Verification completed

- Complete repository test suite: **431 passed**.
- Focused Stage II suite used during review: **120 passed** before the final P1
  regression additions.
- Ruff passes for the Stage II job and Stage II tests.
- `jobs/20_deidentify_stage2.py --version` reports `0.4.0`; the thin RunPod
  wrapper delegates to that same parser.
- Shell syntax and whitespace checks passed.
- The new fingerprint tests were mutation-tested: they failed when the runtime
  binding was removed and passed after restoration.
- The Milestone 4 Stage I-only source and render-fingerprint tests were also
  mutation-tested: each failed when its protection was removed, then passed
  after restoration.

The complete repository Ruff run still reports three unrelated, pre-existing
items outside the Stage II change:

- `jobs/10_blur_egoblur.py:2746` — `C420`
- `tests/test_blur_job.py:1716` — `F841`
- `tests/test_prompt_render.py:101` — `I001`

Do not mix those unrelated cleanup items into a Stage II milestone commit.

## Next milestone

Continue with **Milestone 5: private labels, review workflow, and operator UX**
in `STAGE2_DEIDENTIFICATION_PLAN.md`.

The renderer is an internal tested layer, not yet a public production command.
Milestone 5 must expose the documented command set, implement explicit render
invalidation/recompute, private `DO-NOT-SHIP` labels and evidence, immutable
human review records, actionable recovery output, release checks, and the
idempotent persistent RunPod setup path. Do not begin real GPU calibration until
that operator workflow and its acceptance gate pass.

## Remaining concerns, not current failing tests

These were identified during review and should be addressed in their planned
milestones:

- Before real SAM2 execution, attest the actual extracted frame-window payload,
  not just loader-supplied metadata. Verify frame count, names/order, and content
  identity so an off-by-one or wrong directory cannot reach SAM2 with plausible
  metadata. This belongs with the Stage II setup/real-adapter path in Milestones
  5-6.
- Profile the real SAM2 adapter's full-resolution GPU-to-CPU mask copies and
  Python-list conversion. Forward/reverse overlap may duplicate work. Measure
  this during the Milestone 6 GPU pilot before optimizing.
- The fallback currently holds a conservative padded DINO box over its local
  interval. Do not describe it as learned tracking or true interpolation.
- Global cross-window person identity tracking remains intentionally deferred.
  It is unnecessary for privacy coverage and would add identity-association
  failure modes.

## EgoBlur context that remains relevant

The current best-known `GX010057` Stage I settings remain:

```bash
uv run jobs/10_blur_egoblur.py --input-dir /workspace/in --output-dir /workspace/out2 \
  --run-id test-run-3 --gen 2 \
  --face-weights-gen2 /workspace/weights/ego_blur_face_gen2.jit \
  --face-threshold 0.30 --hold-frames 45 \
  --gpu-rate-usd-per-hr <actual $/hr> --skip-shutdown
```

Do not lower the EgoBlur threshold to `0.20`; that real experiment masked the
wearer's hands heavily. The Stage I fill-integrity findings were measured and
confirmed to be mild H.264 quantization, not exposed face pixels. Preserve the
`NEEDS_REVIEW` evidence rather than treating the clip as an invalid Stage II
input.

Two older EgoBlur issues remain outside current Stage II scope:

- a resumed multi-clip batch can mix completed manifests produced by different
  redaction settings;
- two-threshold hysteresis must stay disabled until high-confidence detections
  are associated before low-confidence detections.

## RunPod reminders

- Only `/workspace` persists across pod stops.
- Re-run `scripts/runpod_setup.sh` and source `/workspace/env.sh` after a restart.
- Use detached execution for long jobs.
- Keep model/checkpoint caches and the future Stage II setup under `/workspace`.
- Use full SSH over an exposed TCP port for file transfer; RunPod Basic SSH does
  not support SCP/SFTP.
- No production Stage II command, persistent model setup, private review
  workflow, or real GPU inference exists yet. The renderer exists only as an
  internal tested layer until Milestone 5 exposes it safely.

## Suggested first message in the new chat

> Work in the `egoannote-stage2` worktree on
> `feature/stage2-deidentification`. Read `AGENTS.md`, `handover.md`, and
> `STAGE2_DEIDENTIFICATION_PLAN.md` completely. Confirm the branch and clean
> status, then start Milestone 5. Do not modify EgoBlur or begin real GPU setup.
