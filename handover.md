# egoannote Stage II session handover

Updated 2026-08-23 after the real GPU smoke test passed on a RunPod pod and
the real (non-fake) pilot pipeline was wired.

## Start here

- Worktree: `/Users/taherpanbiharwala/Desktop/Annotated_Data/egoannote-stage2`
- Branch: `feature/stage2-deidentification` (pushed to `origin`; **not**
  merged into `master` — keep it that way unless explicitly told otherwise)
- Latest pushed HEAD: `eb8b45c` — pilot CLI wiring (`parse_args`/`main()`)
- Real pipeline wiring commits: `3308e50` (real DINO frame loader), `7b4accc`
  (`run_real_pipeline`), `c6d7f4b` (real-GPU readiness gate + calibration
  summary), `eb8b45c` (pilot CLI wiring) — see
  `/Users/taherpanbiharwala/.claude/plans/encapsulated-hugging-peacock.md`
  for the plan these came from, if it's still present locally.
- EgoBlur sync commit: `2b3c5b9` pulled in `--min-track-confirmations`
  support from `master`; `f005062` pinned it to `2` in Stage II's
  `EXPECTED_STAGE1`.
- Milestone 6 local preparation checkpoint: `39d23ed`
- Milestone 5 implementation HEAD: `dba85a9`
- Baseline HEAD before Milestone 5: `f0754f1`
- Base merged from `master`: `9acfe07`
- Milestones through 5 are implemented, tested, and documented on this branch.
- Milestone 6: the offline GPU smoke test has **passed for real** on a RunPod
  L4 pod (see "Real GPU smoke test result" below). The real DINO/SAM2
  orchestration and a runnable `pilot` command are now implemented, tested
  (with fake adapters standing in for actual inference — the sequencing,
  state transitions, and readiness gate are all real, GPU-free-testable
  code), and pushed. **What has not run yet: `pilot` against a real trimmed
  slice on the pod.** That is the next and only remaining step before the
  full threshold sweep.

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
- `f0754f1` — Milestone 4 verified Stage I-only rendering and atomic output
  promotion.
- `dba85a9` — Milestone 5 private artifacts, immutable review/release gates,
  operator commands, persistent setup, tests, and documentation.

Milestone 5 adds content-addressed private labels, manual seeds, evidence, and
review flags beneath `DO-NOT-SHIP`; immutable human reviews bound to exact
processing/output hashes; correction-driven review invalidation; fail-closed
release scans; cooperative stop, compatible resume, and explicit layered
recompute; the complete operator command surface; and an idempotent persistent
RunPod setup script with verified assets. The job is now version `0.5.0` with
code version `milestone-5`. The setup and all commands are documented in
`docs/stage2-operator.md`.

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
Milestone 5 job reports version `0.5.0` and code version `milestone-5`.

## Verification completed

- Complete repository test suite: **446 passed**.
- Focused Stage II job/render/operator suite: **65 passed**.
- Ruff passes for the Stage II job and Stage II tests.
- `jobs/20_deidentify_stage2.py --version` reports `0.5.0`; the thin RunPod
  wrapper delegates to that same parser.
- Shell syntax and whitespace checks passed.
- The new fingerprint tests were mutation-tested: they failed when the runtime
  binding was removed and passed after restoration.
- The Milestone 4 Stage I-only source and render-fingerprint tests were also
  mutation-tested: each failed when its protection was removed, then passed
  after restoration.
- Milestone 5's review-ID content binding, strict review-record schema,
  post-review private-correction invalidation, recompute symlink refusal, and
  setup-root canonicalization were mutation-tested the same way. Every targeted
  test failed with its guard removed and passed after restoration.
- Setup was exercised only in non-mutating dry-run mode. No weights were
  downloaded and no CUDA/model execution occurred locally.

The complete repository Ruff run still reports three unrelated, pre-existing
items outside the Stage II change:

- `jobs/10_blur_egoblur.py:2746` — `C420`
- `tests/test_blur_job.py:1716` — `F841`
- `tests/test_prompt_render.py:101` — `I001`

Do not mix those unrelated cleanup items into a Stage II milestone commit.

## Milestone 6 progress and next external gate

Local code (all pushed to `origin/feature/stage2-deidentification`):

- makes persistent setup run DINO and SAM2 sequentially with networking
  disabled and saves `models/stage2/gpu-smoke.json` atomically;
- chooses BF16 on compute capability 8+ and FP16 on older CUDA devices;
- records device/model/runtime identity, runtime, peak VRAM, and residual DINO
  memory before SAM loads;
- loads DINO from the exact persistent snapshot whose weights were hash
  verified;
- atomically extracts numeric SAM JPEG windows, verifies exact frame count,
  names, dimensions, hashes, source/range binding, and revalidates the payload
  before inference and reuse; the same extraction path (`extract_attested_sam_window`)
  now also backs a real single-frame DINO loader (`real_dino_frame_loader`) —
  a DINO anchor is just a size-1 SAM window, so there is one attested
  extraction path, not two;
- wires the real (non-fake) DINO → SAM2 → render → finalize sequence end to
  end (`run_real_pipeline`), with the same sequential load-then-free GPU
  discipline (`del`+`gc.collect()`+`cuda.empty_cache()`+`cuda.synchronize()`
  between DINO and SAM2) `run_offline_gpu_smoke` already used; and
- gates `pilot` behind `_require_real_gpu_execution_ready`, which re-derives
  the doctor/asset check fresh and compares it against the persisted smoke
  evidence — not just "does gpu-smoke.json exist," but "does it still match
  what's actually installed right now" (`GPU_SMOKE_EVIDENCE_STALE` if not).

### Real GPU smoke test result (RunPod L4 pod, passed for real)

`bash scripts/runpod_setup_stage2.sh --workspace-root /workspace` completed
with `status: PASS_OFFLINE_GPU_SMOKE`. Both DINO and SAM2 loaded and ran
offline (networking disabled) on CUDA, sequentially, with all pinned assets
hash-verified:

- device: L4, compute capability `[8, 9]` → BF16 precision chosen, as
  designed;
- DINO peak VRAM: ~2 GB; SAM2 peak VRAM: ~1.9 GB;
- DINO residual VRAM before SAM2 loads: ~8 MB — confirms the sequential
  free actually released the model, not just deleted a Python reference.

Two real pod-side issues were hit and fixed along the way, in case they
recur on a fresh pod/volume:

- `SAM_RUNTIME_UNVERIFIABLE` on a symlink inside the installed SAM2 package
  (`sam2/sam2_hiera_t.yaml` → `configs/sam2/sam2_hiera_t.yaml`, a real,
  benign, in-tree symlink shipped by upstream SAM2 revision `2b90b9f`).
  `sha256_directory_tree` originally rejected *any* symlink; fixed (commit
  `80c3359`) to allow one only if its resolved target stays inside the
  hashed root.
- `/workspace/.uv-cache` silently consumed ~12 GB (invisible to
  `du -sh /workspace/*`, which doesn't glob dotfiles) and tripped the
  network-volume disk-usage warning. Safe to `rm -rf /workspace/.uv-cache/*`
  — it's a cache, not data. Note also: switching git branches is a cheap
  metadata operation and does **not** itself consume additional disk space;
  don't delete `master` locally to "make room" for a branch switch.

### Next external gate: run `pilot` on a real trimmed slice

This is now pure operator work, not more Python. On the pod:

1. Trim a 30-60 second slice of `GX010057`.
2. Re-run `jobs/10_blur_egoblur.py` on that trim (same settings as the
   "EgoBlur context" section below, with a distinct `--run-id`/`clip_id`,
   e.g. `GX010057-pilot`) to get a real, self-consistent Stage I manifest.
   Stage II never fabricates a Stage I manifest for content it didn't
   itself process — `validate_stage1`'s fail-closed checks need genuinely
   consistent audit/integrity fields.
3. Run `pilot` against that trim's source video, Stage I video, and Stage I
   manifest, with an explicit `--window-size`/`--window-overlap` (no
   default — forcing a value each time is the point of the calibration
   loop) and `--run-id`/`--workspace-root`. It reports
   `{manifest, calibration_summary, window_size, window_overlap}` on
   success, or fails closed with one of `STAGE2_SETUP_INCOMPLETE`,
   `GPU_SMOKE_EVIDENCE_MISSING`, `GPU_SMOKE_NOT_PASSED`, or
   `GPU_SMOKE_EVIDENCE_STALE` from the readiness gate — distinct from
   `REAL_GPU_EXECUTION_DEFERRED`, which `sweep` and non-fake `run` still
   unconditionally raise.
4. Use `calibration_summary`'s per-window runtime/VRAM numbers to pick a
   safe SAM window size, then move on to the full threshold sweep (`0.15`,
   `0.20`, `0.25`, `0.30`) — that tooling does not exist yet and is
   deliberately out of scope until there's real pilot output to design it
   against.

## Remaining concerns, not current failing tests

These were identified during review and should be addressed in their planned
milestones:

- ~~Before real SAM2 execution in Milestone 6, attest the actual extracted
  frame-window payload, not just loader-supplied metadata.~~ Done —
  `extract_attested_sam_window` verifies frame count, names/order, and
  content identity for both the real SAM window loader and the real DINO
  frame loader.
- Profile the real SAM2 adapter's full-resolution GPU-to-CPU mask copies and
  Python-list conversion. Forward/reverse overlap may duplicate work. Measure
  this during the real pod pilot run before optimizing.
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
  --face-threshold 0.30 --hold-frames 45 --min-track-confirmations 2 \
  --gpu-rate-usd-per-hr <actual $/hr> --skip-shutdown
```

Do not lower the EgoBlur threshold to `0.20`; that real experiment masked the
wearer's hands heavily. The Stage I fill-integrity findings were measured and
confirmed to be mild H.264 quantization, not exposed face pixels. Preserve the
`NEEDS_REVIEW` evidence rather than treating the clip as an invalid Stage II
input.

`--min-track-confirmations 2` (EgoBlur commit `39055e2` on `master`, after
this worktree's base) suppresses hold/fill for tracks with fewer than 2
confident detections — single-hit tracks were 335 of 392 face tracks and
77.7% of all redacted area on this clip, almost entirely noise. This
worktree's `EXPECTED_STAGE1` now pins it to `2`; a Stage I manifest produced
without this flag (or with a different value) fails Stage II input
validation closed.

Two older EgoBlur issues remain outside current Stage II scope:

- a resumed multi-clip batch can mix completed manifests produced by different
  redaction settings;
- two-threshold hysteresis must stay disabled until high-confidence detections
  are associated before low-confidence detections.

## RunPod reminders

- Only `/workspace` persists across pod stops.
- Run `scripts/runpod_setup_stage2.sh` for Stage II and source the generated
  `/workspace/stage2-env.sh` after a restart.
- Use detached execution for long jobs.
- Keep model/checkpoint caches and Stage II setup under `/workspace`.
- Use full SSH over an exposed TCP port for file transfer; RunPod Basic SSH does
  not support SCP/SFTP.
- The persistent setup's offline GPU smoke test has passed for real on a
  RunPod L4 pod (see "Real GPU smoke test result" above). `pilot` is now
  wired to the real DINO/SAM2 pipeline and gated by
  `_require_real_gpu_execution_ready`, but has not yet been run against a
  real clip slice on the pod — that is the next step. `sweep` and non-fake
  `run` still deliberately return `REAL_GPU_EXECUTION_DEFERRED`.
- After a fresh pod boot: `cd /workspace/egoannote && git pull origin
  feature/stage2-deidentification`, then `bash scripts/runpod_setup_stage2.sh
  --workspace-root /workspace` (idempotent — re-verifies assets and re-checks
  the smoke test even if it already ran once), then `source
  /workspace/stage2-env.sh`.

## Suggested first message in the new chat

> Work in the `egoannote-stage2` worktree on
> `feature/stage2-deidentification`. Read `AGENTS.md`, `handover.md`, and
> `STAGE2_DEIDENTIFICATION_PLAN.md` completely. Confirm the branch and clean
> status. The real GPU smoke test has already passed on the pod, and `pilot`
> is fully wired and pushed — the next step is trimming a 30-60 second
> `GX010057` slice, producing its own Stage I manifest via EgoBlur, and
> running `pilot` against it to pick a safe SAM window size. Do not modify
> EgoBlur, merge to `master`, or begin the full threshold sweep before that
> pilot run has real results.
