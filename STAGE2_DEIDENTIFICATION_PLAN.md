# Stage II EgoBlur + DINO/SAM2 De-identification Plan

Status: Milestone 1 complete; Milestone 2 next

Saved: 2026-08-18; milestone structure updated 2026-08-20

| Milestone | Status |
|---|---|
| 1. Contracts, validation, and local skeleton | Complete — 2026-08-20 |
| 2. DINO proposal generation and reuse | Not started |
| 3. SAM2 propagation and mask shards | Not started |
| 4. Rendering and technical verification | Not started |
| 5. Labels, review workflow, and operator UX | Not started |
| 6. Real-GPU smoke test and `GX010057` calibration | Not started |
| 7. Canary and production rollout | Not started |

## Worktree and execution boundary

- Work in `/Users/taherpanbiharwala/Desktop/Annotated_Data/egoannote-stage2`, the dedicated sibling Git worktree.
- Use branch `feature/stage2-deidentification`; current `master` through `9acfe07` has been merged into it.
- Leave `/Users/taherpanbiharwala/Desktop/Annotated_Data/egoannote-batch` and `batch/16-clip-run` untouched.
- Keep all implementation, tests, configuration, documentation, and commits in this one worktree.
- Run EgoBlur and Stage II sequentially on the same RunPod GPU pod and persistent `/workspace` network volume, using separate locked PEP 723 environments.

## Milestone delivery rules

- Build milestones in order. A later milestone may start only after the previous milestone's acceptance gate passes.
- Keep each milestone in its own small commit series so it can be reviewed and reverted independently.
- Run the existing repository test suite at every milestone boundary. Also run that milestone's focused tests.
- Do not require a GPU until Milestone 6. Milestones 1-5 must be testable locally with deterministic fake model adapters.
- A milestone is complete only when its code, tests, manifest fields, operator documentation, and failure behavior are reviewed together.
- Privacy or provenance failures block advancement. Convenience features may be deferred without weakening fail-closed behavior.

## Milestone 1 local operator commands

Milestone 1 exposes only the contract-testing commands below. They load no real DINO/SAM models and cannot produce a publishable video:

```bash
# Validate the original, EgoBlur output, and EgoBlur manifest without writing a run.
bash scripts/runpod_stage2.sh validate \
  --source-video /path/to/GX010057.MP4 \
  --stage1-video /path/to/GX010057.blurred.mp4 \
  --stage1-manifest /path/to/GX010057.manifest.json

# Exercise deterministic local artifacts, state transitions, and resume behavior.
bash scripts/runpod_stage2.sh fake-run \
  --source-video /path/to/GX010057.MP4 \
  --stage1-video /path/to/GX010057.blurred.mp4 \
  --stage1-manifest /path/to/GX010057.manifest.json \
  --work-dir /path/to/stage2-work \
  --run-id milestone-1-smoke

# Read state without modifying the run.
bash scripts/runpod_stage2.sh status \
  --work-dir /path/to/stage2-work \
  --run-id milestone-1-smoke \
  --clip-id GX010057
```

Add `--json` before the subcommand for machine-readable output. A fake run records `NOT_RUN_FAKE` and `NOT_REVIEWABLE_FAKE`; it may never be accepted or released.

## Architecture

Use Approach B first:

1. EgoBlur produces the Stage I video.
2. Grounding DINO runs on the original video to find additional faces.
3. SAM2.1 converts DINO boxes into masks and propagates them through nearby frames.
4. Stage II applies the additional masks to the Stage I video.
5. Stage II can add privacy coverage but cannot undo an EgoBlur false positive.

Defer Approach C, where Stage I exports masks and both stages are rendered once from the original. Reconsider it only if the pilot rejects double-encoding quality or exact Stage I mask attribution becomes necessary.

## Stage I input validation

Do not modify EgoBlur's known completed-manifest config-matching behavior in this implementation. Stage II compensates by validating every selected Stage I artifact before GPU work:

- original source hash and Stage I output hash;
- dimensions, constant frame rate, frame count, and rotation/display facts;
- technically complete Stage I manifest and output;
- Gen2 face detector;
- face threshold `0.30`;
- hold frames `45`;
- hysteresis disabled;
- plate detector disabled;
- expected dilation/motion-margin settings;
- fill-integrity check ran across the complete clip and reported its findings. Preserve nonzero legacy exact-pixel findings as Stage II review reasons rather than rejecting `GX010057`: the measured evidence in `AGENTS.md`/`handover.md` established that its findings are mild H.264 quantization, not exposed pixels. Stage II's own renderer will use its new verification contract before promotion.

Any mismatch fails closed. Carry `NEEDS_REVIEW` and its reasons into Stage II rather than treating it as failure, because Stage II exists to investigate and recover misses. Record that historical Stage I model-weight identity is weaker when the upstream manifest did not store its hash.

## Stage II implementation

- Add a self-contained GPU PEP 723 job using pinned Grounding DINO Base and SAM2.1 Hiera Large revisions.
- Persist model assets, Hugging Face/Torch/uv caches, checkpoints, masks, and logs under `/workspace`.
- Setup may download public weights; processing runs offline and never uploads video or frames.
- Run DINO with:
  - prompt `face.`;
  - proposal floor `0.10`;
  - text threshold `0.25`;
  - anchor spacing `20` frames;
  - full-frame inference plus overlapping 2x2 tiled inference;
  - union of full-frame and tiled proposals.
- Run DINO and SAM sequentially so both large models are not resident on the GPU together.
- Feed accepted proposals to SAM2 in bounded overlapping windows. Select a safe window size during the pre-pilot GPU memory check, record it, and hold it fixed throughout threshold comparison.
- Local object IDs are diagnostic only. Correctness comes from combining all masks; global cross-window person identity is deferred.
- Preserve the fail-closed mask rule:

  `final Stage II mask = padded accepted DINO boxes OR valid SAM2 masks OR manual-seed masks`

- SAM failure may not remove a valid DINO redaction. Empty, undersized, exploding, or otherwise suspicious SAM masks retain their DINO fallback and create a review flag.
- Store immutable compressed mask shards per temporal window rather than a clip-wide dense mask map.
- Use layered fingerprints:
  - DINO layer depends on original/model/prompt/preprocessing/tiling/anchor settings;
  - SAM layer depends on DINO output/model/window/threshold/manual-seed settings;
  - render layer depends on masks/Stage I video/dilation/fill/encoder settings.
- Manual-seed changes invalidate intersecting SAM windows and render. Dilation or encoder changes invalidate render only.
- Render using Stage I frames as the only pixel source. Use constant YUV fill with an initial `8px` safety dilation at 1080p, scaled with frame height.
- Strip audio, subtitles, data streams, and metadata. Verify frame/media/fill integrity before atomic output promotion.

## Operator interface

- Add an idempotent Stage II setup script that persists `UV_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, model assets, and checkpoints under `/workspace`, verifies hashes, primes the locked environment, and runs an offline CUDA/model-load smoke test.
- Add a thin runner with `doctor`, `smoke`, `pilot`, `sweep`, `run`, `status`, `resume`, `stop`, `review`, and `release-check`.
- The normal golden path after checkout is:

  ```bash
  bash scripts/runpod_setup_stage2.sh
  bash scripts/runpod_stage2.sh pilot GX010057
  ```

- Compatible resume is the default.
- Use `--recompute-from dino|sam|render` to explicitly invalidate a layer and all downstream work.
- Provide `--dry-run`, `--json`, and `--version`.
- Every error reports an error code, what failed, the cause and actual values, the exact recovery command, log/artifact locations, and which layers remain reusable.
- Keep machine `processing_state`, automated `audit_status`, and human `review_status` separate.

## Whole-clip GX010057 calibration

Create a private, versioned label file covering every face event across the full `GX010057` clip. Record:

- clip, frame, or frame range;
- conservative face box or mask;
- visibility and category;
- human Stage I verdict: `covered`, `partial`, or `missed`;
- DINO proposal verdict: real face or false positive;
- final-mask coverage;
- reviewer disposition.

Include frontal, profile, occluded, frame-edge truncated, small/distant, brief, motion-blurred, and relevant reflection/screen faces. Include negative examples such as hands, skin-colored objects, product labels, and background patterns. Store labels and all pre-acceptance evidence under a private `DO-NOT-SHIP` directory.

## DINO threshold sweep

Collect DINO proposals once at the `0.10` floor using the fixed full-frame-plus-tiled union. Hold EgoBlur output, text threshold, DINO/SAM revisions, SAM settings, window layout, dilation, and encoder settings constant.

Render four complete `GX010057` candidates whose only changed result-affecting parameter is the DINO box threshold:

- `0.15`
- `0.20`
- `0.25`
- `0.30`

Changing the operating threshold at or above `0.10` reuses DINO proposals but reruns affected SAM work and rendering. Going below `0.10` invalidates DINO and all downstream layers.

Compare Stage I and all four candidates using:

- face-event and labeled-frame recall;
- incremental recall over the human Stage I verdict;
- temporal holes;
- full-frame-only, tiled-only, and shared DINO proposals;
- true/false DINO proposals and rejected SAM masks;
- added mask area and new tracks per minute;
- hand/tool overlap and reviewer rejection rate;
- runtime, peak VRAM, mask storage, and cost per source-video hour;
- outside-mask quality loss from the second encode.

## Threshold selection and acceptance

Use a balanced privacy/utility rule:

1. Eliminate a candidate with an unaccounted labeled face event, a visible labeled frame below 95% conservative-face coverage, or a temporal hole.
2. Among passing candidates, choose the one with the least unnecessary hand/tool masking and reviewer rejection.
3. Break a practical tie in favor of the higher DINO threshold.
4. If no automatic candidate passes, choose the highest-recall/lowest-damage candidate, add manual seeds for every residual miss, rerun, and keep it unaccepted until the corrected result passes.
5. Freeze the accepted threshold, model revisions, window layout, dilation, and numeric damage/resource ceilings in the production configuration.

`processing complete` means inference, rendering, and technical verification succeeded. It does not mean the clip is safe to publish.

Before publication, watch every complete final clip at accelerated speed and inspect every flagged interval at normal speed or frame-by-frame. Record a separate immutable review verdict linked to the exact processing-manifest and output hashes. A correction invalidates the previous acceptance and requires rerendering and full-clip-plus-flags review again.

## Tests

- Unit-test input validation, configuration precedence, safe paths, full/tiled coordinate mapping, proposal union, threshold separation, anchor/window boundaries, forward/reverse propagation, mask union/dilation, DINO fallback, manual seeds, compressed shards, layered fingerprints, state transitions, and stale-review invalidation.
- Use deterministic fake DINO/SAM adapters for empty, exploding, crossing, occluded, boundary, interrupted, multiple-face, and no-face cases.
- Add real ffmpeg round-trip tests proving Stage I is the only render source, filled luma/chroma survives encoding, frame synchronization is exact, media/color facts remain valid, and audio/metadata are removed.
- Add RunPod GPU smoke tests for offline DINO inference, SAM anchor masking, forward/reverse propagation, and peak VRAM reporting.
- Mutation-test every privacy, fingerprint, resume, and fail-closed regression.
- Run the existing full repository suite before and after implementation.

## Implementation milestones

### Milestone 1 — Contracts, validation, and local skeleton

**Build**

- Add the self-contained Stage II PEP 723 job skeleton and thin runner command structure without loading real models.
- Define versioned schemas for processing state, manifests, DINO artifacts, SAM mask shards, render artifacts, manual seeds, labels, and immutable review records.
- Implement Stage I artifact validation, safe path handling, atomic writes, hashes, layered fingerprints, and state transitions.
- Add deterministic fake DINO/SAM adapters and fixture videos so all later orchestration can be tested locally.

**Review checkpoint**

- Review public commands, schemas, fingerprint boundaries, safe storage paths, and fail-closed errors before model-specific logic is added.

**Acceptance gate**

- Invalid or incompatible Stage I input is rejected before GPU work; interrupted writes cannot appear complete; fake processing can resume deterministically; focused tests and the full repository suite pass.

### Milestone 2 — DINO proposal generation and reuse

**Build**

- Implement full-frame plus overlapping 2x2 tiled DINO inference, coordinate conversion, proposal union/NMS, anchor scheduling, and immutable proposal artifacts.
- Store all proposals down to the fixed `0.10` floor so thresholds `0.15`-`0.30` can reuse the same DINO computation.
- Record model revision/hash, exact prompt, preprocessing, tiling, scores, accepted/rejected proposal counts, runtime, and peak VRAM fields.

**Review checkpoint**

- Inspect tiled coordinate mapping, border faces, duplicate proposals, threshold separation, checkpoint reuse, and DINO artifact provenance using fake adapters and hand-checkable fixtures.

**Acceptance gate**

- Identical inputs reuse byte-identifiable proposal artifacts; any DINO-affecting change invalidates them; operating-threshold changes at or above `0.10` do not rerun DINO; all mapping and boundary tests pass.

### Milestone 3 — SAM2 propagation, fallback safety, and mask shards

**Build**

- Implement accepted DINO boxes as SAM2 prompts in bounded overlapping windows with forward and reverse propagation.
- Validate masks for empty, undersized, off-prompt, interrupted, or near-full-frame expansion behavior.
- Preserve padded DINO boxes whenever SAM fails, union valid SAM/manual masks, and persist immutable compressed per-window mask shards.
- Implement targeted invalidation for threshold and manual-seed changes; keep object IDs local to a window.

**Review checkpoint**

- Review privacy behavior for occlusion, brief appearances, window boundaries, crossing faces, failed propagation, pathological masks, and manual-seed correction.

**Acceptance gate**

- A valid DINO fallback can never disappear because of SAM behavior; suspicious masks create review flags; overlapping shards reconstruct the expected masks without a clip-wide dense map; focused and full tests pass.

### Milestone 4 — Rendering and technical verification

**Build**

- Stream Stage I frames as the only pixel source and apply the union of Stage II masks using constant YUV fill and scaled safety dilation.
- Strip audio, subtitles, data streams, and metadata; encode to a temporary artifact; verify it; then promote it atomically.
- Implement frame-count/timing, media/color, fill-integrity, complete-shard, manifest-hash, and stale-output checks.
- Record added mask area, hand/tool overlap hooks, runtime, storage, encoder configuration, and outside-mask quality measurements.

**Review checkpoint**

- Compare decoded input/output frames, confirm that unmasked pixels came from Stage I, inspect mask edges after encoding, and test interruption/restart behavior.

**Acceptance gate**

- Frame synchronization is exact; required masks survive encoding; forbidden streams/metadata are absent; incomplete or unverifiable output is never promoted or marked `processing complete`; all ffmpeg and repository tests pass.

### Milestone 5 — Private labels, review workflow, and operator UX

**Build**

- Implement the private versioned face-event label format, negative-example labels, manual seeds, evidence extracts, review flags, and `DO-NOT-SHIP` enforcement.
- Implement `doctor`, `smoke`, `pilot`, `sweep`, `run`, `status`, `resume`, `stop`, `review`, and `release-check`, including dry-run/JSON output and actionable recovery errors.
- Keep `processing_state`, automated `audit_status`, and immutable human `review_status` separate; bind acceptance to exact output and manifest hashes.
- Add the idempotent Stage II RunPod setup path with persistent `/workspace` caches and verified model/checkpoint assets.

**Review checkpoint**

- Walk through setup, a failed run, resume, manual correction, rerender, review invalidation, and release check using fake models and synthetic evidence.

**Acceptance gate**

- Private artifacts cannot enter a release package; any correction invalidates earlier acceptance; errors identify reusable layers and exact recovery commands; the documented golden path works up to the real-GPU boundary.

### Milestone 6 — Real-GPU smoke test and full `GX010057` calibration

**Build/run**

- Run setup and offline model-load checks on RunPod, then choose and freeze a safe SAM window size using a 30-60 second slice.
- Validate the current `GX010057` Stage I artifacts and create the private full-clip answer sheet covering every face event and representative negatives.
- Compute DINO proposals once, render complete candidates at `0.15`, `0.20`, `0.25`, and `0.30`, and collect all privacy, utility, quality, runtime, VRAM, disk, and cost metrics.
- Apply the threshold-selection rule; if none passes, add manual seeds for every residual miss and rerun.

**Review checkpoint**

- Review the smoke slice before paying for the full sweep. Then compare Stage I and all four complete candidates against the same labels and inspect every failure/flag.

**Acceptance gate**

- The selected candidate has no unaccounted face event, no visible labeled frame below 95% conservative coverage, and no temporal hole; a human completes full-clip review; selected settings and measured ceilings are frozen in a versioned production configuration.

### Milestone 7 — Canary and controlled production rollout

**Build/run**

- Process one additional representative clip using the frozen configuration and enforce the calibrated damage/resource stop ceilings.
- Compare canary metrics with `GX010057`, inspect all flags, and watch the complete final clip before acceptance.
- Only after canary acceptance, process remaining clips sequentially with per-clip immutable manifests, review records, and release checks.

**Review checkpoint**

- Review canary drift in recall evidence, false positives, mask area, hand/tool overlap, runtime, VRAM, storage, and cost before authorizing the remaining batch.

**Acceptance gate**

- Canary stays within frozen ceilings and is manually accepted; every production clip independently reaches `processing complete` and then `review accepted`; release packaging excludes every `DO-NOT-SHIP` artifact.

## Deferred

- automatic removal of EgoBlur false positives;
- Stage I mask export and single-render Approach C;
- global cross-window identity tracking;
- alternative detector research;
- web dashboard or hosted service;
- automatic publication without human acceptance.
