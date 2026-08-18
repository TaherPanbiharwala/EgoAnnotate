# Stage II EgoBlur + DINO/SAM2 De-identification Plan

Status: ready for later implementation

Saved: 2026-08-18

## Worktree and execution boundary

- Create `/Users/taherpanbiharwala/Desktop/Annotated_Data/egoannote-stage2` as a sibling Git worktree.
- Use branch `feature/stage2-deidentification` based on clean `master` commit `545396050b5fcbe8fa8efdd2d2241d95e6176b4d`.
- Leave `/Users/taherpanbiharwala/Desktop/Annotated_Data/egoannote-batch` and `batch/16-clip-run` untouched.
- Keep all implementation, tests, configuration, documentation, and commits in this one worktree.
- Run EgoBlur and Stage II sequentially on the same RunPod GPU pod and persistent `/workspace` network volume, using separate locked PEP 723 environments.

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
- fill-integrity check ran and reported zero violations.

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

## Rollout

1. Create the dedicated worktree.
2. Validate the current `GX010057` Stage I integrity report.
3. Implement setup, pure contracts, and fake-model tests.
4. Implement DINO proposals and checkpointing.
5. Implement SAM windows, fallback masks, and mask shards.
6. Implement rendering and media verification.
7. Implement labels, evidence, review records, and manual-seed remediation.
8. Run a 30-60 second real GPU smoke slice.
9. Label the whole `GX010057` clip and render the four threshold candidates.
10. Accept and freeze a production configuration.
11. Canary one additional clip.
12. Process the remaining clips under the accepted damage/resource stop ceilings and full-clip review contract.

## Deferred

- automatic removal of EgoBlur false positives;
- Stage I mask export and single-render Approach C;
- global cross-window identity tracking;
- alternative detector research;
- web dashboard or hosted service;
- automatic publication without human acceptance.
