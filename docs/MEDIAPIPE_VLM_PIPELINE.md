# MediaPipe + dense VLM caption pipeline

## Scope

This pipeline starts from `master`, after EgoBlur. Stage 2 segmentation is
paused and is not invoked, merged, or used as a dependency.

MediaPipe and the VLM always read the **redacted video**. The original is
registered as a private asset and archived for the later WiLoR, SAM2, and
DepthV3 runs. Those future stages must read the original because redaction
removes image evidence that depth, masks, and hand reconstruction may need.

## Caption representation

Prompt v4 produces three synchronized tracks from each six-second window:

1. `activity.caption`: one holistic sentence describing the combined activity
   in that window. Across ordered windows, these captions form the activity
   narrative “as it goes.”
2. `actions[]`: a variable-length list of atomic actions with sampled-frame
   start/end boundaries, a complete action caption, verb/noun-style task label,
   what/how/why, tool, coordination, and handover state.
3. `actions[].left_hand.caption` and `actions[].right_hand.caption`: one sentence
   for each visible anatomical hand, alongside structured verb, object, target,
   contact, and visibility fields.

This follows the useful common structure across egocentric datasets: Ego4D
stores dense narrations plus summaries; EPIC-KITCHENS stores narration with
start/stop time, verb, and noun; action-scene-graph work adds explicit
hand-object relations. The tracks remain separate rather than overloading one
caption, which makes them directly queryable and joinable to later perception
outputs.

Primary references:

- [Ego4D annotation schema](https://ego4d-data.org/docs/data/annotations-schemas/)
- [EPIC-KITCHENS-100 annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations)
- [LaViLa dense video narration](https://arxiv.org/abs/2212.04501)
- [Action Scene Graphs for egocentric video](https://openaccess.thecvf.com/content/CVPR2024/html/Rodin_Action_Scene_Graphs_for_Long-Form_Understanding_of_Egocentric_Videos_CVPR_2024_paper.html)

## Privacy and storage boundary

```text
private/
  annotations.db             resumable VLM state; never publish
  caption_frames/            extracted from redacted video; never publish
  models/                    MediaPipe model cache
  run_manifest.json          local and Drive paths; never publish
  redaction_reviews/         hash-bound named human approvals; never publish
  drive_receipts/            verified archive receipts

work/<video_id>/
  hands.parquet              restartable stage output

publish/
  annotations/<video_id>/hands.parquet
  annotations/<video_id>/caption_windows.parquet
  annotations/<video_id>/caption_actions.parquet
  manifests/<video_id>.json  contains hashes, no local/original path
```

Only `publish/` and the redacted video are uploaded to Hugging Face. Originals,
EgoBlur checkpoints, original-derived preview images, and private manifests are
never included. Later WiLoR/SAM2/DepthV3 numeric outputs can be added to
`publish/` only after a verification pass confirms that no original RGB crops or
debug renderings are embedded.

## Pre-redaction pose-prior shadow pilot

Before a new EgoBlur run, a private MediaPipe Pose pass can label likely
**wearer** hand/arm/leg overlap. It never defines a face-search ROI, changes an
EgoBlur threshold, changes a fill box, or approves a redaction. The role of the
pilot is to measure which detector/YuNet candidates are likely the camera
wearer's limbs, since preserving those hands matters for later hand tracking;
blur on other people is not down-ranked merely because their pose was found.

Run it only against the original and keep both model and artifact private:

```bash
uv run egoannote-run pose-prior \
  --original-video /private/GX010057.MP4 \
  --video-id GX010057 \
  --output private/pose-prior/GX010057.pose_prior.json \
  --models-dir private/models \
  --preview-video private/pose-preview/GX010057.pose.mp4
```

The first call downloads the versioned MediaPipe **Pose Landmarker Full** task
model. The artifact records the original hash, resolution, source timing, model
hash, every sampled frame (including frames with no pose), and limb-only 2D
landmarks/masks. Face and torso landmarks are not retained. The heuristic
labels a pose only as a `wearer_candidate` when reliable limb landmarks are
camera-near; this is a review hint, not an identity claim.

`--preview-video` is an original-only private diagnostic: limb masks for a
camera-near wearer candidate are amber; other detected poses are blue. The
latest 10 Hz pose is held between samples so tracking errors are easy to see.
It is for improving the pilot only and must never be uploaded or published.

Pass that artifact to the pod alongside a private report destination:

```bash
... jobs/10_blur_egoblur.py ... \
  --checkpoint-dir /workspace/private/checkpoints/GX010057-shadow \
  --pose-prior /workspace/private/pose-prior/GX010057.pose_prior.json \
  --pose-shadow-report /workspace/private/pose-shadow/GX010057.pose_shadow.json
```

The GPU job rejects a stale, incomplete, geometry-mismatched, or non-private
artifact before it spends GPU time. In this default shadow pilot, the output
video and normal EgoBlur manifest must match the same command run without pose
input; only the private overlap report is new.

### Experimental active wearer-hand suppression

Pose remains a private shadow diagnostic; it never changes redaction. The
behavior-changing prior is a separate MediaPipe **Hand Landmarker** pass over
the original. It defaults to **30 fps** to preserve fast hand motion. Its
artifact is complete at that cadence (including no-hand frames), holds all 21
2D landmarks and short-continuity anchors, and stays private.

Use two requested hands for the active artifact. A four-hand prior/preview is
valuable for diagnosis, but its different model configuration is rejected for
active suppression. Preview colours are evidence labels only: amber is a
temporally stable likely wearer hand, magenta/pink is a provisional likely
wearer hand, and blue is another/unknown hand. MediaPipe Left/Right, entry
zone, screen side, and `H#` are private review hints, not identity proof.

EgoBlur still detects faces full-frame at its calibrated 10 Hz. Before it
acts, it validates the full 30-fps artifact and selects only the exact 10-Hz
detector frames; therefore the hand prior never becomes an ROI or changes
detector input, thresholds, association, or treatment of other people.

- Amber (`--hand-suppress-wearer-hands`) can withhold a raw face track only
  when one stable hand track explains at least three raw face detections and a
  two-thirds majority of that track, at >=90% box overlap.
- Pink (`--pink-suppress-wearer-hands`) applies the same three-hit and
  two-thirds rule to one provisional likely-wearer hand, at >=92% overlap.
  It is deliberately only slightly tighter than amber, so it can suppress
  convincing fast-hand false positives rather than merely report them.
- Blue changes nothing. Missing hand evidence remains unknown, so the normal
  face-redaction decision is retained.

Any active amber or pink action writes a private per-clip report and forces
`NEEDS_REVIEW`. `--pink-demote-generated-fills` remains an optional separate
experiment for capping interpolation/hold frames; it never removes a raw
detection. For a batch, `--hand-suppression-report` must be a private
directory so no clip loses its evidence.

```bash
uv run egoannote-run hand-prior \
  --original-video /workspace/in/GX010057.MP4 \
  --video-id GX010057 \
  --num-hands 2 \
  --output /workspace/private/hand-prior/GX010057.hand_prior.json \
  --models-dir /workspace/private/models \
  --preview-video /workspace/private/hand-preview/GX010057.hands.mp4

... jobs/10_blur_egoblur.py ... \
  --hand-prior /workspace/private/hand-prior/GX010057.hand_prior.json \
  --hand-suppression-report /workspace/private/hand-suppression/GX010057.json \
  --hand-suppress-wearer-hands \
  --pink-suppress-wearer-hands
```

## Independent YuNet verification

YuNet is an optional post-redaction verifier. It is intentionally much more
relaxed than EgoBlur for wearer-hand false positives: when given the private
30-fps hand prior, it expands amber and pink hand circles by 1.5x and removes
an uncovered hit from the *actionable review queue* at >=10% overlap. The raw
YuNet result/count stays in the private report; non-hand residuals stay
actionable. This filtering cannot change an already-redacted video.

YuNet runs **after** redaction. It is a CPU-only, independent detector over the
redacted video: any face it finds outside EgoBlur's reconstructed fill map is
reported for review. It never opens the original video and does not alter the
redacted artifact.

Download the matching redacted video, EgoBlur manifest, and private
`checkpoints/` directory. Obtain a YuNet ONNX model separately, then run:

```bash
uv run egoannote-run verify-yunet \
  --redacted-video /private/GX010057.blurred.mp4 \
  --blur-manifest /private/GX010057.manifest.json \
  --checkpoint-dir /private/checkpoints \
  --yunet-model /private/face_detection_yunet.onnx \
  --job-script jobs/10_blur_egoblur.py \
  --ffmpeg ffmpeg \
  --report /private/GX010057.yunet_review.json \
  --preview-video /private/GX010057.yunet_full_debug.mp4 \
  --hand-prior private/hand-prior/GX010057.hand_prior.json \
  --pose-prior private/pose-prior/GX010057.pose_prior.json \
  --candidate-contact-sheet-dir private/GX010057.yunet_candidates
```

The command verifies the video hash, checkpoint fingerprint and sampled-frame
coverage before it starts. `PASS_NO_UNCOVERED_YUNET` validates this one
independent check only; it does **not** override other EgoBlur audit reasons,
approve publication, or make the private review report uploadable.

`--preview-video` is optional full diagnostics: it creates a 10-FPS, real-time
review copy from the **redacted video only**, without audio. Green boxes are
YuNet faces already covered by EgoBlur and red boxes are uncovered hits.

The candidate contact-sheet directory is the normal review aid. It groups all
uncovered YuNet hits into short temporal candidates. Normal candidates are red;
only candidates that repeatedly overlap a likely **wearer** limb are amber and
lower priority. They are still present, counted, and keep the run in
`NEEDS_REVIEW`. An overlap with another person's hand/leg remains normal
priority. The report, preview, contact sheets, and pose input must all be in a
`private` or `DO-NOT-SHIP` directory; none may enter `publish/` or Hugging
Face.

Create a review template, change each value to `confirmed_face`,
`false_positive`, or `uncertain`, then produce only confirmed remediation boxes:

```bash
uv run egoannote-run init-yunet-decisions \
  --report private/GX010057.yunet_review.json \
  --output private/GX010057.yunet_decisions.json

uv run egoannote-run decisions-to-forced-boxes \
  --report private/GX010057.yunet_review.json \
  --decisions private/GX010057.yunet_decisions.json \
  --output private/GX010057.forced_boxes.json
```

The decision file is bound to the exact report hash and must contain a decision
for every candidate. The generated JSON is directly compatible with EgoBlur's
`--forced-boxes`; rerun with `--force-reprocess`, then inspect the corrected
video and rerun YuNet. Uncertain candidates are never turned into a claim that
the video is ready to publish.

## Record the final human redaction decision

An active amber/pink suppression run deliberately remains `NEEDS_REVIEW` even
after the owner has watched and accepted the final redacted video. Do **not**
edit that EgoBlur manifest. Instead, retain the final redacted video, its
EgoBlur manifest, its private hand-suppression report, and its private YuNet
report locally, then record the named decision once:

```bash
uv run egoannote-run approve-redaction \
  --run-dir runs/gx010057-final \
  --video-id GX010057 \
  --redacted-video /private-input/GX010057.blurred.mp4 \
  --blur-manifest /private-input/GX010057.manifest.json \
  --hand-suppression-report /private-input/GX010057.hand_suppression.json \
  --yunet-report /private-input/GX010057.yunet_review.json \
  --reviewer "Dataset owner"
```

This writes `runs/gx010057-final/private/redaction_reviews/GX010057.json`.
It stores the reviewer and SHA-256/size binding of all four inputs. Annotation
and upload re-hash those files every time, so changing, removing, or replacing
any input invalidates the approval and requires a fresh human decision. The
approval file and the two review reports are private evidence only; none are
included in `publish/` or uploaded to Hugging Face.

## Runtime

The locked local runtime is Python 3.12 with MediaPipe 0.10.35. MediaPipe needs
normal macOS graphics-service access even though inference uses the CPU; it may
fail inside a restricted app sandbox but runs from a normal terminal.

Set the real model entries in `models.toml` and export the API key. For a
measurement run, the two models should come from different labs and each
OpenRouter provider should be pinned. The CLI defaults to serial VLM calls and a
$10 cap **per model**, so the maximum overshoot is one completed call.

## One-video pilot

Use five windows spread across the clip. The current 271-second clip has 46
windows, so the proposed sample is `0,11,23,34,45`.

```bash
uv run egoannote-run annotate \
  --run-dir runs/mediapipe-vlm-pilot \
  --redacted-video GX010057.blurred.mp4 \
  --blur-manifest GX010057.manifest.json \
  --redaction-review runs/gx010057-final/private/redaction_reviews/GX010057.json \
  --video-id GX010057 \
  --model MODEL_A \
  --model MODEL_B \
  --pilot-windows 0,11,23,34,45 \
  --workers 1
```

Review all 10 model/window results for:

- valid JSON/schema rate;
- correct wearer-left versus wearer-right assignment;
- whether each visible hand caption names the observed action and object;
- atomic-action splitting and sampled-frame boundaries;
- holistic activity caption specificity without long-horizon guessing;
- blur-related loss or hallucination;
- realized provider staying pinned;
- cost per window and projected full-batch cost.

Do not run the full batch until this gate passes. If one model is clearly worse,
replace it and repeat the same windows so the comparison is controlled.

## Resume into the full video

Run the same command without `--pilot-windows`. SQLite skips the completed pilot
windows and captions only the remaining 41. MediaPipe also skips its existing
Parquet output.

```bash
uv run egoannote-run annotate \
  --run-dir runs/mediapipe-vlm-pilot \
  --redacted-video GX010057.blurred.mp4 \
  --blur-manifest GX010057.manifest.json \
  --redaction-review runs/gx010057-final/private/redaction_reviews/GX010057.json \
  --video-id GX010057 \
  --model MODEL_A \
  --model MODEL_B \
  --workers 1 \
  --prune-caption-frames
```

`--prune-caption-frames` is allowed only on a full run. It removes the
reproducible JPEG cache after Parquet export, not the database or annotations.

## Batch manifest

For multiple clips, create JSONL with one object per line:

```json
{"video_id":"GX010057","redacted_video":"/data/GX010057.blurred.mp4","blur_manifest":"/data/GX010057.manifest.json","redaction_review":"private/redaction_reviews/GX010057.json"}
```

Then replace the single-video inputs with:

```bash
uv run egoannote-run annotate \
  --run-dir runs/batch-001 \
  --batch-manifest batch.jsonl \
  --model MODEL_A \
  --model MODEL_B \
  --workers 1
```

When an EgoBlur manifest is supplied, anything other than a `PASS...` status
blocks annotation unless the same command supplies a valid private,
hash-bound `redaction_review`. This is for recorded owner review of an exact
`NEEDS_REVIEW` derivative, not an automatic override. Supplying a manifest
also derives a collision-resistant ID from its clip ID and original SHA-256.

## Hugging Face upload

The uploader creates or reuses a dataset repository, uploads the small bundle
resumably, then streams each redacted video without making a second local copy.
It defaults to a private dataset.

```bash
uv run egoannote-run publish-hf \
  --run-dir runs/batch-001 \
  --repo-id OWNER/DATASET
```

Publishing is blocked when the run has no passing EgoBlur manifest or a valid
human redaction approval. The explicit `--allow-unverified-redaction` override
exists only for already-reviewed legacy clips with unavailable evidence; the
normal path is to preserve the manifest and private review record.

The private bundle is marked `license: other` and includes private prerelease
terms. Public release requires terms approved by the rights holder; pass them
explicitly so the bundled `LICENSE` and card metadata are updated together:

```bash
uv run egoannote-run publish-hf \
  --run-dir runs/batch-001 \
  --repo-id OWNER/DATASET \
  --public \
  --license-file /secure/path/EGOANNOTE-DATA-LICENSE.txt \
  --license-name "Egoannote Data Use Terms v1.0"
```

## Verified Drive archive and local-space release

Configure an rclone Google Drive remote first. Archiving uses copy, then compares
remote size and MD5, writes a receipt, and only then honors an exact-file deletion
flag. It never uses `rclone move`.

```bash
uv run egoannote-run archive-drive \
  --run-dir runs/batch-001 \
  --drive-root gdrive:nbt-videos \
  --delete-local-originals \
  --delete-local-redacted
```

Original deletion needs a verified Drive receipt. Redacted deletion additionally
requires a successful Hugging Face upload receipt. Annotation Parquet and private
state remain local because they are small; they can also be backed up separately.

## Later original-video stages

The private manifest already records `future_original_stages = [wilor, sam2,
depth_v3]`. When those jobs are added, they should:

1. restore the original from `private/originals/<video_id>/` on Drive;
2. write immutable arrays under their own stage directory;
3. join by `video_id` and timestamp/frame index;
4. keep original RGB crops and overlays in a `DO-NOT-SHIP` tree;
5. add verified numeric/structured outputs to the Hugging Face bundle;
6. archive, checksum, receipt, then release local space again.
