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

Pose remains a useful private diagnostic, but it does **not** make the active
decision: its camera-near amber label occasionally changed between the wearer
and a nearby person, and the conservative Pose-only gate consequently withheld
zero tracks. The active gate instead uses a separate MediaPipe **Hand
Landmarker** prior. YuNet is paused and is not part of this protocol.

Build the private hand prior from the original at exactly EgoBlur's 10 Hz
cadence. Active suppression rejects any other cadence, so do not override
either command's `--detect-hz` value. Its preview makes the evidence reviewable: amber is a stable wearer
candidate, magenta is a provisional candidate that cannot suppress anything,
and blue is another hand.

For the first human review, request four hands. This enables the preview to
show a potential third/fourth hand rather than silently dropping it at the
detector limit. This four-hand artifact is **diagnostic only** and must not be
passed to `--hand-suppress-wearer-hands`: active suppression remains pinned to
the reviewed two-hand configuration. If active suppression is approved after
review, make a separate, fresh two-hand artifact.

```bash
uv run egoannote-run hand-prior \
  --original-video /workspace/in/GX010057.MP4 \
  --video-id GX010057 \
  --num-hands 4 \
  --output /workspace/private/hand-prior/GX010057.review4.hand_prior.json \
  --models-dir /workspace/private/models \
  --preview-video /workspace/private/hand-preview/GX010057.review4.mp4
```

The first call downloads the versioned official Hand Landmarker model to the
private model cache. The artifact stores source/model hashes, every 10 Hz
sample (including no-hand frames), 21 two-dimensional hand landmarks, its
short temporal hand-track identifier, and the wearer-candidate/stable state.
It is original-derived and must never be published.

Each preview label additionally exposes review-only identity evidence:

- `H23` is a **short-continuity anchor**, not a person ID. A hand that is
  absent longer than the small tracking grace period gets a new anchor; the
  system records its identity as unknown rather than pretending it has
  re-identified the hand.
- `MP: Left 0.94` or `MP: Right 0.94` is MediaPipe's anatomical-handedness
  hint and its confidence. It is displayed, but never used to merge anchors,
  decide wearer identity, or change blur.
- `entry: left | now: right` is the image-side/edge evidence for that short
  anchor. It helps a human inspect a suspicious re-entry, but a change of side
  is not proof of a different person in a moving egocentric camera.
- `3+ hands: review only` means three or more hands were detected in that
  sample. It is strong evidence to inspect an extra hand, not an automatic
  outsider or redaction decision.

Only a stable candidate can affect the fill map. The active gate additionally
requires all of the following:

- a close, lower-frame or side-edge hand candidate observed for at least five
  temporally continuous 10 Hz samples (a single missed sample may bridge the
  same hand track);
- at least four raw EgoBlur face detections in the proposed face track; and
- at least 98% overlap between **every** raw face box and one same stable hand
  track's circle. A track with even one uncertain/non-hand hit is retained and
  blurred.

EgoBlur still runs full-frame; the hand prior cannot change detector inputs,
thresholds, tracking association, or the treatment of other people. It only
filters an already-built, exceptionally well-supported false-positive track.
A nonzero suppression count always yields `NEEDS_REVIEW`; it is never a
publication approval.

Magenta evidence has a separate, private reporting role. If every raw
EgoBlur face box in a track overlaps the same provisional wearer-hand track,
and that face track then gains generated interpolation or hold fills, the
private hand-suppression report records it as a
`pink_amplification_candidate`. This lets review target brief hand detections
that may have produced a visibly long false-positive fill. It is strictly
shadow-only: magenta never suppresses a track or changes detection, fill-map,
encoded pixels, or run status.

Use fresh output/checkpoint/report destinations, preserving the ordinary run
for byte/hash comparison:

```bash
... jobs/10_blur_egoblur.py ... \
  --output-dir /workspace/out-gx010057-active-hand \
  --checkpoint-dir /workspace/private/checkpoints/GX010057-active-hand \
  --pose-prior /workspace/private/pose-prior/GX010057.pose_prior.json \
  --pose-shadow-report /workspace/private/pose-shadow/GX010057.hand_pose_shadow.json \
  --hand-prior /workspace/private/hand-prior/GX010057.hand_prior.json \
  --hand-suppression-report /workspace/private/hand-suppression/GX010057.json \
  --hand-suppress-wearer-hands
```

For a batch, pass `--hand-suppression-report` a private **directory**, not a
single `.json` file. The job writes one review report per clip and rejects a
single report path that would overwrite earlier evidence.

## Independent YuNet verification

YuNet is an optional separate verifier, but is not part of the current
Pose-plus-EgoBlur pilot.

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
{"video_id":"GX010057","redacted_video":"/data/GX010057.blurred.mp4","original_video":"/data/GX010057.MP4","blur_manifest":"/data/GX010057.manifest.json"}
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
blocks annotation. Supplying a manifest also derives a collision-resistant ID
from its clip ID and original SHA-256.

## Hugging Face upload

The uploader creates or reuses a dataset repository, uploads the small bundle
resumably, then streams each redacted video without making a second local copy.
It defaults to a private dataset.

```bash
uv run egoannote-run publish-hf \
  --run-dir runs/batch-001 \
  --repo-id OWNER/DATASET
```

Publishing is blocked when the run has no passing EgoBlur manifest. The explicit
`--allow-unverified-redaction` override exists for already-reviewed legacy clips,
but the safer batch path is to preserve and supply every EgoBlur manifest.

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
