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
