# Experiment 1: DA3 Metric Large vs MoGe-3 ViT-L

This is a private, one-clip, depth-only RunPod pilot. It compares exactly:

- `depth-anything/DA3METRIC-LARGE` at immutable Hugging Face revision `4010e39f3634a45bc60553321fb49fb760bd594e` (Apache-2.0); and
- `Ruicheng/moge-3-vitl` at immutable Hugging Face revision `184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be` (MIT).

The job is [30_depth_compare_da3_moge3.py](../jobs/30_depth_compare_da3_moge3.py). Its model code is also pinned: DA3 to `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` and MoGe to `74fbce054ebed49800de42d0ad0e83495065719a`.

It is deliberately separate from EgoBlur, MediaPipe hands, VLM captioning, segmentation, SLAM, reconstruction, fusion, production annotations, the public release, and every upload path. There is no `rclone copy` destination, Hugging Face upload, or paid API operation in this job.

## Privacy and input gate

The input must be a short, already-redacted, public-safe continuous child. A privacy cut is a hard sequence boundary: do not send an edited multi-segment video and do not ask the job to bridge it.

The job cannot prove redaction from pixels. It therefore requires a private sidecar attestation. Make it beside the copied input on the mounted volume, replacing the placeholders only with facts you have verified:

```json
{
  "schema_version": 1,
  "privacy_status": "redacted_public_safe",
  "continuous_child": true,
  "contains_privacy_cuts": false,
  "projection": "dewarped_rectilinear",
  "input_sha256": "<sha256 printed after rclone copy>",
  "redaction_evidence": "<private reviewed redaction manifest/decision identifier>"
}
```

Use `"rectilinear"` only when that is the actual projection. If the input is known fisheye, the job stops rather than inventing lens correction; provide an approved redacted dewarped child. If the Drive file is unredacted source footage, do not run it—ask for a redacted child.

## Google Drive to RunPod (read-only transfer)

On the pod, after the standard setup, configure the existing volume-persisted rclone credential if it has not already been configured:

```bash
cd /workspace/egoannote
bash egoblur/runpod_setup.sh
source /workspace/env.sh

# One-time only if no remote is already configured. This writes credentials to
# /workspace/.rclone.conf, never to this repository.
/workspace/bin/rclone config
```

Copy exactly one supplied child into a private mounted-volume location. This is a **download only** command; do not add a reverse `rclone copy` command.

```bash
mkdir -p /workspace/private-input/experiment-1
/workspace/bin/rclone copy \
  "<DRIVE_REMOTE>:<PATH_TO_ONE_REDACTED_CONTINUOUS_CHILD>" \
  /workspace/private-input/experiment-1/ \
  --immutable --checksum --transfers 1 --checkers 4 --progress

find /workspace/private-input/experiment-1 -maxdepth 1 -type f -print
sha256sum /workspace/private-input/experiment-1/<REDACTED_CHILD>.mp4
/workspace/bin/ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,nb_frames,nb_read_frames \
  -of json /workspace/private-input/experiment-1/<REDACTED_CHILD>.mp4
```

Put that SHA-256 in the private attestation JSON. The job repeats SHA-256, `ffprobe`, a full decode/frame-count comparison, frame-rate, resolution, source PTS, and local-path capture before it resolves a model. It writes these immediately to:

```text
/workspace/private-experiments/experiment-1-da3-moge3/transfer_preflight_manifest.json
```

## Calibration / dewarp facts

Do not guess GoPro intrinsics or FOV. If an approved calibration is available for the exact redacted/dewarped pixel geometry, keep it private and provide, for example:

```json
{
  "schema_version": 1,
  "projection": "dewarped_rectilinear",
  "image_width": 1920,
  "image_height": 1080,
  "fx": 1450.2,
  "fy": 1451.0,
  "cx": 960.1,
  "cy": 540.0,
  "fov_x_deg": 67.0
}
```

All fields are facts, not defaults. DA3 uses a supplied calibrated focal for the documented `focal_px * canonical_depth / 300` conversion and receives the full matrix only when all `fx`, `fy`, `cx`, and `cy` are supplied. MoGe receives the supplied horizontal FOV. It also saves an unconditioned predicted-intrinsics pass so the calibration-conditioned result is distinguishable from the model's own prediction.

Without calibration, both models still run but all metre-valued outputs and metric previews are marked **exploratory**. There is no ground-truth depth in this pilot.

## Preflight first

Before reserving meaningful GPU time, use a dry run. It makes no model download or inference call.

```bash
cd /workspace/egoannote
source /workspace/env.sh
env PATH="/workspace/bin:$PATH" UV_CACHE_DIR=/workspace/.uv-cache \
  /workspace/bin/uv run jobs/30_depth_compare_da3_moge3.py \
  --input /workspace/private-input/experiment-1/<REDACTED_CHILD>.mp4 \
  --redacted-input-attestation /workspace/private-input/experiment-1/<REDACTED_CHILD>.attestation.json \
  --output-dir /workspace/private-experiments \
  --run-id experiment-1-da3-moge3 \
  --camera-calibration /workspace/private-input/experiment-1/<REDACTED_CHILD>.calibration.json \
  --dry-run
```

Omit `--camera-calibration` only when no approved rectilinear/dewarped calibration exists. Inspect the resulting preflight manifest before GPU inference. If it says fisheye, fails a full decode, or the attestation does not bind the copied SHA-256, stop.

## Detached GPU execution

The default processes every source frame. It uses two isolated `uv` workers because the upstream DA3 and MoGe-3 packages have incompatible NumPy constraints. Each worker is sourced at an immutable Git revision and verifies its Hugging Face checkpoint revision before inference.

```bash
cd /workspace/egoannote
source /workspace/env.sh
mkdir -p /workspace/private-experiments/logs

setsid nohup env PATH="/workspace/bin:$PATH" UV_CACHE_DIR=/workspace/.uv-cache \
  /workspace/bin/uv run jobs/30_depth_compare_da3_moge3.py \
  --input /workspace/private-input/experiment-1/<REDACTED_CHILD>.mp4 \
  --redacted-input-attestation /workspace/private-input/experiment-1/<REDACTED_CHILD>.attestation.json \
  --output-dir /workspace/private-experiments \
  --run-id experiment-1-da3-moge3 \
  --models both \
  --camera-calibration /workspace/private-input/experiment-1/<REDACTED_CHILD>.calibration.json \
  --precision fp16 \
  --metric-min-m 0.20 --metric-max-m 8.00 \
  --keyframe-count 6 \
  --export-point-clouds \
  > /workspace/private-experiments/logs/experiment-1-da3-moge3.log 2>&1 < /dev/null &

tail -f /workspace/private-experiments/logs/experiment-1-da3-moge3.log
```

For a deliberate later cost-controlled trial, use `--fps 10`; source-frame indices and exact `(PTS, time_base)` are still retained. `--max-resolution 1280` preserves aspect ratio for model input; `--resize 1280x720` is explicit and can distort geometry, so use it only deliberately. Stored maps are upsampled to the source frame size and record the model input size.

`--precision` controls MoGe (`fp16` or `fp32`). The pinned DA3 API selects its own automatic bf16-or-fp16 autocast mode, so the worker records DA3's effective runtime dtype separately and never represents `--precision fp32` as a DA3 fp32 run.

Re-run the identical command after an interruption to resume verified frame archives. A changed input, model revision, calibration, selection FPS, resize, precision, metric scale, or inference option refuses to reuse the same run ID.

## Private output contract

```text
/workspace/private-experiments/experiment-1-da3-moge3/
  transfer_preflight_manifest.json      input SHA, ffprobe, full-decode result
  experiment_manifest.json              run-level provenance and completion status
  worker-requests/                      private immutable worker requests
  workers/{da3,moge3}/worker_manifest.json
  arrays/da3/frame_XXXXXXXX.npz         float32 depth, canonical depth, confidence
  arrays/moge3/frame_XXXXXXXX.npz       float32 depth, valid mask, predicted/used intrinsics
  keyframes/moge3/                      selected point/normal maps and optional estimated PLYs
  previews/                             metric, structural-only, and 2x2 comparison MP4s
  REPORT.md                             concise performance/provenance/review report
```

All `.npz` archives use lossless ZIP compression. They carry source frame index and exact PTS/time base. DA3 confidence is the model-provided output (not asserted to be a calibrated probability). MoGe point clouds are camera-space model estimates—never a ground-truth reconstruction.

The side-by-side metric video is: redacted RGB, DA3 fixed-range metric depth, MoGe fixed-range metric depth, and DA3-confidence/MoGe-validity quality overlay. The structural-only video is clearly watermarked as per-frame normalized and not a metric comparison. Every preview is decoded after encode before the run can be marked complete.

## Review checklist

- Confirm the redacted RGB panel remains the expected already-redacted continuous child; stop if a privacy cut or privacy concern appears.
- Compare fingertips, hand-object boundaries, and contact-adjacent motion rather than only broad scene layout.
- Stress glossy paint cans, thin tools, shelves, clutter, occlusions, edge bleeding, and motion blur; note frame indices/timestamps for every failure.
- Watch the metric video for fixed-scale plausibility and the structural-only video only for shape/edge inspection. Do not turn either into a ground-truth accuracy claim.
- Watch temporal consistency across motion. The report's same-pixel depth-delta value is only a flicker proxy; real camera and object motion contribute.
- Review median/p95 latency, peak allocated/reserved GPU memory, archive/preview sizes, decode checks, private manifest failures, exact package versions, GPU name, model/checkpoint/code revisions, and calibration status.
