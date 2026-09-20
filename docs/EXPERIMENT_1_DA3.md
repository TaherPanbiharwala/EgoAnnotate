# Experiment 1: Depth Anything 3 Metric Large

This is a private, one-clip, depth-only RunPod pilot for
`depth-anything/DA3METRIC-LARGE` only. The checkpoint is pinned to immutable
Hugging Face revision `4010e39f3634a45bc60553321fb49fb760bd594e` and is
Apache-2.0. The DA3 source code is pinned to
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.

Run [30_depth_da3_metric.py](../jobs/30_depth_da3_metric.py). It has no SLAM,
reconstruction, fusion, segmentation, production annotation, VLM, Google
Drive upload, or Hugging Face upload path. Existing historical multi-model
private runs are not altered; use a fresh DA3-only run ID for a new execution.

## Privacy and input gate

The standard input is one short, already-redacted, public-safe continuous
child. A privacy cut is a hard sequence boundary. The job requires a private
hash-bound attestation because it cannot establish redaction from pixels.

```json
{
  "schema_version": 1,
  "privacy_status": "redacted_public_safe",
  "continuous_child": true,
  "contains_privacy_cuts": false,
  "projection": "dewarped_rectilinear",
  "camera_setup_fingerprint_sha256": "<matching-private-inventory-setup-fingerprint>",
  "input_sha256": "<sha256-of-the-copied-clip>",
  "redaction_evidence": "<private-reviewed-redaction-identifier>"
}
```

For the separately authorized one-off private source-footage experiment only,
use the truthful exception below with `--allow-private-unredacted-input`.
It never makes the clip or results public-safe or publishable.

```json
{
  "schema_version": 1,
  "privacy_status": "private_unredacted_user_authorized",
  "continuous_child": true,
  "contains_privacy_cuts": false,
  "projection": "rectilinear",
  "camera_setup_fingerprint_sha256": "<matching-private-inventory-setup-fingerprint>",
  "input_sha256": "<sha256-of-the-copied-clip>",
  "publication_permitted": false,
  "private_exception_reason": "Explicit user-authorized, one-off private source-footage DA3 depth experiment."
}
```

Do not label an unknown or fisheye clip as rectilinear. The job rejects known
fisheye input rather than inventing a lens correction. Obtain an approved
dewarped continuous child before making a metric claim.

## Google Drive to RunPod: download only

After a new pod starts, install the persistent volume tools and load their
environment. This setup script is under `egoblur/`, not `scripts/`.

```bash
cd /workspace/egoannote
git pull --ff-only origin experiment-1-da3-moge3
bash egoblur/runpod_setup.sh
source /workspace/env.sh
```

Copy exactly one supplied clip to the private mounted volume. This command is
read-only with respect to Drive; never reverse it to upload any input, output,
manifest, or report.

```bash
mkdir -p /workspace/private-input/experiment-1
/workspace/bin/rclone copy \
  "<DRIVE_REMOTE>:nbt-videos/GX010079.MP4" \
  /workspace/private-input/experiment-1/ \
  --immutable --checksum --transfers 1 --checkers 4 --progress

sha256sum /workspace/private-input/experiment-1/GX010079.MP4
/workspace/bin/ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,nb_frames,nb_read_frames,duration \
  -of json /workspace/private-input/experiment-1/GX010079.MP4
```

Put the printed SHA-256 in the attestation. Before any model download the job
records local path, hash, ffprobe metadata, resolution, frame rate, source
clock, and full-decode frame count in the private transfer preflight manifest.

## Simple camera segments for this pilot

You do **not** need to run COLMAP before this DA3 pilot. Select the matching
private GoPro group with one flag; do not supply `--camera-calibration` at the
same time.

| Flag | Clips / recorded setup | DA3 handling |
| --- | --- | --- |
| `--camera-segment 1` | `GX010072`, `073`, `075`–`079`, `081`, `082`, `084`, `087`: Linear, EIS off, 1920x1080, 120000/1001 fps | Uses the recorded 100.5836° diagonal FOV to derive an approximate centered focal. Metre values and the metric preview remain **exploratory**, not measurements. |
| `--camera-segment 2` | `GX010059`, `GX010061`, `GX010063`: Linear, EIS off, 1920x1080, 30000/1001 fps | Same exploratory recorded-FOV conditioning. |
| `--camera-segment 3` | `GX010057`: Wide, EIS on, 1920x1080, 30000/1001 fps | Records the group but supplies no pinhole matrix. DA3 uses its own inferred focal and labels the result exploratory. |

Use a segment only when the input preserves that clip's 1920x1080 camera
geometry—no crop, dewarp, or resize before the job. A redacted child at the
same geometry is fine. The source frame rate may have been normalized for the
curated child; choose the segment by the original recording group.

## Optional approved calibration facts

Do not invent GoPro intrinsics or FOV. When an approved calibration matches
the exact rectilinear/dewarped pixel geometry, pass it privately:

```json
{
  "schema_version": 1,
  "approval_status": "approved_cross_probe_consensus_v1",
  "projection": "dewarped_rectilinear",
  "image_width": 1920,
  "image_height": 1080,
  "fx": 1450.2,
  "fy": 1451.0,
  "cx": 960.1,
  "cy": 540.0,
  "fov_x_deg": 67.0,
  "camera_setup_fingerprint_sha256": "<matching-private-inventory-setup-fingerprint>"
}
```

If you later obtain an approved calibration from the private COLMAP
cross-probe workflow, pass it with `--camera-calibration` instead of a camera
segment. The job accepts it only when its setup fingerprint equals the input
attestation. It receives a complete K only when all `fx`, `fy`, `cx`, and
`cy` are present. It uses the supplied focal for the documented conversion
`metres = focal_px * canonical_depth / 300`. Without an approved calibration,
any metre-valued depth map and fixed-scale preview are explicitly labelled
**exploratory**, not measurements. This pilot has no ground-truth depth.

## Dry run

The dry run does no model download or inference. It verifies the attestation,
private paths, hash, ffprobe result, exact source PTS clock, full decode,
calibration mode, and planned output location.

```bash
cd /workspace/egoannote
source /workspace/env.sh

env PATH="/workspace/bin:$PATH" UV_CACHE_DIR=/workspace/.uv-cache \
  HF_HOME=/workspace/.hf-cache \
  /workspace/bin/uv run jobs/30_depth_da3_metric.py \
  --input /workspace/private-input/experiment-1/GX010079.MP4 \
  --input-attestation /workspace/private-input/experiment-1/GX010079.private-attestation.json \
  --allow-private-unredacted-input \
  --output-dir /workspace/private-experiments \
  --run-id experiment-1-da3-private-v1 \
  --camera-segment 1 \
  --fps 30 \
  --max-resolution 1280 \
  --metric-min-m 0.20 \
  --metric-max-m 8.00 \
  --keyframe-count 6 \
  --dry-run
```

For a redacted normal run, omit `--allow-private-unredacted-input` and use a
redacted attestation. Omit `--fps` to process every source frame; an explicit
lower value preserves the original source frame index, PTS, time base, and
timestamp for each selected frame.

## Detached GPU execution

Use a new run ID after a configuration or code change. DA3 resolves its pinned
checkpoint from the Hugging Face cache on first execution, then reuses that
private pod cache. The worker records the resolved model revision, source
revision, package versions, GPU name, CUDA version, input hash, calibration,
and inference settings.

```bash
cd /workspace/egoannote
source /workspace/env.sh
mkdir -p /workspace/private-experiments/logs

setsid nohup env PATH="/workspace/bin:$PATH" \
  UV_CACHE_DIR=/workspace/.uv-cache \
  HF_HOME=/workspace/.hf-cache \
  /workspace/bin/uv run jobs/30_depth_da3_metric.py \
  --input /workspace/private-input/experiment-1/GX010079.MP4 \
  --input-attestation /workspace/private-input/experiment-1/GX010079.private-attestation.json \
  --allow-private-unredacted-input \
  --output-dir /workspace/private-experiments \
  --run-id experiment-1-da3-private-v1 \
  --camera-segment 1 \
  --fps 30 \
  --max-resolution 1280 \
  --metric-min-m 0.20 \
  --metric-max-m 8.00 \
  --keyframe-count 6 \
  > /workspace/private-experiments/logs/experiment-1-da3-private-v1.log 2>&1 < /dev/null &

tail -f /workspace/private-experiments/logs/experiment-1-da3-private-v1.log
```

Re-run the identical command after an interruption to resume verified numeric
arrays. A changed input, code revision, selection FPS, resize, calibration, or
metric range refuses to mix outputs under the same run ID. A partial run never
receives a complete status.

## Private output contract

```text
/workspace/private-experiments/experiment-1-da3-private-v1/
  transfer_preflight_manifest.json       copied input metadata and full-decode check
  experiment_manifest.json               run provenance and terminal status
  worker-requests/da3.json               immutable private worker request
  workers/da3/worker_manifest.json       DA3 model/environment/frame evidence
  arrays/da3/frame_XXXXXXXX.npz          lossless float32 depth/confidence/intrinsics
  previews/da3_metric_depth.mp4          fixed metric-scale DA3 preview
  previews/da3_structural_only.mp4       per-frame-normalised, non-metric preview
  REPORT.md                              private review and performance report
```

Each `.npz` is lossless and includes source frame index, PTS, and time base.
`confidence` is the upstream DA3 map only when emitted. If unavailable, it is
an all-`NaN` float32 sentinel accompanied by `confidence_available=0`; it is
never replaced by a synthetic proxy. DA3 point clouds are not produced by this
job.

## Review checklist

- Check that the source is the expected continuous private/redacted child.
- In the fixed-scale preview, assess whether depth scale is plausible without
  claiming accuracy.
- In the structural-only preview, inspect hands, fingertips, hand-object
  separation, contact-adjacent motion, glossy paint cans, thin tools, shelves,
  clutter, occlusions, edge bleeding, and motion blur.
- Watch for temporal flicker; same-pixel depth deltas in the report are only a
  proxy because camera and object motion also contribute.
- Record failure frame indices/timestamps, per-frame latency, peak allocated
  and reserved GPU memory, output size, decode checks, calibration status, and
  the absence of ground truth before drawing conclusions.
