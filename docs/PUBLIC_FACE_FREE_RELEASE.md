# Public face-free release

This document is only for the owner-approved, 13-clip face-free release. It
does not change the private-by-default rules for other original-derived runs,
redaction diagnostics, or future work.

## What is public

For each approved video, the release contains:

- the normalized clean MP4;
- the final MP4 with MediaPipe hands and Qwen captions burned in;
- per-frame hands in `hands.json` and `hands.parquet`;
- dense caption/event data in `captions.json`; and
- a public curation manifest with hashes, timing, and annotation continuity.

The public curation manifest deliberately excludes source paths and private
review evidence. It exposes output-frame segment boundaries instead. A new
segment always means MediaPipe tracks and caption windows restarted after a
privacy-curation edit. The current 13 clips all have one segment and no
discontinuities: their source/output frame-count differences come from 29.97
fps normalization, not face-frame removal.

## Build the release folder

The artifact directories are ordered deliberately: the 0.55 household run is
selected where it exists, then the 0.4 fast-motion run supplies `GX010059` and
`GX010063`.

```bash
uv run egoannote-run prepare-public-release \
  --children-dir runs/face-free-c40-2026-09-01/private/face_free_children \
  --annotation-run-dir runs/face-free-c55-household-2026-09-02 \
  --annotation-run-dir runs/face-free-c40-2026-09-01 \
  --annotated-video-dir runs/face-free-c55-household-2026-09-02/private/annotated-videos \
  --annotated-video-dir runs/face-free-c40-2026-09-01/private/annotated-videos \
  --output-dir public-release/egoannote-v1 \
  --model qwen3.8-max \
  --approved-by "Taher Panbiharwala" \
  --release-version v1.0 \
  --video-id GX010059 \
  --video-id GX010063 \
  --video-id GX010072 \
  --video-id GX010073 \
  --video-id GX010075 \
  --video-id GX010076 \
  --video-id GX010077 \
  --video-id GX010078 \
  --video-id GX010079 \
  --video-id GX010081 \
  --video-id GX010082 \
  --video-id GX010084 \
  --video-id GX010087
```

The command refuses an incomplete, hash-mismatched, or non-face-free input.
It full-probes every clean and overlay video, requires one complete hand row
per annotated frame, and copies into a new `public-release/` folder. It never
modifies the private inputs.

## Verify and upload

First inspect the generated files and retain the private annotations and the
verified Drive archive. Create the public Hub repository, then upload only the
release directory:

```bash
hf repos create TaherPanbiharwala/EgoAnnotate --repo-type dataset --public --exist-ok
hf upload TaherPanbiharwala/EgoAnnotate public-release/egoannote-v1 . \
  --repo-type dataset --commit-message "Release EgoAnnotate v1.0"
```

After upload, compare the Hub files with `release-manifest.json` and validate
the generated dataset card, clean videos, overlay videos, JSON, and Parquet
downloads. Only then delete local MP4 copies; preserve public manifests,
annotation files, and the verified Drive backup.
