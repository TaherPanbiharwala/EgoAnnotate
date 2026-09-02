# Private original-curated annotation workflow

This branch (`original-trim-mediapipe-vlm`) is intentionally separate from the
redaction workflow. It contains no new EgoBlur or YuNet step and does not make
a Hugging Face bundle. Its only path is:

```text
original video -> selected frame cuts -> private MediaPipe + private VLM records
```

The original-derived child, timeline, extracted frames, hand Parquet files,
SQLite captions, and run summary must remain under `private/` or
`DO-NOT-SHIP`. Do not upload them with the normal publishing command.

## 1. Make a zero-based inclusive cut list

```text
# remove these source frames
3150-3278
4500-4620
```

If using extracted image filenames, extract them with `-start_number 0` so
the numeric name is also the zero-based source-frame index.

## 2. Create the private curated child

```bash
uv run egoannote-run curate-original \
  --original-video /data/GX010057.MP4 \
  --video-id GX010057 \
  --cut-list private/cuts/GX010057.txt \
  --output-video private/curated/GX010057.curated.mp4 \
  --manifest private/curated/GX010057.timeline.json
```

The command removes audio and metadata, verifies the decode, and records the
source-to-output timeline privately.

## 3. Annotate isolated retained segments

```bash
uv run egoannote-run annotate-curated-original \
  --run-dir runs/gx010057-original-curated \
  --curated-video private/curated/GX010057.curated.mp4 \
  --timeline-manifest private/curated/GX010057.timeline.json \
  --video-id GX010057 \
  --model MODEL_ID \
  --workers 1
```

Each segment receives its own internal annotation ID (`GX010057.segment.0000`,
then `...0001`, and so on). The private run summary maps that ID back to the
curated child's output-frame range. MediaPipe runs at its normal 30-fps policy;
VLM windows begin fresh in each segment and can never span a cut.
