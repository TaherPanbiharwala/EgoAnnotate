# EgoAnnotate

**An egocentric-video annotation pipeline using MediaPipe hand tracking and vision-language-model (VLM) activity captions.**

EgoAnnotate turns first-person video into structured annotation artifacts that are useful for computer-vision research and analysis. The active workflow curates selected video frames into retained segments, tracks hands at full frame rate with MediaPipe, and generates dense, time-bounded activity descriptions with a VLM.

[Explore the public dataset](https://huggingface.co/datasets/TaherPanbiharwala/EgoAnnotate)

## What I built

- A Python CLI workflow for curating first-person video and preserving an auditable source-to-output timeline.
- A 30 fps MediaPipe hand-landmark layer that records 21 landmarks per detected hand in columnar Parquet files.
- A VLM captioning layer that turns sampled video windows into validated, structured activity and hand-state records.
- A persistence and validation layer for resumable annotation runs, bounded VLM spend, artifact hashes, and rendered review outputs.
- A public v1 dataset that makes the resulting videos, annotations, and documentation inspectable.

## Project at a glance

| Area | Evidence |
| --- | --- |
| Public release | 13 manually reviewed face-free first-person video clips on [Hugging Face](https://huggingface.co/datasets/TaherPanbiharwala/EgoAnnotate) |
| Annotation artifacts | Clean and annotated MP4s, per-frame hand data in JSON and Parquet, caption/event JSON, and curation manifests |
| Hand tracking | MediaPipe Hand Landmarker at 30 fps, with continuity-based assignment when handedness labels are unreliable |
| Captioning | Dense VLM activity captions, atomic actions, and structured left/right hand state for each sampled window |
| Engineering focus | Reproducible video processing, validation, provenance, resumability, and human-reviewable outputs |

## How it works

```text
First-person video
        |
        v
Manual frame curation and a cut list
        |
        v
Retained video segments + timeline manifest
        |
        +-----------------------------+
        |                             |
        v                             v
MediaPipe hand landmarks          VLM activity captions
(30 fps, per frame)               (8 frames / 6-second window)
        |                             |
        +-------------+---------------+
                      v
Parquet, SQLite, JSON, and optional annotated-video outputs
```

The active original-derived workflow materializes each retained segment independently. Hand tracking and caption windows restart at every intentional edit, so no annotation can silently span a removed portion of the source video.

## Engineering details

### Hand tracking that does not trust a single label

MediaPipe can label both visible hands as the same side in egocentric footage. EgoAnnotate assigns detections jointly: frame-to-frame wrist continuity is the primary signal, and MediaPipe's handedness label is used only to break an initial tie. This prevents a repeated label error from silently dropping a visible hand or swapping the two slots. Each stored frame includes landmark confidence and a track ID so interruptions are visible downstream.

### Structured VLM annotations instead of free-form prose

For each six-second window, the captioning layer samples eight frames and asks the selected VLM for a holistic activity description, temporally bounded atomic actions, and separate left/right hand state. Responses are parsed through a repair-and-validation path and checked against a strict schema. Invalid frame references, malformed output, partial-window mistakes, and unusable fields are recorded rather than treated as successful annotations.

### Reproducible video and data workflow

- One `ffprobe` result supplies the clock used by both hand and caption tracks.
- Frame-exact `ffmpeg` cuts create retained segments and a hash-bound timeline manifest.
- Dense hand landmarks stream directly to Parquet; VLM window records are stored in SQLite and can resume after interruption.
- The model registry supports provider pinning and per-model spend caps, while stored prompt hashes and realized-provider metadata make an annotation run traceable.

## Quick start

### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` on your `PATH` (`brew install ffmpeg` on macOS)

```bash
git clone https://github.com/TaherPanbiharwala/EgoAnnotate.git
cd EgoAnnotate
uv sync
uv run pipeline/demo.py
```

The demo requires no account, GPU, or VLM API key. It runs the real media probe, frame extraction, MediaPipe hand-tracking, and persistence code against a bundled synthetic clip. A deterministic fake backend stands in for the VLM, so the command produces local hand and caption artifacts without a network model call.

The first run may download the MediaPipe task model. The demo is a plumbing smoke test: it proves the local workflow runs end to end, but it does not measure caption quality on real footage.

## Run the pipeline on your own footage

The active workflow is intended for privately curated original footage. Start by creating a zero-based, inclusive cut list for source-frame ranges you want to remove:

```text
3150-3278
4500-4620
```

Create a curated child and its timeline manifest:

```bash
uv run egoannote-run curate-original \
  --original-video /path/to/input.mp4 \
  --video-id my-video \
  --cut-list private/cuts/my-video.txt \
  --output-video private/curated/my-video.mp4 \
  --manifest private/curated/my-video.timeline.json
```

To add VLM captions, configure a model entry in [`pipeline/models.toml`](pipeline/models.toml), export the matching provider credential in your shell, then run:

```bash
export OPENROUTER_API_KEY="<your-key>"

uv run egoannote-run annotate-curated-original \
  --run-dir runs/my-video \
  --curated-video private/curated/my-video.mp4 \
  --timeline-manifest private/curated/my-video.timeline.json \
  --video-id my-video \
  --model YOUR_MODEL_ID \
  --workers 1
```

See the [private original-curated workflow guide](docs/PRIVATE_ORIGINAL_CURATED_PIPELINE.md) for the full command contract, output layout, and preview-rendering commands. Run `uv run egoannote-run --help` to inspect every supported command.

## Outputs

| Output | Purpose |
| --- | --- |
| Curated MP4 + timeline manifest | Records the retained source ranges and output-frame mapping |
| Hand Parquet | Stores per-frame landmarks, confidence, track IDs, and no-hand gaps; the public release also includes JSON exports |
| SQLite caption store + event JSON | Stores window-level VLM responses and source-bound segment summaries |
| Annotated MP4 previews | Burns stored hand landmarks and captions into a reviewable video |
| Run manifests and hashes | Bind inputs, prompts, models, settings, and generated artifacts to a run |

## Repository layout

| Path | Purpose |
| --- | --- |
| [`pipeline/`](pipeline/README.md) | Active installable package, prompts, model registry, and `egoannote-run` CLI |
| [`docs/`](docs/) | Workflow, release, and technical documentation |
| [`tests/`](tests/) | Pytest regression suite for media, tracking, captioning, validation, and packaging behavior |
| [`egoblur/`](egoblur/README.md) | Reference-only GPU face-redaction workflow with its own isolated environment |
| [`public-release-tools/`](public-release-tools/README.md) | Reference-only tooling used to package and publish the public release |

## Status and limitations

- EgoAnnotate v1 is public: [browse the 13-clip dataset](https://huggingface.co/datasets/TaherPanbiharwala/EgoAnnotate).
- The active pipeline supports curation, MediaPipe hands, VLM captioning, storage, previews, and archiving. Subtask segmentation is intentionally not implemented until it can be calibrated against measured annotation data.
- The local demo uses a fake VLM backend. Real captioning requires a configured model and provider credential, and output quality should be evaluated on the footage and task of interest.
- The public release includes only allowlisted public artifacts. Original-derived intermediate data and review evidence remain outside that release.

## Verification

```bash
uv run --extra test pytest tests/ -q
uv run --with ruff ruff check pipeline/src pipeline/demo.py egoblur public-release-tools
```

## License

The code is available under the [MIT License](LICENSE). The approved EgoAnnotate v1 dataset release uses CC BY 4.0.
