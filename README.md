# egoannote

**Dense subtask annotation for egocentric video — measured, not asserted.**

Turns raw first-person video into verb/noun subtask segments with hand
pose, object masks, relative depth and camera pose — and reports real
numbers for how good each field actually is, instead of just shipping a
demo. Runs on one laptop plus a small amount of rented GPU time.

## Repository layout

The active pipeline is deliberately small: `curate-original` ffmpeg-cuts out
frames where a face is visible from a manually-authored cut list, then
`annotate-curated-original` runs MediaPipe hands and dense VLM captioning on
what's retained. Everything else is parked but kept working, isolated by
runtime boundary, as public references:

- [`pipeline/`](pipeline/README.md) is the installable `egoannote` package —
  the active pipeline above, plus its shared media/backend/storage
  infrastructure. Its command is `egoannote-run`.
- [`egoblur/`](egoblur/README.md) is the automated GPU face-redaction job
  (an alternative to manual curation) and its pre/post-redaction review
  tooling. It has an isolated PEP 723 environment and is not a dependency of
  `pipeline/`.
- [`public-release-tools/`](public-release-tools/README.md) is the historical
  mechanism for turning an EgoBlur-redacted video into a published Hugging
  Face dataset (MediaPipe + captioning on redacted video, bundle packaging,
  upload). Not staged as an active public pipeline right now, but kept
  working since this repo is public.

> **Status: early build.** The active pipeline (frame curation, hand
> tracking, dense VLM captioning, storage, verified Drive archiving) is
> implemented and tested. Stage-2 segmentation is deliberately unimplemented
> — see `docs/` for current status. This README will grow a results table
> and a hero clip once real footage has been annotated.

## Setup

Prerequisites:

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — dependency management and the `uv run` entry point used throughout this README
- `ffmpeg` / `ffprobe` on `PATH` (`brew install ffmpeg` / `apt install ffmpeg`) — used for probing and frame extraction, not bundled as a Python dependency

```bash
git clone git@github.com:TaherPanbiharwala/EgoAnnotate.git egoannote
cd egoannote
uv sync
```

`uv sync` creates a `.venv` and installs everything in `pyproject.toml`
(MediaPipe, OpenCV, PyArrow, Pydantic, httpx). There is no separate GPU
install step — the layers implemented so far (hand tracking, VLM captioning)
run on CPU. The isolated GPU privacy workflow lives under `egoblur/` and is
not part of this environment.

## Try it — no accounts, no GPU, no API key

```bash
uv run pipeline/demo.py
```

Runs the real pipeline (probe → frame extraction → MediaPipe hand tracking →
VLM captioning) against a bundled synthetic clip, using a deterministic fake
backend in place of a real API call. It's a plumbing smoke test, not proof
of caption quality — see the script's own output for what it does and
doesn't demonstrate.

## Usage — annotate your own footage

The active workflow — a manual cut list, `curate-original`, then
`annotate-curated-original` for MediaPipe hands and dense VLM captioning on
the retained segments — is documented in
[`docs/PRIVATE_ORIGINAL_CURATED_PIPELINE.md`](docs/PRIVATE_ORIGINAL_CURATED_PIPELINE.md).
The Python layer APIs below remain available for custom experiments.

The historical EgoBlur-redacted-video annotate/publish workflow (automated
redaction instead of a manual cut list) is parked but documented as a
reference in [`docs/MEDIAPIPE_VLM_PIPELINE.md`](docs/MEDIAPIPE_VLM_PIPELINE.md)
— its commands now live in `egoblur/cli.py` and `public-release-tools/cli.py`,
not `egoannote-run`.

### Release the approved face-free batch

The approved public release is intentionally built from a fresh, allowlisted
folder rather than uploading a run directory. It contains the clean videos,
final hand-and-caption overlay videos, per-frame hand annotations in JSON and
Parquet, caption/event JSON, and public curation manifests. See
[`docs/PUBLIC_FACE_FREE_RELEASE.md`](docs/PUBLIC_FACE_FREE_RELEASE.md) for the
exact build, verification, and Hugging Face upload commands (now run via
`public-release-tools/cli.py`).

### Hand tracking (local CPU, free — no API key needed)

```python
from pathlib import Path
from egoannote import config
from egoannote.media.probe import probe
from egoannote.layers import hands as hands_layer
from egoannote.store import write_hands_parquet_streaming

video = Path("my_clip.mp4")
video_id = config.video_id_from_content(video, video.stem)  # disambiguates repeated GoPro filenames
info = probe(video)  # single source of truth for fps/duration — both tracks must share it

n_frames, gaps = write_hands_parquet_streaming(
    hands_layer.run(video, video_id, config.MODELS_DIR, info=info),
    Path("runs") / video_id / "hands.parquet",
)
print(f"{n_frames} frames written, {len(gaps)} no-hands gaps")
```

The MediaPipe hand-landmarker model (~7.8 MB) is downloaded once to
`config.MODELS_DIR` and cached on disk after that.

### VLM captioning (needs an API key and a model in `pipeline/models.toml`)

1. Pick or add a model entry in `pipeline/models.toml` under `[models.<id>]` — fill
   in real prices from your provider before setting a spend cap (a `0.0`
   price disables the cap silently, by design of the validation, so it's
   loud if you forget: `build_backend` will refuse a spend cap against a
   zero-priced model).
2. Export your key: `export OPENROUTER_API_KEY=sk-...` (never put it in
   `pipeline/models.toml` — that file is committed).
3. Run:

```python
from egoannote.backends.registry import build_backend
from egoannote.layers import caption as caption_layer
from egoannote.store import Store

backend = build_backend("example-a")  # the [models.example-a] id from pipeline/models.toml
store = Store(Path("runs") / video_id / "annotations.db")

n = caption_layer.caption_video(video, video_id, backend, store)
print(f"wrote {n} caption window(s)")
```

Calls run concurrently (`config.CAPTION_MAX_WORKERS`, default 8) — pass
`max_workers=1` to `caption_video` for strictly serial, deterministic
execution. Progress is resumable: re-running `caption_video` against the
same `store` skips windows already recorded and only retries error rows.

### Config knobs

Everything tunable lives in `pipeline/src/egoannote/config.py`, each with a comment
explaining where the number came from. Two are environment overrides rather
than constants, for a checkout that isn't the repo root or a non-editable
install:

- `EGOANNOTE_DATA_DIR` — where runtime data (models cache, frame cache, SQLite,
  reports) lives. Defaults to `./runs`.
- `EGOANNOTE_MODELS_TOML` — path to `pipeline/models.toml`. Defaults to that
  repository copy; a wheel install doesn't bundle it, so this is required there.

### Tests and lint

```bash
uv run pytest
uv run --with ruff ruff check pipeline/src pipeline/demo.py egoblur public-release-tools tests
```

## What's here right now

- `pipeline/src/egoannote/media/` — ffprobe/ffmpeg wrappers. One `probe()` call is the
  single source of truth for a video's fps/duration, shared by every stage
  so two tracks never assume two different clocks.
- `pipeline/src/egoannote/layers/hands.py` — MediaPipe hand tracking, local CPU, free.
  Fixes a real bug from the pipeline this was rebuilt from: MediaPipe often
  labels *both* detected hands "Right" in egocentric footage, and the naive
  fix (trust the first label match) silently drops one hand. This version
  assigns both detections jointly using frame-to-frame continuity, with
  handedness labels only breaking ties when there's no prior frame to anchor
  to.
- `pipeline/src/egoannote/parse.py` — a 4-tier JSON repair ladder (fence-strip →
  `json.loads` → `json_repair` → regex fallback) for VLM responses, with a
  `pydantic`-validated schema. Distinguishes "the JSON parsed" from "the JSON
  matched the shape we needed" — a response can be valid JSON and still
  carry zero usable data, and that distinction matters downstream.
- `pipeline/src/egoannote/backends/` — one interface, swappable implementations. The
  real captioning backend talks to any OpenAI-compatible endpoint
  (OpenRouter, Ollama, vLLM, etc.), loaded from `models.toml` by
  `backends/registry.py`; a deterministic fake backend makes the whole
  pipeline testable with no network access. Each image is sent with a
  `Frame N` text label immediately before it, because the caption prompt
  asks the model to return frame indices referring to those labels.
- `pipeline/src/egoannote/store.py` — one SQLite database (laptop-only — GPU pods
  write immutable artifact shards, never a shared database file, so two
  machines can never silently overwrite each other's rows).
- `tests/` — pytest, no GPU or network required. A large share are
  regression guards for specific bugs found in review; each names the defect
  it pins.

## What's not here yet

- The segmentation algorithm — the design is finalized but deliberately not
  implemented until real hand-tracking data exists to calibrate its
  thresholds against.
- The review UI, GPU perception layers (objects/depth/camera-pose), and the
  published benchmarks.

## Known limitations (stated here, not buried)

- A real redacted clip is available locally, but the real two-model caption
  pilot still needs final model/provider entries and credentials.
- The annotation quality claims in the project plan are targets, not
  results, until real data runs through the pipeline.

## License

Code: MIT (see `LICENSE`). The approved face-free v1 dataset release uses
CC BY 4.0. Other original-derived workflows and operator diagnostics remain
private by default and are not part of that public release.
