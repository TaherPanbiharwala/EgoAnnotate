# egoannote

**Dense subtask annotation for egocentric video — measured, not asserted.**

Turns raw first-person video into verb/noun subtask segments with hand
pose, object masks, relative depth and camera pose — and reports real
numbers for how good each field actually is, instead of just shipping a
demo. Runs on one laptop plus a small amount of rented GPU time.

> **Status: early build.** The core pipeline (frame extraction, hand
> tracking, VLM captioning, storage) is implemented and tested. Segmentation,
> the review UI, GPU perception layers (objects/depth/pose), and HF packaging
> are in progress — see `docs/` and the task list in this repo for current
> status. This README will grow a results table and a hero clip once real
> footage has been annotated (currently blocked on Google Drive access — see
> Known limitations below).

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
run on CPU; the future GPU perception layers (objects/depth/pose) are
isolated PEP 723 job scripts under `jobs/`, not part of this environment.

## Try it — no accounts, no GPU, no API key

```bash
uv run scripts/demo.py
```

Runs the real pipeline (probe → frame extraction → MediaPipe hand tracking →
VLM captioning) against a bundled synthetic clip, using a deterministic fake
backend in place of a real API call. It's a plumbing smoke test, not proof
of caption quality — see the script's own output for what it does and
doesn't demonstrate.

## Usage — annotate your own footage

There is deliberately no CLI or package to install — fork the repo and call
the layers directly. This mirrors what `scripts/demo.py` does, pointed at a
real clip instead of the bundled synthetic one.

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

### VLM captioning (needs an API key and a model in `models.toml`)

1. Pick or add a model entry in `models.toml` under `[models.<id>]` — fill
   in real prices from your provider before setting a spend cap (a `0.0`
   price disables the cap silently, by design of the validation, so it's
   loud if you forget: `build_backend` will refuse a spend cap against a
   zero-priced model).
2. Export your key: `export OPENROUTER_API_KEY=sk-...` (never put it in
   `models.toml` — that file is committed).
3. Run:

```python
from egoannote.backends.registry import build_backend
from egoannote.layers import caption as caption_layer
from egoannote.store import Store

backend = build_backend("example-a")  # the [models.example-a] id from models.toml
store = Store(Path("runs") / video_id / "annotations.db")

n = caption_layer.caption_video(video, video_id, backend, store)
print(f"wrote {n} caption window(s)")
```

Calls run concurrently (`config.CAPTION_MAX_WORKERS`, default 8) — pass
`max_workers=1` to `caption_video` for strictly serial, deterministic
execution. Progress is resumable: re-running `caption_video` against the
same `store` skips windows already recorded and only retries error rows.

### Config knobs

Everything tunable lives in `src/egoannote/config.py`, each with a comment
explaining where the number came from. Two are environment overrides rather
than constants, for a checkout that isn't the repo root or a non-editable
install:

- `EGOANNOTE_DATA_DIR` — where runtime data (models cache, frame cache, SQLite,
  reports) lives. Defaults to `./runs`.
- `EGOANNOTE_MODELS_TOML` — path to `models.toml`. Defaults to the repo-root
  copy; a wheel install doesn't bundle it, so this is required there.

### Tests and lint

```bash
uv run pytest
uv run --with ruff ruff check src scripts jobs tests
```

## What's here right now

- `src/egoannote/media/` — ffprobe/ffmpeg wrappers. One `probe()` call is the
  single source of truth for a video's fps/duration, shared by every stage
  so two tracks never assume two different clocks.
- `src/egoannote/layers/hands.py` — MediaPipe hand tracking, local CPU, free.
  Fixes a real bug from the pipeline this was rebuilt from: MediaPipe often
  labels *both* detected hands "Right" in egocentric footage, and the naive
  fix (trust the first label match) silently drops one hand. This version
  assigns both detections jointly using frame-to-frame continuity, with
  handedness labels only breaking ties when there's no prior frame to anchor
  to.
- `src/egoannote/parse.py` — a 4-tier JSON repair ladder (fence-strip →
  `json.loads` → `json_repair` → regex fallback) for VLM responses, with a
  `pydantic`-validated schema. Distinguishes "the JSON parsed" from "the JSON
  matched the shape we needed" — a response can be valid JSON and still
  carry zero usable data, and that distinction matters downstream.
- `src/egoannote/backends/` — one interface, swappable implementations. The
  real captioning backend talks to any OpenAI-compatible endpoint
  (OpenRouter, Ollama, vLLM, etc.), loaded from `models.toml` by
  `backends/registry.py`; a deterministic fake backend makes the whole
  pipeline testable with no network access. Each image is sent with a
  `Frame N` text label immediately before it, because the caption prompt
  asks the model to return frame indices referring to those labels.
- `src/egoannote/store.py` — one SQLite database (laptop-only — GPU pods
  write immutable artifact shards, never a shared database file, so two
  machines can never silently overwrite each other's rows).
- `tests/` — 103 tests, pytest, no GPU or network required. A large share are
  regression guards for specific bugs found in review; each names the defect
  it pins.

## What's not here yet

- The segmentation algorithm (`src/egoannote/layers/segment.py`) — the
  design is finalized but deliberately not implemented until real hand-
  tracking data exists to calibrate its thresholds against. See that file's
  docstring for the full design and why it's staged this way.
- The review UI, GPU perception layers (objects/depth/camera-pose), dataset
  packaging, and the published benchmarks.

## Known limitations (stated here, not buried)

- No real egocentric footage has been processed yet — the source video is
  on Google Drive and getting local access is an open item.
- The annotation quality claims in the project plan are targets, not
  results, until real data runs through the pipeline.

## License

Code: MIT (see `LICENSE`). The eventual published dataset will be licensed
separately under CC BY 4.0 — see `docs/DATASHEET.md` once written.
