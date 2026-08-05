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

## Try it — no accounts, no GPU, no API key

```bash
git clone <this-repo> && cd egoannote
uv sync
uv run scripts/demo.py
```

Runs the real pipeline (probe → frame extraction → MediaPipe hand tracking →
VLM captioning) against a bundled synthetic clip, using a deterministic fake
backend in place of a real API call. It's a plumbing smoke test, not proof
of caption quality — see the script's own output for what it does and
doesn't demonstrate.

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
- `tests/` — 74 tests, pytest, no GPU or network required. A large share are
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
