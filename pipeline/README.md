# Annotation pipeline

This directory contains the installable `egoannote` package for local
annotation, curation, validation, and public-release export. It deliberately
does not depend on the GPU-only EgoBlur workflow.

## Run it

From the repository root:

```bash
uv sync
uv run pipeline/demo.py
uv run egoannote-run --help
```

`pipeline/src/egoannote/` is the Python package source. `prompts/` holds the
caption prompts, and `models.toml` is the local VLM registry. Override its
location for a wheel install with `EGOANNOTE_MODELS_TOML`.

The approved public release process is documented in
[`../docs/PUBLIC_FACE_FREE_RELEASE.md`](../docs/PUBLIC_FACE_FREE_RELEASE.md).
The public dataset is available at
<https://huggingface.co/datasets/TaherPanbiharwala/EgoAnnotate>.
