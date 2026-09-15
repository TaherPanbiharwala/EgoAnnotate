# EgoBlur privacy-redaction workflow

This directory is the GPU-only, self-contained privacy-redaction workflow.
It produces redacted video and private review artifacts before footage enters
the annotation pipeline. It does not import or require `pipeline/`.

## Run on a GPU pod

From the repository root on the pod:

```bash
bash egoblur/runpod_setup.sh
uv run egoblur/job.py --help
uv run egoblur/review.py --help
```

`job.py` uses its own PEP 723 dependency block, including CUDA PyTorch. It is
intended for a Linux GPU pod; use the main `uv sync` environment for the
local annotation pipeline instead.

The job's manifests, source footage, timelines, and review artifacts remain
private. Only the separate, allowlisted public-release exporter may create
files for Hugging Face.
