# EgoBlur privacy-redaction workflow

This directory holds the automated face-redaction workflow. It's parked, not
part of the active annotation pipeline (see the repo root README for what is)
— kept working and self-contained as a reference for anyone who wants to
follow the same mechanism. Nothing here imports or requires `pipeline/`.

## Run on a GPU pod

From the repository root on the pod:

```bash
bash egoblur/runpod_setup.sh
uv run egoblur/job.py --help
uv run egoblur/review.py --help
```

`job.py` and `review.py` are standalone PEP 723 scripts (`job.py`'s block
includes CUDA PyTorch). They are intended for a Linux GPU pod; use the main
`uv sync` environment for the local annotation pipeline instead.

## Pre/post-redaction review tooling (local CPU, no pod needed)

`pose_prior.py`, `hand_prior.py`, and `verify_yunet.py` are a real Python
package (`egoblur/`, note the `__init__.py`) — pre-redaction MediaPipe priors
that `job.py`'s wearer-hand suppression consumes, and an independent
post-redaction YuNet face audit. `probe.py` is a vendored copy of
`pipeline/src/egoannote/media/probe.py`, kept in sync by hand, for the same
self-containment reason `job.py` vendors its own dependencies instead of
importing `pipeline/`.

```bash
uv run egoblur/cli.py --help
uv run egoblur/cli.py pose-prior --help
uv run egoblur/cli.py hand-prior --help
uv run egoblur/cli.py verify-yunet --help
```

The job's manifests, source footage, timelines, and review artifacts remain
private. Only the separate, allowlisted public-release exporter may create
files for Hugging Face.
