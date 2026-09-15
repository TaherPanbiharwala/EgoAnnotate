# Historical Hugging Face publishing pipeline

This directory holds the mechanism that turned an EgoBlur-redacted video into
a published Hugging Face dataset: MediaPipe + dense VLM captioning on the
redacted video, human redaction approval, privacy-safe bundle packaging, and
upload. It's parked, not part of the active annotation pipeline (see the repo
root README for what is) — kept working as a reference for anyone who wants
to follow the same publishing mechanism, since this repo is public. Nothing
here is wired into the main `egoannote-run` CLI.

## Commands

```bash
uv run public-release-tools/cli.py --help
uv run public-release-tools/cli.py annotate --help
uv run public-release-tools/cli.py approve-redaction --help
uv run public-release-tools/cli.py publish-hf --help
uv run public-release-tools/cli.py prepare-public-release --help
```

`annotate_redacted.py` (MediaPipe hands + dense VLM captioning on a
`--redacted-video`, plus Hugging Face bundle packaging) and `public_release.py`
(builds an owner-approved public release folder from curated face-free
artifacts) import the installed `egoannote` package normally — unlike
`egoblur/`, this folder has no CUDA/GPU isolation requirement, so there's no
need to vendor code by copy.

`pack/` (`huggingface.py`) is the privacy-safe bundle builder: parquet/JSON
export, license installation, and the actual Hub upload call.

`export_frame_previews.py` and `extract_preview.sh` cut short, web-friendly
preview clips (used for the project blog and README preview videos) from
already-published release videos — see their own `--help` / usage comments.

See [`../docs/MEDIAPIPE_VLM_PIPELINE.md`](../docs/MEDIAPIPE_VLM_PIPELINE.md)
and [`../docs/PUBLIC_FACE_FREE_RELEASE.md`](../docs/PUBLIC_FACE_FREE_RELEASE.md)
for the full workflow documentation.
