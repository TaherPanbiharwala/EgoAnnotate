# Project decisions

## 2026-09-03 — Public release and website direction

### Dataset status

- **EgoAnnotate v1 is public:** https://huggingface.co/datasets/TaherPanbiharwala/EgoAnnotate
- The release contains 13 manually reviewed face-free clips, each with a clean MP4, hand/caption overlay MP4, hand JSON/Parquet, captions JSON, and a public manifest.
- The release package was verified against the published Hub metadata: 42/42 LFS SHA-256 values and 40/40 regular Git blob hashes match.
- Retain local copies until there is a separate backup that the owner trusts.

### Website decision: Dataset Atlas

Build a quiet, research-lab, video-first public site rather than a generic startup landing page or a dense internal dashboard.

The homepage should include:

1. A concise EgoAnnotate introduction with a strong video preview.
2. A browsable grid of all 13 clips.
3. Per-clip preview with clean/overlay video access, captions, hand annotations, public metadata, and a Hugging Face download link.
4. A curation-and-annotations section explaining manually reviewed face-free footage, output segments, and discontinuities.
5. A reserved blog section marked as writing in progress. Blog content will be supplied later.

Website work is separate from the private annotation pipeline. It may link to the public Hugging Face repository but must not expose private artifacts, local paths, private review evidence, source timelines, or private-only labels.
