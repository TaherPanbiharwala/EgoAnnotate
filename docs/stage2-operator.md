# Stage II operator guide

Stage II adds face redaction to an already-redacted EgoBlur video. DINO and
SAM2 inspect the original, but rendering reads pixels only from Stage I.
Technical completion is not permission to publish: an exact output needs a
separate immutable human acceptance record.

## Persistent RunPod setup

After every pod restart:

```bash
cd /workspace/egoannote
bash scripts/runpod_setup_stage2.sh
source /workspace/stage2-env.sh
bash scripts/runpod_stage2.sh doctor --workspace-root /workspace
```

The setup is idempotent. It stores `uv`, Hugging Face, Torch, model, runtime,
and job caches under `/workspace`; verifies the pinned DINO and SAM2 hashes;
and rejects a changed SAM2 commit, remote, tracked source, or configuration.
Use `--dry-run` to inspect paths without writing and `--verify-only` to forbid
downloads. Real CUDA/model loading remains locked until the Milestone 6 smoke
test.

## Commands

Global `--json` emits one machine-readable object. Global `--dry-run` reports
planned writes without changing the run.

| Command | Milestone 5 behavior |
|---|---|
| `doctor` | Checks persistent tools, caches, exact weights, runtime, and config. |
| `smoke` | Runs deterministic fake DINO/SAM plumbing. Its output is never reviewable or releasable. |
| `pilot` | Describes the fixed real-GPU pilot and stops at the Milestone 6 boundary. |
| `sweep` | Describes the `0.15/0.20/0.25/0.30` threshold sweep and stops at the GPU boundary. |
| `run` | Supports `--fake`; production execution remains gated to Milestone 6. |
| `status` | Reports processing, automated audit, and human review independently. |
| `resume` | Clears a cooperative stop and preserves compatible completed layers. |
| `stop` | Writes an idempotent cooperative stop marker without deleting artifacts. |
| `review` | Creates a content-bound immutable `ACCEPTED` or `REJECTED` human record. |
| `release-check` | Requires exact accepted hashes and scans staged release content for private bytes. |

The golden path available now ends deliberately at the GPU boundary:

```bash
bash scripts/runpod_setup_stage2.sh
source /workspace/stage2-env.sh
bash scripts/runpod_stage2.sh doctor --workspace-root /workspace
bash scripts/runpod_stage2.sh --dry-run pilot GX010057
```

## Private artifacts

Labels, manual seeds, evidence extracts, and generated review flags are
content-addressed below each run's `DO-NOT-SHIP/` directory. They may never be
copied into a public package. `release-check` rejects both the directory name
and renamed files whose content hash matches any private artifact.

The versioned labels envelope is:

```json
{
  "schema_version": 1,
  "artifact_type": "labels",
  "clip_id": "GX010057",
  "labels": [
    {
      "schema_version": 1,
      "event_id": "face-0001",
      "clip_id": "GX010057",
      "frame_start": 120,
      "frame_end": 135,
      "conservative_box": [100, 80, 180, 190],
      "label_kind": "face_event",
      "visibility": "partial",
      "category": "profile",
      "stage1_verdict": "missed",
      "dino_proposal_verdicts": [{"proposal_id": "p1", "verdict": "face"}],
      "final_mask_coverage": null,
      "reviewer_disposition": "pending"
    }
  ]
}
```

Use `label_kind: "negative_example"` for hands, tools, product labels, and
background patterns that should not be treated as faces. The manual-seed
envelope uses `artifact_type: "manual-seeds"` and a `seeds` list containing
`schema_version`, `seed_id`, `clip_id`, `frame_idx`, `box`, and `reason`.

Every private artifact used to produce an output is recorded in that exact
processing manifest. Creating a correction afterward immediately makes the
effective review status `PENDING` and blocks release.

## Correction and review

Explicit recomputation invalidates only the named layer and downstream work:

```bash
bash scripts/runpod_stage2.sh resume \
  --work-dir /workspace/stage2 --run-id RUN --clip-id CLIP \
  --recompute-from sam
```

`dino` removes DINO/SAM/render evidence, `sam` preserves DINO, and `render`
preserves DINO and SAM. Every choice invalidates the processing manifest and
all earlier human review records. Private evidence remains under
`DO-NOT-SHIP` so the corrected run can bind its full history.

Acceptance requires explicit attestations that the reviewer watched the full
clip and inspected every flagged interval:

```bash
bash scripts/runpod_stage2.sh review \
  --work-dir /workspace/stage2 --run-id RUN --clip-id CLIP \
  --decision ACCEPTED --reviewer REVIEWER \
  --reviewed-at 2026-08-20T12:00:00Z \
  --full-clip-reviewed --flagged-intervals-reviewed
```

The record binds the exact processing-manifest and rendered-output SHA-256.
Editing the output, manifest, correction set, or review content invalidates it.

## Release gate

Stage only public artifacts, including the exact accepted video, in a separate
directory and run:

```bash
bash scripts/runpod_stage2.sh release-check \
  --work-dir /workspace/stage2 --run-id RUN --clip-id CLIP \
  --release-root /workspace/release-candidate
```

Passing means technical processing is complete, a current human acceptance
exists, the exact accepted output is staged, and no known private content is
present. It does not package or upload anything automatically.
