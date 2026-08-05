# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""The ONE thing every jobs/*.py script agrees on. Vendored BY COPY into
each job script (not imported — jobs run as self-contained PEP-723 scripts
on ephemeral pods with no shared environment), with a CI test asserting all
copies stay byte-identical to this canonical file.

Why vendored-by-copy instead of a shared package: PEP-723 scripts declare
their own dependencies and run in an isolated `uv run` environment with
nothing else on the path — that isolation is the whole point (see the
plan's S9 fix for why: it's what stops one model's CUDA/torch requirements
from ever coexisting with another's, or with the laptop's torch-free venv).
A real import would break that isolation. Byte-identical copies + a CI
check give the same anti-drift guarantee without it.

Every job writes ONLY this: raw arrays + this metadata. NO schema
knowledge, no database, no interpretation of what the numbers mean — that
all lives laptop-side in src/egoannote/layers/*.py, which is the only place
that ingests these shards into the SQLite store.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "1"


@dataclass(slots=True)
class ShardMeta:
    contract_version: str
    video_id: str
    pod_role: str  # "cpu" | "gpu"
    stage: str  # "hands" | "blur" | "wilor" | "sam3" | "depth_v3" | "mast3r_slam"
    shard_index: int
    frame_indices: list[int]
    timestamp_ms: list[int]  # SAME clock as media.probe.probe() — do not
    # re-derive fps independently in a job script; pass it in from the
    # manifest the laptop already computed, or you reintroduce the
    # two-independent-clocks problem this project explicitly fixed.
    model_name: str
    model_version: str


def write_shard(
    out_dir: Path,
    meta: ShardMeta,
    arrays: dict[str, Any],
) -> Path:
    """Write one immutable shard: `<shard_index>.npz` + `<shard_index>_meta.json`.

    Uniquely prefixed per (pod_role, stage, shard_index) by the caller's
    out_dir convention (`<volume>/runs/<run_id>/<pod_role>/<stage>/`) — this
    is what makes two pods writing concurrently safe with NO locking: they
    never share a destination path.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{meta.shard_index}.npz"
    meta_path = out_dir / f"{meta.shard_index}_meta.json"

    np.savez_compressed(npz_path, **arrays)
    meta_path.write_text(json.dumps(asdict(meta)), encoding="utf-8")
    return npz_path


def read_shard(out_dir: Path, shard_index: int) -> tuple[ShardMeta, dict[str, Any]]:
    import numpy as np

    meta_path = out_dir / f"{shard_index}_meta.json"
    npz_path = out_dir / f"{shard_index}.npz"

    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta_dict.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"shard {npz_path} was written with contract_version="
            f"{meta_dict.get('contract_version')!r}, expected {CONTRACT_VERSION!r}. "
            f"A pod is running an out-of-date copy of _contract.py."
        )
    meta = ShardMeta(**meta_dict)
    arrays = dict(np.load(npz_path))
    return meta, arrays
