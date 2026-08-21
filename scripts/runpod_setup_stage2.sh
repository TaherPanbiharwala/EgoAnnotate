#!/usr/bin/env bash
# Idempotent persistent setup for the Stage II DINO/SAM2 environment.
set -euo pipefail

WORKSPACE_ROOT="${EGOANNOTE_STAGE2_WORKSPACE:-/workspace}"
DRY_RUN=0
VERIFY_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --workspace-root)
            if [ "$#" -lt 2 ]; then
                echo "FATAL: --workspace-root requires a path" >&2
                exit 2
            fi
            WORKSPACE_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=1
            shift
            ;;
        *)
            echo "FATAL: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$WORKSPACE_ROOT" ]; then
    echo "FATAL: --workspace-root must be a dedicated persistent directory" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "FATAL: python3 is required to validate --workspace-root safely" >&2
    exit 2
fi
WORKSPACE_ROOT="$(python3 - "$WORKSPACE_ROOT" <<'PY'
import os
import pathlib
import sys

candidate = pathlib.Path(sys.argv[1]).expanduser()
if candidate.is_symlink():
    print("FATAL: --workspace-root may not be a symbolic link", file=sys.stderr)
    raise SystemExit(2)
resolved_text = os.path.realpath(candidate)
if resolved_text.startswith("//"):
    resolved_text = "/" + resolved_text.lstrip("/")
resolved = pathlib.Path(resolved_text)
if resolved == pathlib.Path("/"):
    print(
        "FATAL: --workspace-root must resolve to a dedicated persistent directory",
        file=sys.stderr,
    )
    raise SystemExit(2)
print(resolved)
PY
)"
if [ "$WORKSPACE_ROOT" = "/" ]; then
    echo "FATAL: --workspace-root must resolve to a dedicated persistent directory" >&2
    exit 2
fi
if [ -e "$WORKSPACE_ROOT" ] && [ ! -d "$WORKSPACE_ROOT" ]; then
    echo "FATAL: --workspace-root exists but is not a directory: $WORKSPACE_ROOT" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$WORKSPACE_ROOT/bin"
UV_CACHE="$WORKSPACE_ROOT/.uv-cache"
HF_CACHE="$WORKSPACE_ROOT/.cache/huggingface"
TORCH_CACHE="$WORKSPACE_ROOT/.cache/torch"
MODEL_ROOT="$WORKSPACE_ROOT/models/stage2"
DINO_DIR="$MODEL_ROOT/dino"
SAM_DIR="$MODEL_ROOT/sam2"
SAM_RUNTIME="$MODEL_ROOT/sam2-runtime"
STAGE2_WORK="$WORKSPACE_ROOT/stage2"
ENV_FILE="$WORKSPACE_ROOT/stage2-env.sh"
ASSET_MANIFEST="$MODEL_ROOT/assets.manifest.json"
GPU_SMOKE_RESULT="$MODEL_ROOT/gpu-smoke.json"

DINO_REVISION="e76a695ed7ae1032a61530cce4b4e9b65f4e368b"
DINO_SHA256="5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
SAM_REVISION="2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM_SHA256="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
SAM_CONFIG_SHA256="1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
DINO_WEIGHTS="$DINO_DIR/model.safetensors"
SAM_CHECKPOINT="$SAM_DIR/sam2.1_hiera_large.pt"

show_plan() {
    echo "Stage II setup plan (no writes):"
    echo "  workspace       $WORKSPACE_ROOT"
    echo "  uv cache        $UV_CACHE"
    echo "  Hugging Face    $HF_CACHE"
    echo "  Torch cache     $TORCH_CACHE"
    echo "  DINO revision   $DINO_REVISION"
    echo "  SAM2 revision   $SAM_REVISION"
    echo "  work directory  $STAGE2_WORK"
    echo "  GPU smoke       $GPU_SMOKE_RESULT"
}
if [ "$DRY_RUN" = "1" ]; then
    show_plan
    exit 0
fi

mkdir -p "$BIN_DIR" "$UV_CACHE" "$HF_CACHE" "$TORCH_CACHE" \
    "$DINO_DIR" "$SAM_DIR" "$STAGE2_WORK"
export PATH="$BIN_DIR:$PATH"
export UV_CACHE_DIR="$UV_CACHE"
export HF_HOME="$HF_CACHE"
export TORCH_HOME="$TORCH_CACHE"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$SAM_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v uv >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1 || \
   ! command -v ffprobe >/dev/null 2>&1; then
    if [ "$VERIFY_ONLY" = "1" ]; then
        echo "FATAL: uv, ffmpeg, and ffprobe are required for --verify-only" >&2
        exit 1
    fi
    EGOANNOTE_BIN="$BIN_DIR" EGOANNOTE_SKIP_RCLONE=1 EGOANNOTE_SKIP_SSH=1 \
        bash "$SCRIPT_DIR/runpod_setup.sh"
    export PATH="$BIN_DIR:$PATH"
fi

sha256_path() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

verify_hash() {
    local path="$1"
    local expected="$2"
    local label="$3"
    if [ ! -f "$path" ] || [ -L "$path" ]; then
        echo "FATAL: $label is missing or is a symbolic link: $path" >&2
        exit 1
    fi
    local actual
    actual="$(sha256_path "$path")"
    if [ "$actual" != "$expected" ]; then
        echo "FATAL: $label hash mismatch: expected $expected, got $actual" >&2
        exit 1
    fi
}

download_atomic() {
    local url="$1"
    local destination="$2"
    local expected="$3"
    local label="$4"
    if [ -f "$destination" ]; then
        verify_hash "$destination" "$expected" "$label"
        echo ">> $label already verified"
        return
    fi
    if [ "$VERIFY_ONLY" = "1" ]; then
        echo "FATAL: $label is absent in verify-only mode: $destination" >&2
        exit 1
    fi
    local partial="$destination.partial"
    if [ -e "$partial" ]; then
        echo "FATAL: interrupted download exists: $partial; inspect and remove it explicitly" >&2
        exit 1
    fi
    echo ">> downloading $label"
    curl -fL --retry 3 --output "$partial" "$url"
    verify_hash "$partial" "$expected" "$label"
    mv "$partial" "$destination"
}

# Populate the exact Transformers snapshot, including processor/config files.
if [ "$VERIFY_ONLY" != "1" ]; then
    export TRANSFORMERS_OFFLINE=0
    export HF_HUB_OFFLINE=0
    DINO_SNAPSHOT="$(uv run --no-project --with 'huggingface-hub==0.29.1' python - \
        "$DINO_REVISION" "$DINO_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="IDEA-Research/grounding-dino-base",
    revision=sys.argv[1],
    local_dir=sys.argv[2],
))
PY
)"
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    verify_hash "$DINO_SNAPSHOT/model.safetensors" "$DINO_SHA256" "Grounding DINO weights"
fi
verify_hash "$DINO_WEIGHTS" "$DINO_SHA256" "Grounding DINO weights"

download_atomic \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
    "$SAM_CHECKPOINT" "$SAM_SHA256" "SAM2.1 Hiera Large checkpoint"

if [ ! -d "$SAM_RUNTIME/.git" ]; then
    if [ "$VERIFY_ONLY" = "1" ]; then
        echo "FATAL: pinned SAM2 runtime is absent: $SAM_RUNTIME" >&2
        exit 1
    fi
    if [ -e "$SAM_RUNTIME" ] || [ -e "$SAM_RUNTIME.partial" ]; then
        echo "FATAL: SAM2 runtime path exists without a verified clone; inspect it manually" >&2
        exit 1
    fi
    echo ">> cloning pinned SAM2 runtime"
    git clone --filter=blob:none https://github.com/facebookresearch/sam2.git \
        "$SAM_RUNTIME.partial"
    git -C "$SAM_RUNTIME.partial" checkout --detach "$SAM_REVISION"
    mv "$SAM_RUNTIME.partial" "$SAM_RUNTIME"
fi

SAM_ACTUAL_REVISION="$(git -C "$SAM_RUNTIME" rev-parse HEAD)"
SAM_REMOTE="$(git -C "$SAM_RUNTIME" remote get-url origin)"
if [ "$SAM_ACTUAL_REVISION" != "$SAM_REVISION" ]; then
    echo "FATAL: SAM2 runtime revision mismatch: $SAM_ACTUAL_REVISION" >&2
    exit 1
fi
case "$SAM_REMOTE" in
    https://github.com/facebookresearch/sam2|https://github.com/facebookresearch/sam2.git)
        ;;
    *)
        echo "FATAL: SAM2 runtime came from unexpected remote: $SAM_REMOTE" >&2
        exit 1
        ;;
esac
if [ -n "$(git -C "$SAM_RUNTIME" status --porcelain --untracked-files=no)" ]; then
    echo "FATAL: SAM2 runtime has tracked modifications; refusing an unpinned runtime" >&2
    exit 1
fi
verify_hash "$SAM_RUNTIME/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" \
    "$SAM_CONFIG_SHA256" "SAM2 model configuration"

cat > "$ENV_FILE" <<EOF
# Generated by scripts/runpod_setup_stage2.sh; source in every new shell.
export PATH="$BIN_DIR:\$PATH"
export UV_CACHE_DIR="$UV_CACHE"
export HF_HOME="$HF_CACHE"
export TORCH_HOME="$TORCH_CACHE"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$SAM_RUNTIME\${PYTHONPATH:+:\$PYTHONPATH}"
export EGOANNOTE_STAGE2_WORK_DIR="$STAGE2_WORK"
export EGOANNOTE_STAGE2_DINO_WEIGHTS="$DINO_WEIGHTS"
export EGOANNOTE_STAGE2_SAM_CHECKPOINT="$SAM_CHECKPOINT"
EOF

uv run "$REPO_DIR/jobs/20_deidentify_stage2.py" --version >/dev/null
uv run --no-project python - "$ASSET_MANIFEST" "$DINO_WEIGHTS" "$SAM_CHECKPOINT" \
    "$SAM_RUNTIME" "$DINO_REVISION" "$SAM_REVISION" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

manifest, dino, sam, runtime, dino_revision, sam_revision = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "schema_version": 1,
    "artifact_type": "stage2_assets",
    "dino": {"path": dino, "revision": dino_revision, "sha256": digest(dino)},
    "sam": {"path": sam, "revision": sam_revision, "sha256": digest(sam)},
    "sam_runtime": {"path": runtime, "revision": sam_revision},
}
path = pathlib.Path(manifest)
temporary = path.with_suffix(path.suffix + ".partial")
with temporary.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY

GPU_SMOKE_PARTIAL="$GPU_SMOKE_RESULT.partial"
if [ -e "$GPU_SMOKE_PARTIAL" ]; then
    echo "FATAL: interrupted GPU smoke result exists: $GPU_SMOKE_PARTIAL" >&2
    exit 1
fi
echo ">> running offline sequential DINO/SAM2 CUDA smoke"
uv run "$REPO_DIR/jobs/20_deidentify_stage2.py" --json doctor \
    --workspace-root "$WORKSPACE_ROOT" --load-models > "$GPU_SMOKE_PARTIAL"
uv run --no-project python - "$GPU_SMOKE_PARTIAL" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "PASS_OFFLINE_GPU_SMOKE":
    raise SystemExit(f"GPU smoke did not pass: {payload}")
# Match the asset-manifest write's fsync-before-rename a few lines above:
# without this, a crash right after the mv below can leave "verified" GPU
# smoke evidence that was never actually flushed to disk.
fd = os.open(str(path), os.O_RDONLY)
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
mv "$GPU_SMOKE_PARTIAL" "$GPU_SMOKE_RESULT"

echo ">> Stage II assets, environment, and offline GPU smoke verified"
echo ">> source $ENV_FILE"
echo ">> bash scripts/runpod_stage2.sh doctor --workspace-root $WORKSPACE_ROOT"
echo ">> GPU smoke evidence: $GPU_SMOKE_RESULT"
