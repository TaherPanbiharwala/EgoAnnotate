#!/usr/bin/env bash
# Idempotent RunPod setup for jobs/10_blur_egoblur.py.
#
# WHY THIS EXISTS: RunPod wipes the CONTAINER disk on stop/start and keeps
# only the VOLUME (mounted at /workspace). Anything installed into /root or
# /usr/local is gone after a restart — including uv and any ffmpeg you built
# or copied there — while the pod still looks fully set up because the repo,
# the weights and the footage all live on the volume and survived. So this
# installs everything into /workspace/bin instead, and is safe to re-run:
# it IS the recovery procedure after every pod restart.
#
# Usage, from anywhere on the pod:
#     bash /workspace/egoannote/scripts/runpod_setup.sh
#     export PATH="/workspace/bin:$PATH"; export UV_CACHE_DIR=/workspace/.uv-cache
#
# The second line matters: a script cannot change its parent shell's PATH.
set -euo pipefail

BIN="${EGOANNOTE_BIN:-/workspace/bin}"
mkdir -p "$BIN"
export PATH="$BIN:$PATH"
# Persist uv's environment cache on the volume too, or every restart re-downloads
# torch and the ~45-package job environment from scratch.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

# --- uv ---------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    echo ">> uv already present: $(uv --version)"
else
    echo ">> installing uv into $BIN"
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$BIN" sh
fi

# --- ffmpeg -----------------------------------------------------------------
# The job passes -fps_mode on every decode, which needs ffmpeg >= 5.0.
# Ubuntu 22.04 ships 4.4, which rejects it — and ffprobe keeps working, so
# clip discovery passes and every decode silently produces zero bytes. That
# failure reached a real run and was reported as "the probe found no faces",
# i.e. a statement about the footage from a binary that never decoded a frame.
# Test the OPTION, not a parsed version string: the option is the thing that
# has to work. Same check runs again in the job's own preflight().
supports_fps_mode() {
    command -v ffmpeg >/dev/null 2>&1 &&
        ffmpeg -hide_banner -loglevel error -f lavfi -i nullsrc=s=16x16:d=0.1 \
            -fps_mode passthrough -f null - >/dev/null 2>&1
}

if supports_fps_mode; then
    echo ">> ffmpeg already supports -fps_mode"
else
    echo ">> system ffmpeg is too old (or absent) — installing a static build into $BIN"
    # imageio-ffmpeg ships a self-contained static ffmpeg and is already a
    # declared test dependency of this project, for exactly this reason:
    # a contributor's system ffmpeg is not guaranteed present or working.
    uv run --no-project --with imageio-ffmpeg python -c \
        "import imageio_ffmpeg, shutil; shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), '$BIN/ffmpeg')"
    chmod +x "$BIN/ffmpeg"
    hash -r 2>/dev/null || true
    supports_fps_mode || {
        echo "FATAL: installed ffmpeg still rejects -fps_mode" >&2
        exit 1
    }
fi

echo
echo "--- ready ---"
echo "uv      : $(command -v uv) — $(uv --version)"
echo "ffmpeg  : $(command -v ffmpeg) — $(ffmpeg -version | head -1)"
echo "ffprobe : $(command -v ffprobe) — $(ffprobe -version | head -1)"
echo
echo "Run this in your shell (a script cannot set its parent's PATH):"
echo "  export PATH=\"$BIN:\$PATH\"; export UV_CACHE_DIR=$UV_CACHE_DIR"
