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

# --- ffprobe ------------------------------------------------------------
# A SEPARATE binary from ffmpeg, and the two do not always ship together.
# imageio-ffmpeg's wheel embeds ffmpeg only — the branch above never put
# ffprobe on PATH. The first pod this script ran on happened to already
# have an old system ffprobe (Ubuntu 22.04's package), so this went
# unnoticed; a different base image with neither ffmpeg nor ffprobe
# preinstalled hit "ffprobe: command not found" the moment the job's
# preflight ran. ffprobe's VERSION does not matter for what this job asks
# of it (see open_decoder's docstring — only ancient, stable options), so
# try the cheapest source first and fall back to a source that needs
# nothing but curl+tar, since we don't know what this base image ships.
if command -v ffprobe >/dev/null 2>&1; then
    echo ">> ffprobe already present"
elif command -v apt-get >/dev/null 2>&1; then
    echo ">> installing ffprobe via apt (its ffmpeg build is irrelevant — $BIN/ffmpeg wins on PATH)"
    apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
else
    echo ">> no apt-get on this image — pulling ffprobe from the same static-build"
    echo "   source imageio-ffmpeg vendors, directly"
    tmp="$(mktemp -d)"
    curl -LsSf https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
        -o "$tmp/ffmpeg-static.tar.xz"
    tar -xf "$tmp/ffmpeg-static.tar.xz" -C "$tmp"
    cp "$tmp"/ffmpeg-*-amd64-static/ffprobe "$BIN/ffprobe"
    chmod +x "$BIN/ffprobe"
    rm -rf "$tmp"
fi
command -v ffprobe >/dev/null 2>&1 || {
    echo "FATAL: still no ffprobe on PATH after both install paths" >&2
    exit 1
}

echo
echo "--- ready ---"
echo "uv      : $(command -v uv) — $(uv --version)"
echo "ffmpeg  : $(command -v ffmpeg) — $(ffmpeg -version | head -1)"
echo "ffprobe : $(command -v ffprobe) — $(ffprobe -version | head -1)"
echo
echo "Run this in your shell (a script cannot set its parent's PATH):"
echo "  export PATH=\"$BIN:\$PATH\"; export UV_CACHE_DIR=$UV_CACHE_DIR"
