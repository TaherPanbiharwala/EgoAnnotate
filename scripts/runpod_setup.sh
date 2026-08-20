#!/usr/bin/env bash
# Idempotent RunPod setup for jobs/10_blur_egoblur.py.
#
# WHY THIS EXISTS: RunPod wipes the CONTAINER disk on stop/start and keeps
# only the VOLUME (mounted at /workspace). Anything installed into /root or
# /usr/local is gone after a restart — including uv and any ffmpeg you built
# or copied there — while the pod still looks fully set up because the repo,
# the weights and the footage all live on the volume and survived. So this
# installs everything into /workspace/bin instead, and is safe to re-run:
# it IS the recovery procedure after every pod restart. SSH access
# (/root/.ssh/authorized_keys) has the identical problem and gets the
# identical fix below — see the SSH section for the one-time setup that
# makes it durable across every future pod on this same volume.
#
# Usage, from anywhere on the pod:
#     bash /workspace/egoannote/scripts/runpod_setup.sh
#     export PATH="/workspace/bin:$PATH"; export UV_CACHE_DIR=/workspace/.uv-cache
#     export RCLONE_CONFIG=/workspace/.rclone.conf
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
# rclone defaults its config to /root/.config/rclone/rclone.conf — container
# disk, so a pod stop silently discards the Google Drive OAuth token and the
# whole copy-paste `rclone authorize` flow has to be repeated. Keep it on the
# volume instead. The token is a credential: 0600, and never in the repo.
export RCLONE_CONFIG="${RCLONE_CONFIG:-/workspace/.rclone.conf}"

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
# A SEPARATE binary from ffmpeg, and the two do not always ship together —
# imageio-ffmpeg's wheel embeds ffmpeg only, so the branch above never puts
# ffprobe anywhere.
#
# It MUST land on the volume, exactly like ffmpeg and uv. An earlier version
# of this preferred `apt-get install ffmpeg` as the cheap path, which
# installs to /usr/bin — CONTAINER disk. That works once and then silently
# evaporates on the next pod restart, and the failure is maximally confusing:
# a pod that looks fully provisioned (repo, weights, footage, uv and ffmpeg
# all present on the volume) rejects the run at preflight with "ffprobe not
# found on PATH — check the pod image". It cost two separate run attempts
# before the pattern was obvious.
#
# So: only $BIN/ffprobe counts. A system ffprobe is deliberately IGNORED
# even when present, because relying on it is what produced the bug — its
# presence is a property of this boot, not of the volume.
if [ -x "$BIN/ffprobe" ]; then
    echo ">> ffprobe already on the volume: $BIN/ffprobe"
else
    echo ">> installing ffprobe into $BIN (NOT via apt — /usr/bin does not survive a stop)"
    tmp="$(mktemp -d)"
    curl -LsSf https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
        -o "$tmp/ffmpeg-static.tar.xz"
    tar -xf "$tmp/ffmpeg-static.tar.xz" -C "$tmp"
    cp "$tmp"/ffmpeg-*-amd64-static/ffprobe "$BIN/ffprobe"
    chmod +x "$BIN/ffprobe"
    rm -rf "$tmp"
    hash -r 2>/dev/null || true
fi
[ -x "$BIN/ffprobe" ] || {
    echo "FATAL: no ffprobe at $BIN/ffprobe after install" >&2
    exit 1
}
"$BIN/ffprobe" -version >/dev/null 2>&1 || {
    echo "FATAL: $BIN/ffprobe exists but will not execute" >&2
    exit 1
}

# --- rclone (optional; skip with EGOANNOTE_SKIP_RCLONE=1) --------------------
# For pulling footage straight from Google Drive to the pod, which runs at
# datacenter speed instead of being capped by a home uplink.
#
# NOT installed via the official `curl https://rclone.org/install.sh | bash`:
# that needs unzip/7z/busybox (absent on minimal images) and installs into
# /usr/bin, which is CONTAINER disk — so it would silently disappear on the
# next pod restart, exactly like uv and ffmpeg did. Python is already here
# via uv, and its zipfile module needs no system packages at all.
if [ "${EGOANNOTE_SKIP_RCLONE:-0}" = "1" ]; then
    echo ">> skipping rclone (EGOANNOTE_SKIP_RCLONE=1)"
elif [ -x "$BIN/rclone" ]; then
    echo ">> rclone already present: $("$BIN/rclone" version | head -1)"
else
    echo ">> installing rclone into $BIN"
    tmp="$(mktemp -d)"
    curl -LsS https://downloads.rclone.org/rclone-current-linux-amd64.zip \
        -o "$tmp/rclone.zip"
    uv run --no-project python - "$tmp/rclone.zip" "$BIN/rclone" <<'PY'
import shutil, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as z:
    inner = [n for n in z.namelist() if n.endswith("/rclone")]
    if len(inner) != 1:
        raise SystemExit(f"expected one rclone binary in the archive, found {inner}")
    with z.open(inner[0]) as f, open(dst, "wb") as out:
        shutil.copyfileobj(f, out)
PY
    chmod +x "$BIN/rclone"
    rm -rf "$tmp"
fi

# Rescue a config written before RCLONE_CONFIG was set, so an existing
# authenticated remote is not thrown away on the next stop.
if [ ! -f "$RCLONE_CONFIG" ] && [ -f /root/.config/rclone/rclone.conf ]; then
    echo ">> migrating rclone config off the container disk onto the volume"
    cp /root/.config/rclone/rclone.conf "$RCLONE_CONFIG"
fi
[ -f "$RCLONE_CONFIG" ] && chmod 600 "$RCLONE_CONFIG"

# --- SSH (optional; skip with EGOANNOTE_SKIP_SSH=1) --------------------------
# /root/.ssh/authorized_keys is CONTAINER disk too — wiped on every stop,
# same failure mode as everything above. Re-pasting a key into a fresh pod's
# web terminal every time is also the wrong fix on its own merits: that
# paste path is unreliable for anything longer than a line or two (this is
# how this section came to exist). Keep the real authorized_keys on the
# VOLUME and make /root/.ssh/authorized_keys a symlink to it. Populate
# /workspace/.ssh/authorized_keys ONCE and every future pod that mounts
# this same volume — different pod, different GPU, doesn't matter — has
# working SSH with zero per-pod setup.
#
# Never seed an actual key into this script or the repo. A public key isn't
# a secret, but this is a forkable public project — a key baked in here
# would mean every fork's pod trusts the original author's key by default.
if [ "${EGOANNOTE_SKIP_SSH:-0}" = "1" ]; then
    echo ">> skipping SSH setup (EGOANNOTE_SKIP_SSH=1)"
else
    VOL_SSH_DIR="/workspace/.ssh"
    mkdir -p "$VOL_SSH_DIR"
    chmod 700 "$VOL_SSH_DIR"
    touch "$VOL_SSH_DIR/authorized_keys"
    chmod 600 "$VOL_SSH_DIR/authorized_keys"

    # Rescue whatever RunPod's own account-level key injection may already
    # have written directly to container disk, before we replace it with a
    # symlink — same rescue-don't-clobber principle as the rclone config
    # migration above, so an already-working key isn't thrown away.
    if [ -f /root/.ssh/authorized_keys ] && [ ! -L /root/.ssh/authorized_keys ] \
       && [ -s /root/.ssh/authorized_keys ]; then
        echo ">> found an existing container-disk authorized_keys — merging onto the volume"
        cat /root/.ssh/authorized_keys >> "$VOL_SSH_DIR/authorized_keys"
        sort -u -o "$VOL_SSH_DIR/authorized_keys" "$VOL_SSH_DIR/authorized_keys"
    fi

    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    # Idempotent: only relink if it isn't already the right symlink, so a
    # re-run doesn't churn the file or fight some other process touching it.
    if [ ! -L /root/.ssh/authorized_keys ] || \
       [ "$(readlink /root/.ssh/authorized_keys)" != "$VOL_SSH_DIR/authorized_keys" ]; then
        rm -f /root/.ssh/authorized_keys
        ln -s "$VOL_SSH_DIR/authorized_keys" /root/.ssh/authorized_keys
    fi

    if [ ! -s "$VOL_SSH_DIR/authorized_keys" ]; then
        echo ">> $VOL_SSH_DIR/authorized_keys is empty — SSH will reject every key until you add one, ONCE:"
        echo "     echo '<paste your public key>' >> $VOL_SSH_DIR/authorized_keys"
        echo "   After that, every pod mounting this volume has working SSH automatically."
    fi

    # sshd is a running SERVICE, not a file that needs to survive a stop —
    # reinstalling/restarting it fresh every boot is normal and cheap, unlike
    # uv/ffmpeg/rclone above where the whole point was avoiding exactly that.
    if ! command -v sshd >/dev/null 2>&1; then
        echo ">> installing openssh-server"
        apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null
    fi
    mkdir -p /run/sshd
    if pgrep -x sshd >/dev/null 2>&1; then
        echo ">> sshd already running"
    else
        echo ">> starting sshd"
        /usr/sbin/sshd
    fi
fi

echo
echo "--- ready ---"
echo "uv      : $(command -v uv) — $(uv --version)"
echo "ffmpeg  : $(command -v ffmpeg) — $(ffmpeg -version | head -1)"
echo "ffprobe : $(command -v ffprobe) — $(ffprobe -version | head -1)"
if [ -x "$BIN/rclone" ]; then
    echo "rclone  : $BIN/rclone — $("$BIN/rclone" version | head -1)"
fi
if [ "${EGOANNOTE_SKIP_SSH:-0}" != "1" ]; then
    n_keys=$(grep -c . /workspace/.ssh/authorized_keys 2>/dev/null || echo 0)
    sshd_state=$(pgrep -x sshd >/dev/null 2>&1 && echo running || echo "NOT running")
    echo "ssh     : $n_keys key(s) on the volume, sshd $sshd_state"
fi
echo
if [ -f "$RCLONE_CONFIG" ]; then
    echo "rclone cfg: $RCLONE_CONFIG (persists across pod stops)"
else
    echo "rclone cfg: none yet — \`rclone config\` will write to $RCLONE_CONFIG"
fi
# Write the env to a file on the VOLUME so a reconnected terminal is one
# `source` away from working, instead of three exports typed from memory.
# Forgetting them fails in two different and equally confusing ways: `uv:
# command not found`, or — worse — a silently working `uv` that uses the
# container-disk cache and re-downloads the whole torch environment.
ENV_FILE="/workspace/env.sh"
cat > "$ENV_FILE" <<EOF
# Generated by scripts/runpod_setup.sh — source this in every new shell:
#     source /workspace/env.sh
export PATH="$BIN:\$PATH"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export RCLONE_CONFIG="$RCLONE_CONFIG"
EOF

echo
echo "In every new shell (a script cannot set its parent's environment), run:"
echo "  source $ENV_FILE"
