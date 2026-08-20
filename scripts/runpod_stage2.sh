#!/usr/bin/env bash
# Thin command wrapper for the self-contained Stage II PEP-723 job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UV_BIN="${EGOANNOTE_UV:-uv}"

exec "$UV_BIN" run "$REPO_DIR/jobs/20_deidentify_stage2.py" "$@"
