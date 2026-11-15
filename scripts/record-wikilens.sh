#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ "${1:-}" == "--offline" ]]; then
  shift
else
  set -- --live "$@"
fi
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m wikilens.recording "$@"
fi
exec python3 -m wikilens.recording "$@"
