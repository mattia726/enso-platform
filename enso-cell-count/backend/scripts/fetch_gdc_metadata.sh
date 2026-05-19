#!/usr/bin/env bash
# Refresh pathologist metadata and slide list from GDC API.
# Run from repository root. Writes data/processed/wedge_mvp_dataset.xlsx.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
python backend/scripts/build_wedge_dataset.py "$@"
