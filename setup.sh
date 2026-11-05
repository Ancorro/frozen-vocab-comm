#!/usr/bin/env bash
# Install from PyPI (CPU is enough for the tests; a GPU is used for the real checkpoints when present).
set -euo pipefail
python -m pip install -q -e ".[test]"
