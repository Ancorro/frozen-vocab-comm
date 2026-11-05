#!/usr/bin/env bash
# CPU checks, under a minute: channel tests on tiny random models.
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q
