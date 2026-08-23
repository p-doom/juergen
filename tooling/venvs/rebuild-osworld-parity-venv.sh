#!/usr/bin/env bash
# Rebuild osworld-parity-venv from tooling/venvs/osworld-parity-venv.requirements.txt.
#
#   ./rebuild-osworld-parity-venv.sh <target-dir>
#
# OSWorld parity evaluation: the one environment where juergen's dependencies and
# OSWorld's coexist. It exists because they could not be merged into an existing
# venv -- a numpy 1.x/2.x ABI conflict forced a dedicated build.
#
# Largest environment in the estate (323 distributions) and the least healthy:
# opencv-python==5.0.0.93 and opencv-python-headless==4.8.1.78 are installed side
# by side, two versions of the same extension module in one import path.
#
# PROVENANCE HOLE: `juergen @ file:///fast/home/franz.srambical/juergen-nn` -- a
# second juergen checkout on branch native-normalized, not this one.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: $0 <target-dir>}"

# --no-config on every uv call: these scripts live inside the juergen
# checkout, whose pyproject.toml carries `[tool.uv] override-dependencies =
# ["openai>=2.26.0"]`. uv discovers it by walking up from the script and
# applies it to an unrelated environment -- and it defeats --require-hashes
# outright, since an override is not `==` pinned.
uv venv --no-config --python 3.12.11 "$TARGET"
uv pip install --no-config --python "$TARGET/bin/python" \
  -r "$HERE/osworld-parity-venv.requirements.txt"

"$TARGET/bin/python" -VV
