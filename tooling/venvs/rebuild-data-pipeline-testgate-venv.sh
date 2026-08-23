#!/usr/bin/env bash
# Rebuild data-pipeline-testgate-venv from tooling/venvs/data-pipeline-testgate-venv.requirements.txt.
#
#   ./rebuild-data-pipeline-testgate-venv.sh <target-dir>
#
# Estate gate suite `data_pipeline` (DATA_PIPELINE_PYTHON: juergen/data_pipeline/tests).
# Marker import: cv2 -- the reason this suite does not run under the juergen
# testgate venv, which has no opencv.
#
# PROVENANCE HOLE: `desktop` is installed from the working tree at
# /fast/home/franz.srambical/desktop, not from a sha. The rebuild takes whatever
# that tree holds when it runs.

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
  -r "$HERE/data-pipeline-testgate-venv.requirements.txt"

"$TARGET/bin/python" -VV
