#!/usr/bin/env bash
# Rebuild eval-venv from tooling/venvs/eval-venv.requirements.txt.
#
#   ./rebuild-eval-venv.sh <target-dir>
#
# Serving/evaluation (sglang 0.5.6.post1 from PyPI, torch 2.9.1, transformers
# 5.0.0rc0). Not on any gate path.
#
# Interpreter is the conda-forge 3.12.9 at /opt/miniforge3, not a uv-managed
# build. Pinned by path on purpose: `uv venv --python 3.12.9` would fetch a
# uv-managed 3.12.9 instead, a different build of the same version.
#
# BROKEN PROVENANCE: `-e file:///fast/home/franz.srambical/eval` points at a
# directory that no longer exists. That line is commented out in the manifest, so
# the rebuild succeeds but is NOT equivalent to the captured venv.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: $0 <target-dir>}"

# --no-config on every uv call: these scripts live inside the juergen
# checkout, whose pyproject.toml carries `[tool.uv] override-dependencies =
# ["openai>=2.26.0"]`. uv discovers it by walking up from the script and
# applies it to an unrelated environment -- and it defeats --require-hashes
# outright, since an override is not `==` pinned.
uv venv --no-config --python /opt/miniforge3/bin/python3.12 "$TARGET"
uv pip install --no-config --python "$TARGET/bin/python" \
  -r "$HERE/eval-venv.requirements.txt"

"$TARGET/bin/python" -VV
