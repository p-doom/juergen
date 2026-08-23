#!/usr/bin/env bash
# Rebuild sglang-eval-venv from tooling/venvs/sglang-eval-venv.requirements.txt.
#
#   ./rebuild-sglang-eval-venv.sh <target-dir>
#
# Serving/evaluation against a forked sglang
# (JustinTong0323/sglang @ 39b37ca6, subdirectory=python). Not on any gate path.
# The oldest venv in the estate (2026-03-31).
#
# Interpreter is the conda-forge 3.12.9 at /opt/miniforge3. Pinned by path for the
# same reason as eval-venv.
#
# The only venv that still has its own pip (24.3.1).
#
# BROKEN PROVENANCE: `-e file:///fast/home/franz.srambical/crowd-cast-eval` points
# at a directory that no longer exists. That line is commented out in the
# manifest, so the rebuild succeeds but is NOT equivalent to the captured venv.

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
  -r "$HERE/sglang-eval-venv.requirements.txt"

"$TARGET/bin/python" -VV
