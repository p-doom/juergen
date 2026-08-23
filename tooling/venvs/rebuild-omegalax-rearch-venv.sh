#!/usr/bin/env bash
# Rebuild omegalax-rearch-venv from tooling/venvs/omegalax-rearch-venv.requirements.txt.
#
#   ./rebuild-omegalax-rearch-venv.sh <target-dir>
#
# The omegalax-rearch working environment. Package set is identical to
# omegalax-rearch-gate-venv -- byte-for-byte, all 166 lines -- so the two agree by
# construction and the gate's isolation costs nothing.
#
# PROVENANCE HOLE: `-e file:///fast/home/franz.srambical/omegalax-rearch`.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: $0 <target-dir>}"

# --no-config on every uv call: these scripts live inside the juergen
# checkout, whose pyproject.toml carries `[tool.uv] override-dependencies =
# ["openai>=2.26.0"]`. uv discovers it by walking up from the script and
# applies it to an unrelated environment -- and it defeats --require-hashes
# outright, since an override is not `==` pinned.
uv venv --no-config --python 3.13.5 "$TARGET"
uv pip install --no-config --python "$TARGET/bin/python" \
  -r "$HERE/omegalax-rearch-venv.requirements.txt"

"$TARGET/bin/python" -VV
