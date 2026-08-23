#!/usr/bin/env bash
# Rebuild omegalax-venv from tooling/venvs/omegalax-venv.requirements.txt.
#
#   ./rebuild-omegalax-venv.sh <target-dir>
#
# The pre-rearchitecture omegalax environment, editable on
# /fast/home/franz.srambical/omegalax @ feat/extra-transforms-hook. No pytest: it
# is a run environment, not a gate.
#
# PROVENANCE HOLE: `-e file:///fast/home/franz.srambical/omegalax`, a branch that
# carried 5 uncommitted files at capture time.

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
  -r "$HERE/omegalax-venv.requirements.txt"

"$TARGET/bin/python" -VV
