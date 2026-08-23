#!/usr/bin/env bash
# Rebuild juergen-testgate-venv from tooling/venvs/juergen-testgate-venv.requirements.txt.
#
#   ./rebuild-juergen-testgate-venv.sh <target-dir>
#
# Estate gate suites `juergen` (JUERGEN_PYTHON: tests/ + grammars/) and `desktop`
# (DESKTOP_PYTHON: the sibling desktop checkout's tests/). Marker imports:
# verifiers, PIL.
#
# The only manifest in this directory with --hash pins, so the install runs under
# --require-hashes: any substituted artifact aborts it.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: $0 <target-dir>}"

# --no-config on every uv call: these scripts live inside the juergen
# checkout, whose pyproject.toml carries `[tool.uv] override-dependencies =
# ["openai>=2.26.0"]`. uv discovers it by walking up from the script and
# applies it to an unrelated environment -- and it defeats --require-hashes
# outright, since an override is not `==` pinned.
uv venv --no-config --python 3.12.11 "$TARGET"
uv pip install --no-config --python "$TARGET/bin/python" --require-hashes \
  -r "$HERE/juergen-testgate-venv.requirements.txt"

"$TARGET/bin/python" -VV
