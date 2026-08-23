#!/usr/bin/env bash
# Rebuild omegalax-rearch-gate-venv from tooling/venvs/omegalax-rearch-gate-venv.requirements.txt.
#
#   ./rebuild-omegalax-rearch-gate-venv.sh <target-dir>
#
# Estate gate suite `omegalax_rearch` (OMEGALAX_PYTHON). Marker import: jax.
# Deliberately separate from omegalax-rearch's own .venv, which the training jobs
# share: a bare `uv sync --locked` there uninstalls torch and torchvision, and
# would do it under a running gate.
#
# Originally built by `uv sync --locked --extra torch-tests --python 3.13` from
# omegalax-rearch's lockfile. That lock remains the upstream source of truth; this
# manifest records what the lock produced, so the two can be diffed.
#
# PROVENANCE HOLE: `-e file:///fast/home/franz.srambical/omegalax-rearch` (the
# editable repo under test) is a path, not a sha.

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
  -r "$HERE/omegalax-rearch-gate-venv.requirements.txt"

"$TARGET/bin/python" -VV
