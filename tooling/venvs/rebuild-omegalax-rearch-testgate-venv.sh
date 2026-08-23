#!/usr/bin/env bash
# Rebuild omegalax-rearch-testgate-venv from tooling/venvs/omegalax-rearch-testgate-venv.requirements.txt.
#
#   ./rebuild-omegalax-rearch-testgate-venv.sh <target-dir>
#
# RETIRED. The predecessor of omegalax-rearch-gate-venv, superseded 2026-08-20 and
# referenced by nothing in the estate. Kept as a manifest because it is the venv
# estate_gate.sh's header blames: it had drifted months ahead of the lock
# (transformers 5.14.1 vs 5.2.0, jax 0.11.0 vs 0.9.2) and failed the Qwen3-VL MoE
# loader on a config key HF renamed after the locked version. Rebuild it only to
# reproduce that failure.
#
# PROVENANCE HOLES: `renderers` and `tokamax` are both installed from working
# trees, not shas.

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
  -r "$HERE/omegalax-rearch-testgate-venv.requirements.txt"

"$TARGET/bin/python" -VV
