#!/usr/bin/env bash
# Rebuild the hash-locked test environment and run Juergen's complete local gate.
# No CI or shared mutable venv contributes to this verdict.

set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "usage: tooling/local_release_gate.sh" >&2
  exit 2
fi

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$HERE/.." && pwd -P)"
DESKTOP_ROOT="$(cd -- "$ROOT/../desktop" 2>/dev/null && pwd -P)" || {
  echo "release gate: exact sibling desktop checkout is missing" >&2
  exit 2
}
REBUILD="$ROOT/tooling/venvs/rebuild-juergen-testgate-venv.sh"
MANIFEST="$ROOT/tooling/venvs/juergen-testgate-venv.requirements.txt"

for command in git uv mktemp; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "release gate: required command is missing: $command" >&2
    exit 2
  }
done

if [ "$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null)" != "$ROOT" ]; then
  echo "release gate: Juergen root is not a canonical checkout top level" >&2
  exit 2
fi
if [ "$(git -C "$DESKTOP_ROOT" rev-parse --show-toplevel 2>/dev/null)" != "$DESKTOP_ROOT" ]; then
  echo "release gate: desktop root is not a canonical checkout top level" >&2
  exit 2
fi
git -C "$ROOT" ls-files --error-unmatch -- \
  tooling/local_release_gate.sh \
  tooling/venvs/rebuild-juergen-testgate-venv.sh \
  tooling/venvs/juergen-testgate-venv.requirements.txt >/dev/null

desktop_pins=()
mapfile -t desktop_pins < <(
  sed -n 's/^desktop = { git = .* rev = "\([0-9a-f]\{40\}\)" }$/\1/p' \
    "$ROOT/pyproject.toml"
)
if [ "${#desktop_pins[@]}" -ne 1 ]; then
  echo "release gate: pyproject must name exactly one full desktop source revision" >&2
  exit 2
fi
desktop_pin="${desktop_pins[0]}"

head_before="$(git -C "$ROOT" rev-parse --verify 'HEAD^{commit}')"
desktop_before="$(git -C "$DESKTOP_ROOT" rev-parse --verify 'HEAD^{commit}')"
desktop_remote="$(git -C "$DESKTOP_ROOT" rev-parse --verify \
  'refs/remotes/origin/main^{commit}' 2>/dev/null)" || {
  echo "release gate: desktop refs/remotes/origin/main is missing" >&2
  exit 2
}
if [ "$desktop_before" != "$desktop_pin" ] || [ "$desktop_remote" != "$desktop_pin" ]; then
  echo "release gate: desktop checkout/remote does not equal pinned $desktop_pin" >&2
  exit 2
fi

clean_tree() {
  local root="$1"
  [ -z "$(git -C "$root" status --porcelain=v1 --untracked-files=all \
    --ignore-submodules=none)" ]
}

if ! clean_tree "$ROOT"; then
  echo "release gate: Juergen checkout is dirty" >&2
  exit 2
fi
if ! clean_tree "$DESKTOP_ROOT"; then
  echo "release gate: desktop checkout is dirty" >&2
  exit 2
fi
git -C "$ROOT" diff --check
git -C "$DESKTOP_ROOT" diff --check
git -C "$ROOT" fsck --strict
git -C "$DESKTOP_ROOT" fsck --strict
uv lock --check --project "$ROOT"

scratch_parent="${TMPDIR:-/var/tmp}"
case "$scratch_parent" in
  /*) ;;
  *) echo "release gate: TMPDIR must be absolute" >&2; exit 2 ;;
esac
if [ "$scratch_parent" = / ] || [ ! -d "$scratch_parent" ] \
  || [ -L "$scratch_parent" ] \
  || [ "$(cd -- "$scratch_parent" && pwd -P)" != "$scratch_parent" ]; then
  echo "release gate: TMPDIR must be an existing canonical non-root directory" >&2
  exit 2
fi
scratch="$(mktemp -d "$scratch_parent/juergen-release-gate.XXXXXX")"
cleanup() {
  case "$scratch" in
    "$scratch_parent"/juergen-release-gate.*) rm -rf -- "$scratch" ;;
    *) echo "release gate: refusing unsafe scratch cleanup: $scratch" >&2 ;;
  esac
}
trap cleanup EXIT

"$REBUILD" "$scratch/venv"
(
  cd -- "$ROOT"
  unset PYTHONHOME PYTHONPATH PYTEST_ADDOPTS
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  "$scratch/venv/bin/python" -m pytest -q -p no:cacheprovider tests grammars
)

if [ "$(git -C "$ROOT" rev-parse --verify 'HEAD^{commit}')" != "$head_before" ] \
  || ! clean_tree "$ROOT"; then
  echo "release gate: Juergen checkout moved or became dirty during the gate" >&2
  exit 1
fi
if [ "$(git -C "$DESKTOP_ROOT" rev-parse --verify 'HEAD^{commit}')" != "$desktop_before" ] \
  || ! clean_tree "$DESKTOP_ROOT"; then
  echo "release gate: desktop checkout moved or became dirty during the gate" >&2
  exit 1
fi

printf 'JUERGEN LOCAL RELEASE GATE: GREEN  head=%s  desktop=%s\n' \
  "$head_before" "$desktop_before"
