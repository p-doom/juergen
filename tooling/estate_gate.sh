#!/usr/bin/env bash
# Run every test suite in the estate, each under the interpreter it needs.
#
# There is no CI. Nothing is pushed. Three repositories hold the code and each
# suite needs a different interpreter, so a single `pytest` cannot cover them:
#
#   juergen/tests + juergen/grammars   verifiers + Pillow, and desktop on
#                                      sys.path (juergen/tests/conftest.py adds
#                                      the sibling checkout).
#   juergen/data_pipeline/tests        needs cv2 (opencv-python-headless), which
#                                      the testgate venv does not have -- so it
#                                      runs under its own. Not `juergen/.venv`:
#                                      that one is uv's project environment, so
#                                      any `uv sync` without the `dev` extra
#                                      prunes pytest out from under a run in
#                                      progress (observed mid-gate).
#   desktop                            Pillow and nothing else.
#   desktop/desktop_fleet              the second distribution in the desktop
#                                      repository: the ZMQ stack plus desktop
#                                      itself, in its own .venv.
#   omegalax-rearch                    jax + tokamax + transformers + a CPU torch
#                                      (transformers' AutoImageProcessor needs
#                                      torchvision) + `renderers` from PyPI.
#
# Every suite runs even if an earlier one fails. Exit status: 0 all green, 1 a
# suite failed, 2 an environment is missing (nothing was measured -- do not read
# that as green).
#
# Five interpreters, each overridable. The defaults are the venvs these suites
# are run under on this cluster; on another machine, set the variables.
#
#   JUERGEN_PYTHON          tests + grammars        (default: shared testgate venv)
#   DATA_PIPELINE_PYTHON    data_pipeline/tests     (default: shared data-pipeline
#                                                   testgate venv)
#   DESKTOP_PYTHON          desktop                 (default: shared testgate venv)
#   DESKTOP_FLEET_PYTHON    desktop/desktop_fleet   (default: that project's .venv)
#   OMEGALAX_PYTHON         omegalax-rearch         (default: shared rearch venv)
#
# Three checkouts, located as siblings of this repository and overridable, plus
# the fleet project inside the desktop one:
#
#   JUERGEN_ROOT  DESKTOP_ROOT  OMEGALAX_REARCH_ROOT  DESKTOP_FLEET_ROOT
#
# `--list` prints the plan and the preflight verdict without running anything.
# `--only <name>` runs one suite (juergen | data_pipeline | desktop |
# desktop_fleet | omegalax_rearch).
#
# omegalax-rearch: only the four test files the rearchitecture touches are run
# (30 tests). The rest of that repo's tests want real GPUs and real checkpoints
# and would fail on a CPU node.

set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${JUERGEN_ROOT:=$(cd -- "$HERE/.." && pwd)}"
SIBLINGS="$(cd -- "$JUERGEN_ROOT/.." && pwd)"
: "${DESKTOP_ROOT:=$SIBLINGS/desktop}"
: "${DESKTOP_FLEET_ROOT:=$DESKTOP_ROOT/desktop_fleet}"
: "${OMEGALAX_REARCH_ROOT:=$SIBLINGS/omegalax-rearch}"

VENVS=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/venvs
: "${JUERGEN_PYTHON:=$VENVS/juergen-testgate-venv/bin/python}"
: "${DATA_PIPELINE_PYTHON:=$VENVS/data-pipeline-testgate-venv/bin/python}"
: "${DESKTOP_PYTHON:=$VENVS/juergen-testgate-venv/bin/python}"
: "${DESKTOP_FLEET_PYTHON:=$DESKTOP_FLEET_ROOT/.venv/bin/python}"
: "${OMEGALAX_PYTHON:=$VENVS/omegalax-rearch-testgate-venv/bin/python}"

# omegalax-rearch's 30-test gate: what the rearchitecture changed, nothing GPU-bound.
OMEGALAX_TESTS=(
  tests/test_sft_collators.py
  tests/test_arrayrecord_image_refs.py
  tests/test_renderers_loss_mask_gate.py
  tests/test_chatml_loss_mask_leakage.py
)

# name | root | interpreter | marker module | pytest targets
SUITES=(
  "juergen|$JUERGEN_ROOT|$JUERGEN_PYTHON|verifiers|tests grammars"
  "data_pipeline|$JUERGEN_ROOT|$DATA_PIPELINE_PYTHON|cv2|data_pipeline/tests"
  "desktop|$DESKTOP_ROOT|$DESKTOP_PYTHON|PIL|tests"
  "desktop_fleet|$DESKTOP_FLEET_ROOT|$DESKTOP_FLEET_PYTHON|zmq|"
  "omegalax_rearch|$OMEGALAX_REARCH_ROOT|$OMEGALAX_PYTHON|jax|${OMEGALAX_TESTS[*]}"
)

ONLY=""
LIST_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }

# preflight: report every missing thing, not just the first.
problems=()
planned=()
for entry in "${SUITES[@]}"; do
  IFS='|' read -r name root python marker targets <<<"$entry"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  planned+=("$name")
  if [ ! -d "$root" ]; then
    problems+=("$name: checkout not found at $root")
    continue
  fi
  if [ ! -x "$python" ]; then
    problems+=("$name: interpreter not found at $python")
    continue
  fi
  if ! "$python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$marker') else 1)" 2>/dev/null; then
    problems+=("$name: $python cannot import '$marker' -- wrong venv for this suite")
    continue
  fi
  if ! "$python" -c "import pytest" 2>/dev/null; then
    problems+=("$name: $python has no pytest")
  fi
done

if [ -n "$ONLY" ] && [ ${#planned[@]} -eq 0 ]; then
  echo "unknown suite: $ONLY" >&2
  exit 2
fi

# tests/test_packaging.py builds the juergen and desktop wheels to prove the five
# flat plugin ids resolve outside a checkout. uv is the estate's build front end
# and is in none of the suites' venvs, so it is preflight, not a suite failure.
case " ${planned[*]} " in
  *" juergen "*)
    command -v uv >/dev/null 2>&1 || problems+=("juergen: uv not on PATH -- tests/test_packaging.py builds a wheel with it")
    ;;
esac

if [ "$LIST_ONLY" = 1 ]; then
  bold "estate gate plan"
  for entry in "${SUITES[@]}"; do
    IFS='|' read -r name root python marker targets <<<"$entry"
    [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
    printf '  %-16s %s\n' "$name" "$root"
    printf '  %-16s %s %s\n' "" "$python" "-m pytest ${targets:-.}"
  done
  if [ ${#problems[@]} -gt 0 ]; then
    bold "preflight: NOT RUNNABLE"
    printf '  - %s\n' "${problems[@]}"
    exit 2
  fi
  bold "preflight: ok"
  exit 0
fi

if [ ${#problems[@]} -gt 0 ]; then
  bold "estate gate: ENVIRONMENT INCOMPLETE -- nothing was measured"
  printf '  - %s\n' "${problems[@]}"
  echo
  echo "Set the interpreter for each suite explicitly; see --help."
  exit 2
fi

log_dir="$(mktemp -d -t estate-gate-XXXXXX)"
results=()
failed=0
started=$SECONDS

for entry in "${SUITES[@]}"; do
  IFS='|' read -r name root python marker targets <<<"$entry"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  bold "==> $name  ($root)"
  suite_started=$SECONDS
  log="$log_dir/$name.log"
  # omegalax-rearch is not installed into its interpreter, so it is imported
  # from the checkout; JAX_PLATFORMS=cpu keeps a GPU node from being claimed by
  # a test suite that does not need one.
  (
    cd "$root" || exit 3
    if [ "$name" = "omegalax_rearch" ]; then
      export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
      export JAX_PLATFORMS=cpu
    fi
    # shellcheck disable=SC2086  # targets is an intentional word list
    exec "$python" -m pytest $targets -q -p no:cacheprovider
  ) 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  elapsed=$((SECONDS - suite_started))
  count="$(grep -Eo '[0-9]+ (passed|failed|error)' "$log" | tr '\n' ' ' | sed 's/ $//')"
  if [ "$status" -eq 0 ]; then
    results+=("PASS|$name|${count:-?}|${elapsed}s")
  else
    results+=("FAIL|$name|${count:-rc=$status}|${elapsed}s")
    failed=1
  fi
done

echo
bold "estate gate summary   ($((SECONDS - started))s total, logs in $log_dir)"
for row in "${results[@]}"; do
  IFS='|' read -r verdict name count elapsed <<<"$row"
  printf '  %-4s  %-16s %-28s %s\n' "$verdict" "$name" "$count" "$elapsed"
done

if [ "$failed" -ne 0 ]; then
  bold "ESTATE GATE: RED"
  exit 1
fi
bold "ESTATE GATE: GREEN"
exit 0
