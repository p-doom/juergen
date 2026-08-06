#!/usr/bin/env bash
#SBATCH --job-name=goalviewer
#SBATCH --partition=standard
#SBATCH --account=hfmi_synergyunit
#SBATCH --qos=low
#SBATCH --gres=none
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
# Slurm parses #SBATCH directives before any shell runs, so `$VAR` / `$(...)`
# CANNOT be expanded here — the only alternatives are a hardcoded absolute path
# (which rotted on the last directory move) or a relative one. Relative is
# correct: Slurm resolves it against the job's working directory, which defaults
# to the directory `sbatch` was invoked from, and `.gitignore` in this directory
# already ignores `viewer_*.log`. So submit from HERE (see Usage below). To put
# the log elsewhere, override on the command line: `sbatch --output=<path> ...`.
#SBATCH --output=viewer_%j.log

# CPU-only goal-timeline viewer on a compute node, reverse-tunneled to the login node.
# HEVC decodes client-side so the server is IO-bound. login->compute ssh is blocked, but
# compute->login works, so the job opens `ssh -R` back to the login node: the viewer ends
# up reachable at hai-login1:PORT, leaving the user's existing mac->login tunnel unchanged.
#   Usage: cd tooling/goal_timeline_viewer
#          sbatch serve_viewer.sbatch [PORT] [LOGIN_HOST] [DATA_SUBDIR]
set -eo pipefail
PORT="${1:-8791}"
LOGIN="${2:-hai-login1.haicore.berlin}"
DATA_SUB="${3:-e2c26556}"
# Viewer directory, never hardcoded. Under Slurm the body runs from a spooled
# COPY of this file, so ${BASH_SOURCE[0]} points into /var/spool/slurmd/... —
# SLURM_SUBMIT_DIR is the only reliable handle on the real directory (and it is
# the same directory the relative --output above resolves against, so the two
# stay consistent). The BASH_SOURCE fallback covers `bash serve_viewer.sbatch`
# runs outside Slurm; VIEWER_DIR overrides both (sbatch exports the env by
# default, so `VIEWER_DIR=... sbatch ...` reaches the job).
VIEW="${VIEWER_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ ! -f "$VIEW/annotator.py" ]]; then
  echo "[viewer] ERROR: no annotator.py under '$VIEW'." >&2
  echo "[viewer]   submit from the viewer directory (cd tooling/goal_timeline_viewer &&" >&2
  echo "[viewer]   sbatch serve_viewer.sbatch), or set VIEWER_DIR=<dir>." >&2
  exit 1
fi
DATA="$VIEW/data/$DATA_SUB"
if [[ ! -f "$DATA/day.json" ]]; then
  echo "[viewer] ERROR: no day.json under '$DATA' — data/ is gitignored and" >&2
  echo "[viewer]   regenerable; run README steps 1-2 (build_recording_day.py," >&2
  echo "[viewer]   build_overlay.py) first, or pass a different DATA_SUBDIR." >&2
  exit 1
fi
cd "$VIEW"
echo "[viewer] node=$(hostname) port=$PORT login=$LOGIN data=$DATA"

# 1) start the viewer on loopback.
# `--script` is load-bearing, and the `python` it replaces was the bug.
# annotator.py is a PEP 723 script: it declares its own deps inline
# (av>=12, msgpack>=1 — see its `# /// script` header and its
# `#!/usr/bin/env -S uv run --script` shebang). Naming an interpreter
# (`uv run python annotator.py`) discards that inline metadata and falls back to
# whatever PROJECT uv discovers by walking up from the cwd — which the directory
# move silently changed: from here it finds the workspace root (juergen: deps
# pixeldesk + av, so a sync PRUNES msgpack — `uv sync --dry-run` here prints
# `- msgpack`), and from the viewer's old home under data_pipeline/ it found
# crowdcast-data-pipeline (msgpack but no av). NEITHER project declares both, so
# no pyproject can be the right answer here; the script's own metadata already
# is. `--script` also keeps the job out of the shared .venv and never touches
# the root uv.lock, and resolves from the uv cache (verified with --offline), so
# it works on a node with no network.
# That isolation — not just the metadata — is the reason to KEEP `--script`.
# A project-mode `uv run` here mutates state other people depend on: it syncs the
# shared .venv down to whatever project it discovered, so a bare
# `uv run --locked` prunes that .venv to the root project and takes torch and
# transformers with it (this already broke a training env once). A read-only
# viewer must never be able to do that. Do not "simplify" this back to
# `uv run python`.
uv run --script annotator.py serve --data_dir "$DATA" --host 127.0.0.1 --port "$PORT" --no_prewarm &
SRV=$!
trap 'kill $SRV $TUN 2>/dev/null' EXIT

# 2) wait until it answers
for i in $(seq 1 40); do
  curl -sf -m2 "http://127.0.0.1:$PORT/api/overlay" >/dev/null 2>&1 && { echo "[viewer] server up"; break; }
  kill -0 $SRV 2>/dev/null || { echo "[viewer] server died early"; exit 1; }
  sleep 1
done

# 3) reverse tunnel: bind login:$PORT -> this node's 127.0.0.1:$PORT, auto-reconnect
echo "[viewer] opening reverse tunnel -> $LOGIN:$PORT"
while kill -0 $SRV 2>/dev/null; do
  ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no \
      -R "$PORT:127.0.0.1:$PORT" "$LOGIN" || echo "[viewer] tunnel dropped (rc=$?), retrying in 5s"
  sleep 5
done &
TUN=$!
wait $SRV
