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
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/goal_timeline_viewer/viewer_%j.log

# CPU-only goal-timeline viewer on a compute node, reverse-tunneled to the login node.
# HEVC decodes client-side so the server is IO-bound. login->compute ssh is blocked, but
# compute->login works, so the job opens `ssh -R` back to the login node: the viewer ends
# up reachable at hai-login1:PORT, leaving the user's existing mac->login tunnel unchanged.
#   Usage: sbatch serve_viewer.sbatch [PORT] [LOGIN_HOST] [DATA_SUBDIR]
set -eo pipefail
PORT="${1:-8791}"
LOGIN="${2:-hai-login1.haicore.berlin}"
DATA_SUB="${3:-e2c26556}"
VIEW=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/goal_timeline_viewer
DATA="$VIEW/data/$DATA_SUB"
cd "$VIEW"
echo "[viewer] node=$(hostname) port=$PORT login=$LOGIN data=$DATA"

# 1) start the viewer on loopback
uv run python annotator.py serve --data_dir "$DATA" --host 127.0.0.1 --port "$PORT" --no_prewarm &
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
