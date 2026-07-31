#!/usr/bin/env bash
set -euo pipefail

GATE_REPORT="${1:?gate report}"
MODEL_PATH="${2:?model path}"
OUTPUT_DIR="${3:?output dir}"
SHARD_INDEX="${4:?shard index}"
REPO_ROOT="${5:?juergen snapshot root}"

BUDGET="$REPO_ROOT/stage5_rft/config/relative_mouse_pilot_budget.json"
RUNTIME_SUMS="$REPO_ROOT/stage5_rft/config/relative_mouse_runtime.sha256"
RFT_STAR=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/rl_scratch/osworld_rl/rft_star
RL_ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/reinforcement-learning
RL_VENV="$RL_ROOT/prime-rl/.venv"
EXPECTED_MODEL_SHA=013f6c62d00d8c101a82d6ecc983f9ddadd5bca8b91664bd4bcf47bc8ddc25c3

jq -e '.threshold_crossed == true and .launch_authorized == false' "$GATE_REPORT" >/dev/null
jq -e '.data.class == "synthetic_training_only"
  and .data.contains_official_heldout == false
  and .data.contains_real_vm_eval == false
  and .data.contains_crowd_cast == false
  and .collection.tasks == 1536
  and .collection.k == 8
  and .collection.shards == 3
  and .collection.adaptive_resampling == false
  and .collection.maximum_rollout_errors == 0' "$BUDGET" >/dev/null
test "$(jq -r '.binding.checkpoint_sha256' "$GATE_REPORT")" = "$EXPECTED_MODEL_SHA"
test "$(jq -r '.binding.checkpoint_uri' "$GATE_REPORT")" = "$MODEL_PATH"
(cd / && sha256sum -c "$RUNTIME_SUMS")
test -f "$MODEL_PATH/model.safetensors"
printf '%s  %s\n' "$EXPECTED_MODEL_SHA" "$MODEL_PATH/model.safetensors" | sha256sum -c -

mkdir -p "$OUTPUT_DIR"
CACHE_ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp/relative_mouse_pilot_${SLURM_JOB_ID:?}
mkdir -p "$CACHE_ROOT/inductor" "$CACHE_ROOT/triton"
export VLLM_CACHE_ROOT="$CACHE_ROOT"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export PROJECT_DIR="$RL_VENV"
export PATH="$RL_VENV/bin:$PATH"
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/rl_scratch/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TMPDIR=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp

PORT=$((8400 + (SLURM_JOB_ID % 1000)))
VLLM_LOG="$LABCTL_RUN_DIR/vllm.log"
vllm serve "$MODEL_PATH" --served-model-name probe --runner generate \
  --port "$PORT" --api-key movebox --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 2}' --trust-remote-code >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 240); do
  if curl -sf -H 'Authorization: Bearer movebox' "http://localhost:$PORT/v1/models" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    tail -100 "$VLLM_LOG"
    exit 1
  fi
  sleep 5
done
if [ "$READY" != 1 ]; then
  tail -100 "$VLLM_LOG"
  exit 1
fi

"$RL_VENV/bin/python" "$RFT_STAR/sample.py" \
  --model probe --base_url "http://localhost:$PORT/v1" --api_key movebox \
  --shard_id "$SHARD_INDEX" --n_shards 3 --n_tasks 1536 --k 8 \
  --label relative_mouse_pilot_v1 --out_dir "$OUTPUT_DIR" \
  --temperature 0.7 --max_steps 8 --concurrency 96 --err_tol 0

STATS="$OUTPUT_DIR/stats_shard${SHARD_INDEX}.json"
jq -e --argjson shard "$SHARD_INDEX" '.summary.status == "OK"
  and .summary.n_tasks == 512
  and .summary.k == 8
  and .summary.n_rollouts_ok == 4096
  and .summary.n_rollouts_err == 0
  and .summary.error_rate == 0
  and .summary.pool == "train"
  and .summary.train_seed == 1001
  and .summary.shard_id == $shard
  and .summary.n_shards == 3' "$STATS" >/dev/null

"$RL_VENV/bin/python" - "$OUTPUT_DIR" "$SHARD_INDEX" "$MODEL_PATH" "$EXPECTED_MODEL_SHA" "$BUDGET" "$STATS" <<'PY'
import hashlib, json, pathlib, sys
out, shard, model, model_sha, budget, stats = sys.argv[1:]
out = pathlib.Path(out)
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
payload = {
    "schema_version": "stage5.relative_mouse_shard.v1",
    "status": "complete",
    "label": "relative_mouse_pilot_v1",
    "shard_id": int(shard),
    "n_shards": 3,
    "policy_checkpoint_uri": model,
    "policy_checkpoint_sha256": model_sha,
    "budget_sha256": sha(budget),
    "stats_sha256": sha(stats),
    "rollouts_sha256": sha(out / f"rollouts_shard{shard}.jsonl"),
    "contains_official_heldout": False,
    "contains_real_vm_eval": False,
    "contains_crowd_cast": False,
    "adaptive_resampling": False,
}
(out / "sample_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
