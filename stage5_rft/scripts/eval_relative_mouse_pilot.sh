#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?model path}"
OUTPUT_DIR="${2:?output dir}"
REPO_ROOT="${3:?juergen snapshot root}"

RUNTIME_SUMS="$REPO_ROOT/stage5_rft/config/relative_mouse_runtime.sha256"
RFT_STAR=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/rl_scratch/osworld_rl/rft_star
RL_ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/reinforcement-learning
RL_VENV="$RL_ROOT/prime-rl/.venv"

(cd / && sha256sum -c "$RUNTIME_SUMS")
jq -e '.status == "complete" and .method == "pure_rejection_sft"
  and .training_steps == 100 and .contains_official_heldout == false
  and .contains_crowd_cast == false' "$MODEL_PATH/pilot_manifest.json" >/dev/null

mkdir -p "$OUTPUT_DIR"
find "$MODEL_PATH" -maxdepth 1 -type f -name '*.safetensors' -print0 \
  | sort -z | xargs -0 -r sha256sum >"$OUTPUT_DIR/checkpoint.sha256"
test -s "$OUTPUT_DIR/checkpoint.sha256"
CHECKPOINT_DIGEST="$(sha256sum "$OUTPUT_DIR/checkpoint.sha256" | cut -d' ' -f1)"
MODEL_ARTIFACT_ID="$(jq -er '.inputs[] | select(.role == "model") | .artifact_id' "$LABCTL_CONTEXT")"

CACHE_ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp/relative_mouse_eval_${SLURM_JOB_ID:?}
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

PORT=$((9400 + (SLURM_JOB_ID % 500)))
VLLM_LOG="$LABCTL_RUN_DIR/vllm.log"
vllm serve "$MODEL_PATH" --served-model-name candidate --runner generate \
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

PROBE="$OUTPUT_DIR/candidate_probe.json"
"$RL_VENV/bin/python" "$RFT_STAR/probe.py" \
  --model candidate --base_url "http://localhost:$PORT/v1" --api_key movebox \
  --n_tasks 128 --offset 0 --k 16 --max_steps 8 --temperature 0.7 \
  --concurrency 96 --err_tol 0 --label relative_mouse_pilot_candidate_v1 \
  --out "$PROBE"

jq -e '.summary.status == "OK" and .summary.n_tasks == 128
  and .summary.k == 16 and .summary.n_rollouts_ok == 2048
  and .summary.n_rollouts_err == 0 and .summary.error_rate == 0
  and .summary.pool == "val" and .summary.val_seed == 7777
  and .summary.offset == 0 and .summary.temperature == 0.7
  and .summary.max_steps == 8' "$PROBE" >/dev/null

python3 - "$OUTPUT_DIR" "$MODEL_PATH" "$MODEL_ARTIFACT_ID" \
  "$CHECKPOINT_DIGEST" "$PROBE" "$SLURM_JOB_ID" <<'PY'
import hashlib, json, pathlib, sys

out, model, artifact_id, checkpoint_digest, probe, job_id = sys.argv[1:]
out = pathlib.Path(out)
probe_path = pathlib.Path(probe)
payload = {
    "schema_version": "stage5.relative_mouse_pilot_eval.v1",
    "status": "complete",
    "model_path": model,
    "model_artifact_id": artifact_id,
    "checkpoint_digest": checkpoint_digest,
    "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
    "slurm_job_id": job_id,
    "data_class": "synthetic_untouched_validation",
    "validation_seed": 7777,
    "task_indices": {"start": 0, "stop_exclusive": 128},
    "tasks": 128,
    "k": 16,
    "attempts": 2048,
    "maximum_rollout_errors": 0,
    "adaptive_resampling": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_steps": 8,
    "contains_official_heldout": False,
    "contains_real_vm_eval": False,
    "contains_crowd_cast": False,
}
(out / "eval_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
