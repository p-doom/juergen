#!/usr/bin/env bash
set -euo pipefail

SOURCE="$(jq -r '.source_path' "$LABCTL_CONTEXT")"
OUTPUT="$(jq -r '.outputs.result.path' "$LABCTL_CONTEXT")"
input_path() { jq -er --arg role "$1" '.inputs[]|select(.role==$role)|.resolved_path' "$LABCTL_CONTEXT"; }
arg() { jq -er --arg key "$1" '.args[$key]' "$LABCTL_CONTEXT"; }

EXECUTOR_READY_ROOT="$(input_path executor_readiness)"
TASK_SETUP_ROOT="$(input_path task_setup_validation)"
NATIVE_CHECKPOINT="$(arg native_checkpoint)"
COMPACT_CHECKPOINT="$(arg compact_checkpoint)"
NATIVE_SHA256="$(arg native_checkpoint_sha256)"
COMPACT_SHA256="$(arg compact_checkpoint_sha256)"
NATIVE_ROOT="$(arg native_checkpoint_root)"
COMPACT_ROOT="$(arg compact_checkpoint_root)"
NATIVE_ARTIFACT_ID="$(arg native_checkpoint_artifact_id)"
COMPACT_ARTIFACT_ID="$(arg compact_checkpoint_artifact_id)"
NATIVE_ALIAS="$(arg native_checkpoint_alias)"
COMPACT_ALIAS="$(arg compact_checkpoint_alias)"
NATIVE_MANIFEST_NAME="$(arg native_checkpoint_manifest_name)"
COMPACT_MANIFEST_NAME="$(arg compact_checkpoint_manifest_name)"
NATIVE_MANIFEST_SHA256="$(arg native_checkpoint_manifest_sha256)"
COMPACT_MANIFEST_SHA256="$(arg compact_checkpoint_manifest_sha256)"
SHARD_INDEX="$(arg shard_index)"
SHARD_COUNT="$(arg shard_count)"

test "$SHARD_COUNT" = 5
case "$SHARD_INDEX" in 0|1|2|3|4) ;; *) echo "FATAL invalid shard index" >&2; exit 2 ;; esac
test -r /dev/kvm && test -w /dev/kvm
test "${SLURM_NTASKS:-1}" = 1
test -f "$EXECUTOR_READY_ROOT/EXECUTOR_READY.json"
test -f "$TASK_SETUP_ROOT/task_setup_validation.json"
test -d "$NATIVE_CHECKPOINT" && test -d "$COMPACT_CHECKPOINT"
test "$NATIVE_CHECKPOINT" != "$COMPACT_CHECKPOINT"

verify_checkpoint() {
  local checkpoint="$1" expected_weights="$2" root="$3" artifact_id="$4"
  local alias="$5" manifest_name="$6" expected_manifest="$7" observed
  test "$checkpoint" = "$root/hf"
  test "$(jq -er '.id' "$root/.meta.json")" = "$artifact_id"
  test "$(jq -er '.alias' "$root/.meta.json")" = "$alias"
  test -f "$root/$manifest_name" && test -f "$checkpoint/model.safetensors"
  observed="$(sha256sum "$root/$manifest_name")"; observed="${observed%% *}"
  test "$observed" = "$expected_manifest"
  observed="$(sha256sum "$checkpoint/model.safetensors")"; observed="${observed%% *}"
  test "$observed" = "$expected_weights"
}
verify_checkpoint "$NATIVE_CHECKPOINT" "$NATIVE_SHA256" "$NATIVE_ROOT" \
  "$NATIVE_ARTIFACT_ID" "$NATIVE_ALIAS" "$NATIVE_MANIFEST_NAME" "$NATIVE_MANIFEST_SHA256"
verify_checkpoint "$COMPACT_CHECKPOINT" "$COMPACT_SHA256" "$COMPACT_ROOT" \
  "$COMPACT_ARTIFACT_ID" "$COMPACT_ALIAS" "$COMPACT_MANIFEST_NAME" "$COMPACT_MANIFEST_SHA256"

IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
test "${#allocated_gpus[@]}" = 2
NATIVE_PORT=$((31000 + SHARD_INDEX))
COMPACT_PORT=$((32000 + SHARD_INDEX))
NATIVE_KEY="paired-native-${SLURM_JOB_ID:-local}-${SHARD_INDEX}"
COMPACT_KEY="paired-compact-${SLURM_JOB_ID:-local}-${SHARD_INDEX}"
mkdir -p "$OUTPUT/model_logs" "$OUTPUT/vm_sessions"

cleanup() {
  status=$?
  test -z "${native_pid:-}" || kill "$native_pid" 2>/dev/null || true
  test -z "${compact_pid:-}" || kill "$compact_pid" 2>/dev/null || true
  for child in $(jobs -pr); do
    wait "$child" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

env CUDA_VISIBLE_DEVICES="${allocated_gpus[0]}" uv run --project "$SOURCE/eval" --locked --no-dev \
  python -m sglang.launch_server --model-path "$NATIVE_CHECKPOINT" \
  --served-model-name native-policy --port "$NATIVE_PORT" --api-key "$NATIVE_KEY" \
  --mem-fraction-static 0.85 --chunked-prefill-size 2048 \
  >"$OUTPUT/model_logs/native.log" 2>&1 &
native_pid=$!
env CUDA_VISIBLE_DEVICES="${allocated_gpus[1]}" uv run --project "$SOURCE/eval" --locked --no-dev \
  python -m sglang.launch_server --model-path "$COMPACT_CHECKPOINT" \
  --served-model-name compact-policy --port "$COMPACT_PORT" --api-key "$COMPACT_KEY" \
  --mem-fraction-static 0.85 --chunked-prefill-size 2048 \
  >"$OUTPUT/model_logs/compact.log" 2>&1 &
compact_pid=$!

for endpoint in "http://127.0.0.1:$NATIVE_PORT/v1/models:$NATIVE_KEY" "http://127.0.0.1:$COMPACT_PORT/v1/models:$COMPACT_KEY"; do
  url="${endpoint%:*}"; key="${endpoint##*:}"
  ready=false
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 5 -H "Authorization: Bearer $key" "$url" >/dev/null; then ready=true; break; fi
    sleep 2
  done
  test "$ready" = true
done

export PAIRED_ENDPOINT_OUTPUT="$OUTPUT/model_endpoints.json"
export PAIRED_NATIVE_PORT="$NATIVE_PORT" PAIRED_COMPACT_PORT="$COMPACT_PORT"
export PAIRED_NATIVE_KEY="$NATIVE_KEY" PAIRED_COMPACT_KEY="$COMPACT_KEY"
export PAIRED_NATIVE_CHECKPOINT="$NATIVE_CHECKPOINT" PAIRED_COMPACT_CHECKPOINT="$COMPACT_CHECKPOINT"
export PAIRED_NATIVE_SHA256="$NATIVE_SHA256" PAIRED_COMPACT_SHA256="$COMPACT_SHA256"
python - <<'PY'
import json, os
value = {
    "native_absolute_control": {
        "base_url": f"http://127.0.0.1:{os.environ['PAIRED_NATIVE_PORT']}/v1",
        "api_key": os.environ["PAIRED_NATIVE_KEY"],
        "served_model": "native-policy",
        "checkpoint": os.environ["PAIRED_NATIVE_CHECKPOINT"],
        "checkpoint_sha256": os.environ["PAIRED_NATIVE_SHA256"],
    },
    "compact_raw_phaseb": {
        "base_url": f"http://127.0.0.1:{os.environ['PAIRED_COMPACT_PORT']}/v1",
        "api_key": os.environ["PAIRED_COMPACT_KEY"],
        "served_model": "compact-policy",
        "checkpoint": os.environ["PAIRED_COMPACT_CHECKPOINT"],
        "checkpoint_sha256": os.environ["PAIRED_COMPACT_SHA256"],
    },
}
with open(os.environ["PAIRED_ENDPOINT_OUTPUT"], "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

# The SGLang children retain their one-GPU environments. The evaluator/KVM
# parent must not expose either GPU to the executor process.
export CUDA_VISIBLE_DEVICES=""
test -z "$CUDA_VISIBLE_DEVICES"
export PAIRED_MODEL_ENDPOINTS_JSON="$OUTPUT/model_endpoints.json"
export PAIRED_VM_QCOW="$(arg qcow)"
export PAIRED_VM_QEMU="$(arg qemu)"
export PAIRED_VM_PROVIDER="$(arg provider)"
export PAIRED_VM_PROVIDER_SHA256="$(arg provider_sha256)"
export PAIRED_VM_SESSION_ROOT="$OUTPUT/vm_sessions"

cd "$SOURCE"
python -m osworld_parity.proper_vm_capability_ladder.paired_eval run \
  --evaluation-manifest="$SOURCE/osworld_parity/proper_vm_capability_ladder/paired_runtime/config/short_task_passk.template.json" \
  --task-manifest="$SOURCE/osworld_parity/proper_vm_capability_ladder/rung2_sameapp/curriculum/manifests/development.json" \
  --executor-ready="$EXECUTOR_READY_ROOT/EXECUTOR_READY.json" \
  --task-setup-validation="$TASK_SETUP_ROOT/task_setup_validation.json" \
  --shard-index="$SHARD_INDEX" --shard-count="$SHARD_COUNT" \
  --output="$OUTPUT/results.jsonl"

test -s "$OUTPUT/results.jsonl"
