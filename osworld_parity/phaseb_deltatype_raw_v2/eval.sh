#!/bin/bash
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2 ;;
  esac
done
for key in model model_manifest model_stage dataset normalized_gold out runtime_dir experiment_dir dataset_kind shard_count shard_index; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done

MODEL="${A[model]}/hf"
MANIFEST="${A[model_manifest]}"
case "${A[dataset_kind]}" in
  val)
    RAW_GOLD="${A[dataset]}/val/chat.jsonl"
    NORMALIZED_GOLD="${A[normalized_gold]}/_normalized/val/chat.jsonl"
    ;;
  train)
    RAW_GOLD="${A[dataset]}/train/chat.jsonl"
    NORMALIZED_GOLD="${A[normalized_gold]}/_normalized/train/chat.jsonl"
    ;;
  *) echo "FATAL invalid --dataset_kind=${A[dataset_kind]}" >&2; exit 2 ;;
esac
OUT="${A[out]}"
[[ -f "$MODEL/config.json" && -f "$MANIFEST" && -f "$RAW_GOLD" \
    && -f "$NORMALIZED_GOLD" ]] || {
  echo "FATAL raw-v2 eval input missing" >&2; exit 3;
}
mkdir -p "$OUT"
PY="${A[runtime_dir]}/bin/python"
"$PY" "${A[experiment_dir]}/../phaseb_oracle_eval.py" \
  --schema raw --val-chat "$RAW_GOLD" --raw-gold "$RAW_GOLD" \
  --normalized-gold "$NORMALIZED_GOLD" --out "$OUT/selftest" --model-dir "$MODEL" \
  --model-manifest "$MANIFEST" --model-stage "${A[model_stage]}" --selftest \
  --dataset-kind "${A[dataset_kind]}"

cache="${VLLM_CACHE_ROOT:-/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp}/phaseb_raw_v2_vllm_${SLURM_JOB_ID:-local}"
export VLLM_CACHE_ROOT="$cache" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" \
  TRITON_CACHE_DIR="$cache/triton"
export CUDA_HOME="${CUDA_HOME:-/fast/service/apps/software/CUDA/12.6.0}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
port=$((20000 + (${SLURM_JOB_ID:-1} % 1000)))
url="http://127.0.0.1:${port}/v1"
cd "${A[runtime_dir]}"
setsid --wait "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port "$port" \
  --served-model-name policy --gpu-memory-utilization 0.85 --max-model-len 16384 --enforce-eager \
  >"$OUT/vllm.log" 2>&1 &
pid=$!
shutdown_vllm() {
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  ! kill -0 -- "-$pid" 2>/dev/null || {
    echo "FATAL vLLM process group survived shutdown" >&2
    return 1
  }
}
trap shutdown_vllm EXIT
"$PY" "${A[experiment_dir]}/../phaseb_relative/server_readiness.py" \
  --base-url "$url" --model policy --grammar deltatype_raw --timeout-s 900 --pid "$pid"
"$PY" "${A[experiment_dir]}/../phaseb_oracle_eval.py" \
  --schema raw --val-chat "$RAW_GOLD" --raw-gold "$RAW_GOLD" \
  --normalized-gold "$NORMALIZED_GOLD" --out "$OUT" --base-url "$url" --model policy \
  --model-dir "$MODEL" --model-manifest "$MANIFEST" --model-stage "${A[model_stage]}" \
  --selftest --concurrency 12 --dataset-kind "${A[dataset_kind]}" \
  --shard-count "${A[shard_count]}" --shard-index "${A[shard_index]}"
shutdown_vllm
trap - EXIT
