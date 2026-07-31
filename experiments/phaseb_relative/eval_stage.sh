#!/bin/bash
set -euo pipefail
declare -A A
for arg in "$@"; do case "$arg" in --*=*) k="${arg%%=*}"; k="${k#--}"; A[$k]="${arg#*=}";;
  *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac; done
for k in model dataset out runtime_dir experiment_dir; do [[ -n "${A[$k]:-}" ]] || exit 2; done
MODEL="${A[model]}/hf"; EXPORT="${A[model]}/export_manifest.json"
VAL="${A[dataset]}/prose_keep/_normalized/val/chat.jsonl"; OUT="${A[out]}"
[[ -f "$MODEL/config.json" && -f "$EXPORT" && -f "$VAL" ]] || exit 3
mkdir -p "$OUT"
PY="${A[runtime_dir]}/.venv/bin/python"
"$PY" "${A[experiment_dir]}/relative_eval.py" --val-chat "$VAL" --out "$OUT/selftest" --selftest
cache="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp/phaseb_relative_vllm_${SLURM_JOB_ID:-local}"
export VLLM_CACHE_ROOT="$cache" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
port=$((20000 + (${SLURM_JOB_ID:-1} % 1000))); url="http://127.0.0.1:${port}/v1"
cd "${A[runtime_dir]}"
uv run --no-sync vllm serve "$MODEL" --host 127.0.0.1 --port "$port" \
 --served-model-name policy --gpu-memory-utilization 0.85 --max-model-len 16384 --enforce-eager \
 >"$OUT/vllm.log" 2>&1 &
pid=$!; trap 'kill -9 "$pid" 2>/dev/null || true' EXIT
"$PY" "${A[experiment_dir]}/../relative_factorial/readiness.py" \
 --base-url "$url" --model policy --grammar move_rel --timeout-s 900 --pid "$pid"
"$PY" "${A[experiment_dir]}/relative_eval.py" --val-chat "$VAL" --out "$OUT" \
 --base-url "$url" --model policy --model-dir "$MODEL" --export-manifest "$EXPORT" \
 --selftest --concurrency 12

