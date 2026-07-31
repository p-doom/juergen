#!/bin/bash
# Evaluate one Phase-B arm on its own held-in validation prompts.
set -euo pipefail

declare -A A
for arg in "$@"; do
    case "$arg" in
        --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
        *) echo "FATAL unexpected positional argument: $arg" >&2; exit 2 ;;
    esac
done
for key in arm source_job_id source_checkpoint_root model_path val_chat \
           training_log training_script audit_dir runtime_dir experiment_dir out; do
    [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done

case "${A[arm]}" in
    prose_keep)
        expected_job=135312
        expected_checkpoint=pb_prose_keep_r32
        expected_val=prose_keep
        ;;
    prose_strip)
        expected_job=135313
        expected_checkpoint=pb_prose_strip_r32
        expected_val=prose_strip
        ;;
    *) echo "FATAL unsupported Phase-B arm: ${A[arm]}" >&2; exit 2 ;;
esac

[[ "${A[source_job_id]}" == "$expected_job" ]] || {
    echo "FATAL ${A[arm]} must use source job $expected_job" >&2; exit 2;
}
[[ "$(basename "${A[source_checkpoint_root]}")" == "$expected_checkpoint" ]] || {
    echo "FATAL checkpoint/arm mismatch: ${A[source_checkpoint_root]}" >&2; exit 2;
}
[[ "${A[model_path]}" == "${A[source_checkpoint_root]}_hf" ]] || {
    echo "FATAL model path is not the exporter-defined checkpoint sibling: ${A[model_path]}" >&2
    exit 2
}
expected_val_path="${A[audit_dir]}/phaseb/${expected_val}/_normalized/val/chat.jsonl"
[[ "${A[val_chat]}" == "$expected_val_path" ]] || {
    echo "FATAL cross-arm or noncanonical validation prompts: ${A[val_chat]}" >&2
    exit 2
}
[[ "${A[training_log]}" == "${A[audit_dir]}/logs/r3sft_${expected_job}.log" ]] || {
    echo "FATAL training log/job mismatch: ${A[training_log]}" >&2; exit 2;
}
[[ "${A[training_script]}" == "${A[audit_dir]}/rung3_sft.sbatch" ]] || {
    echo "FATAL unexpected training script: ${A[training_script]}" >&2; exit 2;
}

source_state="$(sacct -X -n -P -j "${A[source_job_id]}" -o State | head -1 | cut -d'|' -f1)"
[[ "$source_state" == COMPLETED* ]] || {
    echo "FATAL source job ${A[source_job_id]} is not COMPLETED: $source_state" >&2; exit 2;
}

source_checkpoint="${A[source_checkpoint_root]}/000900"
[[ -f "$source_checkpoint/_CHECKPOINT_METADATA" ]] || {
    echo "FATAL final Orbax checkpoint is incomplete: $source_checkpoint" >&2; exit 2;
}
[[ -f "${A[source_checkpoint_root]}/lora_metadata.json" ]] || {
    echo "FATAL LoRA metadata missing: ${A[source_checkpoint_root]}" >&2; exit 2;
}
[[ -f "${A[training_log]}" ]] || { echo "FATAL training log missing" >&2; exit 2; }
grep -Fq "=== R3 SFT DONE (${expected_checkpoint}, rank=32, steps=900)" "${A[training_log]}" || {
    echo "FATAL training completion line missing or mismatched" >&2; exit 2;
}
grep -Fq "=== R3 EXPORT DONE -> ${A[model_path]} ===" "${A[training_log]}" || {
    echo "FATAL export completion line missing or mismatched" >&2; exit 2;
}

python3 - "${A[model_path]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = root / "config.json"
if not config_path.is_file():
    raise SystemExit(f"FATAL export config missing: {config_path}")
config = json.loads(config_path.read_text())
if not config.get("architectures"):
    raise SystemExit("FATAL export config lacks architectures; vLLM may start in pooling mode")
weights = sorted(root.glob("*.safetensors"))
if not weights:
    raise SystemExit(f"FATAL no safetensor weights in {root}")
for path in weights:
    if path.stat().st_size <= 0:
        raise SystemExit(f"FATAL empty weight file: {path}")
for required in ("tokenizer_config.json", "preprocessor_config.json"):
    if not (root / required).is_file():
        raise SystemExit(f"FATAL exported runtime file missing: {root / required}")
PY

OUT="${A[out]}"
mkdir -p "$OUT"
rm -f "$OUT/eval_manifest.json" "$OUT/report.json"

PY="${A[runtime_dir]}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "FATAL prime-rl Python missing: $PY" >&2; exit 2; }

# Validate the scorer against the known-answer teacher outputs before loading a model.
"$PY" "${A[audit_dir]}/phaseb_eval.py" \
    --val_chat "${A[val_chat]}" --out "$OUT/selftest" --selftest \
    --tag "phaseb/${A[arm]}/selftest"

cache="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp/phaseb_vllm_${SLURM_JOB_ID:-local}"
export VLLM_CACHE_ROOT="$cache"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
export TRITON_CACHE_DIR="$cache/triton"
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

port=$((19000 + (${SLURM_JOB_ID:-1} % 1000)))
base_url="http://127.0.0.1:${port}/v1"
cd "${A[runtime_dir]}"
uv run --no-sync vllm serve "${A[model_path]}" \
    --host 127.0.0.1 --port "$port" --served-model-name policy \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --enforce-eager \
    >"$OUT/vllm.log" 2>&1 &
server_pid=$!
trap 'kill -9 "$server_pid" 2>/dev/null || true' EXIT

"$PY" "${A[experiment_dir]}/../relative_factorial/readiness.py" \
    --base-url "$base_url" --model policy --grammar absolute_toolcall \
    --timeout-s 900 --pid "$server_pid"

"$PY" "${A[audit_dir]}/phaseb_eval.py" \
    --val_chat "${A[val_chat]}" --base_url "$base_url" --model policy \
    --out "$OUT" --max_tokens 256 --concurrency 12 --selftest \
    --tag "phaseb/${A[arm]}"

"$PY" "${A[experiment_dir]}/finalize.py" \
    --arm "${A[arm]}" --source-job-id "${A[source_job_id]}" \
    --source-checkpoint "$source_checkpoint" \
    --source-checkpoint-root "${A[source_checkpoint_root]}" \
    --model-dir "${A[model_path]}" --val-chat "${A[val_chat]}" \
    --training-log "${A[training_log]}" --training-script "${A[training_script]}" \
    --evaluator "${A[audit_dir]}/phaseb_eval.py" --out "$OUT"

echo "Phase-B ${A[arm]} own-val evaluation complete: $OUT"
