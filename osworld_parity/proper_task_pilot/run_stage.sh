#!/bin/bash
# Serve one pinned checkpoint, preflight native KVM, and run one proper-task arm.
set -euo pipefail

declare -A A
for arg in "$@"; do
    case "$arg" in
        --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
        *) echo "FATAL unexpected positional argument: $arg" >&2; exit 2 ;;
    esac
done
required=(
    mode arm action_format checkpoint checkpoint_manifest
    checkpoint_manifest_sha256 expected_lora_rank expected_lora_alpha
    runtime_repo runtime_files_sha256 tasks_file tasks_sha256 canonical_tasks_file reverse_tasks
    task_base train_split train_split_sha256 heldout_split heldout_split_sha256
    output qcow_path qemu_bin provider_source provider_sha256
    port_lock_dir port_base
    osworld_root host_python apptainer_image screen_width screen_height
    snapshot_name max_steps n_history_frames pause server_max_model_len
    max_completion_tokens temperature
)
for key in "${required[@]}"; do
    [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
[[ "${A[screen_width]}x${A[screen_height]}" == "1920x1080" ]] || {
    echo "FATAL screen must be pinned to 1920x1080" >&2; exit 2;
}
[[ "${A[snapshot_name]}" == "osworld_ready" ]] || {
    echo "FATAL snapshot name drift" >&2; exit 2;
}
[[ "${A[server_max_model_len]}" == "16384" ]] || {
    echo "FATAL server_max_model_len must be 16384" >&2; exit 2;
}
[[ "${A[port_lock_dir]}" == "/tmp/osworld_port_locks" ]] || {
    echo "FATAL OSWorld port lock directory drift" >&2; exit 2;
}
[[ "${A[port_base]}" == "30000" ]] || {
    echo "FATAL OSWorld port base drift" >&2; exit 2;
}

REPO="$(jq -r '.source_path' "$LABCTL_CONTEXT")"
OUT="${A[output]}"
TASKS_FILE="$REPO/osworld_parity/proper_task_pilot/${A[tasks_file]}"
CANONICAL_TASKS_FILE="$REPO/osworld_parity/proper_task_pilot/${A[canonical_tasks_file]}"
TRAIN_SPLIT="$REPO/osworld_parity/split/${A[train_split]}"
HELDOUT_SPLIT="$REPO/osworld_parity/split/${A[heldout_split]}"
[[ -f "$TASKS_FILE" && -f "$CANONICAL_TASKS_FILE" ]] || {
    echo "FATAL source-snapshotted task inputs are missing" >&2; exit 2;
}
mkdir -p "$OUT" "$OUT/vm_logs"
rm -f "$OUT/run_manifest.json"

JOB_TMP="$(mktemp -d "/tmp/ptp_${SLURM_JOB_ID:-local}_XXXXXX")"
case "$JOB_TMP" in
    /tmp/ptp_*) ;;
    *) echo "FATAL unsafe mktemp result: $JOB_TMP" >&2; exit 2 ;;
esac
(( ${#JOB_TMP} + 1 + 36 < 107 )) || {
    echo "FATAL TMPDIR too long for vLLM ZMQ IPC: $JOB_TMP" >&2; exit 2;
}
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    case "$JOB_TMP" in
        /tmp/ptp_*) rm -rf -- "$JOB_TMP" ;;
    esac
}
trap cleanup EXIT
trap 'exit 143' TERM INT

[[ -r /dev/kvm && -w /dev/kvm ]] || { echo "FATAL /dev/kvm unavailable" >&2; exit 2; }
[[ -x "${A[qemu_bin]}" ]] || { echo "FATAL qemu missing: ${A[qemu_bin]}" >&2; exit 2; }
[[ -f "${A[qcow_path]}" ]] || { echo "FATAL qcow missing: ${A[qcow_path]}" >&2; exit 2; }
[[ -x "${A[host_python]}" ]] || { echo "FATAL host Python missing" >&2; exit 2; }
[[ -f "${A[apptainer_image]}" ]] || { echo "FATAL Apptainer image missing" >&2; exit 2; }
grep -q 'snapshot=on' "${A[provider_source]}" || {
    echo "FATAL provider is not snapshot=on" >&2; exit 2;
}
grep -q '"-enable-kvm"' "${A[provider_source]}" || {
    echo "FATAL provider is not KVM-pinned" >&2; exit 2;
}
[[ "$(sha256sum "${A[provider_source]}" | awk '{print $1}')" == \
    "${A[provider_sha256]}" ]] || {
    echo "FATAL provider SHA-256 mismatch" >&2; exit 2;
}
! grep -q '_free_port' "${A[provider_source]}" || {
    echo "FATAL provider contains racy free-port selection" >&2; exit 2;
}
for variable in \
    OSWORLD_APPTAINER_SERVER_PORT OSWORLD_APPTAINER_CHROMIUM_PORT \
    OSWORLD_APPTAINER_VNC_PORT OSWORLD_APPTAINER_VLC_PORT; do
    grep -q "$variable" "${A[provider_source]}" || {
        echo "FATAL provider does not consume $variable" >&2; exit 2;
    }
done
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#visible_gpus[@]}" -eq 1 && -n "${visible_gpus[0]}" ]] || {
    echo "FATAL expected one CUDA_VISIBLE_DEVICES allocation" >&2; exit 2;
}
nvidia-smi -i "${visible_gpus[0]}" \
    --query-gpu=uuid,name,memory.total --format=csv,noheader

export OSWORLD_USE_KVM_PROVIDER=1
export OSWORLD_HOST_PYTHON="${A[host_python]}"
export OSWORLD_ROOT="${A[osworld_root]}"
export OSWORLD_QCOW_PATH="${A[qcow_path]}"
export OSWORLD_QCOW2="${A[qcow_path]}"
export OSWORLD_QEMU_BIN="${A[qemu_bin]}"
export OSWORLD_VM_LOG_DIR="$OUT/vm_logs"
export OSWORLD_PORT_BASE="${A[port_base]}"
export SCRATCH="$OUT/runtime_scratch"
export TMPDIR="$JOB_TMP"
export PYTHONPATH="$(dirname "${A[provider_source]}"):${A[runtime_repo]}:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
cache="$OUT/vllm_cache"
export VLLM_CACHE_ROOT="$cache"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
export TRITON_CACHE_DIR="$cache/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$SCRATCH"
mkdir -m 700 -p "${A[port_lock_dir]}"
[[ -d "${A[port_lock_dir]}" && ! -L "${A[port_lock_dir]}" \
    && -O "${A[port_lock_dir]}" && -w "${A[port_lock_dir]}" ]] || {
    echo "FATAL shared OSWorld port lock directory is unusable" >&2; exit 2;
}

PRIME="${A[runtime_repo]}/prime-rl"
PY="$PRIME/.venv/bin/python"
[[ -x "$PY" ]] || { echo "FATAL prime-rl Python missing: $PY" >&2; exit 2; }
port=$((18000 + (${SLURM_JOB_ID:-1} % 1000)))
vllm_lock_dir="/tmp/proper_task_vllm_port_locks_961800067"
mkdir -m 700 -p "$vllm_lock_dir"
[[ -d "$vllm_lock_dir" && ! -L "$vllm_lock_dir" \
    && -O "$vllm_lock_dir" && -w "$vllm_lock_dir" ]] || {
    echo "FATAL shared vLLM port lock directory is unusable" >&2; exit 2;
}
exec {vllm_port_lock_fd}>"$vllm_lock_dir/port_${port}.lock"
flock -n "$vllm_port_lock_fd" || {
    echo "FATAL vLLM port $port is already reserved on $(hostname)" >&2; exit 2;
}
base_url="http://127.0.0.1:${port}/v1"
cd "$PRIME"
uv run --no-sync vllm serve "${A[checkpoint]}" \
    --host 127.0.0.1 --port "$port" --served-model-name policy \
    --gpu-memory-utilization 0.85 --max-model-len "${A[server_max_model_len]}" --enforce-eager \
    >"$OUT/vllm.log" 2>&1 &
server_pid=$!

grammar=move_rel
[[ "${A[action_format]}" == "absolute" ]] && grammar=absolute_toolcall
"$PY" "$REPO/experiments/relative_factorial/readiness.py" \
    --base-url "$base_url" --model policy --grammar "$grammar" \
    --timeout-s 900 --pid "$server_pid"

"$PY" "$REPO/osworld_parity/proper_task_pilot/run.py" "$@" \
    --tasks_file="$TASKS_FILE" --canonical_tasks_file="$CANONICAL_TASKS_FILE" \
    --train_split="$TRAIN_SPLIT" --heldout_split="$HELDOUT_SPLIT" \
    --base_url="$base_url" --model=policy --api_key=x
