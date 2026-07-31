#!/bin/bash
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected positional argument: $arg" >&2; exit 2 ;;
  esac
done
required=(
  arm grammar chunk_index chunk_start chunk_stop checkpoint output live_smoke_manifest runtime_repo
  provider_source qcow qemu_bin osworld_root port_lock_dir host_python
)
for key in "${required[@]}"; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[arm]}:${A[grammar]}" in
  absolute_phase_a:absolute_toolcall|normalized_phase_a:move_rel|raw_a_to_b:deltatype_raw) ;;
  *) echo "FATAL arm/grammar mismatch: ${A[arm]}:${A[grammar]}" >&2; exit 2 ;;
esac

REPO="$(jq -r '.source_path' "$LABCTL_CONTEXT")"
OUT="${A[output]}"
[[ ! -e "$OUT/arm_manifest.json" ]] || { echo "FATAL refusing to overwrite arm marker" >&2; exit 2; }
[[ -r /dev/kvm && -w /dev/kvm ]] || { echo "FATAL /dev/kvm unavailable" >&2; exit 2; }
[[ -x "${A[qemu_bin]}" && -f "${A[qcow]}" && -f "${A[provider_source]}" ]] || {
  echo "FATAL KVM substrate missing" >&2; exit 2;
}
[[ -f "${A[checkpoint]}/config.json" && -f "${A[checkpoint]}/model.safetensors" ]] || {
  echo "FATAL checkpoint HF directory incomplete" >&2; exit 2;
}
[[ -f "${A[live_smoke_manifest]}" ]] || { echo "FATAL live-smoke evidence missing" >&2; exit 2; }

PRIME="${A[runtime_repo]}/prime-rl"
PY="$PRIME/.venv/bin/python"
[[ -x "$PY" ]] || { echo "FATAL prime-rl Python missing" >&2; exit 2; }
HOST_PY="${A[host_python]}/bin/python"
[[ -x "$HOST_PY" ]] || { echo "FATAL pinned OSWorld Python missing" >&2; exit 2; }
"$HOST_PY" - "$REPO" "${A[osworld_root]}" "$(dirname "${A[provider_source]}")" <<'PY'
import importlib, json, pathlib, sys
repo, osworld, provider = map(pathlib.Path, sys.argv[1:4])
sys.path[:0] = [str(repo), str(osworld), str(provider)]
modules = {
    name: importlib.import_module(name)
    for name in (
        "gymnasium", "openai", "PIL", "desktop_env.desktop_env",
        "experiments.synthetic_multistep.proper_vm_stage2.run_arm",
    )
}
desktop_path = pathlib.Path(modules["desktop_env.desktop_env"].__file__).resolve()
runner_path = pathlib.Path(
    modules["experiments.synthetic_multistep.proper_vm_stage2.run_arm"].__file__
).resolve()
if not desktop_path.is_relative_to(osworld.resolve()):
    raise SystemExit(f"FATAL desktop_env loaded outside pinned OSWorld: {desktop_path}")
expected_runner = (
    repo / "experiments/synthetic_multistep/proper_vm_stage2/run_arm.py"
).resolve()
if runner_path != expected_runner:
    raise SystemExit(f"FATAL runner module path drift: {runner_path} != {expected_runner}")
print(json.dumps({
    "status": "pinned_osworld_imports_pass",
    "python": sys.executable,
    "gymnasium": modules["gymnasium"].__version__,
    "openai": modules["openai"].__version__,
    "PIL": modules["PIL"].__version__,
    "desktop_env": str(desktop_path),
    "runner": str(runner_path),
}, sort_keys=True))
PY
"$HOST_PY" "$REPO/experiments/synthetic_multistep/proper_vm_stage2/run_arm.py" \
  --preflight-only \
  --arm "${A[arm]}" \
  --chunk-index "${A[chunk_index]}" \
  --chunk-start "${A[chunk_start]}" \
  --chunk-stop "${A[chunk_stop]}" \
  --model-dir "${A[checkpoint]}" \
  --base-url "http://127.0.0.1:1/v1" \
  --served-model policy \
  --out "$OUT" \
  --live-smoke-manifest "${A[live_smoke_manifest]}" \
  --provider-source "${A[provider_source]}" \
  --qcow "${A[qcow]}" \
  --qemu-bin "${A[qemu_bin]}" \
  --osworld-root "${A[osworld_root]}" \
  --port-lock-dir "${A[port_lock_dir]}"

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#visible_gpus[@]}" -eq 1 && -n "${visible_gpus[0]}" ]] || {
  echo "FATAL expected exactly one allocated GPU" >&2; exit 2;
}
nvidia-smi -i "${visible_gpus[0]}" \
  --query-gpu=uuid,name,memory.total --format=csv,noheader
mkdir -p "$OUT" "$OUT/vm_logs"

JOB_TMP="$(mktemp -d "/tmp/pvm_${SLURM_JOB_ID:-local}_XXXXXX")"
case "$JOB_TMP" in /tmp/pvm_*) ;; *) echo "FATAL unsafe temp path" >&2; exit 2 ;; esac
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  case "$JOB_TMP" in /tmp/pvm_*) rm -rf -- "$JOB_TMP" ;; esac
}
trap cleanup EXIT
trap 'exit 143' TERM INT

port=$((19000 + (${SLURM_JOB_ID:-1} % 1000)))
vllm_lock_dir="/tmp/proper_vm_stage2_vllm_locks"
mkdir -m 700 -p "$vllm_lock_dir"
[[ -d "$vllm_lock_dir" && ! -L "$vllm_lock_dir" && -O "$vllm_lock_dir" ]] || {
  echo "FATAL unsafe vLLM lock directory" >&2; exit 2;
}
exec {vllm_lock_fd}>"$vllm_lock_dir/port_${port}.lock"
flock -n "$vllm_lock_fd" || { echo "FATAL vLLM port already leased: $port" >&2; exit 2; }
base_url="http://127.0.0.1:${port}/v1"

export OSWORLD_USE_KVM_PROVIDER=1
export OSWORLD_QEMU_BIN="${A[qemu_bin]}"
export OSWORLD_QCOW2="${A[qcow]}"
export OSWORLD_VM_LOG_DIR="$OUT/vm_logs"
export TMPDIR="$JOB_TMP"
export PYTHONPATH="$(dirname "${A[provider_source]}"):${A[runtime_repo]}:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export VLLM_CACHE_ROOT="$OUT/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/vllm_cache/inductor"
export TRITON_CACHE_DIR="$OUT/vllm_cache/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$PRIME"
uv run --no-sync vllm serve "${A[checkpoint]}" \
  --host 127.0.0.1 --port "$port" --served-model-name policy \
  --gpu-memory-utilization 0.85 --max-model-len 4096 --enforce-eager \
  >"$OUT/vllm.log" 2>&1 &
server_pid=$!

"$PY" "$REPO/experiments/relative_factorial/readiness.py" \
  --base-url "$base_url" --model policy --grammar "${A[grammar]}" \
  --timeout-s 900 --pid "$server_pid"

"$HOST_PY" "$REPO/experiments/synthetic_multistep/proper_vm_stage2/run_arm.py" \
  --arm "${A[arm]}" \
  --chunk-index "${A[chunk_index]}" \
  --chunk-start "${A[chunk_start]}" \
  --chunk-stop "${A[chunk_stop]}" \
  --model-dir "${A[checkpoint]}" \
  --base-url "$base_url" \
  --served-model policy \
  --out "$OUT" \
  --live-smoke-manifest "${A[live_smoke_manifest]}" \
  --provider-source "${A[provider_source]}" \
  --qcow "${A[qcow]}" \
  --qemu-bin "${A[qemu_bin]}" \
  --osworld-root "${A[osworld_root]}" \
  --port-lock-dir "${A[port_lock_dir]}"
