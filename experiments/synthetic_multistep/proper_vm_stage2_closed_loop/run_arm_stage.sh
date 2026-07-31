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
  arm grammar checkpoint output runtime_repo provider_source qcow qemu_bin
  osworld_root port_lock_dir host_python
)
for key in "${required[@]}"; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[arm]}:${A[grammar]}" in
  absolute_matched_control:absolute_toolcall|normalized_relative:move_rel|raw_relative:deltatype_raw) ;;
  *) echo "FATAL arm/grammar mismatch: ${A[arm]}:${A[grammar]}" >&2; exit 2 ;;
esac
MODE="${A[mode]:-full}"
case "$MODE" in full|one_cell_preflight|chunk0_pilot) ;; *) echo "FATAL invalid --mode: $MODE" >&2; exit 2 ;; esac

REPO="$(jq -r '.source_path' "$LABCTL_CONTEXT")"
OUT="${A[output]}"
if [[ "$MODE" == full ]]; then
  [[ ! -e "$OUT/arm_manifest.json" && ! -e "$OUT/rows.jsonl" ]] || {
    echo "FATAL refusing to overwrite trusted stage-2 output" >&2; exit 2;
  }
elif [[ "$MODE" == one_cell_preflight ]]; then
  [[ ! -e "$OUT/preflight_manifest.json" && ! -e "$OUT/one_cell_unit.json" ]] || {
    echo "FATAL refusing to overwrite trusted one-cell preflight output" >&2; exit 2;
  }
else
  [[ ! -e "$OUT/chunk0_pilot_manifest.json" && ! -e "$OUT/rows.jsonl" ]] || {
    echo "FATAL refusing to overwrite trusted chunk-0 pilot output" >&2; exit 2;
  }
fi
[[ -r /dev/kvm && -w /dev/kvm ]] || { echo "FATAL /dev/kvm unavailable" >&2; exit 2; }
[[ -x "${A[qemu_bin]}" && -f "${A[qcow]}" && -f "${A[provider_source]}" ]] || {
  echo "FATAL KVM substrate missing" >&2; exit 2;
}
[[ -f "${A[checkpoint]}/config.json" && -f "${A[checkpoint]}/model.safetensors" ]] || {
  echo "FATAL checkpoint HF directory incomplete" >&2; exit 2;
}

PRIME="${A[runtime_repo]}/prime-rl"
PY="$PRIME/.venv/bin/python"
HOST_PY="${A[host_python]}/bin/python"
[[ -x "$PY" && -x "$HOST_PY" ]] || { echo "FATAL pinned Python environment missing" >&2; exit 2; }

"$HOST_PY" - "$REPO" "${A[osworld_root]}" "$(dirname "${A[provider_source]}")" <<'PY'
import importlib, json, pathlib, sys
repo, osworld, provider = map(pathlib.Path, sys.argv[1:4])
sys.path[:0] = [str(repo), str(osworld), str(provider)]
names = (
    "gymnasium", "openai", "PIL", "desktop_env.desktop_env",
    "experiments.synthetic_multistep.proper_vm_stage2_closed_loop.runner",
)
modules = {name: importlib.import_module(name) for name in names}
desktop_path = pathlib.Path(modules["desktop_env.desktop_env"].__file__).resolve()
runner_path = pathlib.Path(
    modules["experiments.synthetic_multistep.proper_vm_stage2_closed_loop.runner"].__file__
).resolve()
if not desktop_path.is_relative_to(osworld.resolve()):
    raise SystemExit(f"FATAL desktop_env loaded outside pinned OSWorld: {desktop_path}")
expected = (
    repo / "experiments/synthetic_multistep/proper_vm_stage2_closed_loop/runner.py"
).resolve()
if runner_path != expected:
    raise SystemExit(f"FATAL runner path drift: {runner_path} != {expected}")
print(json.dumps({
    "status": "pinned_stage2_host_imports_pass",
    "python": sys.executable,
    "gymnasium": modules["gymnasium"].__version__,
    "openai": modules["openai"].__version__,
    "PIL": modules["PIL"].__version__,
    "desktop_env": str(desktop_path),
    "runner": str(runner_path),
}, sort_keys=True))
PY

runner=(
  "$HOST_PY" "$REPO/experiments/synthetic_multistep/proper_vm_stage2_closed_loop/runner.py"
  --arm "${A[arm]}"
  --model-dir "${A[checkpoint]}"
  --served-model policy
  --out "$OUT"
  --provider-source "${A[provider_source]}"
  --qcow "${A[qcow]}"
  --qemu-bin "${A[qemu_bin]}"
  --osworld-root "${A[osworld_root]}"
  --port-lock-dir "${A[port_lock_dir]}"
  --launch-scope "$MODE"
)
if [[ -n "${A[resume_from]:-}" ]]; then
  [[ -d "${A[resume_from]}" ]] || { echo "FATAL resume source missing" >&2; exit 2; }
  runner+=(--resume-from "${A[resume_from]}")
fi
"${runner[@]}" --preflight-only --base-url "http://127.0.0.1:1/v1"

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#visible_gpus[@]}" -eq 1 && -n "${visible_gpus[0]}" ]] || {
  echo "FATAL expected exactly one allocated GPU" >&2; exit 2;
}
nvidia-smi -i "${visible_gpus[0]}" --query-gpu=uuid,name,memory.total --format=csv,noheader
mkdir -p "$OUT" "$OUT/vm_logs"

JOB_TMP="$(mktemp -d "/tmp/pvm_stage2_${SLURM_JOB_ID:-local}_XXXXXX")"
case "$JOB_TMP" in /tmp/pvm_stage2_*) ;; *) echo "FATAL unsafe temp path" >&2; exit 2 ;; esac
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  case "$JOB_TMP" in /tmp/pvm_stage2_*) rm -rf -- "$JOB_TMP" ;; esac
}
trap cleanup EXIT
trap 'exit 143' TERM INT

port=$((20000 + (${SLURM_JOB_ID:-1} % 1000)))
lock_dir="/tmp/proper_vm_roadmap_stage2_vllm_locks"
mkdir -m 700 -p "$lock_dir"
[[ -d "$lock_dir" && ! -L "$lock_dir" && -O "$lock_dir" ]] || {
  echo "FATAL unsafe vLLM lock directory" >&2; exit 2;
}
exec {vllm_lock_fd}>"$lock_dir/port_${port}.lock"
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

if [[ "$MODE" == one_cell_preflight ]]; then
  "${runner[@]}" --base-url "$base_url" --one-cell-preflight
elif [[ "$MODE" == chunk0_pilot ]]; then
  "${runner[@]}" --base-url "$base_url" --chunk0-pilot
else
  "${runner[@]}" --base-url "$base_url"
fi
