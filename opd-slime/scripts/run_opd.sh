#!/usr/bin/env bash
# OPD training launcher. Invoked from the labctl OPD-train recipe's bash;
# expects the slime-stack env to be available at $ENV_ROOT and reads the
# remaining knobs from CLI args.
#
# Usage: bash run_opd.sh \
#   --env-root <path>          # the slime-stack environment artifact
#   --student-hf <path>        # HF dir of the BC student checkpoint
#   --prompt-data <path>       # JSONL with {"prompt": "..."} per line
#   --save-dir <path>          # output checkpoint_stream root (HF dirs land at $save-dir/rollout_{N})
#   --num-rollout <int>
#   --rollout-batch-size <int>
#   --n-samples-per-prompt <int>
#   --rollout-max-response-len <int>
#   --global-batch-size <int>
#   --save-interval <int>
#   --max-tokens-per-gpu <int>
#   --lr <float>
#   --opd-kl-coef <float>
#   --num-gpus <int>           # gpus on this node
#   [--wandb-group <str>]      # forwarded to slime if set

set -euxo pipefail

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-root)                ENV_ROOT="$2"; shift 2 ;;
        --student-hf)              STUDENT_HF="$2"; shift 2 ;;
        --prompt-data)             PROMPT_DATA="$2"; shift 2 ;;
        --save-dir)                SAVE_DIR="$2"; shift 2 ;;
        --num-rollout)             NUM_ROLLOUT="$2"; shift 2 ;;
        --rollout-batch-size)      ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
        --n-samples-per-prompt)    N_SAMPLES_PER_PROMPT="$2"; shift 2 ;;
        --rollout-max-response-len) ROLLOUT_MAX_RESPONSE_LEN="$2"; shift 2 ;;
        --global-batch-size)       GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --save-interval)           SAVE_INTERVAL="$2"; shift 2 ;;
        --max-tokens-per-gpu)      MAX_TOKENS_PER_GPU="$2"; shift 2 ;;
        --lr)                      LR="$2"; shift 2 ;;
        --opd-kl-coef)             OPD_KL_COEF="$2"; shift 2 ;;
        --num-gpus)                NUM_GPUS="$2"; shift 2 ;;
        --wandb-group)             WANDB_GROUP="$2"; shift 2 ;;
        --teacher-hf)              TEACHER_HF="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Required env vars from labctl/cluster.toml: HF_HOME, UV_CACHE_DIR, PATH(cuda).
: "${HF_HOME:?HF_HOME must be set}"
: "${ENV_ROOT:?--env-root required}"

# Source the slime-stack venv.
# shellcheck disable=SC1091
source "${ENV_ROOT}/venv/bin/activate"

# PYTHONPATH: Megatron-LM (slime's examples use it as PYTHONPATH, not pip pkg);
# plus opd-slime's own dir so opd_plugins.null_rm resolves for --custom-rm-path.
OPD_SLIME_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ENV_ROOT}/Megatron-LM:${OPD_SLIME_ROOT}:${PYTHONPATH:-}"

# CUDA runtime libs ship with torch wheels under venv/lib/.../nvidia/<pkg>/lib;
# sglang's scheduler subprocesses don't pick them up automatically.
NV_LIB_BASE="${ENV_ROOT}/venv/lib/python3.12/site-packages/nvidia"
NV_LD_PATHS=$(find "${NV_LIB_BASE}" -maxdepth 3 -type d -name lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LD_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PYTHONUNBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
export NCCL_NVLS_ENABLE="$HAS_NVLINK"

# slime's model-args helper. Qwen3-VL-2B's language model has Qwen3-1.7B
# architecture; --rotary-base 5e6 is the VL family setting.
# shellcheck disable=SC1091
MODEL_ARGS_ROTARY_BASE=5000000 source "${ENV_ROOT}/slime/scripts/models/qwen3-1.7B.sh"

# Best-effort cleanup of ray/sglang/slime — runs both at startup (stale
# processes from previous jobs) and on exit (trap covers normal exit AND
# failure paths; without the trap, `set -e` would skip post-train cleanup
# when train.py exits non-zero, leaking processes on the node).
cleanup() {
    pkill -9 sglang 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 ray   2>/dev/null || true
    pkill -9 slime 2>/dev/null || true
}
cleanup
sleep 3
trap cleanup EXIT

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# `ray start --head` without --num-cpus inherits the node's full CPU count
# (224 on the H100 boxes here, regardless of slurm cpus-per-task). Ray's
# raylet then pre-spawns one Python worker per CPU; on the shared-FS venv
# that 224-process import storm intermittently DoSes the raylet — driver
# RegisterClient calls time out behind worker-not-registered retries and the
# whole bring-up deadlocks (observed on jobs 107421 / 107423). Capping
# --num-cpus caps `--num_prestart_python_workers` to the same value. 32 is
# generous: slime's placement group needs 8 CPU bundles (1/GPU); SGLang and
# Ray dashboard/runtime-env agents add a handful more. Bump only if a real
# resource error surfaces.
RAY_HEAD_NUM_CPUS=${RAY_HEAD_NUM_CPUS:-32}
# Belt-and-suspenders: extend the registration timeout so any straggler
# worker (slow Lustre stat) still makes it in before the raylet escalates.
export RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds:-180}

ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --num-cpus "${RAY_HEAD_NUM_CPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

# Direct python invocation against the running ray head — bypasses
# `ray job submit` which has unreliable dashboard-JobHead startup on this
# cluster (intermittent 504s before JobHead is ready).
export RAY_ADDRESS="${MASTER_ADDR}:6379"

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-opd
        --wandb-key "${WANDB_API_KEY}"
        --disable-wandb-random-suffix
    )
    if [ -n "${WANDB_GROUP:-}" ]; then
        WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
    fi
fi

# --save-hf takes a Python format string; we keep the literal "{rollout_id}"
# token in bash so labctl's template engine never tries to resolve it (which
# would error since {rollout_id} isn't a labctl-known variable).
python3 "${ENV_ROOT}/slime/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${TEACHER_HF}" \
    --load          "${STUDENT_HF}" \
    --save          "${SAVE_DIR}/megatron" \
    --save-hf       "${SAVE_DIR}/rollout_{rollout_id}" \
    --save-interval "${SAVE_INTERVAL}" \
    --rotary-base   5000000 \
    --prompt-data            "${PROMPT_DATA}" \
    --input-key              prompt \
    --apply-chat-template \
    --rollout-shuffle \
    --num-rollout            "${NUM_ROLLOUT}" \
    --rollout-batch-size     "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt   "${N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-temperature    1.0 \
    --global-batch-size      "${GLOBAL_BATCH_SIZE}" \
    --balance-data \
    --advantage-estimator    grpo \
    --use-opd \
    --opd-type               megatron \
    --opd-kl-coef            "${OPD_KL_COEF}" \
    --opd-teacher-load       "${TEACHER_HF}" \
    --custom-rm-path         opd_plugins.null_rm.reward_func \
    --kl-loss-coef           0.0 \
    --kl-coef                0.0 \
    --kl-loss-type           low_var_kl \
    --entropy-coef           0.0 \
    --eps-clip               0.2 \
    --eps-clip-high          0.28 \
    --optimizer              adam \
    --lr                     "${LR}" \
    --lr-decay-style         constant \
    --weight-decay           0.1 \
    --adam-beta1             0.9 \
    --adam-beta2             0.98 \
    --tensor-model-parallel-size 1 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu     "${MAX_TOKENS_PER_GPU}" \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.6 \
    --colocate \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --megatron-to-hf-mode bridge \
    "${WANDB_ARGS[@]}"
# Cleanup happens via `trap cleanup EXIT` set above — covers both success
# (this line reached) and failure (set -e bails train.py non-zero).
