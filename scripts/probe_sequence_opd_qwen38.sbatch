#!/usr/bin/env bash
#SBATCH --job-name=oev3_seq_opd_probe
#SBATCH --qos=low
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/eval_logs/sequence_opd_qwen38/slurm-%j.out
#SBATCH --exclude=hai001,hai002,hai006,hai007,hai008

set -euo pipefail

WT=/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-oev3-cuagym
PY="$WT/.venv/bin/python"
STUDENT=/fast/project/HFMI_SynergyUnit/yll/checkpoints/oev3_exports/qwen35_9b_lora_success_cuagym_oev3_stage_06_seqlen_24576_lr5e5_run_01a05470bbf27863bed02222c8fb3bea/010000
STUDENT_TOKENIZER=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
TEACHER=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
VAL=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu/cuagym_p2_stage_06_perstep_records_seqlen_24576_v2/val/part-00000.array_record
MANIFEST=/fast/project/HFMI_SynergyUnit/yll/eval_logs/bc_cursor_probe/pairs_seq24/manifest.jsonl
OUTDIR=/fast/project/HFMI_SynergyUnit/yll/eval_logs/sequence_opd_qwen38
API_KEY=sequence-opd-probe
STUDENT_PORT=$((32000 + SLURM_JOB_ID % 1000))
TEACHER_PORT=$((STUDENT_PORT + 1000))

mkdir -p "$OUTDIR"
export HF_HOME=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export SGLANG_DISABLE_CUDNN_CHECK=1

cleanup() {
    kill "${STUDENT_PID:-}" "${TEACHER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 "$PY" -m sglang.launch_server \
    --model-path "$STUDENT" --host 127.0.0.1 --port "$STUDENT_PORT" \
    --api-key "$API_KEY" --served-model-name oev3-student \
    --mem-fraction-static 0.78 --chunked-prefill-size 2048 \
    >"$OUTDIR/student-server-$SLURM_JOB_ID.log" 2>&1 &
STUDENT_PID=$!

CUDA_VISIBLE_DEVICES=1,2 "$PY" -m sglang.launch_server \
    --model-path "$TEACHER" --host 127.0.0.1 --port "$TEACHER_PORT" \
    --api-key "$API_KEY" --served-model-name qwen38-teacher \
    --tp-size 2 --mem-fraction-static 0.78 --chunked-prefill-size 2048 \
    >"$OUTDIR/teacher-server-$SLURM_JOB_ID.log" 2>&1 &
TEACHER_PID=$!

for port in "$STUDENT_PORT" "$TEACHER_PORT"; do
    ready=0
    # Concurrent cold starts can spend >10 minutes reading both checkpoints
    # from the shared filesystem even though inference itself is healthy.
    for _ in $(seq 1 900); do
        if curl -sf -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$port/health_generate" >/dev/null; then
            ready=1
            break
        fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        echo "server on port $port did not become healthy" >&2
        exit 1
    fi
done

cd "$WT/eval"
"$PY" probe_sequence_opd.py \
    --student-url "http://127.0.0.1:$STUDENT_PORT" \
    --teacher-url "http://127.0.0.1:$TEACHER_PORT" \
    --student-model oev3-student \
    --student-tokenizer "$STUDENT_TOKENIZER" \
    --teacher-tokenizer "$TEACHER" \
    --api-key "$API_KEY" \
    --val-shard "$VAL" \
    --manifest "$MANIFEST" \
    --num-records 12 \
    --temperature 0.8 \
    --max-tokens 4096 \
    --concurrency 2 \
    --out "$OUTDIR/result.json"
