#!/usr/bin/env bash
#SBATCH --job-name=pilot_qwen36
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=02:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pilot/output_qwen/slurm_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pilot/output_qwen/slurm_%j.log

# Single-recording pilot: 1 GPU, 1 sglang replica (TP=1).
# For production (394 recordings): request 8 GPUs, launch 8 replicas,
# and fan out recordings across them.

set -euo pipefail

# ── Environment ──
export HF_HOME=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface
export CUDA_HOME=/usr/local/cuda-12
export SGLANG_DISABLE_CUDNN_CHECK=1

PIPELINE=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
EVAL_PROJECT=/fast/project/HFMI_SynergyUnit/yll/juergen/eval
OUTPUT=$PIPELINE/annotation_pilot/output_qwen
STAGE_B_DIR=$PIPELINE/annotation_pilot/stage_a_2fps_540p
RECORDING_ID=d64dc788-6466-4ade-8f4e-b55c21d8311b

MODEL=Qwen/Qwen3.6-27B
MODEL_NAME=qwen
PORT=$((30000 + SLURM_JOB_ID % 10000))

cd "$PIPELINE"
mkdir -p "$OUTPUT"

# ── Launch sglang server (TP=1, single GPU) ──
echo "[sglang] Starting server on port $PORT ..."
uv run --project="$EVAL_PROJECT" \
  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --chunked-prefill-size 4096 \
    --max-running-requests 16 \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
  > "$OUTPUT/sglang_server.log" 2>&1 &
SGLANG_PID=$!

cleanup() {
  echo "[sglang] Shutting down (pid=$SGLANG_PID)"
  kill "$SGLANG_PID" 2>/dev/null || true
  wait "$SGLANG_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[sglang] Waiting for server to be ready..."
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/health_generate" > /dev/null 2>&1; then
    echo "[sglang] Ready after ~$((i*5))s"
    break
  fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "[sglang] Server died. Last 30 lines of log:"
    tail -30 "$OUTPUT/sglang_server.log"
    exit 1
  fi
  sleep 5
done

if ! curl -sf "http://localhost:$PORT/health_generate" > /dev/null 2>&1; then
  echo "[sglang] Server not ready after 30 min, aborting"
  tail -50 "$OUTPUT/sglang_server.log"
  exit 1
fi

# Point stages at the local sglang endpoint (OpenAI-compatible)
export AZURE_AI_FOUNDRY_ENDPOINT="http://localhost:$PORT/v1"
export AZURE_AI_FOUNDRY_API_KEY="local"

# ── Stage C: Captioning ──
echo ""
echo "============================================================"
echo "STAGE C: Keyframe captioning"
echo "============================================================"
time uv run python stage_c_caption.py \
  --source_path="$STAGE_B_DIR" \
  --output_dir="$OUTPUT" \
  --split=train \
  --recording_id="$RECORDING_ID" \
  --keyframe_every_n=75 \
  --target_height=720 \
  --batch_size=15 \
  --model="$MODEL_NAME" \
  --temperature=0.3 \
  --max_tokens=8192

# ── Stage D: Task tree ──
echo ""
echo "============================================================"
echo "STAGE D: Task tree"
echo "============================================================"
time uv run python stage_d_task_tree.py \
  --captions_path="$OUTPUT/captions.json" \
  --output_dir="$OUTPUT" \
  --model="$MODEL_NAME" \
  --temperature=0.3 \
  --max_tokens=16384

# ── Stage E: Chunk buckets ──
echo ""
echo "============================================================"
echo "STAGE E: Chunk buckets"
echo "============================================================"
time uv run python stage_e_chunk_buckets.py \
  --stage_a_dir="$STAGE_B_DIR" \
  --task_tree_path="$OUTPUT/task_tree.json" \
  --captions_path="$OUTPUT/captions.json" \
  --keyframes_meta_path="$OUTPUT/keyframes_meta.json" \
  --output_dir="$OUTPUT/bucket_samples" \
  --recording_id="$RECORDING_ID" \
  --split=train \
  --model="$MODEL_NAME"

echo ""
echo "============================================================"
echo "PILOT COMPLETE"
echo "Output: $OUTPUT"
echo "============================================================"
