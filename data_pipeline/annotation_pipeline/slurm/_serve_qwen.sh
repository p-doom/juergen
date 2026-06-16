#!/usr/bin/env bash
# Sourced by the pipeline slurm scripts. Defines serve_qwen(): start sglang
# serving Qwen3.6-27B (BF16, TP=2 DP=4, hardened: mem-fraction 0.70 for vision-
# encoder activation headroom under concurrent 720p prefills, cuDNN 9.16 from
# setup_env), wait until healthy, export V3_VLM_BASE_URL, and set SERVER_PID
# so the caller can `trap 'kill $SERVER_PID' EXIT`. One place for the serve config.

serve_qwen() {
  local YLL=/fast/project/HFMI_SynergyUnit/yll
  local SERVE_VENV=$YLL/venvs/vllm-annotate
  local MODEL=Qwen/Qwen3.6-27B
  local PORT=8011
  export HF_HOME=$YLL/.cache/huggingface

  "$SERVE_VENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$MODEL" --port $PORT \
    --tp-size 2 --dp-size 4 --context-length 98304 \
    --mem-fraction-static 0.65 --reasoning-parser qwen3 &
  SERVER_PID=$!

  local i
  for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "sglang died during startup" >&2; return 1; }
    sleep 10
  done
  curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "sglang never healthy" >&2; return 1; }
  echo "sglang serving $MODEL on :$PORT (TP=2, DP=4)"
  export V3_VLM_BASE_URL="http://localhost:$PORT/v1"
  export V3_VLM_TIMEOUT_S=600
}
