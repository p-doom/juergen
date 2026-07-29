#!/bin/bash
#SBATCH --job-name=fmt_validity_only
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/fmt_validity_only_%j.out
set -euo pipefail
# Track-1 OFFLINE format-validity re-run (the combined fmt_eval OOM'd on export+sglang; the _hf
# exports already succeeded, so here we run ONLY the sglang validity probe on the existing merged HF
# -> no double model-load -> no OOM). FMT=absolute|moverel|diffabs.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
EVAL=/fast/home/franz.srambical/juergen/eval
SCR=$ROOT/onpolicy_distill/scripts
FMT="${FMT:?set FMT=absolute|moverel|diffabs}"
CKPT=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_${FMT}_hf
VAL=$ROOT/onpolicy_distill/converted/teacher_8b_v4_${FMT}/_normalized/val/chat.jsonl
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0; export PATH=$CUDA_HOME/bin:$PATH
[ -f "$CKPT/model.safetensors" ] || { echo "missing $CKPT"; exit 1; }
PORT=$(( 32100 + (SLURM_JOB_ID % 100) * 10 ))
cd "$EVAL"
uv run --project "$EVAL" python "$SCR/fmt_validity_check.py" \
  --model_path "$CKPT" --val_jsonl "$VAL" --format "$FMT" --n 10 \
  --out_json "$ROOT/onpolicy_distill/eval/fmt_validity_${FMT}.json" \
  --sglang_port "$PORT" --mem_fraction_static 0.70 --max_tokens 256
echo "=== $FMT val-loss (wandb-offline summary, if present in SFT log) ==="
grep -ihE "val/loss|eval/loss" $ROOT/onpolicy_distill/logs/fmt_sft_*.out 2>/dev/null | tail -3 || true
echo "=== VALIDITY-ONLY ($FMT) DONE ==="
