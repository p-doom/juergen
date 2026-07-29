#!/bin/bash
#SBATCH --job-name=dt_norm_distill
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=4:00:00
#SBATCH --nodelist=hai005
#SBATCH --signal=USR1@600
#SBATCH --requeue
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/dt_norm_distill_%j.out
set -euo pipefail
# DELTATYPE-NORMALIZED distillation SFT: the format-fix baseline (coalesced type("...") +
# documented TERMINATE/FAIL + normalized 0-999 bare-token deltas), IDENTICAL fmt_sft recipe
# to fmt_sft_8b_osw_diffabs (the diffabs 1.0% baseline) so the held-out eval delta isolates
# the format changes. tokenize -> train LoRA -> export merged HF.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
CONV=$ROOT/onpolicy_distill/converted/osworld_train_deltatype_normalized
SAVE=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_osw_deltatype_norm
OUT=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_osw_deltatype_norm_hf
STOCK=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)

export HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp
JAXC=$ROOT/jax_cache/dt_norm_${SLURM_JOB_ID}
mkdir -p "$JAXC" "$SAVE" "$OUT"

# ---------------- (1) tokenize ----------------
for split in train val; do
  echo "=== tokenizing deltatype_normalized $split @ $(date) ==="
  uv run --project "$OMX" -- python "$OMX/scripts/build_sft_records_from_chat.py" \
    --data_path "$CONV/_normalized/$split/chat.jsonl" \
    --out_dir "$CONV/$split" \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --tokenizer Qwen/Qwen3-VL-2B-Instruct \
    --processor Qwen/Qwen3-VL-2B-Instruct \
    --max_length 16384 --overflow_mode truncate \
    --records_per_shard 8000 --num_workers 8 --overwrite
done
[ -f "$CONV/train/metadata.json" ] || { echo "tokenize failed"; exit 1; }

# ---------------- (2) train (IDENTICAL recipe to fmt_sft_8b_osw_diffabs) ----------------
cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAXC" \
  --model_id=Qwen/Qwen3-VL-8B-Instruct \
  --processor=Qwen/Qwen3-VL-8B-Instruct \
  --data_path="$CONV/train" \
  --val_data_path="$CONV/val" \
  --save_dir="$SAVE" \
  --enable_lora=true --lora_rank=32 --lora_alpha=32 --freeze_vision_tower=false \
  --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --batch_size=1 --grad_accum_steps=8 \
  --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.7 --lr_end_factor=0.0 --warmup_steps=30 \
  --weight_decay=0.01 --max_grad_norm=1.0 --max_length=16384 \
  --num_steps=300 --num_loss_tiles=16 \
  --keep_latest=1 --keep_period=150 --save_every=150 --val_every=150 --val_steps=15 \
  --log_every=10 --log_memory=true --resume=if_present --gc_period=3000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=29 --max_vision_patches_per_sample=64000 \
  --grain_read_threads=4 --grain_read_buffer_size=2 --grain_workers=2 --grain_worker_buffer_size=2 \
  --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group=fmt_sft_8b_osw_deltatype_norm_${SLURM_JOB_ID} \
  --wandb_name=fmt_sft_8b_osw_deltatype_norm_${SLURM_JOB_ID} \
  --wandb_tags berlin lora distillation deltatype normalized_bare coalesced_type single_gpu

# ---------------- (3) export orbax LoRA -> merged HF ----------------
CKPT=$(ls -d "$SAVE"/[0-9]*/ 2>/dev/null | sort | tail -1); CKPT=${CKPT%/}
[ -n "$CKPT" ] && [ -d "$CKPT" ] || { echo "no checkpoint to export at $SAVE"; exit 1; }
echo "=== exporting $CKPT -> $OUT @ $(date) ==="
JAX_PLATFORMS=cpu srun uv run --project="$OMX" -- python scripts/export_to_hf.py \
  --model_id=Qwen/Qwen3-VL-8B-Instruct --checkpoint_path="$CKPT" --out_dir="$OUT" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
         special_tokens_map.json chat_template.json generation_config.json \
         preprocessor_config.json video_preprocessor_config.json; do
  [ -f "$STOCK/$f" ] && [ ! -f "$OUT/$f" ] && cp "$STOCK/$f" "$OUT/" || true
done
python3 - "$STOCK/config.json" "$OUT/config.json" <<'PY'
import json,sys
b=json.load(open(sys.argv[1])); o=json.load(open(sys.argv[2]))
for k in ("architectures","transformers_version","vision_end_token_id"):
    if k in b and k not in o: o[k]=b[k]
json.dump(o,open(sys.argv[2],"w"),indent=2)
PY
test -f "$OUT/model.safetensors" || test -f "$OUT/model.safetensors.index.json"
echo "=== DELTATYPE_NORM DISTILL DONE -> $OUT @ $(date) ==="
