#!/bin/bash
# Matched stage-2 B training from one of the two frozen merged stage-1 models.
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}";;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac
done
for key in omegalax_repo dataset source_model output; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done

OMX="${A[omegalax_repo]}"
DATA="${A[dataset]}"
SOURCE="${A[source_model]}"
SOURCE_HF="$SOURCE/hf"
OUT="${A[output]}"
ORBAX="$OUT/orbax"
HF="$OUT/hf"
SOURCE_ARM="$(jq -r '.arm' "$SOURCE/train_export_manifest.json")"
case "$SOURCE_ARM" in reltool_pre) BRANCH=A_to_B;; relraw_pre) BRANCH=B_to_B;;
  *) echo "FATAL frozen source is not a preamble arm: $SOURCE_ARM" >&2; exit 2;; esac
EXPECTED_SOURCE_ARM="$SOURCE_ARM"
python3 - "$DATA" "$SOURCE" "$EXPECTED_SOURCE_ARM" <<'PY'
import json, sys
from pathlib import Path
dataset, source, expected_arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
d = json.loads((dataset / "curriculum_dataset_manifest.json").read_text())
expected = {
    "artifact_type": "synthetic_multistep_curriculum_stage2_tokenized",
    "status": "complete", "format": "deltatype_raw_pre",
    "train_records": 2000, "validation_records": 200,
    "seeds": {"train": 2026073101, "val": 2026073102}, "max_length": 4096,
}
bad = {key: (d.get(key), value) for key, value in expected.items() if d.get(key) != value}
if bad:
    raise SystemExit(f"FATAL wrong stage-2 dataset: {bad}")
report = json.loads((dataset / d["overlap_report"]).read_text())
def nonzero(value):
    if isinstance(value, dict): return any(nonzero(x) for x in value.values())
    return value != 0
if report.get("status") != "pass" or nonzero(report.get("overlap_counts", {})):
    raise SystemExit(f"FATAL stage-2 overlap gate failed: {report}")
for split, expected_count in (("train", 2000), ("val", 200)):
    meta = json.loads((dataset / "deltatype_raw_pre" / split / "metadata.json").read_text())
    if meta.get("num_records") != expected_count or meta.get("max_length") != 4096:
        raise SystemExit(f"FATAL tokenized split mismatch {split}: {meta}")
s = json.loads((source / "train_export_manifest.json").read_text())
source_expected = {
    "artifact_type": "relative_factorial_hf_checkpoint", "schema_version": 1,
    "status": "complete", "arm": expected_arm,
    "model_id": "Qwen/Qwen3-VL-8B-Instruct", "step": 750,
    "lora_rank": 256, "lora_alpha": 256, "hf_subdir": "hf",
}
bad = {key: (s.get(key), value) for key, value in source_expected.items()
       if s.get(key) != value}
if bad:
    raise SystemExit(f"FATAL wrong frozen stage-1 source: {bad}")
hf = source / s["hf_subdir"]
for name in ("config.json", "model.safetensors", "tokenizer_config.json",
             "chat_template.json", "preprocessor_config.json"):
    if not (hf / name).is_file():
        raise SystemExit(f"FATAL incomplete merged stage-1 source: {hf / name}")
PY

[[ ! -e "$OUT/curriculum_train_export_manifest.json" ]] || {
  echo "FATAL refusing to overwrite completed stage-2 export: $OUT" >&2; exit 2;
}
mkdir -p "$ORBAX"
job_tag="${SLURM_JOB_ID:-local}_${BRANCH}_raw_pre_r256"
JAX_CACHE="${JAX_CACHE_ROOT:-/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/jax_cache}/curriculum_${job_tag}"
mkdir -p "$JAX_CACHE"

cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAX_CACHE" \
  --model_id="$SOURCE_HF" --processor="$SOURCE_HF" \
  --data_path="$DATA/deltatype_raw_pre/train" \
  --val_data_path="$DATA/deltatype_raw_pre/val" \
  --save_dir="$ORBAX" \
  --enable_lora=true --lora_rank=256 --lora_alpha=256 --freeze_vision_tower=false \
  --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --batch_size=1 --grad_accum_steps=8 \
  --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.7 \
  --lr_end_factor=0.0 --warmup_steps=30 --weight_decay=0.01 --max_grad_norm=1.0 \
  --max_length=4096 --num_steps=750 --num_loss_tiles=8 \
  --keep_latest=3 --keep_period=250 --save_every=250 \
  --val_every=250 --val_steps=15 \
  --log_every=10 --log_memory=false --resume=if_present --gc_period=3000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=2 --max_vision_patches_per_sample=16000 \
  --grain_read_threads=4 --grain_read_buffer_size=4 \
  --grain_workers=2 --grain_worker_buffer_size=2 \
  --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group="curriculum_raw_pre_r256" \
  --wandb_name="curriculum_${job_tag}" \
  --wandb_tags="berlin,rung3,curriculum,raw_preamble,lora,r256"

CKPT="$ORBAX/000750"
[[ -d "$CKPT" && -f "$CKPT/_CHECKPOINT_METADATA" ]] || {
  echo "FATAL stage-2 step-750 checkpoint missing: $CKPT" >&2; exit 3;
}
[[ -f "$ORBAX/lora_metadata.json" ]] || {
  echo "FATAL stage-2 LoRA metadata missing" >&2; exit 3;
}
mkdir -p "$HF"
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="$OMX" -- \
  python scripts/export_to_hf.py --model_id="$SOURCE_HF" \
  --checkpoint_path="$CKPT" --out_dir="$HF" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8

for file in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
  special_tokens_map.json chat_template.json generation_config.json \
  preprocessor_config.json video_preprocessor_config.json; do
  [[ ! -f "$SOURCE_HF/$file" || -f "$HF/$file" ]] || cp "$SOURCE_HF/$file" "$HF/"
done
OMX_SHA="$(git -C "$OMX" rev-parse HEAD)"
OMX_DIFF_SHA="$(git -C "$OMX" diff --binary | sha256sum | awk '{print $1}')"
python3 - "$DATA" "$SOURCE" "$OUT" "$BRANCH" "$EXPECTED_SOURCE_ARM" \
  "$CKPT" "$OMX_SHA" "$OMX_DIFF_SHA" <<'PY'
import hashlib, json, sys
from pathlib import Path
dataset, source, out, branch, source_arm, ckpt = map(Path, sys.argv[1:7])
branch = branch.name
source_arm = source_arm.name
sha, diff = sys.argv[7:9]
hf = out / "hf"
meta = json.loads((ckpt.parent / "lora_metadata.json").read_text())
if int(meta.get("lora_rank", -1)) != 256 or float(meta.get("lora_alpha", -1)) != 256:
    raise SystemExit(f"FATAL stage-2 LoRA metadata mismatch: {meta}")
config = json.loads((hf / "config.json").read_text())
if not config.get("architectures"):
    source_config = json.loads((source / "hf" / "config.json").read_text())
    config["architectures"] = source_config.get("architectures")
    (hf / "config.json").write_text(json.dumps(config, indent=2) + "\n")
for name in ("model.safetensors", "config.json", "tokenizer_config.json",
             "chat_template.json", "preprocessor_config.json"):
    if not (hf / name).is_file(): raise SystemExit(f"FATAL export file missing: {hf / name}")
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "artifact_type": "synthetic_multistep_curriculum_hf_checkpoint",
    "schema_version": 1, "status": "complete", "branch": branch,
    "source_stage1_arm": source_arm, "source_model": str(source.resolve()),
    "source_manifest_sha256": digest(source / "train_export_manifest.json"),
    "dataset": str(dataset.resolve()),
    "dataset_manifest_sha256": digest(dataset / "curriculum_dataset_manifest.json"),
    "target_format": "deltatype_raw_pre", "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "step": 750, "fresh_optimizer": True, "lora_rank": 256, "lora_alpha": 256,
    "max_length": 4096, "hf_subdir": "hf", "source_checkpoint": str(ckpt),
    "omegalax_commit": sha, "omegalax_tracked_diff_sha256": diff,
}
(out / "curriculum_train_export_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY
echo "curriculum stage-2 train+export complete: $BRANCH -> $HF"
