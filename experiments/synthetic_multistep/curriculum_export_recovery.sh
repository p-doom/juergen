#!/bin/bash
# Export an intact stage-2 Orbax checkpoint whose original post-train shell failed.
set -euo pipefail
declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}";;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac
done
for key in omegalax_repo dataset source_model partial_root failed_run_dir failed_run_id failed_job_id output; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
OMX="${A[omegalax_repo]}"; DATA="${A[dataset]}"; SOURCE="${A[source_model]}"
PARTIAL="${A[partial_root]}"; OUT="${A[output]}"; SOURCE_HF="$SOURCE/hf"
CKPT="$PARTIAL/orbax/000750"; HF="$OUT/hf"
FAILED_RUN_DIR="${A[failed_run_dir]}"
TRAIN_LOG="$(find "$FAILED_RUN_DIR/.lab" -maxdepth 1 -type f -name "*_${A[failed_job_id]}.log" | head -1)"
[[ -f "$TRAIN_LOG" ]] || { echo "FATAL failed-run training log missing" >&2; exit 2; }
SOURCE_ARM="$(jq -r '.arm' "$SOURCE/train_export_manifest.json")"
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"
case "$SOURCE_ARM" in reltool_pre) BRANCH=A_to_B;; relraw_pre) BRANCH=B_to_B;;
  *) echo "FATAL wrong source arm: $SOURCE_ARM" >&2; exit 2;; esac
python3 - "$DATA" "$SOURCE" "$PARTIAL" "$CKPT" "$TRAIN_LOG" <<'PY'
import json, math, re, sys
from pathlib import Path
dataset, source, partial, checkpoint, log_path = map(Path, sys.argv[1:6])
d = json.loads((dataset / "curriculum_dataset_manifest.json").read_text())
if (d.get("status") != "complete" or d.get("train_records") != 2000
        or d.get("validation_records") != 200):
    raise SystemExit(f"FATAL wrong stage-2 dataset: {d}")
s = json.loads((source / "train_export_manifest.json").read_text())
if (s.get("status") != "complete" or s.get("lora_rank") != 256
        or s.get("arm") not in ("reltool_pre", "relraw_pre")):
    raise SystemExit(f"FATAL wrong stage-1 source: {s}")
if not (checkpoint / "_CHECKPOINT_METADATA").is_file():
    raise SystemExit(f"FATAL intact step750 checkpoint missing: {checkpoint}")
meta = json.loads((checkpoint.parent / "lora_metadata.json").read_text())
if int(meta.get("lora_rank", -1)) != 256 or float(meta.get("lora_alpha", -1)) != 256:
    raise SystemExit(f"FATAL LoRA metadata mismatch: {meta}")
if (partial / "hf" / "model.safetensors").exists():
    raise SystemExit(f"FATAL partial root unexpectedly already has HF weights: {partial}")
text = log_path.read_text(errors="replace")
matches = re.findall(r"step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)", text)
steps = [(int(step), float(loss), float(grad)) for step, loss, grad in matches]
if [step for step, _, _ in steps] != list(range(10, 751, 10)):
    raise SystemExit(f"FATAL incomplete logged finite-step sequence: {[x[0] for x in steps]}")
if not all(math.isfinite(loss) and math.isfinite(grad) for _, loss, grad in steps):
    raise SystemExit("FATAL non-finite logged training endpoint")
if "finished step=750" not in text or "=mosaic_gpu: command not found" not in text:
    raise SystemExit("FATAL failed-run log does not match the audited post-train export failure")
PY
mkdir -p "$HF"
cd "$OMX"
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="$OMX" -- \
  python scripts/export_to_hf.py --model_id="$BASE_MODEL" \
  --checkpoint_path="$CKPT" --out_dir="$HF" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8
for file in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
  special_tokens_map.json chat_template.json generation_config.json \
  preprocessor_config.json video_preprocessor_config.json; do
  [[ ! -f "$SOURCE_HF/$file" || -f "$HF/$file" ]] || cp "$SOURCE_HF/$file" "$HF/"
done
OMX_SHA="$(git -C "$OMX" rev-parse HEAD)"
OMX_DIFF_SHA="$(git -C "$OMX" diff --binary | sha256sum | awk '{print $1}')"
python3 - "$DATA" "$SOURCE" "$PARTIAL" "$OUT" "$BRANCH" "$SOURCE_ARM" \
  "$CKPT" "$OMX_SHA" "$OMX_DIFF_SHA" "$TRAIN_LOG" \
  "${A[failed_run_id]}" "${A[failed_job_id]}" <<'PY'
import hashlib, json, math, re, sys
from pathlib import Path
dataset, source, partial, out = map(Path, sys.argv[1:5])
branch, source_arm = sys.argv[5:7]
checkpoint, sha, diff = Path(sys.argv[7]), sys.argv[8], sys.argv[9]
log_path, failed_run_id, failed_job_id = Path(sys.argv[10]), sys.argv[11], int(sys.argv[12])
hf = out / "hf"
config = json.loads((hf / "config.json").read_text())
source_config = json.loads((source / "hf" / "config.json").read_text())
for key in ("architectures", "transformers_version", "vision_end_token_id"):
    if key in source_config and key not in config: config[key] = source_config[key]
if not config.get("architectures"): raise SystemExit("FATAL export lacks architectures")
(hf / "config.json").write_text(json.dumps(config, indent=2) + "\n")
for name in ("model.safetensors", "config.json", "tokenizer_config.json",
             "chat_template.json", "preprocessor_config.json"):
    if not (hf / name).is_file(): raise SystemExit(f"FATAL export file missing: {hf / name}")
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
log_text = log_path.read_text(errors="replace")
finite_steps = [(int(step), float(loss), float(grad)) for step, loss, grad in
                re.findall(r"step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)", log_text)]
endpoint_hashes = {
    "checkpoint_metadata_sha256": digest(checkpoint / "_CHECKPOINT_METADATA"),
    "train_state_metadata_sha256": digest(checkpoint / "train_state" / "_METADATA"),
    "input_iterator_sha256": digest(checkpoint / "input_iter" / "process_0-of-1.json"),
    "lora_metadata_sha256": digest(checkpoint.parent / "lora_metadata.json"),
    "failed_run_log_sha256": digest(log_path),
}
manifest = {
    "artifact_type": "synthetic_multistep_curriculum_hf_checkpoint",
    "schema_version": 1, "status": "complete", "branch": branch,
    "source_stage1_arm": source_arm, "source_model": str(source.resolve()),
    "source_manifest_sha256": digest(source / "train_export_manifest.json"),
    "dataset": str(dataset.resolve()),
    "dataset_manifest_sha256": digest(dataset / "curriculum_dataset_manifest.json"),
    "target_format": "deltatype_raw_pre", "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "step": 750, "fresh_optimizer": True, "lora_rank": 256, "lora_alpha": 256,
    "learning_rate": 1e-4, "max_length": 4096, "hf_subdir": "hf",
    "source_checkpoint": str(checkpoint.resolve()),
    "recovered_from_failed_run_root": str(partial.resolve()),
    "recovered_from_failed_run_id": failed_run_id,
    "recovered_from_failed_job_id": failed_job_id,
    "logged_finite_step_checks": len(finite_steps),
    "logged_step_sequence": [step for step, _, _ in finite_steps],
    "final_logged_loss": finite_steps[-1][1],
    "final_logged_grad_norm": finite_steps[-1][2],
    "endpoint_hashes": endpoint_hashes,
    "omegalax_commit": sha, "omegalax_tracked_diff_sha256": diff,
}
(out / "curriculum_train_export_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY
echo "recovered curriculum export complete: $BRANCH -> $HF"
