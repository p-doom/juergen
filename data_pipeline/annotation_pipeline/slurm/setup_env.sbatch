#!/usr/bin/env bash
#SBATCH --job-name=v3_setup_env
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%j.log

# One-time environment setup for the v3 pipeline. Builds two venvs:
#   1. vllm-annotate  - sglang serving stack (torch cu126, cuDNN 9.16), from the
#                       proven slime pins; serves Qwen3.6-27B.
#   2. juergen workspace venv - pipeline stages 00-04 via data_pipeline's uv
#                       project. Pinned transformers==5.3.0 to match the
#                       workspace/sglang dependency solution (no omegalax
#                       dependency).
# Also downloads the annotator (Qwen3.6-27B) and trainee (Qwen3-VL-2B) weights.

set -euo pipefail

YLL=/fast/project/HFMI_SynergyUnit/yll
V3=$YLL/juergen/data_pipeline/annotation_pipeline
DATA_PIPELINE=$YLL/juergen/data_pipeline
VENV=$YLL/venvs/vllm-annotate
SLIME_VENV=$YLL/venvs/slime
SGL_SRC=$YLL/deps/sglang/python
export HF_HOME=$YLL/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
UV=/home/yll.kryeziu/.local/bin/uv

CONSTRAINTS=$(mktemp /tmp/sglang_constraints_XXXX.txt)
# Pin the GPU/serving stack to the exact versions proven in the slime venv
# (where Qwen3.6-27B serving works on this cluster's CUDA 12.8 driver);
# sglang's own pins resolve everything else. A full venv freeze is NOT usable
# as constraints: the slime venv holds pip-tolerated conflicts (e.g.
# openai 2.38.0 vs sglang's pin of 2.6.1) that uv rejects.
cat > "$CONSTRAINTS" <<'PINS'
torch==2.9.1
torchvision==0.24.1
torchaudio==2.9.1
torchao==0.9.0
sgl-kernel==0.3.21
flashinfer-python==0.6.3
flashinfer-cubin==0.6.3
transformers==4.57.1
xgrammar==0.1.27
timm==1.0.16
decord2==3.3.0
PINS

rm -rf "$VENV"
"$UV" venv "$VENV" --python 3.12
"$UV" pip install --python "$VENV/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match \
  -c "$CONSTRAINTS" \
  -e "$SGL_SRC[all]" hf_transfer "huggingface_hub[cli]"

# sglang's startup check requires cuDNN >= 9.15 with torch 2.9.1 (Conv3d bug,
# pytorch#168167) — the VLM vision tower uses Conv3d, so upgrade rather than
# set SGLANG_DISABLE_CUDNN_CHECK.
"$UV" pip install --python "$VENV/bin/python" "nvidia-cudnn-cu12==9.16.0.29"

"$VENV/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
rm -f "$CONSTRAINTS"

# --- 2. pipeline env (stages 00-04; stage 05 optional) --------------------
# Pin lives in data_pipeline/pyproject.toml and juergen/uv.lock. Keep this
# locked so SLURM jobs do not silently re-resolve the annotation environment.
"$UV" sync --project "$DATA_PIPELINE" --locked
"$UV" run --project "$DATA_PIPELINE" --locked python -c "import transformers, annotation_pipeline.qwen3_encoding; print('data_pipeline transformers', transformers.__version__)"

# --- 3. weights ----------------------------------------------------------
"$VENV/bin/hf" download Qwen/Qwen3.6-27B >/dev/null        # annotator (BF16)
"$VENV/bin/hf" download Qwen/Qwen3-VL-2B-Instruct >/dev/null  # trainee tokenizer/processor
echo "SETUP_DONE"
