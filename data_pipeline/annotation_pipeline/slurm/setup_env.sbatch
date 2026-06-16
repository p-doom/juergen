#!/usr/bin/env bash
#SBATCH --job-name=v3_setup_env
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation-pipeline/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation-pipeline/logs/%x_%j.log

# One-time environment setup for the v3 pipeline. Builds two venvs:
#   1. vllm-annotate  - sglang serving stack (torch cu126, cuDNN 9.16), from the
#                       proven slime pins; serves Qwen3.6-27B.
#   2. v3/.venv        - pipeline stages 00-04. Pinned transformers==5.2.0 so the
#                       slow (numpy/PIL) Qwen3-VL image processor builds torch-free
#                       for exact stage-04 token counts (no omegalax dependency).
# Also downloads the annotator (Qwen3.6-27B) and trainee (Qwen3-VL-2B) weights.

set -euo pipefail

YLL=/fast/project/HFMI_SynergyUnit/yll
V3=$YLL/juergen/data_pipeline/annotation-pipeline
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

# --- 2. pipeline venv (stages 00-04) -------------------------------------
# Pin transformers 5.2.0: the slow Qwen3-VL image processor builds without
# torch there, which the vendored qwen3_encoding uses for exact token counts.
"$UV" sync --project "$V3" 2>/dev/null || true
"$UV" pip install --python "$V3/.venv/bin/python" "transformers==5.2.0" pillow
"$V3/.venv/bin/python" -c "import transformers, qwen3_encoding; print('v3 transformers', transformers.__version__)"

# --- 3. weights ----------------------------------------------------------
"$VENV/bin/hf" download Qwen/Qwen3.6-27B >/dev/null        # annotator (BF16)
"$VENV/bin/hf" download Qwen/Qwen3-VL-2B-Instruct >/dev/null  # trainee tokenizer/processor
echo "SETUP_DONE"
