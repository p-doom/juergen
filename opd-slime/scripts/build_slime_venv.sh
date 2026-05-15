#!/usr/bin/env bash
# Idempotent slime build-stack constructor.
#
# Usage: bash build_slime_venv.sh --out_dir=<env_root>
#
# (labctl appends recipe [args] as --key=value flags after a `--` separator,
# so this script reads its arg in that form rather than positionally.)
#
# Produces, inside <env_root>:
#   venv/             # uv venv with all CUDA build extensions installed
#   slime/            # clone of THUDM/slime at $SLIME_SHA (editable-installed into venv)
#   Megatron-LM/      # clone of NVIDIA/Megatron-LM at $MEGATRON_SHA, patched with
#                     # slime/docker/patch/$PATCH_VERSION/megatron.patch (editable-installed)
#   sglang/           # clone of sgl-project/sglang at $SGLANG_SHA, patched with sglang.patch (editable-installed)
#   apex/             # clone of NVIDIA/apex at $APEX_SHA, patched with apex.patch, built+installed as a wheel
#   manifest.json     # runtime-captured provenance: all SHAs, patch sha256s, key env vars
#   .ready            # marker; written LAST after everything succeeds
#
# We build DIRECTLY at the final location (not via a .partial sibling +
# rename) because uv venvs are not relocatable — the activate script and
# every .pth file in site-packages bake in absolute paths to their venv
# root, so renaming breaks the venv. The cost is that a crashed mid-build
# leaves partial state in <env_root> without `.ready`; the next submission
# of this recipe will see the missing marker, proceed, and the clone/patch/
# install steps are idempotent by design (git apply --check, uv pip install
# audit-if-present).
#
# Required env vars set by the recipe / labctl:
#   UV_CACHE_DIR        — shared uv wheel cache (skip pre-compiled deps on warm cache)
#   HF_HOME             — HuggingFace cache root
#   CUDA_HOME           — usually /usr/local/cuda-12
#   PATH                — must include nvcc + uv

set -euxo pipefail

ENV_ROOT=""
for arg in "$@"; do
    case "$arg" in
        --out_dir=*) ENV_ROOT="${arg#*=}" ;;
        --) ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done
: "${ENV_ROOT:?--out_dir=<env_root> required}"
ENV_ROOT="$(realpath "$ENV_ROOT")"

# Pinned SHAs — bumping any of these invalidates labctl provenance (the file's
# git_diff captures the change) and the colleague's next pipeline run
# rebuilds against the new combo.
SGLANG_SHA=bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2
MEGATRON_SHA=3714d81d418c9f1bca4594fc35f9e8289f652862
MBRIDGE_SHA=89eb10887887bc74853f89a4de258c0702932a1c
APEX_SHA=10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
TORCH_MEM_SAVER_SHA=dc6876905830430b5054325fa4211ff302169c6b
SLIME_SHA=41dc3b6d21d3c75b212965077a1cc4117932f06d   # bump when slime upstream advances and we've verified the patches still apply
PATCH_VERSION=v0.5.9

# Build-only env vars.
export TORCH_CUDA_ARCH_LIST="9.0a"             # H100-only; halves build time vs default arch list
export STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1    # nvidia-resiliency-ext needs libcupti — skip the CUPTI ext build
export NVCC_APPEND_FLAGS="--threads 4"

OPD_SLIME_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APEX_PATCH="${OPD_SLIME_ROOT}/patches/apex.patch"

# Build at the final location (uv venvs aren't relocatable, see header).
mkdir -p "$ENV_ROOT"
cd "$ENV_ROOT"
# If a previous attempt left a stale .ready, drop it so we don't claim done
# before this run actually completes.
rm -f "$ENV_ROOT/.ready"

# --- 1. Clones at pinned SHAs --------------------------------------------
clone_at_sha() {
    local url="$1" dir="$2" sha="$3"
    if [ ! -d "$dir/.git" ]; then
        git clone "$url" "$dir"
    fi
    (cd "$dir" && git fetch --depth=1 origin "$sha" && git checkout "$sha")
}
clone_at_sha https://github.com/THUDM/slime.git slime "$SLIME_SHA"
clone_at_sha https://github.com/NVIDIA/Megatron-LM.git Megatron-LM "$MEGATRON_SHA"
clone_at_sha https://github.com/sgl-project/sglang.git sglang "$SGLANG_SHA"
clone_at_sha https://github.com/NVIDIA/apex.git apex "$APEX_SHA"
(cd Megatron-LM && git submodule update --init --recursive)
(cd apex && git submodule update --init --recursive)

# --- 2. Apply patches idempotently ---------------------------------------
# Slime-shipped patches against Megatron-LM and sglang. We use `git apply
# --check` first to detect a pre-applied state and skip silently.
apply_idempotent() {
    local dir="$1" patch="$2"
    # patch path must be absolute since we cd into $dir before invoking git
    patch="$(realpath "$patch")"
    (cd "$dir" && {
        if git apply --check --reverse "$patch" >/dev/null 2>&1; then
            echo "[$dir] patch $patch already applied — skip"
        elif git apply --check "$patch" >/dev/null 2>&1; then
            git apply "$patch"
            echo "[$dir] patch $patch applied"
        else
            echo "[$dir] patch $patch neither applies cleanly nor is already applied — bailing" >&2
            exit 1
        fi
    })
}
apply_idempotent Megatron-LM "${ENV_ROOT}/slime/docker/patch/${PATCH_VERSION}/megatron.patch"
apply_idempotent sglang      "${ENV_ROOT}/slime/docker/patch/${PATCH_VERSION}/sglang.patch"

# Apex CUDA-version-check bypass — our cluster has nvcc 12.8 but torch
# wheels are built against 12.9, and apex's setup.py refuses minor-mismatch.
# The patch is a one-liner: replace the check call with `pass`.
apply_idempotent apex "$APEX_PATCH"

# --- 3. Create venv + install deps ---------------------------------------
if [ ! -d venv ]; then
    uv venv --python 3.12 venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

UVPIP="uv pip install"

# torch first (cu129) so subsequent CUDA-ext builds find correct libs.
$UVPIP --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
$UVPIP cuda-python==12.9 cmake ninja packaging wheel
$UVPIP "numpy<2"
$UVPIP -e "sglang/python[all]"
MAX_JOBS=32 $UVPIP --no-build-isolation flash-attn==2.7.4.post1
$UVPIP --no-deps "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_SHA}"
$UVPIP --no-build-isolation "transformer_engine[pytorch]==2.10.0"
$UVPIP flash-linear-attention==0.4.1

# Apex: built via pip (uv pip lacks --build-option for setup.py's --cpp_ext / --cuda_ext).
$UVPIP pip   # ensure pip is in the venv for the next step
NVCC_APPEND_FLAGS="--threads 4" python -m pip install \
    --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    ./apex

$UVPIP "git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl"
$UVPIP --no-cache-dir --force-reinstall \
    "git+https://github.com/fzyzcjy/torch_memory_saver.git@${TORCH_MEM_SAVER_SHA}"
$UVPIP --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"
$UVPIP --force-reinstall \
    https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
$UVPIP -e Megatron-LM
$UVPIP -e slime
$UVPIP nvidia-cudnn-cu12==9.16.0.29

# --- 4. Smoke import check -----------------------------------------------
PYTHONPATH="${ENV_ROOT}/Megatron-LM" python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)
import sglang; import flash_attn; import megatron; import megatron.bridge
from megatron.bridge import AutoBridge  # noqa: F401
import transformer_engine; import slime  # noqa: F401
print("imports ok")
PY

# --- 5. Manifest + .ready marker -----------------------------------------
python - <<PY
import hashlib, json, os, pathlib, subprocess
root = pathlib.Path("${ENV_ROOT}")
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
manifest = {
    "schema_version": 1,
    "pinned_shas": {
        "slime": "${SLIME_SHA}",
        "Megatron-LM": "${MEGATRON_SHA}",
        "sglang": "${SGLANG_SHA}",
        "apex": "${APEX_SHA}",
        "mbridge": "${MBRIDGE_SHA}",
        "torch_memory_saver": "${TORCH_MEM_SAVER_SHA}",
    },
    "patches": {
        "megatron.patch": sha256(root / "slime/docker/patch/${PATCH_VERSION}/megatron.patch"),
        "sglang.patch":   sha256(root / "slime/docker/patch/${PATCH_VERSION}/sglang.patch"),
        "apex.patch":     sha256("${APEX_PATCH}"),
    },
    "build_env": {
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD": os.environ.get("STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD"),
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
    },
    "uv_pip_freeze": subprocess.check_output(["uv", "pip", "freeze"], cwd=str(root)).decode("utf-8"),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
PY

touch "$ENV_ROOT/.ready"

echo "=== build complete: $ENV_ROOT ==="
