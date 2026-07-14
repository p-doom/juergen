#!/usr/bin/env bash
#SBATCH --job-name=annotation_smoke
#SBATCH --partition=standard
#SBATCH --account=hfmi_synergyunit
#SBATCH --qos=low
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --time=04:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%j.log

# CPU-only concurrency-ramp smoke test for the full-dataset driver (run_dataset),
# so the login node is never hit with ffmpeg. Builds the 2026-06-18 manifest if
# missing, probes both model names, then lets the TPM governor schedule the run.
# Submit from a login shell so AZURE_OPENAI_* propagate:
#   sbatch annotation_pipeline/slurm/run_dataset_smoke.sbatch

set -eo pipefail
DATA_PIPELINE=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
DATASET_ROOT=/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-06-18
MANIFEST="${SMOKE_MANIFEST:-$DATA_PIPELINE/manifest.crowd-cast-2026-06-18.jsonl}"
export PYTHONPATH="$DATA_PIPELINE"
export JUERGEN_ANNOTATION_FFMPEG_BIN="${JUERGEN_ANNOTATION_FFMPEG_BIN:-/fast/service/apps/software/FFmpeg/7.0.2-GCCcore-13.3.0/bin/ffmpeg}"
# Many ffmpeg decodes can run at once (up to clip_concurrency*models segments);
# keep per-ffmpeg threads modest so 32 cores aren't oversubscribed.
export JUERGEN_ANNOTATION_FFMPEG_THREADS="${JUERGEN_ANNOTATION_FFMPEG_THREADS:-3}"
export LABELER_MAX_TOKENS="${LABELER_MAX_TOKENS:-32000}"

RUN_NAME="${SMOKE_RUN_NAME:-smoke}"
WANT_MODELS="${SMOKE_MODELS:-Kimi-K2.6,Kimi-K2.5}"
LIMIT="${SMOKE_LIMIT:-}"            # blank = whole manifest (full run)
SEED="${SMOKE_SEED:-0}"
TARGET_TPM="${SMOKE_TARGET_TPM:-1800000}"
MAX_WORKERS="${SMOKE_MAX_WORKERS:-64}"
INIT_CALL_S="${SMOKE_INIT_CALL_S:-90}"

if [[ -z "${AZURE_OPENAI_ENDPOINT:-}" || -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  source "$HOME/.bashrc" 2>/dev/null || true
fi

cd "$DATA_PIPELINE"
mkdir -p annotation_pipeline/logs
echo "host=$(hostname)  run=$RUN_NAME  want_models=$WANT_MODELS  limit=${LIMIT:-ALL}  target_tpm=$TARGET_TPM"
"$JUERGEN_ANNOTATION_FFMPEG_BIN" -version 2>/dev/null | head -1 || echo "WARN: ffmpeg not runnable"
python3 -c "from array_record.python.array_record_module import ArrayRecordWriter; print('array_record OK')" || echo "WARN: array_record missing"

# Azure reachability (compute nodes may lack egress -> Stage 03 would fail).
code=$(curl -s -o /dev/null -m 20 -w "%{http_code}" \
  "${AZURE_OPENAI_ENDPOINT%/}/models" -H "api-key: ${AZURE_OPENAI_API_KEY:-none}" 2>/dev/null || echo "000")
echo "azure /models reachability: HTTP $code"

# --- manifest (CPU, free) -------------------------------------------------
if [[ ! -s "$MANIFEST" ]]; then
  echo "building manifest -> $MANIFEST"
  python3 -m annotation_pipeline.build_manifest --dataset-root "$DATASET_ROOT" \
    --out "$MANIFEST" --workers 32
else
  echo "manifest exists: $(wc -l < "$MANIFEST") rows"
fi

# --- probe each model name with one tiny call; keep only the live ones ----
echo "probing model names..."
GOOD_MODELS=$(python3 - "$WANT_MODELS" <<'PY'
import os, sys
from annotation_pipeline.labeler import Labeler, LabelerConfig
good = []
for m in sys.argv[1].split(","):
    m = m.strip()
    if not m:
        continue
    try:
        lab = Labeler(LabelerConfig.from_env(model=m))
        out = lab.call_text("You are concise.", "Reply with the single word OK.", no_cache=True)
        print(f"  probe {m}: OK ({out[:40]!r})", file=sys.stderr)
        good.append(m)
    except Exception as exc:  # noqa: BLE001
        print(f"  probe {m}: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
print(",".join(good))
PY
)
echo "live models: ${GOOD_MODELS:-<none>}"
if [[ -z "$GOOD_MODELS" ]]; then
  echo "ERROR: no model name resolved; aborting before spending tokens." >&2
  exit 1
fi

# --- run: TPM governor self-regulates throughput per model ----------------
python3 -m annotation_pipeline.run_dataset \
  --manifest "$MANIFEST" --run-name "$RUN_NAME" --models "$GOOD_MODELS" \
  --out-root annotation_pipeline/dataset_runs \
  --target-tpm "$TARGET_TPM" --max-workers "$MAX_WORKERS" --init-call-s "$INIT_CALL_S" \
  --shuffle-seed "$SEED" ${LIMIT:+--limit "$LIMIT"} \
  ${SMOKE_IDLE_KEEP_HEAD:+--idle-keep-head "$SMOKE_IDLE_KEEP_HEAD"} \
  ${SMOKE_IDLE_KEEP_TAIL:+--idle-keep-tail "$SMOKE_IDLE_KEEP_TAIL"}
echo "DONE run=$RUN_NAME"
