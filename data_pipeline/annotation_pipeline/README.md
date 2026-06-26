# Annotation pipeline — hindsight instruction annotation

Turns human screen recordings (MP4 + msgpack keylog) into instruction-annotated
computer-use SFT data. Each sample is `(instruction, trajectory)` where the
**instruction** is the prompt a user would type to make an agent do this and the
**trajectory** is the screen→action sequence the human performed. We recover the
prompt in **hindsight** (label the goal the trajectory actually achieved).

Labeler: any OpenAI-compatible VLM, selected by env (`LABELER_MODEL`, default
`Kimi-K2.6`). The full-dataset driver (`run_dataset.py`) annotates with
**Kimi-K2.6 + Kimi-K2.5 in parallel** (`--models Kimi-K2.6,Kimi-K2.5`), both
served off the same Azure `mihir-4710` `/openai/v1/` surface and passed by name;
a closed-loop TPM governor routes each window-unit to whichever model has the
most headroom (per-model run dirs, so caches never cross models). Repoint to a
local model to distill. No sglang / no GPU needed for annotation.

## Pipeline

```
stage_01_frames_actions   raw MP4 → 1fps/720p frames (grain store) + per-frame
                          action strings from the keylog. Idle is thinned by a
                          NO_OP head/tail keep (first 3 + last 3 of each idle run,
                          so a wait's start and end stay visible).
keylog_transcript         keylog → typed text + chords + click/scroll bursts + app
                          (exact ground truth for what was typed; fused into stage 02)
stage_02_annotate         A perceive  frames+transcript → timeline
                          B segment   timeline → goal-coherent intervals
                          C label      hindsight user-prompt instruction + variants
                          D verify+repair  achieved/monotonic/grounded/tight/register;
                                           trim+relabel loose intervals, else drop
stage_03_assemble         verified trajectories → SFT rows (variants fanned out)
stage_04_build_canonical  portable canonical SFT JSONL (+ split/sample manifests)
stage_05_length_buckets   optional token/length bucket inspector
```

Frames go to the VLM **clean** (no burned-in overlay); per-frame timestamps are
passed as **interleaved text** (`original_t=..s` before each image). Stage 01
stores frames as one `images.array_record` (grain) per segment, referenced as
`ar://…#idx`.

Supporting modules: `config`, `common`, `image_store` (grain), `frames_render`
(VLM frame sampling/render), `labeler` (VLM client), `prompts` + `prompts.yaml`
(all prompt text), `qwen3_encoding` (stage 04/05 token counts).

## Iterate

`run_iteration.py` runs the loop on a fixed curated clip set
(`iteration_clips.json`) and writes an HTML review + an independent LLM judge.

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
# CPU-only slurm job (keeps ffmpeg off the login node), resumable:
bash -lc 'sbatch annotation_pipeline/slurm/run_iteration.sbatch clean'
# outputs: annotation_pipeline/iteration_runs/clean/{review.html, judge.json}
```

`run_iteration` is resumable (done clips are skipped) and stage 02 caches every
labeler response, so a prompt edit re-spends only the changed step:
`--refresh verify` (or `label,verify`) re-runs just those calls.

## Inspect

```bash
PYTHONPATH=. python3 -m annotation_pipeline.visualize_run --port 8910
# ssh -L 8910:127.0.0.1:8910 <node>  then open http://127.0.0.1:8910/
```

Per clip: the raw recording; the **kept (NO_OP-capped) frame stream** as a player;
every VLM call (perceive/segment/label/verify) with the exact frames sent, the
prompt, and the raw response; and the final instructions, each with its trajectory
played back from the actual sample frames. Reads straight from the grain store.

## Environment

The labeler is configured by env (see `labeler.py`): `LABELER_MODEL`,
`LABELER_BASE_URL` (defaults to `$AZURE_OPENAI_ENDPOINT`), `LABELER_API_KEY`
(defaults to `$AZURE_OPENAI_API_KEY`), `LABELER_REASONING_EFFORT`. ffmpeg via
`$JUERGEN_ANNOTATION_FFMPEG_BIN` or PATH. Deps are in the `data_pipeline`
`pyproject.toml` (`array-record` for grain, `PyYAML` for prompts, `openai`,
`opencv-python-headless`, `msgpack`).
