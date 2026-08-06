# Annotation pipeline — hindsight instruction annotation

> **SUPERSEDED — scheduled for deletion.** The current generation is `pipeline/`
> at the repo root (stages 00–06 + `pipeline/annotation/`). **Nothing outside this
> directory imports this package any more.** Two modules were ported out and are
> now thin aliases kept only so recorded labctl `submit.sh` scripts keep
> resolving:
>
> | old | new |
> |---|---|
> | `annotation_pipeline.build_manifest` | `pipeline.stage_00_clip_manifest` (byte-identical) |
> | `annotation_pipeline.stage_00_realign` | `pipeline.stage_02_realign` (byte-identical body) |
> | `annotation_pipeline.stage_04_build_canonical_sft` | `pipeline.stage_04_build_canonical_sft` (byte-identical body) |
>
> What is still real here is the LEGACY ANNOTATION ENGINE — `run_dataset`,
> `stage_02_annotate`, `stage_02b_plans`, `stage_03_assemble_trajectories`,
> `stage_00_realign`, plus `labeler`/`prompts`/`frames_render`/`config`/`common`
> — and its two drivers `build_sft` and `reextract_run`, which only ever run over
> the `dataset_runs/<run>/<model>/clips/<uid>` layout `run_dataset` produces.
> `pipeline/annotation/stage_annotate.py` is the port of that engine
> (`describe_extract` = `stage_02_annotate`, incl. `clean_goals` +
> `snap_goal_starts`; `plans` = `stage_02b_plans`; the prompt packs differ only
> by parameterising the hardcoded 2 s frame period as `${frame_period_s}`), and
> `pipeline/stage_04_build_conversations.py` is the port of
> `stage_03_assemble` + message assembly. Deleting this directory is a decision
> about whether that port is trusted, not a porting task.

Turns human screen recordings (MP4 + msgpack keylog) into instruction-annotated
computer-use SFT data. Each sample is `(instruction, trajectory)` where the
**instruction** is the prompt a user would type to make an agent do this and the
**trajectory** is the screen→action sequence the human performed. We recover the
prompt in **hindsight** (label the goal the trajectory actually achieved).

Labeler: any OpenAI-compatible VLM, selected by env (`LABELER_MODEL`, default
`Kimi-K2.6`). The full-dataset driver (`run_dataset.py`) annotates with
**Kimi-K2.6 + Kimi-K2.5 in parallel** (`--models Kimi-K2.6,Kimi-K2.5`), both
served off the same Azure `mihir-4710` `/openai/v1/` surface and passed by name;
a closed-loop TPM governor (AIMD on measured tokens/min) routes each window-unit
to whichever model has the most headroom (per-model run dirs, so caches never
cross models). Repoint to a local model to distill. No sglang / no GPU needed
for annotation.

## Pipeline

```
build_manifest            walk a raw crowd-cast uploads tree, probe each MP4,
                          pair it with its keylog → per-segment JSONL manifest
                          (PORTED: alias for pipeline.stage_00_clip_manifest)
stage_00_realign          (optional) recover the keylog→video time-map broken by
                          the OBS pause-clock bug; emits corrected keylogs + a
                          realigned manifest that stage 01 consumes unchanged
(stage 01: DELETED — it was a byte-duplicate of the current generation's
 pipeline/lib/frames_actions.py, 8 import lines apart. Decode frames with
 pipeline/stage_01_master_frames.py + pipeline/stage_02_realign.py; the
 downstream annotate/assemble/canonical stages here read the resulting
 frame_records unchanged.)
stage_02_annotate         vision-only, two passes over the clip's kept frames:
                          A describe  all frames → faithful factual prose
                                      narration (no goals/intent)
                          B extract   narration + the same frames (each labelled
                                      `frame <N>`) → the instruction(s) a person
                                      would type to a computer-use agent, each
                                      with start_frame/end_frame bounds, register
                                      variants, anchor + grounding
                          then typed goals' starts are snapped back to the first
                          keystroke of their input burst via the keylog (the
                          vision model anchors ~1 frame late, where text renders)
stage_02b_plans           reason-before-action prose: per window-unit, ONE cached
                          call (narration + goals in time order + each goal's
                          start-frame screenshot) → a 1-2 sentence first-person
                          PLAN per goal, written from the information state at
                          the goal's start (no outcome/clairvoyance, no
                          restatement — situation + method). Reads a run_dataset
                          output (read-only) and writes a mirrored tree that
                          build_sft consumes as its --run-dir
stage_03_assemble         goals + frame bounds → SFT rows: slice
                          [start_frame_idx, end_frame_idx] from frame_records and
                          emit image→action chat turns (instruction on the first
                          user turn; a stage-02b plan prefixes the FIRST
                          assistant turn as `<plan>\n<first action>`); rejects
                          generic/contentless instructions, and plan-less goals
                          under --require-plan. Default one sample per goal;
                          --include-variants fans each paraphrase into its own
                          sample (variants share the goal's plan)
stage_04_build_canonical  portable canonical SFT JSONL (chat.jsonl + split/sample
                          manifests; grouped train/val split; optional system
                          prompt + terminal-token policy; ar:// URIs pass through)
                          (PORTED: alias for
                          pipeline.stage_04_build_canonical_sft)
(stage 05: DELETED — superseded by omegalax's
 measure_message_lengths_from_chat.py / build_sft_records_from_chat.py, which is
 the code path training actually uses. `build_sft.py --buckets` now shells out
 to pipeline/stage_05_measure_lengths.py; pass --omegalax-repo /
 --bucket-model-id.)
```

Frames go to the VLM **clean** (no burned-in overlay, no timestamps). The
describe pass is label-free; only the extract pass interleaves a `frame <N>`
text label before each image so goals can carry frame bounds. Stage 01 stores
frames as one `images.array_record` (grain) per segment, referenced as
`ar://…#idx`; stage 02 feeds the VLM these same stored frames (no re-render).

Supporting modules: `config`, `common` (keylog parsing, action formatting,
message shapes), `image_store` (grain), `frames_render` (frame data-URLs, token
estimate `ceil(h/28)*ceil(w/28)`, window planning), `labeler` (VLM client),
`prompts` + `prompts.yaml` (all prompt text). Token counting is no longer
vendored here: `qwen3_encoding` is deleted, and message serialization / loss
masking now come from the `renderers` library via omegalax.

## Full-dataset run

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen           # repo root, not data_pipeline
python3 -m pipeline.stage_00_clip_manifest \
    --dataset-root /fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-06-18 \
    --out manifest.crowd-cast-2026-06-18.jsonl --workers 32
cd data_pipeline
PYTHONPATH=. python3 -m annotation_pipeline.run_dataset \
    --manifest manifest.crowd-cast-2026-06-18.jsonl \
    --run-name full --models Kimi-K2.6,Kimi-K2.5 \
    --target-tpm 1800000 --max-workers 64
PYTHONPATH=. python3 -m annotation_pipeline.build_sft \
    --run-dir annotation_pipeline/dataset_runs/full \
    --out annotation_pipeline/dataset_runs/full/sft
```

`run_dataset` runs stage 01 once per segment (model-agnostic) into a shared
`<run>/_frames/` dir, then stage 02 per window-unit under `<run>/<model>/clips/`.
A segment is split into `__wN` window-units only when it exceeds the labeler
context budget (`--context-limit` minus completion reserve); cuts snap to a
command submission or real time-gap (never mid typing-burst), and each non-final
window gets a small trailing context buffer whose goals belong to the next
window. Resumable (`progress.jsonl`; finished units are skipped), shardable for
multi-node fan-out (`--shard i/N`), and splittable into a CPU-only frames pass
and an API-only annotate pass (`--phase frames|annotate|all`). CPU-only slurm
wrapper: `slurm/run_dataset_smoke.sbatch` (qos=low, keeps ffmpeg off the login
node).

`build_sft` groups window-units back into their parent segment, concatenates
their goals in window order, feeds the parent's full frame_records through stage
03's `assemble_samples`, then stage 04 → `<out>/canonical/` plus per-split
`<out>/<split>/chat.jsonl` (drop-in source for omegalax stage_c). Every sample
carries provenance (recording_id, clip_id, parent_segment_id, user_id, version,
video_path, frame/time spans, source goal). `--buckets` adds stage 05.

## Iterate on prompts

Stage 02 caches every labeler response per call
(`cache/<name>.txt` + `.reasoning.txt` + `.meta.json`), so a prompt edit
re-spends only the changed step: pass `--refresh extract_from_prose` (or
`describe`, `extract`) to stage 02 to invalidate just that call. To validate an
EXTRACT prompt change over a whole existing run without re-paying for describe:

```bash
PYTHONPATH=. python3 -m annotation_pipeline.reextract_run \
    --run-dir annotation_pipeline/dataset_runs/qc30 --concurrency 8
# or: REEX_RUN_DIR=... sbatch annotation_pipeline/slurm/reextract.sbatch
```

## Plan prose (reason-before-action)

Each SFT sample's first assistant turn is `<plan>\n<first action>` — the agent
states, in one or two first-person sentences, what it is about to do and why,
then acts (format contract in `config.SYSTEM_PROMPT`, byte-identical with
`hindsight_fold/scripts/assemble_sft.py`). Plans are generated by
`stage_02b_plans` as a separate cached call AFTER goal extraction (never inside
extract — re-extracting would re-roll already-QC'd goals), so they can be
backfilled onto any existing annotation run:

```bash
PYTHONPATH=. python3 -m annotation_pipeline.stage_02b_plans \
    --run-root <annotations artifact>/<tag> --out-root <plans out>/<tag> \
    --concurrency 16
PYTHONPATH=. python3 -m annotation_pipeline.build_sft \
    --run-dir <plans out>/<tag> --frames-root <frames artifact>/<tag>/_frames \
    --out <sft out> [--require-plan]
```

Plans carry deterministic quality flags (`restates_instruction`, `empty`, …);
flagged-unusable plans fall back to a plan-less first turn unless
`--require-plan` rejects them.

## Fixing misaligned actions after the fact

`realign_patch_canonical.py` was a ONE-OFF migration script (its own docstring
said so) for canonical artifacts built before the realignment fix; it is
deleted. Use `stage_00_realign` when (re)running the pipeline from scratch on
misaligned recordings — it now imports the single copy of the realignment math
from `pipeline/lib/realign.py` (the duplicate `realign_lib.py` here is gone).

## Inspect

```bash
uv run python tooling/visualize_run.py --port 8765 \
    [--run-root data_pipeline/annotation_pipeline/iteration_runs]
# ssh -L 8765:127.0.0.1:8765 <node>  then open http://127.0.0.1:8765/
```

Per clip: the raw recording (range-streamed); the kept (NO_OP-thinned) frame
stream as a player; the describe pass (prompt, reasoning, raw response,
narration); and the extracted goals (instruction + variants + anchor +
grounding + frame bounds), each playable from the actual sample frames. Reads
`stage_02/stage02_result.json` straight from the grain store.

The viewers/annotator UIs were split out to `tooling/` (repo root):
`tooling/goal_timeline_viewer/` (full-recording goal-hierarchy timeline + the
manual human goal annotation server, see its README),
`tooling/frame_stepper.py` (frame-by-frame action↔screen alignment for one
clip), `tooling/action_video_viewer.py` (realtime video/action overlay, raw
timeline), `tooling/visualize_frame_records.py` (the `pipeline/` frame-record
inspector). `frames_render.py` stayed here — it is a library the annotate
stages import, not a viewer.

## Environment

The labeler is configured by env (see `labeler.py`): `LABELER_MODEL`,
`LABELER_BASE_URL` (defaults to `$AZURE_OPENAI_ENDPOINT`), `LABELER_API_KEY`
(defaults to `$AZURE_OPENAI_API_KEY`), `LABELER_MAX_TOKENS`,
`LABELER_REASONING_EFFORT`. ffmpeg via `$JUERGEN_ANNOTATION_FFMPEG_BIN` or PATH
(`$JUERGEN_ANNOTATION_FFMPEG_THREADS` caps per-decode threads on shared nodes).
Deps are in the `data_pipeline` `pyproject.toml` (`array-record` for grain,
`PyYAML` for prompts, `openai`, `opencv-python-headless`, `msgpack`).
