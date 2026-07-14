# Hindsight annotation pipeline

This pipeline turns human screen recordings and raw input logs into structured
computer-use trajectories, then projects those trajectories into SFT messages.
The stages have strict contracts; outputs from the previous pipeline layout are
not accepted.

## Stages

| Stage | Entry point | Contract |
| --- | --- | --- |
| 00 | `stage_00_realign.py` | Optionally realign raw keylog timestamps to video time and emit a corrected segment manifest. |
| 01 | `stage_01_base_modalities.py` | Extract frames at `base_fps` and normalize the raw keylog into one ordered, unbinned `events.jsonl` timeline. |
| 02 | `stage_02_observation_view.py` | Create a named FPS/idle-thinning view by joining Stage-01 frames with timestamped events. Emits structured `action_bin` values but no action strings. |
| 03 | `stage_03_annotate.py` | Run vision-only describe and goal-extraction passes over the annotation view. Emits visual goal proposals. |
| 04 | `stage_04_refine_boundaries.py` | Apply an explicit `vision_only` or `keylog_refined` boundary policy. Emits half-open timestamp bounds that transfer across observation FPS values. |
| 05 | `stage_05_assemble_trajectories.py` | Slice the selected training view into structured, message-format-neutral trajectories. |
| 06 | `stage_06_project_sft.py` | Project Stage-05 trajectories into the current image/action SFT message format, split the dataset, and optionally apply prompt/terminal policy. |

`build_manifest.py` discovers source segments before Stage 00. `run_dataset.py`
orchestrates Stages 01–04, and `build_sft.py` materializes a training view and
runs Stages 05–06.

## Independent annotation and training FPS

Stage 01 should be run at the highest FPS needed by any downstream ablation.
Stage 02 can then create multiple views without decoding the video again:

- the annotation view, normally 0.5 FPS, is consumed by Stages 03–04;
- the training view is materialized by `build_sft --training-fps ...` and is
  consumed by Stage 05.

Stage-04 goals retain their annotation-frame indices for auditability and also
carry `[start_time_s, end_time_s)` bounds. Stage 05 uses those timestamps when
slicing the training view, so changing training FPS does not require another
VLM annotation pass. Both requested FPS values must divide `base_fps` exactly.

Idle thinning is independently configured for annotation and training views.
Keep those settings fixed when the intended ablation variable is FPS alone.

## Action ownership

Stage 01 preserves ordered raw events. Stage 02 currently projects each
observation interval into the existing aggregate action structure:

```json
{"move_dx": 10.0, "move_dy": -2.0, "scroll": 0.0, "events": [["+", "LMB"], ["-", "LMB"]]}
```

Stage 05 retains both the ordered events and this structured aggregate. Only
Stage 06 renders the current action text (`<dx> <dy> <scroll> ; ...` or
`NO_OP`). Mouse scaling and a new action representation are intentionally not
implemented here.

## Full run

From `data_pipeline/`:

```bash
PYTHONPATH=. python3 -m annotation_pipeline.build_manifest \
  --dataset-root /path/to/uploads \
  --out manifest.jsonl --workers 32

# Choose base_fps high enough for every later training-FPS ablation.
PYTHONPATH=. python3 -m annotation_pipeline.run_dataset \
  --manifest manifest.jsonl \
  --run-name full \
  --models Kimi-K2.6,Kimi-K2.5 \
  --base-fps 2 \
  --annotation-fps 0.5 \
  --target-tpm 1800000 \
  --max-workers 64

PYTHONPATH=. python3 -m annotation_pipeline.build_sft \
  --run-dir annotation_pipeline/dataset_runs/full \
  --out annotation_pipeline/dataset_runs/full/sft_1fps \
  --training-fps 1
```

The run layout is:

```text
<run>/
  _modalities/clips/<segment>/
    stage_00/manifest.jsonl
    stage_01_base/{frames,events}.jsonl
    stage_02_views/annotation/observations.jsonl
  <model>/clips/<unit>/
    stage_02_annotation_view/observations.jsonl
    stage_03_annotation/{annotation.json,goal_proposals.jsonl}
    stage_04_boundaries/goals.jsonl

<sft-output>/
  stage_02_training_views/clips/<segment>/observations.jsonl
  stage_05_trajectories/trajectories.jsonl
  stage_06_sft/{chat.jsonl,train/chat.jsonl,val/chat.jsonl}
```

Large segments are split into annotation window-units. Window cuts prefer
submission boundaries or real time gaps, and each non-final window receives a
small visual look-ahead buffer. Stage 05 merges window goals back into their
parent segment and assigns parent-wide unique goal IDs.

`run_dataset --phase prepare` performs Stages 01–02 without API calls;
`--phase annotate` performs Stages 03–04 from prepared artifacts. Progress is
tracked separately per phase and shard.

## Boundary-policy ablation

Choose the Stage-04 policy when running annotation:

```bash
PYTHONPATH=. python3 -m annotation_pipeline.run_dataset \
  ... --boundary-policy vision_only
```

`vision_only` retains VLM bounds. `keylog_refined` moves typed-goal starts to
the beginning of the detected typing burst. The annotation prompt and action
representation are otherwise identical.

## Prompt iteration

Stage 03 caches the describe and extract responses. To rerun only extraction
and then regenerate Stage-04 bounds:

```bash
PYTHONPATH=. python3 -m annotation_pipeline.reextract_run \
  --run-dir annotation_pipeline/dataset_runs/full \
  --concurrency 8
```

Inspect a model directory with:

```bash
PYTHONPATH=. python3 -m annotation_pipeline.visualize_run \
  --run-root annotation_pipeline/dataset_runs/full --port 8765
```

The viewer reads the Stage-02 annotation view, Stage-03 responses, and finalized
Stage-04 goals directly from the new artifacts.
