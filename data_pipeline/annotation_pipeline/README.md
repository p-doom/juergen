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
| 02 | `stage_02_observation_view.py` | Join Stage-01 frames with timestamped events at the configured FPS and apply idle thinning. Emits structured `action_bin` values but no action strings. |
| 03 | `stage_03_annotate.py` | Run vision-only describe and goal-extraction passes over the observation view. Emits visual goal proposals. |
| 04 | `stage_04_refine_boundaries.py` | Apply an explicit `vision_only` or `keylog_refined` boundary policy and emit finalized frame bounds. |
| 05 | `stage_05_assemble_trajectories.py` | Slice the Stage-02 observation view into structured, message-format-neutral trajectories. |
| 06 | `stage_06_project_sft.py` | Project Stage-05 trajectories into the current image/action SFT message format, split the dataset, and optionally apply prompt/terminal policy. |

`build_manifest.py` discovers source segments before Stage 00. `run_dataset.py`
orchestrates Stages 01–04, and `build_sft.py` runs Stages 05–06 over the same
Stage-02 observation view.

## Action ownership

Stage 01 preserves ordered raw events. Stage 02 currently projects each
observation interval into the existing aggregate action structure:

```json
{"move_dx": 10.0, "move_dy": -2.0, "scroll": 0.0, "events": [["+", "LMB"], ["-", "LMB"]]}
```

Stage 05 retains both the ordered events and this structured aggregate. Only
Stage 06 renders action text. Its default `ordered_events_v2` schema emits an
ordered mini-program with these primitives:

```text
move(dx,dy); scroll(dx,dy); down(INPUT); up(INPUT)
```

For example, movement on both sides of a click remains ordered:

```text
move(4,-1); down(LMB); move(6,-1); up(LMB)
```

Continuous events are accumulated on a configurable internal motor grid whose
default is 10 Hz. The grid does not add screenshots or assistant turns, and a
discrete transition always splits the surrounding continuous events even when
they occur in the same motor tick. Zero-valued `move` and `scroll` primitives
are omitted. A turn with no remaining primitives is `NO_OP`.

Select the old aggregate projection explicitly with
`--action-schema aggregate_delta_keys_v1`. Coordinate normalization,
quantization, mouse scaling, evaluator parsing, and runtime execution of v2 are
not implemented by this data-pipeline change.

## Full run

From `data_pipeline/`:

```bash
PYTHONPATH=. python3 -m annotation_pipeline.build_manifest \
  --dataset-root /path/to/uploads \
  --out manifest.jsonl --workers 32

PYTHONPATH=. python3 -m annotation_pipeline.run_dataset \
  --manifest manifest.jsonl \
  --run-name full \
  --models Kimi-K2.6,Kimi-K2.5 \
  --base-fps 0.5 \
  --observation-fps 0.5 \
  --target-tpm 1800000 \
  --max-workers 64

PYTHONPATH=. python3 -m annotation_pipeline.build_sft \
  --run-dir annotation_pipeline/dataset_runs/full \
  --out annotation_pipeline/dataset_runs/full/sft \
  --action-schema ordered_events_v2 \
  --continuous-action-hz 10
```

To generate the aggregate action-format ablation from the same Stage-05
trajectories, rerun only Stage 06 through `build_sft.py` with:

```bash
--action-schema aggregate_delta_keys_v1
```

The run layout is:

```text
<run>/
  _modalities/clips/<segment>/
    stage_00/manifest.jsonl
    stage_01_base/{frames,events}.jsonl
    stage_02_view/observations.jsonl
  <model>/clips/<unit>/
    stage_02_view/observations.jsonl
    stage_03_annotation/{annotation.json,goal_proposals.jsonl}
    stage_04_boundaries/goals.jsonl

<sft-output>/
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
