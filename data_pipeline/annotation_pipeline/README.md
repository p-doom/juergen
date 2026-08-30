# Hindsight annotation pipeline

The annotation pipeline turns screen recordings and input logs into instruction
and trajectory pairs for computer-use SFT. Frames are stored once in
ArrayRecord and reused by annotation and assembly.

## Stages

| stage | output |
| --- | --- |
| `build_manifest` | validated recording/keylog manifest |
| `stage_00_realign` | optional corrected keylogs and manifest |
| `stage_01_frames_actions` | sampled frames and aligned actions |
| `stage_02_annotate` | factual narration and bounded instructions |
| `stage_02b_plans` | optional cached plan text |
| `stage_03_assemble_trajectories` | instruction/trajectory samples |
| `stage_04_build_canonical_sft` | portable train and validation chat records |
| `stage_05_length_buckets` | optional length report |

The labeler is any OpenAI-compatible vision endpoint configured with
`LABELER_MODEL`, `LABELER_BASE_URL`, and `LABELER_API_KEY`. Azure-compatible
endpoint and key variables are accepted by `labeler.py`.

## Run

Run from `data_pipeline/` and keep datasets outside the repository:

```bash
PYTHONPATH=. python -m annotation_pipeline.build_manifest \
    --dataset-root /absolute/recordings \
    --out /absolute/manifests/recordings.jsonl

PYTHONPATH=. python -m annotation_pipeline.run_dataset \
    --manifest /absolute/manifests/recordings.jsonl \
    --run-name annotation \
    --out-root /absolute/annotation-runs

PYTHONPATH=. python -m annotation_pipeline.build_sft \
    --run-dir /absolute/annotation-runs/annotation \
    --out /absolute/sft-output
```

`run_dataset` records progress per work unit and supports explicit phase and
shard selection. `build_sft` groups windows by their parent recording before
splitting train and validation records.

Use `visualize_run.py`, `frame_stepper.py`, or `action_video_viewer.py` for
read-only inspection. The full-day viewer is documented in
`goal_timeline_viewer/README.md`.
