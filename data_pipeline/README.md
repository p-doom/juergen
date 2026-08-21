# crowdcast-data-pipeline

labctl/pmanager launch configs and tests for the crowd-cast SFT data pipeline.
The stages themselves live at the repo root, in [`pipeline/`](../pipeline); this
directory holds the chain configs that schedule them and the tests that cover
them. It is a separate uv project because those tests need
`opencv-python-headless`, which the training venv does not carry.

A recipe injects params as `--flag=value` args and polls
`<output_dir>/manifest.json` for stage completion.

## Chains (`configs/`)

| Config | Stages |
| --- | --- |
| `chain_annotate.py` | `pipeline/stage_03_filter.py` → `pipeline/annotation/stage_annotate.py` (goal annotation at K fps) |
| `chain_train.py` | `pipeline/stage_03_filter.py` → `stage_04_build_conversations.py` → `stage_05_measure_lengths.py` → `stage_06_training_records.py` |

Each stage is nested as its parent's `on_complete` child, so launching the head
runs the whole chain. `chain_train` imports `stage_03_filter` from
`chain_annotate`, which makes that one function the single place a dataset family
is named (`MASTER_DIR` / `CLIPS_MANIFEST`).

Both configs derive `PROJECT_REPO` from their own location (override with
`JUERGEN_REPO`) and assert a root `pipeline/` directory exists, so a checkout
without one fails at config-build time rather than after the job is scheduled.

## Running

```bash
pmanager launch <checkout>/data_pipeline/configs/chain_annotate.py
pmanager launch <checkout>/data_pipeline/configs/chain_train.py
```

Both are STUBs: the dataset paths at the top of each file name one specific
generation and are asserted to exist at `get_config()`. Point them at your own
family before launching.

`pipeline/stage_05_measure_lengths.py` and `stage_06_training_records.py`
subprocess-wrap omegalax scripts, which run in omegalax's own venv via
`uv run --project <omegalax_repo>`.

Dataset browsing (contributor index, day-grouped segments, timeline heatmap,
frame-by-frame viewer) is not here — it lives in the labctl UI's artifact panel,
under any `dataset` artifact's Browse section.

## Output contract

Every stage entrypoint writes `<output_dir>/manifest.json` before exiting (per
`pipeline_task()` in `pmanager.configs.schema`); the launcher polls for that file
to mark the dataset complete and register it. The schema captures the stage name,
every flag the entrypoint received, input fingerprints (paths plus key file
hashes), and output statistics.

## Development

`pmanager` is loaded by the outer `pmanager launch` process, not by this venv;
the config tests stub it.

```bash
uv sync
uv run pytest tests
uvx ruff check . && uvx ruff format .
```
