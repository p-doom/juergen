# crowdcast-data-pipeline

pmanager/labctl launch configs and tests for the crowd-cast SFT data pipeline.

**The stages themselves live at the repo root, in [`pipeline/`](../pipeline).**
This directory holds only the two chain configs that schedule them and the test
suite that covers them; it exists as a separate uv project because those tests
need `opencv-python-headless`, which the training venv deliberately does not
carry.

Consumed by [`pmanager`][pmanager]/[`labctl`][labctl] recipes that inject params
as `--flag=value` args and poll `<output_dir>/manifest.json` for stage
completion.

[pmanager]: https://github.com/anthropics/pmanager
[labctl]: https://github.com/anthropics/labctl

## Layout

### Chains (`configs/`)

| Config | Stages |
| --- | --- |
| `chain_annotate.py` | `pipeline/stage_03_filter.py` → `pipeline/annotation/stage_annotate.py` (goal annotation at K fps) |
| `chain_train.py` | `pipeline/stage_03_filter.py` → `pipeline/stage_04_build_conversations.py` → `pipeline/stage_05_measure_lengths.py` → `pipeline/stage_06_training_records.py` |

Each stage is nested as its parent's `on_complete` child, so launching the head
runs the whole chain. `chain_train` imports `stage_03_filter` from
`chain_annotate`, which makes that one function the single place a dataset
family is named (`MASTER_DIR` / `CLIPS_MANIFEST`).

Both configs derive `PROJECT_REPO` from their own location (override with
`JUERGEN_REPO`) and assert a root `pipeline/` directory exists, so a
pre-rearchitecture checkout fails at config-build time rather than after the job
is already scheduled.

### Tests (`tests/`)

Cover `pipeline.*` — the action formatter, filter, views, dead zones, goal
projection, annotation-method registry, and the stage-04 conversation builder.

> Dataset browsing (contributor index, day-grouped segments, timeline heatmap,
> frame-by-frame viewer) lives in the labctl UI's artifact panel — open any
> `dataset` artifact and use the **Browse** section.

## Setup

```bash
uv sync  # creates .venv with msgpack, Pillow, opencv, array-record, ...
```

`pipeline/stage_05_measure_lengths.py` and `stage_06_training_records.py`
subprocess-wrap omegalax scripts, which run in omegalax's own venv via
`uv run --project <omegalax_repo>`.

## Running

```bash
pmanager launch <checkout>/data_pipeline/configs/chain_annotate.py
pmanager launch <checkout>/data_pipeline/configs/chain_train.py
```

Both are STUBs: the dataset paths at the top of each file name one specific
generation and are asserted to exist at `get_config()`. Point them at your own
family before launching.

## Output contract

Every stage entrypoint writes `<output_dir>/manifest.json` before exiting (per
`pipeline_task()` in `pmanager.configs.schema`). pmanager polls for this file to
mark the dataset complete and register it. The schema captures the stage name,
every flag the entrypoint received, input fingerprints (paths + key file
hashes), and output statistics.

## Development

```bash
uvx ruff check .       # lint
uvx ruff format .      # format
uv run pytest tests    # 89 tests
```

`pyproject.toml` carries the strict rule set (pycodestyle, pyflakes, isort,
bugbear, pyupgrade, simplify, ruff, pylint, tidy-imports, use-pathlib, return,
comprehensions, pep8-naming).
