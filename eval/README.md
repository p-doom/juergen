# Crowd-Cast evaluation

This workspace contains IFEval and OSWorld runners, SGLang process management,
and result aggregation. Dispatch systems pass parameters to the entrypoints and
consume their `result.json` files.

## Setup

```bash
uv sync
```

OSWorld runners require an upstream checkout:

```bash
export OSWORLD_ROOT=/absolute/path/to/OSWorld
```

## Entrypoints

- `ifeval.py` runs IFEval against an HF/SGLang-loadable model.
- `roundtrip_ifeval.py` exports a checkpoint before running IFEval.
- `osworld_one_task_runner.py` runs one desktop task.
- `osworld_fullbench_runner.py` maps one task to each Slurm array index.
- `osworld_score.py` aggregates completed task results.
- `bc_offline_score.py` and `bc_roundtrip.py` score recorded behavior-cloning
  outputs.

Each entrypoint exposes its required paths and runtime settings through
`--help`. Model artifacts and output directories must be supplied by the
caller; the repository does not select a production model.

Successful runners write `result.json` atomically through `result.write_result`.
The record includes task identity, metrics, input parameters, sample count,
elapsed time, and dispatch identifiers when present.

## Checks

```bash
uvx ruff check .
uvx ruff format --check .
```
