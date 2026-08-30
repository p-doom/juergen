# Crowd-Cast data pipeline

This workspace converts screen recordings and input logs into chat records and
ArrayRecord training data. It also contains annotation, replay-source, and
post-processing utilities used by the same dataset build.

## Pipelines

The four-stage pipeline is:

1. `stage_a_prepare.py` extracts frames and actions.
2. `stage_b_run_length_cap.py` bounds idle runs.
3. `stage_c_grain_payload.py` compiles chat records into Grain ArrayRecord.
4. `stage_d_chunk_index.py` builds the offline chunk index.

`realigned_pipeline/` provides the recording-alignment flow.
`annotation_pipeline/` builds hindsight instructions and canonical SFT records.
The `prep_*`, `generate_*`, and `filter_*` entrypoints prepare auxiliary
corpora or filter generated rows.

## Setup and checks

Run from `data_pipeline/`:

```bash
uv sync
uv run pytest tests
uvx ruff check .
uvx ruff format --check .
```

## Dispatch

The checked-in pmanager configs resolve their repository paths from the config
file. Pass pmanager an absolute config path:

```bash
pmanager launch "$(realpath configs/chain_smoke.py)"
pmanager launch "$(realpath configs/chain_v1.py)"
```

Each scheduled stage writes `<output_dir>/manifest.json` before it is considered
complete.
