# crowdcast-data-pipeline

SFT data pipeline for crowd-cast: turns raw S3-synced screen recordings + keylogs into omegalax-ingestable Grain ArrayRecord shards.

Consumed by [`pmanager`][pmanager]/[`labctl`][labctl] recipes that inject params via `absl` flags and poll `<output_dir>/manifest.json` for stage completion.

[pmanager]: https://github.com/anthropics/pmanager
[labctl]: https://github.com/anthropics/labctl

## Layout

### Crowd-cast 4-stage chain (`stage_*`)

| Stage | Script | Role |
| --- | --- | --- |
| A | `stage_a_prepare.py` | S3 sync → per-segment JPEG frames at target fps/height + per-frame action strings + per-split `chat.jsonl`. Pass `--image_store_format=arrayrecord` to store each segment's JPEGs as records in `images.array_record` and emit `ar:///...#idx` image refs instead of `frames/frame_*.jpg` files. |
| B | `stage_b_run_length_cap.py` | Cap NO_OP runs (`k = round(k_seconds · target_fps)`). Rewrites `chat_line.json` only — frames are referenced in place. |
| C | `stage_c_grain_payload.py` | Compile per-split `chat.jsonl` → Grain ArrayRecord shards. Wraps `omegalax/scripts/compile_sft_dataset.py` (one subprocess per split, in omegalax's uv venv). |
| D | `stage_d_chunk_index.py` | Build the offline chunk index from the Grain payload. Wraps `omegalax/scripts/build_sft_chunk_index.py`. |

Chain configs in `configs/chain_v1.py` (full) and `configs/chain_smoke.py` (10 segments — end-to-end pmanager validation without burning ~17 h of cluster time).

### Misc

| Script | Role |
| --- | --- |
| `_manifest.py` | `write_manifest()` — shared helper for the `<output_dir>/manifest.json` completion marker that pmanager polls for. |

> Dataset browsing (contributor index, day-grouped segments, timeline heatmap, frame-by-frame viewer) lives in the labctl UI's artifact panel — open any `dataset` artifact and use the **Browse** section.

## Setup

```bash
uv sync  # creates .venv with msgpack, Pillow, opencv, ...
```

Scripts that run in *other* venvs (see `pyproject.toml` notes):

| Script(s) | Venv |
| --- | --- |
| `stage_c_*`, `stage_d_*` | omegalax — `uv run --project <omegalax_repo>` |

## Running

### The chain (via pmanager)

```bash
pmanager launch /fast/home/franz.srambical/data_pipeline/configs/chain_smoke.py   # 10-segment smoke
pmanager launch /fast/home/franz.srambical/data_pipeline/configs/chain_v1.py      # full
```

Each stage's `cfg.children` triggers the next on `on_complete`. To launch a single stage standalone, point pmanager at e.g. `configs/stage_a_v1_5fps_360p.py`.

## Output contract

Every stage entrypoint writes `<output_dir>/manifest.json` before exiting (per `pipeline_task()` in `pmanager.configs.schema`). pmanager polls for this file to mark the dataset complete and register it. Schema is owned by `_manifest.write_manifest()` and captures: stage name, every flag the entrypoint received, input fingerprints (paths + key file hashes), and output statistics.

## Development

```bash
uvx ruff check .       # lint
uvx ruff format .      # format
```

`pyproject.toml` carries the strict rule set (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, ruff, pylint, tidy-imports, use-pathlib, return, comprehensions, pep8-naming).
