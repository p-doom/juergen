# crowdcast-data-pipeline

SFT data pipeline for crowd-cast: turns raw S3-synced screen recordings + keylogs into omegalax-ingestable Grain ArrayRecord shards, and prepares replay-source corpora (SmolTalk2, FineVision, Tulu3-Persona-IF) in the same chat-format.

Consumed by [`pmanager`][pmanager]/[`labctl`][labctl] recipes that inject params via `absl` flags and poll `<output_dir>/manifest.json` for stage completion.

[pmanager]: https://github.com/anthropics/pmanager
[labctl]: https://github.com/anthropics/labctl

## Layout

### Realigned crowd-cast chain (`realigned_pipeline/`)

The current chain (stage 00 → 06): decode-once master frame store, keylog↔video
realignment, fps sampling, conversation assembly (per segment, per same-app run, or
per goal), tokenizer cache, inline SFT records. Documented in
[`realigned_pipeline/README.md`](realigned_pipeline/README.md), which also covers
**application filtering** — selecting or splitting conversations by the foreground
app recorded in the keylog.

### Crowd-cast 4-stage chain (`stage_*`)

| Stage | Script | Role |
| --- | --- | --- |
| A | `stage_a_prepare.py` | S3 sync → per-segment JPEG frames at target fps/height + per-frame action strings + per-split `chat.jsonl`. Pass `--image_store_format=arrayrecord` to store each segment's JPEGs as records in `images.array_record` and emit `ar:///...#idx` image refs instead of `frames/frame_*.jpg` files. |
| B | `stage_b_run_length_cap.py` | Cap NO_OP runs (`k = round(k_seconds · target_fps)`). Rewrites `chat_line.json` only — frames are referenced in place. |
| C | `stage_c_grain_payload.py` | Compile per-split `chat.jsonl` → Grain ArrayRecord shards. Wraps `omegalax/scripts/compile_sft_dataset.py` (one subprocess per split, in omegalax's uv venv). |
| D | `stage_d_chunk_index.py` | Build the offline chunk index from the Grain payload. Wraps `omegalax/scripts/build_sft_chunk_index.py`. |

Chain configs in `configs/chain_v1.py` (full) and `configs/chain_smoke.py` (10 segments — end-to-end pmanager validation without burning ~17 h of cluster time).

### Replay-source preps (`prep_*`)

| Script | Source | Notes |
| --- | --- | --- |
| `prep_smoltalk2.py` | `HuggingFaceTB/smoltalk2` (SFT) | Pass-through of HF `messages`; stashes sub-corpus tag under `_source`. |
| `prep_finevision.py` | `HuggingFaceM4/FineVision` | Stratified-sample N configs; materializes PIL images to disk and rewrites turns with inline `{"type":"image","url":...}` blocks (omegalax `qwen3_encoding.py` contract). |
| `prep_tulu3_persona_if.py` | `allenai/tulu-3-sft-personas-instruction-following` | Verbatim `messages`; stashes `constraints` under `_constraints`. |

### On-policy completion gen (runs in the eval venv)

| Script | Role |
| --- | --- |
| `generate_onpolicy_completions.py` | Text-only: regenerates assistant turns from a teacher via SGLang OAI endpoint. Single-turn protocol (prefix up to first assistant, one teacher completion). |
| `generate_onpolicy_completions_mm.py` | Multimodal sibling over FineVision prompts; base64-data-URL image blocks; captures `finish_reason` at the source and optionally drops truncated rows. |
| `_smoke_mm_sglang.py` | Wire-format smoke test for the SGLang OAI vision API before scaling up. |

### Post-hoc filters

| Script | Role |
| --- | --- |
| `filter_truncated.py` | Drops rows whose assistant turn hit the `max_tokens` cap (heuristic recovery of OpenAI-style `finish_reason == "length"`). Runs in the omegalax venv to share its pinned tokenizer. |
| `preprocess_smoltalk_prompts.py` | smoltalk2 `chat.jsonl` → prompts-only JSONL for slime OPD rollouts. Templated-length filter (`--max_prompt_tokens`) to avoid budget waste on OOL prompts. |

### Misc

| Script | Role |
| --- | --- |
| `_manifest.py` | `write_manifest()` — shared helper for the `<output_dir>/manifest.json` completion marker that pmanager polls for. |

> Dataset browsing (contributor index, day-grouped segments, timeline heatmap, frame-by-frame viewer) lives in the labctl UI's artifact panel — open any `dataset` artifact and use the **Browse** section.

## Setup

```bash
uv sync  # creates .venv with msgpack, Pillow, datasets, transformers, ...
```

Scripts that run in *other* venvs (see `pyproject.toml` notes):

| Script(s) | Venv |
| --- | --- |
| `filter_truncated.py`, `stage_c_*`, `stage_d_*` | omegalax — `uv run --project <omegalax_repo>` |
| `generate_onpolicy_completions{,_mm}.py`, `_smoke_mm_sglang.py` | crowdcast-eval — `cd ../eval && uv run python …` |

## Running

### The chain (via pmanager)

```bash
pmanager launch /fast/home/franz.srambical/data_pipeline/configs/chain_smoke.py   # 10-segment smoke
pmanager launch /fast/home/franz.srambical/data_pipeline/configs/chain_v1.py      # full
```

Each stage's `cfg.children` triggers the next on `on_complete`. To launch a single stage standalone, point pmanager at e.g. `configs/stage_a_v1_5fps_360p.py`.

### Replay-source preps (standalone)

```bash
uv run python prep_smoltalk2.py --output_dir=/path/to/smoltalk2_chat --max_rows=200000 --seed=0
uv run python prep_finevision.py --output_dir=/path/to/finevision_chat --configs=DoclingMatrix,SynthChartNet,GroundUI --per_config_max=5000 --seed=0
uv run python prep_tulu3_persona_if.py --output_dir=/path/to/tulu3_chat
```

## Output contract

Every stage entrypoint writes `<output_dir>/manifest.json` before exiting (per `pipeline_task()` in `pmanager.configs.schema`). pmanager polls for this file to mark the dataset complete and register it. Schema is owned by `_manifest.write_manifest()` and captures: stage name, every flag the entrypoint received, input fingerprints (paths + key file hashes), and output statistics.

## Development

```bash
uvx ruff check .       # lint
uvx ruff format .      # format
```

`pyproject.toml` carries the strict rule set (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, ruff, pylint, tidy-imports, use-pathlib, return, comprehensions, pep8-naming).
