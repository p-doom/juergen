# crowdcast-data-pipeline

SFT data pipeline for crowd-cast: turns raw S3-synced screen recordings + keylogs into omegalax-ingestable Grain ArrayRecord shards, and prepares replay-source corpora (SmolTalk2, FineVision, Tulu3-Persona-IF) in the same chat-format.

Consumed by [`pmanager`][pmanager]/[`labctl`][labctl] recipes that inject params via `absl` flags and poll `<output_dir>/manifest.json` for stage completion.

[pmanager]: https://github.com/anthropics/pmanager
[labctl]: https://github.com/anthropics/labctl

## Layout

### Crowd-cast 7-stage chain (`stage_*`)

```
A  Frame extraction + keylog parsing
|
B  Blackscreen removal + NO_OP run-length cap
|
C  Keyframe captioning (vision LLM)
|
D  Hierarchical task tree (text LLM)
|
E  Token-budget chunking + per-chunk instruction generation
|
F  Grain ArrayRecord packing
|
G  Chunk index for dataloader
```

Stages A-B clean the raw data. Stages C-E add synthetic annotations via LLM. Stages F-G compile into training format. **B must run before C** so the vision LLM never sees blackscreens or idle stretches.

| Stage | Script | Role |
| --- | --- | --- |
| A | `stage_a_prepare.py` | S3 sync → per-segment JPEG frames at target fps/height + per-frame action strings + per-split `chat.jsonl`. |
| B | `stage_b_run_length_cap.py` | Cap NO_OP runs (`k = round(k_seconds · target_fps)`). Rewrites `chat_line.json` only — frames are referenced in place. |
| C | `stage_c_caption.py` | Sample keyframes from Stage B output, send batched filmstrips to a vision LLM for per-keyframe captioning. |
| D | `stage_d_task_tree.py` | Text-only LLM call: reads captions, produces hierarchical task tree (goal → sub-goals → steps). |
| E | `stage_e_chunk_buckets.py` | Slice recording into fixed token-budget chunks (8k-128k), generate per-chunk instructions via text-only LLM. |
| F | `stage_f_grain_payload.py` | Compile per-split `chat.jsonl` → Grain ArrayRecord shards. Wraps `omegalax/scripts/compile_sft_dataset.py`. |
| G | `stage_g_chunk_index.py` | Build the offline chunk index from the Grain payload. Wraps `omegalax/scripts/build_sft_chunk_index.py`. |

> Legacy aliases `stage_c_grain_payload.py` / `stage_d_chunk_index.py` are kept for existing pmanager configs.

Chain configs in `configs/chain_v1.py` (full) and `configs/chain_smoke.py` (10 segments — end-to-end pmanager validation without burning ~17 h of cluster time).

### Annotation details (Stages C-E)

#### Stage C: Captioning

**Keyframe sampling:** Every 75th frame from the cleaned Stage B stream (`--keyframe_every_n=75`). At 2fps this is one keyframe every ~37.5s. A 2-hour recording yields ~530 keyframes.

**Batching:** 15 keyframes per LLM call (`--batch_size=15`), resized to 720p (`--target_height=720`). ~35 calls for a 2-hour recording.

**Captioning prompt** (`_build_caption_messages`):
```
These are N chronological screenshots from a screen recording of a software
engineer working. They are sampled roughly every 5 seconds.

For each screenshot (numbered 1-N):
1. Describe what application is visible and what content is on screen
   (read text, code, terminal output, URLs, file names -- be specific).
2. Describe what the user appears to be doing.
3. If the activity changed from the previous screenshot, note the transition.

Be concise -- one or two sentences per screenshot.
[1] <description>
[2] <description>
```

**Output:** `captions.json`, `keyframes_meta.json`, `keyframes/`

#### Stage D: Task tree

Single text-only LLM call. Sends all captions as a timeline and asks for a hierarchical JSON decomposition.

**Task tree prompt** (`_build_task_tree_messages`):
```
From this timeline, produce a hierarchical task decomposition as JSON:
1. goal: One imperative sentence for the overall session task.
2. sub_goals: Major phases, each with instruction, keyframe range, steps.

Rules:
- Instructions must be imperative ("Open the file", NOT "The user opens")
- Every keyframe belongs to exactly one step and one sub-goal
- Be specific: reference actual file names, URLs, commands
```

**Output:** `task_tree.json`

#### Stage E: Token-budget chunking

Slices the recording into chunks sized to fill each token bucket. Per-chunk instructions are generated via text-only LLM calls (captions only, no images).

**Token math** (Qwen3-VL at 540p, 2fps):
- Vision tokens per frame: 620 (`ceil(540/28) * ceil(864/28)`)
- Total tokens per frame (vision + chat overhead): **634**
- System message overhead: ~60 tokens

| Bucket | Frames | Duration | Instruction granularity |
|--------|--------|----------|------------------------|
| 8k | 12 | ~6s | Atomic UI action (1-5 words): "Open a new browser tab" |
| 16k | 25 | ~12.5s | Short action sequence (5-12 words): "Search Google for X" |
| 32k | 51 | ~25s | Step-level task (8-20 words): "Inspect temperature hyperparameters" |
| 64k | 103 | ~50s | Multi-step goal (10-25 words): "Compare validation metrics across runs" |
| 128k | 206 | ~103s | Sub-goal (10-30 words): "Research temperature effects on distillation" |

**Instruction prompt** (`_generate_instructions_batch`, 40 chunks per call):
```
You are generating training instructions for a computer-use agent.
The recording session's overall goal: "{goal}"

Below are N chunks. For each chunk, write ONE imperative instruction at
this granularity: {granularity_description}

Rules:
- Imperative ('Open X', 'Click Y', NOT 'The user opens X')
- Be specific: use actual file names, URLs, app names
- Each instruction must be different
```

**Output:** `samples_{8k,16k,32k,64k,128k}.jsonl` — one JSONL per bucket. Each line:
```json
{
  "sample_id": "8k_0042",
  "bucket": "8k",
  "bucket_tokens": 8192,
  "instruction": "Click the Config button",
  "n_frames": 12,
  "est_tokens": 7668,
  "duration_s": 6.0,
  "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
}
```

### LLM backends

Stages C-E use the OpenAI client, configured via environment variables:
```bash
export AZURE_AI_FOUNDRY_ENDPOINT="..."   # Azure URL or http://localhost:PORT/v1
export AZURE_AI_FOUNDRY_API_KEY="..."    # API key or "local" for sglang
```

| Backend | Notes |
|---------|-------|
| **Kimi K2.6** (Azure AI Foundry) | Thinking model — output in `reasoning_content` field. ~$1.60/recording. |
| **Qwen3.6-27B** (local sglang) | Natively multimodal, fits on 1x H100 (TP=1, ~54GB bf16). Thinking via `<think>...</think>` tags (stripped automatically). Free. |

For local inference, see `run_pilot_qwen.sbatch`. For production: 8 GPUs → 8 independent sglang replicas (TP=1), fan out recordings across them.

### Viewer (`annotation_pilot/viewer.py`)

Browser-based inspector for annotated samples.
```bash
uv run python annotation_pilot/viewer.py --port=8500
```
- Bucket tabs (8k / 16k / 32k / 64k / 128k)
- **Player view**: frame-by-frame playback, scrubber, speed control (0.25x-8x), action overlay
- **Chat view**: paginated messages with Hide NO_OP toggle
- Keyboard: Space (play/pause), arrows (step), `[`/`]` (speed), Home/End (jump)

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
| `filter_truncated.py`, `stage_f_*`, `stage_g_*` | omegalax — `uv run --project <omegalax_repo>` |
| `stage_c_caption.py`, `stage_d_task_tree.py`, `stage_e_chunk_buckets.py` | data_pipeline (this project) — needs `openai`, `opencv-python-headless` |
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
