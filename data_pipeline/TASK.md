# Data Pipeline Facts

## Raw Dataset

Inspected dataset path:

```text
/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-04-09/uploads/1.0.2
```

Observed structure:

- The directory contains five contributor directories.
- Each contributor directory contains `recordings/` and `keylogs/`.
- Recording files are MP4 files named like
  `recording_<recording_id>_seg0000.mp4`.
- Keylog files are msgpack files named like
  `input_<recording_id>_seg0000.msgpack`.
- `uploads/1.0.2` contains 1,588 MP4 files.
- `uploads/1.0.2` contains 1,587 msgpack files.
- There are 394 unique recording IDs.
- Some recording IDs have multiple segment files.
- The largest observed number of segments for one recording ID is 70.
- Total observed MP4 size is about 90.59 GiB.
- One MP4 has no matching keylog:

```text
87776db2-8265-4b9a-8199-ba43f1baf479/keylogs/input_a92f8703-c4fe-4329-94bb-5a99171e6952_seg0009.msgpack
```

Observed keylog event types:

- `ContextChanged`
- `MouseMove`
- `MouseScroll`
- `KeyPress`
- `KeyRelease`
- `MousePress`
- `MouseRelease`

Example keylog event shapes:

```text
[0, ["ContextChanged", ["com.apple.Terminal"]]]
[441599995, ["MouseMove", [0.0, -5.0]]]
[442433328, ["MouseScroll", [0, 0, 0.0, 0.0]]]
[471966661, ["KeyPress", [54, "KeyE"]]]
[472033328, ["KeyRelease", [54, "KeyE"]]]
[4399999, ["MousePress", ["Left", 0.0, 0.0]]]
[4499999, ["MouseRelease", ["Left", 0.0, 0.0]]]
```

## Stage A Config

File:

```text
configs/stage_a_v1_5fps_360p.py
```

Configured source path:

```text
/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-04-09/uploads
```

Configured output dataset version:

```text
v1_raw_event_stream_5fps_360p_2026_04_09
```

Configured parameters:

- `target_fps = 5`
- `target_height = 360`
- `jpeg_quality = 85`
- `train_ratio = 0.8`
- `val_ratio = 0.1`
- `seed = 0`
- `num_workers = 32`
- `max_segments = 0`

## Stage A Behavior

File:

```text
stage_a_prepare.py
```

Stage A:

- Recursively collects all `*.mp4` files under `--source_path`.
- Uses `np.random.default_rng(seed)` to shuffle videos.
- Splits videos into train/val/test by video file.
- Computes `test_ratio` as `1.0 - train_ratio - val_ratio`.
- Extracts frames from each MP4 with ffmpeg.
- Uses ffmpeg filter:

```text
fps=<target_fps>:round=up,scale=<computed_width>:<target_height>
```

- Writes frames as JPEG files named `frame_000000.jpg`,
  `frame_000001.jpg`, etc.
- Computes output frame width from original aspect ratio and target height.
- Forces output frame width to be even.
- Finds each keylog path from the recording filename by replacing
  `recording_` with `input_` and `.mp4` with `.msgpack`.
- Parses keylogs with `msgpack.unpackb(raw, raw=False)`.
- Buckets events into frame indices with:

```text
bucket_idx = (timestamp_us * target_fps) // 1_000_000
```

Stage A output layout:

```text
<output_dir>/
  manifest.json
  train/
    chat.jsonl
    <segment_id>/
      frames/
        frame_000000.jpg
      chat_line.json
      meta.json
  val/
  test/
```

Per-frame message pattern:

```json
{"role": "user", "content": [{"type": "image", "image": "/path/to/frame.jpg"}]}
{"role": "assistant", "content": [{"type": "text", "text": "NO_OP"}]}
```

Per-segment `chat_line.json` top-level fields:

- `segment_id`
- `contributor_hash`
- `messages`

Per-segment `meta.json` fields include:

- `segment_id`
- `contributor_hash`
- `split`
- `video_path`
- `keylog_path`
- `n_frames`
- `frame_height`
- `frame_width`
- `target_fps`
- `n_no_op`
- `stats`
- `recorder_emits_context_events`

## Stage A Action Format

Stage A action strings have these forms:

```text
NO_OP
<dx> <dy> <scroll>
<dx> <dy> <scroll> ; +K1 -K2
```

Observed examples:

```text
NO_OP
4 -2 0
0 0 0 ; +LMB
0 0 0 ; -LMB +KeyA
```

Formatting behavior:

- `dx`, `dy`, and `scroll` are rounded with Python `round`.
- If `dx == 0`, `dy == 0`, `scroll == 0`, and there are no key/button
  transition events, the action string is `NO_OP`.
- If there are key/button transition events, they are appended after ` ; `.
- Transition markers use `+` for press and `-` for release.

## Key and Button Handling

Stage A maintains a `held` set inside `_aggregate_events`.

For `KeyPress`:

- `_resolve_key_name(payload)` is called.
- If the key name is not in `held`, Stage A appends `("+", name)` to the
  current frame events.
- The key name is added to `held`.

For `MousePress`:

- `_resolve_button_name(payload)` is called.
- If the button name is not in `held`, Stage A appends `("+", name)` to the
  current frame events.
- The button name is added to `held`.

For `KeyRelease`:

- `_resolve_key_name(payload)` is called.
- If the key name is in `held`, Stage A appends `("-", name)` to the current
  frame events.
- The key name is removed from `held`.
- If the key name is not in `held`, `n_dangling_release` is incremented.

For `MouseRelease`:

- `_resolve_button_name(payload)` is called.
- If the button name is in `held`, Stage A appends `("-", name)` to the
  current frame events.
- The button name is removed from `held`.
- If the button name is not in `held`, `n_dangling_release` is incremented.

At the end of `_aggregate_events`:

- `n_held_at_end` is set to `len(held)`.
- No additional key/button release action is emitted at segment end.

The `held` set is local to one `_aggregate_events` call.

## Key Name Handling

Stage A special-cases some macOS `Unknown(N)` key names.

Mapping in `MACOS_UNKNOWN_NAME_BY_CODE`:

```text
10  -> ISO_Section
62  -> ControlRight
84  -> Keypad2
86  -> Keypad4
88  -> Keypad6
91  -> Keypad8
114 -> Help
115 -> Home
116 -> PageUp
117 -> ForwardDelete
119 -> End
121 -> PageDown
```

If a key name is `Unknown(N)` and `N` is not in that mapping, Stage A returns
`KC_N`.

## Mouse Button Name Handling

Stage A maps mouse button payloads as follows:

```text
Left   -> LMB
Right  -> RMB
Middle -> MMB
```

Other string button names become:

```text
M_<button>
```

Dictionary button payloads become:

```text
M_<key>_<value>
```

## Stage B Config

File:

```text
configs/stage_b_v1_run_length_cap_k0p4.py
```

Configured input dataset version:

```text
v1_raw_event_stream_5fps_360p_2026_04_09
```

Configured output dataset version:

```text
v1_run_length_capped_k0p4_5fps_360p_2026_04_09
```

Configured parameters:

- `k_seconds = 0.4`
- `num_workers = 32`

## Stage B Behavior

File:

```text
stage_b_run_length_cap.py
```

Stage B:

- Reads Stage A segment directories.
- Reads each segment's `chat_line.json` and `meta.json`.
- Computes:

```text
k_frames = max(1, round(k_seconds * target_fps))
```

- Keeps the first `k_frames` actions in each contiguous `NO_OP` run.
- Keeps all non-`NO_OP` actions.
- Drops the paired user and assistant messages for removed frames.
- Writes filtered `chat_line.json`.
- Writes updated `meta.json`.
- Adds `meta["filter"]`.
- Adds `meta["kept_indices"]`.
- Adds `meta["source_segment_dir"]`.
- Regenerates per-split `chat.jsonl`.
- Does not copy frame files.

## Stage C Config

File:

```text
configs/stage_c_v1_grain_payload.py
```

Configured input dataset version:

```text
v1_run_length_capped_k0p4_5fps_360p_2026_04_09
```

Configured output dataset version:

```text
v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09
```

Configured parameters:

- `omegalax_repo = /fast/home/franz.srambical/omegalax`
- `messages_per_record = 128`
- `records_per_shard = 10000`

## Stage C Behavior

File:

```text
stage_c_grain_payload.py
```

Stage C:

- Loops over `train`, `val`, and `test`.
- For each split with a `chat.jsonl`, calls:

```text
uv run --project <omegalax_repo> python scripts/compile_sft_dataset.py
```

- Passes:

```text
--data_path=<split>/chat.jsonl
--out_dir=<output>/<split>
--messages_per_record=<messages_per_record>
--records_per_shard=<records_per_shard>
--overwrite
```

## Stage D Config

File:

```text
configs/stage_d_v1_chunk_index_len4096.py
```

Configured input dataset version:

```text
v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09
```

Configured output dataset version:

```text
v1_chunk_index_qwen3vl2b_len4096_msgs128_k0p4_5fps_360p_2026_04_09
```

Configured parameters:

- `omegalax_repo = /fast/home/franz.srambical/omegalax-main`
- `model_id = Qwen/Qwen3-VL-2B-Instruct`
- `processor = Qwen/Qwen3-VL-2B-Instruct`
- `max_length = 4096`
- `records_per_shard = 100000`
- `num_workers = 16`

## Stage D Behavior

File:

```text
stage_d_chunk_index.py
```

Stage D:

- Loops over `train`, `val`, and `test`.
- For each split with a payload directory, calls:

```text
uv run --project <omegalax_repo> python scripts/build_sft_chunk_index.py
```

- Passes:

```text
--data_path=<payload>/<split>
--out_dir=<output>/<split>
--model_id=<model_id>
--processor=<processor>
--max_length=<max_length>
--records_per_shard=<records_per_shard>
--num_workers=<num_workers>
--overwrite
```

The wrapper defines a `system_message_text` flag and appends
`--system_message_text=<value>` if the flag is non-empty.

## Local Omegalax Clone

Inspected local clone:

```text
/fast/project/HFMI_SynergyUnit/yll/omegalax
```

The local file:

```text
/fast/project/HFMI_SynergyUnit/yll/omegalax/scripts/build_sft_chunk_index.py
```

does not define a `system_message_text` flag.

The local file:

```text
/fast/project/HFMI_SynergyUnit/yll/omegalax/scripts/compile_sft_dataset.py
```

calls:

```text
omegalax.data.grain_pipeline.compile_jsonl_to_arrayrecord
```

The local file:

```text
/fast/project/HFMI_SynergyUnit/yll/omegalax/scripts/build_sft_chunk_index.py
```

calls:

```text
omegalax.data.grain_pipeline.build_chunk_index
```

with:

```text
measure_message=make_message_length_fn(tokenizer, image_processor)
```

## Local Omegalax JSONL Compilation

File:

```text
/fast/project/HFMI_SynergyUnit/yll/omegalax/omegalax/data/grain_pipeline.py
```

`compile_jsonl_to_arrayrecord`:

- Reads a JSONL file.
- Expects each non-empty line to contain a top-level `messages` list.
- Creates `session_id` as:

```text
<jsonl_filename_stem>-<line_number_9_digits>
```

- Stores all top-level fields except `messages` and `session_id` in
  `session_meta`.
- Splits each session into payload records by message count.
- Uses `messages_per_record` as the maximum number of messages per payload
  record.
- Writes ArrayRecord shards.
- Writes `metadata.json`.

Payload block record fields:

- `session_id`
- `source_line`
- `block_idx`
- `message_start`
- `message_end`
- `session_meta`
- `messages`

## Local Omegalax Chunk Indexing

File:

```text
/fast/project/HFMI_SynergyUnit/yll/omegalax/omegalax/data/grain_pipeline.py
```

`build_chunk_index`:

- Loads payload ArrayRecord data.
- Measures every message with the provided `measure_message` function.
- Groups messages greedily into chunks until adding the next message would
  exceed `max_length`.
- Preserves `session_id` boundaries.
- Emits a chunk descriptor when the session changes.
- Emits a chunk descriptor when adding the next message would exceed
  `max_length`.
- Raises an error if a single message exceeds `max_length`.
- Writes ArrayRecord shards.
- Writes `metadata.json`.
- Writes `token_stats.json` when message measurement returns vision stats.

Chunk descriptor fields include:

- `session_id`
- `start_record_idx`
- `start_message_offset`
- `end_record_idx`
- `end_message_offset`
- `num_messages`
- `measured_length`

When reading a chunk-index dataset, `_ChunkDescriptorResolver` reconstructs
the example by collecting messages from the referenced payload records and
adding:

- `_omegalax_session_id`
- `_omegalax_start_record_idx`
- `_omegalax_end_record_idx`
- `_omegalax_measured_length`

## Local Qwen3 Encoding

File:

```text
/fast/project/HFMI_SynergyUnit/yll/omegalax/omegalax/data/qwen3_encoding.py
```

`build_chatml_text` serializes each message as:

```text
<|im_start|>{role}
{content}<|im_end|>
```

For string content, it appends the string directly.

For structured content lists:

- `{"type": "text", "text": "..."}` appends text.
- `{"type": "image", ...}` appends Qwen vision pad tokens.

`extract_images` accepts image blocks with either:

```json
{"type": "image", "image": "..."}
```

or:

```json
{"type": "image", "url": "..."}
```

`make_message_length_fn`:

- Calls `encode_qwen_messages([message], ...)`.
- Returns message length.
- Also returns vision stats when images are present.

## Local Qwen3 Collator

File:

```text
/fast/project/HFMI_SynergyUnit/yll/omegalax/omegalax/data/collator_qwen3.py
```

The VLM SFT collator:

- Reads `ex["messages"]`.
- Encodes messages with `encode_qwen_messages`.
- Raises an error if encoded length exceeds `max_length`.
- Builds `loss_mask_BT` with `_build_assistant_loss_mask`.
- Includes loss on assistant content tokens.
- Includes loss on the assistant `<|im_end|>` token.
- Does not include loss on user or system content tokens.

