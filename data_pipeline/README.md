# Juergen data pipelines

This project owns the stage code for exactly two SFT streams:

- canonical Crowd-Cast `describe_extract` goals and `deltatype_v2` actions;
- CUA-Gym action-format parity with
  `ordered_events_v3_relative_1000_grid_v1` actions.

Both chains finish through the shared Stage05 message-length and Stage06
training-record builders. Their Labctl recipes bind the Juergen and Omegalax
checkouts and the pinned Qwen processor snapshot. Stage05 and Stage06 attest
the exact processor files they consume and run the attested Omegalax project
locked and offline.
Stage06 rejects any full conversation above `max_length`; it never splits away
conditioning context.

The Crowd-Cast Labctl pipeline owns the complete Stage00--06 chain: raw-source
inventory, 720p/q92 master frames, strict realignment, filtering, goal
annotation, conversations, message lengths, and training records. Every raw
video and keylog is sealed as accepted or explicitly excluded. The parity
pipeline begins from the recorded CUA-Gym screenshots and native
`computer_use` trajectory JSONL. Its curator groups recorded multi-call turns,
verifies executed metadata, records non-action and nonrepresentable
dispositions, and emits the sole schema accepted by its conversation builder.

Labctl owns deployment. Its two TOML pipelines live in the Slurm repository at
`dev/franz/berlin/crowd-cast-bc/labctl/pipelines/crowdcast_canonical_v1.toml`
and
`dev/franz/berlin/crowd-cast-bc/labctl/pipelines/cuagym_action_format_parity_v1.toml`.
Juergen contains no scheduler compatibility layer.

```bash
uv run --project data_pipeline --locked pytest -q data_pipeline/tests
```

The Omegalax compiler must use one structural Qwen encoder for per-message
measurement, full-example encoding, and VLM collation. Assistant messages with
`loss: false` must contribute no supervised tokens. Run the real consumer
contract against the checkout and immutable processor snapshot selected for a
job:

```bash
export OMEGALAX_REPO=/path/to/omegalax
export PROCESSOR_SNAPSHOT=/path/to/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/<revision>
uv run --project data_pipeline --locked pytest -q \
  data_pipeline/runtime_tests/test_omegalax_encoder_contract.py
```

Each stage publishes `manifest.json` only after its outputs are complete.

CUA-Gym Stage01 first seals a source inventory, then runs one independently
resumable job per tar and a finalizer. Workers take the inventory path, its
printed `inventory_sha256`, and one `--tar_index`; `--finalize` accepts the
same sealed inventory and refuses missing, stale, or corrupt shard receipts.
Schema-v1 image stores must be rebuilt; Stage04 accepts only the schema-v2
receipt layout. The local loop below can be dispatched as independent tar jobs:

```bash
metadata=$(uv run --project data_pipeline --locked python pipeline/cua_gym/stage_01_image_store.py \
  --screenshots_dir /data/screenshots --output_dir /data/images)
digest=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["inventory_sha256"])' <<<"$metadata")
count=$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)["sources"]))' <<<"$metadata")
inventory=/data/images/source_inventory-$digest.json
for ((index=0; index<count; index++)); do
  uv run --project data_pipeline --locked python pipeline/cua_gym/stage_01_image_store.py \
    --inventory "$inventory" --inventory_sha256 "$digest" --tar_index "$index" \
    --workers 8 --output_dir /data/images
done
uv run --project data_pipeline --locked python pipeline/cua_gym/stage_01_image_store.py \
  --inventory "$inventory" --inventory_sha256 "$digest" --finalize --output_dir /data/images
```

### Sharded message-length measurement

Run one Stage05 worker per shard against a shared output directory, then run
the finalizer after every worker receipt exists. This local loop can also be
dispatched as independent jobs, one `--shard_index` per job:

```bash
lengths=/path/to/message-lengths
chat=/path/to/stage04
omegalax=/path/to/omegalax
snapshot=/path/to/processor/snapshots/REVISION
common=(--output_dir="$lengths" --source_path="$chat" --omegalax_repo="$omegalax" \
  --processor_snapshot="$snapshot" --num_workers=8 --num_shards=8)
for index in {0..7}; do
  uv run --project data_pipeline --locked python pipeline/stage_05_measure_lengths.py \
    "${common[@]}" --shard_index="$index"
done
uv run --project data_pipeline --locked python pipeline/stage_05_measure_lengths.py \
  "${common[@]}" --merge
```

A shard with no assigned conversations publishes an empty receipt without
launching the compiler. With `--num_shards=1`, the worker uses the same shard
contract and finalizes automatically.
