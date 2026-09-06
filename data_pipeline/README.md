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

Each stage publishes `manifest.json` only after its outputs are complete.

CUA-Gym Stage01 first seals a source inventory, then runs one independently
resumable job per tar and a finalizer. Workers take the inventory path, its
printed `inventory_sha256`, and one `--tar_index`; `--finalize` accepts the
same sealed inventory and refuses missing, stale, or corrupt shard receipts.

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
