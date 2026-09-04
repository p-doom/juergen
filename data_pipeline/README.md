# Juergen data pipelines

This project owns exactly two SFT streams:

- `configs/chain_crowdcast.py`: canonical Crowd-Cast `describe_extract` goals
  and `deltatype_v2` actions.
- `configs/chain_cua_gym_parity.py`: CUA-Gym action-format parity with
  `ordered_events_v3_relative_1000_grid_v1` actions.

Both chains finish through the shared Stage05 message-length and Stage06
training-record builders in an explicit Omegalax checkout. Set
`JUERGEN_REPO`, `LABCTL_DATASETS_ROOT`, and `OMEGALAX_REPO` to existing
directories before loading either config. `SFT_PROCESSOR_SNAPSHOT` must name a
weight-free local Hugging Face `snapshots/<40-hex-revision>` directory containing
the complete tokenizer and image-processor files. The chain records every file
digest and runs the attested Omegalax project locked and offline.

The Crowd-Cast chain begins at Stage03 and requires two immutable upstream
artifacts: `CROWDCAST_MASTER_DIR` must be a 720p/q92 Stage01 master image store,
and `CROWDCAST_CLIPS_MANIFEST` must be the canonical file from a Stage02
realigned artifact. The parity chain begins at its raw screenshot tar source.
`CUA_GYM_TRAJECTORIES` must be the recorded native `computer_use` JSONL. Its
Stage03 curator groups recorded multi-call turns, verifies executed metadata,
records non-action and nonrepresentable dispositions, and emits the sole schema
accepted by Stage04.

```bash
pmanager launch data_pipeline/configs/chain_crowdcast.py
pmanager launch data_pipeline/configs/chain_cua_gym_parity.py
uv run --project data_pipeline --locked pytest -q data_pipeline/tests
```

Each stage publishes `manifest.json` only after its outputs are complete.
