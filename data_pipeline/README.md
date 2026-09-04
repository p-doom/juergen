# Juergen data pipelines

This project owns exactly two SFT streams:

- `configs/chain_crowdcast.py`: canonical Crowd-Cast `describe_extract` goals
  and `deltatype_v2` actions.
- `configs/chain_cua_gym_parity.py`: CUA-Gym action-format parity with
  `ordered_events_v3_relative_1000_grid_v1` actions.

Both chains finish through the shared Stage05 message-length and Stage06
training-record builders in an explicit Omegalax checkout. Set
`JUERGEN_REPO`, `LABCTL_DATASETS_ROOT`, and `OMEGALAX_REPO` to existing
directories before loading either config.

The Crowd-Cast chain begins at Stage03 and requires two immutable upstream
artifacts: `CROWDCAST_MASTER_DIR` must be a 720p/q92 Stage01 master image store,
and `CROWDCAST_CLIPS_MANIFEST` must be the canonical file from a Stage02
realigned artifact. The parity chain begins at its raw screenshot tar source.
`CUA_GYM_TRAJECTORIES` must be a curated native `computer_use` JSONL in which
every rollout has at least one successfully parsed, executed action. Parse
failures may remain only as non-action steps inside an otherwise valid rollout.

```bash
pmanager launch data_pipeline/configs/chain_crowdcast.py
pmanager launch data_pipeline/configs/chain_cua_gym_parity.py
uv run --project data_pipeline --locked pytest -q data_pipeline/tests
```

Each stage publishes `manifest.json` only after its outputs are complete.
