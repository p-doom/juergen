# Paired development evaluation

This package compares the native absolute control system with the compact
raw-relative system on the same real-VM development tasks.  It is an evaluation
scaffold, not a job launcher, model trainer, or source of held-out tasks.
The production CLI accepts only the sealed
`proper_vm_sameapp_semantic_curriculum_v1` `development.json` through that
curriculum package's own loader. It does not discover task directories or fall
back to another schema.

The comparison is explicitly a **complete-system comparison**.  Checkpoint,
prompt, generation settings, and action interface are recorded for each arm and
may differ.  Task ID, reset snapshot, parameter seed, gold-prefix cursor,
generation seed, and inference/action budget are common pair properties and
cannot be overridden inside an arm.

## Development modes

- `gold_history_one_step` resets and replays an action-format-independent gold
  semantic prefix, executes one model action, and scores the semantic next
  state with the fresh-process task verifier.
- `gold_prefix_horizon` starts from every registered semantic prefix and runs
  fixed action horizons 2, 4, and 8.
- `natural_closed_loop` starts from the reset state and runs naturally on tasks
  registered with two to four semantic steps.

Every cell can have multiple independent generation attempts.  The aggregate
always reports whether pass@1, pass@4, and pass@8 are actually estimable; it
does not invent pass@k values when there are too few complete attempts or when
multi-sample generation is deterministic.

## Fail-closed execution gate

`plan` and `validate` never import a runtime.  `run` requires an explicit
`--executor-ready /registered/artifact/EXECUTOR_READY.json`.  The file SHA-256
must match the sealed evaluation manifest, its status and all checks must pass,
and it must bind the executor commit, capability report, both action
interfaces, and VM snapshot.  The CLI consumes this marker before importing
the runtime factory; it never searches the repository for or creates a marker.

```text
python -m osworld_parity.proper_vm_capability_ladder.paired_eval validate \
  --evaluation-manifest paired-development.json \
  --task-manifest curriculum-development.json

python -m osworld_parity.proper_vm_capability_ladder.paired_eval plan \
  --evaluation-manifest paired-development.json \
  --task-manifest curriculum-development.json \
  --shard-index 0 --shard-count 4
```

No `run` command is part of repository validation.  Scientific or held-out
execution is rejected by manifest validation.

## Result contract

Each pair row records observation hashes (not screenshots), raw requested
actions, canonical semantic operations, lowered operations, executed operation
traces, cursor before/after, executor evidence, verifier semantic state and
hash, parse/dispatch status, and both raw and semantic first divergence.
Known VM/setup/observation/model-service/verifier failures can exclude only the
whole pair.  Parse and executor-dispatch errors remain scored system failures.

Aggregation rejects duplicate, missing, or unmatched pairs.  It emits arm
rates, compact-minus-native paired differences, a deterministic task-clustered
paired bootstrap interval, descriptive McNemar discordance, mode/horizon
strata, and the pass@1/4/8 feasibility report.
