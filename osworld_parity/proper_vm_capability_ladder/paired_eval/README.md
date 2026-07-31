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
  semantic prefix, permits a bounded multi-action/event plan for exactly one
  logical semantic step, and scores its next state in a fresh oracle process.
- `gold_prefix_horizon` starts from every registered semantic prefix and runs
  fixed action horizons 2, 4, and 8.
- `natural_closed_loop` starts from the reset state and runs naturally on tasks
  registered with two to four semantic steps.

Every cell can have multiple independent generation attempts.  The aggregate
always reports whether pass@1, pass@4, and pass@8 are actually estimable; it
does not invent pass@k values when there are too few complete attempts or when
multi-sample generation is deterministic.
The sealed config requires at least eight attempts, sampling enabled with a
positive temperature, and a unique deterministic generation seed per attempt.

## Fail-closed execution gate

`plan` and `validate` never import a runtime.  `run` requires an explicit
`--executor-ready /registered/artifact/EXECUTOR_READY.json` and an explicit
`--task-setup-validation /registered/setup/task_setup_validation.json`. The
two raw file SHA-256 values and artifact IDs must match the sealed evaluation
manifest. The setup artifact must bind the exact development task-manifest
payload, VM snapshot, setup commit, fixture and asset hashes, and full task
coverage; the evaluator only consumes it and never reruns mutable setup
validation. The readiness file SHA-256
must match the sealed evaluation manifest, its status and all checks must pass,
and it must bind the executor commit, capability report, both action
interfaces, and VM snapshot.  The CLI consumes this marker before importing
the runtime factory; it never searches the repository for or creates a marker.
The accepted certification schema is exactly `proper_vm_executor_cert_v1`,
including its ordered four-interface list, eight frozen checks, canonical
report self-hash, and `osworld_ready` snapshot binding.
The two registered artifact identities are independently matched through the
exact `executor_readiness` and `task_setup_validation` inputs in the JSON file
named by `LABCTL_CONTEXT`; no input alias, path search, or marker self-ID is
accepted.

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
The result JSONL is written only after every planned pair completes, using one
atomic final rename; there is no partial success marker.

Production `run` pins the independently approved curriculum commit and its
version-1 runtime binding, provider-reset, refresh, compiled-segment, dispatch,
and executed-segment receipt schemas. It remains fail-closed on both registered
dependencies; in particular, no scored execution can begin without the exact
sealed `EXECUTOR_READY.json` named by the evaluation manifest.

## Result contract

Each pair row records observation hashes (not screenshots), raw requested
actions, canonical semantic operations, lowered operations, executed operation
traces, cursor before/after, executor evidence, verifier semantic state and
hash, parse/dispatch status, and both raw and semantic first divergence.
Task coordinates are live-probe references resolved independently after every
reset. Both arms must resolve the same initial cursor and reset signature.
The runtime must expose the reset cursor as an unmodified live probe and may
not pre-center it on a target. It must also prove the true active window and
that no hidden keyboard/mouse intervention occurred between policy turns.
For native clicks, requested absolute coordinates must exactly match the
lowered dispatch trace and per-click cursor readback; scoring uses only the
post-action fresh semantic state, so an exact-action mismatch cannot override
semantic success or failure.
Model turns, logical steps, primitive action lines, emitted events, output
tokens, and wall time are separately decremented and logged. Any overrun or
parse/dispatch error is a scored failure.
Known VM/setup/observation/model-service/verifier failures can exclude only the
whole pair.  Parse and executor-dispatch errors remain scored system failures.

Aggregation rejects duplicate, missing, or unmatched pairs.  It emits arm
rates, compact-minus-native paired differences, a deterministic task-clustered
paired bootstrap interval, descriptive McNemar discordance, mode/horizon
strata, and the pass@1/4/8 feasibility report.
It also re-hashes every sealed row/turn/verifier state and recomputes success
from trace plus fresh-process oracle evidence instead of trusting stored score
fields.
