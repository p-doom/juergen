# ROADMAP 3.4 controlled-perturbation recovery

This package constructs recovery-task VM evaluation and training infrastructure
on audited base commit `48a54e8`. It is intentionally isolated from the earlier
capability rungs and is not authorized for activation until ROADMAP 3.1, 3.2,
and 3.3 each provide a passing, content-addressed gate artifact.

## Task contract

Each task starts from a rung 1b real-application fixture. A controller applies a
deterministic perturbation before the policy observation:

- VS Code: focus a known wrong surface.
- Chrome: scroll in the opposite direction.
- Files: drag a separately created wrong file to the requested destination.

The intended Files source remains available, so recovery is a real policy
action rather than a privileged reset. The perturbation has matched
`native_absolute_control` and `compact_raw_phaseb` encodings and does not consume
the policy budget. The recovery horizon is exactly the base horizon plus two.
Opposite-scroll demonstrations spend one added step cancelling a real opposing
displacement before running the base solution.

Controller dispatches and policy actions have separate origins. Public rollout
labels distinguish `injected_perturbation`, `natural_ineffective_action`,
`executor_failure`, and `effective_recovery_action`. A failed controller
dispatch is always `executor_failure`, never an injected perturbation.

## Data and oracle boundaries

Train and development manifests contain newly authored recovery IDs in disjoint
namespaces. Evaluation is represented only by an opaque count/hash commitment;
there are no sealed task rows or seeds in this repository, and training/build
loaders reject `evaluation_sealed` before reading any path.

The policy observation is instruction plus screenshot. Reward and hidden state
are retained by `TrainerOnlyOracle` and are forbidden recursively by the public
JSONL validator. `on_policy_rollout.schema.json` covers scripted recovery demos
and future on-policy data without exporting expected state, reward, or oracle
state.

## CPU-only construction checks

```bash
python3 -m pytest -q \
  osworld_parity/proper_vm_capability_ladder/rung34_recovery/tests
CUDA_VISIBLE_DEVICES= python3 -m \
  osworld_parity.proper_vm_capability_ladder.rung34_recovery.build \
  --output /tmp/rung34-recovery-build
```

The matching labctl recipes are:

- `rung34_recovery_contract_build_cpu.toml`
- `rung34_recovery_rollout_validate_cpu.toml`
- `rung34_recovery_vm_replay_cpu_kvm.toml`

All allocate zero GPUs and invoke no model runtime. The KVM replay recipe also
requires explicit earlier-gate evidence, restores the pinned VM for every
episode, compares two post-perturbation reset hashes, replays both action arms,
and exports only public rollout fields. Its placeholder gate-artifact path must
not be replaced or submitted until the earlier rungs are promoted.
