# Proper-VM rung 2: deterministic same-application compositions

This directory implements roadmap item 3.2 only. It authors deterministic
Writer, Calc, Files, and Chrome tasks with 2–4 semantic steps, fixed maximum
horizons, hidden state oracles, reset signatures, paired action encodings, and
scripted gold/near-miss replay. It contains no model invocation, GPU path, or
official OSWorld fixture material.

## Split boundary

`manifests/train.json`, `development.json`, and `sealed_eval.json` have disjoint
IDs and parameter seeds and independent payload seals. Training collection is
hard-coded to `train`; replay permits `train` and `development`; both reject
`sealed_eval` before VM startup. The sealed manifest is validated by unit tests
but is not replayed or collected by the supplied recipes.

The semantic-step/horizon contract is:

| Application | Composition | Semantic steps | Max action turns |
|---|---|---:|---:|
| Writer | replace text, format, save | 3 | 4 |
| Calc | select cell, enter formula, confirm, save | 4 | 6 |
| Files | select, move, rename | 3 | 8 |
| Chrome | navigate, scroll, toggle | 3 | 6 |

`action_schemas.json` freezes the native absolute operation-sequence envelope
and compact raw-delta grammar. `trajectory.py` produces the same semantic gold
and near-miss programs for both encodings.

## Determinism and readiness

Every VM cell restores `osworld_ready`, creates a private fixture root, verifies
the exact negative state and geometry, rejects a scripted near miss, restores
again, compares the reset signature, and requires a scripted gold to pass in a
fresh oracle process. Writer and Calc oracles inspect ODF state, Files inspects
the filesystem, and Chrome's local settings fixture writes a guest-private state
file. Screenshots and instructions are the policy-visible observation surface;
the state probes are oracle-only.

Readiness does not use a larger generic timeout. Failures carry the last proven
phase and preserve app logs, guest processes, windows, Chrome-debug output, and
the last state/geometry error plus a hashed screenshot before guest teardown.
`READINESS_EVIDENCE_JOB_135883.json` preserves the hashes and causal localization
of the prior development-only failure. `VM_DEVELOPMENT_VALIDATION_20260731.json`
records the passing two-arm CPU/KVM validation for all four apps and the causal
Calc/Files readiness diagnostics. No timeout was raised.

## Commands

Run the local contract tests:

```bash
python3 -m pytest -q \
  osworld_parity/proper_vm_capability_ladder/rung2_sameapp/curriculum/tests \
  osworld_parity/proper_vm_capability_ladder/rung2_sameapp/tests
```

The production entry point is the VM-only development replay documented in
`curriculum/README.md`. Both checked-in rung-2 CPU recipes use that path,
request zero GPUs, and require a registered independent task-setup-validation
artifact. The former build replay and train collector fail closed. No sealed
evaluation or model run is part of this item.
