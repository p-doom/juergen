# Clean-room natural VM development corpus

This package contains forty novel, deterministic, development-only GUI tasks:
ten each for Writer, Calc, Files, and Chrome. The tasks were parameterized from
local application primitives and safe development harness APIs. They were not
derived from benchmark task text, tests, evaluation splits, model rollouts, or
mixed corpora.

Each task has a unique human-readable ID, a sealed fixture payload, an exact
snapshot-and-seed reset contract, a private guest root, a fresh-process hidden
state verifier, a scripted near miss, a documented recovery opportunity, and a
difficulty tag. The corpus is balanced across applications. Its aggregate
capabilities include clicking, coalesced typing, signed vertical scrolling,
dragging, hotkeys, and multi-step state changes.

Regenerate and validate the sealed corpus:

```bash
python3 -m osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.build_corpus
python3 -m osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.qualify \
  --mode static --output /tmp/cleanroom-static.json
```

Run CPU/KVM qualification without a model:

```bash
python3 -m osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.qualify \
  --mode vm --per-app 10 --output /tmp/cleanroom-vm.json
```

The VM qualification restores `osworld_ready` for every task, seeds only that
task's private fixture root, checks the exact reset state, performs the scripted
gold interaction through the production input transport, and evaluates the
result in a fresh host process. The receipt retains per-task reset attestation,
readiness, action, input-audit, and oracle evidence. It requests no GPU and
invokes no model.

## Disjoint plumbing smoke

`plumbing_smoke.json` is a separate five-task inventory for end-to-end plumbing
checks. Its IDs and seeds are disjoint from the forty-task corpus, and its
top-level eligibility contract fixes `stage0=false`, `final=false`, and
`purpose=plumbing_smoke_only`. It must never be promoted into Stage0 or final
measurement. Build and qualify it with:

```bash
python3 -m osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.build_smoke
python3 -m osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.qualify \
  --inventory plumbing-smoke --mode vm --output /tmp/cleanroom-smoke-vm.json
```
