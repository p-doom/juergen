# Clean-room natural VM development corpus

This package contains three disjoint clean-room development inventories. None
was derived from benchmark task text, tests, evaluation splits, model rollouts,
or mixed corpora.

`stage0_inventory.json` is the Stage0 inventory. It contains exactly forty
records balanced as five anchor applications (Writer, Calc, Files, Chrome, and
VS Code) by two composition modes by four cells. Its twenty multi-app records
cover every ordered anchor-to-distinct-partner pairing once and require a
visible Alt+Tab between their two ordered source tasks. Its eligibility is
fixed to `stage0=true` and `final=false`. Every multi-app symbolic program,
including that switch, is sealed below the frozen limits of eight primitive
actions and twenty-five emitted input events. The primitive-action field counts
the compiled `ActionTurn` payloads (three or four per record), whose cardinality
must agree between the native and compact adapters.

`corpus.json` is an older forty-task, four-application auxiliary inventory. Its
eligibility is permanently fixed to `stage0=false` and `final=false`; it does
not satisfy the Stage0 balance contract and must not be relabeled or promoted.

Each source task has a unique human-readable ID, a sealed fixture payload, an exact
snapshot-and-seed reset contract, a private guest root, a fresh-process hidden
state verifier, a scripted near miss, a documented recovery opportunity, and a
difficulty tag. The inventories' aggregate
capabilities include clicking, coalesced typing, signed vertical scrolling,
dragging, hotkeys, and multi-step state changes.

Regenerate and validate the sealed Stage0 inventory:

```bash
python3 -m osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_generate_inventory
python3 -m osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_loader
python3 -m osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_qualify \
  --mode static --output /tmp/cleanroom-stage0-static.json
```

Run its CPU/KVM native-gold qualification through
`osworld_parity/labctl/recipes/natural_dev_cleanroom_stage0_cpu_kvm.toml`. Each
record executes twice from distinct provider-attested resets. A multi-app run
sets up both private components, executes both in order, uses one policy-visible
Alt+Tab, and passes the ordered states to one fresh-process composed verifier.
This native scripted qualification is not a paired-adapter receipt and does not
substitute for one.

Regenerate and validate the sealed auxiliary corpus:

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
