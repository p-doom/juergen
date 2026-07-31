# Same-app semantic curriculum scaffold

This is a small train/development-only bridge between isolated Phase-B mouse
skills and eventual full-VM parity. It stays on newly authored deterministic
Writer, Calc, Files, Chrome, and VS Code fixtures; it is not an OSWorld task
ingestion or offline-training pipeline.

## Contract

`semantic_task.schema.json` defines task identity independently of action
format. Each task pins a parameter seed, deterministic asset recipe/hash,
snapshot/reset strategy, 2–4 ordered semantic steps, semantic cursor milestones,
hidden final-state verifier, reset/near-miss/gold requirements, and a fixture
seal. `program.py` is a runtime bridge to the existing rung-2 symbolic compiler;
native absolute and compact raw-relative actions are never stored in task
identity.

The `budget_contract` declares conservative admission caps for semantic steps,
emitted primitive actions, and lowered primitive events, keyed by the real
compiler interface IDs `native_absolute_sequence_v1` and
`compact_raw_phaseb_v1`. It does not claim coordinate-independent exact event
counts. After live binding, every compiled segment records its resolved action
and event counts, binding hash, and resolved-budget hash; the aggregate receipt
sums and re-hashes its segments. For example, Files has three semantic steps
but permits up to eight compact action lines and 22 lowered compact events.

Coordinates are never seeded or invented. Each task carries a versioned
`geometry_contract` and live-cursor contract. Setup must return exact initial
state plus all required targets from the existing VM probes. At least two exact
reset probes must agree on geometry, cursor, and viewport before a sealed
`ValidatedRuntimeBinding` can compile anything. They must also be distinct,
single-use reset generations issued by the active VM session, with ordered
monotonic timestamps, setup/snapshot provenance, content seals, and bounded
freshness. The issuing ledger consumes a one-use HMAC receipt created only
after the KVM provider's real `load_state(osworld_ready)` call; that receipt
attests the provider session, prior/new generation chain, before/after provider
state hashes, and reset interval. Provider state is copied into immutable
canonical observations before the call, generation IDs are those observation
hashes rather than host-invented UUIDs, and the receipt is issued only when the
provider's append-only telemetry adds the expected `loadvm[osworld_ready]` and
`loadvm_guest_ready` records. Immediately before reset the session also writes
a nonce sentinel inside the guest and requires the restored VM to prove that
the snapshot removed it. Equal/no-op and telemetry-only transitions are
rejected. The production CLI also pins the provider module's content SHA-256,
and the recipes place that fixed argument after all labctl arguments. A raw observation has a unique ID and capture
time and cannot be re-signed. Cursor history stores semantic refs, and evaluated rows log each ref
with its resolved live value. Chrome is re-probed after its executed scroll
receipt; its later click compiles only from a refreshed binding that proves the
signed scroll delta, active generation, causal timestamps, and changed binding
revision. The ledger independently replays the exact dispatch-cardinality,
ordering, adapter, operation-content, cursor, atomic-result, and result-seal
checks before recording an executed receipt. Compiled segments pin their
binding-derived start/final cursor, and one cursor chain must span every
operation and action in the segment. Chrome refresh accepts only that
recorded step-2 receipt object, never a caller-provided digest string.

The initial materialized matrix contains one train and one development seed for
each family:

| App | Family | Steps | Phase-B coverage | Explicit edge/thin labels |
|---|---|---:|---|---|
| Writer | replace, bold, save | 3 | click, type, hotkey | Ctrl+S |
| Calc | select, formula, confirm, save | 4 | click, type, hotkey | Ctrl+S |
| Files | select, drag, rename | 3 | click, drag, type, hotkey | file-drag (thin) |
| Chrome | navigate, signed scroll, toggle | 3 | click, vscroll | each sign (thin) |
| VS Code | focus, Unicode replace, save | 3 | click, type, hotkey | Unicode (thin), Ctrl+S |

Train Chrome scrolls down and development Chrome scrolls up, so the complete
materialized scaffold covers both signs. Horizontal scroll and timing-sensitive
double-click are frozen exclusions until their transport contracts are proven.
Unicode, real file drag, Ctrl+S, and every single-family thin case remain
visible in `coverage`; they are not silently promoted into broad evidence.

Every fixture must reject exact reset, reject its deterministic near miss,
accept gold, reproduce the same reset signature, run its oracle in a new
process, and finish with zero held inputs. Reset, near-miss, and gold checks
each require `oracle_status == "ok"`; an oracle error cannot count as a
successful rejection. `verify_fixture_contract` requires
four real artifact roots and invokes the declared independent extractor; it
does not manufacture scripted expected states. Writer/Calc/Files/Chrome delegate to
the existing same-app state oracles; VS Code delegates to the existing UTF-8
file oracle. `state_extraction.py` independently reads ODF, filesystem, Chrome
event-state, or UTF-8 artifacts without consulting expected values. The
declared verifier is the CLI in `oracle.py`; evaluator-owned processes may run
final mode or semantic mode with `--expected-step-index` and
`--expected-target-ref` and verify PID, identity, and semantic-state hash.
Model termination is not a task-state transition and cannot set `MOUSE_SOLVED`.

An external setup run may produce exactly `task_setup_validation.json`, with
schema ID `multistep_sameapp_task_setup_validation_v1`. The reader accepts only
development-only, no-heldout, full-fixture coverage bound to the manifest hash,
all fixture/asset hashes, `osworld_ready`, and a lowercase 40-hex setup commit.
It reads an explicit immutable artifact path and never synthesizes or reruns
setup. Evaluators separately pin its artifact ID and raw file SHA.

`rung2_sameapp/replay.py` is the production boundary. It requires that pinned
setup dependency, performs distinct reset cycles, exports actual guest files for
the declared extractor/fresh-process oracle, and compiles one semantic segment
at a time. Each successful executor dispatch becomes an executed segment
receipt carrying the exact binding revision used. Aggregation hashes those
ordered receipts and revisions without recompiling earlier Chrome segments
against the post-scroll binding. The former build replay and direct teacher
collector fail closed; neither is an alternate compiler/execution path.
Both checked-in rung-2 CPU recipes invoke this VM-only development replay and
declare the VM, QEMU, provider, and `task_setup_validation` inputs. The setup
input intentionally carries a conspicuous artifact placeholder until an
independent producer output is registered and pinned; labctl resolves its
artifact ID and the launcher hashes the exact custody-resolved JSON.

The upward Chrome fixture establishes its nonzero initial offset in page load
with `scrollTo`, reports it through the local event endpoint, and cannot pass
runtime setup validation if the probed offset differs.

## Split boundary

Only `manifests/train.json` and `manifests/development.json` exist. They share
the five family IDs and are explicitly a within-family, disjoint-seed
development split—not family-heldout validation. Future family-heldout and
sealed sets require new family IDs assigned externally and disjointly; neither
set is materialized and neither has seeds or inputs. `load_manifest("sealed_eval")`
rejects before any file access. No official task IDs, benchmark assets, GPU
recipes, or training stages are in this directory.

Run the local contract suite with:

```bash
python3 -m pytest -q \
  osworld_parity/proper_vm_capability_ladder/rung2_sameapp/curriculum/tests \
  osworld_parity/proper_vm_capability_ladder/rung2_sameapp/tests
```
