# Rung 1a: instrumented browser mechanics

This directory implements only the first, non-scientific mechanics gate for the
proper-VM ladder. It runs newly authored local pages in the real Chrome GUI on
the pinned Ubuntu QCOW. It does not reuse official OSWorld task IDs, setup
configs, instructions, assets, or held-out material.

## Interpretation boundary

Rung 1a is an **instrumented browser microbenchmark**. It establishes that the
two action adapters can drive a real GUI closed loop and that reset, dispatch,
and state-oracle contracts work. It does **not** establish transfer to Chrome
settings, Files, VS Code, Writer, Calc, or official OSWorld tasks.

The policy-visible contract is exactly the natural-language instruction and
desktop screenshot. DOM geometry, event logs, expected values, fixture hashes,
and oracle results remain in the host process. The fixture server exposes no
HTTP route that reads oracle state; `/state`, `/oracle`, and GET `/event/*` are
404. Final scoring runs in a fresh host process and `MOUSE_SOLVED` is true only
when that process accepts the final hidden state.

## Frozen fixture inventory

`fixtures.json` schema v2 contains four templates: click, focus plus Unicode-safe
coalesced type, bidirectional scroll, and native range-slider drag. Each has two
development seeds and eight frozen evaluation seeds: 8 development and 32
evaluation fixtures total. Development fixtures are the only fixtures used by
the CPU/KVM selfcheck. Evaluation fixtures are sealed but are not exercised by
the selfcheck or exposed to a model.

Schema v2 is a pre-science coordinate-calibration amendment. Development-only
job 135823 showed that treating design-space pixels as viewport pixels could put
a card outside a smaller Chrome viewport. No evaluation outcome was observed.
Every row was resealed with a common contract that scales its 1920×1080 design
coordinate into the measured Chrome viewport and clamps the whole card on
screen. Before dispatch, the VM selfcheck validates the actual development DOM
rectangles and mathematically checks all 40 manifest rows against the measured
window; evaluation pages and oracles remain unopened.

Both adapters compile to the same transport primitives. In particular,
`compile_unicode_coalesced_type()` is the sole typing compiler for both arms: it
performs one exact UTF-8 clipboard write and one Ctrl+V in a single guest
process. `pyautogui.write` is not used.

### Pre-science compact-dispatch diagnostic amendment

Development-only job 135826 established that the v2 geometry contract itself
held: `r1a-click-dev-1101` passed under the native absolute arm, and the compact
arm's measured target rectangle was fully visible and centered at the requested
endpoint. The compact gold nevertheless left the checkbox unchanged. Its old
browser trace had neither pointer coordinates nor a causal ordering guarantee,
because independent asynchronous event POSTs could arrive at the host out of
order. It therefore does not prove that the compact compiler emitted button-down
before movement.

The specific adapter-dependent failure mechanism predicted for the next check
is relative-baseline drift: compact deltas were created from one observed cursor
sample, while the executor read the cursor again at dispatch. Any intervening
movement would offset the endpoint; the absolute arm would be unaffected. The
harness now requires the planned observed baseline to equal the dispatch-time
baseline and the final cursor to equal the geometry-derived endpoint before an
oracle result is interpreted. It also journals each action's cursor before and
after dispatch, persists the active cell and final host state before calling the
oracle, and serializes browser event reports with client sequence numbers,
browser/host timestamps, pointer screen coordinates, and `elementFromPoint`
identity. Thus a future CPU/KVM retry can distinguish baseline drift, endpoint
error, wrong hit element, and oracle/reporting delay. These diagnostics do not
alter fixture contents, hashes, evaluation exposure, or model policy inputs.

Development-only job 135830 then falsified baseline drift for its failing cell:
the compact baseline and endpoint both matched, and browser `elementFromPoint`
confirmed that pointer-down hit the target. The separate guest mouse-up command
reported success, but no pointer-up or click acknowledgement reached the
browser, whose final button state remained held. The failure was therefore at
the split-process release/browser-ack boundary.

Compact actions now execute in one guest Python process. That process preserves
the raw move/scroll/event order, synchronizes with X11, and returns its final
cursor plus actual pointer-button mask. If an operation or mask check fails, it
releases all supported buttons and touched keys before returning a fail-loud
result. The host journals one guest-process result per compact action and
requires a zero button mask before any near-miss or gold oracle. Intentional
button-hold reset probes declare and verify the left-button mask instead.

Fixed post-action sleeps were replaced with a bounded host wait keyed to the
browser's monotonic client sequence. A click trajectory must causally report
both pointer-up with zero buttons and the checkbox-change event; other templates
likewise require their state event, and pointer templates require release. The
wait returns only after the acknowledged sequence is quiescent, and stale or
missing acknowledgements fail loudly. There is no input retry, state correction,
or oracle-dependent action: the original compact action executes exactly once.

Development-only job 135839 stopped before compact dispatch because Reset A and
Reset B produced different aggregate setup hashes. The old report did not retain
both source snapshots, so it could not identify the differing component. Fixture
readiness is now gated by an explicit layout handshake: after
`document.fonts.ready`, the page samples rounded window and DOM geometry on
distinct animation frames, reports every observation with its monotonic client
sequence, and emits `ready` only after three consecutive observations are exactly
identical. The host independently rejects early readiness. Stabilization is
bounded at 120 animation frames; it uses neither a fixed sleep nor a coordinate
tolerance.

For each reset, the selfcheck now checkpoints the complete browser state, cursor,
and button state. It separately hashes exact fixture identity, logical state,
cursor/button state, window geometry, and DOM geometry, then persists both
component payloads and a field-level hash diff before enforcing equality. Raw
event timestamps and stabilization histories remain available in the full
snapshots but are not confused with the exact final-state components. Any future
reset mismatch therefore fails before dispatch with the precise differing
component and both values already durable in `progress.json`.

Every development cell performs:

1. restore the full-RAM/disk `osworld_ready` QMP snapshot and deterministic setup;
2. require the reset-state oracle to reject;
3. execute a near miss and require rejection;
4. deliberately leave LMB held;
5. restore `osworld_ready` again and require identical initial state, cursor,
   released buttons, initial text/scroll/slider state, and oracle rejection;
6. execute the scripted gold trajectory and require a fresh oracle process to pass.

Any setup, parse, dispatch, reset, VM, or oracle exception fails the artifact;
it is never converted to a task/model failure.

## Gates

Matched rung-1a model evaluation is authorized only after all of the following:

- unit tests and the build selfcheck pass;
- the labctl CPU/KVM development selfcheck passes all 16 fixture/arm cells;
- its report proves endpoint denial, two-reset equivalence, Unicode exactness,
  signed scroll, button hold/release, and positive/negative oracle behavior;
- an owner explicitly approves opening the sealed 32 evaluation fixtures.

Model evaluation must then pair every evaluation fixture in
`native_absolute_control` and `compact_raw_phaseb`, enforce the preregistered
horizon, and refuse unmatched fixture hashes/seeds. No GPU model job is part of
this implementation.

### Declared rung 1b real-application transfer gate

Rung 1b is not implemented here. It requires a separate explicit authorization
and frozen addendum defining non-official fixtures that change underlying state
in real applications (for example a Chrome/desktop preference, a VS Code file,
Chrome scroll state, and Files or a native browser control). Its setup and
oracles must be validated on development seeds with the same two-reset and
fresh-process requirements before either model runs. Rung 1a success alone does
not authorize or predict rung 1b, later rungs, or official held-out evaluation.

## Local checks

```bash
python3 -m pytest -q osworld_parity/proper_vm_capability_ladder/rung1/tests
python3 -m osworld_parity.proper_vm_capability_ladder.rung1.selfcheck \
  --mode build --output /tmp/rung1a-build
```

The real-VM check is intentionally registered and launched through labctl using
`rung1a_vm_selfcheck_cpu_kvm.toml`; it allocates zero GPUs.
