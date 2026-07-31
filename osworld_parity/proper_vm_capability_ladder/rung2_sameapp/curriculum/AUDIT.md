# Reuse audit

This scaffold was added after inspecting the existing proper-VM ladder and the
Stage-4 teacher-SFT branch. It intentionally reuses these contracts:

- `rung1/curriculum.py` and `curriculum_spec.json`: Phase-B's primary coverage
  vocabulary (click, signed vertical scroll, drag, coalesced typing, and the
  keyboard events needed for hotkeys), deterministic seeds, paired-format
  identity, and balanced final pointer state.
- `rung1b_realapps`: real VS Code UTF-8 file, Chrome scroll, and Files
  filesystem verifier shapes, including hidden-state rather than screenshot
  scoring and fresh-process oracle execution.
- `rung2_sameapp/actions.py`: the existing symbolic operation model and the two
  native-absolute/compact-relative compilers. Semantic task identity is above
  this layer; action encoding is selected only after a repeated-reset live
  binding. Resolved action/event counts and hashes are compiler receipts, while
  manifest counts are conservative admission caps.
- `rung2_sameapp/fixtures.py`, `oracle.py`, and `trajectory.py`: sealed task
  hashes, 2–4-step/horizon checks, exact reset signatures, reset/near-miss/gold
  oracle polarity, and Writer/Calc/Files/Chrome state probes.
- Stage-4 `experiments/teacher_sft` (commit `009d6fb`): train-derived split
  fail-closure, hash-addressed provenance, balanced input rejection, and
  separation of task records from action conversion. Its offline data pipeline
  is not copied or invoked here.

The old rung-2 package includes an earlier sealed-eval manifest. This scaffold
does not load, copy, enumerate, hash, or depend on that file. Its loader rejects
anything except `train` and `development` before constructing a path. Those two
materialized splits reuse family IDs only as a clearly labelled within-family
development split with disjoint seeds. Future family-heldout validation and
sealed evaluation require externally assigned family IDs disjoint from the five
materialized families. The registry contains only those abstract commitments;
it contains no future family ID, seed, instruction, asset, expected state, or
near miss.

The initial implementation also generated seed-based coordinates. The
successor removes them: tasks now name targets only, existing VM probes provide
versioned live geometry/cursor values, and runtime validation rejects setup
state or repeated-reset geometry/cursor/viewport drift before compilation.
Chrome refreshes geometry and cursor after scrolling before its final target is
compiled. Fixture contract checks read application artifacts through the
declared extractor and use isolated oracle processes rather than scripted
expected-state echoes.

The production replay no longer imports or calls the direct symbolic
`compile_native`/`compile_compact` path. It consumes session-issued reset
generation evidence backed by a single-use receipt emitted after the provider's
actual snapshot transition. Provider receipts bind before/after state hashes,
prior/new generation IDs, append-only native `loadvm` telemetry indices and
records, reset ordering, and the target snapshot. Provider state is canonicalized
before the call so a returned live dictionary cannot alias the pre-reset value;
generation IDs derive from the immutable observations and equal/no-op transitions
are rejected. A pre-reset guest nonce sentinel must also disappear after
`loadvm`, so a provider cannot satisfy the receipt with telemetry alone; the
production CLI independently pins the provider source hash. Identical
raw observations cannot be re-attributed. Replay compiles and executes semantic
segments sequentially, and aggregates only sealed executor receipts whose full
dispatch journals have exact cardinality, ordering, adapter, operation, cursor,
and atomic evidence. Cursor evidence starts at the compiled binding cursor,
continues across action boundaries, and ends at the compiled final cursor. The Chrome refresh transition records pre/post binding
revisions and accepts only the ledger-recorded executed step-2 receipt object.
Fixture polarity requires successful fresh-oracle status as well as the expected
boolean result. The two legacy-named labctl recipes now enter the same hardened
VM replay and declare all runtime/setup inputs; their setup artifact placeholder
must be replaced by an independently registered producer artifact before use.
The superseded build replay and legacy collector are disabled so these checks
cannot be bypassed through a second production entry point.
