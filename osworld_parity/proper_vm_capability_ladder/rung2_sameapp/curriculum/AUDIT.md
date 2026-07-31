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
  this layer; action encoding is selected only by `program.compile_program`.
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
state or repeated-reset geometry drift before compilation.
