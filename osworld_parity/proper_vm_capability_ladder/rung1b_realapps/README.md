# Rung 1b real-application development and training contract

This package is ROADMAP 3.1: isolated focus/type, scroll, and drag primitives in
real desktop applications. It is separate from rung 1a click transport and uses
only newly authored deterministic seeds.

The three state-changing tasks are:

- VS Code: focus the editor, replace a file through the shared coalesced
  clipboard compiler, and save. The Phase-B-aligned primary gate uses ASCII;
  exact Unicode remains an explicitly labelled capability probe rather than a
  silently weakened primary claim. The hidden oracle reads bytes.
- Chrome: signed scroll in a guest-local long development document. A guest-local
  recorder accepts write-only scroll events; no HTTP state/oracle read route
  exists. The host oracle reads the private state file through the VM agent.
- Files/Nautilus: drag a file into a named directory. The hidden oracle reads the
  filesystem. There is no browser-slider fallback.

`fixtures.json` has six DEVELOPMENT rows only. Every VM cell restores the pinned
`osworld_ready` RAM/disk snapshot, proves reset and near-miss rejection, restores
again, compares exact initial hidden state/UI geometry/cursor, and requires a
fresh-process gold oracle. The KVM provider must hash to
`76a8f44fab16c6dd38a4378a270e38758ba8d31885f244baedb95d8178f588d7`.

Scroll and Files cells are explicitly labelled thin-coverage capability probes.
The scroll negative is a correct-direction undershoot, distinct from both reset
and gold. Setup readiness records phase-specific guest evidence and accepts UI
geometry only after three identical observations. After each scripted action,
the harness requires a hidden-state acknowledgement followed by three fresh,
identical probes before launching the oracle process; timeout raises
`AppSettleTimeout` and cannot return a stale state.

VM cells are counterbalanced by `fixture_seed_parity_v1` (even seeds run native
then compact; odd seeds compact then native). The journal atomically records a
cell before assertions. A failing cell retains its state, stable geometry,
cursor, dispatch and settle polls, screenshots, guest/QEMU logs, traceback and
failure phase, resets cleanly, and does not stop later cells from being attempted.
Any failed cell suppresses `selfcheck.json`; details remain in `progress.json`
and the cell's `failure_context.json`.

## Training environments

`training/splits.json` separately seals 18 train, six development, and 12
evaluation IDs. Training/build tools fail closed if asked to materialize the
sealed evaluation split. Train seeds can be extended only through the collision
checked proposal helper.

`Rung1bTrainingEnv` exposes only instruction and screenshot to the policy. Its
backend retains hidden state and computes reward/done through the oracle for the
trainer. Scripted gold export, native-absolute teacher collection with rejection
sampling, and deterministic native-to-compact conversion never export expected
state, oracle state, or hidden reward. No GPU training is authorized by this
package.

## Gates

```bash
python3 -m pytest -q osworld_parity/proper_vm_capability_ladder/rung1b_realapps/tests
python3 -m osworld_parity.proper_vm_capability_ladder.rung1b_realapps.selfcheck \
  --mode=build --output=/tmp/rung1b-build
```

The real VM check is CPU/KVM-only and must run through
`rung1b_realapps_vm_selfcheck_cpu_kvm.toml`. The teacher collector remains a
data job; neither recipe allocates a GPU or opens the sealed evaluation IDs.
