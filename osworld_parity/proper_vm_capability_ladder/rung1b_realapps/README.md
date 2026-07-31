# Rung 1b real-application development and training contract

This package is ROADMAP 3.1: isolated focus/type, scroll, and drag primitives in
real desktop applications. It is separate from rung 1a click transport and uses
only newly authored deterministic seeds.

The three state-changing tasks are:

- VS Code: focus the editor, replace a file with exact Unicode text through the
  shared coalesced clipboard compiler, and save. The hidden oracle reads bytes.
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

