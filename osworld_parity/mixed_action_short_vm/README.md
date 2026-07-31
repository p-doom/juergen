# ROADMAP 3.3: short mixed-action VM infrastructure

Status: **pre-gate infrastructure only**. No scientific evaluation row, model
inference, GPU job, or sealed evaluation payload was generated or opened while
building this package.

This package covers newly authored 2--4 semantic-step tasks combining focus or
click, Unicode-safe coalesced type, signed scroll, and explicit drag. It imports
the audited native-absolute and compact-raw executors and transport operations
from `proper_vm_capability_ladder.rung1`; it does not fork an action grammar or
guest input implementation.

## Split and oracle boundary

- `manifests/train.json` and `manifests/development.json` contain sealed,
  deterministic generator cells with disjoint parameter seeds.
- `manifests/sealed_evaluation.json` contains only reserved cell metadata. It has
  no parameter seed, instruction, expected state, or generated task payload.
  Evaluation payload custody remains in an external owner vault and
  `materialize_tasks()` fails closed for this split.
- A policy sees only instruction, opaque frame reference/digest, step index,
  horizon, reward, and done/truncated. Expected text, geometry, ordered progress,
  and final state remain trainer-side.
- Final oracle replay runs in a fresh process. Parse, dispatch, reset, and oracle
  errors are infrastructure errors; they are never rewritten as model failure.

`runtime.Episode` is the deterministic CPU contract backend used by tests.
`vm_runtime.VmEpisode` accepts owner-supplied reset, screenshot, and hidden
host-oracle hooks while dispatching through the same production executors. A
scientific VM episode cannot be constructed without a sealed, owner-approved
gate receipt. No such receipt is committed here.

## Teacher and replay contract

`NativeTeacherCollector` records absolute native actions against the visible VM
observation stream. Deterministic conversion derives compact raw-delta actions
from the recorded cursor baseline. Focus/click and Unicode typing remain one
action each; drag lowers to move-and-hold, held movement, and release. Every
compact trace seals the source native trace hash. Because drag expands into
additional turns, conversion is replayed from the identical reset and captures
a fresh observation before every compact action; native screenshots are never
blindly duplicated into the derivative trace.

Gold and near-miss replay is available for both formats. A replay is valid only
when gold passes the fresh hidden oracle, the matched near miss is a clean
negative, the final pointer mask is zero, and the common frozen action horizon
is respected.

The teacher artifact builder produces CPU contract records only. Its manifest
states that it is not scientific VM data and does not authorize a training or
model launch.

## Gates

The following order is frozen:

1. Unit tests and the CPU build selfcheck pass.
2. Native-first teacher conversion, compact round trip, split leakage checks,
   reset equivalence, gold replay, and near-miss rejection pass.
3. An owner wires `VmHostHooks` to the pinned ready snapshot and runs only the
   development manifest through CPU/KVM, proving two-reset equivalence and a
   fresh host-oracle result. This package does not auto-launch that run.
4. The owner separately materializes and seals evaluation payloads outside this
   repository, audits leakage, and signs a gate receipt bound to the committed
   sealed-metadata hash.
5. Only then may scientific paired model jobs be authored or launched. They are
   intentionally absent from the provided labctl pipeline.

## CPU-only checks

```bash
python3 -m pytest -q osworld_parity/mixed_action_short_vm/tests
python3 -m osworld_parity.mixed_action_short_vm.selfcheck \
  --output /tmp/roadmap33-selfcheck
python3 -m osworld_parity.mixed_action_short_vm.dataset \
  --split development --output /tmp/roadmap33-teacher-development
```

The corresponding labctl recipes are
`roadmap33_mixed_action_build_cpu.toml` and
`roadmap33_mixed_action_teacher_cpu.toml`. Both request `gpus = 0` and reject a
non-empty `CUDA_VISIBLE_DEVICES`.
