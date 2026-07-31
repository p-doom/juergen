# Stage 5: closed-loop parity and on-policy RFT

Status: **construction only; no model, GPU, official-heldout, or promotion run is
authorized by this package.** Every emitted learner plan has
`launch_authorized=false`.

This package defines the missing task-level contract between a VM actor and a
learner. It is intentionally separate from the historical `rft/` sample-level
toolkit. A rollout batch pins one checkpoint fingerprint, restarts incomplete
episodes from a deterministic reset, and records every transition:

- policy version, checkpoint digest, source commit, action schema, and sampling;
- reset snapshot/setup/task digests and seed;
- screenshot and structured-state hashes before and after every action;
- raw output, parsed action, parser, dispatch result, log probability, reward,
  `done`, task success, timing, and failure class;
- an atomically committed complete episode and content-addressed artifacts.

The first learner method is rejection SFT over complete successful on-policy
episodes with unit weight. `reward_weighted` exists only as an explicit
experimental mode. The learner output is never allowed to overwrite or hot-swap
the collection actor; it may become an actor only in a new iteration.

## Package map

- `schema.py`: versioned episode, step, policy, reset, action, state, and image
  records.
- `collector.py`: adapter protocols, deterministic reset gate, content-addressed
  artifacts, and episode-atomic resume.
- `replay.py`: independent offline artifact audit and live action replay.
- `contamination.py`: task-id plus task-content-digest exclusion.
- `rft.py`: task-level rejection or explicitly experimental reward-weighted data.
- `metrics.py`: strictly separate single-step/multi-step summaries and exact
  matching to a native-absolute control.
- `learner.py`: launch-disabled actor/learner handoff and exact resume identity.
- `gates.py`: missing-metric-fails preregistered decision rules.
- `pipeline.py`: idempotent stage receipts used within labctl stages.

The VM/model integration is injected as `module:function`. A collection factory
receives `policy=` and `rollout_root=` and returns `(environment, actor)` matching
the protocols in `collector.py`. This keeps VM-provider code out of the scientific
record contract while forcing it to expose deterministic reset and state.

Production blocklists must contain at least one official-evaluation task ID or
content digest. An empty blocklist fails closed; only a file explicitly marked
`testing_only=true` may be empty for CPU mocks.

## Local construction checks

```bash
cd stage5_rft
python -m pytest -q
python -m stage5_rft.cli --help
```

The labctl templates in `labctl/` are not launch-ready: every external path and
adapter marked `REPLACE_ME` must be pinned, construction gates must pass, and a
separate authorization artifact is required before any model/GPU work.

See `PREREGISTRATION.md` for the promotion decision rules and `SCHEMA_AUDIT.md`
for why the older runtime records are adapted rather than imported.
