# Action/runtime schema audit

Audit source: `origin/franz/training-eval-reproduction-20260731` at
`8f20ceb9d08184ec02ff1c43939c18c2f809e43d`, read-only.

## Existing records

`eval/action_parser.py` has useful action parsing dataclasses (`Action`,
`ComputerUseCall`, and `DeltaTypeAction`) but its representation is an executor
format, not a rollout record. It has no checkpoint identity, reset identity,
observation/state hashes, reward, terminal flag, or learner provenance.

`eval/osworld_vm_client.py::StepResult` carries screenshots and evaluator scores
at runtime, but it is ephemeral and does not pin raw model output, parsed action,
state before/after, reset seed/snapshot, checkpoint digest, or atomic episode
completion.

`eval/freeroll.py::StepLog` is evaluation-oriented. It records prompt/output and
some execution detail, but not a deterministic reset proof, content-addressed
screenshots, exact state joins, per-step reward/done, actor checkpoint digest, or
task-level resume boundary.

`experiments/synthetic_multistep/proper_vm_stage2_closed_loop/closed_loop_contract.py`
correctly preserves actual cursor state after misses and fails closed on invalid
dispatch. It is intentionally specialized to synthetic cursor targets and cannot
represent arbitrary task-level VM state or learner provenance.

The historical `rft/` package at commit `b8731fc` is a reusable offline
sample/completion rejection toolkit. Its records and task-level split controls are
valuable precedent, but it does not collect task-level closed-loop VM episodes or
enforce one immutable actor over an episode batch.

## Stage-5 decision

Stage 5 therefore does not import any of those dataclasses. VM adapters may use
the existing parsers and executors internally, but they must serialize the result
into `ActionTrace` and declare the exact parser/action schema string. This avoids
coupling Stage 5 to a branch-only runtime while retaining the provenance needed to
reconstruct how an action was interpreted.

The adapter boundary requires:

1. raw screenshot bytes and structured state from reset and every transition;
2. raw model output plus parsed JSON action and parser name;
3. exact served-policy fingerprint on every response;
4. reward, environment `done`, task success, and typed failure;
5. deterministic reset inputs and preregistered expected initial hashes.

Changing an existing parser or executor is therefore visible as an action-schema
or source-commit change, not silently absorbed by a rollout labeled with the old
policy version.
