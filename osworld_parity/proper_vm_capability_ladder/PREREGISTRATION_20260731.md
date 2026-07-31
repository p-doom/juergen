# Proper-VM capability ladder preregistration

Status: design frozen before scientific rollout. No ladder result has been
generated or interpreted. Broad official OSWorld evaluation is not the next
training gate.

## Estimand and arms

The primary estimand at every rung is the paired task-success difference between:

1. `native_absolute_control`: the pinned off-the-shelf
   Qwen3-VL-8B-Instruct snapshot using its native absolute computer-use action
   interface; and
2. `compact_raw_phaseb`: the exported compact raw-delta Phase-B checkpoint using
   the raw-pixel delta/event/coalesced-typing executor.

There is no learned normalized-coordinate or tool-call treatment in the primary
comparison. Every task instance is run in both arms. Aggregation must fail if a
task, parameter seed, perturbation, or horizon is unmatched. Control shards may
finish before the raw export exists, but no result may be interpreted until its
registered raw mate is complete.

## Task source and separation

The ladder uses newly authored, parameterized task fixtures, not official
OSWorld task IDs, instructions, setup configs, or held-out assets. Fixtures run
inside the same resettable Ubuntu desktop VM and exercise normal applications:
Chrome, Files/Nautilus, LibreOffice Writer/Calc, and VS Code. Each template has
development seeds used only for executor/oracle self-validation and frozen
evaluation seeds generated afterward. Evaluation seeds and expected states are
sealed before either model runs. None may enter training data.

Each episode restores the pinned `osworld_ready` QCOW snapshot, runs a deterministic
setup script, verifies the initial state with a hidden host-side oracle, exposes
only the visible desktop and instruction to the policy, and evaluates final VM
state with a fresh oracle process. Setup/oracle secrets and expected values are
never displayed to the model.

## Sequential rungs

### Rung 1 — isolated state-changing primitives

Each template isolates one semantic primitive. The horizon is the minimum needed
by both action interfaces rather than forcing all primitives into one model turn.

- Click: toggle a real Chrome or desktop setting; oracle reads the underlying
  preference. Horizon 2.
- Focus plus coalesced type: focus a VS Code or browser field and enter a seeded
  string in one coalesced typing operation; oracle reads autosaved file or form
  state. Horizon 3.
- Scroll: change the scroll position of a seeded long local document in Chrome;
  an instrumented local fixture exposes scroll state only to the host oracle.
  Horizon 2.
- Drag: move a seeded file icon into a seeded Files folder, or set a native
  browser slider; oracle reads filesystem or DOM state. Horizon 4, allowing
  press/move/release for raw deltas.

Eight frozen evaluation seeds per template vary location, direction, distance,
label, and typed content without changing the requested primitive.

### Rung 2 — two-to-four-step same-application compositions

- Writer: focus, replace seeded text, apply one formatting property, save.
- Calc: select a seeded cell, enter a value or formula, confirm, save.
- Files: select a seeded file, drag it into a folder, and rename it.
- Chrome: navigate within an already-open settings page, scroll, and toggle a
  seeded setting.

Oracles inspect ODF/file/preference state, not screenshots. Horizons are 4, 6,
8, and 6 respectively and are recorded per template.

### Rung 3 — short mixed-action tasks

Tasks require at least three action classes among click, drag, scroll, keyboard
chord, and coalesced type. Planned templates include Chrome form navigation,
VS Code find/edit/save, and Writer select/replace/format/save. Horizons are
frozen per template in the range 8–12. These remain authored fixtures rather
than official OSWorld tasks.

### Rung 4 — recovery

Every clean task has a matched recovery version. A deterministic perturbation is
applied after a preregistered step and before the next screenshot: wrong field
focus, one opposite-direction scroll, selecting the wrong file, or moving a file
to the wrong folder. The same perturbation occurs in both arms. Natural
ineffective actions are also tagged when executor dispatch succeeds but the
action-specific state probe is unchanged. `recovered_after_error` is true only
when the final task oracle passes within the original horizon plus two recovery
steps. Perturbations, natural errors, and executor failures are reported
separately.

### Rung 5 — coarse official-task diagnostic

Only after rungs 1–4 pass, run a small paired pilot from the untouched held-out
OSWorld split. It remains a coarse diagnostic, never a training gate by itself.
Broad held-out OSWorld shards require a separate decision after this pilot.

## Executor and oracle contract

Before scientific shards, a non-model self-check must prove on a freshly reset
VM that both adapters can dispatch click, button hold/release, drag, positive and
negative scroll, key chords, and Unicode-safe coalesced type. It must also prove
that each setup is deterministic, each positive oracle accepts a scripted gold
trajectory, each negative oracle rejects the reset state and a near miss, and a
second reset removes all first-episode state. Any parse, dispatch, VM, setup, or
oracle exception is fail-loud and never converted to a model failure.

Every scientific row contains an explicit `MOUSE_SOLVED` boolean. It is true if
and only if the end-to-end state oracle passes; model termination text cannot set
it. Rows also contain model parse status, executor dispatch status, reset/setup
status, oracle status, action classes attempted, horizon used, natural ineffective
actions, controlled perturbation, recovery status, model/export provenance, and
the sealed task/fixture hashes.

## Shards, metrics, and stopping rule

Scientific arm/rung/shard jobs and CPU aggregation jobs all run through labctl.
Tasks are deterministically sharded by task ID; no shard owns different task
instances across arms. The aggregate refuses duplicate or missing pairs.

Primary metrics are end-to-end `MOUSE_SOLVED` rate and the paired raw-minus-control
difference. Secondary metrics are success by horizon, parse failures, executor
failures, action-class success, natural-error recovery, controlled-perturbation
recovery, and VM/setup/oracle infrastructure exclusions. Report Wilson intervals
for arm rates and a 10,000-resample paired cluster bootstrap, seed 20260731,
resampling template then parameter seed, for differences and horizon curves.

A rung is non-inferior only if the lower 95% confidence bound for the paired
success difference is above -0.15, raw point success is at least 0.60, and raw
parse-plus-executor failure is at most 0.05. The first sequential rung that fails
any criterion is the preregistered compact-raw break point. Action strata are
reported even when the overall rung passes. Later rungs are diagnostic after the
first break and cannot retroactively redefine it.
