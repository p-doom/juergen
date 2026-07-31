# Stage-5 preregistration

Status: construction-only draft frozen before any official-heldout, model, GPU,
or learner run. Thresholds are machine-readable in
`config/promotion_gates.json`; prose cannot override that file after observation.

## Scope and units

The experimental unit is a complete task-level VM episode. Incomplete episodes
are never learner examples and are restarted from reset on resume. All actions in
one collection come from one immutable policy/checkpoint fingerprint. A learner
checkpoint may act only in a new collection iteration.

The task source is train or train-adjacent. Official evaluation task IDs and task
content digests form a blocklist. Both identity types are checked before
collection and again before learner-data emission. Labels such as `heldout`,
`test`, and `official_eval` are rejected even if a blocklist is empty.
An empty production blocklist is itself a gate failure; `testing_only=true` is
reserved for synthetic CPU tests.

## Conditions and control

Single-step sentinel episodes (`max_steps=1`) measure immediate grounding.
Multi-step episodes measure recovery and state propagation. Their numerators and
denominators are reported separately and are never pooled.

The control is the matched native-absolute policy. Every candidate cell must have
exactly one control cell with the same task content, instruction, VM snapshot,
setup, reset protocol, seed, horizon condition, maximum steps, and sampling tuple.
The control policy must declare `role=native_absolute_baseline` and an absolute
action schema. Missing or duplicate pairs fail closed.

Primary metrics are task success for each condition and the paired candidate minus
native-absolute difference in percentage points. Diagnostics are return, episode
length, parse and dispatch rates, first-step terminal success, multi-step recovery,
infrastructure failure rate, and the full failure taxonomy.

## Failure taxonomy

Failures are one of reset failure/non-determinism, policy timeout/error/provenance
mismatch, parse/schema/invalid action, dispatch/observation/reward/VM error, task
failure, maximum steps, actor interruption, replay divergence, or contamination.
Infrastructure failures are reported separately and are never counted as ordinary
task-policy failures without disclosure.

## Construction gates

Before any model or official evaluation run:

- 100% trace completeness and state/screenshot joins;
- 100% deterministic-reset agreement in CPU mocks and then a separately
  authorized VM smoke;
- 100% independent replay agreement;
- zero task-ID or content-digest contamination;
- zero served-policy provenance mismatches;
- 100% episode-atomic kill/resume behavior.

Missing metrics fail. Passing construction gates does not authorize a launch.

## Closed-loop parity gates

On a separately authorized, frozen evaluation design:

- pair coverage is 100% in both conditions;
- candidate minus native-absolute task success is at least -5 percentage points
  separately in single-step and multi-step conditions;
- parse rate is at least 99% separately in both conditions;
- infrastructure failure rate is at most 1% separately in both conditions.

Exact trajectory matching is not a gate. The historical stack has material
multi-step decoding non-determinism; task-level outcomes and paired distributions
are the estimands. Confidence intervals and candidate-only/baseline-only successes
are reported, but cannot replace the frozen finite-benchmark decision rule.

## Learner and promotion gates

The first eligible learner is rejection SFT: keep complete successful on-policy
episodes above the frozen return floor and assign unit weight. No reward-model,
importance-ratio, advantage, or policy-gradient claim is made. Reward weighting is
an explicit later experiment and is not eligible for the first promotion.

The dataset split is stable at task level. Parent actor checkpoint digest and data
manifest digest jointly key resume. Checkpoints are written outside both parent
actor and dataset paths. A child is eligible for a new rollout iteration only if:

- learner method is rejection and contamination remains zero;
- versus its frozen parent, single-step success changes by at least -2pp;
- versus its frozen parent, multi-step success improves by at least +2pp;
- versus matched native absolute, child multi-step success is at least -5pp.

Those comparisons require a new on-policy child collection. Parent rollouts cannot
be relabeled as child rollouts, and post-training teacher-forced loss is not a
substitute.

## Stop and amendment rules

Any gate failure stops the downstream pipeline. An amendment changes the config
version and is documented before observing the affected result; the old report is
retained. Infrastructure corrections may rerun exact cells only after recording
the failure class and preserving the failed trace. No threshold is tuned on the
official heldout set.
