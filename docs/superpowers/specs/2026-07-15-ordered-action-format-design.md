# Ordered Computer-Use Action Format

Date: 2026-07-15

## Objective

Introduce an ordered action representation that preserves the relative order of
mouse movement, scrolling, keyboard transitions, and mouse-button transitions
inside each observation interval. The representation must remain a Stage-06
projection so it can be ablated without rerunning vision annotation or
trajectory assembly.

This change covers the data-pipeline action projection only. It does not modify
the evaluator, or implement coordinate normalization, coarse quantization,
random mouse scaling, or context-aware trajectory trimming.

## Existing pipeline boundary

Stage 01 retains a timestamped, globally ordered event timeline. Stage 02 joins
those events to observation intervals. Stage 05 retains both the ordered events
and the existing aggregate `action_bin`. Stage 06 is the only stage that renders
action text.

The existing aggregate representation loses sequences such as
`move -> click -> move`, because all movement is reduced to one interval-wide
delta and all discrete transitions are appended afterward. The new format will
derive its output from Stage-05 `events`, while the aggregate remains available
for an explicit v1 ablation.

## Considered approaches

### Interval-wide aggregate

Keep the existing `<dx> <dy> <scroll> ; +KEY -KEY` representation. This is
compact but cannot represent ordering between continuous and discrete events.
It remains available only as `aggregate_delta_keys_v1`.

### Raw ordered events

Serialize every recorded mouse and scroll event. This is maximally faithful but
far too verbose. In a deterministic sample of 1,000 source keylogs, raw mouse
events outnumbered keyboard transitions by 7.52 to 1.

### Ordered events on a motor grid

Project continuous events onto a finer internal grid inside each observation
interval, while retaining every discrete transition at its exact ordered
position. This preserves useful paths and reversals without creating additional
screenshots or assistant turns. This is the selected design.

## V2 action language

An assistant turn contains one ordered mini-program. The grammar is:

```text
action      := "NO_OP" | primitive (";" primitive)*
primitive   := move | scroll | down | up
move        := "move(" integer "," integer ")"
scroll      := "scroll(" integer "," integer ")"
down        := "down(" input_name ")"
up          := "up(" input_name ")"
integer     := "-"? digit+
input_name  := canonical keyboard or mouse-button name
```

The canonical serializer uses `; ` between primitives and no spaces inside
function arguments:

```text
move(4,-1); down(LMB); move(6,-1); up(LMB); scroll(0,-3)
```

There are no `click`, `tap`, `type`, `wait`, or timing primitives. A click is:

```text
down(LMB); up(LMB)
```

The executable input names remain the existing normalized names, including
`LMB`, `RMB`, `MMB`, and the current rdev-derived keyboard names. The parser
must accept the complete set produced by the normalizer rather than assuming
only ASCII alphanumeric key names.

## Continuous-action projection

The ordered v2 default uses `continuous_action_hz = 10`. This is an internal
100 ms motor grid within each observation interval; it is not another frame or
observation FPS. A two-second observation interval can therefore contain up to
20 ordinary motor ticks, while still producing one assistant turn.

Stage 06 processes each step's events in their existing timestamp and source
order. A pending continuous accumulator is identified by:

- observation interval;
- 100 ms motor-tick index relative to the interval start; and
- event kind (`move` or `scroll`).

The accumulator is flushed when any of the following occurs:

- the motor tick changes;
- the continuous event kind changes;
- a `press` or `release` event occurs; or
- the observation interval ends.

Consequently, a discrete transition is always an ordering barrier, even when
movement occurs on both sides within the same 100 ms tick:

```text
move samples -> press LMB -> move samples
```

becomes:

```text
move(4,-1); down(LMB); move(2,0)
```

Movement accumulates both `dx` and `dy`. Scrolling independently accumulates
both recorded axes and renders as `scroll(dx,dy)`. V2 does not collapse scroll
to one scalar.

Each accumulator sums in floating point and rounds each component to an integer
only when flushed. `move(0,0)` and `scroll(0,0)` are omitted. A one-axis action,
such as `move(0,5)`, remains valid. Empty motor ticks are omitted and do not
create `NO_OP` primitives. If no primitive remains for the complete observation
interval, the assistant target is the single token `NO_OP`.

The v2 baseline does not normalize, clip, bucket, or randomly scale continuous
values. These transformations remain independent future ablations.

## Ordering and state semantics

`press` maps to `down`, and `release` maps to `up`. Their order relative to all
other events is preserved. Equal-timestamp events retain `source_event_idx`
order.

The action language is stateful within an executing trajectory:

- relative movement depends on current cursor position;
- `down(X)` adds an input to the held set;
- `up(X)` removes it;
- held state persists across assistant turns and `NO_OP`; and
- ordinary application changes do not reset input state.

Independent tasks and training samples must not share executor state. Enforcing
that rule in the evaluator/runtime is explicitly outside this implementation.
The data projection reports state anomalies so later runtime work can apply
cleanup without silently changing training labels.

The formatter does not invent missing transitions, synthesize releases, or
silently rebalance demonstrations. It reports duplicate downs, dangling ups,
and non-neutral trajectory endings as diagnostics. Context-aware trimming and
the treatment of `UNCAPTURED` spans require Stage 02/05 to retain context
metadata and are intentionally a separate data-validity change.

## Stage-06 selection and provenance

Stage 06 and `build_sft.py` expose an action-schema selection with these values:

- `ordered_events_v2`, the new default;
- `aggregate_delta_keys_v1`, the explicit aggregate ablation.

They also expose a positive `--continuous-action-hz` value whose default is
10. Stage 06 applies it only to `ordered_events_v2`; the v1 manifest records
the field as null. This makes 5, 15, and 30 Hz projections available without
changing the observation view or regenerating Stage 05.

For v2, Stage 06 renders from `step["events"]`. For v1, it renders from
`step["action_bin"]`. Stage 05 remains message-format-neutral and continues to
retain both fields.

The Stage-06 manifest records:

- `action_schema`;
- `continuous_action_hz` for v2, otherwise null;
- `primitive_counts` with `move`, `scroll`, `down`, and `up` counts;
- the number of `NO_OP` turns; and
- `state_diagnostics` with duplicate-down, dangling-up, non-neutral-trajectory,
  and held-at-trajectory-end counts.

Splitting, prompt policy, image handling, and terminal-token policy remain
unchanged. Idle thinning and `NO_OP` sampling also remain unchanged in this
change.

## Error handling

Projection errors use the existing Stage-06 rejected-record mechanism and
include the trajectory, variant, exception type, and detail.

Source events missing required continuous components or discrete input names
are invalid. Unknown executable event kinds are rejected rather than silently
reordered. Non-executable context metadata is not an action and is never
serialized.

## Verification

Unit tests will cover:

- `move -> click -> move` ordering;
- movement and click on both sides of one motor-tick boundary;
- movement and click inside the same motor tick;
- independent ordered 2D scroll projection;
- 10 Hz motor-grid aggregation;
- integer rounding after accumulation;
- omission of zero movement and scroll;
- stationary clicks without a leading `move(0,0)`;
- `NO_OP` for an empty mini-program;
- preservation of discrete transition order;
- explicit v1 and v2 Stage-06 selection and manifest provenance; and
- held-state anomaly diagnostics without label mutation.

The full data-pipeline test suite, Ruff formatting and linting, Python
compilation, and relevant shell syntax checks must pass before the
implementation is considered complete. No evaluator file is changed.

## Deferred ablations

The following remain separate from the ordered-format implementation:

- experiments comparing the supported 5, 10, 15, and 30 Hz projections;
- coordinate normalization and bounded representations;
- coarse or non-uniform quantization;
- random trajectory-level mouse scaling during training;
- scroll scaling;
- context-aware neutral-boundary trimming; and
- policies for `UNCAPTURED` spans;
- evaluator parsing and left-to-right execution of v2 actions; and
- runtime held-input tracking and cleanup.
