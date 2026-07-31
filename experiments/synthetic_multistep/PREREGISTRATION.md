# Phase-A synthetic multi-step bridge (frozen before GPU evaluation)

This bridge asks whether the synthetic relative-factorial tool-call result survives
stateful use.  The primary comparison is exactly:

- absolute: `relative_factorial_abstool_act_hf_v1_run_019fb44ae67d76e1b022d4521c233dfb`
- relative: `relative_factorial_reltool_act_r32_s750_v1_run_019fb44ae67d76e1b022d4dbea28c4dd`

The legacy OSWorld `fmt_sft` step-300/900 checkpoints are explicitly outside
Phase A.  A validated r64/r256 relative capacity winner may be added later only as
a labeled sensitivity with a new recipe and manifest; it cannot replace the r32
primary after results are seen.

## Frozen design

The 80 seed-0 rung-2 scenes are the heldout episode starts.  Their first
observation is copied byte-for-byte, and cursor/bbox/center geometry is asserted
equal to the current single-step record.  Each episode exposes four successive
green-box targets.  Later targets are deterministically selected from the same
heldout bbox pool, at least 400 px from the preceding oracle cursor, and rendered
with the exact 1920x1080 rung-2 canvas, colors, 150 px box, and arrow-tip cursor.
Train and validation scene ids, bboxes, centers, and `(cursor,bbox)` tuples must
all have empty intersection with heldout.

The CPU build emits two oracle-prefix/teacher-forced views.  At each prefix the
absolute oracle emits `left_click [x,y]`; the relative oracle emits `move_rel
[dx,dy]`.  Both use the same tool-call JSON template and exact rung-2 prompts,
with only the coordinate semantics and values changed.  The resulting cursor,
next image bytes, goals, prose, and history must remain identical between
oracles, and both oracles must hit 320/320 targets.  Any failure aborts the build.

Closed-loop evaluation starts from those same episode specs.  A parsed coordinate
updates the cursor using the parity harness's normalized-grid rule: quantize the
current cursor to 0--999, add a relative delta there (or use the absolute
coordinate), map back to clipped pixels, re-render, and restate the current
cursor.  A target advances only after the model lands inside it.  Three misses on
one target end the episode.  The most recent three complete user/assistant turns
are replayed.  Raw assistant prose is neither removed nor truncated.  The two
arms have identical target schedules, sampling tuple, token limit, and
semantic-independent request seeds; after the first action, differing cursor
pixels are permitted only as a causal consequence of differing outputs.

## Outcomes fixed in advance

For attempts 1--3, report the planned-target reach CDF.  Also report episode
completion, recovery after a first miss and after each eligible miss event,
normalized trapezoidal distance-AUC (failure padded at its final distance),
positive progress/regression/stall rates, direction-reversal oscillation,
tolerant parse rate, exact tool schema rate, and coordinate-unit violations.
Request errors invalidate an artifact rather than counting as misses.

Primary directional hypothesis: relative multi-step completion and recovery are
lower and normalized distance-AUC is higher than absolute, even if first-attempt
reach improves over the original off-the-shelf relative baseline.  This is a
directional bridge, not a new causal training comparison: checkpoint provenance
differs by coordinate semantics.  All metrics are reported regardless of sign;
no metric or episode subset will be changed after seeing GPU output.

The optional preamble sensitivity uses the exact `abstool_pre` and `reltool_pre`
aliases recorded in `frozen_manifest.json`, preserves their prose byte-for-byte,
and is not part of the primary decision.

## Frozen capacity bridge (added before r64/r256 multi-step evaluation)

The single-step capacity analysis selected the tool/action-only relative cell
before any r64 or r256 output on the generated later targets was observed.  The
secondary bridge therefore evaluates both prespecified capacity levels:

- rank 64: `relative_factorial_reltool_act_r64_s750_capacity_v1_run_019fb4bf589e7bc184a34976f58a658a`
- rank 256: `relative_factorial_reltool_act_r256_s750_capacity_v1_run_019fb4becd3c75f0809cb9acce69ba5e`

Both use the already registered episode artifact
`synthetic_multistep_phasea_episodes_v1_run_019fb4f69ebe76a3aef087cbaafb92fa`
and its frozen manifest hash.  No episode, target, request seed, sampling value,
history rule, prompt, semantic, or evaluation setting is rebuilt or changed.
The pinned absolute primary artifact remains the common reference, and the
observed r32 primary artifact remains the rank-32 point of the capacity curve.

The ordered primary endpoints are (1) first-attempt planned-target reach and
(2) planned-target completion by attempts 2 and 3.  Report each rank against
the exact same absolute baseline with paired episode bootstrap intervals and
paired target McNemar counts, then report the descriptive 32→64→256 curve.
Episode completion, recovery, distance-AUC, progress/regression/stall,
oscillation, parse, strict schema, and coordinate-unit violations remain
required secondary endpoints.  There is no post-hoc composite threshold: each
endpoint and rank is reported regardless of sign, and neither capacity result
may replace the frozen r32 primary comparison.

## Frozen r256 production-format movement bridge

Before either production-format multi-step output was observed, the r256
preamble pair was frozen as a movement-only A/B bridge:

- A, normalized tool call: `relative_factorial_reltool_pre_r256_s750_capacity_v1_run_019fb4beda3472b289ae60fc612c1cea`
- B, raw deltatype: `relative_factorial_relraw_pre_r256_s750_capacity_v1_run_019fb4bec0167122b655a36ea3a4237e`

Both consume the exact already registered 80-episode/320-target artifact
`synthetic_multistep_phasea_episodes_v1_run_019fb4f69ebe76a3aef087cbaafb92fa`.
Both use their own frozen rung-2 system prompt and preamble byte-for-byte.  Full
assistant output is stored and replayed without prose stripping or truncation.
The only semantic difference in the closed loop is the trained action format:
A applies normalized `move_rel`; B parses the last action line under the audited
deltatype parser and applies `dx,dy` as clipped screen-pixel offsets.

The ordered primary endpoints are first-attempt planned-target reach and
planned-target completion by attempts 2 and 3.  Required secondary endpoints
are tolerant parse, strict schema, recovery, oscillation, episode completion,
distance-AUC, progress/regression/stall, and coordinate-unit violations.  Use
paired episode-cluster bootstrap intervals and exact paired-target McNemar
counts.  This is evidence about movement behavior only; it is explicitly not a
validation of coalesced typing or any non-movement production action.

## Frozen r256 A→B curriculum versus B→B continuation

This matched training contrast was frozen before generating the fresh stage-2
dataset or observing either stage-2 model. Both arms receive exactly 750 new
optimizer steps on one shared 2,000-train/200-validation raw-deltatype dataset
with the qualitative preamble retained. Both start from a merged r256 stage-1
model and install a fresh rank/alpha 256/256 LoRA with a fresh optimizer:

- A→B curriculum starts from
  `relative_factorial_reltool_pre_r256_s750_capacity_v1_run_019fb4beda3472b289ae60fc612c1cea`.
- B→B continuation starts from
  `relative_factorial_relraw_pre_r256_s750_capacity_v1_run_019fb4bec0167122b655a36ea3a4237e`.

Thus both lineages receive 1,500 total steps and the same number of examples;
the only stage-2 recipe difference is the merged stage-1 source checkpoint.
Stage 2 always teaches B: a prose preamble followed by the exact bare relative
pixel line `dx dy 0 ; +LMB -LMB`. Prose is never stripped. The stage-2 train
and validation geometry seeds are respectively 2026073101 and 2026073102,
independent of the stage-1 seed. Before either training may launch, the build
must prove zero exact bbox, center, `(cursor,bbox)`, and rendered-image-hash
overlap with stage-1 train/validation, frozen single-step evaluation, and every
planned multi-step target; it must also prove train/validation disjointness.

All optimizer, schedule, batching, validation, checkpoint, export, and LoRA
settings are byte-matched across arms. The launch gate additionally requires
validated tests/recipes, a conservative aggregate storage peak below 500 GB,
and a three-hour SLURM limit submitted early enough to end before 05:09 Europe/
Berlin on 2026-07-31. Failure of any gate blocks both launches.

After export, first report teacher-forced B assistant-token NLL and action-line
token NLL on the shared fresh validation split. Then evaluate both models in
the frozen 80-episode B raw-pixel/preamble closed loop. Its ordered primary
endpoints are first-attempt reach, reach by attempt 2, and reach by attempt 3;
episode completion, recovery, distance-AUC, progress/regression/stall,
oscillation, parsing, strict schema, unit violations, and miss types remain
required. Pair all differences by the exact validation example or episode/
target as appropriate. The directional hypothesis is that A→B improves B
generalization over B→B; every endpoint is reported regardless of sign. This
is a movement-only curriculum test, not evidence about coalesced typing.

### Prepared matched low-LR rescue (not authorized for submission)

The frozen 1e-4 A→B run showed finite, clipped optimization shocks at steps
440--480, recovered near step 500, then showed renewed gradient/loss spikes at
steps 510--560; the exact matched B→B batches remained stable. Before seeing
either final evaluation, a clean 2×2 lineage-by-learning-rate sensitivity is
therefore prepared at 5e-5. This is exactly half the original rate: it preserves
the WSD schedule, warmup, optimizer, clipping, data order, seed, 750 steps,
fresh rank-256 LoRA, validation, checkpoint, and export settings while reducing
the update scale implicated by the lineage-specific shock. Both A→B and B→B
must run; changing only A is forbidden. The pair remains unsubmitted unless
explicitly approved after the original matched outcomes are audited.

The rescue pair uses the same registered stage-2 artifact and original merged
stage-1 sources. Its conservative incremental source+two-output+cache bound is
the same sub-500-GB pair envelope as the original launch. Retaining both current
merged HF outputs raises the all-lineage aggregate above 500 GB even after the
current Orbax sources are removed; this must be reported rather than hidden.
Current Orbax removal remains forbidden until both current exports, both
teacher-forced evaluations, both closed-loop evaluations, and both paired
analyses verify. A rescue submission, if authorized, uses a two-hour limit and
the same hard 05:09 Europe/Berlin deadline; it must not be submitted after the
deadline feasibility gate closes.

At 01:35 Europe/Berlin, after the original A→B instability recurred beyond
step 500, explicit approval was received to submit exactly this matched 5e-5
pair. No other learning-rate or single-lineage variant was authorized.

## Frozen matched typing-format factorial

Before any typing-cell GPU training or evaluation output was observed, a
separate 2×2 lineage-by-target-format transfer experiment was authorized. Its
lineages are the frozen r256 stage-1 preamble checkpoints used above: A is
`relative_factorial_reltool_pre_r256_s750_capacity_v1_run_019fb4beda3472b289ae60fc612c1cea`
and B is
`relative_factorial_relraw_pre_r256_s750_capacity_v1_run_019fb4bec0167122b655a36ea3a4237e`.
Each lineage is continued independently on target format `coalesced` or
`perkey`. Every cell receives exactly 750 steps from a fresh rank/alpha
256/256 LoRA and fresh optimizer at learning rate 5e-5. All other optimizer,
schedule, batching, seed, validation, checkpoint, and export settings are
byte-matched.

The shared dataset contains exactly 2,000 training and 200 validation pairs.
Within each split, coalesced and per-key records have identical sample order,
images, target strings, and prompt prose; only the action serialization
differs. The immutable build must demonstrate 4,400/4,400 production-parser
parse/format/execute round trips and 4,400/4,400 exact typed-string outcomes,
train/validation disjointness, and zero sample-id or rendered-image overlap
with stage-1, single-step evaluation, multi-step episodes, or curriculum data.
Failure of any gate blocks all four cells.

The primary endpoint is first-attempt exact typed-string success. Let
`I = [p(A,coalesced)-p(B,coalesced)] -
[p(A,perkey)-p(B,perkey)]`. A lineage-by-format interaction is claimed only
if the paired interval for I excludes zero and `|I| >= 5` percentage points.
A coalescing capacity benefit is claimed only if the lineage-averaged paired
coalesced-minus-per-key effect is at least 5 percentage points, its lower
interval bound is above zero, and action-token NLL points in the same
direction. Report all four cell rates, all simple effects, the interaction,
paired intervals, action-token NLL, parser/schema failures, and exact typed
strings regardless of sign. This typing study is separate from, and cannot
replace, the movement curriculum evidence.

Each model is evaluated on the same ordered 200 validation examples exactly
once with greedy generation (`temperature=0`, `max_tokens=256`). The final
nonblank action line is executed only after parsing with the pinned production
`action_parser.py` (SHA-256
`f916757d17e4a5f53627510616ffff411e9109e8737d1309067c6338caae4a9a`).
Exact success requires the executed string to equal the target, zero mouse and
scroll deltas, the target-format element schema, and a canonical final action
line. Parse, schema, zero-delta, and request failures are reported separately.
Assistant- and action-token NLL are teacher-forced on the same 200 examples.

The launch requires a conservative incremental storage bound no greater than
700 GB and a four-cell projected training-plus-export completion no later than
04:40 Europe/Berlin on 2026-07-31. At 02:19 Europe/Berlin the deadline was
tightened from 04:45 to 04:40 before any cell was submitted.

### Authorized step-250 typing recovery amendment

At 02:49 Europe/Berlin, all four matched cells had completed and atomically
finalized update 250, then failed in the first in-loop validation forward with
the same GPU `RESOURCE_EXHAUSTED` error. The trainer saves before validation;
validation consumes a separate iterator, calls the model without the training
RNG, and does not mutate the optimizer. Exact resume from each immutable
step-250 optimizer, RNG, and training-iterator state was therefore authorized.
Each recovery must hash-gate the checkpoint metadata, train-state metadata,
training iterator, LoRA metadata, and parent log; restore global gradient step
250 and optimizer microstep 2000; retain the same dataset/order, optimizer,
schedule, seed, and step-750 endpoint; and use `resume=required`. The only
allowed change is removal of in-loop validation. The frozen external ordered
200-example greedy evaluation and all claim gates remain unchanged. Recovery
launch is blocked unless a conservative 75-minute projection still finishes
before 04:40 Europe/Berlin.

At 02:58 Europe/Berlin, the three not-yet-started recovery jobs were cancelled
while still pending and replaced by new immutable recipes that additionally
allow healthy node hai008. It is the same H100 80 GB class and uses the same
shared filesystem and environment; concurrent probe jobs confirmed node
health. This is an operational capacity amendment only. No data, checkpoint,
optimizer, schedule, seed, runtime limit, or evaluation field changed. The
already-running A/coalesced recovery was not altered.

At 03:02 Europe/Berlin, the assumed 04:51 Nishant reservation was found not to
exist: job 134957 was an ordinary pending job on other nodes. The three still
pending recovery submissions were cancelled and replaced with snapshots that
allow only hai003/hai007/hai008 and enforce a corrected 09:00 Europe/Berlin
hard finish deadline with the same 75-minute start projection. This deadline
precedes the user's 10:00 return and changes no scientific field. The running
A/coalesced snapshot retained its original 04:40 gate.

The first 09:00-gated replacements exited before JAX startup because labctl
garbage-collected `source/juergen_rft` for the terminal failed parent runs
between the earlier successful A/coalesced preflight and their starts. This
was a shared-filesystem provenance-path failure, not a hai008 compatibility
failure. Before GC, all four parent trainer snapshots had independently
matched SHA-256 `4bff34fa17dfd7b22d215aaa18828cbfac2d9f463f3798c5dd70ca7c12c84aa9`.
The surviving immutable parent contexts all seal source hash
`ae6a90779440de25021a0b9e05743bb9ee9889474e7fbb5d23e173aee7c9de8f`.
Replacement preflight may use that sealed context hash only when the previously
audited snapshot has been GCed; every checkpoint, iterator, LoRA, and log hash
gate remains mandatory.

### Frozen TP1-to-TP2 typing recovery amendment

At 04:57 Europe/Berlin on 2026-07-31, before any GPU output from this recovery
was observed, explicit approval was received for one further operational
amendment. All four cells retain their immutable finalized first 250 updates
from TP1/FSDP1/DP1, but their last 500 updates may run on exactly two local
H100s as TP2/FSDP1/DP1. This changes parameter and optimizer sharding and the
order of tensor-parallel floating-point operations. Even with a bit-exact
restore, the last-500 update trajectory is therefore not numerically identical
to a TP1 continuation and must be reported as a TP1-first-250 to TP2-last-500
run. It must not be described as a single-topology replay.

The topology transition is byte-matched across the complete 2x2 factorial.
No dataset, target serialization, model, LoRA rank/alpha, optimizer, learning
rate or schedule, global batch size, accumulation grouping, seed, update
count, sample order, checkpoint endpoint, external evaluation, or claim rule
changes. In particular, the frozen primary interaction remains
`I = [p(A,coalesced)-p(B,coalesced)] -
[p(A,perkey)-p(B,perkey)]`, with the same paired interval and five-percentage-
point threshold, and the lineage-averaged coalescing-benefit estimand and NLL
direction gate remain unchanged. Thus the topology change is a matched
operational nuisance across all four cells rather than a change to the
estimand.

Restore must construct every target leaf from the checkpoint's global shape
and dtype plus the fresh TP2 leaf's target sharding. The physically stored RNG
must be requested as replicated `uint32[2]` and reconstructed by Orbax as its
typed scalar key. The prior fresh-optimizer dtype canonicalization is forbidden
because it would round the checkpoint's 1,512 fp32 accumulator and Adam-moment
leaves to bf16. Before training, the recovery must prove exact source/target
path identity, expected TP1 and TP2 mesh shapes, checkpoint-driven dtypes,
bitwise equality of all restored train-state leaves, RNG key data, counters
(gradient step 250, optimizer microstep 2000, accumulation remainder zero),
the exact unadvanced Grain iterator state, endpoint hashes, provenance, and an
immutable byte-identical private Orbax clone. Any mismatch fails closed.

Only A/coalesced may launch initially. It may continue after restore, but no
other cell may launch until the live pilot has passed all restore gates and
logged a finite update 260 without any in-loop validation invocation or
validation OOM. After that gate, the other three use exactly the same code,
TP2 topology, restore contract, and 09:00 Europe/Berlin hard deadline. The
ordered external 200-example greedy and teacher-forced evaluations remain the
only validation used for the factorial. Every immutable TP1 source parent is
retained until all four HF exports and matched external evaluations validate.
