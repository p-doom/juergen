# Handoff: standalone sequential goal-memory annotation

Date: 2026-07-31  
Repository: `/fast/project/HFMI_SynergyUnit/yll/juergen`  
Branch: `thinking-training`  
HEAD before this work: `e263560 feat(cua): add strict semantic relative-step training pipeline`

All changes described below are currently uncommitted. Do not work in the Franz tree. The user explicitly corrected the working repository to this Juergen checkout.

## Objective

Add `sequential_goal_memory` as a standalone, resumable Stage 03b annotation method. It consumes only Stage 02 realigned events plus the Stage 03 filter artifact, infers hierarchical goals and causal decision thoughts, maintains rolling memory/checkpoints, links missions across days for the same user, and publishes a Stage 04-compatible artifact. It must never invent or modify computer actions.

The intended flow is:

```text
02 realign --+-- raw events ------------------+
03 filter ---+-- 03b sequential annotations -+--> 04 conversations --> 05 lengths --> 06 records
             +-- keep/drop/dead-zone mask ----+
```

## User decisions and discussion so far

- `sequential_goal_memory` must not depend on `hindsight_fold`. It is independently dispatched by the existing generic annotation framework.
- Existing annotation methods and existing Stage 04 action/thinking modes must remain compatible.
- Kimi-K2.6 was used for the real annotation smoke tests. The output manifests record that provenance.
- Thoughts should be produced immediately at the causal decision anchor, using the current screenshot, rolling memory, active goal path, prior real actions, and the upcoming real Stage 02 action. They should not be retrospective justifications.
- Thoughts should be free-form and sparse. They should appear for meaningful orientation, strategy, observed outcomes, goal switches, errors, or exact constraints—not routine motor execution. The user explicitly rejected hardcoded thought fields.
- The rolling memory is the important long-horizon state. It should be used when generating thoughts and should be saved at every semantic event. The current implementation does save `memory_before` and `memory_after` for every event.
- Checkpoints use the required seven-field tagged format, but free-form rolling memory does not use those headings.
- The user requested a downloadable, standalone HTML review with screenshots embedded at good resolution. That exists and embeds original JPEG bytes by default.
- The user questioned whether five-decision samples are genuinely long-horizon. After researching long-context and agent-memory training—and UI-TARS-2 in particular—the conclusion is that fixed five-decision/stride-four windows must not be the primary projection.
- Most recent decision: training must mirror inference compaction. Raw chronological context should continue until about 70% of configured screenshot/context capacity, then produce a checkpoint-control target, clear old image history, and resume from the same goal, checkpoint, and current screenshot. Segment length is variable and capacity-driven. Five recent decisions may be an auxiliary working-memory view, but must not imply checkpointing every five turns.

## Research conclusion relevant to the redesign

UI-TARS-2 formalizes each decision as:

```text
instruction + recent high-fidelity working memory + current observation
+ compressed episodic memory
-> immediate thought + next action
```

Its SFT data is collected in a live environment: the current model proposes a thought/action, a human accepts or replaces it, the selected action executes, and the true next state is captured. Successful trajectories are recycled into SFT. However, the exact serialized UI-TARS-2 SFT sample, loss mask, value of `N`, and episodic-memory update rule are not published. ArXiv v2 has no appendix; its LaTeX source has the appendix command commented out. The original UI-TARS repository does publish an older `data/training_example.json` showing historical turns masked and only the final next action trained, but that is not proof of UI-TARS-2's exact serialization.

Primary sources discussed:

- UI-TARS-2: https://arxiv.org/abs/2509.02544
- Original UI-TARS: https://arxiv.org/abs/2501.12326
- Original released sample: https://github.com/bytedance/UI-TARS/blob/main/data/training_example.json
- LongAlign: https://arxiv.org/abs/2401.18058
- Recurrent Memory Transformer: https://arxiv.org/abs/2304.11062
- MEM1: https://arxiv.org/abs/2506.15841
- ReSum: https://arxiv.org/abs/2509.13313
- AgentFold: https://arxiv.org/abs/2510.24699

The practical lesson is that a short raw working-memory suffix can still participate in long-horizon training if it is causally conditioned on the correct rolling memory. But our runtime only compacts near 70% capacity, so the training projection must expose the same distribution and cadence.

## Implemented work

### Generic Stage 03b framework

- Added optional method-owned `goal_rows_from_result`, `finalize_dataset`, and `requires_pilot_review` hooks to the annotation registry.
- Added an optional dataset-finalize callback to the generic concurrent driver. Existing methods do not provide it, so their behavior is unchanged.
- Added `DatasetFinalizeContext` for cross-unit reduce/finalize work.
- Stage annotation now discovers the Stage 02 clips manifest from the Stage 03 filter manifest when possible.
- Added full-run pilot gating. Partial runs (`--limit`, `--day-filter`, `--day-t1`, or shard runs) count as pilots; an unrestricted run of this method requires a review JSON with all six gates and `reviewed_by`.
- Dataset finalization is skipped if any unit failed.

### Standalone `sequential_goal_memory` method

Location: `data_pipeline/realigned_pipeline/annotation/methods/sequential_goal_memory/`

The method has resumable, input-hashed passes:

1. `01_prepare.json`: constructs stable semantic events and filter/dead-zone dispositions from Stage 02/03.
2. `02_goal_tree.json`: hindsight goal tree for a day, with long/mid/short nodes, stable IDs, nested boundaries, grounding, and explicit/proactive provenance.
3. `03_causal_replay.json` plus per-event `causal/*.json`: chronological replay, optional immediate thought, free-form `memory_after`, references, and checkpoint projections.
4. Per-user finalization under `finalize/users/`: cross-day mission-link reduce, cached by input hash.
5. `05_publish.json` and dataset-level JSONL artifacts.

Important behavior:

- Goal-tree validation rejects cycles, invalid nesting, missing three-level active paths, unresolved boundaries, and motor-only short goals.
- A repair prompt is used if the first goal tree contains short goals such as “click/focus/scroll”.
- Causal replay sees only the current/past screenshots, rolling memory, active goal path, prior actions, and the already-recorded upcoming action packet.
- One free-form rolling-memory snapshot is saved for every semantic event.
- Stable IDs and raw-event/master-frame provenance are published.
- Per-user cross-day linking cannot cross users.
- No `hindsight_fold` reference exists in the new method or projection.

Published files include:

```text
manifest.json
days.jsonl
goal_nodes.jsonl
decision_thoughts.jsonl
checkpoints.jsonl
memory_snapshots.jsonl
mission_links.jsonl
semantic_events.jsonl
event_dispositions.jsonl
stage04_index.jsonl
goals.jsonl
goals_active.jsonl
memory.jsonl                 # compatibility projection
```

### Real action projection

`data_pipeline/realigned_pipeline/lib/semantic_actions.py` derives semantic action packets from the full Stage 02 event stream after applying the Stage 03 label/dead-zone policy.

- Uses `computer_use_rel_norm_v1`.
- Normalizes relative pointer deltas to the resolution-independent `[-1000, 1000]` screen-fraction scale.
- Coalesces uninterrupted mouse-motion bursts while preserving the endpoint.
- Coalesces printable typing and splits it at Enter, shortcuts, focus/mouse/scroll boundaries, and genuine gaps.
- Preserves drag/button/modifier/scroll ordering.
- Every emitted action maps to raw Stage 02 event IDs.
- Every non-emitted event receives an explicit filtered/dead-zone disposition.
- It does not generate 10 Hz/fixed-tick actions and never synthesizes an input action.

### Rolling memory, thoughts, and checkpoint contract

`prompts.yaml` currently instructs Kimi to:

- update the rolling memory at every event;
- produce a natural thought only when useful;
- naturally mention useful observations without headings/checklists;
- keep future outcomes out of thoughts and memory;
- optionally project the rolling memory into the exact checkpoint schema.

The shared versioned system prompt and strict parser support either:

```text
<think>optional immediate thought</think>
<tool_call>...</tool_call>
[more ordered tool calls]
```

or exactly one checkpoint:

```text
<checkpoint>
## Long-term goal
...

## Mid-term objective
...

## Short-term objective
...

## Completed
...

## Current state
...

## Next step
...

## Critical details
...
</checkpoint>
```

Training and evaluation load the same prompt bytes from:

`data_pipeline/realigned_pipeline/system_prompts/sequential_goal_memory_v1.txt`

### Evaluation/runtime work

- Added strict `parse_sequential_reply` handling for action replies versus exact checkpoint replies.
- Added normalized-delta and wait-duration validation.
- Registered `sequential_goal_memory_v1` in freeroll/evaluation.
- Added a screenshot-capacity checkpoint controller. Runtime requests checkpoint control at `ceil(capacity * fraction)`, default fraction `0.7`.
- After a valid checkpoint, runtime stores it in `checkpoints.jsonl`, retains the current screenshot, clears older image/action history, and resumes with the same system prompt plus goal/checkpoint conditioning.
- Visible completion still requires a parsed `terminate`; uncertainty alone does not terminate.

### HTML review

Generator:

`data_pipeline/realigned_pipeline/annotation/review_sequential_goal_memory.py`

It embeds screenshots as data URLs. `--image-width 0` preserves original JPEG bytes without decode/re-encode. It shows the goal path, thought, ordered real actions, memory before/after, checkpoint, event provenance, and per-event review controls. Ratings are stored in browser local storage and can be exported as JSON.

Generated review:

`_smoke/sequential_goal_memory/html_review/review.html`

It is approximately 1.1 MB and contains nine embedded full-resolution pilot screenshots. `_smoke/` is ignored through `.git/info/exclude` and is not part of the patch.

## Smoke tests and artifacts

### Latest rolling-memory pilot

Artifact:

`_smoke/sequential_goal_memory/rolling_pilot`

Configuration/provenance:

- labeler: `Kimi-K2.6`
- one user/day, first 60 seconds
- nine semantic events
- five goal nodes
- four non-empty decision thoughts
- nine rolling-memory snapshots
- three checkpoints
- 1,672 raw event dispositions
- action format: `computer_use_rel_norm_v1`
- pilot parameter `checkpoint_every=4`

The pilot shows that memory is genuinely chained: each event's `memory_before` equals the prior event's `memory_after`. It also shows useful screen-grounded orientation, such as noticing that a horizontally clipped instruction panel needs horizontal scrolling. However, several memories are too verbose, and some thoughts are attached to essentially cursor-positioning actions. This still needs human qualitative review and prompt tuning.

The pilot's inferred goals are all `proactive`, so the Stage 04 attempt requested a 70/30 schedule but fell back to proactive twice. The pipeline must never fabricate an explicit goal to satisfy the ratio; future pilot selection or dataset-level sampling must provide enough explicit examples.

### Earlier two-day/cross-day pilots

- `_smoke/sequential_goal_memory/annotation`
- `_smoke/sequential_goal_memory/link_pilot`

These exercised two day-level workers and the idempotent per-user finalizer. The selected slices did not contain a clearly continuing mission, so `mission_links.jsonl` was empty. Structural non-cross-user validation ran, but useful positive cross-day linking is not yet smoke-tested.

### Stages 04–06 compatibility smoke

The latest pilot was projected and passed unchanged through measurement and record conversion:

```text
_smoke/sequential_goal_memory/rolling_conversations/chat.jsonl
_smoke/sequential_goal_memory/rolling_lengths/
_smoke/sequential_goal_memory/rolling_records/train/part-00000.array_record
```

The obsolete current Stage 04 projection produced two records from nine events. The records measured 6,357 and 7,920 tokens and were converted successfully to ArrayRecord.

### Tests rerun at handoff

```text
pytest -q \
  data_pipeline/tests/test_sequential_goal_memory.py \
  eval/test_sequential_reply_contract.py \
  data_pipeline/tests/test_annotation_registry.py
# 24 passed

pytest -q \
  data_pipeline/tests/test_action_format.py \
  data_pipeline/tests/test_stage04_action_identity.py \
  data_pipeline/tests/test_stage04_goal_windows.py
# 120 passed, 3 skipped
```

The second group covers existing Stage 04/action behavior and is evidence that legacy modes have not been broken by the current patch.

## Critical unresolved design issue: remove fixed five/stride-four projection

The current code in `data_pipeline/realigned_pipeline/lib/sequential_conversations.py` still contains:

```python
WINDOW_DECISIONS = 5
WINDOW_STRIDE = 4
```

`stage_04_conversations.py`, its docstring/help text, summaries, and tests also describe five decisions with stride four. This is now explicitly rejected by the user and must be redesigned before considering the work complete.

Why it is wrong:

- It makes training resets far more frequent than inference resets.
- Checkpoints can appear twice because overlapping windows repeat an anchor.
- It teaches local behavior rather than the runtime's capacity-driven context-management distribution.
- The annotation pilot currently generates checkpoints every fixed number of semantic events (`checkpoint_every`, default 8; pilot 4), which also mismatches runtime's 70% capacity trigger.

Required replacement:

1. Build a chronological segment from the start of a task/day or the previous resume boundary.
2. Keep real screenshot/decision turns until roughly 70% of configured screenshot/context capacity.
3. At the boundary's current screenshot, emit the checkpoint-control user turn and exact checkpoint assistant target.
4. End that training record.
5. Start the continuation record with the identical system prompt, same `GOAL:`, the emitted checkpoint, and the same boundary/current screenshot.
6. Continue with the real next action and subsequent chronological turns until the next capacity boundary.
7. Preserve linkage metadata such as `episode_id`, `segment_index`, `memory_in_id`, `memory_out_id`, and checkpoint anchor.
8. Keep short next-decision records only as optional auxiliary action/motor balancing, not the long-horizon recipe.

Stage 03b already stores free-form memory at every event, which is sufficient for auditing and thought annotation. A remaining design choice is how to provide exact checkpoint projections at arbitrary capacity boundaries:

- generate a structured checkpoint snapshot at every semantic event during the existing causal labeler call, then select only capacity-boundary checkpoints in Stage 04; or
- parameterize Stage 03b with the deployment context capacity/fraction and generate checkpoints only at matching boundaries.

The first is more flexible and matches the user's desire to store every state snapshot, but produces more annotation text. Do not silently keep `checkpoint_every=8`.

Tests that explicitly encode the obsolete behavior must be rewritten, especially:

- `test_stage04_windows_round_trip_and_preserve_every_action`
- `test_stage04_provenance_schedule_is_exactly_seven_three_per_ten`

## Other unresolved issues

- Human review has not been completed. Do not approve a full run based only on structural tests.
- The HTML export schema is an event-level review. The full-run gate expects a separate JSON containing `reviewed_by` plus boolean gates: `goal_grounding`, `causal_thoughts`, `cross_day_links`, `checkpoints`, `action_provenance`, and `parser_validity`. These are not currently generated directly from the HTML export.
- Checkpoint and memory prose should be shortened. The latest Kimi pilot often repeats large portions of prior memory.
- Thoughts need stronger sparsity around pure cursor movement.
- The 70/30 explicit/proactive mixture must be achieved through eligible real examples at dataset sampling time. Current fallback behavior records the mismatch but does not guarantee the target distribution.
- A positive cross-day mission-continuation pilot is still required.
- Run a fresh end-to-end pilot and regenerate the HTML after the Stage 04/checkpoint-cadence redesign.
- Verify paired pre-checkpoint and resume records round-trip through the same strict evaluator parser and still pass unchanged through Stages 05 and 06.

## Files currently modified or added

Tracked modifications:

```text
data_pipeline/realigned_pipeline/annotation/lib/driver.py
data_pipeline/realigned_pipeline/annotation/lib/registry.py
data_pipeline/realigned_pipeline/annotation/stage_annotate.py
data_pipeline/realigned_pipeline/stage_04_conversations.py
eval/action_parser.py
eval/freeroll.py
eval/osworld_runtime.py
eval/osworld_system_prompts.py
```

New files:

```text
data_pipeline/realigned_pipeline/annotation/methods/sequential_goal_memory/__init__.py
data_pipeline/realigned_pipeline/annotation/methods/sequential_goal_memory/annotator.py
data_pipeline/realigned_pipeline/annotation/methods/sequential_goal_memory/prompts.yaml
data_pipeline/realigned_pipeline/annotation/review_sequential_goal_memory.py
data_pipeline/realigned_pipeline/lib/semantic_actions.py
data_pipeline/realigned_pipeline/lib/sequential_conversations.py
data_pipeline/realigned_pipeline/lib/sequential_goal_memory_contract.py
data_pipeline/realigned_pipeline/system_prompts/sequential_goal_memory_v1.txt
data_pipeline/tests/test_sequential_goal_memory.py
eval/test_sequential_reply_contract.py
handoff.md
```
