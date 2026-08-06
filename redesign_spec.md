# Sequential goal-memory redesign — implementation spec

Working doc for the capacity-driven packing redesign. Owner: yll. This file is the
single source of truth for cross-module interfaces; if an implementation detail
conflicts with this spec, the spec wins unless it is impossible, in which case the
deviation must be recorded at the bottom under "Deviations".

Repository: this repo, branch `thinking-training`. Do NOT commit. Do NOT touch the
Franz tree. Run tests with `uv run pytest -q <paths>`.

## Design summary

Stage 03b annotates deployment-independent STATE (goal tree, chained per-event
rolling memory, sparse causal thoughts). Training serialization is produced by
simulating the runtime context manager ("the packer") over that state:

- Segments are cut where the runtime's `ScreenshotCheckpointController` would fire,
  with a per-segment jittered trigger fraction (seeded, deterministic).
- At each boundary the record ends with the CHECKPOINT CONTROL user turn and a
  seven-field checkpoint target; the next record opens with the same GOAL, the
  byte-identical checkpoint text, and the same boundary screenshot.
- Checkpoints are lazy text-only LLM projections of the annotated rolling memory at
  exactly the boundary anchors the packer chooses (new annotation pass 03c).
- Explicit/proactive is a rendering choice at Stage 04 (hindsight goal relabeling),
  not a data property. The old SHA-scheduled 70/30 and `desired_provenance` die.
- Thought sparsity is structural: a deterministic decision-boundary pre-gate plus a
  predict-then-reveal agreement gate. No style begging.
- Quality is functional: resumability probe, leak detector, metric-drafted review
  gates (new QA script).

Decisions already made (do not re-litigate):
- No `terminate` synthesis. Human recordings contain none; flag deferred.
- Agreement-gate predictor = the configured annotation labeler (swappable later).
- `n_packings` default 1.
- Stage 04 `--capacity` is REQUIRED for the sequential recipe. No silent default.
- Fixed `checkpoint_every` and `WINDOW_DECISIONS/WINDOW_STRIDE` are removed, not
  deprecated.
- Legacy annotation methods and legacy Stage 04 recipes must remain byte-identical
  in behavior.

## Phase 1 — shared foundation (blocks everything else)

### New file: `data_pipeline/realigned_pipeline/lib/sequential_packing.py`

Pure, deterministic, no I/O, no LLM. Mirrors `eval/osworld_runtime.py`
`ScreenshotCheckpointController` semantics exactly.

```python
@dataclass(frozen=True)
class PackingConfig:
    capacity: int                 # runtime screenshot capacity
    fraction_low: float = 0.5     # per-segment trigger fraction jitter range
    fraction_high: float = 0.85
    seed: int = 0
    n_packings: int = 1

def packing_config_hash(cfg: PackingConfig) -> str  # sha256 of canonical json, hex

def boundary_events(n_events: int, *, day_tag: str, cfg: PackingConfig,
                    packing_index: int = 0) -> list[int]
```

`boundary_events` semantics:
- A segment starts with screenshot count 1 at its first event; each subsequent
  event increments the count (one screenshot per semantic event — identical to the
  controller counting screenshots, `note_screenshot` per new frame).
- Per-segment threshold: `max(2, ceil(cfg.capacity * fraction_k))` where
  `fraction_k` is drawn uniformly from `[fraction_low, fraction_high]` using
  `random.Random(sha256((seed, day_tag, packing_index)) -> int)` with one draw per
  segment in order. Threshold floor 2 guarantees progress. Validate
  `capacity >= 3`, `0 < fraction_low <= fraction_high <= 1`.
- The event at which the count reaches the threshold is the boundary anchor. The
  next segment starts AT that same event with count reset to 1 (the boundary
  screenshot carries over, exactly like `reset_to_current`).
- A boundary that would land on the day's final event is dropped (no continuation
  to train). Returns sorted unique indices in `[1, n_events-2]`.

```python
def segments_from_boundaries(n_events: int, boundaries: list[int]) -> list[tuple[int, int]]
```
Returns inclusive ACTION spans per segment: segment 0 = `(0, b0 - 1)`, segment k =
`(b_{k-1}, b_k - 1)`, final = `(b_last, n_events - 1)`. The boundary event's action
belongs to the NEXT segment; its screenshot appears in both records (control turn
of the earlier record, opening turn of the later) — this mirrors the runtime, where
the control request interrupts before the action for the current screenshot.

```python
def eligible_modes(span: tuple[int, int], goal_nodes: list[dict]) -> list[str]
```
Subset of `["explicit_mid", "explicit_long", "proactive"]`. `explicit_mid` is
eligible iff one single mid-level node covers every event in the span; same for
`explicit_long` with a long node. `proactive` is always eligible.

```python
def sample_mode(eligible: list[str], weights: dict[str, float], *, seed: int,
                day_tag: str, packing_index: int, segment_index: int) -> str
```
Deterministic seeded categorical draw over eligible modes with renormalized
weights. Default weights (module constant `DEFAULT_MODE_WEIGHTS`):
`{"explicit_mid": 0.45, "explicit_long": 0.25, "proactive": 0.30}`.

```python
def actions_agree(predicted: list[dict], actual: list[dict]) -> bool
```
Deterministic comparison of ordered `computer_use` tool-call lists
(`{"name": "computer_use", "arguments": {"action": ..., ...}}`):
- Different lengths or any pairwise action-name mismatch → disagree.
- `key`: normalized (casefolded, stripped) key lists equal.
- `key_down`/`key_up`: same key, casefolded.
- `type`: whitespace-normalized exact text equality.
- `mouse_move_rel`: agree iff per-axis sign matches (|delta| < 40 counts as zero
  and matches either sign) AND euclidean magnitude ratio within `[0.4, 2.5]`
  (both-near-zero, i.e. both magnitudes < 40, agrees).
- `scroll`/`hscroll`: same sign of `pixels`.
- `left_click`/`right_click`/`middle_click`/`double_click`/`triple_click`/
  `button_down`/`button_up`: name (and `button` where present) match.
- `wait`: name match only.
- `terminate`: same `status`.
Constants (`MOVE_ZERO_DELTA = 40`, `MOVE_RATIO_LOW/HIGH`) as module-level names.

### Additions to `lib/sequential_goal_memory_contract.py`

```python
PROACTIVE_GOAL_TEXT = (
    "Continue the user's work on this computer. Infer what they are doing from "
    "the screen and prior context, and advance it."
)
THOUGHT_MAX_WORDS = 60
CHECKPOINT_MAX_WORDS = 180   # total across the seven field bodies
RESUME_UPWEIGHT_TURNS = 3    # first assistant action turns after a checkpoint_in
```
Everything already in the contract stays unchanged (system prompt path, checkpoint
fields, `CHECKPOINT_CONTROL_REQUEST`, `goal_conditioning`, `render_checkpoint`).

### New tests: `data_pipeline/tests/test_sequential_packing.py`

Cover: determinism per (seed, day_tag, packing_index); different seeds move
boundaries; threshold floor/edges (tiny days: 0, 1, 2 events); boundary-on-final
dropped; spans partition `[0, n)` exactly once; eligibility on synthetic trees;
`actions_agree` per action type including tolerance edges.

## Phase 2A — Stage 04 packer (owner: agent A)

Files: rewrite `lib/sequential_conversations.py`; update `stage_04_conversations.py`
(sequential recipe path ONLY — legacy recipes byte-identical); rewrite obsolete
tests (`test_stage04_windows_round_trip_and_preserve_every_action`,
`test_stage04_provenance_schedule_is_exactly_seven_three_per_ten`, wherever they
live) and add packer invariant tests.

Kill entirely: `WINDOW_DECISIONS`, `WINDOW_STRIDE`, `_window_starts`, the SHA-256
`desired_provenance` schedule, `_goal_for_event`'s deepest-node preference. Short
goals never appear as `GOAL:`.

`build_sequential_conversations(days, *, system_prompt, parse_reply, cfg:
PackingConfig, mode_weights=DEFAULT_MODE_WEIGHTS, mission_links=None)`.

Per day, per `packing_index in range(cfg.n_packings)`:
- `boundaries = boundary_events(...)`; `spans = segments_from_boundaries(...)`.
- Every boundary anchor (and, for cross-day, the day-final anchor) MUST have a
  checkpoint row in the artifact whose `packing_config_hash` matches
  `packing_config_hash(cfg)`. If missing → raise with a message naming the anchor,
  the expected hash, and the remedy (rerun annotation pass 03c with this config).
- Mode per segment via `eligible_modes` + `sample_mode`. GOAL text: covering node's
  `text` for explicit modes; `PROACTIVE_GOAL_TEXT` for proactive. `goal_id` null
  for proactive.
- Record for segment k (action span `(s, e)`):
  - system turn (unchanged prompt bytes).
  - user turn: `goal_conditioning(goal_text, checkpoint_in)` + image of event `s`.
    `checkpoint_in` = boundary `s` checkpoint text for k > 0, else None (or the
    linked prior-day final checkpoint for the cross-day variant).
  - alternating assistant action turns (`<think>` + tool calls when the event has
    a non-empty annotated thought) and user image turns for events `s..e`,
    exactly the existing `_messages` inner shape.
  - if `e + 1` is a boundary: user turn `CHECKPOINT_CONTROL_REQUEST` + image of
    event `e + 1`; assistant turn = checkpoint text at that anchor. Final segment
    of a day: no control turn, record simply ends after action `e`.
  - `parse_reply` every assistant turn (`expected="action"` / `"checkpoint"`).
- Record metadata: `episode_id` = `{day_tag}_p{packing_index}`, `segment_index`,
  `conversation_id` = `{episode_id}_s{segment_index:03d}`, `mode`,
  `goal_id`, `instruction`, `goal_provenance` (the node's annotated provenance;
  null for proactive), `checkpoint_in_id`/`checkpoint_out_id` (nullable),
  `semantic_event_ids` (action span), `n_images`,
  `resume_upweight_turns`: indices (into `messages`) of the first
  `RESUME_UPWEIGHT_TURNS` assistant action turns when `checkpoint_in` is present,
  else `[]`, plus existing provenance fields (day_tag, user_id, date, recipe,
  action_format).
- Cross-day variants: for each mission link (same user, date A < date B), locate
  day B / packing 0 / segment 0. Emit ONE extra record with suffix `_xday`:
  `checkpoint_in` = day A's day-final checkpoint text, mode = `explicit_long` with
  the linked day-B long goal if it covers the span else `proactive`,
  `cross_day: true`, `mission_link_id`. Excluded from the exactly-once invariant.

Invariants (enforced in code, covered by tests):
1. Per packing, base records' `semantic_event_ids` partition the day's events
   exactly once (the existing missing/unresolved checks adapt to per-packing).
2. Checkpoint handoff byte-identical: `checkpoint_out` text of segment k ==
   embedded `checkpoint_in` of segment k+1.
3. `n_images` per record ≤ `cfg.capacity`.
4. Every assistant turn round-trips through `parse_reply`.
5. Same cfg → identical output (no wall-clock, no unseeded RNG).
6. Explicit-mode GOAL node covers the record's whole action span.

Summary dict: replace window fields with `packing_config` (asdict + hash),
`mode_counts`, `n_episodes`, `n_segments`, `n_cross_day_records`,
`mean_segment_events`, plus retained counters. Update stage_04 CLI: `--capacity`
(required for this recipe), `--fraction-low`, `--fraction-high`,
`--packing-seed`, `--n-packings`, `--mode-weights` (JSON), and docstrings/help
that mention five-decision windows.

## Phase 2B — annotation method changes + 03c (owner: agent B)

Files: `annotation/methods/sequential_goal_memory/annotator.py`, `prompts.yaml`,
`annotation/stage_annotate.py` (param plumbing only), and
`data_pipeline/tests/test_sequential_goal_memory.py`.

### Causal replay changes
- Delete `checkpoint_every` / `checkpoint_due` and the inline checkpoint branch
  from the causal pass and from `causal_event` in prompts.yaml. Causal replies are
  `{"thought", "memory_after", "references"}` only. Bump
  `PROMPT_VERSIONS["causal_replay"]` to `causal_replay_v4`.
- New pure pre-gate (new file `annotation/methods/sequential_goal_memory/gate.py`):
  `is_decision_boundary(events, index, goal_nodes, *, gap_s=5.0) -> bool` — True
  iff any of: `index == 0`; any goal node (any level) has
  `start_event_index == index`; `segment_id` differs from the previous event;
  inter-event time gap > `gap_s`; the PREVIOUS event's calls include `key` with
  Enter or a modifier shortcut, or `wait`; consecutive `scroll` events reverse
  sign. Unit-test each rule.
- Motor events (gate False): memory-only prompt variant (`causal_event_motor`,
  no thought field offered; reply `{"memory_after", "references"}`, thought = "").
- Decision events, param `thought_gating` (default `"agreement"`, alt
  `"boundary"`):
  - `agreement`: FIRST a predict call (`predict_action` prompt: same causal
    context — screenshots, memory_before, goal path, prior actions — but NOT the
    true upcoming action; reply is the ordered tool-call JSON list it would
    execute next). Compare with the true packet via `actions_agree`.
    Agree → memory-only variant, thought stays empty. Disagree → reveal variant
    (`causal_event_reveal`): true action shown, requires a non-empty decisive
    thought ≤ `THOUGHT_MAX_WORDS` grounded in visible state (one retry on
    violation, then hard error). The thought must not merely paraphrase the
    action.
  - `boundary`: no predict call; optional thought at decision events (existing
    causal_event behavior).
- Per-event causal record gains: `is_decision_boundary`, `thought_gating`,
  `predicted_calls` (nullable), `agreed` (nullable bool). Memory chaining,
  reference validation, caching, and `memory_before`/`memory_after` semantics are
  UNCHANGED. `decisions` rows gain `gate: "divergence" | "offered"`.

### New pass 03c — checkpoint projection (`04_checkpoints.json`)
- Params via `ctx.params`, plumbed from new stage_annotate flags:
  `--checkpoint-capacity` (int, REQUIRED for this method),
  `--checkpoint-fraction-low/high`, `--packing-seed`, `--n-packings`. Build a
  `PackingConfig`; anchors = union over packings of
  `boundary_events(...)` PLUS the day-final event (flagged `is_day_final`).
- Per anchor, one TEXT-ONLY labeler call (`checkpoint_projection` prompt):
  inputs `memory_after` at the anchor, the active goal-path texts, and the
  previous anchor's checkpoint values when one exists (for folding). Instructions:
  exact seven fields; total across field bodies ≤ `CHECKPOINT_MAX_WORDS`;
  `Completed` must FOLD the previous checkpoint's `Completed` into at most two
  sentences plus the new interval — never restate full history; `Next step` states
  intent only, never outcomes; nothing beyond what the memory knows. Validate
  seven non-empty fields + word budget, one corrective retry, then error. Render
  with `render_checkpoint`. Cache per anchor keyed on (memory snapshot input hash,
  prompt version, `packing_config_hash`). `PROMPT_VERSIONS["checkpoint_projection"]
  = "checkpoint_projection_v1"`.
- `checkpoints.jsonl` rows keep the existing shape plus `packing_config_hash`,
  `is_day_final`, `source_memory_snapshot_id`. Publish/finalize/manifest carry
  the packing config, gating params, and divergence stats
  (`n_decision_boundaries`, `n_predicted`, `n_divergent`).
- Checkpoint anchor semantics: project from `memory_after(anchor)` — written with
  the upcoming action recorded as intended, never completed; identical convention
  to the previous inline checkpoints.

### Goal-tree prompt tweak
Add to `goal_tree`: goal `text` must be a short imperative task statement (eval
instruction style), not a description of the user. Bump to `goal_tree_v4`.

### Tests
Update `test_sequential_goal_memory.py` for all the above with a fake labeler:
gate rules; agreement path (agree → empty thought, no reveal call; disagree →
reveal call, non-empty bounded thought); motor path never offered a thought;
projection budget + folding validation + retry; anchor determinism vs
`sequential_packing`; publish/validation schema updates (validators that assumed
inline checkpoints must accept the 03c shape).

## Phase 2C — functional QA (owner: agent C)

New files: `annotation/qa_sequential_goal_memory.py` + tests
(`data_pipeline/tests/test_sequential_qa.py`).

CLI: `--artifact <dir>` (published dataset dir), optional `--stage04-chat
<chat.jsonl>`, `--sample N` (default 20), `--no-llm` (skip probe/judge), labeler
config mirroring stage_annotate. Reuse `annotation/lib/labeler.py` client. Writes
`qa_report.json` and `review_draft.json` into the artifact dir.

Deterministic checks (always run):
- memory chain integrity; references never future (re-run, do not trust).
- leak detector: for each memory/thought/checkpoint text, collect candidate
  strings from FUTURE events' action packets (typed `type` text tokens ≥ 4 chars,
  key-combo names) that never occur in past events' packets or in active goal
  texts; case-insensitive substring hits are leaks. Report rate + worst examples.
- thought metrics: density overall, density on motor events (must be 0), word-
  length distribution vs `THOUGHT_MAX_WORDS`, divergence rate, thought/action
  n-gram overlap (parrot score).
- checkpoint metrics: word counts vs budget; `Completed` growth across chained
  anchors (folding regression check: non-monotonic-growth heuristic).
- if `--stage04-chat`: every assistant turn parses via `parse_sequential_reply`;
  handoff byte-identity; capacity respected (recompute from record metadata).

LLM checks (skipped under `--no-llm`):
- resumability probe on `--sample` boundary checkpoints: fresh context = system
  prompt + `goal_conditioning(goal, checkpoint)` + boundary screenshot ONLY →
  "describe the situation and propose the next step". A judge call scores
  compatibility against the true next 3 semantic action packets → bool + reason.
  Report pass rate.

`review_draft.json`: the six gate keys. Auto-draft: `checkpoints` = probe pass
rate ≥ 0.8 (null under `--no-llm`); `action_provenance` and `parser_validity`
from deterministic checks; `goal_grounding`, `causal_thoughts`, `cross_day_links`
= null (human), each with a `basis` string of supporting metrics. `reviewed_by`:
null — a human must fill it; the file must NOT satisfy the full-run gate as
emitted (verify against the gate-checking code in stage_annotate/driver and keep
it that way).

## Phase 2D — runtime/eval consistency (owner: agent D)

Files: `eval/freeroll.py`, `eval/osworld_runtime.py`,
`eval/test_sequential_reply_contract.py`.

- Single-eviction guarantee: when the sequential recipe runs with a
  `ScreenshotCheckpointController`, StreamingLLM block eviction in `append_turn`
  must never fire first — fail fast at setup if `n_history_frames <
  controller.capacity` (or wire the sequential path so eviction is disabled
  outright). A training segment contains every frame since the last compaction;
  the runtime must too.
- Serialization identity test: build the runtime message list for a tiny
  synthetic flow (goal turn → actions → control → checkpoint → resume) via the
  actual runtime helpers, and the Stage-04 record shape for the same flow, and
  assert turn-by-turn structural identity (roles, text blocks incl.
  `goal_conditioning` output and `CHECKPOINT_CONTROL_REQUEST`, image positions,
  checkpoint bytes). Import from `data_pipeline` the way existing eval tests do.
- Jitter note: runtime keeps fixed fraction 0.7; training covers [0.5, 0.85].
  No runtime change needed beyond the guard; document in the controller docstring.

## Cross-cutting rules for all agents

- Python, no type-annotation-free zones — match surrounding style exactly
  (this codebase uses compact, heavily-validated code with trailing-comma'd
  call sites; mirror it).
- Never invent or modify computer actions. Never weaken existing validators.
- All randomness seeded and keyed by stable ids; no wall-clock, no `random`
  module default state.
- Bump every prompt version you touch; hash-cached passes must invalidate.
- Legacy behavior (other annotation methods, other stage-04 recipes) must remain
  byte-identical; the legacy test group
  (`test_action_format.py`, `test_stage04_action_identity.py`,
  `test_stage04_goal_windows.py` minus the two rewritten tests) must pass.
- Run your own tests plus the adjacent suites before reporting done. Report
  the exact pytest commands + results.

## Deviations

- Phase 2A, cross-day `conversation_id`: the spec fixes the suffix at `_xday`, but two
  mission links can target the same day B (`day1 -> day3` and `day2 -> day3`), and both
  are legitimate records that differ only in the incoming checkpoint. Emitting both with
  the literal suffix would duplicate a `conversation_id`, which
  `write_conversation_artifact` sorts on. Links are therefore ordered by
  `mission_link_id` and the k-th record into one day gets `_xday` for k == 0 and
  `_xday{k}` after that. Covered by
  `test_two_missions_resuming_into_one_day_keep_unique_ids`.
- Phase 2A, cross-day resolution: a mission link whose `from_goal_id` or `to_goal_id`
  belongs to a day that is not in the `days` passed to the packer (Stage 04 run with
  `--day-filter`/`--limit`, or a day with no events) is SKIPPED and counted in the
  summary as `n_cross_day_links_unresolved`, not an error. A link that resolves on both
  sides but crosses users or is non-causal is still fatal.
- Phase 2A, note (not a deviation) for 2B/2C: `packing_config_hash` is the hash of the
  whole `PackingConfig`, so it covers `n_packings`. Raising `--n-packings` therefore
  invalidates every 03c checkpoint even though the anchors of the existing packings do
  not move. Pass 03c must be rerun with the identical five values Stage 04 uses; the
  Stage-04 error message prints them as 03c flags.
- Phase 2B, `predict_action` reply shape: the spec says the predictor's reply "is the
  ordered tool-call JSON list it would execute next". `labeler.call_json_full` /
  `common.extract_json_object` can only return a JSON **object** (a bare top-level list
  is silently reduced to one element), so the prompt asks for
  `{"calls":[{"name":"computer_use","arguments":{...}}, ...]}` and the annotator unwraps
  `calls` before `actions_agree`. Semantics unchanged; a malformed entry is kept as an
  empty call rather than dropped, so a bad prediction can only disagree.
- Phase 2B, `Completed` folding: the spec states the folding constraint as a prompt
  instruction and lists "seven non-empty fields + word budget" as the validated
  properties, while the test list asks for "folding validation". `Completed` is therefore
  also validated at <= `COMPLETED_MAX_SENTENCES = 3` sentences (at most two folded plus
  the new interval) whenever a previous checkpoint exists — same one-corrective-retry-
  then-error path, and re-checked in `_validate_days`.
- Phase 2B, note (not a deviation): the causal pass no longer knows checkpoint ids, so
  `memory_snapshots[*].checkpoint_id` is filled in at publish time from the 03c anchors
  (one checkpoint per anchor index, independent of packing index). The review viewer and
  `_validate_days`' snapshot->checkpoint resolution keep working unchanged.
