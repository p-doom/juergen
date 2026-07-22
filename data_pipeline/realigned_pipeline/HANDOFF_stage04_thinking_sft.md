# Handoff: stage-04 thinking-SFT mode (lumine_thinking → training samples)

**Task for the next session**: extend `stage_04_build_conversations.py` with a
WINDOW/THINKING mode that turns the `lumine_thinking` stage-03b artifact into
SFT conversations — `<think>` before the anchored action, earlier thoughts as
context — following the lumine-validated sample shape, in the pipeline's own
conventions. (An AD-HOC sibling already exists and produced interim data:
`stage_04_thinking_conversations.py`; whether think-insertion becomes a mode
inside the real stage 04 or stays a sibling is the open design decision.)

**Governing rule (same as the annotation port, it worked)**: the validated
*sample shape* comes from lumine (`hindsight_fold/lumine/assemble.py`,
read-only reference); the *implementation* is stage 04's own idiom — its
selector, its projection, its chat.jsonl schema, its manifest/join-guard
conventions. Import nothing from lumine. Stages 05/06 must keep running
untouched on the output.

## What already exists (do not rebuild)

- **The annotation artifact** (corpus run):
  `p-doom_shared/labctl/datasets/yll.kryeziu/realigned_ccast0618d_v3_goals_lumine_thinking_fps0.5/`
  - `goals.jsonl` — one row per VERIFIED thought. Single-tick half-open master
    interval: `end_master_idx == start_master_idx + 1`. Thought text in
    `instruction`; extra fields: `kind` (plan/reorient/decide/react/monitor/
    wait), `verify` (verdict/violations/reason), `day_tag`, `t_day_s`,
    `anchor` ("day_tag +HH:MM:SS"), `model`, `prompt_pack_sha`, `unit_id`
    (= day_tag), `annotation_fps` (0.5). ~100k rows expected, 73% pass rate.
  - `units/<day_tag>/clip_*.json` — per-clip ledger (all thoughts incl. fails,
    memory_in/out, log, chunk_index, day_idx_range). Audit trail + densify
    substrate.
  - `memory/<day_tag>.jsonl` — memory+log trajectory per clip. NOT for
    samples (inference-consistency rule below); raw material for future
    compaction/annotation passes.
  - `manifest.json` — `master_store_id` + `filter_id`; stage 04 must
    `assert_same_artifact` against `--filter-dir` exactly as goal mode does.
- **Partial snapshot** (built while the corpus ran, for the interim datasets):
  `..._fps0.5_snap390d/` — 390 days, 50,178 verified thoughts, plus copied
  per-clip ledgers; `manifest.json` marked `snapshot_partial: true`.
- **Densified layer** (text-only second pass over the snapshot ledgers):
  method `lumine_thinking_densify` — additional thoughts written from
  log+memory+actions with NO frames sent; intent-not-narration rules; rows
  stamped `method: lumine_thinking_densify` and `verify.mode` `"text"`
  (audited) or `"none"` (verify=off run). Keep provenance separate from
  Track-A rows in any analysis; they merge for training only.
- **Filter artifact** (join target):
  `p-doom_shared/labctl/datasets/yll.kryeziu/realigned_ccast0618d_v3_filter`.
- **Day machinery**: `annotation/lib/days.py` (day index via mvhd + stage-00/02
  clips manifest, DayStream with chunks at >180s gaps, canonical actions);
  `--day-index-cache` avoids the ~3 min probe.
- **Validation context**: Franz-day h1 vs the 3-track reference — density
  0.090 thoughts/frame (ref 0.088), pass 67–73% (ref 67–74%), self-test
  green, batched≈per-thought verify (77% agreement, no future-leak).
  Reference lumine samples:
  `hindsight_fold/lumine/runs/u54196854_20260616/clip_annotator/smoke/sft/`
  and `.../assemble.py` (~125 lines — read first, it IS the spec).

## The validated sample shape (lumine assemble.py, adapt don't port)

1. **Windows, not goals.** Fixed frame-windows tiled over each CHUNK of the
   day stream (lumine: 64 frames, no overlap; never straddle chunks). One
   sample per window; `thinking` stream = only windows with ≥1 verified
   thought (`bulk` = every window, no thoughts — existing goal-free mode is
   the per-segment analog).
2. **Think-then-act.** Anchor frame's assistant turn:
   `<think>\n{thought}\n</think>\n{action}`; actions verbatim. Training fps
   == annotation fps (0.5, LOCKED v1): the anchor tick IS a selected frame —
   placement is 1:1, no snapping.
3. **Context block.** First user turn: `"Your thoughts so far this session:"`
   + last **K=8** earlier VERIFIED same-chunk thoughts, one
   `[+HH:MM:SS] {text}` line each. Never across a recording gap.
4. **System prompt / terminal token stay CLI policy.** Ad-hoc decision made
   with the user: goal-free prompt + think-sentence (see
   `THINKING_SYSTEM_PROMPT` in the ad-hoc script), NO terminal token
   (windows are arbitrary cuts, not completions — terminate would train
   spurious stopping).

## Design decisions already made with the user (don't relitigate)

- **Inference-consistency rule:** context must be reproducible at inference.
  The model's own earlier thoughts qualify; writer memory/logs do NOT
  (annotation scaffolding). Memory-as-context is an ablation flag committing
  to a runtime summarizer — never the default. Logs never enter samples.
- **Evidence boundary:** every verified thought is locally evidenced within
  ~12 pre-anchor frames (that's what the gate checked). A window must not
  start closer than ~12 frames before one of its anchors — ad-hoc script
  drops the <think> (keeps the action) and counts `n_demoted`; a real
  implementation may snap boundaries instead.
- **Window size from the token budget:** ~1.2k tok per 720p frame. 16k seq
  cap → 12 frames/window; 64k → 48. (Lumine's 64 was their GPU budget, not
  a validated optimum.) Stage 05 measures; sizes are flags.
- **Guard:** stage 04's GOAL mode must never consume a lumine_thinking
  artifact (single-tick thought rows, not task intervals) — branch on the
  manifest `method` field.
- **Day streams in stage 04:** windows tile day-chunks, which cross segment
  boundaries — the thinking mode needs the same day inputs 03b uses
  (`--clips-manifest`, tz, gap-cut, day-index cache). Deterministic rebuild
  gives identical day indices.

## Open points for the real (non-ad-hoc) implementation

1. Sibling stage vs mode inside stage 04.
2. Bulk stream from the same runs (day-window-shaped vs per-segment).
3. Cross-fps thought projection (unlock training fps ≠ 0.5) — use
   `lib/goals.project_goals` window-ownership semantics.
4. Context block content ablations (K, kinds, timestamps; memory-block flag).
5. Densified rows: merge policy (goal_id de-collision needed — both passes
   number rows `<day_tag>_tNNNN`), and whether verify=off rows need a text
   audit pass before non-ad-hoc training.

## Validation plan (mirror the annotation port's playbook)

1. Dry-run the builder on a Franz day slice; compare shape/counts vs the
   lumine reference `sft/thinking/` samples.
2. Alignment hand-check (~20 samples): <think> on the anchor's assistant
   turn; thought precedes ITS action; context block strictly earlier +
   same-chunk + verified.
3. Stage 05 measure for token-length distributions vs the 16k/64k caps;
   stage 06 must run unchanged.

## Operational state (as of writing — check live state, this goes stale)

- Corpus annotate: labctl job 130052, ~450/679 days, 73% pass, ~3.3M TPM
  dual-Kimi (recipe: `yll/slurm/.../recipes/realigned_pipeline/stage_03b_annotate.toml`,
  `[inputs.filter]` temporarily external; orphan `created` runs in the
  registry need cleanup). Mop-up: re-run the recipe once after job end
  (LABELER_MAX_TOKENS=65536 now in recipe env), then finalize writes
  goals.jsonl + manifest.
- Seven labeler/extractor failure classes were found and fixed during the
  corpus run (all merged to main via PR #11): generic self-test plants,
  invalid JSON escapes, malformed JSON, trailing prose, content_filter,
  reasoning spirals, control chars. The labeler is heavily hardened —
  reuse, don't rediscover.
- labctl snapshots the repo per run (`run_dir/source/juergen`); running jobs
  are immune to working-tree changes.
- Cost anchors (measured): Track A 2,113 in + 737 out tok/frame (~€6.4k
  corpus, dual-Kimi 60/40); densify ~27.6k tok/clip with text verify
  (~€2.1k corpus) or ~half with verify=off — Kimi burns ~11.5k reasoning
  tokens even on tiny text calls; that dominates.
