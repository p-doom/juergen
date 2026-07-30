# Realigned CUA data pipeline — DAG reference

Reference map of the clean (realigned) CUA data pipeline: the real stage DAG,
the stage→script mapping, and — the point of this doc — every **cross-lineage
input edge** and the pinned artifact alias that makes it reproducible.

Recipes live in `slurm/dev/yll/berlin/labctl/recipes/realigned_pipeline/`.
Scripts live in `juergen/data_pipeline/realigned_pipeline/`.
Cross-lineage inputs are pinned by `slurm/dev/yll/berlin/labctl/pin_external_inputs.sh`.

## Linear DAG

```
00 clip_manifest ─▶ 01 master_frames ─▶ 02 realign ─▶ 03 filter ─┐
                                                                  │
        ┌── 03b annotate (side branch off the filter) ───────────┤
        │                                                         ▼
        └──────────────────────────────────────────▶ 04 conversations ─▶ 05 measure ─▶ 06 records
                                                        (--mode action | thinking)                (param max_length)
```

04 is the single **join point**: it fuses the filter mask (frames) + realigned
keylog actions, and — in thinking mode — the 03b goal annotations. 05 and 06
run unchanged regardless of what 04 emitted (chat.jsonl shape is stable).

## Stage → script → output-alias

| Stage | Recipe | Script (`realigned_pipeline/…`) | Canonical output alias |
|-------|--------|--------------------------------|------------------------|
| 00 clip_manifest | `stage_00_clip_manifest.toml` | `stage_00_clip_manifest.py` | `realigned_ccast0618d_stage_00_clip_manifest` |
| 01 master_frames | `stage_01_master_frames.toml` (+ `_merge`) | `stage_01_master_frames.py` (`--merge`) | `realigned_ccast0618d_stage_01_master_frames_fps15` |
| 02 realign | `stage_02_realign.toml` | `stage_02_realign.py` | `realigned_ccast0618d_stage_02_realign_manifest` |
| 03 filter | `stage_03_filter.toml` | `stage_03_filter.py` | `realigned_ccast0618d_v3_filter` |
| 03b annotate | see annotation sub-DAG | `annotation/stage_annotate.py --method …` | `realigned_ccast0618d_v3_stage_03b_*` |
| 04 conversations | `stage_04_conversations.toml` / `stage_04t_*.toml` | `stage_04_conversations.py --mode action\|thinking` | `realigned_ccast0618d[_v3]_stage_04[t]_*` |
| 05 measure | `stage_05_measure*.toml` | `stage_05_measure_lengths.py` | `realigned_ccast0618d_stage_05_measure_fps{fps}` |
| 06 records | `stage_06_records*.toml` | `stage_06_training_records.py` | `realigned_ccast0618d_stage_06_records_fps{fps}_len{max_length}` |

## 04 — one script, two modes (`--mode`, NOT a code fork)

`stage_04_conversations.py` is the merge of the two former scripts
(`stage_04_build_conversations.py` + `stage_04_thinking_conversations.py`).
Format-agnostic plumbing — chat content blocks, day-index build/cache/selection,
the four-file artifact writer, the window↔clip-stride guard — lives in
`lib/conversations.py` and is shared verbatim by both modes.

| Mode | Recipes | Action format | Goals? | Emits |
|------|---------|---------------|--------|-------|
| **action** | `stage_04_conversations.toml` | `canonical` (byte-identical to legacy on clean stretches) | optional (goal-free today) | plain CUA chat.jsonl |
| **thinking** | `stage_04t_thinking_goals_oev2_w{30,60}.toml` (+ `_heldout`) | `ordered_events_v2` — the only live generation; `computer_use_rel_v1` and the other formatters still work, they just have no live recipe | required (`--goals-dir` = a 03b `lumine_thinking_goals` artifact) | goal-conditioned `<think>` chat.jsonl, TERMINATE at goal boundaries |

Both modes require `--filter-dir`, `--clips-manifest` and `--day-index-cache`
(the mvhd day probe is minutes, so it is cached by `filter_id` + tz and reused);
the thinking recipes additionally consume the 03b goals corpus (and optionally a
`lumine_goal_boundaries` corpus via `--boundaries-dir` for the `verified`
terminate variant).

Action-mode semantics are pinned by `tests/test_stage04_action_identity.py`: it
runs the merged script's `--mode action` against the pre-merge
`stage_04_build_conversations.py` (recovered from git history) and requires
**byte-identical** `conversations.jsonl` / `chat.jsonl`.

**Only the oev2 generation is live** (2026-07-30). Everything else was moved to
`labctl/archive/` — a mirror of the live layout, so archived pipelines/policies
still resolve internally and restoring is a plain `mv`. Archived: **curel
w30/w60** (`computer_use_rel_v1`, written to this CLI but never built),
**w24 native_v1/v2**, and the goal-free **w12 / w48** lineage — recipes,
pipelines, HF exports, training jobs, eval recipes and policies alike.

The w24 recipes are not just superseded but **unrunnable** against the current
script: they invoke the deleted `stage_04_thinking_conversations.py`, pass the
now-rejected `window-frames = 24` (not a multiple of 15), and v2 passes the
removed `--memory-update-samples` / `--no-memory-at-goal-start` flags. Their
built artifacts remain in the registry; reproducing them requires the juergen
tree at the last pre-merge commit (`deda409`), not the current script.

## 06 — max_length is a PARAMETER, not a code variant

All `stage_06_records_*.toml` run the **same** `stage_06_training_records.py`.
They differ only in `[params].max_length`, which flows into both the arg
(`max_length`) and the output alias (`…_len{params.max_length}`):

| Recipe | max_length | overflow_mode |
|--------|-----------|---------------|
| `stage_06_records.toml` (base) | `{params.max_length}` | `truncate` |
| `…_thinking_goals_oev2_w30_len48000` | 48000 | (per recipe) |
| `…_thinking_goals_oev2_w60_len96000` | 96000 | (per recipe) |

(The archived generations — `curel_w30/w60` at 48000/96000, `*_native_v1/v2` at
32768 `drop`, `w12` at 16384, `w48` at 65536 — are under
`labctl/archive/recipes/realigned_pipeline/`; see the archive README.)

05 (measure) is likewise one script (`stage_05_measure_lengths.py`) tokenizing
chat.jsonl once into a split-agnostic length cache; its `_thinking_*` variants
differ only in which 04 output they point at.

## 03b — annotation sub-DAG

`annotation/stage_annotate.py` is method-dispatched (`--method`, methods under
`annotation/methods/`). All run off the **filter** + **clips manifest**.

```
filter ─▶ describe_extract ─┐
       ─▶ lumine_thinking ──┴─▶ lumine_thinking_densify (--param source_dir=<parent>)
       ─▶ lumine_thinking_goals ─▶ lumine_goal_boundaries (--param goals_dir=<goals corpus>)
       ─▶ plans
```

| Method | Recipe | Extra input (`--param`) | Produces |
|--------|--------|-------------------------|----------|
| `describe_extract` | `stage_03b_annotate.toml` (method-swappable) | — | describe→extract goals |
| `lumine_thinking` | `stage_03b_annotate.toml` (active config) | — | Track-A sequential thinking goals |
| `lumine_thinking_densify` | (uncommitted) | `source_dir=<parent artifact>` | text-only densification pass |
| **`lumine_thinking_goals`** | **`stage_03b_annotate_goals.toml`** (canonical producer) | `goals_fold_dir=<hindsight_fold days>` | goal-conditioned dense goals corpus |
| `lumine_goal_boundaries` | `stage_03b_annotate_goal_boundaries.toml` | `goals_dir=<lumine_thinking_goals artifact>` | verified goal-END boundaries |
| `plans` | `stage_03b_annotate.toml` (method-swappable) | — | plan\naction first-turn prose |

`lumine_thinking_goals` reads its short-horizon goals from
`<goals_fold_dir>/<day_tag>/goals/goals.json` — that fold tree is a
cross-lineage input (see below) and is the **root provenance** of every goal.

## Cross-lineage input edges (the pinned aliases)

Every edge below reaches OUTSIDE this pipeline's own stage lineage and is
therefore pinned with `labctl register-external --kind dataset` so builds are
reproducible. `dataset` is the only correct kind (the berlin
`artifact_roots` are `dataset|checkpoint|eval_result|environment` — there is no
`manifest`/`external` kind).

| Pinned alias | Source path | Source lineage / user | Consumed by |
|--------------|-------------|-----------------------|-------------|
| `ext_ccast0618d_stage01_master_fps15` | `…/datasets/alfred.nguyen/ccast0618d_dataset_full_v3_stage_01_master_frames_fps_15` | **alfred.nguyen** stage-01 (15 fps JPEG master) | 03 filter |
| `ext_ccast0618d_stage02_manifest` | `…/datasets/alfred.nguyen/ccast0618d_dataset_full_v3_stage_02_realign_manifest` | **alfred.nguyen** stage-02 (realign manifest + corrected keylogs) | 03 filter, all 03b annotators, 04 (day probe) |
| `ext_ccast0618d_v3_filter` | `…/datasets/yll.kryeziu/realigned_ccast0618d_v3_filter` | this pipeline (03 filter), self-produced | 03b annotators, 04t |
| `ext_ccast0618d_v3_goals` | `…/datasets/yll.kryeziu/realigned_ccast0618d_v3_goals_lumine_thinking_goals_fps0.5` | this pipeline (03b, job 130763), self-produced | 04t thinking, 03b boundaries |
| `ext_ccast0618d_v3_boundaries` | `…/datasets/yll.kryeziu/realigned_ccast0618d_v3_boundaries_lumine_goal_boundaries_fps0.5` | this pipeline (03b side-branch), self-produced | 04t thinking (`verified` terminate variant) |
| `ext_hindsight_fold_days` | `…/yll/hindsight_fold/pipeline_runs/days` | **hindsight_fold** lineage (Kimi day-goal fold) | 03b `lumine_thinking_goals` (`goals_fold_dir`) |

**`ext_hindsight_fold_days` caveat:** its path is outside every
`artifact_root`, so it cannot be registered in place — the pin script stages a
goals-only projection under the datasets root first. See
`pin_external_inputs.sh` and the provenance audit.

**`ext_hindsight_fold_days` provenance:** produced by hindsight_fold
`pipeline.run_annotate_dataset` (SLURM 125749/125801/125807, 2026-07-13,
hai001), goals-extract finalized at hindsight_fold commit **160b9e7**
(2026-07-14, branch main). Of the 624 day-units in `ext_ccast0618d_v3_goals`,
620 trace to a present fold `goals.json`; 4 ran unconditioned (no fold goal).

## Invariants (do not violate)

- **Master grid / stride.** The frame master runs at **15 fps**. Sampling uses
  `resolve_stride(master_fps=15, fps, mode)` (`lib/views.py`). In `exact` mode
  `15 / fps` **must be an integer** tick stride (fails loudly otherwise): 0.5
  fps ⇒ stride 30, 1 fps ⇒ 15, 3/5/15 fps ⇒ 5/3/1. `nearest` mode allows any
  fps ≤ 15 with ≤half-tick jitter. A stage's fps must match the fps of the
  artifact it consumes (e.g. 04t `fps` MUST equal the goals corpus fps, 0.5).
- **Window size is a multiple of the annotation clip stride (15).** The 03b
  sidecars (`goals_active.jsonl`, `memory/<day>.jsonl`) tile each chunk in
  `day_idx_range` steps of `CLIP_STRIDE = 15` sampled frames. Thinking windows
  are counted in the same selected frames, so `--window-frames` must be a
  positive multiple of 15 — `require_window_alignment()` (`lib/conversations.py`)
  fails the run otherwise. Only then does every window edge land on a clip edge,
  which is what makes the memory predecessor below well defined. Built sizes are
  w30 / w60 @ 0.5 fps (60 s / 120 s); the pre-merge w24/w48 sizes are NOT
  aligned and are rejected by the current script.
- **Leak-free "So far".** Memory conditioning in thinking mode is strictly
  **past-only and exact**: a window's `So far:` is the `memory_out` of the clip
  whose `day_idx_range` END is exactly `window_first_frame_idx - 1`. Never a clip
  that overlaps or extends into the window (future leak), and never a
  farther-back clip when the exact predecessor is missing from the sidecar —
  that case is omitted and counted as `n_windows_memory_omitted_boundary`. A
  span start or chunk start carries a bare `GOAL: …` with no memory
  (unconditional since the merge — the old `--no-memory-at-goal-start` knob is
  gone), matching the freeroll episode-start shape. Annotation itself is
  future-blind (the verify gate sees no downstream frames).
- **MEMORY UPDATE turns are no longer emitted.** The merged stage 04 dropped the
  memory-update sample appendix (`--memory-update-samples` and its prompt). The
  `cua_v3_thinking` / `cua_v4_thinking` / `cua_oev2_thinking` prompts still carry
  a `# Memory` section describing the `MEMORY UPDATE:` turn, so that instruction
  is currently **untrained**. The prompt bytes are frozen because they are baked
  into built artifacts (`…_stage_04t_thinking_goals_oev2_w30` and its 05/06
  descendants) and `eval/osworld_system_prompts.py` loads the same files to keep
  eval byte-identical to training — removing the section means a new prompt id
  plus a rebuild, not an in-place edit.
- **Join fingerprints.** 04 refuses mismatched joins: `filter_id` and
  `master_store_id` fingerprints recorded in the filter and goals manifests
  must agree, so a thinking build can only fuse a goals corpus built from the
  same filter.
