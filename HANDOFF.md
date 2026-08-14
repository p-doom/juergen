# Handoff — dataset jitter fix, cua_micro_eval, ChatML loss-mask fix

Status as of 2026-08-14. Covers three independent threads. Deliberately
**excludes** the stage_04b/04c/04d key-combo/annotation/goal-stripping work —
that's still local-only WIP in this same worktree and isn't part of what got
pushed alongside this doc (see "Explicitly out of scope" below).

## 1. Dataset: `--jitter-deadband-px` for the ordered action formatters

Already committed and pushed (it ships with `feat/conversational-memory`,
which `feat/cua-micro-evals` is based on and rejoined at the same commit).

- **What**: sums maximal single-axis move/scroll-only runs and drops the
  result when it's pure hand-tension jitter (never empties a window, never
  drops a run's last primitive).
- **Where**: `data_pipeline/realigned_pipeline/lib/action_format.py`, wired
  into `OrderedFormatter`/`OrderedTypingFormatter` via `get_formatter`; plumbed
  through `stage_04_build_conversations.py`'s CLI and recorded in stage-04's
  output manifest as `jitter_deadband_px`. Tests in
  `data_pipeline/tests/test_action_format.py`.
- **Why**: needed by the `dataset_v6_fix_jitter` recipes.
- **Status**: done, no open follow-up.

## 2. `cua_micro_eval` — new verifiable per-turn eval suite (`eval/`)

Ported from `yll/cua-micro-evals` and extended. Design: every task is **one
sampled model turn with an automatic verifier** — not a long freeroll — so
checkpoint progress is visible without eyeballing rollouts. Headline metrics:
exact task success (pass@1, empirical pass@4), strict action parse-valid rate,
expected-primitive rate, and for movement tasks, distance-to-target-bbox
reduction with partial credit.

**Files**: `cua_micro_eval.py` (runner), `cua_micro_tasks.json` (20-task
suite: native_launch / native_app / chrome_control / multi_turn),
`cua_micro_fixture.py` (stdlib Tk fixture for editor/terminal/calculator/
files/settings, writes exact widget bboxes + semantic state to guest JSON),
`cua_micro_action_parser.py` (strict parsers for `computer_use_rel_step_v1`
and `qwen3vl_native_cua_v1` — kept separate from `action_parser.py` because
both files independently evolved a class named `OrderedPrimitive`/
`OrderedAction` into incompatible shapes), `test_cua_micro_eval.py`,
`sampling.py`/`test_sampling.py` (Qwen sampling-tuple module), `eval/patches/`
(native-runner sampling patch). Supporting spec/prompt: `data_pipeline/
realigned_pipeline/action_specs/computer_use_rel_step_v1.json`,
`data_pipeline/realigned_pipeline/system_prompts/cua_rel_step_v1_thinking.txt`.

Also see `MICRO_EVAL_HANDOFF.md` (yll's original handoff, kept for context —
its "test state at handoff" section describes an earlier point than yll's
actual HEAD) and `MICRO_EVAL_PORT_NOTES.md` (this branch's own porting notes:
why it wasn't a plain merge, what was adapted vs. written new).

**Extended past yll's original scope**:
- Added support for this branch's own `ordered_events_v3` action format
  alongside yll's two (`native_ordered_to_relstep` /
  `denormalize_native_ordered_action` adapters in `cua_micro_eval.py`).
- Added `--vms_per_sglang` (default 4, validated 1..10): runs that many VMs
  concurrently against one shared sglang instance instead of one VM at a
  time, since a single VM's step time is dominated by sglang prefill and the
  server otherwise only ever sees batch size 1.

**A real scoring bug found and fixed during validation**: the first real GPU
run against `ckpt_35k` completed with zero errors but `expected_action_rate=0.0`
and `pass_at_1=0.0` across all 20 tasks despite `parse_valid_rate=0.8` — the
model was parsing/dispatching fine, but `action_matches_expected` only
recognized rel-step/qwen3vl-native's atomic `click`/`key_combo` primitives,
never `ordered_events_v3`'s `down`/`up` pairs. Fixed with
`_canonicalize_native_ordered_action`, gated to only that format (rel-step's
"no leading move before a click" check is intentional and stays a violation
there — they have a dedicated atomic click primitive).

**Validated**:
- `python -m unittest test_cua_micro_eval.py test_sampling.py` from `eval/`:
  59/59 pass.
- `cua_micro_eval.py --help` resolves; `--action_format` lists all three
  formats; `--vms_per_sglang` is wired.
- Real GPU/VM runs against `ckpt_35k` (`cua_ordered_typing_v1`, `--attempts
  1`, full 20-task suite) via the
  `osworld_freeroll_manual_continuous_action_ckpt_35k_ylli.toml` labctl
  recipe:
  - Single-VM (job 139171): COMPLETED, 12m36s, zero errors.
    `pass_at_1=0.2, expected_action_rate=0.25, parse_valid_rate=0.8`.
  - `--vms_per_sglang=4` (job 139172): COMPLETED, 6m28s (~2x faster — not the
    full 4x, sglang's ~160s load doesn't parallelize), zero errors,
    `result.json` integrity confirmed (20/20 unique subdirs/task_ids, no
    duplicate or lost writes from the concurrent state_lock path).
    `pass_at_1=0.15, expected_action_rate=0.2` — same ballpark, ordinary
    sampling variance (temperature=0.7).
  - Both runs' `trajectory.jsonl` + `steps/step_NNN.png` + `runs[]` manifest
    confirmed to match labctl's rollout-viewer contract — browsable via
    `labctl show`.

**Not yet validated — flagged as open risk**:
1. All fixed/eyeballed bboxes (dock icons, Chrome toolbar/tabs, window
   controls) are still provisional; not cross-checked against ground truth
   on this VM image's actual resolution/theme.
2. Only `--attempts 1` has been run for real; `--attempts 4` (the script's
   own default, for real pass@4/all_4_success numbers) hasn't.
3. Only `cua_ordered_typing_v1` has a real run; `computer_use_rel_step_v1`/
   `qwen3vl_native_cua_v1` are still unit-tested only.
4. Disk space was tight while building this (`/fast/home` hit 100% full at
   one point) — this worktree still has no `.venv` of its own; tests were run
   against `feat/conversational-memory`'s existing `.venv` with `PYTHONPATH=.`
   to avoid writing more data.

## 3. omegalax: `fix/chatml-loss-mask-leakage`

Branch `fix/chatml-loss-mask-leakage` (worktree:
`/home/alfred.nguyen/projects/worktrees/omegalax/fix-chatml-loss-mask-leakage`).

- **Bug**: the assistant loss mask was built by scanning the final token
  stream for `<|im_start|>`/`<|im_end|>` pairs in sequence order. Samples
  whose user/context text embeds literal ChatML markers (e.g. screen notes
  describing the chat format) inject spurious special tokens that break the
  pairing, silently flipping later user turns — including image pad tokens —
  to supervised. Symptom: `train/supervised_tokens` spikes with anomalously
  low loss in run `lq3fgwvd` at steps 8980/12380/14390/23470.
- **Fix**: encode each ChatML turn independently in `encode_qwen_messages`
  and mask assistant content structurally by block position
  (`block_ids[3:-1]`) instead of scanning. ChatML specials are hard BPE split
  points, so per-block encoding reproduces the full-sequence token ids
  exactly — `input_ids` are byte-identical, no dataset rebuild needed. Both
  collators now consume `encoded["loss_mask"]`; the old scanning helper is
  removed. Regression tests cover literal-marker injection (text + VLM
  image-pad cases) and the per-block-equals-full-encode property.
- **Status**: **pushed to `origin`, working tree clean — but NOT merged into
  `origin/main` and no PR is currently open.** This is ready for review; the
  main action item is to open the PR and get it merged.

## Repo / branch map

| Repo | Branch | Push status |
| --- | --- | --- |
| `p-doom/juergen` | `feat/cua-micro-evals` | Pushed as part of this handoff (see below) |
| `p-doom/omegalax` | `fix/chatml-loss-mask-leakage` | Already pushed; not merged, no PR open |

## Explicitly out of scope of this push/handoff

`stage_04b_filter_key_combo.py` (chord-triggered window cutting),
`stage_04c_annotate_conversations.py` (hindsight goal annotation of an
already-cut artifact), `stage_04d_strip_goal_text.py` (goal-text ablation
stripping), their tests, and the full `data_pipeline/realigned_pipeline/
README.md` rewrite (which documents 04b/04c alongside everything else) remain
**local, uncommitted** in this worktree
(`/fast/home/alfred.nguyen/projects/worktrees/juergen/cua-micro-evals`) — not
pushed anywhere yet. Also left uncommitted: two stale `cua_micro_tasks.json`
backup files (`.BAK`/`.BAKK`, safe to delete) and a `logs/` directory of local
eval run output.

## Immediate next steps

1. Open a PR for `omegalax` `fix/chatml-loss-mask-leakage` → `main` — the fix
   is tested and already pushed, it just needs review/merge.
2. Before trusting `cua_micro_eval` as a policy gate: audit the fixed bboxes
   against the real VM resolution/theme, run `--attempts 4`, and get a real
   run on the other two action formats.
3. Decide what to do with the local-only stage_04b/c/d work in the
   `cua-micro-evals` worktree — it's uncommitted and not described here on
   purpose; commit/push separately once ready.
