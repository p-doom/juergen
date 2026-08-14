# Porting yll/cua-micro-evals onto feat/conversational-memory

This worktree (`feat/cua-micro-evals`) was branched from `feat/conversational-memory`
and now carries ONLY the CUA micro-eval suite ported from `yll/cua-micro-evals`
(no freeroll/grounding/fullbench/smoke -- those are this branch's own evals
and were deliberately removed from this worktree; they're untouched on
`feat/conversational-memory` itself). The suite has been extended to also
support this branch's own `ordered_events_v3` (`cua_ordered_typing_v1`)
action format, alongside yll's original two.

## Why this wasn't a merge

The two branches share a common ancestor 38 (yll side) / 21 (this side)
commits back and independently touched the SAME files in incompatible ways
-- both sides evolved `action_parser.py`'s `OrderedPrimitive`/`OrderedAction`
classes under the same names into different shapes (this branch:
`input_name`, `render()`, `as_key_event()`; yll's: `name`/`keys`/`count`/
`status`), and gave `osworld_runtime._call_model` an incompatible signature
and return type. A `git merge` would have produced large, semantically-loaded
conflicts in exactly the code this branch's real evals depend on. So this was
a manual, dependency-traced port instead: only `cua_micro_eval.py`'s actual
import graph was brought over, and this branch's existing shared files were
touched only additively (new functions/dict entries, nothing removed or
changed in place) -- see "What was adapted" below.

## What's in this worktree

- `eval/cua_micro_eval.py`, `eval/cua_micro_tasks.json`, `eval/cua_micro_fixture.py`,
  `eval/test_cua_micro_eval.py` -- the ported runner + 20-task suite + Tk
  fixture + tests.
- `eval/cua_micro_action_parser.py` -- yll's `computer_use_rel_step_v1` /
  `qwen3vl_native_cua_v1` parsers, with the colliding classes renamed
  `RelStepPrimitive`/`RelStepAction` (aliased back to `OrderedAction`/
  `OrderedPrimitive` on import in `cua_micro_eval.py`).
- `eval/sampling.py`, `eval/test_sampling.py` -- Qwen sampling-tuple module.
- `eval/patches/` -- native-runner sampling patch + README.
- `data_pipeline/realigned_pipeline/action_specs/computer_use_rel_step_v1.json`,
  `data_pipeline/realigned_pipeline/system_prompts/cua_rel_step_v1_thinking.txt`
  -- rel-step's binding spec + system prompt.
- This branch's own `action_parser.py`, `osworld_vm_client.py`,
  `osworld_system_prompts.py`, `osworld_runtime.py` -- kept (see below),
  additive edits only.
- `MICRO_EVAL_HANDOFF.md` -- yll's original handoff doc, kept as-is for
  context. Its worktree paths and "test state at handoff" describe an
  EARLIER point than `yll/cua-micro-evals`'s actual HEAD (later commits
  there did real VM/model debugging -- Chrome CDP tab activation,
  terminal/save-dialog matching, a native-Qwen3VL baseline -- but none of
  those runs' artifacts are committed anywhere to point to).

## Removed (this branch's OWN evals -- not yll's, out of scope here)

`freeroll.py`, `osworld_grounding_runner.py`, `osworld_fullbench_runner.py`,
`osworld_one_task_runner.py`, `test_freeroll_helpers.py`,
`test_ordered_action.py`, `test_play_env_action_format.py`, `play_env.py`,
`play_env_ui.html`, `play_env_web.py`, `eval/smoke/` (the other closed-form
canary suite -- also this branch's own, not yll's), `bc_offline_score.py`,
`bc_roundtrip.py`, `ifeval.py`, `roundtrip_ifeval.py`, `result.py`,
`hf_complete.py`, `sglang_runner.py`, `inspect_runner.py`,
`inspect_ai_patches.py`, `osworld_score.py`. None of these are needed by
`cua_micro_eval.py`; they're all fully intact on `feat/conversational-memory`
itself, this worktree just doesn't carry copies of them anymore.

## What was written new (not a straight copy)

- `eval/cua_micro_action_parser.py` (see above).
- `native_ordered_to_relstep()` + `denormalize_native_ordered_action()` in
  `cua_micro_eval.py` -- the third action-format's adapter, mirroring the
  existing `qwen3vl_native_to_ordered()` pattern:
  - `native_ordered_to_relstep` renames this branch's real
    `action_parser.OrderedAction`/`OrderedPrimitive` (`input_name`) into the
    canonical `RelStepAction`/`RelStepPrimitive` shape (`name`) the rest of
    the file already scores/serializes/dispatches uniformly. It's a field
    rename, not a reinterpretation -- `move`/`scroll`/`down`/`up`/`type` mean
    the same thing on both sides.
  - `denormalize_native_ordered_action` scales `move` deltas from
    `--model_resolution` pixels to VM-native pixels (screen/model_resolution
    ratio), mirroring `denormalize_action`'s job for rel-step's fixed
    0..1000 grid. No-ops when `--model_resolution` is unset. This is needed
    because the ported `dispatch_ordered_action()` (unlike this branch's own
    `dispatch_ordered()`) does NOT scale internally -- yll's two formats
    never needed it (they express absolute positions on a resolution-
    independent 0..1000 grid), but this branch's `ordered_events_v3` deltas
    are literal model-resolution pixels, so scaling has to happen before
    dispatch instead.

## What was adapted (existing files, additive edits only)

- `eval/osworld_vm_client.py`: added `run_command()` and `dispatch_ordered_action()`
  (+ `_type_write_command`, `_cua_v4_key_to_pyautogui`) verbatim from yll's
  branch. `dispatch_ordered_action` is the ACTUAL dispatch path
  `cua_micro_eval.py` uses for all three formats (it understands
  `move`/`scroll`/`down`/`up`/`type` -- this branch's own vocabulary --
  as well as yll's `click`/`button_down`/`key_combo`/etc.). This branch's
  own `dispatch_ordered()`/`dispatch_action()`/`dispatch_computer_use()`,
  `_event_to_pyautogui`, `_rdev_to_pyautogui`, `_computer_use_key_to_pyautogui`
  were reused as-is (verified byte-identical to yll's copies before relying
  on them) -- NOT duplicated.
- `eval/osworld_system_prompts.py`: added `cua_rel_step_v1_thinking` and
  `qwen3vl_native_cua_v1` entries. `cua_ordered_typing_v1` already existed
  here natively -- nothing to add for the new format.
- `eval/cua_micro_eval.py` / `eval/test_cua_micro_eval.py`: `action_parser`
  imports redirected to `cua_micro_action_parser` for the rel-step/qwen3vl
  formats; given a LOCAL `_call_model` (copied from yll's version, distinct
  from this branch's shared one -- see prior note) plus the new
  `_NATIVE_ORDERED_FORMAT` branch wired into both `run_attempt` and
  `run_multiturn_attempt`'s parse+dispatch sites and into `--action_format`'s
  choices.

## Concurrency: --vms_per_sglang (added after the initial port)

Ported the concurrency model from `feat/multi-vm-sglang`'s (uncommitted)
`freeroll.py` rewrite: `--vms_per_sglang` (default 4, validated 1..10) runs
that many VMs concurrently against one shared sglang instance instead of one
VM at a time -- a single VM's step time is dominated by sglang prefill and
the server only ever sees one in-flight request (batch size 1), badly
underusing the GPU.

Implementation differs from freeroll's in one way: freeroll boots VMs
*while* sglang is still loading (a `threading.Event` gates the first real
request per slot on sglang readiness). This suite already waited for sglang
synchronously before touching any VM, so that overlap optimization was
skipped as unnecessary complexity -- the actual goal (N concurrent requests
in flight) is unaffected.

`_RunContext` (a dataclass bundling everything invariant across attempts)
+ `_run_one_task_attempt` + `_run_vm_slot` replace the old sequential
`for task in tasks: for attempt in range(attempts):` loop:
(task, attempt) pairs are round-robined across N slots (`work[i::n_vms]`),
each slot boots/tears down its own VM sequentially on a dedicated port
offset (`5000 + job_mod + slot_id`), and slots run concurrently via
`ThreadPoolExecutor`. `_RunContext.state_lock` guards the shared
`attempts`/`runs` lists and the live `result.json` rewrite -- multiple slots
finish attempts at unpredictable times, so appending to those lists and
recomputing the aggregate is a real data race without it, not just a
cosmetic log-ordering issue.

## Validated

- `python -m unittest test_cua_micro_eval.py test_sampling.py` from `eval/`:
  **59/59 pass** (48 ported + 11 new: the `cua_ordered_typing_v1`
  parse/convert/scale path, its `_PROMPT_FORMATS` wiring, and the
  `_canonicalize_native_ordered_action` click/key-chord matching -- see the
  bug note below).
- `cua_micro_eval.py --help` resolves cleanly; `--action_format` lists all
  three formats; `--vms_per_sglang` is wired.
- **Real GPU/VM runs against ckpt_35k** (`cua_ordered_typing_v1`,
  `--attempts 1`, full 20-task suite), via the
  `osworld_freeroll_manual_continuous_action_ckpt_35k_ylli.toml` labctl
  recipe:
  - Single-VM (job 139171): COMPLETED, 12m36s, zero errors. `overall`:
    `pass_at_1=0.2, expected_action_rate=0.25, parse_valid_rate=0.8`.
  - `--vms_per_sglang=4` (job 139172): COMPLETED, 6m28s (~2x faster --
    not the full 4x, since sglang's ~160s load doesn't parallelize and
    multi_turn tasks unevenly load the 4 slots), zero errors, `result.json`
    integrity confirmed (20/20 unique subdirs and task_ids, no duplicate or
    lost writes from the concurrent state_lock path). `overall`:
    `pass_at_1=0.15, expected_action_rate=0.2` -- same ballpark as the
    single-VM run; the difference is ordinary run-to-run sampling variance
    (temperature=0.7, not greedy), not a regression.
  - Both runs' `trajectory.jsonl` + `steps/step_NNN.png` + top-level
    `runs[]` manifest confirmed on disk to match labctl's rollout-viewer
    contract exactly (`labctl::server::resolve_rollout_paths`) -- these
    runs should be browsable via `labctl show` the same way freeroll's are.
- **A real scoring bug found and fixed via this process**:
  `action_matches_expected` originally required exactly one primitive and
  only recognized rel-step/qwen3vl-native's atomic `click`/`key_combo`
  primitives -- never `cua_ordered_typing_v1`'s `down`/`up` pairs. The first
  real run (before this fix) completed with zero errors but
  `expected_action_rate=0.0` and `pass_at_1=0.0` across all 20 tasks despite
  `parse_valid_rate=0.8` -- i.e. the model was parsing and dispatching fine,
  but nothing could ever recognize a native-format click as correct.
  Fixed with `_canonicalize_native_ordered_action`, gated to only that
  format (rel-step/qwen3vl-native's "no leading move before a click" check
  is intentional there -- they have a dedicated atomic click primitive, so
  needing two tool calls really is a contract violation; caught this as a
  regression on `test_qwen3vl_native_multiple_calls_are_parse_valid_but_not_atomic`
  while first generalizing the fix, now covered by a dedicated test).

## Remaining unvalidated items

1. All fixed/eyeballed bboxes (dock icons, Chrome toolbar/tabs, window
   controls) are still "provisional" per yll's original handoff and should
   be visually audited against this VM image's actual resolution/theme --
   the real runs above show `parse_valid_rate=0.8`/`expected_action_rate`
   in the 0.2-0.25 range, plausible numbers but not yet cross-checked
   against ground truth bbox correctness specifically.
2. Only `--attempts 1` has been run for real; `--attempts 4` (the script's
   own default, for real pass@4/all_4_success numbers) hasn't yet.
3. Only `cua_ordered_typing_v1` has a real run; `computer_use_rel_step_v1`/
   `qwen3vl_native_cua_v1` are still unit-tested only.

## Disk space note

`/fast/home` was reported 100% full / 0 bytes available (`df`) while
building this port -- `uv run` failed trying to create a fresh `.venv` for
this worktree; a partial 79MB `.venv` from that failed attempt was removed.
Tests above were run against `feat/conversational-memory`'s existing `.venv`
directly (`PYTHONPATH=. <that venv>/bin/python -m unittest ...`) to avoid
writing more data. This worktree still has no `.venv` of its own -- creating
one will likely hit the same wall until space is freed cluster-wide (one
concrete candidate: `feat/conversational-memory`'s own `play_env_out/`,
572MB of local run output).
