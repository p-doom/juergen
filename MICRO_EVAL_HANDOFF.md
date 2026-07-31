# CUA micro-eval handoff

## Goal

Build a fast, interpretable evaluation ladder for the strict
`computer_use_rel_step_v1` CUA model. Every task is one sampled model turn and
has an automatic verifier, so checkpoint progress is visible without eyeballing
long freerolls.

The intended headline metrics are:

- exact task success (`pass@1` over all attempts and empirical `pass@4`);
- strict action parse-valid rate;
- expected primitive/payload rate;
- for movement: distance-to-target-bbox reduction and best-legal-step
  optimality, including useful partial credit when the cursor does not reach the
  box;
- for click/type/scroll: semantic post-action state, not screenshot similarity.

## Isolation and branches

Do all further work in these worktrees. Do **not** edit the training worktrees.

- Juergen code:
  `/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-cua-micro-evals`
  - branch: `yll/cua-micro-evals`
  - based on `e2635605286442bb546f56b07f136840be8a81f3`
- Slurm/labctl wiring:
  `/fast/project/HFMI_SynergyUnit/yll/.worktrees/slurm-cua-micro-evals`
  - branch: `yll/cua-micro-evals`
  - based on `abb5e4c24d54f4572ee76ac8d792166dd9487f16`
  - currently no changes

The live/original trees were deliberately left untouched:

- `/fast/project/HFMI_SynergyUnit/yll/juergen` remains on `thinking-training`.
- `/fast/project/HFMI_SynergyUnit/yll/slurm` contains unrelated user changes;
  do not copy or overwrite it.

## Current implementation

Unfinished WIP lives under `eval/`:

- `cua_micro_eval.py`
  - loads a versioned JSON task suite;
  - boots a fresh snapshot VM per task attempt;
  - starts SGLang once for the batch;
  - sends exactly one fresh-context user turn with the real
    `cua_rel_step_v1_thinking` system prompt;
  - passes a deterministic seed to SGLang and runs four attempts by default;
  - uses the strict transactional rel-step parser;
  - denormalizes fixed 0..1000 relative steps to VM pixels;
  - saves before/after frames, prompt, response, result, and an overlay with
    bbox/start/end/vector;
  - incrementally rewrites the aggregate `result.json` after each attempt;
  - reports per-task/category/overall pass@1, empirical pass@4, mean/best
    progress, parse-valid rate, and expected-action rate.
- `cua_micro_tasks_v1.json`
  - 24 tasks across `move`, `click`, and `app_atomic`;
  - desktop/dock targets: Chrome and Files;
  - window controls: minimize/maximize/close;
  - Chrome controls: tab, new-tab, back, reload, address bar;
  - deterministic web button and scroll tasks;
  - editor, terminal, calculator, files, and settings tasks;
  - four typing cases, including punctuation/escaping and a longer coalesced
    string.
- `cua_micro_fixture.py`
  - a standard-library Tk fixture copied into the VM;
  - renders deterministic editor/terminal/calculator/files/settings windows;
  - atomically writes semantic state and exact widget bboxes to guest JSON.
- `test_cua_micro_eval.py`
  - suite/schema coverage, geometry, fixed-step optimality, action strictness,
    exact typing payload checks, pass@4 aggregation, and SGLang seed plumbing.
- `osworld_runtime.py`
  - `_call_model(..., seed=...)` forwards an optional seed.
- `osworld_vm_client.py`
  - `run_command()` exposes structured guest command output for setup/verifiers;
  - existing `execute()` now delegates to it.

Important typing design: a typing task only passes if both layers agree:

1. the strict parsed response is exactly one `type` primitive with the exact
   requested text;
2. the focused fixture contains that exact text afterward.

This separates model/action-format failure from dispatch/keyboard-layout
failure.

## Test state at handoff

The relevant unit suite passed before pausing:

```bash
cd /fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-cua-micro-evals/eval
uv run --project . python -m unittest \
  test_cua_micro_eval.py test_freeroll_helpers.py test_sampling.py
```

Result: **132 tests passed**.

The same test command from the repository root currently fails to import
`cua_micro_eval` because these are flat eval scripts, not an installed package.
Run it from `eval/` as above or explicitly set `PYTHONPATH=eval`.

`ruff check` passes for the newly touched files after auto-fixing imports, but
`ruff format --check` still reports four files requiring formatting:

- `eval/cua_micro_eval.py`
- `eval/cua_micro_fixture.py`
- `eval/osworld_runtime.py`
- `eval/test_cua_micro_eval.py`

No actual VM/model smoke has been run yet. Therefore the harness is **not ready
to trust or launch as a policy** despite the unit tests.

## Known risks and unfinished work

Treat all fixed desktop/Chrome bboxes as provisional. They were estimated from
known 1920x1080 OSWorld screenshots and normalized to 0..1000. The fixture
widgets expose live bboxes, but native window-control geometry is inferred from
the content window and must be checked on the real GNOME theme.

Specific items to validate/fix:

1. Run formatting and rerun the 132 tests.
2. Boot one VM without a model and exercise every setup/verifier directly:
   desktop reset, each Chrome variant, each Tk fixture, active-title checks,
   widget bbox extraction, and exact state writes.
3. Confirm the guest image contains `tkinter`, `xdotool`, and `wmctrl`. Add a
   setup fallback if any is absent.
4. Verify Chrome launch flags and file URLs on the image. `tabs` should start on
   BETA; clicking ALPHA should change the active title. `reload`, history/back,
   button, and scroll title markers must transition exactly once.
5. Inspect real screenshots and correct all fixed normalized bboxes, especially
   the Files dock icon and Chrome toolbar/tab geometry.
6. Verify window-control bboxes on GNOME. If theme geometry is unstable, replace
   these three tasks with a custom decorated test window whose controls expose
   exact bboxes and state.
7. Check relative-start geometry for every move task. The start must be outside
   the bbox and one legal 8/32/128 step should be optimal/reachable as intended.
8. Run a one-task, one-attempt off-the-shelf model smoke. Confirm strict parsing,
   dispatch, overlay, conversation, per-attempt result, and aggregate result.
9. Run `attempts=4` for a small move/click/type subset and confirm seeds produce
   independent sampled completions and empirical pass@4 is correct.
10. Add README documentation and a compact result-table/CSV artifact suitable
    for comparing the base model and checkpoints.
11. Implement labctl recipe and checkpoint policy in the isolated slurm
    worktree. Use the source revision containing this branch, the exact rel-step
    prompt/action format, temperature 1.0/top_p 0.95/top_k 20, and `attempts=4`.
12. Validate the recipe with labctl before enabling policy dispatch.

Also review the VM process lifecycle carefully. Each attempt currently boots a
fresh VM on the same forwarded ports; shutdown waits for QEMU, but a failed
guest background process or slow port release should be exercised in the smoke.

## Suggested next commands

```bash
cd /fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-cua-micro-evals/eval

uvx ruff format \
  cua_micro_eval.py cua_micro_fixture.py osworld_runtime.py \
  osworld_vm_client.py test_cua_micro_eval.py

uvx ruff check \
  cua_micro_eval.py cua_micro_fixture.py osworld_runtime.py \
  osworld_vm_client.py test_cua_micro_eval.py

uv run --project . python -m unittest \
  test_cua_micro_eval.py test_freeroll_helpers.py test_sampling.py
```

Then build a setup-only VM smoke before spending a GPU/model allocation. The
runner currently starts SGLang unconditionally, so the cleanest route is to add
`--validate_setups_only` (or a separate small script) that boots each selected
task, resolves its bbox/cursor, applies a known correct synthetic primitive,
and asserts the verifier.

For a later model smoke, select a tiny representative set:

```text
move.desktop.chrome.scale128
click.chrome.deterministic_button
type.editor.punctuation
scroll.chrome.down
```

Use `--attempts 1` first, then `--attempts 4` after the result contract is
verified.

## Definition of done

The sidequest is complete only when:

- every task setup and verifier passes under synthetic known-correct actions;
- fixed targets have visually audited bboxes at the actual VM resolution;
- one real checkpoint run produces valid artifacts and overlays;
- pass@1/pass@4 and distance metrics are independently checked from raw attempt
  results;
- a labctl recipe and lineage-scoped policy are validated and committed;
- original training worktrees remain untouched.
