# Ordered Action Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ordered Stage-06 action projection with a configurable 10 Hz continuous-action grid while retaining the aggregate v1 format as an explicit ablation.

**Architecture:** A focused `annotation_pipeline.action_format` module will own schema constants and the pure event-to-mini-program projection. Stage 05 will retain observation interval bounds, and Stage 06 will select v1 or v2, collect projection/state diagnostics, and record exact provenance. `build_sft.py` will pass the two projection controls through; evaluator code is explicitly out of scope.

**Tech Stack:** Python 3.11, dataclasses, argparse, unittest, Ruff.

---

## File structure

- Create `data_pipeline/annotation_pipeline/action_format.py`: pure ordered-action projection and state diagnostics.
- Create `data_pipeline/tests/test_action_format.py`: focused projection tests.
- Modify `data_pipeline/annotation_pipeline/stage_05_assemble_trajectories.py`: retain interval bounds in format-neutral steps.
- Modify `data_pipeline/annotation_pipeline/stage_06_project_sft.py`: schema selection, rendering, statistics, and manifest provenance.
- Modify `data_pipeline/annotation_pipeline/build_sft.py`: CLI passthrough for action schema and motor rate.
- Modify `data_pipeline/tests/test_structured_pipeline.py`: end-to-end Stage-05/06 coverage for v2 default and v1 ablation.
- Modify `data_pipeline/annotation_pipeline/README.md`: document the action language and commands.
- Do not modify any file under `eval/`.

### Task 1: Pure ordered-action projection

**Files:**
- Create: `data_pipeline/annotation_pipeline/action_format.py`
- Create: `data_pipeline/tests/test_action_format.py`

- [ ] **Step 1: Write failing projection tests**

Create `data_pipeline/tests/test_action_format.py` with helpers that construct normalized events and tests for ordering, motor ticks, 2D scroll, zero omission, names, and state diagnostics:

```python
import unittest

from annotation_pipeline.action_format import (
    HeldStateDiagnostics,
    project_ordered_action,
    update_held_state,
)


def _event(index: int, time_s: float, kind: str, **fields):
    return {
        "source_event_idx": index,
        "local_time_s": time_s,
        "kind": kind,
        **fields,
    }


class OrderedActionFormatTest(unittest.TestCase):
    def test_discrete_event_splits_movement_inside_one_motor_tick(self):
        result = project_ordered_action(
            [
                _event(0, 2.01, "move", dx=1.0, dy=0.0),
                _event(1, 2.02, "move", dx=3.0, dy=-1.0),
                _event(2, 2.03, "press", key="LMB"),
                _event(3, 2.04, "move", dx=2.0, dy=0.0),
                _event(4, 2.05, "release", key="LMB"),
            ],
            interval_start_s=2.0,
            continuous_action_hz=10.0,
        )
        self.assertEqual(
            result.text,
            "move(4,-1); down(LMB); move(2,0); up(LMB)",
        )

    def test_motor_tick_boundary_splits_continuous_actions(self):
        result = project_ordered_action(
            [
                _event(0, 4.01, "move", dx=1.0, dy=0.0),
                _event(1, 4.11, "move", dx=2.0, dy=0.0),
            ],
            interval_start_s=4.0,
            continuous_action_hz=10.0,
        )
        self.assertEqual(result.text, "move(1,0); move(2,0)")

    def test_scroll_is_ordered_and_two_dimensional(self):
        result = project_ordered_action(
            [
                _event(0, 0.01, "scroll", dx=2.0, dy=-3.0),
                _event(1, 0.02, "scroll", dx=1.0, dy=-2.0),
                _event(2, 0.03, "press", key="KeyA"),
                _event(3, 0.04, "scroll", dx=-1.0, dy=4.0),
            ],
            interval_start_s=0.0,
            continuous_action_hz=10.0,
        )
        self.assertEqual(
            result.text,
            "scroll(3,-5); down(KeyA); scroll(-1,4)",
        )

    def test_zero_continuous_actions_are_omitted(self):
        result = project_ordered_action(
            [
                _event(0, 0.01, "move", dx=0.4, dy=0.4),
                _event(1, 0.02, "scroll", dx=0.0, dy=0.0),
                _event(2, 0.03, "press", key="LMB"),
                _event(3, 0.04, "release", key="LMB"),
            ],
            interval_start_s=0.0,
            continuous_action_hz=10.0,
        )
        self.assertEqual(result.text, "down(LMB); up(LMB)")

    def test_empty_projection_is_no_op(self):
        result = project_ordered_action(
            [], interval_start_s=0.0, continuous_action_hz=10.0
        )
        self.assertEqual(result.text, "NO_OP")
        self.assertEqual(result.primitives, ())

    def test_invalid_rate_and_event_kind_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "continuous_action_hz"):
            project_ordered_action([], interval_start_s=0.0, continuous_action_hz=0.0)
        with self.assertRaisesRegex(ValueError, "Unsupported action event kind"):
            project_ordered_action(
                [_event(0, 0.0, "context")],
                interval_start_s=0.0,
                continuous_action_hz=10.0,
            )

    def test_state_diagnostics_do_not_mutate_primitives(self):
        result = project_ordered_action(
            [
                _event(0, 0.01, "release", key="LMB"),
                _event(1, 0.02, "press", key="KeyA"),
                _event(2, 0.03, "press", key="KeyA"),
            ],
            interval_start_s=0.0,
            continuous_action_hz=10.0,
        )
        diagnostics = HeldStateDiagnostics()
        held = set()
        update_held_state(result.primitives, held=held, diagnostics=diagnostics)
        diagnostics.finish_trajectory(held)
        self.assertEqual(result.text, "up(LMB); down(KeyA); down(KeyA)")
        self.assertEqual(diagnostics.dangling_up, 1)
        self.assertEqual(diagnostics.duplicate_down, 1)
        self.assertEqual(diagnostics.non_neutral_trajectory, 1)
        self.assertEqual(diagnostics.held_at_trajectory_end, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify the missing-module failure**

Run:

```bash
cd data_pipeline
PYTHONPATH=. python -m unittest tests.test_action_format -v
```

Expected: `ModuleNotFoundError: No module named 'annotation_pipeline.action_format'`.

- [ ] **Step 3: Implement the projection module**

Create `data_pipeline/annotation_pipeline/action_format.py` with:

```python
"""Ordered action projection for Stage-05 event records."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

AGGREGATE_ACTION_SCHEMA = "aggregate_delta_keys_v1"
ORDERED_ACTION_SCHEMA = "ordered_events_v2"
ACTION_SCHEMAS = (AGGREGATE_ACTION_SCHEMA, ORDERED_ACTION_SCHEMA)
DEFAULT_ACTION_SCHEMA = ORDERED_ACTION_SCHEMA
DEFAULT_CONTINUOUS_ACTION_HZ = 10.0

PrimitiveKind = Literal["move", "scroll", "down", "up"]
_INPUT_NAME_RE = re.compile(r"^[^\s(),;]+$")


@dataclass(frozen=True)
class ActionPrimitive:
    kind: PrimitiveKind
    dx: int | None = None
    dy: int | None = None
    input_name: str | None = None

    def render(self) -> str:
        if self.kind in {"move", "scroll"}:
            return f"{self.kind}({self.dx},{self.dy})"
        return f"{self.kind}({self.input_name})"


@dataclass(frozen=True)
class ProjectedAction:
    text: str
    primitives: tuple[ActionPrimitive, ...]


@dataclass
class HeldStateDiagnostics:
    duplicate_down: int = 0
    dangling_up: int = 0
    non_neutral_trajectory: int = 0
    held_at_trajectory_end: int = 0

    def finish_trajectory(self, held: set[str]) -> None:
        if held:
            self.non_neutral_trajectory += 1
            self.held_at_trajectory_end += len(held)

    def update(self, other: "HeldStateDiagnostics") -> None:
        self.duplicate_down += other.duplicate_down
        self.dangling_up += other.dangling_up
        self.non_neutral_trajectory += other.non_neutral_trajectory
        self.held_at_trajectory_end += other.held_at_trajectory_end

    def to_dict(self) -> dict[str, int]:
        return {
            "duplicate_down": self.duplicate_down,
            "dangling_up": self.dangling_up,
            "non_neutral_trajectory": self.non_neutral_trajectory,
            "held_at_trajectory_end": self.held_at_trajectory_end,
        }


def _continuous_primitive(kind: str, dx: float, dy: float) -> ActionPrimitive | None:
    rounded_dx = round(dx)
    rounded_dy = round(dy)
    if rounded_dx == 0 and rounded_dy == 0:
        return None
    return ActionPrimitive(kind=kind, dx=rounded_dx, dy=rounded_dy)


def _input_primitive(kind: str, value: Any) -> ActionPrimitive:
    name = str(value)
    if not _INPUT_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid input name: {name!r}")
    projected_kind: PrimitiveKind = "down" if kind == "press" else "up"
    return ActionPrimitive(kind=projected_kind, input_name=name)


def project_ordered_action(
    events: list[dict[str, Any]],
    *,
    interval_start_s: float,
    continuous_action_hz: float,
) -> ProjectedAction:
    if not math.isfinite(continuous_action_hz) or continuous_action_hz <= 0:
        raise ValueError("continuous_action_hz must be finite and positive")
    if not math.isfinite(interval_start_s):
        raise ValueError("interval_start_s must be finite")

    primitives: list[ActionPrimitive] = []
    pending: tuple[int, str, float, float] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        _tick, kind, dx, dy = pending
        primitive = _continuous_primitive(kind, dx, dy)
        if primitive is not None:
            primitives.append(primitive)
        pending = None

    for event in events:
        kind = str(event["kind"])
        if kind in {"move", "scroll"}:
            event_time_s = float(event["local_time_s"])
            dx = float(event["dx"])
            dy = float(event["dy"])
            if not all(math.isfinite(value) for value in (event_time_s, dx, dy)):
                raise ValueError(f"Non-finite continuous event: {event!r}")
            tick = math.floor((event_time_s - interval_start_s) * continuous_action_hz)
            if pending is not None and pending[0] == tick and pending[1] == kind:
                pending = (tick, kind, pending[2] + dx, pending[3] + dy)
            else:
                flush()
                pending = (tick, kind, dx, dy)
        elif kind in {"press", "release"}:
            flush()
            primitives.append(_input_primitive(kind, event["key"]))
        else:
            raise ValueError(f"Unsupported action event kind: {kind!r}")
    flush()
    frozen = tuple(primitives)
    return ProjectedAction(
        text="; ".join(primitive.render() for primitive in frozen) if frozen else "NO_OP",
        primitives=frozen,
    )


def update_held_state(
    primitives: tuple[ActionPrimitive, ...],
    *,
    held: set[str],
    diagnostics: HeldStateDiagnostics,
) -> None:
    for primitive in primitives:
        if primitive.kind == "down":
            assert primitive.input_name is not None
            if primitive.input_name in held:
                diagnostics.duplicate_down += 1
            held.add(primitive.input_name)
        elif primitive.kind == "up":
            assert primitive.input_name is not None
            if primitive.input_name not in held:
                diagnostics.dangling_up += 1
            held.discard(primitive.input_name)
```

- [ ] **Step 4: Run focused tests and Ruff**

Run:

```bash
cd data_pipeline
PYTHONPATH=. python -m unittest tests.test_action_format -v
uv run ruff format --check annotation_pipeline/action_format.py tests/test_action_format.py
uv run ruff check annotation_pipeline/action_format.py tests/test_action_format.py
```

Expected: all seven tests pass; formatter and linter exit zero.

- [ ] **Step 5: Commit the projection module**

```bash
git add data_pipeline/annotation_pipeline/action_format.py data_pipeline/tests/test_action_format.py
git commit -m "feat: add ordered action projection"
```

### Task 2: Integrate ordered actions into Stages 05 and 06

**Files:**
- Modify: `data_pipeline/annotation_pipeline/stage_05_assemble_trajectories.py`
- Modify: `data_pipeline/annotation_pipeline/stage_06_project_sft.py`
- Modify: `data_pipeline/tests/test_structured_pipeline.py`

- [ ] **Step 1: Extend structured-pipeline fixtures and write failing integration tests**

Update `_observation` in `data_pipeline/tests/test_structured_pipeline.py` so frame zero includes matching ordered move/press/release events with `local_time_s` inside its interval. Add assertions that Stage 05 retains `interval_start_s` and `interval_end_s`. Replace the current-format projection test with two tests:

```python
def test_sft_projection_uses_ordered_v2_by_default(self) -> None:
    observations = [_observation(0), _observation(1)]
    goals = refine_boundaries([_proposal(0, 1)], observations, policy="vision_only")
    trajectories, _ = assemble_trajectories(observations, goals)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        trajectories_path = root / "trajectories.jsonl"
        trajectories_path.write_text(json.dumps(trajectories[0]) + "\n")
        manifest = project_sft(
            trajectories_path=trajectories_path,
            output_dir=root / "sft",
            val_frac=0.0,
        )
        sample = json.loads((root / "sft" / "chat.jsonl").read_text())

    assistant_texts = [
        message["content"][0]["text"]
        for message in sample["messages"]
        if message["role"] == "assistant"
    ]
    self.assertEqual(
        assistant_texts,
        ["move(15,-2); down(LMB); up(LMB)", "NO_OP"],
    )
    self.assertEqual(manifest["action_schema"], "ordered_events_v2")
    self.assertEqual(manifest["continuous_action_hz"], 10.0)
    self.assertEqual(
        manifest["primitive_counts"],
        {"down": 1, "move": 1, "scroll": 0, "up": 1},
    )
    self.assertEqual(manifest["n_no_op_turns"], 1)


def test_sft_projection_keeps_aggregate_v1_as_explicit_ablation(self) -> None:
    observations = [_observation(0), _observation(1)]
    goals = refine_boundaries([_proposal(0, 1)], observations, policy="vision_only")
    trajectories, _ = assemble_trajectories(observations, goals)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        trajectories_path = root / "trajectories.jsonl"
        trajectories_path.write_text(json.dumps(trajectories[0]) + "\n")
        manifest = project_sft(
            trajectories_path=trajectories_path,
            output_dir=root / "sft",
            action_schema="aggregate_delta_keys_v1",
            val_frac=0.0,
        )
        sample = json.loads((root / "sft" / "chat.jsonl").read_text())

    assistant_texts = [
        message["content"][0]["text"]
        for message in sample["messages"]
        if message["role"] == "assistant"
    ]
    self.assertEqual(assistant_texts, ["15 -2 0 ; +LMB -LMB", "NO_OP"])
    self.assertEqual(manifest["action_schema"], "aggregate_delta_keys_v1")
    self.assertIsNone(manifest["continuous_action_hz"])
```

Add a trajectory with `up(LMB); down(KeyA); down(KeyA)` across its steps and assert the Stage-06 `state_diagnostics` values from Task 1 are recorded without altering the assistant text.

- [ ] **Step 2: Run the integration tests and verify failures**

Run:

```bash
cd data_pipeline
PYTHONPATH=. python -m unittest tests.test_structured_pipeline -v
```

Expected: failures because Stage 05 drops interval bounds and Stage 06 has no schema selection or v2 rendering.

- [ ] **Step 3: Retain interval bounds in Stage 05**

Add these fields to each assembled step in `stage_05_assemble_trajectories.py`:

```python
"interval_start_s": float(item["interval_start_s"]),
"interval_end_s": float(item["interval_end_s"]),
```

- [ ] **Step 4: Add Stage-06 schema parameters and rendering**

Import `ActionPrimitive`, `ProjectedAction`, the schema constants,
`HeldStateDiagnostics`, `project_ordered_action`, and `update_held_state`.
Extend `render_messages` and `project_sft` with `action_schema` and
`continuous_action_hz` keyword arguments. Validate the schema against
`ACTION_SCHEMAS` and the rate as finite and positive before creating the output
directory.

Add this local adapter so the v1 ablation contributes comparable manifest
statistics without changing its text:

```python
def _project_aggregate_action(value: dict[str, Any]) -> ProjectedAction:
    action_bin = action_bin_from_dict(value)
    primitives: list[ActionPrimitive] = []
    dx = round(action_bin.move_dx)
    dy = round(action_bin.move_dy)
    scroll = round(action_bin.scroll)
    if dx != 0 or dy != 0:
        primitives.append(ActionPrimitive(kind="move", dx=dx, dy=dy))
    if scroll != 0:
        primitives.append(ActionPrimitive(kind="scroll", dx=0, dy=scroll))
    primitives.extend(
        ActionPrimitive(
            kind="down" if sign == "+" else "up",
            input_name=name,
        )
        for sign, name in action_bin.events
    )
    return ProjectedAction(
        text=format_action(action_bin),
        primitives=tuple(primitives),
    )
```

For each step:

```python
if action_schema == ORDERED_ACTION_SCHEMA:
    projected = project_ordered_action(
        step["events"],
        interval_start_s=float(step["interval_start_s"]),
        continuous_action_hz=continuous_action_hz,
    )
    action = projected.text
    primitives = projected.primitives
else:
    projected = _project_aggregate_action(step["action_bin"])
    action = projected.text
    primitives = projected.primitives
```

Return the per-step `ProjectedAction` objects from `render_messages` alongside
messages and image paths. After a sample renders successfully, compute its
statistics exactly once:

```python
sample_counts: Counter[str] = Counter()
sample_no_ops = 0
sample_diagnostics = HeldStateDiagnostics()
held: set[str] = set()
for projected in projected_actions:
    if projected.text == "NO_OP":
        sample_no_ops += 1
    sample_counts.update(primitive.kind for primitive in projected.primitives)
    update_held_state(
        projected.primitives,
        held=held,
        diagnostics=sample_diagnostics,
    )
sample_diagnostics.finish_trajectory(held)

primitive_counts.update(sample_counts)
n_no_op_turns += sample_no_ops
state_diagnostics.update(sample_diagnostics)
```

Set the record's rendered non-noop count with:

```python
"n_non_noop": len(projected_actions) - sample_no_ops,
```

Write the manifest fields exactly as specified:

```python
"action_schema": action_schema,
"continuous_action_hz": (
    continuous_action_hz if action_schema == ORDERED_ACTION_SCHEMA else None
),
"primitive_counts": {
    kind: primitive_counts.get(kind, 0)
    for kind in ("move", "scroll", "down", "up")
},
"n_no_op_turns": n_no_op_turns,
"state_diagnostics": state_diagnostics.to_dict(),
```

- [ ] **Step 5: Add Stage-06 CLI arguments**

Add:

```python
parser.add_argument(
    "--action-schema",
    choices=ACTION_SCHEMAS,
    default=DEFAULT_ACTION_SCHEMA,
)
parser.add_argument(
    "--continuous-action-hz",
    type=float,
    default=DEFAULT_CONTINUOUS_ACTION_HZ,
)
```

Pass both values from `main()` into `project_sft`.

- [ ] **Step 6: Run focused integration tests and Ruff**

Run:

```bash
cd data_pipeline
PYTHONPATH=. python -m unittest tests.test_action_format tests.test_structured_pipeline -v
uv run ruff format --check annotation_pipeline/action_format.py annotation_pipeline/stage_05_assemble_trajectories.py annotation_pipeline/stage_06_project_sft.py tests/test_action_format.py tests/test_structured_pipeline.py
uv run ruff check annotation_pipeline/action_format.py annotation_pipeline/stage_05_assemble_trajectories.py annotation_pipeline/stage_06_project_sft.py tests/test_action_format.py tests/test_structured_pipeline.py
```

Expected: all focused tests pass; formatter and linter exit zero.

- [ ] **Step 7: Commit Stage-05/06 integration**

```bash
git add data_pipeline/annotation_pipeline/stage_05_assemble_trajectories.py data_pipeline/annotation_pipeline/stage_06_project_sft.py data_pipeline/tests/test_structured_pipeline.py
git commit -m "feat: project ordered actions in stage 06"
```

### Task 3: Wire the top-level builder and document the contract

**Files:**
- Modify: `data_pipeline/annotation_pipeline/build_sft.py`
- Modify: `data_pipeline/annotation_pipeline/README.md`
- Modify: `data_pipeline/tests/test_structured_pipeline.py`

- [ ] **Step 1: Write a failing build_sft passthrough test**

Extend `test_build_sft_uses_the_prepared_observation_view` to invoke:

```python
argv = [
    "build_sft",
    "--run-dir", str(run_dir),
    "--out", str(output_dir),
    "--val-frac", "0",
    "--action-schema", "ordered_events_v2",
    "--continuous-action-hz", "5",
]
```

Then read `stage_06_sft/manifest.json` and assert:

```python
self.assertEqual(sft_manifest["action_schema"], "ordered_events_v2")
self.assertEqual(sft_manifest["continuous_action_hz"], 5.0)
```

- [ ] **Step 2: Run the passthrough test and verify argparse failure**

Run:

```bash
cd data_pipeline
PYTHONPATH=. python -m unittest tests.test_structured_pipeline.StructuredPipelineTest.test_build_sft_uses_the_prepared_observation_view -v
```

Expected: failure because `build_sft.py` does not recognize the two new arguments.

- [ ] **Step 3: Add build_sft arguments and passthrough**

Import the four schema/rate constants from `annotation_pipeline.action_format`. Add the same two argparse definitions used by Stage 06, then pass:

```python
action_schema=args.action_schema,
continuous_action_hz=args.continuous_action_hz,
```

to `project_sft`.

- [ ] **Step 4: Update the pipeline README**

Replace the old action-ownership paragraph with the v2 grammar, a `move -> click -> move` example, the 10 Hz internal-grid rule, and explicit statements that zero movement is omitted and empty turns use `NO_OP`. Add command examples:

```bash
PYTHONPATH=. python3 -m annotation_pipeline.build_sft \
  --run-dir annotation_pipeline/dataset_runs/full \
  --out annotation_pipeline/dataset_runs/full/sft \
  --action-schema ordered_events_v2 \
  --continuous-action-hz 10
```

and the v1 ablation:

```bash
--action-schema aggregate_delta_keys_v1
```

State explicitly that coordinate transforms, mouse scaling, evaluator parsing, and runtime execution are not part of this change.

- [ ] **Step 5: Run focused tests and documentation checks**

Run:

```bash
cd data_pipeline
PYTHONPATH=. python -m unittest tests.test_structured_pipeline -v
uv run ruff format --check annotation_pipeline/build_sft.py tests/test_structured_pipeline.py
uv run ruff check annotation_pipeline/build_sft.py tests/test_structured_pipeline.py
```

Expected: all structured-pipeline tests pass; formatter and linter exit zero.

- [ ] **Step 6: Commit builder and documentation changes**

```bash
git add data_pipeline/annotation_pipeline/build_sft.py data_pipeline/annotation_pipeline/README.md data_pipeline/tests/test_structured_pipeline.py
git commit -m "docs: expose ordered action projection controls"
```

### Task 4: Full verification and scope audit

**Files:**
- Verify only; modify a failing file only if a verification command exposes a defect in the preceding tasks.

- [ ] **Step 1: Run the complete data-pipeline test suite**

```bash
cd data_pipeline
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Format and lint the complete data-pipeline scope**

```bash
cd data_pipeline
uv run ruff format --check annotation_pipeline tests
uv run ruff check annotation_pipeline tests
```

Expected: both commands exit zero.

- [ ] **Step 3: Compile Python sources**

```bash
python -m compileall -q data_pipeline/annotation_pipeline data_pipeline/tests
```

Expected: exit zero with no output.

- [ ] **Step 4: Confirm evaluator isolation and inspect the final diff**

```bash
git diff origin/yll/action-format...HEAD --name-only
git diff --check origin/yll/action-format...HEAD
```

Expected: no path begins with `eval/`, and `git diff --check` exits zero.

- [ ] **Step 5: Confirm no mouse-scaling or coordinate-transform code was introduced**

```bash
rg -n "mouse.*scal|random.*scal|normalize.*(dx|dy)|quantiz" data_pipeline/annotation_pipeline data_pipeline/tests
```

Expected: only documentation statements declaring those features out of scope, or no matches; no implementation match.

- [ ] **Step 6: Commit any verification-only corrections**

If Steps 1-5 required a correction, stage only those correction files and commit:

```bash
git commit -m "fix: harden ordered action projection"
```

If no correction was required, do not create an empty commit.
