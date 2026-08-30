"""Tests for the scripted oracle and the harvest record builder.

The load-bearing property here is a round trip: every action line the oracle
emits must survive ``parse_ordered_action_tolerant`` ->
``native_ordered_to_relstep`` -> ``denormalize_native_ordered_action``, the exact
path the eval puts a model reply through. If that holds, harvested targets are
parseable and dispatch to what the oracle intended -- which is the whole
argument for generating the data through the harness instead of offline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import cua_micro_harvest
import cua_micro_oracle
from action_parser import parse_ordered_action_tolerant
from cua_micro_eval import (
    denormalize_native_ordered_action,
    in_bbox,
    load_suite,
    native_ordered_to_relstep,
)
from cua_micro_oracle import OracleEnv, OracleError, OracleRuntime

_SCREEN = (1920, 1080)
_MODEL = (1280, 720)
# The LibreOffice Writer dock icon, in VM pixels at _SCREEN.
_ICON = (4, 308, 73, 373)


def _runtime(cursor=(960, 21), bbox=_ICON, *, seed=7, title="") -> OracleRuntime:
    return OracleRuntime(
        env=OracleEnv(active_title=lambda: title),
        screen=_SCREEN,
        model_resolution=_MODEL,
        rng=random.Random(seed),
        cursor=cursor,
        bbox=bbox,
    )


def _drive(plan, rt, *, max_turns=32):
    """Pull the plan, applying each action's motion to the runtime cursor.

    Stands in for the VM: the harness re-reads the real cursor between turns, so
    a test that did not move the cursor would let compounding errors pass.
    """
    lines = []
    for line in cua_micro_oracle.run_plan(rt, plan):
        lines.append(line)
        assert len(lines) <= max_turns, f"plan ran away: {lines}"
        action = denormalize_native_ordered_action(
            native_ordered_to_relstep(parse_ordered_action_tolerant(line)),
            _SCREEN,
            _MODEL,
        )
        for primitive in action.primitives:
            if primitive.kind == "move":
                rt.cursor = (rt.cursor[0] + primitive.dx, rt.cursor[1] + primitive.dy)
        rt.turn_index += 1
    return lines


# ---------------------------------------------------------------------------
# approach geometry
# ---------------------------------------------------------------------------


def test_approach_lands_on_target_and_clicks():
    rt = _runtime()
    lines = _drive([{"op": "approach", "click": "LMB"}], rt)
    assert 2 <= len(lines) <= 3, lines
    assert in_bbox(rt.cursor, _ICON), (rt.cursor, _ICON)
    assert lines[-1].endswith("down(LMB); up(LMB)")
    # Only the final turn clicks -- the approach turns are pure motion.
    assert all("LMB" not in line for line in lines[:-1])


@pytest.mark.parametrize("seed", range(25))
def test_approach_converges_from_many_seeds(seed):
    rt = _runtime(seed=seed)
    _drive([{"op": "approach", "click": "LMB"}], rt)
    assert in_bbox(rt.cursor, _ICON), f"seed {seed} missed: {rt.cursor}"


@pytest.mark.parametrize("seed", range(25))
def test_intermediate_steps_do_not_arrive_early(seed):
    """The point of staging is that early moves get CLOSE, not that they land.

    If a first move already sat on the target, the later steps would be
    demonstrating nothing and the model would learn to teleport.
    """
    rt = _runtime(seed=seed)
    lines = _drive([{"op": "approach", "click": "LMB", "steps": 3}], rt)
    assert len(lines) == 3
    start = _runtime(seed=seed).cursor
    # Re-drive one turn at a time to inspect the cursor between steps.
    rt2 = _runtime(seed=seed)
    positions = []
    for line in lines[:-1]:
        action = denormalize_native_ordered_action(
            native_ordered_to_relstep(parse_ordered_action_tolerant(line)), _SCREEN, _MODEL
        )
        for primitive in action.primitives:
            if primitive.kind == "move":
                rt2.cursor = (rt2.cursor[0] + primitive.dx, rt2.cursor[1] + primitive.dy)
        positions.append(rt2.cursor)
    assert not any(in_bbox(point, _ICON) for point in positions), positions
    # ... but each step must strictly reduce the distance to the target.
    def distance(point):
        cx, cy = (_ICON[0] + _ICON[2]) // 2, (_ICON[1] + _ICON[3]) // 2
        return (point[0] - cx) ** 2 + (point[1] - cy) ** 2

    assert distance(positions[0]) < distance(start)
    assert distance(positions[1]) < distance(positions[0])


def test_approach_step_count_and_offsets_vary_across_seeds():
    """Jitter is the feature: identical demos would teach three fixed deltas."""
    shapes = set()
    firsts = set()
    for seed in range(40):
        rt = _runtime(seed=seed)
        lines = _drive([{"op": "approach", "click": "LMB"}], rt)
        shapes.add(len(lines))
        firsts.add(lines[0])
    assert shapes == {2, 3}, shapes
    assert len(firsts) > 30, len(firsts)


def test_move_deltas_are_in_model_resolution_pixels():
    """A delta is expressed in the frame the model SAW, not VM pixels.

    denormalize_native_ordered_action scales model->VM at dispatch, so emitting
    VM pixels here would overshoot every target by the resolution ratio.
    """
    rt = _runtime(cursor=(960, 21), bbox=(4, 308, 73, 373), seed=3)
    line = next(iter(cua_micro_oracle.run_plan(rt, [{"op": "approach", "steps": 2}])))
    parsed = native_ordered_to_relstep(parse_ordered_action_tolerant(line))
    model_dx = parsed.primitives[0].dx
    vm_dx = denormalize_native_ordered_action(parsed, _SCREEN, _MODEL).primitives[0].dx
    assert model_dx != vm_dx
    assert vm_dx == pytest.approx(model_dx * _SCREEN[0] / _MODEL[0], abs=1)


def test_no_zero_move_is_ever_emitted():
    """The system prompt forbids move(0,0) outright."""
    center = ((_ICON[0] + _ICON[2]) // 2, (_ICON[1] + _ICON[3]) // 2)
    rt = _runtime(cursor=center)
    lines = _drive([{"op": "approach", "click": "LMB"}], rt)
    assert all("move(0,0)" not in line for line in lines)
    assert lines[-1].endswith("down(LMB); up(LMB)")


# ---------------------------------------------------------------------------
# other ops
# ---------------------------------------------------------------------------


def test_key_chord_presses_in_order_and_releases_in_reverse():
    rt = _runtime()
    (line,) = _drive([{"op": "key", "keys": ["ControlLeft", "ShiftLeft", "KeyT"]}], rt)
    assert line == (
        "down(ControlLeft); down(ShiftLeft); down(KeyT); "
        "up(KeyT); up(ShiftLeft); up(ControlLeft)"
    )


def test_type_then_confirm_keeps_return_out_of_the_string():
    rt = _runtime()
    (line,) = _drive([{"op": "type", "text": "wikipedia", "then": ["Return"]}], rt)
    assert line == 'type("wikipedia"); down(Return); up(Return)'


def test_type_escapes_quotes_and_backslashes():
    rt = _runtime()
    (line,) = _drive([{"op": "type", "text": 'a "b" \\c'}], rt)
    assert parse_ordered_action_tolerant(line).primitives[0].text == 'a "b" \\c'


def test_double_click_emits_two_press_release_pairs():
    rt = _runtime()
    lines = _drive([{"op": "approach", "click": "LMB", "count": 2}], rt)
    assert lines[-1].count("down(LMB); up(LMB)") == 2


def test_wait_title_stops_as_soon_as_the_title_appears():
    rt = _runtime(title="0x1 LibreOffice Writer")
    lines = _drive([{"op": "wait_title", "pattern": "LibreOffice Writer"}], rt)
    assert lines == []


def test_wait_title_raises_when_the_window_never_appears():
    rt = _runtime(title="something else")
    with pytest.raises(OracleError, match="never appeared"):
        _drive([{"op": "wait_title", "pattern": "Nope", "max_turns": 3}], rt)


def test_every_emitted_line_round_trips_through_the_eval_parser():
    """The property that makes harvested targets safe to train on."""
    plan = [
        {"op": "approach", "click": "LMB"},
        {"op": "key", "keys": ["ControlLeft", "KeyS"]},
        {"op": "type", "text": "report.pdf", "then": ["Return"]},
        {"op": "scroll", "dx": 0, "dy": -5},
        {"op": "wait", "turns": 2},
    ]
    lines = _drive(plan, _runtime())
    for line in lines:
        parsed = parse_ordered_action_tolerant(line)
        if line == "NO_OP":
            assert parsed.no_op
        else:
            assert parsed.primitives


# ---------------------------------------------------------------------------
# plan validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        {"plan": []},
        {"plan": [{"op": "nope"}]},
        {"plan": [{"op": "approach", "steps": 5}]},
        {"plan": [{"op": "type"}]},
        {"plan": [{"op": "wait_title"}]},
        {"plan": [{"op": "approach"}], "extra": 1},
        [{"op": "approach"}],
    ],
)
def test_bad_plans_are_rejected_at_load_time(spec):
    with pytest.raises(ValueError):
        cua_micro_oracle.validate_plan(spec, where="task 0")


def test_suite_without_oracle_still_loads(tmp_path: Path):
    """Adding the field must not change what an existing suite means."""
    suite = {
        "schema_version": 1,
        "suite": "t",
        "coordinate_grid": 1000,
        "tasks": [
            {
                "id": "a",
                "category": "c",
                "instruction": "i",
                "setup": {"kind": "desktop"},
                "turn_mode": "multiturn",
                "max_turns": 4,
                "turn": {
                    "target": {"kind": "fixed_norm", "bbox": [1, 1, 9, 9]},
                    "cursor": {"kind": "target_center"},
                    "expected": {"kind": "any"},
                    "verifier": {"kind": "active_title_regex", "pattern": "x"},
                },
            }
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    _, tasks = load_suite(path)
    assert tasks[0].oracle_plan is None


def test_oracle_block_is_parsed_off_the_suite(tmp_path: Path):
    suite = {
        "schema_version": 1,
        "suite": "t",
        "coordinate_grid": 1000,
        "tasks": [
            {
                "id": "a",
                "category": "c",
                "instruction": "i",
                "setup": {"kind": "desktop"},
                "turn_mode": "multiturn",
                "max_turns": 4,
                "oracle": {"plan": [{"op": "approach", "click": "LMB"}]},
                "turn": {
                    "target": {"kind": "fixed_norm", "bbox": [1, 1, 9, 9]},
                    "cursor": {"kind": "normalized", "point": [500, 20]},
                    "expected": {"kind": "any"},
                    "verifier": {"kind": "active_title_regex", "pattern": "x"},
                },
            }
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    _, tasks = load_suite(path)
    assert tasks[0].oracle_plan == ({"op": "approach", "click": "LMB"},)


# ---------------------------------------------------------------------------
# harvest records
# ---------------------------------------------------------------------------


def _fake_attempt(tmp_path: Path, n_turns: int, *, success=True) -> tuple[dict, Path]:
    steps = tmp_path / "steps"
    steps.mkdir()
    for index in range(n_turns + 1):
        (steps / f"step_{index:03d}.png").write_bytes(b"png")
        (steps / f"step_{index:03d}_after.png").write_bytes(b"after")
    result = {
        "task_id": "click.desktop.libreoffice_writer",
        "category": "native_launch",
        "instruction": "Click the LibreOffice Writer icon.",
        "seed": 41000,
        "success": success,
        "action_format": "cua_ordered_typing_v1",
        "model_resolution": [1280, 720],
        "screen_size": [1920, 1080],
        "turns": [
            {"response": f"move({-10 * (i + 1)},{5 * (i + 1)})"} for i in range(n_turns)
        ],
    }
    return result, tmp_path


def test_chat_record_matches_the_stage04_message_shape(tmp_path: Path):
    result, attempt_dir = _fake_attempt(tmp_path, 3)
    record = cua_micro_harvest.build_chat_record(
        result=result,
        attempt_dir=attempt_dir,
        suite_name="cua_micro_tasks",
        system_prompt="SYS",
        system_prompt_id="cua_ordered_typing_v1",
        plan=({"op": "approach"},),
    )
    messages = record["messages"]
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": "SYS"}]}
    # 4 frames (3 turns + the terminal one), one user+assistant pair each.
    assert record["n_frames"] == 4
    assert len(messages) == 1 + 2 * 4
    # Instruction text precedes the image on the first user turn only.
    assert messages[1]["content"][0] == {
        "type": "text",
        "text": "Click the LibreOffice Writer icon.",
    }
    assert messages[1]["content"][1]["type"] == "image"
    assert [block["type"] for block in messages[3]["content"]] == ["image"]
    assert messages[-1]["content"] == [{"type": "text", "text": "NO_OP"}]
    # Every content field is a list of typed blocks, never a bare string.
    assert all(isinstance(message["content"], list) for message in messages)


def test_chat_record_uses_before_frames_not_after_frames(tmp_path: Path):
    result, attempt_dir = _fake_attempt(tmp_path, 2)
    record = cua_micro_harvest.build_chat_record(
        result=result,
        attempt_dir=attempt_dir,
        suite_name="s",
        system_prompt="SYS",
        system_prompt_id="cua_ordered_typing_v1",
        plan=(),
    )
    images = [
        block["image"]
        for message in record["messages"]
        if message["role"] == "user"
        for block in message["content"]
        if block["type"] == "image"
    ]
    assert [Path(image).name for image in images] == [
        "step_000.png",
        "step_001.png",
        "step_002.png",
    ]


def test_unverified_trajectories_are_dropped(tmp_path: Path):
    result, attempt_dir = _fake_attempt(tmp_path, 2, success=False)
    assert (
        cua_micro_harvest.build_chat_record(
            result=result,
            attempt_dir=attempt_dir,
            suite_name="s",
            system_prompt="SYS",
            system_prompt_id="cua_ordered_typing_v1",
            plan=(),
        )
        is None
    )


def test_terminal_frame_can_be_dropped(tmp_path: Path):
    result, attempt_dir = _fake_attempt(tmp_path, 3)
    record = cua_micro_harvest.build_chat_record(
        result=result,
        attempt_dir=attempt_dir,
        suite_name="s",
        system_prompt="SYS",
        system_prompt_id="cua_ordered_typing_v1",
        plan=(),
        terminal_action=None,
    )
    assert record["n_frames"] == 3
    assert record["messages"][-1]["content"] == [{"type": "text", "text": "move(-30,15)"}]


def test_harvest_summary_flags_tasks_that_produced_nothing():
    summary = cua_micro_harvest.harvest_summary(
        [{"task_id": "a", "n_frames": 4}],
        attempted=4,
        suite_name="s",
        task_ids=["a", "b"],
    )
    assert summary["n_kept"] == 1
    assert summary["n_dropped"] == 3
    assert summary["tasks_with_no_data"] == ["b"]


# ---------------------------------------------------------------------------
# the shipped suites
# ---------------------------------------------------------------------------


# The oracle-carrying suites live in osworld_freeroll_v22_claude: a plain
# "osworld_freeroll_v22" (a fresh copy of v21, no oracle blocks) was created
# alongside them, so the directory name has to be explicit here.
_V22 = (
    Path(__file__).resolve().parents[1]
    / "slurm/dev/alfred/berlin/labctl/recipes/eval/osworld_freeroll_v22_claude"
)


@pytest.mark.parametrize("name", ["easy", "mid"])
def test_v22_suites_are_harvestable(name):
    _, tasks = load_suite(_V22 / f"cua_micro_tasks_{name}.json")
    assert tasks
    for task in tasks:
        assert task.oracle_plan, task.task_id
        assert task.turn_mode == "multiturn", task.task_id
        # 16 everywhere except the two genuinely multi-step Chrome tasks, whose
        # plans need room for a Chrome cold start plus two approaches.
        expected_turns = 20 if task.task_id.startswith("multi.chrome.") else 16
        assert len(task.turns) == expected_turns, (task.task_id, len(task.turns))
        # No task may start with the pointer already on its target: the
        # approach is the skill these suites are meant to exercise.
        assert task.turns[0].cursor == {"kind": "normalized", "point": [500, 20]}


def _worst_case_turns(plan) -> int:
    """Upper bound on the turns a plan can consume: every approach 3 steps and
    every wait burning its full budget."""
    total = 0
    for op in plan:
        kind = op["op"]
        if kind == "approach":
            total += max(_APPROACH_SHAPES_MAX, op.get("steps") or _APPROACH_SHAPES_MAX)
        elif kind == "wait":
            total += max(1, int(op.get("turns", 1)))
        elif kind == "wait_title":
            total += int(op.get("max_turns", 6))
        else:
            total += 1
    return total


_APPROACH_SHAPES_MAX = 3


@pytest.mark.parametrize("name", ["easy", "mid"])
def test_v22_plans_fit_their_turn_budget(name):
    """A plan that cannot finish inside max_turns yields nothing at all.

    The trajectory is truncated mid-plan, the verifier never fires, and
    build_chat_record drops it -- so an over-budget plan shows up as a task with
    zero training data rather than as an error.
    """
    _, tasks = load_suite(_V22 / f"cua_micro_tasks_{name}.json")
    for task in tasks:
        worst = _worst_case_turns(task.oracle_plan)
        assert worst <= len(task.turns), (task.task_id, worst, len(task.turns))
