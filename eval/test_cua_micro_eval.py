"""Unit tests for the deterministic one-turn CUA micro-eval contract."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from contextlib import suppress
from pathlib import Path
from typing import ClassVar
from unittest import mock

from PIL import Image

import cua_micro_eval as micro
from action_parser import parse_ordered_action_tolerant
from cua_micro_action_parser import (
    parse_computer_use_rel_step_action,
    parse_qwen3vl_computer_use_action,
)
from cua_micro_eval import _call_model
from osworld_system_prompts import SYSTEM_PROMPTS
from sampling import qwen_sampling

_SUITE = Path(__file__).with_name("cua_micro_tasks.json")


def _tool(arguments: dict) -> str:
    return (
        "<tool_call>"
        + json.dumps({"name": "computer_use", "arguments": arguments}, separators=(",", ":"))
        + "</tool_call>"
    )


class SuiteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw, cls.tasks = micro.load_suite(_SUITE)

    def test_suite_identity_and_composition_are_pinned(self) -> None:
        self.assertEqual(self.raw["schema_version"], 1)
        self.assertEqual(self.raw["suite"], "cua_micro_tasks")
        self.assertEqual(self.raw["coordinate_grid"], 1000)
        self.assertEqual(
            {
                task.task_id: task.turn_mode if task.turns else "atomic"
                for task in self.tasks
            },
            {
                "click.desktop.libreoffice_writer": "atomic",
                "click.desktop.libreoffice_impress": "atomic",
                "key.desktop.open_terminal": "multiturn",
                "type.terminal.native_exact": "multiturn",
                "type.text_editor.native_exact": "multiturn",
                "key.writer.open_save_as": "atomic",
                "key.impress.open_save_as": "atomic",
                "click.files.open_eval_target": "atomic",
                "key.calculator.digit7": "atomic",
                "click.chrome.back": "atomic",
                "click.chrome.reload": "atomic",
                "click.chrome.deterministic_button": "atomic",
                "scroll.chrome.down": "atomic",
                "multi.calculator.73_plus_19": "multiturn",
                "multi.chrome.search_3blue1brown": "multiturn",
                "multi.chrome.open_chrome_search_wikipedia": "multiturn",
                "multi.terminal.vim_hello_world_script": "multiturn",
                "multi.terminal.hello_world_script": "multiturn",
            },
        )

    def test_requested_families_are_present(self) -> None:
        ids = {task.task_id for task in self.tasks}
        required_fragments = {
            "desktop.libreoffice_writer",
            "desktop.libreoffice_impress",
            "chrome.reload",
            "chrome.back",
            "terminal.native_exact",
            "text_editor.native_exact",
            "calculator.digit7",
            "files.open_eval_target",
            "scroll.chrome.down",
        }
        for fragment in required_fragments:
            self.assertTrue(
                any(fragment in task_id for task_id in ids),
                f"missing task family {fragment}",
            )

    def test_all_turns_expect_one_atomic_primitive(self) -> None:
        for task in self.tasks:
            for turn in micro.task_turns(task):
                self.assertIn(
                    turn.expected["kind"],
                    # "any" marks outcome-only turns (turn_mode="multiturn"):
                    # no specific primitive is prescribed, only the verifier
                    # decides success.
                    {"move", "click", "type", "scroll", "key", "any"},
                )

    def test_typing_checks_exact_action_and_exact_app_state(self) -> None:
        typing = {
            task.task_id: task
            for task in self.tasks
            if task.task_id in {"type.terminal.native_exact", "type.text_editor.native_exact"}
        }
        self.assertEqual(
            set(typing),
            {"type.terminal.native_exact", "type.text_editor.native_exact"},
        )
        for task in typing.values():
            self.assertEqual(task.turn_mode, "multiturn")
            self.assertEqual(len(task.turns), 32)
            expected = task.turns[0].expected
            verifier = task.turns[0].verifier
            self.assertTrue(all(turn.expected == expected for turn in task.turns))
            self.assertTrue(all(turn.verifier == verifier for turn in task.turns))
            self.assertEqual(expected["kind"], "type")
            self.assertIn(verifier["kind"], {"guest_json_equals", "saved_file_equals"})
            self.assertEqual(expected["text"], verifier["value"])
            parsed = parse_computer_use_rel_step_action(
                _tool({"action": "type", "text": expected["text"]})
            )
            self.assertTrue(micro.action_matches_expected(parsed, expected))

    def test_multiturn_tasks_have_semantically_verified_steps(self) -> None:
        tasks = [task for task in self.tasks if task.turns]
        self.assertEqual(
            {
                task.task_id: (task.category, task.turn_mode, len(task.turns))
                for task in tasks
            },
            {
                "key.desktop.open_terminal": ("multi_turn", "multiturn", 4),
                "type.terminal.native_exact": ("native_app", "multiturn", 32),
                "type.text_editor.native_exact": ("native_app", "multiturn", 32),
                "multi.calculator.73_plus_19": ("multi_turn", "multiturn", 64),
                "multi.chrome.search_3blue1brown": ("multi_turn", "multiturn", 64),
                "multi.chrome.open_chrome_search_wikipedia": (
                    "multi_turn",
                    "multiturn",
                    64,
                ),
                "multi.terminal.vim_hello_world_script": (
                    "multi_turn",
                    "multiturn",
                    64,
                ),
                "multi.terminal.hello_world_script": ("multi_turn", "multiturn", 64),
            },
        )
        for task in tasks:
            self.assertGreaterEqual(len(task.turns), 2)
            for turn in task.turns:
                self.assertTrue(turn.turn_id)
                self.assertIn("kind", turn.verifier)

    def test_outcome_only_multiturn_tasks_are_explicitly_pinned(self) -> None:
        outcome_only = {
            task.task_id: task
            for task in self.tasks
            if task.turns and all(turn.expected == {"kind": "any"} for turn in task.turns)
        }
        self.assertEqual(
            set(outcome_only),
            {
                "key.desktop.open_terminal",
                "multi.calculator.73_plus_19",
                "multi.chrome.search_3blue1brown",
                "multi.chrome.open_chrome_search_wikipedia",
                "multi.terminal.vim_hello_world_script",
                "multi.terminal.hello_world_script",
            },
        )
        for task in outcome_only.values():
            self.assertEqual(task.turn_mode, "multiturn")
            self.assertGreaterEqual(len(task.turns), 2)
            verifiers = {json.dumps(turn.verifier, sort_keys=True) for turn in task.turns}
            self.assertEqual(len(verifiers), 1)

    def test_native_suite_is_balanced_across_real_apps(self) -> None:
        raw, tasks = micro.load_suite(_SUITE)
        self.assertEqual(raw["suite"], "cua_micro_tasks")
        ids = {task.task_id for task in tasks}
        for fragment in ("files", "terminal", "text_editor", "writer", "calc", "impress"):
            self.assertTrue(any(fragment in task_id for task_id in ids), fragment)
        self.assertLessEqual(sum("chrome" in task_id for task_id in ids), len(tasks) // 3)

    def _suite_with(self, task: dict) -> dict:
        return {
            "schema_version": 1,
            "suite": "cua_micro_tasks",
            "coordinate_grid": 1000,
            "description": "test",
            "tasks": [task],
        }

    def _write_suite(self, directory: Path, raw: dict) -> Path:
        path = Path(directory) / "suite.json"
        path.write_text(json.dumps(raw))
        return path

    _MULTITURN_TASK_BASE: ClassVar[dict] = {
        "id": "multi.test.multiturn",
        "category": "multi_turn",
        "instruction": "test",
        "setup": {"kind": "desktop"},
        "turns": [
            {
                "id": "a",
                "target": {"kind": "fixed_norm", "bbox": [0, 0, 10, 10], "label": "x"},
                "cursor": {"kind": "target_center"},
                "expected": {"kind": "any"},
                "verifier": {"kind": "bbox_hit"},
            },
            {
                "id": "b",
                "target": {"kind": "fixed_norm", "bbox": [0, 0, 10, 10], "label": "x"},
                "cursor": {"kind": "target_center"},
                "expected": {"kind": "any"},
                "verifier": {"kind": "bbox_hit"},
            },
        ],
    }

    def test_turn_mode_defaults_to_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_suite(directory, self._suite_with(dict(self._MULTITURN_TASK_BASE)))
            _, tasks = micro.load_suite(path)
        self.assertEqual(tasks[0].turn_mode, "prefix")

    def test_turn_mode_multiturn_is_accepted(self) -> None:
        task = {**self._MULTITURN_TASK_BASE, "turn_mode": "multiturn"}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_suite(directory, self._suite_with(task))
            _, tasks = micro.load_suite(path)
        self.assertEqual(tasks[0].turn_mode, "multiturn")

    def test_turn_mode_rejects_unknown_value(self) -> None:
        task = {**self._MULTITURN_TASK_BASE, "turn_mode": "bogus"}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_suite(directory, self._suite_with(task))
            with self.assertRaisesRegex(ValueError, "turn_mode"):
                micro.load_suite(path)

    def test_turn_mode_is_rejected_on_atomic_tasks(self) -> None:
        atomic_task = {
            "id": "click.test",
            "category": "click",
            "instruction": "test",
            "setup": {"kind": "desktop"},
            "target": {"kind": "fixed_norm", "bbox": [0, 0, 10, 10], "label": "x"},
            "cursor": {"kind": "target_center"},
            "expected": {"kind": "click"},
            "verifier": {"kind": "bbox_hit"},
            "turn_mode": "multiturn",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_suite(directory, self._suite_with(atomic_task))
            with self.assertRaisesRegex(ValueError, "extra"):
                micro.load_suite(path)

    _TURN_TEMPLATE: ClassVar[dict] = {
        "target": {"kind": "fixed_norm", "bbox": [0, 0, 10, 10], "label": "x"},
        "cursor": {"kind": "target_center"},
        "expected": {"kind": "any"},
        "verifier": {"kind": "bbox_hit"},
    }

    def _multiturn_budget_task(self, **overrides: object) -> dict:
        task = {
            "id": "multi.test.budget",
            "category": "multi_turn",
            "instruction": "test",
            "setup": {"kind": "desktop"},
            "turn_mode": "multiturn",
            "turn": dict(self._TURN_TEMPLATE),
            "max_turns": 60,
        }
        task.update(overrides)
        return task

    def test_compact_turn_template_expands_to_max_turns_identical_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_suite(directory, self._suite_with(self._multiturn_budget_task()))
            _, tasks = micro.load_suite(path)
        turns = tasks[0].turns
        self.assertEqual(len(turns), 60)
        self.assertEqual(
            [turn.turn_id for turn in turns[:3]], ["attempt_1", "attempt_2", "attempt_3"]
        )
        self.assertEqual(turns[-1].turn_id, "attempt_60")
        self.assertTrue(all(turn.target == self._TURN_TEMPLATE["target"] for turn in turns))
        self.assertTrue(all(turn.verifier == self._TURN_TEMPLATE["verifier"] for turn in turns))

    def test_compact_turn_template_requires_multiturn_mode(self) -> None:
        task = self._multiturn_budget_task(turn_mode="prefix")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_suite(directory, self._suite_with(task))
            with self.assertRaisesRegex(ValueError, "turn_mode='multiturn'"):
                micro.load_suite(path)

    def test_compact_turn_template_rejects_turns_list_combined(self) -> None:
        task = self._multiturn_budget_task(turns=self._MULTITURN_TASK_BASE["turns"])
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_suite(directory, self._suite_with(task))
            with self.assertRaisesRegex(ValueError, "either 'turns' or 'turn'"):
                micro.load_suite(path)

    def test_compact_turn_template_rejects_bad_max_turns(self) -> None:
        for bad in (1, 0, -1, "60", 2.5, True):
            task = self._multiturn_budget_task(max_turns=bad)
            with tempfile.TemporaryDirectory() as directory:
                path = self._write_suite(directory, self._suite_with(task))
                with self.assertRaisesRegex(ValueError, "max_turns"):
                    micro.load_suite(path)

    def test_compact_turn_template_requires_all_turn_fields(self) -> None:
        incomplete = dict(self._TURN_TEMPLATE)
        del incomplete["verifier"]
        task = self._multiturn_budget_task(turn=incomplete)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_suite(directory, self._suite_with(task))
            with self.assertRaisesRegex(ValueError, "'turn' fields must be"):
                micro.load_suite(path)


class HistoryWindowTests(unittest.TestCase):
    """``evict_history``/``window_frame_labels`` (osworld_runtime.py), used by
    run_multiturn_attempt's optional ``n_history_frames`` cap."""

    def test_evict_history_noop_while_under_window(self) -> None:
        frames = ["f0", "f1", "f2"]
        actions = ["a0", "a1"]
        micro.evict_history(frames, actions, n_history_frames=5)
        self.assertEqual(frames, ["f0", "f1", "f2"])
        self.assertEqual(actions, ["a0", "a1"])

    def test_evict_history_keeps_newest_half_once_over_window(self) -> None:
        frames = [f"f{i}" for i in range(17)]
        actions = [f"a{i}" for i in range(16)]
        micro.evict_history(frames, actions, n_history_frames=16)
        self.assertEqual(frames, [f"f{i}" for i in range(9, 17)])
        self.assertEqual(actions, [f"a{i}" for i in range(9, 16)])
        # Invariant preserved: every frame but the newest has an action.
        self.assertEqual(len(actions), len(frames) - 1)

    def test_evict_history_minimum_keep_is_one_frame_no_actions(self) -> None:
        frames = ["f0", "f1"]
        actions = ["a0"]
        micro.evict_history(frames, actions, n_history_frames=1)
        self.assertEqual(frames, ["f1"])
        self.assertEqual(actions, [])

    def test_window_frame_labels_contiguous_tail_after_eviction(self) -> None:
        # 21 steps have happened (step_000..step_020); the window currently
        # holds the newest 9 of them.
        self.assertEqual(
            micro.window_frame_labels(21, 9),
            [f"step_{i:03d}.png" for i in range(12, 21)],
        )

    def test_window_frame_labels_first_turn(self) -> None:
        self.assertEqual(micro.window_frame_labels(1, 1), ["step_000.png"])


class GeometryTests(unittest.TestCase):
    def test_distance_to_bbox_is_zero_inside_and_nearest_edge_outside(self) -> None:
        bbox = (100, 100, 200, 200)
        self.assertEqual(micro.distance_to_bbox((150, 150), bbox), 0)
        self.assertEqual(micro.distance_to_bbox((90, 150), bbox), 10)
        self.assertEqual(micro.distance_to_bbox((210, 150), bbox), 11)
        self.assertEqual(micro.distance_to_bbox((90, 90), bbox), 10 * 2**0.5)

    def test_normalized_bbox_scales_per_axis(self) -> None:
        self.assertEqual(
            micro.norm_bbox_to_px([100, 200, 300, 400], (1920, 1080)),
            (192, 216, 576, 432),
        )

    def test_relative_cursor_start_uses_screen_fraction(self) -> None:
        bbox = (900, 500, 1000, 600)
        start = micro.resolve_cursor_start(
            {"kind": "relative_to_target", "delta_norm": [128, 0]},
            bbox,
            (1920, 1080),
        )
        self.assertEqual(start, (1195, 549))

    def test_best_legal_step_and_continuous_progress(self) -> None:
        screen = (1000, 1000)
        bbox = (480, 480, 520, 520)
        exact = micro.movement_metrics((647, 500), (519, 500), bbox, screen)
        self.assertTrue(exact["bbox_hit"])
        self.assertEqual(exact["legal_step_optimality"], 1.0)
        halfway = micro.movement_metrics((647, 500), (615, 500), bbox, screen)
        self.assertFalse(halfway["bbox_hit"])
        self.assertGreater(halfway["legal_step_optimality"], 0)
        self.assertLess(halfway["legal_step_optimality"], 1)

    def test_denormalize_scales_move_only(self) -> None:
        parsed = parse_computer_use_rel_step_action(
            _tool({"action": "mouse_move_rel", "delta": [128, -128]})
        )
        scaled = micro.denormalize_action(parsed, (1920, 1080))
        primitive = scaled.primitives[0]
        self.assertEqual((primitive.dx, primitive.dy), (246, -138))


class ActionAndAggregationTests(unittest.TestCase):
    def test_official_qwen3vl_native_prompt_contract_is_available(self) -> None:
        prompt = SYSTEM_PROMPTS["qwen3vl_native_cua_v1"]
        self.assertIn("You are a helpful assistant.", prompt)
        self.assertIn('"name": "computer_use"', prompt)
        self.assertIn("The screen's resolution is 1000x1000.", prompt)
        self.assertIn("<tool_call>", prompt)

    def test_qwen3vl_native_parser_accepts_thinking_and_absolute_move(self) -> None:
        response = "<think>locate target</think>\n" + _tool(
            {"action": "mouse_move", "coordinate": [750, 250]}
        )
        calls = parse_qwen3vl_computer_use_action(response)
        parsed = micro.qwen3vl_native_to_ordered(calls, (1920, 1080), (960, 540))
        primitive = parsed.primitives[0]
        self.assertEqual(primitive.kind, "move")
        self.assertEqual((primitive.dx, primitive.dy), (480, -270))

    def test_qwen3vl_native_parser_rejects_prose_and_bad_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside <tool_call>"):
            parse_qwen3vl_computer_use_action("Action: move\n" + _tool({"action": "left_click"}))
        with self.assertRaisesRegex(ValueError, "0..1000"):
            parse_qwen3vl_computer_use_action(
                _tool({"action": "mouse_move", "coordinate": [1001, 0]})
            )

    def test_qwen3vl_native_multiple_calls_are_parse_valid_but_not_atomic(self) -> None:
        calls = parse_qwen3vl_computer_use_action(
            _tool({"action": "mouse_move", "coordinate": [500, 500]})
            + _tool({"action": "left_click"})
        )
        parsed = micro.qwen3vl_native_to_ordered(calls, (1000, 1000), (0, 0))
        self.assertEqual(len(parsed.primitives), 2)
        self.assertFalse(micro.action_matches_expected(parsed, {"kind": "click", "button": "left"}))

    def test_chrome_tab_activation_uses_exact_cdp_target(self) -> None:
        client = mock.Mock()
        client.run_command.return_value = {"status": "success"}

        result = micro._activate_chrome_target(client, "BETA")

        self.assertEqual(result, {"status": "success"})
        command = client.run_command.call_args.args[0]
        self.assertEqual(command[:2], ["python3", "-c"])
        self.assertIn("t.get('title') == 'BETA'", command[2])
        self.assertIn("/json/activate/", command[2])

    def test_expected_action_is_strict_about_payload(self) -> None:
        click = parse_computer_use_rel_step_action(_tool({"action": "left_click"}))
        double = parse_computer_use_rel_step_action(_tool({"action": "double_click"}))
        typed = parse_computer_use_rel_step_action(_tool({"action": "type", "text": "rollout-ok"}))
        self.assertTrue(micro.action_matches_expected(click, {"kind": "click", "button": "left"}))
        self.assertFalse(micro.action_matches_expected(double, {"kind": "click", "button": "left"}))
        self.assertTrue(
            micro.action_matches_expected(typed, {"kind": "type", "text": "rollout-ok"})
        )
        self.assertFalse(micro.action_matches_expected(typed, {"kind": "type", "text": "other"}))

    def test_expected_action_supports_exact_key_chords_and_double_clicks(self) -> None:
        chord = micro.OrderedAction(
            primitives=(micro.OrderedPrimitive(kind="key_combo", keys=("CTRL", "SHIFT", "S")),),
            no_op=False,
        )
        double = parse_computer_use_rel_step_action(_tool({"action": "double_click"}))
        self.assertTrue(
            micro.action_matches_expected(
                chord,
                {"kind": "key", "keys": ["ctrl", "shift", "s"]},
            )
        )
        self.assertTrue(
            micro.action_matches_expected(
                double,
                {"kind": "click", "button": "left", "count": 2},
            )
        )

    def test_any_expected_kind_accepts_any_nonempty_action_regardless_of_format(self) -> None:
        click = parse_computer_use_rel_step_action(_tool({"action": "left_click"}))
        self.assertTrue(micro.action_matches_expected(click, {"kind": "any"}))
        multi_primitive = micro.OrderedAction(
            primitives=(
                micro.OrderedPrimitive(kind="type", text="73"),
                micro.OrderedPrimitive(kind="key_combo", keys=("ENTER",)),
            ),
            no_op=False,
        )
        self.assertTrue(micro.action_matches_expected(multi_primitive, {"kind": "any"}))
        native_click = micro.native_ordered_to_relstep(
            parse_ordered_action_tolerant("down(LMB); up(LMB)")
        )
        self.assertTrue(
            micro.action_matches_expected(
                native_click, {"kind": "any"}, micro._NATIVE_ORDERED_FORMAT
            )
        )

    def test_any_expected_kind_rejects_no_op_and_missing_action(self) -> None:
        wait_only = micro.OrderedAction(
            primitives=(micro.OrderedPrimitive(kind="wait"),), no_op=True
        )
        self.assertFalse(micro.action_matches_expected(wait_only, {"kind": "any"}))
        self.assertFalse(micro.action_matches_expected(None, {"kind": "any"}))

    def test_pass_at_four_and_best_progress(self) -> None:
        task = micro.Task(
            task_id="move.test",
            category="move",
            instruction="move",
            setup={},
            target={},
            cursor={},
            expected={},
            verifier={},
        )
        attempts = [
            {
                "task_id": task.task_id,
                "success": success,
                "validity": "valid",
                "progress": progress,
                "parse_valid": parse_valid,
                "expected_action_ok": expected,
            }
            for success, progress, parse_valid, expected in (
                (False, 0.2, True, True),
                (False, 0.5, False, False),
                (True, 1.0, True, True),
                (False, 0.7, True, True),
            )
        ]
        aggregate = micro.aggregate_results([task], attempts)
        row = aggregate["per_task"][task.task_id]
        self.assertEqual(row["pass_at_1"], 0.25)
        self.assertTrue(row["pass_at_4"])
        self.assertEqual(row["best_of_4_progress"], 1.0)
        self.assertEqual(row["parse_valid_rate"], 0.75)

    def test_infrastructure_failures_have_raw_and_valid_denominators(self) -> None:
        task = micro.Task(
            task_id="click.test",
            category="click",
            instruction="click",
            setup={},
            target={},
            cursor={},
            expected={},
            verifier={},
        )
        attempts = [
            {
                "task_id": task.task_id,
                "success": True,
                "validity": "valid",
                "progress": 1.0,
                "parse_valid": True,
                "expected_action_ok": True,
            },
            {
                "task_id": task.task_id,
                "success": False,
                "validity": "valid",
                "progress": 0.0,
                "parse_valid": True,
                "expected_action_ok": False,
            },
            {
                "task_id": task.task_id,
                "success": None,
                "validity": "infra_invalid",
                "progress": 0.0,
                "parse_valid": False,
                "expected_action_ok": False,
            },
        ]
        aggregate = micro.aggregate_results([task], attempts)
        for row in (aggregate["per_task"][task.task_id], aggregate["overall"]):
            self.assertEqual(row["n_attempts_raw"], 3)
            self.assertEqual(row["n_attempts_valid"], 2)
            self.assertEqual(row["n_infrastructure_failures"], 1)
            self.assertEqual(row["pass_at_1_raw"], 1 / 3)
            self.assertEqual(row["pass_at_1_valid"], 1 / 2)
        self.assertEqual(aggregate["overall"]["pass_at_1"], 1 / 3)
        self.assertEqual(aggregate["primary"], "overall/pass_at_1")

    def test_empty_calculator_clipboard_is_a_failed_verifier_not_an_exception(self) -> None:
        class EmptyClipboardClient:
            def execute(self, _command: str) -> None:
                pass

            def run_command(self, command: list[str]) -> dict[str, str]:
                if "except tk.TclError" not in command[-1]:
                    raise RuntimeError("_tkinter.TclError: CLIPBOARD selection doesn't exist")
                return {"output": "\n"}

        with mock.patch.object(micro, "_wait_until", side_effect=TimeoutError):
            passed, state = micro.verifier_passed(
                EmptyClipboardClient(),
                {"kind": "calculator_clipboard_equals", "value": "7"},
            )
        self.assertFalse(passed)
        self.assertEqual(state, "")

    def test_active_title_read_failure_is_not_a_negative_verifier(self) -> None:
        client = mock.Mock()
        client.run_command.side_effect = RuntimeError("guest command failed")
        with (
            mock.patch.object(micro, "_wait_until", side_effect=TimeoutError),
            self.assertRaisesRegex(RuntimeError, "guest command failed"),
        ):
            micro.verifier_passed(
                client,
                {"kind": "active_title_regex", "pattern": "target"},
            )

    def test_attempt_exception_is_infrastructure_invalid_not_a_model_failure(self) -> None:
        task = micro.Task(
            task_id="click.test",
            category="click",
            instruction="click",
            setup={},
            target={},
            cursor={},
            expected={},
            verifier={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            ctx = micro._RunContext(
                tasks=[task],
                suite_raw={"suite": "test"},
                output_dir=output_dir,
                sglang_url="http://model.invalid/v1",
                args=Namespace(
                    attempts=1,
                    seed_base=1,
                    qemu_bin="qemu",
                    qcow2="vm.qcow2",
                    sglang_api_key="key",
                    model_path="model",
                    no_frames=True,
                    settle_s=0.0,
                    n_history_frames=None,
                    system_prompt_id="test",
                ),
                action_format=micro._REL_STEP_FORMAT,
                system_prompt="prompt",
                sampling=qwen_sampling("thinking"),
                model_resolution=None,
                total=1,
                started=0.0,
                state_lock=threading.Lock(),
                attempts=[],
                runs=[],
            )
            with (
                mock.patch.object(micro, "_launch_vm", return_value=mock.Mock()),
                mock.patch.object(
                    micro,
                    "_assert_qemu_alive",
                    side_effect=ConnectionError("guest reset"),
                ),
                mock.patch.object(micro._LOGGER, "exception"),
                mock.patch.object(micro, "_terminate"),
            ):
                micro._run_one_task_attempt(
                    ctx,
                    task_index=0,
                    task=task,
                    attempt_index=0,
                    vm_port=5000,
                    vnc_port=5900,
                )
            result = json.loads(next(output_dir.glob("task_*/result.json")).read_text())
            summary = json.loads((output_dir / "result.json").read_text())
        self.assertEqual(result["validity"], "infra_invalid")
        self.assertEqual(result["schema_version"], 2)
        self.assertIsNone(result["success"])
        self.assertEqual(result["infra_error"]["type"], "ConnectionError")
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["overall"]["n_attempts_raw"], 1)
        self.assertEqual(summary["overall"]["n_attempts_valid"], 0)
        self.assertIsNone(summary["overall"]["pass_at_1_valid"])

    def test_guest_dispatch_failure_escapes_the_model_outcome_path(self) -> None:
        class BrokenGuestClient:
            def execute(self, _command: str) -> None:
                pass

            def screen_size(self) -> tuple[int, int]:
                return (1000, 1000)

            def cursor_position(self) -> tuple[int, int]:
                return (500, 500)

            def screenshot_settled(self, **_kwargs: object) -> Image.Image:
                return Image.new("RGB", (1000, 1000))

            def dispatch_ordered_action(self, _action: object) -> None:
                raise RuntimeError("guest transport failed")

            def release_all_inputs(self) -> None:
                pass

        task = micro.Task(
            task_id="click.test",
            category="click",
            instruction="click",
            setup={},
            target={"kind": "fixed_norm", "bbox": [400, 400, 600, 600]},
            cursor={"kind": "target_center"},
            expected={"kind": "click", "button": "left"},
            verifier={"kind": "bbox_hit"},
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(micro, "prepare_task", return_value={}),
            mock.patch.object(
                micro,
                "_call_model",
                return_value=(_tool({"action": "left_click"}), "stop"),
            ),
            self.assertRaisesRegex(RuntimeError, "guest transport failed"),
        ):
            micro.run_attempt(
                client=BrokenGuestClient(),
                task=task,
                output_dir=Path(tmp),
                sglang_url="http://model.invalid/v1",
                api_key="key",
                model="model",
                system_prompt="prompt",
                action_format=micro._REL_STEP_FORMAT,
                sampling=qwen_sampling("thinking"),
                seed=1,
                model_resolution=None,
                save_frames=False,
                settle_s=0.0,
            )

    def test_finalize_multiturn_result_prefix_mode_requires_full_ordered_run(self) -> None:
        turns = tuple(
            micro.Turn(turn_id=f"t{i}", target={}, cursor={}, expected={}, verifier={})
            for i in range(3)
        )
        # Stops after turn 2's failure (as run_multiturn_attempt's break does):
        # only 2 of 3 turns were ever attempted, and the first failed.
        results = [{"success": False}, {"success": False}]
        verified_prefix, completed, success, progress = micro._finalize_multiturn_result(
            results, turns, "prefix"
        )
        self.assertEqual(verified_prefix, 0)
        self.assertFalse(completed)
        self.assertFalse(success)
        self.assertEqual(progress, 0.0)

        full_run = [{"success": True}, {"success": True}, {"success": True}]
        verified_prefix, completed, success, progress = micro._finalize_multiturn_result(
            full_run, turns, "prefix"
        )
        self.assertEqual(verified_prefix, 3)
        self.assertTrue(completed)
        self.assertTrue(success)
        self.assertEqual(progress, 1.0)

    def test_finalize_multiturn_result_multiturn_mode_succeeds_on_any_attempt(self) -> None:
        turns = tuple(
            micro.Turn(turn_id=f"t{i}", target={}, cursor={}, expected={}, verifier={})
            for i in range(5)
        )
        # Failed twice, then succeeded on the third try -- the loop would
        # have broken right there, so only 3 attempts were ever recorded.
        results = [{"success": False}, {"success": False}, {"success": True}]
        verified_prefix, completed, success, progress = micro._finalize_multiturn_result(
            results, turns, "multiturn"
        )
        self.assertTrue(success)
        self.assertTrue(completed)
        self.assertEqual(progress, 1.0)
        self.assertEqual(verified_prefix, 3)

        exhausted = [{"success": False} for _ in range(5)]
        verified_prefix, completed, success, progress = micro._finalize_multiturn_result(
            exhausted, turns, "multiturn"
        )
        self.assertFalse(success)
        self.assertTrue(completed)  # budget exhausted counts as "completed"
        self.assertEqual(progress, 0.0)
        self.assertEqual(verified_prefix, 0)

    def test_multiturn_aggregation_scores_prefix_and_all_four(self) -> None:
        turns = tuple(
            micro.Turn(
                turn_id=f"turn_{index}",
                target={},
                cursor={},
                expected={},
                verifier={},
            )
            for index in range(3)
        )
        task = micro.Task(
            task_id="multi.test",
            category="multi_turn",
            instruction="test",
            setup={},
            target={},
            cursor={},
            expected={},
            verifier={},
            turns=turns,
        )
        attempts = [
            {
                "task_id": task.task_id,
                "validity": "valid",
                "multi_turn": True,
                "turns_total": 3,
                "turns_attempted": attempted,
                "verified_prefix": prefix,
                "turns": [
                    {
                        "parse_valid": True,
                        "expected_action_ok": True,
                    }
                    for _ in range(attempted)
                ],
                "success": success,
                "progress": prefix / 3,
                "parse_valid": success,
                "expected_action_ok": success,
            }
            for attempted, prefix, success in (
                (2, 1, False),
                (3, 3, True),
                (1, 0, False),
                (3, 2, False),
            )
        ]
        row = micro.aggregate_results([task], attempts)["per_task"][task.task_id]
        self.assertTrue(row["pass_at_4"])
        self.assertFalse(row["all_4_success"])
        self.assertEqual(row["best_of_4_progress"], 1.0)
        self.assertEqual(row["verified_turn_rate"], 0.5)
        self.assertEqual(row["turn_completion_rate"], 0.75)

    @mock.patch("cua_micro_eval.requests.post")
    def test_model_seed_is_forwarded_to_sglang(self, post: mock.Mock) -> None:
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        post.return_value.raise_for_status.return_value = None
        _call_model(
            sglang_url="http://localhost:30000/v1",
            api_key="key",
            model="model",
            system_prompt="<think>",
            instruction="GOAL: test",
            recent_frames=[Image.new("RGB", (4, 4))],
            fresh_visual_context=True,
            sampling=qwen_sampling("thinking"),
            seed=1234,
        )
        self.assertEqual(post.call_args.kwargs["json"]["seed"], 1234)


class NativeOrderedFormatTests(unittest.TestCase):
    """cua_ordered_typing_v1 (this branch's own ordered_events_v3) support."""

    def test_native_ordered_parses_and_converts_move_and_click(self) -> None:
        native = parse_ordered_action_tolerant("move(50,30); down(LMB); up(LMB)")
        converted = micro.native_ordered_to_relstep(native)
        self.assertEqual(len(converted.primitives), 3)
        move, down, up = converted.primitives
        self.assertEqual((move.kind, move.dx, move.dy), ("move", 50, 30))
        # input_name -> name is a rename, not a reinterpretation.
        self.assertEqual((down.kind, down.name, down.mouse_button), ("down", "LMB", 1))
        self.assertEqual((up.kind, up.name, up.mouse_button), ("up", "LMB", 1))

    def test_native_ordered_converts_typing_and_no_op(self) -> None:
        native = parse_ordered_action_tolerant('type("hi")')
        converted = micro.native_ordered_to_relstep(native)
        self.assertEqual(converted.primitives[0].kind, "type")
        self.assertEqual(converted.primitives[0].text, "hi")
        self.assertFalse(converted.no_op)

        no_op_native = parse_ordered_action_tolerant("NO_OP")
        self.assertTrue(micro.native_ordered_to_relstep(no_op_native).no_op)

    def test_denormalize_native_ordered_scales_move_by_model_resolution(self) -> None:
        native = parse_ordered_action_tolerant("move(128,-64); down(LMB); up(LMB)")
        converted = micro.native_ordered_to_relstep(native)
        scaled = micro.denormalize_native_ordered_action(
            converted, screen=(1920, 1080), model_resolution=(1280, 720)
        )
        move, down, up = scaled.primitives
        # 1920/1280 = 1.5, 1080/720 = 1.5 -- move scales, down/up (no dx/dy) don't change.
        self.assertEqual((move.dx, move.dy), (192, -96))
        self.assertEqual((down.name, up.name), ("LMB", "LMB"))

    def test_denormalize_native_ordered_is_noop_without_model_resolution(self) -> None:
        native = parse_ordered_action_tolerant("move(50,30)")
        converted = micro.native_ordered_to_relstep(native)
        scaled = micro.denormalize_native_ordered_action(
            converted, screen=(1920, 1080), model_resolution=None
        )
        self.assertEqual((scaled.primitives[0].dx, scaled.primitives[0].dy), (50, 30))

    def test_native_ordered_format_is_a_selectable_action_format(self) -> None:
        self.assertEqual(
            micro._PROMPT_FORMATS["cua_ordered_typing_v1"], micro._NATIVE_ORDERED_FORMAT
        )
        self.assertIn("cua_ordered_typing_v1", SYSTEM_PROMPTS)

    def test_native_ordered_move_then_click_matches_expected_click(self) -> None:
        # The exact shape ckpt_35k actually emitted for click.desktop.files:
        # a small position adjustment folded into the same atomic response.
        converted = micro.native_ordered_to_relstep(
            parse_ordered_action_tolerant("move(-1,-2); down(LMB); up(LMB)")
        )
        self.assertTrue(
            micro.action_matches_expected(
                converted, {"kind": "click", "button": "left"}, micro._NATIVE_ORDERED_FORMAT
            )
        )

    def test_native_ordered_click_without_leading_move_matches(self) -> None:
        converted = micro.native_ordered_to_relstep(
            parse_ordered_action_tolerant("down(RMB); up(RMB)")
        )
        self.assertTrue(
            micro.action_matches_expected(
                converted, {"kind": "click", "button": "right"}, micro._NATIVE_ORDERED_FORMAT
            )
        )

    def test_native_ordered_key_chord_matches_expected_key(self) -> None:
        converted = micro.native_ordered_to_relstep(
            parse_ordered_action_tolerant("down(ControlLeft); down(KeyS); up(KeyS); up(ControlLeft)")
        )
        self.assertTrue(
            micro.action_matches_expected(
                converted,
                {"kind": "key", "keys": ["ControlLeft", "KeyS"]},
                micro._NATIVE_ORDERED_FORMAT,
            )
        )

    def test_native_ordered_single_key_press_matches_expected_key(self) -> None:
        # key.calculator.digit7's actual expected shape: one key, not a chord.
        converted = micro.native_ordered_to_relstep(
            parse_ordered_action_tolerant("down(7); up(7)")
        )
        self.assertTrue(
            micro.action_matches_expected(
                converted, {"kind": "key", "keys": ["7"]}, micro._NATIVE_ORDERED_FORMAT
            )
        )

    def test_native_ordered_drag_does_not_canonicalize_to_click(self) -> None:
        # down; move; up is a drag, not a click -- must NOT collapse to one.
        converted = micro.native_ordered_to_relstep(
            parse_ordered_action_tolerant("down(LMB); move(50,0); up(LMB)")
        )
        self.assertFalse(
            micro.action_matches_expected(
                converted, {"kind": "click", "button": "left"}, micro._NATIVE_ORDERED_FORMAT
            )
        )

    def test_move_then_click_stays_a_hard_mismatch_for_other_formats(self) -> None:
        # Regression guard: computer_use_rel_step_v1 / qwen3vl_native_cua_v1 have
        # a dedicated atomic click primitive, so two separate tool calls
        # (move, then click) is a genuine "should have been one call"
        # contract violation there -- unlike cua_ordered_typing_v1, it must
        # NOT canonicalize away. Regressed once already while generalizing
        # _canonicalize_native_ordered_action; caught by the existing
        # test_qwen3vl_native_multiple_calls_are_parse_valid_but_not_atomic.
        calls = parse_qwen3vl_computer_use_action(
            _tool({"action": "mouse_move", "coordinate": [500, 500]})
            + _tool({"action": "left_click"})
        )
        parsed = micro.qwen3vl_native_to_ordered(calls, (1000, 1000), (0, 0))
        self.assertFalse(
            micro.action_matches_expected(
                parsed, {"kind": "click", "button": "left"}, micro._QWEN3VL_NATIVE_FORMAT
            )
        )


class ResultPersistenceTests(unittest.TestCase):
    def _run_validation_main(self, success: bool) -> tuple[int, dict, bool, dict]:
        task = micro.Task(
            task_id="click.test",
            category="click",
            instruction="click",
            setup={},
            target={},
            cursor={},
            expected={},
            verifier={},
        )
        validation = {
            "schema_version": 1,
            "mode": "validate_setups_only",
            "task_id": task.task_id,
            "category": task.category,
            "success": success,
        }

        def validate(**kwargs: object) -> dict:
            output_dir = kwargs["output_dir"]
            assert isinstance(output_dir, Path)
            (output_dir / "result.json").write_text(json.dumps(validation))
            return validation

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            argv = [
                "cua_micro_eval.py",
                "--output_dir",
                str(output_dir),
                "--validate_setups_only",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(micro, "load_suite", return_value=({"suite": "test"}, [task])),
                mock.patch.object(micro, "_preflight_ports"),
                mock.patch.object(micro, "_wait_ports_free"),
                mock.patch.object(micro, "_launch_vm", return_value=mock.Mock()),
                mock.patch.object(micro, "_assert_qemu_alive"),
                mock.patch.object(micro, "OSWorldClient"),
                mock.patch.object(micro, "validate_task_setup", side_effect=validate),
                mock.patch.object(micro, "_terminate"),
            ):
                return_code = micro.main()
            result = json.loads((output_dir / "result.json").read_text())
            marker_exists = (output_dir / "completed.json").exists()
            validation_result = json.loads(
                (output_dir / "tasks" / "click.test" / "validation" / "result.json").read_text()
            )
        return return_code, result, marker_exists, validation_result

    def test_successful_setup_validation_publishes_completion(self) -> None:
        return_code, result, marker_exists, validation = self._run_validation_main(True)
        self.assertEqual(return_code, 0)
        self.assertTrue(result["success"])
        self.assertTrue(marker_exists)
        self.assertTrue(validation["success"])

    def test_failed_setup_validation_preserves_results_without_completion(self) -> None:
        return_code, result, marker_exists, validation = self._run_validation_main(False)
        self.assertEqual(return_code, 1)
        self.assertFalse(result["success"])
        self.assertFalse(marker_exists)
        self.assertFalse(validation["success"])

    def _run_main(self, attempt: dict) -> tuple[int, dict, bool, dict]:
        task = micro.Task(
            task_id="click.test",
            category="click",
            instruction="click",
            setup={},
            target={},
            cursor={},
            expected={},
            verifier={},
        )

        def run_slot(
            _slot_id: int,
            _assigned: list[tuple[int, micro.Task, int]],
            ctx: micro._RunContext,
            **_ports: int,
        ) -> None:
            attempt_dir = ctx.output_dir / "task_000_click.test"
            attempt_dir.mkdir()
            (attempt_dir / "result.json").write_text(json.dumps(attempt))
            ctx.attempts.append(attempt)
            ctx.runs.append(
                {
                    "index": 0,
                    "slug": attempt_dir.name,
                    "task_id": task.task_id,
                    "subdir": attempt_dir.name,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            argv = [
                "cua_micro_eval.py",
                "--model_path",
                "model",
                "--output_dir",
                str(output_dir),
                "--attempts",
                "1",
                "--vms_per_sglang",
                "1",
                "--sglang_url",
                "http://model.invalid/v1",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(micro, "load_suite", return_value=({"suite": "test"}, [task])),
                mock.patch.object(micro, "_preflight_ports"),
                mock.patch.object(micro.signal, "signal"),
                mock.patch.object(micro.sampling_mod, "from_cli", return_value=qwen_sampling("thinking")),
                mock.patch.object(micro, "_run_vm_slot", side_effect=run_slot),
            ):
                return_code = micro.main()
            result = json.loads((output_dir / "result.json").read_text())
            marker_exists = (output_dir / "completed.json").exists()
            attempt_result = json.loads(
                (output_dir / "task_000_click.test" / "result.json").read_text()
            )
        return return_code, result, marker_exists, attempt_result

    def test_all_valid_attempts_publish_completion_and_exit_zero(self) -> None:
        attempt = {
            "task_id": "click.test",
            "validity": "valid",
            "success": True,
            "progress": 1.0,
            "parse_valid": True,
            "expected_action_ok": True,
        }
        return_code, result, marker_exists, _ = self._run_main(attempt)
        self.assertEqual(return_code, 0)
        self.assertTrue(marker_exists)
        self.assertEqual(result["overall"]["n_attempts_raw"], 1)
        self.assertEqual(result["overall"]["n_attempts_valid"], 1)

    def test_valid_model_failure_still_completes_and_exits_zero(self) -> None:
        attempt = {
            "task_id": "click.test",
            "validity": "valid",
            "success": False,
            "progress": 0.0,
            "parse_valid": True,
            "expected_action_ok": False,
        }
        return_code, result, marker_exists, _ = self._run_main(attempt)
        self.assertEqual(return_code, 0)
        self.assertTrue(marker_exists)
        self.assertEqual(result["overall"]["pass_at_1_raw"], 0.0)
        self.assertEqual(result["overall"]["pass_at_1_valid"], 0.0)

    def test_infrastructure_invalid_preserves_results_but_does_not_complete(self) -> None:
        attempt = {
            "task_id": "click.test",
            "validity": "infra_invalid",
            "infra_error": {"type": "ConnectionError", "message": "guest reset"},
            "success": None,
            "progress": 0.0,
            "parse_valid": False,
            "expected_action_ok": False,
        }
        return_code, result, marker_exists, attempt_result = self._run_main(attempt)
        self.assertEqual(return_code, 1)
        self.assertFalse(marker_exists)
        self.assertEqual(result["overall"]["n_attempts_raw"], 1)
        self.assertEqual(result["overall"]["n_attempts_valid"], 0)
        self.assertEqual(result["overall"]["pass_at_1_raw"], 0.0)
        self.assertIsNone(result["overall"]["pass_at_1_valid"])
        self.assertEqual(attempt_result, attempt)

    def test_suite_round_trips_from_an_arbitrary_path(self) -> None:
        raw = json.loads(_SUITE.read_text())
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "suite.json"
            copied.write_text(json.dumps(raw))
            loaded, tasks = micro.load_suite(copied)
        self.assertEqual(loaded["suite"], "cua_micro_tasks")
        self.assertEqual(len(tasks), len(raw["tasks"]))


class SchedulerContractTests(unittest.TestCase):
    @staticmethod
    def _task(task_id: str, *, turns: int = 0) -> micro.Task:
        turn_rows = tuple(
            micro.Turn(
                turn_id=f"turn_{index}",
                target={},
                cursor={},
                expected={},
                verifier={},
            )
            for index in range(turns)
        )
        return micro.Task(
            task_id=task_id,
            category="test",
            instruction=task_id,
            setup={"kind": "desktop"},
            target={},
            cursor={},
            expected={},
            verifier={},
            turns=turn_rows,
        )

    def test_free_slot_takes_next_index_instead_of_static_round_robin(self) -> None:
        tasks = [self._task(f"task.{index}") for index in range(5)]
        work: queue.Queue[tuple[int, micro.Task, int]] = queue.Queue()
        for task_index, task in enumerate(tasks):
            work.put((task_index, task, 0))

        slow_started = threading.Event()
        fast_drained = threading.Event()
        assignments: dict[int, list[int]] = {5000: [], 5001: []}

        def run_one(
            _ctx: object,
            *,
            task_index: int,
            task: micro.Task,
            attempt_index: int,
            vm_port: int,
            vnc_port: int,
        ) -> None:
            del task, attempt_index, vnc_port
            assignments[vm_port].append(task_index)
            if task_index == 0:
                slow_started.set()
                self.assertTrue(fast_drained.wait(timeout=2))
            elif task_index == 4:
                fast_drained.set()

        with mock.patch.object(micro, "_run_one_task_attempt_isolated", side_effect=run_one):
            slow = threading.Thread(
                target=micro._run_vm_slot,
                args=(0, work, mock.Mock()),
                kwargs={"vm_port": 5000, "vnc_port": 5900},
            )
            fast = threading.Thread(
                target=micro._run_vm_slot,
                args=(1, work, mock.Mock()),
                kwargs={"vm_port": 5001, "vnc_port": 5901},
            )
            slow.start()
            self.assertTrue(slow_started.wait(timeout=2))
            fast.start()
            slow.join(timeout=2)
            fast.join(timeout=2)

        self.assertFalse(slow.is_alive())
        self.assertFalse(fast.is_alive())
        self.assertEqual(assignments[5000], [0])
        self.assertEqual(assignments[5001], [1, 2, 3, 4])

    def test_attempt_indices_and_seeds_do_not_depend_on_slot(self) -> None:
        self.assertEqual(
            micro._attempt_identity(
                task_index=2,
                attempt_index=3,
                attempts_per_task=4,
                seed_base=41000,
            ),
            (11, 41203),
        )

    def test_attempt_deadline_kills_process_and_records_typed_invalid_result(self) -> None:
        task = self._task("task.timeout", turns=64)
        process = mock.Mock(pid=81234, exitcode=None)
        process.is_alive.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            attempt_dir = output_dir / "task_000_task.timeout"
            attempt_dir.mkdir()
            (attempt_dir / "prompt_034.json").write_text("partial")
            ctx = micro._RunContext(
                tasks=[task],
                suite_raw={"suite": "test"},
                output_dir=output_dir,
                sglang_url="http://model.invalid/v1",
                args=Namespace(
                    attempts=1,
                    seed_base=41000,
                    qemu_bin="qemu",
                    qcow2="vm.qcow2",
                    sglang_api_key="key",
                    model_path="model",
                    no_frames=True,
                    settle_s=0.0,
                    n_history_frames=None,
                    system_prompt_id="test",
                ),
                action_format=micro._REL_STEP_FORMAT,
                system_prompt="prompt",
                sampling=qwen_sampling("thinking"),
                model_resolution=None,
                total=1,
                started=0.0,
                state_lock=threading.Lock(),
                attempts=[],
                runs=[],
            )
            with (
                mock.patch.object(micro, "_spawn_attempt_process", return_value=process),
                mock.patch.object(micro, "_terminate_attempt_process_group") as terminate,
                mock.patch.object(micro, "_attempt_wall_bound_s", return_value=123.0),
            ):
                micro._run_one_task_attempt_isolated(
                    ctx,
                    task_index=0,
                    task=task,
                    attempt_index=0,
                    vm_port=5000,
                    vnc_port=5900,
                )

            result = json.loads((attempt_dir / "result.json").read_text())
            summary = json.loads((output_dir / "result.json").read_text())
            partial_prompt = (attempt_dir / "prompt_034.json").read_text()

        process.join.assert_called_once_with(timeout=123.0)
        terminate.assert_called_once_with(process)
        self.assertEqual(result["validity"], "infra_invalid")
        self.assertEqual(result["infra_error"]["type"], "AttemptWallTimeout")
        self.assertEqual(result["attempt_wall_bound_s"], 123.0)
        self.assertEqual(summary["overall"]["n_attempts_valid"], 0)
        self.assertEqual(partial_prompt, "partial")

    def test_slurm_allocation_shorter_than_suite_bound_is_rejected(self) -> None:
        tasks = [self._task("task.atomic")]
        required = micro._suite_wall_bound_s(
            tasks,
            attempts=1,
            n_vms=1,
            local_sglang=True,
        )
        with (
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "142074"}),
            mock.patch.object(micro, "_slurm_remaining_wall_s", return_value=required - 1),
            self.assertRaisesRegex(RuntimeError, "cannot cover declared suite"),
        ):
            micro._preflight_slurm_wall_budget(
                tasks,
                attempts=1,
                n_vms=1,
                local_sglang=True,
            )

    def test_timeout_cleanup_kills_the_attempt_process_group(self) -> None:
        child = subprocess.Popen(
            ["bash", "-c", "trap '' TERM; echo ready; sleep 60 & wait"],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert child.stdout is not None
        self.assertEqual(child.stdout.readline().strip(), "ready")

        class ProcessHandle:
            pid = child.pid

            @staticmethod
            def join(timeout: float | None = None) -> None:
                with suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=timeout)

            @staticmethod
            def is_alive() -> bool:
                return child.poll() is None

        try:
            with mock.patch.object(micro, "_QEMU_SHUTDOWN_TIMEOUT_S", 0.05):
                micro._terminate_attempt_process_group(ProcessHandle())
        finally:
            if child.poll() is None:
                os.killpg(child.pid, 9)
                child.wait()
            child.stdout.close()
        self.assertIsNotNone(child.returncode)

    def test_slurm_duration_parser_accepts_scheduler_formats(self) -> None:
        self.assertEqual(micro._parse_slurm_duration_s("03:00:00"), 10_800)
        self.assertEqual(micro._parse_slurm_duration_s("1-02:03:04"), 93_784)
        self.assertEqual(micro._parse_slurm_duration_s("05:30"), 330)


if __name__ == "__main__":
    unittest.main()
