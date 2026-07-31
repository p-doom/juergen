"""Unit tests for the deterministic one-turn CUA micro-eval contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import cua_micro_eval as micro
from action_parser import (
    parse_computer_use_rel_step_action,
    parse_qwen3vl_computer_use_action,
)
from osworld_runtime import _call_model
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

    def test_suite_is_nonempty_unique_and_versioned(self) -> None:
        self.assertEqual(self.raw["schema_version"], 1)
        self.assertEqual(self.raw["coordinate_grid"], 1000)
        ids = [task.task_id for task in self.tasks]
        self.assertGreaterEqual(len(ids), 16)
        self.assertEqual(len(ids), len(set(ids)))

    def test_requested_families_are_present(self) -> None:
        ids = {task.task_id for task in self.tasks}
        required_fragments = {
            "desktop.files",
            "desktop.libreoffice_writer",
            "desktop.libreoffice_calc",
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
                    {"move", "click", "type", "scroll", "key"},
                )

    def test_typing_checks_exact_action_and_exact_app_state(self) -> None:
        typing = [task for task in self.tasks if not task.turns and task.expected["kind"] == "type"]
        self.assertGreaterEqual(len(typing), 2)
        for task in typing:
            self.assertIn(task.verifier["kind"], {"guest_json_equals", "saved_file_equals"})
            self.assertEqual(task.expected["text"], task.verifier["value"])
            parsed = parse_computer_use_rel_step_action(
                _tool({"action": "type", "text": task.expected["text"]})
            )
            self.assertTrue(micro.action_matches_expected(parsed, task.expected))

    def test_multiturn_tasks_have_semantically_verified_steps(self) -> None:
        tasks = [task for task in self.tasks if task.turns]
        self.assertEqual(len(tasks), 4)
        self.assertEqual(sum(len(task.turns) for task in tasks), 15)
        for task in tasks:
            self.assertGreaterEqual(len(task.turns), 3)
            self.assertEqual(task.category, "multi_turn")
            for turn in task.turns:
                self.assertTrue(turn.turn_id)
                self.assertIn("kind", turn.verifier)

    def test_native_suite_is_balanced_across_real_apps(self) -> None:
        raw, tasks = micro.load_suite(_SUITE)
        self.assertEqual(raw["suite"], "cua_micro_tasks")
        ids = {task.task_id for task in tasks}
        for fragment in ("files", "terminal", "text_editor", "writer", "calc", "impress"):
            self.assertTrue(any(fragment in task_id for task_id in ids), fragment)
        self.assertLessEqual(sum("chrome" in task_id for task_id in ids), len(tasks) // 3)


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

    @mock.patch("osworld_runtime.requests.post")
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


class ResultPersistenceTests(unittest.TestCase):
    def test_suite_round_trips_from_an_arbitrary_path(self) -> None:
        raw = json.loads(_SUITE.read_text())
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "suite.json"
            copied.write_text(json.dumps(raw))
            loaded, tasks = micro.load_suite(copied)
        self.assertEqual(loaded["suite"], "cua_micro_tasks")
        self.assertEqual(len(tasks), len(raw["tasks"]))


if __name__ == "__main__":
    unittest.main()
