import io
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from cuagym_reward import parse_reward
from cuagym_rollout_runner import (
    predict_with_noop_resample,
    screen_hash,
    should_abort_repetition,
    should_abort_stale_screen,
)
from cuagym_task_adapter import (
    NOOP_EVALUATOR,
    UnsupportedTaskError,
    cache_relpath,
    load_task,
    seed_cache,
)
from cuagym_traj_export import export_episode
from oev3_agent import Oev3Agent

TASK_ID = "11111111-2222-5333-8444-555555555555"


def _png_bytes(color=(0, 0, 0)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


class RewardParseTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(parse_reward("checks...\nREWARD: 0.75\n"), 0.75)

    def test_last_line_wins(self):
        self.assertEqual(parse_reward("REWARD: 0.0\nmore\nREWARD: 1.0"), 1.0)

    def test_scientific_and_negative(self):
        self.assertEqual(parse_reward("REWARD: 2.5e-1"), 0.25)
        self.assertEqual(parse_reward("REWARD: -1"), -1.0)

    def test_inline_prefix(self):
        self.assertEqual(parse_reward("FINAL REWARD: 0.30 (partial)"), 0.30)

    def test_missing(self):
        self.assertIsNone(parse_reward("no reward here"))
        self.assertIsNone(parse_reward(""))
        self.assertIsNone(parse_reward(None))


class AbortGuardTests(unittest.TestCase):
    def test_repetition_abort_fires(self):
        lines = ["a", "x", "x", "x", "x"]
        hashes = ["h0", "h1", "h2", "h2", "h2", "h2"]
        self.assertTrue(should_abort_repetition(lines, hashes))

    def test_repetition_needs_identical_screens(self):
        lines = ["x", "x", "x", "x"]
        hashes = ["h0", "h1", "h2", "h3", "h4"]
        self.assertFalse(should_abort_repetition(lines, hashes))

    def test_repetition_needs_four_actions(self):
        lines = ["x", "x", "x"]
        hashes = ["h", "h", "h", "h"]
        self.assertFalse(should_abort_repetition(lines, hashes))

    def test_repetition_none_line_breaks_streak(self):
        lines = ["x", None, "x", "x"]
        hashes = ["h", "h", "h", "h", "h"]
        self.assertFalse(should_abort_repetition(lines, hashes))

    def test_stale_screen_abort_fires(self):
        hashes = ["h0"] + ["h"] * 8
        self.assertTrue(should_abort_stale_screen(hashes))

    def test_stale_screen_needs_eight(self):
        hashes = ["h"] * 7
        self.assertFalse(should_abort_stale_screen(hashes))

    def test_stale_screen_none_never_counts(self):
        hashes = [None] * 8
        self.assertFalse(should_abort_stale_screen(hashes))

    def test_screen_hash(self):
        a = _png_bytes()
        self.assertEqual(screen_hash(a), screen_hash(bytes(a)))
        self.assertNotEqual(screen_hash(_png_bytes((1, 2, 3))), screen_hash(a))
        self.assertIsNone(screen_hash(None))
        self.assertIsNone(screen_hash(b""))


PREV_LINE = "move(1,2); down(LMB); up(LMB)"


class NoopResampleTests(unittest.TestCase):
    def _agent(self, replies, temperature=0.8):
        a = Oev3Agent(model="m", history_n=4, temperature=temperature)
        a.reset()
        a.screenshots.append("S0")
        a.stripped_responses.append(f"resp0\n{PREV_LINE}")
        a.action_lines.append(PREV_LINE)
        self.temps = []

        def fake_call(messages, _replies=replies):
            self.temps.append(a.temperature)
            return _replies.pop(0)

        a._call_llm = fake_call
        return a

    def _obs(self):
        return {"screenshot": _png_bytes()}

    def test_resample_replaces_repeat(self):
        a = self._agent(
            [
                (f"t</think>\n{PREV_LINE}", "stop"),
                ("t</think>\nmove(3,4)", "stop"),
            ]
        )
        response, actions, n = predict_with_noop_resample(
            a, "goal", self._obs(), PREV_LINE, True
        )
        self.assertEqual(n, 1)
        self.assertEqual(a.action_lines, [PREV_LINE, "move(3,4)"])
        self.assertEqual(len(actions), 1)
        self.assertIn("move(3,4)", response)
        self.assertEqual(a.temperature, 0.8)
        self.assertAlmostEqual(self.temps[1], 1.0)

    def test_resample_exhaustion_accepts_repeat(self):
        a = self._agent([(f"t</think>\n{PREV_LINE}", "stop")] * 3)
        response, actions, n = predict_with_noop_resample(
            a, "goal", self._obs(), PREV_LINE, True
        )
        self.assertEqual(n, 2)
        self.assertEqual(a.action_lines, [PREV_LINE, PREV_LINE])
        self.assertTrue(actions)
        self.assertEqual(a.temperature, 0.8)
        self.assertAlmostEqual(self.temps[1], 1.0)
        self.assertAlmostEqual(self.temps[2], 1.2)

    def test_resample_parse_failure_falls_back(self):
        a = self._agent(
            [
                (f"t</think>\n{PREV_LINE}", "stop"),
                ("t</think>\nnot_a_primitive(1)", "stop"),
            ]
        )
        response, actions, n = predict_with_noop_resample(
            a, "goal", self._obs(), PREV_LINE, True
        )
        self.assertEqual(n, 1)
        self.assertEqual(a.action_lines, [PREV_LINE, PREV_LINE])
        self.assertTrue(actions)
        self.assertIn(PREV_LINE, response)
        self.assertEqual(a.temperature, 0.8)

    def test_no_resample_when_screen_changed(self):
        a = self._agent([(f"t</think>\n{PREV_LINE}", "stop")])
        response, actions, n = predict_with_noop_resample(
            a, "goal", self._obs(), PREV_LINE, False
        )
        self.assertEqual(n, 0)
        self.assertEqual(a.action_lines, [PREV_LINE, PREV_LINE])

    def test_no_resample_when_line_differs(self):
        a = self._agent([("t</think>\nmove(9,9)", "stop")])
        response, actions, n = predict_with_noop_resample(
            a, "goal", self._obs(), PREV_LINE, True
        )
        self.assertEqual(n, 0)
        self.assertEqual(a.action_lines[-1], "move(9,9)")

    def test_no_resample_without_prev_line(self):
        a = self._agent([(f"t</think>\n{PREV_LINE}", "stop")])
        response, actions, n = predict_with_noop_resample(
            a, "goal", self._obs(), None, True
        )
        self.assertEqual(n, 0)

    def test_parse_failure_first_draw_passes_through(self):
        a = self._agent([("t</think>\nnot_a_primitive(1)", "stop")])
        response, actions, n = predict_with_noop_resample(
            a, "goal", self._obs(), PREV_LINE, True
        )
        self.assertEqual(n, 0)
        self.assertEqual(actions, [])


def _write_bundle(root: Path, task_id: str, evaluator=None) -> Path:
    bundle = root / task_id
    bundle.mkdir(parents=True)
    (bundle / "initial_setup.py").write_text("print('setup')\n")
    (bundle / "reward.py").write_text("print('REWARD: 0.0')\n")
    task = {
        "evaluator": evaluator or {"type": "python", "url": "./reward.py"},
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {"url": "./initial_setup.py", "path": "/home/user/initial_setup.py"}
                    ]
                },
            },
            {
                "type": "execute",
                "parameters": {"command": "python3 /home/user/initial_setup.py"},
            },
        ],
        "id": task_id,
        "difficulty": "hard",
        "instruction": "do the thing",
        "app_type": "vscode",
    }
    (bundle / "task.json").write_text(json.dumps(task))
    return bundle


class AdapterTests(unittest.TestCase):
    def test_adapts_config_verbatim_and_stubs_evaluator(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(root, TASK_ID)
            adapted = load_task(TASK_ID, "desktop", tasks_root=root)
            original = json.loads((root / TASK_ID / "task.json").read_text())
            self.assertEqual(adapted.task_config["config"], original["config"])
            self.assertEqual(adapted.task_config["id"], TASK_ID)
            self.assertEqual(adapted.task_config["instruction"], "do the thing")
            stub = adapted.task_config["evaluator"]
            self.assertEqual(stub["func"], "exact_match")
            self.assertEqual(stub["result"]["type"], "rule")
            self.assertEqual(stub["expected"]["type"], "rule")
            self.assertNotEqual(stub["result"]["rules"], stub["expected"]["rules"]["expected"])
            self.assertNotIn("postconfig", stub)
            self.assertEqual(adapted.reward_script, root / TASK_ID / "reward.py")

    def test_cache_seed_matches_pinned_download_cache_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(root, TASK_ID)
            adapted = load_task(TASK_ID, "desktop_office", tasks_root=root)
            self.assertEqual(len(adapted.cache_seeds), 1)
            seed = adapted.cache_seeds[0]
            url = "./initial_setup.py"
            expected = "{}/{}_{}".format(
                TASK_ID, uuid.uuid5(uuid.NAMESPACE_URL, url), "initial_setup.py"
            )
            self.assertEqual(seed.cache_relpath, expected)
            self.assertEqual(cache_relpath(TASK_ID, url, "/home/user/initial_setup.py"), expected)

    def test_seed_cache_copies_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(root, TASK_ID)
            adapted = load_task(TASK_ID, "multi_apps", tasks_root=root)
            cache_dir = root / "cache"
            seeded = seed_cache(adapted, cache_dir)
            self.assertEqual(len(seeded), 1)
            self.assertTrue(seeded[0].is_file())
            self.assertEqual(seeded[0].read_text(), "print('setup')\n")

    def test_postconfig_carried_over(self):
        postconfig = [
            {
                "type": "execute",
                "parameters": {
                    "command": ["python", "-c", "import pyautogui; pyautogui.hotkey('ctrl','s')"]
                },
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(
                root,
                TASK_ID,
                evaluator={"type": "python", "url": "./reward.py", "postconfig": postconfig},
            )
            adapted = load_task(TASK_ID, "other", tasks_root=root)
            self.assertEqual(adapted.task_config["evaluator"]["postconfig"], postconfig)
            self.assertNotIn("postconfig", NOOP_EVALUATOR)

    def test_mock_web_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(root, TASK_ID)
            with self.assertRaises(UnsupportedTaskError):
                load_task(TASK_ID, "mock_web", tasks_root=root)

    def test_url_escape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = _write_bundle(root, TASK_ID)
            task = json.loads((bundle / "task.json").read_text())
            task["config"][0]["parameters"]["files"][0]["url"] = "../outside.py"
            (bundle / "task.json").write_text(json.dumps(task))
            with self.assertRaises(UnsupportedTaskError):
                load_task(TASK_ID, "desktop", tasks_root=root)


class ExportTests(unittest.TestCase):
    def _episode(self, root: Path, stop_reason: str, reward):
        episode = root / "desktop" / TASK_ID
        (episode / "steps").mkdir(parents=True)
        result = {
            "task": "cuagym_rollout",
            "scores": {"reward": reward, "n_steps_taken": 2, "stop_reason_code": 1},
            "params": {
                "task_id": TASK_ID,
                "app_family": "desktop",
                "app_type": "vscode",
                "sample_index": 2,
                "instruction": "do the thing",
                "screen_width": 1920,
                "screen_height": 1080,
                "stop_reason": stop_reason,
            },
            "inputs": {},
        }
        (episode / "result.json").write_text(json.dumps(result))
        rows = [
            {"step_num": 0, "action": "<reset>", "response": "<reset>", "reward": 0.0, "done": False, "info": {}},
            {
                "step_num": 1,
                "action": "import pyautogui\npyautogui.moveRel(2, 2)",
                "response": f"think</think>\nAction: move.\n{PREV_LINE}",
                "reward": 0,
                "done": False,
                "info": {},
            },
            {"step_num": 2, "action": "DONE", "response": "done</think>\nTERMINATE", "reward": 0, "done": True, "info": {"done": True}},
        ]
        with (episode / "traj.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return episode

    def test_export_row_shape(self):
        with tempfile.TemporaryDirectory() as td:
            episode = self._episode(Path(td), "agent_terminate", 0.75)
            row = export_episode(episode)
            self.assertEqual(row["task_id"], TASK_ID)
            self.assertEqual(row["app_family"], "desktop")
            self.assertEqual(row["app_type"], "vscode")
            self.assertEqual(row["instruction"], "do the thing")
            self.assertEqual(row["reward"], 0.75)
            self.assertTrue(row["terminated"])
            self.assertEqual(row["stop_reason"], "agent_terminate")
            self.assertEqual(row["screen"], [1920, 1080])
            self.assertEqual(row["sample_index"], 2)
            self.assertEqual(len(row["steps"]), 2)
            first, last = row["steps"]
            self.assertEqual(first["step"], 1)
            self.assertEqual(first["image_path"], "steps/step_000.png")
            self.assertEqual(first["action_line"], PREV_LINE)
            self.assertIn("</think>", first["assistant_raw"])
            self.assertIn("moveRel", first["dispatched"])
            self.assertEqual(last["image_path"], "steps/step_001.png")
            self.assertEqual(last["action_line"], "TERMINATE")
            self.assertEqual(last["dispatched"], "DONE")

    def test_export_fail_counts_as_terminated(self):
        with tempfile.TemporaryDirectory() as td:
            row = export_episode(self._episode(Path(td), "agent_fail", 0.0))
            self.assertTrue(row["terminated"])

    def test_export_abort_not_terminated(self):
        with tempfile.TemporaryDirectory() as td:
            row = export_episode(self._episode(Path(td), "repetition_abort", None))
            self.assertFalse(row["terminated"])
            self.assertIsNone(row["reward"])


if __name__ == "__main__":
    unittest.main()
