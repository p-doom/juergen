import json
import sys
import tempfile
import unittest
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from cuagym_pipeline.onpolicy_accept import (
    classify,
    default_options,
    episode_from_run_dir,
    iter_rollout_root,
    process_rows,
)
from cuagym_pipeline.stage_04_build_conversations import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    _is_target_sampled,
)
from cuagym_pipeline.stage_04o_onpolicy_conversations import run as run_stage_04o

REAL_ROW_PATH = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu/"
    "cuagym_p2_stage_04_perstep_conversations_v2/chat.jsonl"
)


def make_step(i, line, image_path=None, **kw):
    step = {
        "step": i,
        "image_path": image_path or f"/nonexistent/steps/step_{i:03d}.png",
        "assistant_raw": f"<think>step {i} reasoning.</think>\n\n{line}",
        "action_line": line,
        "dispatched": True,
    }
    step.update(kw)
    return step


def make_episode(task_id, reward, stop_reason, lines, **kw):
    rec = {
        "task_id": task_id,
        "app_family": "chrome",
        "app_type": "chrome",
        "instruction": "Do the thing.",
        "reward": reward,
        "terminated": stop_reason == "agent_terminate",
        "stop_reason": stop_reason,
        "screen": [1920, 1080],
        "sample_index": 0,
        "steps": [make_step(i, line) for i, line in enumerate(lines)],
    }
    rec.update(kw)
    return rec


class ClassifyTest(unittest.TestCase):
    def test_false_done_rejected(self):
        opts = default_options()
        rec = make_episode("t1", 0.5, "agent_terminate", ["move(1,2); down(LMB); up(LMB)", "TERMINATE"])
        verdict = classify(rec, opts)
        self.assertFalse(verdict["accept"])
        self.assertIn("false_done", verdict["reasons"])
        self.assertEqual(verdict["stratum"], "failed")

    def test_false_done_zero_reward_terminate(self):
        verdict = classify(
            make_episode("t1", 0.0, "agent_terminate", ["TERMINATE"]), default_options()
        )
        self.assertFalse(verdict["accept"])
        self.assertIn("false_done", verdict["reasons"])

    def test_solved_terminate_accepted(self):
        verdict = classify(
            make_episode("t1", 1.0, "agent_terminate", ['type("x")', "TERMINATE"]),
            default_options(),
        )
        self.assertTrue(verdict["accept"])
        self.assertEqual(verdict["stratum"], "solved")

    def test_fail_on_infeasible_accepted(self):
        opts = default_options()
        opts["infeasible_task_ids"] = {"t_inf"}
        rec = make_episode("t_inf", 1.0, "agent_terminate", ["NO_OP", "FAIL"])
        verdict = classify(rec, opts)
        self.assertTrue(verdict["accept"])
        self.assertEqual(verdict["stratum"], "solved")

    def test_fail_on_feasible_rejected(self):
        rec = make_episode("t_feasible", 1.0, "agent_terminate", ["NO_OP", "FAIL"])
        verdict = classify(rec, default_options())
        self.assertFalse(verdict["accept"])
        self.assertIn("fail_on_feasible", verdict["reasons"])

    def test_fail_with_low_reward_rejected_even_if_infeasible(self):
        opts = default_options()
        opts["infeasible_task_ids"] = {"t_inf"}
        rec = make_episode("t_inf", 0.0, "agent_terminate", ["FAIL"])
        verdict = classify(rec, opts)
        self.assertFalse(verdict["accept"])
        self.assertIn("false_done", verdict["reasons"])

    def test_bad_stop_reasons_rejected(self):
        for sr in ("repetition_abort", "stale_screen_abort", "no_actions_parsed", "truncated_think"):
            verdict = classify(make_episode("t1", 1.0, sr, ["NO_OP"]), default_options())
            self.assertFalse(verdict["accept"])
            self.assertIn(f"stop_reason:{sr}", verdict["reasons"])

    def test_bad_stop_toggle_off(self):
        opts = default_options()
        opts["reject_bad_stop"] = False
        verdict = classify(make_episode("t1", 1.0, "repetition_abort", ["NO_OP"]), opts)
        self.assertTrue(verdict["accept"])

    def test_quarantine_null_reward(self):
        rec = make_episode("t1", None, "max_steps", ["NO_OP"])
        accepted, rejected, report = process_rows([rec], default_options())
        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["stratum"], "quarantine")
        self.assertFalse(rejected[0]["accept"])
        self.assertEqual(rejected[0]["quarantine_reasons"], ["reward_null"])
        self.assertEqual(rejected[0]["reasons"], [])
        self.assertEqual(report["quarantine_reason_histogram"], {"reward_null": 1})

    def test_quarantine_toggle_off_treats_as_zero(self):
        opts = default_options()
        opts["quarantine_null_reward"] = False
        verdict = classify(make_episode("t1", None, "max_steps", ["NO_OP"]), opts)
        self.assertFalse(verdict["accept"])
        self.assertEqual(verdict["stratum"], "failed")
        self.assertIn("zero_reward", verdict["reasons"])

    def test_partial_disabled_by_default(self):
        verdict = classify(make_episode("t1", 0.5, "max_steps", ["NO_OP"]), default_options())
        self.assertFalse(verdict["accept"])
        self.assertEqual(verdict["stratum"], "partial")
        self.assertIn("partial_disabled", verdict["reasons"])


class StepAnnotationTest(unittest.TestCase):
    def test_run_trimming(self):
        rec = make_episode("t1", 1.0, "max_steps", ["scroll(0,-2)"] + ["NO_OP"] * 5 + ['type("x")'])
        accepted, _, _ = process_rows([rec], default_options())
        steps = accepted[0]["steps"]
        excluded = [s["step"] for s in steps if s["excluded"]]
        self.assertEqual(excluded, [3, 4, 5])
        self.assertTrue(all(s["excluded_reason"] == "run_trim" for s in steps if s["excluded"]))

    def test_run_of_two_not_trimmed(self):
        rec = make_episode("t1", 1.0, "max_steps", ["NO_OP", "NO_OP", 'type("x")'])
        accepted, _, _ = process_rows([rec], default_options())
        self.assertFalse(any(s["excluded"] for s in accepted[0]["steps"]))

    def test_run_trim_hash_gated(self):
        lines = ["NO_OP"] * 5
        rec = make_episode("t1", 1.0, "max_steps", lines)
        for i, s in enumerate(rec["steps"]):
            s["image_hash"] = f"h{i % 2}"
        accepted, _, _ = process_rows([rec], default_options())
        self.assertFalse(any(s["excluded"] for s in accepted[0]["steps"]))

        rec2 = make_episode("t2", 1.0, "max_steps", lines)
        for s in rec2["steps"]:
            s["image_hash"] = "same"
        accepted2, _, _ = process_rows([rec2], default_options())
        self.assertEqual([s["step"] for s in accepted2[0]["steps"] if s["excluded"]], [2, 3, 4])

    def test_run_trim_without_hashes_toggle(self):
        opts = default_options()
        opts["trim_without_hashes"] = False
        rec = make_episode("t1", 1.0, "max_steps", ["NO_OP"] * 5)
        accepted, _, _ = process_rows([rec], opts)
        self.assertFalse(any(s["excluded"] for s in accepted[0]["steps"]))

    def test_partial_sampling_deterministic(self):
        opts = default_options()
        opts["accept_partial"] = True
        rec = make_episode("t_part", 0.4, "max_steps", [f"move({i},1)" for i in range(50)])
        accepted_a, _, _ = process_rows([json.loads(json.dumps(rec))], opts)
        accepted_b, _, _ = process_rows([json.loads(json.dumps(rec))], opts)
        self.assertEqual(accepted_a[0]["stratum"], "partial")
        excluded_a = {s["step"] for s in accepted_a[0]["steps"] if s["excluded"]}
        excluded_b = {s["step"] for s in accepted_b[0]["steps"] if s["excluded"]}
        self.assertEqual(excluded_a, excluded_b)
        expected = {i for i in range(50) if not _is_target_sampled("t_part#k0", i, 40)}
        self.assertEqual(excluded_a, expected)
        self.assertGreater(len(excluded_a), 0)
        self.assertLess(len(excluded_a), 50)


class RolloutRootTest(unittest.TestCase):
    def test_walk_and_assemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "chrome" / "task_w"
            run_dir.mkdir(parents=True)
            result = {
                "scores": {"reward": 1.0},
                "params": {
                    "task_id": "task_w",
                    "app": "chrome",
                    "task_instruction": "Open the page.",
                    "stop_reason": "agent_terminate",
                    "sample_index": 0,
                    "screen_width": 1920,
                    "screen_height": 1080,
                },
            }
            (run_dir / "result.json").write_text(json.dumps(result))
            with (run_dir / "traj.jsonl").open("w") as fh:
                for i, line in enumerate(["NO_OP", "TERMINATE"]):
                    fh.write(
                        json.dumps(
                            {
                                "step": i,
                                "image_path": f"{run_dir}/steps/step_{i:03d}.png",
                                "assistant_raw": f"<think>r{i}</think>\n\n{line}",
                                "action_line": line,
                                "dispatched": True,
                            }
                        )
                        + "\n"
                    )
            sample_dir = run_dir / "sample_1"
            sample_dir.mkdir()
            result2 = json.loads(json.dumps(result))
            result2["params"]["sample_index"] = 1
            result2["scores"]["reward"] = 0.0
            result2["params"]["stop_reason"] = "max_steps"
            (sample_dir / "result.json").write_text(json.dumps(result2))
            (sample_dir / "traj.jsonl").write_text("")

            rows = list(iter_rollout_root(Path(tmp)))
            self.assertEqual(len(rows), 2)
            base = rows[0]
            self.assertEqual(base["task_id"], "task_w")
            self.assertEqual(base["app_family"], "chrome")
            self.assertEqual(base["instruction"], "Open the page.")
            self.assertEqual(base["reward"], 1.0)
            self.assertTrue(base["terminated"])
            self.assertEqual(base["stop_reason"], "agent_terminate")
            self.assertEqual(base["screen"], [1920, 1080])
            self.assertEqual(len(base["steps"]), 2)
            self.assertEqual(base["steps"][1]["action_line"], "TERMINATE")
            self.assertEqual(rows[1]["sample_index"], 1)

    def test_missing_result_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "chrome" / "task_x"
            run_dir.mkdir(parents=True)
            (run_dir / "traj.jsonl").write_text("")
            self.assertIsNone(episode_from_run_dir(run_dir, "chrome", "task_x"))
            self.assertEqual(list(iter_rollout_root(Path(tmp))), [])


def _message_signature(row):
    sig = []
    for msg in row["messages"]:
        blocks = []
        for block in msg["content"]:
            keys = tuple(sorted(block.keys()))
            blocks.append((block["type"], keys))
        sig.append((msg["role"], "loss" in msg, msg.get("loss"), tuple(blocks)))
    return sig


def _write_png(path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (128, 0, 255)).save(path, format="PNG")


class EndToEndStage04oTest(unittest.TestCase):
    def _build_fixture(self, tmp: Path):
        img_dir = tmp / "imgs"
        ep_a = make_episode(
            "task_a",
            1.0,
            "agent_terminate",
            ["move(10,12); down(LMB); up(LMB)", 'type("hi")', "TERMINATE"],
        )
        ep_b = make_episode(
            "task_b",
            1.0,
            "max_steps",
            ["scroll(0,-2)", "NO_OP", "NO_OP", "NO_OP", 'type("x")'],
        )
        for rec in (ep_a, ep_b):
            for step in rec["steps"]:
                png = img_dir / rec["task_id"] / f"step_{step['step']:03d}.png"
                _write_png(png)
                step["image_path"] = str(png)
        accepted, rejected, report = process_rows([ep_a, ep_b], default_options())
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(rejected), 0)
        accepted_path = tmp / "accepted.jsonl"
        with accepted_path.open("w") as fh:
            for row in accepted:
                fh.write(json.dumps(row) + "\n")
        return accepted_path

    def test_direct_paths_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            accepted_path = self._build_fixture(tmp)
            out_dir = tmp / "stage_04o"
            report = run_stage_04o(
                accepted_path,
                out_dir,
                image_index_root=None,
                history_n=4,
                system_prompt_path=DEFAULT_SYSTEM_PROMPT_PATH,
            )
            rows = [json.loads(l) for l in (out_dir / "chat.jsonl").read_text().splitlines()]
            self.assertEqual(report["records"], 7)
            self.assertEqual(len(rows), 7)
            self.assertEqual(report["skipped_excluded_run_trim"], 1)

            by_task = {}
            for row in rows:
                by_task.setdefault(row["task_id"], []).append(row)
            self.assertEqual([r["target_step"] for r in by_task["task_a"]], [0, 1, 2])
            self.assertEqual([r["target_step"] for r in by_task["task_b"]], [0, 1, 2, 4])

            required_keys = {
                "conversation_id",
                "recording_id",
                "task_id",
                "app",
                "reward",
                "terminated",
                "pool",
                "target_step",
                "n_history_turns",
                "action_format",
                "messages",
            }
            if REAL_ROW_PATH.exists():
                with REAL_ROW_PATH.open() as fh:
                    real_row = json.loads(fh.readline())
                required_keys = set(real_row.keys())
                zero_hist = next(r for r in rows if r["n_history_turns"] == 0)
                self.assertEqual(real_row["n_history_turns"], 0)
                self.assertEqual(_message_signature(zero_hist), _message_signature(real_row))
                self.assertEqual(real_row["action_format"], zero_hist["action_format"])
            for row in rows:
                self.assertTrue(required_keys.issubset(row.keys()), required_keys - row.keys())
                self.assertEqual(row["action_format"], "ordered_events_v3")
                self.assertEqual(row["pool"], "solved")
                msgs = row["messages"]
                self.assertEqual(msgs[0]["role"], "system")
                self.assertEqual([b["type"] for b in msgs[0]["content"]], ["text"])
                body = msgs[1:]
                self.assertEqual(len(body), 2 * (row["n_history_turns"] + 1))
                for i, msg in enumerate(body):
                    self.assertEqual(msg["role"], "user" if i % 2 == 0 else "assistant")
                user_turns = [m for m in body if m["role"] == "user"]
                for i, m in enumerate(user_turns):
                    self.assertEqual(m["content"][0]["type"], "image")
                    self.assertTrue(m["content"][0]["image"].endswith(".png"))
                    if i == 0:
                        self.assertEqual(m["content"][1]["type"], "text")
                        self.assertIn("Instruction: Do the thing.", m["content"][1]["text"])
                    else:
                        self.assertEqual(len(m["content"]), 1)
                assistant_turns = [m for m in body if m["role"] == "assistant"]
                for m in assistant_turns[:-1]:
                    self.assertIs(m["loss"], False)
                self.assertNotIn("loss", assistant_turns[-1])
                target_text = assistant_turns[-1]["content"][0]["text"]
                self.assertTrue(target_text.startswith("<think>"))
                self.assertIn("</think>", target_text)

            deep = next(r for r in by_task["task_b"] if r["target_step"] == 4)
            self.assertEqual(deep["n_history_turns"], 4)
            hist_assistants = [
                m for m in deep["messages"] if m["role"] == "assistant" and m.get("loss") is False
            ]
            self.assertEqual([m["content"][0]["text"] for m in hist_assistants][-1], "NO_OP")
            final = deep["messages"][-1]["content"][0]["text"]
            self.assertTrue(final.endswith('type("x")'))

    def test_relative_paths_and_dispatched(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            root = tmp / "rollouts"
            ep = make_episode("task_rel", 1.0, "max_steps", ["NO_OP", 'type("x")'])
            for step in ep["steps"]:
                rel = f"steps/step_{step['step']:03d}.png"
                _write_png(root / "chrome" / "task_rel" / rel)
                step["image_path"] = rel
            ep["steps"][0]["dispatched"] = None
            accepted, _, _ = process_rows([ep], default_options())
            accepted_path = tmp / "accepted.jsonl"
            accepted_path.write_text(json.dumps(accepted[0]) + "\n")
            out_dir = tmp / "out"
            report = run_stage_04o(
                accepted_path,
                out_dir,
                image_index_root=None,
                history_n=4,
                system_prompt_path=DEFAULT_SYSTEM_PROMPT_PATH,
                rollout_root=root,
            )
            rows = [json.loads(l) for l in (out_dir / "chat.jsonl").read_text().splitlines()]
            self.assertEqual(report["records"], 1)
            self.assertEqual(report["skipped_not_dispatched"], 1)
            self.assertEqual(rows[0]["target_step"], 1)
            image_ref = rows[0]["messages"][1]["content"][0]["image"]
            self.assertEqual(image_ref, str(root / "chrome" / "task_rel" / "steps" / "step_000.png"))
            self.assertTrue(Path(image_ref).exists())
            hist = [
                m
                for m in rows[0]["messages"]
                if m["role"] == "assistant" and m.get("loss") is False
            ]
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0]["content"][0]["text"], "NO_OP")

    def test_ar_uri_mode(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            accepted_path = self._build_fixture(tmp)
            store = tmp / "image_store" / "screenshots-0000"
            store.mkdir(parents=True)
            with (store / "index.jsonl").open("w") as fh:
                i = 0
                for line in accepted_path.read_text().splitlines():
                    rec = json.loads(line)
                    for step in rec["steps"]:
                        member = f"{rec['task_id']}/{Path(step['image_path']).name}"
                        uri = f"ar://{tmp}/image_store/screenshots-0000/images.array_record#{i}"
                        fh.write(json.dumps({"member": member, "uri": uri}) + "\n")
                        i += 1
            out_dir = tmp / "stage_04o_ar"
            report = run_stage_04o(
                accepted_path,
                out_dir,
                image_index_root=tmp / "image_store",
                history_n=4,
                system_prompt_path=DEFAULT_SYSTEM_PROMPT_PATH,
            )
            rows = [json.loads(l) for l in (out_dir / "chat.jsonl").read_text().splitlines()]
            self.assertEqual(report["records"], 7)
            for row in rows:
                for msg in row["messages"]:
                    for block in msg["content"]:
                        if block["type"] == "image":
                            self.assertTrue(block["image"].startswith("ar://"))


if __name__ == "__main__":
    unittest.main()
