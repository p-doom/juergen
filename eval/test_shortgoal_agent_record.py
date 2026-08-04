"""Agent-driver tests: action JSON in, the recorder's own step rows out.

No VM: one stub client stands in for the guest (tracked cursor, recorded
dispatches, canned ``/execute`` replies), which is enough to drive every part of
``shortgoal_agent_record`` that decides what a recording MEANS — the action
vocabulary, the chord mapping, the grid/pixel twin, the step cap, the session
round trip — plus the two recorder-side gates added for the pilot failures (the
fixture keyboard probe and the browser popup sweep).

The load-bearing assertions are the contracts the driver claims: a published
agent recording loads and builds through ``shortgoal_build`` for BOTH arms (with
the bbox gate exempted only because its source says ``sonnet_agent``), and it
replays through ``shortgoal_record.replay_recording`` unchanged.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shortgoal_agent_record as agent  # noqa: E402
import shortgoal_build as build  # noqa: E402
import shortgoal_fixture as fixture  # noqa: E402
import shortgoal_golden as golden  # noqa: E402
import shortgoal_grammar as grammar  # noqa: E402
import shortgoal_record as sr  # noqa: E402
import shortgoal_templates as templates  # noqa: E402
from osworld_vm_client import StepResult  # noqa: E402
from shortgoal_grammar import ARM_ABS, ARM_REL, ARMS  # noqa: E402

SCREEN = templates.SCREEN_WH
TASK_ID = templates.OVERFIT1_TASK_ID
FIXTURE_TASK_ID = "fx_click_button__s00"
QUOTED_THOUGHT = "Ich klick's auf »Delta« — \"jetzt\" größer, sagt's der Plan."
FIXTURE_STATE = {"ready": True, "keyboard": True, "keys_seen": 0, "kind": "buttons"}
WMCTRL_WINDOWS = (
    "0x02200003 -1 user-virtual-machine @!0,0;BDHF\n"
    f"0x03800004  0 user-virtual-machine {fixture.PAGE_READY_TITLE} - Google Chrome\n"
    "0x03800015  0 user-virtual-machine Can't update Chrome\n"
)


class FakeClient:
    """The stub guest: a tracked cursor, recorded dispatches, canned replies."""

    def __init__(
        self,
        *,
        cursor: tuple[int, int] = (0, 0),
        screen_wh: tuple[int, int] = SCREEN,
        state: dict[str, Any] | None = None,
        probe: dict[str, Any] | None = None,
        windows: str = WMCTRL_WINDOWS,
        tools: str = "wmctrl",
        focused: bool = True,
    ) -> None:
        self.cursor = tuple(cursor)
        self.screen_wh = tuple(screen_wh)
        self.state = dict(state or FIXTURE_STATE)
        self.probe = dict(probe or {})
        self.windows = windows
        self.tools = tools
        self.focused = focused
        self.dispatched: list[Any] = []
        self.executed: list[str] = []
        self.commands: list[Any] = []
        self.closed: list[str] = []
        self.activated: list[str] = []
        self.settles: list[dict[str, Any]] = []
        self.shots = 0

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def screen_size(self) -> tuple[int, int]:
        return self.screen_wh

    def screenshot(self) -> Image.Image:
        return Image.new("RGB", (16, 9), (self.shots * 11 % 256, 40, 80))

    def screenshot_settled(self, **kwargs: Any) -> Image.Image:
        self.shots += 1
        self.settles.append(dict(kwargs))
        return self.screenshot()

    def execute(self, command: str) -> None:
        self.executed.append(command)
        if "keyDown" in command and sr.KEY_PROBE_NAME in command and self.focused:
            self.state["keys_seen"] = int(self.state["keys_seen"]) + 1
        if "moveTo(" in command:
            x, _, y = command.partition("moveTo(")[2].partition(")")[0].partition(",")
            self.cursor = (int(x), int(y))

    def run_command(self, command: Any, *, shell: bool = False) -> dict[str, Any]:
        self.commands.append(command)
        if isinstance(command, list):
            code = command[2] if len(command) > 2 else ""
            if code == sr._READ_TEXT_CODE:
                return {"output": json.dumps(self.state)}
            if code == sr._PATH_PROBE_CODE:
                return {"output": json.dumps(self.probe)}
            if code == sr._FILE_PREP_CODE:
                return {"output": json.dumps({"home": "/home/user", "dirs": [], "files": []})}
            if code in (sr._PROCESS_PROBE_CODE, sr._PTS_PROBE_CODE):
                return {"output": "[]"}
            if code == sr._TITLE_COMMAND:
                return {"output": self.windows}
            return {"output": "{}"}
        text = str(command)
        if "wmctrl -l" in text:
            return {"output": self.windows}
        if "wmctrl -i -c" in text:
            self.closed.append(text.split()[-1])
            return {"output": ""}
        if "wmctrl -a" in text:
            self.activated.append(text)
            return {"output": ""}
        if "command -v" in text:
            return {"output": self.tools}
        return {"output": ""}

    def dispatch_ordered_action(self, action: Any) -> StepResult:
        before = self.cursor
        scroll = 0
        for prim in action.primitives:
            if prim.kind == "move_to":
                self.cursor = (prim.x, prim.y)
            elif prim.kind == "move":
                self.cursor = (self.cursor[0] + prim.dx, self.cursor[1] + prim.dy)
            elif prim.kind == "scroll":
                scroll += int(prim.dy or 0)
        self.dispatched.append(action)
        return StepResult(
            cursor_before=before,
            cursor_after=self.cursor,
            intended_target=self.cursor,
            delta=(self.cursor[0] - before[0], self.cursor[1] - before[1]),
            scroll=scroll,
            events_dispatched=[],
            parse_ok=True,
            action_text="",
        )


def _kinds(step: list[dict[str, Any]]) -> list[str]:
    return [prim["kind"] for prim in step]


def _names(step: list[dict[str, Any]]) -> list[str]:
    return [prim["name"] for prim in step if "name" in prim]


class ActionVocabularyTests(unittest.TestCase):
    def test_every_kind_converts_to_the_oracles_own_turn_shape(self) -> None:
        cases = {
            "click": ({"kind": "click", "at": [960, 540]}, ["move", "down", "up"]),
            "middle_double_click": (
                {"kind": "click", "at": [10, 20], "button": "middle", "count": 2},
                ["move", "down", "up", "down", "up"],
            ),
            "right_click": (
                {"kind": "click", "at": [10, 20], "button": "right"},
                ["move", "down", "up"],
            ),
            "move": ({"kind": "move", "to": [4, 5]}, ["move"]),
            "drag": (
                {"kind": "drag", "from": [10, 10], "to": [200, 10]},
                ["move", "down", "move", "up"],
            ),
            "type": ({"kind": "type", "text": "touch a.txt"}, ["type"]),
            "key": ({"kind": "key", "keys": ["ctrl", "s"]}, ["down", "down", "up", "up"]),
            "scroll": ({"kind": "scroll", "notches": -3}, ["scroll"]),
            "no_op": ({"kind": "no_op"}, ["no_op"]),
        }
        for name, (action, kinds) in cases.items():
            with self.subTest(action=name):
                step = agent.action_step(action)
                self.assertEqual(_kinds(step), kinds)
                self.assertEqual(golden.validate_step(step), step)
        self.assertEqual(
            _names(agent.action_step({"kind": "click", "at": [1, 2], "button": "right"})),
            ["RMB", "RMB"],
        )
        self.assertEqual(
            _names(agent.action_step({"kind": "drag", "from": [1, 2], "to": [3, 4]})),
            ["LMB", "LMB"],
        )

    def test_chords_map_to_the_v4_name_set_and_release_in_reverse(self) -> None:
        action = {"kind": "key", "keys": ["ctrl", "shift", "t"]}
        self.assertEqual(_names(agent.action_step(action)), [
            "ControlLeft", "ShiftLeft", "KeyT", "KeyT", "ShiftLeft", "ControlLeft",
        ])
        self.assertEqual(
            agent.plan_step(action, (10, 10), SCREEN).abs_line,
            "down(ControlLeft); down(ShiftLeft); down(KeyT); up(KeyT); up(ShiftLeft); "
            "up(ControlLeft)",
        )

    def test_key_names_cover_letters_digits_aliases_and_rdev_passthrough(self) -> None:
        wanted = {
            "s": "KeyS", "S": "KeyS", "7": "Num7", "ctrl": "ControlLeft",
            "Control": "ControlLeft", "ctrl_right": "ControlRight", "enter": "Return",
            "Return": "Return", "escape": "Escape", "page_down": "PageDown",
            "up": "ArrowUp", "delete": "Delete", "super": "MetaLeft", "KeyA": "KeyA",
            "Num0": "Num0", "space": "Space",
        }
        for key, name in wanted.items():
            with self.subTest(key=key):
                self.assertEqual(agent.key_name(key), name)
        for key in ("", "f13", "hyper", "ctrl+s", "unknownkey", 5, None):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    agent.key_name(key)

    def test_malformed_actions_are_refused(self) -> None:
        for action in (
            {"kind": "jump", "at": [1, 2]},
            {"kind": "click"},
            {"kind": "click", "at": [1, 2], "where": "here"},
            {"kind": "click", "at": [1, 2, 3]},
            {"kind": "click", "at": [1.5, 2]},
            {"kind": "click", "at": [1, 2], "button": "extra"},
            {"kind": "click", "at": [1, 2], "count": 4},
            {"kind": "move", "to": "middle"},
            {"kind": "drag", "from": [1, 2]},
            {"kind": "type", "text": ""},
            {"kind": "type", "text": "with\nnewline"},
            {"kind": "key", "keys": []},
            {"kind": "key", "keys": "ctrl"},
            {"kind": "scroll", "notches": 0},
            {"kind": "scroll", "notches": 1.5},
            {"kind": "no_op", "notches": 1},
            "click",
        ):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    agent.action_step(action)

    def test_targets_off_the_screen_and_zero_delta_moves_are_refused(self) -> None:
        for at in ([SCREEN[0], 10], [-1, 10], [10, SCREEN[1]]):
            with self.subTest(at=at):
                with self.assertRaises(ValueError):
                    agent.plan_step({"kind": "click", "at": at}, (5, 5), SCREEN)
        held = (grammar.snap_point_px(960, SCREEN[0]), grammar.snap_point_px(540, SCREEN[1]))
        with self.assertRaises(ValueError) as caught:
            agent.plan_step({"kind": "click", "at": list(held)}, held, SCREEN)
        self.assertIn("move(0,0)", str(caught.exception))

    def test_the_plan_is_grid_snapped_and_renders_on_both_arms(self) -> None:
        plan = agent.plan_step({"kind": "click", "at": [963, 541]}, (100, 100), SCREEN)
        target = (
            grammar.snap_point_px(963, SCREEN[0]), grammar.snap_point_px(541, SCREEN[1]),
        )
        self.assertEqual(
            [(p.x, p.y) for p in plan.px_action.primitives if p.kind == "move_to"], [target],
        )
        self.assertEqual(plan.abs_line, "move_to(502,501); down(LMB); up(LMB)")
        self.assertEqual(
            plan.rel_line,
            f"move({grammar.norm_delta(target[0] - 100, SCREEN[0])},"
            f"{grammar.norm_delta(target[1] - 100, SCREEN[1])}); down(LMB); up(LMB)",
        )
        self.assertEqual(plan.zero_deltas, 0)
        for arm, line in ((ARM_ABS, plan.abs_line), (ARM_REL, plan.rel_line)):
            with self.subTest(arm=arm):
                self.assertEqual(build.validate_line(line, arm), line)


class ThoughtCaptureTests(unittest.TestCase):
    def test_plain_absent_and_empty_thoughts(self) -> None:
        self.assertEqual(agent.parse_thought(), "")
        self.assertEqual(agent.parse_thought(None, None), "")
        self.assertEqual(agent.parse_thought(""), "")
        self.assertEqual(agent.parse_thought(b64=""), "")
        self.assertEqual(agent.parse_thought("I click Delta, then confirm."),
                         "I click Delta, then confirm.")
        limit = "x" * grammar.THOUGHT_MAX_CHARS
        self.assertEqual(agent.parse_thought(limit), limit)

    def test_base64_survives_quotes_and_umlauts(self) -> None:
        encoded = base64.b64encode(QUOTED_THOUGHT.encode("utf-8")).decode("ascii")
        self.assertEqual(agent.parse_thought(None, encoded), QUOTED_THOUGHT)
        self.assertEqual(agent.parse_thought(b64=f"  {encoded}  "), QUOTED_THOUGHT)
        self.assertIn("'", QUOTED_THOUGHT)
        self.assertIn('"', QUOTED_THOUGHT)
        self.assertNotIn("'", encoded)
        self.assertNotIn('"', encoded)

    def test_thoughts_that_must_be_refused(self) -> None:
        over = "x" * (grammar.THOUGHT_MAX_CHARS + 1)
        for kwargs in (
            {"text": "both", "b64": base64.b64encode(b"both").decode("ascii")},
            {"text": over},
            {"b64": base64.b64encode(over.encode("utf-8")).decode("ascii")},
            {"text": "two\nlines"},
            {"text": "a\ttab"},
            {"text": "nul\x00byte"},
            {"text": 5},
            {"b64": "not base64!!"},
            {"b64": "aGk"},
            {"b64": base64.b64encode(b"\xff\xfe").decode("ascii")},
            {"b64": 7},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    agent.parse_thought(**kwargs)

    def test_the_cli_takes_one_thought_flag_at_most(self) -> None:
        base = ["step", "--session_dir=/tmp/session", "--action={}"]
        self.assertEqual(agent._parse_args([*base, "--thought=hi"]).thought, "hi")
        self.assertEqual(agent._parse_args([*base, "--thought_b64=aGk="]).thought_b64, "aGk=")
        parsed = agent._parse_args(base)
        self.assertIsNone(parsed.thought)
        self.assertIsNone(parsed.thought_b64)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                agent._parse_args([*base, "--thought=hi", "--thought_b64=aGk="])


class ScrollSettleTests(unittest.TestCase):
    def _prims(self, action: dict[str, Any]) -> Any:
        return agent.plan_step(action, (10, 10), SCREEN).px_action.primitives

    def test_only_scroll_turns_wait_longer(self) -> None:
        live = sr.Settle()
        click = self._prims({"kind": "click", "at": [960, 540]})
        scroll = self._prims({"kind": "scroll", "notches": -5})
        self.assertEqual(agent.settle_for(live, click), live)
        self.assertEqual(agent.settle_for(live, ()), live)
        extended = agent.settle_for(live, scroll)
        self.assertEqual(extended.delay_s, agent.SCROLL_SETTLE_DELAY_S)
        self.assertEqual(extended.stable_timeout_s, agent.SCROLL_SETTLE_STABLE_TIMEOUT_S)
        self.assertEqual(extended.poll_s, live.poll_s)
        self.assertGreater(extended.delay_s, live.delay_s)
        self.assertGreater(extended.stable_timeout_s, live.stable_timeout_s)

    def test_an_explicit_no_settle_and_a_generous_settle_are_left_alone(self) -> None:
        scroll = self._prims({"kind": "scroll", "notches": 3})
        off = sr.Settle(delay_s=0.0, stable_timeout_s=0.0, poll_s=0.0)
        self.assertEqual(agent.settle_for(off, scroll), off)
        generous = sr.Settle(delay_s=2.0, stable_timeout_s=9.0, poll_s=0.2)
        self.assertEqual(agent.settle_for(generous, scroll), generous)


class PortTests(unittest.TestCase):
    def test_slots_are_stable_and_never_collide_with_the_recorders_grid(self) -> None:
        self.assertEqual(agent.agent_ports(0), (5003, 5903))
        self.assertEqual(agent.agent_ports(1), (5013, 5913))
        recorder_ports = {5000 + (job % 200) * 10 for job in range(200)}
        seen: set[int] = set()
        for slot in range(agent.MAX_SLOT + 1):
            vm_port, vnc_port = agent.agent_ports(slot)
            self.assertNotIn(vm_port, recorder_ports)
            self.assertLess(vm_port, agent.VNC_BASE)
            self.assertNotIn(vm_port, seen)
            seen.add(vm_port)
            self.assertEqual(vnc_port - vm_port, agent.VNC_BASE - agent.PORT_BASE)
        for slot in (-1, agent.MAX_SLOT + 1, "0", 1.0, True):
            with self.subTest(slot=slot):
                with self.assertRaises(ValueError):
                    agent.agent_ports(slot)

    def test_a_dead_vm_is_never_mistaken_for_a_live_one(self) -> None:
        self.assertFalse(agent._vm_alive(0, 5003))
        self.assertFalse(agent._vm_alive(os.getpid(), 5003))
        self.assertFalse(agent._kill_vm(0, port=5003, label="none"))


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.settle = sr.Settle(delay_s=0.0, stable_timeout_s=0.0, poll_s=0.0)

    def _session(self, task_id: str = TASK_ID, *, max_steps: int = 12) -> agent.Session:
        task = agent._task(task_id)
        setup: dict[str, Any] = {"setup_id": task.setup_id}
        if "fixture_spec" in task.params:
            setup["widgets"] = fixture.spec_widgets(task.params["fixture_spec"])
        start = sr.cursor_start_px(task.task_id, SCREEN)
        session = agent.Session.create(
            self.root / task.task_id,
            task,
            screen_wh=SCREEN,
            cursor_start=start,
            cursor=start,
            setup=setup,
            settle=self.settle,
            max_steps=max_steps,
            n_attempt=3,
        )
        session.frames_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 9)).save(session.frame_path(0))
        session.save()
        return session

    def _client(self, session: agent.Session, **kwargs: Any) -> FakeClient:
        return FakeClient(cursor=tuple(session.data["cursor"]), **kwargs)

    def _episode(
        self,
        task_id: str,
        actions: list[dict[str, Any]],
        *,
        thoughts: list[str] | None = None,
    ) -> agent.Session:
        session = self._session(task_id)
        client = self._client(session)
        for index, action in enumerate(actions):
            session.apply_step(
                client, action, settle=self.settle,
                thought="" if thoughts is None else thoughts[index],
            )
        return session

    def test_a_step_writes_the_recorders_row_and_the_next_frame(self) -> None:
        session = self._session()
        client = self._client(session)
        action = {"kind": "click", "at": [960, 540]}
        plan = agent.plan_step(action, tuple(session.data["cursor_start"]), SCREEN)
        status = session.apply_step(client, action, settle=self.settle)
        self.assertEqual((status["step"], status["steps_left"]), (1, 11))
        self.assertEqual(status["frame"], str(session.frame_path(1).resolve()))
        self.assertEqual(status["line"], plan.abs_line)
        self.assertTrue(session.frame_path(1).is_file())
        row = session.steps[0]
        self.assertEqual(row["frame"], "step_000.png")
        self.assertEqual(row["cursor_before"], list(session.data["cursor_start"]))
        self.assertEqual(row["cursor_after"], list(client.cursor))
        self.assertEqual(row["primitives_px"], [asdict(p) for p in plan.px_action.primitives])
        self.assertEqual(
            row["primitives_grid"], [asdict(p) for p in plan.grid_action.primitives],
        )
        self.assertEqual(status["cursor"], row["cursor_after"])

    def test_a_scroll_step_captures_its_frame_under_the_longer_settle(self) -> None:
        session = self._session()
        client = self._client(session)
        live = sr.Settle(delay_s=0.05, stable_timeout_s=0.1, poll_s=0.01)
        session.apply_step(client, {"kind": "click", "at": [960, 540]}, settle=live)
        self.assertEqual(client.settles[-1]["min_delay_s"], live.delay_s)
        session.apply_step(client, {"kind": "scroll", "notches": -5}, settle=live)
        self.assertEqual(client.settles[-1], {
            "min_delay_s": agent.SCROLL_SETTLE_DELAY_S,
            "stability_timeout_s": agent.SCROLL_SETTLE_STABLE_TIMEOUT_S,
            "poll_s": live.poll_s,
        })
        session.apply_step(client, {"kind": "no_op"}, settle=live)
        self.assertEqual(client.settles[-1]["min_delay_s"], live.delay_s)

    def test_the_session_round_trips_through_disk(self) -> None:
        session = self._session()
        client = self._client(session)
        session.apply_step(client, {"kind": "type", "text": "touch a.txt"}, settle=self.settle)
        session.apply_step(client, {"kind": "key", "keys": ["enter"]}, settle=self.settle)
        reloaded = agent.Session.load(session.session_dir)
        self.assertEqual(reloaded.data, session.data)
        self.assertEqual(
            reloaded.data["lines"], ['type("touch a.txt")', "down(Return); up(Return)"],
        )
        reloaded.apply_step(client, {"kind": "no_op"}, settle=self.settle)
        self.assertEqual(reloaded.steps[-1]["primitives_px"], [])
        self.assertEqual(len(agent.Session.load(session.session_dir).steps), 3)
        self.assertTrue(session.frame_path(3).is_file())

    def test_the_step_cap_is_hard(self) -> None:
        session = self._session(max_steps=2)
        client = self._client(session)
        for step in range(2):
            with self.subTest(step=step):
                session.apply_step(client, {"kind": "scroll", "notches": -2}, settle=self.settle)
        self.assertEqual(session.status()["steps_left"], 0)
        with self.assertRaises(agent.StepCapReached):
            session.apply_step(client, {"kind": "scroll", "notches": -2}, settle=self.settle)
        self.assertEqual((len(session.steps), len(client.dispatched)), (2, 2))

    def test_a_finished_session_refuses_further_steps(self) -> None:
        session = self._session()
        client = self._client(session)
        session.apply_step(client, {"kind": "scroll", "notches": 3}, settle=self.settle)
        session.publish(
            {"kind": "guest_path_exists", "passed": True, "detail": {}},
            status=agent.STATUS_PASSED,
        )
        with self.assertRaises(RuntimeError):
            session.apply_step(client, {"kind": "scroll", "notches": 3}, settle=self.settle)

    def test_bad_actions_never_dispatch_and_never_grow_the_episode(self) -> None:
        session = self._session()
        client = self._client(session)
        for action in ({"kind": "nope"}, {"kind": "key", "keys": ["hyper"]}):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    session.apply_step(client, action, settle=self.settle)
        self.assertEqual(client.dispatched, [])
        self.assertEqual(session.steps, [])
        self.assertFalse(session.frame_path(1).exists())

    def test_a_published_recording_matches_the_recorders_schema(self) -> None:
        session = self._episode(TASK_ID, [
            {"kind": "type", "text": "touch report.cfg"},
            {"kind": "key", "keys": ["enter"]},
        ])
        verifier = {"kind": "guest_path_exists", "passed": True, "detail": {}}
        path = session.publish(verifier, status=agent.STATUS_PASSED)
        self.assertEqual(path.name, sr.RECORDING_NAME)
        published = json.loads(path.read_text())
        task = agent._task(TASK_ID)
        oracle_keys = {
            "schema_version", "task_id", "template_id", "seed", "category", "tier_b",
            "single_action", "setup_id", "policy_id", "params", "instruction",
            "screen_size", "cursor_start", "steps", "n_steps", "n_frames",
            "zero_delta_moves", "setup", "verifier", "elapsed_s",
        }
        self.assertEqual(set(published), oracle_keys | {"source", "n_attempt"})
        self.assertEqual(published["source"], agent.SOURCE)
        self.assertEqual(published["n_attempt"], 3)
        self.assertEqual(published["instruction"], task.instruction)
        self.assertEqual(published["params"], task.params)
        self.assertEqual(published["policy_id"], task.policy_id)
        self.assertEqual(published["n_frames"], published["n_steps"] + 1)
        self.assertEqual(published["zero_delta_moves"], 0)
        self.assertEqual(published["verifier"], verifier)
        self.assertEqual(
            agent.Session.load(session.session_dir).data["status"], agent.STATUS_PASSED,
        )

    def test_an_agent_episode_builds_for_both_arms_despite_a_missed_click(self) -> None:
        session = self._episode(FIXTURE_TASK_ID, [
            {"kind": "click", "at": [30, 40]},
            {"kind": "click", "at": [654, 578]},
            {"kind": "key", "keys": ["enter"]},
        ])
        path = session.publish(
            {"kind": "fixture_state", "passed": True, "detail": {}},
            status=agent.STATUS_PASSED,
        )
        rec = build.load_recording(self.root, FIXTURE_TASK_ID)
        self.assertEqual(rec.source, build.SOURCE_AGENT)
        self.assertEqual(rec.n_frames, 4)
        self.assertTrue(rec.widgets)
        for arm in ARMS:
            with self.subTest(arm=arm):
                lines = build.episode_lines(rec, arm)
                self.assertEqual(len(lines), 4)
                self.assertEqual(lines[-1], grammar.TERMINATE_LINE)
                for line in lines[:-1]:
                    self.assertEqual(build.validate_line(line, arm), line)
        manifest = build.build_arm(
            recordings_root=self.root,
            output_root=self.root / "chat",
            arm=ARM_REL,
            subset="full",
            splits={"train": [FIXTURE_TASK_ID], "tier_a": [], "tier_b": []},
            resolution=(64, 36),
        )
        self.assertEqual(manifest["counts"]["n_tasks"], 1)
        self.assertEqual(manifest["counts"]["tasks_by_source"], {build.SOURCE_AGENT: 1})
        as_oracle = json.loads(path.read_text())
        as_oracle["source"] = build.SOURCE_ORACLE
        path.write_text(json.dumps(as_oracle))
        with self.assertRaises(ValueError):
            build.episode_lines(build.load_recording(self.root, FIXTURE_TASK_ID), ARM_ABS)

    def test_a_captured_thought_rides_the_step_row_into_the_recording(self) -> None:
        session = self._session()
        client = self._client(session)
        status = session.apply_step(
            client, {"kind": "scroll", "notches": -2}, settle=self.settle,
            thought=QUOTED_THOUGHT,
        )
        self.assertEqual(status["thought_chars"], len(QUOTED_THOUGHT))
        self.assertEqual(session.steps[0]["thought"], QUOTED_THOUGHT)
        session.apply_step(client, {"kind": "no_op"}, settle=self.settle)
        self.assertEqual(session.steps[1]["thought"], "")
        self.assertEqual(
            [row["thought"] for row in agent.Session.load(session.session_dir).steps],
            [QUOTED_THOUGHT, ""],
        )
        with self.assertRaises(ValueError):
            session.apply_step(
                client, {"kind": "no_op"}, settle=self.settle,
                thought="x" * (grammar.THOUGHT_MAX_CHARS + 1),
            )
        self.assertEqual(len(session.steps), 2)
        path = session.publish(
            {"kind": "guest_path_exists", "passed": True, "detail": {}},
            status=agent.STATUS_PASSED,
        )
        published = json.loads(path.read_text())
        self.assertEqual(
            [row["thought"] for row in published["steps"]], [QUOTED_THOUGHT, ""],
        )
        self.assertEqual(published["schema_version"], sr.SCHEMA_VERSION)
        self.assertEqual(build.load_recording(self.root, TASK_ID).n_frames, 3)

    def test_a_thought_is_ignored_byte_identically_by_the_no_think_builder(self) -> None:
        marker = "SENTINELTHOUGHT the tile under the pointer is Delta"
        session = self._episode(
            FIXTURE_TASK_ID,
            [{"kind": "click", "at": [654, 578]}, {"kind": "key", "keys": ["enter"]}],
            thoughts=[marker, f"{marker} again"],
        )
        path = session.publish(
            {"kind": "fixture_state", "passed": True, "detail": {}},
            status=agent.STATUS_PASSED,
        )
        with_thoughts = self._built_chat("out_thoughts")
        self.assertNotIn("SENTINELTHOUGHT", with_thoughts)
        recording = json.loads(path.read_text())
        for row in recording["steps"]:
            row.pop("thought")
        path.write_text(json.dumps(recording))
        self.assertNotIn("thought", path.read_text())
        self.assertEqual(self._built_chat("out_plain"), with_thoughts)

    def _built_chat(self, name: str) -> str:
        out = self.root / name
        build.build_arm(
            recordings_root=self.root,
            output_root=out,
            arm=ARM_REL,
            subset="full",
            splits={"train": [FIXTURE_TASK_ID], "tier_a": [], "tier_b": []},
            resolution=(64, 36),
        )
        chat = (out / build.CHAT_RELPATH).read_text()
        return chat.replace(str(out.resolve()), "OUT")

    def test_a_published_recording_replays_through_the_recorder(self) -> None:
        session = self._episode(TASK_ID, [
            {"kind": "type", "text": "touch report.cfg"},
            {"kind": "key", "keys": ["enter"]},
        ], thoughts=[QUOTED_THOUGHT, "now confirm with Enter"])
        path = session.publish(
            {"kind": "guest_path_exists", "passed": True, "detail": {}},
            status=agent.STATUS_PASSED,
        )
        client = FakeClient(probe={
            "path": "/home/user/report.cfg", "exists": True, "is_dir": False,
            "is_file": True, "executable": False, "content": None,
        })
        recording = json.loads(path.read_text())
        self.assertEqual(recording["steps"][0]["thought"], QUOTED_THOUGHT)
        replay = sr.replay_recording(
            client,
            recording,
            self.root / "replay",
            settle=self.settle,
            verify_timeout_s=0.0,
        )
        self.assertTrue(replay["passed"])
        self.assertEqual(replay["n_steps"], 2)
        self.assertEqual(replay["max_cursor_drift_px"], 0)
        self.assertEqual(
            [row["frame"] for row in replay["steps"]], ["step_000.png", "step_001.png"],
        )

    def test_a_rejected_episode_publishes_a_failure_only(self) -> None:
        session = self._episode(TASK_ID, [{"kind": "scroll", "notches": 2}])
        path = session.publish(
            {"kind": "guest_path_exists", "passed": False, "detail": {}},
            status=agent.STATUS_FAILED,
            reason="verifier guest_path_exists failed",
        )
        self.assertEqual(path.name, sr.FAILURE_NAME)
        self.assertFalse((session.session_dir / sr.RECORDING_NAME).exists())
        failure = json.loads(path.read_text())
        self.assertEqual(failure["rejected_reason"], "verifier guest_path_exists failed")
        self.assertEqual(failure["source"], agent.SOURCE)
        with self.assertRaises(FileNotFoundError):
            build.load_recording(self.root, TASK_ID)

    def test_create_rejects_a_useless_budget_or_attempt(self) -> None:
        task = agent._task(TASK_ID)
        for kwargs in ({"max_steps": 0}, {"max_steps": True}, {"n_attempt": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    agent.Session.create(
                        self.root / TASK_ID,
                        task,
                        screen_wh=SCREEN,
                        cursor_start=(1, 1),
                        cursor=(1, 1),
                        setup={},
                        settle=self.settle,
                        **kwargs,
                    )


class RecorderGateTests(unittest.TestCase):
    """The two pilot fixes, exercised against the stub guest."""

    def test_the_keyboard_probe_passes_when_the_fixture_sees_the_key(self) -> None:
        client = FakeClient(focused=True)
        probe = sr._probe_fixture_keyboard(client, FIXTURE_TASK_ID)
        self.assertEqual(probe, {"key_probe_attempts": 1, "keys_seen": 1})
        self.assertEqual(client.activated, [])

    def test_the_keyboard_probe_escalates_then_fails_loudly(self) -> None:
        client = FakeClient(focused=False)
        with mock.patch.object(sr, "KEY_PROBE_TIMEOUT_S", 0.05):
            with self.assertRaises(RuntimeError) as caught:
                sr._probe_fixture_keyboard(client, FIXTURE_TASK_ID)
        self.assertIn("never received a probe keypress", str(caught.exception))
        self.assertEqual(len(client.activated), sr.KEY_PROBE_ATTEMPTS)
        self.assertTrue(all(fixture.FIXTURE_TITLE in text for text in client.activated))

    def test_browser_popups_are_closed_and_the_page_window_is_kept(self) -> None:
        client = FakeClient()
        self.assertEqual(
            sr._close_browser_popups(client, fixture.PAGE_READY_TITLE),
            ["Can't update Chrome"],
        )
        self.assertEqual(client.closed, ["0x03800015"])
        self.assertEqual(sr._close_browser_popups(FakeClient(windows=""), "x"), [])

    def test_the_page_ready_title_is_a_post_paint_signal(self) -> None:
        params = templates.draw_params("web_click_link", 0)
        page = fixture.make_html_page("link_grid", params)
        self.assertIn(f'document.title="{fixture.PAGE_READY_TITLE}"', page)
        self.assertIn("requestAnimationFrame", page)
        self.assertNotIn(f"<title>{fixture.PAGE_READY_TITLE}", page)
        self.assertNotIn(params["expect"]["title"], fixture.PAGE_READY_TITLE)
        self.assertEqual(sr.PAGE_READY_TITLE, fixture.PAGE_READY_TITLE)


if __name__ == "__main__":
    unittest.main()
