from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import osworld_runtime as rt  # noqa: E402
import osworld_vm_client  # noqa: E402
import sampling  # noqa: E402
import shortgoal_grammar as sg  # noqa: E402
from action_parser import (  # noqa: E402
    OrderedAction,
    OrderedPrimitive,
    parse_ordered_action,
    parse_ordered_v4_action,
)

_GOAL = "GOAL: create the file report.txt on the Desktop"
_DATA_URL_PREFIX = "data:image/jpeg;base64,"

_EVICTION_TABLE = {
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [6],
    8: [6],
    9: [6],
    10: [6],
    11: [6, 10],
    12: [6, 10],
}


def _action(i: int) -> str:
    return f'type("step{i}"); down(Return); up(Return)'


class _FakeFrame:
    """PIL-Image stand-in whose JPEG bytes are its own tag (stub-venv safe)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def save(self, buf, **kwargs) -> None:
        buf.write(self.tag.encode("ascii"))


def _frame_tag(part: dict) -> str:
    url = part["image_url"]["url"]
    if not url.startswith(_DATA_URL_PREFIX):
        raise AssertionError(f"not a JPEG data URL: {url[:40]!r}")
    return base64.b64decode(url[len(_DATA_URL_PREFIX):]).decode("ascii")


def _episode(n_frames: int, **kw) -> rt.KeepTextWindow:
    w = rt.KeepTextWindow(_FakeFrame("f000"), **kw)
    for i in range(1, n_frames):
        w.append_turn(_action(i - 1), _FakeFrame(f"f{i:03d}"))
    return w


def _simulate(n_frames: int, k: int, keep: int) -> tuple[list[int], list[bool], list[int]]:
    """Independent keep-text simulation: (eviction points, liveness, live counts)."""
    live = []
    points: list[int] = []
    counts: list[int] = []
    for j in range(n_frames):
        live.append(True)
        if sum(live) > k:
            alive = [i for i, v in enumerate(live) if v]
            for i in alive[:-keep]:
                live[i] = False
            points.append(j)
        counts.append(sum(live))
    return points, live, counts


class EvictionPointTests(unittest.TestCase):
    def test_defaults_come_from_the_grammar_contract(self) -> None:
        self.assertEqual((rt.K_IMAGES, rt.KEEP_IMAGES), (sg.K_IMAGES, sg.KEEP_IMAGES))
        self.assertEqual(rt.IMAGE_PLACEHOLDER, sg.IMAGE_PLACEHOLDER)
        self.assertEqual((rt.K_IMAGES, rt.KEEP_IMAGES), (6, 3))
        for n in (1, 7, 11, 23):
            self.assertEqual(
                rt.keep_text_eviction_points(n),
                rt.keep_text_eviction_points(n, sg.K_IMAGES, sg.KEEP_IMAGES),
            )

    def test_hand_computed_table(self) -> None:
        for n_frames, expected in _EVICTION_TABLE.items():
            with self.subTest(n_frames=n_frames):
                self.assertEqual(rt.keep_text_eviction_points(n_frames), expected)

    def test_single_frame_episode_never_evicts(self) -> None:
        self.assertEqual(rt.keep_text_eviction_points(1), [])

    def test_record_count_is_one_plus_evictions(self) -> None:
        for n_frames in range(1, 7):
            self.assertEqual(len(rt.keep_text_eviction_points(n_frames)), 0)
        for n_frames in (7, 8, 9):
            self.assertEqual(len(rt.keep_text_eviction_points(n_frames)), 1)
        self.assertEqual(len(rt.keep_text_eviction_points(11)), 2)

    def test_matches_independent_simulation(self) -> None:
        for k, keep in ((6, 3), (4, 2), (3, 1), (8, 4), (2, 1)):
            for n_frames in range(1, 26):
                with self.subTest(k=k, keep=keep, n_frames=n_frames):
                    points, _, _ = _simulate(n_frames, k, keep)
                    self.assertEqual(rt.keep_text_eviction_points(n_frames, k, keep), points)

    def test_is_pure(self) -> None:
        first = rt.keep_text_eviction_points(12)
        first.append(999)
        self.assertEqual(rt.keep_text_eviction_points(12), [6, 10])

    def test_rejects_bad_arguments(self) -> None:
        for args in ((0,), (-1,), (2.0,), ("9",)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    rt.keep_text_eviction_points(*args)
        for k, keep in ((6, 6), (6, 7), (6, 0), (1, 1), (6, 3.0), (6.0, 3)):
            with self.subTest(k=k, keep=keep):
                with self.assertRaises(ValueError):
                    rt.keep_text_eviction_points(9, k, keep)


class KeepTextWindowTests(unittest.TestCase):
    def test_eviction_fires_only_past_k_and_keeps_newest_three(self) -> None:
        w = rt.KeepTextWindow(_FakeFrame("f000"))
        counts = [w.live_count()]
        for i in range(1, 13):
            w.append_turn(_action(i - 1), _FakeFrame(f"f{i:03d}"))
            self.assertLessEqual(w.live_count(), sg.K_IMAGES)
            counts.append(w.live_count())
        self.assertEqual(counts, [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5])
        self.assertEqual(w.evicted_at, [6, 10])

    def test_eviction_keeps_the_newest_keep_frames_only(self) -> None:
        w = _episode(7)
        self.assertEqual(w.live_count(), sg.KEEP_IMAGES)
        self.assertEqual(w.liveness(), [False, False, False, False, True, True, True])
        self.assertEqual(
            [f.tag for f in w.frames if f is not None], ["f004", "f005", "f006"]
        )

    def test_window_agrees_with_the_pure_eviction_function(self) -> None:
        for n_frames in range(1, 20):
            with self.subTest(n_frames=n_frames):
                w = _episode(n_frames)
                points, live, _ = _simulate(n_frames, sg.K_IMAGES, sg.KEEP_IMAGES)
                self.assertEqual(w.evicted_at, points)
                self.assertEqual(w.evicted_at, rt.keep_text_eviction_points(n_frames))
                self.assertEqual(w.liveness(), live)

    def test_text_history_is_never_truncated(self) -> None:
        w = _episode(13)
        self.assertEqual(w.actions, [_action(i) for i in range(12)])
        self.assertEqual(len(w.actions), len(w.frames) - 1)

    def test_current_frame_is_always_live(self) -> None:
        w = rt.KeepTextWindow(_FakeFrame("f000"))
        for i in range(1, 15):
            w.append_turn(_action(i - 1), _FakeFrame(f"f{i:03d}"))
            self.assertIsNotNone(w.frames[-1])
            self.assertEqual(len(w.actions), len(w.frames) - 1)

    def test_custom_k_and_keep(self) -> None:
        w = _episode(9, k=4, keep=2)
        self.assertEqual(w.evicted_at, rt.keep_text_eviction_points(9, 4, 2))
        self.assertEqual(w.evicted_at, [4, 7])
        self.assertEqual(w.live_count(), 3)
        self.assertEqual(w.liveness(), [False] * 6 + [True] * 3)

    def test_frame_labels_cover_the_whole_episode(self) -> None:
        self.assertEqual(
            _episode(4).frame_labels(),
            ["step_000.png", "step_001.png", "step_002.png", "step_003.png"],
        )
        self.assertEqual(rt.keep_text_frame_labels(1), ["step_000.png"])
        with self.assertRaises(ValueError):
            rt.keep_text_frame_labels(0)

    def test_rejects_bad_turns_and_windows(self) -> None:
        w = rt.KeepTextWindow(_FakeFrame("f000"))
        with self.assertRaises(ValueError):
            w.append_turn("", _FakeFrame("f001"))
        with self.assertRaises(ValueError):
            w.append_turn(None, _FakeFrame("f001"))
        with self.assertRaises(ValueError):
            w.append_turn(_action(0), None)
        self.assertEqual((len(w.frames), w.actions), (1, []))
        with self.assertRaises(ValueError):
            rt.KeepTextWindow(None)
        with self.assertRaises(ValueError):
            rt.KeepTextWindow(_FakeFrame("f000"), k=3, keep=3)


class KeepTextAssemblyTests(unittest.TestCase):
    def _messages(self, n_frames: int = 9) -> tuple[rt.KeepTextWindow, list[dict]]:
        w = _episode(n_frames)
        return w, rt.build_keep_text_messages(
            system_prompt="SYS",
            goal=_GOAL,
            frames=w.frames,
            actions=w.actions,
        )

    def test_nine_frame_episode_turn_layout(self) -> None:
        w, messages = self._messages()
        self.assertEqual(len(messages), 1 + 9 + 8)
        self.assertEqual(messages[0], {"role": "system", "content": "SYS"})
        self.assertEqual(
            [m["role"] for m in messages[1:]],
            ["user", "assistant"] * 8 + ["user"],
        )
        self.assertEqual(
            [m["content"] for m in messages if m["role"] == "assistant"],
            [_action(i) for i in range(8)],
        )
        self.assertEqual(w.liveness(), [False] * 4 + [True] * 5)

    def test_placeholders_land_exactly_on_evicted_turns(self) -> None:
        w, messages = self._messages()
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        placeholder = {"type": "text", "text": sg.IMAGE_PLACEHOLDER}
        for i, (content, live) in enumerate(zip(user_turns, w.liveness(), strict=True)):
            with self.subTest(frame=i):
                part = content[-1]
                if live:
                    self.assertEqual(_frame_tag(part), f"f{i:03d}")
                else:
                    self.assertEqual(part, placeholder)
        self.assertEqual(
            sum(c[-1] == placeholder for c in user_turns), 4,
        )
        self.assertEqual(
            sum("image_url" in c[-1] for c in user_turns), sg.K_IMAGES - 1,
        )

    def test_goal_rides_the_first_user_turn_only(self) -> None:
        _, messages = self._messages()
        first = messages[1]["content"]
        self.assertEqual(first[0], {"type": "text", "text": _GOAL})
        self.assertEqual(len(first), 2)
        for m in messages[2:]:
            if m["role"] == "user":
                self.assertEqual(len(m["content"]), 1)
        texts = [
            p["text"]
            for m in messages
            if isinstance(m["content"], list)
            for p in m["content"]
            if p.get("type") == "text"
        ]
        self.assertEqual(texts.count(_GOAL), 1)

    def test_current_frame_is_the_last_message(self) -> None:
        w, messages = self._messages()
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(_frame_tag(messages[-1]["content"][-1]), "f008")
        self.assertEqual(w.frames[-1].tag, "f008")

    def test_frames_and_actions_stay_aligned_across_evictions(self) -> None:
        for n_frames in (1, 2, 6, 7, 9, 11, 14):
            with self.subTest(n_frames=n_frames):
                w = _episode(n_frames)
                messages = rt.build_keep_text_messages(
                    system_prompt="SYS", goal=_GOAL, frames=w.frames, actions=w.actions,
                )
                users = [m for m in messages if m["role"] == "user"]
                assistants = [m for m in messages if m["role"] == "assistant"]
                self.assertEqual(len(users), n_frames)
                self.assertEqual(len(assistants), n_frames - 1)
                self.assertEqual(
                    [m["content"] for m in assistants],
                    [_action(i) for i in range(n_frames - 1)],
                )
                self.assertLessEqual(
                    sum("image_url" in u["content"][-1] for u in users), sg.K_IMAGES,
                )

    def test_no_goal_leaves_the_first_turn_image_only(self) -> None:
        w = _episode(2)
        messages = rt.build_keep_text_messages(
            system_prompt="SYS", goal=None, frames=w.frames, actions=w.actions,
        )
        self.assertEqual(_frame_tag(messages[1]["content"][0]), "f000")
        self.assertEqual(len(messages[1]["content"]), 1)

    def test_builder_reuses_the_same_assembly_with_path_refs(self) -> None:
        w = _episode(9)
        refs = [
            None if f is None else {"type": "image", "image": f"/root/{f.tag}.png"}
            for f in w.frames
        ]
        record = rt.keep_text_messages("SYS", _GOAL, refs, w.actions)
        sent = rt.build_keep_text_messages(
            system_prompt="SYS", goal=_GOAL, frames=w.frames, actions=w.actions,
        )
        self.assertEqual([m["role"] for m in record], [m["role"] for m in sent])
        self.assertEqual(
            [m["content"] for m in record if m["role"] == "assistant"],
            [m["content"] for m in sent if m["role"] == "assistant"],
        )
        for a, b in zip(record, sent, strict=True):
            if isinstance(a["content"], list):
                self.assertEqual(
                    [p for p in a["content"] if p.get("type") == "text"],
                    [p for p in b["content"] if p.get("type") == "text"],
                )

    def test_record_shape_allows_a_trailing_terminate_turn(self) -> None:
        w = _episode(9)
        record = rt.build_keep_text_messages(
            system_prompt="SYS",
            goal=_GOAL,
            frames=w.frames,
            actions=w.actions + [sg.TERMINATE_LINE],
        )
        self.assertEqual(len(record), 1 + 9 + 9)
        self.assertEqual(record[-1], {"role": "assistant", "content": sg.TERMINATE_LINE})
        self.assertEqual(record[-2]["role"], "user")
        self.assertEqual(_frame_tag(record[-2]["content"][-1]), "f008")
        self.assertEqual(
            record[:-1],
            rt.build_keep_text_messages(
                system_prompt="SYS", goal=_GOAL, frames=w.frames, actions=w.actions,
            ),
        )

    def test_assembly_rejects_misaligned_or_empty_input(self) -> None:
        parts = [{"type": "image", "image": "a"}, {"type": "image", "image": "b"}]
        with self.assertRaises(ValueError):
            rt.keep_text_messages("SYS", _GOAL, parts, ["one", "two", "three"])
        with self.assertRaises(ValueError):
            rt.keep_text_messages("SYS", _GOAL, parts, [])
        with self.assertRaises(ValueError):
            rt.keep_text_messages("SYS", _GOAL, [], [])


class KeepTextLoggableTests(unittest.TestCase):
    def test_loggable_twin_matches_the_sent_shape(self) -> None:
        w = _episode(9)
        sent = rt.build_keep_text_messages(
            system_prompt="SYS", goal=_GOAL, frames=w.frames, actions=w.actions,
        )
        logged = rt.build_loggable_keep_text_messages(
            system_prompt="SYS",
            goal=_GOAL,
            actions=w.actions,
            frame_labels=w.frame_labels(),
            liveness=w.liveness(),
        )
        self.assertEqual([m["role"] for m in logged], [m["role"] for m in sent])
        for a, b in zip(logged, sent, strict=True):
            if isinstance(a["content"], list):
                self.assertEqual(len(a["content"]), len(b["content"]))
                self.assertEqual(
                    [p for p in a["content"] if p.get("type") == "text"],
                    [p for p in b["content"] if p.get("type") == "text"],
                )
            else:
                self.assertEqual(a, b)

    def test_loggable_labels_live_frames_and_keeps_placeholders(self) -> None:
        w = _episode(9)
        logged = rt.build_loggable_keep_text_messages(
            system_prompt="SYS",
            goal=_GOAL,
            actions=w.actions,
            frame_labels=w.frame_labels(),
            liveness=w.liveness(),
        )
        users = [m["content"] for m in logged if m["role"] == "user"]
        self.assertEqual(
            [c[-1] for c in users[:4]],
            [{"type": "text", "text": sg.IMAGE_PLACEHOLDER}] * 4,
        )
        self.assertEqual(
            [c[-1]["image"] for c in users[4:]],
            [f"<image step_{i:03d}.png>" for i in range(4, 9)],
        )

    def test_loggable_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            rt.build_loggable_keep_text_messages(
                system_prompt="SYS",
                goal=_GOAL,
                actions=["a"],
                frame_labels=["step_000.png", "step_001.png"],
                liveness=[True],
            )


class _ChatResponse:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.content = content
        self.finish_reason = finish_reason

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {"message": {"content": self.content}, "finish_reason": self.finish_reason}
            ]
        }


class _PostRecorder:
    def __init__(self, content: str = "TERMINATE", finish_reason: str = "stop") -> None:
        self.calls: list[dict] = []
        self.content = content
        self.finish_reason = finish_reason

    def __call__(self, url, *, headers, json, timeout) -> _ChatResponse:
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _ChatResponse(self.content, self.finish_reason)


_LEGACY_PAYLOAD_KEYS = {
    "model", "messages", "max_tokens", "temperature", "top_p", "top_k",
    "repetition_penalty", "presence_penalty",
}


class SeedInjectionTests(unittest.TestCase):
    def _call(self, **kw) -> tuple[_PostRecorder, tuple[str, str | None]]:
        post = _PostRecorder()
        with patch.object(rt.requests, "post", post):
            out = rt._call_model(
                sglang_url="http://sg/v1/",
                api_key="key",
                model="qwen",
                system_prompt="SYS",
                instruction=_GOAL,
                recent_frames=[],
                sampling=sampling.qwen_sampling("instruct", max_tokens=64),
                **kw,
            )
        return post, out

    def test_seed_is_forwarded(self) -> None:
        post, (content, finish_reason) = self._call(seed=17)
        self.assertEqual(post.calls[0]["json"]["seed"], 17)
        self.assertEqual((content, finish_reason), ("TERMINATE", "stop"))
        self.assertEqual(post.calls[0]["url"], "http://sg/v1/chat/completions")
        self.assertEqual(post.calls[0]["headers"], {"Authorization": "Bearer key"})

    def test_seed_zero_is_forwarded_not_dropped(self) -> None:
        post, _ = self._call(seed=0)
        self.assertEqual(post.calls[0]["json"]["seed"], 0)

    def test_seed_defaults_to_absent_legacy_payload(self) -> None:
        post, _ = self._call()
        self.assertEqual(set(post.calls[0]["json"]), _LEGACY_PAYLOAD_KEYS)
        self.assertEqual(
            post.calls[0]["json"]["messages"], [{"role": "system", "content": "SYS"}]
        )

    def test_greedy_payload_keeps_sampling_module_as_source_of_truth(self) -> None:
        post = _PostRecorder()
        with patch.object(rt.requests, "post", post):
            rt._call_model(
                sglang_url="http://sg/v1",
                api_key="key",
                model="qwen",
                system_prompt="SYS",
                instruction=None,
                recent_frames=[],
                sampling=sampling.qwen_sampling("instruct", max_tokens=8, greedy=True),
                seed=3,
            )
        self.assertEqual(
            post.calls[0]["json"],
            {
                "model": "qwen",
                "messages": [{"role": "system", "content": "SYS"}],
                "max_tokens": 8,
                "temperature": 0.0,
                "seed": 3,
            },
        )

    def test_call_model_messages_sends_prebuilt_keep_text_messages(self) -> None:
        w = _episode(9)
        messages = rt.build_keep_text_messages(
            system_prompt="SYS", goal=_GOAL, frames=w.frames, actions=w.actions,
        )
        post = _PostRecorder(content="move_to(500,500); down(LMB); up(LMB)")
        with patch.object(rt.requests, "post", post):
            content, finish_reason = rt.call_model_messages(
                sglang_url="http://sg/v1",
                api_key="key",
                model="qwen",
                messages=messages,
                sampling=sampling.qwen_sampling("instruct", max_tokens=64),
                seed=5,
            )
        sent = post.calls[0]["json"]
        self.assertEqual(sent["messages"], messages)
        self.assertEqual(sent["seed"], 5)
        self.assertEqual(set(sent), _LEGACY_PAYLOAD_KEYS | {"seed"})
        self.assertEqual((content, finish_reason), (post.content, "stop"))

    def test_call_model_messages_omits_seed_when_unset(self) -> None:
        post = _PostRecorder()
        with patch.object(rt.requests, "post", post):
            rt.call_model_messages(
                sglang_url="http://sg/v1",
                api_key="key",
                model="qwen",
                messages=[{"role": "system", "content": "SYS"}],
                sampling=sampling.qwen_sampling("instruct", max_tokens=64),
            )
        self.assertNotIn("seed", post.calls[0]["json"])


class _ExecResponse:
    def __init__(self, payload: dict, *, http_error: Exception | None = None) -> None:
        self.payload = payload
        self.http_error = http_error

    def raise_for_status(self) -> None:
        if self.http_error is not None:
            raise self.http_error

    def json(self) -> dict:
        return self.payload


class _ExecSession:
    def __init__(self, response: _ExecResponse) -> None:
        self.response = response
        self.posts: list[dict] = []

    def post(self, url, *, json, timeout) -> _ExecResponse:
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return self.response


class _ExecClient(osworld_vm_client.OSWorldClient):
    """Client whose /execute session returns one canned agent result."""

    def __init__(self, response: _ExecResponse) -> None:
        super().__init__("http://fake")
        self._sess = _ExecSession(response)


def _ok(**extra) -> _ExecResponse:
    return _ExecResponse({"status": "success", "returncode": 0, "output": "", **extra})


class RunCommandTests(unittest.TestCase):
    def test_returns_the_structured_result_and_posts_a_list_command(self) -> None:
        client = _ExecClient(_ok(output="hello\n"))
        result = client.run_command(["echo", "hello"])
        self.assertEqual(result["output"], "hello\n")
        self.assertEqual(
            client._sess.posts,
            [{
                "url": "http://fake/execute",
                "json": {"command": ["echo", "hello"], "shell": False},
                "timeout": client.timeout,
            }],
        )

    def test_shell_string_command_passes_through(self) -> None:
        client = _ExecClient(_ok())
        client.run_command("test -f /root/Desktop/report.txt", shell=True)
        self.assertEqual(
            client._sess.posts[0]["json"],
            {"command": "test -f /root/Desktop/report.txt", "shell": True},
        )

    def test_nonzero_return_code_raises(self) -> None:
        client = _ExecClient(
            _ExecResponse({"status": "success", "returncode": 1, "error": "no such file"})
        )
        with self.assertRaises(RuntimeError) as ctx:
            client.run_command(["test", "-f", "/nope"])
        self.assertIn("rc=1", str(ctx.exception))
        self.assertIn("no such file", str(ctx.exception))

    def test_failed_status_raises_even_with_rc_zero(self) -> None:
        client = _ExecClient(
            _ExecResponse({"status": "error", "returncode": 0, "message": "agent died"})
        )
        with self.assertRaises(RuntimeError) as ctx:
            client.run_command(["true"])
        self.assertIn("agent died", str(ctx.exception))

    def test_missing_return_code_is_treated_as_zero(self) -> None:
        client = _ExecClient(_ExecResponse({"status": "success", "output": "x"}))
        self.assertEqual(client.run_command(["true"])["output"], "x")

    def test_http_error_propagates(self) -> None:
        client = _ExecClient(
            _ExecResponse({"status": "success", "returncode": 0}, http_error=OSError("500"))
        )
        with self.assertRaises(OSError):
            client.run_command(["true"])

    def test_execute_keeps_its_legacy_pyautogui_wrapper_and_no_rc_check(self) -> None:
        client = _ExecClient(_ExecResponse({"status": "error", "returncode": 1}))
        client.execute("pyautogui.moveTo(1, 2)")
        self.assertEqual(
            client._sess.posts[0]["json"],
            {
                "command": [
                    "python",
                    "-c",
                    osworld_vm_client.OSWorldClient._PYAUTOGUI_PREFIX
                    + "pyautogui.moveTo(1, 2)",
                ],
                "shell": False,
            },
        )


class _FakeVMClient(osworld_vm_client.OSWorldClient):
    """Records execute() calls; simulates cursor tracking via moveTo."""

    def __init__(self, *, pos=(100, 100), screen=(1920, 1080)) -> None:
        super().__init__("http://fake")
        self._pos = pos
        self._screen = screen
        self.commands: list[str] = []

    def cursor_position(self):  # type: ignore[override]
        return self._pos

    def screen_size(self):  # type: ignore[override]
        return self._screen

    def execute(self, command: str) -> None:  # type: ignore[override]
        self.commands.append(command)


def _move_to(x: int, y: int) -> OrderedAction:
    return OrderedAction(
        primitives=(OrderedPrimitive(kind="move_to", x=x, y=y),), no_op=False,
    )


class MoveToDispatchTests(unittest.TestCase):
    def test_click_after_move_to_lands_at_the_tracked_cursor(self) -> None:
        client = _FakeVMClient()
        action = sg.denorm_v4(
            parse_ordered_v4_action("move_to(500,500); down(LMB); up(LMB)", arm=sg.ARM_ABS),
            (1920, 1080),
        )
        sr = client.dispatch_ordered_action(action)
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(960, 540)",
            "pyautogui.mouseDown(button='left')",
            "pyautogui.mouseUp(button='left')",
        ])
        self.assertEqual(sr.intended_target, (960, 540))
        self.assertEqual(sr.delta, (860, 440))
        self.assertTrue(sr.parse_ok)

    def test_pixels_in_no_hidden_scaling(self) -> None:
        client = _FakeVMClient()
        client.dispatch_ordered_action(_move_to(37, 941))
        self.assertEqual(client.commands, ["pyautogui.moveTo(37, 941)"])

    def test_target_is_clipped_to_the_screen(self) -> None:
        client = _FakeVMClient(screen=(1920, 1080))
        client.dispatch_ordered_action(_move_to(5000, 5000))
        self.assertEqual(client.commands, ["pyautogui.moveTo(1919, 1079)"])
        client.commands.clear()
        client.dispatch_ordered_action(_move_to(-5, -5))
        self.assertEqual(client.commands, ["pyautogui.moveTo(0, 0)"])

    def test_move_to_current_position_dispatches_nothing(self) -> None:
        client = _FakeVMClient(pos=(640, 480))
        sr = client.dispatch_ordered_action(_move_to(640, 480))
        self.assertEqual(client.commands, [])
        self.assertEqual(sr.intended_target, (640, 480))
        self.assertEqual(sr.delta, (0, 0))

    def test_cursor_tracks_across_several_move_tos(self) -> None:
        client = _FakeVMClient(pos=(0, 0))
        action = OrderedAction(
            primitives=(
                OrderedPrimitive(kind="move_to", x=10, y=20),
                OrderedPrimitive(kind="down", name="LMB", mouse_button=1),
                OrderedPrimitive(kind="move_to", x=30, y=25),
                OrderedPrimitive(kind="up", name="LMB", mouse_button=1),
            ),
            no_op=False,
        )
        sr = client.dispatch_ordered_action(action)
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(10, 20)",
            "pyautogui.mouseDown(button='left')",
            "pyautogui.moveTo(30, 25)",
            "pyautogui.mouseUp(button='left')",
        ])
        self.assertEqual(sr.intended_target, (30, 25))
        self.assertEqual(sr.delta, (30, 25))

    def test_missing_coordinate_raises(self) -> None:
        client = _FakeVMClient()
        for prim in (
            OrderedPrimitive(kind="move_to", x=5),
            OrderedPrimitive(kind="move_to", y=5),
        ):
            with self.subTest(prim=prim):
                with self.assertRaises(ValueError):
                    client.dispatch_ordered_action(
                        OrderedAction(primitives=(prim,), no_op=False)
                    )

    def test_legacy_move_branch_is_unchanged(self) -> None:
        client = _FakeVMClient()
        sr = client.dispatch_ordered_action(
            parse_ordered_action("move(12,-4); down(LMB); up(LMB)")
        )
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(112, 96)",
            "pyautogui.mouseDown(button='left')",
            "pyautogui.mouseUp(button='left')",
        ])
        self.assertEqual(sr.delta, (12, -4))

    def test_no_op_still_dispatches_nothing(self) -> None:
        client = _FakeVMClient()
        sr = client.dispatch_ordered_action(parse_ordered_action("NO_OP"))
        self.assertEqual((client.commands, sr.action_text), ([], "NO_OP"))


if __name__ == "__main__":
    unittest.main()
