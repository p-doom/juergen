"""THE short-goal train/eval contract: runtime keep-text prompt == builder record.

One synthetic 9-frame episode (8 golden turns plus the closing ``TERMINATE``, so
exactly one eviction and two records) is driven three times. Once through
``shortgoal_eval.run_episode`` itself — the real closed loop, on a stub model that
replies with the golden lines and a stub VM client that serves the recording's own
frames — which is the only replay that also covers how the evaluator uses the
window (frame scaling, what text is appended, when the screenshot arrives, the
JPEG bytes it sends). Once as a hand-rolled ``KeepTextWindow`` replay, a second
opinion on the runtime helpers alone. Once through ``shortgoal_build``, whose
records are read back off disk as the trainer will see them. Every decision
context, plus the reply that followed it, must be identical to the prefix of the
OWNING record that ends at that assistant turn — same roles, same text bytes, same
GOAL placement, the same ``<Image collapsed>`` literals in the same
positions, the same frames in the same turns — for both records and both arms.

The only permitted difference is how a frame is carried: the runtime inlines a
base64 JPEG data URL, a record references the materialized ``step_NNN.jpg`` by
path, and the persisted prompt trace names it ``<image step_003.png>``. The
evaluator replay compares the DECODED data URL against the record's file bytes, so
a frame that reaches the model at another resolution or JPEG quality than training
fails here; the other two replays reduce a frame to its identity (``step_003``).
Anything else that differs means every assistant turn trains under a context the
evaluator never builds, which no parser or grammar check can catch.
"""

from __future__ import annotations

import base64
import json
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import osworld_runtime as rt  # noqa: E402
import shortgoal_build as build  # noqa: E402
import shortgoal_eval as se  # noqa: E402
import shortgoal_record as sr  # noqa: E402
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402
from shortgoal_grammar import (  # noqa: E402
    ARMS,
    FRAME_JPEG_QUALITY,
    IMAGE_PLACEHOLDER,
    K_IMAGES,
    KEEP_IMAGES,
    PROMPT_IDS,
    TERMINATE_LINE,
)
from test_shortgoal_build import (  # noqa: E402
    RESOLUTION,
    SCREEN,
    SOURCE_FRAME_WH,
    SPLITS,
    TASK_ID,
    write_recording,
)

N_STEPS = 8
N_FRAMES = N_STEPS + 1
_DATA_URL_PREFIX = "data:image/jpeg;base64,"
_LEGACY_JPEG_QUALITY = 85
_LABEL_PREFIX = "<image "
_LABEL_SUFFIX = ">"


class _FakeFrame:
    """PIL-Image stand-in whose JPEG bytes are its own frame id."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def save(self, buf, **kwargs) -> None:
        buf.write(self.tag.encode("ascii"))


def _frame_id(index: int) -> str:
    return build.FRAME_STEM.format(index)


def _data_url_bytes(url: str) -> bytes:
    if not url.startswith(_DATA_URL_PREFIX):
        raise AssertionError(f"not a JPEG data URL: {url[:40]!r}")
    return base64.b64decode(url[len(_DATA_URL_PREFIX):])


def _block(block: dict[str, Any]) -> tuple[str, str]:
    if block["type"] == "text":
        return ("text", block["text"])
    if block["type"] == "image":
        label = block.get("url")
        if label is None:
            label = block["image"].removeprefix(_LABEL_PREFIX).removesuffix(_LABEL_SUFFIX)
        return ("image", Path(label).stem)
    return ("image", _data_url_bytes(block["image_url"]["url"]).decode("ascii"))


def _frame_bytes(block: dict[str, Any]) -> tuple[str, Any]:
    """A frame reduced to the JPEG bytes it carries, whichever way it is carried."""
    if block["type"] == "text":
        return ("text", block["text"])
    if block["type"] == "image":
        return ("image", Path(block["url"]).read_bytes())
    return ("image", _data_url_bytes(block["image_url"]["url"]))


def _turns(
    messages: list[dict[str, Any]], part: Callable[[dict[str, Any]], tuple[str, Any]],
) -> list[tuple[str, tuple]]:
    canonical: list[tuple[str, tuple]] = []
    for message in messages:
        content = message["content"]
        blocks = (
            (("text", content),) if isinstance(content, str)
            else tuple(part(block) for block in content)
        )
        canonical.append((message["role"], blocks))
    return canonical


def _canonical(messages: list[dict[str, Any]]) -> list[tuple[str, tuple]]:
    """``(role, blocks)`` per turn; a frame reduces to its id in every serialization."""
    return _turns(messages, _block)


def _canonical_bytes(messages: list[dict[str, Any]]) -> list[tuple[str, tuple]]:
    """``(role, blocks)`` per turn; a frame reduces to its exact JPEG bytes."""
    return _turns(messages, _frame_bytes)


def _runtime_contexts(
    prompt: str, goal: str, lines: tuple[str, ...],
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    """Replay the keep-text loop by hand; one context per decision, plus live counts.

    The runtime helpers on their own, without the evaluator: assemble the context,
    take the reply, append ``(reply, next frame)`` to the window, stop after the
    reply to the post-success frame (``TERMINATE``). ``EvaluatorContractTests``
    drives ``shortgoal_eval.run_episode`` itself over the same episode.
    """
    window = rt.KeepTextWindow(_FakeFrame(_frame_id(0)))
    contexts: list[list[dict[str, Any]]] = []
    live_counts: list[int] = []
    for index, line in enumerate(lines):
        contexts.append(rt.build_keep_text_messages(
            system_prompt=prompt,
            goal=goal,
            frames=window.frames,
            actions=window.actions,
        ))
        live_counts.append(window.live_count())
        if index < len(lines) - 1:
            window.append_turn(line, _FakeFrame(_frame_id(index + 1)))
    return contexts, live_counts


def _arm_invariant(messages: list[dict[str, Any]], arm: str) -> list[tuple[str, tuple]]:
    """A canonical turn list with the two arm-divergent parts masked away."""
    invariant: list[tuple[str, tuple]] = []
    for role, blocks in _canonical(messages):
        if role == "system":
            invariant.append((role, (("text", build.SYSTEM_MASK),)))
        elif role == "assistant":
            invariant.append((role, (("text", build.masked_line(blocks[0][1], arm)),)))
        else:
            invariant.append((role, blocks))
    return invariant


def _owner(starts: list[int], decision: int) -> int:
    return max(index for index, start in enumerate(starts) if start <= decision)


class _ReplayClient(se._StubClient):
    """The evaluator's own stub VM client, serving the recording's frames in order."""

    def __init__(self, frames: tuple[Path, ...]) -> None:
        super().__init__(screen_wh=SCREEN, frame_wh=SOURCE_FRAME_WH)
        self.frame_paths = list(frames)

    def screenshot(self) -> Image.Image:
        path = self.frame_paths[self.shots]
        self.shots += 1
        with Image.open(path) as raw:
            return raw.convert("RGB")


class _ReplayModel(se._StubModel):
    """The evaluator's own stub model, keeping every context it was really called with."""

    def __init__(self, replies: tuple[str, ...]) -> None:
        super().__init__(replies)
        self.contexts: list[list[dict[str, Any]]] = []

    def __call__(
        self, messages: list[dict[str, Any]], *, seed: int | None = None,
    ) -> tuple[str, str | None]:
        self.contexts.append(messages)
        return super().__call__(messages, seed=seed)


class _BuiltEpisode(unittest.TestCase):
    """Shared setUp: the synthetic 9-frame episode, built for both arms."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self.root = root
        self.recordings = root / "recordings"
        write_recording(self.recordings, n_steps=N_STEPS)
        self.recording = build.load_recording(self.recordings, TASK_ID)
        self.goal = build.GOAL_PREFIX + self.recording.instruction
        self.points = rt.keep_text_eviction_points(N_FRAMES)
        self.starts = [0, *self.points]
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.lines: dict[str, tuple[str, ...]] = {}
        for arm in ARMS:
            build.build_arm(
                recordings_root=self.recordings,
                output_root=root / build.ARM_SLUGS[arm],
                arm=arm,
                subset="overfit1",
                splits=SPLITS,
                resolution=RESOLUTION,
                allow_resupervision=True,
            )
            self.records[arm] = build.read_records(
                root / build.ARM_SLUGS[arm] / build.CHAT_RELPATH
            )
            self.lines[arm] = build.episode_lines(self.recording, arm)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _prompt(self, arm: str) -> str:
        return SYSTEM_PROMPTS[PROMPT_IDS[arm]]

    def _record_for(self, arm: str, decision: int) -> dict[str, Any]:
        return self.records[arm][_owner(self.starts, decision)]

    def _prefix(self, arm: str, decision: int) -> list[dict[str, Any]]:
        return self._record_for(arm, decision)["messages"][: 2 * decision + 3]


class KeepTextContractTests(_BuiltEpisode):
    def test_the_episode_spans_exactly_one_eviction_and_two_records(self) -> None:
        self.assertEqual(self.recording.n_frames, N_FRAMES)
        self.assertEqual(self.points, [K_IMAGES])
        for arm in ARMS:
            with self.subTest(arm=arm):
                self.assertEqual(len(self.records[arm]), 2)
                self.assertEqual([r["n_frames"] for r in self.records[arm]], [6, 9])
                self.assertEqual([r["n_live_images"] for r in self.records[arm]], [6, KEEP_IMAGES + 2])
                self.assertEqual(len(self.lines[arm]), N_FRAMES)
                self.assertEqual(self.lines[arm][-1], TERMINATE_LINE)

    def test_every_decision_context_is_the_owning_record_prefix(self) -> None:
        for arm in ARMS:
            contexts, _ = _runtime_contexts(self._prompt(arm), self.goal, self.lines[arm])
            self.assertEqual(len(contexts), N_FRAMES)
            owned: dict[int, list[int]] = {}
            for decision, context in enumerate(contexts):
                index = _owner(self.starts, decision)
                record = self.records[arm][index]
                owned.setdefault(index, []).append(decision)
                reply = {"role": "assistant", "content": self.lines[arm][decision]}
                with self.subTest(arm=arm, decision=decision, record=index):
                    self.assertGreater(record["n_frames"], decision)
                    self.assertEqual(
                        _canonical([*context, reply]),
                        _canonical(record["messages"][: 2 * decision + 3]),
                    )
            self.assertEqual(owned, {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8]})

    def test_goal_and_placeholder_literals_land_identically(self) -> None:
        placeholder = ("text", IMAGE_PLACEHOLDER)
        for arm in ARMS:
            contexts, _ = _runtime_contexts(self._prompt(arm), self.goal, self.lines[arm])
            for decision, context in enumerate(contexts):
                record = self.records[arm][_owner(self.starts, decision)]
                runtime_users = [t for t in _canonical(context) if t[0] == "user"]
                record_users = [
                    t for t in _canonical(record["messages"][: 2 * decision + 3])
                    if t[0] == "user"
                ]
                with self.subTest(arm=arm, decision=decision):
                    self.assertEqual(runtime_users, record_users)
                    self.assertEqual(runtime_users[0][1][0], ("text", self.goal))
                    self.assertTrue(all(
                        ("text", self.goal) not in turn[1] for turn in runtime_users[1:]
                    ))
                    self.assertEqual(
                        [i for i, t in enumerate(runtime_users) if placeholder in t[1]],
                        [0, 1, 2, 3] if decision >= K_IMAGES else [],
                    )
                    self.assertEqual(runtime_users[-1][1][-1], ("image", _frame_id(decision)))

    def test_live_image_counts_agree_across_the_eviction(self) -> None:
        for arm in ARMS:
            _, live_counts = _runtime_contexts(self._prompt(arm), self.goal, self.lines[arm])
            with self.subTest(arm=arm):
                self.assertEqual(live_counts, [1, 2, 3, 4, 5, 6, 3, 4, 5])
                self.assertLessEqual(max(live_counts), K_IMAGES)
                for index, record in enumerate(self.records[arm]):
                    last = self.starts[index + 1] - 1 if index + 1 < len(self.starts) else N_FRAMES - 1
                    self.assertEqual(record["n_live_images"], live_counts[last])

    def test_second_record_reopens_with_the_full_text_history(self) -> None:
        for arm in ARMS:
            first, second = self.records[arm]
            first_actions = [m["content"] for m in first["messages"] if m["role"] == "assistant"]
            second_actions = [m["content"] for m in second["messages"] if m["role"] == "assistant"]
            with self.subTest(arm=arm):
                self.assertEqual(second_actions[: len(first_actions)], first_actions)
                self.assertEqual(second_actions, list(self.lines[arm]))
                self.assertEqual(second_actions[-1], TERMINATE_LINE)
                first_users = [t for t in _canonical(first["messages"]) if t[0] == "user"]
                second_users = [t for t in _canonical(second["messages"]) if t[0] == "user"]
                self.assertTrue(all(
                    ("text", IMAGE_PLACEHOLDER) not in turn[1] for turn in first_users
                ))
                self.assertEqual(
                    [("text", IMAGE_PLACEHOLDER) in turn[1] for turn in second_users],
                    [True] * 4 + [False] * 5,
                )

    def test_both_arms_share_the_whole_turn_structure(self) -> None:
        structures = {}
        for arm in ARMS:
            contexts, _ = _runtime_contexts(self._prompt(arm), self.goal, self.lines[arm])
            structures[arm] = [_arm_invariant(context, arm) for context in contexts]
        self.assertEqual(structures[ARMS[0]], structures[ARMS[1]])
        self.assertEqual(
            build.check_arms_identity(
                Path(self._tmp.name) / build.ARM_SLUGS[ARMS[0]] / build.CHAT_RELPATH,
                Path(self._tmp.name) / build.ARM_SLUGS[ARMS[1]] / build.CHAT_RELPATH,
            )["n_records"],
            2,
        )

    def test_a_drifted_runtime_context_is_caught(self) -> None:
        arm = ARMS[0]
        contexts, _ = _runtime_contexts(self._prompt(arm), self.goal, self.lines[arm])
        drifted = rt.build_keep_text_messages(
            system_prompt=self._prompt(arm),
            goal=None,
            frames=[_FakeFrame(_frame_id(0))],
            actions=[],
        )
        self.assertNotEqual(_canonical(drifted), _canonical(contexts[0]))
        reply = {"role": "assistant", "content": self.lines[arm][0]}
        self.assertNotEqual(
            _canonical([*drifted, reply]),
            _canonical(self.records[arm][0]["messages"][:3]),
        )


class EvaluatorContractTests(_BuiltEpisode):
    """The same identity, driven through ``shortgoal_eval.run_episode`` itself."""

    def _replay(self, arm: str) -> tuple[_ReplayModel, se.Episode]:
        task = se.task_of(TASK_ID)
        recording = se.load_recording(self.recordings, TASK_ID)
        se.check_recording(recording, task)
        model = _ReplayModel(self.lines[arm])
        episode, _ = se.run_episode(
            task=task,
            tier="train",
            arm=arm,
            attempt=0,
            seed=None,
            client=_ReplayClient(self.recording.frames),
            call=model,
            system_prompt=self._prompt(arm),
            out_dir=self.root / f"episode_{arm}",
            max_steps=N_FRAMES,
            settle=sr.Settle(delay_s=0.0, stable_timeout_s=0.0, poll_s=0.0),
            model_resolution=RESOLUTION,
            jpeg_quality=build.DEFAULT_JPEG_QUALITY,
            save_frames=True,
            setup=lambda _client, _task, screen_wh: {"screen_wh": list(screen_wh)},
            verify=lambda _client, run, state, *, timeout_s: {
                "kind": run.verifier_id, "passed": True, "detail": state,
            },
            verify_timeout_s=1.0,
            recording=recording,
        )
        return model, episode

    def test_the_evaluator_sends_the_owning_record_prefix_byte_for_byte(self) -> None:
        for arm in ARMS:
            model, episode = self._replay(arm)
            with self.subTest(arm=arm):
                self.assertEqual(len(model.contexts), N_FRAMES)
                self.assertEqual(episode.stop_reason, se._STOP_TERMINATE)
                self.assertEqual(episode.steps_used, N_FRAMES)
                self.assertEqual(episode.blind_history_steps, N_FRAMES - self.starts[1])
                self.assertTrue(episode.success)
            for decision, context in enumerate(model.contexts):
                reply = {"role": "assistant", "content": self.lines[arm][decision]}
                with self.subTest(arm=arm, decision=decision):
                    self.assertEqual(
                        _canonical_bytes([*context, reply]),
                        _canonical_bytes(self._prefix(arm, decision)),
                    )

    def test_the_persisted_prompt_trace_carries_the_same_turns(self) -> None:
        for arm in ARMS:
            _model, episode = self._replay(arm)
            artifacts = Path(episode.artifact_dir)
            trace = [
                json.loads(line)
                for line in (artifacts / "conversation.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["step"] for row in trace], list(range(1, N_FRAMES + 1)))
            for decision, row in enumerate(trace):
                reply = {"role": "assistant", "content": row["response"]}
                with self.subTest(arm=arm, decision=decision):
                    self.assertEqual(
                        _canonical([*row["messages"], reply]),
                        _canonical(self._prefix(arm, decision)),
                    )
            sidecars = sorted((artifacts / "steps").glob("prompt_*.json"))
            with self.subTest(arm=arm):
                self.assertEqual(len(sidecars), N_FRAMES)
                self.assertEqual(
                    [_canonical(json.loads(path.read_text())) for path in sidecars],
                    [_canonical(row["messages"]) for row in trace],
                )

    def test_the_legacy_jpeg_quality_would_break_the_frame_identity(self) -> None:
        arm = ARMS[0]
        model, _ = self._replay(arm)
        trained = _frame_bytes(self._prefix(arm, 0)[1]["content"][-1])
        self.assertEqual(_frame_bytes(model.contexts[0][1]["content"][-1]), trained)
        with Image.open(self.recording.frames[0]) as raw:
            frame = raw.convert("RGB").resize(RESOLUTION, Image.LANCZOS)
        default = rt.build_keep_text_messages(
            system_prompt=self._prompt(arm), goal=self.goal, frames=[frame], actions=[],
        )
        self.assertEqual(_frame_bytes(default[1]["content"][-1]), trained)
        self.assertEqual(
            trained[1],
            _data_url_bytes(rt._pil_to_data_url(frame, quality=FRAME_JPEG_QUALITY)),
        )
        self.assertNotEqual(
            trained[1],
            _data_url_bytes(rt._pil_to_data_url(frame, quality=_LEGACY_JPEG_QUALITY)),
        )


if __name__ == "__main__":
    unittest.main()
