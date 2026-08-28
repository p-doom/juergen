"""`HistoryPolicy` / `History`."""

from __future__ import annotations

import base64

import pytest

from history_policy import replay_training_messages

from agent.history import (
    IMAGE_PLACEHOLDER,
    POLICIES,
    History,
    ImageBudget,
    InterleavedFrames,
    LatestImageOnly,
    ProseSummarisedWindow,
    StatelessSingleTurn,
    history_policy,
    prose_summary,
)
from image_domain import OSWORLD_CURSOR_JPEG_DOMAIN
from juergen_doubles import jpeg, png


def _f(i: int) -> bytes:
    return jpeg(colour=(i % 250, 0, 0))


def _drive(history: History, steps: int, first: bytes | None = None) -> None:
    history.start(first or _f(0))
    for step in range(1, steps + 1):
        history.append(f"action {step}", _f(step))


def _part_type(part) -> str:
    return part.get("type", "") if isinstance(part, dict) else getattr(part, "type", "")


def _text_of(message) -> str:
    """All text on a message, whether pydantic coerced the parts or not."""
    content = message.content
    if isinstance(content, str) or content is None:
        return content or ""
    out = []
    for part in content:
        if _part_type(part) != "text":
            continue
        out.append(part["text"] if isinstance(part, dict) else part.text)
    return "\n".join(out)


def _images(messages) -> int:
    """Image parts on the wire. `vf.UserMessage` coerces dicts into `ContentPart`
    models, so a dict-only count would silently read 0 and pass every budget test."""
    total = 0
    for message in messages:
        content = message.content
        if isinstance(content, list):
            total += sum(1 for part in content if _part_type(part) == "image_url")
    return total


def test_the_window_invariant_holds_at_every_step() -> None:
    history = History(n_history_frames=6)
    history.start(_f(0))
    for step in range(1, 20):
        assert history.turns[-1].output is None, "the newest frame has no action yet"
        assert len(history.outputs) == len(history.turns) - 1
        history.append(f"a{step}", _f(step))


def test_append_before_start_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="before History.start"):
        History().append("a", _f(1))


def test_start_resets_the_window_and_the_evicted_log() -> None:
    history = History(n_history_frames=2)
    _drive(history, 6)
    assert history.evicted
    history.start(_f(99))
    assert history.turns == [history.turns[0]] and history.evicted == []


def test_the_window_slides_one_frame_at_a_time() -> None:
    history = History(n_history_frames=4)
    history.start(_f(0))
    sizes = [len(history.turns)]
    for step in range(1, 9):
        history.append(f"a{step}", _f(step))
        sizes.append(len(history.turns))
    assert sizes == [1, 2, 3, 4, 4, 4, 4, 4, 4]
    assert history.evicted == [f"a{i}" for i in range(1, 6)]
    assert history.all_outputs == [f"a{i}" for i in range(1, 9)], (
        "no action is ever lost: evicted + in-window is the whole history"
    )


def test_offline_replay_matches_the_online_rendered_messages_exactly() -> None:
    turns = [(f"frame-{index}", f"action-{index}") for index in range(6)]
    offline = replay_training_messages(
        turns=turns,
        n_history_frames=4,
        system="SYSTEM",
        instruction="GOAL",
        image_part=lambda image: {"type": "image_url", "image_url": {"url": image}},
    )
    assert [
        message["content"][-1]["image_url"]["url"]
        for message in offline[4]
        if message["role"] == "user"
    ] == ["frame-1", "frame-2", "frame-3", "frame-4"]
    history: History = History(n_history_frames=4)
    history.start(turns[0][0])
    budget = type(
        "Budget",
        (),
        {
            "max_images": 4,
            "image_part": staticmethod(
                lambda image: {"type": "image_url", "image_url": {"url": image}}
            ),
        },
    )()
    for index, (_, target) in enumerate(turns):
        online = InterleavedFrames().render(
            history=history,
            system="SYSTEM",
            instruction="GOAL",
            step=index + 1,
            budget=budget,
        )
        rendered = [
            {
                "role": message.role,
                "content": [
                    part.model_dump() if hasattr(part, "model_dump") else part
                    for part in message.content
                ]
                if isinstance(message.content, list)
                else message.content,
            }
            for message in online
        ]
        assert rendered == offline[index][:-1]
        assert offline[index][-1] == {"role": "assistant", "content": target}
        if index + 1 < len(turns):
            history.append(target, turns[index + 1][0])


@pytest.mark.parametrize("n", [1, 2, 3, 16])
def test_eviction_never_empties_the_window(n: int) -> None:
    history = History(n_history_frames=n)
    _drive(history, 40)
    assert history.turns, f"n_history_frames={n} must keep at least one turn"
    assert len(history.turns) <= max(n, 1)
    assert history.current is history.turns[-1].image
    assert len(history.all_outputs) == 40


def test_interleaved_alternates_user_and_assistant_turns() -> None:
    history = History(n_history_frames=16)
    _drive(history, 2)
    messages = InterleavedFrames().render(
        history=history,
        system="SYS",
        instruction="GOAL",
        step=3,
        budget=ImageBudget(max_images=16),
    )
    roles = [m.role for m in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert _images(messages) == 3


def test_interleaved_respects_the_image_budget_and_keeps_outputs_aligned() -> None:
    history = History(n_history_frames=32)
    _drive(history, 10)  # 11 frames, 10 outputs
    messages = InterleavedFrames().render(
        history=history, system="S", instruction="G", step=11, budget=ImageBudget(max_images=4)
    )
    assert _images(messages) == 4, "the budget is a hard cap"
    assistants = [m for m in messages if m.role == "assistant"]
    assert [m.content for m in assistants] == ["action 8", "action 9", "action 10"], (
        "the outputs kept must be the ones between the frames kept"
    )


def test_interleaved_honours_the_image_budget_after_eviction() -> None:
    history = History(n_history_frames=6)
    _drive(history, 20)
    messages = InterleavedFrames().render(
        history=history, system="S", instruction="G", step=21, budget=ImageBudget(max_images=2)
    )
    assert _images(messages) == 2


def test_interleaved_re_anchors_the_goal_every_step() -> None:
    history = History(n_history_frames=16)
    _drive(history, 3)
    persisted = InterleavedFrames().render(
        history=history, system="S", instruction="GOAL", step=4, budget=ImageBudget()
    )
    assert "GOAL" in _text_of(persisted[1])


def test_prose_window_keeps_five_images_and_summarises_the_rest() -> None:
    history = History(n_history_frames=32)
    _drive(history, 7)  # 8 frames, 7 outputs
    messages = ProseSummarisedWindow().render(
        history=history, system="S", instruction="GOAL", step=8, budget=ImageBudget(max_images=5)
    )
    assert _images(messages) == 5, "the sealed contract is five images"
    first_user = messages[1]
    text = _text_of(first_user)
    assert "Instruction: GOAL" in text
    assert "Previous actions:" in text
    assert "Step 1: action 1" in text and "Step 3: action 3" in text
    assert "Step 4:" not in text, "only actions older than the image window are prose"
    assert _part_type(first_user.content[0]) == "image_url", (
        "the image comes before the text on the first turn, matching the sealed evaluator"
    )


def test_prose_window_says_none_when_nothing_has_been_evicted() -> None:
    history = History(n_history_frames=32)
    _drive(history, 2)
    messages = ProseSummarisedWindow().render(
        history=history, system="S", instruction="G", step=3, budget=ImageBudget(max_images=5)
    )
    assert "Previous actions:\nNone" in _text_of(messages[1])


def test_prose_window_survives_the_window_evicting() -> None:
    history = History(n_history_frames=8)
    _drive(history, 7)
    ProseSummarisedWindow().render(  # 8 frames, no eviction yet
        history=history, system="S", instruction="G", step=8, budget=ImageBudget(max_images=5)
    )
    history.append("action 8", _f(8))
    assert history.evicted, "precondition: the window has evicted"
    messages = ProseSummarisedWindow().render(
        history=history, system="S", instruction="G", step=9, budget=ImageBudget(max_images=5)
    )
    assert _images(messages) == 5


def test_prose_window_pairs_the_right_action_with_each_frame_after_eviction() -> None:
    """The alignment half of the same defect: an off-by-`len(evicted)` would show the
    model somebody else's action under its own screenshot."""
    history = History(n_history_frames=8)
    _drive(history, 20)
    assert history.evicted, "precondition"
    messages = ProseSummarisedWindow().render(
        history=history, system="S", instruction="G", step=21, budget=ImageBudget(max_images=3)
    )
    assistants = [m.content for m in messages if m.role == "assistant"]
    # The window holds the newest frames; the newest frame has no action yet.
    expected = history.outputs[-len(assistants):] if assistants else []
    assert assistants == expected, (assistants, expected)
    assert assistants == ["action 19", "action 20"], assistants


def test_prose_window_summarises_every_action_older_than_the_visible_window() -> None:
    history = History(n_history_frames=8)
    _drive(history, 20)
    messages = ProseSummarisedWindow().render(
        history=history, system="S", instruction="G", step=21, budget=ImageBudget(max_images=3)
    )
    text = _text_of(messages[1])
    assert "Step 1: action 1" in text, "the prose block starts at the very first action"
    assert "Step 18: action 18" in text
    assert "Step 19:" not in text, "action 19 is visible, so it is not prose"
    assert len(history.all_outputs) == 20


def test_prose_summary_recovers_the_action_description() -> None:
    assert prose_summary("I will click the icon.\n0 0 0 ; +LMB -LMB") == "I will click the icon."
    assert prose_summary("Action: click it\n0 0 0 ;") == "click it"
    assert prose_summary("only one line") == "only one line"
    assert prose_summary("") == "No parseable action description."
    assert prose_summary("Action:\nline") == "No action description."


def test_latest_image_only_sends_exactly_one_image() -> None:
    history = History(n_history_frames=32)
    _drive(history, 6)
    messages = LatestImageOnly().render(
        history=history, system="S", instruction="GOAL", step=7, budget=ImageBudget(max_images=1)
    )
    assert _images(messages) == 1
    assert _text_of(messages[1]).count("GOAL") == 1
    placeholders = sum(1 for m in messages if IMAGE_PLACEHOLDER in _text_of(m))
    assert placeholders == 5, "every older image becomes one text placeholder"
    assert "Newest observation." in _text_of(messages[-1])


def test_stateless_single_turn_has_no_history() -> None:
    history = History(n_history_frames=32)
    _drive(history, 5)
    messages = StatelessSingleTurn().render(
        history=history, system="S", instruction="GOAL", step=6, budget=ImageBudget(max_images=1)
    )
    assert [m.role for m in messages] == ["system", "user"]
    assert _images(messages) == 1
    assert all("action 1" not in _text_of(m) for m in messages)


def test_image_budget_wraps_the_received_jpeg_byte_for_byte() -> None:
    raw = jpeg(20, 10)
    url = ImageBudget().data_url(raw)
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw
    assert OSWORLD_CURSOR_JPEG_DOMAIN == "osworld_cursor_jpeg_q85_420_1920x1080_v1"


def test_image_budget_refuses_a_non_jpeg_observation() -> None:
    with pytest.raises(ValueError, match=OSWORLD_CURSOR_JPEG_DOMAIN):
        ImageBudget().data_url(png())


def test_history_policy_builds_by_name_and_refuses_an_unknown_one() -> None:
    assert history_policy("interleaved_frames").name == "interleaved_frames"
    with pytest.raises(ValueError, match="unknown history policy"):
        history_policy("no_such_shape")


def test_every_registered_policy_name_matches_its_key() -> None:
    for name, factory in POLICIES.items():
        assert factory().name == name
