"""Item 3 — `HistoryPolicy` / `History`.

Four faithful ports, but **block eviction and `render_all` are new**, so the
boundaries are what matters here: the window invariant
(`turns[-1].output is None`), the block-eviction arithmetic, and that every policy
respects `ImageBudget.max_images` however long the window has grown.
"""

from __future__ import annotations

import base64
import io

import pytest

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
    render_all,
)
from juergen_doubles import png


def _f(i: int) -> bytes:
    return png(colour=(i % 250, 0, 0))


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


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #


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


def test_eviction_is_block_not_slide_by_one() -> None:
    """Block eviction keeps the server-side prefix cache: N/2 refills per N/2 steps."""
    history = History(n_history_frames=8)
    history.start(_f(0))
    for step in range(1, 8):
        history.append(f"a{step}", _f(step))
    assert len(history.turns) == 8, "no eviction at exactly n_history_frames"
    assert history.evicted == []
    history.append("a8", _f(8))  # ninth frame trips it
    assert len(history.turns) == 4, "keep the newest n_history_frames // 2"
    assert history.evicted == [f"a{i}" for i in range(1, 6)], history.evicted
    assert history.all_outputs == [f"a{i}" for i in range(1, 9)], (
        "no action is ever lost: evicted + in-window is the whole history"
    )


@pytest.mark.parametrize("n", [1, 2, 3, 16])
def test_eviction_never_empties_the_window(n: int) -> None:
    history = History(n_history_frames=n)
    _drive(history, 40)
    assert history.turns, f"n_history_frames={n} must keep at least one turn"
    assert len(history.turns) <= max(n, 1)
    assert history.current is history.turns[-1].image
    assert len(history.all_outputs) == 40


def test_frame_labels_name_the_window_at_the_start_of_a_step() -> None:
    history = History(n_history_frames=16)
    _drive(history, 3)  # frames 0..3 in the window, next step is 4
    assert history.frame_labels(4) == [
        "step_000.png",
        "step_001.png",
        "step_002.png",
        "step_003.png",
    ]


# --------------------------------------------------------------------------- #
# InterleavedFrames
# --------------------------------------------------------------------------- #


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


def test_persist_instruction_re_anchors_the_goal_every_step() -> None:
    history = History(n_history_frames=16)
    _drive(history, 3)
    persisted = InterleavedFrames(persist_instruction=True).render(
        history=history, system="S", instruction="GOAL", step=4, budget=ImageBudget()
    )
    assert "GOAL" in _text_of(persisted[1])
    once = InterleavedFrames(persist_instruction=False).render(
        history=history, system="S", instruction="GOAL", step=4, budget=ImageBudget()
    )
    assert "GOAL" not in _text_of(once[1]), "goal-on-step-1 is the training distribution"
    fresh = History(n_history_frames=16)
    fresh.start(_f(0))
    assert "GOAL" in _text_of(
        InterleavedFrames(persist_instruction=False).render(
            history=fresh, system="S", instruction="GOAL", step=1, budget=ImageBudget()
        )[1]
    )


def test_interleaved_loggable_mirrors_the_wire_structure() -> None:
    history = History(n_history_frames=16)
    _drive(history, 2)
    policy = InterleavedFrames()
    wire = policy.render(
        history=history, system="S", instruction="G", step=3, budget=ImageBudget(max_images=16)
    )
    logged = policy.loggable(history=history, system="S", instruction="G", step=3)
    assert [m.role for m in logged] == [m.role for m in wire]
    assert "<image step_002.png>" in _text_of(logged[-1])


def test_interleaved_loggable_ignores_the_image_budget() -> None:
    """A recorded defect, not a fixed one: `loggable` claims to be structurally
    identical to the wire payload, but it renders the whole window regardless of
    `max_images`, so a sidecar written for a budgeted arm would over-report the
    window. It is unused by the harness today (which writes `dump_prompt(build_body)`
    instead), which is why this is recorded rather than repaired."""
    history = History(n_history_frames=16)
    _drive(history, 8)
    policy = InterleavedFrames()
    wire = policy.render(
        history=history, system="S", instruction="G", step=9, budget=ImageBudget(max_images=3)
    )
    logged = policy.loggable(history=history, system="S", instruction="G", step=9)
    assert _images(wire) == 3
    assert len([m for m in logged if m.role == "user"]) == 9, (
        "loggable renders 9 frames where the wire carries 3"
    )


# --------------------------------------------------------------------------- #
# ProseSummarisedWindow
# --------------------------------------------------------------------------- #


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
    """DEFECT (fixed, `agent/history.py:329-334`).

    The invariant was `len(all_outputs) == len(images) - 1`, but `all_outputs` is
    global (evicted + in-window) while `images` is in-window only. The moment
    `History` block-evicts the two live in different index spaces: the check raised —
    and had it not, `outputs[first:]` would have paired the wrong action with each
    visible frame. `len(history.evicted)` is the missing term.

    Harmless for the sign-of-life gate, whose `max_steps <= 12` never reaches the
    default `n_history_frames=16` — so the published Phase-B-compact 2/4 is unaffected.
    Fatal for any longer rollout under this policy, where it surfaced as
    `infrastructure_error`.
    """
    history = History(n_history_frames=8)
    _drive(history, 7)
    ProseSummarisedWindow().render(  # 8 frames, no eviction yet
        history=history, system="S", instruction="G", step=8, budget=ImageBudget(max_images=5)
    )
    history.append("action 8", _f(8))  # trips block eviction
    assert history.evicted, "precondition: the window has evicted"
    messages = ProseSummarisedWindow().render(
        history=history, system="S", instruction="G", step=9, budget=ImageBudget(max_images=5)
    )
    assert _images(messages) == 4, "the window holds 4 frames after eviction"


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


# --------------------------------------------------------------------------- #
# LatestImageOnly / StatelessSingleTurn
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# the budget itself
# --------------------------------------------------------------------------- #


def test_image_budget_encodes_jpeg_by_default_and_png_when_asked() -> None:
    raw = png(20, 10)
    assert ImageBudget().data_url(raw).startswith("data:image/jpeg;base64,")
    assert ImageBudget(media="png").data_url(raw) == (
        "data:image/png;base64," + base64.b64encode(raw).decode()
    ), "PNG with no downscale must pass the bytes through untouched"


def test_image_budget_downscales_to_max_pixels() -> None:
    from PIL import Image

    raw = png(400, 200)
    url = ImageBudget(media="png", max_pixels=5000).data_url(raw)
    payload = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as handle:
        assert handle.width * handle.height <= 5000
        assert abs(handle.width / handle.height - 2.0) < 0.1, "aspect is preserved"


# --------------------------------------------------------------------------- #
# registry + render_all
# --------------------------------------------------------------------------- #


def test_history_policy_builds_by_name_and_refuses_an_unknown_one() -> None:
    assert history_policy("interleaved_frames").name == "interleaved_frames"
    assert history_policy("interleaved_frames", persist_instruction=False).persist_instruction is False
    with pytest.raises(ValueError, match="unknown history policy"):
        history_policy("no_such_shape")


def test_every_registered_policy_name_matches_its_key() -> None:
    for name, factory in POLICIES.items():
        assert factory().name == name


def test_render_all_renders_one_window_under_every_policy() -> None:
    history = History(n_history_frames=32)
    _drive(history, 3)
    rendered = render_all(
        [InterleavedFrames(), LatestImageOnly(), StatelessSingleTurn()],
        history=history,
        system="S",
        instruction="G",
        step=4,
        budget=ImageBudget(max_images=4),
    )
    assert set(rendered) == {"interleaved_frames", "latest_image_only", "stateless_single_turn"}
    assert _images(rendered["interleaved_frames"]) == 4
    assert _images(rendered["latest_image_only"]) == 1
    assert _images(rendered["stateless_single_turn"]) == 1
    assert len(rendered["stateless_single_turn"]) == 2


def test_render_all_keyed_by_name_collapses_two_configurations_of_one_policy() -> None:
    """Recorded, not fixed: `render_all` keys by `policy.name`, and `name` is a class
    default, so an A/B between `persist_instruction=True` and `False` silently keeps
    only the last one. The two shapes it is documented for (different classes) are
    unaffected."""
    history = History(n_history_frames=32)
    _drive(history, 2)
    rendered = render_all(
        [InterleavedFrames(persist_instruction=True), InterleavedFrames(persist_instruction=False)],
        history=history,
        system="S",
        instruction="G",
        step=3,
        budget=ImageBudget(),
    )
    assert len(rendered) == 1, "two policies in, one rendering out"
