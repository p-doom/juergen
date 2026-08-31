from __future__ import annotations

import asyncio
import copy
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import verifiers.v1 as vf
from desktop.geometry import DisplayGeometry
from juergen_doubles import make_ctx
from PIL import Image
from test_cua_gym_dataset import _synthetic_snapshot
from test_cua_gym_runtime import (
    _CuaSession,
    _harness_config,
    _task,
)

import agent.desktop as desktop_pools
import stream_cuagym_qwen35 as stream_render
from agent.agent import Agent, EffectiveSampling
from data_pipeline.cuagym_pipeline.stage_04_build_conversations import (
    build_episode_records,
)
from data_pipeline.cuagym_pipeline.translate import translate_step
from evals.cua_gym import TaskPlatform, runtime
from evals.harness import ArtifactConfig, DesktopHarness
from evals.tasks import RESULT_KEY, DesktopState
from grammars.ordered_events_v3.codec import CODEC
from harness_render import HarnessRenderer


class _UnusedTransport:
    async def complete(self, *args, **kwargs):
        raise AssertionError("transport is not used by Agent.decide")

    async def close(self) -> None:
        return None


def _jpeg_q92(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1920, 1080), color).save(
        buffer, format="JPEG", quality=92, subsampling=2, optimize=False
    )
    return buffer.getvalue()


def _image_count(messages: list[dict[str, Any]]) -> int:
    return sum(
        part.get("type") == "image"
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, dict)
    )


class _Images:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames

    def uri(self, shard: str, member: str) -> bytes:
        assert shard == "screenshots-0000.tar"
        return self.frames[int(member.removeprefix("frame-").removesuffix(".png"))]


def _trajectory(turns: int) -> dict[str, Any]:
    return {
        "task_id": "synthetic-task",
        "instruction": "Add a two-column table.",
        "screen": [1920, 1080],
        "reward": 1,
        "steps": [
            {
                "step": index + 1,
                "shard": "screenshots-0000.tar",
                "member": f"frame-{index}.png",
                "assistant_raw": (f"reason {index}</think>\n<tool_call>{{}}"),
                "raw_action_args": {"action": "wait"},
                "cursor_before": [960, 540],
            }
            for index in range(turns)
        ],
    }


def test_desktop_harness_and_stage_04_use_one_render_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import juergen_harness_pool

    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    monkeypatch.setattr(runtime, "_snapshot", lambda *_: snapshot)
    monkeypatch.setattr(
        runtime,
        "_run_setup_steps",
        lambda session, steps: session.record_setup_steps(steps),
    )
    frames = [_jpeg_q92((index, 0, 0)) for index in range(7)]
    replies = [f"<think>reason {index}</think>\n\nNO_OP" for index in range(6)]
    online_prompts: list[list[dict[str, Any]]] = []
    offline_prompts: list[list[dict[str, Any]]] = []
    destination = online_prompts
    original_render = HarnessRenderer.render_prompt

    def capture(self, *, instruction):
        prompt = original_render(self, instruction=instruction)
        destination.append(copy.deepcopy(prompt))
        return prompt

    monkeypatch.setattr(HarnessRenderer, "render_prompt", capture)
    artifacts = ArtifactConfig(
        output_dir=str(tmp_path / "artifacts"),
        save_frames=True,
        save_prompts=False,
        write_gif=False,
    )
    config = _harness_config(tmp_path).model_copy(
        update={"artifacts": artifacts, "max_steps": 6}
    )
    session = _CuaSession(frames=frames)
    juergen_harness_pool.Pool.session = session
    trace = vf.Trace(
        task=vf.TraceTask(type="CuaGymDesktopTask", data=_task(root)),
        state=DesktopState(),
    )
    context = make_ctx(replies=replies)
    try:
        asyncio.run(
            DesktopHarness(config).launch(
                context, trace, SimpleNamespace(is_local=True), "", "", {}
            )
        )
    finally:
        desktop_pools.close_all_pools()
        juergen_harness_pool.Pool.session = None

    destination = offline_prompts
    rows = build_episode_records(
        _trajectory(6), _Images(frames), failure_step_percent=100
    )

    assert online_prompts == offline_prompts
    assert [_image_count(prompt) for prompt in online_prompts] == [1, 2, 3, 4, 5, 5]
    assert "Step 1: NO_OP" in online_prompts[5][1]["content"][1]["text"]
    assert all(
        "<think>" not in part["text"]
        for message in online_prompts[5]
        if message["role"] == "assistant"
        for part in message["content"]
    )
    assert rows[5]["messages"][-1]["content"][0]["text"] == replies[5]
    assert [row["n_history_turns"] for row in rows] == [0, 1, 2, 3, 4, 4]
    result = trace.info[RESULT_KEY]
    assert result["render"]["max_completed_turns"] == 4

    assert result["render"]["render_spec_sha256"] == stream_render.SPEC_SHA256
    assert (
        result["render"]["system_prompt_sha256"] == stream_render.SYSTEM_PROMPT_SHA256
    )
    assert result["render"]["action_contract"] == stream_render.ACTION_CONTRACT
    assert result["images"] == {
        "image_domain": stream_render.OBSERVATION_CONTRACT,
        **stream_render.OBSERVATION_METADATA,
    }


def test_stage_04_keeps_non_target_turns_in_the_completed_turn_window() -> None:
    frames = [_jpeg_q92((index, 0, 0)) for index in range(6)]
    trajectory = _trajectory(6)
    trajectory["steps"][0]["raw_action_args"] = {"action": "answer"}

    rows = build_episode_records(
        trajectory,
        _Images(frames),
        failure_step_percent=100,
    )

    assert [row["target_step"] for row in rows] == [2, 3, 4, 5, 6]
    assert [row["n_history_turns"] for row in rows] == [1, 2, 3, 4, 4]
    assert [_image_count(row["messages"]) for row in rows] == [2, 3, 4, 5, 5]
    assert "Step 1:" not in rows[-1]["messages"][1]["content"][1]["text"]


@pytest.mark.parametrize("mismatch", ["spec", "prompt"])
def test_desktop_harness_refuses_render_digest_mismatch_before_pool_acquire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    _, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    if mismatch == "spec":
        monkeypatch.setattr(stream_render, "SPEC_SHA256", "0" * 64)
        expected = "render spec digest mismatch"
    else:
        monkeypatch.setattr(stream_render, "system_prompt", lambda: "wrong prompt")
        expected = "system prompt digest mismatch"
    trace = vf.Trace(
        task=vf.TraceTask(type="CuaGymDesktopTask", data=_task(root)),
        state=DesktopState(),
    )
    with pytest.raises(ValueError, match=expected):
        asyncio.run(
            DesktopHarness(_harness_config(tmp_path)).launch(
                make_ctx(replies=["NO_OP"]),
                trace,
                SimpleNamespace(is_local=True),
                "",
                "",
                {},
            )
        )
    assert "desktop_session" not in trace.info
    with pytest.raises(ValueError, match=expected):
        build_episode_records(
            _trajectory(1), _Images([_jpeg_q92((0, 0, 0))]), failure_step_percent=100
        )


def test_desktop_harness_refuses_a_remote_runtime_before_pool_acquire(
    tmp_path: Path,
) -> None:
    _, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    trace = vf.Trace(
        task=vf.TraceTask(type="CuaGymDesktopTask", data=_task(root)),
        state=DesktopState(),
    )

    with pytest.raises(ValueError, match="host-local verifiers runtime"):
        asyncio.run(
            DesktopHarness(_harness_config(tmp_path)).launch(
                make_ctx(replies=["NO_OP"]),
                trace,
                SimpleNamespace(is_local=False),
                "",
                "",
                {},
            )
        )

    assert "desktop_session" not in trace.info


def test_offline_translation_and_runtime_compile_share_the_relative_1000_grid() -> None:
    translated = translate_step(
        {"action": "mouse_move", "coordinate": [750, 500]},
        (480, 540),
        (1920, 1080),
    )
    assert translated.line == "move(500,0)"
    (operation,) = CODEC.compile(
        translated.line,
        DisplayGeometry(1920, 1080),
        (480, 540),
    )
    assert operation.kind == "move_to"
    assert operation.args == (1440, 540)
    assert CODEC.action_contract == stream_render.ACTION_CONTRACT


def test_online_action_canonicalization_matches_or_fails_loud() -> None:
    renderer = stream_render.renderer()
    renderer.start(b"frame-0")
    agent = Agent(
        codec=CODEC,
        renderer=renderer,
        transport=_UnusedTransport(),
    )
    sampling = EffectiveSampling(
        model="test",
        temperature=0.0,
        max_tokens=256,
        top_p=1.0,
        stop=(),
        temperature_source="test",
        wire_body_keys=(),
    )

    terminate = agent.decide(
        "<think>done</think>\nTERMINATE",
        step=1,
        geometry=DisplayGeometry(1920, 1080),
        cursor=(960, 540),
        sampling=sampling,
    )
    assert terminate.control == "terminate"
    assert CODEC.format(terminate.action) == "TERMINATE"
    renderer.complete(
        assistant=terminate.text,
        action=CODEC.format(terminate.action),
        next_image=b"frame-1",
    )

    malformed = agent.decide(
        "<think>uncertain</think>\nnot-an-action",
        step=2,
        geometry=DisplayGeometry(1920, 1080),
        cursor=(960, 540),
        sampling=sampling,
    )
    assert malformed.parse_error is not None
    renderer.complete(
        assistant=malformed.text,
        action=None,
        next_image=b"frame-2",
    )

    alternate_control = agent.decide(
        "<think>done</think>\nTERMINATE: success",
        step=3,
        geometry=DisplayGeometry(1920, 1080),
        cursor=(960, 540),
        sampling=sampling,
    )
    assert alternate_control.parse_error is not None
    assert alternate_control.action is None
    renderer.complete(
        assistant=alternate_control.text,
        action=None,
        next_image=b"frame-3",
    )
