from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest
from desktop.execute.protocol import build_action_request
from desktop.geometry import DisplayGeometry
from grammars.ordered_events_v3_relative_1000_grid_v1.codec import (
    CODEC,
    action_from_dict,
    grid_delta,
    pixels_from_grid,
)
from PIL import Image

from pipeline.cua_gym.stage_01_image_store import build_store
from pipeline.cua_gym.stage_03_curate_trajectories import (
    build_curated_dataset,
    curate_rollout,
)
from pipeline.cua_gym.stage_04_build_conversations import (
    ImageIndex,
    build_dataset,
    build_episode_records,
    render_contract,
)
from pipeline.cua_gym.translate import (
    UnsupportedSourceAction,
    translate_step,
)
from pipeline.lib.image_store import parse_arrayrecord_image_uri, read_jpeg_bytes
from pipeline.lib.manifest import resolve_chat_artifact

GEOMETRY = DisplayGeometry(desktop_width=1920, desktop_height=1080)


def _png(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1920, 1080), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _screenshot_tar(root: Path, count: int = 6) -> Path:
    root.mkdir(parents=True)
    path = root / "screenshots-0000.tar"
    with tarfile.open(path, "w") as archive:
        for index in range(count):
            payload = _png((index, 20, 30))
            info = tarfile.TarInfo(f"task/step_{index:03d}.png")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"action": "right_click"}, "down(RMB); up(RMB)"),
        ({"action": "click"}, "down(LMB); up(LMB)"),
        ({"action": "double_click"}, "down(LMB); up(LMB); down(LMB); up(LMB)"),
        (
            {"action": "key", "keys": ["ctrl", "c"]},
            "down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)",
        ),
        (
            {"action": "key", "keys": ["ctrl", "k", "ctrl", "s"]},
            "down(ControlLeft); down(KeyK); up(KeyK); up(ControlLeft); "
            "down(ControlLeft); down(KeyS); up(KeyS); up(ControlLeft)",
        ),
        (
            {"action": "key", "keys": ["backspace", "backspace", "backspace"]},
            "down(Backspace); up(Backspace); down(Backspace); up(Backspace); "
            "down(Backspace); up(Backspace)",
        ),
        (
            {"action": "type", "text": "a\nb\tc"},
            'type("a"); down(Return); up(Return); type("b"); down(Tab); up(Tab); type("c")',
        ),
        ({"action": "scroll", "pixels": -5}, "scroll(0,-5)"),
        ({"action": "hscroll", "pixels": 4}, "scroll(4,0)"),
        ({"action": "wait", "time": 1}, "NO_OP"),
        ({"action": "screenshot"}, "NO_OP"),
        ({"action": "type", "text": ""}, "NO_OP"),
        (
            {
                "action": "key",
                "keys": ["mod", "comma", "grave", "print"],
            },
            "down(ControlLeft); down(Comma); down(Backquote); down(PrintScreen); "
            "up(PrintScreen); up(Backquote); up(Comma); up(ControlLeft)",
        ),
    ],
)
def test_native_action_translation_matrix(arguments, expected):
    translated = translate_step(arguments, (960, 540), GEOMETRY)
    assert translated.text == expected
    CODEC.parse(expected)


def test_coordinate_translation_and_compile_share_one_grid():
    translated = translate_step(
        {"action": "left_click", "coordinate": [389, 308]},
        (1728, 972),
        GEOMETRY,
    )
    assert translated.target_pixel == (747, 333)
    assert translated.text == "move(-511,-592); down(LMB); up(LMB)"
    operations = CODEC.compile(translated.text, GEOMETRY, (1728, 972))
    assert operations[0].kind == "move_to"
    assert operations[0].args == (747, 333)


def test_relative_grid_boundary_uses_the_pixel_delta():
    translated = translate_step(
        {"action": "mouse_move", "coordinate": [1000, 0]},
        (1, 0),
        GEOMETRY,
    )
    assert translated.target_pixel == (1919, 0)
    assert translated.text == "move(999,0)"
    assert CODEC.compile(translated.text, GEOMETRY, (1, 0))[0].args == (1919, 0)

    from_origin = translate_step(
        {"action": "mouse_move", "coordinate": [1000, 1000]},
        (0, 0),
        GEOMETRY,
    )
    assert from_origin.text == "move(1000,999)"
    assert CODEC.compile(from_origin.text, GEOMETRY, (0, 0))[0].args == (1919, 1079)


def _landing(cursor: int, unit: int, dimension: int) -> int:
    return min(max(cursor + pixels_from_grid(unit, dimension), 0), dimension - 1)


@pytest.mark.parametrize("dimension", range(1, 51))
def test_integer_thousandth_delta_is_nearest_on_bounded_domains(dimension):
    for cursor in range(dimension):
        for target in range(dimension):
            encoded = grid_delta(cursor, target, dimension)
            error = abs(_landing(cursor, encoded, dimension) - target)
            assert error == min(
                abs(_landing(cursor, unit, dimension) - target)
                for unit in range(encoded - 3, encoded + 4)
            )


@pytest.mark.parametrize("dimension", [720, 1080, 1920, 2560, 3840])
def test_integer_thousandth_delta_samples_large_displays(dimension):
    positions = sorted({0, 1, dimension // 4, dimension // 2, dimension - 2, dimension - 1})
    for cursor in positions:
        for target in positions:
            encoded = grid_delta(cursor, target, dimension)
            error = abs(_landing(cursor, encoded, dimension) - target)
            assert error == min(
                abs(_landing(cursor, unit, dimension) - target)
                for unit in range(encoded - 3, encoded + 4)
            )


@pytest.mark.parametrize("width,height", [(1280, 720), (1920, 1080), (2560, 1440)])
def test_coordinate_grid_matrix(width, height):
    geometry = DisplayGeometry(desktop_width=width, desktop_height=height)
    cursor = (width // 4, height // 2)
    translated = translate_step(
        {"action": "mouse_move", "coordinate": [750, 500]},
        cursor,
        geometry,
    )
    (operation,) = CODEC.compile(translated.text, geometry, cursor)
    assert operation.args == (round(0.75 * width), round(0.5 * height))


def test_source_policy_and_geometry_fail_loudly():
    for action in ("call_user", "answer"):
        with pytest.raises(UnsupportedSourceAction, match=action):
            translate_step({"action": action, "text": "unsupported"}, (0, 0), GEOMETRY)
    with pytest.raises(TypeError, match="numeric"):
        translate_step({"action": "mouse_move", "coordinate": [True, 2]}, (0, 0), GEOMETRY)
    with pytest.raises(ValueError, match="positive"):
        CODEC.compile("move(1,1)", DisplayGeometry(0, 1080), (0, 0))


def test_source_arguments_are_exact():
    with pytest.raises(ValueError, match=r"missing arguments.*pixels"):
        translate_step({"action": "scroll"}, (0, 0), GEOMETRY)
    with pytest.raises(ValueError, match=r"missing arguments.*status"):
        translate_step({"action": "terminate"}, (0, 0), GEOMETRY)
    with pytest.raises(ValueError, match="unexpected arguments"):
        translate_step({"action": "screenshot", "coordinate": [1, 2]}, (0, 0), GEOMETRY)
    with pytest.raises(ValueError, match="unsupported pyautogui key"):
        translate_step({"action": "key", "keys": ["invented_key"]}, (0, 0), GEOMETRY)
    with pytest.raises(ValueError, match="unsupported pyautogui key"):
        translate_step({"action": "key", "keys": ["center"]}, (0, 0), GEOMETRY)


def test_print_key_compiles_to_the_pinned_desktop_execution_target():
    translated = translate_step({"action": "key", "keys": ["print"]}, (0, 0), GEOMETRY)
    operations = tuple(CODEC.compile(translated.text, GEOMETRY, (0, 0)))
    request, _, held = build_action_request(
        operations,
        initial_buttons=set(),
        initial_keys=set(),
    )
    assert held == set()
    assert [(row["kind"], row["key"], row["keysym"]) for row in request["rows"]] == [
        ("key_down", "printscreen", 0xFF61),
        ("key_up", "printscreen", 0xFF61),
    ]


def test_termination_preserves_status_and_does_not_use_reward():
    assert (
        translate_step({"action": "terminate", "status": "success"}, (0, 0), GEOMETRY).text
        == "NO_OP\nTERMINATE: success"
    )
    assert (
        translate_step({"action": "terminate", "status": "failure"}, (0, 0), GEOMETRY).text
        == "NO_OP\nTERMINATE: failure"
    )


def test_stage_01_writes_q92_rgb_arrayrecord_and_manifest(tmp_path: Path):
    screenshots = tmp_path / "screenshots"
    _screenshot_tar(screenshots, count=2)
    output = tmp_path / "images"
    manifest = build_store(screenshots, output, workers=1)
    assert manifest["artifact_type"] == "cuagym_stage_01_image_store"
    assert manifest["jpeg_quality"] == 92
    assert manifest["total_images"] == 2
    generation = output / manifest["generation"]
    rows = [
        json.loads(line)
        for line in (generation / "screenshots-0000" / "index.jsonl").read_text().splitlines()
    ]
    shard, index = parse_arrayrecord_image_uri(rows[0]["uri"])
    assert shard == (generation / "screenshots-0000" / "images.array_record").resolve()
    assert index == 0
    with Image.open(io.BytesIO(read_jpeg_bytes(rows[0]["uri"]))) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (1920, 1080)
    assert build_store(screenshots, output, workers=1) == manifest


def test_stage_01_rebuilds_corruption_and_closes_the_source_set(tmp_path: Path):
    screenshots = tmp_path / "screenshots"
    _screenshot_tar(screenshots, count=1)
    output = tmp_path / "images"
    first = build_store(screenshots, output, workers=1)
    first_generation = output / first["generation"]
    shard = first_generation / "screenshots-0000" / "images.array_record"
    shard.write_bytes(b"corrupt")

    repaired = build_store(screenshots, output, workers=1)
    assert repaired == first
    assert shard.read_bytes() != b"corrupt"

    second_tar = screenshots / "screenshots-0001.tar"
    first_tar = screenshots / "screenshots-0000.tar"
    first_tar.replace(second_tar)
    changed = build_store(screenshots, output, workers=1)
    assert set(changed["source_tars"]) == {"screenshots-0001.tar"}
    current = output / changed["generation"]
    assert {path.name for path in current.iterdir()} == {"screenshots-0001"}
    assert {path.name for path in output.glob("generation-*")} == {changed["generation"]}

    orphan = output / "generation-orphan"
    orphan.mkdir()
    assert build_store(screenshots, output, workers=1) == changed
    assert not orphan.exists()


def test_stage_01_rejects_duplicate_png_members(tmp_path: Path):
    screenshots = tmp_path / "screenshots"
    path = _screenshot_tar(screenshots, count=1)
    payload = _png()
    with tarfile.open(path, "a") as archive:
        info = tarfile.TarInfo("task/step_000.png")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="duplicate PNG member"):
        build_store(screenshots, tmp_path / "images", workers=1)


def test_stage_01_failure_cannot_leave_a_completion_manifest(tmp_path: Path):
    screenshots = tmp_path / "screenshots"
    _screenshot_tar(screenshots, count=1)
    output = tmp_path / "images"
    build_store(screenshots, output, workers=1)
    (output / "generation-orphan").mkdir()
    with (
        mock.patch(
            "pipeline.cua_gym.stage_01_image_store.shutil.rmtree",
            side_effect=OSError("cleanup failed"),
        ),
        pytest.raises(OSError, match="cleanup failed"),
    ):
        build_store(screenshots, output, workers=1)
    assert not (output / "manifest.json").exists()


class _Images:
    def uri(self, shard: str, member: str) -> str:
        return f"ar:///images/{shard.removesuffix('.tar')}.array_record#{member[-7:-4]}"


def _raw_trajectory(*, task_id: str = "task", turns: int = 6) -> dict:
    steps = []
    for index in range(turns):
        arguments = {"action": "wait", "time": 1}
        assistant = (
            f"reason {index}</think>\n\n<tool_call>\n"
            + json.dumps({"name": "computer_use", "arguments": arguments})
            + "\n</tool_call>"
        )
        steps.append(
            {
                "step": index,
                "latency_s": 1.0,
                "screenshot": f"screenshots/step_{index:03d}.png",
                "cursor_before": [960, 540],
                "raw": assistant,
                "action": "wait",
                "meta": {"action": "wait", "time": 1.0},
                "raw_action_args": arguments,
                "coordinate_screen": None,
                "assistant_raw": assistant,
                "shard": "screenshots-0000.tar",
                "member": f"{task_id}/step_{index:03d}.png",
            }
        )
    return {
        "task_id": task_id,
        "instruction": "Edit the document.",
        "app": "writer",
        "worker": "worker",
        "started": 1.0,
        "steps": steps,
        "reward": 1.0,
        "setup_ok": True,
        "complete": True,
        "screen": [1920, 1080],
        "terminated": False,
        "steps_taken": turns,
        "reward_raw": "PASS",
        "finished": 2.0,
        "duration_s": 1.0,
        "_shard": "screenshots-0000.tar",
        "_members": [step["member"] for step in steps],
    }


def _curated_trajectory(*, task_id: str = "task", turns: int = 6) -> dict:
    counters = Counter()
    curated = curate_rollout(_raw_trajectory(task_id=task_id, turns=turns), counters, [])
    assert curated is not None
    return curated


def _assistant(reasoning: str, *calls: dict) -> str:
    return reasoning + "".join(
        "\n<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": call})
        + "\n</tool_call>"
        for call in calls
    )


def test_curator_composes_multicall_turn_in_execution_order():
    trajectory = _raw_trajectory(turns=1)
    move = {"action": "mouse_move", "coordinate": [389, 308]}
    click = {"action": "left_click", "coordinate": [389, 308]}
    assistant = _assistant("reason</think>", move, click)
    first = trajectory["steps"][0]
    first |= {
        "cursor_before": [1728, 972],
        "raw": assistant,
        "action": "mouse_move",
        "meta": {"action": "mouse_move", "pixel": [747, 333]},
        "raw_action_args": move,
        "coordinate_screen": [747, 333],
        "assistant_raw": assistant,
    }
    second = {key: value for key, value in first.items() if key != "latency_s"}
    second |= {
        "sub": 1,
        "screenshot": "screenshots/step_000_1.png",
        "cursor_before": [747, 333],
        "action": "left_click",
        "meta": {"action": "left_click", "pixel": [747, 333]},
        "raw_action_args": click,
        "member": "task/step_000_1.png",
    }
    trajectory["steps"] = [first, second]
    trajectory["_members"] = [first["member"], second["member"]]
    counters = Counter()
    curated = curate_rollout(trajectory, counters, [])
    assert curated is not None
    assert len(curated["steps"]) == 1
    assert curated["steps"][0]["member"] == "task/step_000.png"
    action = action_from_dict(curated["steps"][0]["action"])
    assert CODEC.format(action) == "move(-511,-592); down(LMB); up(LMB)"
    assert counters["executed_calls"] == 2
    assert counters["multicall_turns"] == 1
    assert counters["multicall_extra_calls"] == 1


@pytest.mark.parametrize(
    ("prefix", "reasoning", "counter"),
    [
        ("plain reason", "plain reason", "reasoning_missing_closer"),
        ("plain reason</thinking>", "plain reason", "reasoning_thinking_closer_typo"),
        ("plain reason</think>\nAction: wait", "plain reason", "reasoning_prose_after_closer"),
        ("<think>plain reason</think>", "plain reason", "reasoning_closed"),
    ],
)
def test_curator_normalizes_the_observed_reasoning_envelopes(
    prefix: str, reasoning: str, counter: str
):
    trajectory = _raw_trajectory(turns=1)
    call = trajectory["steps"][0]["raw_action_args"]
    assistant = _assistant(prefix, call)
    trajectory["steps"][0]["raw"] = assistant
    trajectory["steps"][0]["assistant_raw"] = assistant
    counters = Counter()
    curated = curate_rollout(trajectory, counters, [])
    assert curated is not None
    assert curated["steps"][0]["reasoning"] == reasoning
    assert counters[counter] == 1


def test_curator_normalizes_the_observed_double_open_tool_tag():
    trajectory = _raw_trajectory(turns=1)
    call = trajectory["steps"][0]["raw_action_args"]
    assistant = _assistant("plain reason</think>", call).replace("<tool_call>", "<<tool_call>", 1)
    trajectory["steps"][0]["raw"] = assistant
    trajectory["steps"][0]["assistant_raw"] = assistant
    counters = Counter()
    curated = curate_rollout(trajectory, counters, [])
    assert curated is not None
    assert curated["steps"][0]["reasoning"] == "plain reason"
    assert counters["reasoning_double_open_tool_tag"] == 1


def test_curator_accounts_for_nonrepresentable_source_key():
    trajectory = _raw_trajectory(turns=2)
    center = {"action": "key", "keys": ["center"]}
    assistant = _assistant("center text</think>", center)
    trajectory["steps"][0] |= {
        "raw": assistant,
        "action": "key",
        "meta": center,
        "raw_action_args": center,
        "assistant_raw": assistant,
    }
    dispositions = []
    counters = Counter()
    curated = curate_rollout(trajectory, counters, dispositions)
    assert curated is not None
    assert [step["step"] for step in curated["steps"]] == [1]
    assert counters["executed_calls"] == 2
    assert counters["logical_targets"] == 2
    assert counters["executable_targets"] == 1
    assert counters["nonexecutable_calls"] == 1
    assert dispositions == [
        {
            "recording_id": "task",
            "step": 0,
            "reason": "source_key_has_no_desktop_execution_identity",
            "source_call_sha256": hashlib.sha256(
                json.dumps(
                    center,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        }
    ]


def _image_count(messages: list[dict]) -> int:
    return sum(part.get("type") == "image" for message in messages for part in message["content"])


def test_stage_04_bounds_history_masks_loss_and_summarizes_evictions():
    rows = build_episode_records(_curated_trajectory(), _Images(), render_contract(), Counter())
    assert [row["n_history_turns"] for row in rows] == [0, 1, 2, 3, 4, 4]
    assert [_image_count(row["messages"]) for row in rows] == [1, 2, 3, 4, 5, 5]
    final = rows[-1]["messages"]
    assert "Step 0: NO_OP" in final[1]["content"][1]["text"]
    assert all(
        message.get("loss") is False for message in final[:-1] if message["role"] == "assistant"
    )
    assert "loss" not in final[-1]
    assert "<tool_call>" not in json.dumps(rows)
    assert all(row["grammar"] == CODEC.name for row in rows)
    assert all(
        "Action:" not in message["content"][0]["text"]
        for message in final
        if message["role"] == "assistant"
    )


def test_evicted_action_summary_has_one_total_bound():
    trajectory = _curated_trajectory(turns=100)
    trajectory["steps"] = [
        {
            **step,
            "action": translate_step(
                {"action": "type", "text": "x" * 500}, (960, 540), GEOMETRY
            ).action.to_dict(),
        }
        for step in trajectory["steps"]
    ]
    rows = build_episode_records(trajectory, _Images(), render_contract(), Counter())
    prompt = rows[-1]["messages"][1]["content"][1]["text"]
    previous = prompt.split("Previous actions:\n", 1)[1]
    assert len(previous) <= 160
    assert previous.startswith("…[earlier actions omitted]")


def test_failure_control_survives_without_reward_inference():
    trajectory = _curated_trajectory(task_id="task-6", turns=1)
    trajectory["steps"][0]["action"] = translate_step(
        {"action": "terminate", "status": "failure"}, (0, 0), GEOMETRY
    ).action.to_dict()
    rows = build_episode_records(trajectory, _Images(), render_contract(), Counter())
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"][0]["text"].endswith("TERMINATE: failure")
    assert build_episode_records(trajectory, _Images(), render_contract(), Counter()) == rows
    assert "reward" not in rows[0]


def test_actual_source_schema_normalizes_executed_type_and_null_steps():
    trajectory = _raw_trajectory(turns=3)
    trajectory["steps"][0] |= {
        "raw_action_args": {
            "action": "type",
            "text": "hello",
            "clear": True,
            "enter": 2,
        },
        "meta": {"action": "type", "text": "hello"},
        "action": "type",
    }
    type_call = {"action": "type", "text": "hello", "clear": True, "enter": 2}
    type_assistant = (
        "type it</think>\n<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": type_call})
        + "\n</tool_call>"
    )
    trajectory["steps"][0]["raw"] = type_assistant
    trajectory["steps"][0]["assistant_raw"] = type_assistant
    trajectory["steps"][1] |= {
        "error": "parse: invalid",
    }
    for key in ("action", "assistant_raw", "meta", "raw_action_args", "coordinate_screen"):
        trajectory["steps"][1].pop(key)
    trajectory["steps"][2] |= {
        "raw_action_args": {"action": "click", "coordinate": [389, 308]},
        "meta": {"action": "click", "pixel": [747, 333]},
        "coordinate_screen": [747, 333],
        "action": "click",
    }
    click_call = {"action": "click", "coordinate": [389, 308]}
    click_assistant = (
        "click it</think>\n<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": click_call})
        + "\n</tool_call>"
    )
    trajectory["steps"][2]["raw"] = click_assistant
    trajectory["steps"][2]["assistant_raw"] = click_assistant
    counters = Counter()
    curated = curate_rollout(trajectory, counters, [])
    assert curated is not None
    rows = build_episode_records(curated, _Images(), render_contract(), Counter())
    assert len(rows) == 2
    assert counters["source_events"] == 3
    assert counters["nonexecuted_events"] == 1
    assert counters["logical_targets"] == 2
    assert counters["executable_targets"] == 2
    assert rows[0]["messages"][-1]["content"][0]["text"].endswith('type("hello")')
    assert rows[1]["messages"][-1]["content"][0]["text"].endswith(
        "move(-111,-192); down(LMB); up(LMB)"
    )


def test_all_null_rollout_is_accounted_by_identity_and_digest(tmp_path: Path):
    trajectory = _raw_trajectory(task_id="broken-recording", turns=1)
    trajectory["steps"][0]["error"] = "parse: invalid trailing comma"
    for key in ("action", "assistant_raw", "meta", "raw_action_args", "coordinate_screen"):
        trajectory["steps"][0].pop(key)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(_raw_trajectory(task_id="kept", turns=1)) + "\n" + json.dumps(trajectory) + "\n"
    )
    manifest = build_curated_dataset(source, tmp_path / "curated")
    assert manifest["stats"]["excluded_rollouts"] == 1
    assert manifest["exclusions"][0]["recording_id"] == "broken-recording"
    assert manifest["exclusions"][0]["reason"] == "no_executed_actions"
    assert len(manifest["exclusions"][0]["source_rollout_sha256"]) == 64


def test_stage_01_to_stage_04_manifest_and_digest_contract(tmp_path: Path):
    screenshots = tmp_path / "screenshots"
    _screenshot_tar(screenshots, count=1)
    image_store = tmp_path / "images"
    build_store(screenshots, image_store, workers=1)
    trajectory = _raw_trajectory(turns=1)
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(json.dumps(trajectory) + "\n")
    curated = tmp_path / "curated"
    curated_manifest = build_curated_dataset(trajectories, curated)
    output = tmp_path / "conversations"
    manifest = build_dataset(curated, image_store, output)
    assert ImageIndex(image_store).uri("screenshots-0000.tar", "task/step_000.png")
    assert manifest["artifact_type"] == "cuagym_stage_04_conversations"
    assert manifest["grammar"] == CODEC.name
    assert manifest["stats"] == {"records": 1, "rollouts": 1}
    assert curated_manifest["stats"]["logical_targets"] == 1
    assert curated_manifest["stats"]["executable_targets"] == 1
    contract = render_contract()
    assert manifest["contract"]["render_spec_sha256"] == contract["render_spec_sha256"]
    row = json.loads((output / "chat.jsonl").read_text())
    target = row["messages"][-1]["content"][0]["text"]
    assert target.count("<think>") == target.count("</think>") == 1
    assert "Action:" not in target and "tool_call" not in target
    for field in (
        "system_prompt_sha256",
        "render_spec_sha256",
        "action_spec_sha256",
        "observation_spec_sha256",
    ):
        assert row[field] == contract[field]
    assert resolve_chat_artifact(output) == output / "chat.jsonl"
    original_id = manifest["chat_sha256"]
    (output / "chat.jsonl").write_text("mutated\n")
    with pytest.raises(ValueError, match="chat digest mismatch"):
        resolve_chat_artifact(output)
    assert json.loads((output / "manifest.json").read_text())["chat_sha256"] == original_id
