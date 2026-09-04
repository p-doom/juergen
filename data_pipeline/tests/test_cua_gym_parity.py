from __future__ import annotations

import importlib
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest
from desktop.geometry import DisplayGeometry
from grammars.ordered_events_v3_relative_1000_grid_v1.codec import (
    CODEC,
    grid_from_pixels,
    pixels_from_grid,
)
from PIL import Image

from pipeline.cua_gym.stage_01_image_store import build_store
from pipeline.cua_gym.stage_04_build_conversations import (
    ImageIndex,
    build_dataset,
    build_episode_records,
    render_contract,
)
from pipeline.cua_gym.translate import (
    UnsupportedSourceAction,
    rewrite_assistant,
    translate_step,
)
from pipeline.lib.image_store import parse_arrayrecord_image_uri, read_jpeg_bytes
from pipeline.lib.manifest import resolve_chat_artifact

GEOMETRY = DisplayGeometry(desktop_width=1920, desktop_height=1080)


class _Bag:
    def __init__(self) -> None:
        object.__setattr__(self, "values", {})

    def __getattr__(self, name: str):
        values = object.__getattribute__(self, "values")
        if name not in values:
            values[name] = _Bag()
        return values[name]

    def __setattr__(self, name: str, value: object) -> None:
        object.__getattribute__(self, "values")[name] = value

    def to_dict(self) -> dict:
        return {
            key: value.to_dict() if isinstance(value, _Bag) else value
            for key, value in object.__getattribute__(self, "values").items()
        }


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
        ({"action": "double_click"}, "down(LMB); up(LMB); down(LMB); up(LMB)"),
        (
            {"action": "key", "keys": ["ctrl", "c"]},
            "down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)",
        ),
        (
            {"action": "type", "text": "a\nb\tc"},
            'type("a"); down(Return); up(Return); type("b"); down(Tab); up(Tab); type("c")',
        ),
        ({"action": "scroll", "pixels": -5}, "scroll(0,-5)"),
        ({"action": "hscroll", "pixels": 4}, "scroll(4,0)"),
        ({"action": "wait", "time": 1}, "NO_OP"),
        ({"action": "screenshot"}, "NO_OP"),
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


@pytest.mark.parametrize("dimension", [1080, 1920, 2560])
def test_integer_thousandth_delta_is_nearest_representable_pixel(dimension):
    for coordinate in range(1001):
        target = min(round(coordinate / 1000 * dimension), dimension - 1)
        for cursor in range(dimension):
            encoded = grid_from_pixels(target - cursor, dimension)
            landed = cursor + pixels_from_grid(encoded, dimension)
            assert abs(landed - target) <= 1


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


class _Images:
    def uri(self, shard: str, member: str) -> str:
        return f"ar:///images/{shard.removesuffix('.tar')}.array_record#{member[-7:-4]}"


def _trajectory(*, task_id: str = "task", reward: int = 1, turns: int = 6) -> dict:
    return {
        "task_id": task_id,
        "instruction": "Edit the document.",
        "screen": [1920, 1080],
        "reward": reward,
        "steps": [
            {
                "step": index,
                "shard": "screenshots-0000.tar",
                "member": f"task/step_{index:03d}.png",
                "assistant_raw": f"reason {index}</think>\nAction: wait\n<tool_call>{{}}</tool_call>",
                "raw_action_args": {"action": "wait", "time": 1},
                "cursor_before": [960, 540],
            }
            for index in range(turns)
        ],
    }


def _image_count(messages: list[dict]) -> int:
    return sum(part.get("type") == "image" for message in messages for part in message["content"])


def test_stage_04_bounds_history_masks_loss_and_summarizes_evictions():
    rows = build_episode_records(_trajectory(), _Images())
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


def test_rewrite_keeps_only_the_thinking_block():
    source = "reason</think>\nAction: wait\n<tool_call>{}</tool_call>"
    action = translate_step({"action": "wait", "time": 1}, (0, 0), GEOMETRY).action
    assert rewrite_assistant(source, action) == "<think>reason</think>\n\nNO_OP"


def test_failure_sampling_is_deterministic_and_failure_control_survives():
    trajectory = _trajectory(task_id="task-6", reward=0, turns=1)
    trajectory["steps"][0]["raw_action_args"] = {"action": "terminate", "status": "failure"}
    rows = build_episode_records(trajectory, _Images())
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"][0]["text"].endswith("TERMINATE: failure")
    assert build_episode_records(trajectory, _Images()) == rows


def test_stage_01_to_stage_04_manifest_and_digest_contract(tmp_path: Path):
    screenshots = tmp_path / "screenshots"
    _screenshot_tar(screenshots, count=1)
    image_store = tmp_path / "images"
    build_store(screenshots, image_store, workers=1)
    trajectory = _trajectory(turns=1)
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(json.dumps(trajectory) + "\n")
    output = tmp_path / "conversations"
    manifest = build_dataset(trajectories, image_store, output)
    assert ImageIndex(image_store).uri("screenshots-0000.tar", "task/step_000.png")
    assert manifest["artifact_type"] == "cuagym_stage_04_conversations"
    assert manifest["grammar"] == CODEC.name
    contract = render_contract()
    assert manifest["contract"]["render_spec_sha256"] == contract["render_spec_sha256"]
    row = json.loads((output / "chat.jsonl").read_text())
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


def test_labctl_chain_requires_and_connects_omegalax_scripts(monkeypatch, tmp_path: Path):
    schema = types.ModuleType("pmanager.configs.schema")
    schema.pipeline_task = _Bag
    monkeypatch.setitem(sys.modules, "pmanager", types.ModuleType("pmanager"))
    monkeypatch.setitem(sys.modules, "pmanager.configs", types.ModuleType("pmanager.configs"))
    monkeypatch.setitem(sys.modules, "pmanager.configs.schema", schema)
    module = importlib.import_module("configs.chain_cua_gym_parity")

    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text("{}\n")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    omegalax = tmp_path / "omegalax"
    omegalax.mkdir()
    monkeypatch.setenv("LABCTL_DATASETS_ROOT", str(datasets))
    monkeypatch.setenv("CUA_GYM_SCREENSHOTS_DIR", str(screenshots))
    monkeypatch.setenv("CUA_GYM_TRAJECTORIES", str(trajectories))
    monkeypatch.setenv("OMEGALAX_REPO", str(omegalax))

    with pytest.raises(RuntimeError, match="measure_message_lengths_from_chat"):
        module.get_config()
    scripts = omegalax / "scripts"
    scripts.mkdir()
    for name in (
        "measure_message_lengths_from_chat.py",
        "build_sft_records_from_chat.py",
    ):
        (scripts / name).touch()
    config = module.get_config().to_dict()
    assert config["entrypoint"]["path"] == "pipeline/cua_gym/stage_01_image_store.py"
    stage_04 = config["children"][0]
    stage_05 = stage_04["children"][0]
    stage_06 = stage_05["children"][0]
    assert stage_04["entrypoint"]["path"] == "pipeline/cua_gym/stage_04_build_conversations.py"
    assert stage_05["entrypoint"]["path"] == "pipeline/stage_05_measure_lengths.py"
    assert stage_06["entrypoint"]["path"] == "pipeline/stage_06_training_records.py"
