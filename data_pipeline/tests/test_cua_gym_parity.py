from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from desktop.geometry import DisplayGeometry
from grammars.ordered_events_v3_relative_1000_grid_v1.codec import CODEC
from PIL import Image

from pipeline.cua_gym.stage_01_image_store import build_store
from pipeline.cua_gym.stage_04_build_conversations import (
    ImageIndex,
    build_dataset,
    build_episode_records,
    render_contract,
)
from pipeline.cua_gym.translate import UnsupportedSourceAction, translate_step
from pipeline.lib.image_store import parse_arrayrecord_image_uri, read_jpeg_bytes

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
        ({"action": "wait"}, "NO_OP"),
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
            translate_step({"action": action}, (0, 0), GEOMETRY)
    with pytest.raises(TypeError, match="numeric"):
        translate_step({"action": "mouse_move", "coordinate": [True, 2]}, (0, 0), GEOMETRY)
    with pytest.raises(ValueError, match="positive"):
        CODEC.compile("move(1,1)", DisplayGeometry(0, 1080), (0, 0))


def test_termination_preserves_status_and_does_not_use_reward():
    assert (
        translate_step({"action": "terminate"}, (0, 0), GEOMETRY).text
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
    rows = [
        json.loads(line)
        for line in (output / "screenshots-0000" / "index.jsonl").read_text().splitlines()
    ]
    shard, index = parse_arrayrecord_image_uri(rows[0]["uri"])
    assert shard == (output / "screenshots-0000" / "images.array_record").resolve()
    assert index == 0
    with Image.open(io.BytesIO(read_jpeg_bytes(rows[0]["uri"]))) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (1920, 1080)
    assert build_store(screenshots, output, workers=1) == manifest


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
                "raw_action_args": {"action": "wait"},
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


def test_failure_sampling_is_deterministic_and_failure_control_survives():
    trajectory = _trajectory(task_id="task-1", reward=0, turns=1)
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
