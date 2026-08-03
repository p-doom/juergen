from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from osworld_parity.proper_vm_capability_ladder.rung1.transport import RecordingTransport
from osworld_parity.sign_of_life_v2.compact_relative import (
    CompactRelativeError,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    build_phaseb_messages,
    compile_compact_relative,
    execute_compact_relative,
    format_compact_relative,
    parse_compact_relative,
    verify_sealed_contract,
)


RECIPE_ROOT = Path(__file__).resolve().parents[2] / "labctl" / "recipes"


def test_sealed_phaseb_prompt_is_byte_exact() -> None:
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == SYSTEM_PROMPT_SHA256
    assert verify_sealed_contract()["system_prompt_sha256"] == SYSTEM_PROMPT_SHA256


def test_model_output_extraction_preserves_prose_but_executes_last_line() -> None:
    raw = "Action: Click the target visible below.\n-65 -40 0 ; +LMB -LMB"
    action = parse_compact_relative(raw)
    assert (action.dx, action.dy) == (-65, -40)
    assert format_compact_relative(action) == "-65 -40 0 ; +LMB -LMB"
    with pytest.raises(CompactRelativeError):
        parse_compact_relative("-65 -40 0 ; +LMB -LMB\nextra commentary")


def test_relative_move_scroll_and_events_keep_exact_order() -> None:
    action = parse_compact_relative(
        '10 -5 3 ; +ControlLeft type("abc") -ControlLeft'
    )
    operations = compile_compact_relative(action)
    assert [(item.kind, item.args) for item in operations] == [
        ("move_relative", (10, -5)),
        ("scroll", (3,)),
        ("key_down", ("ControlLeft",)),
        ("ascii_type", ("abc",)),
        ("key_up", ("ControlLeft",)),
    ]


def test_click_uses_relative_raw_pixels_and_clips_to_screen() -> None:
    transport = RecordingTransport(screen=(100, 80))
    execute_compact_relative(transport, "150 90 0 ; +LMB -LMB")
    assert transport.cursor_position() == (99, 79)
    assert transport.atomic_inputs[0][0].kind == "move_relative"
    assert [item.kind for item in transport.audit.operations] == [
        "move_to",
        "mouse_down",
        "mouse_up",
    ]
    assert not transport.audit.held_buttons


def test_drag_move_is_ordered_and_other_move_forms_fail_closed() -> None:
    action = parse_compact_relative("5 7 0 ; +LMB MOVE(30,-20) -LMB")
    assert [item.kind for item in compile_compact_relative(action)] == [
        "move_relative",
        "mouse_down",
        "move_relative",
        "mouse_up",
    ]
    for malformed in (
        "0 0 1 ; +LMB MOVE(3,4) -LMB",
        "0 0 0 ; MOVE(3,4)",
        "0 0 0 ; +LMB MOVE(3,4)",
    ):
        with pytest.raises(CompactRelativeError):
            parse_compact_relative(malformed)


def test_special_lines_and_type_fail_closed() -> None:
    assert execute_compact_relative(RecordingTransport(), "NO_OP")["no_op"] is True
    assert execute_compact_relative(RecordingTransport(), "TERMINATE")["terminated"] is True
    assert execute_compact_relative(RecordingTransport(), "FAIL")["failed"] is True
    with pytest.raises(CompactRelativeError):
        compile_compact_relative(parse_compact_relative('0 0 0 ; type("line\\n")'))
    with pytest.raises(CompactRelativeError):
        parse_compact_relative('<tool_call>{"action":"left_click"}</tool_call>')


def test_phaseb_message_shape_matches_image_first_training_scaffold() -> None:
    frames = [Image.new("RGB", (4, 4), color=(index, 0, 0)) for index in range(6)]
    actions = [f"Action: prior {index + 1}.\n0 0 0" for index in range(5)]
    messages = build_phaseb_messages(
        instruction="Do the task.", frames=frames, actions=actions
    )
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    users = [row for row in messages if row["role"] == "user"]
    assistants = [row for row in messages if row["role"] == "assistant"]
    assert len(users) == 5
    assert len(assistants) == 4
    assert users[0]["content"][0]["type"] == "image_url"
    prefix = users[0]["content"][1]["text"]
    assert "Instruction: Do the task." in prefix
    assert "Previous actions:\nStep 1: prior 1." in prefix
    assert assistants[0]["content"] == [{"type": "text", "text": actions[1]}]


def test_four_gpu_cells_select_the_one_fixed_suite_without_alias_collisions() -> None:
    expected_artifact = (
        "phaseb_raw_deltatype_v2_A_to_B_r256_s900_continuation_hf_v4_"
        "run_019fba52e90778e0b8ae170058c814e7"
    )
    aliases: set[str] = set()
    for index in range(4):
        recipe = tomllib.loads(
            (RECIPE_ROOT / f"sign_of_life_v2_phaseb_compact_t{index}_gpu_kvm.toml").read_text()
        )
        assert recipe["resources"]["gpus"] == 1
        assert recipe["args"]["mode"] == "compact-model"
        assert recipe["args"]["task-index"] == str(index)
        assert recipe["inputs"]["model"]["artifact"] == expected_artifact
        assert recipe["inputs"]["runtime"]["path"].endswith(
            "/juergen-sign-of-life-eval-v2/.venv"
        )
        assert recipe["env"]["CUDA_HOME"] == "/fast/service/apps/software/CUDA/12.6.0"
        assert '[[ -x "$CUDA_HOME/bin/nvcc" ]]' in recipe["command"][2]
        assert '"$RUNTIME/bin/python"' in recipe["command"][2]
        alias = recipe["outputs"]["result"]["alias"]
        assert alias not in aliases
        aliases.add(alias)
