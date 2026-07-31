from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.teacher_sft.contracts import ContractError
from experiments.teacher_sft.teacher import load_teacher_spec, parse_native_actions


def test_teacher_spec_and_multi_tool_calls(tmp_path: Path) -> None:
    path = tmp_path / "teacher.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "openai_chat",
                "base_url": "http://127.0.0.1:30000",
                "model_id": "teacher",
                "model_revision": "revision",
                "system_prompt": "Use absolute actions.",
                "action_space": "native_absolute",
                "coordinate_space": "absolute_grid",
                "coordinate_grid": 1000,
                "temperature": 0,
            }
        )
    )
    spec = load_teacher_spec(path)
    response = """
<tool_call>{"name":"computer_use","arguments":{"action":"mouse_down","button":"left"}}</tool_call>
<tool_call>{"name":"computer_use","arguments":{"action":"mouse_move","coordinate":[500,250]}}</tool_call>
<tool_call>{"name":"computer_use","arguments":{"action":"mouse_up","button":"left"}}</tool_call>
"""
    actions = parse_native_actions(response, spec)
    assert [action["action"] for action in actions] == [
        "mouse_down",
        "mouse_move",
        "mouse_up",
    ]
    assert actions[1]["coordinate_space"] == "absolute_grid"
    assert actions[1]["coordinate_grid"] == 1000


def test_nondeterministic_teacher_spec_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "teacher.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "openai_chat",
                "base_url": "http://localhost",
                "model_id": "teacher",
                "model_revision": "revision",
                "system_prompt": "prompt",
                "action_space": "native_absolute",
                "coordinate_space": "absolute_px",
                "temperature": 0.2,
            }
        )
    )
    with pytest.raises(ContractError, match="temperature=0"):
        load_teacher_spec(path)
