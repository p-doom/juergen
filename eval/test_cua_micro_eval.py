from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from grammars.ordered_events_v3_relative_1000_grid_v1.codec import CODEC

import cua_micro_eval as micro

SUITE = Path(__file__).with_name("cua_micro_tasks.json")


def test_suite_is_the_exact_18_task_contract():
    raw, tasks = micro.load_suite(SUITE)

    assert raw["suite"] == "cua_micro_tasks"
    assert len(tasks) == 18
    assert len({task.task_id for task in tasks}) == 18
    assert sum(task.max_turns == 1 for task in tasks) == 12
    assert sum(task.max_turns > 1 for task in tasks) == 6
    assert all(task.expected == {"kind": "any"} for task in tasks if task.max_turns > 1)


def test_suite_keeps_both_exact_atomic_typing_tasks():
    _, tasks = micro.load_suite(SUITE)
    typing = [task for task in tasks if task.expected.get("kind") == "type"]

    assert {task.task_id for task in typing} == {
        "type.terminal.native_exact",
        "type.text_editor.native_exact",
    }
    assert all(task.max_turns == 1 for task in typing)
    assert all(task.expected["text"] == task.verifier["value"] for task in typing)


def test_wikipedia_task_requires_the_opened_article_outcome():
    _, tasks = micro.load_suite(SUITE)
    task = next(
        task for task in tasks if task.task_id == "multi.chrome.open_chrome_search_wikipedia"
    )

    assert task.setup == {"kind": "desktop", "chrome_startup": "wikipedia"}
    assert task.verifier["pattern"] == "PASS_TRANSFORMERS_ARTICLE"


def test_attempt_plan_is_exactly_four_fixed_seeds_per_task():
    _, tasks = micro.load_suite(SUITE)
    plan = micro.attempt_plan(tasks)

    assert len(plan) == 72
    for task in tasks:
        assert [seed for _, candidate, _, seed in plan if candidate == task] == [
            41000,
            41001,
            41002,
            41003,
        ]


def test_attempt_matrix_rejects_missing_or_duplicate_rows():
    _, tasks = micro.load_suite(SUITE)
    attempts = [
        {"task_id": task.task_id, "seed": seed} for task in tasks for seed in range(41000, 41004)
    ]
    micro._validate_attempts(tasks, attempts)

    with pytest.raises(RuntimeError, match="incomplete attempt matrix"):
        micro._validate_attempts(tasks, attempts[:-1])
    with pytest.raises(RuntimeError, match="incomplete attempt matrix"):
        micro._validate_attempts(tasks, [*attempts[:-1], attempts[0]])


@pytest.mark.parametrize(
    ("text", "expected", "matches"),
    [
        (
            "move(2,-3); down(LMB); up(LMB)",
            {"kind": "click", "button": "left"},
            True,
        ),
        ('type("exact")', {"kind": "type", "text": "exact"}, True),
        (
            "down(ControlLeft); down(KeyS); up(KeyS); up(ControlLeft)",
            {"kind": "key", "keys": ["CTRL", "S"]},
            True,
        ),
        ("scroll(0,-3)", {"kind": "scroll", "sign": "down"}, True),
        ('type("wrong")', {"kind": "type", "text": "exact"}, False),
        ("NO_OP", {"kind": "any"}, True),
    ],
)
def test_expected_action_matching(text, expected, matches):
    assert micro.action_matches_expected(CODEC.parse(text), expected) is matches


def test_response_parser_accepts_only_the_shared_grammar_and_control():
    action = micro._parse_response(
        "<think>done</think>\nmove(1,2); down(LMB); up(LMB)\nTERMINATE: success"
    )
    assert action.terminate == "success"
    assert CODEC.format(action).endswith("TERMINATE: success")

    for invalid in (
        '<tool_call>{"name":"computer_use","arguments":{"action":"terminate"}}</tool_call>',
        '{"action":"terminate","status":"success"}',
        "TERMINATE: success\nNO_OP",
        "TERMINATE",
    ):
        with pytest.raises(ValueError):
            micro._parse_response(invalid)


def test_image_part_forwards_raw_jpeg_bytes():
    payload = b"\xff\xd8raw-desktop-jpeg\xff\xd9"
    part = micro._image_part(payload)

    encoded = part["image_url"]["url"].removeprefix("data:image/jpeg;base64,")
    assert base64.b64decode(encoded) == payload


def test_model_call_uses_exact_local_request_contract(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "/models/checkpoint",
                "choices": [{"message": {"content": "NO_OP"}, "finish_reason": "stop"}],
            }

    def post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr(micro.requests, "post", post)
    output = micro._call_model(
        sglang_url="http://127.0.0.1:31000/v1",
        api_key="owned",
        model="/models/checkpoint",
        instruction="Do it",
        history=[
            {
                "step": 0,
                "image": micro._image_part(b"\xff\xd8raw\xff\xd9"),
            }
        ],
        seed=41000,
    )

    assert output == ("NO_OP", "stop")
    assert captured["url"] == "http://127.0.0.1:31000/v1/chat/completions"
    assert captured["json"] == {
        "model": "/models/checkpoint",
        "messages": captured["json"]["messages"],
        **micro._SAMPLING,
        "seed": 41000,
    }
    assert captured["timeout"] == 120


def test_model_call_rejects_wrong_serving_model(monkeypatch):
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "model": "wrong",
            "choices": [{"message": {"content": "NO_OP"}, "finish_reason": "stop"}],
        },
    )
    monkeypatch.setattr(micro.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="not '/models/checkpoint'"):
        micro._call_model(
            sglang_url="http://127.0.0.1:31000/v1",
            api_key="owned",
            model="/models/checkpoint",
            instruction="Do it",
            history=[{"step": 0, "image": {"type": "image_url"}}],
            seed=41000,
        )


def test_guest_and_execution_failures_are_not_scored_as_model_errors():
    guest = SimpleNamespace(
        run_guest=lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout="", stderr="failed")
    )
    with pytest.raises(RuntimeError, match="rc=3"):
        micro._run_guest(guest, ["false"])

    executor = SimpleNamespace(
        execute=lambda operations: SimpleNamespace(ok=False, error="rejected")
    )
    with pytest.raises(RuntimeError, match="desktop action failed"):
        micro._execute(executor, (micro.Operation("wait", (1.0,)),))


def test_guest_timeout_is_not_treated_as_a_failed_verifier(monkeypatch):
    monkeypatch.setattr(
        micro,
        "read_verifier_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("guest")),
    )
    with pytest.raises(TimeoutError, match="guest"):
        micro.verifier_passed(object(), {"kind": "active_title_regex", "pattern": "x"})


def test_xcursor_checkpoint_setup_is_fail_closed():
    outputs = iter(("already-patched\n", ""))
    calls = []

    def run_guest(argv, *, timeout_s=None):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=next(outputs), stderr="")

    micro._require_xcursor_repair(SimpleNamespace(run_guest=run_guest))
    assert len(calls) == 2
    assert "pyxcursor.py" in calls[0][-1]
    assert "Xcursor.display" in calls[1][-1]


def test_xcursor_checkpoint_setup_rejects_unknown_status():
    client = SimpleNamespace(
        run_guest=lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="unexpected\n", stderr=""
        )
    )
    with pytest.raises(RuntimeError, match="Xcursor repair returned"):
        micro._require_xcursor_repair(client)


def test_wandb_failure_propagates_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_PROJECT", "cua")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("wandb"))),
    )

    with pytest.raises(RuntimeError, match="wandb"):
        micro._init_wandb(tmp_path, {})


def test_model_attestation_uses_the_pinned_sglang_api(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"model_path": "/models/wrong"},
    )

    def get(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return response

    monkeypatch.setattr(micro.requests, "get", get)
    with pytest.raises(RuntimeError, match="not '/models/checkpoint'"):
        micro._assert_serving_model(31000, "owned", "/models/checkpoint")
    assert captured["url"] == "http://127.0.0.1:31000/model_info"


def test_owned_sglang_exit_is_fatal_without_network_polling():
    process = SimpleNamespace(poll=lambda: 7, returncode=7)
    with pytest.raises(RuntimeError, match="rc=7"):
        micro._wait_for_sglang(process, 31000, "owned")
