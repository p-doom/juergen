from __future__ import annotations

import base64
import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from grammars.ordered_events_v3_relative_1000_grid_v1.codec import CODEC
from PIL import Image

import cua_micro_eval as micro

SUITE = Path(__file__).with_name("cua_micro_tasks.json")
EXPECTED_TASKS = {
    "click.desktop.libreoffice_writer": ("native_launch", 1),
    "click.desktop.libreoffice_impress": ("native_launch", 1),
    "key.desktop.open_terminal": ("multi_turn", 4),
    "type.terminal.native_exact": ("native_app", 1),
    "type.text_editor.native_exact": ("native_app", 1),
    "key.writer.open_save_as": ("native_app", 1),
    "key.impress.open_save_as": ("native_app", 1),
    "click.files.open_eval_target": ("native_app", 1),
    "key.calculator.digit7": ("native_app", 1),
    "click.chrome.back": ("chrome_control", 1),
    "click.chrome.reload": ("chrome_control", 1),
    "click.chrome.deterministic_button": ("chrome_control", 1),
    "scroll.chrome.down": ("chrome_control", 1),
    "multi.calculator.73_plus_19": ("multi_turn", 64),
    "multi.chrome.search_3blue1brown": ("multi_turn", 64),
    "multi.chrome.open_chrome_search_wikipedia": ("multi_turn", 64),
    "multi.terminal.vim_hello_world_script": ("multi_turn", 64),
    "multi.terminal.hello_world_script": ("multi_turn", 64),
}


def test_suite_is_the_exact_18_task_contract():
    raw, tasks = micro.load_suite(SUITE)

    assert raw["suite"] == "cua_micro_tasks"
    assert [task.task_id for task in tasks] == list(EXPECTED_TASKS)
    assert {task.task_id: (task.category, task.max_turns) for task in tasks} == EXPECTED_TASKS
    assert all(task.expected == {"kind": "any"} for task in tasks if task.max_turns > 1)


def test_suite_payload_digest_rejects_any_benchmark_drift(tmp_path):
    payload = SUITE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == micro._SUITE_SHA256
    changed = tmp_path / SUITE.name
    changed.write_bytes(payload.replace(b"Click the LibreOffice Writer", b"Click Writer", 1))
    with pytest.raises(ValueError, match="suite digest mismatch"):
        micro.load_suite(changed)


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

    assert [(task.task_id, seed) for _, task, _, seed in plan] == [
        ("click.desktop.libreoffice_writer", 41000),
        ("click.desktop.libreoffice_writer", 41001),
        ("click.desktop.libreoffice_writer", 41002),
        ("click.desktop.libreoffice_writer", 41003),
        ("click.desktop.libreoffice_impress", 41000),
        ("click.desktop.libreoffice_impress", 41001),
        ("click.desktop.libreoffice_impress", 41002),
        ("click.desktop.libreoffice_impress", 41003),
        ("key.desktop.open_terminal", 41000),
        ("key.desktop.open_terminal", 41001),
        ("key.desktop.open_terminal", 41002),
        ("key.desktop.open_terminal", 41003),
        ("type.terminal.native_exact", 41000),
        ("type.terminal.native_exact", 41001),
        ("type.terminal.native_exact", 41002),
        ("type.terminal.native_exact", 41003),
        ("type.text_editor.native_exact", 41000),
        ("type.text_editor.native_exact", 41001),
        ("type.text_editor.native_exact", 41002),
        ("type.text_editor.native_exact", 41003),
        ("key.writer.open_save_as", 41000),
        ("key.writer.open_save_as", 41001),
        ("key.writer.open_save_as", 41002),
        ("key.writer.open_save_as", 41003),
        ("key.impress.open_save_as", 41000),
        ("key.impress.open_save_as", 41001),
        ("key.impress.open_save_as", 41002),
        ("key.impress.open_save_as", 41003),
        ("click.files.open_eval_target", 41000),
        ("click.files.open_eval_target", 41001),
        ("click.files.open_eval_target", 41002),
        ("click.files.open_eval_target", 41003),
        ("key.calculator.digit7", 41000),
        ("key.calculator.digit7", 41001),
        ("key.calculator.digit7", 41002),
        ("key.calculator.digit7", 41003),
        ("click.chrome.back", 41000),
        ("click.chrome.back", 41001),
        ("click.chrome.back", 41002),
        ("click.chrome.back", 41003),
        ("click.chrome.reload", 41000),
        ("click.chrome.reload", 41001),
        ("click.chrome.reload", 41002),
        ("click.chrome.reload", 41003),
        ("click.chrome.deterministic_button", 41000),
        ("click.chrome.deterministic_button", 41001),
        ("click.chrome.deterministic_button", 41002),
        ("click.chrome.deterministic_button", 41003),
        ("scroll.chrome.down", 41000),
        ("scroll.chrome.down", 41001),
        ("scroll.chrome.down", 41002),
        ("scroll.chrome.down", 41003),
        ("multi.calculator.73_plus_19", 41000),
        ("multi.calculator.73_plus_19", 41001),
        ("multi.calculator.73_plus_19", 41002),
        ("multi.calculator.73_plus_19", 41003),
        ("multi.chrome.search_3blue1brown", 41000),
        ("multi.chrome.search_3blue1brown", 41001),
        ("multi.chrome.search_3blue1brown", 41002),
        ("multi.chrome.search_3blue1brown", 41003),
        ("multi.chrome.open_chrome_search_wikipedia", 41000),
        ("multi.chrome.open_chrome_search_wikipedia", 41001),
        ("multi.chrome.open_chrome_search_wikipedia", 41002),
        ("multi.chrome.open_chrome_search_wikipedia", 41003),
        ("multi.terminal.vim_hello_world_script", 41000),
        ("multi.terminal.vim_hello_world_script", 41001),
        ("multi.terminal.vim_hello_world_script", 41002),
        ("multi.terminal.vim_hello_world_script", 41003),
        ("multi.terminal.hello_world_script", 41000),
        ("multi.terminal.hello_world_script", 41001),
        ("multi.terminal.hello_world_script", 41002),
        ("multi.terminal.hello_world_script", 41003),
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
        ("move(2,-3)", {"kind": "scroll", "sign": "down"}, False),
        ("scroll(0,-3); scroll(0,-3)", {"kind": "scroll", "sign": "down"}, False),
        ("down(LMB); up(LMB)", {"kind": "scroll", "sign": "down"}, False),
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
        "TERMINATE:success",
        "TERMINATE:  success",
        "TERMINATE:\tsuccess",
        "TERMINATE: success ",
        " TERMINATE: success",
    ):
        with pytest.raises(ValueError):
            micro._parse_response(invalid)


def test_image_part_forwards_raw_jpeg_bytes():
    payload = b"\xff\xd8raw-desktop-jpeg\xff\xd9"
    part = micro._image_part(payload)

    encoded = part["image_url"]["url"].removeprefix("data:image/jpeg;base64,")
    assert base64.b64decode(encoded) == payload


def _jpeg(*, quality=92, subsampling=2):
    encoded = io.BytesIO()
    Image.new("RGB", micro.OBSERVATION_SIZE, "white").save(
        encoded,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=False,
    )
    return encoded.getvalue()


def test_observation_attestation_requires_q92_420_rgb_before_consumption():
    assert micro._jpeg_image(_jpeg()).size == micro.OBSERVATION_SIZE
    with pytest.raises(RuntimeError, match=micro.OBSERVATION_CONTRACT):
        micro._jpeg_image(_jpeg(quality=85))
    with pytest.raises(RuntimeError, match=micro.OBSERVATION_CONTRACT):
        micro._jpeg_image(_jpeg(subsampling=0))


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


def test_json_outputs_are_atomic_and_failed_completion_leaves_no_marker(monkeypatch, tmp_path):
    result = tmp_path / "result.json"
    micro._write_json_atomic(result, {"completed": False})
    assert result.read_text() == '{\n  "completed": false\n}\n'

    completed = tmp_path / "completed.json"
    monkeypatch.setattr(
        micro.os,
        "replace",
        lambda source, target: (_ for _ in ()).throw(OSError("injected write failure")),
    )
    with pytest.raises(OSError, match="injected write failure"):
        micro._write_json_atomic(completed, {"completed": True})
    assert not completed.exists()
    assert not (tmp_path / "completed.json.tmp").exists()


def _run_one_turn(
    monkeypatch,
    tmp_path,
    *,
    response,
    action_error=None,
    receipt_error=None,
    held_keys=(),
    pointer_button_mask=0,
    events=None,
    expected=None,
):
    receipt = SimpleNamespace(
        ok=True,
        error=None,
        cursor_readback_verified=True,
        held_keys=held_keys,
        pointer_button_mask=pointer_button_mask,
    )

    class Client:
        def __init__(self):
            self.execution_count = 0

        def screen_size(self):
            return micro.OBSERVATION_SIZE

        def cursor_position(self):
            return 960, 540

        def screenshot_settled(self, **kwargs):
            return b"\xff\xd8raw-desktop-jpeg\xff\xd9"

        def execute(self, operations):
            self.execution_count += 1
            if self.execution_count == 2 and action_error is not None:
                raise action_error
            if self.execution_count == 2 and receipt_error is not None:
                return SimpleNamespace(ok=False, error=receipt_error)
            return receipt

    monkeypatch.setattr(micro, "prepare_task", lambda *args: None)
    monkeypatch.setattr(micro, "read_verifier_state", lambda *args: "before")
    monkeypatch.setattr(micro, "verifier_passed", lambda *args: (True, "after"))

    def call_model(**kwargs):
        if events is not None:
            events.append("model")
        return response, "stop"

    def attest_image(value):
        if events is not None:
            events.append("attest")
        return Image.new("RGB", micro.OBSERVATION_SIZE)

    monkeypatch.setattr(micro, "_call_model", call_model)
    monkeypatch.setattr(
        micro,
        "_jpeg_image",
        attest_image,
    )
    task = micro.Task(
        task_id="test.one_turn",
        category="native_app",
        instruction="Do it",
        setup={"kind": "desktop"},
        target={"kind": "fixed_norm", "bbox": [0, 0, 1000, 1000], "label": "screen"},
        expected={"kind": "any"} if expected is None else expected,
        verifier={"kind": "active_title_regex", "pattern": "after"},
        max_turns=1,
    )
    return micro.run_attempt(
        client=Client(),
        task=task,
        output_dir=tmp_path / "attempt",
        sglang_url="http://127.0.0.1:31000/v1",
        api_key="owned",
        model="/models/checkpoint",
        seed=41000,
        reset_receipt_sha256="reset-receipt",
    )


def test_run_attests_observation_before_model_consumption(monkeypatch, tmp_path):
    events = []
    result = _run_one_turn(monkeypatch, tmp_path, response="NO_OP", events=events)
    assert events[:2] == ["attest", "model"]
    assert result["reset_receipt_sha256"] == "reset-receipt"


@pytest.mark.parametrize(
    "response",
    ["move(2,-3)", "scroll(0,-3); scroll(0,-3)", "down(LMB); up(LMB)"],
)
def test_wrong_scroll_actions_are_scored_as_failed_attempts(monkeypatch, tmp_path, response):
    result = _run_one_turn(
        monkeypatch,
        tmp_path,
        response=response,
        expected={"kind": "scroll", "sign": "down"},
    )

    assert result["expected_action_ok"] is False
    assert result["success"] is False


def test_unknown_expected_action_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown expected action kind"):
        micro.action_matches_expected(CODEC.parse("move(1,1)"), {"kind": "unknown"})


def test_model_compile_and_held_state_failures_are_scored(monkeypatch, tmp_path):
    monkeypatch.setattr(
        type(micro.CODEC),
        "compile_action",
        lambda *args: (_ for _ in ()).throw(ValueError("unsupported model action")),
    )
    compile_result = _run_one_turn(
        monkeypatch,
        tmp_path / "compile",
        response="down(NotAKey)",
    )
    assert compile_result["success"] is False
    assert compile_result["turns"][0]["parse_error"] == "unsupported model action"

    monkeypatch.undo()
    held_result = _run_one_turn(
        monkeypatch,
        tmp_path / "held",
        response="up(KeyA)",
    )
    assert held_result["success"] is False
    assert held_result["turns"][0]["parse_error"] == "key not held: a"

    monkeypatch.undo()
    keymap_result = _run_one_turn(
        monkeypatch,
        tmp_path / "keymap",
        response="down(NotAKey)",
    )
    assert keymap_result["success"] is False
    assert "unsupported X11 key" in keymap_result["turns"][0]["parse_error"]

    monkeypatch.undo()
    overflow_result = _run_one_turn(
        monkeypatch,
        tmp_path / "overflow",
        response=f"move({10**400},0)",
    )
    assert overflow_result["success"] is False
    assert overflow_result["turns"][0]["parse_error"]

    monkeypatch.undo()
    control_text_result = _run_one_turn(
        monkeypatch,
        tmp_path / "control-text",
        response='type("\x80")',
    )
    assert control_text_result["success"] is False
    assert control_text_result["turns"][0]["parse_error"] == (
        "typing text contains unsupported U+0080"
    )


def test_action_receipt_failure_remains_fatal(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="desktop action failed: rejected"):
        _run_one_turn(
            monkeypatch,
            tmp_path,
            response="down(KeyA); up(KeyA)",
            receipt_error="rejected",
        )

    monkeypatch.undo()
    with pytest.raises(micro.ExecutionError, match="dispatch failed"):
        _run_one_turn(
            monkeypatch,
            tmp_path / "dispatch",
            response="down(KeyA); up(KeyA)",
            action_error=micro.ExecutionError("dispatch failed"),
        )


def test_terminal_failure_and_held_inputs_cannot_score_success(monkeypatch, tmp_path):
    terminated = _run_one_turn(
        monkeypatch,
        tmp_path / "failure",
        response="NO_OP\nTERMINATE: failure",
    )
    assert terminated["turns"][0]["verifier_pass"] is True
    assert terminated["success"] is False
    assert terminated["stop_reason"] == "model terminated: failure"

    monkeypatch.undo()
    held = _run_one_turn(
        monkeypatch,
        tmp_path / "held",
        response="down(KeyA)\nTERMINATE: success",
        held_keys=("a",),
    )
    assert held["turns"][0]["input_state_ok"] is False
    assert held["success"] is False


def test_guest_timeout_is_not_treated_as_a_failed_verifier(monkeypatch):
    monkeypatch.setattr(
        micro,
        "read_verifier_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("guest")),
    )
    with pytest.raises(TimeoutError, match="guest"):
        micro.verifier_passed(object(), {"kind": "active_title_regex", "pattern": "x"})


def test_xcursor_checkpoint_provisions_once_then_restores_and_reattests(monkeypatch):
    old = "self.display = self.xlib.XOpenDisplay(display)"
    new = "Xcursor.display = self.xlib.XOpenDisplay(display)"
    fresh_source = f"class Xcursor:\n    {old}\n"
    events = []

    class Session:
        base_url = "http://desktop.test"

        def __init__(self):
            self.source = fresh_source
            self.checkpoint = None
            self.checkpoint_name = "base"
            self.reset_sequence = 0
            self.runtime = SimpleNamespace(has_checkpoint=lambda name: self.checkpoint is not None)

        def run_guest(self, argv, *, timeout_s=None):
            if argv[:2] == ["python3", "-c"] and argv[-1] == micro._XCURSOR_REPAIR:
                events.append("patch")
                if self.source.count(old) != 1 or new in self.source:
                    return SimpleNamespace(returncode=1, stdout="", stderr="bad preimage")
                self.source = self.source.replace(old, new)
            elif argv[:2] == ["bash", "-lc"]:
                events.append("restart")
                assert new in self.source and old not in self.source
            else:
                events.append("verify-bytes")
                if self.source.count(new) != 1 or old in self.source:
                    return SimpleNamespace(returncode=1, stdout="", stderr="bad patch")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def reset_to_checkpoint(self, name, *, setup):
            assert name == "cua_micro_xcursor_v1"
            assert self.checkpoint is None
            events.append("reset-fresh")
            assert self.source == fresh_source
            setup(self)
            events.append("save-checkpoint")
            self.checkpoint = self.source

        def reset_with_receipt(self):
            events.append("reset-with-receipt")
            self.source = self.checkpoint
            self.reset_sequence += 1
            return self, SimpleNamespace(
                checkpoint_name=self.checkpoint_name,
                receipt_sha256=f"receipt-{self.reset_sequence}",
            )

        def consume_receipt(self, receipt):
            events.append(("consume-receipt", receipt.receipt_sha256))

    class Probe:
        def __init__(self, base_url):
            assert base_url == "http://desktop.test"

        def wait_ready(self, *, timeout_s):
            assert timeout_s == 120
            events.append("wait-ready")

        def verify_actions_contract(self):
            events.append("verify-actions")

    monkeypatch.setattr(micro, "DesktopClient", Probe)
    monkeypatch.setattr(micro.time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    session = Session()
    first = micro._reset_xcursor_checkpoint(session)
    session.source = fresh_source
    second = micro._reset_xcursor_checkpoint(session)

    assert session.source == fresh_source.replace(old, new)
    assert (first.receipt_sha256, second.receipt_sha256) == ("receipt-1", "receipt-2")
    assert events == [
        "reset-fresh",
        "patch",
        "restart",
        ("sleep", 9),
        "wait-ready",
        "verify-actions",
        "verify-bytes",
        "save-checkpoint",
        "reset-with-receipt",
        ("consume-receipt", "receipt-1"),
        "verify-bytes",
        "reset-with-receipt",
        ("consume-receipt", "receipt-2"),
        "verify-bytes",
    ]


def test_xcursor_checkpoint_rejects_an_already_patched_base():
    client = SimpleNamespace(
        run_guest=lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="unexpected pyxcursor preimage"
        )
    )
    with pytest.raises(RuntimeError, match="guest command failed"):
        micro._require_xcursor_repair(client)

    preexisting = SimpleNamespace(
        checkpoint_name="base",
        runtime=SimpleNamespace(has_checkpoint=lambda name: True),
    )
    with pytest.raises(RuntimeError, match="unexpected pre-existing Xcursor checkpoint"):
        micro._reset_xcursor_checkpoint(preexisting)


def test_four_owned_slots_stop_after_infrastructure_failure():
    started = []
    barrier = micro.threading.Barrier(micro._VM_SLOTS)

    def worker(value, session):
        assert session == value % 4
        started.append(value)
        barrier.wait()
        if value == 0:
            raise RuntimeError("infrastructure")

    with pytest.raises(RuntimeError, match="infrastructure"):
        micro._run_slots(tuple(range(4)), tuple(range(5)), worker)
    assert set(started) == {0, 1, 2, 3}


def test_pool_close_waits_for_every_tracked_resource(monkeypatch):
    closing = {
        "closed": True,
        "ready": 0,
        "starting": 1,
        "leased": 0,
        "retiring": 1,
        "total_failed": 0,
        "sessions": [],
        "starting_sessions": [{"session_id": "starting"}],
    }
    closed = {
        **closing,
        "starting": 0,
        "retiring": 0,
        "starting_sessions": [],
    }

    class Pool:
        def __init__(self):
            self.states = iter((closing, closed))
            self.closed = False

        def close(self):
            self.closed = True

        def snapshot(self):
            return next(self.states)

    monkeypatch.setattr(micro.time, "sleep", lambda seconds: None)
    pool = Pool()
    assert micro._close_desktop_pool(pool) == closed
    assert pool.closed is True


def test_wandb_failure_propagates_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_PROJECT", "cua")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("wandb"))),
    )

    with pytest.raises(RuntimeError, match="wandb"):
        micro._init_wandb(tmp_path, {})


def test_runtime_preflight_requires_ffmpeg6_before_output_creation(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    desktop_image = tmp_path / "desktop.qcow2"
    desktop_image.write_bytes(b"qcow2")
    output_dir = tmp_path / "output"
    monkeypatch.setitem(sys.modules, "torchcodec", SimpleNamespace(ffmpeg_major_version=5))
    monkeypatch.setattr(
        micro.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preflight must precede process startup")
        ),
    )

    with pytest.raises(RuntimeError, match="requires FFmpeg 6, got 5"):
        micro.main(
            [
                "--model-path",
                str(model_path),
                "--desktop-image",
                str(desktop_image),
                "--output-dir",
                str(output_dir),
            ]
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("failure", [None, "init", "run", "log", "finish"])
def test_wandb_lifecycle_controls_completion_marker(monkeypatch, tmp_path, failure):
    model_path = tmp_path / "model"
    model_path.mkdir()
    desktop_image = tmp_path / "desktop.qcow2"
    desktop_image.write_bytes(b"qcow2")
    output_dir = tmp_path / "output"
    events = []

    class PortLease:
        start = 31000

        def release(self):
            events.append("release-port")

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("stop-model")
            self.returncode = 0

        def wait(self, *, timeout):
            return self.returncode

        def kill(self):
            raise AssertionError("graceful model termination should succeed")

    class Checked:
        env = object()

        def tracked_env(self):
            return self.env

    class Pool:
        def __init__(self):
            self.started = False
            self.closed = False
            self.checked = [Checked() for _ in range(micro._VM_SLOTS)]

        def start(self):
            events.append("start-pool")
            self.started = True

        def checkout(self, *, timeout_s):
            assert timeout_s == 1200
            return self.checked.pop()

        def close(self):
            events.append("close-pool")
            self.closed = True

        def snapshot(self):
            if self.closed:
                return {
                    "closed": True,
                    "ready": 0,
                    "starting": 0,
                    "leased": 0,
                    "retiring": 0,
                    "total_failed": 0,
                    "sessions": [],
                    "starting_sessions": [],
                }
            assert self.started
            return {
                "closed": False,
                "ready": 0,
                "starting": 0,
                "leased": micro._VM_SLOTS,
                "retiring": 0,
                "total_failed": 0,
                "sessions": [{} for _ in range(micro._VM_SLOTS)],
                "starting_sessions": [],
            }

    class WandbRun:
        def log(self, scores):
            events.append("wandb-log")
            if failure == "log":
                raise RuntimeError("wandb log")

        def finish(self, *, exit_code):
            events.append(("wandb-finish", exit_code))
            if failure == "finish":
                raise RuntimeError("wandb finish")

    pool = Pool()
    process = Process()
    monkeypatch.setattr(micro, "_preflight_runtime", lambda: None)
    monkeypatch.setattr(micro, "acquire_port_range", lambda **kwargs: PortLease())
    monkeypatch.setattr(micro.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(micro, "build_desktop_pool", lambda **kwargs: pool)
    monkeypatch.setattr(micro, "_wait_for_sglang", lambda *args: None)
    monkeypatch.setattr(micro, "_assert_serving_model", lambda *args: None)

    def init_wandb(*args):
        events.append("wandb-init")
        if failure == "init":
            raise RuntimeError("wandb init")
        return WandbRun()

    monkeypatch.setattr(micro, "_init_wandb", init_wandb)
    monkeypatch.setattr(
        micro,
        "attempt_plan",
        lambda tasks: [(index, task, 0, 41000) for index, task in enumerate(tasks[:4])],
    )
    monkeypatch.setattr(
        micro,
        "_reset_xcursor_checkpoint",
        lambda client: SimpleNamespace(receipt_sha256="reset-receipt"),
    )

    def run_attempt(**kwargs):
        events.append("run-attempt")
        if failure == "run":
            raise RuntimeError("attempt infrastructure")
        return {"task_id": kwargs["task"].task_id, "seed": kwargs["seed"]}

    monkeypatch.setattr(micro, "run_attempt", run_attempt)
    monkeypatch.setattr(
        micro,
        "aggregate_results",
        lambda tasks, attempts: {"scores": {"overall": 1.0}},
    )
    write_json_atomic = micro._write_json_atomic

    def record_write(path, payload):
        write_json_atomic(path, payload)
        events.append(("write", path.name))

    monkeypatch.setattr(micro, "_write_json_atomic", record_write)
    argv = [
        "--model-path",
        str(model_path),
        "--desktop-image",
        str(desktop_image),
        "--output-dir",
        str(output_dir),
    ]

    if failure is None:
        assert micro.main(argv) == 0
        assert output_dir.joinpath("completed.json").is_file()
        assert events.index(("wandb-finish", 0)) < events.index(("write", "completed.json"))
    else:
        expected_error = BaseExceptionGroup if failure == "finish" else RuntimeError
        with pytest.raises(expected_error):
            micro.main(argv)
        assert not output_dir.joinpath("completed.json").exists()
        if failure in {"run", "log"}:
            assert ("wandb-finish", 1) in events
        if failure == "init":
            assert not any(
                isinstance(event, tuple) and event[0] == "wandb-finish" for event in events
            )


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
