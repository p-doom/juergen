from __future__ import annotations

from osworld_parity.proper_vm_capability_ladder.rung1.transport import RecordingTransport
from osworld_parity.sign_of_life_v2.actions import compile_native_absolute, execute_native_absolute
from osworld_parity.sign_of_life_v2.oracle import evaluate
from osworld_parity.sign_of_life_v2.runner import _select_tasks
from osworld_parity.sign_of_life_v2.suite import load_suite


def _base(task_id: str) -> dict:
    return {"schema_version": 1, "task_id": task_id}


def test_single_fixed_suite_contract() -> None:
    suite = load_suite()
    assert suite.role == "single_fixed_development_gate"
    assert suite.final_benchmark is False
    assert len(suite.tasks) == 4
    assert len(suite.manifest_sha256) == 64


def test_task_index_is_only_an_execution_selector() -> None:
    suite = load_suite()
    assert [task.id for task in _select_tasks(suite.tasks, None)] == [task.id for task in suite.tasks]
    assert [task.id for task in _select_tasks(suite.tasks, 2)] == ["desktop_open_chrome"]
    for invalid in (-1, len(suite.tasks)):
        try:
            _select_tasks(suite.tasks, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"task index {invalid} should have failed")


def test_ls_requires_command_and_real_output() -> None:
    task = load_suite().by_id("terminal_ls")
    marker = task.expected["listing_marker"]
    gold = {**_base(task.id), "history": "    1  ls\n", "transcript": f"SOLV2-LS$ ls\n{marker}\nSOLV2-LS$ ", "prompt_count": 2}
    assert evaluate(task, gold).success
    assert not evaluate(task, {**gold, "history": "pwd\n"}).success
    assert not evaluate(task, {**gold, "transcript": "SOLV2-LS$ ls\nSOLV2-LS$ "}).success


def test_exact_text_wrong_text_fails_closed() -> None:
    task = load_suite().by_id("terminal_exact_text")
    gold = {**_base(task.id), "capture_file_exists": True, "captured_text": task.expected["text"]}
    assert evaluate(task, gold).success
    assert not evaluate(task, {**gold, "captured_text": "wrong text"}).success
    assert not evaluate(task, {**gold, "capture_file_exists": False}).success


def test_chrome_requires_process_and_foreground_not_click_ack() -> None:
    task = load_suite().by_id("desktop_open_chrome")
    gold = {**_base(task.id), "chrome_process": True, "active_window": 'WM_CLASS = "google-chrome", "Google-chrome"', "windows": "chrome"}
    assert evaluate(task, gold).success
    assert not evaluate(task, {**gold, "chrome_process": False}).success
    assert not evaluate(task, {**gold, "active_window": 'WM_CLASS = "gnome-shell"'}).success


def test_compound_requires_focus_history_and_exact_file() -> None:
    task = load_suite().by_id("focus_terminal_and_type")
    gold = {
        **_base(task.id),
        "active_window": 'WM_CLASS = "gnome-terminal-server", "Gnome-terminal"',
        "history": task.expected["command"],
        "proof_file_exists": True,
        "proof_file_content": task.expected["content"],
    }
    assert evaluate(task, gold).success
    assert not evaluate(task, {**gold, "active_window": 'WM_CLASS = "xmessage"'}).success
    assert not evaluate(task, {**gold, "proof_file_content": "wrong"}).success


def test_native_absolute_model_and_gold_share_atomic_primitives() -> None:
    type_args = {"action": "type", "text": "ls"}
    click_args = {"action": "left_click", "coordinate": [35, 60]}
    assert [op.kind for op in compile_native_absolute(type_args, (1920, 1080))] == ["ascii_type"]
    assert [op.kind for op in compile_native_absolute(click_args, (1920, 1080))] == ["move_to", "mouse_down", "mouse_up"]
    transport = RecordingTransport(screen=(1920, 1080))
    receipt = execute_native_absolute(transport, click_args)
    assert receipt["receipt"]["ok"] is True
    assert transport.atomic_invocations == 1
    assert transport.cursor_position() == (35, 60)


def test_missing_or_malformed_evidence_is_an_oracle_error() -> None:
    task = load_suite().by_id("terminal_ls")
    result = evaluate(task, {"schema_version": 99, "task_id": task.id})
    assert result.oracle_status == "error"
    assert result.success is False
