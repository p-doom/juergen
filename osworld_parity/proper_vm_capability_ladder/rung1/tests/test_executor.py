from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.executor import (
    CompactRawExecutor,
    NativeAbsoluteExecutor,
)
from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import load_manifest
from osworld_parity.proper_vm_capability_ladder.rung1.trajectory import GoldTrajectory
from osworld_parity.proper_vm_capability_ladder.rung1.selfcheck import (
    SelfcheckError,
    _assert_dispatch_journal,
    _execute,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import (
    CLIPBOARD_OWNER_LIFETIME_MS,
    CLIPBOARD_PASTE_DELAY_MS,
    CLICK_BACKENDS,
    CLICK_DWELL_S,
    DIRECT_XTEST_CLICK_BACKEND,
    ATOMIC_RESULT_PREFIX,
    HttpVmTransport,
    Operation,
    PASSIVE_X_OBSERVER_LIMITATION,
    PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    RecordingTransport,
    TransportError,
    compile_atomic_guest_program,
    compile_unicode_coalesced_type,
    lower_guest_operations,
)


def _kinds(transport: RecordingTransport) -> list[str]:
    return [operation.kind for operation in transport.audit.operations]


class StatefulClickTransport(RecordingTransport):
    """Tiny GUI-state model: a click requires down/up on the same hitbox."""

    def __init__(self, *, cursor: tuple[int, int]) -> None:
        super().__init__(cursor=cursor)
        self.target = (290, 390, 310, 410)
        self.checked = False
        self._down_hit = ""

    def _hit(self) -> str:
        x, y = self.cursor_position()
        left, top, right, bottom = self.target
        return "target" if left <= x < right and top <= y < bottom else ""

    def mouse_down(self, button: str = "left") -> None:
        super().mouse_down(button)
        if button == "left":
            self._down_hit = self._hit()

    def mouse_up(self, button: str = "left") -> None:
        up_hit = self._hit()
        super().mouse_up(button)
        if button == "left" and self._down_hit == up_hit == "target":
            self.checked = not self.checked
        self._down_hit = ""


class SingleProcessHttpTransport(HttpVmTransport):
    def __init__(self) -> None:
        super().__init__("http://not-used.invalid")
        self.execute_calls: list[list[str]] = []
        self.cursor = (10, 20)

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def execute_argv(self, argv: list[str], *, check: bool = True) -> dict:
        self.execute_calls.append(argv)
        self.cursor = (300, 400)
        payload = {
            "_r1a_schema": 1,
            "ok": True,
            "cursor": [300, 400],
            "cursor_before": [10, 20],
            "cursor_after": [300, 400],
            "pointer_button_mask": 0,
            "observed_pointer_button_mask": 0,
            "expected_pointer_button_mask": 0,
            "guest_process_count": 1,
            "cleanup_attempted": False,
            "error": None,
            "failure_kind": None,
            "operations": [
                {"kind": "move_to", "args": [300, 400]},
                {"kind": "mouse_down", "args": ["left"]},
                {"kind": "mouse_up", "args": ["left"]},
            ],
            "semantic_operations": [
                {"kind": "move_relative", "args": [290, 380]},
                {"kind": "mouse_down", "args": ["left"]},
                {"kind": "mouse_up", "args": ["left"]},
            ],
            "lowered_operations": [
                {"kind": "move_relative", "args": [290, 380]},
                {"kind": "click", "args": ["left"]},
            ],
            "backend_primitives": [
                {
                    "kind": "click",
                    "click_backend": PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
                    "release_side_motion_notify": True,
                    "injection_attempt_count": 1,
                    "retry_count": 0,
                    "dwell_ms": 50,
                    "press_xtest_sequence": ["motion_notify", "button_press"],
                    "release_xtest_sequence": [
                        "motion_notify",
                        "button_release",
                    ],
                }
            ],
            "x_event_sync_evidence": [],
            "click_backend": PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
            "x_injection_evidence": [],
            "x_injection_timestamps": [
                {
                    "click_backend": PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
                    "backend_identity": "fake_x11",
                    "release_side_motion_notify": True,
                    "clock": "time.monotonic_ns",
                    "dwell_requested_ns": 50_000_000,
                    "click_started_guest_monotonic_ns": 1,
                    "press_call_before_guest_monotonic_ns": 2,
                    "press_call_after_guest_monotonic_ns": 3,
                    "press_sync_completed_guest_monotonic_ns": 4,
                    "dwell_started_guest_monotonic_ns": 5,
                    "dwell_completed_guest_monotonic_ns": 6,
                    "dwell_duration_ns": 1,
                    "release_call_before_guest_monotonic_ns": 7,
                    "release_call_after_guest_monotonic_ns": 8,
                    "release_sync_completed_guest_monotonic_ns": 9,
                    "click_completed_guest_monotonic_ns": 10,
                    "press_xtest_sequence": ["motion_notify", "button_press"],
                    "release_xtest_sequence": [
                        "motion_notify",
                        "button_release",
                    ],
                }
            ],
            "passive_x_observer": {
                "installed": False,
                "observer_process_count": 0,
                "additional_x_connection_count": 0,
                "assessment": "omitted_not_demonstrably_non_perturbing",
                "limitation": PASSIVE_X_OBSERVER_LIMITATION,
            },
        }
        return {
            "status": "success",
            "returncode": 0,
            "output": ATOMIC_RESULT_PREFIX + json.dumps(payload),
        }


def _execute_compiled_click(
    operations: tuple[Operation, ...],
    click_backend: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> dict:
    state = {"x": 10, "y": 20, "mask": 0}

    class FakeX:
        ButtonPress = 4
        ButtonRelease = 5
        MotionNotify = 6

    class FakeDisplay:
        def flush(self) -> None:
            return None

        def sync(self) -> None:
            return None

        def screen(self):
            root = SimpleNamespace(
                query_pointer=lambda: SimpleNamespace(
                    root_x=state["x"], root_y=state["y"], mask=state["mask"]
                )
            )
            return SimpleNamespace(root=root)

    backend = ModuleType("fake_pyautogui_x11")
    backend.X = FakeX
    backend.BUTTON_NAME_MAPPING = {
        "left": 1,
        "middle": 2,
        "right": 3,
        1: 1,
        2: 2,
        3: 3,
    }
    backend._display = FakeDisplay()

    def fake_input(_display, event_type, detail=0, **kwargs) -> None:
        if event_type == FakeX.MotionNotify:
            state["x"] = int(kwargs["x"])
            state["y"] = int(kwargs["y"])
        elif event_type == FakeX.ButtonPress:
            state["mask"] |= 1 << (7 + int(detail))
        elif event_type == FakeX.ButtonRelease:
            state["mask"] &= ~(1 << (7 + int(detail)))

    backend.fake_input = fake_input

    def move_to(x, y) -> None:
        backend.fake_input(backend._display, FakeX.MotionNotify, x=x, y=y)
        backend._display.sync()

    def mouse_down(x, y, button) -> None:
        move_to(x, y)
        backend.fake_input(
            backend._display, FakeX.ButtonPress, backend.BUTTON_NAME_MAPPING[button]
        )
        backend._display.sync()

    def mouse_up(x, y, button) -> None:
        move_to(x, y)
        backend.fake_input(
            backend._display, FakeX.ButtonRelease, backend.BUTTON_NAME_MAPPING[button]
        )
        backend._display.sync()

    def click(x, y, button) -> None:
        mapped = backend.BUTTON_NAME_MAPPING[button]
        backend._mouseDown(x, y, mapped)
        backend._mouseUp(x, y, mapped)

    backend._moveTo = move_to
    backend._mouseDown = mouse_down
    backend._mouseUp = mouse_up
    backend._click = click

    pyautogui = ModuleType("pyautogui")
    pyautogui.platformModule = backend
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    pyautogui.position = lambda: (state["x"], state["y"])
    pyautogui.size = lambda: (1920, 1080)
    pyautogui.moveTo = lambda x, y: move_to(int(x), int(y))
    pyautogui.click = lambda *, clicks, interval, button: backend._click(
        state["x"], state["y"], button
    )
    pyautogui.keyUp = lambda _key: None
    pyautogui.mouseUp = lambda *, button: mouse_up(
        state["x"], state["y"], button
    )
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    program, expected_mask = compile_atomic_guest_program(
        operations,
        initial_buttons=set(),
        initial_keys=set(),
        click_backend=click_backend,
    )
    assert expected_mask == 0
    compile(program, "<rung1a-atomic-ab>", "exec")
    exec(program, {})
    markers = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(ATOMIC_RESULT_PREFIX)
    ]
    assert len(markers) == 1
    return json.loads(markers[0][len(ATOMIC_RESULT_PREFIX) :])

def test_click_adapters_match_cursor_and_button_transitions() -> None:
    native_transport = RecordingTransport(cursor=(10, 20))
    native_result = NativeAbsoluteExecutor(native_transport).execute(
        {"action": "left_click", "coordinate": [300, 400]}
    )
    raw_transport = RecordingTransport(cursor=(10, 20))
    raw_result = CompactRawExecutor(raw_transport).execute("290 380 0 ; +LMB -LMB")
    assert native_transport.cursor_position() == raw_transport.cursor_position() == (300, 400)
    assert _kinds(native_transport) == _kinds(raw_transport) == [
        "move_to",
        "mouse_down",
        "mouse_up",
    ]
    assert not native_transport.audit.held_buttons
    assert not raw_transport.audit.held_buttons
    assert native_result.atomic_state is not None
    assert raw_result.atomic_state is not None
    assert native_result.atomic_state["pointer_button_mask"] == 0
    assert raw_result.atomic_state["pointer_button_mask"] == 0
    expected_semantics = (
        Operation("mouse_down", ("left",)),
        Operation("mouse_up", ("left",)),
    )
    assert native_transport.atomic_inputs == [
        (Operation("move_to", (300, 400)),) + expected_semantics
    ]
    assert raw_transport.atomic_inputs == [
        (Operation("move_relative", (290, 380)),) + expected_semantics
    ]
    assert lower_guest_operations(native_transport.atomic_inputs[0]) == (
        Operation("move_to", (300, 400)),
        Operation("click", ("left",)),
    )
    assert lower_guest_operations(raw_transport.atomic_inputs[0]) == (
        Operation("move_relative", (290, 380)),
        Operation("click", ("left",)),
    )


def test_compact_click_is_one_guest_process_with_compiled_order() -> None:
    transport = SingleProcessHttpTransport()
    result = CompactRawExecutor(transport).execute("290 380 0 ; +LMB -LMB")
    assert len(transport.execute_calls) == 1
    argv = transport.execute_calls[0]
    assert argv[:2] == ["python", "-c"]
    program = argv[2]
    compile(program, "<rung1a-atomic>", "exec")
    assert program.index("RUNG1A_ATOMIC_STEP_0:move_relative") < program.index(
        "RUNG1A_ATOMIC_STEP_1:click"
    )
    assert "pyautogui.click(clicks=1,interval=0.05,button=_button)" in program
    assert "_r1a_time.sleep(0.05)" in program
    assert "_r1a_sync_after_x_event('mouse_down')" in program
    assert "_r1a_sync_after_x_event('mouse_up')" in program
    assert "'x_event_sync_evidence':_r1a_x_event_sync" in program
    assert result.executor_dispatch_status == "ok"
    assert result.atomic_state is not None
    assert result.atomic_state["guest_process_count"] == 1
    assert result.atomic_state["pointer_button_mask"] == 0
    assert result.atomic_state["cursor_readback_verified"] is True


@pytest.mark.parametrize("click_backend", sorted(CLICK_BACKENDS))
def test_click_backend_is_action_format_neutral(
    click_backend: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    native = _execute_compiled_click(
        (
            Operation("move_to", (300, 400)),
            Operation("mouse_down", ("left",)),
            Operation("mouse_up", ("left",)),
        ),
        click_backend,
        monkeypatch,
        capsys,
    )
    compact = _execute_compiled_click(
        (
            Operation("move_relative", (290, 380)),
            Operation("mouse_down", ("left",)),
            Operation("mouse_up", ("left",)),
        ),
        click_backend,
        monkeypatch,
        capsys,
    )

    assert native["semantic_operations"][-2:] == compact["semantic_operations"][-2:] == [
        {"kind": "mouse_down", "args": ["left"]},
        {"kind": "mouse_up", "args": ["left"]},
    ]
    assert native["lowered_operations"][-1] == compact["lowered_operations"][-1] == {
        "kind": "click",
        "args": ["left"],
    }
    native_click = [
        item for item in native["backend_primitives"] if item["kind"] == "click"
    ]
    compact_click = [
        item for item in compact["backend_primitives"] if item["kind"] == "click"
    ]
    assert native_click == compact_click
    assert native_click[0]["injection_attempt_count"] == 1
    assert native_click[0]["retry_count"] == 0
    assert native["guest_process_count"] == compact["guest_process_count"] == 1
    assert [
        (item["phase"], item["event"], item["detail"])
        for item in native["x_injection_evidence"]
    ] == [
        (item["phase"], item["event"], item["detail"])
        for item in compact["x_injection_evidence"]
    ]
    assert native["passive_x_observer"] == compact["passive_x_observer"] == {
        "installed": False,
        "observer_process_count": 0,
        "additional_x_connection_count": 0,
        "assessment": "omitted_not_demonstrably_non_perturbing",
        "limitation": PASSIVE_X_OBSERVER_LIMITATION,
    }


def test_click_backends_differ_only_by_release_side_motion_notify(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    operations = (
        Operation("move_to", (300, 400)),
        Operation("mouse_down", ("left",)),
        Operation("mouse_up", ("left",)),
    )
    current = _execute_compiled_click(
        operations,
        PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
        monkeypatch,
        capsys,
    )
    direct = _execute_compiled_click(
        operations,
        DIRECT_XTEST_CLICK_BACKEND,
        monkeypatch,
        capsys,
    )

    assert current["semantic_operations"] == direct["semantic_operations"]
    assert current["lowered_operations"] == direct["lowered_operations"]
    assert [item["event"] for item in current["x_injection_evidence"] if item["phase"] == "press"] == [
        "motion_notify",
        "button_press",
    ]
    assert [item["event"] for item in direct["x_injection_evidence"] if item["phase"] == "press"] == [
        "motion_notify",
        "button_press",
    ]
    current_release = [
        item for item in current["x_injection_evidence"] if item["phase"] == "release"
    ]
    direct_release = [
        item for item in direct["x_injection_evidence"] if item["phase"] == "release"
    ]
    assert [item["event"] for item in current_release] == [
        "motion_notify",
        "button_release",
    ]
    assert [item["event"] for item in direct_release] == ["button_release"]
    assert current_release[1]["detail"] == direct_release[0]["detail"] == 1
    assert current_release[1]["x"] == direct_release[0]["x"] is None
    assert current_release[1]["y"] == direct_release[0]["y"] is None
    current_primitive = next(
        item for item in current["backend_primitives"] if item["kind"] == "click"
    )
    direct_primitive = next(
        item for item in direct["backend_primitives"] if item["kind"] == "click"
    )
    for invariant in (
        "call",
        "dwell_ms",
        "ordering",
        "injection_attempt_count",
        "retry_count",
        "press_xtest_sequence",
    ):
        assert current_primitive[invariant] == direct_primitive[invariant]
    assert [item["event"] for item in current["x_event_sync_evidence"]] == [
        "mouse_down",
        "mouse_up",
    ]
    assert [item["event"] for item in direct["x_event_sync_evidence"]] == [
        "mouse_down",
        "mouse_up",
    ]

    timestamp_fields = (
        "press_call_before_guest_monotonic_ns",
        "press_call_after_guest_monotonic_ns",
        "press_sync_completed_guest_monotonic_ns",
        "dwell_started_guest_monotonic_ns",
        "dwell_completed_guest_monotonic_ns",
        "release_call_before_guest_monotonic_ns",
        "release_call_after_guest_monotonic_ns",
        "release_sync_completed_guest_monotonic_ns",
    )
    for payload, backend, release_motion in (
        (current, PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND, True),
        (direct, DIRECT_XTEST_CLICK_BACKEND, False),
    ):
        assert len(payload["x_injection_timestamps"]) == 1
        timing = payload["x_injection_timestamps"][0]
        assert timing["click_backend"] == backend
        assert timing["backend_identity"] == "fake_pyautogui_x11"
        assert timing["release_side_motion_notify"] is release_motion
        assert timing["dwell_requested_ns"] == int(CLICK_DWELL_S * 1e9)
        assert timing["dwell_duration_ns"] >= int(CLICK_DWELL_S * 1e9)
        timestamps = [timing[field] for field in timestamp_fields]
        assert timestamps == sorted(timestamps)


def test_unknown_click_backend_fails_before_guest_dispatch() -> None:
    transport = SingleProcessHttpTransport()
    with pytest.raises(TransportError, match="unsupported click backend"):
        transport.execute_atomic(
            (
                Operation("mouse_down", ("left",)),
                Operation("mouse_up", ("left",)),
            ),
            click_backend="unknown",
        )
    assert transport.execute_calls == []


def test_atomic_exception_releases_preexisting_and_new_buttons() -> None:
    transport = RecordingTransport()
    transport.mouse_down("right")
    result = transport.execute_atomic(
        (
            Operation("mouse_down", ("left",)),
            Operation("raise_for_test", ("injected failure",)),
        )
    )
    assert result.ok is False
    assert result.cleanup_attempted is True
    assert result.pointer_button_mask == 0
    assert result.guest_returncode == 1
    assert result.failure_kind == "injected"
    assert result.raw_result_marker.startswith(ATOMIC_RESULT_PREFIX)
    assert transport.audit.held_buttons == set()
    assert "injected failure" in (result.error or "")

    program, _ = compile_atomic_guest_program(
        (
            Operation("mouse_down", ("left",)),
            Operation("raise_for_test", ("injected failure",)),
        ),
        initial_buttons=set(),
        initial_keys=set(),
    )
    assert "except BaseException" in program
    assert "pyautogui.mouseUp(button=_button)" in program
    assert "sys.exit(1)" in program


def test_http_atomic_failure_preserves_nonzero_marker_evidence() -> None:
    class FailedHttpTransport(HttpVmTransport):
        def __init__(self) -> None:
            super().__init__("http://not-used.invalid")

        def execute_argv(self, argv: list[str], *, check: bool = True) -> dict:
            payload = {
                "_r1a_schema": 1,
                "ok": False,
                "cursor": [10, 20],
                "cursor_before": [10, 20],
                "cursor_after": [10, 20],
                "pointer_button_mask": 0,
                "observed_pointer_button_mask": -1,
                "expected_pointer_button_mask": 0,
                "guest_process_count": 1,
                "cleanup_attempted": True,
                "error": "RuntimeError: injected failure",
                "failure_kind": "injected",
                "operations": [],
                "semantic_operations": [
                    {"kind": "raise_for_test", "args": ["injected failure"]}
                ],
                "lowered_operations": [
                    {"kind": "raise_for_test", "args": ["injected failure"]}
                ],
                "backend_primitives": [],
                "x_event_sync_evidence": [],
                "click_backend": PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
                "x_injection_evidence": [],
                "x_injection_timestamps": [],
                "passive_x_observer": {
                    "installed": False,
                    "observer_process_count": 0,
                    "additional_x_connection_count": 0,
                    "assessment": "omitted_not_demonstrably_non_perturbing",
                    "limitation": PASSIVE_X_OBSERVER_LIMITATION,
                },
            }
            return {
                "status": "error",
                "returncode": 1,
                "output": ATOMIC_RESULT_PREFIX + json.dumps(payload),
                "error": "",
            }

    result = FailedHttpTransport().execute_atomic(
        (Operation("raise_for_test", ("injected failure",)),)
    )
    assert result.ok is False
    assert result.guest_returncode == 1
    assert result.failure_kind == "injected"
    assert result.raw_result_marker.startswith(ATOMIC_RESULT_PREFIX)


def test_intervening_move_prevents_click_coalescing() -> None:
    semantic = (
        Operation("mouse_down", ("left",)),
        Operation("move_relative", (25, -10)),
        Operation("mouse_up", ("left",)),
    )
    assert lower_guest_operations(semantic) == semantic
    program, expected_mask = compile_atomic_guest_program(
        semantic,
        initial_buttons=set(),
        initial_keys=set(),
    )
    assert expected_mask == 0
    assert "RUNG1A_ATOMIC_STEP_0:mouse_down" in program
    assert "RUNG1A_ATOMIC_STEP_1:move_relative" in program
    assert "RUNG1A_ATOMIC_STEP_2:mouse_up" in program
    assert "RUNG1A_ATOMIC_STEP_0:click" not in program


def test_nonzero_clamped_compact_delta_remains_auditable() -> None:
    transport = RecordingTransport(cursor=(1919, 100), screen=(1920, 1080))
    result = CompactRawExecutor(transport).execute("10 0 0")
    assert result.executor_dispatch_status == "ok"
    assert result.atomic_state is not None
    state = result.atomic_state
    assert state["semantic_operations"] == [
        {"kind": "move_relative", "args": [10, 0]}
    ]
    assert state["lowered_operations"] == state["semantic_operations"]
    assert state["requested_relative_delta"] == [10, 0]
    assert state["executed_cursor_delta"] == [0, 0]
    assert state["backend_primitives"] == [
        {
            "kind": "move_to",
            "call": "recording.move_to",
            "requested_delta": [10, 0],
            "cursor_before": [1919, 100],
            "cursor_after": [1919, 100],
            "actual_delta": [0, 0],
            "clamped": True,
        }
    ]


def test_compact_turn_fails_cursor_readback_verification() -> None:
    class StaleReadbackTransport(RecordingTransport):
        def __init__(self) -> None:
            super().__init__(cursor=(10, 20))
            self.read_count = 0

        def cursor_position(self) -> tuple[int, int]:
            self.read_count += 1
            if self.read_count == 1:
                return self._cursor
            return self._cursor[0] + 1, self._cursor[1]

    transport = StaleReadbackTransport()
    result = CompactRawExecutor(transport).execute("5 0 0")
    assert transport.read_count == 2
    assert result.executor_dispatch_status == "error"
    assert result.atomic_state is not None
    assert result.atomic_state["cursor_readback_verified"] is False
    assert result.atomic_state["failure_kind"] == "verification"


@pytest.mark.parametrize(
    "action",
    [
        {"action": "mouse_move", "coordinate": [20, 30]},
        {"action": "left_click", "coordinate": [20, 30]},
        {"action": "left_click_drag", "coordinate": [20, 30]},
        {"action": "scroll", "clicks": -3},
        {"action": "key", "keys": ["ControlLeft", "KeyA"]},
        {"action": "type", "text": "λ"},
        {"action": "wait", "time": 0},
    ],
)
def test_each_native_logical_action_uses_one_guest_process(action) -> None:
    transport = RecordingTransport(cursor=(10, 20))
    result = NativeAbsoluteExecutor(transport).execute(action)
    assert transport.atomic_invocations == 1
    assert result.atomic_state is not None
    assert result.atomic_state["guest_process_count"] == 1


def test_atomic_compiler_rejects_impossible_held_button_transitions() -> None:
    with pytest.raises(TransportError, match="button not held"):
        compile_atomic_guest_program(
            (Operation("mouse_up", ("left",)),),
            initial_buttons=set(),
            initial_keys=set(),
        )
    with pytest.raises(TransportError, match="button already held"):
        compile_atomic_guest_program(
            (Operation("mouse_down", ("left",)),),
            initial_buttons={"left"},
            initial_keys=set(),
        )


def test_stateful_raw_click_requires_observed_baseline_and_endpoint() -> None:
    observed = (10, 20)
    exact = StatefulClickTransport(cursor=observed)
    trajectory = GoldTrajectory(
        arm="compact_raw_phaseb",
        actions=("290 380 0 ; +LMB -LMB",),
        observed_cursor_baseline=observed,
        expected_endpoint=(300, 400),
    )
    _, journal = _execute("compact_raw_phaseb", exact, trajectory)
    assert journal["baseline_matches"] is True
    assert journal["endpoint_matches"] is True
    assert journal["atomic_guest_process_count"] == 1
    assert journal["final_pointer_button_mask"] == 0
    assert journal["actions"][0]["guest_cursor_after"] == [300, 400]
    assert exact.checked is True

    # The same raw delta is a genuine negative when it was compiled against an
    # assumed/stale baseline rather than the cursor observed at dispatch.
    stale = StatefulClickTransport(cursor=(100, 100))
    CompactRawExecutor(stale).execute("290 380 0 ; +LMB -LMB")
    assert stale.cursor_position() == (390, 480)
    assert stale.checked is False

    # The production selfcheck now catches that mismatch before dispatch.
    guarded = StatefulClickTransport(cursor=(100, 100))
    records, stale_journal = _execute("compact_raw_phaseb", guarded, trajectory)
    assert records == []
    assert stale_journal["dispatch_status"] == "blocked_baseline_drift"
    assert stale_journal["baseline_matches"] is False
    assert stale_journal["endpoint_matches"] is False
    assert guarded.cursor_position() == (100, 100)
    assert guarded.checked is False


def test_pre_oracle_mask_requires_release_except_declared_hold_probe() -> None:
    fixture = next(
        item
        for item in load_manifest().select(split="development")
        if item.template == "click"
    )
    transport = RecordingTransport(cursor=(300, 400))
    trajectory = GoldTrajectory(
        arm="compact_raw_phaseb",
        actions=("0 0 0 ; +LMB",),
        observed_cursor_baseline=(300, 400),
        expected_endpoint=(300, 400),
    )
    _, journal = _execute("compact_raw_phaseb", transport, trajectory)
    assert journal["final_pointer_button_mask"] == 1 << 8
    with pytest.raises(SelfcheckError, match="pointer button mask"):
        _assert_dispatch_journal(fixture, "compact_raw_phaseb", "gold", journal)
    _assert_dispatch_journal(
        fixture,
        "compact_raw_phaseb",
        "held-button injection",
        journal,
        required_pointer_button_mask=1 << 8,
    )


def test_stateful_click_rejects_button_down_before_move() -> None:
    transport = StatefulClickTransport(cursor=(10, 20))
    raw = CompactRawExecutor(transport)
    raw.execute("0 0 0 ; +LMB")
    raw.execute("290 380 0")
    raw.execute("0 0 0 ; -LMB")
    assert transport.cursor_position() == (300, 400)
    assert transport.checked is False


def test_drag_adapters_match_hold_move_release_state() -> None:
    native_transport = RecordingTransport(cursor=(50, 50))
    native = NativeAbsoluteExecutor(native_transport)
    native.execute({"action": "mouse_move", "coordinate": [200, 300]})
    native.execute({"action": "left_click_drag", "coordinate": [700, 300]})
    raw_transport = RecordingTransport(cursor=(50, 50))
    raw = CompactRawExecutor(raw_transport)
    raw.execute("150 250 0 ; +LMB")
    assert raw_transport.audit.held_buttons == {"left"}
    raw.execute("500 0 0")
    raw.execute("0 0 0 ; -LMB")
    assert native_transport.cursor_position() == raw_transport.cursor_position() == (700, 300)
    assert _kinds(native_transport) == _kinds(raw_transport) == [
        "move_to",
        "mouse_down",
        "move_to",
        "mouse_up",
    ]
    assert not native_transport.audit.held_buttons
    assert not raw_transport.audit.held_buttons
    assert all(
        Operation("click", ("left",)) not in lower_guest_operations(operations)
        for operations in raw_transport.atomic_inputs
    )


def test_unicode_coalesced_type_is_one_shared_operation() -> None:
    text = "東京 Grüße λ ✓"
    native_transport = RecordingTransport()
    NativeAbsoluteExecutor(native_transport).execute({"action": "type", "text": text})
    raw_transport = RecordingTransport()
    CompactRawExecutor(raw_transport).execute('0 0 0 ; type("東京 Grüße λ ✓")')
    assert native_transport.audit.typed_texts == raw_transport.audit.typed_texts == [text]
    assert _kinds(native_transport) == _kinds(raw_transport) == ["coalesced_type"]
    compiled = compile_unicode_coalesced_type(text)
    # Newest (lineage B) GTK clipboard owner.
    assert "gi.require_version('Gtk','3.0')" in compiled
    assert "clipboard.set_text(value,-1)" in compiled
    assert "clipboard.wait_for_text()!=value" in compiled
    assert "pyautogui.hotkey('ctrl','a')" in compiled
    assert "pyautogui.hotkey('ctrl','v')" in compiled
    assert f"GLib.timeout_add({CLIPBOARD_PASTE_DELAY_MS},paste)" in compiled
    assert f"GLib.timeout_add({CLIPBOARD_OWNER_LIFETIME_MS},Gtk.main_quit)" in compiled
    # Both superseded backends must stay out of the compiled program.
    assert "pyperclip" not in compiled
    assert "tkinter" not in compiled
    assert "clipboard_append" not in compiled
    assert "pyautogui.write" not in compiled
    atomic_program, _ = compile_atomic_guest_program(
        (Operation("coalesced_type", (text,)),),
        initial_buttons=set(),
        initial_keys=set(),
    )
    assert compiled in atomic_program


def test_coalesced_type_fails_loudly_when_the_clipboard_owner_expires() -> None:
    """A clipboard owner that dies before pasting must not exit successfully.

    The GTK compiler schedules the paste and the owner teardown as two
    independent GLib timeouts.  If the paste callback never runs the guest
    process would otherwise return rc=0 having typed nothing, which is exactly
    the silent-success class this ladder exists to eliminate.
    """
    compiled = compile_unicode_coalesced_type("x")
    assert "if not _r1a_pasted:" in compiled
    assert "clipboard owner expired before the paste callback ran" in compiled
    assert CLIPBOARD_PASTE_DELAY_MS < CLIPBOARD_OWNER_LIFETIME_MS


def test_signed_scroll_state_is_preserved_by_both_adapters() -> None:
    for clicks in (-7, 6):
        native_transport = RecordingTransport()
        NativeAbsoluteExecutor(native_transport).execute(
            {"action": "scroll", "clicks": clicks}
        )
        raw_transport = RecordingTransport()
        CompactRawExecutor(raw_transport).execute(f"0 0 {clicks}")
        assert native_transport.audit.scroll_total == clicks
        assert raw_transport.audit.scroll_total == clicks


def test_reset_model_recreates_cursor_button_type_and_scroll_state() -> None:
    dirty = RecordingTransport(cursor=(50, 50))
    dirty.move_to(900, 700)
    dirty.mouse_down("left")
    dirty.scroll(-8)
    dirty.coalesced_type("leak")
    assert dirty.audit.held_buttons == {"left"}
    clean = RecordingTransport(cursor=(50, 50))
    assert clean.cursor_position() == (50, 50)
    assert clean.audit.held_buttons == set()
    assert clean.audit.scroll_total == 0
    assert clean.audit.typed_texts == []
