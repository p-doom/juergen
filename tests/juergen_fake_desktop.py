"""A desktop pool that boots nothing, so the whole dispatcher can be run.

The only thing standing between the dispatcher and a booted VM is
`desktop.vm.factory.build_desktop_pool`, so this substitutes that and only that.
`evals.vm.kvm_desktop_pool` runs for real — argument assembly, `DesktopPoolConfig`,
`_AdaptedPool`, reset-on-reuse, `DesktopFacade` — over a pool whose checkouts are
these fakes. This is the path every real gate run takes, and where a missing
`evaluate()` once hid.

The guest is a VM where nothing happened, deliberately. It answers the four cells'
probes with well-formed evidence in which every postcondition is false, which is
the negative control's calibrated reading (0/4), so the dispatcher can be run end
to end without touching what a cell scores. Making this guest report success would
fabricate the oracle arm's 4/4, which is a calibration, not a fixture.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

__all__ = ["FakeCheckout", "FakeDesktopPool", "FakeGuestSession", "build_desktop_pool"]

_TASK_ID = re.compile(r"'task_id':\s*(?P<value>'[^']*'|\"[^\"]*\")")

PANEL_STATE = {
    "schema_version": 1,
    "title": "SOLV2 panel",
    "clicked": [],
    "entry_text": "",
    "submitted": False,
    "screen": [1920, 1080],
    "widgets": {
        "entry": [815, 494, 1045, 517],
        "button:Commit B1": [884, 580, 1056, 611],
        "button:Commit B2": [884, 617, 1056, 648],
        "button:Commit B3": [884, 654, 1056, 685],
        "button:Commit B4": [884, 691, 1056, 722],
        "button:Save draft": [844, 530, 1016, 561],
        "button:Submit": [844, 567, 1016, 598],
    },
}
"""The Tk fixture's published measurement, verbatim from a real run (job 141317).

The panel cells resolve every scripted click through this, so a double that
omitted it would make the two promoted scored cells unrunnable without a VM. The
bboxes are the guest's own `winfo_root*` numbers rather than invented ones, so the
lattice premise this suite asserts at setup is the one a VM actually produces."""

WINDOW_GEOMETRY = {
    "window_id": "0x2000001",
    "x": 80,
    "y": 120,
    "width": 1120,
    "height": 720,
    "window_line": "0x2000001  0 80 120 1120 720 xterm.XTerm host SOLV2",
}


def _state_line(task_id: str) -> str:
    """`probe_state`'s contract: exactly one `SOLV2_STATE=` line, schema 1.

    Every postcondition clause is absent rather than false-y by accident —
    an unreadable probe is `status="error"`, which the harness publishes as
    infrastructure-invalid, and that is a different outcome from a clean miss.
    """
    return "SOLV2_STATE=" + json.dumps(
        {
            "schema_version": 1,
            "task_id": task_id,
            "active_window": 'WM_CLASS(STRING) = "xterm", "XTerm"',
            "windows": WINDOW_GEOMETRY["window_line"],
            "chrome_process": False,
            "history": None,
            "transcript": None,
            "prompt_count": 0,
            "capture_file_exists": False,
            "captured_text": None,
            "proof_file_exists": False,
            "proof_file_content": None,
            "keystroke_state": None,
            "stage_one_text": None,
            "commit_text": None,
            "panel_state": PANEL_STATE,
        },
        sort_keys=True,
    )


class FakeGuestTransport:
    """`HttpGuiTransport`'s half: input, geometry and guest commands."""

    base_url = "http://127.0.0.1:5999"

    def __init__(self, screen: tuple[int, int] = (1920, 1080)) -> None:
        self.screen = screen
        self.cursor = (960, 540)
        self.argv: list[list[str]] = []
        self.operations: list[Any] = []
        self.pyautogui: list[str] = []

    def screen_size(self) -> tuple[int, int]:
        return self.screen

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def execute_atomic(self, operations: Any) -> dict[str, Any]:
        ops = tuple(operations)
        self.operations.append(ops)
        return {"dispatched": len(ops)}

    def execute_pyautogui(self, code: str) -> None:
        self.pyautogui.append(code)

    def execute_argv(self, argv: list[str], *, check: bool = True) -> dict[str, Any]:
        self.argv.append(list(argv))
        script = " ".join(argv)
        if "SOLV2_GEOMETRY" in script:
            return {"output": "SOLV2_GEOMETRY=" + json.dumps(WINDOW_GEOMETRY, sort_keys=True)}
        if "panel.py" in script or script.startswith("cat ") and "panel.json" in script:
            # The panel setup script cats its state file, and a click re-reads it.
            return {"output": json.dumps(PANEL_STATE, sort_keys=True)}
        if "SOLV2_STATE=" in script:
            match = _TASK_ID.search(script)
            task_id = ast.literal_eval(match.group("value")) if match else ""
            return {"output": _state_line(task_id)}
        # Everything else is a setup script. Empty stdout is a real answer: the
        # compound cell asserts the active window is not a terminal after setup,
        # and "" satisfies that the way a focused xmessage note does.
        return {"output": ""}


class FakeGuestClient:
    """`OSWorldClient`'s half: pixels."""

    def __init__(self) -> None:
        self.screenshots = 0

    def _png(self) -> bytes:
        from PIL import Image
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), (self.screenshots % 250, 10, 20)).save(buffer, format="PNG")
        return buffer.getvalue()

    def screenshot(self) -> bytes:
        self.screenshots += 1
        return self._png()

    def screenshot_settled(self, **kwargs: float) -> bytes:
        return self.screenshot()


class FakeGuestSession:
    """`DesktopSession`'s shape: the two halves side by side, unmerged.

    Merging them is `DesktopFacade`'s job, so a pre-merged double would let a
    wrong adapter pass.
    """

    def __init__(self) -> None:
        self.transport = FakeGuestTransport()
        self.client = FakeGuestClient()
        self.resets = 0

    def reset(self) -> Any:
        self.resets += 1
        self.transport = FakeGuestTransport()
        return self.transport


class FakeCheckout:
    """`CheckedOutDesktopSession`: a lease, and nothing the harness can drive."""

    def __init__(self, session: FakeGuestSession, session_id: str) -> None:
        self.env = session
        self.session_id = session_id
        self.touches = 0
        self.released: list[tuple[bool, str | None]] = []

    def touch(self) -> None:
        self.touches += 1

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        self.released.append((failed, error))


class FakeDesktopPool:
    """`DesktopSessionPool`'s surface: start / checkout / close."""

    instances: list["FakeDesktopPool"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.started = 0
        self.closed = 0
        self.checkouts: list[FakeCheckout] = []
        FakeDesktopPool.instances.append(self)

    def start(self) -> None:
        self.started += 1

    def checkout(self) -> FakeCheckout:
        checkout = FakeCheckout(FakeGuestSession(), f"fake-{len(self.checkouts):02d}")
        self.checkouts.append(checkout)
        return checkout

    def close(self) -> None:
        self.closed += 1


def build_desktop_pool(**kwargs: Any) -> FakeDesktopPool:
    """The one substituted function. Signature-compatible with the real one."""
    return FakeDesktopPool(**kwargs)
