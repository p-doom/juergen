"""Shared doubles for the agent/ suite.

A `FakeSession` records the argv it was handed and replays canned stdout, because
every guest interaction in this module is "run one command, read one marker line".

Nothing here needs a VM, a GPU or a network. The one test file that needs a real
qemu is marked `kvm` and skips itself.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import verifiers.v1 as vf

_CONVERT = "juergen_datasets_convert"


def load_convert():
    """The real BC reader, `datasets/convert.py`, loaded by path and then cached.

    Not `from datasets import convert`: HuggingFace's `datasets` distribution owns
    that import name in every venv here, and the repo's `datasets/` is a script
    directory with no `__init__.py`, so the name resolves to theirs. Same hazard
    `grammars.load` explains for `desktop`. Registered in `sys.modules` before
    execution because `@dataclass` resolves its own module by name — which is also
    why it is cached: two live copies of one module means two `Step` classes that
    are not each other.
    """
    if _CONVERT not in sys.modules:
        path = Path(__file__).resolve().parent.parent / "datasets" / "convert.py"
        spec = importlib.util.spec_from_file_location(_CONVERT, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_CONVERT] = module
        spec.loader.exec_module(module)
    return sys.modules[_CONVERT]


def png(width: int = 8, height: int = 6, colour: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class FakeGuestReceipt:
    """What `execute_atomic` returns: desktop's `AtomicExecutionResult` surface.

    Only the four fields the harness publishes. Returning a bare
    `{"dispatched": n}` here would let a receipt-shaped contract pass on a shape
    the real transport never produces, which is how the guest's verdict came to be
    dropped in the first place.
    """

    ok: bool
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    failure_kind: str | None = None


class FakeSession:
    """The session surface the harness and the preparers actually touch.

    `argv_responses` maps a substring of the joined argv to the stdout to return,
    so a test pins the guest protocol (one `SOLV2_STATE=` line, one
    `FIXTURE_JSON=` line).
    """

    def __init__(
        self,
        *,
        screen: tuple[int, int] = (1920, 1080),
        cursor: tuple[int, int] = (100, 100),
        argv_responses: dict[str, str] | None = None,
        frames: list[bytes] | None = None,
    ) -> None:
        self.screen = screen
        self.cursor = cursor
        self.argv_responses = dict(argv_responses or {})
        self.frames = list(frames or [])
        self.session_id = "fake-session"
        self.argv_log: list[list[str]] = []
        self.pyautogui_log: list[str] = []
        self.operations_log: list[Any] = []
        self.released: list[tuple[bool, str | None]] = []
        self.screenshots = 0
        self.evaluate_value: float | None = None
        self.task_config: dict[str, Any] | None = None
        self.declared_terminal: list[str | None] = []

    def screen_size(self) -> tuple[int, int]:
        return self.screen

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def screenshot(self) -> bytes:
        self.screenshots += 1
        if self.frames:
            return self.frames[min(self.screenshots - 1, len(self.frames) - 1)]
        return png(colour=(self.screenshots % 250, 0, 0))

    def execute_atomic(self, operations: Any) -> FakeGuestReceipt:
        ops = list(operations)
        self.operations_log.append(ops)
        return FakeGuestReceipt(
            ok=True, cursor_before=self.cursor, cursor_after=self.cursor
        )

    def execute_pyautogui(self, code: str) -> None:
        self.pyautogui_log.append(code)
        import re

        match = re.search(r"moveTo\(\s*(-?\d+)\s*,\s*(-?\d+)", code)
        if match:
            self.cursor = (int(match.group(1)), int(match.group(2)))

    def execute_argv(self, argv: list[str]) -> dict[str, Any]:
        self.argv_log.append(list(argv))
        joined = " ".join(argv)
        for needle, output in self.argv_responses.items():
            if needle in joined:
                return {"output": output}
        return {"output": ""}

    def setup(self, task_config: dict[str, Any]) -> int:
        """The whole OSWorld task JSON, matching `DesktopFacade.setup`.

        Not just the `config` list: the `evaluator` block has to reach the
        session, or a no-argument `evaluate()` has nothing to score.
        """
        steps = list(task_config.get("config") or [])
        self.task_config = dict(task_config)
        self.argv_log.append(["<osworld-setup>", json.dumps(steps)])
        return len(steps)

    def declare_terminal(self, control: str | None) -> None:
        self.declared_terminal.append(control)

    def evaluate(self) -> float:
        if self.evaluate_value is None:
            raise RuntimeError("no evaluate configured")
        return self.evaluate_value

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        self.released.append((failed, error))


class FakePool:
    """A `DesktopSessionPool`-shaped pool over `FakeSession`s."""

    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory or FakeSession
        self.started = 0
        self.closed = 0
        self.checked_out: list[Any] = []

    def start(self) -> None:
        self.started += 1

    def checkout(self) -> Any:
        session = self._factory()
        self.checked_out.append(session)
        return session

    def close(self) -> None:
        self.closed += 1


class FakeClient:
    """A `Client` stand-in for `ContextTransport`. Records every wire body.

    A reply is either the content or a `(content, finish_reason)` pair; a bare
    string finishes on `"stop"`, which is what a whole turn reports.
    """

    def __init__(self, replies: list[str | tuple[str, str]] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self, dialect: Any, body: dict[str, Any], model: str, sampling: Any, **kwargs: Any
    ) -> Any:
        self.calls.append(
            {"body": body, "model": model, "sampling": sampling, "kwargs": kwargs}
        )
        reply = self.replies.pop(0) if self.replies else ""
        text, finish_reason = reply if isinstance(reply, tuple) else (reply, "stop")
        return type(
            "Response",
            (),
            {"message": type("M", (), {"content": text})(), "finish_reason": finish_reason},
        )()


def make_ctx(
    *,
    model: str = "test-model",
    replies: list[str | tuple[str, str]] | None = None,
    **sampling: Any,
) -> vf.ModelContext:
    return vf.ModelContext(
        model=model,
        client=FakeClient(replies),  # type: ignore[arg-type]
        sampling=vf.Sampling(**sampling),
    )
