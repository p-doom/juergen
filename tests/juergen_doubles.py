"""Shared doubles for the agent/ evals/ rl/ suite.

A `FakeSession` records typed desktop operation batches and supplies deterministic
screenshots.

Nothing here needs a VM, a GPU or a network. The one test file that needs a real
qemu is marked `kvm` and skips itself.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

from evals.tasks import DesktopState, DesktopTaskData

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


def jpeg(width: int = 8, height: int = 6, colour: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(
        buffer, format="JPEG", quality=85, subsampling=2, optimize=False
    )
    return buffer.getvalue()


@dataclass(frozen=True)
class FakeGuestReceipt:
    """What `execute` returns: desktop's `ExecutionReceipt` surface.

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
    """The session surface the harness and the preparers actually touch."""

    def __init__(
        self,
        *,
        screen: tuple[int, int] = (1920, 1080),
        cursor: tuple[int, int] = (100, 100),
        frames: list[bytes] | None = None,
    ) -> None:
        self.screen = screen
        self.cursor = cursor
        self.frames = list(frames or [])
        self.session_id = "fake-session"
        self.operations_log: list[Any] = []
        self.released: list[tuple[bool, str | None]] = []
        self.screenshots = 0

    def screen_size(self) -> tuple[int, int]:
        return self.screen

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def screenshot(self) -> bytes:
        self.screenshots += 1
        if self.frames:
            return self.frames[min(self.screenshots - 1, len(self.frames) - 1)]
        return jpeg(colour=(self.screenshots % 250, 0, 0))

    def execute(self, operations: Any) -> FakeGuestReceipt:
        ops = list(operations)
        self.operations_log.append(ops)
        return FakeGuestReceipt(
            ok=True, cursor_before=self.cursor, cursor_after=self.cursor
        )

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        self.released.append((failed, error))


class FakeCheckout:
    """Desktop's checkout handle around one fake session."""

    def __init__(self, session: FakeSession) -> None:
        self.session = session

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def tracked_env(self) -> FakeSession:
        return self.session

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        self.session.release(failed=failed, error=error)


class FakePool:
    """A `DesktopSessionPool`-shaped pool over `FakeSession`s."""

    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory or FakeSession
        self.started = 0
        self.closed = 0
        self.checked_out: list[FakeCheckout] = []

    def start(self) -> None:
        self.started += 1

    def checkout(self) -> FakeCheckout:
        session = self._factory()
        checkout = FakeCheckout(session)
        self.checked_out.append(checkout)
        return checkout

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


def make_task_data(**kwargs: Any) -> DesktopTaskData:
    base: dict[str, Any] = {
        "idx": 0,
        "name": "cell",
        "prompt": "do the thing",
        "instruction": "do the thing",
        "kind": "none",
        "max_steps": 3,
    }
    base.update(kwargs)
    return DesktopTaskData(**base)


def make_trace(
    data: DesktopTaskData | None = None,
    *,
    task_type: str = "DesktopTask",
    episode: dict[str, Any] | None = None,
) -> vf.Trace:
    trace = vf.Trace(
        task=vf.TraceTask(type=task_type, data=data or make_task_data()),
        state=DesktopState(),
    )
    if episode is not None:
        from evals.tasks import RESULT_KEY

        trace.info[RESULT_KEY] = episode
    return trace
