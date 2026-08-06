"""Shared doubles for the agent/ evals/ rl/ suite.

The doubles are deliberately thin and *behavioural*: a `FakeSession` records the
argv it was handed and replays canned stdout, because every guest interaction in
this module is "run one command, read one marker line". A mock that accepted
anything would let a broken marker protocol pass.

Nothing here needs a VM, a GPU or a network. The one test file that needs a real
qemu is marked `kvm` and skips itself.
"""

from __future__ import annotations

import io
import json
from typing import Any, Callable

import verifiers.v1 as vf

from evals.tasks import DesktopState, DesktopTaskData


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #


def png(width: int = 8, height: int = 6, colour: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #


class FakeSession:
    """The session surface the harness and the preparers actually touch.

    `argv_responses` maps a substring of the joined argv to the stdout to return, so
    a test pins the guest protocol (one `SOLV2_STATE=` line, one `FIXTURE_JSON=`
    line) rather than the fact that *some* command ran.
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

    # -- surface -------------------------------------------------------- #

    def screen_size(self) -> tuple[int, int]:
        return self.screen

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def screenshot(self) -> bytes:
        self.screenshots += 1
        if self.frames:
            return self.frames[min(self.screenshots - 1, len(self.frames) - 1)]
        return png(colour=(self.screenshots % 250, 0, 0))

    def execute_atomic(self, operations: Any) -> dict[str, Any]:
        ops = list(operations)
        self.operations_log.append(ops)
        return {"dispatched": len(ops)}

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
        """The **whole** OSWorld task JSON, matching `DesktopFacade.setup`.

        It used to take just the `config` list, which is what the preparer used to
        pass — and that was exactly the reason nothing could implement
        `evaluate()`: the `evaluator` block never reached the session, so a
        no-argument scorer had nothing to score.
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


# --------------------------------------------------------------------------- #
# model context
# --------------------------------------------------------------------------- #


class FakeClient:
    """A `Client` stand-in for `ContextTransport`. Records every wire body."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self, dialect: Any, body: dict[str, Any], model: str, sampling: Any, **kwargs: Any
    ) -> Any:
        self.calls.append(
            {"body": body, "model": model, "sampling": sampling, "kwargs": kwargs}
        )
        text = self.replies.pop(0) if self.replies else ""
        return type("Response", (), {"message": type("M", (), {"content": text})()})()


def make_ctx(
    *,
    model: str = "test-model",
    replies: list[str] | None = None,
    **sampling: Any,
) -> vf.ModelContext:
    return vf.ModelContext(
        model=model,
        client=FakeClient(replies),  # type: ignore[arg-type]
        sampling=vf.Sampling(**sampling),
    )


# --------------------------------------------------------------------------- #
# traces
# --------------------------------------------------------------------------- #


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
