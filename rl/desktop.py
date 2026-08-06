"""An in-process desktop for the container-free envs.

`movebox` and `grounding` need no VM: their whole observation is a static
background with a target box drawn on it and a cursor marker composited at the
current position. Giving that canvas the *same session surface* a real desktop has
means both envs run under `evals.harness.DesktopHarness` instead of carrying their
own rollout loops — which is where their divergences came from (movebox reset the
prompt every step, grounding sent exactly one turn, and the two disagreed about
whether a coordinate-less click was a no-op or a parse failure).

The operation vocabulary is `pixeldesk.ir.Operation`, in **absolute screen
pixels**. This class never divides by 1000: the normalized 0-999 convention is the
codec's business, and every copy of `round(delta/1000 * screen_dim)` that used to
live in an env is gone.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from rl.geometry import png_bytes, render_cursor

_LOGGER = logging.getLogger(__name__)

__all__ = ["VirtualDesktop", "VirtualDesktopPool", "canvas_pool"]


def _op(operation: Any) -> tuple[str, tuple[Any, ...]]:
    kind = getattr(operation, "kind", None)
    args = getattr(operation, "args", None)
    if kind is None and isinstance(operation, dict):
        kind, args = operation.get("kind"), operation.get("args")
    return str(kind), tuple(args or ())


@dataclass
class VirtualDesktop:
    """A canvas with a cursor. Applies pixel operations, renders on demand.

    `canvas` is the pre-composited background+box image, loaded once per episode
    because compositing the marker is cheap and reloading the background is not.
    """

    canvas: Any = None
    cursor: tuple[int, int] = (0, 0)
    screen: tuple[int, int] = (1920, 1080)
    buttons: set[str] = field(default_factory=set)
    keys: list[str] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    scrolled: int = 0
    dispatched: int = 0
    session_id: str = "virtual"

    # -- per-episode configuration ---------------------------------------- #

    def configure(
        self, *, canvas: Any, cursor: tuple[int, int], screen: tuple[int, int]
    ) -> None:
        """Install this episode's scene. Called by the env's `Preparer.prepare`.

        The pool hands out desktops with no scene because a pool cannot know which
        task a rollout drew; the preparer is the one place that does.
        """
        self.canvas = canvas
        self.screen = screen
        self.buttons.clear()
        self.keys.clear()
        self.typed.clear()
        self.scrolled = 0
        self.dispatched = 0
        self.cursor = (0, 0)
        self._move_to(*cursor)

    # -- session surface -------------------------------------------------- #

    def screen_size(self) -> tuple[int, int]:
        return self.screen

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def screenshot(self) -> bytes:
        return png_bytes(render_cursor(self.canvas, self.cursor))

    def execute_atomic(self, operations: Sequence[Any]) -> dict[str, Any]:
        before = self.cursor
        applied: list[str] = []
        for operation in operations:
            kind, args = _op(operation)
            applied.append(kind)
            self._apply(kind, args)
        self.dispatched += len(applied)
        return {
            "cursor_before": list(before),
            "cursor_after": list(self.cursor),
            "operations": applied,
        }

    def execute_pyautogui(self, code: str) -> None:
        """Only `moveTo` is meaningful on a canvas; anything else is a no-op.

        Kept so the shared preparers (which place a cursor with a pyautogui
        expression) work unchanged against a virtual desktop.
        """
        import re

        match = re.search(r"moveTo\(\s*(-?\d+)\s*,\s*(-?\d+)", code)
        if match:
            self._move_to(int(match.group(1)), int(match.group(2)))

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        del failed, error

    # -- operation application -------------------------------------------- #

    def _apply(self, kind: str, args: tuple[Any, ...]) -> None:
        """Apply one IR operation.

        The vocabulary is closed and absolute: `move_to(x, y)`,
        `glide_to(x, y, seconds)`, `mouse_down/up(button)`, `scroll(dx, dy)`,
        `key_down/up(name)`, `coalesced_type(text)`, `wait(seconds)`. There is
        deliberately **no relative move** — every codec resolves its own convention
        against the cursor and emits clamped absolute pixels, so a desktop that
        accepted a relative op would be re-opening the door this refactor closed.
        An unknown kind is logged and skipped rather than raising: a grammar may
        legitimately emit an operation a *canvas* cannot honour (a window manager
        call, say), and that is not a malformed action.
        """
        if kind in {"move_to", "glide_to"} and len(args) >= 2:
            self._move_to(int(args[0]), int(args[1]))
        elif kind == "mouse_down":
            self.buttons.add(str(args[0]) if args else "left")
        elif kind == "mouse_up":
            self.buttons.discard(str(args[0]) if args else "left")
        elif kind == "key_down" and args:
            self.keys.append(str(args[0]))
        elif kind == "key_up":
            pass
        elif kind == "coalesced_type" and args:
            self.typed.append(str(args[0]))
        elif kind == "scroll" and len(args) >= 2:
            self.scrolled += int(args[1])
        elif kind == "hscroll" and args:
            # `glide_to` and `hscroll` are OPTIONAL backend capabilities, probed
            # rather than required. A canvas can honour both, so it does; a
            # transport that cannot must not be assumed to.
            self.scrolled += int(args[0])
        elif kind == "wait":
            pass
        else:
            _LOGGER.debug("virtual desktop ignoring operation %r", kind)

    def _move_to(self, x: int, y: int) -> None:
        self.cursor = (
            max(0, min(self.screen[0] - 1, x)),
            max(0, min(self.screen[1] - 1, y)),
        )


class VirtualDesktopPool:
    """A `DesktopSessionPool`-shaped pool over `VirtualDesktop`s.

    It has the checkout/close surface `agent.desktop.LeasedDesktopPool` expects, so
    the container-free envs go through the same lease, node-slot and idle-reaper
    machinery. That is not ceremony: the slot cap keeps a scaled-up worker from
    rendering 56 canvases concurrently and thrashing memory, for the same reason it
    keeps it from booting 56 VMs.
    """

    def __init__(self, factory: Callable[[], VirtualDesktop]) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._live = 0

    def start(self) -> None:
        return None

    def checkout(self) -> VirtualDesktop:
        with self._lock:
            self._live += 1
        return self._factory()

    def close(self) -> None:
        with self._lock:
            self._live = 0


def canvas_pool(factory: Callable[[], VirtualDesktop]) -> Callable[[], VirtualDesktopPool]:
    """`pool_factory()` return value for a container-free env."""

    def build() -> VirtualDesktopPool:
        return VirtualDesktopPool(factory)

    return build
