"""A fake session pool the harness can be pointed at via `pool_target`.

Importable by `module:attribute`, so `DesktopPoolConfig.pool_target` can name it —
which is exactly what that field exists for: injecting a fake, never selecting a VM
backend.
"""

from __future__ import annotations

from typing import Any


class Pool:
    session: Any = None
    """Set by the test before `launch`; every checkout returns it."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.started = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def checkout(self) -> Any:
        if Pool.session is not None:
            return Pool.session
        from juergen_doubles import FakeSession

        return FakeSession()

    def close(self) -> None:
        self.closed += 1
