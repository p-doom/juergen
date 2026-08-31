"""A fake session pool the harness can be pointed at via `pool_target`.

Importable by `module:attribute`, so `DesktopPoolConfig.pool_target` can name it;
that field injects a fake, it does not select a VM backend.
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
        from juergen_doubles import FakeCheckout, FakeSession

        if Pool.session is not None:
            return FakeCheckout(Pool.session)

        return FakeCheckout(FakeSession())

    def close(self) -> None:
        self.closed += 1
