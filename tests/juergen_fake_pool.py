"""A fake pool constructor for `pool_target` to be pointed at.

`pool_target` names a constructor (`module:attribute`), not a provider; nothing
here selects a VM backend.
"""

from __future__ import annotations

from typing import Any


class Recorder:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.started = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def checkout(self) -> Any:
        from juergen_doubles import FakeSession

        return FakeSession()

    def close(self) -> None:
        self.closed += 1
