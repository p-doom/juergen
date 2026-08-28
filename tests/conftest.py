"""Import bootstrap and pytest fixtures.

The doubles live in `juergen_doubles.py` rather than here because
`desktop/tests/` is also importable as `tests` on this cluster's layout, and
`from tests.conftest import ...` resolves to the wrong package. A unique module
name makes the import unambiguous regardless of sys.path order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_DESKTOP = _REPO.parent / "desktop"

for candidate in (_HERE, _REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
if str(_DESKTOP) not in sys.path:
    sys.path.append(str(_DESKTOP))

from juergen_doubles import FakeSession, jpeg  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: drives a whole `python -m evals.signoflife` run in-process "
        "(seconds, not milliseconds). Still no VM, no GPU and no network.",
    )


@pytest.fixture
def frame() -> bytes:
    return jpeg()


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture(autouse=True)
def _isolated_slot_dir(tmp_path_factory, monkeypatch):
    """Never let a test lease a slot in the shared node directory."""
    directory = tmp_path_factory.mktemp("vm-slots")
    monkeypatch.setenv("JUERGEN_VM_SLOT_DIR", str(directory))
    yield directory
