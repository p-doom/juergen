from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


PATH = Path(__file__).parents[1] / "cleanup_orbax.py"
SPEC = importlib.util.spec_from_file_location("phaseb_cleanup_orbax", PATH)
assert SPEC and SPEC.loader
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def test_completed_rejects_running() -> None:
    result = type("Result", (), {"returncode": 0, "stdout": "135403|RUNNING|\n",
                                  "stderr": ""})()
    with patch.object(cleanup.subprocess, "run", return_value=result):
        with pytest.raises(cleanup.CleanupError, match="in-flight/failed"):
            cleanup.completed("135403")


def test_completed_accepts_completed() -> None:
    result = type("Result", (), {"returncode": 0, "stdout": "135403|COMPLETED|\n",
                                  "stderr": ""})()
    with patch.object(cleanup.subprocess, "run", return_value=result):
        cleanup.completed("135403")


def test_completed_rejects_non_numeric_job_id() -> None:
    with pytest.raises(cleanup.CleanupError, match="invalid Slurm job id"):
        cleanup.completed("")
