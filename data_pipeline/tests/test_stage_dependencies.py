"""Every numbered stage must import under the interpreter this suite runs in.

The stages are dispatched as file paths (``python pipeline/stage_NN_*.py --flags``)
from whatever cwd the scheduler picks, and their dependency set is declared here,
in ``data_pipeline/pyproject.toml`` -- not in the root project, which packages no
``pipeline/`` and carries neither cv2 nor array-record. Nothing tied the two
together, so ``absl`` reached stage 05/06 declared in one venv and dispatched
against another: job 141103 died with ``ModuleNotFoundError: No module named
'absl'`` after being scheduled, and stage 06 followed it into
DependencyNeverSatisfied.

Reading the TOML cannot catch that. Each stage is executed in a subprocess under
``-I`` from a temporary directory, so nothing resolves out of the checkout except
the repo root the stage puts on ``sys.path`` itself -- which is the only
arrangement that measures the declared dependency set rather than the cwd.
``run_name`` is not ``__main__``, so the flag definitions run and ``app.run`` does
not.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = sorted((REPO_ROOT / "pipeline").glob("stage_0*.py"))

_PROBE = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='_dep_probe')"


def test_the_probe_found_stages():
    # An empty parametrize list reports as one skip, which reads as green.
    assert STAGES, f"no pipeline/stage_0*.py under {REPO_ROOT}"


@pytest.mark.parametrize("stage", STAGES, ids=lambda p: p.stem)
def test_stage_imports_under_this_interpreter(stage: Path, tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE, str(stage)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{stage.name} does not import under {sys.executable}:\n{proc.stderr}"
    )
