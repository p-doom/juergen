"""What the estate gate's own reading says it measured.

`tooling/estate_gate.sh` is the only gate over the five suites and nothing
covered it. In one day it produced four misleading readings: two REDs that were
mid-edit snapshots of another agent's working tree (a `build/lib` a `pip
install` had just left behind, and a symbol deleted out from under
`conftest.py`), a run starved out and killed at ten minutes, and an all-skipped
suite that printed GREEN having executed nothing. Every one would have been
diagnosable at a glance if the reading had identified the tree, the interpreter
and the CPU it measured.

These drive the real script over a throwaway one-suite estate, so they assert on
the reading a caller gets rather than on the script's source text.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from itertools import takewhile
from pathlib import Path

GATE = Path(__file__).resolve().parents[1] / "tooling" / "estate_gate.sh"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_COMMITTER = (
    "-c",
    "user.email=gate@estate.test",
    "-c",
    "user.name=estate gate",
    "-c",
    "commit.gpgsign=false",
)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ("git", *_COMMITTER, *args), cwd=root, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _estate(tmp_path: Path, test_body: str | None) -> Path:
    """One suite's checkout: a `tests/` directory under git, and nothing else."""
    root = tmp_path / "siblings" / "desktop"
    (root / "tests").mkdir(parents=True)
    (root / "README").write_text("the tree state the gate must report\n")
    if test_body is not None:
        (root / "tests" / "test_fake.py").write_text(test_body)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "committed")
    return root


def _read(root: Path, gate: Path = GATE) -> tuple[int, str]:
    """The `desktop` suite is the one whose marker and target list are minimal."""
    env = dict(os.environ, DESKTOP_ROOT=str(root), DESKTOP_PYTHON=sys.executable)
    env.pop("PYTEST_ADDOPTS", None)
    done = subprocess.run(
        ("bash", str(gate), "--only", "desktop"),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return done.returncode, _ANSI.sub("", done.stdout + done.stderr)


def _verdict(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith("ESTATE GATE:")]
    assert lines, out
    return lines[-1]


def test_a_reading_names_the_sha_the_interpreter_and_the_cleanliness_it_measured(
    tmp_path: Path,
) -> None:
    root = _estate(tmp_path, "def test_one():\n    assert True\n")
    code, out = _read(root)
    assert code == 0, out
    sha = _git(root, "rev-parse", "--short=12", "HEAD")
    assert re.search(
        rf"desktop\s+{sha}\s+py{re.escape(platform.python_version())}\s+clean", out
    ), out
    assert _verdict(out) == "ESTATE GATE: GREEN", out


def test_a_dirty_tree_is_named_in_the_reading_and_weakens_its_green(tmp_path: Path) -> None:
    """The two misattributed REDs were this: a verdict over uncommitted bytes.

    An untracked directory is one porcelain entry, which is how the transient
    `build/lib` shows up.
    """
    root = _estate(tmp_path, "def test_one():\n    assert True\n")
    (root / "README").write_text("edited under the gate's feet\n")
    (root / "build" / "lib").mkdir(parents=True)
    (root / "build" / "lib" / "stale.py").write_text("")
    code, out = _read(root)
    assert code == 0, out
    assert "DIRTY: 1 tracked, 1 untracked" in out, out
    assert _verdict(out) == (
        "ESTATE GATE: GREEN  (DIRTY TREE: desktop -- uncommitted bytes, not a commit)"
    ), out


def test_a_tree_that_moves_mid_run_is_named_as_having_moved(tmp_path: Path) -> None:
    """Suites take twenty minutes, over which other agents land commits, so the
    state a run began on is not the state it ended on."""
    root = _estate(
        tmp_path,
        "import pathlib\n\n\ndef test_lands_a_change():\n"
        "    (pathlib.Path(__file__).resolve().parents[1] / 'landed.py').write_text('')\n",
    )
    code, out = _read(root)
    assert code == 0, out
    assert "MOVED: the tree changed while this ran" in out, out
    assert re.search(r"^\s+< desktop\|\w+\|0\|0\|", out, re.MULTILINE), out
    assert re.search(r"^\s+> desktop\|\w+\|0\|[1-9]", out, re.MULTILINE), out


def test_a_run_whose_own_script_was_replaced_says_so(tmp_path: Path) -> None:
    """An editor replaces this file rather than rewriting it, so a run in flight
    keeps executing the inode it started on -- the stray `tooling/.nfs*` files are
    exactly that -- and its output describes a gate no longer on disk."""
    gate = tmp_path / "estate_gate.sh"
    shutil.copy(GATE, gate)
    root = _estate(
        tmp_path,
        "import os\nimport pathlib\nimport shutil\n\n\ndef test_replaces_the_gate():\n"
        f"    gate = pathlib.Path({str(gate)!r})\n"
        "    swap = gate.with_suffix('.swap')\n"
        "    shutil.copy(gate, swap)\n"
        "    os.replace(swap, gate)\n",
    )
    code, out = _read(root, gate=gate)
    assert code == 0, out
    assert "REPLACED: this script was rewritten mid-run" in out, out


def test_a_root_whose_tree_state_cannot_be_read_measures_nothing(tmp_path: Path) -> None:
    """Exit 2, and no suite runs: an unrecordable tree is an incomplete environment."""
    root = _estate(tmp_path, "def test_one():\n    assert True\n")
    shutil.rmtree(root / ".git")
    code, out = _read(root)
    assert code == 2, out
    assert "not a git checkout" in out, out
    assert "==> desktop" not in out, out


def test_an_all_skipped_suite_is_not_a_pass(tmp_path: Path) -> None:
    """pytest exits 0 when a collected test skips; rc=5 only covers zero collected.

    The skip is inside the test, not at module level: a module-level skip collects
    nothing and so already exits 5, which is the case below.
    """
    root = _estate(
        tmp_path, "import pytest\n\n\ndef test_one():\n    pytest.skip('nothing to run')\n"
    )
    code, out = _read(root)
    assert code == 1, out
    assert re.search(r"FAIL\s+desktop\s+no test executed", out), out


def test_a_suite_that_collected_nothing_is_not_a_pass(tmp_path: Path) -> None:
    root = _estate(tmp_path, None)
    code, out = _read(root)
    assert code == 1, out
    assert re.search(r"FAIL\s+desktop\s+rc=5", out), out


def test_a_collection_error_is_not_a_pass(tmp_path: Path) -> None:
    """Zero executed tests wearing different clothes: rc=2 with nothing collected."""
    root = _estate(tmp_path, "import a_module_that_is_not_installed\n")
    code, out = _read(root)
    assert code == 1, out
    assert "PASS" not in out, out
    assert re.search(r"FAIL\s+desktop\s+1 error", out), out


def test_a_printed_pass_count_cannot_inflate_the_reading(tmp_path: Path) -> None:
    """The count is pytest's own summary line, not the first match in the log."""
    root = _estate(tmp_path, "def test_prints():\n    print('12 passed')\n    assert False\n")
    code, out = _read(root)
    assert code == 1, out
    summary = out.split("estate gate summary")[1]
    assert re.search(r"FAIL\s+desktop\s+1 failed\s", summary), summary
    assert "12 passed" not in summary, summary


def test_the_reading_says_how_much_cpu_the_run_actually_got(tmp_path: Path) -> None:
    """A starved run and a bigger suite both take longer; only one loses CPU."""
    root = _estate(tmp_path, "import time\n\n\ndef test_waits():\n    time.sleep(3)\n")
    code, out = _read(root)
    assert code == 0, out
    row = re.search(r"PASS\s+desktop\s+1 passed\s+(\d+)s wall\s+(\d+)s cpu\s+([\d.]+) core", out)
    assert row, out
    assert int(row.group(1)) >= 3, out
    assert float(row.group(3)) < 0.5, out
    assert "CONTENDED" in out, out


def test_help_prints_the_whole_header_and_stops_at_the_code() -> None:
    """A line-numbered range silently truncates this header as it grows.

    Counted rather than quoted: an assertion on the last line's wording has to be
    edited whenever the prose is, which is how a truncation guard stops guarding.
    """
    lines = GATE.read_text(encoding="utf-8").splitlines()
    header = list(takewhile(lambda line: line.startswith("#"), lines[1:]))
    done = subprocess.run(
        ("bash", str(GATE), "--help"), capture_output=True, text=True, check=True
    )
    assert len(done.stdout.splitlines()) == len(header), "the header is not printed whole"
    assert "DIRTY TREE" in done.stdout, "the dirty-tree contract belongs in --help"
    assert "set -uo pipefail" not in done.stdout, "the header stops at the code"
