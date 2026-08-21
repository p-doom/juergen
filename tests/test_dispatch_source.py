"""A dispatched job reads a pinned checkout, and a live edit cannot reach it.

Three jobs died in one night because every solv2r recipe resolves
`provenance.repo_path`, the live checkout: two read a half-edited `suite.py`, and
one returned a clean 6/6 on trial 1 then failed trial 2 because `suite.json` was
reverted underneath it. Those three ran this file's central experiment by
accident. It runs it on purpose.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

RESOLVER = Path(__file__).resolve().parents[1] / "tooling" / "dispatch_source.sh"
_COMMITTER = (
    "-c",
    "user.email=pin@dispatch.test",
    "-c",
    "user.name=dispatch pin",
    "-c",
    "commit.gpgsign=false",
)
MEASURED = "suite.json"


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ("git", *_COMMITTER, *args), cwd=cwd, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def live(tmp_path: Path) -> Path:
    """A checkout that stands in for the one agents edit while jobs run."""
    root = tmp_path / "live"
    root.mkdir()
    (root / MEASURED).write_text("committed\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "the state a job was dispatched against")
    return root


def _resolve(live: Path, sha: str, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("bash", str(RESOLVER), str(live), sha, str(root)),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_it_prints_a_checkout_pinned_at_the_recorded_sha(live: Path, tmp_path: Path) -> None:
    sha = _git(live, "rev-parse", "HEAD")
    done = _resolve(live, sha, tmp_path / "pins")
    assert done.returncode == 0, done.stderr
    pin = Path(done.stdout.strip())
    assert pin.is_dir()
    assert _git(pin, "rev-parse", "HEAD") == sha
    assert (pin / MEASURED).read_text() == "committed\n"


def test_an_edit_to_the_live_tree_cannot_reach_the_pinned_one(live: Path, tmp_path: Path) -> None:
    """The accidental experiment, run on purpose.

    A job holding the pinned path reads the commit it was dispatched against, for
    as long as it runs, no matter what happens to the checkout it came from.
    """
    sha = _git(live, "rev-parse", "HEAD")
    pin = Path(_resolve(live, sha, tmp_path / "pins").stdout.strip())

    # What the three failures were: the tree changing under a running job.
    (live / MEASURED).write_text("half-edited, mid-revert\n")

    assert (live / MEASURED).read_text() == "half-edited, mid-revert\n", "the live tree did change"
    assert (pin / MEASURED).read_text() == "committed\n", "the pinned tree must not have"


def test_two_jobs_at_one_sha_share_one_tree(live: Path, tmp_path: Path) -> None:
    sha = _git(live, "rev-parse", "HEAD")
    root = tmp_path / "pins"
    first = _resolve(live, sha, root)
    second = _resolve(live, sha, root)
    assert first.returncode == 0 and second.returncode == 0, (first.stderr, second.stderr)
    assert first.stdout == second.stdout
    trees = [p for p in root.iterdir() if p.is_dir()]
    assert len(trees) == 1, trees


def test_it_refuses_an_edited_pin_rather_than_running_on_it(live: Path, tmp_path: Path) -> None:
    """The pin's whole value is that nobody writes to it, so that is checked per job."""
    sha = _git(live, "rev-parse", "HEAD")
    root = tmp_path / "pins"
    pin = Path(_resolve(live, sha, root).stdout.strip())
    (pin / MEASURED).write_text("someone edited the pin\n")
    done = _resolve(live, sha, root)
    assert done.returncode == 2, done.stdout
    assert "has been edited" in done.stderr, done.stderr


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unknown_sha", "not a commit"),
        ("short_sha", "not a full 40-character sha"),
        ("not_a_checkout", "not a git checkout"),
        ("wrong_arity", "usage:"),
    ],
)
def test_it_refuses_every_way_it_can_fail(
    live: Path, tmp_path: Path, case: str, expected: str
) -> None:
    sha = _git(live, "rev-parse", "HEAD")
    root = tmp_path / "pins"
    if case == "unknown_sha":
        done = _resolve(live, "0" * 40, root)
    elif case == "short_sha":
        done = _resolve(live, sha[:12], root)
    elif case == "not_a_checkout":
        done = _resolve(tmp_path, sha, root)
    else:
        done = subprocess.run(
            ("bash", str(RESOLVER), str(live)), capture_output=True, text=True, timeout=120
        )
    assert done.returncode == 2, done.stdout
    assert expected in done.stderr, done.stderr


def test_no_refusal_ever_prints_the_live_path(live: Path, tmp_path: Path) -> None:
    """`SOURCE="$(dispatch_source.sh ...)"` must not silently become the live tree.

    A fallback here would have turned three loud crashes into three plausible
    wrong numbers, which is the one outcome worse than a failed job.
    """
    sha = _git(live, "rev-parse", "HEAD")
    root = tmp_path / "pins"
    pin = Path(_resolve(live, sha, root).stdout.strip())
    (pin / MEASURED).write_text("edited\n")
    refusals = [
        _resolve(live, "0" * 40, root),
        _resolve(live, sha[:12], root),
        _resolve(tmp_path, sha, root),
        _resolve(live, sha, root),
    ]
    for done in refusals:
        assert done.returncode == 2, done.stdout
        assert str(live) not in done.stdout, done.stdout
        assert done.stdout.strip() == "", done.stdout


def test_the_recipe_prologue_resolves_the_pin_and_dies_rather_than_fall_back(
    live: Path, tmp_path: Path
) -> None:
    """The two lines each solv2r recipe would change, run against a real context shape.

    Field names taken from a live labctl context: `provenance.repo_path` and
    `provenance.git_head` (a full 40-character sha).
    """
    sha = _git(live, "rev-parse", "HEAD")
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"provenance": {"repo_path": str(live), "git_head": sha}}))
    root = tmp_path / "pins"
    prologue = f"""
set -euo pipefail
CONTEXT_REPO="$(jq -er '.provenance.repo_path' "$LABCTL_CONTEXT")"
SOURCE="$(bash {RESOLVER} "$CONTEXT_REPO" \
  "$(jq -er '.provenance.git_head' "$LABCTL_CONTEXT")" {root})"
cd "$SOURCE"
cat {MEASURED}
"""

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("bash", "-c", prologue),
            env={"LABCTL_CONTEXT": str(context), "PATH": os.environ["PATH"]},
            capture_output=True,
            text=True,
            timeout=120,
        )

    done = run()
    assert done.returncode == 0, done.stderr
    assert done.stdout == "committed\n", done.stdout

    # The live tree moves under the job, as it did for all three failures.
    (live / MEASURED).write_text("half-edited\n")
    assert run().stdout == "committed\n", "the job must still read the commit it was sent with"

    # And if the pin itself is touched, the job dies instead of reading the live tree.
    (Path(_resolve(live, sha, root).stdout.strip()) / MEASURED).write_text("touched\n")
    broken = run()
    assert broken.returncode != 0, broken.stdout
    assert "has been edited" in broken.stderr, broken.stderr
    assert "half-edited" not in broken.stdout, broken.stdout
