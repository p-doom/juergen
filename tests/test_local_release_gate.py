"""The release verdict comes from a fresh hash-locked local environment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "tooling" / "local_release_gate.sh"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*args], cwd=cwd, text=True, capture_output=True, check=True
    )


def _commit(root: Path, message: str) -> str:
    _run("git", "add", ".", cwd=root)
    _run(
        "git",
        "-c",
        "user.name=Release Gate Test",
        "-c",
        "user.email=release-gate@example.invalid",
        "commit",
        "-m",
        message,
        cwd=root,
    )
    return _run("git", "rev-parse", "HEAD", cwd=root).stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path / "juergen"
    desktop = tmp_path / "desktop"
    root.mkdir()
    desktop.mkdir()
    _run("git", "init", "-q", cwd=root)
    _run("git", "init", "-q", cwd=desktop)
    (desktop / "desktop.py").write_text("PINNED = True\n")
    desktop_sha = _commit(desktop, "desktop")
    _run(
        "git",
        "update-ref",
        "refs/remotes/origin/main",
        desktop_sha,
        cwd=desktop,
    )

    tooling = root / "tooling"
    venvs = tooling / "venvs"
    venvs.mkdir(parents=True)
    shutil.copy2(SCRIPT, tooling / SCRIPT.name)
    (root / "pyproject.toml").write_text(
        "desktop = { git = \"file:///sealed/desktop.git\", "
        f"rev = \"{desktop_sha}\" }}\n"
    )
    (venvs / "juergen-testgate-venv.requirements.txt").write_text(
        "pytest==9.1.1 --hash=sha256:test\n"
    )
    rebuild = venvs / "rebuild-juergen-testgate-venv.sh"
    rebuild.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$1/bin\"\n"
        "ln -s \"$FAKE_RELEASE_PYTHON\" \"$1/bin/python\"\n"
    )
    rebuild.chmod(0o755)
    _commit(root, "release gate")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "[ \"${1:-}\" = lock ] || exit 91\n"
    )
    fake_uv.chmod(0o755)
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_RELEASE_LOG\"\n"
        "printf '999 passed\\n'\n"
    )
    fake_python.chmod(0o755)
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_RELEASE_PYTHON": str(fake_python),
        "FAKE_RELEASE_LOG": str(tmp_path / "pytest.log"),
        "TMPDIR": str(scratch_parent),
    }
    return root, desktop, environment


def test_gate_rebuilds_then_runs_the_complete_local_suite(tmp_path) -> None:
    root, _, environment = _fixture(tmp_path)

    result = subprocess.run(
        ["bash", "tooling/local_release_gate.sh"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "JUERGEN LOCAL RELEASE GATE: GREEN" in result.stdout
    assert (tmp_path / "pytest.log").read_text().strip() == (
        "-m pytest -q -p no:cacheprovider tests grammars"
    )
    assert not list((tmp_path / "scratch").iterdir())


def test_gate_refuses_a_dirty_tree_before_rebuilding(tmp_path) -> None:
    root, _, environment = _fixture(tmp_path)
    (root / "untracked").write_text("not a commit\n")

    result = subprocess.run(
        ["bash", "tooling/local_release_gate.sh"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Juergen checkout is dirty" in result.stderr
    assert not (tmp_path / "pytest.log").exists()


def test_gate_refuses_a_desktop_other_than_the_published_pin(tmp_path) -> None:
    root, desktop, environment = _fixture(tmp_path)
    (desktop / "desktop.py").write_text("PINNED = False\n")
    _commit(desktop, "moved desktop")

    result = subprocess.run(
        ["bash", "tooling/local_release_gate.sh"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "desktop checkout/remote does not equal pinned" in result.stderr
    assert not (tmp_path / "pytest.log").exists()
