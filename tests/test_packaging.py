"""What a dispatched run gets: the built distribution, not the checkout.

Resolving an id in-process proves nothing: the repo root is already
`sys.path[0]` there, so it passes whether or not the module is packaged at all.
It did: `py-modules = []` shipped none of the flat ids, and `top_level.txt` read
`grammars` alone. Nothing noticed, because every caller so far has run with the
checkout as its working directory.

So this file builds the wheels and imports out of them with the checkout nowhere
on `sys.path`, which is the only arrangement that can tell the two apart.

`desktop` is built here too rather than taken from the sibling directory: it is
the dependency whose import name PyPI's unrelated `desktop` 0.4.2 also owns, and
asserting every module resolves *out of our wheel* is what distinguishes ours
from the index's.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_DESKTOP = _REPO.parent / "desktop"

# The one flat module left in `[tool.setuptools] py-modules`. It is not a plugin
# id -- it is the image-domain spelling `agent/` and `pipeline/` both write, flat
# because `pipeline/` runs in a venv without `verifiers` and so can import nothing
# under `agent/`.
FLAT_MODULES = ("image_domain",)

PROBED = FLAT_MODULES + (
    "grammars",
    "desktop.geometry",
    "desktop.ir",
)

# Nothing a build reads, and `build/` in particular is poison: setuptools copies
# the packages into it and reuses whatever is already there, so a wheel built in a
# tree with a stale one carries files the current declaration does not name.
_NOT_SOURCE = shutil.ignore_patterns(
    ".git",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    "*.egg-info",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
)

# Every module in one interpreter: `verifiers` costs six seconds to import, so a
# subprocess per module is a minute of gate for nothing.
_PROBE = """
import importlib, json, sys
print(json.dumps({name: importlib.import_module(name).__file__ for name in sys.argv[1:]}))
"""


@pytest.fixture(scope="module")
def resolved(tmp_path_factory) -> tuple[Path, dict[str, Path]]:
    """Both wheels unpacked into one directory that is not a checkout, and the
    file that answered each import made from a working directory outside both."""
    uv = shutil.which("uv")
    assert uv, "uv is the estate's build front end and is in none of the suites' venvs"
    root = tmp_path_factory.mktemp("dist")
    site = root / "site"
    for project in (_REPO, _DESKTOP):
        # Built from a copy, never in place: a build writes `build/` and an
        # egg-info into the project directory, and `tests/test_suite_and_gate.py`
        # walks the repo looking for duplicated files.
        source = root / f"src-{project.name}"
        shutil.copytree(project, source, ignore=_NOT_SOURCE, symlinks=True)
        subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(root), str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
    wheels = sorted(root.glob("*.whl"))
    assert len(wheels) == 2, f"expected a juergen and a desktop wheel, got {wheels}"
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(site)

    process = subprocess.run(
        [sys.executable, "-c", _PROBE, *PROBED],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(site)),
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, f"importing {PROBED} failed:\n{process.stderr}"
    return site, {name: Path(path) for name, path in json.loads(process.stdout).items()}


@pytest.mark.parametrize("flat_module", FLAT_MODULES)
def test_every_flat_module_imports_out_of_the_wheel(resolved, flat_module: str) -> None:
    site, files = resolved
    assert files[flat_module].is_relative_to(site)


def test_grammars_imports_with_only_the_wheels_on_the_path(resolved) -> None:
    # grammars hard-imports desktop, and `desktop` is installed in neither shared
    # testgate venv: every suite resolves it by a sys.path append in conftest, so a
    # declared dependency that does not actually install cannot fail a test.
    site, files = resolved
    assert files["grammars"].is_relative_to(site)


def test_the_desktop_that_answers_is_ours_and_not_the_index_one(resolved) -> None:
    # PyPI's `desktop` 0.4.2 imports fine and has neither submodule, so resolving
    # the two `grammars/_support.py` needs out of our wheel is the whole check.
    site, files = resolved
    assert files["desktop.geometry"].is_relative_to(site)
    assert files["desktop.ir"].is_relative_to(site)


def test_the_pinned_desktop_source_resolves_where_a_path_could_not() -> None:
    """A dispatched run resolves a recorded desktop revision."""
    import tomllib

    source = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    pin = source["tool"]["uv"]["sources"]["desktop"]
    url, rev = pin["git"], pin["rev"]
    assert url == "https://github.com/p-doom/desktop.git"
    assert len(rev) == 40 and not set(rev) - set("0123456789abcdef"), rev
