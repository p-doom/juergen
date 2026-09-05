"""Exercise the published wheel contract outside the checkout."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_DESKTOP_REQUIREMENT = (
    "desktop @ git+https://github.com/p-doom/desktop.git@"
    "1db6ae2499afc16d87dee15453a57042dff13f64"
)
_PROBED = (
    "cua_parity_contract",
    "image_domain",
    "grammars",
    "desktop.geometry",
    "desktop.ir",
)
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
_PROBE = """
import importlib, json, sys
print(json.dumps({name: importlib.import_module(name).__file__ for name in sys.argv[1:]}))
"""


def _metadata(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        return archive.read(name).decode()


def _members(wheel: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(wheel) as archive:
        return tuple(archive.namelist())


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> tuple[Path, Path]:
    uv = shutil.which("uv")
    assert uv is not None
    root = tmp_path_factory.mktemp("dist")
    source = root / "source"
    shutil.copytree(_REPO, source, ignore=_NOT_SOURCE, symlinks=True)
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(root), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return (
        root,
        root / next(path.name for path in root.glob("juergen-*.whl")),
    )


def test_published_wheel_and_pipeline_runtime_pin_exact_desktop(wheel):
    _, juergen = wheel
    assert f"Requires-Dist: {_DESKTOP_REQUIREMENT}" in _metadata(juergen)
    pipeline_project = tomllib.loads(
        (_REPO / "data_pipeline" / "pyproject.toml").read_text()
    )["project"]
    assert _DESKTOP_REQUIREMENT in pipeline_project["dependencies"]


def test_published_juergen_wheel_contains_no_test_modules(wheel):
    _, juergen = wheel
    assert not [
        name for name in _members(juergen) if Path(name).name.startswith("test_")
    ]


def test_normal_wheel_install_resolves_the_pinned_desktop(wheel):
    root, juergen = wheel
    uv = shutil.which("uv")
    environment = root / "venv"
    subprocess.run(
        [uv, "venv", "--python", sys.executable, str(environment)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = environment / "bin" / "python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(juergen)],
        check=True,
        capture_output=True,
        text=True,
    )
    process = subprocess.run(
        [str(python), "-I", "-c", _PROBE, *_PROBED],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    files = {name: Path(path) for name, path in json.loads(process.stdout).items()}
    site_packages = environment / "lib" / "python3.12" / "site-packages"
    assert all(path.is_relative_to(site_packages) for path in files.values())
