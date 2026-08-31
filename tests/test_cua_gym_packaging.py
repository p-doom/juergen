"""The dispatched CUA-Gym runtime imports from the built wheel."""

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
_MODULES = (
    "harness_render",
    "cua_gym",
    "cua_gym_web",
    "evals.cua_gym.manifest",
    "evals.cua_gym.runtime",
    "evals.cua_gym.web.gateway",
    "evals.cua_gym.web.hub",
    "evals.cua_gym.web.image",
    "evals.cua_gym.web.manifest",
    "evals.cua_gym.web.runtime",
    "grammars.ordered_events_v3.codec",
    "stream_cuagym_qwen35",
)
_PROBE = """
import importlib, json, sys
print(json.dumps({name: importlib.import_module(name).__file__ for name in sys.argv[1:]}))
"""
_NOT_SOURCE = shutil.ignore_patterns(
    ".git",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    "*.egg-info",
    ".pytest_cache",
)


@pytest.fixture(scope="module")
def built_wheel(
    tmp_path_factory,
) -> tuple[Path, frozenset[str], dict[str, Path]]:
    uv = shutil.which("uv")
    assert uv is not None
    root = tmp_path_factory.mktemp("cua-gym-wheel")
    source = root / "source"
    shutil.copytree(_REPO, source, ignore=_NOT_SOURCE, symlinks=True)
    wheel_dir = root / "dist"
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_dir), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = tuple(wheel_dir.glob("juergen-*.whl"))
    site = root / "site"
    with zipfile.ZipFile(wheel) as archive:
        names = frozenset(archive.namelist())
        archive.extractall(site)
    process = subprocess.run(
        [sys.executable, "-c", _PROBE, *_MODULES],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(site)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    files = {name: Path(path) for name, path in json.loads(process.stdout).items()}
    return site, names, files


def test_cua_gym_runtime_imports_from_the_wheel(built_wheel) -> None:
    site, _, files = built_wheel
    assert all(path.is_relative_to(site) for path in files.values())


def test_wheel_contains_the_pinned_manifest_and_no_excluded_front_doors(
    built_wheel,
) -> None:
    _, names, _ = built_wheel
    from evals.cua_gym import PINNED_REVISION

    assert f"evals/cua_gym/compatibility/{PINNED_REVISION}.json" in names
    assert "rl_grounding.py" not in names
    assert "rl_movebox.py" not in names
    assert "rl_target_box.py" not in names
    assert f"evals/cua_gym/web/compatibility/{PINNED_REVISION}.json" in names
    assert "evals/cua_gym/web/gateway.py" in names
    assert "evals/cua_gym/web/hub.py" in names
    assert "evals/cua_gym/web/image.py" in names
    assert "evals/cua_gym/web/runtime.py" in names
    assert "cua_gym_web.py" in names
    assert "image_domain.py" not in names
    assert not any(name.startswith("reinforcement_learning/") for name in names)


def test_desktop_dependency_is_pinned_to_the_reviewed_runtime_revision() -> None:
    import tomllib

    source = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert source["tool"]["uv"]["sources"]["desktop"] == {
        "git": "https://github.com/p-doom/desktop.git",
        "rev": "1db6ae2499afc16d87dee15453a57042dff13f64",
    }


def test_wheel_contains_the_candidate_render_resources(built_wheel) -> None:
    _, names, _ = built_wheel
    assert "stream_cuagym_qwen35/render_spec.json" in names
    assert "stream_cuagym_qwen35/system_prompt.txt" in names
