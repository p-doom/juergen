import os
import subprocess
import tomllib
from pathlib import Path

from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.vm import (
    GUEST_ROOT_NAME,
    resolve_guest_root,
)


REPO_ROOT = Path(__file__).parents[4]
RECIPE_ROOT = REPO_ROOT / "osworld_parity" / "labctl" / "recipes"
RUNG1B_RECIPES = (
    "rung1b_realapps_build_selfcheck_cpu.toml",
    "rung1b_realapps_vm_selfcheck_cpu_kvm.toml",
    "rung1b_training_data_build_cpu.toml",
    "rung1b_teacher_collect_cpu_kvm.toml",
)


class LocalExecuteTransport:
    def __init__(self, env):
        self.env = env
        self.calls = 0

    def execute_argv(self, argv):
        self.calls += 1
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            env=self.env,
        )
        return {
            "status": "success" if completed.returncode == 0 else "error",
            "returncode": completed.returncode,
            "output": completed.stdout,
            "error": completed.stderr,
        }


def test_guest_root_falls_back_to_validated_temp_when_home_is_missing(tmp_path):
    guest_tmp = tmp_path / "guest-tmp"
    guest_tmp.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "missing-home"),
            "XDG_RUNTIME_DIR": str(tmp_path / "missing-runtime"),
            "TMPDIR": str(guest_tmp),
        }
    )
    transport = LocalExecuteTransport(env)

    root = resolve_guest_root(transport)

    assert root == Path(guest_tmp / GUEST_ROOT_NAME)
    assert Path(root).is_dir()
    assert transport.calls == 1
    assert resolve_guest_root(transport) == root
    assert transport.calls == 1


def test_recipes_use_the_declared_locked_runtime_not_ambient_python():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(
        requirement.startswith("requests") for requirement in project["project"]["dependencies"]
    )
    assert any(
        requirement.startswith("pytest")
        for requirement in project["dependency-groups"]["dev"]
    )
    for name in RUNG1B_RECIPES:
        recipe = tomllib.loads((RECIPE_ROOT / name).read_text(encoding="utf-8"))
        script = "\n".join(recipe["command"])
        assert "uv run --locked" in script
        assert "python3 -m pytest" not in script
    build = (RECIPE_ROOT / RUNG1B_RECIPES[0]).read_text(encoding="utf-8")
    assert "uv run --locked --group dev python -m pytest" in build
