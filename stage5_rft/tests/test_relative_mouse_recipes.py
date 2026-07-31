from __future__ import annotations

import tomllib
from pathlib import Path

from stage5_rft.cli import SELF_WRITING_COMMANDS


RECIPE_DIR = Path(__file__).parents[1] / "labctl" / "recipes"


def test_relative_mouse_commands_do_not_use_unrendered_command_templates():
    """The deployed labctl only renders templates from ``[args]`` values.

    Paths used directly by a shell command must therefore come from the
    registered ``LABCTL_CONTEXT`` rather than command-vector placeholders.
    """

    recipes = sorted(RECIPE_DIR.glob("relative_mouse_*.toml"))
    assert recipes
    for path in recipes:
        payload = tomllib.loads(path.read_text())
        command = payload["command"]
        assert all("{inputs." not in part for part in command), path.name
        assert all("{outputs." not in part for part in command), path.name
        assert "LABCTL_CONTEXT" in "\n".join(command), path.name


def test_directory_producing_relative_mouse_commands_skip_generic_file_write():
    assert "seal-relative-mouse-batch" in SELF_WRITING_COMMANDS
    assert "build-relative-mouse-rft" in SELF_WRITING_COMMANDS
