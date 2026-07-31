from __future__ import annotations

import ast
import tomllib
from pathlib import Path


RECIPE_ROOT = Path(__file__).parents[3] / "labctl" / "recipes"


def test_release_and_aggregate_recipes_are_cpu_only_and_have_no_source_inputs() -> None:
    recipes = {
        path.name: tomllib.loads(path.read_text(encoding="utf-8"))
        for path in RECIPE_ROOT.glob("rung5_official_pilot_*_cpu.toml")
    }
    assert set(recipes) == {
        "rung5_official_pilot_release_cpu.toml",
        "rung5_official_pilot_aggregate_cpu.toml",
    }
    forbidden_inputs = {
        "task",
        "tasks",
        "source",
        "split",
        "model",
        "checkpoint",
        "vm",
        "osworld",
        "runtime",
        "provider",
    }
    for recipe in recipes.values():
        assert recipe["resources"]["gpus"] == 0
        assert "CUDA_VISIBLE_DEVICES" in " ".join(recipe["command"])
        assert not (set(recipe.get("inputs", {})) & forbidden_inputs)
        args = recipe["args"]
        assert {
            "prerequisites_gate",
            "prerequisites_signature",
            "pilot_release_gate",
            "pilot_release_signature",
            "allowed_signers",
            "signer_identity",
        } <= set(args)
        assert "release_trust" in recipe["inputs"]
        assert args["allowed_signers"].startswith("{inputs.release_trust.path}")


def test_source_module_has_protocol_only_and_no_filesystem_access() -> None:
    source_path = Path(__file__).parents[1] / "source.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "pathlib" not in imported_modules | imported_names
    assert "os" not in imported_modules | imported_names
    assert "glob" not in imported_modules | imported_names
    assert "subprocess" not in imported_modules | imported_names
    assert not any(isinstance(node, ast.FunctionDef) for node in tree.body)
