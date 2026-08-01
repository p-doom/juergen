from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .build_corpus import _base
from .schema import sha256_value
from .smoke_schema import SMOKE_PATH


def _wrap(source_task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(source_task["id"])
    app = str(source_task["app"])
    payload = {
        "id": task_id,
        "mode": "single",
        "bridge": "single_app",
        "anchor_app": app,
        "ordered_components": [
            {
                "order": 1,
                "app": app,
                "task_id": task_id,
                "semantic_steps": int(source_task["semantic_steps"]),
            }
        ],
        "eligibility": {"purpose": "plumbing_smoke_only", "stage0": False, "final": False},
        "source_task": source_task,
        "source_task_payload_sha256": sha256_value(source_task),
    }
    return {**payload, "record_sha256": sha256_value(payload)}


def _tasks() -> list[dict[str, Any]]:
    rows = [
        _base(
            task_id="cln-smoke-writer-01-pine-check",
            app="writer",
            seed=5101,
            index=0,
            semantic_steps=3,
            horizon=4,
            instruction=(
                "In the open Writer document, replace all text with exactly “Pine smoke: input path ready.” "
                "Make the complete replacement bold, then save the document."
            ),
            capabilities=["click", "coalesced_type", "hotkey", "multi_step_state_change"],
            params={"file_name": "smoke_writer_pine.odt", "initial_text": "Temporary smoke draft"},
            expected={"text": "Pine smoke: input path ready.", "bold": True},
            near_miss={"text": "Pine smoke: input path ready.", "bold": False},
            recovery={
                "near_miss_class": "correct_text_missing_bold_format",
                "corrective_action": "select the replacement, apply bold, and save again",
            },
        ),
        _base(
            task_id="cln-smoke-vscode-01-oak-check",
            app="vscode",
            seed=5102,
            index=1,
            semantic_steps=3,
            horizon=4,
            instruction=(
                "In the open VS Code editor, replace all text with exactly “Oak smoke: receipt path ready.” "
                "Then save the file."
            ),
            capabilities=["click", "coalesced_type", "hotkey", "multi_step_state_change"],
            params={"file_name": "smoke_vscode_oak.txt", "initial_text": "Another temporary smoke draft"},
            expected={"text": "Oak smoke: receipt path ready."},
            near_miss={"text": "Oak smoke: receipt path almost ready."},
            recovery={
                "near_miss_class": "near_exact_text_not_saved_as_requested",
                "corrective_action": "replace the near miss with the exact requested text and save",
            },
        ),
        _base(
            task_id="cln-smoke-calc-01-three-plus-five",
            app="calc",
            seed=5103,
            index=2,
            semantic_steps=4,
            horizon=6,
            instruction="In Calc, go to cell C3, enter a formula that adds 3 and 5, confirm it, and save the spreadsheet.",
            capabilities=["click", "coalesced_type", "hotkey", "multi_step_state_change"],
            params={"file_name": "smoke_calc_sum.ods", "cell": "C3", "initial_value": "0"},
            expected={"formula": "of:=SUM(3;5)", "display_value": "8"},
            near_miss={"formula": "of:=SUM(3;4)", "display_value": "7"},
            recovery={
                "near_miss_class": "formula_second_operand_off_by_one",
                "corrective_action": "correct the second operand, confirm, and save",
            },
        ),
        _base(
            task_id="cln-smoke-files-01-spruce-move",
            app="files",
            seed=5104,
            index=1,
            semantic_steps=3,
            horizon=8,
            instruction=(
                "In Files, move z_source_spruce.txt into the folder b_archive_spruce, "
                "then rename the moved file to spruce_ready.txt."
            ),
            capabilities=["click", "drag", "coalesced_type", "hotkey", "multi_step_state_change"],
            params={
                "source_name": "z_source_spruce.txt",
                "destination_name": "b_archive_spruce",
                "decoy_name": "a_review_spruce",
                "content": "Clean-room plumbing smoke payload: spruce.\n",
            },
            expected={"destination": "b_archive_spruce", "final_name": "spruce_ready.txt"},
            near_miss={"destination": "a_review_spruce", "final_name": "spruce_ready.txt"},
            recovery={
                "near_miss_class": "correct_rename_in_decoy_folder",
                "corrective_action": "move the renamed file into the requested archive folder",
            },
        ),
        _base(
            task_id="cln-smoke-chrome-01-network-toggle",
            app="chrome",
            seed=5105,
            index=2,
            semantic_steps=3,
            horizon=6,
            instruction=(
                "On the local settings page, open network, scroll down to the controls, "
                "and enable “Report offline transitions”."
            ),
            capabilities=["click", "signed_vertical_scroll", "multi_step_state_change"],
            params={
                "port": 18505,
                "section": "network",
                "setting": "Report offline transitions",
                "initial_scroll_y": 0,
                "scroll_direction": "down",
                "minimum_scroll_delta": 500,
            },
            expected={"section": "network", "setting_enabled": True},
            near_miss={"section": "appearance", "setting_enabled": False},
            recovery={
                "near_miss_class": "decoy_section_and_decoy_toggle",
                "corrective_action": "open network, scroll to its control, and enable the requested setting",
            },
        ),
    ]
    return [_wrap(row) for row in rows]


def build() -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "suite": "natural_dev_disjoint_smoke_inventory_v1",
        "status": "authored",
        "split": "development",
        "development_only": True,
        "role": "plumbing_smoke_only",
        "source_policy": {
            "construction": "first_principles_parameterized_local_app_primitives",
            "deny_before_open": True,
            "external_benchmark_material_consumed": False,
            "external_rollout_systems_used": False,
            "model_runs": False,
            "source_scope": "explicit_safe_development_fixture_apis_only",
            "test_derived": False,
        },
        "tasks": _tasks(),
    }
    return {**payload, "inventory_payload_sha256": sha256_value(payload)}


def write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(build(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SMOKE_PATH)
    args = parser.parse_args(argv)
    write(args.output)
    print(json.dumps({"output": str(args.output), "task_count": 5}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
