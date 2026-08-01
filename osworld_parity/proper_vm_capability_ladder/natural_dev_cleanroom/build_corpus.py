from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .schema import CORPUS_PATH, canonical_json, sha256_value


RESET = {
    "snapshot": "osworld_ready",
    "strategy": "restore_then_seed_private_fixture",
    "reproducible_signature_required": True,
    "state_isolation": "unique_guest_root_per_task",
}
VERIFIER = {
    "module": "osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.oracle",
    "fresh_process": True,
    "machine_readable": True,
    "reset_reject": True,
    "near_miss_reject": True,
    "gold_pass": True,
}
DIFFICULTIES = ("easy", "medium", "hard", "medium", "hard", "easy", "medium", "hard", "easy", "medium")


def _seal(row: dict[str, Any]) -> dict[str, Any]:
    fixture_keys = (
        "id",
        "app",
        "split",
        "parameter_seed",
        "semantic_steps",
        "horizon",
        "instruction",
        "params",
        "expected",
        "near_miss",
    )
    fixture = {key: row[key] for key in fixture_keys}
    row = {**row, "fixture_sha256": sha256_value(fixture)}
    row["task_sha256"] = sha256_value(row)
    return row


def _base(
    *,
    task_id: str,
    app: str,
    seed: int,
    instruction: str,
    semantic_steps: int,
    horizon: int,
    capabilities: list[str],
    params: dict[str, Any],
    expected: dict[str, Any],
    near_miss: dict[str, Any],
    recovery: dict[str, str],
    index: int,
) -> dict[str, Any]:
    return _seal(
        {
            "id": task_id,
            "app": app,
            "split": "development",
            "parameter_seed": seed,
            "difficulty": DIFFICULTIES[index],
            "semantic_steps": semantic_steps,
            "horizon": horizon,
            "instruction": instruction,
            "capabilities": capabilities,
            "params": params,
            "expected": expected,
            "near_miss": near_miss,
            "recovery": recovery,
            "reset": RESET,
            "verifier": VERIFIER,
        }
    )


def _writer_tasks() -> list[dict[str, Any]]:
    notes = (
        "Lantern review: confirm Room Cedar at 14:30.",
        "Harbor checklist: pack badges, cables, and spare labels.",
        "Orchid brief: publish the draft after the noon review.",
        "Juniper memo: send the revised estimate before Friday.",
        "Copper plan: archive receipts after the budget check.",
        "Nimbus note: reserve two quiet desks for the visitors.",
        "Maple update: route the sample box through reception.",
        "Quartz log: record the calibration result as 7.25.",
        "Willow agenda: discuss hiring, travel, then security.",
        "Saffron recap: share action items with the design group.",
    )
    rows = []
    for index, text in enumerate(notes):
        number = index + 1
        rows.append(
            _base(
                task_id=f"cln-dev-writer-{number:02d}-bold-note",
                app="writer",
                seed=4100 + number,
                index=index,
                semantic_steps=3,
                horizon=4,
                instruction=(
                    f"In the open Writer document, replace all text with exactly “{text}” "
                    "Make the complete replacement bold, then save the document."
                ),
                capabilities=["click", "coalesced_type", "hotkey", "multi_step_state_change"],
                params={
                    "file_name": f"cleanroom_writer_{number:02d}.odt",
                    "initial_text": f"Temporary Writer draft {number:02d}",
                },
                expected={"text": text, "bold": True},
                near_miss={"text": text, "bold": False},
                recovery={
                    "near_miss_class": "correct_text_missing_bold_format",
                    "corrective_action": "select the full replacement, apply bold, and save again",
                },
            )
        )
    return rows


def _calc_tasks() -> list[dict[str, Any]]:
    operands = ((4, 9), (6, 13), (8, 17), (11, 7), (12, 19), (15, 8), (18, 14), (21, 16), (24, 5), (27, 12))
    cells = ("B3", "C4", "D5", "E6", "F7", "G8", "H9", "I10", "J11", "K12")
    rows = []
    for index, ((left, right), cell) in enumerate(zip(operands, cells, strict=True)):
        number = index + 1
        value = left + right
        rows.append(
            _base(
                task_id=f"cln-dev-calc-{number:02d}-sum-cell",
                app="calc",
                seed=4200 + number,
                index=index,
                semantic_steps=4,
                horizon=6,
                instruction=(
                    f"In Calc, go to cell {cell}, enter a formula that adds {left} and {right}, "
                    "confirm it, and save the spreadsheet."
                ),
                capabilities=["click", "coalesced_type", "hotkey", "multi_step_state_change"],
                params={
                    "file_name": f"cleanroom_calc_{number:02d}.ods",
                    "cell": cell,
                    "initial_value": "0",
                },
                expected={"formula": f"of:=SUM({left};{right})", "display_value": str(value)},
                near_miss={"formula": f"of:=SUM({left};{right - 1})", "display_value": str(value - 1)},
                recovery={
                    "near_miss_class": "formula_second_operand_off_by_one",
                    "corrective_action": "return to the named cell, correct the second operand, confirm, and save",
                },
            )
        )
    return rows


def _files_tasks() -> list[dict[str, Any]]:
    labels = ("amber", "birch", "coral", "dune", "ember", "fjord", "grove", "hazel", "ivory", "jade")
    rows = []
    for index, label in enumerate(labels):
        number = index + 1
        source = f"z_source_{label}.txt"
        destination = f"b_archive_{label}"
        decoy = f"a_review_{label}"
        final_name = f"{label}_approved.txt"
        rows.append(
            _base(
                task_id=f"cln-dev-files-{number:02d}-{label}-archive",
                app="files",
                seed=4300 + number,
                index=index,
                semantic_steps=3,
                horizon=8,
                instruction=(
                    f"In Files, move {source} into the folder {destination}, then rename the "
                    f"moved file to {final_name}."
                ),
                capabilities=["click", "drag", "coalesced_type", "hotkey", "multi_step_state_change"],
                params={
                    "source_name": source,
                    "destination_name": destination,
                    "decoy_name": decoy,
                    "content": f"Clean-room archive payload {number:02d}: {label}.\n",
                },
                expected={"destination": destination, "final_name": final_name},
                near_miss={"destination": decoy, "final_name": final_name},
                recovery={
                    "near_miss_class": "correct_rename_in_decoy_folder",
                    "corrective_action": "move the renamed file from the review folder into the requested archive folder",
                },
            )
        )
    return rows


def _chrome_tasks() -> list[dict[str, Any]]:
    pairs = (
        ("privacy", "Allow local diagnostics"),
        ("updates", "Notify before restart"),
        ("downloads", "Ask where to save files"),
        ("accessibility", "Show focus indicators"),
        ("language", "Offer page translations"),
        ("security", "Warn about unsafe forms"),
        ("startup", "Restore the last workspace"),
        ("content", "Block intrusive popups"),
        ("performance", "Pause sleeping tabs"),
        ("appearance", "Use compact controls"),
    )
    rows = []
    for index, (section, setting) in enumerate(pairs):
        number = index + 1
        rows.append(
            _base(
                task_id=f"cln-dev-chrome-{number:02d}-{section}-toggle",
                app="chrome",
                seed=4400 + number,
                index=index,
                semantic_steps=3,
                horizon=6,
                instruction=(
                    f"On the local settings page, open {section}, scroll down to the controls, "
                    f"and enable “{setting}”."
                ),
                capabilities=["click", "signed_vertical_scroll", "multi_step_state_change"],
                params={
                    "port": 18400 + number,
                    "section": section,
                    "setting": setting,
                    "initial_scroll_y": 0,
                    "scroll_direction": "down",
                    "minimum_scroll_delta": 500,
                },
                expected={"section": section, "setting_enabled": True},
                near_miss={"section": "appearance", "setting_enabled": False},
                recovery={
                    "near_miss_class": "decoy_section_and_decoy_toggle",
                    "corrective_action": "return to the requested section, scroll to its control, and enable the requested setting",
                },
            )
        )
    return rows


def build() -> dict[str, Any]:
    tasks = _writer_tasks() + _calc_tasks() + _files_tasks() + _chrome_tasks()
    payload = {
        "schema_version": 1,
        "suite": "cleanroom_natural_multistep_vm_development_v1",
        "split": "development",
        "task_count": len(tasks),
        "observation_contract": "instruction_and_screenshot_only",
        "oracle_visibility": "fresh_host_process_only",
        "model_runs": False,
        "eligibility": {
            "purpose": "auxiliary_development_only",
            "stage0": False,
            "final": False,
        },
        "provenance": {
            "construction": "first_principles_parameterized_local_app_primitives",
            "source_scope": "explicit_safe_development_fixture_apis_only",
            "external_benchmark_material_consumed": False,
            "test_derived": False,
        },
        "tasks": tasks,
    }
    return {**payload, "manifest_payload_sha256": sha256_value(payload)}


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
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    args = parser.parse_args(argv)
    write(args.output)
    print(json.dumps({"output": str(args.output), "sha256": sha256_value(build())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
