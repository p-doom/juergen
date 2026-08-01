from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .stage0_loader import (
    ANCHOR_APPS,
    CELLS_PER_ANCHOR_MODE,
    DEVELOPMENT,
    DIFFICULTY_BY_CELL,
    MODES,
    RECORD_ELIGIBILITY,
    RESET_CONTRACT,
    SOURCE_POLICY,
    SOURCE_VERIFIER,
    STAGE0_INVENTORY_PATH,
    canonical_json,
    sha256_value,
)


def _seal_source(row: dict[str, Any]) -> dict[str, Any]:
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
    sealed = {**row, "fixture_sha256": sha256_value(fixture)}
    return {**sealed, "task_sha256": sha256_value(sealed)}


def _source_task(
    *, task_id: str, app: str, seed: int, cell: int, code: int, mode: str
) -> dict[str, Any]:
    suffix = task_id.removeprefix("cln-s0-src-").replace("-", "_")
    if app == "writer":
        text = f"Handoff code {code}: confirm the clean-room review."
        instruction = (
            f'In the open Writer document, replace all text with exactly “{text}” '
            "Make the complete replacement bold, then save the document."
        )
        semantic_steps, horizon = 3, 4
        capabilities = ["click", "coalesced_type", "hotkey", "multi_step_state_change"]
        params = {
            "file_name": f"stage0_{suffix}.odt",
            "initial_text": f"Temporary Writer draft for code {code}",
        }
        expected = {"text": text, "bold": True}
        near_miss = {"text": text, "bold": False}
        recovery = {
            "near_miss_class": "correct_text_missing_bold_format",
            "corrective_action": "select the full replacement, apply bold, and save again",
        }
    elif app == "calc":
        left = code - cell
        right = cell
        target = ("B3", "C4", "D5", "E6")[cell - 1]
        instruction = (
            f"In Calc, go to cell {target}, enter a formula that adds {left} and {right}, "
            "confirm it, and save the spreadsheet."
        )
        semantic_steps, horizon = 4, 6
        capabilities = ["click", "coalesced_type", "hotkey", "multi_step_state_change"]
        params = {
            "file_name": f"stage0_{suffix}.ods",
            "cell": target,
            "initial_value": "0",
        }
        expected = {"formula": f"of:=SUM({left};{right})", "display_value": str(left + right)}
        near_miss = {"formula": f"of:=SUM({left};{max(0, right - 1)})", "display_value": str(left + max(0, right - 1))}
        recovery = {
            "near_miss_class": "formula_second_operand_off_by_one",
            "corrective_action": "return to the named cell, correct the second operand, confirm, and save",
        }
    elif app == "files":
        source = f"incoming_{code}_{cell}.txt"
        destination = f"archive_{code}_{cell}"
        decoy = f"review_{code}_{cell}"
        final_name = f"handoff_{code}.txt"
        instruction = (
            f'In Files, move “{source}” into “{destination}”, then rename the moved '
            f'file to “{final_name}”.'
        )
        semantic_steps, horizon = 3, 8
        capabilities = ["click", "drag", "coalesced_type", "hotkey", "multi_step_state_change"]
        params = {
            "source_name": source,
            "destination_name": destination,
            "decoy_name": decoy,
            "content": f"Clean-room handoff payload {code}.\n",
        }
        expected = {"destination": destination, "final_name": final_name}
        near_miss = {"destination": decoy, "final_name": final_name}
        recovery = {
            "near_miss_class": "correct_rename_in_decoy_folder",
            "corrective_action": "move the renamed file from the review folder into the requested archive folder",
        }
    elif app == "chrome":
        section = f"handoff-{code}"
        setting = f"Confirm code {code}"
        instruction = (
            f'On the private local settings page, open “{section}”, scroll down to the '
            f'controls, and enable “{setting}”.'
        )
        semantic_steps, horizon = 3, 6
        capabilities = ["click", "signed_vertical_scroll", "multi_step_state_change"]
        params = {
            "port": 19000 + (seed - 920000),
            "section": section,
            "setting": setting,
            "initial_scroll_y": 0,
            "scroll_direction": "down",
            "minimum_scroll_delta": 500,
        }
        expected = {"section": section, "setting_enabled": True}
        near_miss = {"section": "appearance", "setting_enabled": False}
        recovery = {
            "near_miss_class": "decoy_section_and_decoy_toggle",
            "corrective_action": "return to the requested section, scroll to its control, and enable the requested setting",
        }
    elif app == "vscode":
        text = f"handoff={code}\nstate=ready\n"
        instruction = (
            "In VS Code, replace all text in the open file with exactly two lines: "
            f'“handoff={code}” and “state=ready”. Save the file.'
        )
        semantic_steps, horizon = 3, 4
        capabilities = ["click", "coalesced_type", "hotkey", "multi_step_state_change"]
        params = {
            "file_name": f"stage0_{suffix}.txt",
            "initial_text": f"draft={code}\nstate=pending\n",
        }
        expected = {"text": text}
        near_miss = {"text": f"handoff={code}\nstate=pending\n"}
        recovery = {
            "near_miss_class": "handoff_present_but_state_not_ready",
            "corrective_action": "replace the whole file with the requested two lines and save again",
        }
    else:  # pragma: no cover - fixed app list
        raise ValueError(f"unsupported app: {app}")
    if mode == "multi":
        program = {
            "writer": "writer_replace_active_body",
            "calc": "calc_replace_selected_a1",
            "files": "files_rename_selected_source",
            "chrome": "chrome_enable_visible_setting",
            "vscode": "vscode_replace_active_file",
        }[app]
        params = {**params, "stage0_program": program}
        semantic_steps = 1
        horizon = 2 if app == "chrome" else 3
        if app == "writer":
            expected_text = f"Handoff code {code}: ready for transfer."
            expected = {"text": expected_text, "bold": False}
            near_miss = {"text": f"Handoff code {code}: pending transfer.", "bold": False}
            instruction = (
                f'In the active Writer document, replace all text with exactly “{expected_text}” '
                "and save the document without adding formatting."
            )
            capabilities = ["coalesced_type", "hotkey", "multi_step_state_change"]
            recovery = {
                "near_miss_class": "handoff_status_pending_instead_of_ready",
                "corrective_action": "replace the complete line with the requested ready status and save again",
            }
        elif app == "calc":
            params = {**params, "cell": "A1"}
            instruction = (
                f"In the selected Calc cell A1, enter a formula that adds {left} and "
                f"{right}, confirm it, and save the spreadsheet."
            )
            capabilities = ["coalesced_type", "hotkey", "multi_step_state_change"]
        elif app == "files":
            expected = {"destination": "root", "final_name": final_name}
            near_miss = {
                "destination": "root",
                "final_name": f"handoff_{code}_draft.txt",
            }
            instruction = (
                f'In Files, rename the selected file “{source}” to “{final_name}”.'
            )
            capabilities = ["coalesced_type", "hotkey", "file_rename", "multi_step_state_change"]
        elif app == "chrome":
            params = {
                **params,
                "initial_scroll_y": 900,
                "minimum_scroll_delta": 0,
            }
            instruction = (
                f'On the private local settings page, open “{section}” and enable '
                f'“{setting}”. Both controls are visible.'
            )
            capabilities = ["click", "local_web_state_change", "multi_step_state_change"]
        else:
            instruction = (
                "In the active VS Code editor, replace all text with exactly two lines: "
                f'“handoff={code}” and “state=ready”. Save the active file.'
            )
            capabilities = ["coalesced_type", "hotkey", "editor_file_save", "multi_step_state_change"]
    return _seal_source(
        {
            "id": task_id,
            "app": app,
            "split": "development",
            "parameter_seed": seed,
            "difficulty": DIFFICULTY_BY_CELL[cell],
            "semantic_steps": semantic_steps,
            "horizon": horizon,
            "instruction": instruction,
            "capabilities": capabilities,
            "params": params,
            "expected": expected,
            "near_miss": near_miss,
            "recovery": recovery,
            "reset": dict(RESET_CONTRACT),
            "verifier": dict(SOURCE_VERIFIER),
        }
    )


def _record(*, anchor_app: str, mode: str, cell: int, app_index: int) -> dict[str, Any]:
    record_id = f"cln-s0-dev-{anchor_app}-{mode}-c{cell:02d}"
    record_seed = 910000 + app_index * 100 + MODES.index(mode) * 10 + cell
    code = 500 + app_index * 50 + cell * 3
    apps = [anchor_app]
    if mode == "multi":
        apps.append(tuple(app for app in ANCHOR_APPS if app != anchor_app)[cell - 1])
    source_tasks = [
        _source_task(
            task_id=f"cln-s0-src-{anchor_app}-{mode}-c{cell:02d}-{order:02d}",
            app=app,
            seed=920000 + app_index * 1000 + MODES.index(mode) * 100 + cell * 10 + order,
            cell=cell,
            code=code,
            mode=mode,
        )
        for order, app in enumerate(apps, start=1)
    ]
    ordered_components = [
        {
            "order": order,
            "app": source["app"],
            "task_id": source["id"],
            "semantic_steps": source["semantic_steps"] if mode == "single" else 1,
        }
        for order, source in enumerate(source_tasks, start=1)
    ]
    source_ids = [source["id"] for source in source_tasks]
    capability_set = {
        capability for source in source_tasks for capability in source["capabilities"]
    }
    capability_set.add("multi_step_state_change")
    if mode == "multi":
        capability_set.add("app_switch")
    instruction = source_tasks[0]["instruction"]
    if mode == "multi":
        instruction = (
            f'Complete both parts in order using handoff code “{code}”. '
            f'{source_tasks[0]["instruction"]} Then switch applications with Alt+Tab. '
            f'{source_tasks[1]["instruction"]}'
        )
    row: dict[str, Any] = {
        "id": record_id,
        "mode": mode,
        "bridge": f"{mode}_app",
        "anchor_app": anchor_app,
        "ordered_components": ordered_components,
        "eligibility": dict(RECORD_ELIGIBILITY),
        "instruction": instruction,
        "parameter_seed": record_seed,
        "difficulty": DIFFICULTY_BY_CELL[cell],
        "semantic_steps": source_tasks[0]["semantic_steps"] if mode == "single" else 2,
        "capabilities": sorted(capability_set),
        "reset": dict(RESET_CONTRACT),
        "verifier": {
            "module": "osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_oracle",
            "kind": "fresh_composed_private_state",
            "fresh_process": True,
            "machine_readable": True,
            "composition": "ordered_all_components",
            "component_source_task_ids": source_ids,
            "reset_reject": True,
            "near_miss_reject": True,
            "gold_pass": True,
        },
    }
    if mode == "single":
        row.update(
            source_task=source_tasks[0],
            source_task_payload_sha256=sha256_value(source_tasks[0]),
        )
    else:
        # (compiled ActionTurn payloads, emitted input events) per component.
        component_counts = {
            "writer": (1, 9),
            "calc": (1, 7),
            "files": (1, 5),
            "chrome": (2, 6),
            "vscode": (1, 9),
        }
        primitive_actions = 1 + sum(component_counts[source["app"]][0] for source in source_tasks)
        emitted_events = 4 + sum(component_counts[source["app"]][1] for source in source_tasks)
        row.update(
            source_tasks=source_tasks,
            source_task_payload_sha256s=[sha256_value(source) for source in source_tasks],
            program_budget={
                "primitive_actions": primitive_actions,
                "primitive_action_ceiling": 8,
                "emitted_events": emitted_events,
                "emitted_event_ceiling": 25,
                "visible_app_switch_included": True,
            },
        )
    return {**row, "record_sha256": sha256_value(row)}


def build() -> dict[str, Any]:
    tasks = [
        _record(anchor_app=app, mode=mode, cell=cell, app_index=app_index)
        for app_index, app in enumerate(ANCHOR_APPS)
        for mode in MODES
        for cell in range(1, CELLS_PER_ANCHOR_MODE + 1)
    ]
    payload = {
        "schema_version": 1,
        "suite": "cleanroom_natural_dev_stage0_v1",
        "split": "development",
        "stage": "stage0",
        "development_only": True,
        "task_count": len(tasks),
        "anchor_apps": list(ANCHOR_APPS),
        "modes": list(MODES),
        "cells_per_anchor_mode": CELLS_PER_ANCHOR_MODE,
        "development": dict(DEVELOPMENT),
        "source_policy": dict(SOURCE_POLICY),
        "tasks": tasks,
    }
    return {**payload, "manifest_payload_sha256": sha256_value(payload)}


def write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(build(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the sealed clean-room Stage0 inventory")
    parser.add_argument("--output", type=Path, default=STAGE0_INVENTORY_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    document = build()
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(document):
            raise SystemExit("Stage0 inventory is not generator-clean")
    else:
        write(args.output)
    print(json.dumps({"output": str(args.output), "task_count": len(document["tasks"]), "manifest_payload_sha256": document["manifest_payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
