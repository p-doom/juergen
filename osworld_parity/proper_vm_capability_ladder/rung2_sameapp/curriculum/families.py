"""Deterministic same-application task-family materialization."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .schema import EXCLUSIONS, SemanticTask, seal_task


FAMILY_IDS = (
    "writer_replace_format_save_v1",
    "calc_formula_confirm_save_v1",
    "files_drag_rename_v1",
    "chrome_navigate_signed_scroll_toggle_v1",
    "vscode_unicode_replace_save_v1",
)

ORACLE_MODULE = (
    "osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle"
)
RESET_CONTRACT = {
    "reset_reject": True,
    "near_miss_reject": True,
    "gold_pass": True,
    "reproducible_reset": True,
    "fresh_process_final_oracle": True,
    "zero_held_inputs": True,
}
SNAPSHOT = {
    "id": "osworld_ready",
    "reset_strategy": "restore_snapshot_then_seeded_setup",
    "fresh_process_per_episode": True,
}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _layout(seed: int) -> tuple[list[int], dict[str, list[int]]]:
    rng = random.Random(seed ^ 0x52454C)
    cursor = [rng.randint(80, 260), rng.randint(90, 260)]
    geometry = {
        "editor": [rng.randint(350, 650), rng.randint(260, 500)],
        "cell": [rng.randint(360, 610), rng.randint(280, 510)],
        "name_box": [rng.randint(90, 210), rng.randint(100, 155)],
        "source": [rng.randint(150, 280), rng.randint(260, 430)],
        "destination": [rng.randint(650, 820), rng.randint(250, 410)],
        "decoy": [rng.randint(650, 820), rng.randint(480, 590)],
        "moved": [rng.randint(650, 820), rng.randint(250, 410)],
        "nav": [rng.randint(110, 240), rng.randint(180, 310)],
        "decoy_nav": [rng.randint(110, 240), rng.randint(390, 500)],
        "scroll_surface": [rng.randint(430, 670), rng.randint(320, 520)],
        "toggle": [rng.randint(740, 870), rng.randint(480, 600)],
        "decoy_toggle": [rng.randint(740, 870), rng.randint(220, 330)],
    }
    return cursor, geometry


def _step(
    step_id: int,
    intent: str,
    target_ref: str,
    capabilities: list[str],
    **arguments: Any,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "intent": intent,
        "target_ref": target_ref,
        "capabilities": capabilities,
        "arguments": arguments,
    }


def _history(
    initial: list[int], entries: list[tuple[str, list[int]]]
) -> list[dict[str, Any]]:
    before = list(initial)
    result: list[dict[str, Any]] = []
    for index, (target_ref, after) in enumerate(entries, 1):
        result.append(
            {
                "prefix_length": index,
                "step_id": index,
                "target_ref": target_ref,
                "cursor_before": list(before),
                "cursor_after": list(after),
            }
        )
        before = list(after)
    return result


def _common(
    *,
    task_id: str,
    family_id: str,
    app: str,
    split: str,
    seed: int,
    instruction: str,
    steps: list[dict[str, Any]],
    horizon: int,
    assets: list[dict[str, Any]],
    params: dict[str, Any],
    expected: dict[str, Any],
    near_miss: dict[str, Any],
    verifier_kind: str,
    initial_cursor: list[int],
    cursor_history: list[dict[str, Any]],
    capabilities: list[str],
    scroll_signs: list[str],
    edge_cases: list[str],
    thin_cases: list[str],
    transport: dict[str, Any],
) -> SemanticTask:
    return seal_task(
        {
            "task_id": task_id,
            "family_id": family_id,
            "app": app,
            "split": split,
            "parameter_seed": seed,
            "instruction": instruction,
            "natural_multistep": True,
            "semantic_steps": steps,
            "semantic_step_count": len(steps),
            "max_action_turns": horizon,
            "snapshot": SNAPSHOT,
            "assets": assets,
            "params": params,
            "expected": expected,
            "near_miss": near_miss,
            "verifier": {
                "kind": verifier_kind,
                "module": ORACLE_MODULE,
                "fresh_process": True,
                "policy_visible": False,
            },
            "initial_cursor": initial_cursor,
            "gold_cursor_history": cursor_history,
            "coverage": {
                "primary_capabilities": capabilities,
                "signed_vertical_scroll": scroll_signs,
                "edge_cases": edge_cases,
                "thin_cases": thin_cases,
            },
            "exclusions": list(EXCLUSIONS),
            "transport_requirements": transport,
            "reset_contract": RESET_CONTRACT,
        }
    )


def build_task(family_id: str, split: str, seed: int) -> SemanticTask:
    if family_id not in FAMILY_IDS:
        raise ValueError(f"unknown curriculum family: {family_id!r}")
    if split not in {"train", "development"}:
        raise ValueError("only train/development tasks can be materialized")
    cursor, geometry = _layout(seed)
    suffix = f"{split}-{seed}"

    if family_id == "writer_replace_format_save_v1":
        replacement = f"Quarterly orbit note {seed % 997:03d}"
        initial = f"replace writer draft {seed}"
        return _common(
            task_id=f"r2c-writer-{suffix}", family_id=family_id, app="writer",
            split=split, seed=seed,
            instruction=(f"In Writer, replace the note with exactly ‘{replacement}’, "
                         "make it bold, and save the document."),
            steps=[
                _step(1, "replace_text", "writer.editor", ["click", "hotkey", "coalesced_type"], text_param="replacement_text", selection_hotkey="ControlLeft+KeyA"),
                _step(2, "apply_bold", "writer.editor", ["hotkey"], hotkey="ControlLeft+KeyB"),
                _step(3, "save_document", "writer.editor", ["hotkey"], hotkey="ControlLeft+KeyS"),
            ],
            horizon=4,
            assets=[{"asset_id": f"writer-{seed}.odt", "kind": "odf_text_document", "generator": "existing_rung2_writer_fixture", "seed": seed, "content_sha256": _sha_text(initial)}],
            params={"file_name": f"writer-{seed}.odt", "initial_text": initial, "replacement_text": replacement, "style": "bold", "geometry": geometry},
            expected={"text": replacement, "bold": True, "saved": True},
            near_miss={"text": replacement[:-1] + "x", "bold": False},
            verifier_kind="writer_odf_state",
            initial_cursor=cursor,
            cursor_history=_history(cursor, [("writer.editor", geometry["editor"]), ("writer.editor", geometry["editor"]), ("writer.editor", geometry["editor"])]),
            capabilities=["click", "coalesced_type", "hotkey"], scroll_signs=[],
            edge_cases=["ctrl_s"], thin_cases=[],
            transport={"coalesced_type": True, "hotkeys": ["ControlLeft+KeyA", "ControlLeft+KeyB", "ControlLeft+KeyS"], "mouse_button_hold": False},
        )

    if family_id == "calc_formula_confirm_save_v1":
        left, right = seed % 19 + 3, seed % 23 + 5
        formula, display = f"={left}+{right}", str(left + right)
        initial = f"seed spreadsheet {seed}"
        return _common(
            task_id=f"r2c-calc-{suffix}", family_id=family_id, app="calc",
            split=split, seed=seed,
            instruction=f"In Calc, enter {formula} in cell C5, confirm it, and save the sheet.",
            steps=[
                _step(1, "select_cell", "calc.name_box", ["click", "hotkey", "coalesced_type"], cell_param="cell"),
                _step(2, "enter_formula", "calc.cell", ["coalesced_type"], formula_param="formula"),
                _step(3, "confirm_formula", "calc.cell", ["hotkey"], hotkey="Return"),
                _step(4, "save_sheet", "calc.cell", ["hotkey"], hotkey="ControlLeft+KeyS"),
            ],
            horizon=6,
            assets=[{"asset_id": f"calc-{seed}.ods", "kind": "odf_spreadsheet", "generator": "existing_rung2_calc_fixture", "seed": seed, "content_sha256": _sha_text(initial)}],
            params={"file_name": f"calc-{seed}.ods", "cell": "C5", "initial_value": initial, "formula": formula, "geometry": geometry},
            expected={"formula": "of:" + formula, "display_value": display, "saved": True},
            near_miss={"formula": f"of:={left}+{right + 1}", "display_value": str(left + right + 1)},
            verifier_kind="calc_odf_state",
            initial_cursor=cursor,
            cursor_history=_history(cursor, [("calc.name_box", geometry["name_box"]), ("calc.cell", geometry["name_box"]), ("calc.cell", geometry["name_box"]), ("calc.cell", geometry["name_box"])]),
            capabilities=["click", "coalesced_type", "hotkey"], scroll_signs=[],
            edge_cases=["ctrl_s"], thin_cases=[],
            transport={"coalesced_type": True, "hotkeys": ["ControlLeft+KeyA", "Return", "ControlLeft+KeyS"], "mouse_button_hold": False},
        )

    if family_id == "files_drag_rename_v1":
        content = f"seeded parcel {seed}\n"
        source, destination = f"parcel-{seed}.txt", f"Delivered-{seed}"
        final = f"receipt-{seed}.txt"
        return _common(
            task_id=f"r2c-files-{suffix}", family_id=family_id, app="files",
            split=split, seed=seed,
            instruction=f"In Files, drag {source} into {destination}, then rename it {final}.",
            steps=[
                _step(1, "select_source", "files.source", ["click"], source_param="source_name"),
                _step(2, "drag_into_folder", "files.destination", ["drag"], source_ref="files.source", destination_param="destination_name"),
                _step(3, "rename_moved_file", "files.moved", ["click", "hotkey", "coalesced_type"], name_param="final_name"),
            ],
            horizon=8,
            assets=[{"asset_id": source, "kind": "plain_text_file", "generator": "existing_rung2_files_fixture", "seed": seed, "content_sha256": _sha_text(content)}],
            params={"source_name": source, "destination_name": destination, "decoy_name": f"Archive-{seed}", "final_name": final, "content": content, "geometry": geometry},
            expected={"destination": destination, "final_name": final},
            near_miss={"destination": f"Archive-{seed}", "final_name": f"receipt-{seed}.tmp"},
            verifier_kind="filesystem_move_and_content",
            initial_cursor=cursor,
            cursor_history=_history(cursor, [("files.source", geometry["source"]), ("files.destination", geometry["destination"]), ("files.moved", geometry["moved"])]),
            capabilities=["click", "drag", "coalesced_type", "hotkey"], scroll_signs=[],
            edge_cases=["file_drag"], thin_cases=["file_drag_single_family"],
            transport={"coalesced_type": True, "hotkeys": ["Return", "F2", "ControlLeft+KeyA"], "mouse_button_hold": True},
        )

    if family_id == "chrome_navigate_signed_scroll_toggle_v1":
        direction = "down" if split == "train" else "up"
        clicks = -6 if direction == "down" else 6
        initial_y = 0 if direction == "down" else 720
        fixture_source = f"chrome local fixture {seed} {direction}"
        return _common(
            task_id=f"r2c-chrome-{suffix}", family_id=family_id, app="chrome",
            split=split, seed=seed,
            instruction=f"In the open Chrome fixture, open Privacy controls, scroll {direction}, and enable Sync previews.",
            steps=[
                _step(1, "navigate_section", "chrome.nav", ["click"], section="privacy"),
                _step(2, "scroll_settings", "chrome.scroll_surface", ["signed_vertical_scroll"], direction=direction, clicks_param="scroll_clicks"),
                _step(3, "toggle_setting", "chrome.toggle", ["click"], setting="Sync previews"),
            ],
            horizon=6,
            assets=[{"asset_id": f"chrome-{seed}.html", "kind": "local_html_settings_fixture", "generator": "existing_rung2_chrome_fixture", "seed": seed, "content_sha256": _sha_text(fixture_source)}],
            params={"port": 18000 + seed % 1000, "section": "privacy", "setting": "Sync previews", "initial_scroll_y": initial_y, "minimum_scroll_delta": 300, "scroll_direction": direction, "scroll_clicks": clicks, "geometry": geometry},
            expected={"section": "privacy", "setting_enabled": True},
            near_miss={"section": "appearance", "setting_enabled": False},
            verifier_kind="local_chrome_state",
            initial_cursor=cursor,
            cursor_history=_history(cursor, [("chrome.nav", geometry["nav"]), ("chrome.scroll_surface", geometry["scroll_surface"]), ("chrome.toggle", geometry["toggle"])]),
            capabilities=["click", "signed_vertical_scroll"], scroll_signs=[direction],
            edge_cases=[], thin_cases=[f"signed_vertical_scroll_{direction}_single_family"],
            transport={"coalesced_type": False, "hotkeys": [], "mouse_button_hold": False, "signed_scroll_clicks": clicks},
        )

    replacement = f"Zürich μ-{seed} — café"
    initial = f"replace vscode draft {seed}\n"
    return _common(
        task_id=f"r2c-vscode-{suffix}", family_id=family_id, app="vscode",
        split=split, seed=seed,
        instruction=f"In VS Code, replace the file contents with exactly ‘{replacement}’ and save the file.",
        steps=[
            _step(1, "focus_editor", "vscode.editor", ["click"]),
            _step(2, "replace_file_text", "vscode.editor", ["hotkey", "coalesced_type"], text_param="replacement_text", selection_hotkey="ControlLeft+KeyA"),
            _step(3, "save_file", "vscode.editor", ["hotkey"], hotkey="ControlLeft+KeyS"),
        ],
        horizon=5,
        assets=[{"asset_id": f"vscode-{seed}.txt", "kind": "utf8_text_file", "generator": "existing_rung1b_vscode_fixture", "seed": seed, "content_sha256": _sha_text(initial)}],
        params={"file_name": f"vscode-{seed}.txt", "initial_text": initial, "replacement_text": replacement, "geometry": geometry},
        expected={"text": replacement, "saved": True},
        near_miss={"text": replacement.replace("μ", "u"), "saved": True},
        verifier_kind="vscode_utf8_file_state",
        initial_cursor=cursor,
        cursor_history=_history(cursor, [("vscode.editor", geometry["editor"]), ("vscode.editor", geometry["editor"]), ("vscode.editor", geometry["editor"])]),
        capabilities=["click", "coalesced_type", "hotkey"], scroll_signs=[],
        edge_cases=["unicode", "ctrl_s"], thin_cases=["unicode_coalesced_type_single_family"],
        transport={"coalesced_type": True, "unicode_safe": True, "hotkeys": ["ControlLeft+KeyA", "ControlLeft+KeyS"], "mouse_button_hold": False},
    )
