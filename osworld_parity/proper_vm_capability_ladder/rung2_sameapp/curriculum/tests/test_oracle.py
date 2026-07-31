from __future__ import annotations

import json
import importlib
import os
import zipfile
from pathlib import Path

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_materialized_curriculum,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle import (
    evaluate_in_fresh_process,
    evaluate_state,
    reset_signature,
)


def _tasks():
    return [
        task
        for manifest in load_materialized_curriculum().values()
        for task in manifest.tasks
    ]


def _artifact(task, root: Path, mode: str) -> Path:
    root.mkdir(parents=True)
    if task.app == "writer":
        text = (
            task.params["initial_text"]
            if mode == "reset"
            else task.near_miss["text"]
            if mode == "near"
            else task.expected["text"]
        )
        bold = mode == "gold"
        content = (
            '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
            'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0">'
            + ('<style fo:font-weight="bold"/>' if bold else "")
            + f"<text:p>{text}</text:p></office:document-content>"
        )
        with zipfile.ZipFile(root / task.params["file_name"], "w") as archive:
            archive.writestr("content.xml", content)
    elif task.app == "calc":
        value = (
            task.params["initial_value"]
            if mode == "reset"
            else task.near_miss["display_value"]
            if mode == "near"
            else task.expected["display_value"]
        )
        formula = (
            None
            if mode == "reset"
            else task.near_miss["formula"]
            if mode == "near"
            else task.expected["formula"]
        )
        formula_attr = f' table:formula="{formula}"' if formula else ""
        content = (
            '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">'
            '<table:table><table:table-row table:number-rows-repeated="5">'
            f'<table:table-cell table:number-columns-repeated="3"{formula_attr} office:value="{value}"/>'
            "</table:table-row></table:table></office:document-content>"
        )
        with zipfile.ZipFile(root / task.params["file_name"], "w") as archive:
            archive.writestr("content.xml", content)
    elif task.app == "files":
        (root / task.params["destination_name"]).mkdir()
        (root / task.params["decoy_name"]).mkdir()
        if mode == "reset":
            path = root / task.params["source_name"]
        elif mode == "near":
            path = root / task.near_miss["destination"] / task.near_miss["final_name"]
        else:
            path = root / task.expected["destination"] / task.expected["final_name"]
        path.write_text(task.params["content"], encoding="utf-8")
    elif task.app == "chrome":
        if mode == "reset":
            section, enabled, scroll_y = "root", False, task.params["initial_scroll_y"]
        elif mode == "near":
            section, enabled = task.near_miss["section"], task.near_miss["setting_enabled"]
            delta = task.params["minimum_scroll_delta"]
            scroll_y = task.params["initial_scroll_y"] + (
                -delta if task.params["scroll_direction"] == "up" else delta
            )
        else:
            section, enabled = task.expected["section"], task.expected["setting_enabled"]
            delta = task.params["minimum_scroll_delta"]
            scroll_y = task.params["initial_scroll_y"] + (
                -delta if task.params["scroll_direction"] == "up" else delta
            )
        (root / "state.json").write_text(
            json.dumps(
                {
                    "ready": True,
                    "section": section,
                    "scroll_y": scroll_y,
                    "setting_enabled": enabled,
                }
            ),
            encoding="utf-8",
        )
    else:
        text = (
            task.params["initial_text"]
            if mode == "reset"
            else task.near_miss["text"]
            if mode == "near"
            else task.expected["text"]
        )
        (root / task.params["file_name"]).write_text(text, encoding="utf-8")
    return root


def test_every_fixture_uses_independent_extraction_and_fresh_oracle(tmp_path: Path) -> None:
    for task in _tasks():
        extractor = getattr(
            importlib.import_module(task.verifier["state_extractor_module"]),
            task.verifier["state_extractor_entrypoint"],
        )
        states = {
            mode: extractor(task, _artifact(task, tmp_path / task.task_id / mode, mode))
            for mode in ("reset", "near", "gold")
        }
        assert reset_signature(task, states["reset"]) == reset_signature(
            task,
            extractor(task, _artifact(task, tmp_path / task.task_id / "reset2", "reset")),
        )
        assert evaluate_state(task, states["reset"]).MOUSE_SOLVED is False
        assert evaluate_state(task, states["near"]).MOUSE_SOLVED is False
        gold = evaluate_in_fresh_process(task, states["gold"])
        assert gold.oracle_pid != os.getpid()
        assert gold.oracle_status == "ok"
        assert gold.MOUSE_SOLVED is True
        assert gold.semantic_state_sha256


def test_final_oracle_rejects_held_input_from_extracted_gold(tmp_path: Path) -> None:
    task = _tasks()[0]
    extractor = getattr(
        importlib.import_module(task.verifier["state_extractor_module"]),
        task.verifier["state_extractor_entrypoint"],
    )
    state = extractor(
        task, _artifact(task, tmp_path / "gold", "gold"), held_inputs=("left",)
    )
    result = evaluate_state(task, state)
    assert result.oracle_status == "ok"
    assert result.MOUSE_SOLVED is False
    assert "not fully released" in result.reason
