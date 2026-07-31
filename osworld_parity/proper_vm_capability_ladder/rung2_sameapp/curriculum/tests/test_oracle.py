from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_materialized_curriculum,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle import (
    evaluate_state,
    verify_fixture_contract,
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
        roots = {
            mode: _artifact(
                task,
                tmp_path / task.task_id / mode,
                "reset" if mode == "reset_repeat" else mode,
            )
            for mode in ("reset", "reset_repeat", "near", "gold")
        }
        result = verify_fixture_contract(task, artifact_roots=roots)
        assert result == {
            "task_id": task.task_id,
            "fixture_sha256": task.fixture_sha256,
            "reset_rejected": True,
            "near_miss_rejected": True,
            "gold_passed": True,
            "reset_reproducible": True,
            "fresh_process_final_oracle": True,
            "zero_held_inputs": True,
        }


def test_fixture_contract_propagates_extraction_failure(tmp_path: Path) -> None:
    task = next(task for task in _tasks() if task.app == "vscode")
    roots = {
        mode: _artifact(
            task,
            tmp_path / mode,
            "reset" if mode == "reset_repeat" else mode,
        )
        for mode in ("reset", "reset_repeat", "near", "gold")
    }
    (roots["gold"] / task.params["file_name"]).write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        verify_fixture_contract(task, artifact_roots=roots)


@pytest.mark.parametrize(
    ("error_call", "failed_field"),
    ((1, "reset_rejected"), (2, "near_miss_rejected"), (3, "gold_passed")),
)
def test_fixture_contract_fails_closed_on_any_oracle_error(
    tmp_path: Path, monkeypatch, error_call: int, failed_field: str
) -> None:
    from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum import oracle

    task = next(task for task in _tasks() if task.app == "vscode")
    roots = {
        mode: _artifact(
            task,
            tmp_path / mode,
            "reset" if mode == "reset_repeat" else mode,
        )
        for mode in ("reset", "reset_repeat", "near", "gold")
    }
    real = oracle.evaluate_in_fresh_process
    calls = 0

    def injected(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real(*args, **kwargs)
        return (
            replace(
                result,
                oracle_status="error",
                MOUSE_SOLVED=False,
                reason="injected oracle failure",
            )
            if calls == error_call
            else result
        )

    monkeypatch.setattr(oracle, "evaluate_in_fresh_process", injected)
    contract = oracle.verify_fixture_contract(task, artifact_roots=roots)
    assert contract[failed_field] is False


def test_final_oracle_rejects_held_input_from_extracted_gold(tmp_path: Path) -> None:
    task = _tasks()[0]
    import importlib

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


def test_vscode_bridge_supplies_merged_harness_gate_fields(monkeypatch) -> None:
    from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum import oracle

    @dataclass(frozen=True)
    class MergedFixture:
        id: str
        template: str
        split: str
        parameter_seed: int
        horizon: int
        instruction: str
        params: dict
        expected: dict
        near_miss: dict
        fixture_sha256: str
        gate_role: str
        coverage_label: str

    monkeypatch.setattr(oracle, "RealAppFixture", MergedFixture)
    task = next(task for task in _tasks() if task.app == "vscode")
    fixture = oracle.as_vscode_fixture(task)
    assert fixture.gate_role == "capability_probe"
    assert fixture.coverage_label == "unicode_coalesced_typing_probe"
