from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.relative_factorial import evaluate
from experiments.relative_factorial.effects import EffectError, _load_cell
from experiments.relative_factorial.readiness import (
    _png_data_url,
    _probe_instruction,
    _probe_schema_ok,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(tmp_path: Path, arm: str) -> Path:
    root = tmp_path / f"model_{arm}"
    hf = root / "hf"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text(json.dumps({"architectures": ["MockVisionModel"]}))
    (hf / "model.safetensors").write_bytes(b"mock-weights")
    checkpoint = root / "orbax/000750"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    (root / "export_manifest.json").write_text(json.dumps({
        "artifact_type": "relative_factorial_hf_checkpoint",
        "schema_version": 1,
        "status": "complete",
        "arm": arm,
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "source_checkpoint": str(checkpoint),
        "step": 750,
        "lora_rank": 32,
        "lora_alpha": 32,
        "max_length": 4096,
        "hf_subdir": "hf",
    }))
    return hf


def _raw(grammar: str, action: str, coord: list[int]) -> str:
    if evaluate.GRAMMAR_LEVELS[grammar]["grammar_wrapper"] == "bare":
        return f"{coord[0]} {coord[1]} 0 ; +LMB -LMB"
    return (
        '<tool_call>{"name":"computer_use","arguments":'
        f'{{"action":"{action}","coordinate":{json.dumps(coord)}}}}}</tool_call>'
    )


def _rows(grammar: str, *, request_error: int | None = None) -> list[dict]:
    spec = evaluate.GRAMMAR_LEVELS[grammar]
    action = spec["expected_action"] if spec["grammar_wrapper"] == "tool" else "delta"
    rows = []
    for index in range(80):
        coord = [10 + index, 20 + index]
        error = index == request_error
        rows.append({
            "scene_id": (f"long_{index:04d}" if index < 40 else f"short_{index:04d}"),
            "kind": "long" if index < 40 else "short",
            "space": spec["space"],
            "ideal_coord": coord,
            "coord": None if error else coord,
            "parse_ok": not error,
            "in_box": not error,
            "endpoint_err_px": None if error else 0.0,
            "on_lattice": False,
            "raw_output": "<ERROR ConnectionError>" if error else _raw(grammar, action, coord),
            "grammar": grammar,
            "k": 0,
            "request_error": error,
            "action": None if error else action,
        })
    return rows


def _canonical_report(grammar: str, preamble: bool) -> dict:
    return {"meta": {
        "rung": 2,
        "model": "policy",
        "tag": f"relative_factorial/{grammar}/{'pre' if preamble else 'act'}",
        "state_cursor": False,
        "preamble": preamble,
        "sampling": {"temperature": 0.0, "top_p": 0.8, "top_k": 20},
        "n_scenes": 80,
        "k": 1,
        "box": 90,
        "n_long": 40,
        "n_short": 40,
        "seed": 0,
    }, "summary": {f"{grammar}/all": {"n": 80}}}


def _write_eval_artifact(tmp_path: Path, cell: str) -> Path:
    from experiments.relative_factorial.effects import CELLS

    rel, grammar_code, pre, grammar, arm, _space = CELLS[cell]
    del rel, grammar_code
    directory = tmp_path / cell
    directory.mkdir()
    model_dir = _model(directory, arm)
    provenance = evaluate._model_provenance(model_dir, grammar, pre == 1)
    rows, report = evaluate._validate_and_harden(
        _rows(grammar), _canonical_report(grammar, pre == 1), grammar=grammar,
        preamble=pre == 1, model="policy",
        tag=f"relative_factorial/{grammar}/{'pre' if pre == 1 else 'act'}",
        model_provenance=provenance,
    )
    rows_path = directory / "rows.jsonl"
    report_path = directory / "report.json"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report_path.write_text(json.dumps(report, sort_keys=True))
    spec = evaluate.GRAMMAR_LEVELS[grammar]
    manifest = {
        "artifact_type": "synthetic_factorial_eval",
        "schema_version": 2,
        "status": "complete",
        "grammar_name": grammar,
        "preamble": pre == 1,
        "relativity": spec["relativity"],
        "grammar_wrapper": spec["grammar_wrapper"],
        "expected_action": spec["expected_action"],
        "sampling": {"temperature": 0.0, "k": 1},
        "known_answer_selftest": {"passing": 80, "total": 80},
        "request_errors": {"count": 0, "total": 80},
        "row_contract": {
            "count": 80, "unique_scenes": 80, "long": 40, "short": 40,
            "k_values": [0],
        },
        "model_provenance": provenance,
        "report": "report.json",
        "report_sha256": _sha256(report_path),
        "rows": "rows.jsonl",
        "rows_sha256": _sha256(rows_path),
    }
    (directory / "eval_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return directory


def _fake_canonical_run(rows: list[dict], report: dict):
    def run(command, check):
        assert check is False
        work = Path(command[command.index("--out") + 1])
        (work / "rows.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (work / "report.json").write_text(json.dumps(report))
        (work / "scenes.jsonl").write_text("{}\n")
        return subprocess.CompletedProcess(command, 0)

    return run


@pytest.mark.parametrize("grammar", ["move_rel", "absolute_toolcall"])
def test_tool_schema_rejects_wrong_action_and_bare_json(grammar):
    expected = evaluate.GRAMMAR_LEVELS[grammar]["expected_action"]
    wrong = "left_click" if expected == "move_rel" else "move_rel"
    coord = [12, 34]
    row = _rows(grammar)[0]
    assert evaluate._schema_ok(row, grammar)
    wrong_row = copy.deepcopy(row)
    wrong_row["action"] = wrong
    wrong_row["raw_output"] = _raw(grammar, wrong, coord)
    wrong_row["coord"] = coord
    assert not evaluate._schema_ok(wrong_row, grammar)
    bare_json = copy.deepcopy(row)
    bare_json["raw_output"] = json.dumps({
        "name": "computer_use",
        "arguments": {"action": expected, "coordinate": bare_json["coord"]},
    })
    assert not evaluate._schema_ok(bare_json, grammar)


@pytest.mark.parametrize("grammar", ["deltatype_raw", "absolute_raw"])
def test_bare_schema_requires_bare_delta_line(grammar):
    row = _rows(grammar)[0]
    assert evaluate._schema_ok(row, grammar)
    row["raw_output"] += " | tool_calls=[]"
    assert not evaluate._schema_ok(row, grammar)


@pytest.mark.parametrize("bad_line", ["12 34", "12 34 0", "12 34 1 ; +LMB -LMB"])
def test_bare_schema_requires_exact_move_and_click_action(bad_line):
    row = _rows("deltatype_raw")[0]
    row["raw_output"] = bad_line
    assert not evaluate._schema_ok(row, "deltatype_raw")


def test_request_error_removes_stale_trusted_outputs(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    for name in evaluate._TRUSTED_ARTIFACTS:
        (out / name).write_text("stale")
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "rung2_scene.py").write_text("# mocked by test\n")
    model = _model(tmp_path, "reltool_act")
    monkeypatch.setattr(
        evaluate.subprocess, "run",
        _fake_canonical_run(_rows("move_rel", request_error=7), _canonical_report("move_rel", False)),
    )
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--audit-dir", str(audit), "--out", str(out),
        "--base-url", "http://mock/v1", "--model-dir", str(model),
        "--grammar", "move_rel",
    ])
    assert evaluate.main() == 4
    assert all(not (out / name).exists() for name in evaluate._TRUSTED_ARTIFACTS)


def test_success_publishes_hashed_manifest_last(tmp_path, monkeypatch):
    out = tmp_path / "out"
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "rung2_scene.py").write_text("# mocked by test\n")
    model = _model(tmp_path, "abstool_pre")
    monkeypatch.setattr(
        evaluate.subprocess, "run",
        _fake_canonical_run(
            _rows("absolute_toolcall"), _canonical_report("absolute_toolcall", True)
        ),
    )
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--audit-dir", str(audit), "--out", str(out),
        "--base-url", "http://mock/v1", "--model-dir", str(model),
        "--grammar", "absolute_toolcall", "--preamble",
    ])
    assert evaluate.main() == 0
    manifest = json.loads((out / "eval_manifest.json").read_text())
    assert manifest["model_provenance"]["arm"] == "abstool_pre"
    assert manifest["report_sha256"] == _sha256(out / "report.json")
    assert manifest["rows_sha256"] == _sha256(out / "rows.jsonl")
    assert not list(out.glob(".eval_work_*"))


def test_effect_loader_accepts_only_complete_hashed_artifact(tmp_path):
    directory = _write_eval_artifact(tmp_path, "rel_tool_act")
    value, provenance = _load_cell("rel_tool_act", directory, "in_box")
    assert value == 1.0
    assert provenance["model_provenance"]["arm"] == "reltool_act"


def test_effect_loader_rejects_partial_rows_even_with_updated_hash(tmp_path):
    directory = _write_eval_artifact(tmp_path, "abs_bare_act")
    rows_path = directory / "rows.jsonl"
    rows_path.write_text("\n".join(rows_path.read_text().splitlines()[:-1]) + "\n")
    manifest_path = directory / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows_sha256"] = _sha256(rows_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EffectError, match="exactly 80"):
        _load_cell("abs_bare_act", directory, "in_box")


def test_effect_loader_rejects_nonfinite_metric_and_wrong_model(tmp_path):
    directory = _write_eval_artifact(tmp_path, "rel_bare_pre")
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    report["summary"]["deltatype_raw/all"]["in_box"] = float("nan")
    report_path.write_text(json.dumps(report))
    manifest_path = directory / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["report_sha256"] = _sha256(report_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EffectError, match="finite and bounded"):
        _load_cell("rel_bare_pre", directory, "in_box")

    directory = _write_eval_artifact(tmp_path, "abs_tool_pre")
    manifest_path = directory / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_provenance"]["arm"] = "wrong_arm"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EffectError, match="model provenance"):
        _load_cell("abs_tool_pre", directory, "in_box")


def test_readiness_probe_requires_vision_cell_schema():
    assert _png_data_url().startswith("data:image/png;base64,iVBOR")
    instruction = _probe_instruction("move_rel")
    payload = instruction.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
    assert json.loads(payload)["arguments"]["action"] == "move_rel"
    assert _probe_schema_ok({
        "content": '<tool_call>{"name":"computer_use","arguments":'
                   '{"action":"move_rel","coordinate":[0,0]}}</tool_call>'
    }, "move_rel")
    assert _probe_schema_ok({
        "content": None,
        "tool_calls": [{"function": {
            "name": "computer_use",
            "arguments": json.dumps({"action": "left_click", "coordinate": [0, 0]}),
        }}],
    }, "absolute_toolcall")
    assert _probe_schema_ok({"content": "0 0 0 ; +LMB -LMB"}, "absolute_raw")
    assert not _probe_schema_ok({"content": "ready"}, "absolute_raw")
