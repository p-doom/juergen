from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "postexport_orbax_cleanup.py"
SPEC = importlib.util.spec_from_file_location("postexport_orbax_cleanup", MODULE_PATH)
assert SPEC and SPEC.loader
cleanup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cleanup
SPEC.loader.exec_module(cleanup)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_safetensors(path: Path) -> None:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1.0, 2.0))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint_root = tmp_path / "checkpoints"
    source = checkpoint_root / "legacy_stream" / "001000"
    export = checkpoint_root / "hf_export" / "001000"
    run_id = "run_0123456789abcdef0123456789abcdef"
    producer_id = "run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    source_id = "artifact_source"
    export_id = "artifact_export"
    source.mkdir(parents=True)
    (source / "train_state").mkdir()
    (source / "input_iter").mkdir()
    (source / "_CHECKPOINT_METADATA").write_text("orbax\n", encoding="utf-8")
    _write_json(
        source / ".meta.json",
        {
            "id": source_id,
            "alias": "legacy_stream/001000",
            "producer_run_id": producer_id,
            "metadata": {"step": 1000, "marker": "_CHECKPOINT_METADATA"},
        },
    )
    export.mkdir(parents=True)
    _write_json(export / "config.json", {"model_type": "test"})
    _write_safetensors(export / "model.safetensors")
    _write_json(
        export / ".meta.json",
        {
            "id": export_id,
            "alias": "hf_export/001000",
            "producer_run_id": run_id,
            "metadata": {"step": 1000, "marker": "model.safetensors"},
        },
    )
    run_root = tmp_path / "runs"
    context_path = run_root / run_id / ".lab" / "context.json"
    _write_json(
        context_path,
        {
            "run_id": run_id,
            "recipe_name": "bc_export_hf_per_checkpoint_test",
            "inputs": [
                {
                    "artifact_id": source_id,
                    "resolved_path": str(source),
                    "role": "checkpoint",
                }
            ],
            "outputs": {
                "hf_checkpoint": {
                    "path": str(export.parent),
                    "marker": "model.safetensors",
                    "role": "hf_checkpoint",
                }
            },
        },
    )
    entry = {
        "source_path": str(source),
        "source_artifact_id": source_id,
        "source_producer_run_id": producer_id,
        "source_producer_status": "succeeded",
        "source_step": 1000,
        "source_meta_sha256": _sha(source / ".meta.json"),
        "source_checkpoint_metadata_sha256": _sha(source / "_CHECKPOINT_METADATA"),
        "expected_allocated_bytes": cleanup._du_bytes(source, apparent=False),
        "expected_logical_bytes": cleanup._du_bytes(source, apparent=True),
        "export_path": str(export),
        "export_artifact_id": export_id,
        "export_run_id": run_id,
        "export_run_status": "succeeded",
        "export_context_path": str(context_path),
        "export_context_sha256": _sha(context_path),
        "export_meta_sha256": _sha(export / ".meta.json"),
        "export_config_sha256": _sha(export / "config.json"),
        "export_model_size": (export / "model.safetensors").stat().st_size,
    }
    allowlist = tmp_path / "allowlist.json"
    _write_json(
        allowlist,
        {
            "schema_version": 1,
            "authorization": "test",
            "checkpoint_root": str(checkpoint_root),
            "run_root": str(run_root),
            "target_count": 1,
            "expected_allocated_bytes": entry["expected_allocated_bytes"],
            "expected_logical_bytes": entry["expected_logical_bytes"],
            "entries": [entry],
        },
    )
    return allowlist, source, export, checkpoint_root


def test_cleanup_removes_only_source_and_retains_export(tmp_path, monkeypatch):
    allowlist, source, export, _checkpoint_root = _fixture(tmp_path)
    monkeypatch.setattr(cleanup, "_active_job_snapshot", lambda **_kwargs: [])
    output = tmp_path / "result" / "cleanup.json"
    result = cleanup.cleanup(
        allowlist_path=allowlist,
        expected_allowlist_sha256=_sha(allowlist),
        output_path=output,
    )
    assert result["status"] == "complete"
    assert not source.exists()
    assert (export / "model.safetensors").is_file()
    assert json.loads(output.read_text())["all_hf_exports_retained"] is True


def test_cleanup_rejects_changed_source_before_any_deletion(tmp_path, monkeypatch):
    allowlist, source, export, _checkpoint_root = _fixture(tmp_path)
    expected_hash = _sha(allowlist)
    (source / "_CHECKPOINT_METADATA").write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(cleanup, "_active_job_snapshot", lambda **_kwargs: [])
    with pytest.raises(cleanup.CleanupError, match="hash changed"):
        cleanup.cleanup(
            allowlist_path=allowlist,
            expected_allowlist_sha256=expected_hash,
            output_path=tmp_path / "cleanup.json",
        )
    assert source.is_dir()
    assert export.is_dir()


def test_safetensors_rejects_uncovered_payload(tmp_path):
    model = tmp_path / "model.safetensors"
    _write_safetensors(model)
    model.write_bytes(model.read_bytes() + b"junk")
    with pytest.raises(cleanup.CleanupError, match="cover"):
        cleanup._validate_safetensors(model)
