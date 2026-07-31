from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


FINALIZE = Path(__file__).parents[1] / "finalize.py"


def _fixture(tmp_path: Path, *, val_arm: str = "prose_keep") -> list[str]:
    checkpoint_root = tmp_path / "pb_prose_keep_r32"
    checkpoint = checkpoint_root / "000900"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("complete")
    (checkpoint_root / "lora_metadata.json").write_text("{}")

    model = tmp_path / "pb_prose_keep_r32_hf"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"architectures": ["Qwen3VLForConditionalGeneration"]}))
    (model / "model.safetensors").write_bytes(b"weights")

    val = tmp_path / "phaseb" / val_arm / "_normalized" / "val" / "chat.jsonl"
    val.parent.mkdir(parents=True)
    val.write_text("{}\n")

    out = tmp_path / "out"
    out.mkdir()
    rows = [
        {"request_error": False, "teacher_action": "left_click"}
        for _ in range(233)
    ]
    (out / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "report.json").write_text(json.dumps({
        "meta": {"valid": True, "val_chat": str(val), "n": 233},
        "summary": {
            "n_rows": 233,
            "n_coord_records": 178,
            "n_request_errors": 0,
            "request_error_rate": 0.0,
        },
    }))
    training_log = tmp_path / "train.log"
    training_log.write_text("complete")
    training_script = tmp_path / "train.sh"
    training_script.write_text("true")
    evaluator = tmp_path / "eval.py"
    evaluator.write_text("pass")

    return [
        sys.executable, str(FINALIZE),
        "--arm", "prose_keep",
        "--source-job-id", "135312",
        "--source-checkpoint", str(checkpoint),
        "--source-checkpoint-root", str(checkpoint_root),
        "--model-dir", str(model),
        "--val-chat", str(val),
        "--training-log", str(training_log),
        "--training-script", str(training_script),
        "--evaluator", str(evaluator),
        "--out", str(out),
    ]


def test_finalize_writes_manifest_last(tmp_path: Path) -> None:
    subprocess.run(_fixture(tmp_path), check=True)
    manifest = json.loads((tmp_path / "out" / "eval_manifest.json").read_text())
    assert manifest["valid"] is True
    assert manifest["own_val_contract"]["cross_arm_prompt_reuse"] is False
    assert manifest["evaluation"]["request_errors"] == 0
    assert manifest["model"]["weights"][0]["sha256"]


def test_finalize_rejects_cross_arm_val(tmp_path: Path) -> None:
    proc = subprocess.run(_fixture(tmp_path, val_arm="prose_strip"), text=True, capture_output=True)
    assert proc.returncode != 0
    assert "cross-arm validation prompt reuse" in proc.stderr
    assert not (tmp_path / "out" / "eval_manifest.json").exists()
