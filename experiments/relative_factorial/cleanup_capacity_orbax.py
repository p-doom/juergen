#!/usr/bin/env python3
"""Delete validated capacity-run Orbax trees after their registered evals."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from .capacity import ARMS, CELL_NAMES
    from .effects import EffectError, _load_cell
    from .uncertainty import _atomic_write
except ImportError:  # Direct execution by a labctl recipe.
    from capacity import ARMS, CELL_NAMES
    from effects import EffectError, _load_cell
    from uncertainty import _atomic_write


RANKS = (64, 256)


class CleanupError(RuntimeError):
    pass


def _logical_bytes(root: Path) -> int:
    total = 0
    for directory, _subdirectories, files in os.walk(root):
        for name in files:
            total += (Path(directory) / name).stat().st_size
    return total


def _context_inputs(path: Path) -> dict[str, dict[str, Any]]:
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanupError(f"cannot read labctl context {path}: {exc}") from exc
    inputs = {item.get("role"): item for item in context.get("inputs", [])}
    expected = {
        f"{kind}_r{rank}_{arm}"
        for rank in RANKS
        for arm in ARMS
        for kind in ("model", "eval")
    }
    if set(inputs) != expected:
        raise CleanupError(
            f"cleanup requires exactly the 16 registered model/eval inputs; "
            f"missing={sorted(expected - set(inputs))}, extra={sorted(set(inputs) - expected)}"
        )
    for role, item in inputs.items():
        if not item.get("artifact_id") or not item.get("resolved_path"):
            raise CleanupError(f"{role}: input is not a registered resolved artifact")
    return inputs


def cleanup(*, labctl_context: Path) -> dict[str, Any]:
    inputs = _context_inputs(labctl_context)
    validated = []
    targets = []
    for rank in RANKS:
        for arm in ARMS:
            model_item = inputs[f"model_r{rank}_{arm}"]
            eval_item = inputs[f"eval_r{rank}_{arm}"]
            model = Path(model_item["resolved_path"]).resolve()
            evaluation = Path(eval_item["resolved_path"]).resolve()
            try:
                _value, provenance = _load_cell(
                    CELL_NAMES[arm],
                    evaluation,
                    "in_box",
                    require_source_checkpoint=True,
                    expected_lora_rank=rank,
                )
            except EffectError as exc:
                raise CleanupError(f"r{rank}/{arm}: eval validation failed: {exc}") from exc
            artifact_manifest = Path(
                provenance["model_provenance"]["artifact_manifest"]
            ).resolve()
            if artifact_manifest != model / "train_export_manifest.json":
                raise CleanupError(f"r{rank}/{arm}: eval does not point to registered model input")
            target = model / "orbax"
            if target.is_symlink() or target.resolve().parent != model:
                raise CleanupError(f"r{rank}/{arm}: unsafe Orbax target {target}")
            if not (target / "000750/_CHECKPOINT_METADATA").is_file():
                raise CleanupError(f"r{rank}/{arm}: intact step-750 source is absent")
            validated.append({
                "rank": rank,
                "arm": arm,
                "model_artifact_id": model_item["artifact_id"],
                "eval_artifact_id": eval_item["artifact_id"],
                "model_path": str(model),
                "eval_path": str(evaluation),
                "orbax_path": str(target),
                "logical_bytes_before": _logical_bytes(target),
            })
            targets.append(target)

    for target in targets:
        shutil.rmtree(target)
        target.mkdir()
    for item in validated:
        target = Path(item["orbax_path"])
        if not target.is_dir() or any(target.iterdir()):
            raise CleanupError(f"postcondition failed for {target}")
    return {
        "artifact_type": "synthetic_relative_factorial_capacity_checkpoint_cleanup",
        "schema_version": 1,
        "status": "complete",
        "validation": "all exports and registered evals validated before any deletion",
        "deleted": "eight Orbax source trees; HF exports retained",
        "logical_bytes_removed": sum(item["logical_bytes_before"] for item in validated),
        "entries": validated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labctl-context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = cleanup(labctl_context=args.labctl_context)
    except CleanupError as exc:
        print(f"FATAL capacity cleanup: {exc}", file=sys.stderr)
        return 2
    _atomic_write(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
