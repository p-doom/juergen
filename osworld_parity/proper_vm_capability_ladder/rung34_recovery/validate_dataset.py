from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import assert_cpu_only
from .rollouts import validate_rollout_record


def validate_dataset(input_path: Path, output: Path) -> dict[str, object]:
    assert_cpu_only()
    rows = 0
    splits: set[str] = set()
    arms: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"rollout row is not an object at line {line_number}")
            validate_rollout_record(row)
            splits.add(str(row["split"]))
            arms.add(str(row["arm"]))
            rows += 1
    if rows == 0:
        raise RuntimeError("rollout dataset is empty")
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "record_count": rows,
        "splits": sorted(splits),
        "arms": sorted(arms),
        "sealed_evaluation_opened": 0,
        "models_run": 0,
        "gpu_count": 0,
        "trainer_only_values_exported": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_dataset(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
