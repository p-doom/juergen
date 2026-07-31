from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..trajectory import UiGeometry, build_trajectory
from .conversion import assert_round_trip
from .splits import materialize_tasks


def scripted_gold_records(split: str) -> list[dict[str, Any]]:
    if split not in {"train", "development"}:
        raise ValueError("demonstration export is train/development only")
    rows: list[dict[str, Any]] = []
    initial_cursor = (73, 91)
    for fixture in materialize_tasks(split):
        native = build_trajectory(
            fixture,
            arm="native_absolute_control",
            cursor=initial_cursor,
            geometry=UiGeometry(),
        )
        native_actions = tuple(action for action in native.actions if isinstance(action, dict))
        compact_actions = assert_round_trip(native_actions, initial_cursor=initial_cursor)
        rows.append(
            {
                "schema_version": 1,
                "source": "scripted_gold_development_contract",
                "task_id": fixture.id,
                "fixture_sha256": fixture.fixture_sha256,
                "split": fixture.split,
                "parameter_seed": fixture.parameter_seed,
                "instruction": fixture.instruction,
                "initial_cursor": list(initial_cursor),
                "native_absolute_actions": list(native_actions),
                "compact_raw_actions": list(compact_actions),
                "hidden_reward_in_record": False,
                "oracle_state_in_record": False,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "development"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = scripted_gold_records(args.split)
    output = args.output / "demonstrations.jsonl"
    write_jsonl(output, rows)
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "split": args.split,
        "record_count": len(rows),
        "evaluation_opened": 0,
        "contains_hidden_reward": False,
        "contains_oracle_state": False,
    }
    (args.output / "result.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
