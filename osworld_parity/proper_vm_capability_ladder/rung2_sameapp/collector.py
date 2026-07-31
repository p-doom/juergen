"""Fail-closed boundary for the superseded direct same-app collector.

Production curriculum replay now requires live reset bindings, immutable setup
validation, real artifact extraction, and executed segment receipts. The old
teacher collector cannot satisfy those dependencies and is intentionally not a
second execution path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..rung1.vm import DEFAULT_PROVIDER, DEFAULT_QCOW, DEFAULT_QEMU


def collect(**_: Any) -> None:
    raise RuntimeError(
        "legacy rung2_sameapp collector is disabled; use hardened development "
        "replay with pinned task_setup_validation.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "vm"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    failure = {
        "schema_version": 2,
        "status": "failed",
        "mode": args.mode,
        "split": args.split,
        "sealed_eval_executed": False,
        "error_type": "RuntimeError",
        "message": (
            "legacy rung2_sameapp collector is disabled; use hardened development "
            "replay with pinned task_setup_validation.json"
        ),
    }
    (args.output / "failure.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(failure, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
