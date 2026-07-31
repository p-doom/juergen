#!/usr/bin/env python3
"""Read the tiny optimizer counters from an Orbax typing checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import orbax.checkpoint as ocp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-gradient-step", type=int, required=True)
    parser.add_argument("--expected-micro-step", type=int, required=True)
    args = parser.parse_args()

    item = {
        "optimizer": {
            "step": {"value": 0},
            "opt_state": {
                "gradient_step": {"value": 0},
                "mini_step": {"value": 0},
            },
        }
    }
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            args.checkpoint / "train_state",
            args=ocp.args.PyTreeRestore(item=item, partial_restore=True),
        )
    optimizer = restored["optimizer"]
    gradient_step = int(optimizer["opt_state"]["gradient_step"]["value"])
    mini_step = int(optimizer["opt_state"]["mini_step"]["value"])
    micro_step = int(optimizer["step"]["value"])
    if gradient_step != args.expected_gradient_step:
        raise SystemExit(
            f"FATAL gradient step {gradient_step} != {args.expected_gradient_step}"
        )
    if micro_step != args.expected_micro_step or mini_step != 0:
        raise SystemExit(
            f"FATAL optimizer counters micro={micro_step} mini={mini_step}; "
            f"expected {args.expected_micro_step}, 0"
        )
    report = {
        "status": "pass",
        "checkpoint": str(args.checkpoint.resolve()),
        "global_gradient_step": gradient_step,
        "optimizer_micro_step": micro_step,
        "gradient_accumulation_remainder": mini_step,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
