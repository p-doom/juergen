from __future__ import annotations

import argparse
import json
import os

from ..contracts import sha256_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--expected-step-index", type=int)
    parser.add_argument("--expected-target-ref")
    args = parser.parse_args()
    with open(args.state, encoding="utf-8") as handle:
        state = json.load(handle)
    index = int(state["semantic_step_index"])
    target = str(state["target"])
    next_state_matches = (
        args.expected_step_index is None
        or (
            index == args.expected_step_index
            and target == args.expected_target_ref
        )
    )
    result = {
        "task_id": args.task_id,
        "fixture_sha256": state["fixture_sha256"],
        "oracle_status": "ok",
        "MOUSE_SOLVED": bool(state["task_solved"] and next_state_matches),
        "semantic_step_index": index,
        "matched_target_ref": target if next_state_matches else None,
        "semantic_state_sha256": sha256_json(state),
        "reason": "fake fresh-process semantic verifier",
        "oracle_pid": os.getpid(),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
