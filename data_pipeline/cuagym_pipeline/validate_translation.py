from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))
EVAL_DIR = DATA_PIPELINE_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from absl import app, flags

from action_parser import parse_ordered_action
from cuagym_pipeline.translate import (
    DropStep,
    reconstruct_target_px,
    rewrite_assistant,
    translate_step,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("trajectories", None, "Path to trajectories.jsonl", required=True)
flags.DEFINE_integer("limit", 0, "Max rollouts to scan (0 = all)")
flags.DEFINE_string("report", "", "Optional JSON report output path")


def main(argv):
    del argv
    n_rollouts = 0
    n_steps = 0
    action_hist = Counter()
    drop_reasons = Counter()
    parse_fail_steps = 0
    strict_parse_failures = []
    invert_err_hist = Counter()
    invert_failures = []
    rewrite_failures = Counter()
    line_len_hist = Counter()
    rewards = Counter()

    with open(FLAGS.trajectories) as fh:
        for raw_line in fh:
            if FLAGS.limit and n_rollouts >= FLAGS.limit:
                break
            rec = json.loads(raw_line)
            n_rollouts += 1
            screen = tuple(rec.get("screen") or (1920, 1080))
            reward = rec.get("reward")
            if reward is None:
                rewards["null"] += 1
            elif reward >= 0.999:
                rewards["perfect"] += 1
            elif reward > 0:
                rewards["partial"] += 1
            else:
                rewards["zero"] += 1
            for step in rec.get("steps") or []:
                n_steps += 1
                if "assistant_raw" not in step or "raw_action_args" not in step:
                    parse_fail_steps += 1
                    continue
                args = step["raw_action_args"]
                cursor = step.get("cursor_before")
                if not cursor:
                    drop_reasons["missing_cursor_before"] += 1
                    continue
                try:
                    t = translate_step(args, tuple(cursor), screen)
                except DropStep as exc:
                    drop_reasons[exc.reason.split(":")[0]] += 1
                    continue
                if t.dropped_reason:
                    drop_reasons[t.dropped_reason] += 1
                    continue
                action_hist[args.get("action") or "?"] += 1
                if t.line not in ("TERMINATE",):
                    try:
                        parse_ordered_action(t.line)
                    except (ValueError, TypeError) as exc:
                        strict_parse_failures.append(
                            {"task_id": rec["task_id"], "step": step.get("step"), "line": t.line[:200], "err": str(exc)[:200]}
                        )
                line_len_hist[min(len(t.line) // 20 * 20, 200)] += 1
                cs = step.get("coordinate_screen")
                if t.move_delta is not None and t.move_delta != (0, 0) and cs:
                    px = reconstruct_target_px(tuple(cursor), t.move_delta, screen)
                    err = max(abs(px[0] - cs[0]), abs(px[1] - cs[1]))
                    invert_err_hist[err] += 1
                    if err > 2:
                        invert_failures.append(
                            {"task_id": rec["task_id"], "step": step.get("step"), "px": px, "coordinate_screen": cs, "cursor_before": cursor, "args": args}
                        )
                try:
                    rewrite_assistant(step["assistant_raw"], t.line)
                except DropStep as exc:
                    rewrite_failures[exc.reason] += 1

    report = {
        "rollouts": n_rollouts,
        "steps": n_steps,
        "rewards": dict(rewards),
        "actions": dict(action_hist.most_common()),
        "drops": dict(drop_reasons.most_common()),
        "harness_parse_failure_steps": parse_fail_steps,
        "strict_parse_failures": len(strict_parse_failures),
        "strict_parse_examples": strict_parse_failures[:10],
        "invert_error_hist": {str(k): v for k, v in sorted(invert_err_hist.items())},
        "invert_gt2px": len(invert_failures),
        "invert_examples": invert_failures[:10],
        "rewrite_failures": dict(rewrite_failures),
        "line_len_hist": {str(k): v for k, v in sorted(line_len_hist.items())},
    }
    out = json.dumps(report, indent=2)
    if FLAGS.report:
        Path(FLAGS.report).write_text(out)
    print(out)


if __name__ == "__main__":
    app.run(main)
