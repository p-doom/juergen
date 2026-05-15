"""Aggregate per-task result.json files into an OSWorld benchmark score.

Usage:
    python osworld_score.py --base_output_dir /path/to/results \
        --test_split_path /path/to/test_all.json

Walks {base_output_dir}/{app}/{task_id}/result.json, prints per-app and
overall success rates, and writes a summary JSON to {output_dir}/score.json
(defaulting to base_output_dir).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _mean(rewards: list[float]) -> float:
    """Mean over non-NaN rewards. NaN when all-NaN or empty."""
    valid = [r for r in rewards if not math.isnan(r)]
    return sum(valid) / len(valid) if valid else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_output_dir",
        required=True,
        help="Root dir under which per-task result.json files live.",
    )
    parser.add_argument(
        "--test_split_path", required=True, help="OSWorld test split JSON (e.g. test_all.json)."
    )
    parser.add_argument(
        "--output_dir", default=None, help="Where to write score.json (default: base_output_dir)."
    )
    args = parser.parse_args()

    base = Path(args.base_output_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.test_split_path).open() as f:
        test_split = json.load(f)

    total_tasks = sum(len(v) for v in test_split.values())
    app_results: dict[str, list[float]] = defaultdict(list)
    missing: list[tuple[str, str]] = []

    for app_name, task_ids in sorted(test_split.items()):
        for tid in task_ids:
            result_path = base / app_name / tid / "result.json"
            if not result_path.exists():
                missing.append((app_name, tid))
                continue
            with result_path.open() as f:
                d = json.load(f)
            reward = d.get("scores", {}).get("reward", float("nan"))
            app_results[app_name].append(reward)

    print(f"\nOSWorld benchmark results: {base}\n")
    print(f"{'App':<25} {'Done':>6} {'Total':>6} {'Success':>8}")
    print("-" * 50)

    all_rewards: list[float] = []
    for app_name, task_ids in sorted(test_split.items()):
        rewards = app_results.get(app_name, [])
        mean = _mean(rewards)
        pct = f"{mean * 100:.1f}%" if not math.isnan(mean) else "N/A"
        print(f"{app_name:<25} {len(rewards):>6} {len(task_ids):>6} {pct:>8}")
        all_rewards.extend(r for r in rewards if not math.isnan(r))

    print("-" * 50)
    overall = sum(all_rewards) / len(all_rewards) if all_rewards else float("nan")
    n_done = sum(len(v) for v in app_results.values())
    print(f"{'OVERALL':<25} {n_done:>6} {total_tasks:>6} {overall * 100:.2f}%\n")

    if missing:
        print(f"Missing results ({len(missing)} tasks):")
        for app_name, tid in missing[:20]:
            print(f"  {app_name}/{tid}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        print()

    summary = {
        "overall_success_rate": overall,
        "n_done": n_done,
        "n_total": total_tasks,
        "n_missing": len(missing),
        "per_app": {
            app_name: {
                "n_done": len(app_results.get(app_name, [])),
                "n_total": len(tids),
                "success_rate": _mean(app_results.get(app_name, [])),
            }
            for app_name, tids in sorted(test_split.items())
        },
    }

    out_path = out_dir / "score.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
