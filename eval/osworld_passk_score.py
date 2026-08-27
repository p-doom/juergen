"""Aggregate a pass@k OSWorld sweep into pass@1 / pass@k / mean-best-of-k.

Sample i of task {app}/{task_id} lives at:
    i == 0:  {base_output_dir}/{app}/{task_id}/result.json
    i  > 0:  {base_output_dir}/{app}/{task_id}/sample_{i}/result.json

Metrics (NaN reward, i.e. an evaluator crash, counts as 0.0):
    pass_at_1       mean reward over every sample that ran
    pass_at_k       fraction of tasks with at least one sample scoring > 0
    mean_best_of_k  mean over tasks of the max reward across that task's samples

The emitted score.json is a superset of the single-sample schema the
strat40 recipes wrote (checkpoint / arm / n / mean_reward / per_task), so
existing consumers keep working against a pass@k run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _reward(scores: dict) -> float:
    r = scores.get("reward", float("nan"))
    try:
        r = float(r)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(r) else r


def _sample_path(base: Path, app: str, task_id: str, i: int) -> Path:
    task_dir = base / app / task_id
    if i:
        task_dir = task_dir / f"sample_{i}"
    return task_dir / "result.json"


def _load_task_list(split_path: str) -> list[tuple[str, str]]:
    with Path(split_path).open() as f:
        d = json.load(f)
    return [(app, tid) for app, tids in sorted(d.items()) for tid in tids]


def collect(base: Path, tasks: list[tuple[str, str]], k: int) -> list[dict]:
    rows = []
    for app, tid in tasks:
        samples = []
        for i in range(k):
            p = _sample_path(base, app, tid, i)
            if not p.exists():
                samples.append(None)
                continue
            with p.open() as f:
                d = json.load(f)
            samples.append(d.get("scores", {}))
        rows.append({"app": app, "task_id": tid, "samples": samples})
    return rows


def summarize(rows: list[dict], k: int) -> dict:
    all_rewards: list[float] = []
    best_per_task: list[float] = []
    solved_any = 0
    n_tasks_with_samples = 0
    n_missing = 0
    per_task_detail = []
    per_app: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        present = [s for s in row["samples"] if s is not None]
        n_missing += k - len(present)
        rewards = [_reward(s) for s in present]
        all_rewards.extend(rewards)
        bucket = per_app.setdefault(row["app"], {"rewards": [], "best": []})
        bucket["rewards"].extend(rewards)
        if rewards:
            n_tasks_with_samples += 1
            best = max(rewards)
            best_per_task.append(best)
            bucket["best"].append(best)
            if best > 0:
                solved_any += 1
        else:
            best = None
        per_task_detail.append(
            {
                "task": f"{row['app']}/{row['task_id']}",
                "n_samples": len(present),
                "rewards": rewards,
                "best": best,
                "any_solved": bool(rewards) and max(rewards) > 0,
            }
        )

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    legacy_per_task = []
    for row in rows:
        if row["samples"] and row["samples"][0] is not None:
            legacy_per_task.append([f"{row['app']}/{row['task_id']}", row["samples"][0]])

    return {
        "k": k,
        "n_tasks": len(rows),
        "n_tasks_scored": n_tasks_with_samples,
        "n_samples": len(all_rewards),
        "n_samples_missing": n_missing,
        "pass_at_1": _mean(all_rewards),
        "pass_at_k": (solved_any / n_tasks_with_samples) if n_tasks_with_samples else None,
        "mean_best_of_k": _mean(best_per_task),
        "n_solved_any": solved_any,
        "n": len(legacy_per_task),
        "mean_reward": _mean([_reward(s) for _, s in legacy_per_task]),
        "per_task": sorted(legacy_per_task),
        "per_task_detail": sorted(per_task_detail, key=lambda r: r["task"]),
        "per_app": {
            app: {
                "pass_at_1": _mean(v["rewards"]),
                "pass_at_k": (
                    sum(1 for b in v["best"] if b > 0) / len(v["best"]) if v["best"] else None
                ),
                "mean_best_of_k": _mean(v["best"]),
                "n_samples": len(v["rewards"]),
            }
            for app, v in sorted(per_app.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_output_dir", required=True)
    parser.add_argument("--test_split_path", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--arm", default="")
    args = parser.parse_args()

    base = Path(args.base_output_dir)
    tasks = _load_task_list(args.test_split_path)
    summary = summarize(collect(base, tasks, args.k), args.k)
    summary["checkpoint"] = args.checkpoint
    summary["arm"] = args.arm

    out_path = Path(args.output_path) if args.output_path else base / "score.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "arm",
                    "k",
                    "n_tasks",
                    "n_samples",
                    "n_samples_missing",
                    "pass_at_1",
                    "pass_at_k",
                    "mean_best_of_k",
                    "n_solved_any",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
