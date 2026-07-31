"""Preregistered closed-loop metrics, factored for deterministic unit tests."""
from __future__ import annotations

import math
from typing import Any


def _target_steps(row: dict[str, Any], target_index: int) -> list[dict[str, Any]]:
    return [step for step in row.get("steps", []) if step["target_index"] == target_index]


def _distance_auc(steps: list[dict[str, Any]], max_attempts: int) -> float | None:
    if not steps:
        return None
    start = float(steps[0]["distance_before"])
    scale = max(start, 1.0)
    distances = [start] + [float(step["distance_after"]) for step in steps]
    final = 0.0 if any(step["hit"] for step in steps) else distances[-1]
    while len(distances) < max_attempts + 1:
        distances.append(final)
    distances = distances[: max_attempts + 1]
    area = sum((a + b) / 2.0 for a, b in zip(distances, distances[1:]))
    return area / (max_attempts * scale)


def summarize(rows: list[dict[str, Any]], *, max_attempts: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero episodes")
    planned: list[tuple[dict[str, Any], int, list[dict[str, Any]]]] = []
    for row in rows:
        for target_index in range(int(row["target_count"])):
            planned.append((row, target_index, _target_steps(row, target_index)))

    reach_cdf = {}
    for attempt in range(1, max_attempts + 1):
        reached = sum(
            any(step["hit"] and step["attempt"] <= attempt for step in steps)
            for _row, _target, steps in planned
        )
        reach_cdf[str(attempt)] = reached / len(planned)

    attempted = [(row, target, steps) for row, target, steps in planned if steps]
    aucs = [value for _row, _target, steps in attempted
            if (value := _distance_auc(steps, max_attempts)) is not None]
    first_miss = [steps for _row, _target, steps in attempted if not steps[0]["hit"]]
    first_miss_recovered = sum(any(step["hit"] for step in steps[1:]) for steps in first_miss)
    miss_events = 0
    recovered_miss_events = 0
    for _row, _target, steps in attempted:
        for index, step in enumerate(steps):
            if step["hit"]:
                continue
            miss_events += 1
            recovered_miss_events += int(any(later["hit"] for later in steps[index + 1 :]))

    steps = [step for row in rows for step in row.get("steps", [])]
    parsed_moves = [step for step in steps if step.get("coord") is not None]
    eps = 1e-9
    progress = sum(step["progress_px"] > eps for step in parsed_moves)
    regression = sum(step["progress_px"] < -eps for step in parsed_moves)
    stalled = len(parsed_moves) - progress - regression
    denominators = {
        "planned_targets": len(planned),
        "attempted_targets": len(attempted),
        "steps": len(steps),
        "parsed_moves": len(parsed_moves),
        "first_miss_targets": len(first_miss),
        "miss_events": miss_events,
    }

    def rate(numerator: int | float, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    output = {
        "n_episodes": len(rows),
        "denominators": denominators,
        "target_reach_cdf_by_attempt": reach_cdf,
        "first_attempt_reach_rate": reach_cdf["1"],
        "episode_completion_rate": sum(bool(row["completed"]) for row in rows) / len(rows),
        "first_miss_recovery_rate": rate(first_miss_recovered, len(first_miss)),
        "miss_event_recovery_rate": rate(recovered_miss_events, miss_events),
        "normalized_distance_auc": sum(aucs) / len(aucs) if aucs else None,
        "progress_rate": rate(progress, len(parsed_moves)),
        "regression_rate": rate(regression, len(parsed_moves)),
        "stall_rate": rate(stalled, len(parsed_moves)),
        "oscillation_rate": rate(sum(bool(step.get("oscillation")) for step in steps), len(steps)),
        "parse_rate": rate(sum(bool(step.get("parse_ok")) for step in steps), len(steps)),
        "strict_schema_rate": rate(sum(bool(step.get("schema_ok")) for step in steps), len(steps)),
        "coordinate_unit_violation_rate": rate(
            sum(step.get("coord") is not None and not step.get("unit_range_ok") for step in steps),
            len(steps),
        ),
        "no_move_rate": rate(sum(step.get("coord") is None for step in steps), len(steps)),
        "request_error_count": sum(
            1 for row in rows for step in row.get("steps", []) if step.get("request_error")
        ),
    }
    numeric = [value for value in output.values() if isinstance(value, float)]
    numeric.extend(value for value in reach_cdf.values())
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("non-finite metric")
    return output
