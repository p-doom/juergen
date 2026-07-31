#!/usr/bin/env python3
"""Audit the preregistered low-LR rescue pair without submitting it."""
from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from .curriculum_launch_gate import DEADLINE, MAX_PEAK_BYTES, _du, _recipe_diff
except ImportError:
    from curriculum_launch_gate import DEADLINE, MAX_PEAK_BYTES, _du, _recipe_diff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-a", required=True, type=Path)
    parser.add_argument("--source-b", required=True, type=Path)
    parser.add_argument("--recipe-a", required=True, type=Path)
    parser.add_argument("--recipe-b", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    frozen = json.loads((Path(__file__).parent / "frozen_manifest.json").read_text())[
        "curriculum_transfer"
    ]
    rescue = frozen["low_lr_rescue_prepared"]
    if rescue["submission_authorized"] is not False or "not_submitted" not in rescue["status"]:
        raise SystemExit("FATAL rescue preparation no longer has a no-submit guard")
    recipes = [tomllib.loads(path.read_text()) for path in (args.recipe_a, args.recipe_b)]
    unexpected = _recipe_diff(*recipes)
    if unexpected:
        raise SystemExit(f"FATAL rescue recipes are not matched: {unexpected}")
    for recipe in recipes:
        if (recipe["args"].get("learning_rate") != "5e-5"
                or recipe["resources"]["time"] != "02:00:00"
                or "--deadline=2026-07-31T05:09:00" not in recipe["resources"]["sbatch_extra"]):
            raise SystemExit(f"FATAL wrong rescue LR/time/deadline: {recipe['name']}")
        if recipe["inputs"]["dataset"]["artifact"] != frozen["stage2_dataset"]["dataset_alias"]:
            raise SystemExit("FATAL rescue dataset differs from original")
    source_sizes = [_du(args.source_a), _du(args.source_b)]
    dataset_size = _du(args.dataset)
    output_bound_each = (17 * max(source_sizes) + 3) // 4
    incremental_pair_bound = (
        sum(source_sizes) + dataset_size + 2 * output_bound_each + 10_000_000_000
    )
    retained_current_hf_bound = 2 * max(source_sizes)
    all_lineage_after_current_orbax_cleanup = incremental_pair_bound + retained_current_hf_bound
    all_lineage_before_current_orbax_cleanup = incremental_pair_bound + 2 * output_bound_each
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    result = {
        "status": "prepared_not_submitted",
        "submission_authorized": False,
        "learning_rate": 5e-5,
        "pair_recipe_differences_allowed": [
            "name", "inputs.source_model.artifact", "outputs.model.alias"
        ],
        "incremental_pair_peak_bound_bytes": incremental_pair_bound,
        "incremental_pair_below_500GB": incremental_pair_bound < MAX_PEAK_BYTES,
        "all_lineage_peak_after_current_orbax_cleanup_bytes":
            all_lineage_after_current_orbax_cleanup,
        "all_lineage_peak_before_current_orbax_cleanup_bytes":
            all_lineage_before_current_orbax_cleanup,
        "two_hour_limit_end_if_submitted_now": (now + timedelta(hours=2)).isoformat(),
        "deadline": DEADLINE.isoformat(),
        "latest_feasible_submission_for_two_hour_limit":
            (DEADLINE - timedelta(hours=2)).isoformat(),
        "requires_current_export_eval_analysis_before_current_orbax_cleanup": True,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
