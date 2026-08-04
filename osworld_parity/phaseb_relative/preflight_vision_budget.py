#!/usr/bin/env python3
"""Fail-loud preflight for Phase-B static vision padding budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def scan_split(split_dir: Path, merge_size: int = 2) -> dict[str, Any]:
    rows = [json.loads(line) for line in (split_dir / "sequence_lengths.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError(f"no sequence-length rows in {split_dir}")
    measured = []
    for row in rows:
        # Qwen grids are divisible by spatial_merge_size, so
        # patches = vision_tokens * spatial_merge_size**2 exactly.
        patches = int(row["vision_tokens"]) * merge_size * merge_size
        measured.append({
            "session_id": row["session_id"],
            "num_images": int(row["num_images"]),
            "vision_tokens": int(row["vision_tokens"]),
            "vision_patches": patches,
        })

    stats = json.loads((split_dir / "token_stats.json").read_text())
    chunks = stats["per_chunk"]
    expected = {
        "records": int(chunks["num_chunks"]),
        "max_images": int(chunks["num_images"]["max"]),
        "max_patches": int(chunks["vision_patches"]["max"]),
        "sum_images": int(chunks["num_images"]["sum"]),
        "sum_patches": int(chunks["vision_patches"]["sum"]),
    }
    observed = {
        "records": len(measured),
        "max_images": max(r["num_images"] for r in measured),
        "max_patches": max(r["vision_patches"] for r in measured),
        "sum_images": sum(r["num_images"] for r in measured),
        "sum_patches": sum(r["vision_patches"] for r in measured),
    }
    if observed != expected:
        raise ValueError(f"full-scan/token-stats disagreement for {split_dir}: {observed} != {expected}")
    max_image_row = max(measured, key=lambda r: r["num_images"])
    max_patch_row = max(measured, key=lambda r: r["vision_patches"])
    return {**observed, "max_image_record": max_image_row, "max_patch_record": max_patch_row,
            "rows": measured}


def check_budget(rows: list[dict[str, Any]], max_images: int, max_patches: int,
                 merge_size: int = 2) -> None:
    ms2 = merge_size * merge_size
    for row in rows:
        images, patches = row["num_images"], row["vision_patches"]
        if images > max_images or patches > max_patches:
            raise ValueError(
                f"vision budget exceeded by {row['session_id']}: real_images={images} > "
                f"max_images={max_images} or real_patches={patches} > max_patches={max_patches}"
            )
        dummies, extra = max_images - images, max_patches - patches
        feasible = (dummies == 0 and extra == 0) or (
            dummies >= 1 and extra >= dummies * ms2 and extra % ms2 == 0
        )
        if not feasible:
            raise ValueError(
                f"infeasible vision padding for {row['session_id']}: real_images={images}, "
                f"real_patches={patches}, max_images={max_images}, max_patches={max_patches}"
            )


def preflight(dataset: Path, arm: str, max_images: int, max_patches: int,
              merge_size: int = 2) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        result = scan_split(dataset / arm / split, merge_size)
        all_rows.extend(result.pop("rows"))
        splits[split] = result
    check_budget(all_rows, max_images, max_patches, merge_size)
    return {
        "status": "pass",
        "dataset": str(dataset),
        "arm": arm,
        "records_scanned": len(all_rows),
        "merge_size": merge_size,
        "configured": {"max_images": max_images, "max_patches": max_patches},
        "observed": {
            "max_images": max(v["max_images"] for v in splits.values()),
            "max_patches": max(v["max_patches"] for v in splits.values()),
        },
        "splits": splits,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--arm", default="prose_keep")
    p.add_argument("--max-images", type=int, required=True)
    p.add_argument("--max-patches", type=int, required=True)
    p.add_argument("--merge-size", type=int, default=2)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    report = preflight(args.dataset, args.arm, args.max_images, args.max_patches, args.merge_size)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
