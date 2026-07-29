#!/usr/bin/env python3
"""Deterministic, no-leak train/eval partition of the OSWorld-Verified 369-task set.

- Source universe: OSWorld $OSWORLD_ROOT/evaluation_examples/test_all.json (369 tasks,
  10 apps) — the exact set the Qwen3-VL tech report scores 8B-Instruct = 33.9% on.
- Stratified by app; per-app ~30% held-out EVAL / ~70% TRAIN.
- Partition is DISJOINT by construction (eval and train are complementary index
  slices of the same per-app sorted+shuffled list) -> zero eval leakage into train.
- Fully reproducible: fixed global seed, per-app sorted task ids, seeded shuffle.

Outputs (in {app: [task_id, ...]} form, directly usable as --test_split_path):
  osworld_eval_heldout.json   the parity metric (NEVER used for rollout generation)
  osworld_train.json          teacher-rollout generation ONLY
  split_manifest.json         counts + per-task instruction text + provenance
  SPLIT_README.txt            human-readable summary
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

OSWORLD_ROOT = Path("/fast/home/franz.srambical/OSWorld")
TEST_ALL = OSWORLD_ROOT / "evaluation_examples" / "test_all.json"
EXAMPLES = OSWORLD_ROOT / "evaluation_examples" / "examples"
OUT = Path("/fast/home/franz.srambical/osworld_parity_split")

EVAL_FRACTION = 0.30
SEED = 20260728  # fixed -> reproducible

def main() -> None:
    universe = json.loads(TEST_ALL.read_text())
    total = sum(len(v) for v in universe.values())
    assert total == 369, f"expected 369 tasks, got {total}"

    eval_split: dict[str, list[str]] = {}
    train_split: dict[str, list[str]] = {}
    manifest_tasks: list[dict] = []

    for app in sorted(universe):
        ids = sorted(universe[app])  # deterministic base order
        rng = random.Random(f"{SEED}:{app}")
        shuffled = ids[:]
        rng.shuffle(shuffled)
        n_eval = round(len(shuffled) * EVAL_FRACTION)
        # guarantee both non-empty when app has >=2 tasks
        n_eval = max(1, min(len(shuffled) - 1, n_eval)) if len(shuffled) >= 2 else 0
        ev = sorted(shuffled[:n_eval])
        tr = sorted(shuffled[n_eval:])
        assert not (set(ev) & set(tr)), f"LEAK in {app}"
        assert set(ev) | set(tr) == set(ids), f"partition incomplete in {app}"
        eval_split[app] = ev
        train_split[app] = tr
        for tid in ids:
            tp = EXAMPLES / app / f"{tid}.json"
            instr = ""
            if tp.exists():
                try:
                    instr = json.loads(tp.read_text()).get("instruction", "")
                except Exception:
                    instr = "<unreadable>"
            manifest_tasks.append(
                {"app": app, "task_id": tid, "split": "eval" if tid in set(ev) else "train",
                 "instruction": instr}
            )

    n_eval = sum(len(v) for v in eval_split.values())
    n_train = sum(len(v) for v in train_split.values())
    # global disjointness check
    ev_all = {(a, t) for a, ts in eval_split.items() for t in ts}
    tr_all = {(a, t) for a, ts in train_split.items() for t in ts}
    assert not (ev_all & tr_all), "GLOBAL LEAK"
    assert len(ev_all) + len(tr_all) == 369

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "osworld_eval_heldout.json").write_text(json.dumps(eval_split, indent=2, sort_keys=True))
    (OUT / "osworld_train.json").write_text(json.dumps(train_split, indent=2, sort_keys=True))

    # content hash so the pipeline subagent can pin the exact split it trained against
    def h(d):
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    manifest = {
        "provenance": {
            "source": str(TEST_ALL), "total_tasks": 369, "eval_fraction": EVAL_FRACTION,
            "seed": SEED, "stratified_by": "app", "disjoint": True,
        },
        "counts": {
            "eval": n_eval, "train": n_train,
            "per_app": {a: {"eval": len(eval_split[a]), "train": len(train_split[a])}
                        for a in sorted(universe)},
        },
        "hashes": {"eval": h(eval_split), "train": h(train_split)},
        "tasks": manifest_tasks,
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = [
        "OSWorld no-leak parity split (source: test_all.json, 369 tasks OSWorld-Verified)",
        f"seed={SEED}  eval_fraction={EVAL_FRACTION}  stratified by app  DISJOINT",
        f"EVAL (held-out parity metric): {n_eval} tasks   hash={manifest['hashes']['eval']}",
        f"TRAIN (rollout generation only): {n_train} tasks  hash={manifest['hashes']['train']}",
        "",
        f"{'app':<20}{'eval':>6}{'train':>7}{'total':>7}",
    ]
    for a in sorted(universe):
        lines.append(f"{a:<20}{len(eval_split[a]):>6}{len(train_split[a]):>7}{len(universe[a]):>7}")
    lines.append(f"{'TOTAL':<20}{n_eval:>6}{n_train:>7}{369:>7}")
    (OUT / "SPLIT_README.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote: {OUT}/osworld_eval_heldout.json, osworld_train.json, split_manifest.json")


if __name__ == "__main__":
    main()
