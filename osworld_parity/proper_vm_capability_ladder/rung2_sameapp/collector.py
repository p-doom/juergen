from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..rung1.vm import DEFAULT_PROVIDER, DEFAULT_QCOW, DEFAULT_QEMU
from .fixtures import assert_collectable_split, load_manifest
from .replay import _compiled_actions, run_build_replay, run_vm_replay


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def collect(
    *,
    mode: str,
    split: str,
    output: Path,
    qcow: Path = DEFAULT_QCOW,
    qemu: Path = DEFAULT_QEMU,
    provider: Path = DEFAULT_PROVIDER,
) -> dict[str, Any]:
    assert_collectable_split(split)
    if split != "train":
        raise ValueError("teacher collection is train-only; development is replay-only")
    manifest = load_manifest(split)
    replay = (
        run_build_replay(split)
        if mode == "build"
        else run_vm_replay(
            split,
            output=output,
            qcow=qcow,
            qemu=qemu,
            provider=provider,
        )
    )
    evidence: dict[str, list[dict[str, Any]]] = {}
    demonstrations: dict[str, list[dict[str, Any]]] = {}
    for row in replay["rows"]:
        evidence.setdefault(row["fixture_id"], []).append(
            {
                "arm": row.get("arm", "schema_build_replay"),
                "reset_signature": row["reset_signature"],
                "near_miss_rejected": row["near_miss_oracle"]["MOUSE_SOLVED"] is False,
                "gold_accepted": row["gold_oracle"]["MOUSE_SOLVED"] is True,
            }
        )
        if mode == "vm":
            demonstrations.setdefault(row["fixture_id"], []).append(
                {
                    "arm": row["arm"],
                    "turns": [
                        {
                            "turn": turn["turn"],
                            "semantic_step": turn["semantic_step"],
                            "screenshot": turn["screenshot"],
                            "action": turn["action"],
                        }
                        for turn in row["gold_journal"]
                    ],
                }
            )
    rows: list[dict[str, Any]] = []
    for fixture in manifest.fixtures:
        rows.append(
            {
                "schema_version": 1,
                "fixture_id": fixture.id,
                "fixture_sha256": fixture.fixture_sha256,
                "manifest_payload_sha256": manifest.manifest_payload_sha256,
                "split": "train",
                "app": fixture.app,
                "instruction": fixture.instruction,
                "semantic_steps": fixture.semantic_steps,
                "horizon": fixture.horizon,
                "observation_contract": "instruction_and_screenshot_only",
                "gold_actions": _compiled_actions(fixture, near_miss=False),
                "scripted_replay_attestations": evidence.get(fixture.id),
                "vm_screenshot_action_demonstrations": demonstrations.get(fixture.id),
                "training_ready": mode == "vm",
                "sealed_eval_material": False,
                "model_generated": False,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "teacher_trajectories.jsonl"
    dataset_sha256 = _write_jsonl(dataset, rows)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "mode": mode,
        "split": split,
        "row_count": len(rows),
        "apps": sorted({row["app"] for row in rows}),
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "sealed_eval_executed": False,
        "model_used": False,
        "gpu_used": False,
        "training_ready": mode == "vm",
    }
    (output / "collection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "vm"), required=True)
    parser.add_argument("--split", choices=("train", "development", "sealed_eval"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    args = parser.parse_args(argv)
    try:
        payload = collect(
            mode=args.mode,
            split=args.split,
            output=args.output,
            qcow=args.qcow,
            qemu=args.qemu,
            provider=args.provider,
        )
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1,
            "status": "failed",
            "mode": args.mode,
            "split": args.split,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "sealed_eval_executed": False,
        }
        (args.output / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
