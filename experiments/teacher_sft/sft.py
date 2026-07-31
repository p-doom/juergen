"""Package verified relative trajectories as hashed Omegalax chat rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.teacher_sft import CONVERSION_VERSION, SCHEMA_VERSION
from experiments.teacher_sft.actions import parse_compact_sequence
from experiments.teacher_sft.contracts import (
    ContractError,
    assert_not_heldout,
    ensure_empty_output,
    file_sha256,
    iter_jsonl,
    load_heldout_denylist,
    object_sha256,
    read_json,
    verify_declared_hash,
    write_json,
    write_jsonl,
)

SYSTEM_PROMPT = """You operate a desktop computer using compact raw relative actions. The first user turn shows the current screen and the user's goal; later user turns show the current screen. Reply with one or more action lines, in execution order.

Each line is one of: NO_OP, TERMINATE, FAIL, or `dx dy scroll` optionally followed by ` ; ELEMENTS`. dx and dy are raw screen-pixel offsets from the CURRENT cursor (positive right/down). scroll is a signed wheel amount. Elements execute after that line's move and scroll: `+X` presses and `-X` releases a key or LMB/RMB/MMB; `type("...")` types one JSON-escaped literal string. Click by moving and then pressing/releasing. Drag causally with three lines: press at the source, move while held, release at the destination. Use TERMINATE only after the goal is complete and FAIL only when impossible."""


def _visible_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    for index, step in enumerate(row["steps"]):
        observation = step["observation"]
        user_content: list[dict[str, str]] = [
            {"type": "image", "image": observation["image_path"]}
        ]
        if index == 0:
            user_content.append({"type": "text", "text": row["task"]["instruction"]})
        messages.append({"role": "user", "content": user_content})
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": step["compact_action"]}],
            }
        )
    return messages


def build_sft(
    converted_dir: Path, denylist_path: Path, output_dir: Path
) -> dict[str, Any]:
    ensure_empty_output(output_dir)
    converted_manifest = read_json(converted_dir / "manifest.json")
    if (
        not isinstance(converted_manifest, dict)
        or converted_manifest.get("construction_scope") != "train_only"
    ):
        raise ContractError("converted artifact is not train-only")
    converted_path = converted_dir / "converted.jsonl"
    if file_sha256(converted_path) != converted_manifest.get("converted_sha256"):
        raise ContractError("converted.jsonl hash mismatch")
    denylist = load_heldout_denylist(denylist_path)
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "train_validation": []}
    row_manifest: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    task_splits: dict[str, str] = {}
    for converted in iter_jsonl(converted_path):
        body = dict(converted)
        declared_converted_hash = body.pop("converted_row_sha256", None)
        if declared_converted_hash != object_sha256(body):
            raise ContractError(
                f"converted row hash mismatch: {converted.get('rollout_id')}"
            )
        if converted.get("conversion_version") != CONVERSION_VERSION:
            raise ContractError("conversion version mismatch")
        task = converted.get("task")
        if not isinstance(task, dict) or task.get("source_split") != "train":
            raise ContractError("SFT input is not sourced from train")
        split = task.get("split")
        if split not in outputs:
            raise ContractError(f"invalid SFT split: {split!r}")
        previous_split = task_splits.setdefault(task["task_key"], split)
        if previous_split != split:
            raise ContractError(f"task appears in multiple splits: {task['task_key']}")
        image_hashes = [
            step["observation"]["image_sha256"] for step in converted["steps"]
        ]
        for step in converted["steps"]:
            observation = step["observation"]
            verify_declared_hash(
                Path(observation["image_path"]),
                observation["image_sha256"],
                context=f"SFT image {converted['rollout_id']}/{step['step_index']}",
            )
        assert_not_heldout(
            denylist=denylist,
            task_key=task["task_key"],
            source_task_id=task["source_task_id"],
            instruction=task["instruction"],
            asset_hashes=image_hashes,
        )
        for step in converted["steps"]:
            parsed = parse_compact_sequence(step["compact_action"])
            if (
                "\n".join(action.render() for action in parsed)
                != step["compact_action"]
            ):
                raise ContractError("non-canonical compact target during SFT build")
        sample_id = f"teacher_sft_{converted['rollout_id']}"
        if sample_id in seen_ids:
            raise ContractError(f"duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        visible = {
            "sample_id": sample_id,
            "recording_id": converted["rollout_id"],
            "task_key": task["task_key"],
            "split": split,
            "format": CONVERSION_VERSION,
            "messages": _visible_messages(converted),
        }
        visible_hash = object_sha256(visible)
        provenance = {
            "source": task["source"],
            "source_revision": task["source_revision"],
            "source_rollout_sha256": converted["source_rollout_sha256"],
            "task_row_sha256": task["task_row_sha256"],
            "converted_row_sha256": declared_converted_hash,
            "visible_row_sha256": visible_hash,
        }
        row_hash = object_sha256({**visible, "_provenance": provenance})
        provenance["sft_row_sha256"] = row_hash
        row = {**visible, "_provenance": provenance}
        outputs[split].append(row)
        row_manifest.append(
            {
                "sample_id": sample_id,
                "task_key": task["task_key"],
                "split": split,
                "visible_row_sha256": visible_hash,
                "sft_row_sha256": row_hash,
                "source_rollout_sha256": converted["source_rollout_sha256"],
                "image_sha256": image_hashes,
            }
        )
    for split, split_rows in outputs.items():
        split_rows.sort(key=lambda row: row["sample_id"])
        write_jsonl(output_dir / "_normalized" / split / "chat.jsonl", split_rows)
    row_manifest.sort(key=lambda row: row["sample_id"])
    write_jsonl(output_dir / "rows.jsonl", row_manifest)
    train_tasks = {row["task_key"] for row in row_manifest if row["split"] == "train"}
    validation_tasks = {
        row["task_key"] for row in row_manifest if row["split"] == "train_validation"
    }
    if train_tasks & validation_tasks:
        raise ContractError("train/train_validation task leakage")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "teacher_sft_chat",
        "construction_scope": "train_only",
        "validation_scope": "train_derived_task_disjoint",
        "format": CONVERSION_VERSION,
        "system_prompt_sha256": object_sha256(SYSTEM_PROMPT),
        "converted_manifest_sha256": file_sha256(converted_dir / "manifest.json"),
        "heldout_denylist_sha256": denylist["denylist_sha256"],
        "rows_sha256": file_sha256(output_dir / "rows.jsonl"),
        "split_files": {
            split: {
                "path": f"_normalized/{split}/chat.jsonl",
                "sha256": file_sha256(
                    output_dir / "_normalized" / split / "chat.jsonl"
                ),
                "n_rows": len(outputs[split]),
                "n_tasks": len({row["task_key"] for row in outputs[split]}),
            }
            for split in outputs
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
