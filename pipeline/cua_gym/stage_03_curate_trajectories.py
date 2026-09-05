"""Curate executed CUA-Gym calls into one canonical action per model turn."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from desktop.execute import BUTTON_MASKS
from desktop.execute.protocol import build_action_request
from desktop.geometry import DisplayGeometry

from grammars.ordered_events_v3_relative_1000_grid_v1.codec import (
    CODEC,
    OrderedEventsV3Action,
)
from pipeline.cua_gym.translate import translate_step

SCREEN = (1920, 1080)
GEOMETRY = DisplayGeometry(desktop_width=SCREEN[0], desktop_height=SCREEN[1])
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_COORDINATE_ACTIONS = {
    "click",
    "double_click",
    "left_click",
    "left_click_drag",
    "mouse_move",
    "right_click",
}
_TYPE_AUXILIARY_FIELDS = {
    "clear",
    "clear_history",
    "clear_screen",
    "clear_text",
    "coordinate",
    "enter",
    "keys",
    "press",
}
_ROLLOUT_FIELDS = {
    "_members",
    "_shard",
    "app",
    "complete",
    "duration_s",
    "finished",
    "instruction",
    "reward",
    "reward_raw",
    "screen",
    "setup_ok",
    "started",
    "steps",
    "steps_taken",
    "task_id",
    "terminated",
    "worker",
}
_EXECUTED_FIELDS = {
    "action",
    "assistant_raw",
    "coordinate_screen",
    "cursor_before",
    "latency_s",
    "member",
    "meta",
    "raw",
    "raw_action_args",
    "screenshot",
    "shard",
    "step",
}
_FAILED_FIELDS = {
    "cursor_before",
    "error",
    "latency_s",
    "member",
    "raw",
    "screenshot",
    "shard",
    "step",
}
_STAT_FIELDS = {
    "excluded_rollouts",
    "executable_targets",
    "executed_calls",
    "logical_targets",
    "multicall_extra_calls",
    "multicall_turns",
    "nonexecutable_calls",
    "nonexecuted_events",
    "reasoning_closed",
    "reasoning_double_open_tool_tag",
    "reasoning_missing_closer",
    "reasoning_prose_after_closer",
    "reasoning_thinking_closer_typo",
    "retained_rollouts",
    "source_events",
    "source_rollouts",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_action(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("raw_action_args")
    if raw is None:
        if set(row) not in (_FAILED_FIELDS, {"step", "error"}):
            raise ValueError(f"invalid nonexecuted source row fields: {sorted(row)}")
        return None
    if not isinstance(raw, dict):
        raise TypeError("raw_action_args must be an object")
    expected_fields = (
        (_EXECUTED_FIELDS - {"latency_s"} | {"sub"})
        if "sub" in row
        else _EXECUTED_FIELDS | ({"terminated"} if "terminated" in row else set())
    )
    if set(row) != expected_fields:
        raise ValueError(f"invalid executed source row fields: {sorted(row)}")
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise TypeError("executed source action requires meta")
    action = raw.get("action")
    if not isinstance(action, str) or row.get("action") != action:
        raise ValueError("source action identity mismatch")
    if action in _COORDINATE_ACTIONS:
        if set(raw) != {"action", "coordinate"}:
            raise ValueError(f"invalid raw fields for {action}: {sorted(raw)}")
        if meta != {"action": action, "pixel": row.get("coordinate_screen")}:
            raise ValueError(f"executed metadata mismatch for {action}")
        return raw
    if action == "type":
        if (
            not {"action", "text"}.issubset(raw)
            or set(raw)
            - {
                "action",
                "text",
            }
            - _TYPE_AUXILIARY_FIELDS
        ):
            raise ValueError(f"invalid raw fields for type: {sorted(raw)}")
        if meta != {"action": "type", "text": raw["text"]}:
            raise ValueError("executed metadata mismatch for type")
        if row.get("coordinate_screen") is not None:
            raise ValueError("type must not carry an executed coordinate")
        return meta
    if action == "wait":
        if set(raw) != {"action", "time"} or set(meta) != {"action", "time"}:
            raise ValueError("wait requires raw and executed time")
        raw_time = raw["time"]
        if (
            isinstance(raw_time, bool)
            or not isinstance(raw_time, (int, float))
            or meta != {"action": "wait", "time": min(float(raw_time), 5.0)}
        ):
            raise ValueError("executed wait must be the source duration capped at 5s")
        if row.get("coordinate_screen") is not None:
            raise ValueError("wait must not carry an executed coordinate")
        return meta
    if raw != meta:
        raise ValueError(f"raw and executed action differ for {action}")
    if row.get("coordinate_screen") is not None:
        raise ValueError(f"{action} must not carry an executed coordinate")
    return raw


def _tool_calls(source: str) -> tuple[str, list[dict[str, Any]], str]:
    matches = list(_TOOL_CALL.finditer(source))
    if not matches:
        raise ValueError("assistant output has no complete tool call")
    prefix = source[: matches[0].start()]
    malformed_tag = prefix.endswith("<")
    if malformed_tag:
        prefix = prefix[:-1]
    if source[matches[-1].end() :].strip():
        raise ValueError("assistant output has text after its final tool call")
    between = "".join(
        source[left.end() : right.start()]
        for left, right in itertools.pairwise(matches)
    )
    if between.strip():
        raise ValueError("assistant output has text between tool calls")
    calls: list[dict[str, Any]] = []
    for match in matches:
        payload = json.loads(match.group(1))
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            raise ValueError("tool call must contain exactly name and arguments")
        if payload["name"] != "computer_use" or not isinstance(
            payload["arguments"], dict
        ):
            raise ValueError("tool call must target computer_use with object arguments")
        calls.append(payload["arguments"])

    kind = "closed"
    if prefix.count("</think>") == 1 and "</thinking>" not in prefix:
        reasoning, trailing = prefix.split("</think>", 1)
        reasoning = reasoning.strip()
        if trailing.strip():
            kind = "prose_after_closer"
    elif prefix.count("</thinking>") == 1 and "</think>" not in prefix:
        reasoning = prefix.replace("</thinking>", "", 1).strip()
        kind = "thinking_closer_typo"
    elif "</think>" not in prefix and "</thinking>" not in prefix:
        reasoning = prefix.strip()
        kind = "missing_closer"
    else:
        raise ValueError("assistant output has an unsupported reasoning envelope")
    if reasoning.startswith("<think>"):
        reasoning = reasoning.removeprefix("<think>").strip()
    if "<think>" in reasoning or "</think>" in reasoning:
        raise ValueError("assistant output has nested reasoning tags")
    if malformed_tag:
        kind = "double_open_tool_tag"
    if not reasoning:
        raise ValueError("assistant output has empty reasoning")
    return reasoning, calls, kind


def _expected_image_identity(
    task_id: str, step: int, sub: int | None
) -> tuple[str, str]:
    suffix = "" if sub is None else f"_{sub}"
    filename = f"step_{step:03d}{suffix}.png"
    return f"screenshots/{filename}", f"{task_id}/{filename}"


def _curate_group(
    task_id: str,
    rows: list[dict[str, Any]],
    source_members: set[str],
    counters: Counter[str],
    dispositions: list[dict[str, Any]],
    pointer_button_mask: int,
    held_keys: set[str],
) -> tuple[dict[str, Any] | None, int, set[str]]:
    step = rows[0].get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"{task_id}: invalid step id {step!r}")
    if any(row.get("step") != step for row in rows):
        raise AssertionError("grouped rows have different step ids")
    expected_subs = [None, *range(1, len(rows))]
    if [row.get("sub") for row in rows] != expected_subs:
        raise ValueError(f"{task_id} step {step}: invalid sub-step sequence")
    actions = [_source_action(row) for row in rows]
    if any(action is None for action in actions):
        if len(rows) != 1 or actions != [None]:
            raise ValueError(f"{task_id} step {step}: mixed executed and failed calls")
        counters["nonexecuted_events"] += 1
        return None, pointer_button_mask, held_keys

    assistant = rows[0].get("assistant_raw")
    if not isinstance(assistant, str) or any(
        row.get("assistant_raw") != assistant or row.get("raw") != assistant
        for row in rows
    ):
        raise ValueError(f"{task_id} step {step}: assistant blobs differ")
    reasoning, parsed_calls, reasoning_kind = _tool_calls(assistant)
    raw_calls = [row["raw_action_args"] for row in rows]
    if parsed_calls != raw_calls:
        raise ValueError(f"{task_id} step {step}: parsed and recorded calls differ")
    counters["executed_calls"] += len(rows)
    counters["logical_targets"] += 1
    if actions == [{"action": "key", "keys": ["center"]}]:
        counters["nonexecutable_calls"] += 1
        dispositions.append(
            {
                "recording_id": task_id,
                "step": step,
                "reason": "source_key_has_no_desktop_execution_identity",
                "source_call_sha256": _canonical_sha256(raw_calls[0]),
            }
        )
        return None, pointer_button_mask, held_keys

    cursor: tuple[int, int] | None = None
    primitives = []
    terminate = None
    for index, (row, arguments) in enumerate(zip(rows, actions, strict=True)):
        sub = None if index == 0 else index
        screenshot, member = _expected_image_identity(task_id, step, sub)
        if (
            row.get("screenshot") != screenshot
            or row.get("member") != member
            or member not in source_members
        ):
            raise ValueError(f"{task_id} step {step}: screenshot identity mismatch")
        before = row.get("cursor_before")
        if not isinstance(before, list) or len(before) != 2:
            raise ValueError(f"{task_id} step {step}: invalid cursor_before")
        before_tuple = tuple(before)
        if cursor is not None and before_tuple != cursor:
            raise ValueError(f"{task_id} step {step}: cursor chain mismatch")
        translation = translate_step(arguments, before, GEOMETRY)
        coordinate = row.get("coordinate_screen")
        if translation.target_pixel is not None:
            if tuple(coordinate or ()) != translation.target_pixel:
                raise ValueError(f"{task_id} step {step}: executed pixel mismatch")
            cursor = translation.target_pixel
        else:
            cursor = before_tuple
        if translation.action.terminate is not None:
            if terminate is not None or index != len(rows) - 1:
                raise ValueError(f"{task_id} step {step}: termination must be final")
            terminate = translation.action.terminate
        primitives.extend(translation.action.primitives)

    counters["executable_targets"] += 1
    counters[f"reasoning_{reasoning_kind}"] += 1
    if len(rows) > 1:
        counters["multicall_turns"] += 1
        counters["multicall_extra_calls"] += len(rows) - 1
    action = OrderedEventsV3Action(
        primitives=tuple(primitives),
        no_op=not primitives,
        terminate=terminate,
    )
    operations = CODEC.compile_action(action, GEOMETRY, tuple(rows[0]["cursor_before"]))
    if operations:
        _, pointer_button_mask, held_keys = build_action_request(
            operations,
            initial_buttons={
                button
                for button, mask in BUTTON_MASKS.items()
                if pointer_button_mask & mask
            },
            initial_keys=held_keys,
        )
    return (
        {
            "step": step,
            "shard": rows[0]["shard"],
            "member": rows[0]["member"],
            "reasoning": reasoning,
            "action": action.to_dict(),
        },
        pointer_button_mask,
        held_keys,
    )


def curate_rollout(
    record: dict[str, Any],
    counters: Counter[str],
    dispositions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if set(record) != _ROLLOUT_FIELDS:
        raise ValueError(f"invalid rollout fields: {sorted(record)}")
    task_id = record["task_id"]
    instruction = record["instruction"]
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be non-empty text")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{task_id}: instruction must be non-empty text")
    if tuple(record["screen"]) != SCREEN:
        raise ValueError(f"{task_id}: screen must be {SCREEN}")
    rows = record["steps"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{task_id}: steps must be non-empty")
    members = record["_members"]
    if not isinstance(members, list) or any(
        not isinstance(member, str) for member in members
    ):
        raise ValueError(f"{task_id}: _members must be a text list")
    member_rows = [row.get("member") for row in rows if "member" in row]
    if members != member_rows or len(members) != len(set(members)):
        raise ValueError(f"{task_id}: member inventory mismatch")
    shard = record["_shard"]
    if not isinstance(shard, str) or any(
        row.get("shard") != shard for row in rows if "shard" in row
    ):
        raise ValueError(f"{task_id}: shard identity mismatch")

    counters["source_events"] += len(rows)
    curated = []
    executed_before = counters["executed_calls"]
    previous_step = -1
    pointer_button_mask = 0
    held_keys: set[str] = set()
    for step, group in itertools.groupby(rows, key=lambda row: row.get("step")):
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
            raise ValueError(f"{task_id}: source steps are not strictly increasing")
        previous_step = step
        item, pointer_button_mask, held_keys = _curate_group(
            task_id,
            list(group),
            set(members),
            counters,
            dispositions,
            pointer_button_mask,
            held_keys,
        )
        if item is not None:
            curated.append(item)
    if not curated:
        if counters["executed_calls"] != executed_before:
            raise ValueError(f"{task_id}: rollout has no representable targets")
        return None
    counters["retained_rollouts"] += 1
    return {
        "task_id": task_id,
        "instruction": instruction,
        "app": record["app"],
        "screen": list(SCREEN),
        "steps": curated,
    }


def build_curated_dataset(source_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    target_path = output_dir / "trajectories.jsonl"
    temporary = output_dir / f".trajectories.{os.getpid()}.jsonl"
    counters: Counter[str] = Counter({field: 0 for field in _STAT_FIELDS})
    exclusions = []
    dispositions: list[dict[str, Any]] = []
    try:
        with (
            source_path.open("rb") as source,
            temporary.open("w", encoding="utf-8") as target,
        ):
            for line_number, raw_line in enumerate(source, 1):
                if not raw_line.strip():
                    raise ValueError(f"blank source row at {source_path}:{line_number}")
                record = json.loads(raw_line)
                counters["source_rollouts"] += 1
                curated = curate_rollout(record, counters, dispositions)
                if curated is None:
                    counters["excluded_rollouts"] += 1
                    exclusions.append(
                        {
                            "recording_id": record.get("task_id"),
                            "reason": "no_executed_actions",
                            "source_rollout_sha256": hashlib.sha256(
                                raw_line.rstrip(b"\r\n")
                            ).hexdigest(),
                        }
                    )
                    continue
                target.write(json.dumps(curated, ensure_ascii=False) + "\n")
        if not counters["source_rollouts"]:
            raise ValueError(f"trajectory file is empty: {source_path}")
        if not counters["retained_rollouts"]:
            raise ValueError("curation retained no rollouts")
        temporary.replace(target_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "artifact_type": "cuagym_stage_03_curated_trajectories",
        "schema_version": 1,
        "trajectories": target_path.name,
        "trajectories_sha256": _sha256(target_path),
        "inputs": {
            "source": str(source_path.resolve()),
            "source_sha256": _sha256(source_path),
        },
        "exclusions": exclusions,
        "exclusions_sha256": _canonical_sha256(exclusions),
        "dispositions": dispositions,
        "dispositions_sha256": _canonical_sha256(dispositions),
        "stats": dict(sorted(counters.items())),
    }
    temporary_manifest = output_dir / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def resolve_curated_artifact(root: Path) -> tuple[Path, dict[str, Any]]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "artifact_type",
        "schema_version",
        "trajectories",
        "trajectories_sha256",
        "inputs",
        "exclusions",
        "exclusions_sha256",
        "dispositions",
        "dispositions_sha256",
        "stats",
    }
    required = {
        "artifact_type": "cuagym_stage_03_curated_trajectories",
        "schema_version": 1,
        "trajectories": "trajectories.jsonl",
    }
    if (
        set(manifest) != expected_fields
        or {key: manifest.get(key) for key in required} != required
    ):
        raise ValueError(f"invalid curated trajectory artifact: {manifest_path}")
    path = root / "trajectories.jsonl"
    expected = manifest.get("trajectories_sha256")
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise ValueError(f"curated trajectory digest mismatch: {path}")
    exclusions = manifest.get("exclusions")
    if not isinstance(exclusions, list) or manifest.get(
        "exclusions_sha256"
    ) != _canonical_sha256(exclusions):
        raise ValueError(f"curated exclusion receipt mismatch: {manifest_path}")
    dispositions = manifest.get("dispositions")
    if not isinstance(dispositions, list) or manifest.get(
        "dispositions_sha256"
    ) != _canonical_sha256(dispositions):
        raise ValueError(f"curated disposition receipt mismatch: {manifest_path}")
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"source", "source_sha256"}
        or not isinstance(inputs.get("source"), str)
        or not Path(inputs["source"]).is_absolute()
        or not isinstance(inputs.get("source_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", inputs["source_sha256"]) is None
    ):
        raise ValueError(f"invalid curated source identity: {manifest_path}")
    stats = manifest.get("stats")
    if (
        not isinstance(stats, dict)
        or set(stats) != _STAT_FIELDS
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in stats.values()
        )
        or stats["source_events"]
        != stats["executed_calls"] + stats["nonexecuted_events"]
        or stats["logical_targets"]
        != stats["executable_targets"] + stats["nonexecutable_calls"]
        or stats["source_rollouts"]
        != stats["retained_rollouts"] + stats["excluded_rollouts"]
        or stats["retained_rollouts"] <= 0
        or stats["executable_targets"] <= 0
    ):
        raise ValueError(f"invalid curated trajectory statistics: {manifest_path}")
    if len(exclusions) != stats["excluded_rollouts"] or any(
        not isinstance(item, dict)
        or set(item) != {"recording_id", "reason", "source_rollout_sha256"}
        or not isinstance(item["recording_id"], str)
        or not item["recording_id"]
        or item["reason"] != "no_executed_actions"
        or not isinstance(item["source_rollout_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", item["source_rollout_sha256"]) is None
        for item in exclusions
    ):
        raise ValueError(f"invalid curated exclusion receipt: {manifest_path}")
    if len(dispositions) != stats["nonexecutable_calls"] or any(
        not isinstance(item, dict)
        or set(item) != {"recording_id", "step", "reason", "source_call_sha256"}
        or not isinstance(item["recording_id"], str)
        or not item["recording_id"]
        or isinstance(item["step"], bool)
        or not isinstance(item["step"], int)
        or item["step"] < 0
        or item["reason"] != "source_key_has_no_desktop_execution_identity"
        or not isinstance(item["source_call_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", item["source_call_sha256"]) is None
        for item in dispositions
    ):
        raise ValueError(f"invalid curated disposition receipt: {manifest_path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise ValueError(f"blank curated row at {path}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"curated row must be an object at {path}:{line_number}")
        rows.append(row)
    if (
        len(rows) != stats["retained_rollouts"]
        or sum(len(row.get("steps", ())) for row in rows) != stats["executable_targets"]
    ):
        raise ValueError(f"curated trajectory counts mismatch: {path}")
    return path, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_curated_dataset(args.source_path, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
