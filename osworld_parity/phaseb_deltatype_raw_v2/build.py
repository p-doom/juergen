#!/usr/bin/env python3
"""Build and audit the exact Phase-B raw-deltatype-v2 dataset."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from action_v2 import format_deltatype_v2, ordered_plan, parse_deltatype_v2
from converter import ConversionError, replace_action_span
from prompt import SYSTEM_PROMPT


SW, SH = 1920, 1080
EXPECTED_RECORDS = {"train": 2383, "val": 233}
EXPECTED_ASSISTANT_SPANS = 10721
EXPECTED_TOOL_CALLS = 11471
EXPECTED_DRAG_SPANS = 444
EXPECTED_LEGACY_SPANS = 10277
EXPECTED_MULTI_CALL_SPANS = 750
EXPECTED_SOURCE_SHA256 = {
    "train": "41f59fa17b866bfca460ae30747a0448c2ed60f542ea91db9f0b068c29ebc2db",
    "val": "866adc0b06ca4badfbec73c47be77ce639b8ea49351e3ee665634864df91c592",
}
EXPECTED_SPLIT_SHA256 = {
    "train": "1a5cb5bf8f27079b50188a3735f5bb7f801b5fd812551d7f328281e6399700ae",
    "heldout": "9bdb3e466738c06d3f372d7ae4ebadb4d4b575175871cb63af0f4c89f8ba7e7c",
}
EXPECTED_CONTRACT_SHA256 = {
    "action_span_conversion.py": "65397c1dcebdd95431bb53918c0117131f24dfc3cd06c5390e4b321202c84497",
    "build_osworld_format_records.py": "28b5cbe1c936d25e3a8871e4de4ff73dc54ab4a5e82061537463af3cdeaf09a5",
    "convert_abs_to_deltatype.py": "e9424d319c18736d043681a857079cb28143081dce720cdd54de0d291e718bf3",
    "production_action_parser.py": "f916757d17e4a5f53627510616ffff411e9109e8737d1309067c6338caae4a9a",
}
EXPECTED_TRAJECTORY_SET_SHA256 = (
    "4ac24eff3069a7bd2bedb8c12fb59ff98807ea6d4ba69b0499ff690f2d226917"
)
EXPECTED_PYTHON_SHA256 = (
    "77f4fced0204779fe0d004881c1c38c08b808eea8516660d2db971007d791cc2"
)
EXPECTED_PYTHON_VERSION = (
    "3.12.9 | packaged by conda-forge | (main, Mar  4 2025, 22:48:41) "
    "[GCC 13.3.0]"
)
EXPECTED_PYTEST_VERSION = "9.0.2"


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_runtime_environment() -> dict[str, str]:
    executable = Path(sys.executable)
    observed = {
        "python_executable": str(executable.resolve()),
        "python_sha256": sha256(executable),
        "python_version": sys.version,
        "pytest_version": importlib.metadata.version("pytest"),
    }
    expected = {
        "python_sha256": EXPECTED_PYTHON_SHA256,
        "python_version": EXPECTED_PYTHON_VERSION,
        "pytest_version": EXPECTED_PYTEST_VERSION,
    }
    mismatches = {
        key: {"observed": observed[key], "expected": value}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise BuildError(f"unpinned Python test/build environment: {mismatches}")
    return observed


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_contract_modules(
    audit_dir: Path, onpolicy_scripts: Path, production_parser: Path
) -> tuple[Any, Any, Any]:
    converter_path = (audit_dir / "action_span_conversion.py").resolve()
    builder_path = (onpolicy_scripts / "build_osworld_format_records.py").resolve()
    old_action = sys.modules.pop("action_span_conversion", None)
    old_path = list(sys.path)
    scripts = str(onpolicy_scripts.resolve())
    try:
        sys.path.insert(0, scripts)
        osw = load_module("raw_v2_osworld_builder", builder_path)
    finally:
        sys.path[:] = old_path
        sys.modules.pop("action_span_conversion", None)
        if old_action is not None:
            sys.modules["action_span_conversion"] = old_action
    conversion = load_module("raw_v2_action_span_conversion", converter_path)
    production = load_module("raw_v2_legacy_parser", production_parser.resolve())
    if conversion._tests() != 0:
        raise BuildError("audited action-span C1-C4 regression suite failed")
    osw._SW, osw._SH = SW, SH
    return conversion, osw, production


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise BuildError(f"blank source line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BuildError(f"non-object source row: {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def text_part(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        raise BuildError("message content must be a list")
    parts = [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    if len(parts) != 1 or not isinstance(parts[0].get("text"), str):
        raise BuildError("assistant/system message must contain exactly one text part")
    return parts[0]


def split_pairs(path: Path) -> set[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (app, task_id)
        for app, task_ids in payload.items()
        for task_id in task_ids
    }


def full_trace_geometry(
    osw: Any, collected_root: Path, app: str, task_id: str
) -> dict[int, list[tuple[dict[str, Any], list[int], list[int]]]]:
    trajectory = collected_root / app / task_id / "traj.jsonl"
    responses: dict[int, str] = {}
    for line in trajectory.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        step = event.get("step_num")
        if isinstance(step, int) and step >= 1 and step not in responses:
            responses[step] = event.get("response") or ""
    cursor = [SW // 2, SH // 2]
    per_step: dict[int, list[tuple[dict[str, Any], list[int], list[int]]]] = {}
    for step in sorted(responses):
        geometries: list[tuple[dict[str, Any], list[int], list[int]]] = []
        for call in osw.parse_computer_use_tool_calls(responses[step]):
            arguments = dict(call.arguments)
            before = list(cursor)
            target = list(cursor)
            action = str(arguments.get("action", "")).lower()
            coordinate = arguments.get("coordinate")
            if (
                action in osw._COORD_ACTIONS
                and isinstance(coordinate, (list, tuple))
                and len(coordinate) == 2
            ):
                target = [
                    max(0, min(SW - 1, round(float(coordinate[0]) * SW / 1000))),
                    max(0, min(SH - 1, round(float(coordinate[1]) * SH / 1000))),
                ]
                cursor = list(target)
            geometries.append((arguments, before, target))
        per_step[step - 1] = geometries
    return per_step


def trajectory_set_hash(
    collected_root: Path, tasks: set[tuple[str, str]]
) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for app, task_id in sorted(tasks):
        relative = f"{app}/{task_id}/traj.jsonl"
        hashes[relative] = sha256(collected_root / relative)
    canonical = "".join(
        f"{digest}  {relative}\n" for relative, digest in hashes.items()
    )
    return hashlib.sha256(canonical.encode()).hexdigest(), hashes


def compose_legacy_label(
    osw: Any, geometries: list[tuple[dict[str, Any], list[int], list[int]]]
) -> str | None:
    actions = tuple(str(item[0].get("action", "")).lower() for item in geometries)
    if "left_click_drag" in actions:
        return None
    if len(geometries) == 1:
        arguments, before, target = geometries[0]
        if actions == ("terminate",):
            status = str(
                arguments.get("computer_use_status")
                or arguments.get("status")
                or "success"
            ).lower()
            return "FAIL" if status == "failure" else "TERMINATE"
        return osw.deltatype_conv.action_to_label(
            arguments, before, target, coord_space="raw", sw=SW, sh=SH
        )
    labels = [
        osw.deltatype_conv.action_to_label(
            arguments, before, target, coord_space="raw", sw=SW, sh=SH
        )
        for arguments, before, target in geometries
    ]
    if any(label is None for label in labels):
        return None
    if actions == ("mouse_move", "scroll"):
        _arguments, before, target = geometries[0]
        scroll = 0 if labels[1] == "NO_OP" else int(labels[1].split()[2])
        dx, dy = target[0] - before[0], target[1] - before[1]
        return "NO_OP" if (dx, dy, scroll) == (0, 0, 0) else f"{dx} {dy} {scroll}"
    if actions == ("type", "key"):
        elements = [label.split(" ; ", 1)[1] for label in labels if " ; " in label]
        return "0 0 0 ; " + " ".join(elements) if elements else "NO_OP"
    return None


def render_drag_label(
    geometries: list[tuple[dict[str, Any], list[int], list[int]]]
) -> tuple[str, tuple[tuple[Any, ...], ...]]:
    actions = tuple(str(item[0].get("action", "")).lower() for item in geometries)
    if actions == ("left_click_drag",):
        _arguments, before, target = geometries[0]
        initial = (0, 0)
        drag = (target[0] - before[0], target[1] - before[1])
        expected = (
            ("press", "LMB"),
            ("moveTo", target[0], target[1], 0.5),
            ("release", "LMB"),
        )
    elif actions == ("mouse_move", "left_click_drag"):
        _move_arguments, before, start = geometries[0]
        _drag_arguments, drag_before, target = geometries[1]
        if start != drag_before:
            raise BuildError("move-then-drag geometry is not contiguous")
        initial = (start[0] - before[0], start[1] - before[1])
        drag = (target[0] - drag_before[0], target[1] - drag_before[1])
        expected_list: list[tuple[Any, ...]] = []
        if start != before:
            expected_list.append(("moveTo", start[0], start[1]))
        expected_list.extend(
            (
                ("press", "LMB"),
                ("moveTo", target[0], target[1], 0.5),
                ("release", "LMB"),
            )
        )
        expected = tuple(expected_list)
    else:
        raise BuildError(f"unsupported drag sequence: {actions}")
    label = (
        f"{initial[0]} {initial[1]} 0 ; +LMB "
        f"MOVE({drag[0]},{drag[1]}) -LMB"
    )
    return label, expected


def legacy_plan(production: Any, label: str, cursor: tuple[int, int]) -> Any:
    action = production.parse_deltatype(label)
    if production.format_deltatype(action) != label:
        raise BuildError(f"legacy parser is not byte-stable: {label!r}")
    if action.no_op or action.terminate or action.fail:
        return ()
    target = (
        max(0, min(SW - 1, cursor[0] + action.dx)),
        max(0, min(SH - 1, cursor[1] + action.dy)),
    )
    plan: list[tuple[Any, ...]] = []
    if target != cursor:
        plan.append(("moveTo", *target))
    if action.scroll:
        plan.append(("scroll", action.scroll))
    for kind, value in action.elements:
        if kind == "type":
            plan.append(("type", value))
        else:
            plan.append((value.kind, value.what))
    return tuple(plan)


def build(
    *,
    source_root: Path,
    collected_root: Path,
    audit_dir: Path,
    onpolicy_scripts: Path,
    production_parser: Path,
    train_split: Path,
    heldout_split: Path,
    output: Path,
) -> dict[str, Any]:
    runtime_environment = pinned_runtime_environment()
    contract_paths = {
        "action_span_conversion.py": audit_dir / "action_span_conversion.py",
        "build_osworld_format_records.py": (
            onpolicy_scripts / "build_osworld_format_records.py"
        ),
        "convert_abs_to_deltatype.py": (
            onpolicy_scripts / "convert_abs_to_deltatype.py"
        ),
        "production_action_parser.py": production_parser,
    }
    contract_hashes = {name: sha256(path) for name, path in contract_paths.items()}
    if contract_hashes != EXPECTED_CONTRACT_SHA256:
        raise BuildError(
            f"contract module hash mismatch: {contract_hashes} "
            f"!= {EXPECTED_CONTRACT_SHA256}"
        )
    split_hashes = {
        "train": sha256(train_split),
        "heldout": sha256(heldout_split),
    }
    if split_hashes != EXPECTED_SPLIT_SHA256:
        raise BuildError(
            f"split hash mismatch: {split_hashes} != {EXPECTED_SPLIT_SHA256}"
        )
    if output.exists() and any(output.iterdir()):
        raise BuildError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    conversion, osw, production = load_contract_modules(
        audit_dir, onpolicy_scripts, production_parser
    )
    train_pairs = split_pairs(train_split)
    heldout_pairs = split_pairs(heldout_split)
    trace_cache: dict[
        tuple[str, str],
        dict[int, list[tuple[dict[str, Any], list[int], list[int]]]],
    ] = {}
    source_tasks_by_split: dict[str, set[tuple[str, str]]] = {
        "train": set(),
        "val": set(),
    }
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    action_sequences: collections.Counter[str] = collections.Counter()
    split_drag_counts: collections.Counter[str] = collections.Counter()
    records = assistant_spans = tool_calls = multi_call_spans = 0
    legacy_spans = drag_spans = exact_plans = 0
    output_orders: dict[str, list[tuple[Any, ...]]] = {}

    for split, expected_count in EXPECTED_RECORDS.items():
        source_path = source_root / "prose_keep" / "_normalized" / split / "chat.jsonl"
        source_rows = read_jsonl(source_path)
        if len(source_rows) != expected_count:
            raise BuildError(f"{split}: {len(source_rows)} records != {expected_count}")
        source_hashes[split] = sha256(source_path)
        if source_hashes[split] != EXPECTED_SOURCE_SHA256[split]:
            raise BuildError(
                f"{split}: source hash {source_hashes[split]} "
                f"!= {EXPECTED_SOURCE_SHA256[split]}"
            )
        converted_rows: list[dict[str, Any]] = []
        order: list[tuple[Any, ...]] = []
        for row_index, source in enumerate(source_rows, 1):
            record = copy.deepcopy(source)
            app, task_id, step = source.get("app"), source.get("task_id"), source.get("step")
            if not isinstance(app, str) or not isinstance(task_id, str) or not isinstance(step, int):
                raise BuildError(f"missing row identity: {split}:{row_index}")
            key = (app, task_id)
            source_tasks_by_split[split].add(key)
            order.append(
                tuple(source.get(field) for field in ("sample_id", "recording_id", "app", "task_id", "step"))
            )
            if key not in trace_cache:
                trace_cache[key] = full_trace_geometry(osw, collected_root, *key)
            geometry = trace_cache[key]
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
                raise BuildError(f"bad message scaffold: {split}:{row_index}")
            text_part(messages[0])["text"] = SYSTEM_PROMPT
            assistants = [message for message in messages if message.get("role") == "assistant"]
            first_step = step - len(assistants) + 1
            row_audit: list[dict[str, Any]] = []
            for turn_index, message in enumerate(assistants):
                mapped_step = first_step + turn_index
                _before_text, action_span, _after_text = conversion.split_assistant_turn(
                    text_part(message)["text"]
                )
                calls = osw.parse_computer_use_tool_calls(action_span)
                geometries = geometry.get(mapped_step)
                if not calls or geometries is None:
                    raise BuildError(
                        f"missing action/geometry: {split}:{row_index}:{mapped_step}"
                    )
                source_arguments = [dict(call.arguments) for call in calls]
                if source_arguments != [item[0] for item in geometries]:
                    raise BuildError(
                        f"source/trajectory mismatch: {split}:{row_index}:{mapped_step}"
                    )
                actions = tuple(
                    str(arguments.get("action", "")).lower()
                    for arguments in source_arguments
                )
                sequence = "+".join(actions)
                action_sequences[sequence] += 1
                assistant_spans += 1
                tool_calls += len(calls)
                multi_call_spans += len(calls) > 1
                cursor_before = tuple(geometries[0][1])
                label = compose_legacy_label(osw, geometries)
                if label is None:
                    label, expected_plan = render_drag_label(geometries)
                    drag_spans += 1
                    split_drag_counts[split] += 1
                else:
                    expected_plan = legacy_plan(production, label, cursor_before)
                    legacy_spans += 1
                action = parse_deltatype_v2(label)
                if format_deltatype_v2(action) != label:
                    raise BuildError(f"v2 label is not byte-stable: {label!r}")
                actual_plan = ordered_plan(action, cursor_before, (SW, SH))
                if actual_plan != expected_plan:
                    raise BuildError(
                        f"command plan mismatch {split}:{row_index}:{mapped_step}: "
                        f"{actual_plan!r} != {expected_plan!r}"
                    )
                exact_plans += 1
                old_text = text_part(message)["text"]
                try:
                    new_text, original_action = replace_action_span(
                        conversion, old_text, label
                    )
                except ConversionError as exc:
                    raise BuildError(
                        f"action-span byte contract failed: {split}:{row_index}"
                    ) from exc
                if original_action != action_span:
                    raise BuildError(
                        f"action-span extraction drift: {split}:{row_index}"
                    )
                text_part(message)["text"] = new_text
                row_audit.append(
                    {
                        "mapped_step": mapped_step,
                        "source_call_count": len(calls),
                        "source_sequence": sequence,
                        "label": label,
                        "command_plan": [list(command) for command in actual_plan],
                    }
                )
            source_users = [message for message in source["messages"] if message.get("role") == "user"]
            output_users = [message for message in record["messages"] if message.get("role") == "user"]
            if source_users != output_users:
                raise BuildError(f"user/image bytes changed: {split}:{row_index}")
            record["format"] = "deltatype_raw_v2"
            record["raw_deltatype_v2_audit"] = row_audit
            converted_rows.append(record)
            records += 1
        output_path = output / split / "chat.jsonl"
        write_jsonl(output_path, converted_rows)
        output_hashes[f"{split}/chat.jsonl"] = sha256(output_path)
        output_orders[split] = order

    source_tasks = source_tasks_by_split["train"] | source_tasks_by_split["val"]
    if not source_tasks <= train_pairs or source_tasks & heldout_pairs:
        raise BuildError("dataset task provenance is not train-only")
    if source_tasks_by_split["train"] & source_tasks_by_split["val"]:
        raise BuildError("train and validation task sets overlap")
    trajectory_digest, trajectory_hashes = trajectory_set_hash(
        collected_root, source_tasks
    )
    if trajectory_digest != EXPECTED_TRAJECTORY_SET_SHA256:
        raise BuildError(
            f"trajectory set hash {trajectory_digest} "
            f"!= {EXPECTED_TRAJECTORY_SET_SHA256}"
        )
    observed = {
        "records": records,
        "assistant_spans": assistant_spans,
        "tool_calls": tool_calls,
        "multi_call_spans": multi_call_spans,
        "legacy_spans": legacy_spans,
        "drag_spans": drag_spans,
        "exact_command_plans": exact_plans,
    }
    expected = {
        "records": sum(EXPECTED_RECORDS.values()),
        "assistant_spans": EXPECTED_ASSISTANT_SPANS,
        "tool_calls": EXPECTED_TOOL_CALLS,
        "multi_call_spans": EXPECTED_MULTI_CALL_SPANS,
        "legacy_spans": EXPECTED_LEGACY_SPANS,
        "drag_spans": EXPECTED_DRAG_SPANS,
        "exact_command_plans": EXPECTED_ASSISTANT_SPANS,
    }
    if observed != expected:
        raise BuildError(f"full-source count mismatch: {observed} != {expected}")
    if split_drag_counts != {"train": 437, "val": 7}:
        raise BuildError(f"drag split mismatch: {dict(split_drag_counts)}")

    manifest = {
        "schema_version": 2,
        "artifact_type": "phaseb_raw_deltatype_v2_dataset",
        "status": "complete",
        "format": "deltatype_raw_v2",
        "grammar": "initial_dx initial_dy 0 ; +LMB MOVE(drag_dx,drag_dy) -LMB",
        "record_counts": EXPECTED_RECORDS,
        **observed,
        "action_sequence_counts": dict(sorted(action_sequences.items())),
        "drag_split_counts": dict(split_drag_counts),
        "source_task_count": len(source_tasks),
        "train_task_count": len(source_tasks_by_split["train"]),
        "val_task_count": len(source_tasks_by_split["val"]),
        "train_val_task_intersection_count": 0,
        "heldout_intersection_count": 0,
        "source_file_sha256": source_hashes,
        "output_file_sha256": output_hashes,
        "split_file_sha256": split_hashes,
        "contract_file_sha256": contract_hashes,
        "trajectory_set_sha256": trajectory_digest,
        "trajectory_file_sha256": trajectory_hashes,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "runtime_environment": runtime_environment,
        "implementation_sha256": {
            "action_v2.py": sha256(Path(__file__).with_name("action_v2.py")),
            "build.py": sha256(Path(__file__)),
            "converter.py": sha256(Path(__file__).with_name("converter.py")),
            "prompt.py": sha256(Path(__file__).with_name("prompt.py")),
            "readiness.py": sha256(Path(__file__).with_name("readiness.py")),
        },
        "source_order_sha256": {
            split: hashlib.sha256(
                json.dumps(order, separators=(",", ":")).encode()
            ).hexdigest()
            for split, order in output_orders.items()
        },
        "legacy_label_byte_invariance": True,
        "legacy_transition_invariance": True,
        "all_drag_command_sequences_exact": True,
        "all_source_calls_consumed": True,
        "production_gpu_training_authorized": False,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--onpolicy-scripts", type=Path, required=True)
    parser.add_argument("--production-parser", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--heldout-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(**vars(args))
    print(
        f"raw deltatype-v2 build PASS: records={manifest['records']} "
        f"spans={manifest['assistant_spans']} calls={manifest['tool_calls']} "
        f"drag_spans={manifest['drag_spans']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
