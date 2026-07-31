#!/usr/bin/env python3
"""Build prose-matched move_rel twins of the immutable Phase-B datasets.

Every assistant history/target turn is converted through the audited C1--C4
action-span contract.  The source arm's prose and all user/image bytes remain
unchanged; only the fixed system grammar and assistant action spans change.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


ARMS = ("prose_keep",)
SPLITS = ("train", "val")
EXPECTED = {"train": 2383, "val": 233}
SW, SH = 1920, 1080
COORD_ACTIONS = {
    "mouse_move", "left_click", "right_click", "middle_click", "double_click",
    "triple_click", "left_click_drag", "mouse_down",
}


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"missing JSONL: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise BuildError(f"blank line: {path}:{line_no}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"malformed JSON: {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise BuildError(f"non-object JSONL row: {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_modules(audit_dir: Path, onpolicy_scripts: Path):
    converter_path = (audit_dir / "action_span_conversion.py").resolve()
    builder_path = (onpolicy_scripts / "build_osworld_format_records.py").resolve()
    for path in (converter_path, builder_path):
        if not path.is_file():
            raise BuildError(f"required audited source missing: {path}")
    # The OSWorld builder imports its own, API-richer action_span_conversion,
    # while this audit intentionally uses the independently pinned converter.
    # Load each by exact path under a distinct module identity so sys.path order
    # and an existing test-process import cannot silently select either copy.
    old_action_module = sys.modules.pop("action_span_conversion", None)
    scripts = str(onpolicy_scripts.resolve())
    old_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != scripts]
        sys.path.insert(0, scripts)
        builder_spec = importlib.util.spec_from_file_location(
            "phaseb_osworld_format_builder", builder_path)
        if builder_spec is None or builder_spec.loader is None:
            raise BuildError(f"cannot load OSWorld converter: {builder_path}")
        osw = importlib.util.module_from_spec(builder_spec)
        builder_spec.loader.exec_module(osw)
    finally:
        sys.path[:] = old_path
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        sys.modules.pop("action_span_conversion", None)
        if old_action_module is not None:
            sys.modules["action_span_conversion"] = old_action_module

    converter_spec = importlib.util.spec_from_file_location(
        "phaseb_audited_action_span_conversion", converter_path)
    if converter_spec is None or converter_spec.loader is None:
        raise BuildError(f"cannot load audited converter: {converter_path}")
    conversion = importlib.util.module_from_spec(converter_spec)
    converter_spec.loader.exec_module(conversion)

    if Path(conversion.__file__).resolve() != converter_path:
        raise BuildError(f"loaded wrong action-span converter: {conversion.__file__}")
    if Path(osw.__file__).resolve() != builder_path:
        raise BuildError(f"loaded wrong OSWorld converter: {osw.__file__}")
    osw._SW, osw._SH = SW, SH
    return conversion, osw


def text_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        raise BuildError("Phase-B contract requires list-valued message content")
    parts = [part for part in content if isinstance(part, dict) and part.get("type") == "text"]
    if len(parts) != 1 or not isinstance(parts[0].get("text"), str):
        raise BuildError("Phase-B contract requires exactly one text part per assistant/system turn")
    return parts


def assistant_texts(record: dict[str, Any]) -> list[str]:
    return [text_parts(message)[0]["text"] for message in record["messages"]
            if message.get("role") == "assistant"]


def outside_action(conversion, text: str) -> tuple[str, str]:
    before, _action, after = conversion.split_assistant_turn(text)
    return before, after


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)


def trace_geometry(osw, collected_root: Path, app: str, task_id: str):
    traj_path = collected_root / app / task_id / "traj.jsonl"
    if not traj_path.is_file():
        raise BuildError(f"source trajectory missing: {traj_path}")
    responses: dict[int, str] = {}
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        step_num = event.get("step_num", 0)
        if isinstance(step_num, int) and step_num >= 1 and step_num not in responses:
            responses[step_num] = event.get("response") or ""
    return osw._telescope({step - 1: response for step, response in responses.items()}, SW, SH)


def normalized_abs_landing(args: dict[str, Any]) -> tuple[int, int] | None:
    action = str(args.get("action", "")).lower()
    coord = args.get("coordinate")
    if action not in COORD_ACTIONS or not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    return round(float(coord[0]) * SW / 1000.0), round(float(coord[1]) * SH / 1000.0)


def relative_landing(osw, action_span: str, cursor_before: list[int]) -> tuple[int, int]:
    try:
        calls = osw.parse_computer_use_tool_calls(action_span)
    except Exception as exc:  # noqa: BLE001
        raise BuildError(f"converted move_rel action does not parse: {action_span!r}: {exc}") from exc
    cursor = [int(cursor_before[0]), int(cursor_before[1])]
    for call in calls:
        args = dict(call.arguments)
        if str(args.get("action", "")).lower() != "move_rel":
            continue
        coord = args.get("coordinate")
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            raise BuildError(f"move_rel lacks a two-vector: {args}")
        cursor[0] += round(float(coord[0]) * SW / 1000.0)
        cursor[1] += round(float(coord[1]) * SH / 1000.0)
    return cursor[0], cursor[1]


def render_relative_action(osw, args: dict[str, Any] | None,
                           cursor_before: list[int], intended_target: list[int]) -> str:
    """Render one absolute action in the one canonical move_rel grammar.

    This deliberately fixes the legacy builder's `left_click_drag` hole: the
    canonical prompt has no such action and specifies drag as mouse_down,
    move_rel, mouse_up.
    """
    if args is None:
        seq = [{"action": "wait", "time": 1}]
    else:
        action = str(args.get("action", "")).lower()
        if action == "terminate":
            seq = [{"action": "terminate",
                    "status": args.get("computer_use_status") or args.get("status") or "success"}]
        else:
            import convert_abs_to_relative as v3  # type: ignore
            relative = v3._normalize_abs_args_to_rel(
                args, cursor_before, intended_target,
                coord_space="normalized", screen=[SW, SH],
            )
            if relative is None:
                seq = [{"action": "wait", "time": 1}]
            else:
                relative = dict(relative)
                delta = relative.pop("coordinate", None)
                move = ({"action": "move_rel", "coordinate": [int(delta[0]), int(delta[1])]}
                        if isinstance(delta, (list, tuple)) and len(delta) == 2
                        and (int(delta[0]) != 0 or int(delta[1]) != 0) else None)
                if action == "mouse_move":
                    seq = [move] if move else [{"action": "wait", "time": 1}]
                elif action == "left_click_drag":
                    seq = [{"action": "mouse_down", "button": "left"}]
                    if move:
                        seq.append(move)
                    seq.append({"action": "mouse_up", "button": "left"})
                elif action in {"left_click", "right_click", "middle_click",
                                "double_click", "triple_click", "mouse_down"}:
                    seq = ([move] if move else []) + [relative]
                else:
                    seq = [relative]
            if not seq:
                seq = [{"action": "wait", "time": 1}]
    import convert_abs_to_relative as v3  # type: ignore
    return v3._render_assistant_text(seq)


def build(*, source_root: Path, out_root: Path, audit_dir: Path,
          onpolicy_scripts: Path, collected_root: Path) -> dict[str, Any]:
    source_root, out_root = source_root.resolve(), out_root.resolve()
    collected_root = collected_root.resolve()
    if not source_root.is_dir() or not collected_root.is_dir():
        raise BuildError("source Phase-B root or collected rollout root is missing")
    if out_root.exists() and any(out_root.iterdir()):
        raise BuildError(f"refusing to overwrite non-empty output: {out_root}")

    conversion, osw = load_modules(audit_dir, onpolicy_scripts)
    if conversion._tests() != 0:
        raise BuildError("action-span C1--C4 regression suite failed")
    fixed_system = osw._FMT_SYSTEM["moverel"]
    if fixed_system != Path("/fast/home/franz.srambical/osworld_parity_split/moverel_system_prompt.txt").read_text():
        raise BuildError("canonical move_rel system prompt disagrees with pinned parity prompt")

    stage = out_root.parent / f".{out_root.name}.building_{os.getpid()}_{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    source_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    output_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    trace_cache: dict[tuple[str, str], Any] = {}
    source_hashes: dict[str, str] = {}
    stats = {
        "records": 0, "assistant_turns": 0, "outside_action_identity": 0,
        "new_numeric_tokens_outside_action": 0, "source_action_trace_match": 0,
        "coordinate_turns": 0, "common_pixel_landing_within_2px": 0,
        "max_common_pixel_linf_error": 0.0, "fallback_turns": 0,
        "prose_turns_retained": 0, "user_image_identity": 0,
        "task_split_order_identity": 0,
    }
    try:
        for arm in ARMS:
            for split in SPLITS:
                src_path = source_root / arm / "_normalized" / split / "chat.jsonl"
                rows = read_jsonl(src_path)
                if len(rows) != EXPECTED[split]:
                    raise BuildError(f"{src_path}: {len(rows)} records, expected {EXPECTED[split]}")
                source_hashes[str(src_path)] = sha256(src_path)
                source_rows[(arm, split)] = rows
                converted_rows = []
                for row_index, source in enumerate(rows, 1):
                    record = copy.deepcopy(source)
                    messages = record.get("messages")
                    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
                        raise BuildError(f"bad message scaffold {src_path}:{row_index}")
                    app, task_id, step = source.get("app"), source.get("task_id"), source.get("step")
                    if not isinstance(app, str) or not isinstance(task_id, str) or not isinstance(step, int):
                        raise BuildError(f"missing app/task/step {src_path}:{row_index}")
                    key = (app, task_id)
                    if key not in trace_cache:
                        trace_cache[key] = trace_geometry(osw, collected_root, app, task_id)
                    per_step = trace_cache[key]

                    text_parts(messages[0])[0]["text"] = fixed_system
                    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
                    first_mapped_step = step - len(assistant_messages) + 1
                    if first_mapped_step < 0:
                        raise BuildError(f"negative history mapping {src_path}:{row_index}")
                    turn_audit = []
                    for turn_index, message in enumerate(assistant_messages):
                        mapped_step = first_mapped_step + turn_index
                        if mapped_step not in per_step:
                            raise BuildError(f"missing telescope step {key} step={mapped_step}")
                        args, cursor_before, intended_target = per_step[mapped_step]
                        part = text_parts(message)[0]
                        old_text = part["text"]
                        old_before, old_action, old_after = conversion.split_assistant_turn(old_text)
                        if not old_action:
                            raise BuildError(f"empty source action span {src_path}:{row_index}")
                        try:
                            old_calls = osw.parse_computer_use_tool_calls(old_action)
                        except Exception:
                            old_calls = []
                        if old_calls and args is not None and dict(old_calls[0].arguments) != dict(args):
                            raise BuildError(
                                f"captured action disagrees with trajectory {key} step={mapped_step}"
                            )
                        if old_calls and args is not None:
                            stats["source_action_trace_match"] += 1
                        new_action = render_relative_action(
                            osw, args, cursor_before, intended_target
                        )
                        new_text = conversion.convert_assistant_turn(
                            old_text, lambda _old, rendered=new_action: rendered, keep_prose=True
                        )
                        new_before, actual_action, new_after = conversion.split_assistant_turn(new_text)
                        if (new_before, new_after) != (old_before, old_after):
                            raise BuildError(f"C1 violation {src_path}:{row_index} turn={turn_index}")
                        if actual_action != new_action:
                            raise BuildError(f"action renderer mismatch {src_path}:{row_index}")
                        if numeric_tokens(new_before + new_after) != numeric_tokens(old_before + old_after):
                            raise BuildError(f"numeric leakage outside action {src_path}:{row_index}")
                        stats["outside_action_identity"] += 1
                        stats["assistant_turns"] += 1
                        if args is None or not old_calls:
                            stats["fallback_turns"] += 1

                        abs_landing = normalized_abs_landing(args or {})
                        rel_landing = relative_landing(osw, new_action, cursor_before)
                        error = None
                        if abs_landing is not None:
                            stats["coordinate_turns"] += 1
                            error = max(abs_landing[0] - rel_landing[0], rel_landing[0] - abs_landing[0],
                                        abs_landing[1] - rel_landing[1], rel_landing[1] - abs_landing[1])
                            error = abs(float(error))
                            stats["max_common_pixel_linf_error"] = max(
                                stats["max_common_pixel_linf_error"], error
                            )
                            if error > 2:
                                raise BuildError(
                                    f"common-pixel landing mismatch {key} step={mapped_step}: "
                                    f"abs={abs_landing} rel={rel_landing} error={error}"
                                )
                            stats["common_pixel_landing_within_2px"] += 1
                        part["text"] = new_text
                        turn_audit.append({
                            "mapped_step": mapped_step,
                            "cursor_before_px": cursor_before,
                            "intended_target_px": intended_target,
                            "absolute_landing_px": list(abs_landing) if abs_landing else None,
                            "relative_landing_px": list(rel_landing),
                            "common_pixel_linf_error": error,
                        })

                    record["format"] = "moverel"
                    record["phaseb_relative_audit"] = turn_audit
                    source_users = [m for m in source["messages"] if m.get("role") == "user"]
                    output_users = [m for m in record["messages"] if m.get("role") == "user"]
                    if source_users != output_users:
                        raise BuildError(f"user/image content changed {src_path}:{row_index}")
                    stats["user_image_identity"] += 1
                    stats["task_split_order_identity"] += 1
                    converted_rows.append(record)
                    stats["records"] += 1
                output_rows[(arm, split)] = converted_rows
                write_jsonl(stage / arm / "_normalized" / split / "chat.jsonl", converted_rows)

        # The full source task split/order and prose must survive unchanged.
        for split in SPLITS:
            source_split = source_rows[("prose_keep", split)]
            output_split = output_rows[("prose_keep", split)]
            for index, (source, output) in enumerate(zip(source_split, output_split), 1):
                for field in ("sample_id", "recording_id", "app", "task_id", "step"):
                    if source.get(field) != output.get(field):
                        raise BuildError(f"task/order mismatch {split}:{index} field={field}")
                old_texts, new_texts = assistant_texts(source), assistant_texts(output)
                if len(old_texts) != len(new_texts):
                    raise BuildError(f"assistant count mismatch {split}:{index}")
                for old, new in zip(old_texts, new_texts):
                    old_before, _old_action, old_after = conversion.split_assistant_turn(old)
                    new_before, _new_action, new_after = conversion.split_assistant_turn(new)
                    if (old_before, old_after) != (new_before, new_after):
                        raise BuildError(f"natural teacher reasoning changed {split}:{index}")
                    if old_before or old_after:
                        stats["prose_turns_retained"] += 1

        report = {
            "status": "pass",
            "artifact": "phaseb_relative_twins",
            "grammar": "move_rel normalized 0-999 tool calls",
            "fixed_system_prompt_sha256": hashlib.sha256(fixed_system.encode()).hexdigest(),
            "counts": {arm: dict(EXPECTED) for arm in ARMS},
            "source_root": str(source_root),
            "collected_root": str(collected_root),
            "source_file_sha256": source_hashes,
            "source_code_sha256": {
                str((audit_dir / "action_span_conversion.py").resolve()): sha256(
                    (audit_dir / "action_span_conversion.py").resolve()),
                str((onpolicy_scripts / "build_osworld_format_records.py").resolve()): sha256(
                    (onpolicy_scripts / "build_osworld_format_records.py").resolve()),
            },
            "converter_regression_groups": {"passing": 7, "total": 7},
            "assistant_outside_action_identity": {
                "passing": stats["outside_action_identity"], "total": stats["assistant_turns"]},
            "new_numeric_tokens_outside_action": {
                "leaking": stats["new_numeric_tokens_outside_action"], "total": stats["assistant_turns"]},
            "source_action_trace_match": stats["source_action_trace_match"],
            "fallback_turns": stats["fallback_turns"],
            "common_pixel_landing": {
                "within_2px": stats["common_pixel_landing_within_2px"],
                "total_coordinate_turns": stats["coordinate_turns"],
                "max_linf_error_px": stats["max_common_pixel_linf_error"],
            },
            "task_split_order_identity": {"passing": stats["task_split_order_identity"],
                                            "total": sum(EXPECTED.values())},
            "user_image_identity": {"passing": stats["user_image_identity"],
                                      "total": sum(EXPECTED.values())},
            "natural_teacher_reasoning_retained": {
                "passing": stats["prose_turns_retained"],
                "total_prose_turns": stats["prose_turns_retained"]},
        }
        (stage / "invariant_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "artifact_type": "phaseb_relative_twins",
            "schema_version": 1,
            "status": "complete",
            "arms": list(ARMS),
            "train_records_per_arm": EXPECTED["train"],
            "val_records_per_arm": EXPECTED["val"],
            "grammar": "move_rel",
            "invariant_report": "invariant_report.json",
        }
        (stage / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        out_root.mkdir(parents=True, exist_ok=True)
        for child in stage.iterdir():
            os.replace(child, out_root / child.name)
        stage.rmdir()
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--onpolicy-scripts", type=Path, required=True)
    parser.add_argument("--collected-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build(source_root=args.source_root, out_root=args.out_root,
                       audit_dir=args.audit_dir, onpolicy_scripts=args.onpolicy_scripts,
                       collected_root=args.collected_root)
    except BuildError as exc:
        print(f"FATAL Phase-B relative invariant: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
