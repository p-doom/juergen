#!/usr/bin/env python3
"""Build the full-call natural-prose Phase-B move_rel control dataset."""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


SW, SH = 1920, 1080
EXPECTED_RECORDS = {"train": 2383, "val": 233}
EXPECTED_SOURCE_SHA256 = {
    "train": "41f59fa17b866bfca460ae30747a0448c2ed60f542ea91db9f0b068c29ebc2db",
    "val": "866adc0b06ca4badfbec73c47be77ce639b8ea49351e3ee665634864df91c592",
}
EXPECTED_RAW_SHA256 = {
    "train": "5f449f3d57b368e55cfe2ba486bcdd9953aa6f9bad343948e0b8653b2ab4de99",
    "val": "a819011d5f8524cad1980d720fcdbc98a838a37b33de499c46eb4c13c94acadd",
}
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "77085ee3c2ea7d780e96ade76efbffc0746139c0c619a5d9cbcec8562a1a25d5"
)
EXPECTED_SPANS = 10_721
EXPECTED_CALLS = 11_471
EXPECTED_MULTI = 750
EXPECTED_DRAGS = 444
EXPECTED_OUTPUT_CALLS = 18_483
EXPECTED_OUTPUT_SHA256 = {
    "train": "4cc72eb35c845ecd1aad5412ee0872f9be784675452677a88face644236c97aa",
    "val": "b51221df5f044f21092fec6a973c6d8164a7119f2b4971841fc5926df6e9ef7c",
}
EXPECTED_CONTRACT_SHA256 = {
    "action_span_conversion.py": "65397c1dcebdd95431bb53918c0117131f24dfc3cd06c5390e4b321202c84497",
    "build_osworld_format_records.py": "28b5cbe1c936d25e3a8871e4de4ff73dc54ab4a5e82061537463af3cdeaf09a5",
    "convert_abs_to_relative.py": "ef3b2ef2b0e6001878dfc5861f7aaf5bff545701e724eb719a518877c0259ac3",
    "convert_abs_to_moverel.py": "e91c588e1a0bdb0f78e7b7dfe5a66ba6d2b22ea35ff2db96dafe637dda760406",
    "move_rel_format.py": "71545f300295b9e1587c9cddabe0c407641af2fddf76fdaddf92a2fb5d6d3e8d",
    "onpolicy_action_span_conversion.py": "cad0c16cf46e119a09ebf72d45132506533098336b56219567cb045308e559f1",
    "moverel_system_prompt.txt": "01e0cd22431ca94bcd661283c65abd5d1fb3e5897da7981338de9da5a6b0a00d",
}
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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_contract(audit_dir: Path, onpolicy_scripts: Path) -> tuple[Any, Any, Any]:
    converter = (audit_dir / "action_span_conversion.py").resolve()
    builder = (onpolicy_scripts / "build_osworld_format_records.py").resolve()
    v3_path = (onpolicy_scripts / "convert_abs_to_relative.py").resolve()
    moverel_path = (onpolicy_scripts / "convert_abs_to_moverel.py").resolve()
    v2_format_path = Path(
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/"
        "franz.srambical/videocua_moverel/move_rel_format.py"
    ).resolve()
    onpolicy_conversion = (onpolicy_scripts / "action_span_conversion.py").resolve()
    prompt_path = Path("/fast/home/franz.srambical/osworld_parity_split/moverel_system_prompt.txt")
    paths = {
        "action_span_conversion.py": converter,
        "build_osworld_format_records.py": builder,
        "convert_abs_to_relative.py": v3_path,
        "convert_abs_to_moverel.py": moverel_path,
        "move_rel_format.py": v2_format_path,
        "onpolicy_action_span_conversion.py": onpolicy_conversion,
        "moverel_system_prompt.txt": prompt_path,
    }
    for path in paths.values():
        if not path.is_file():
            raise BuildError(f"contract source missing: {path}")
    bad_hashes = {name: (sha256(path), EXPECTED_CONTRACT_SHA256[name])
                  for name, path in paths.items()
                  if sha256(path) != EXPECTED_CONTRACT_SHA256[name]}
    if bad_hashes:
        raise BuildError(f"contract source hash changed: {bad_hashes}")
    old_action = sys.modules.pop("action_span_conversion", None)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(onpolicy_scripts.resolve()))
        osw = load_module("phaseb_normalized_v2_osw", builder)
    finally:
        sys.path[:] = old_path
        sys.modules.pop("action_span_conversion", None)
        if old_action is not None:
            sys.modules["action_span_conversion"] = old_action
    conversion = load_module("phaseb_normalized_v2_conversion", converter)
    scripts = str(onpolicy_scripts.resolve())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    moverel = load_module("phaseb_normalized_v2_moverel", moverel_path)
    imported_conversion = sys.modules.get(moverel.convert_assistant_turn.__module__)
    if (Path(moverel.v3.__file__).resolve() != v3_path
            or Path(moverel.v2enc.__file__).resolve() != v2_format_path
            or imported_conversion is None
            or Path(imported_conversion.__file__).resolve() != onpolicy_conversion):
        raise BuildError("exact move_rel converter imported an unpinned dependency")
    if conversion._tests() != 0:
        raise BuildError("action-span regression contract failed")
    osw._SW, osw._SH = SW, SH
    return conversion, osw, moverel


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise BuildError(f"blank line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BuildError(f"non-object row: {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def text_part(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        raise BuildError("message content is not a list")
    parts = [part for part in content
             if isinstance(part, dict) and part.get("type") == "text"]
    if len(parts) != 1 or not isinstance(parts[0].get("text"), str):
        raise BuildError("message lacks exactly one text part")
    return parts[0]


def trace_geometry(osw: Any, collected: Path, app: str, task: str) -> dict[int, list[Any]]:
    responses: dict[int, str] = {}
    path = collected / app / task / "traj.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        step = event.get("step_num")
        if isinstance(step, int) and step >= 1 and step not in responses:
            responses[step] = event.get("response") or ""
    cursor = [SW // 2, SH // 2]
    result: dict[int, list[Any]] = {}
    for step in sorted(responses):
        items = []
        for call in osw.parse_computer_use_tool_calls(responses[step]):
            args = dict(call.arguments)
            before, target = list(cursor), list(cursor)
            action, coordinate = str(args.get("action", "")).lower(), args.get("coordinate")
            if (action in COORD_ACTIONS and isinstance(coordinate, (list, tuple))
                    and len(coordinate) == 2):
                target = [
                    max(0, min(SW - 1, round(float(coordinate[0]) * SW / 1000))),
                    max(0, min(SH - 1, round(float(coordinate[1]) * SH / 1000))),
                ]
                cursor = list(target)
            items.append((args, before, target))
        result[step - 1] = items
    return result


def render_full(osw: Any, moverel: Any,
                geometries: list[Any]) -> tuple[str, int]:
    rendered = []
    source_calls_validated = 0
    for args, before, target in geometries:
        source_action = str(args.get("action", "")).lower()
        if source_action == "terminate":
            converted = [{"action": "terminate", "status": (
                args.get("computer_use_status") or args.get("status") or "success"
            )}]
        else:
            converted = moverel.absolute_to_moverel_args(
                args, before, target, screen=[SW, SH]
            )
        if converted is None:
            raise BuildError(f"exact move_rel converter rejected source call: {args}")
        converted = [dict(item) for item in converted]
        if (source_action == "left_click_drag"
                and not any(item.get("action") == "move_rel" for item in converted)):
            if [item.get("action") for item in converted] != ["mouse_down", "mouse_up"]:
                raise BuildError(f"unexpected zero-drag plan: {converted}")
            converted.insert(1, {"action": "move_rel", "coordinate": [0, 0]})
        segment = moverel.v3._render_assistant_text(converted)
        actual = [dict(call.arguments)
                  for call in osw.parse_computer_use_tool_calls(segment)]
        if actual != converted:
            raise BuildError(f"rendered ordered call plan changed: {actual} != {converted}")
        rendered.append(segment)
        source_calls_validated += 1
    return "\n".join(rendered), source_calls_validated


def simulate(osw: Any, action_span: str, start: list[int]) -> tuple[list[int], int]:
    cursor = list(start)
    calls = osw.parse_computer_use_tool_calls(action_span)
    for call in calls:
        args = dict(call.arguments)
        if str(args.get("action", "")).lower() == "move_rel":
            coordinate = args.get("coordinate")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise BuildError("move_rel without two-vector")
            cursor[0] += round(float(coordinate[0]) * SW / 1000)
            cursor[1] += round(float(coordinate[1]) * SH / 1000)
            cursor[0] = max(0, min(SW - 1, cursor[0]))
            cursor[1] = max(0, min(SH - 1, cursor[1]))
    return cursor, len(calls)


def validate_trajectory_set(raw_manifest: dict[str, Any], collected_root: Path) -> dict[str, Any]:
    expected = raw_manifest.get("trajectory_file_sha256")
    if not isinstance(expected, dict) or not expected:
        raise BuildError("trusted raw-v2 manifest lacks trajectory file seals")
    observed: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        path = collected_root / relative
        actual = sha256(path)
        if actual != digest:
            raise BuildError(f"mutable trajectory changed: {relative}")
        observed[relative] = actual
    canonical = "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(observed.items())
    )
    set_hash = hashlib.sha256(canonical.encode()).hexdigest()
    if (set_hash != raw_manifest.get("trajectory_set_sha256")
            or set_hash != "4ac24eff3069a7bd2bedb8c12fb59ff98807ea6d4ba69b0499ff690f2d226917"):
        raise BuildError(f"trajectory-set seal mismatch: {set_hash}")
    return {"files": len(observed), "trajectory_set_sha256": set_hash}


def build(source_root: Path, raw_twin: Path, collected_root: Path, audit_dir: Path,
          onpolicy_scripts: Path, output: Path) -> dict[str, Any]:
    conversion, osw, moverel = load_contract(audit_dir, onpolicy_scripts)
    raw_manifest_path = raw_twin / "dataset_manifest.json"
    if sha256(raw_manifest_path) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise BuildError("trusted raw-v2 dataset manifest changed")
    raw_manifest = json.loads(raw_manifest_path.read_text())
    trajectory_seal = validate_trajectory_set(raw_manifest, collected_root)
    system = Path("/fast/home/franz.srambical/osworld_parity_split/moverel_system_prompt.txt").read_text()
    if system != osw._FMT_SYSTEM["moverel"] or system != moverel.MOVEREL_SYSTEM_PROMPT:
        raise BuildError("move_rel system prompt is not pinned parity prompt")
    if output.exists() and any(output.iterdir()):
        raise BuildError(f"refusing to overwrite non-empty output: {output}")
    stage = output.parent / f".{output.name}.building_{os.getpid()}_{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    stats = {"records": 0, "assistant_spans": 0, "source_tool_calls": 0,
             "source_multi_call_spans": 0, "source_drag_spans": 0,
             "output_tool_calls": 0, "content_twin_rows": 0,
             "prose_twin_spans": 0, "semantic_cursor_spans": 0,
             "ordered_source_calls_validated": 0}
    trace_cache: dict[tuple[str, str], Any] = {}
    action_sequences: collections.Counter[str] = collections.Counter()
    source_hashes, raw_hashes, output_hashes = {}, {}, {}
    try:
        for split, expected in EXPECTED_RECORDS.items():
            source_path = source_root / "prose_keep/_normalized" / split / "chat.jsonl"
            raw_path = raw_twin / split / "chat.jsonl"
            if sha256(source_path) != EXPECTED_SOURCE_SHA256[split]:
                raise BuildError(f"{split}: absolute source hash changed")
            if sha256(raw_path) != EXPECTED_RAW_SHA256[split]:
                raise BuildError(f"{split}: raw-v2 twin hash changed")
            source_hashes[split], raw_hashes[split] = sha256(source_path), sha256(raw_path)
            source_rows, raw_rows = read_jsonl(source_path), read_jsonl(raw_path)
            if len(source_rows) != expected or len(raw_rows) != expected:
                raise BuildError(f"{split}: record count mismatch")
            converted: list[dict[str, Any]] = []
            for index, (source, raw) in enumerate(zip(source_rows, raw_rows, strict=True), 1):
                for field in ("sample_id", "recording_id", "app", "task_id", "step"):
                    if source.get(field) != raw.get(field):
                        raise BuildError(f"{split}:{index}: raw/absolute identity differs: {field}")
                source_meta = {k: v for k, v in source.items()
                               if k not in {"messages", "format", "raw_deltatype_v2_audit"}}
                raw_meta = {k: v for k, v in raw.items()
                            if k not in {"messages", "format", "raw_deltatype_v2_audit"}}
                if source_meta != raw_meta:
                    raise BuildError(f"{split}:{index}: raw/absolute non-message fields differ")
                source_users = [m for m in source["messages"] if m.get("role") == "user"]
                raw_users = [m for m in raw["messages"] if m.get("role") == "user"]
                if source_users != raw_users:
                    raise BuildError(f"{split}:{index}: raw/absolute user or image content differs")
                record = copy.deepcopy(source)
                text_part(record["messages"][0])["text"] = system
                src_assist = [m for m in source["messages"] if m.get("role") == "assistant"]
                raw_assist = [m for m in raw["messages"] if m.get("role") == "assistant"]
                dst_assist = [m for m in record["messages"] if m.get("role") == "assistant"]
                if not (len(src_assist) == len(raw_assist) == len(dst_assist)):
                    raise BuildError(f"{split}:{index}: assistant history mismatch")
                raw_audit = raw.get("raw_deltatype_v2_audit")
                if not isinstance(raw_audit, list) or len(raw_audit) != len(src_assist):
                    raise BuildError(f"{split}:{index}: raw-v2 assistant audit mismatch")
                app, task, step = source["app"], source["task_id"], source["step"]
                key = (app, task)
                if key not in trace_cache:
                    trace_cache[key] = trace_geometry(osw, collected_root, app, task)
                first_step = step - len(src_assist) + 1
                audit = []
                for turn, (src_msg, raw_msg, dst_msg, raw_item) in enumerate(
                        zip(src_assist, raw_assist, dst_assist, raw_audit, strict=True)):
                    mapped = first_step + turn
                    src_text, raw_text = text_part(src_msg)["text"], text_part(raw_msg)["text"]
                    src_before, src_action, src_after = conversion.split_assistant_turn(src_text)
                    raw_before, _raw_action, raw_after = conversion.split_assistant_turn(raw_text)
                    if (src_before, src_after) != (raw_before, raw_after):
                        raise BuildError(f"{split}:{index}:{mapped}: raw/absolute prose differs")
                    calls = osw.parse_computer_use_tool_calls(src_action)
                    geometries = trace_cache[key].get(mapped)
                    if not calls or geometries is None:
                        raise BuildError(f"{split}:{index}:{mapped}: source geometry missing")
                    source_args = [dict(call.arguments) for call in calls]
                    if source_args != [item[0] for item in geometries]:
                        raise BuildError(f"{split}:{index}:{mapped}: full source calls differ from trajectory")
                    actions = [str(item.get("action", "")).lower() for item in source_args]
                    sequence = "+".join(actions)
                    if (raw_item.get("mapped_step") != mapped
                            or raw_item.get("source_call_count") != len(source_args)
                            or raw_item.get("source_sequence") != sequence):
                        raise BuildError(f"{split}:{index}:{mapped}: raw-v2 source audit differs")
                    new_action, calls_validated = render_full(osw, moverel, geometries)
                    endpoint, output_calls = simulate(osw, new_action, geometries[0][1])
                    if endpoint != geometries[-1][2]:
                        raise BuildError(f"{split}:{index}:{mapped}: relative endpoint mismatch")
                    new_text = conversion.convert_assistant_turn(
                        src_text, lambda _old, value=new_action: value, keep_prose=True)
                    before, actual, after = conversion.split_assistant_turn(new_text)
                    if (before, after) != (src_before, src_after) or actual != new_action:
                        raise BuildError(f"{split}:{index}:{mapped}: C1 action-span contract failed")
                    text_part(dst_msg)["text"] = new_text
                    action_sequences[sequence] += 1
                    stats["assistant_spans"] += 1
                    stats["source_tool_calls"] += len(source_args)
                    stats["source_multi_call_spans"] += len(source_args) > 1
                    stats["source_drag_spans"] += "left_click_drag" in actions
                    stats["output_tool_calls"] += output_calls
                    stats["prose_twin_spans"] += 1
                    stats["semantic_cursor_spans"] += 1
                    stats["ordered_source_calls_validated"] += calls_validated
                    audit.append({"mapped_step": mapped, "source_call_count": len(source_args),
                                  "source_sequence": sequence,
                                  "output_call_count": output_calls})
                record["format"] = "move_rel_full_v2"
                record["phaseb_normalized_v2_audit"] = audit
                stats["records"] += 1
                stats["content_twin_rows"] += 1
                converted.append(record)
            out_path = stage / "_normalized" / split / "chat.jsonl"
            write_jsonl(out_path, converted)
            output_hashes[split] = sha256(out_path)
            if output_hashes[split] != EXPECTED_OUTPUT_SHA256[split]:
                raise BuildError(
                    f"{split}: deterministic normalized-v2 output hash changed: "
                    f"{output_hashes[split]}"
                )
        if dict(sorted(action_sequences.items())) != raw_manifest.get("action_sequence_counts"):
            raise BuildError("source action-sequence counts differ from trusted raw-v2 audit")
        expected_stats = {"records": 2616, "assistant_spans": EXPECTED_SPANS,
                          "source_tool_calls": EXPECTED_CALLS,
                          "source_multi_call_spans": EXPECTED_MULTI,
                          "source_drag_spans": EXPECTED_DRAGS,
                          "content_twin_rows": 2616, "prose_twin_spans": EXPECTED_SPANS,
                          "semantic_cursor_spans": EXPECTED_SPANS}
        expected_stats["ordered_source_calls_validated"] = EXPECTED_CALLS
        expected_stats["output_tool_calls"] = EXPECTED_OUTPUT_CALLS
        bad = {key: (stats[key], value) for key, value in expected_stats.items()
               if stats[key] != value}
        if bad:
            raise BuildError(f"full-call normalized-v2 audit mismatch: {bad}")
        manifest = {
            "artifact_type": "phaseb_normalized_move_rel_v2_dataset",
            "schema_version": 1, "status": "complete", "format": "move_rel_full_v2",
            **stats, "source_file_sha256": source_hashes,
            "raw_twin_file_sha256": raw_hashes, "output_file_sha256": output_hashes,
            "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "contract_source_sha256": EXPECTED_CONTRACT_SHA256,
            "trajectory_seal": trajectory_seal,
            "action_sequence_counts": dict(sorted(action_sequences.items())),
            "logical_semantics_limit": (
                "exact ordered calls/payloads/endpoints; zero move_rel is explicit, but the "
                "current VM suppresses its pyautogui call and move_rel has no raw drag duration, "
                "so exact runtime transition/timing equivalence is not claimed"
            ),
            "full_source_calls_preserved_in_audit": True,
            "calls_0_collapse_forbidden": True,
        }
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        manifest["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        (stage / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        output.mkdir(parents=True, exist_ok=True)
        for child in stage.iterdir():
            os.replace(child, output / child.name)
        stage.rmdir()
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--raw-twin", type=Path, required=True)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--onpolicy-scripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(**vars(args))
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL normalized-v2 full-call build audit: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
