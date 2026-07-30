#!/usr/bin/env python3
"""Build the relative half of the synthetic 2x2x2 factorial.

The absolute r3data_2k records are the immutable source of record order, image
paths, scenes, and assistant prose.  Only the assistant action span is converted
with audit_operand/action_span_conversion.py; system and user text deliberately
change to the exact relative rung2_scene evaluation prompts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


ARMS = {
    "reltool_act": ("abstool_act", "move_rel", False),
    "relraw_act": ("absraw_act", "deltatype_raw", False),
    "reltool_pre": ("abstool_pre", "move_rel", True),
    "relraw_pre": ("absraw_pre", "deltatype_raw", True),
}
SPLITS = ("train", "val")
EXPECTED_COUNTS = {"train": 2000, "val": 200}


class BuildError(RuntimeError):
    pass


def _load_modules(audit_dir: Path):
    audit_dir = audit_dir.resolve()
    required = (audit_dir / "action_span_conversion.py", audit_dir / "rung2_scene.py")
    for path in required:
        if not path.is_file():
            raise BuildError(f"required audited source missing: {path}")
    sys.path.insert(0, str(audit_dir))
    import action_span_conversion as conversion  # type: ignore
    import rung2_scene as rung2  # type: ignore

    if Path(conversion.__file__).resolve() != required[0]:
        raise BuildError(f"loaded wrong converter: {conversion.__file__}")
    if Path(rung2.__file__).resolve() != required[1]:
        raise BuildError(f"loaded wrong eval prompt source: {rung2.__file__}")
    return conversion, rung2


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"missing required JSONL: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise BuildError(f"blank line in {path}:{line_no}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"malformed JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise BuildError(f"non-object record in {path}:{line_no}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assistant_text(record: dict[str, Any]) -> str:
    try:
        message = record["messages"][-1]
        content = message["content"]
        if message["role"] != "assistant" or len(content) != 1:
            raise KeyError
        item = content[0]
        if item["type"] != "text" or not isinstance(item["text"], str):
            raise KeyError
        return item["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BuildError(f"record {record.get('sample_id')!r} has no single text assistant turn") from exc


def _image_path(record: dict[str, Any]) -> str:
    try:
        items = record["messages"][1]["content"]
        images = [x["image"] for x in items if x.get("type") == "image"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BuildError(f"record {record.get('sample_id')!r} has malformed user content") from exc
    if len(images) != 1 or not isinstance(images[0], str):
        raise BuildError(f"record {record.get('sample_id')!r} must have exactly one image")
    return images[0]


def _action_for(rung2, grammar: str, scene: dict[str, Any]) -> str:
    cursor = tuple(scene["cursor"])
    target = tuple(scene["target_center"])
    g = rung2.GRAMMARS[grammar]
    dx, dy = rung2.ideal(g["space"], cursor, target)
    if grammar == "move_rel":
        return (
            '<tool_call>\n{"name": "computer_use", "arguments": '
            f'{{"action": "move_rel", "coordinate": [{dx}, {dy}]}}}}\n</tool_call>'
        )
    if grammar == "deltatype_raw":
        return f"{dx} {dy} 0 ; +LMB -LMB"
    raise BuildError(f"unsupported relative grammar: {grammar}")


def _absolute_action_for(rung2, source_arm: str, scene: dict[str, Any]) -> str:
    target = tuple(scene["target_center"])
    if source_arm.startswith("abstool"):
        x, y = rung2.to_norm(*target)
        return (
            '<tool_call>\n{"name": "computer_use", "arguments": '
            f'{{"action": "left_click", "coordinate": [{x}, {y}]}}}}\n</tool_call>'
        )
    return f"{target[0]} {target[1]} 0 ; +LMB -LMB"


def _scene_key(scene: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(scene["cursor"]), tuple(scene["bbox"])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(*, source_root: Path, out_root: Path, audit_dir: Path, eval_scenes: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    out_root = out_root.resolve()
    eval_scenes = eval_scenes.resolve()
    if not source_root.is_dir():
        raise BuildError(f"source r3data_2k directory missing: {source_root}")
    if not eval_scenes.is_file():
        raise BuildError(f"seed-0 eval scenes missing: {eval_scenes}")
    if (out_root / "build_manifest.json").exists() or any((out_root / arm).exists() for arm in ARMS):
        raise BuildError(f"refusing to overwrite an existing build: {out_root}")

    conversion, rung2 = _load_modules(audit_dir)
    if conversion._tests() != 0:
        raise BuildError("audited converter regression suite failed")

    source_scenes: dict[str, list[dict[str, Any]]] = {}
    scene_by_id: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = source_root / f"scenes_{split}.jsonl"
        scenes = _read_jsonl(path)
        expected = EXPECTED_COUNTS[split]
        if len(scenes) != expected:
            raise BuildError(f"{path} has {len(scenes)} scenes, expected {expected}")
        ids = [s.get("scene_id") for s in scenes]
        if len(set(ids)) != len(ids):
            raise BuildError(f"duplicate scene_id in {path}")
        source_scenes[split] = scenes
        scene_by_id.update({s["scene_id"]: s for s in scenes})

    eval_rows = _read_jsonl(eval_scenes)
    eval_geometry = {_scene_key(row) for row in eval_rows}
    leak_counts = {}
    for split in SPLITS:
        overlap = {_scene_key(s) for s in source_scenes[split]} & eval_geometry
        leak_counts[split] = len(overlap)
        if overlap:
            raise BuildError(f"geometry leak: {split} intersects seed-0 eval in {len(overlap)} scenes")

    stage = out_root / f".building_{os.getpid()}_{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    sources: dict[tuple[str, str], list[dict[str, Any]]] = {}
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    outside_identity = 0
    gold_parse_and_land = 0
    prompt_equality = 0
    total_records = 0
    source_hashes: dict[str, str] = {}
    try:
        for arm, (source_arm, grammar, preamble) in ARMS.items():
            g = rung2.GRAMMARS[grammar]
            for split in SPLITS:
                src_path = source_root / source_arm / "_normalized" / split / "chat.jsonl"
                rows = _read_jsonl(src_path)
                sources[(arm, split)] = rows
                source_hashes[str(src_path)] = _sha256(src_path)
                expected_scenes = source_scenes[split]
                if len(rows) != EXPECTED_COUNTS[split]:
                    raise BuildError(f"{src_path} has {len(rows)} records, expected {EXPECTED_COUNTS[split]}")
                out_rows = []
                for index, (src, ordered_scene) in enumerate(zip(rows, expected_scenes, strict=True)):
                    sid = src.get("scene_id")
                    if sid != ordered_scene["scene_id"]:
                        raise BuildError(f"scene order mismatch {src_path}:{index + 1}: {sid!r}")
                    scene = scene_by_id[sid]
                    for field in ("recording_id", "scene_id", "kind"):
                        expected_value = scene["scene_id"] if field == "recording_id" else scene[field]
                        if src.get(field) != expected_value:
                            raise BuildError(f"source metadata mismatch {src_path}:{index + 1} field={field}")
                    image = _image_path(src)
                    if image != scene["image_path"] or not Path(image).is_file():
                        raise BuildError(f"source image mismatch/missing {src_path}:{index + 1}: {image}")

                    old_text = _assistant_text(src)
                    old_before, old_action, old_after = conversion.split_assistant_turn(old_text)
                    expected_old_action = _absolute_action_for(rung2, source_arm, scene)
                    if old_action != expected_old_action:
                        raise BuildError(
                            f"absolute source action mismatch {src_path}:{index + 1}: "
                            f"{old_action!r} != {expected_old_action!r}"
                        )
                    relative_action = _action_for(rung2, grammar, scene)
                    # Non-negotiable: every assistant conversion goes through the audited converter.
                    new_text = conversion.convert_assistant_turn(
                        old_text, lambda _old, action=relative_action: action, keep_prose=True
                    )
                    new_before, new_action, new_after = conversion.split_assistant_turn(new_text)
                    if new_before != old_before or new_after != old_after:
                        raise BuildError(f"assistant prefix/suffix changed {src_path}:{index + 1}")
                    if new_action != relative_action:
                        raise BuildError(f"converted action mismatch {src_path}:{index + 1}")
                    outside_identity += 1

                    rec = copy.deepcopy(src)
                    rec["sample_id"] = f"{arm}_{sid}"
                    rec["format"] = arm
                    rec["messages"][0]["content"][0]["text"] = g["system"]
                    expected_user = rung2.build_user_text(g, scene, False, preamble)
                    rec["messages"][1]["content"][-1]["text"] = expected_user
                    rec["messages"][-1]["content"][0]["text"] = new_text
                    if rec["messages"][0]["content"][0]["text"] != g["system"]:
                        raise BuildError(f"system prompt differs from rung2 eval {arm}/{split}/{sid}")
                    if rec["messages"][1]["content"][-1]["text"] != rung2.build_user_text(
                        g, scene, False, preamble
                    ):
                        raise BuildError(f"user prompt differs from rung2 eval {arm}/{split}/{sid}")
                    prompt_equality += 1

                    move = g["parse"](new_text, None)
                    scored = rung2.score_row(g, scene, move.coord, move.parse_ok, new_text)
                    if not (move.parse_ok and scored["in_box"]):
                        raise BuildError(f"gold relative action does not parse/land {arm}/{split}/{sid}: {scored}")
                    gold_parse_and_land += 1
                    total_records += 1
                    out_rows.append(rec)
                outputs[(arm, split)] = out_rows
                _write_jsonl(stage / arm / "_normalized" / split / "chat.jsonl", out_rows)

        # Exact record/image/order matching against each absolute source twin.
        exact_matching = 0
        for arm, (source_arm, _grammar, _pre) in ARMS.items():
            for split in SPLITS:
                for index, (src, dst) in enumerate(zip(sources[(arm, split)], outputs[(arm, split)], strict=True)):
                    if src["recording_id"] != dst["recording_id"] or src["scene_id"] != dst["scene_id"]:
                        raise BuildError(f"record identity changed {arm}/{split}/{index}")
                    if src["kind"] != dst["kind"] or _image_path(src) != _image_path(dst):
                        raise BuildError(f"record image/order changed {arm}/{split}/{index}")
                    exact_matching += 1

        # Prose must be grammar-independent; no digit may leak the target.
        prose_identity = 0
        preamble_digit_leaks = 0
        action_twin_identity = 0
        for split in SPLITS:
            for pre_a, pre_b in zip(outputs[("reltool_pre", split)], outputs[("relraw_pre", split)], strict=True):
                pa, _, _ = conversion.split_assistant_turn(_assistant_text(pre_a))
                pb, _, _ = conversion.split_assistant_turn(_assistant_text(pre_b))
                if pa != pb:
                    raise BuildError(f"prose differs across grammars {split}/{pre_a['scene_id']}")
                prose_identity += 1
                preamble_digit_leaks += int(bool(re.search(r"\d", pa)))
            for action_arm, pre_arm in (("reltool_act", "reltool_pre"), ("relraw_act", "relraw_pre")):
                for act, pre in zip(outputs[(action_arm, split)], outputs[(pre_arm, split)], strict=True):
                    _, action_a, _ = conversion.split_assistant_turn(_assistant_text(act))
                    _, action_p, _ = conversion.split_assistant_turn(_assistant_text(pre))
                    if action_a != action_p:
                        raise BuildError(f"action span differs across preamble twins {split}/{act['scene_id']}")
                    action_twin_identity += 1
        if preamble_digit_leaks:
            raise BuildError(f"preamble digit leak in {preamble_digit_leaks} records")

        report = {
            "status": "pass",
            "converter_regression_groups": {"passing": 7, "total": 7},
            "counts": {arm: dict(EXPECTED_COUNTS) for arm in ARMS},
            "source_root": str(source_root),
            "source_file_sha256": source_hashes,
            "eval_scenes": str(eval_scenes),
            "geometry_leak": {
                "train_vs_seed0_eval": leak_counts["train"],
                "val_vs_seed0_eval": leak_counts["val"],
            },
            "exact_record_image_order_matching": {"passing": exact_matching, "total": total_records},
            "assistant_outside_action_identity": {"passing": outside_identity, "total": total_records},
            "prose_grammar_identity": {"passing": prose_identity, "total": sum(EXPECTED_COUNTS.values())},
            "preamble_digit_leak": {"leaking": preamble_digit_leaks, "total": sum(EXPECTED_COUNTS.values())},
            "action_span_identity_across_preamble_twins": {
                "passing": action_twin_identity,
                "total": 2 * sum(EXPECTED_COUNTS.values()),
            },
            "gold_relative_action_parse_and_land": {"passing": gold_parse_and_land, "total": total_records},
            "prompt_equality_to_rung2_eval": {"passing": prompt_equality, "total": total_records},
            "arms": {
                arm: {"absolute_twin": src, "grammar": grammar, "preamble": pre}
                for arm, (src, grammar, pre) in ARMS.items()
            },
        }
        (stage / "invariant_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "artifact_type": "synthetic_relative_factorial_2x2",
            "schema_version": 1,
            "status": "complete",
            "counts": {"train_per_arm": 2000, "val_per_arm": 200},
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
    parser.add_argument("--eval-scenes", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build(
            source_root=args.source_root,
            out_root=args.out_root,
            audit_dir=args.audit_dir,
            eval_scenes=args.eval_scenes,
        )
    except BuildError as exc:
        print(f"FATAL relative-factorial build invariant: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
