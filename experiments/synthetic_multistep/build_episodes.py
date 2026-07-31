#!/usr/bin/env python3
"""Build frozen heldout episode specs and 100%-checked oracle prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from .contract import (
        Contract,
        ContractError,
        DEFAULT_AUDIT_DIR,
        FROZEN_PATH,
        SEMANTICS,
        heldout_image_aggregate,
        load_frozen,
        load_jsonl,
        serialize_action,
        sha256_bytes,
        sha256_file,
        strict_schema_ok,
        unit_range_ok,
        verify_frozen_sources,
    )
except ImportError:  # direct script execution
    from contract import (
    Contract,
    ContractError,
    DEFAULT_AUDIT_DIR,
    FROZEN_PATH,
    SEMANTICS,
    heldout_image_aggregate,
    load_frozen,
    load_jsonl,
    serialize_action,
    sha256_bytes,
    sha256_file,
    strict_schema_ok,
    unit_range_ok,
    verify_frozen_sources,
    )


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _scene_sets(rows: list[dict[str, Any]]) -> dict[str, set[Any]]:
    return {
        "scene_id": {row["scene_id"] for row in rows},
        "bbox": {tuple(row["bbox"]) for row in rows},
        "center": {tuple(row["target_center"]) for row in rows},
        "geometry": {(tuple(row["cursor"]), tuple(row["bbox"])) for row in rows},
    }


def _leak_report(
    heldout: list[dict[str, Any]], train: list[dict[str, Any]], val: list[dict[str, Any]]
) -> dict[str, Any]:
    sets = {name: _scene_sets(rows) for name, rows in (
        ("heldout", heldout), ("train", train), ("val", val)
    )}
    report: dict[str, Any] = {}
    for key in ("scene_id", "bbox", "center", "geometry"):
        report[key] = {
            "heldout_train": len(sets["heldout"][key] & sets["train"][key]),
            "heldout_val": len(sets["heldout"][key] & sets["val"][key]),
            "train_val": len(sets["train"][key] & sets["val"][key]),
        }
    if any(
        values[pair]
        for key, values in report.items()
        for pair in ("heldout_train", "heldout_val")
    ):
        raise ContractError(f"heldout task/geometry leak: {report}")
    return report


def _candidate_order(scene_id: str, target_index: int, rows: list[dict[str, Any]], seed: int):
    def key(row: dict[str, Any]) -> str:
        raw = f"synthetic-multistep-v1|{seed}|{scene_id}|{target_index}|{row['scene_id']}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return sorted(rows, key=key)


def _select_targets(
    initial: dict[str, Any],
    heldout: list[dict[str, Any]],
    *,
    count: int,
    minimum_transition: float,
    seed: int,
) -> list[dict[str, Any]]:
    selected = [initial]
    current = tuple(initial["target_center"])
    for target_index in range(1, count):
        chosen = None
        for candidate in _candidate_order(initial["scene_id"], target_index, heldout, seed):
            if candidate["scene_id"] in {row["scene_id"] for row in selected}:
                continue
            target = tuple(candidate["target_center"])
            if math.dist(current, target) < minimum_transition:
                continue
            chosen = candidate
            break
        if chosen is None:
            raise ContractError(f"cannot choose target {target_index} for {initial['scene_id']}")
        selected.append(chosen)
        current = tuple(chosen["target_center"])
    return selected


def _serialization_shape(text: str) -> str:
    # A diagnostic only: erase exactly the semantic payload fields.
    import re

    text = re.sub(r'"action": "(?:left_click|move_rel)"', '"action": "<SEMANTIC>"', text)
    return re.sub(r'"coordinate": \[-?\d+, -?\d+\]', '"coordinate": [<COORD>]', text)


def build(
    out: Path,
    *,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    preamble: bool = False,
) -> dict[str, Any]:
    frozen = load_frozen()
    cfg = frozen["episode_contract"]
    if preamble != bool(cfg["preamble"]):
        raise ContractError(
            "primary frozen episode artifact is action-only; preamble sensitivity requires "
            "a separately frozen manifest"
        )
    source_hashes = verify_frozen_sources(audit_dir)
    contract = Contract(audit_dir, verify=False)
    heldout_path = audit_dir / "runs/rung2_offshelf/px/scenes.jsonl"
    train_path = audit_dir / "r3data_2k/scenes_train.jsonl"
    val_path = audit_dir / "r3data_2k/scenes_val.jsonl"
    heldout, train, val = map(load_jsonl, (heldout_path, train_path, val_path))
    if len(heldout) != cfg["heldout_scenes"]:
        raise ContractError(f"heldout count drift: {len(heldout)}")
    aggregate = heldout_image_aggregate(heldout)
    if aggregate != frozen["sources"]["heldout_image_aggregate"]:
        raise ContractError(f"heldout PNG aggregate drift: {aggregate}")
    leaks = _leak_report(heldout, train, val)
    train_geometry = _scene_sets(train)["geometry"]
    val_geometry = _scene_sets(val)["geometry"]

    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ContractError(f"refusing to overwrite non-empty output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.building_", dir=out.parent))
    try:
        images = stage / "images"
        images.mkdir()
        specs: list[dict[str, Any]] = []
        oracle: dict[str, list[dict[str, Any]]] = {semantic: [] for semantic in SEMANTICS}
        identity_rows = []
        oracle_hits = {semantic: 0 for semantic in SEMANTICS}
        generated_geometry: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        expected_targets = len(heldout) * int(cfg["targets_per_episode"])

        for episode_index, initial in enumerate(heldout):
            episode_id = f"phasea_{initial['scene_id']}"
            targets = _select_targets(
                initial,
                heldout,
                count=int(cfg["targets_per_episode"]),
                minimum_transition=float(cfg["minimum_target_transition_px"]),
                seed=int(cfg["target_selection_seed"]),
            )
            step1_source = Path(initial["image_path"])
            step1_dest = images / f"{episode_id}_t00.png"
            shutil.copyfile(step1_source, step1_dest)
            cursor0 = tuple(initial["cursor"])
            rendered = contract.render_png(initial["bbox"], cursor0)
            source_bytes = step1_source.read_bytes()
            if rendered != source_bytes or step1_dest.read_bytes() != source_bytes:
                raise ContractError(f"step-1 PNG identity failed: {initial['scene_id']}")
            identity_rows.append({
                "episode_id": episode_id,
                "source_scene_id": initial["scene_id"],
                "cursor": list(cursor0),
                "bbox": initial["bbox"],
                "target_center": initial["target_center"],
                "source_image": str(step1_source.resolve()),
                "copied_image": str(step1_dest.relative_to(stage)),
                "png_sha256": sha256_bytes(source_bytes),
                "geometry_equal": True,
                "bytes_equal": True,
            })
            spec = {
                "episode_id": episode_id,
                "episode_index": episode_index,
                "kind": initial["kind"],
                "initial_cursor": initial["cursor"],
                "step1_image": str(step1_dest.relative_to(stage)),
                "step1_png_sha256": sha256_bytes(source_bytes),
                "targets": [
                    {
                        "target_index": index,
                        "source_scene_id": target["scene_id"],
                        "bbox": target["bbox"],
                        "target_center": target["target_center"],
                    }
                    for index, target in enumerate(targets)
                ],
            }
            specs.append(spec)

            per_semantic_turns: dict[str, list[dict[str, Any]]] = {
                semantic: [] for semantic in SEMANTICS
            }
            cursors = {semantic: cursor0 for semantic in SEMANTICS}
            for target_index, target_row in enumerate(targets):
                bbox = target_row["bbox"]
                target = tuple(target_row["target_center"])
                if len(set(cursors.values())) != 1:
                    raise ContractError(f"oracle cursor states diverged before {episode_id}/{target_index}")
                common_cursor = next(iter(cursors.values()))
                geometry_key = (tuple(common_cursor), tuple(bbox))
                if geometry_key in train_geometry or geometry_key in val_geometry:
                    raise ContractError(
                        f"generated oracle-prefix geometry leaks into train/val: "
                        f"{episode_id}/{target_index} {geometry_key}"
                    )
                generated_geometry.add(geometry_key)
                if target_index == 0:
                    image_path = step1_dest
                    png = image_path.read_bytes()
                else:
                    image_path = images / f"{episode_id}_t{target_index:02d}.png"
                    png = contract.render_png(bbox, common_cursor)
                    image_path.write_bytes(png)
                outputs: dict[str, str] = {}
                for semantic in SEMANTICS:
                    coord = contract.ideal_coord(semantic, cursors[semantic], target)
                    prose = contract.preamble_text(cursors[semantic], target) if preamble else None
                    assistant = serialize_action(semantic, coord, prose=prose)
                    parsed = contract.parse(semantic, assistant)
                    schema_ok = strict_schema_ok(semantic, assistant, parsed.coord)
                    if not parsed.parse_ok or parsed.coord != coord or not schema_ok:
                        raise ContractError(
                            f"oracle parse/schema failure {episode_id}/{target_index}/{semantic}"
                        )
                    landing = contract.apply_coord(semantic, cursors[semantic], coord)
                    if not contract.in_bbox(landing, bbox):
                        raise ContractError(
                            f"oracle miss {episode_id}/{target_index}/{semantic}: {landing} {bbox}"
                        )
                    if not unit_range_ok(semantic, coord):
                        raise ContractError(f"oracle unit failure: {semantic} {coord}")
                    user = contract.user_text(
                        semantic,
                        cursors[semantic],
                        target,
                        target_index=target_index,
                        target_count=len(targets),
                        preamble=preamble,
                        prior=[turn["assistant"] for turn in per_semantic_turns[semantic]][
                            -int(cfg["history_turns"]):
                        ],
                    )
                    turn = {
                        "target_index": target_index,
                        "cursor_before": list(cursors[semantic]),
                        "bbox": bbox,
                        "target_center": list(target),
                        "image": str(image_path.relative_to(stage)),
                        "image_sha256": sha256_bytes(png),
                        "system": contract.system_prompt(semantic),
                        "user": user,
                        "assistant": assistant,
                        "coord": list(coord),
                        "landing": list(landing),
                        "parse_ok": True,
                        "schema_ok": True,
                        "hit": True,
                    }
                    per_semantic_turns[semantic].append(turn)
                    cursors[semantic] = landing
                    outputs[semantic] = assistant
                    oracle_hits[semantic] += 1
                if cursors[SEMANTICS[0]] != cursors[SEMANTICS[1]]:
                    raise ContractError(f"oracle landing mismatch {episode_id}/{target_index}: {cursors}")
                if _serialization_shape(outputs[SEMANTICS[0]]) != _serialization_shape(
                    outputs[SEMANTICS[1]]
                ):
                    raise ContractError(f"action serialization mismatch {episode_id}/{target_index}")
            for semantic in SEMANTICS:
                oracle[semantic].append({
                    "episode_id": episode_id,
                    "semantic": semantic,
                    "preamble": preamble,
                    "turns": per_semantic_turns[semantic],
                })

        if any(count != expected_targets for count in oracle_hits.values()):
            raise ContractError(f"oracle is not 100%: {oracle_hits}/{expected_targets}")
        _write_jsonl(stage / "episode_specs.jsonl", specs)
        _write_jsonl(stage / "step1_identity.jsonl", identity_rows)
        for semantic in SEMANTICS:
            _write_jsonl(stage / f"oracle_{semantic}.jsonl", oracle[semantic])
        artifact_hashes = {
            name: sha256_file(stage / name)
            for name in (
                "episode_specs.jsonl",
                "step1_identity.jsonl",
                "oracle_absolute_toolcall.jsonl",
                "oracle_move_rel.jsonl",
            )
        }
        manifest = {
            "schema_version": 1,
            "artifact_type": "synthetic_multistep_phasea_episodes",
            "status": "complete",
            "frozen_manifest": str(FROZEN_PATH),
            "frozen_manifest_sha256": sha256_file(FROZEN_PATH),
            "source_sha256": source_hashes,
            "heldout_image_aggregate": aggregate,
            "n_episodes": len(specs),
            "targets_per_episode": cfg["targets_per_episode"],
            "n_oracle_targets": expected_targets,
            "oracle_hits": oracle_hits,
            "oracle_rate": {semantic: 1.0 for semantic in SEMANTICS},
            "preamble": preamble,
            "leak_report": leaks,
            "generated_oracle_geometry": {
                "unique": len(generated_geometry),
                "train_overlap": len(generated_geometry & train_geometry),
                "val_overlap": len(generated_geometry & val_geometry),
            },
            "step1_identity": {
                "checked": len(identity_rows), "byte_equal": len(identity_rows),
                "geometry_equal": len(identity_rows),
            },
            "action_serialization_matched_except_semantics": True,
            "oracle_observations_and_states_identical": True,
            "artifact_sha256": artifact_hashes,
        }
        _atomic_text(stage / "build_manifest.json", json.dumps(manifest, indent=2) + "\n")
        if out.exists():
            out.rmdir()
        stage.replace(out)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--preamble", action="store_true")
    args = parser.parse_args()
    manifest = build(args.out, audit_dir=args.audit_dir, preamble=args.preamble)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
