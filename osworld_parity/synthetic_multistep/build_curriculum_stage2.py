#!/usr/bin/env python3
"""Build the frozen fresh B-format dataset for the r256 curriculum contrast."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


COUNTS = {"train": 2000, "val": 200}
SEEDS = {"train": 2026073101, "val": 2026073102}
FORMAT = "deltatype_raw_pre"


class BuildError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"missing JSONL: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise BuildError(f"blank JSONL line: {path}:{line_no}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BuildError(f"malformed JSON: {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise BuildError(f"non-object JSONL row: {path}:{line_no}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rung2(audit_dir: Path):
    expected = (audit_dir / "rung2_scene.py").resolve()
    if not expected.is_file():
        raise BuildError(f"audited rung2 source missing: {expected}")
    sys.path.insert(0, str(audit_dir.resolve()))
    import rung2_scene as rung2  # type: ignore

    if Path(rung2.__file__).resolve() != expected:
        raise BuildError(f"loaded wrong rung2 module: {rung2.__file__}")
    return rung2


def _empty_keys() -> dict[str, set[Any]]:
    return {"bbox": set(), "center": set(), "cursor_bbox": set(), "image_sha256": set()}


def _add_scene(keys: dict[str, set[Any]], scene: dict[str, Any], image: Path | None) -> None:
    bbox = tuple(int(x) for x in scene["bbox"])
    center = tuple(int(x) for x in scene["target_center"])
    cursor = tuple(int(x) for x in scene["cursor"])
    keys["bbox"].add(bbox)
    keys["center"].add(center)
    keys["cursor_bbox"].add((cursor, bbox))
    if image is not None:
        if not image.is_file():
            raise BuildError(f"referenced image missing: {image}")
        keys["image_sha256"].add(_sha256(image))


def _reference_keys(
    stage1_root: Path, single_eval_scenes: Path, episode_root: Path
) -> dict[str, dict[str, set[Any]]]:
    refs = {name: _empty_keys() for name in ("stage1", "single_step_eval", "multistep_eval")}
    for split in ("train", "val"):
        for scene in _jsonl(stage1_root / f"scenes_{split}.jsonl"):
            _add_scene(refs["stage1"], scene, Path(scene["image_path"]))
    for scene in _jsonl(single_eval_scenes):
        _add_scene(refs["single_step_eval"], scene, Path(scene["image_path"]))
    specs = _jsonl(episode_root / "episode_specs.jsonl")
    for spec in specs:
        cursor = tuple(spec["initial_cursor"])
        for target in spec["targets"]:
            scene = {
                "cursor": cursor,
                "bbox": target["bbox"],
                "target_center": target["target_center"],
            }
            image = episode_root / "images" / (
                f"{spec['episode_id']}_t{int(target['target_index']):02d}.png"
            )
            _add_scene(refs["multistep_eval"], scene, image)
            cursor = tuple(target["target_center"])
    return refs


def _overlaps(
    left: dict[str, set[Any]], right: dict[str, set[Any]]
) -> dict[str, int]:
    return {key: len(left[key] & right[key]) for key in left}


def _union_keys(*groups: dict[str, set[Any]]) -> dict[str, set[Any]]:
    return {key: set().union(*(group[key] for group in groups)) for key in groups[0]}


def _collision_types(
    *, cursor: tuple[int, int], bbox: tuple[int, int, int, int],
    image_sha256: str, forbidden: dict[str, set[Any]],
) -> set[str]:
    center = (bbox[0] + (bbox[2] - bbox[0]) // 2,
              bbox[1] + (bbox[3] - bbox[1]) // 2)
    values = {
        "bbox": bbox,
        "center": center,
        "cursor_bbox": (cursor, bbox),
        "image_sha256": image_sha256,
    }
    return {key for key, value in values.items() if value in forbidden[key]}


def _render_candidate(rung2, cursor: tuple[int, int], bbox: tuple[int, int, int, int]) -> bytes:
    image = rung2.Image.new("RGB", (rung2.SW, rung2.SH), rung2.BG)
    draw = rung2.ImageDraw.Draw(image)
    draw.rectangle(bbox, fill=rung2.BOX_FILL, outline=rung2.BOX_EDGE, width=4)
    rung2.draw_arrow(image, cursor)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_fresh_scenes(
    rung2, *, count: int, seed: int, out_dir: Path,
    forbidden: dict[str, set[Any]], split: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate exact-count scenes, deterministically rejecting every collision."""
    if count % 2:
        raise BuildError(f"split count must be even: {split}={count}")
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_keys = _empty_keys()
    scenes: list[dict[str, Any]] = []
    rejections = {key: 0 for key in forbidden}
    rejections["distance_or_bounds"] = 0
    margin = 60

    for kind, wanted in (("long", count // 2), ("short", count // 2)):
        accepted_kind = 0
        while accepted_kind < wanted:
            if kind == "long":
                cursor = (
                    rng.randrange(margin, rung2.SW - margin),
                    rng.randrange(margin, rung2.SH - margin),
                )
                bx = rng.randrange(margin, rung2.SW - rung2.BOX - margin)
                by = rng.randrange(margin, rung2.SH - rung2.BOX - margin)
                center = (bx + rung2.BOX // 2, by + rung2.BOX // 2)
                if math.dist(cursor, center) < 400:
                    rejections["distance_or_bounds"] += 1
                    continue
            else:
                bx = rng.randrange(
                    margin + 200, rung2.SW - rung2.BOX - margin - 200
                )
                by = rng.randrange(
                    margin + 200, rung2.SH - rung2.BOX - margin - 200
                )
                center = (bx + rung2.BOX // 2, by + rung2.BOX // 2)
                angle, radius = rng.uniform(0, 2 * math.pi), rng.uniform(120, 300)
                cursor = (
                    int(center[0] + radius * math.cos(angle)),
                    int(center[1] + radius * math.sin(angle)),
                )
                if not (margin <= cursor[0] < rung2.SW - margin
                        and margin <= cursor[1] < rung2.SH - margin):
                    rejections["distance_or_bounds"] += 1
                    continue
            bbox = (bx, by, bx + rung2.BOX, by + rung2.BOX)
            image_bytes = _render_candidate(rung2, cursor, bbox)
            image_sha = hashlib.sha256(image_bytes).hexdigest()
            all_forbidden = _union_keys(forbidden, accepted_keys)
            collisions = _collision_types(
                cursor=cursor, bbox=bbox, image_sha256=image_sha,
                forbidden=all_forbidden,
            )
            if collisions:
                for key in collisions:
                    rejections[key] += 1
                continue
            scene_index = len(scenes)
            sid = f"stage2_{split}_{kind}_{scene_index:04d}"
            path = out_dir / f"{kind}_{scene_index:04d}.png"
            path.write_bytes(image_bytes)
            scene = {
                "scene_id": sid, "kind": kind, "cursor": list(cursor),
                "bbox": list(bbox), "target_center": list(center),
                "image_path": str(path), "distance_px": math.dist(cursor, center),
            }
            scenes.append(scene)
            accepted_kind += 1
            accepted_keys["bbox"].add(bbox)
            accepted_keys["center"].add(center)
            accepted_keys["cursor_bbox"].add((cursor, bbox))
            accepted_keys["image_sha256"].add(image_sha)
    return scenes, rejections


def _scene_keys(scenes: list[dict[str, Any]]) -> dict[str, set[Any]]:
    result = _empty_keys()
    for scene in scenes:
        _add_scene(result, scene, Path(scene["image_path"]))
    return result


def _record(rung2, scene: dict[str, Any], final_image: Path) -> dict[str, Any]:
    grammar = rung2.GRAMMARS["deltatype_raw"]
    cursor = tuple(scene["cursor"])
    target = tuple(scene["target_center"])
    dx, dy = rung2.ideal(grammar["space"], cursor, target)
    action = f"{dx} {dy} 0 ; +LMB -LMB"
    prose = rung2.preamble_text(scene)
    if any(character.isdigit() for character in prose):
        raise BuildError(f"numeric preamble leak: {scene['scene_id']}: {prose!r}")
    raw = f"{prose}\n{action}"
    move = grammar["parse"](raw, None)
    if not move.parse_ok or move.coord != (dx, dy):
        raise BuildError(f"gold B action did not parse: {scene['scene_id']}: {raw!r}")
    if rung2.landing(grammar["space"], cursor, move.coord) != target:
        raise BuildError(f"gold B action did not land: {scene['scene_id']}")
    user = rung2.build_user_text(grammar, scene, False, True)
    sid = scene["scene_id"]
    return {
        "sample_id": f"stage2_raw_pre_{sid}",
        "recording_id": sid,
        "scene_id": sid,
        "kind": scene["kind"],
        "format": FORMAT,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": grammar["system"]}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(final_image)},
                {"type": "text", "text": user},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": raw}]},
        ],
    }


def build(
    *,
    out_root: Path,
    audit_dir: Path,
    stage1_root: Path,
    single_eval_scenes: Path,
    episode_root: Path,
) -> dict[str, Any]:
    out_root = out_root.resolve()
    if (out_root / "curriculum_dataset_manifest.json").exists() or any(
        (out_root / name).exists() for name in (FORMAT, "images")
    ):
        raise BuildError(f"refusing to overwrite stage-2 dataset: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    rung2 = _load_rung2(audit_dir.resolve())
    refs = _reference_keys(
        stage1_root.resolve(), single_eval_scenes.resolve(), episode_root.resolve()
    )
    frozen_forbidden = _union_keys(*refs.values())
    stage = out_root / f".building_{os.getpid()}_{uuid.uuid4().hex}"
    stage.mkdir()
    built_scenes: dict[str, list[dict[str, Any]]] = {}
    built_keys: dict[str, dict[str, set[Any]]] = {}
    rejection_report: dict[str, dict[str, int]] = {}
    try:
        for split, count in COUNTS.items():
            image_stage = stage / "images" / split
            split_forbidden = frozen_forbidden
            if split == "val":
                split_forbidden = _union_keys(frozen_forbidden, built_keys["train"])
            scenes, rejections = _build_fresh_scenes(
                rung2, count=count, seed=SEEDS[split], out_dir=image_stage,
                forbidden=split_forbidden, split=split,
            )
            if len(scenes) != count:
                raise BuildError(f"wrong generated count: {split}={len(scenes)}")
            final_scenes = []
            records = []
            for scene in scenes:
                scene = dict(scene)
                staged_image = Path(scene["image_path"])
                final_image = out_root / "images" / split / staged_image.name
                scene["image_path"] = str(staged_image)
                final_scenes.append(scene)
                records.append(_record(rung2, scene, final_image))
            keys = _scene_keys(final_scenes)
            for reference, reference_keys in refs.items():
                overlap = _overlaps(keys, reference_keys)
                if any(overlap.values()):
                    raise BuildError(f"{split} overlaps {reference}: {overlap}")
            if any(len(values) != count for values in keys.values()):
                sizes = {key: len(value) for key, value in keys.items()}
                raise BuildError(f"duplicate stage-2 geometry in {split}: {sizes}")
            for scene in final_scenes:
                scene["image_path"] = str(
                    out_root / "images" / split / Path(scene["image_path"]).name
                )
            built_scenes[split] = final_scenes
            built_keys[split] = keys
            rejection_report[split] = rejections
            _write_jsonl(stage / f"scenes_{split}.jsonl", final_scenes)
            _write_jsonl(stage / FORMAT / "_normalized" / split / "chat.jsonl", records)

        cross = _overlaps(built_keys["train"], built_keys["val"])
        if any(cross.values()):
            raise BuildError(f"stage-2 train/validation overlap: {cross}")
        overlap_report = {
            split: {reference: _overlaps(built_keys[split], keys)
                    for reference, keys in refs.items()}
            for split in COUNTS
        }
        overlap_report["train_vs_validation"] = cross
        report = {
            "status": "pass",
            "format": FORMAT,
            "counts": COUNTS,
            "seeds": SEEDS,
            "geometry_key_types": ["bbox", "center", "cursor_bbox", "image_sha256"],
            "overlap_counts": overlap_report,
            "deterministic_rejections": rejection_report,
            "oracle_parse_and_land": {"passing": sum(COUNTS.values()), "total": sum(COUNTS.values())},
            "preamble_retained": {"passing": sum(COUNTS.values()), "total": sum(COUNTS.values())},
            "references": {
                "stage1_root": str(stage1_root.resolve()),
                "single_eval_scenes": str(single_eval_scenes.resolve()),
                "episode_root": str(episode_root.resolve()),
                "episode_manifest_sha256": _sha256(episode_root / "build_manifest.json"),
                "rung2_scene_sha256": _sha256(audit_dir / "rung2_scene.py"),
            },
        }
        (stage / "overlap_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "artifact_type": "synthetic_multistep_curriculum_stage2_dataset",
            "schema_version": 1,
            "status": "complete",
            "format": FORMAT,
            "train_records": COUNTS["train"],
            "validation_records": COUNTS["val"],
            "seeds": SEEDS,
            "overlap_report": "overlap_report.json",
        }
        (stage / "curriculum_source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for child in stage.iterdir():
            os.replace(child, out_root / child.name)
        stage.rmdir()
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--stage1-root", required=True, type=Path)
    parser.add_argument("--single-eval-scenes", required=True, type=Path)
    parser.add_argument("--episode-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = build(
            out_root=args.out,
            audit_dir=args.audit_dir,
            stage1_root=args.stage1_root,
            single_eval_scenes=args.single_eval_scenes,
            episode_root=args.episode_root,
        )
    except BuildError as exc:
        print(f"FATAL curriculum stage-2 build: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
