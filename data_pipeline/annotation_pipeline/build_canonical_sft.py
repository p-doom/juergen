#!/usr/bin/env python3
"""Build a portable canonical SFT artifact from an annotation pipeline run."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any


VALID_IMAGE_MODES = ("hardlink", "copy", "symlink")
VALID_SPLIT_GROUPS = ("recording_id", "clip_id")
DEFAULT_SYSTEM_PROMPT_VERSION = "annotation_pipeline_v3"


def ensure_empty_dir(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Refusing to overwrite non-empty output dir: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_num}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def iter_stage03_outputs(run_dir: Path) -> list[tuple[str, Path]]:
    run_dir = run_dir.expanduser().resolve()
    direct = run_dir / "trajectories.jsonl"
    if direct.is_file():
        return [(run_dir.name, direct)]

    found: list[tuple[str, Path]] = []
    for clip_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        path = clip_dir / "stage_03_assemble" / "trajectories.jsonl"
        if path.is_file():
            found.append((clip_dir.name, path))
    if not found:
        raise FileNotFoundError(
            f"No stage_03_assemble/trajectories.jsonl files found under {run_dir}"
        )
    return found


def sanitize_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return value.strip("_") or "sample"


def resolve_image_path(image: str, *, run_dir: Path, image_base: Path | None) -> Path:
    path = Path(image).expanduser()
    candidates = [path] if path.is_absolute() else []
    if image_base is not None and not path.is_absolute():
        candidates.append(image_base / path)
    if not path.is_absolute():
        candidates.extend([run_dir / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates) or str(path)
    raise FileNotFoundError(f"image not found: {image} (tried {tried})")


def materialize_image(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    raise ValueError(f"unknown image mode: {mode}")


def transform_messages(
    raw: dict[str, Any],
    *,
    sample_id: str,
    run_dir: Path,
    image_base: Path | None,
    images_root: Path,
    image_mode: str,
    terminate_token: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    messages = raw.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages list")

    instruction = str(raw.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("missing instruction")

    rewritten: list[dict[str, Any]] = []
    rel_images: list[str] = []
    first_user_seen = False
    last_assistant_idx = -1

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message is not an object")
        role = message.get("role")
        content = message.get("content", [])
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role!r}")

        new_content: str | list[dict[str, Any]]
        if isinstance(content, str):
            new_content = content
        elif isinstance(content, list):
            new_blocks: list[dict[str, Any]] = []
            image_blocks: list[dict[str, Any]] = []
            text_blocks: list[dict[str, Any]] = []
            other_blocks: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError("content block is not an object")
                block_type = block.get("type")
                if block_type == "image":
                    image_value = block.get("image") or block.get("path") or block.get("url")
                    if not image_value:
                        raise ValueError("image block missing image/path/url")
                    src = resolve_image_path(str(image_value), run_dir=run_dir, image_base=image_base)
                    index = len(rel_images)
                    suffix = src.suffix or ".jpg"
                    rel_path = Path("images") / sanitize_component(sample_id) / f"frame_{index:05d}{suffix}"
                    materialize_image(src, images_root.parent / rel_path, mode=image_mode)
                    image_blocks.append({"type": "image", "image": rel_path.as_posix()})
                    rel_images.append(rel_path.as_posix())
                elif block_type == "text":
                    text = str(block.get("text", ""))
                    if text:
                        text_blocks.append({"type": "text", "text": text})
                else:
                    other_blocks.append(dict(block))

            if role == "user" and not first_user_seen:
                new_blocks.append({"type": "text", "text": instruction})
                new_blocks.extend(image_blocks)
                # Drop duplicate instruction text from the old stage-03 first
                # user turn, but preserve any extra non-identical text blocks.
                new_blocks.extend(block for block in text_blocks if block.get("text") != instruction)
                first_user_seen = True
            else:
                new_blocks.extend(image_blocks)
                new_blocks.extend(text_blocks)
            new_blocks.extend(other_blocks)
            new_content = new_blocks
        else:
            raise ValueError("message content must be string or list")

        rewritten.append({"role": role, "content": new_content})
        if role == "assistant":
            last_assistant_idx = len(rewritten) - 1

    if not first_user_seen:
        raise ValueError("no user turn found")
    if last_assistant_idx < 0:
        raise ValueError("no assistant turn found")
    rewritten[last_assistant_idx] = {
        "role": "assistant",
        "content": [{"type": "text", "text": terminate_token}],
    }
    return rewritten, rel_images


def assign_splits(
    records: list[dict[str, Any]],
    *,
    split_group: str,
    seed: int,
    val_frac: float,
) -> dict[str, str]:
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("--val-frac must be >= 0 and < 1")
    groups = sorted({str(record[split_group]) for record in records})
    if not groups:
        return {}
    rng = random.Random(seed)
    rng.shuffle(groups)
    if len(groups) == 1 or val_frac == 0.0:
        val_groups: set[str] = set()
    else:
        n_val = min(len(groups) - 1, max(1, round(len(groups) * val_frac)))
        val_groups = set(groups[:n_val])
    return {group: ("val" if group in val_groups else "train") for group in groups}


def build_canonical_sft(
    *,
    run_dir: Path,
    output_dir: Path,
    image_base: Path | None = None,
    split_group: str = "recording_id",
    val_frac: float = 0.1,
    seed: int = 0,
    image_mode: str = "hardlink",
    terminate_token: str = "TERMINATE",
    overwrite: bool = False,
) -> dict[str, Any]:
    if split_group not in VALID_SPLIT_GROUPS:
        raise ValueError(f"split_group must be one of {VALID_SPLIT_GROUPS}")
    if image_mode not in VALID_IMAGE_MODES:
        raise ValueError(f"image_mode must be one of {VALID_IMAGE_MODES}")

    run_dir = run_dir.expanduser().resolve()
    output_dir = ensure_empty_dir(output_dir, overwrite=overwrite)
    images_root = output_dir / "images"

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for clip_id, trajectories_path in iter_stage03_outputs(run_dir):
        for raw_index, raw in enumerate(read_jsonl(trajectories_path)):
            raw_sample_id = str(raw.get("sample_id") or f"sample{raw_index:05d}")
            sample_id = sanitize_component(f"{clip_id}__{raw_sample_id}")
            try:
                messages, image_paths = transform_messages(
                    raw,
                    sample_id=sample_id,
                    run_dir=run_dir,
                    image_base=image_base,
                    images_root=images_root,
                    image_mode=image_mode,
                    terminate_token=terminate_token,
                )
                recording_id = str(raw.get("recording_id") or "")
                if not recording_id:
                    raise ValueError("missing recording_id")
                record = dict(raw)
                record.update(
                    {
                        "sample_id": sample_id,
                        "raw_sample_id": raw_sample_id,
                        "clip_id": clip_id,
                        "recording_id": recording_id,
                        "group_id": recording_id if split_group == "recording_id" else clip_id,
                        "messages": messages,
                        "image_paths": image_paths,
                    }
                )
                records.append(record)
            except Exception as exc:  # noqa: BLE001 - keep bad samples auditable.
                rejected.append(
                    {
                        "clip_id": clip_id,
                        "raw_sample_id": raw_sample_id,
                        "source_path": str(trajectories_path),
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )

    split_of = assign_splits(records, split_group="group_id", seed=seed, val_frac=val_frac)
    for record in records:
        record["split"] = split_of[str(record["group_id"])]

    split_manifest = [
        {
            "sample_id": record["sample_id"],
            "group_id": record["group_id"],
            "split": record["split"],
            "recording_id": record["recording_id"],
            "clip_id": record["clip_id"],
        }
        for record in records
    ]
    sample_manifest = [
        {
            "sample_id": record["sample_id"],
            "raw_sample_id": record["raw_sample_id"],
            "split": record["split"],
            "group_id": record["group_id"],
            "recording_id": record["recording_id"],
            "clip_id": record["clip_id"],
            "n_messages": len(record["messages"]),
            "n_images": len(record["image_paths"]),
            "n_frames": record.get("n_frames"),
            "duration_s": record.get("duration_s"),
        }
        for record in records
    ]

    write_jsonl(output_dir / "chat.jsonl", records)
    write_jsonl(output_dir / "split_manifest.jsonl", split_manifest)
    write_jsonl(output_dir / "sample_manifest.jsonl", sample_manifest)
    write_jsonl(output_dir / "rejected.jsonl", rejected)

    counts_by_split = {
        split: sum(1 for record in records if record["split"] == split)
        for split in ("train", "val")
    }
    manifest = {
        "artifact_type": "juergen_canonical_sft",
        "schema_version": 1,
        "source_run_dir": str(run_dir),
        "split_group": split_group,
        "val_frac": val_frac,
        "seed": seed,
        "image_mode": image_mode,
        "terminate_token": terminate_token,
        "system_prompt_version": DEFAULT_SYSTEM_PROMPT_VERSION,
        "n_samples": len(records),
        "n_rejected": len(rejected),
        "counts_by_split": counts_by_split,
        "files": {
            "chat": "chat.jsonl",
            "split_manifest": "split_manifest.jsonl",
            "sample_manifest": "sample_manifest.jsonl",
            "rejected": "rejected.jsonl",
            "images": "images/",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", "--run_dir", dest="run_dir", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--image-base", "--image_base", dest="image_base", type=Path)
    parser.add_argument("--split-group", "--split_group", dest="split_group", default="recording_id")
    parser.add_argument("--val-frac", "--val_frac", dest="val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-mode", "--image_mode", dest="image_mode", default="hardlink")
    parser.add_argument(
        "--terminate-token", "--terminate_token", dest="terminate_token", default="TERMINATE"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_canonical_sft(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        image_base=args.image_base,
        split_group=args.split_group,
        val_frac=args.val_frac,
        seed=args.seed,
        image_mode=args.image_mode,
        terminate_token=args.terminate_token,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {manifest['n_samples']} canonical SFT samples "
        f"({manifest['n_rejected']} rejected) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
