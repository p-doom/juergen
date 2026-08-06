#!/usr/bin/env python3
"""Build a portable canonical SFT artifact from an annotation pipeline run.

One of two stage-04 variants, and the only one that emits the labctl-facing
``juergen_canonical_sft`` contract (``message_policy.owner = "labctl"``,
``split_manifest.jsonl`` + ``sample_manifest.jsonl``, a grouped train/val split
decided HERE, and the full ``terminal_mode`` matrix incl.
``replace_final_assistant``). Input is a run dir holding
``stage_03_assemble/trajectories.jsonl``.

The other variant — ``stage_04_build_conversations.py`` — is the current
generation's single injection point: it joins the stage-03 filter mask with the
stage-03b goals artifact, applies the system prompt and a terminal token that
always rides the final assistant turn, and emits
``artifact_type = "juergen_annotation_conversations"``; its train/val split is
applied downstream in stage 06 (``--val_fraction``). Neither subsumes the other:
this file owns the canonical-SFT artifact contract and the terminal-mode matrix;
that file owns the filter+goals join. Prefer that one for new datasets.

    PYTHONPATH=. python3 -m pipeline.stage_04_build_canonical_sft \\
        --run-dir <run with stage_03_assemble/> --output-dir <dest>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# Make the ``pipeline`` package importable when this stage is run directly
# as a script (mirrors the other stages' PYTHONPATH setup).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.lib.image_store import is_arrayrecord_image_uri  # noqa: E402


VALID_IMAGE_PATH_MODES = ("absolute", "preserve")
VALID_SPLIT_GROUPS = ("recording_id", "clip_id")
VALID_TERMINAL_MODES = ("none", "replace_final_assistant", "append_to_final_assistant",
                        "append_assistant")  # append_assistant DEPRECATED (OOD)


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


def stage03_output(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    path = run_dir / "stage_03_assemble" / "trajectories.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing stage 03 dataset: {path}")
    return path


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


def load_system_prompt(
    *,
    system_prompt_text: str | None,
    system_prompt_file: Path | None,
) -> tuple[str | None, dict[str, Any]]:
    if system_prompt_text and system_prompt_file:
        raise ValueError("pass only one of --system-prompt-text or --system-prompt-file")
    if system_prompt_file:
        text = system_prompt_file.expanduser().read_text().rstrip("\n")
        source: dict[str, Any] = {"kind": "file", "path": str(system_prompt_file)}
    elif system_prompt_text:
        text = system_prompt_text
        source = {"kind": "arg"}
    else:
        return None, {"kind": "none"}
    if not text.strip():
        raise ValueError("system prompt is empty")
    source["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source["text"] = text
    return text, source


def render_image_path(raw_value: str, resolved: Path, *, image_path_mode: str) -> str:
    if image_path_mode == "absolute":
        return str(resolved)
    if image_path_mode == "preserve":
        return raw_value
    raise ValueError(f"unknown image_path_mode: {image_path_mode}")


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def apply_terminal_policy(
    messages: list[dict[str, Any]],
    *,
    terminal_token: str | None,
    terminal_mode: str,
) -> None:
    if terminal_mode == "none":
        return
    if not terminal_token:
        raise ValueError("terminal_token is required when terminal_mode is not 'none'")
    terminal_message = {"role": "assistant", "content": text_content(terminal_token)}
    if terminal_mode == "append_to_final_assistant":
        # Correct terminal policy: ride the token at the END of the final action's own turn
        # ("<action>\n<TERMINATE>"). Keeps user/assistant alternation — at inference the model
        # always sees an observation between its outputs, so a STANDALONE terminal assistant
        # turn (append_assistant) trains an out-of-distribution state.
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                parts = messages[idx].get("content") or []
                for c in reversed(parts):
                    if c.get("type") == "text":
                        c["text"] = f"{c['text']}\n{terminal_token}"
                        return
                messages[idx]["content"] = list(parts) + text_content(terminal_token)
                return
        raise ValueError("no assistant turn found")
    if terminal_mode == "append_assistant":
        # DEPRECATED (OOD): creates assistant->assistant adjacency; use append_to_final_assistant.
        messages.append(terminal_message)
        return
    if terminal_mode == "replace_final_assistant":
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                messages[idx] = terminal_message
                return
        raise ValueError("no assistant turn found")
    raise ValueError(f"unknown terminal_mode: {terminal_mode}")


def transform_messages(
    raw: dict[str, Any],
    *,
    sample_id: str,
    run_dir: Path,
    image_base: Path | None,
    image_path_mode: str,
    system_prompt_text: str | None,
    terminal_token: str | None,
    terminal_mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    messages = raw.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages list")

    instruction = str(raw.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("missing instruction")

    rewritten: list[dict[str, Any]] = []
    image_paths: list[str] = []
    first_user_seen = False

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message is not an object")
        role = message.get("role")
        content = message.get("content", [])
        if role == "system":
            raise ValueError("stage-03 samples must not contain system messages")
        if role not in {"user", "assistant"}:
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
                    raw_image = str(image_value)
                    if is_arrayrecord_image_uri(raw_image):
                        # Grain store: ar:///abs/path/images.array_record#idx is
                        # already portable/absolute. Pass through verbatim; do
                        # not run filesystem resolution or path rewriting.
                        image_path = raw_image
                    else:
                        src = resolve_image_path(raw_image, run_dir=run_dir, image_base=image_base)
                        image_path = render_image_path(
                            raw_image,
                            src,
                            image_path_mode=image_path_mode,
                        )
                    image_blocks.append({"type": "image", "image": image_path})
                    image_paths.append(image_path)
                elif block_type == "text":
                    text = str(block.get("text", ""))
                    if text:
                        text_blocks.append({"type": "text", "text": text})
                else:
                    other_blocks.append(dict(block))

            if role == "user" and not first_user_seen:
                new_blocks.append({"type": "text", "text": instruction})
                new_blocks.extend(image_blocks)
                # Drop instruction text duplicated from the stage-03 first user
                # turn, but preserve any extra non-identical text blocks.
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

    if not first_user_seen:
        raise ValueError("no user turn found")
    if not any(message.get("role") == "assistant" for message in rewritten):
        raise ValueError("no assistant turn found")
    apply_terminal_policy(
        rewritten,
        terminal_token=terminal_token,
        terminal_mode=terminal_mode,
    )
    if system_prompt_text is not None:
        rewritten.insert(0, {"role": "system", "content": text_content(system_prompt_text)})
    return rewritten, image_paths


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
    image_path_mode: str = "absolute",
    system_prompt_text: str | None = None,
    system_prompt_file: Path | None = None,
    terminal_token: str | None = None,
    terminal_mode: str = "none",
    overwrite: bool = False,
) -> dict[str, Any]:
    if split_group not in VALID_SPLIT_GROUPS:
        raise ValueError(f"split_group must be one of {VALID_SPLIT_GROUPS}")
    if image_path_mode not in VALID_IMAGE_PATH_MODES:
        raise ValueError(f"image_path_mode must be one of {VALID_IMAGE_PATH_MODES}")
    if terminal_mode not in VALID_TERMINAL_MODES:
        raise ValueError(f"terminal_mode must be one of {VALID_TERMINAL_MODES}")

    run_dir = run_dir.expanduser().resolve()
    output_dir = ensure_empty_dir(output_dir, overwrite=overwrite)
    system_prompt_text, system_prompt_source = load_system_prompt(
        system_prompt_text=system_prompt_text,
        system_prompt_file=system_prompt_file,
    )
    if (
        system_prompt_text is not None
        and terminal_mode != "none"
        and terminal_token
        and terminal_token not in system_prompt_text
    ):
        raise ValueError("terminal_token is not mentioned in the selected system prompt")

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    trajectories_path = stage03_output(run_dir)
    for raw_index, raw in enumerate(read_jsonl(trajectories_path)):
        clip_id = str(raw.get("clip_id") or "").strip()
        raw_sample_id = str(raw.get("sample_id") or f"sample{raw_index:05d}")
        sample_id = sanitize_component(f"{clip_id}__{raw_sample_id}")
        try:
            if not clip_id:
                raise ValueError("missing clip_id")
            messages, image_paths = transform_messages(
                raw,
                sample_id=sample_id,
                run_dir=run_dir,
                image_base=image_base,
                image_path_mode=image_path_mode,
                system_prompt_text=system_prompt_text,
                terminal_token=terminal_token,
                terminal_mode=terminal_mode,
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
            "start_frame_idx": record.get("start_frame_idx"),
            "end_frame_idx": record.get("end_frame_idx"),
            "start_time_s": record.get("start_time_s"),
            "end_time_s": record.get("end_time_s"),
            "source_frame_start": record.get("source_frame_start"),
            "source_frame_end": record.get("source_frame_end"),
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
        "image_path_mode": image_path_mode,
        "message_policy": {
            "owner": "labctl",
            "system_prompt": system_prompt_source,
            "terminal_mode": terminal_mode,
            "terminal_token": terminal_token,
        },
        "n_samples": len(records),
        "n_rejected": len(rejected),
        "counts_by_split": counts_by_split,
        "files": {
            "chat": "chat.jsonl",
            "split_manifest": "split_manifest.jsonl",
            "sample_manifest": "sample_manifest.jsonl",
            "rejected": "rejected.jsonl",
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
    parser.add_argument(
        "--image-path-mode",
        "--image_path_mode",
        dest="image_path_mode",
        default="absolute",
        choices=VALID_IMAGE_PATH_MODES,
    )
    parser.add_argument(
        "--system-prompt-text",
        "--system_prompt_text",
        dest="system_prompt_text",
    )
    parser.add_argument(
        "--system-prompt-file",
        "--system_prompt_file",
        dest="system_prompt_file",
        type=Path,
    )
    parser.add_argument("--terminal-token", "--terminal_token", dest="terminal_token")
    parser.add_argument(
        "--terminal-mode",
        "--terminal_mode",
        dest="terminal_mode",
        default="none",
        choices=VALID_TERMINAL_MODES,
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
        image_path_mode=args.image_path_mode,
        system_prompt_text=args.system_prompt_text,
        system_prompt_file=args.system_prompt_file,
        terminal_token=args.terminal_token,
        terminal_mode=args.terminal_mode,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {manifest['n_samples']} canonical SFT samples "
        f"({manifest['n_rejected']} rejected) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
