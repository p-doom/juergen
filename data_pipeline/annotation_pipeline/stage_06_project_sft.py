#!/usr/bin/env python3
"""Stage 06: project structured trajectories into the current SFT message format."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from annotation_pipeline.common import action_bin_from_dict, format_action, read_jsonl
from annotation_pipeline.image_store import is_arrayrecord_image_uri

VALID_IMAGE_PATH_MODES = ("absolute", "preserve")
VALID_SPLIT_GROUPS = ("recording_id", "clip_id")
VALID_TERMINAL_MODES = ("none", "replace_final_assistant", "append_to_final_assistant")
PROVENANCE_KEYS = ("parent_segment_id", "user_id", "version", "video_path")


def ensure_empty_dir(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Refusing to overwrite non-empty output dir: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sanitize_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return value.strip("_") or "sample"


def is_generic_instruction(instruction: str) -> bool:
    text = instruction.strip().lower().rstrip(".")
    if not text or len(text.split()) < 3:
        return True
    return text in {
        "complete the visible desktop task in this interval",
        "complete the task",
        "perform the visible task",
        "continue the current task",
        "do the task on screen",
    }


def instruction_variants(trajectory: dict[str, Any], include_variants: bool) -> list[str]:
    candidates = [trajectory["instruction"]]
    if include_variants:
        candidates.extend(trajectory["instruction_variants"])
    output: list[str] = []
    for candidate in candidates:
        text = str(candidate).strip()
        if text and text not in output:
            output.append(text)
    return output


def load_system_prompt(
    *, system_prompt_text: str | None, system_prompt_file: Path | None
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
    source["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    source["text"] = text
    return text, source


def _image_path(value: str, *, source_dir: Path, mode: str) -> str:
    if is_arrayrecord_image_uri(value):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_dir / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"image not found: {resolved}")
    return str(resolved) if mode == "absolute" else value


def _text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def apply_terminal_policy(
    messages: list[dict[str, Any]], *, terminal_token: str | None, terminal_mode: str
) -> None:
    if terminal_mode == "none":
        return
    if not terminal_token:
        raise ValueError("terminal_token is required when terminal_mode is not none")
    if terminal_mode == "replace_final_assistant":
        messages[-1] = {"role": "assistant", "content": _text_content(terminal_token)}
        return
    if terminal_mode == "append_to_final_assistant":
        text = messages[-1]["content"][0]["text"]
        messages[-1]["content"][0]["text"] = f"{text}\n{terminal_token}"
        return
    raise ValueError(f"Unknown terminal mode: {terminal_mode}")


def render_messages(
    trajectory: dict[str, Any],
    *,
    instruction: str,
    source_dir: Path,
    image_path_mode: str,
    system_prompt: str | None,
    terminal_token: str | None,
    terminal_mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    messages: list[dict[str, Any]] = []
    image_paths: list[str] = []
    for step_idx, step in enumerate(trajectory["steps"]):
        image = _image_path(str(step["image_path"]), source_dir=source_dir, mode=image_path_mode)
        image_paths.append(image)
        user_content: list[dict[str, str]] = []
        if step_idx == 0:
            user_content.append({"type": "text", "text": instruction})
        user_content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": user_content})
        action = format_action(action_bin_from_dict(step["action_bin"]))
        messages.append({"role": "assistant", "content": _text_content(action)})
    if not messages:
        raise ValueError("trajectory has no steps")
    apply_terminal_policy(messages, terminal_token=terminal_token, terminal_mode=terminal_mode)
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": _text_content(system_prompt)})
    return messages, image_paths


def assign_splits(
    records: list[dict[str, Any]], *, split_group: str, seed: int, val_frac: float
) -> dict[str, str]:
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be >= 0 and < 1")
    groups = sorted({str(record[split_group]) for record in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    if len(groups) <= 1 or val_frac == 0:
        val_groups: set[str] = set()
    else:
        n_val = min(len(groups) - 1, max(1, round(len(groups) * val_frac)))
        val_groups = set(groups[:n_val])
    return {group: ("val" if group in val_groups else "train") for group in groups}


def project_sft(
    *,
    trajectories_path: Path,
    output_dir: Path,
    include_variants: bool = False,
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
    system_prompt, system_prompt_source = load_system_prompt(
        system_prompt_text=system_prompt_text, system_prompt_file=system_prompt_file
    )
    if (
        system_prompt
        and terminal_token
        and terminal_mode != "none"
        and terminal_token not in system_prompt
    ):
        raise ValueError("terminal_token is not mentioned in the selected system prompt")

    output_dir = ensure_empty_dir(output_dir, overwrite=overwrite)
    source_dir = trajectories_path.resolve().parent
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for trajectory in read_jsonl(trajectories_path):
        for variant_idx, instruction in enumerate(
            instruction_variants(trajectory, include_variants)
        ):
            if is_generic_instruction(instruction):
                rejected.append(
                    {
                        "trajectory_id": trajectory["trajectory_id"],
                        "variant_idx": variant_idx,
                        "reason": "generic_or_empty_instruction",
                    }
                )
                continue
            sample_id = sanitize_component(
                f"{trajectory['clip_id']}__{trajectory['trajectory_id']}__v{variant_idx}"
            )
            try:
                messages, image_paths = render_messages(
                    trajectory,
                    instruction=instruction,
                    source_dir=source_dir,
                    image_path_mode=image_path_mode,
                    system_prompt=system_prompt,
                    terminal_token=terminal_token,
                    terminal_mode=terminal_mode,
                )
                record = {
                    "sample_id": sample_id,
                    "trajectory_id": trajectory["trajectory_id"],
                    "variant_idx": variant_idx,
                    "instruction": instruction,
                    "recording_id": trajectory["recording_id"],
                    "clip_id": trajectory["clip_id"],
                    "start_frame_idx": trajectory["start_frame_idx"],
                    "end_frame_idx": trajectory["end_frame_idx"],
                    "start_time_s": trajectory["start_time_s"],
                    "end_time_s": trajectory["end_time_s"],
                    "n_frames": trajectory["n_observations"],
                    "n_non_noop": trajectory["n_non_noop"],
                    "source_goal": trajectory["source_goal"],
                    "image_paths": image_paths,
                    "messages": messages,
                }
                record.update(
                    {key: trajectory[key] for key in PROVENANCE_KEYS if key in trajectory}
                )
                records.append(record)
            except Exception as exc:
                rejected.append(
                    {
                        "trajectory_id": trajectory["trajectory_id"],
                        "variant_idx": variant_idx,
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )

    split_of = assign_splits(records, split_group=split_group, seed=seed, val_frac=val_frac)
    for record in records:
        record["group_id"] = record[split_group]
        record["split"] = split_of[str(record[split_group])]
    write_jsonl(output_dir / "chat.jsonl", records)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    for split, split_records in by_split.items():
        write_jsonl(output_dir / split / "chat.jsonl", split_records)
    manifest = {
        "stage": "sft_projection",
        "artifact_type": "juergen_sft_view",
        "schema_version": 1,
        "source_trajectories": str(trajectories_path.resolve()),
        "action_schema": "aggregate_delta_keys_v1",
        "include_variants": include_variants,
        "split_group": split_group,
        "val_frac": val_frac,
        "seed": seed,
        "image_path_mode": image_path_mode,
        "message_policy": {
            "system_prompt": system_prompt_source,
            "terminal_mode": terminal_mode,
            "terminal_token": terminal_token,
        },
        "n_samples": len(records),
        "n_rejected": len(rejected),
        "counts_by_split": {split: len(items) for split, items in sorted(by_split.items())},
        "files": {"chat": "chat.jsonl", "rejected": "rejected.jsonl"},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument("--split-group", choices=VALID_SPLIT_GROUPS, default="recording_id")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-path-mode", choices=VALID_IMAGE_PATH_MODES, default="absolute")
    parser.add_argument("--system-prompt-text")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--terminal-token")
    parser.add_argument("--terminal-mode", choices=VALID_TERMINAL_MODES, default="none")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = project_sft(
        trajectories_path=args.trajectories,
        output_dir=args.output_dir,
        include_variants=args.include_variants,
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
    print(f"Wrote {manifest['n_samples']} SFT samples to {args.output_dir}")


if __name__ == "__main__":
    main()
