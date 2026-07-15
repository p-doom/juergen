#!/usr/bin/env python3
"""Stage 06: project structured trajectories into the current SFT message format."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from annotation_pipeline.action_format import (
    ACTION_SCHEMAS,
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_CONTINUOUS_ACTION_HZ,
    ORDERED_ACTION_SCHEMA,
    ActionPrimitive,
    HeldStateDiagnostics,
    ProjectedAction,
    project_ordered_action,
    update_held_state,
)
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


def _project_aggregate_action(value: dict[str, Any]) -> ProjectedAction:
    action_bin = action_bin_from_dict(value)
    primitives: list[ActionPrimitive] = []
    dx = round(action_bin.move_dx)
    dy = round(action_bin.move_dy)
    scroll = round(action_bin.scroll)
    if dx != 0 or dy != 0:
        primitives.append(ActionPrimitive(kind="move", dx=dx, dy=dy))
    if scroll != 0:
        primitives.append(ActionPrimitive(kind="scroll", dx=0, dy=scroll))
    primitives.extend(
        ActionPrimitive(
            kind="down" if sign == "+" else "up",
            input_name=name,
        )
        for sign, name in action_bin.events
    )
    return ProjectedAction(
        text=format_action(action_bin),
        primitives=tuple(primitives),
    )


def render_messages(
    trajectory: dict[str, Any],
    *,
    instruction: str,
    source_dir: Path,
    image_path_mode: str,
    system_prompt: str | None,
    terminal_token: str | None,
    terminal_mode: str,
    action_schema: str,
    continuous_action_hz: float,
) -> tuple[list[dict[str, Any]], list[str], list[ProjectedAction]]:
    messages: list[dict[str, Any]] = []
    image_paths: list[str] = []
    projected_actions: list[ProjectedAction] = []
    for step_idx, step in enumerate(trajectory["steps"]):
        image = _image_path(str(step["image_path"]), source_dir=source_dir, mode=image_path_mode)
        image_paths.append(image)
        user_content: list[dict[str, str]] = []
        if step_idx == 0:
            user_content.append({"type": "text", "text": instruction})
        user_content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": user_content})
        if action_schema == ORDERED_ACTION_SCHEMA:
            projected = project_ordered_action(
                step["events"],
                interval_start_s=float(step["interval_start_s"]),
                continuous_action_hz=continuous_action_hz,
            )
        else:
            projected = _project_aggregate_action(step["action_bin"])
        projected_actions.append(projected)
        messages.append({"role": "assistant", "content": _text_content(projected.text)})
    if not messages:
        raise ValueError("trajectory has no steps")
    apply_terminal_policy(messages, terminal_token=terminal_token, terminal_mode=terminal_mode)
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": _text_content(system_prompt)})
    return messages, image_paths, projected_actions


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
    action_schema: str = DEFAULT_ACTION_SCHEMA,
    continuous_action_hz: float = DEFAULT_CONTINUOUS_ACTION_HZ,
    overwrite: bool = False,
) -> dict[str, Any]:
    if split_group not in VALID_SPLIT_GROUPS:
        raise ValueError(f"split_group must be one of {VALID_SPLIT_GROUPS}")
    if image_path_mode not in VALID_IMAGE_PATH_MODES:
        raise ValueError(f"image_path_mode must be one of {VALID_IMAGE_PATH_MODES}")
    if terminal_mode not in VALID_TERMINAL_MODES:
        raise ValueError(f"terminal_mode must be one of {VALID_TERMINAL_MODES}")
    if action_schema not in ACTION_SCHEMAS:
        raise ValueError(f"action_schema must be one of {ACTION_SCHEMAS}")
    if not math.isfinite(continuous_action_hz) or continuous_action_hz <= 0:
        raise ValueError("continuous_action_hz must be finite and positive")
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
    primitive_counts: Counter[str] = Counter()
    n_no_op_turns = 0
    state_diagnostics = HeldStateDiagnostics()
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
                messages, image_paths, projected_actions = render_messages(
                    trajectory,
                    instruction=instruction,
                    source_dir=source_dir,
                    image_path_mode=image_path_mode,
                    system_prompt=system_prompt,
                    terminal_token=terminal_token,
                    terminal_mode=terminal_mode,
                    action_schema=action_schema,
                    continuous_action_hz=continuous_action_hz,
                )
                sample_counts: Counter[str] = Counter()
                sample_no_ops = 0
                sample_diagnostics = HeldStateDiagnostics()
                held: set[str] = set()
                for projected in projected_actions:
                    if projected.text == "NO_OP":
                        sample_no_ops += 1
                    sample_counts.update(primitive.kind for primitive in projected.primitives)
                    update_held_state(
                        projected.primitives,
                        held=held,
                        diagnostics=sample_diagnostics,
                    )
                sample_diagnostics.finish_trajectory(held)
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
                    "n_non_noop": len(projected_actions) - sample_no_ops,
                    "source_goal": trajectory["source_goal"],
                    "image_paths": image_paths,
                    "messages": messages,
                }
                record.update(
                    {key: trajectory[key] for key in PROVENANCE_KEYS if key in trajectory}
                )
                records.append(record)
                primitive_counts.update(sample_counts)
                n_no_op_turns += sample_no_ops
                state_diagnostics.update(sample_diagnostics)
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
        "action_schema": action_schema,
        "continuous_action_hz": (
            continuous_action_hz if action_schema == ORDERED_ACTION_SCHEMA else None
        ),
        "primitive_counts": {
            kind: primitive_counts.get(kind, 0) for kind in ("move", "scroll", "down", "up")
        },
        "n_no_op_turns": n_no_op_turns,
        "state_diagnostics": state_diagnostics.to_dict(),
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
    parser.add_argument(
        "--action-schema",
        choices=ACTION_SCHEMAS,
        default=DEFAULT_ACTION_SCHEMA,
    )
    parser.add_argument(
        "--continuous-action-hz",
        type=float,
        default=DEFAULT_CONTINUOUS_ACTION_HZ,
    )
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
        action_schema=args.action_schema,
        continuous_action_hz=args.continuous_action_hz,
        overwrite=args.overwrite,
    )
    print(f"Wrote {manifest['n_samples']} SFT samples to {args.output_dir}")


if __name__ == "__main__":
    main()
