"""Render curated CUA-Gym turns as loss-masked parity conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grammars.ordered_events_v3_relative_1000_grid_v1.codec import (
    CODEC,
    action_from_dict,
)
from pipeline.cua_gym.stage_01_image_store import validate_image_store
from pipeline.cua_gym.stage_03_curate_trajectories import resolve_curated_artifact
from pipeline.lib.image_store import parse_arrayrecord_image_uri
from pipeline.lib.manifest import make_artifact_id

SCREEN = (1920, 1080)
MAX_COMPLETED_TURNS = 4
PREVIOUS_ACTIONS_MAX_CHARS = 160
OBSERVATION_CONTRACT = {
    "image_domain": "jpeg_q92_1920x1080",
    "media_type": "image/jpeg",
    "jpeg_quality": 92,
    "width": 1920,
    "height": 1080,
}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def render_contract() -> dict[str, Any]:
    system_prompt = CODEC.describe()
    action = {
        "grammar": CODEC.name,
        "prompt_sha256": CODEC.digest,
        "control": "TERMINATE: success|failure",
    }
    render = {
        "max_completed_turns": MAX_COMPLETED_TURNS,
        "previous_actions_max_chars": PREVIOUS_ACTIONS_MAX_CHARS,
        "history_assistant_loss": False,
    }
    return {
        "grammar": CODEC.name,
        "system_prompt": system_prompt,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "action_spec_sha256": _canonical_digest(action),
        "observation_spec_sha256": _canonical_digest(OBSERVATION_CONTRACT),
        "render_spec_sha256": _canonical_digest(render),
        "observation": OBSERVATION_CONTRACT,
        "render": render,
    }


class ImageResolver(Protocol):
    def uri(self, shard: str, member: str) -> str: ...


class ImageIndex:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest = validate_image_store(self.root)
        self.generation = self.root / str(manifest["generation"])
        self.source_tars = set(manifest["source_tars"])
        self._shards: dict[str, dict[str, str]] = {}

    def uri(self, shard: str, member: str) -> str:
        if shard not in self.source_tars:
            raise ValueError(f"unknown screenshot shard: {shard!r}")
        shard_name = shard.removesuffix(".tar")
        if shard_name not in self._shards:
            expected_path = (
                self.generation / shard_name / "images.array_record"
            ).resolve()
            mapping: dict[str, str] = {}
            index_path = self.generation / shard_name / "index.jsonl"
            try:
                lines = index_path.read_text(encoding="utf-8").splitlines()
                for expected_index, line in enumerate(lines):
                    row = json.loads(line)
                    if set(row) != {"member", "uri", "jpeg_sha256"}:
                        raise ValueError(f"invalid image index row {expected_index}")
                    name = row["member"]
                    uri = row["uri"]
                    if not isinstance(name, str) or not isinstance(uri, str):
                        raise TypeError("image index fields must be text")
                    if name in mapping:
                        raise ValueError(f"duplicate image member {name!r}")
                    path, record_index = parse_arrayrecord_image_uri(uri)
                    if (
                        path.resolve() != expected_path
                        or record_index != expected_index
                    ):
                        raise ValueError(
                            f"image index URI does not match row {expected_index}: {uri}"
                        )
                    mapping[name] = uri
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"cannot read image index {index_path}: {exc}"
                ) from exc
            if not mapping:
                raise ValueError(f"image index is empty: {index_path}")
            self._shards[shard_name] = mapping
        try:
            return self._shards[shard_name][member]
        except KeyError as exc:
            raise ValueError(
                f"image member {member!r} is absent from {shard!r}"
            ) from exc


def _text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def _image(value: str) -> dict[str, str]:
    return {"type": "image", "image": value}


def _previous_actions(evicted: list[tuple[int, str]]) -> str:
    if not evicted:
        return "None"
    value = "\n".join(
        f"Step {step}: {action.replace(chr(10), ' | ')}" for step, action in evicted
    )
    if len(value) <= PREVIOUS_ACTIONS_MAX_CHARS:
        return value
    marker = "…[earlier actions omitted]\n"
    return marker + value[-(PREVIOUS_ACTIONS_MAX_CHARS - len(marker)) :]


def _instruction(instruction: str, evicted: list[tuple[int, str]]) -> str:
    return (
        "Please generate the next move according to the UI screenshot, instruction "
        "and previous actions.\n\n"
        f"Instruction: {instruction}\n\nPrevious actions:\n{_previous_actions(evicted)}"
    )


def _messages(
    instruction: str,
    steps: list[dict[str, Any]],
    target_index: int,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    window_start = max(0, target_index - MAX_COMPLETED_TURNS)
    evicted = [
        (steps[index]["step"], steps[index]["action_text"])
        for index in range(window_start)
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [_text(contract["system_prompt"])]}
    ]
    for index in range(window_start, target_index + 1):
        content = [_image(steps[index]["image"])]
        if index == window_start:
            content.append(_text(_instruction(instruction, evicted)))
        messages.append({"role": "user", "content": content})
        assistant = {
            "role": "assistant",
            "content": [_text(steps[index]["assistant"])],
        }
        if index < target_index:
            assistant["loss"] = False
        messages.append(assistant)
    return messages


def _action_text(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {
        "primitives",
        "no_op",
        "terminate",
    }:
        raise ValueError("curated action must use the canonical action object")
    action = action_from_dict(value)
    if action.to_dict() != value or action.no_op != (not action.primitives):
        raise ValueError("curated action object is not canonical")
    return CODEC.format(action)


def build_episode_records(
    record: dict[str, Any],
    images: ImageResolver,
    contract: dict[str, Any],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    if set(record) != {"task_id", "instruction", "app", "screen", "steps"}:
        raise ValueError(f"invalid curated rollout fields: {sorted(record)}")
    if tuple(record["screen"]) != SCREEN:
        raise ValueError(f"CUA-Gym parity requires screen {list(SCREEN)}")
    task_id = record["task_id"]
    instruction = record["instruction"]
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("curated task_id must be non-empty text")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"curated trajectory {task_id!r} has no instruction")
    source_steps = record["steps"]
    if not isinstance(source_steps, list) or not source_steps:
        raise ValueError(f"curated trajectory {task_id!r} has no steps")
    translated = []
    previous_step = -1
    for source in source_steps:
        if not isinstance(source, dict) or set(source) != {
            "step",
            "shard",
            "member",
            "reasoning",
            "action",
        }:
            raise ValueError(f"{task_id}: invalid curated step")
        step = source["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
            raise ValueError(f"{task_id}: curated steps are not strictly increasing")
        previous_step = step
        reasoning = source["reasoning"]
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError(f"{task_id} step {step}: reasoning is empty")
        shard = source["shard"]
        member = source["member"]
        if not isinstance(shard, str) or not isinstance(member, str):
            raise TypeError(f"{task_id} step {step}: image identity is invalid")
        action_text = _action_text(source["action"])
        translated.append(
            {
                "step": step,
                "image": images.uri(shard, member),
                "action_text": action_text,
                "assistant": f"<think>{reasoning.strip()}</think>\n\n{action_text}",
            }
        )
    counters["rollouts"] += 1
    rows = []
    for index, step in enumerate(translated):
        messages = _messages(instruction, translated, index, contract)
        if "<tool_call>" in json.dumps(messages, ensure_ascii=False):
            raise AssertionError(
                "native computer_use syntax leaked into parity history"
            )
        rows.append(
            {
                "conversation_id": f"{task_id}__s{step['step']:03d}",
                "recording_id": task_id,
                "task_id": task_id,
                "app": record["app"],
                "target_step": step["step"],
                "n_history_turns": min(MAX_COMPLETED_TURNS, index),
                "grammar": CODEC.name,
                "system_prompt_sha256": contract["system_prompt_sha256"],
                "render_spec_sha256": contract["render_spec_sha256"],
                "action_spec_sha256": contract["action_spec_sha256"],
                "observation_spec_sha256": contract["observation_spec_sha256"],
                "messages": messages,
            }
        )
        counters["records"] += 1
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    curated_trajectories: Path, image_store: Path, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").unlink(missing_ok=True)
    trajectory_path, curated_manifest = resolve_curated_artifact(curated_trajectories)
    images = ImageIndex(image_store)
    contract = render_contract()
    image_store_id = make_artifact_id(image_store)
    curated_id = make_artifact_id(curated_trajectories)
    temporary = output_dir / f".chat.{os.getpid()}.jsonl"
    counters: Counter[str] = Counter()
    try:
        with (
            trajectory_path.open(encoding="utf-8") as source,
            temporary.open("w", encoding="utf-8") as target,
        ):
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise ValueError(
                        f"blank curated row at {trajectory_path}:{line_number}"
                    )
                for row in build_episode_records(
                    json.loads(line), images, contract, counters
                ):
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not counters["records"]:
            raise ValueError("curated artifact produced no training records")
        temporary.replace(output_dir / "chat.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "artifact_type": "cuagym_stage_04_conversations",
        "schema_version": 1,
        "chat": "chat.jsonl",
        "chat_sha256": _file_sha256(output_dir / "chat.jsonl"),
        "grammar": CODEC.name,
        "inputs": {
            "curated_trajectories": str(curated_trajectories.resolve()),
            "curated_trajectories_id": curated_id,
            "source_sha256": curated_manifest["inputs"]["source_sha256"],
            "image_store": str(image_store.resolve()),
            "image_store_id": image_store_id,
        },
        "contract": {
            key: value for key, value in contract.items() if key != "system_prompt"
        },
        "stats": dict(sorted(counters.items())),
    }
    temporary_manifest = output_dir / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(output_dir / "manifest.json")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated_trajectories", type=Path, required=True)
    parser.add_argument("--image_store", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_dataset(args.curated_trajectories, args.image_store, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
