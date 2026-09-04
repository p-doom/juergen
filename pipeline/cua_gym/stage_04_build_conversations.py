"""Build the single-target CUA-Gym action-format parity conversations."""

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

from desktop.geometry import DisplayGeometry

from grammars.ordered_events_v3_relative_1000_grid_v1.codec import CODEC
from pipeline.cua_gym.stage_01_image_store import validate_image_store
from pipeline.cua_gym.translate import (
    rewrite_assistant,
    translate_step,
)
from pipeline.lib.image_store import parse_arrayrecord_image_uri
from pipeline.lib.manifest import make_artifact_id

SCREEN = (1920, 1080)
GEOMETRY = DisplayGeometry(desktop_width=SCREEN[0], desktop_height=SCREEN[1])
FAILURE_STEP_PERCENT = 25
MAX_COMPLETED_TURNS = 4
PREVIOUS_ACTIONS_MAX_CHARS = 160
OBSERVATION_CONTRACT = {
    "image_domain": "jpeg_q92_1920x1080",
    "media_type": "image/jpeg",
    "jpeg_quality": 92,
    "width": 1920,
    "height": 1080,
}
_COORDINATE_ACTIONS = {
    "click",
    "double_click",
    "left_click",
    "left_click_drag",
    "mouse_move",
    "right_click",
}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
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
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "action_spec_sha256": _canonical_digest(action),
        "observation_spec_sha256": _canonical_digest(OBSERVATION_CONTRACT),
        "render_spec_sha256": _canonical_digest(render),
        "observation": OBSERVATION_CONTRACT,
        "render": render,
    }


class ImageResolver(Protocol):
    def uri(self, shard: str, member: str) -> str: ...


def _source_action(step: dict[str, Any]) -> dict[str, Any] | None:
    raw = step.get("raw_action_args")
    if raw is None:
        if any(
            step.get(key) is not None
            for key in ("assistant_raw", "meta", "coordinate_screen")
        ):
            raise ValueError("null source action must not carry action metadata")
        return None
    if not isinstance(raw, dict):
        raise TypeError("raw_action_args must be an object or null")
    meta = step.get("meta")
    if not isinstance(meta, dict):
        raise TypeError("non-null source action requires executed meta")
    action = raw.get("action")
    if not isinstance(action, str):
        raise TypeError("source action must be text")
    if action in _COORDINATE_ACTIONS:
        if set(raw) != {"action", "coordinate"}:
            raise ValueError(f"unexpected source fields for {action}: {sorted(raw)}")
        if set(meta) != {"action", "pixel"} or meta.get("action") != action:
            raise ValueError(f"executed metadata mismatch for {action}")
        if meta["pixel"] != step.get("coordinate_screen"):
            raise ValueError(f"executed pixel mismatch for {action}")
        return raw
    if action == "type":
        if set(meta) != {"action", "text"} or meta != {
            "action": "type",
            "text": raw.get("text"),
        }:
            raise ValueError("executed metadata mismatch for type")
        return meta
    if action == "wait":
        if set(raw) != {"action", "time"} or set(meta) != {"action", "time"}:
            raise ValueError("wait requires raw and executed time")
        if meta.get("action") != action:
            raise ValueError("executed metadata mismatch for wait")
        return meta
    if raw != meta:
        raise ValueError(f"raw and executed action differ for {action}")
    return raw


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


def _sample_failure(task_id: str, step: int) -> bool:
    digest = hashlib.sha256(f"{task_id}:{step}".encode()).digest()
    value = int.from_bytes(digest[:8], "big")
    return value < (1 << 64) * FAILURE_STEP_PERCENT // 100


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
    previous = _previous_actions(evicted)
    return (
        "Please generate the next move according to the UI screenshot, instruction "
        "and previous actions.\n\n"
        f"Instruction: {instruction}\n\nPrevious actions:\n{previous}"
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


def build_episode_records(
    record: dict[str, Any],
    images: ImageResolver,
    contract: dict[str, Any],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    if tuple(record.get("screen") or ()) != SCREEN:
        raise ValueError(
            f"CUA-Gym parity requires screen {list(SCREEN)}, got {record.get('screen')!r}"
        )
    task_id = record.get("task_id")
    instruction = record.get("instruction")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("trajectory task_id must be non-empty text")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("trajectory instruction must be non-empty text")
    source_steps = record.get("steps")
    if not isinstance(source_steps, list) or not source_steps:
        raise ValueError(f"trajectory {task_id!r} has no steps")
    translated: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    previous_step = -1
    for source in source_steps:
        if not isinstance(source, dict):
            raise TypeError(f"trajectory step must be an object, got {source!r}")
        step = source.get("step")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step in seen_steps
        ):
            raise ValueError(
                f"trajectory step id must be unique and non-negative, got {step!r}"
            )
        if step <= previous_step:
            raise ValueError(
                f"trajectory steps must be strictly increasing, got {previous_step}, {step}"
            )
        seen_steps.add(step)
        previous_step = step
        counters["source_steps"] += 1
        arguments = _source_action(source)
        if arguments is None:
            counters["source_null_steps"] += 1
            continue
        shard = source.get("shard")
        member = source.get("member")
        if (
            not isinstance(shard, str)
            or not shard
            or not isinstance(member, str)
            or not member
        ):
            raise ValueError(f"trajectory step {step} has no screenshot identity")
        assistant_raw = source.get("assistant_raw")
        if not isinstance(assistant_raw, str):
            raise TypeError(f"trajectory step {step} has no assistant_raw")
        translation = translate_step(arguments, source.get("cursor_before"), GEOMETRY)
        if translation.target_pixel is not None and translation.target_pixel != tuple(
            source["coordinate_screen"]
        ):
            raise ValueError(
                f"trajectory step {step} coordinate does not match executed pixel"
            )
        action_text = translation.text
        translated.append(
            {
                "step": step,
                "image": images.uri(shard, member),
                "action_text": action_text,
                "assistant": rewrite_assistant(assistant_raw, translation.action),
            }
        )
        counters["translated_steps"] += 1
    if not translated:
        raise ValueError(
            f"recording_id={task_id!r} has no successfully parsed, executed actions"
        )
    counters["translated_rollouts"] += 1
    reward = record.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError(f"trajectory reward must be numeric, got {reward!r}")
    success = reward > 0
    rows = []
    for index, step in enumerate(translated):
        if not success and not _sample_failure(task_id, step["step"]):
            continue
        messages = _messages(instruction, translated, index, contract)
        serialized = json.dumps(messages, ensure_ascii=False)
        if "<tool_call>" in serialized:
            raise AssertionError(
                "native computer_use syntax leaked into parity history"
            )
        rows.append(
            {
                "conversation_id": f"{task_id}__s{step['step']:03d}",
                "recording_id": task_id,
                "task_id": task_id,
                "app": record.get("app"),
                "reward": reward,
                "pool": "success" if success else "failure",
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
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    trajectories: Path, image_store: Path, output_dir: Path
) -> dict[str, Any]:
    images = ImageIndex(image_store)
    contract = render_contract()
    image_store_id = make_artifact_id(image_store)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".chat.{os.getpid()}.jsonl"
    counters: Counter[str] = Counter()
    try:
        with (
            trajectories.open(encoding="utf-8") as source,
            temporary.open("w", encoding="utf-8") as target,
        ):
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                counters["rollouts"] += 1
                rows = build_episode_records(record, images, contract, counters)
                for row in rows:
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counters["records"] += 1
                    counters[f"records_{row['pool']}"] += 1
        if counters["rollouts"] == 0:
            raise ValueError(f"trajectory file is empty: {trajectories}")
        if counters["records"] == 0:
            raise ValueError(
                "no target records survived translation and deterministic sampling"
            )
        temporary.replace(output_dir / "chat.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "artifact_type": "cuagym_stage_04_conversations",
        "schema_version": 1,
        "chat": "chat.jsonl",
        "chat_sha256": _file_sha256(output_dir / "chat.jsonl"),
        "grammar": CODEC.name,
        "failure_step_percent": FAILURE_STEP_PERCENT,
        "inputs": {
            "trajectories": str(trajectories.resolve()),
            "trajectories_sha256": _file_sha256(trajectories),
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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(output_dir / "manifest.json")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--image_store", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_dataset(args.trajectories, args.image_store, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
