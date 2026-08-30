from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

DATA_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DATA_PIPELINE_ROOT.parent
for import_root in (DATA_PIPELINE_ROOT, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import stream_cuagym_qwen35 as stream_render  # noqa: E402

from cuagym_pipeline.translate import (  # noqa: E402
    DropStepError,
    rewrite_assistant,
    translate_step,
)


class ImageResolver(Protocol):
    def uri(self, shard: str, member: str) -> Any: ...


class ImageIndex:
    def __init__(self, root: Path) -> None:
        self._root = root
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read image-store manifest {manifest_path}: {exc}") from exc
        expected = {
            "artifact_type": "cuagym_stage_01_image_store",
            "schema_version": 1,
            "uri_scheme": "ar:///abs/path/images.array_record#idx",
            "jpeg_quality": 92,
        }
        observed = {key: manifest.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"image-store contract mismatch: expected {expected!r}, got {observed!r}"
            )
        self._by_tar: dict[str, dict[str, str]] = {}

    def uri(self, shard: str, member: str) -> str:
        tar = shard.removesuffix(".tar")
        if tar not in self._by_tar:
            index_path = self._root / tar / "index.jsonl"
            mapping: dict[str, str] = {}
            try:
                with index_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        name = row["member"]
                        if name in mapping:
                            raise ValueError(f"duplicate image member {name!r}")
                        mapping[name] = row["uri"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"cannot read image index {index_path}: {exc}") from exc
            self._by_tar[tar] = mapping
        try:
            return self._by_tar[tar][member]
        except KeyError as exc:
            raise ValueError(f"image member {member!r} is absent from shard {shard!r}") from exc


def _translate_episode(record: dict[str, Any], stats: Counter) -> list[dict[str, Any]]:
    screen = tuple(record.get("screen") or ())
    if screen != (1920, 1080):
        raise ValueError(f"CUA-Gym SFT requires screen [1920, 1080], got {screen!r}")
    translated: list[dict[str, Any]] = []
    for step in record.get("steps") or []:
        shard = step.get("shard")
        member = step.get("member")
        if not isinstance(shard, str) or not isinstance(member, str):
            stats["drop_missing_screenshot"] += 1
            continue
        raw = step.get("assistant_raw") or step.get("raw") or ""
        entry = {
            "shard": shard,
            "member": member,
            "step": int(step["step"]),
            "line": None,
            "target": None,
            "history_assistant": raw.strip() or "NO_OP",
        }
        if "assistant_raw" in step and "raw_action_args" in step and step.get("cursor_before"):
            try:
                action = translate_step(
                    step["raw_action_args"], tuple(step["cursor_before"]), screen
                )
                if action.dropped_reason:
                    stats[f"drop_{action.dropped_reason}"] += 1
                else:
                    target = rewrite_assistant(step["assistant_raw"], action.line)
                    entry.update(
                        line=action.line,
                        target=target,
                        history_assistant=target,
                    )
            except (DropStepError, KeyError, TypeError, ValueError) as exc:
                reason = exc.reason if isinstance(exc, DropStepError) else type(exc).__name__
                stats[f"drop_{str(reason).split(':', 1)[0].replace(' ', '_')}"] += 1
        else:
            stats["drop_harness_parse_failure"] += 1
        translated.append(entry)
    return translated


def _sample_failure(task_id: str, step: int, percent: int) -> bool:
    digest = hashlib.sha256(f"{task_id}:{step}".encode()).digest()
    return digest[0] % 100 < percent


def build_episode_records(
    record: dict[str, Any],
    images: ImageResolver,
    *,
    failure_step_percent: int = 25,
    stats: Counter | None = None,
) -> list[dict[str, Any]]:
    if not 0 <= failure_step_percent <= 100:
        raise ValueError("failure_step_percent must be in [0, 100]")
    instruction = record.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("CUA-Gym trajectory instruction must be non-empty text")
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("CUA-Gym trajectory task_id must be non-empty text")
    counters = stats if stats is not None else Counter()
    steps = _translate_episode(record, counters)
    if not steps:
        return []
    frames = [images.uri(step["shard"], step["member"]) for step in steps]
    renderer = stream_render.renderer()
    renderer.start(frames[0])
    reward = record.get("reward")
    success = isinstance(reward, (int, float)) and not isinstance(reward, bool) and reward > 0
    contract = stream_render.metadata()
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        sampled = success or _sample_failure(task_id, step["step"], failure_step_percent)
        if step["target"] is not None and sampled:
            conversation = renderer.render_prompt(instruction=instruction)
            for message in conversation:
                if message["role"] == "assistant":
                    message["loss"] = False
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": step["target"]}],
                }
            )
            rows.append(
                {
                    "conversation_id": f"{task_id}__s{step['step']:03d}",
                    "recording_id": task_id,
                    "task_id": task_id,
                    "app": record.get("app"),
                    "reward": reward,
                    "terminated": record.get("terminated"),
                    "pool": "success" if success else "failure",
                    "target_step": step["step"],
                    "n_history_turns": min(contract["max_completed_turns"], index),
                    "render": contract,
                    "messages": conversation,
                }
            )
            counters[f"records_{'success' if success else 'failure'}"] += 1
        elif step["target"] is not None:
            counters["skipped_failure_subsample"] += 1
        if index + 1 < len(steps):
            renderer.complete(
                assistant=step["history_assistant"],
                action=step["line"],
                next_image=frames[index + 1],
            )
    return rows


def build_dataset(
    trajectories: Path,
    image_index_root: Path,
    output_dir: Path,
    *,
    failure_step_percent: int = 25,
    limit: int = 0,
) -> dict[str, Any]:
    images = ImageIndex(image_index_root)
    contract = stream_render.metadata()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".chat.{os.getpid()}.jsonl"
    stats: Counter = Counter()
    rollouts = 0
    records = 0
    try:
        with (
            trajectories.open(encoding="utf-8") as source,
            temporary.open("w", encoding="utf-8") as target,
        ):
            for line in source:
                if limit and rollouts >= limit:
                    break
                record = json.loads(line)
                rollouts += 1
                for row in build_episode_records(
                    record,
                    images,
                    failure_step_percent=failure_step_percent,
                    stats=stats,
                ):
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    records += 1
        temporary.replace(output_dir / "chat.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "rollouts": rollouts,
        "records": records,
        "failure_step_percent": failure_step_percent,
        "render": contract,
        **dict(sorted(stats.items())),
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--image-index-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-step-percent", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dataset(
        args.trajectories,
        args.image_index_root,
        args.output_dir,
        failure_step_percent=args.failure_step_percent,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
