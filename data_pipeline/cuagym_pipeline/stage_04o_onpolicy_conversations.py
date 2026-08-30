from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from absl import app, flags

from cuagym_pipeline.onpolicy_accept import derive_action_line
from cuagym_pipeline.stage_04_build_conversations import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    build_step_record,
    strip_think,
)

FLAGS = flags.FLAGS

SOLVED_THRESHOLD = 0.999


def define_flags() -> None:
    flags.DEFINE_string(
        "accepted", None, "Path to accepted.jsonl from onpolicy_accept", required=True
    )
    flags.DEFINE_string("output_dir", None, "Output dataset dir", required=True)
    flags.DEFINE_string(
        "image_index_root",
        "",
        "stage_01 image store root over the rollout screenshots (empty = direct png paths)",
    )
    flags.DEFINE_string(
        "rollout_root",
        "",
        "Runner output root, used to resolve relative step image paths in direct-path mode",
    )
    flags.DEFINE_integer("history_n", 4, "Live history turns per record")
    flags.DEFINE_integer("limit", 0, "Max episodes (0 = all)")
    flags.DEFINE_string(
        "system_prompt_path", str(DEFAULT_SYSTEM_PROMPT_PATH), "System prompt file"
    )


class _RefImages:
    def uri(self, shard: str, member: str) -> str:
        return member


class DirectImageRefs:
    def __init__(self, rollout_root: Path | None = None):
        self._root = Path(rollout_root) if rollout_root else None

    def ref(self, rec: dict, step: dict) -> str | None:
        image_path = step.get("image_path")
        if not image_path:
            return None
        path = Path(image_path)
        if path.is_absolute() or self._root is None:
            return str(path)
        episode_dir = self._root / (rec.get("app_family") or "") / rec["task_id"]
        sample_index = rec.get("sample_index") or 0
        if sample_index:
            episode_dir = episode_dir / f"sample_{sample_index}"
        return str(episode_dir / path)


class ArImageRefs:
    def __init__(self, root: Path):
        self._by_member: dict[str, str] = {}
        for index_path in sorted(Path(root).glob("*/index.jsonl")):
            with index_path.open() as fh:
                for line in fh:
                    row = json.loads(line)
                    self._by_member[row["member"]] = row["uri"]
        if not self._by_member:
            raise SystemExit(f"no */index.jsonl entries under {root}")

    def ref(self, rec: dict, step: dict) -> str | None:
        candidates = []
        if step.get("member"):
            candidates.append(step["member"])
        image_path = step.get("image_path")
        if image_path:
            parts = Path(image_path).parts
            candidates.append("/".join(parts[-2:]))
            candidates.append(f"{rec.get('task_id')}/{parts[-1]}")
            if len(parts) >= 3:
                candidates.append("/".join(parts[-3:]))
        for candidate in candidates:
            uri = self._by_member.get(candidate)
            if uri:
                return uri
        return None


def episode_entries(rec: dict, refs, stats: Counter) -> list[dict]:
    entries = []
    for step in rec.get("steps") or []:
        raw = step.get("assistant_raw") or ""
        ref = refs.ref(rec, step)
        if not ref:
            stats["drop_missing_screenshot"] += 1
            continue
        if not raw:
            stats["drop_empty_assistant_raw"] += 1
            continue
        line = step.get("action_line") or derive_action_line(raw)
        entries.append(
            {
                "shard": ref,
                "member": ref,
                "step": step.get("step"),
                "target": raw,
                "line": line or None,
                "history_text": strip_think(raw).strip() or "NO_OP",
                "excluded": bool(step.get("excluded")),
                "excluded_reason": step.get("excluded_reason"),
                "dispatched": step.get("dispatched", True),
            }
        )
    return entries


def run(
    accepted_path: Path,
    out_dir: Path,
    *,
    image_index_root: Path | None,
    history_n: int,
    system_prompt_path: Path,
    rollout_root: Path | None = None,
    limit: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = Path(system_prompt_path).read_text().strip()
    refs = ArImageRefs(image_index_root) if image_index_root else DirectImageRefs(rollout_root)
    images = _RefImages()
    stats: Counter = Counter()
    n_rollouts = 0
    n_records = 0

    with open(accepted_path) as fh, open(out_dir / "chat.jsonl", "w") as out:
        for raw_line in fh:
            if not raw_line.strip():
                continue
            if limit and n_rollouts >= limit:
                break
            rec = json.loads(raw_line)
            if rec.get("accept") is False:
                stats["skipped_not_accepted"] += 1
                continue
            n_rollouts += 1
            reward = rec.get("reward")
            stratum = rec.get("stratum") or (
                "solved" if reward is not None and reward >= SOLVED_THRESHOLD else "partial"
            )
            task_id = rec["task_id"]
            sample_index = rec.get("sample_index") or 0
            recording_id = task_id if not sample_index else f"{task_id}__k{sample_index}"
            entries = episode_entries(rec, refs, stats)
            for t, entry in enumerate(entries):
                if entry["excluded"]:
                    stats[f"skipped_excluded_{entry['excluded_reason'] or 'unspecified'}"] += 1
                    continue
                if not entry["dispatched"]:
                    stats["skipped_not_dispatched"] += 1
                    continue
                messages = build_step_record(
                    rec,
                    entries,
                    t,
                    system_prompt=system_prompt,
                    images=images,
                    history_n=history_n,
                )
                step_idx = entry["step"] if entry["step"] is not None else t
                row = {
                    "conversation_id": f"{recording_id}__s{step_idx:03d}",
                    "recording_id": recording_id,
                    "task_id": task_id,
                    "app": rec.get("app_family") or rec.get("app"),
                    "app_type": rec.get("app_type"),
                    "reward": reward,
                    "terminated": rec.get("terminated"),
                    "stop_reason": rec.get("stop_reason"),
                    "sample_index": sample_index,
                    "pool": stratum,
                    "target_step": step_idx,
                    "n_history_turns": min(history_n, t),
                    "action_format": "ordered_events_v3",
                    "messages": messages,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_records += 1
                stats[f"records_{stratum}"] += 1

    report = {
        "rollouts": n_rollouts,
        "records": n_records,
        "history_n": history_n,
        "image_mode": "ar" if image_index_root else "path",
        **{k: v for k, v in sorted(stats.items())},
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv):
    del argv
    report = run(
        Path(FLAGS.accepted),
        Path(FLAGS.output_dir),
        image_index_root=Path(FLAGS.image_index_root) if FLAGS.image_index_root else None,
        history_n=FLAGS.history_n,
        system_prompt_path=Path(FLAGS.system_prompt_path),
        rollout_root=Path(FLAGS.rollout_root) if FLAGS.rollout_root else None,
        limit=FLAGS.limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    define_flags()
    app.run(main)
