from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))
EVAL_DIR = DATA_PIPELINE_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from absl import app, flags

from action_parser import parse_ordered_action
from cuagym_pipeline.translate import DropStep, rewrite_assistant, translate_step

FLAGS = flags.FLAGS
flags.DEFINE_string("trajectories", None, "Path to trajectories.jsonl", required=True)
flags.DEFINE_string(
    "image_index_root", None, "stage_01 image store root (per-tar index.jsonl)", required=True
)
flags.DEFINE_string("output_dir", None, "Output dataset dir", required=True)
flags.DEFINE_integer("chunk_size", 5, "Steps per conversation chunk")
flags.DEFINE_integer("limit", 0, "Max rollouts (0 = all)")
flags.DEFINE_string(
    "system_prompt_path",
    str(DATA_PIPELINE_DIR / "realigned_pipeline" / "system_prompts" / "cua_v3_cuagym.txt"),
    "System prompt file",
)

INSTRUCTION_TEMPLATE = """
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions}"""


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _image_block(image: str) -> dict:
    return {"type": "image", "image": image}


class ImageIndex:
    def __init__(self, root: Path):
        self._root = root
        self._by_tar: dict[str, dict[str, str]] = {}

    def uri(self, shard: str, member: str) -> str:
        tar = shard.removesuffix(".tar")
        if tar not in self._by_tar:
            index_path = self._root / tar / "index.jsonl"
            mapping: dict[str, str] = {}
            with open(index_path) as fh:
                for line in fh:
                    row = json.loads(line)
                    mapping[row["member"]] = row["uri"]
            self._by_tar[tar] = mapping
        return self._by_tar[tar][member]


def translate_episode(rec: dict, stats: Counter) -> list[dict]:
    screen = tuple(rec.get("screen") or (1920, 1080))
    kept = []
    for step in rec.get("steps") or []:
        if "assistant_raw" not in step or "raw_action_args" not in step:
            stats["drop_harness_parse_failure"] += 1
            continue
        cursor = step.get("cursor_before")
        if not cursor:
            stats["drop_missing_cursor"] += 1
            continue
        try:
            t = translate_step(step["raw_action_args"], tuple(cursor), screen)
        except DropStep as exc:
            stats[f"drop_{exc.reason.split(':')[0].replace(' ', '_')}"] += 1
            continue
        if t.dropped_reason:
            stats[f"drop_{t.dropped_reason}"] += 1
            continue
        if t.line != "TERMINATE":
            try:
                parse_ordered_action(t.line)
            except (ValueError, TypeError):
                stats["drop_unparseable_line"] += 1
                continue
        try:
            target = rewrite_assistant(step["assistant_raw"], t.line)
        except DropStep:
            stats["drop_rewrite_failure"] += 1
            continue
        kept.append(
            {
                "line": t.line,
                "target": target,
                "shard": step["shard"],
                "member": step["member"],
                "step": step.get("step"),
            }
        )
    return kept


def build_chunk_messages(
    steps: list[dict],
    chunk_start: int,
    chunk: list[dict],
    *,
    instruction: str,
    system_prompt: str,
    images: ImageIndex,
) -> list[dict]:
    previous = [f"Step {i + 1}: {steps[i]['line']}" for i in range(chunk_start)]
    previous_str = "\n".join(previous) if previous else "None"
    first_text = INSTRUCTION_TEMPLATE.format(
        instruction=instruction, previous_actions=previous_str
    )
    messages = [{"role": "system", "content": [_text_block(system_prompt)]}]
    for offset, step in enumerate(chunk):
        content = [_image_block(images.uri(step["shard"], step["member"]))]
        if offset == 0:
            content.append(_text_block(first_text))
        messages.append({"role": "user", "content": content})
        messages.append(
            {"role": "assistant", "content": [_text_block(step["target"])]}
        )
    return messages


def main(argv):
    del argv
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = Path(FLAGS.system_prompt_path).read_text().strip()
    images = ImageIndex(Path(FLAGS.image_index_root))
    stats: Counter = Counter()
    n_rollouts = 0
    n_conversations = 0

    with open(FLAGS.trajectories) as fh, open(out_dir / "chat.jsonl", "w") as out:
        for raw_line in fh:
            if FLAGS.limit and n_rollouts >= FLAGS.limit:
                break
            rec = json.loads(raw_line)
            n_rollouts += 1
            steps = translate_episode(rec, stats)
            if not steps:
                stats["episodes_empty"] += 1
                continue
            stats["steps_kept"] += len(steps)
            n_chunks = (len(steps) + FLAGS.chunk_size - 1) // FLAGS.chunk_size
            for ci in range(n_chunks):
                start = ci * FLAGS.chunk_size
                chunk = steps[start : start + FLAGS.chunk_size]
                messages = build_chunk_messages(
                    steps,
                    start,
                    chunk,
                    instruction=rec["instruction"],
                    system_prompt=system_prompt,
                    images=images,
                )
                row = {
                    "conversation_id": f"{rec['task_id']}__c{ci:02d}",
                    "task_id": rec["task_id"],
                    "app": rec.get("app"),
                    "reward": rec.get("reward"),
                    "terminated": rec.get("terminated"),
                    "chunk_index": ci,
                    "n_chunks": n_chunks,
                    "chunk_step_start": start,
                    "n_steps_in_chunk": len(chunk),
                    "action_format": "ordered_events_v3",
                    "messages": messages,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_conversations += 1

    report = {
        "rollouts": n_rollouts,
        "conversations": n_conversations,
        "chunk_size": FLAGS.chunk_size,
        **{k: v for k, v in sorted(stats.items())},
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    app.run(main)
