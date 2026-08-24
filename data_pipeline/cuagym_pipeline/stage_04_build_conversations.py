from __future__ import annotations

import hashlib
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
flags.DEFINE_integer("history_n", 4, "Live history turns per record")
flags.DEFINE_integer("failure_step_percent", 25, "Percent of failure-episode steps kept as targets")
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


def strip_think(text: str) -> str:
    head, sep, tail = text.partition("</think>")
    if not sep:
        return text
    return tail.lstrip("\n")


def translate_episode(rec: dict, stats: Counter) -> list[dict]:
    screen = tuple(rec.get("screen") or (1920, 1080))
    steps = []
    for step in rec.get("steps") or []:
        entry = {
            "shard": step.get("shard"),
            "member": step.get("member"),
            "step": step.get("step"),
            "target": None,
            "line": None,
            "history_text": None,
        }
        raw = step.get("assistant_raw") or step.get("raw") or ""
        entry["history_text"] = strip_think(raw).strip() or "NO_OP"
        if "assistant_raw" in step and "raw_action_args" in step and step.get("cursor_before"):
            try:
                t = translate_step(
                    step["raw_action_args"], tuple(step["cursor_before"]), screen
                )
                if not t.dropped_reason:
                    if t.line != "TERMINATE":
                        parse_ordered_action(t.line)
                    entry["target"] = rewrite_assistant(step["assistant_raw"], t.line)
                    entry["line"] = t.line
                    entry["history_text"] = strip_think(entry["target"])
                else:
                    stats[f"drop_{t.dropped_reason}"] += 1
            except DropStep as exc:
                stats[f"drop_{exc.reason.split(':')[0].replace(' ', '_')}"] += 1
            except (ValueError, TypeError):
                stats["drop_unparseable_line"] += 1
        else:
            stats["drop_harness_parse_failure"] += 1
        if entry["shard"] and entry["member"]:
            steps.append(entry)
        else:
            stats["drop_missing_screenshot"] += 1
    return steps


def _is_target_sampled(task_id: str, step_idx, percent: int) -> bool:
    digest = hashlib.sha256(f"{task_id}:{step_idx}".encode()).digest()
    return digest[0] % 100 < percent


def build_step_record(
    rec: dict,
    steps: list[dict],
    t: int,
    *,
    system_prompt: str,
    images: ImageIndex,
    history_n: int,
) -> dict:
    window_start = max(0, t - history_n)
    previous = [
        f"Step {i + 1}: {steps[i]['line']}"
        for i in range(window_start)
        if steps[i]["line"] is not None
    ]
    previous_str = "\n".join(previous) if previous else "None"
    instruction_text = INSTRUCTION_TEMPLATE.format(
        instruction=rec["instruction"], previous_actions=previous_str
    )
    messages = [{"role": "system", "content": [_text_block(system_prompt)]}]
    for offset, i in enumerate(range(window_start, t)):
        content = [_image_block(images.uri(steps[i]["shard"], steps[i]["member"]))]
        if offset == 0:
            content.append(_text_block(instruction_text))
        messages.append({"role": "user", "content": content})
        messages.append(
            {
                "role": "assistant",
                "loss": False,
                "content": [_text_block(steps[i]["history_text"])],
            }
        )
    content = [_image_block(images.uri(steps[t]["shard"], steps[t]["member"]))]
    if window_start == t:
        content.append(_text_block(instruction_text))
    messages.append({"role": "user", "content": content})
    messages.append(
        {"role": "assistant", "content": [_text_block(steps[t]["target"])]}
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
    n_records = 0

    with open(FLAGS.trajectories) as fh, open(out_dir / "chat.jsonl", "w") as out:
        for raw_line in fh:
            if FLAGS.limit and n_rollouts >= FLAGS.limit:
                break
            rec = json.loads(raw_line)
            n_rollouts += 1
            reward = rec.get("reward")
            is_success = reward is not None and reward > 0
            pool = "success" if is_success else "failure"
            steps = translate_episode(rec, stats)
            for t, entry in enumerate(steps):
                if entry["target"] is None:
                    continue
                if not is_success and not _is_target_sampled(
                    rec["task_id"], entry["step"], FLAGS.failure_step_percent
                ):
                    stats["skipped_failure_subsample"] += 1
                    continue
                messages = build_step_record(
                    rec,
                    steps,
                    t,
                    system_prompt=system_prompt,
                    images=images,
                    history_n=FLAGS.history_n,
                )
                row = {
                    "conversation_id": f"{rec['task_id']}__s{entry['step']:03d}",
                    "task_id": rec["task_id"],
                    "app": rec.get("app"),
                    "reward": reward,
                    "terminated": rec.get("terminated"),
                    "pool": pool,
                    "target_step": entry["step"],
                    "n_history_turns": min(FLAGS.history_n, t),
                    "action_format": "ordered_events_v3",
                    "messages": messages,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_records += 1
                stats[f"records_{pool}"] += 1

    report = {
        "rollouts": n_rollouts,
        "records": n_records,
        "history_n": FLAGS.history_n,
        "failure_step_percent": FLAGS.failure_step_percent,
        **{k: v for k, v in sorted(stats.items())},
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    app.run(main)
