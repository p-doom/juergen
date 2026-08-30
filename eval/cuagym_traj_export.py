"""Convert finished CUA-Gym episode dirs into trajectories.jsonl rows for
downstream conversion. One row per episode; steps pair each assistant
response with the screenshot it observed (the previous step's PNG). Episodes
are already in oev3 format, so no action translation happens here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from absl import app, flags

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from oev3_agent import extract_action_line

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("episode_dir", [], "Episode dir(s) containing result.json + traj.jsonl.")
flags.DEFINE_string("runs_root", "", "Scan this root for cuagym_rollout episode dirs.")
flags.DEFINE_string("out", "", "Output trajectories.jsonl path.")

TERMINAL_STOP_REASONS = ("agent_terminate", "agent_fail")


def export_episode(episode_dir: Path | str) -> dict:
    episode_dir = Path(episode_dir)
    result = json.loads((episode_dir / "result.json").read_text())
    params = result["params"]
    scores = result["scores"]

    steps = []
    with (episode_dir / "traj.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step_num = row["step_num"]
            if step_num == 0:
                continue
            response = row.get("response") or ""
            try:
                action_line = extract_action_line(response)
            except ValueError:
                action_line = None
            steps.append(
                {
                    "step": step_num,
                    "image_path": f"steps/step_{step_num - 1:03d}.png",
                    "assistant_raw": response,
                    "action_line": action_line,
                    "dispatched": row.get("action"),
                }
            )

    stop_reason = params.get("stop_reason", "")
    return {
        "task_id": params["task_id"],
        "app_family": params.get("app_family", ""),
        "app_type": params.get("app_type", ""),
        "instruction": params.get("instruction", ""),
        "reward": scores.get("reward"),
        "terminated": stop_reason in TERMINAL_STOP_REASONS,
        "stop_reason": stop_reason,
        "screen": [params.get("screen_width", 1920), params.get("screen_height", 1080)],
        "sample_index": params.get("sample_index", 0),
        "steps": steps,
    }


def _find_episode_dirs(runs_root: Path) -> list[Path]:
    dirs = []
    for result_path in sorted(runs_root.rglob("result.json")):
        episode_dir = result_path.parent
        if not (episode_dir / "traj.jsonl").exists():
            continue
        try:
            if json.loads(result_path.read_text()).get("task") != "cuagym_rollout":
                continue
        except (json.JSONDecodeError, OSError):
            continue
        dirs.append(episode_dir)
    return dirs


def main(_) -> None:
    if not FLAGS.out:
        raise ValueError("--out is required")
    episode_dirs = [Path(d) for d in FLAGS.episode_dir]
    if FLAGS.runs_root:
        episode_dirs.extend(_find_episode_dirs(Path(FLAGS.runs_root)))
    if not episode_dirs:
        raise ValueError("no episode dirs (pass --episode_dir and/or --runs_root)")

    out_path = Path(FLAGS.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for episode_dir in episode_dirs:
            f.write(json.dumps(export_episode(episode_dir)) + "\n")
            n += 1
    print(f"wrote {n} trajectory row(s) to {out_path}")


if __name__ == "__main__":
    app.run(main)
