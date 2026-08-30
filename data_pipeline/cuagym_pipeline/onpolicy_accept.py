from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from absl import app, flags

from cuagym_pipeline.stage_04_build_conversations import _is_target_sampled, strip_think

FLAGS = flags.FLAGS

SOLVED_THRESHOLD = 0.999
DEFAULT_REJECT_STOP_REASONS = (
    "repetition_abort",
    "stale_screen_abort",
    "no_actions_parsed",
    "truncated_think",
)
IMAGE_HASH_KEYS = ("image_hash", "image_sha256", "frame_hash")


def define_flags() -> None:
    flags.DEFINE_string("trajectories", None, "Path to trajectories.jsonl (episode rows)")
    flags.DEFINE_string(
        "rollout_root",
        None,
        "Runner output tree: <root>/<app_family>/<task_id>/[sample_N/]{result.json, traj.jsonl}",
    )
    flags.DEFINE_string("output_dir", None, "Output dir for accepted/rejected jsonl", required=True)
    flags.DEFINE_boolean("quarantine_null_reward", True, "Quarantine episodes with null reward")
    flags.DEFINE_boolean("reject_bad_stop", True, "Reject episodes with a bad stop_reason")
    flags.DEFINE_list(
        "reject_stop_reasons", list(DEFAULT_REJECT_STOP_REASONS), "stop_reasons that reject"
    )
    flags.DEFINE_boolean(
        "reject_false_done", True, "Reject TERMINATE-ended episodes with reward < 0.999"
    )
    flags.DEFINE_boolean(
        "fail_requires_infeasible",
        True,
        "Solved FAIL-terminated episodes accept only on tasks flagged infeasible",
    )
    flags.DEFINE_list("infeasible_task_ids", [], "task_ids flagged infeasible")
    flags.DEFINE_string(
        "infeasible_task_ids_path", "", "File with infeasible task_ids (json list or one per line)"
    )
    flags.DEFINE_boolean(
        "accept_partial", False, "Accept 0<reward<0.999 max_steps episodes (reward-prop. steps)"
    )
    flags.DEFINE_boolean("trim_runs", True, "Trim runs of byte-identical action lines")
    flags.DEFINE_integer("trim_run_min", 3, "Minimum run length that triggers trimming")
    flags.DEFINE_boolean(
        "trim_without_hashes",
        True,
        "Trim on action identity alone when image hashes are absent; hashes gate runs when present",
    )
    flags.DEFINE_integer("limit", 0, "Max episodes (0 = all)")


def default_options() -> dict:
    return {
        "quarantine_null_reward": True,
        "reject_bad_stop": True,
        "reject_stop_reasons": set(DEFAULT_REJECT_STOP_REASONS),
        "reject_false_done": True,
        "fail_requires_infeasible": True,
        "infeasible_task_ids": set(),
        "accept_partial": False,
        "trim_runs": True,
        "trim_run_min": 3,
        "trim_without_hashes": True,
    }


def options_from_flags() -> dict:
    infeasible = set(FLAGS.infeasible_task_ids)
    if FLAGS.infeasible_task_ids_path:
        text = Path(FLAGS.infeasible_task_ids_path).read_text().strip()
        if text.startswith("["):
            infeasible.update(json.loads(text))
        else:
            infeasible.update(line.strip() for line in text.splitlines() if line.strip())
    return {
        "quarantine_null_reward": FLAGS.quarantine_null_reward,
        "reject_bad_stop": FLAGS.reject_bad_stop,
        "reject_stop_reasons": set(FLAGS.reject_stop_reasons),
        "reject_false_done": FLAGS.reject_false_done,
        "fail_requires_infeasible": FLAGS.fail_requires_infeasible,
        "infeasible_task_ids": infeasible,
        "accept_partial": FLAGS.accept_partial,
        "trim_runs": FLAGS.trim_runs,
        "trim_run_min": FLAGS.trim_run_min,
        "trim_without_hashes": FLAGS.trim_without_hashes,
    }


def _is_null_reward(reward) -> bool:
    return reward is None or (isinstance(reward, float) and math.isnan(reward))


def derive_action_line(assistant_raw: str) -> str:
    tail = strip_think(assistant_raw or "").strip()
    if not tail:
        return ""
    return tail.splitlines()[-1].strip()


def step_action_line(step: dict) -> str:
    return step.get("action_line") or derive_action_line(step.get("assistant_raw") or "")


def _ends_with_fail(steps: list[dict]) -> bool:
    return bool(steps) and step_action_line(steps[-1]) == "FAIL"


def _step_hash(step: dict):
    for key in IMAGE_HASH_KEYS:
        if step.get(key):
            return step[key]
    return None


def classify(rec: dict, opts: dict) -> dict:
    reward = rec.get("reward")
    quarantine_reasons: list[str] = []
    if _is_null_reward(reward):
        if opts["quarantine_null_reward"]:
            return {
                "accept": False,
                "stratum": "quarantine",
                "reasons": [],
                "quarantine_reasons": ["reward_null"],
            }
        reward = 0.0
    reward = float(reward)
    stop_reason = rec.get("stop_reason") or ""
    terminate_ended = (
        stop_reason == "agent_terminate" if stop_reason else bool(rec.get("terminated"))
    )
    solved = reward >= SOLVED_THRESHOLD
    if solved:
        stratum = "solved"
    elif 0 < reward and not terminate_ended:
        stratum = "partial"
    else:
        stratum = "failed"

    reasons: list[str] = []
    if opts["reject_bad_stop"] and stop_reason in opts["reject_stop_reasons"]:
        reasons.append(f"stop_reason:{stop_reason}")
    if opts["reject_false_done"] and terminate_ended and not solved:
        reasons.append("false_done")
    if (
        opts["fail_requires_infeasible"]
        and solved
        and _ends_with_fail(rec.get("steps") or [])
        and rec.get("task_id") not in opts["infeasible_task_ids"]
    ):
        reasons.append("fail_on_feasible")
    if stratum == "partial" and not opts["accept_partial"]:
        reasons.append("partial_disabled")
    if stratum == "failed" and not reasons:
        reasons.append("zero_reward")

    return {
        "accept": not reasons,
        "stratum": stratum,
        "reasons": reasons,
        "quarantine_reasons": quarantine_reasons,
    }


def annotate_steps(rec: dict, stratum: str, opts: dict, stats: Counter) -> list[dict]:
    steps = [dict(s) for s in rec.get("steps") or []]
    for s in steps:
        s.setdefault("excluded", False)

    if stratum == "partial":
        reward = float(rec.get("reward") or 0.0)
        percent = max(1, min(99, int(round(reward * 100))))
        key = f"{rec.get('task_id')}#k{rec.get('sample_index') or 0}"
        for i, s in enumerate(steps):
            step_idx = s.get("step") if s.get("step") is not None else i
            if not _is_target_sampled(key, step_idx, percent):
                s["excluded"] = True
                s["excluded_reason"] = "partial_subsample"
                stats["steps_excluded_partial_subsample"] += 1

    if opts["trim_runs"] and steps:
        hashes = [_step_hash(s) for s in steps]
        hashes_available = all(h is not None for h in hashes)
        if hashes_available or opts["trim_without_hashes"]:
            run_start = 0
            for i in range(1, len(steps) + 1):
                same = False
                if i < len(steps):
                    same = step_action_line(steps[i]) == step_action_line(steps[run_start])
                    if same and hashes_available:
                        same = hashes[i] == hashes[i - 1]
                if not same:
                    run_len = i - run_start
                    if run_len >= opts["trim_run_min"]:
                        for j in range(run_start + 2, i):
                            if not steps[j]["excluded"]:
                                steps[j]["excluded"] = True
                                steps[j]["excluded_reason"] = "run_trim"
                                stats["steps_excluded_run_trim"] += 1
                    run_start = i

    stats["steps_total_accepted"] += len(steps)
    stats["steps_excluded_total"] += sum(1 for s in steps if s["excluded"])
    return steps


def process_rows(rows, opts: dict, limit: int = 0):
    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    stats: Counter = Counter()
    reasons_hist: Counter = Counter()
    quarantine_hist: Counter = Counter()
    per_family: dict[str, Counter] = {}
    per_stratum: dict[str, Counter] = {}
    n = 0
    for rec in rows:
        if limit and n >= limit:
            break
        n += 1
        verdict = classify(rec, opts)
        out = dict(rec)
        out["accept"] = verdict["accept"]
        out["reasons"] = verdict["reasons"]
        out["stratum"] = verdict["stratum"]
        if verdict["quarantine_reasons"]:
            out["quarantine_reasons"] = verdict["quarantine_reasons"]
            quarantine_hist.update(verdict["quarantine_reasons"])
        reasons_hist.update(verdict["reasons"])

        family = rec.get("app_family") or rec.get("app") or "unknown"
        per_family.setdefault(family, Counter())["episodes"] += 1
        per_stratum.setdefault(verdict["stratum"], Counter())["episodes"] += 1

        if verdict["accept"]:
            out["steps"] = annotate_steps(rec, verdict["stratum"], opts, stats)
            accepted_rows.append(out)
            per_family[family]["accepted"] += 1
            per_stratum[verdict["stratum"]]["accepted"] += 1
        else:
            rejected_rows.append(out)

    def _rates(counters: dict[str, Counter]) -> dict:
        return {
            k: {
                "episodes": c["episodes"],
                "accepted": c["accepted"],
                "acceptance_rate": round(c["accepted"] / c["episodes"], 4) if c["episodes"] else 0.0,
            }
            for k, c in sorted(counters.items())
        }

    report = {
        "episodes": n,
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "acceptance_rate": round(len(accepted_rows) / n, 4) if n else 0.0,
        "per_app_family": _rates(per_family),
        "per_stratum": _rates(per_stratum),
        "reject_reason_histogram": dict(sorted(reasons_hist.items())),
        "quarantine_reason_histogram": dict(sorted(quarantine_hist.items())),
        **{k: v for k, v in sorted(stats.items())},
    }
    return accepted_rows, rejected_rows, report


def _legacy_step_rows(rows: list[dict], run_dir: Path) -> list[dict]:
    steps = []
    for r in rows:
        if r.get("step_num", 0) < 1 or r.get("response") in (None, "<reset>"):
            continue
        idx = r["step_num"] - 1
        raw = r.get("response") or ""
        steps.append(
            {
                "step": idx,
                "image_path": str(run_dir / "steps" / f"step_{idx:03d}.png"),
                "assistant_raw": raw,
                "action_line": derive_action_line(raw),
                "dispatched": True,
            }
        )
    return steps


def episode_from_run_dir(run_dir: Path, app_family: str, task_id: str) -> dict | None:
    result_path = run_dir / "result.json"
    traj_path = run_dir / "traj.jsonl"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    params = result.get("params") or {}
    scores = result.get("scores") or {}
    reward = scores.get("reward")
    if _is_null_reward(reward):
        reward = None
    rows: list[dict] = []
    if traj_path.exists():
        with traj_path.open() as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    if rows and any("assistant_raw" in r for r in rows):
        steps = [
            {
                "step": r.get("step"),
                "image_path": r.get("image_path"),
                "assistant_raw": r.get("assistant_raw") or "",
                "action_line": r.get("action_line") or derive_action_line(r.get("assistant_raw") or ""),
                "dispatched": r.get("dispatched", True),
                **{k: r[k] for k in IMAGE_HASH_KEYS if k in r},
            }
            for r in rows
            if "assistant_raw" in r
        ]
    else:
        steps = _legacy_step_rows(rows, run_dir)
    stop_reason = params.get("stop_reason") or ""
    return {
        "task_id": params.get("task_id") or task_id,
        "app_family": params.get("app") or app_family,
        "app_type": params.get("app_type"),
        "instruction": params.get("task_instruction"),
        "reward": reward,
        "terminated": stop_reason == "agent_terminate",
        "stop_reason": stop_reason,
        "screen": [params.get("screen_width") or 1920, params.get("screen_height") or 1080],
        "sample_index": params.get("sample_index") or 0,
        "steps": steps,
    }


def iter_rollout_root(root: Path):
    for task_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        app_family, task_id = task_dir.parts[-2], task_dir.parts[-1]
        run_dirs = [task_dir] + sorted(p for p in task_dir.glob("sample_*") if p.is_dir())
        for run_dir in run_dirs:
            rec = episode_from_run_dir(run_dir, app_family, task_id)
            if rec is not None:
                yield rec


def iter_trajectories(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main(argv):
    del argv
    if bool(FLAGS.trajectories) == bool(FLAGS.rollout_root):
        raise SystemExit("provide exactly one of --trajectories / --rollout_root")
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = options_from_flags()
    rows = (
        iter_trajectories(Path(FLAGS.trajectories))
        if FLAGS.trajectories
        else iter_rollout_root(Path(FLAGS.rollout_root))
    )
    accepted_rows, rejected_rows, report = process_rows(rows, opts, limit=FLAGS.limit)
    with open(out_dir / "accepted.jsonl", "w") as fh:
        for row in accepted_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "rejected.jsonl", "w") as fh:
        for row in rejected_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["options"] = {
        k: sorted(v) if isinstance(v, set) else v for k, v in sorted(opts.items())
    }
    (out_dir / "accept_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    define_flags()
    app.run(main)
