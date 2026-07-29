"""Format-AWARE closed-loop OSWorld task-success eval for move_rel / diffabs.

Same eval BACKEND as the absolute control (DesktopEnv+apptainer, env.reset setup,
env.evaluate() reward -> path A, apples-to-apples), but the AGENT uses each
format's own system prompt + parser + VM dispatch (so treatment checkpoints are
scored on their real behaviour, not zero'd by convention-mismatch):

  move_rel : moverel_system_prompt.txt + parse_computer_use_tool_calls +
             dispatch_computer_use(relative=True), 0-999 deltas scaled by rel_coord_grid=1000.
  diffabs  : diffabs_system_prompt.txt + parse_action_tolerant (bare "dx dy scroll ; +KEY")
             + dispatch_action.

Reuses freeroll._run_rollout (the validated format-aware rollout loop) pointed at
the DesktopEnv VM's in-VM Flask agent (http://localhost:{env.server_port}); then
env.evaluate() scores task-success on the SAME VM. Infeasible handling matches
DesktopEnv: action_history=["FAIL"] iff the agent explicitly declared failure
(stop_reason=="terminate_failure"; move_rel only — the diffabs grammar cannot
express failure, a real format property).

Result layout matches the baseline (aggregate.py reads scores.reward):
  {base_output_dir}/{app}/{task_id}/result.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

from absl import app, flags

_OSWORLD_ROOT = os.environ["OSWORLD_ROOT"]
if _OSWORLD_ROOT not in sys.path:
    sys.path.insert(0, _OSWORLD_ROOT)
_EVAL_DIR = "/fast/home/franz.srambical/juergen/eval"
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from result import write_result  # noqa: E402
from sglang_runner import sglang_server  # noqa: E402

_SPLIT_DIR = "/fast/home/franz.srambical/osworld_parity_split"
_PROMPTS = {
    "move_rel": (f"{_SPLIT_DIR}/moverel_system_prompt.txt", "computer_use_relative", 1000),
    "diffabs": (f"{_SPLIT_DIR}/diffabs_system_prompt.txt", "delta", 0),
    # deltatype = crowd-cast-native bare-token diffabs grammar + coalesced type("...") +
    # documented TERMINATE/FAIL. raw = raw-px deltas; norm = 0-999-of-screen grid (grid=1000).
    "deltatype_raw": (f"{_SPLIT_DIR}/deltatype_raw_system_prompt.txt", "deltatype", 0),
    "deltatype_norm": (f"{_SPLIT_DIR}/deltatype_norm_system_prompt.txt", "deltatype", 1000),
}

FLAGS = flags.FLAGS
flags.DEFINE_string("base_output_dir", None, "Root output dir.", required=True)
flags.DEFINE_string("test_split_path", f"{_SPLIT_DIR}/osworld_eval_heldout.json", "Split JSON.")
flags.DEFINE_integer("shard_index", 0, "Shard index.")
flags.DEFINE_integer("num_shards", 1, "Total shards.")
flags.DEFINE_enum("action_format", None, ["move_rel", "diffabs", "deltatype_raw", "deltatype_norm"], "Treatment format.", required=True)
flags.DEFINE_string("model_path", None, "HF checkpoint dir/id.", required=True)
flags.DEFINE_string("served_model_name", "policy", "sglang alias.")
flags.DEFINE_string("path_to_vm", "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2", "qcow2.")
flags.DEFINE_integer("max_steps", 15, "Max steps.")
flags.DEFINE_float("temperature", 0.0, "Temp.")
flags.DEFINE_integer("max_tokens", 256, "Max tokens/turn.")
flags.DEFINE_integer("n_history_frames", 4, "History frames (default 4 = match abs control; set to training value).")
flags.DEFINE_float("settle_s", 1.0, "Post-action settle.")
flags.DEFINE_float("settle_stable_timeout_s", 2.0, "Settle stability timeout.")
flags.DEFINE_float("settle_poll_s", 0.1, "Settle poll.")
flags.DEFINE_integer("sglang_port", 0, "0=auto.")
flags.DEFINE_string("sglang_api_key", "osworld", "key.")
flags.DEFINE_float("mem_fraction_static", 0.85, "sglang mem frac.")
flags.DEFINE_integer("chunked_prefill_size", 2048, "sglang chunked prefill.")
flags.DEFINE_float("top_p", -1.0, "Nucleus top_p (<0 = let sglang use generation_config).")
flags.DEFINE_integer("top_k", -1, "top_k (<0 = let sglang use generation_config).")
flags.DEFINE_float("presence_penalty", 0.0, "Presence penalty (0 = default/no anti-repetition).")


def _load_tasks(p):
    d = json.loads(Path(p).read_text())
    return [(a, t) for a, ts in sorted(d.items()) for t in ts]


def _run_one(app_name, task_id, sp_text, freeroll_fmt, rel_grid, server_url, log):
    from desktop_env.desktop_env import DesktopEnv
    from freeroll import _run_rollout

    task = json.loads((Path(_OSWORLD_ROOT) / "evaluation_examples" / "examples" / app_name / f"{task_id}.json").read_text())
    out = Path(FLAGS.base_output_dir) / app_name / task_id
    if (out / "result.json").exists():
        log.info("skip (done) %s/%s", app_name, task_id); return
    out.mkdir(parents=True, exist_ok=True)

    env = DesktopEnv(provider_name="apptainer", path_to_vm=FLAGS.path_to_vm, action_space="pyautogui",
                     screen_size=(1920, 1080), headless=True, os_type="Ubuntu",
                     require_a11y_tree=False, cache_dir=str(out / "cache"))
    t0 = time.time(); reward = float("nan"); n_steps = 0; stop_reason = "setup_error"; parse_errors = None
    try:
        env.reset(task_config=task)
        result = _run_rollout(
            sglang_url=server_url, api_key=FLAGS.sglang_api_key, model=FLAGS.served_model_name,
            osworld_url=f"http://localhost:{env.server_port}", output_dir=out,
            max_steps=FLAGS.max_steps, instruction=task["instruction"], system_prompt=sp_text,
            n_history_frames=FLAGS.n_history_frames, persist_instruction=True,
            max_tokens=FLAGS.max_tokens, temperature=FLAGS.temperature, save_frames=True,
            stop_on_click=False, desktop_setup="none", settle_s=FLAGS.settle_s,
            settle_stable_timeout_s=FLAGS.settle_stable_timeout_s, settle_poll_s=FLAGS.settle_poll_s,
            action_format=freeroll_fmt, rel_coord_grid=rel_grid,
            top_p=(FLAGS.top_p if FLAGS.top_p >= 0 else None),
            top_k=(FLAGS.top_k if FLAGS.top_k >= 0 else None),
            presence_penalty=(FLAGS.presence_penalty if FLAGS.presence_penalty else None),
        )
        stop_reason = result.get("stop_reason", "?"); n_steps = result.get("n_steps", 0)
        parse_errors = result.get("parse_errors", None)
        # DesktopEnv infeasible/FAIL semantics: FAIL only on an explicit failure declaration.
        env.action_history = ["FAIL"] if stop_reason == "terminate_failure" else []
        time.sleep(6)
        try:
            reward = float(env.evaluate())
        except Exception as e:
            log.exception("evaluate raised: %s", e); reward = float("nan")
        log.info("[%s/%s] fmt=%s reward=%.3f stop=%s parse_err=%s", app_name, task_id,
                 FLAGS.action_format, reward, stop_reason, result.get("parse_errors"))
    finally:
        try: env.close()
        except Exception: log.warning("close failed:\n%s", traceback.format_exc())

    # Robustness: model unreachable (sglang OOM/death/preempt) -> don't write a fake
    # result; leave unscored so a resumable relaunch retries (no fake-0 poisoning).
    if stop_reason == "model_error":
        log.warning("[%s/%s] model_error (sglang unreachable) -> skip write; retry on relaunch", app_name, task_id)
        return
    write_result(
        out / "result.json", task="osworld_format_eval",
        scores={"reward": reward, "n_steps_taken": n_steps,
                "stop_reason_code": (1 if str(stop_reason).startswith("terminate") else 2 if stop_reason == "max_steps" else 0)},
        params={"task_id": task_id, "app": app_name, "task_instruction": task["instruction"],
                "action_format": FLAGS.action_format, "freeroll_action_format": freeroll_fmt,
                "rel_coord_grid": rel_grid, "n_history_frames": FLAGS.n_history_frames,
                "max_steps": FLAGS.max_steps, "temperature": FLAGS.temperature,
                "max_tokens": FLAGS.max_tokens, "stop_reason": stop_reason,
                "parse_errors": parse_errors},
        inputs={"model_path": FLAGS.model_path, "served_model_name": FLAGS.served_model_name,
                "system_prompt_file": _PROMPTS[FLAGS.action_format][0], "test_split_path": FLAGS.test_split_path},
        n_samples=1, elapsed_s=int(time.time() - t0), extra={},
    )


def main(_):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("osworld_format_eval")
    sp_file, freeroll_fmt, rel_grid = _PROMPTS[FLAGS.action_format]
    sp_text = Path(sp_file).read_text()
    log.info("format=%s freeroll_fmt=%s rel_grid=%d sysprompt_len=%d", FLAGS.action_format, freeroll_fmt, rel_grid, len(sp_text))
    tasks = _load_tasks(FLAGS.test_split_path)
    mine = [(i, a, t) for i, (a, t) in enumerate(tasks) if i % FLAGS.num_shards == FLAGS.shard_index]
    log.info("shard %d/%d -> %d tasks", FLAGS.shard_index, FLAGS.num_shards, len(mine))
    base = Path(FLAGS.base_output_dir); base.mkdir(parents=True, exist_ok=True)
    with sglang_server(model_path=FLAGS.model_path, port=FLAGS.sglang_port, api_key=FLAGS.sglang_api_key,
                       log_path=base / f"sglang_shard{FLAGS.shard_index}.log",
                       mem_fraction_static=FLAGS.mem_fraction_static,
                       chunked_prefill_size=FLAGS.chunked_prefill_size,
                       served_model_name=FLAGS.served_model_name) as server_url:
        for i, a, t in mine:
            log.info("=== task %d (%s/%s) ===", i, a, t)
            try:
                _run_one(a, t, sp_text, freeroll_fmt, rel_grid, server_url, log)
            except Exception:
                log.error("task %s/%s crashed:\n%s", a, t, traceback.format_exc())
    log.info("shard %d done", FLAGS.shard_index)


if __name__ == "__main__":
    app.run(main)
