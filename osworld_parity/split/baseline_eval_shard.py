"""OSWorld closed-loop task-success eval, sharded to amortize the sglang load.

Same per-task logic as juergen/eval/osworld_fullbench_runner.py (DesktopEnv +
upstream mm_agents.qwen3vl_agent.Qwen3VLAgent, env.evaluate() -> reward) but
starts ONE sglang server and runs a shard of tasks against it (sglang cold JIT
is ~6.5 min, so per-task servers are wasteful for a 110-task set).

Shard selection: global sorted task index i is run by this shard iff
    i % num_shards == shard_index
Result layout matches the fullbench runner (osworld_score.py aggregates it):
    {base_output_dir}/{app}/{task_id}/result.json    (scores.reward in [0,1])
Resumable: a task whose result.json exists is skipped.
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from absl import app, flags
from PIL import Image

_OSWORLD_ROOT = os.environ["OSWORLD_ROOT"]
if _OSWORLD_ROOT not in sys.path:
    sys.path.insert(0, _OSWORLD_ROOT)
_EVAL_DIR = "/fast/home/franz.srambical/juergen/eval"
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from result import write_result  # noqa: E402
from sglang_runner import sglang_server  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string("base_output_dir", None, "Root output dir.", required=True)
flags.DEFINE_string("test_split_path", None, "Split JSON {app:[task_id]}.", required=True)
flags.DEFINE_integer("shard_index", 0, "This shard's index.")
flags.DEFINE_integer("num_shards", 1, "Total shards.")
flags.DEFINE_string("model_path", None, "HF dir / id.", required=True)
flags.DEFINE_string("served_model_name", "Qwen3-VL-8B-Instruct", "sglang alias.")
flags.DEFINE_string("path_to_vm", "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2", "qcow2.")
flags.DEFINE_string("provider_name", "apptainer", "DesktopEnv provider.")
flags.DEFINE_integer("max_steps", 15, "Max agent steps.")
flags.DEFINE_float("temperature", 0.0, "Temp.")
flags.DEFINE_float("top_p", 0.9, "top_p.")
flags.DEFINE_integer("max_tokens", 1024, "Max tokens/turn.")
flags.DEFINE_integer("history_n", 4, "History frames.")
flags.DEFINE_string("coordinate_type", "relative", "absolute|relative.")
flags.DEFINE_integer("screen_width", 1920, "W.")
flags.DEFINE_integer("screen_height", 1080, "H.")
flags.DEFINE_float("sleep_after_execution", 1.0, "Sleep between actions (s).")
flags.DEFINE_integer("sglang_port", 0, "0=auto free port.")
flags.DEFINE_string("sglang_api_key", "osworld", "sglang key.")
flags.DEFINE_float("mem_fraction_static", 0.85, "sglang mem frac.")
flags.DEFINE_integer("chunked_prefill_size", 2048, "sglang chunked prefill.")


def _load_task_list(p):
    d = json.loads(Path(p).read_text())
    return [(a, t) for a, ts in sorted(d.items()) for t in ts]


def _save_step(step_idx, obs, response, action, reward, done, info, steps_dir, traj_file):
    img = None
    if obs.get("screenshot"):
        img = Image.open(io.BytesIO(obs["screenshot"]))
        buf = io.BytesIO(); img.save(buf, format="PNG")
        (steps_dir / f"step_{step_idx:03d}.png").write_bytes(buf.getvalue())
    traj_file.write(json.dumps({"step_num": step_idx, "action": action, "response": response,
                                "reward": reward, "done": done, "info": info or {}}) + "\n")
    traj_file.flush()
    return img


def _run_task(app_name, task_id, log):
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.qwen3vl_agent import Qwen3VLAgent

    task_path = Path(_OSWORLD_ROOT) / "evaluation_examples" / "examples" / app_name / f"{task_id}.json"
    task = json.loads(task_path.read_text())
    out = Path(FLAGS.base_output_dir) / app_name / task_id
    result_path = out / "result.json"
    if result_path.exists():
        log.info("skip (done): %s/%s", app_name, task_id)
        return
    out.mkdir(parents=True, exist_ok=True)
    steps_dir = out / "steps"; steps_dir.mkdir(exist_ok=True)
    traj_path = out / "traj.jsonl"

    agent = Qwen3VLAgent(
        platform="ubuntu", model=FLAGS.served_model_name, max_tokens=FLAGS.max_tokens,
        top_p=FLAGS.top_p, temperature=FLAGS.temperature, action_space="pyautogui",
        observation_type="screenshot", history_n=FLAGS.history_n,
        coordinate_type=FLAGS.coordinate_type, api_backend="openai",
    )
    env = DesktopEnv(
        provider_name=FLAGS.provider_name, path_to_vm=FLAGS.path_to_vm, action_space="pyautogui",
        screen_size=(FLAGS.screen_width, FLAGS.screen_height), headless=True, os_type="Ubuntu",
        require_a11y_tree=False, cache_dir=str(out / "cache"),
    )
    t0 = time.time(); final_reward = float("nan"); n_steps = 0; stop_reason = "max_steps"
    try:
        env.reset(task_config=task)
        agent.reset(log)
        time.sleep(15)
        obs = env._get_obs()
        with traj_path.open("w") as tf:
            _save_step(0, obs, "<reset>", "<reset>", 0.0, False, {}, steps_dir, tf)
            done = False; step_idx = 0
            while not done and step_idx < FLAGS.max_steps:
                try:
                    response, actions = agent.predict(task["instruction"], obs)
                except Exception as e:
                    log.error("predict failed step %d: %s", step_idx + 1, e); stop_reason = f"agent_error: {e}"; break
                log.info("[%s/%s] step %d actions=%s", app_name, task_id, step_idx + 1, actions)
                # Persist the EXACT messages Qwen3VLAgent just sent, VERBATIM, for
                # a4b609f6's record-builder (no reconstruction of the eval input).
                # The agent writes ./draft/message_cache/messages_step_{len(actions)-1}.json
                # each predict; len(agent.actions) has just incremented by 1, so the
                # file for THIS step is messages_step_{step_idx}.json (agent current_step
                # == loop step_idx, both 0-based and +1 per predict). cwd is isolated
                # per-shard (see main) so concurrent shards don't clobber the cache.
                try:
                    _mc = Path("draft/message_cache") / f"messages_step_{step_idx}.json"
                    if _mc.is_file():
                        shutil.copy2(_mc, steps_dir / f"messages_step_{step_idx:03d}.json")
                except Exception as _e:
                    log.warning("[%s/%s] message-cache persist failed step %d: %s",
                                app_name, task_id, step_idx, _e)
                if not actions:
                    stop_reason = "no_actions_parsed"; break
                for action in actions:
                    obs, reward, done, info = env.step(action, FLAGS.sleep_after_execution)
                    _save_step(step_idx + 1, obs, response, action, reward, done, info, steps_dir, tf)
                    if done:
                        stop_reason = "agent_terminate"; break
                step_idx += 1; n_steps = step_idx
        time.sleep(8)
        try:
            final_reward = float(env.evaluate())
        except Exception as e:
            log.exception("evaluate raised: %s", e); final_reward = float("nan")
        log.info("[%s/%s] reward=%.4f stop=%s", app_name, task_id, final_reward, stop_reason)
    finally:
        try:
            env.close()
        except Exception:
            log.warning("env.close failed:\n%s", traceback.format_exc())

    # Robustness: if the model was unreachable (predict raised -> sglang OOM/death/
    # preempt), do NOT write a fake result -> leave the task unscored so a resumable
    # relaunch retries it (avoids poisoning the parity number with spurious 0s).
    if str(stop_reason).startswith("agent_error"):
        log.warning("[%s/%s] agent_error (model unreachable) -> skip write; retry on relaunch", app_name, task_id)
        return
    write_result(
        result_path, task="osworld_fullbench",
        scores={"reward": final_reward, "n_steps_taken": n_steps,
                "stop_reason_code": (1 if stop_reason == "agent_terminate" else 2 if stop_reason == "max_steps" else 0)},
        params={"task_id": task_id, "app": app_name, "task_instruction": task["instruction"],
                "max_steps": FLAGS.max_steps, "temperature": FLAGS.temperature, "top_p": FLAGS.top_p,
                "max_tokens": FLAGS.max_tokens, "history_n": FLAGS.history_n,
                "coordinate_type": FLAGS.coordinate_type, "stop_reason": stop_reason,
                "provider_name": FLAGS.provider_name},
        inputs={"model_path": FLAGS.model_path, "served_model_name": FLAGS.served_model_name,
                "test_split_path": FLAGS.test_split_path},
        n_samples=1, elapsed_s=int(time.time() - t0), extra={"traj_path": str(traj_path)},
    )


def main(_):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("osworld_baseline_shard")
    # Isolate Qwen3VLAgent's hardcoded ./draft/message_cache per shard so
    # concurrent array shards don't clobber each other's message captures.
    # All output paths (base_output_dir, cache_dir, sglang log) are absolute,
    # so chdir only affects the agent's relative ./draft dir. Safe.
    _mc_cwd = Path(FLAGS.base_output_dir).resolve() / "_run_cwd" / f"shard{FLAGS.shard_index}"
    _mc_cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(_mc_cwd)
    tasks = _load_task_list(FLAGS.test_split_path)
    mine = [(i, a, t) for i, (a, t) in enumerate(tasks) if i % FLAGS.num_shards == FLAGS.shard_index]
    log.info("shard %d/%d -> %d tasks", FLAGS.shard_index, FLAGS.num_shards, len(mine))
    port = FLAGS.sglang_port
    logdir = Path(FLAGS.base_output_dir); logdir.mkdir(parents=True, exist_ok=True)
    with sglang_server(model_path=FLAGS.model_path, port=port, api_key=FLAGS.sglang_api_key,
                       log_path=logdir / f"sglang_shard{FLAGS.shard_index}.log",
                       mem_fraction_static=FLAGS.mem_fraction_static,
                       chunked_prefill_size=FLAGS.chunked_prefill_size,
                       served_model_name=FLAGS.served_model_name) as server_url:
        os.environ["OPENAI_BASE_URL"] = server_url
        os.environ["OPENAI_API_KEY"] = FLAGS.sglang_api_key
        for i, a, t in mine:
            log.info("=== task %d (%s/%s) ===", i, a, t)
            try:
                _run_task(a, t, log)
            except Exception:
                log.error("task %s/%s crashed:\n%s", a, t, traceback.format_exc())
    log.info("shard %d done", FLAGS.shard_index)


if __name__ == "__main__":
    app.run(main)
