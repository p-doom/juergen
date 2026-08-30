"""OSWorld full-benchmark runner — one task per SLURM array task.

Reads test_all.json (or another split), picks task at --task_index, runs
it end-to-end, and writes result.json to:
  {base_output_dir}/{app}/{task_id}/result.json            (--sample_index=0)
  {base_output_dir}/{app}/{task_id}/sample_{i}/result.json (--sample_index=i>0)

Sample 0 keeps the historical layout so pass@1 aggregation over
{app}/{task_id}/result.json is unaffected; pass@k runs add siblings.

Designed to be called from a SLURM --array job: --task_index must equal
$SLURM_ARRAY_TASK_ID, which the recipe wires up. When --sglang_port=0,
the runner derives a deterministic per-array-task sglang port from the
same env var. VM-host port collisions on multi-task nodes are handled
by ApptainerProvider's scan-based port allocator (no env var needed).
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from absl import app, flags
from PIL import Image

_OSWORLD_ROOT = os.environ.get("OSWORLD_ROOT")
if not _OSWORLD_ROOT:
    raise RuntimeError(
        "OSWORLD_ROOT env var is not set. Point it at your OSWorld checkout "
        "(the dir containing desktop_env/ and mm_agents/)."
    )
if _OSWORLD_ROOT not in sys.path:
    sys.path.insert(0, _OSWORLD_ROOT)

from result import write_result  # noqa: E402
from sglang_runner import sglang_server  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "base_output_dir",
    None,
    "Root output dir; task results go under {app}/{task_id}/.",
    required=True,
)
flags.DEFINE_string(
    "test_split_path", None, "Path to test split JSON (e.g. test_all.json).", required=True
)
flags.DEFINE_integer(
    "task_index", None, "Index into the sorted task list (0-based).", required=True
)
flags.DEFINE_integer(
    "sample_index",
    0,
    "pass@k sample index. 0 writes to {app}/{task_id}/ (historical layout); "
    "i>0 writes to {app}/{task_id}/sample_{i}/.",
)

flags.DEFINE_string("model_path", None, "HF model_id or local HF dir.", required=True)
flags.DEFINE_string("served_model_name", None, "Name sglang serves the model under.", required=True)
flags.DEFINE_string("path_to_vm", None, "Path to Ubuntu.qcow2.", required=True)
flags.DEFINE_string("provider_name", None, "DesktopEnv provider name.", required=True)
flags.DEFINE_integer("max_steps", None, "Max agent steps.", required=True)
flags.DEFINE_float("temperature", None, "Generation temperature.", required=True)
flags.DEFINE_float("top_p", None, "Nucleus sampling top_p.", required=True)
flags.DEFINE_integer("max_tokens", None, "Max tokens per turn.", required=True)
flags.DEFINE_integer("history_n", None, "History frames per turn.", required=True)
flags.DEFINE_string("coordinate_type", None, "'absolute' or 'relative'.", required=True)
flags.DEFINE_integer("screen_width", None, "VM screen width.", required=True)
flags.DEFINE_integer("screen_height", None, "VM screen height.", required=True)
flags.DEFINE_float("sleep_after_execution", None, "Sleep between actions (s).", required=True)

flags.DEFINE_integer(
    "sglang_port", None, "SGLang port. 0=auto from SLURM_ARRAY_TASK_ID.", required=True
)
flags.DEFINE_string("sglang_api_key", None, "SGLang API key.", required=True)
flags.DEFINE_float("mem_fraction_static", None, "SGLang --mem-fraction-static.", required=True)
flags.DEFINE_integer("chunked_prefill_size", None, "SGLang --chunked-prefill-size.", required=True)
flags.DEFINE_string(
    "cache_dir", "cache", "OSWorld task asset cache dir (default: 'cache' relative to cwd)."
)
flags.DEFINE_bool(
    "retry_on_env_error",
    False,
    "On env death (evaluate() exception or trailing missing-screenshot streak) "
    "write result_enverror.json instead of result.json so an outer attempt loop "
    "re-runs the sample; a second env death writes result.json as before.",
)


def _load_task_list(split_path: str) -> list[tuple[str, str]]:
    """Return sorted list of (app, task_id) pairs from the split JSON."""
    with Path(split_path).open() as f:
        d = json.load(f)
    return [(app, tid) for app, tids in sorted(d.items()) for tid in tids]


def _save_step_artifacts(
    *, step_idx, obs, response, action, reward, done, info, steps_dir, traj_file
):
    img = None
    if obs.get("screenshot"):
        img = Image.open(io.BytesIO(obs["screenshot"]))
        # Save via BytesIO → write_bytes to avoid PIL's internal os.getcwd()
        # call (abspath("") in the same-file collision check), which fails
        # when the process CWD is on a temporarily stale NFS mount.
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        (steps_dir / f"step_{step_idx:03d}.png").write_bytes(buf.getvalue())
    traj_file.write(
        json.dumps(
            {
                "step_num": step_idx,
                "action": action,
                "response": response,
                "reward": reward,
                "done": done,
                "info": info or {},
            }
        )
        + "\n"
    )
    traj_file.flush()
    return img


def _write_gif(frames: list[Image.Image], path: Path) -> None:
    if len(frames) < 2:
        return
    target_w = 960
    scaled = []
    for f in frames:
        if f.width <= target_w:
            scaled.append(f.convert("RGB"))
        else:
            h = int(f.height * target_w / f.width)
            scaled.append(f.resize((target_w, h), Image.LANCZOS).convert("RGB"))
    scaled[0].save(
        path, save_all=True, append_images=scaled[1:], duration=900, loop=0, optimize=True
    )


def main(_) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("osworld_fullbench")

    task_list = _load_task_list(FLAGS.test_split_path)
    if FLAGS.task_index >= len(task_list):
        raise ValueError(
            f"task_index {FLAGS.task_index} out of range (split has {len(task_list)} tasks)"
        )

    app_name, task_id = task_list[FLAGS.task_index]
    task_path = (
        Path(_OSWORLD_ROOT) / "evaluation_examples" / "examples" / app_name / f"{task_id}.json"
    )
    with task_path.open() as f:
        task = json.load(f)

    log.info(
        "task_index=%d  sample_index=%d  app=%s  task_id=%s",
        FLAGS.task_index,
        FLAGS.sample_index,
        app_name,
        task_id,
    )
    log.info("instruction: %r", task["instruction"])

    output_dir = Path(FLAGS.base_output_dir) / app_name / task_id
    if FLAGS.sample_index:
        output_dir = output_dir / f"sample_{FLAGS.sample_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = output_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    sglang_log = output_dir / "sglang_server.log"
    traj_path = output_dir / "traj.jsonl"

    # Skip if already completed (allows resuming interrupted array runs).
    result_path = output_dir / "result.json"
    if result_path.exists():
        log.info(
            "result.json already exists — skipping (task_index=%d sample_index=%d)",
            FLAGS.task_index,
            FLAGS.sample_index,
        )
        return

    if FLAGS.sglang_port == 0:
        array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", FLAGS.task_index))
        sglang_port = 30000 + (array_task_id % 1000)
        log.info(
            "auto-derived sglang_port=%d from SLURM_ARRAY_TASK_ID=%d", sglang_port, array_task_id
        )
    else:
        sglang_port = FLAGS.sglang_port

    t_start = time.time()
    final_reward: float = float("nan")
    n_steps_taken = 0
    stop_reason = "max_steps"
    env_error = ""
    missing_streak = 0

    with sglang_server(
        model_path=FLAGS.model_path,
        port=sglang_port,
        api_key=FLAGS.sglang_api_key,
        log_path=sglang_log,
        mem_fraction_static=FLAGS.mem_fraction_static,
        chunked_prefill_size=FLAGS.chunked_prefill_size,
        served_model_name=FLAGS.served_model_name,
    ) as server_url:
        os.environ["OPENAI_BASE_URL"] = server_url
        os.environ["OPENAI_API_KEY"] = FLAGS.sglang_api_key

        # Lazy import: OPENAI_BASE_URL / _API_KEY must already be set in env
        # before mm_agents' openai backend reads them at import time.
        from desktop_env.desktop_env import DesktopEnv  # noqa: PLC0415
        from mm_agents.qwen3vl_agent import Qwen3VLAgent  # noqa: PLC0415

        agent = Qwen3VLAgent(
            platform="ubuntu",
            model=FLAGS.served_model_name,
            max_tokens=FLAGS.max_tokens,
            top_p=FLAGS.top_p,
            temperature=FLAGS.temperature,
            action_space="pyautogui",
            observation_type="screenshot",
            history_n=FLAGS.history_n,
            coordinate_type=FLAGS.coordinate_type,
            api_backend="openai",
        )

        env = DesktopEnv(
            provider_name=FLAGS.provider_name,
            path_to_vm=FLAGS.path_to_vm,
            action_space="pyautogui",
            screen_size=(FLAGS.screen_width, FLAGS.screen_height),
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=False,
            cache_dir=FLAGS.cache_dir,
        )

        try:
            log.info("resetting env")
            env.reset(task_config=task)
            agent.reset(log)
            time.sleep(15)
            obs = env._get_obs()
            frames: list[Image.Image] = []

            with traj_path.open("w") as traj_f:
                initial_img = _save_step_artifacts(
                    step_idx=0,
                    obs=obs,
                    response="<reset>",
                    action="<reset>",
                    reward=0.0,
                    done=False,
                    info={},
                    steps_dir=steps_dir,
                    traj_file=traj_f,
                )
                if initial_img is not None:
                    frames.append(initial_img.convert("RGB"))

                done = False
                step_idx = 0
                while not done and step_idx < FLAGS.max_steps:
                    try:
                        response, actions = agent.predict(task["instruction"], obs)
                    except Exception as e:
                        log.error("agent.predict failed at step %d: %s", step_idx + 1, e)
                        stop_reason = f"agent_error: {e}"
                        break
                    log.info("step %d response: %r", step_idx + 1, (response or "")[:200])
                    log.info("step %d actions: %s", step_idx + 1, actions)

                    if not actions:
                        if response == "<truncated_think>":
                            log.warning("step %d: truncated think twice", step_idx + 1)
                            stop_reason = "truncated_think"
                        else:
                            log.warning("step %d: no actions parsed", step_idx + 1)
                            stop_reason = "no_actions_parsed"
                        break

                    for action in actions:
                        obs, reward, done, info = env.step(action, FLAGS.sleep_after_execution)
                        missing_streak = missing_streak + 1 if not obs.get("screenshot") else 0
                        img = _save_step_artifacts(
                            step_idx=step_idx + 1,
                            obs=obs,
                            response=response,
                            action=action,
                            reward=reward,
                            done=done,
                            info=info,
                            steps_dir=steps_dir,
                            traj_file=traj_f,
                        )
                        if img is not None:
                            frames.append(img.convert("RGB"))
                        if done:
                            stop_reason = "agent_terminate"
                            break
                    step_idx += 1
                    n_steps_taken = step_idx

            time.sleep(10)
            try:
                final_reward = float(env.evaluate())
            except Exception as e:
                log.exception("env.evaluate raised: %s", e)
                final_reward = float("nan")
                env_error = f"evaluate_exception: {e}"
            log.info("env.evaluate -> %.4f", final_reward)
            if not env_error and missing_streak >= 8:
                env_error = f"missing_screenshot_streak: {missing_streak}"
        finally:
            try:
                env.close()
            except Exception:
                log.warning("env.close failed:\n%s", traceback.format_exc())

    elapsed_s = int(time.time() - t_start)

    gif_path = output_dir / "rollout.gif"
    try:
        _write_gif(frames, gif_path)
    except Exception as e:
        log.warning("GIF write failed: %s", e)

    enverror_path = output_dir / "result_enverror.json"
    write_target = result_path
    if FLAGS.retry_on_env_error and env_error and not enverror_path.exists():
        write_target = enverror_path
    write_result(
        write_target,
        task="osworld_fullbench",
        scores={
            "reward": final_reward,
            "n_steps_taken": n_steps_taken,
            "stop_reason_code": (
                1 if stop_reason == "agent_terminate" else 2 if stop_reason == "max_steps" else 0
            ),
        },
        params={
            "task_index": FLAGS.task_index,
            "sample_index": FLAGS.sample_index,
            "task_id": task_id,
            "app": app_name,
            "task_instruction": task["instruction"],
            "task_snapshot": task.get("snapshot", ""),
            "provider_name": FLAGS.provider_name,
            "max_steps": FLAGS.max_steps,
            "temperature": FLAGS.temperature,
            "top_p": FLAGS.top_p,
            "max_tokens": FLAGS.max_tokens,
            "history_n": FLAGS.history_n,
            "coordinate_type": FLAGS.coordinate_type,
            "screen_width": FLAGS.screen_width,
            "screen_height": FLAGS.screen_height,
            "sleep_after_execution": FLAGS.sleep_after_execution,
            "stop_reason": stop_reason,
            "env_error": env_error,
        },
        inputs={
            "model_path": FLAGS.model_path,
            "served_model_name": FLAGS.served_model_name,
            "test_split_path": FLAGS.test_split_path,
            "path_to_vm": FLAGS.path_to_vm,
        },
        n_samples=1,
        elapsed_s=elapsed_s,
        extra={"gif_path": str(gif_path), "traj_path": str(traj_path)},
    )
    if write_target is not result_path:
        log.error("env death (%s): wrote %s; result.json withheld for retry", env_error, enverror_path)
    log.info(
        "done in %ds task_index=%d sample_index=%d app=%s task_id=%s reward=%.4f",
        elapsed_s,
        FLAGS.task_index,
        FLAGS.sample_index,
        app_name,
        task_id,
        final_reward,
    )


if __name__ == "__main__":
    app.run(main)
