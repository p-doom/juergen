"""Off-shelf OSWorld single-task runner.

End-to-end:
  1. Spawn sglang serving Qwen3-VL-4B-Instruct (OpenAI-compat).
  2. Boot the OSWorld VM via the upstream ``ApptainerProvider``
     (native qemu+KVM extracted from tianon SIF — see
     ``project_osworld_kvm_path`` memory for why; this sidesteps
     the apptainer-userns KVM block on hai-*).
  3. Run a single OSWorld task with the verbatim
     ``mm_agents.qwen3vl_agent.Qwen3VLAgent`` from OSWorld upstream,
     pointed at our local sglang via ``api_backend="openai"``.
  4. Score via ``env.evaluate()`` and persist:
     - per-step screenshots + actions to ``output_dir/steps/``
     - rollout GIF for at-a-glance inspection
     - ``result.json`` consumed by labctl/pmanager

NOT a multi-task benchmark — for the off-shelf baseline number on the
full OSWorld test set, this would loop over the dataset. We're doing
one task end-to-end to validate the harness; multi-task is a small
extension.
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

# OSWorld upstream lives outside the eval venv. Caller must set
# OSWORLD_ROOT to the OSWorld checkout; PYTHONPATH-injection here keeps
# imports of ``desktop_env`` / ``mm_agents`` working without an install.
_OSWORLD_ROOT = os.environ.get("OSWORLD_ROOT")
if not _OSWORLD_ROOT:
    raise RuntimeError(
        "OSWORLD_ROOT env var is not set. Point it at your OSWorld checkout "
        "(the dir containing desktop_env/ and mm_agents/)."
    )
if _OSWORLD_ROOT not in sys.path:
    sys.path.insert(0, _OSWORLD_ROOT)

from result import write_result  # noqa: E402  must follow sys.path injection above
from sglang_runner import sglang_server  # noqa: E402
import sampling as sampling_mod  # noqa: E402

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Eval task dir.", required=True)

# Model:
flags.DEFINE_string("model_path", None, "HF model_id or local HF dir.", required=True)
flags.DEFINE_string(
    "served_model_name",
    None,
    "Name sglang serves the model under (sent in chat-completion requests). "
    "Empty = use model_path verbatim.",
    required=True,
)

# Task / agent params:
flags.DEFINE_string(
    "task_path",
    None,
    "Path to OSWorld task JSON under evaluation_examples/examples/.",
    required=True,
)
flags.DEFINE_string(
    "path_to_vm",
    None,
    "Path to the OSWorld Ubuntu.qcow2.",
    required=True,
)
flags.DEFINE_string(
    "provider_name",
    None,
    "DesktopEnv provider. 'apptainer' uses our native-qemu provider "
    "(see desktop_env/providers/apptainer/provider.py).",
    required=True,
)
flags.DEFINE_integer("max_steps", None, "Max agent rollout length.", required=True)
# Sampling: default to the Qwen-recommended tuple for the detected regime
# (see eval/sampling.py) instead of requiring a hand-set value. None = "use the
# Qwen recommendation"; pass a value to override. NOTE: the vendored
# mm_agents.qwen3vl_agent OpenAI backend COMMENTS OUT temperature/top_p
# (qwen3vl_agent.py:646-648) and never sends top_k/repetition_penalty/
# presence_penalty, so these only reach the sampler once the checkout patch in
# eval/patches/qwen3vl_agent_sampling.patch is applied (a warning fires below if
# it is not).
flags.DEFINE_float("temperature", None, "Sampling temperature (default: Qwen tuple).")
flags.DEFINE_float("top_p", None, "Nucleus top_p (default: Qwen tuple).")
flags.DEFINE_integer("top_k", None, "top_k (default: Qwen tuple = 20).")
flags.DEFINE_float("repetition_penalty", None, "repetition_penalty (default: Qwen tuple = 1.0).")
flags.DEFINE_float(
    "presence_penalty", None,
    "presence_penalty (default: Qwen Instruct = 1.5; pass 0 for our OSWorld runs "
    "— our A/B found 1.5 a near no-op for closed-loop repetition).")
flags.DEFINE_string("sampling_mode", "auto", "Regime: 'auto'|'instruct'|'thinking'.")
flags.DEFINE_bool("greedy", False, "Decode greedily (temperature 0). DISCOURAGED.")
flags.DEFINE_integer("max_tokens", None, "Max tokens per agent turn.", required=True)
flags.DEFINE_integer("history_n", None, "Frames of history per turn.", required=True)
flags.DEFINE_string(
    "coordinate_type",
    None,
    "'absolute' (raw pixels) or 'relative' (0-999 grid). Qwen3VLAgent default is 'relative'.",
    required=True,
)
flags.DEFINE_integer("screen_width", None, "VM screen width.", required=True)
flags.DEFINE_integer("screen_height", None, "VM screen height.", required=True)
flags.DEFINE_float(
    "sleep_after_execution",
    None,
    "Sleep between each dispatched action (seconds).",
    required=True,
)

# SGLang server:
flags.DEFINE_integer("sglang_port", None, "SGLang server port. 0=auto.", required=True)
flags.DEFINE_string("sglang_api_key", None, "SGLang server API key.", required=True)
flags.DEFINE_float(
    "mem_fraction_static",
    None,
    "SGLang --mem-fraction-static.",
    required=True,
)
flags.DEFINE_integer(
    "chunked_prefill_size",
    None,
    "SGLang --chunked-prefill-size.",
    required=True,
)


def _load_task(path: str) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def _save_step_artifacts(
    *,
    step_idx: int,
    obs: dict,
    response: str,
    action: str,
    reward: float,
    done: bool,
    info: dict,
    steps_dir: Path,
    traj_file,
) -> Image.Image | None:
    """Mirror OSWorld's per-step logging shape but under output_dir/steps/."""
    img = None
    if obs.get("screenshot"):
        img = Image.open(io.BytesIO(obs["screenshot"]))
        img.save(steps_dir / f"step_{step_idx:03d}.png")
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
    # Downscale for portable GIF size.
    target_w = 960
    scaled = []
    for f in frames:
        if f.width <= target_w:
            scaled.append(f.convert("RGB"))
            continue
        h = int(f.height * target_w / f.width)
        scaled.append(f.resize((target_w, h), Image.LANCZOS).convert("RGB"))
    scaled[0].save(
        path,
        save_all=True,
        append_images=scaled[1:],
        duration=900,
        loop=0,
        optimize=True,
    )


def main(_) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("osworld_runner")

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = output_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    sglang_log = output_dir / "sglang_server.log"
    traj_path = output_dir / "traj.jsonl"

    # Auto-derive sglang port from SLURM_JOB_ID to avoid collisions when
    # multiple eval jobs land on the same node.
    if FLAGS.sglang_port == 0:
        jid = int(os.environ.get("SLURM_JOB_ID", "0"))
        sglang_port = 30000 + (jid % 10000)
        log.info("auto-derived sglang_port=%d from SLURM_JOB_ID=%d", sglang_port, jid)
    else:
        sglang_port = FLAGS.sglang_port

    # Validate task + qcow2 paths early.
    task = _load_task(FLAGS.task_path)
    log.info("task id=%s instruction=%r", task["id"], task["instruction"])
    if not Path(FLAGS.path_to_vm).is_file():
        raise FileNotFoundError(f"path_to_vm not found: {FLAGS.path_to_vm}")

    served_name = FLAGS.served_model_name or FLAGS.model_path

    t_start = time.time()
    scores: list[float] = []
    n_steps_taken = 0
    stop_reason = "max_steps"

    with sglang_server(
        model_path=FLAGS.model_path,
        port=sglang_port,
        api_key=FLAGS.sglang_api_key,
        log_path=sglang_log,
        mem_fraction_static=FLAGS.mem_fraction_static,
        chunked_prefill_size=FLAGS.chunked_prefill_size,
        served_model_name=FLAGS.served_model_name or None,
    ) as server_url:
        sglang_base_url = server_url  # e.g. http://localhost:30000/v1
        log.info("sglang ready at %s", sglang_base_url)

        # Qwen3VLAgent reads OPENAI_BASE_URL + OPENAI_API_KEY from env in
        # its openai backend path.
        os.environ["OPENAI_BASE_URL"] = sglang_base_url
        os.environ["OPENAI_API_KEY"] = FLAGS.sglang_api_key

        # Lazy import: OPENAI_BASE_URL / _API_KEY must already be set in env
        # before mm_agents' openai backend reads them at import time.
        from desktop_env.desktop_env import DesktopEnv  # noqa: PLC0415
        from mm_agents.qwen3vl_agent import Qwen3VLAgent  # noqa: PLC0415

        sampling = sampling_mod.qwen_sampling(
            sampling_mod.detect_mode(
                model_path=FLAGS.model_path,
                mode=(None if FLAGS.sampling_mode == "auto" else FLAGS.sampling_mode),
            ),
            max_tokens=FLAGS.max_tokens,
            greedy=FLAGS.greedy,
            temperature=FLAGS.temperature,
            top_p=FLAGS.top_p,
            top_k=FLAGS.top_k,
            repetition_penalty=FLAGS.repetition_penalty,
            presence_penalty=FLAGS.presence_penalty,
        )
        log.info("sampling: %s", sampling.to_dict())
        agent = Qwen3VLAgent(**sampling_mod.openai_agent_kwargs(
            Qwen3VLAgent,
            sampling,
            base=dict(
                platform="ubuntu",
                model=served_name,
                action_space="pyautogui",
                observation_type="screenshot",
                history_n=FLAGS.history_n,
                coordinate_type=FLAGS.coordinate_type,
                api_backend="openai",
            ),
            logger=log,
        ))

        env = DesktopEnv(
            provider_name=FLAGS.provider_name,
            path_to_vm=FLAGS.path_to_vm,
            action_space="pyautogui",
            screen_size=(FLAGS.screen_width, FLAGS.screen_height),
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=False,
        )

        # Reset to the task starting state. ``env.reset(task_config=...)``
        # runs the per-task setup commands (e.g., click center of screen
        # to focus). Returns the initial observation dict.
        log.info("resetting env to task starting state")
        env.reset(task_config=task)
        agent.reset(log)

        # Brief settle delay matches lib_run_single (60s in upstream;
        # tightened since our VM boots clean each time).
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
                    log.warning("step %d: no actions parsed from response", step_idx + 1)
                    stop_reason = "no_actions_parsed"
                    break

                for action in actions:
                    obs, reward, done, info = env.step(action, FLAGS.sleep_after_execution)
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
            result = float(env.evaluate())
        except Exception as e:
            log.exception("env.evaluate raised: %s", e)
            result = float("nan")
        log.info("env.evaluate -> %.4f", result)
        scores.append(result)

        try:
            env.close()
        except Exception:
            log.warning("env.close failed:\n%s", traceback.format_exc())

    elapsed_s = int(time.time() - t_start)

    gif_path = output_dir / "rollout.gif"
    try:
        _write_gif(frames, gif_path)
        log.info("wrote rollout GIF (%d frames) → %s", len(frames), gif_path)
    except Exception as e:
        log.warning("GIF write failed: %s", e)

    write_result(
        output_dir / "result.json",
        task="osworld_one_task",
        scores={
            "reward": scores[0] if scores else float("nan"),
            "n_steps_taken": n_steps_taken,
            "stop_reason_code": (
                1 if stop_reason == "agent_terminate" else 2 if stop_reason == "max_steps" else 0
            ),
        },
        params={
            "task_id": task["id"],
            "task_instruction": task["instruction"],
            "task_snapshot": task.get("snapshot", ""),
            "provider_name": FLAGS.provider_name,
            "max_steps": FLAGS.max_steps,
            "sampling": sampling.to_dict(),
            # back-compat scalar keys (resolved values, not the raw flags)
            "temperature": 0.0 if sampling.greedy else sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "history_n": FLAGS.history_n,
            "coordinate_type": FLAGS.coordinate_type,
            "screen_width": FLAGS.screen_width,
            "screen_height": FLAGS.screen_height,
            "sleep_after_execution": FLAGS.sleep_after_execution,
            "mem_fraction_static": FLAGS.mem_fraction_static,
            "chunked_prefill_size": FLAGS.chunked_prefill_size,
            "stop_reason": stop_reason,
        },
        inputs={
            "model_path": FLAGS.model_path,
            "served_model_name": served_name,
            "task_path": FLAGS.task_path,
            "path_to_vm": FLAGS.path_to_vm,
        },
        n_samples=1,
        elapsed_s=elapsed_s,
        extra={"gif_path": str(gif_path), "traj_path": str(traj_path)},
    )
    log.info("done in %ds → %s", elapsed_s, output_dir / "result.json")


if __name__ == "__main__":
    app.run(main)
