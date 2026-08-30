"""On-policy CUA-Gym rollout driver for Oev3Agent.

Modeled on osworld_fullbench_runner's episode loop; output layout matches the
pass@k eval dirs: {base_output_dir}/{app_family}/{task_id}/[sample_N/]
{result.json, traj.jsonl, steps/}. The model is expected to be already served
(OPENAI_BASE_URL); rewards come from each bundle's reward.py via
cuagym_reward. --dry_run validates task adaptation without a VM.
"""

from __future__ import annotations

import hashlib
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

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

os.environ.setdefault("OSWORLD_ROOT", "/fast/project/HFMI_SynergyUnit/yll/osworld-pinned")
_OSWORLD_ROOT = os.environ["OSWORLD_ROOT"]
if _OSWORLD_ROOT not in sys.path:
    sys.path.insert(0, _OSWORLD_ROOT)

import cuagym_reward
from cuagym_task_adapter import (
    DEFAULT_TASKS_PARQUET,
    DEFAULT_TASKS_ROOT,
    AdaptedTask,
    UnsupportedTaskError,
    load_metadata,
    load_task,
    seed_cache,
)
from oev3_agent import Oev3Agent
from result import write_result

FLAGS = flags.FLAGS

flags.DEFINE_string("base_output_dir", "", "Root output dir; episodes go under {app_family}/{task_id}/.")
flags.DEFINE_string("tasks_root", str(DEFAULT_TASKS_ROOT), "CUA-Gym tasks_v1 bundle root.")
flags.DEFINE_string("tasks_parquet", str(DEFAULT_TASKS_PARQUET), "CUA-Gym tasks.parquet with task metadata.")
flags.DEFINE_string("task_id", "", "Single task id to roll out.")
flags.DEFINE_string("tasks_file", "", "JSONL file of task ids (bare string or object with task_id/id).")
flags.DEFINE_integer("task_index", -1, "Index into tasks_file (0-based); -1 runs all listed tasks.")
flags.DEFINE_integer("sample_index", 0, "First pass@k sample index. 0 uses the historical task-dir layout.")
flags.DEFINE_integer("k", 1, "Number of samples per task, sample_index..sample_index+k-1.")

flags.DEFINE_string("served_model_name", "oev3", "Model name the OPENAI_BASE_URL server exposes.")
flags.DEFINE_string("provider_name", "", "DesktopEnv provider name.")
flags.DEFINE_string("path_to_vm", "", "Path to Ubuntu.qcow2.")
flags.DEFINE_bool("use_qemu_kvm", False, "Lease VM ports and install the local qemu KVM provider.")

flags.DEFINE_integer("max_steps", 80, "Max agent steps per episode.")
flags.DEFINE_float("temperature", 0.8, "Generation temperature.")
flags.DEFINE_float("presence_penalty", 0.0, "Presence penalty (OEV3_PRESENCE_PENALTY passthrough; 0 disables).")
flags.DEFINE_float("top_p", 0.95, "Nucleus sampling top_p.")
flags.DEFINE_integer("max_tokens", 32768, "Max tokens per turn.")
flags.DEFINE_integer("history_n", 4, "History frames per turn.")
flags.DEFINE_integer("screen_width", 1920, "VM screen width.")
flags.DEFINE_integer("screen_height", 1080, "VM screen height.")
flags.DEFINE_float("sleep_after_execution", 2.0, "Sleep between actions (s).")
flags.DEFINE_string("cache_dir", "cache", "OSWorld setup cache dir (seeded from the bundle before reset).")
flags.DEFINE_string("reward_guest_path", cuagym_reward.DEFAULT_GUEST_PATH, "Guest path for the uploaded reward.py.")
flags.DEFINE_float("reward_timeout_s", cuagym_reward.DEFAULT_TIMEOUT_S, "Reward script execution timeout (s).")
flags.DEFINE_string("reward_wheels_dir", "", "Host dir of guest wheels to pip-install before the reward step; empty disables.")

flags.DEFINE_bool("dry_run", False, "Adapt task(s), print setup/evaluator/reward plan, exit without a VM.")

REPEAT_ACTION_WINDOW = 4
REPEAT_SCREEN_WINDOW = 3
STALE_SCREEN_WINDOW = 8
NOOP_RESAMPLE_MAX = 2
NOOP_RESAMPLE_TEMP_BUMP = 0.2


def screen_hash(png_bytes: bytes | None) -> str | None:
    if not png_bytes:
        return None
    return hashlib.sha1(png_bytes).hexdigest()


def _all_equal_non_none(values) -> bool:
    return bool(values) and values[0] is not None and all(v == values[0] for v in values)


def should_abort_repetition(action_lines: list, screen_hashes: list) -> bool:
    if len(action_lines) < REPEAT_ACTION_WINDOW or len(screen_hashes) < REPEAT_SCREEN_WINDOW:
        return False
    return _all_equal_non_none(action_lines[-REPEAT_ACTION_WINDOW:]) and _all_equal_non_none(
        screen_hashes[-REPEAT_SCREEN_WINDOW:]
    )


def should_abort_stale_screen(screen_hashes: list) -> bool:
    if len(screen_hashes) < STALE_SCREEN_WINDOW:
        return False
    return _all_equal_non_none(screen_hashes[-STALE_SCREEN_WINDOW:])


def _pop_history(agent) -> tuple:
    return (
        agent.screenshots.pop(),
        agent.stripped_responses.pop(),
        agent.action_lines.pop(),
    )


def _push_history(agent, entry: tuple) -> None:
    shot, resp, line = entry
    agent.screenshots.append(shot)
    agent.stripped_responses.append(resp)
    agent.action_lines.append(line)


def predict_with_noop_resample(
    agent,
    instruction: str,
    obs: dict,
    prev_line: str | None,
    screen_unchanged: bool,
    max_resamples: int = NOOP_RESAMPLE_MAX,
    temp_bump: float = NOOP_RESAMPLE_TEMP_BUMP,
) -> tuple[str, list, int]:
    n_before = len(agent.action_lines)
    response, actions = agent.predict(instruction, obs)
    n_resamples = 0
    appended = len(agent.action_lines) > n_before
    if not actions or not appended or prev_line is None or not screen_unchanged:
        return response, actions, n_resamples
    base_temperature = agent.temperature
    try:
        while n_resamples < max_resamples and agent.action_lines[-1] == prev_line:
            saved = _pop_history(agent)
            agent.temperature = base_temperature + temp_bump * (n_resamples + 1)
            n_resamples += 1
            new_response, new_actions = agent.predict(instruction, obs)
            if not new_actions or len(agent.action_lines) <= n_before:
                _push_history(agent, saved)
                break
            response, actions = new_response, new_actions
    finally:
        agent.temperature = base_temperature
    return response, actions, n_resamples


def _save_step_artifacts(
    *, step_idx, obs, response, action, reward, done, info, steps_dir, traj_file
):
    img = None
    if obs.get("screenshot"):
        img = Image.open(io.BytesIO(obs["screenshot"]))
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


def _write_gif(frames: list, path: Path) -> None:
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


def run_episode(
    env,
    agent,
    adapted: AdaptedTask,
    output_dir: Path,
    *,
    max_steps: int,
    sleep_after_execution: float,
    reward_guest_path: str,
    reward_timeout_s: float,
    reward_wheels_dir: str,
    log,
) -> dict:
    steps_dir = output_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / "traj.jsonl"

    log.info("resetting env for task %s", adapted.task_id)
    env.reset(task_config=adapted.task_config)
    agent.reset(log)
    time.sleep(15)
    obs = env._get_obs()

    frames = []
    emitted_lines: list = []
    screen_hashes = [screen_hash(obs.get("screenshot"))]
    stop_reason = "max_steps"
    resample_count = 0
    missing_streak = 0
    n_steps_taken = 0

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
        while not done and step_idx < max_steps:
            prev_line = emitted_lines[-1] if emitted_lines else None
            screen_unchanged = (
                len(screen_hashes) >= 2
                and screen_hashes[-1] is not None
                and screen_hashes[-1] == screen_hashes[-2]
            )
            n_before = len(agent.action_lines)
            try:
                response, actions, n_resamples = predict_with_noop_resample(
                    agent, adapted.instruction, obs, prev_line, screen_unchanged
                )
            except Exception as e:
                log.error("agent.predict failed at step %d: %s", step_idx + 1, e)
                stop_reason = f"agent_error: {e}"
                break
            resample_count += n_resamples
            log.info("step %d response: %r", step_idx + 1, (response or "")[:200])
            log.info("step %d actions: %s", step_idx + 1, actions)

            if not actions:
                if response == "<truncated_think>":
                    stop_reason = "truncated_think"
                else:
                    stop_reason = "no_actions_parsed"
                log.warning("step %d: %s", step_idx + 1, stop_reason)
                break

            emitted_lines.append(
                agent.action_lines[-1] if len(agent.action_lines) > n_before else None
            )

            for action in actions:
                obs, reward, done, info = env.step(action, sleep_after_execution)
                missing_streak = missing_streak + 1 if not obs.get("screenshot") else 0
                screen_hashes.append(screen_hash(obs.get("screenshot")))
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
                    stop_reason = "agent_fail" if (info or {}).get("fail") else "agent_terminate"
                    break
            step_idx += 1
            n_steps_taken = step_idx
            if done:
                break
            if should_abort_repetition(emitted_lines, screen_hashes):
                stop_reason = "repetition_abort"
                log.warning("step %d: repetition abort", step_idx)
                break
            if should_abort_stale_screen(screen_hashes):
                stop_reason = "stale_screen_abort"
                log.warning("step %d: stale screen abort", step_idx)
                break

    time.sleep(5)
    evaluate_stub = None
    evaluate_error = ""
    try:
        evaluate_stub = float(env.evaluate())
    except Exception as e:
        log.exception("env.evaluate raised: %s", e)
        evaluate_error = f"evaluate_exception: {e}"

    if reward_wheels_dir:
        try:
            _, wheel_err = cuagym_reward.bootstrap_wheels(env, reward_wheels_dir)
            if wheel_err:
                log.warning("wheel bootstrap stderr tail: %s", wheel_err[-400:])
        except Exception as exc:
            log.warning("wheel bootstrap failed: %s", exc)
    outcome = cuagym_reward.compute_reward(
        env,
        adapted.reward_script,
        output_dir,
        guest_path=reward_guest_path,
        timeout_s=reward_timeout_s,
    )
    log.info("reward=%s error=%s", outcome.reward, outcome.error)

    gif_path = output_dir / "rollout.gif"
    try:
        _write_gif(frames, gif_path)
    except Exception as e:
        log.warning("GIF write failed: %s", e)

    return {
        "reward": outcome.reward,
        "reward_error": outcome.error or "",
        "reward_quarantined": outcome.reward is None,
        "evaluate_stub": evaluate_stub,
        "evaluate_error": evaluate_error,
        "n_steps_taken": n_steps_taken,
        "stop_reason": stop_reason,
        "resample_count": resample_count,
        "missing_streak": missing_streak,
        "gif_path": str(gif_path),
        "traj_path": str(traj_path),
    }


def _select_task_ids(log) -> list[str]:
    if FLAGS.task_id and FLAGS.tasks_file:
        raise ValueError("pass exactly one of --task_id / --tasks_file")
    if FLAGS.task_id:
        return [FLAGS.task_id]
    if not FLAGS.tasks_file:
        raise ValueError("pass exactly one of --task_id / --tasks_file")
    task_ids: list[str] = []
    with Path(FLAGS.tasks_file).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, str):
                task_ids.append(row)
            else:
                task_ids.append(row.get("task_id") or row["id"])
    if FLAGS.task_index >= 0:
        if FLAGS.task_index >= len(task_ids):
            raise ValueError(
                f"task_index {FLAGS.task_index} out of range (tasks_file has {len(task_ids)})"
            )
        task_ids = [task_ids[FLAGS.task_index]]
    log.info("selected %d task(s)", len(task_ids))
    return task_ids


def _load_adapted(task_id: str, log, strict: bool) -> AdaptedTask | None:
    meta = load_metadata(task_id, FLAGS.tasks_parquet)
    try:
        return load_task(
            task_id,
            meta["app_family"],
            tasks_root=FLAGS.tasks_root,
            difficulty=str(meta.get("difficulty", "")),
        )
    except UnsupportedTaskError:
        if strict:
            raise
        log.warning("skipping out-of-scope task %s (app_family=%s)", task_id, meta["app_family"])
        return None


def _print_dry_run(adapted: AdaptedTask) -> None:
    print(f"== task {adapted.task_id} ==")
    print(f"app_family: {adapted.app_family}")
    print(f"app_type:   {adapted.app_type}")
    print(f"difficulty: {adapted.difficulty}")
    print(f"bundle_dir: {adapted.bundle_dir}")
    print(f"instruction: {adapted.instruction}")
    print("setup steps:")
    for i, step in enumerate(adapted.task_config["config"]):
        print(f"  [{i}] {json.dumps(step)}")
    print("cache seeds:")
    for seed in adapted.cache_seeds:
        print(f"  {{cache_dir}}/{seed.cache_relpath} <- {seed.source_path}")
    print(f"evaluator stub: {json.dumps(adapted.task_config['evaluator'])}")
    print(
        "reward plan: upload {} -> {} ; execute ['python3', '{}'] via /setup/execute "
        "(timeout {}s) ; parse last 'REWARD: <float>' from stdout ; None => quarantine".format(
            adapted.reward_script,
            FLAGS.reward_guest_path,
            FLAGS.reward_guest_path,
            FLAGS.reward_timeout_s,
        )
    )
    print()


def _episode_output_dir(base: Path, adapted: AdaptedTask, sample_index: int) -> Path:
    output_dir = base / adapted.app_family / adapted.task_id
    if sample_index:
        output_dir = output_dir / f"sample_{sample_index}"
    return output_dir


def main(_) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("cuagym_rollout")

    strict = bool(FLAGS.task_id)
    task_ids = _select_task_ids(log)

    if FLAGS.dry_run:
        for task_id in task_ids:
            adapted = _load_adapted(task_id, log, strict)
            if adapted is not None:
                _print_dry_run(adapted)
        return

    if not FLAGS.base_output_dir:
        raise ValueError("--base_output_dir is required")
    if not FLAGS.provider_name or not FLAGS.path_to_vm:
        raise ValueError("--provider_name and --path_to_vm are required")
    if not os.environ.get("OPENAI_BASE_URL"):
        raise RuntimeError("OPENAI_BASE_URL is not set; serve the model first")
    os.environ.setdefault("OPENAI_API_KEY", "sk-123")
    if FLAGS.presence_penalty:
        os.environ["OEV3_PRESENCE_PENALTY"] = str(FLAGS.presence_penalty)

    if FLAGS.use_qemu_kvm:
        from osworld_fullbench_kvm import _lease_vm_ports

        _lease_vm_ports()

    from desktop_env.desktop_env import DesktopEnv

    if FLAGS.use_qemu_kvm:
        import qemu_kvm_provider

        qemu_kvm_provider.install()

    agent = Oev3Agent(
        platform="ubuntu",
        model=FLAGS.served_model_name,
        max_tokens=FLAGS.max_tokens,
        top_p=FLAGS.top_p,
        temperature=FLAGS.temperature,
        action_space="pyautogui",
        observation_type="screenshot",
        history_n=FLAGS.history_n,
        coordinate_type="relative",
        api_backend="openai",
        screen_size=(FLAGS.screen_width, FLAGS.screen_height),
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

    base = Path(FLAGS.base_output_dir)
    try:
        for task_id in task_ids:
            adapted = _load_adapted(task_id, log, strict)
            if adapted is None:
                continue
            for sample_index in range(FLAGS.sample_index, FLAGS.sample_index + FLAGS.k):
                output_dir = _episode_output_dir(base, adapted, sample_index)
                result_path = output_dir / "result.json"
                if result_path.exists():
                    log.info("result.json exists, skipping %s", result_path)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                seeded = seed_cache(adapted, FLAGS.cache_dir)
                log.info("seeded %d cache file(s) for %s", len(seeded), task_id)
                t_start = time.time()
                stats = run_episode(
                    env,
                    agent,
                    adapted,
                    output_dir,
                    max_steps=FLAGS.max_steps,
                    sleep_after_execution=FLAGS.sleep_after_execution,
                    reward_guest_path=FLAGS.reward_guest_path,
                    reward_timeout_s=FLAGS.reward_timeout_s,
                    reward_wheels_dir=FLAGS.reward_wheels_dir,
                    log=log,
                )
                elapsed_s = int(time.time() - t_start)
                stop_reason = stats["stop_reason"]
                write_result(
                    result_path,
                    task="cuagym_rollout",
                    scores={
                        "reward": stats["reward"],
                        "n_steps_taken": stats["n_steps_taken"],
                        "stop_reason_code": (
                            1
                            if stop_reason in ("agent_terminate", "agent_fail")
                            else 2
                            if stop_reason == "max_steps"
                            else 0
                        ),
                    },
                    params={
                        "task_id": adapted.task_id,
                        "app_family": adapted.app_family,
                        "app_type": adapted.app_type,
                        "difficulty": adapted.difficulty,
                        "sample_index": sample_index,
                        "instruction": adapted.instruction,
                        "provider_name": FLAGS.provider_name,
                        "max_steps": FLAGS.max_steps,
                        "temperature": FLAGS.temperature,
                        "presence_penalty": FLAGS.presence_penalty,
                        "top_p": FLAGS.top_p,
                        "max_tokens": FLAGS.max_tokens,
                        "history_n": FLAGS.history_n,
                        "screen_width": FLAGS.screen_width,
                        "screen_height": FLAGS.screen_height,
                        "sleep_after_execution": FLAGS.sleep_after_execution,
                        "stop_reason": stop_reason,
                        "resample_count": stats["resample_count"],
                        "missing_streak": stats["missing_streak"],
                        "reward_error": stats["reward_error"],
                        "reward_quarantined": stats["reward_quarantined"],
                        "reward_guest_path": FLAGS.reward_guest_path,
                        "reward_timeout_s": FLAGS.reward_timeout_s,
                        "evaluate_stub": stats["evaluate_stub"],
                        "evaluate_error": stats["evaluate_error"],
                    },
                    inputs={
                        "served_model_name": FLAGS.served_model_name,
                        "tasks_root": FLAGS.tasks_root,
                        "tasks_parquet": FLAGS.tasks_parquet,
                        "tasks_file": FLAGS.tasks_file,
                        "path_to_vm": FLAGS.path_to_vm,
                        "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
                    },
                    n_samples=1,
                    elapsed_s=elapsed_s,
                    extra={"gif_path": stats["gif_path"], "traj_path": stats["traj_path"]},
                )
                log.info(
                    "done in %ds task_id=%s sample_index=%d reward=%s stop_reason=%s",
                    elapsed_s,
                    adapted.task_id,
                    sample_index,
                    stats["reward"],
                    stop_reason,
                )
    finally:
        try:
            env.close()
        except Exception:
            log.warning("env.close failed:\n%s", traceback.format_exc())


if __name__ == "__main__":
    app.run(main)
