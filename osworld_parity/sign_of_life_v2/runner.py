from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

from ..proper_vm_capability_ladder.rung1.transport import HttpVmTransport
from ..proper_vm_capability_ladder.rung1.vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    KvmFixtureSession,
    sha256_file,
)
from .actions import execute_native_absolute
from .guest import capture_screenshot, probe_task, setup_task
from .oracle import as_json, evaluate
from .suite import DevelopmentTask, load_suite


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))


OFFSHELF_MODEL = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface/hub/"
    "models--Qwen--Qwen3-VL-4B-Instruct/snapshots/"
    "ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
SERVED_MODEL = "qwen3-vl-4b-native-absolute"
SYSTEM_PROMPT_ID = "computer_use_v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


@contextmanager
def _hide_gpu_from_vm_parent() -> Iterator[None]:
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _state_check(task: DevelopmentTask, transport: HttpVmTransport) -> dict[str, Any]:
    state = probe_task(transport, task)
    return {"state": state, "postcondition": as_json(evaluate(task, state))}


def _scripted_actions(task: DevelopmentTask, setup: dict[str, Any], *, negative: bool) -> list[dict[str, Any]]:
    if task.kind == "terminal_command":
        return [
            {"action": "type", "text": "pwd" if negative else task.expected["command"]},
            {"action": "key", "keys": ["ENTER"]},
        ]
    if task.kind == "terminal_exact_text":
        return [
            {"action": "type", "text": "wrong text" if negative else task.expected["text"]},
            {"action": "key", "keys": ["ENTER"]},
        ]
    if task.kind == "open_chrome":
        coordinate = [960, 540] if negative else setup["dock_chrome_coordinate"]
        return [{"action": "left_click", "coordinate": coordinate}]
    if task.kind == "focus_terminal_and_type":
        rows: list[dict[str, Any]] = []
        if not negative:
            rows.append({"action": "left_click", "coordinate": setup["terminal_click_coordinate"]})
        rows.extend(
            [
                {"action": "type", "text": task.expected["command"]},
                {"action": "key", "keys": ["ENTER"]},
            ]
        )
        return rows
    raise ValueError(task.kind)


def _run_scripted_task(
    task: DevelopmentTask,
    transport: HttpVmTransport,
    task_dir: Path,
    *,
    negative: bool,
) -> dict[str, Any]:
    setup = setup_task(transport, task)
    initial = _state_check(task, transport)
    if initial["postcondition"]["oracle_status"] != "ok" or initial["postcondition"]["success"]:
        raise RuntimeError("task reset/setup did not begin in a valid unsolved state")
    before = capture_screenshot(transport, task_dir / "before.png")
    steps: list[dict[str, Any]] = []
    stop_reason = "script_exhausted"
    for index, arguments in enumerate(_scripted_actions(task, setup, negative=negative), start=1):
        receipt = execute_native_absolute(transport, arguments)
        time.sleep(2.0 if task.kind == "open_chrome" else 0.75)
        screenshot = capture_screenshot(transport, task_dir / f"step_{index:02d}.png")
        check = _state_check(task, transport)
        steps.append(
            {
                "step": index,
                "requested_native_absolute_action": arguments,
                "execution": receipt,
                "screenshot_after": screenshot,
                **check,
            }
        )
        if check["postcondition"]["success"]:
            stop_reason = "postcondition_reached"
            break
    final = _state_check(task, transport)
    after = capture_screenshot(transport, task_dir / "after.png")
    success = bool(final["postcondition"]["success"])
    return {
        "task_id": task.id,
        "kind": task.kind,
        "instruction": task.instruction,
        "mode": "negative_control" if negative else "scripted_native_absolute_oracle",
        "success": success,
        "stop_reason": stop_reason,
        "model_termination": None,
        "steps": steps,
        "n_steps": len(steps),
        "parse_errors": [],
        "action_errors": [],
        "executor_errors": [],
        "setup": setup,
        "initial_postcondition": initial["postcondition"],
        "final_postcondition": final["postcondition"],
        "final_state": final["state"],
        "screenshots": {"before": before, "after": after},
    }


def _model_frame(transport: HttpVmTransport, path: Path) -> Any:
    from PIL import Image

    capture_screenshot(transport, path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _run_model_task(
    task: DevelopmentTask,
    transport: HttpVmTransport,
    task_dir: Path,
    *,
    model_url: str,
    api_key: str,
) -> dict[str, Any]:
    from action_parser import parse_computer_use_tool_calls
    from osworld_runtime import _call_model, append_turn
    from osworld_system_prompts import SYSTEM_PROMPTS

    setup = setup_task(transport, task)
    initial = _state_check(task, transport)
    if initial["postcondition"]["oracle_status"] != "ok" or initial["postcondition"]["success"]:
        raise RuntimeError("task reset/setup did not begin in a valid unsolved state")
    frame = _model_frame(transport, task_dir / "before.png")
    recent_frames = [frame]
    recent_actions: list[str] = []
    steps: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    action_errors: list[dict[str, Any]] = []
    executor_errors: list[dict[str, Any]] = []
    model_termination: dict[str, Any] | None = None
    stop_reason = "max_steps"
    for index in range(1, task.max_steps + 1):
        raw = _call_model(
            sglang_url=model_url,
            api_key=api_key,
            model=SERVED_MODEL,
            system_prompt=SYSTEM_PROMPTS[SYSTEM_PROMPT_ID],
            instruction=task.instruction,
            recent_frames=recent_frames,
            recent_actions=recent_actions,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            request_timeout_s=180.0,
        )
        row: dict[str, Any] = {
            "step": index,
            "raw_model_output": raw,
            "parsed_calls": [],
            "executions": [],
            "parse_error": None,
            "action_error": None,
            "executor_error": None,
        }
        try:
            calls = parse_computer_use_tool_calls(raw)
            row["parsed_calls"] = [dict(call.arguments) for call in calls]
        except (TypeError, ValueError) as exc:
            error = {"step": index, "type": type(exc).__name__, "message": str(exc)}
            row["parse_error"] = error
            parse_errors.append(error)
            calls = []
        for call in calls:
            arguments = dict(call.arguments)
            action = str(arguments.get("action", "")).strip().lower()
            if action == "terminate":
                model_termination = {
                    "step": index,
                    "status": str(arguments.get("status", "success")).strip().lower(),
                    "raw": arguments,
                }
                continue
            try:
                row["executions"].append(execute_native_absolute(transport, arguments))
            except (TypeError, ValueError) as exc:
                error = {"step": index, "type": type(exc).__name__, "message": str(exc), "arguments": arguments}
                row["action_error"] = error
                action_errors.append(error)
            except Exception as exc:  # transport failures are kept distinct and fail closed
                error = {"step": index, "type": type(exc).__name__, "message": str(exc), "arguments": arguments}
                row["executor_error"] = error
                executor_errors.append(error)
                break
        time.sleep(2.0 if task.kind == "open_chrome" else 0.75)
        next_frame = _model_frame(transport, task_dir / f"step_{index:02d}.png")
        check = _state_check(task, transport)
        row.update({"screenshot_after": str(task_dir / f"step_{index:02d}.png"), **check})
        steps.append(row)
        append_turn(recent_frames, recent_actions, next_frame, raw, n_history_frames=8)
        if check["postcondition"]["success"]:
            stop_reason = "postcondition_reached"
            break
        if executor_errors:
            stop_reason = "executor_error"
            break
        if model_termination is not None:
            stop_reason = f"model_terminate_{model_termination['status']}_without_postcondition"
            break
    final = _state_check(task, transport)
    after = capture_screenshot(transport, task_dir / "after.png")
    return {
        "task_id": task.id,
        "kind": task.kind,
        "instruction": task.instruction,
        "mode": "offshelf_native_absolute",
        "success": bool(final["postcondition"]["success"]),
        "stop_reason": stop_reason,
        "model_termination": model_termination,
        "steps": steps,
        "n_steps": len(steps),
        "parse_errors": parse_errors,
        "action_errors": action_errors,
        "executor_errors": executor_errors,
        "setup": setup,
        "initial_postcondition": initial["postcondition"],
        "final_postcondition": final["postcondition"],
        "final_state": final["state"],
        "screenshots": {
            "before": {"path": str(task_dir / "before.png")},
            "after": after,
        },
    }


def run_suite(
    *,
    mode: str,
    output_dir: Path,
    model_path: Path,
    qcow: Path,
    qemu: Path,
    provider: Path,
    model_url: str | None = None,
    api_key: str = "sign-of-life",
) -> dict[str, Any]:
    suite = load_suite()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    infrastructure_errors: list[dict[str, Any]] = []
    with _hide_gpu_from_vm_parent():
        with KvmFixtureSession(
            qcow=qcow,
            qemu=qemu,
            provider_path=provider,
            vm_log_dir=output_dir / "vm_logs",
        ) as session:
            for task in suite.tasks:
                task_dir = output_dir / task.id
                task_dir.mkdir(parents=True, exist_ok=True)
                try:
                    transport = session.reset_to_ready()
                    if mode == "model":
                        if model_url is None:
                            raise RuntimeError("model URL is missing")
                        row = _run_model_task(task, transport, task_dir, model_url=model_url, api_key=api_key)
                    else:
                        row = _run_scripted_task(task, transport, task_dir, negative=mode == "negative")
                except Exception as exc:
                    error = {
                        "task_id": task.id,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    infrastructure_errors.append(error)
                    row = {
                        "task_id": task.id,
                        "kind": task.kind,
                        "instruction": task.instruction,
                        "mode": mode,
                        "success": False,
                        "stop_reason": "infrastructure_error",
                        "infrastructure_error": error,
                        "steps": [],
                        "n_steps": 0,
                        "parse_errors": [],
                        "action_errors": [],
                        "executor_errors": [],
                    }
                rows.append(row)
                _atomic_json(task_dir / "result.json", row)
    passed = sum(bool(row["success"]) for row in rows)
    expected_passed = 0 if mode == "negative" else len(rows)
    controls_ok = passed == expected_passed and not infrastructure_errors
    return {
        "schema_version": 2,
        "suite_id": suite.suite_id,
        "suite_role": suite.role,
        "final_benchmark": suite.final_benchmark,
        "suite_manifest_sha256": suite.manifest_sha256,
        "mode": mode,
        "status": "complete" if not infrastructure_errors else "infrastructure_failure",
        "controls_ok": controls_ok,
        "model": (
            {
                "path": str(model_path),
                "served_model": SERVED_MODEL,
                "config_sha256": sha256_file(model_path / "config.json"),
                "system_prompt_id": SYSTEM_PROMPT_ID,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 256,
                "action_format": "native_absolute_computer_use_v1",
            }
            if mode == "model"
            else None
        ),
        "vm": {
            "qcow": str(qcow),
            "qemu": str(qemu),
            "provider": str(provider),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "labctl_run_id": os.environ.get("LABCTL_RUN_ID"),
        },
        "aggregate": {
            "task_count": len(rows),
            "passed_count": passed,
            "success_rate": passed / len(rows),
            "all_pass": passed == len(rows),
        },
        "infrastructure_errors": infrastructure_errors,
        "tasks": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crowd-Cast sign-of-life v2 VM gate")
    parser.add_argument("--mode", choices=("oracle", "negative", "model"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=OFFSHELF_MODEL)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--api-key", default="sign-of-life")
    parser.add_argument("--sglang-port", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode == "model" and not args.model_path.joinpath("config.json").is_file():
        raise SystemExit(f"model path is incomplete: {args.model_path}")
    server_context = (
        _model_server(args)
        if args.mode == "model"
        else nullcontext(None)
    )
    with server_context as model_url:
        result = run_suite(
            mode=args.mode,
            output_dir=args.output,
            model_path=args.model_path,
            qcow=args.qcow,
            qemu=args.qemu,
            provider=args.provider,
            model_url=model_url,
            api_key=args.api_key,
        )
    _atomic_json(args.output / "result.json", result)
    print(json.dumps(result["aggregate"], sort_keys=True), flush=True)
    if args.mode in {"oracle", "negative"} and not result["controls_ok"]:
        return 2
    return 3 if result["infrastructure_errors"] else 0


def _model_server(args: argparse.Namespace) -> Any:
    from sglang_runner import sglang_server

    return sglang_server(
        model_path=str(args.model_path),
        port=args.sglang_port,
        api_key=args.api_key,
        log_path=args.output / "sglang.log",
        mem_fraction_static=0.65,
        chunked_prefill_size=2048,
        ready_timeout_s=1500,
        served_model_name=SERVED_MODEL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
