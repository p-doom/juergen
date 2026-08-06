"""`python -m evals.signoflife` — the gate's dispatcher.

`verifiers`' own `vf eval` CLI cannot stand in, because three things this gate
needs are not expressible as CLI flags —

  * **the arm.** An arm is a whole `DesktopHarnessConfig` (codec, history policy,
    image budget, settle profile, scripted/negative, artifact policy) that lives in
    `cells.py` precisely so an arm cannot be redefined at a command line. `--arm`
    names one; it does not rebuild one.
  * **the VM.** `DesktopPoolConfig.session_kwargs` has to be filled with an image,
    a qemu binary and a pool target that the harness can actually drive
    (`evals/vm.py`).
  * **the aggregate.** labctl's `eval_result` output wants one `result.json` at a
    fixed marker path, and the thing we need out of a multi-trial run is
    **pass_rate per cell**, not one pass count.

Everything else is verifiers': task loading, the episode, interception, the
client, `traces.jsonl`.

**Trials are separate `run_eval` passes, not `num_rollouts`.** Both give N draws
per cell, but `DesktopHarness._artifact_dir` keys on the task name alone, so N
rollouts of one task overwrite each other's frames,
prompts, GIF and `result.json` and the run keeps only the last. A pass per trial
with its own `artifacts.output_dir` keeps all N. The sglang server and the desktop
pool are process-global and survive across passes, so the extra passes cost
nothing.

**`controls_ok` is emitted for control arms only, and is `null` for a model arm.**
The old runner computed `expected_passed = 0 if negative else len(rows)` for every
mode, so a model arm's `controls_ok` was just "every selected cell passed" wearing
the word *control* — a green light that restated the measurement instead of
checking it. Nothing derived from a model arm can calibrate that arm; cite the
scripted oracle/negative runs.

Exit status: 0 fine, 2 a control arm did not read its calibrated value, 3
infrastructure failure (a result that must not be read as a model number).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    # The flat plugin id `signoflife` is a module at the repo root; verifiers
    # imports it by name, so the root must be importable however we were invoked.
    sys.path.insert(0, str(_REPO_ROOT))

import verifiers.v1 as vf  # noqa: E402
from verifiers.v1.cli.eval.runner import run_eval  # noqa: E402
from verifiers.v1.configs.eval import EvalConfig  # noqa: E402

from evals.signoflife.cells import ARMS, verify_phaseb_provenance  # noqa: E402
from evals.signoflife.suite import load_suite  # noqa: E402
from evals.tasks import RESULT_KEY  # noqa: E402
from signoflife import PLUGIN_ID  # noqa: E402

_LOGGER = logging.getLogger("signoflife")

SERVED_MODEL = "sign-of-life"
"""The alias sglang serves under, so the wire model id does not encode a path."""

API_KEY_VAR = "SIGN_OF_LIFE_API_KEY"
"""`resolve_api_key` reads the key from an env var named by the client config
(`clients/config.py:91-102`); it is never a CLI field, so we name our own."""


@contextlib.contextmanager
def _sglang(
    *,
    python: str,
    model_path: Path,
    api_key: str,
    log_path: Path,
    port: int,
    mem_fraction_static: float,
    ready_timeout_s: float,
) -> Iterator[str]:
    """Serve `model_path` and yield its OpenAI base URL.

    `python` is an explicit interpreter, not `sys.executable`, and that is the
    whole point: the harness needs `verifiers`, sglang needs a 14 GB CUDA stack,
    and the two do not have to be the same venv. Keeping them apart means the
    already-built, already-referenced sglang environment stays byte-unchanged
    instead of being mutated to add a framework it does not use.
    """
    if port == 0:
        # sglang derives grpc_port = port + 10000 and rejects > 65535, and its own
        # warmup probes the *requested* port, so `--port 0` cannot be handed
        # through: pick a real free port in the safe sub-range.
        for _ in range(64):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                candidate = sock.getsockname()[1]
            if candidate + 10000 <= 65535:
                port = candidate
                break
        else:
            raise RuntimeError("no sglang-safe free port (port + 10000 <= 65535)")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--served-model-name",
        SERVED_MODEL,
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        "2048",
    ]
    _LOGGER.info("sglang: %s", " ".join(command))
    handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            env={**os.environ, "SGLANG_DISABLE_CUDNN_CHECK": "1"},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    except BaseException:
        handle.close()
        raise
    try:
        deadline = time.monotonic() + ready_timeout_s
        probe = f"http://127.0.0.1:{port}/health_generate"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = "\n".join(log_path.read_text().splitlines()[-40:])
                raise RuntimeError(
                    f"sglang exited before ready (rc={process.returncode}):\n{tail}"
                )
            try:
                request = urllib.request.Request(
                    probe, headers={"Authorization": f"Bearer {api_key}"}
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    if response.status < 500:
                        break
            except Exception:  # noqa: BLE001 - not up yet is the normal case
                pass
            time.sleep(2.0)
        else:
            raise TimeoutError(f"sglang not ready after {ready_timeout_s}s")
        url = f"http://127.0.0.1:{port}/v1"
        _LOGGER.info("sglang ready at %s", url)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        handle.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_provenance(model_path: Path | None) -> dict[str, Any] | None:
    """Record which checkpoint served the arm — recorded, not enforced.

    `cells.verify_phaseb_provenance` fails closed against the *step-900* export and
    is the right check for that one arm; hard-coding a second, third and fourth
    expected manifest here would make every new arm a code change. What must never
    be lost is *which bytes answered*, so both registration files are hashed and
    embedded, and `--verify-phaseb` opts into the strict check when the arm is the
    one it describes.
    """
    if model_path is None:
        return None
    root = model_path.parent
    record: dict[str, Any] = {"path": str(model_path)}
    config = model_path / "config.json"
    if config.is_file():
        record["config_sha256"] = _sha256_file(config)
    for name in (".meta.json", "export_manifest.json"):
        candidate = root / name
        if not candidate.is_file():
            continue
        record[name.strip(".").replace(".json", "")] = json.loads(candidate.read_text())
        record[f"{name.strip('.').replace('.json', '')}_sha256"] = _sha256_file(candidate)
    return record


def _harness_payload(arm: str, *, artifacts: Path, pool: dict[str, Any]) -> dict[str, Any]:
    """One arm's `DesktopHarnessConfig` as verifiers wants it.

    `id` is overwritten with the plugin id, and it has to be: `HarnessConfig.id` is
    the *plugin* id `harness_class()` imports (`loaders.py:87-88`), while `cells.py`
    uses it as the arm's human name (`sol_native_oracle`, ...). Resolving
    `sol_native_oracle` as a package fails, so the arm name moves into the run
    record and the field goes back to meaning what verifiers means by it.
    """
    payload = ARMS[arm].model_dump()
    payload["id"] = PLUGIN_ID
    payload["artifacts"] = {**payload["artifacts"], "output_dir": str(artifacts)}
    payload["pool"] = {**payload["pool"], **pool}
    return payload


def _eval_config(
    *,
    arm: str,
    task_ids: list[str],
    artifacts: Path,
    traces_dir: Path,
    pool: dict[str, Any],
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> EvalConfig:
    return EvalConfig(
        taskset={"id": PLUGIN_ID, "task_ids": task_ids},
        harness=_harness_payload(arm, artifacts=artifacts, pool=pool),
        model=SERVED_MODEL,
        client={"base_url": base_url, "api_key_var": API_KEY_VAR},
        sampling={"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
        num_rollouts=1,
        # One episode at a time. Concurrency here would put N desktops on the node
        # at once, and the node budget is a slot count, not a promise of memory.
        max_concurrent=1,
        output_dir=traces_dir,
        rich=False,  # a live dashboard in a batch job is a log full of escape codes
        push=False,  # never upload a gate run to the Prime platform
    )


def _harness_error(trace: Any) -> dict[str, Any] | None:
    """The reason an episode published nothing at all.

    `DesktopHarness._run` turns an episode failure into `validity="infra_invalid"`
    with an `infra_error`, but a raise *before* it — a bad pool spec, an unknown
    grammar, an unregistered task kind — never reaches that code and leaves
    `trace.info` empty. The row then said `validity: null, infra_error: null`, so
    the run exited 3 with no reason recorded anywhere a reader would look, and the
    only copy of the message was a log line in a batch job's stdout.

    Reads `trace.errors` / `trace.stop_condition`, which verifiers fills in for
    exactly this case. Diagnostic only: an episode with no result was already
    excluded from every rate (`_aggregate` counts `validity == "valid"`), so
    nothing here can move a pass count.
    """
    errors = getattr(trace, "errors", None) or []
    if not errors:
        return None
    first = errors[0]
    if not isinstance(first, dict):
        first = getattr(first, "model_dump", lambda: {"message": str(first)})()
    return {
        "stage": "harness",
        "type": str(first.get("type") or "HarnessError"),
        "message": str(first.get("message") or first),
        "stop_condition": getattr(trace, "stop_condition", None),
    }


def _episode_row(trace: Any, trial: int) -> dict[str, Any]:
    episode = dict(trace.info.get(RESULT_KEY) or {})
    prompt = dict(trace.info.get("prompt") or {})
    return {
        "trial": trial,
        "cell": trace.task.data.name,
        "kind": trace.task.data.kind,
        "trace_id": trace.id,
        "success": episode.get("success"),
        "validity": episode.get("validity"),
        "outcome": episode.get("outcome"),
        "steps": episode.get("steps"),
        "parse_errors": episode.get("parse_errors"),
        "action_errors": episode.get("action_errors"),
        "executor_errors": episode.get("executor_errors"),
        "control_terminate": episode.get("control_terminate"),
        "terminate_step": episode.get("terminate_step"),
        "control_ok": episode.get("control_ok"),
        "infra_error": episode.get("infra_error") or _harness_error(trace),
        "final_probe": episode.get("final_probe"),
        "sampling": episode.get("sampling"),
        "host": episode.get("host"),
        "prompt_sha256": prompt.get("prompt_sha256"),
        "comparable_to_sealed_baseline": prompt.get("comparable_to_sealed_baseline"),
        "steps_detail": episode.get("steps_detail"),
    }


def _aggregate(
    rows: list[dict[str, Any]], *, cell_ids: list[str], scripted: bool, negative: bool
) -> dict[str, Any]:
    """pass_rate per cell over trials — the reading this gate is supposed to give.

    A single-trial score cannot be read on `desktop_open_chrome`: the suite's own
    `instrument_limits` note says a Chrome that starts but never maps a window
    flips PASS to FAIL and that the cell must be read as a rate over trials. A
    scalar `passed/4` hides exactly that, which is how one race became a published
    arm difference.
    """
    per_cell: dict[str, Any] = {}
    for cell in cell_ids:
        draws = [row for row in rows if row["cell"] == cell]
        valid = [row for row in draws if row["validity"] == "valid"]
        passed = sum(1 for row in valid if row["success"] is True)
        per_cell[cell] = {
            "trials": len(draws),
            "valid_trials": len(valid),
            "passed": passed,
            "pass_rate": (passed / len(valid)) if valid else None,
            "outcomes": [row["outcome"] for row in draws],
        }
    valid_rows = [row for row in rows if row["validity"] == "valid"]
    conformant = (
        None
        if not scripted
        else all(row.get("control_ok") == 1.0 for row in rows) and len(valid_rows) == len(rows)
    )
    return {
        "per_cell": per_cell,
        "episodes": len(rows),
        "valid_episodes": len(valid_rows),
        "expected_per_cell_pass_rate": (0.0 if negative else 1.0) if scripted else None,
        "controls_ok": conformant,
        "controls_ok_note": (
            "null for a model arm on purpose. A model arm has no expected value, so "
            "any 'controls_ok' computed from its own rows only restates the pass "
            "count; calibration comes from the separate scripted oracle/negative runs."
        ),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.signoflife")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--task-index",
        type=int,
        action="append",
        default=None,
        help="zero-based suite cell; repeatable. Omitted = all four (the gate).",
    )
    selection.add_argument("--cell", action="append", default=None, help="cell id; repeatable")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="independent draws per cell. >=3 for any model arm: the gate is "
        "single-trial only by historical accident, and one race-prone cell makes a "
        "single draw uninterpretable.",
    )
    parser.add_argument("--qcow", type=Path, required=True)
    parser.add_argument("--qemu", type=Path, default=None)
    parser.add_argument("--qemu-img", type=Path, default=None)
    parser.add_argument("--vm-smp", type=int, default=None)
    parser.add_argument("--vm-mem", default=None)
    parser.add_argument("--vm-slots", type=int, default=1)
    parser.add_argument("--vm-rollouts-per-session", type=int, default=1)
    parser.add_argument(
        "--scoring-grace-s",
        type=float,
        default=120.0,
        help="how long the desktop stays leased after the episode so a "
        "runtime-declaring reward can probe live guest state. Pure wall clock per "
        "cell, and until now it was pinned in code with no way to name it.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--base-url", default=None, help="serve externally instead")
    parser.add_argument("--sglang-python", default=None)
    parser.add_argument("--sglang-port", type=int, default=0)
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.65)
    parser.add_argument("--sglang-ready-timeout-s", type=float, default=1500.0)
    parser.add_argument("--api-key", default="sign-of-life")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--verify-phaseb",
        action="store_true",
        help="fail closed unless --model-path is the registered step-900 export",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    arm = ARMS[args.arm]
    scripted = arm.scripted.enabled
    negative = arm.scripted.negative
    suite = load_suite()

    selected = list(suite.tasks)
    if args.task_index:
        selected = [suite.tasks[index] for index in args.task_index]
    if args.cell:
        selected = [suite.by_id(cell) for cell in args.cell]
    cell_ids = [task.id for task in selected]

    if not scripted and args.trials < 3:
        _LOGGER.warning(
            "model arm %s with trials=%d: a single draw cannot separate a model "
            "difference from the open_chrome window-mapping race",
            args.arm,
            args.trials,
        )
    if scripted and (args.model_path or args.base_url):
        raise SystemExit(
            f"arm {args.arm} is scripted and never calls a model; --model-path / "
            "--base-url would be recorded in the run and ignored"
        )
    if not scripted and args.model_path is None and args.base_url is None:
        raise SystemExit(f"arm {args.arm} is a model arm: pass --model-path or --base-url")
    if args.verify_phaseb:
        if args.model_path is None:
            raise SystemExit("--verify-phaseb checks a checkpoint: pass --model-path")
        verify_phaseb_provenance(args.model_path)

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    os.environ[API_KEY_VAR] = args.api_key

    # Omit unset knobs rather than passing None. `session_kwargs` is a plain
    # `dict[str, Any]`, and pydantic's `exclude_none` does not recurse into one, so
    # a None inside it reaches `tomli_w.dumps` when verifiers saves the run's
    # config.toml and kills the run before the first episode with
    # "Object of type 'NoneType' is not TOML serializable". `kvm_desktop_pool`
    # treats absent and None identically, so this loses nothing.
    session_kwargs: dict[str, Any] = {
        "image": str(args.qcow),
        "root_dir": str(output / "vm"),
        "accelerator": "kvm",
        "max_rollouts_per_session": args.vm_rollouts_per_session,
        "max_sessions": args.vm_slots,
    }
    for key, value in (
        ("qemu_binary", args.qemu),
        ("qemu_img_binary", args.qemu_img),
        ("smp", args.vm_smp),
        ("memory", args.vm_mem),
    ):
        if value is not None:
            session_kwargs[key] = str(value) if isinstance(value, Path) else value
    pool = {
        "key": f"signoflife-{args.arm}",
        "max_node_slots": args.vm_slots,
        "slot_dir": str(output / "vm_slots"),
        "scoring_grace_s": args.scoring_grace_s,
        "pool_target": "evals.vm:kvm_desktop_pool",
        "session_kwargs": session_kwargs,
    }

    rows: list[dict[str, Any]] = []
    infrastructure_errors: list[dict[str, Any]] = []

    def _run_trials(base_url: str) -> None:
        for trial in range(1, args.trials + 1):
            config = _eval_config(
                arm=args.arm,
                task_ids=cell_ids,
                artifacts=output / f"trial_{trial:02d}",
                traces_dir=output / f"trial_{trial:02d}" / "traces",
                pool=pool,
                base_url=base_url,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            environment = vf.Environment(config)
            traces = asyncio.run(run_eval(environment, config))
            for trace in traces:
                row = _episode_row(trace, trial)
                rows.append(row)
                if row["validity"] != "valid":
                    infrastructure_errors.append(
                        {"trial": trial, "cell": row["cell"], "error": row["infra_error"]}
                    )
            _LOGGER.info(
                "trial %d/%d: %s",
                trial,
                args.trials,
                {row["cell"]: row["success"] for row in rows if row["trial"] == trial},
            )

    if scripted or args.base_url:
        _run_trials(args.base_url or "http://127.0.0.1:1/v1")
    else:
        with _sglang(
            python=args.sglang_python or sys.executable,
            model_path=args.model_path,
            api_key=args.api_key,
            log_path=output / "sglang.log",
            port=args.sglang_port,
            mem_fraction_static=args.sglang_mem_fraction,
            ready_timeout_s=args.sglang_ready_timeout_s,
        ) as base_url:
            _run_trials(base_url)

    aggregate = _aggregate(rows, cell_ids=cell_ids, scripted=scripted, negative=negative)
    result = {
        "schema_version": 3,
        "arm": args.arm,
        "arm_id": arm.id,
        "arm_kind": "scripted_negative" if negative else "scripted_oracle" if scripted else "model",
        "codec": arm.codec,
        "history_policy": arm.history.name,
        "trials": args.trials,
        "suite_id": suite.suite_id,
        "suite_role": suite.role,
        "final_benchmark": suite.final_benchmark,
        "suite_manifest_sha256": suite.manifest_sha256,
        "selection": {"task_ids": cell_ids, "full_suite_task_count": len(suite.tasks)},
        "status": "complete" if not infrastructure_errors else "infrastructure_failure",
        "aggregate": aggregate,
        "model": _model_provenance(args.model_path),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "vm": {
            "qcow": str(args.qcow),
            "qemu": str(args.qemu) if args.qemu else None,
            "rollouts_per_session": args.vm_rollouts_per_session,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "labctl_run_id": os.environ.get("LABCTL_RUN_ID"),
        },
        "indicators_note": (
            "A/B/C/D do not exist in this schema. The eov3 indicator set is 16 flat "
            "named keys and is not computed here; what this file carries per episode "
            "is parse_errors / action_errors / executor_errors. The over-submission "
            "indicator is structurally zero on this suite regardless -- see "
            "evals/signoflife/suite.py, which is the one place that classification "
            "lives and the only module allowed to name it."
        ),
        "baseline_note": (
            "Re-baseline, not a reproduction. codec.describe() is not byte-identical "
            "to the sealed prompts, so every episode carries "
            "comparable_to_sealed_baseline=false and a difference from a sealed "
            "number is not a regression."
        ),
        "infrastructure_errors": infrastructure_errors,
        "episodes": rows,
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps({"arm": args.arm, **{k: v["pass_rate"] for k, v in aggregate["per_cell"].items()}}, sort_keys=True), flush=True)

    if infrastructure_errors:
        return 3
    if scripted and aggregate["controls_ok"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
