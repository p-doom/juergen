"""Real-OSWorld full-behavior teacher rollout collection (re-scope 2026-07-26).

Distill the teacher's FULL agentic behavior (planning, screen-reading, reasoning
prose, action selection, termination) on the REAL OSWorld task distribution —
re-expressed in our relative action format. Same pipeline shape as the custom-task
collector (absolute rollout -> diff-of-absolute -> relative, KEEP PROSE), but:

  * task distribution = real OSWorld task JSONs (evaluation_examples/examples/),
  * per-task initial-state SETUP via OSWorld SetupController against freeroll's OWN
    qemu VM (reuses osworld_grounding_runner._launch_vm[+chromium hostfwd] +
    run_task_setup; NO DesktopEnv/apptainer fork needed),
  * a setup-success prefilter (roll out only on tasks that set up cleanly),
  * optional THINKING preamble to elicit per-step reasoning from an Instruct teacher
    (--thinking), or serve a Thinking model under the plain prompt.

Reuses the validated abs rollout loop + coord scaling (coord_grid=1000) from
collect_absolute_rollouts.
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_SCR = str(Path(__file__).resolve().parent)
if _SCR not in sys.path:
    sys.path.insert(0, _SCR)
_JUERGEN_EVAL = "/fast/home/franz.srambical/juergen/eval"
if _JUERGEN_EVAL not in sys.path:
    sys.path.insert(0, _JUERGEN_EVAL)

from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402
from osworld_vm_client import OSWorldClient  # noqa: E402
from osworld_runtime import _EVAL_DIR, _wait_for  # noqa: E402
# grounding runner = the proven SetupController-against-freeroll-qemu path
import osworld_grounding_runner as gr  # noqa: E402
# my validated abs rollout loop + coord scaling
from collect_absolute_rollouts import _run_rollout  # noqa: E402
# CANONICAL OSWorld prompt (verbatim from qwen3vl_agent.py) — keeps the teacher
# exactly in-distribution (the prompt behind published 33.9%). NOT a hand-rolled one.
from canonical_prompt import build_canonical_system_prompt, THINK_PREAMBLE_FALLBACK  # noqa: E402

_LOGGER = logging.getLogger("collect_osworld")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks_file", required=True,
                   help="one 'app/task_id' per line (real OSWorld tasks)")
    p.add_argument("--run_prefix", default="")
    p.add_argument("--samples_per_task", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=25)
    p.add_argument("--system_prompt_id", default="canonical",
                   help="'canonical' = verbatim OSWorld qwen3vl_agent prompt (default, "
                        "keeps the teacher in-distribution). Or any id in osworld_system_prompts.")
    p.add_argument("--thinking", action="store_true",
                   help="FALLBACK ONLY: append a minimal <think> preamble to the canonical "
                        "prompt. Primary path = serve the 8B-THINKING model under the plain "
                        "canonical prompt (reasons natively, no preamble).")
    p.add_argument("--coord_type", choices=["relative", "absolute"], default="relative",
                   help="'relative' = canonical 0-999 grid (robust, validated). 'absolute' = "
                        "processed-pixel canonical variant (fast-follow; needs frame-resize).")
    p.add_argument("--coord_grid", type=int, default=1000)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n_history_frames", type=int, default=6)
    p.add_argument("--persist_instruction", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--settle_s", type=float, default=0.6)
    p.add_argument("--settle_stable_timeout_s", type=float, default=2.5)
    p.add_argument("--settle_poll_s", type=float, default=0.1)
    p.add_argument("--sglang_port", type=int, default=30000)
    p.add_argument("--sglang_api_key", default="osworld")
    p.add_argument("--sglang_tp", type=int, default=1, help="sglang tensor-parallel size (2 for 32B)")
    p.add_argument("--evaluate", action="store_true",
                   help="run OSWorld's DETERMINISTIC evaluate() (getters/checkers vs the freeroll VM) "
                        "at episode end -> gold task-success label for filtering. Default off so the "
                        "in-flight sweep is unaffected; the winning-config scale-up round opts in.")
    p.add_argument("--mem_fraction_static", type=float, default=0.85)
    p.add_argument("--qcow2", default=gr._DEFAULT_QCOW2 if hasattr(gr, "_DEFAULT_QCOW2") else None)
    p.add_argument("--qemu_bin", default=None)
    args = p.parse_args()

    # defaults from the runtime module (same VM image / wrapped qemu the grounding uses)
    from osworld_runtime import _DEFAULT_QCOW2, _DEFAULT_QEMU_BIN
    qcow2 = args.qcow2 or _DEFAULT_QCOW2
    qemu_bin = args.qemu_bin or _DEFAULT_QEMU_BIN

    if args.system_prompt_id == "canonical":
        system_prompt = build_canonical_system_prompt()
    elif args.system_prompt_id in SYSTEM_PROMPTS:
        system_prompt = SYSTEM_PROMPTS[args.system_prompt_id]
    else:
        print(f"Unknown system_prompt_id {args.system_prompt_id!r} (use 'canonical' or an osworld id)", file=sys.stderr)
        return 1
    if args.thinking:  # fallback only — append the minimal <think> preamble
        system_prompt = THINK_PREAMBLE_FALLBACK + system_prompt

    # coordinate handling. relative = canonical 0-999 grid, per-axis grid = coord_grid.
    grid_x = grid_y = args.coord_grid
    if args.coord_type == "absolute":
        print("coord_type=absolute is a fast-follow (needs processed-dim prompt + frame-resize + "
              "per-axis orig/processed scaling + absolute->normalized conversion). Not yet wired; "
              "run coord_type=relative for now.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(output_dir / "collect_osworld.log"))

    tasks = []
    for line in Path(args.tasks_file).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        app, task_id = s.split("/", 1)
        tasks.append((app, task_id))

    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
    base_vm, base_vnc, base_chr, base_sglang = 5000, 5900, 9300, 30000
    sglang_port = (base_sglang + job_mod) if args.sglang_port == 30000 else args.sglang_port

    _procs: list[subprocess.Popen] = []

    def _cleanup():
        for pr in _procs:
            if pr.poll() is None:
                pr.terminate()
                try:
                    pr.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pr.kill()
    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))

    _LOGGER.info("starting sglang model=%s thinking=%s prompt=%s", args.model_path, args.thinking, args.system_prompt_id)
    sglang_proc = subprocess.Popen(
        ["uv", "run", "--project", str(_EVAL_DIR), "python", "-m", "sglang.launch_server",
         "--model-path", args.model_path, "--host", "0.0.0.0", "--port", str(sglang_port),
         "--api-key", args.sglang_api_key, "--mem-fraction-static", str(args.mem_fraction_static),
         "--tp-size", str(args.sglang_tp), "--chunked-prefill-size", "2048"],
        cwd=str(_EVAL_DIR), stdout=open(output_dir / "sglang.log", "w"), stderr=subprocess.STDOUT)
    _procs.append(sglang_proc)

    runs = []
    first = True
    idx = -1
    for (app, task_id) in tasks:
        for s in range(args.samples_per_task):
            idx += 1
            slug = f"{args.run_prefix}task_{idx:03d}_{app}_{task_id[:12]}" + (f"_s{s}" if args.samples_per_task > 1 else "")
            run_dir = output_dir / slug
            run_dir.mkdir(parents=True, exist_ok=True)
            offset = idx % 40
            vm_port = base_vm + job_mod + offset
            vnc_port = base_vnc + job_mod + offset
            chromium_port = base_chr + job_mod + offset
            _LOGGER.info("=== %s : %s/%s ===", slug, app, task_id)

            vm_proc = gr._launch_vm(qemu_bin=qemu_bin, qcow2=qcow2, vm_port=vm_port,
                                    vnc_port=vnc_port, chromium_port=chromium_port,
                                    log_path=run_dir / "qemu.log")
            _procs.append(vm_proc)
            rec = {"slug": slug, "app": app, "task_id": task_id, "sample": s}
            try:
                _wait_for(f"http://localhost:{vm_port}/screenshot", proc=vm_proc,
                          poll_s=5, max_polls=72, label="VM")
                if first:
                    _wait_for(f"http://localhost:{sglang_port}/health_generate",
                              headers={"Authorization": f"Bearer {args.sglang_api_key}"},
                              proc=sglang_proc, poll_s=10, max_polls=180, label="sglang")
                    first = False
                client = OSWorldClient(f"http://localhost:{vm_port}")
                client.wait_ready(timeout_s=300)
                sw, sh = client.screen_size()
                task = gr.load_osworld_task(app, task_id)
                instruction = task.get("instruction")
                # ---- setup-success prefilter ----
                t0 = time.time()
                setup_ok = True
                setup_err = None
                try:
                    gr.run_task_setup(task=task, vm_port=vm_port, chromium_port=chromium_port,
                                      vlc_port=base_vnc + 100 + job_mod + offset,
                                      cache_dir=run_dir / "setup_cache", screen_w=sw, screen_h=sh)
                except Exception as e:
                    setup_ok = False
                    setup_err = str(e)[:300]
                    _LOGGER.warning("SETUP FAILED %s: %s", slug, setup_err)
                rec["setup_ok"] = setup_ok
                rec["setup_err"] = setup_err
                rec["setup_s"] = round(time.time() - t0, 1)
                if not setup_ok:
                    (run_dir / "result.json").write_text(json.dumps(
                        {**rec, "instruction": instruction, "stop_reason": "setup_failed"}, indent=2))
                    runs.append(rec)
                    continue
                # ---- rollout (abs teacher + thinking) ----
                result = _run_rollout(
                    sglang_url=f"http://localhost:{sglang_port}/v1", api_key=args.sglang_api_key,
                    model=args.model_path, osworld_url=f"http://localhost:{vm_port}", output_dir=run_dir,
                    max_steps=args.max_steps, instruction=instruction, system_prompt=system_prompt,
                    n_history_frames=args.n_history_frames, persist_instruction=args.persist_instruction,
                    max_tokens=args.max_tokens, temperature=args.temperature, coord_grid=args.coord_grid,
                    coord_grid_x=grid_x, coord_grid_y=grid_y,
                    settle_s=args.settle_s, settle_stable_timeout_s=args.settle_stable_timeout_s,
                    settle_poll_s=args.settle_poll_s)
                result.update({**rec, "instruction": instruction, "app": app, "task_id": task_id,
                               "thinking": args.thinking, "system_prompt_id": args.system_prompt_id})
                # DETERMINISTIC task-success (VM still up, in final state) — gold filter signal.
                if args.evaluate:
                    from osworld_evaluate import evaluate_task  # lazy: OSWorld import
                    ah = ["FAIL"] if str(result.get("stop_reason", "")).startswith("terminate_") else []
                    score, everr = evaluate_task(
                        task, vm_port=vm_port, cache_dir=run_dir / "eval_cache",
                        chromium_port=chromium_port, vlc_port=base_vnc + 100 + job_mod + offset,
                        screen_w=sw, screen_h=sh, action_history=ah)
                    result["task_success"] = score
                    result["eval_error"] = everr
                    rec["task_success"] = score
                    _LOGGER.info("evaluate %s -> success=%s err=%s", slug, score, everr)
                (run_dir / "result.json").write_text(json.dumps(result, indent=2))
                rec.update({"stop_reason": result["stop_reason"], "n_steps": result["n_steps"],
                            "parse_errors": result["parse_errors"]})
                runs.append(rec)
            except Exception as e:
                _LOGGER.error("rollout %s failed: %s", slug, e)
                rec["error"] = str(e)[:300]
                runs.append(rec)
            finally:
                if vm_proc.poll() is None:
                    vm_proc.terminate()
                    try:
                        vm_proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        vm_proc.kill()
                if vm_proc in _procs:
                    _procs.remove(vm_proc)

    n_setup_ok = sum(1 for r in runs if r.get("setup_ok"))
    (output_dir / "index.json").write_text(json.dumps(
        {"schema_version": 1, "model": args.model_path, "thinking": args.thinking,
         "system_prompt_id": args.system_prompt_id, "coord_grid": args.coord_grid,
         "n_runs": len(runs), "n_setup_ok": n_setup_ok, "runs": runs}, indent=2))
    _LOGGER.info("done. %d runs, %d setup_ok -> %s", len(runs), n_setup_ok, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
