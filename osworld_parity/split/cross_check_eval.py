"""Metric-correctness cross-check: DesktopEnv.evaluate() (path A, the validated
metric) vs the standalone osworld_evaluate.evaluate_task() port (path B) scored
on the SAME VM state — isolating the SCORER from agent behavior.

For each task: boot a fresh apptainer VM, run the task's setup via
DesktopEnv.reset(), then score the identical post-setup state with both
evaluators (neutral action_history so no FAIL-injection). A faithful port MUST
return the same value as DesktopEnv on identical state.
No GPU / no model needed.
"""
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
os.environ.setdefault("OSWORLD_VM_LOGDIR", "/tmp/osworld_xcheck")
OSWORLD_ROOT = "/fast/home/franz.srambical/OSWorld"
sys.path.insert(0, OSWORLD_ROOT)
sys.path.insert(0, "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/scripts")
QCOW2 = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2"

from desktop_env.desktop_env import DesktopEnv  # noqa: E402
from osworld_evaluate import evaluate_task  # noqa: E402

tasks_file = sys.argv[1] if len(sys.argv) > 1 else \
    "/fast/home/franz.srambical/osworld_parity_split/crosscheck_tasks.txt"
tasks = [l.strip().split("/", 1) for l in Path(tasks_file).read_text().splitlines() if l.strip()]

rows = []
for app, tid in tasks:
    tp = Path(OSWORLD_ROOT) / "evaluation_examples" / "examples" / app / f"{tid}.json"
    task = json.loads(tp.read_text())
    a = b = None; berr = None; note = ""
    env = None
    try:
        env = DesktopEnv(provider_name="apptainer", path_to_vm=QCOW2, action_space="pyautogui",
                         screen_size=(1920, 1080), headless=True, os_type="Ubuntu",
                         require_a11y_tree=False, cache_dir=f"/tmp/osworld_xcheck/{tid}/cache")
        env.reset(task_config=task)  # per-task setup, identical state for both scorers
        # Path B (port) FIRST, neutral action_history — same VM, before A mutates via its own postconfig
        b, berr = evaluate_task(task, vm_port=env.server_port,
                                cache_dir=f"/tmp/osworld_xcheck/{tid}/bcache",
                                chromium_port=env.chromium_port, vlc_port=env.vlc_port,
                                screen_w=1920, screen_h=1080, action_history=[])
        # Path A (DesktopEnv canonical scorer)
        try:
            a = float(env.evaluate())
        except Exception as e:
            a = None; note = f"A_raised={type(e).__name__}:{str(e)[:80]}"
    except Exception as e:
        note = f"boot/setup_err={type(e).__name__}:{str(e)[:120]}"
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
    agree = (a is not None and b is not None and abs(a - b) < 1e-6)
    rows.append((app, tid[:12], a, b, berr, agree, note))
    print(f"{app:<18} {tid[:12]}  A={a}  B={b}  agree={agree}  berr={berr or ''}  {note}", flush=True)

print("\n=== CROSS-CHECK SUMMARY ===")
n = len(rows)
both = [(a, b) for _, _, a, b, _, _, _ in rows if a is not None and b is not None]
agreed = sum(1 for *_, ag, _ in rows if ag)
print(f"tasks={n}  both-scored={len(both)}  scorer-agreements={agreed}/{len(both)}")
disagreements = [r for r in rows if r[2] is not None and r[3] is not None and not r[5]]
if disagreements:
    print("DISAGREEMENTS:")
    for r in disagreements:
        print("  ", r)
else:
    print("ALL scored tasks AGREE (port == DesktopEnv on identical state)")
