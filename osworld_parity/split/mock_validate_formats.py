"""GPU-free validation of the move_rel + diffabs PARSE+DISPATCH path.

A real format checkpoint is hours out, so validate the harness's action half now:
boot one apptainer VM, feed canned move_rel + diffabs model outputs through the
EXACT parser + VM dispatch the format eval uses, and confirm the VM cursor moves
as expected (non-garbage). Uses the same functions format_eval_shard drives via
freeroll._run_rollout.
"""
import logging, os, sys
logging.basicConfig(level=logging.WARNING)
os.environ.setdefault("OSWORLD_VM_LOGDIR", "/tmp/osworld_mockfmt")
sys.path.insert(0, "/fast/home/franz.srambical/OSWorld")
sys.path.insert(0, "/fast/home/franz.srambical/juergen/eval")
from desktop_env.desktop_env import DesktopEnv
from osworld_vm_client import OSWorldClient
from action_parser import parse_computer_use_tool_calls, parse_action_tolerant

QCOW2 = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2"
env = DesktopEnv(provider_name="apptainer", path_to_vm=QCOW2, action_space="pyautogui",
                 screen_size=(1920, 1080), headless=True, os_type="Ubuntu",
                 require_a11y_tree=False, cache_dir="/tmp/osworld_mockfmt/cache")
ok = True
try:
    client = OSWorldClient(f"http://localhost:{env.server_port}")
    client.wait_ready()
    sw, sh = client.screen_size()
    print(f"VM screen {sw}x{sh}")

    # ---- move_rel: <tool_call> move_rel [dx,dy] on 0-999 grid, then coordinate-less click ----
    client.execute("pyautogui.moveTo(400, 400)")
    c0 = client.cursor_position()
    mr_text = ('<tool_call>\n{"name": "computer_use", "arguments": {"action": "move_rel", "coordinate": [100, 50]}}\n</tool_call>\n'
               '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click"}}\n</tool_call>')
    calls = parse_computer_use_tool_calls(mr_text)
    print(f"move_rel parsed {len(calls)} calls: {[c.arguments for c in calls]}")
    for c in calls:
        raw = c.arguments.get("coordinate")
        if isinstance(raw, (list, tuple)) and len(raw) == 2:  # freeroll rel_coord_grid=1000 scaling
            c.arguments["coordinate"] = [round(float(raw[0]) * sw / 1000), round(float(raw[1]) * sh / 1000)]
        client.dispatch_computer_use(c.arguments, relative=True)
    c1 = client.cursor_position()
    exp = (c0[0] + round(100 * sw / 1000), c0[1] + round(50 * sh / 1000))
    dmr = abs(c1[0] - exp[0]) + abs(c1[1] - exp[1])
    print(f"move_rel cursor {c0} -> {c1}  expected ~{exp}  |err|={dmr}  {'OK' if dmr <= 3 else 'MISMATCH'}")
    ok &= (len(calls) == 2 and dmr <= 3)

    # ---- diffabs: bare 'dx dy scroll' (moveTo cursor+delta) + a click-event label ----
    client.execute("pyautogui.moveTo(600, 500)")
    c2 = client.cursor_position()
    act = parse_action_tolerant("150 -80 0")
    print(f"diffabs parsed: dx={act.dx} dy={act.dy} scroll={act.scroll} no_op={act.no_op} events={act.events}")
    client.dispatch_action(act)
    c3 = client.cursor_position()
    exp2 = (c2[0] + 150, c2[1] - 80)
    dda = abs(c3[0] - exp2[0]) + abs(c3[1] - exp2[1])
    print(f"diffabs cursor {c2} -> {c3}  expected ~{exp2}  |err|={dda}  {'OK' if dda <= 3 else 'MISMATCH'}")
    ok &= (act.dx == 150 and act.dy == -80 and dda <= 3)

    act2 = parse_action_tolerant("0 0 0 ; +LMB -LMB")
    evs = [(e.kind, e.what, e.mouse_button) for e in act2.events]
    print(f"diffabs click-event parse: events={evs}")
    ok &= (len(act2.events) == 2 and act2.events[0].kind == "press")

    print("\nMOCK_FORMAT_VALIDATE:", "PASS" if ok else "FAIL")
finally:
    try: env.close()
    except Exception: pass
