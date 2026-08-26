"""Measure whether the oev3 eval dispatch moves the guest cursor by the requested delta.

Training labels for ``move(dx,dy)`` mean "cursor displacement in screen pixels"
(``target_px - cursor_before_px``, teleport semantics, recorded from CUA-Gym
rollouts whose harness executed an absolute ``pyautogui.moveTo`` / click).

Two dispatch styles are in the codebase:

``agent``   -- what ``oev3_agent.compile_primitives`` emits today:
              ``pyautogui.moveRel(dx_px, dy_px)`` with ``dx_px = dx/1000*W``,
              wrapped in OSWorld's ``PYAUTOGUI_PKGS_PREFIX`` and shipped to the
              in-VM ``/execute`` endpoint exactly as ``PythonController`` does.

``tracked`` -- the team's earlier ``osworld_vm_client.dispatch_ordered_action``
              semantics: read ``/cursor_position``, compute the absolute target
              ``clip(cursor + delta)``, then ``pyautogui.moveTo(tx, ty)``.

If the guest applies pointer acceleration to relative motion, or ``moveRel``
takes a different X11 path than ``moveTo``, the achieved displacement differs
from the requested one and every model action is silently distorted. This probe
boots one OSWorld KVM VM and measures both styles over the same delta set,
plus click-landing and screen-edge clamping.

Usage (inside a CPU SLURM allocation -- never on the login node):

    python eval/probe_move_dispatch.py --out /path/to/probe_dir
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from action_parser import parse_ordered_action
from oev3_agent import compile_primitives

PYAUTOGUI_PKGS_PREFIX = (
    "import pyautogui; import time; import platform; "
    "pyautogui.FAILSAFE = False; "
    "_osworld_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:\"<>?'; "
    "_osworld_linux_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:\">?'; "
    "pyautogui.isShiftCharacter = lambda character: character.isupper() or "
    "character in (_osworld_linux_shift_chars if platform.system() == 'Linux' else _osworld_shift_chars); "
    "{command}"
)

CLIENT_PREFIX = (
    "import pyautogui; import time; pyautogui.FAILSAFE = False; pyautogui.PAUSE = 0; "
)

XLIB_POS_CODE = (
    "from Xlib import display as _d\n"
    "_p = _d.Display().screen().root.query_pointer()\n"
    "print(_p.root_x, _p.root_y)\n"
)

CLICK_CATCHER_CODE = r'''
import json, signal, sys
from Xlib import X, display

d = display.Display()
root = d.screen().root
out = {"caught": False}


def _bail(*_a):
    try:
        d.ungrab_pointer(X.CurrentTime)
        d.sync()
    except Exception:
        pass
    with open("/tmp/oev3_click.json", "w") as fh:
        json.dump(out, fh)
    sys.exit(0)


signal.signal(signal.SIGALRM, _bail)
signal.alarm(40)
root.grab_pointer(
    True, X.ButtonPressMask | X.ButtonReleaseMask,
    X.GrabModeAsync, X.GrabModeAsync, X.NONE, X.NONE, X.CurrentTime,
)
d.sync()
with open("/tmp/oev3_click_ready", "w") as fh:
    fh.write("1")
while True:
    ev = d.next_event()
    if ev.type == X.ButtonPress:
        out = {
            "caught": True,
            "root_x": ev.root_x,
            "root_y": ev.root_y,
            "button": ev.detail,
        }
        _bail()
'''


class Guest:
    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sess = requests.Session()

    def screen_size(self) -> tuple[int, int]:
        r = self._sess.post(f"{self.base_url}/screen_size", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return int(d["width"]), int(d["height"])

    def cursor(self) -> tuple[int, int]:
        r = self._sess.get(f"{self.base_url}/cursor_position", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return int(d[0]), int(d[1])

    def run(self, command: list[str], *, check: bool = True) -> dict:
        r = self._sess.post(
            f"{self.base_url}/execute",
            json={"command": command, "shell": False},
            timeout=self.timeout,
        )
        r.raise_for_status()
        res = r.json()
        if check and (res.get("status") != "success" or int(res.get("returncode", 0)) != 0):
            raise RuntimeError(f"guest command failed: {command!r} -> {res!r}")
        return res

    def py(self, code: str, *, check: bool = True) -> dict:
        return self.run(["python", "-c", code], check=check)

    def exec_agent_program(self, program: str) -> dict:
        return self.py(PYAUTOGUI_PKGS_PREFIX.format(command=program))

    def exec_client(self, command: str) -> dict:
        return self.py(CLIENT_PREFIX + command)

    def xlib_cursor(self) -> tuple[int, int] | None:
        res = self.py(XLIB_POS_CODE, check=False)
        if res.get("status") != "success" or int(res.get("returncode", 1)) != 0:
            return None
        parts = (res.get("output") or "").split()
        if len(parts) != 2:
            return None
        return int(parts[0]), int(parts[1])


def clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def agent_program_for(th_x: int, th_y: int, screen: tuple[int, int]) -> tuple[str, str, int, int]:
    line = f"move({th_x},{th_y})"
    parsed = parse_ordered_action(line)
    program = compile_primitives(parsed.primitives, screen)
    req_x = round(th_x / 1000 * screen[0])
    req_y = round(th_y / 1000 * screen[1])
    return line, program, req_x, req_y


def build_trials(sw: int, sh: int) -> list[dict]:
    cx, cy = sw // 2, sh // 2
    center = (cx, cy)
    specs: list[tuple[str, tuple[int, int], int, int]] = [
        ("tiny_pp", center, 5, 5),
        ("tiny_mm", center, -5, -5),
        ("tiny_pm", center, 5, -5),
        ("tiny_mp", center, -5, 5),
        ("unit_p", center, 1, 1),
        ("unit_m", center, -1, -1),
        ("small_x_p", center, 20, 0),
        ("small_y_p", center, 0, 20),
        ("small_x_m", center, -20, 0),
        ("small_y_m", center, 0, -20),
        ("mid_pp", center, 150, 150),
        ("mid_mm", center, -150, -150),
        ("mid_pm", center, 150, -150),
        ("mid_mp", center, -150, 150),
        ("typ250_x_p", center, 250, 0),
        ("typ250_y_p", center, 0, 250),
        ("typ250_x_m", center, -250, 0),
        ("typ250_y_m", center, 0, -250),
        ("large_pp", center, 800, 400),
        ("large_mm", center, -800, -400),
        ("large_pm", center, 800, -400),
        ("large_mp", center, -800, 400),
        ("osc_fwd", center, 734, -200),
        ("osc_back", center, -734, 200),
        ("odd_mixed", center, 37, -113),
        ("odd_mixed2", center, -409, 61),
        ("edge_tl_out", (10, 10), -800, -800),
        ("edge_br_out", (sw - 10, sh - 10), 800, 800),
        ("edge_tl_in", (0, 0), 5, 5),
        ("edge_br_in", (sw - 1, sh - 1), -5, -5),
        ("edge_top_clampy", (cx, 5), 100, -200),
        ("edge_left_clampx", (5, cy), -200, 100),
        ("edge_far_right", (sw - 50, cy), 1500, 0),
        ("edge_far_bottom", (cx, sh - 50), 0, 900),
    ]
    trials = []
    for name, start, dpx, dpy in specs:
        th_x = round(dpx * 1000 / sw)
        th_y = round(dpy * 1000 / sh)
        trials.append(
            {
                "name": name,
                "start": [start[0], start[1]],
                "target_px": [dpx, dpy],
                "thousandths": [th_x, th_y],
            }
        )
    return trials


def set_cursor(g: Guest, x: int, y: int) -> tuple[int, int]:
    g.exec_client(f"pyautogui.moveTo({x}, {y})")
    return g.cursor()


def run_trials(g: Guest, screen: tuple[int, int], trials: list[dict]) -> list[dict]:
    sw, sh = screen
    rows: list[dict] = []
    for t in trials:
        th_x, th_y = t["thousandths"]
        line, program, req_x, req_y = agent_program_for(th_x, th_y, screen)
        row = dict(t)
        row["action_line"] = line
        row["program"] = program
        row["requested_px"] = [req_x, req_y]

        sx, sy = set_cursor(g, t["start"][0], t["start"][1])
        row["start_actual"] = [sx, sy]
        exp_x = clip(sx + req_x, 0, sw - 1) - sx
        exp_y = clip(sy + req_y, 0, sh - 1) - sy
        row["clipped_expected_px"] = [exp_x, exp_y]
        row["clamped"] = [exp_x != req_x, exp_y != req_y]

        t0 = time.time()
        g.exec_agent_program(program)
        row["agent_exec_s"] = round(time.time() - t0, 3)
        ax, ay = g.cursor()
        row["agent_end"] = [ax, ay]
        row["agent_achieved_px"] = [ax - sx, ay - sy]
        row["agent_err_vs_requested"] = [(ax - sx) - req_x, (ay - sy) - req_y]
        row["agent_err_vs_clipped"] = [(ax - sx) - exp_x, (ay - sy) - exp_y]
        xl = g.xlib_cursor()
        row["agent_end_xlib"] = list(xl) if xl else None

        sx2, sy2 = set_cursor(g, t["start"][0], t["start"][1])
        row["start_actual_tracked"] = [sx2, sy2]
        tx = clip(sx2 + req_x, 0, sw - 1)
        ty = clip(sy2 + req_y, 0, sh - 1)
        row["tracked_target"] = [tx, ty]
        t0 = time.time()
        if (tx, ty) != (sx2, sy2):
            g.exec_client(f"pyautogui.moveTo({tx}, {ty})")
        row["tracked_exec_s"] = round(time.time() - t0, 3)
        bx, by = g.cursor()
        row["tracked_end"] = [bx, by]
        row["tracked_achieved_px"] = [bx - sx2, by - sy2]
        exp_x2 = clip(sx2 + req_x, 0, sw - 1) - sx2
        exp_y2 = clip(sy2 + req_y, 0, sh - 1) - sy2
        row["tracked_err_vs_requested"] = [(bx - sx2) - req_x, (by - sy2) - req_y]
        row["tracked_err_vs_clipped"] = [(bx - sx2) - exp_x2, (by - sy2) - exp_y2]

        rows.append(row)
    return rows


def run_scale_sweep(g: Guest, screen: tuple[int, int]) -> list[dict]:
    sw, sh = screen
    rows = []
    for d in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 900):
        for axis in ("x", "y"):
            start = (100, 100) if d > 0 else (sw // 2, sh // 2)
            sx, sy = set_cursor(g, start[0], start[1])
            code = (
                "import pyautogui\npyautogui.FAILSAFE = False\n"
                + (f"pyautogui.moveRel({d}, 0)" if axis == "x" else f"pyautogui.moveRel(0, {d})")
            )
            g.py(PYAUTOGUI_PKGS_PREFIX.format(command=code))
            ex, ey = g.cursor()
            rows.append(
                {
                    "axis": axis,
                    "requested": d,
                    "achieved": (ex - sx) if axis == "x" else (ey - sy),
                    "start": [sx, sy],
                    "end": [ex, ey],
                }
            )
    return rows


def run_oscillation(g: Guest, screen: tuple[int, int], cycles: int = 6) -> dict:
    sw, sh = screen
    th_fwd = (round(734 * 1000 / sw), round(-200 * 1000 / sh))
    th_back = (round(-734 * 1000 / sw), round(200 * 1000 / sh))
    _, prog_fwd, rfx, rfy = agent_program_for(th_fwd[0], th_fwd[1], screen)
    _, prog_back, rbx, rby = agent_program_for(th_back[0], th_back[1], screen)
    start = set_cursor(g, sw // 2, sh // 2)
    path = [list(start)]
    for _ in range(cycles):
        g.exec_agent_program(prog_fwd)
        path.append(list(g.cursor()))
        g.exec_agent_program(prog_back)
        path.append(list(g.cursor()))
    return {
        "requested_fwd_px": [rfx, rfy],
        "requested_back_px": [rbx, rby],
        "start": list(start),
        "path": path,
        "net_drift": [path[-1][0] - start[0], path[-1][1] - start[1]],
    }


def run_click_test(g: Guest, screen: tuple[int, int]) -> dict:
    sw, sh = screen
    out: dict = {"available": False}
    probe = g.py("import Xlib; print('ok')", check=False)
    if int(probe.get("returncode", 1)) != 0:
        out["reason"] = f"python-xlib unavailable in guest: {probe!r}"
        return out
    g.run(["bash", "-c", "rm -f /tmp/oev3_click.json /tmp/oev3_click_ready"], check=False)
    g.run(["bash", "-c", f"cat > /tmp/oev3_click_catcher.py <<'EOF'\n{CLICK_CATCHER_CODE}\nEOF"])
    g.run(
        [
            "bash",
            "-c",
            "nohup python /tmp/oev3_click_catcher.py "
            "> /tmp/oev3_click_catcher.log 2>&1 & echo started",
        ]
    )
    ready = False
    for _ in range(40):
        res = g.run(["bash", "-c", "test -f /tmp/oev3_click_ready && echo yes || echo no"], check=False)
        if "yes" in (res.get("output") or ""):
            ready = True
            break
        time.sleep(0.5)
    if not ready:
        log = g.run(["bash", "-c", "cat /tmp/oev3_click_catcher.log || true"], check=False)
        out["reason"] = f"click catcher never grabbed pointer: {log.get('output')!r}"
        return out

    start = set_cursor(g, sw // 2, sh // 2)
    th_x = round(300 * 1000 / sw)
    th_y = round(-180 * 1000 / sh)
    line = f"move({th_x},{th_y});down(LMB);up(LMB)"
    parsed = parse_ordered_action(line)
    program = compile_primitives(parsed.primitives, screen)
    req_x = round(th_x / 1000 * sw)
    req_y = round(th_y / 1000 * sh)
    g.exec_agent_program(program)
    time.sleep(1.0)
    after = g.cursor()
    res = g.run(["bash", "-c", "cat /tmp/oev3_click.json 2>/dev/null || echo '{}'"], check=False)
    try:
        caught = json.loads((res.get("output") or "{}").strip() or "{}")
    except json.JSONDecodeError:
        caught = {}
    g.run(["bash", "-c", "pkill -f oev3_click_catcher.py || true"], check=False)
    out.update(
        {
            "available": True,
            "action_line": line,
            "program": program,
            "start": list(start),
            "requested_px": [req_x, req_y],
            "expected_landing": [start[0] + req_x, start[1] + req_y],
            "cursor_after": list(after),
            "click_event": caught,
        }
    )
    if caught.get("caught"):
        out["click_minus_readback"] = [
            caught["root_x"] - after[0],
            caught["root_y"] - after[1],
        ]
        out["click_minus_expected"] = [
            caught["root_x"] - (start[0] + req_x),
            caught["root_y"] - (start[1] + req_y),
        ]
    return out


def _stats(vals: list[int]) -> dict:
    if not vals:
        return {"n": 0}
    a = [abs(v) for v in vals]
    return {
        "n": len(vals),
        "mean_abs": round(statistics.fmean(a), 4),
        "median_abs": round(statistics.median(a), 4),
        "max_abs": max(a),
        "mean_signed": round(statistics.fmean(vals), 4),
        "n_nonzero": sum(1 for v in vals if v != 0),
    }


def _scale(pairs: list[tuple[int, int]]) -> float | None:
    num = sum(r * a for r, a in pairs)
    den = sum(r * r for r, _ in pairs)
    return round(num / den, 6) if den else None


def summarize(rows: list[dict], sweep: list[dict]) -> dict:
    unclamped_x = [r for r in rows if not r["clamped"][0]]
    unclamped_y = [r for r in rows if not r["clamped"][1]]
    summary: dict = {
        "n_trials": len(rows),
        "agent": {
            "x_err_vs_clipped": _stats([r["agent_err_vs_clipped"][0] for r in rows]),
            "y_err_vs_clipped": _stats([r["agent_err_vs_clipped"][1] for r in rows]),
            "x_err_unclamped": _stats([r["agent_err_vs_requested"][0] for r in unclamped_x]),
            "y_err_unclamped": _stats([r["agent_err_vs_requested"][1] for r in unclamped_y]),
            "x_scale": _scale(
                [(r["requested_px"][0], r["agent_achieved_px"][0]) for r in unclamped_x]
            ),
            "y_scale": _scale(
                [(r["requested_px"][1], r["agent_achieved_px"][1]) for r in unclamped_y]
            ),
        },
        "tracked": {
            "x_err_vs_clipped": _stats([r["tracked_err_vs_clipped"][0] for r in rows]),
            "y_err_vs_clipped": _stats([r["tracked_err_vs_clipped"][1] for r in rows]),
            "x_err_unclamped": _stats([r["tracked_err_vs_requested"][0] for r in unclamped_x]),
            "y_err_unclamped": _stats([r["tracked_err_vs_requested"][1] for r in unclamped_y]),
            "x_scale": _scale(
                [(r["requested_px"][0], r["tracked_achieved_px"][0]) for r in unclamped_x]
            ),
            "y_scale": _scale(
                [(r["requested_px"][1], r["tracked_achieved_px"][1]) for r in unclamped_y]
            ),
        },
        "agent_vs_tracked_disagreements": [
            r["name"]
            for r in rows
            if r["agent_achieved_px"] != r["tracked_achieved_px"]
        ],
        "xlib_readback_disagreements": [
            r["name"]
            for r in rows
            if r["agent_end_xlib"] is not None and r["agent_end_xlib"] != r["agent_end"]
        ],
    }
    sx = [(s["requested"], s["achieved"]) for s in sweep if s["axis"] == "x"]
    sy = [(s["requested"], s["achieved"]) for s in sweep if s["axis"] == "y"]
    summary["sweep_x_scale"] = _scale(sx)
    summary["sweep_y_scale"] = _scale(sy)
    summary["sweep_x_max_abs_err"] = max((abs(a - r) for r, a in sx), default=None)
    summary["sweep_y_max_abs_err"] = max((abs(a - r) for r, a in sy), default=None)
    return summary


def render_table(rows: list[dict]) -> str:
    hdr = (
        f"{'trial':<18}{'start':>12}{'th(dx,dy)':>14}{'req_px':>14}"
        f"{'clip_exp':>14}{'agent_got':>14}{'a_err':>10}{'tracked_got':>14}{'t_err':>10}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['name']:<18}"
            f"{str(tuple(r['start_actual'])):>12}"
            f"{str(tuple(r['thousandths'])):>14}"
            f"{str(tuple(r['requested_px'])):>14}"
            f"{str(tuple(r['clipped_expected_px'])):>14}"
            f"{str(tuple(r['agent_achieved_px'])):>14}"
            f"{str(tuple(r['agent_err_vs_clipped'])):>10}"
            f"{str(tuple(r['tracked_achieved_px'])):>14}"
            f"{str(tuple(r['tracked_err_vs_clipped'])):>10}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--qcow2", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--settle-s", type=float, default=0.35)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = None
    vm_path = None
    if args.base_url:
        base_url = args.base_url
    else:
        import os

        from osworld_fullbench_kvm import _lease_vm_ports

        _lease_vm_ports()
        os.environ.setdefault("OSWORLD_VM_LOG_DIR", str(out_dir))
        import qemu_kvm_provider

        vm_path = args.qcow2 or qemu_kvm_provider.DEFAULT_QCOW2
        provider = qemu_kvm_provider.KvmProvider()
        print(f"booting VM {vm_path}", flush=True)
        t0 = time.time()
        provider.start_emulator(vm_path)
        print(f"VM ready in {time.time() - t0:.0f}s", flush=True)
        ports = provider._vms[vm_path]["ports"]
        base_url = f"http://localhost:{ports['server']}"

    try:
        g = Guest(base_url)
        sw, sh = g.screen_size()
        print(f"guest screen_size = {sw}x{sh}", flush=True)
        platform_res = g.run(["python", "-c", "import pyautogui, sys; print(pyautogui.__version__); print(sys.version)"], check=False)
        print(f"guest pyautogui/python: {platform_res.get('output')!r}", flush=True)
        time.sleep(args.settle_s)

        trials = build_trials(sw, sh)
        rows = run_trials(g, (sw, sh), trials)
        sweep = run_scale_sweep(g, (sw, sh))
        osc = run_oscillation(g, (sw, sh))
        click = run_click_test(g, (sw, sh))
        summary = summarize(rows, sweep)

        payload = {
            "screen_size": [sw, sh],
            "agent_screen_size_default": [1920, 1080],
            "guest_python": platform_res.get("output"),
            "trials": rows,
            "scale_sweep": sweep,
            "oscillation": osc,
            "click_test": click,
            "summary": summary,
        }
        (out_dir / "probe_move_dispatch.json").write_text(json.dumps(payload, indent=2))
        table = render_table(rows)
        (out_dir / "probe_move_dispatch_table.txt").write_text(
            table + "\n\n" + json.dumps(summary, indent=2) + "\n"
        )
        print(table, flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        print("oscillation:", json.dumps(osc), flush=True)
        print("click_test:", json.dumps(click), flush=True)
    finally:
        if provider is not None and vm_path is not None:
            provider.stop_emulator(vm_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
