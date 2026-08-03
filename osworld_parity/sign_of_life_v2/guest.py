from __future__ import annotations

import json
import shlex
import time
import urllib.request
from pathlib import Path
from typing import Any

from ..proper_vm_capability_ladder.rung1.transport import HttpVmTransport
from .suite import DevelopmentTask


STATE_PREFIX = "SOLV2_STATE="
ROOT = Path("/tmp/crowdcast_sign_of_life_v2")


def _stdout(result: dict[str, Any]) -> str:
    value = result.get("output")
    if not isinstance(value, str):
        raise RuntimeError("guest command returned no stdout")
    return value


def _run_bash(transport: HttpVmTransport, script: str) -> str:
    return _stdout(transport.execute_argv(["bash", "-lc", script]))


def _task_root(task: DevelopmentTask) -> Path:
    return ROOT / task.id


def _active_window_script() -> str:
    return """
wid=$(xprop -root _NET_ACTIVE_WINDOW | sed -n 's/.*# //p')
test -n "$wid"; test "$wid" != "0x0"
xprop -id "$wid" WM_CLASS _NET_WM_NAME WM_NAME 2>/dev/null || true
""".strip()


def _launch_terminal_script(root: Path, title: str, rc_body: str, *, geometry: str) -> str:
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}
rm -rf -- "$root"
mkdir -p -- "$root"
rc="$root/rc.sh"
cat >"$rc" <<'SOLV2_RC'
{rc_body}
SOLV2_RC
chmod 700 "$rc"
terminal=$(command -v gnome-terminal || command -v xfce4-terminal || command -v xterm)
test -n "$terminal"
case "$(basename "$terminal")" in
  gnome-terminal) nohup "$terminal" --geometry={shlex.quote(geometry)} --title={shlex.quote(title)} -- bash --noprofile --rcfile "$rc" -i >"$root/terminal-launch.log" 2>&1 </dev/null & ;;
  xfce4-terminal) nohup "$terminal" --geometry={shlex.quote(geometry)} --title={shlex.quote(title)} --command="bash --noprofile --rcfile '$rc' -i" >"$root/terminal-launch.log" 2>&1 </dev/null & ;;
  *) nohup "$terminal" -geometry {shlex.quote(geometry)} -title {shlex.quote(title)} -e bash --noprofile --rcfile "$rc" -i >"$root/terminal-launch.log" 2>&1 </dev/null & ;;
esac
for _ in $(seq 1 120); do
  win=$(wmctrl -l | awk -v title={shlex.quote(title)} 'index($0,title){{print $1; exit}}')
  [ -n "${{win:-}}" ] && break
  sleep 0.25
done
test -n "${{win:-}}"
wmctrl -ir "$win" -b remove,maximized_vert,maximized_horz,hidden,shaded || true
wmctrl -ir "$win" -e 0,80,120,1120,720 || true
wmctrl -ia "$win"
sleep 1
printf 'WINDOW_ID=%s\n' "$win"
{_active_window_script()}
""".strip()


def setup_task(transport: HttpVmTransport, task: DevelopmentTask) -> dict[str, Any]:
    root = _task_root(task)
    title = f"SOLV2 {task.id}"
    if task.kind == "terminal_command":
        marker = str(task.expected["listing_marker"])
        rc_body = f"""
mkdir -p {shlex.quote(str(root / 'listing'))}
touch {shlex.quote(str(root / 'listing' / marker))}
cd {shlex.quote(str(root / 'listing'))}
export HISTFILE={shlex.quote(str(root / 'history'))}
export HISTCONTROL=
export HISTSIZE=100
set -o history
shopt -s histappend
exec > >(tee -a {shlex.quote(str(root / 'transcript'))}) 2>&1
export PS1='SOLV2-LS$ '
PROMPT_COMMAND='history -a'
""".strip()
        script = _launch_terminal_script(root, title, rc_body, geometry="110x34+80+120")
        script += f"\nwmctrl -a {shlex.quote(title)}\nsleep 0.5"
        output = _run_bash(transport, script)
        geometry = _window_geometry(transport, title)
        return {"title": title, "window": geometry, "setup_output": output}
    if task.kind == "terminal_exact_text":
        expected = str(task.expected["text"])
        rc_body = f"""
export PS1='SOLV2-TEXT$ '
printf 'Type the requested exact text, then press Enter:\n> '
IFS= read -r SOLV2_LINE
printf '%s' "$SOLV2_LINE" > {shlex.quote(str(root / 'captured.txt'))}
printf '\nCaptured %s bytes.\n' "${{#SOLV2_LINE}}"
""".strip()
        output = _run_bash(
            transport,
            _launch_terminal_script(root, title, rc_body, geometry="110x34+80+120"),
        )
        geometry = _window_geometry(transport, title)
        return {"title": title, "window": geometry, "expected_text": expected, "setup_output": output}
    if task.kind == "open_chrome":
        script = f"""
set -euo pipefail
root={shlex.quote(str(root))}
rm -rf -- "$root"; mkdir -p -- "$root"
pkill -x chrome 2>/dev/null || true
pkill -x google-chrome 2>/dev/null || true
pkill -x chromium 2>/dev/null || true
pkill -x chromium-browser 2>/dev/null || true
for _ in $(seq 1 40); do
  wmctrl -lx 2>/dev/null | grep -Eqi 'google-chrome|chromium' || break
  sleep 0.25
done
wmctrl -k on || true
python3 -c "import pyautogui,time; pyautogui.moveTo(960,540); time.sleep(1)"
wmctrl -lx || true
""".strip()
        output = _run_bash(transport, script)
        return {
            "dock_chrome_coordinate": [35, 60],
            "setup_output": output,
            "chrome_absent_before": not probe_task(transport, task)["chrome_process"],
        }
    if task.kind == "focus_terminal_and_type":
        command = str(task.expected["command"])
        rc_body = f"""
export HISTFILE={shlex.quote(str(root / 'history'))}
export HISTCONTROL=
export HISTSIZE=100
set -o history
shopt -s histappend
export PS1='SOLV2-COMPOUND$ '
PROMPT_COMMAND='history -a'
""".strip()
        script = _launch_terminal_script(root, title, rc_body, geometry="100x30+80+160")
        script += f"""
cat >{shlex.quote(str(root / 'focus-note.sh'))} <<'SOLV2_NOTE'
#!/usr/bin/env bash
exec xmessage -geometry 420x180+1350+180 -title 'SOLV2 desktop note' 'The terminal is visible. Click it before typing.'
SOLV2_NOTE
chmod 700 {shlex.quote(str(root / 'focus-note.sh'))}
if command -v xmessage >/dev/null; then
  nohup {shlex.quote(str(root / 'focus-note.sh'))} >{shlex.quote(str(root / 'note.log'))} 2>&1 </dev/null &
  for _ in $(seq 1 40); do wmctrl -l | grep -Fq 'SOLV2 desktop note' && break; sleep 0.25; done
  wmctrl -a 'SOLV2 desktop note'
else
  nohup nautilus --new-window {shlex.quote(str(root))} >{shlex.quote(str(root / 'note.log'))} 2>&1 </dev/null &
  sleep 2
  win=$(wmctrl -lx | awk 'tolower($0) ~ /nautilus/ {{print $1; exit}}')
  test -n "$win"; wmctrl -ia "$win"
fi
sleep 1
"""
        output = _run_bash(transport, script)
        geometry = _window_geometry(transport, title)
        active = _run_bash(transport, _active_window_script())
        if "terminal" in active.casefold():
            raise RuntimeError("compound setup did not remove terminal focus")
        return {
            "title": title,
            "window": geometry,
            "terminal_click_coordinate": [geometry["x"] + geometry["width"] // 2, geometry["y"] + 100],
            "expected_command": command,
            "active_window_after_setup": active,
            "setup_output": output,
        }
    raise ValueError(f"unsupported task kind: {task.kind}")


def _window_geometry(transport: HttpVmTransport, title: str) -> dict[str, int | str]:
    code = f"""
import json,subprocess
title={title!r}
rows=subprocess.run(['wmctrl','-lGx'],capture_output=True,text=True,check=True).stdout.splitlines()
row=next((line for line in rows if title in line),None)
if row is None: raise RuntimeError('window not found')
parts=row.split(None,8)
value={{'window_id':parts[0],'x':int(parts[2]),'y':int(parts[3]),'width':int(parts[4]),'height':int(parts[5]),'window_line':row}}
print('SOLV2_GEOMETRY='+json.dumps(value,sort_keys=True))
""".strip()
    output = _stdout(transport.execute_argv(["python3", "-c", code]))
    lines = [line for line in output.splitlines() if line.startswith("SOLV2_GEOMETRY=")]
    if len(lines) != 1:
        raise RuntimeError("window geometry evidence missing")
    return json.loads(lines[0].removeprefix("SOLV2_GEOMETRY="))


def probe_task(transport: HttpVmTransport, task: DevelopmentTask) -> dict[str, Any]:
    root = _task_root(task)
    expected_file = str(task.expected.get("file", ""))
    code = f"""
import json,pathlib,subprocess
root=pathlib.Path({str(root)!r})
def text(path):
 try: return pathlib.Path(path).read_text(encoding='utf-8')
 except (FileNotFoundError,UnicodeDecodeError,OSError): return None
def run(argv):
 return subprocess.run(argv,capture_output=True,text=True,check=False).stdout
active_id=run(['xprop','-root','_NET_ACTIVE_WINDOW']).strip().split()[-1]
active=run(['xprop','-id',active_id,'WM_CLASS','_NET_WM_NAME','WM_NAME']) if active_id not in ('','0x0') else ''
windows=run(['wmctrl','-lGx'])
processes=run(['ps','-eo','comm='])
proof=pathlib.Path({expected_file!r}) if {bool(expected_file)!r} else None
transcript=text(root/'transcript')
value={{
 'schema_version':1,
 'task_id':{task.id!r},
 'active_window':active,
 'windows':windows,
 'chrome_process':any(row.strip().lower() in ('chrome','google-chrome','chromium','chromium-browser') for row in processes.splitlines()),
 'history':text(root/'history'),
 'transcript':transcript,
 'prompt_count':0 if transcript is None else transcript.count('SOLV2-LS$'),
 'capture_file_exists':(root/'captured.txt').is_file(),
 'captured_text':text(root/'captured.txt'),
 'proof_file_exists':False if proof is None else proof.is_file(),
 'proof_file_content':None if proof is None else text(proof),
}}
print({STATE_PREFIX!r}+json.dumps(value,ensure_ascii=False,sort_keys=True))
""".strip()
    output = _stdout(transport.execute_argv(["python3", "-c", code]))
    lines = [line for line in output.splitlines() if line.startswith(STATE_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError("guest state evidence missing or ambiguous")
    value = json.loads(lines[0].removeprefix(STATE_PREFIX))
    if not isinstance(value, dict):
        raise RuntimeError("guest state evidence is not an object")
    return value


def capture_screenshot(transport: HttpVmTransport, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(transport.base_url + "/screenshot", timeout=30) as response:
        payload = response.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("invalid screenshot payload")
    path.write_bytes(payload)
    time.sleep(0.05)
    return {"path": str(path), "bytes": len(payload)}
