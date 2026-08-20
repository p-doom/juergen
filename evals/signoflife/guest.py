"""In-guest setup, read-only state extraction, and the scripted control arms.

Everything family-specific about the cells lives here, behind the `Preparer` seam,
so the harness never branches on task kind.

Three properties are load-bearing:

  * Setup is hermetic per cell. Each cell wipes and rebuilds
    `/tmp/crowdcast_sign_of_life_v2/<cell>`, launches its own terminal with its own
    rcfile (own `HISTFILE`, own `PS1`, `tee`'d transcript) or its own Tk panel, and
    positions the window at a fixed geometry. Two cells that shared a shell would
    share history.
  * Extraction is read-only and input-free. `probe` runs one `python3 -c`
    inside the guest and prints a single `SOLV2_STATE=` line; missing or ambiguous
    evidence raises rather than degrading to `success=False`.
  * The control arms go through the codec. `render_step` produces codec text, not
    operations, so the oracle (expected all-pass) and negative (expected all-fail)
    arms exercise the same `parse` and `compile` the model arm does. It renders one
    intent at a time, because the relative encodings resolve a click against a
    cursor read that must be fresh.

Every click on a panel cell is resolved from the fixture's own runtime
measurement (`evals/fixtures/tk.py`), never from a coordinate typed into this
file: a fixture that measures its widgets and is then never asked for the
measurement is the same defect as an eyeballed bbox.
"""

from __future__ import annotations

import base64
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from grammars.ordered_events_v3.codec import escape

from evals.fixtures import tk
from evals.signoflife.oracle import evaluate_postcondition
from evals.signoflife.suite import ALLOWED_KINDS
from evals.tasks import DesktopTaskData, register_preparer

__all__ = [
    "Intent",
    "ROOT",
    "SCRIPT_RENDERERS",
    "STATE_PREFIX",
    "SignOfLifePreparer",
    "register_preparers",
    "render_step",
    "script_plan",
]

STATE_PREFIX = "SOLV2_STATE="
ROOT = Path("/tmp/crowdcast_sign_of_life_v2")
DOCK_CHROME_COORDINATE = (35, 60)


def _stdout(result: dict[str, Any]) -> str:
    value = result.get("output")
    if not isinstance(value, str):
        raise RuntimeError("guest command returned no stdout")
    return value


def _bash(session: Any, script: str) -> str:
    return _stdout(session.execute_argv(["bash", "-lc", script]))


def _task_root(task_id: str) -> Path:
    return ROOT / task_id


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


def _window_geometry(session: Any, title: str) -> dict[str, Any]:
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
    output = _stdout(session.execute_argv(["python3", "-c", code]))
    lines = [line for line in output.splitlines() if line.startswith("SOLV2_GEOMETRY=")]
    if len(lines) != 1:
        raise RuntimeError("window geometry evidence missing")
    return json.loads(lines[0].removeprefix("SOLV2_GEOMETRY="))


def probe_state(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    """One read-only guest probe. Ambiguous evidence fails closed."""
    root = _task_root(task.name or "")
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
 'task_id':{(task.name or '')!r},
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
 'keystroke_state':json.loads(text(root/'keys.json') or 'null'),
 'stage_one_text':text(root/'stage_one.txt'),
 'commit_text':text(root/'committed.txt'),
 'panel_state':json.loads(text(root/{tk.STATE_NAME!r}) or 'null'),
}}
print({STATE_PREFIX!r}+json.dumps(value,ensure_ascii=False,sort_keys=True))
""".strip()
    output = _stdout(session.execute_argv(["python3", "-c", code]))
    lines = [line for line in output.splitlines() if line.startswith(STATE_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError("guest state evidence missing or ambiguous")
    value = json.loads(lines[0].removeprefix(STATE_PREFIX))
    if not isinstance(value, dict):
        raise RuntimeError("guest state evidence is not an object")
    return value


def _setup_terminal_command(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    root = _task_root(task.name or "")
    title = f"SOLV2 {task.name}"
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
    output = _bash(session, script)
    return {"title": title, "window": _window_geometry(session, title), "setup_output": output}


def _setup_terminal_exact_text(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    root = _task_root(task.name or "")
    title = f"SOLV2 {task.name}"
    rc_body = f"""
export PS1='SOLV2-TEXT$ '
printf 'Type the requested exact text, then press Enter:\n> '
IFS= read -r SOLV2_LINE
printf '%s' "$SOLV2_LINE" > {shlex.quote(str(root / 'captured.txt'))}
printf '\nCaptured %s bytes.\n' "${{#SOLV2_LINE}}"
""".strip()
    output = _bash(
        session, _launch_terminal_script(root, title, rc_body, geometry="110x34+80+120")
    )
    return {
        "title": title,
        "window": _window_geometry(session, title),
        "expected_text": str(task.expected["text"]),
        "setup_output": output,
    }


def _setup_open_chrome(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    root = _task_root(task.name or "")
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
    output = _bash(session, script)
    return {
        "dock_chrome_coordinate": list(DOCK_CHROME_COORDINATE),
        "setup_output": output,
        "chrome_absent_before": not probe_state(session, task)["chrome_process"],
    }


def _setup_focus_terminal_and_type(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    root = _task_root(task.name or "")
    title = f"SOLV2 {task.name}"
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
    output = _bash(session, script)
    geometry = _window_geometry(session, title)
    active = _bash(session, _active_window_script())
    if "terminal" in active.casefold():
        raise RuntimeError("compound setup did not remove terminal focus")
    return {
        "title": title,
        "window": geometry,
        "terminal_click_coordinate": [
            geometry["x"] + geometry["width"] // 2,
            geometry["y"] + 100,
        ],
        "expected_command": str(task.expected["command"]),
        "active_window_after_setup": active,
        "setup_output": output,
    }


def _keystroke_reader(state: Path) -> str:
    """A reader that records what arrived, not just whether it succeeded.

    `ICANON` off with `ECHO` left on: the reader sees every character as it lands
    (so a literal `\\n` typed instead of a Return is published as a two-character
    prefix that never completed) while the terminal still shows the model what it
    typed. Publishing on every character, not only on completion, is what makes a
    failure diagnosable without re-running the VM.
    """
    return f"""
import json,os,sys,tempfile,termios,time
STATE={str(state)!r}
def publish(prefix,completed):
 fd,tmp=tempfile.mkstemp(dir=os.path.dirname(STATE))
 with os.fdopen(fd,'w') as handle:
  json.dump({{'schema_version':1,'prefix':prefix,'prefix_len':len(prefix),
   'completed':completed}},handle,sort_keys=True)
 os.replace(tmp,STATE)
fd=sys.stdin.fileno()
saved=termios.tcgetattr(fd)
mode=termios.tcgetattr(fd)
mode[3]&=~termios.ICANON
mode[6][termios.VMIN]=1
mode[6][termios.VTIME]=0
termios.tcsetattr(fd,termios.TCSANOW,mode)
prefix=''
publish(prefix,False)
sys.stdout.write('Press Enter. Type nothing else.\\r\\n')
sys.stdout.flush()
try:
 while True:
  data=os.read(fd,1)
  if not data: break
  char=data.decode('utf-8','replace')
  if char in ('\\r','\\n'):
   publish(prefix,True)
   break
  prefix+=char
  publish(prefix,False)
finally:
 termios.tcsetattr(fd,termios.TCSADRAIN,saved)
sys.stdout.write('\\r\\nKeypress recorded.\\r\\n')
sys.stdout.flush()
time.sleep(600)
""".strip()


def _setup_submit_only(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    root = _task_root(task.name or "")
    title = f"SOLV2 {task.name}"
    reader = root / "reader.py"
    encoded = base64.b64encode(
        _keystroke_reader(root / "keys.json").encode("utf-8")
    ).decode("ascii")
    rc_body = f"""
printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(str(reader))}
exec python3 {shlex.quote(str(reader))}
""".strip()
    script = _launch_terminal_script(root, title, rc_body, geometry="90x18+80+120")
    # Wait for the reader's first publish. Without it the initial probe can find no
    # keystroke evidence at all, which `evaluate_postcondition` reports as
    # `status="error"` and `require_unsolved_start` then raises on: a race that
    # presents as an infrastructure failure rather than as the cell it is.
    keys = shlex.quote(str(root / "keys.json"))
    script += f"\nfor _ in $(seq 1 80); do [ -s {keys} ] && break; sleep 0.25; done\ntest -s {keys}"
    output = _bash(session, script)
    return {
        "title": title,
        "window": _window_geometry(session, title),
        "expected_prefix": str(task.expected["keystroke_prefix"]),
        "setup_output": output,
    }


def _setup_staged_confirm(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    """Two stages, and only the screen says there is a second one.

    The instruction names the goal, not the steps, so a policy that stops when the
    first sub-goal looks done fails the cell instead of scoring it.
    """
    root = _task_root(task.name or "")
    title = f"SOLV2 {task.name}"
    confirmation = str(task.expected["confirmation"])
    rc_body = f"""
export PS1='SOLV2-STAGE$ '
printf 'Report id: '
IFS= read -r SOLV2_ID
printf '%s' "$SOLV2_ID" > {shlex.quote(str(root / 'stage_one.txt'))}
printf '\\nStage 1 recorded %s. Now type {confirmation} and press Enter to commit.\\n> ' "$SOLV2_ID"
IFS= read -r SOLV2_CONFIRM
if [ "$SOLV2_CONFIRM" = {shlex.quote(confirmation)} ]; then
  printf '%s' "$SOLV2_ID" > {shlex.quote(str(root / 'committed.txt'))}
  printf '\\nCommitted.\\n'
else
  printf '\\nAborted: nothing committed.\\n'
fi
sleep 600
""".strip()
    output = _bash(
        session, _launch_terminal_script(root, title, rc_body, geometry="100x24+80+120")
    )
    return {
        "title": title,
        "window": _window_geometry(session, title),
        "expected_report_id": str(task.expected["report_id"]),
        "setup_output": output,
    }


def _launch_panel(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    root = _task_root(task.name or "")
    title = f"SOLV2 {task.name}"
    panel = tk.panel_from_expected(task.expected.get("panel"), title=title)
    cursor_start = _cursor_start(task)
    output = _bash(session, tk.setup_script(panel, root=root, cursor_start=cursor_start))
    if tk.TK_MISSING_MARKER in output:
        raise RuntimeError(
            "the guest image cannot import tkinter, so the panel cells cannot run; "
            "re-bake the image with python3-tk"
        )
    state = tk.parse_state(output)
    measured = state.get("widgets") or {}
    missing = [name for name in panel.widgets if name not in measured]
    if missing:
        raise RuntimeError(f"the panel published no measured bbox for {missing}")
    return {
        "title": title,
        "cursor_start": list(cursor_start),
        "panel_state": state,
        "setup_output": output,
    }


def _setup_tk_target_click(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    evidence = _launch_panel(session, task)
    evidence["single_move_reach"] = _assert_off_lattice(task, evidence["panel_state"])
    return evidence


def _setup_tk_no_submit_entry(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    return _launch_panel(session, task)


def _cursor_start(task: DesktopTaskData) -> tuple[int, int]:
    x, y = (int(value) for value in task.expected["cursor_start"])
    return (x, y)


def _lattice_move_count(low: int, high: int, support: tuple[int, ...]) -> int:
    """Fewest support-sized steps that land the cursor inside `[low, high]`.

    `0` means the interval already contains the cursor. Exact rather than greedy
    over one target, because the admissible landing zone is the whole measured
    widget and the cheapest point in it is the one a policy would find.
    """
    units = sorted({abs(int(value)) for value in support} - {0}, reverse=True)
    if not units:
        raise ValueError("the single-move support must contain a non-zero step")
    best: int | None = None
    for value in range(low, high + 1):
        count = 0
        rest = abs(value)
        for unit in units:
            count += rest // unit
            rest %= unit
        if rest:
            continue
        best = count if best is None else min(best, count)
    if best is None:
        raise ValueError(f"no point in [{low}, {high}] is reachable from {support}")
    return best


def _assert_off_lattice(task: DesktopTaskData, state: dict[str, Any]) -> int:
    """The cell's premise, checked against the measured target.

    A collapsed policy's observed output support is {0, +-1, +-10, +-100} per
    axis, so a cell that means to require chained moves must name a target no
    single support value reaches — and one a chain still reaches inside the step
    budget, or it measures nothing but the budget. Both are properties of the
    measured bbox, so both are asserted here instead of assumed from the nominal
    window position.
    """
    support = tuple(int(value) for value in task.expected["single_move_support"])
    cursor = _cursor_start(task)
    box = tk.widget_bbox(state, tk.button_widget(str(task.expected["target_label"])))
    moves = max(
        _lattice_move_count(box[0] - cursor[0], box[2] - 1 - cursor[0], support),
        _lattice_move_count(box[1] - cursor[1], box[3] - 1 - cursor[1], support),
    )
    if not 2 <= moves <= task.max_steps:
        raise RuntimeError(
            f"{task.name}: the measured target at {box} is {moves} support-sized "
            f"moves from the cursor start {cursor}; the cell requires 2 to "
            f"{task.max_steps}, so it measures a single move or an impossible one. "
            "Move `expected.cursor_start`, not the assertion."
        )
    return moves


_SETUPS: dict[str, Callable[[Any, DesktopTaskData], dict[str, Any]]] = {
    "terminal_command": _setup_terminal_command,
    "terminal_exact_text": _setup_terminal_exact_text,
    "open_chrome": _setup_open_chrome,
    "focus_terminal_and_type": _setup_focus_terminal_and_type,
    "submit_only": _setup_submit_only,
    "staged_confirm": _setup_staged_confirm,
    "tk_target_click": _setup_tk_target_click,
    "tk_no_submit_entry": _setup_tk_no_submit_entry,
}


@dataclass(frozen=True)
class Intent:
    """One scripted step, grammar-neutral.

    `kind` is `click` (at `target`, a screen coordinate, or at `widget`, a name the
    fixture measured), `type` (`text`) or `submit`. The plan is the cell's gold (or
    plausibly-wrong) behaviour; the renderer is the encoding.
    """

    kind: str
    target: tuple[int, int] | None = None
    text: str = ""
    widget: str = ""


def _panel_state(session: Any, task: DesktopTaskData) -> dict[str, Any]:
    return tk.parse_state(
        _stdout(
            session.execute_argv(
                ["cat", str(_task_root(task.name or "") / tk.STATE_NAME)]
            )
        )
    )


def _click_target(
    session: Any, task: DesktopTaskData, intent: Intent
) -> tuple[int, int]:
    """Where the click lands, recomputed from the guest rather than cached.

    One `Preparer` instance is shared by every concurrent rollout in the process, so
    caching a coordinate on `self` would let one cell's geometry leak into another's.

    Three sources, in falling order of specificity: the plan's own coordinate, the
    fixture's measured widget, the window rectangle. A named widget resolves through
    `tk.widget_centre`, which raises when the measurement is missing — a control arm
    that silently fell back to a nominal pixel would certify nothing.
    """
    if intent.target is not None:
        return intent.target
    if intent.widget:
        return tk.widget_centre(_panel_state(session, task), intent.widget)
    if task.kind == "open_chrome":
        return DOCK_CHROME_COORDINATE
    geometry = _window_geometry(session, f"SOLV2 {task.name}")
    return (geometry["x"] + geometry["width"] // 2, geometry["y"] + 100)


def script_plan(task: DesktopTaskData, *, negative: bool) -> list[Intent]:
    """The gold plan for a cell, or the negative control's wrong-but-real plan.

    The negatives are plausible actions that cannot succeed, not no-ops: `pwd`
    instead of the required command, the wrong paragraph, a click at screen centre
    instead of the dock icon, and typing without focusing first. On the candidate
    cells each negative is a defect we have measured a checkpoint commit: a literal
    `\\n` instead of a Return, stopping when the first stage looks done, clicking the
    confusable neighbour, and the reflexive submit.
    """
    if task.kind == "terminal_command":
        return [
            Intent("type", text="pwd" if negative else str(task.expected["command"])),
            Intent("submit"),
        ]
    if task.kind == "terminal_exact_text":
        return [
            Intent("type", text="wrong text" if negative else str(task.expected["text"])),
            Intent("submit"),
        ]
    if task.kind == "open_chrome":
        # Screen centre for the negative: a real click on empty desktop. The
        # relative encodings resolve it against the live cursor, so this is also the
        # one cell where a stale cursor read would silently turn the negative into a
        # different action.
        return [Intent("click", target=(960, 540) if negative else None)]
    if task.kind == "focus_terminal_and_type":
        rows: list[Intent] = []
        if not negative:
            rows.append(Intent("click"))
        rows.extend(
            [Intent("type", text=str(task.expected["command"])), Intent("submit")]
        )
        return rows
    if task.kind == "submit_only":
        # The negative IS the defect: `type("\\n")` puts two literal characters in
        # the reader's buffer and no newline, in every grammar — the bare-line ones
        # unescape it to a backslash and an `n`, and the tool-call one has to, since
        # a real newline inside type() is refused by `lower_typing`.
        return [Intent("type", text="\\n") if negative else Intent("submit")]
    if task.kind == "staged_confirm":
        rows = [
            Intent("type", text=str(task.expected["report_id"])),
            Intent("submit"),
        ]
        if negative:
            return rows
        return rows + [
            Intent("type", text=str(task.expected["confirmation"])),
            Intent("submit"),
        ]
    if task.kind == "tk_target_click":
        label = (
            str(task.expected["decoy_labels"][0])
            if negative
            else str(task.expected["target_label"])
        )
        return [Intent("click", widget=tk.button_widget(label))]
    if task.kind == "tk_no_submit_entry":
        rows = [
            Intent("click", widget=tk.ENTRY_WIDGET),
            Intent("type", text=str(task.expected["text"])),
        ]
        if negative:
            return rows + [Intent("submit")]
        return rows + [
            Intent("click", widget=tk.button_widget(str(task.expected["draft_label"])))
        ]
    raise ValueError(task.kind)


def _tool_call(arguments: dict[str, Any]) -> str:
    payload = {"name": "computer_use", "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def _render_native_absolute(
    intent: Intent, session: Any, task: DesktopTaskData
) -> str:
    """`native_absolute` — the tool-call arm. Coordinates are absolute pixels."""
    if intent.kind == "type":
        return _tool_call({"action": "type", "text": intent.text})
    if intent.kind == "submit":
        return _tool_call({"action": "key", "keys": ["ENTER"]})
    if intent.kind == "click":
        target = _click_target(session, task, intent)
        return _tool_call({"action": "left_click", "coordinate": [target[0], target[1]]})
    raise ValueError(intent.kind)


def _render_compact_absolute(
    intent: Intent, session: Any, task: DesktopTaskData
) -> str:
    """`compact_absolute` — the bare-line absolute arm: `x y scroll ; EVENTS`.

    `from_target` here needs only element geometry: no cursor read, so nothing about
    this rendering can go stale. That is the asymmetry against `compact_raw` below.

    The paired grammars share a semantic trap: `0 0 0 ; +LMB -LMB` is a click at the
    top-left corner in this arm and "don't move, click here" in `compact_raw`. Same
    bytes, different action, so a control arm must be rendered per grammar and never
    copied between them.
    """
    if intent.kind == "type":
        return "0 0 0 ; type(" + json.dumps(intent.text, ensure_ascii=False) + ")"
    if intent.kind == "submit":
        return "0 0 0 ; +Return -Return"
    if intent.kind == "click":
        target = _click_target(session, task, intent)
        return f"{int(target[0])} {int(target[1])} 0 ; +LMB -LMB"
    raise ValueError(intent.kind)


def _render_relative(intent: Intent, session: Any, task: DesktopTaskData) -> str:
    """`deltatype_v2` / `compact_raw` — bare-line raw relative pixels.

    The click delta is `target - cursor` with the cursor read now. A stale read
    makes this rendering wrong, silently and by exactly the drift; the harness
    therefore renders one intent per step rather than a whole script up front.
    """
    if intent.kind == "type":
        return "0 0 0 ; type(" + json.dumps(intent.text, ensure_ascii=False) + ")"
    if intent.kind == "submit":
        return "0 0 0 ; +Return -Return"
    if intent.kind == "click":
        cursor = tuple(session.cursor_position())
        target = _click_target(session, task, intent)
        return f"{int(target[0]) - cursor[0]} {int(target[1]) - cursor[1]} 0 ; +LMB -LMB"
    raise ValueError(intent.kind)


def _render_ordered_events_v3(
    intent: Intent, session: Any, task: DesktopTaskData
) -> str:
    """`ordered_events_v3` — the ordered mini-program: `move(dx,dy); down(EV); up(EV)`.

    Relative like `_render_relative`, so the click delta needs the cursor read now.
    The `type()` payload is escaped by the grammar's own `escape`, not by
    `json.dumps`: this grammar accepts only `\\\\` and `\\"`, so a JSON-escaped
    payload is a parse error the moment a cell's text contains anything else.
    """
    if intent.kind == "type":
        return f'type("{escape(intent.text)}")'
    if intent.kind == "submit":
        return "down(Return); up(Return)"
    if intent.kind == "click":
        cursor = tuple(session.cursor_position())
        target = _click_target(session, task, intent)
        delta = (int(target[0]) - cursor[0], int(target[1]) - cursor[1])
        return f"move({delta[0]},{delta[1]}); down(LMB); up(LMB)"
    raise ValueError(intent.kind)


SCRIPT_RENDERERS: dict[str, Callable[[Intent, Any, DesktopTaskData], str]] = {
    "native_absolute": _render_native_absolute,
    "compact_absolute": _render_compact_absolute,
    "deltatype_v2": _render_relative,
    "compact_raw": _render_relative,
    "ordered_events_v3": _render_ordered_events_v3,
}
"""Exact codec name -> renderer.

An exact table, not a substring heuristic: `compact_raw` and `compact_absolute`
share a prefix while meaning opposite things, `native_absolute` and
`compact_absolute` both contain "absolute", and `move_rel` contains neither
"compact" nor "absolute" while being relative, so a heuristic could pick the wrong
encoding and give a control arm that fails for the wrong reason — or, for that
first pair, one that clicks in the wrong place without failing at all.
`move_rel` and `diffabs` are absent on purpose: no cell
has a scripted arm in them yet, and a missing renderer must be a loud `LookupError`,
not a silently substituted grammar."""


def render_step(
    session: Any, task: DesktopTaskData, *, codec: Any, intent: Intent
) -> str:
    """Render one intent into `codec`'s text, reading the guest as needed."""
    name = str(getattr(codec, "name", "") or "")
    renderer = SCRIPT_RENDERERS.get(name)
    if renderer is None:
        raise LookupError(
            f"no scripted control arm for grammar {name!r}; register one in "
            f"SCRIPT_RENDERERS (known: {sorted(SCRIPT_RENDERERS)})"
        )
    return renderer(intent, session, task)


class SignOfLifePreparer:
    """One `Preparer` per cell kind; the harness resolves it from `task.kind`."""

    def __init__(self, kind: str) -> None:
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"unsupported sign-of-life kind: {kind!r}")
        self.kind = kind

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        return _SETUPS[self.kind](session, task)

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        state = probe_state(session, task)
        outcome = evaluate_postcondition(
            task.name or "", self.kind, dict(task.expected), state
        )
        # The verdict rides the probe so the harness can stop as soon as the
        # postcondition is reached (the original runner's behaviour) while the
        # authoritative judgement still happens in the oracle reward.
        return {
            **state,
            "postcondition_status": outcome.status,
            "postcondition_success": outcome.success,
            "postcondition_reason": outcome.reason,
            "postcondition_evidence": outcome.evidence,
        }

    def script_plan(self, task: DesktopTaskData, *, negative: bool) -> list[Intent]:
        return script_plan(task, negative=negative)

    def render_step(
        self, session: Any, task: DesktopTaskData, *, codec: Any, intent: Intent
    ) -> str:
        return render_step(session, task, codec=codec, intent=intent)


def register_preparers() -> None:
    for kind in sorted(ALLOWED_KINDS):
        register_preparer(SignOfLifePreparer(kind))


register_preparers()
