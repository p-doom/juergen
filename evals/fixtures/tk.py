"""A stdlib-Tk panel that measures its own widgets and publishes them as JSON.

The bboxes are ground truth. Every one comes from `winfo_rootx/rooty/width/height`
on the mapped widget, so a different theme, font, scale factor or window
decoration changes the number the oracle reads instead of silently invalidating
it — which is what a bbox hand-labelled off one screenshot does. Measuring and
then not using the measurement is the same defect one step later, so
`signoflife.guest._click_target` resolves every scripted click on a panel cell
through `widget_centre` and nothing else.

Three properties are load-bearing:

  * `panel.json` is the only channel: widget bboxes, the click order, the entry
    text and whether the form was submitted, written with `mkstemp` +
    `os.replace` so the concurrent read-only probe never sees a half-written
    file.
  * `submitted` is sticky, and Return anywhere in the window sets it. A cell
    whose success requires *not* submitting therefore cannot be passed by a
    policy that submits and keeps going, and the reflexive Return the training
    data over-weights is exactly the event this records.
  * One program, one state shape. Which widgets exist is data (`buttons`,
    `submit_labels`); a cell chooses which clauses of the same state its oracle
    reads. A second program would be a second thing to drift.

Guest requirement: `python3 -c "import tkinter"` must succeed in the VM. The
setup script prints `TK_MISSING_MARKER` and fails rather than proceeding, because
a panel that never maps looks exactly like a model that never clicked.
"""

from __future__ import annotations

import base64
import json
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

__all__ = [
    "ENTRY_WIDGET",
    "STATE_NAME",
    "TK_MISSING_MARKER",
    "TkPanel",
    "button_widget",
    "panel_from_expected",
    "panel_program",
    "parse_state",
    "setup_script",
    "widget_bbox",
    "widget_centre",
]

STATE_NAME = "panel.json"
TK_MISSING_MARKER = "SOLV2_TK_MISSING"
ENTRY_WIDGET = "entry"


def button_widget(label: str) -> str:
    return f"button:{label}"


@dataclass(frozen=True)
class TkPanel:
    """One panel's declared widgets and window rectangle."""

    title: str
    x: int
    y: int
    width: int
    height: int
    entry_label: str
    buttons: tuple[str, ...]
    submit_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title or not self.entry_label:
            raise ValueError("a panel needs a title and an entry label")
        if not self.buttons or len(set(self.buttons)) != len(self.buttons):
            raise ValueError(f"panel buttons must be unique and non-empty: {self.buttons}")
        unknown = sorted(set(self.submit_labels) - set(self.buttons))
        if unknown:
            raise ValueError(f"submit_labels name buttons the panel does not have: {unknown}")
        if min(self.width, self.height) < 120 or min(self.x, self.y) < 0:
            raise ValueError("a panel must be on-screen and large enough to click")

    @property
    def widgets(self) -> tuple[str, ...]:
        return (ENTRY_WIDGET, *(button_widget(label) for label in self.buttons))


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def panel_program(panel: TkPanel, *, state_path: PurePosixPath) -> str:
    """The in-guest Tk program. Every coordinate it publishes it measured."""
    return f"""
import json,os,tempfile,tkinter
STATE={str(state_path)!r}
TITLE={panel.title!r}
BUTTONS={list(panel.buttons)!r}
SUBMIT={list(panel.submit_labels)!r}
root=tkinter.Tk()
root.title(TITLE)
root.geometry('{panel.width}x{panel.height}+{panel.x}+{panel.y}')
root.resizable(False,False)
clicked=[]
submitted=[False]
tkinter.Label(root,text={panel.entry_label!r}).pack(pady=(14,2))
entry=tkinter.Entry(root,width=28)
entry.pack(pady=(0,10))
widgets={{{ENTRY_WIDGET!r}:entry}}
def write():
 root.update_idletasks()
 value={{'schema_version':1,'title':TITLE,'clicked':list(clicked),
  'entry_text':entry.get(),'submitted':bool(submitted[0]),
  'screen':[root.winfo_screenwidth(),root.winfo_screenheight()],
  'widgets':{{name:[w.winfo_rootx(),w.winfo_rooty(),
   w.winfo_rootx()+w.winfo_width(),w.winfo_rooty()+w.winfo_height()]
   for name,w in widgets.items()}}}}
 fd,tmp=tempfile.mkstemp(dir=os.path.dirname(STATE))
 with os.fdopen(fd,'w') as handle: json.dump(value,handle,sort_keys=True)
 os.replace(tmp,STATE)
def press(label):
 clicked.append(label)
 if label in SUBMIT: submitted[0]=True
 write()
for label in BUTTONS:
 button=tkinter.Button(root,text=label,width=18,
  command=(lambda name=label: press(name)))
 button.pack(pady=3)
 widgets['button:'+label]=button
def submit(event):
 submitted[0]=True
 write()
root.bind('<Return>',submit)
root.bind('<KP_Enter>',submit)
entry.bind('<KeyRelease>',lambda event: write())
root.after(300,write)
root.mainloop()
""".strip()


def setup_script(
    panel: TkPanel, *, root: PurePosixPath, cursor_start: tuple[int, int]
) -> str:
    """Wipe the cell root, run the panel, focus it, park the cursor, print state.

    The cursor is parked last: a cell whose premise is a fixed offset between the
    cursor and a measured widget has no premise if the cursor is wherever the
    previous cell left it.
    """
    state = root / STATE_NAME
    program = panel_program(panel, state_path=state)
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}
rm -rf -- "$root"
mkdir -p -- "$root"
python3 -c 'import tkinter' 2>/dev/null || {{ printf '%s\\n' {TK_MISSING_MARKER}; exit 3; }}
printf '%s' {shlex.quote(_b64(program))} | base64 -d > "$root/panel.py"
nohup python3 "$root/panel.py" >"$root/panel.log" 2>&1 </dev/null &
for _ in $(seq 1 120); do
  win=$(wmctrl -l | awk -v title={shlex.quote(panel.title)} 'index($0,title){{print $1; exit}}')
  [ -n "${{win:-}}" ] && break
  sleep 0.25
done
test -n "${{win:-}}"
wmctrl -ir "$win" -b remove,maximized_vert,maximized_horz,hidden,shaded || true
wmctrl -ia "$win"
for _ in $(seq 1 80); do [ -s {shlex.quote(str(state))} ] && break; sleep 0.25; done
test -s {shlex.quote(str(state))}
python3 -c "import pyautogui,time; pyautogui.moveTo({int(cursor_start[0])},{int(cursor_start[1])}); time.sleep(0.5)"
cat {shlex.quote(str(state))}
""".strip()


def widget_bbox(state: object, name: str) -> tuple[int, int, int, int]:
    """One measured widget rectangle, or raise.

    Fails closed rather than falling back to a nominal coordinate: a scripted arm
    that silently clicks an eyeballed pixel when the measurement is missing is a
    control that certifies nothing.
    """
    if not isinstance(state, dict):
        raise RuntimeError("panel state is missing")
    widgets = state.get("widgets")
    box = widgets.get(name) if isinstance(widgets, dict) else None
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, int) for v in box):
        raise RuntimeError(f"panel state has no measured bbox for {name!r}")
    x1, y1, x2, y2 = (int(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"panel widget {name!r} measured an empty rectangle: {box}")
    return (x1, y1, x2, y2)


def widget_centre(state: object, name: str) -> tuple[int, int]:
    x1, y1, x2, y2 = widget_bbox(state, name)
    return (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)


def panel_from_expected(value: object, *, title: str) -> TkPanel:
    """Build the panel a cell declares in its `expected` block."""
    if not isinstance(value, dict):
        raise ValueError("a panel cell must declare an `expected.panel` object")
    return TkPanel(
        title=title,
        x=int(value["x"]),
        y=int(value["y"]),
        width=int(value["width"]),
        height=int(value["height"]),
        entry_label=str(value["entry_label"]),
        buttons=tuple(str(label) for label in value["buttons"]),
        submit_labels=tuple(str(label) for label in value.get("submit_labels", ())),
    )


def parse_state(output: str) -> dict:
    """The panel state from a guest command's stdout: the last JSON object in it."""
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            value = json.loads(stripped)
            if isinstance(value, dict) and value.get("schema_version") == 1:
                return value
    raise RuntimeError("panel state evidence missing from guest output")
