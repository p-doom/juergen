"""One-time recorder for the short-goal golden trajectories (plan section 2).

One task = one fresh OSWorld VM (``snapshot=on``, so nothing mutates the shared
qcow2): boot -> template setup -> seeded cursor start -> one settled screenshot
per golden turn, each turn dispatched from the oracle policy -> post-success
screenshot -> verifier. A recording is only published (``recording.json``) when
its verifier passes; a rejected task keeps its frames plus a ``failure.json``
and the process exits nonzero, so a broken template can never leak into
training data.

The output is FORMAT-AGNOSTIC: every turn stores the pixel primitives that were
actually dispatched (``primitives_px``, moves grid-snapped via
``shortgoal_grammar.snap_point_px``) together with their 0-1000 grid twin
(``primitives_grid``) and the live cursor before/after, which is everything
``shortgoal_build`` needs to render either arm — ``move_to(x,y)`` straight from
the grid twin, ``move(dx,dy)`` from the pixel deltas against ``cursor_before``.
Turn boundaries are semantic by construction (one turn = one ``GoldenStep`` =
one v4 line), so there is no action binning anywhere in this file.

Every dispatched turn is routed through the real contract before it touches the
VM: grid primitives -> ``render_line`` -> ``parse_ordered_v4_action`` ->
byte-identity re-render -> ``denorm_v4``, and the rel arm's delta line is
rendered and re-parsed too. A conversion bug therefore fails at record time
rather than surfacing as a mysterious training-data defect.

Modes:
  (default)      record the selected tasks, one fresh VM each.
  --replay_from  re-dispatch existing recordings and re-run their verifiers on
                 fresh VMs (rung 0 of the ladder), no model involved.
  --dry_run      no VM at all: walk templates, policies and the golden-step
                 conversion with a stub context and print per-task step counts.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import shortgoal_fixture as fixture
import shortgoal_golden as golden
import shortgoal_grammar as grammar
import shortgoal_templates as templates
from action_parser import OrderedAction, OrderedPrimitive, parse_ordered_v4_action
from osworld_runtime import _DEFAULT_QCOW2, _DEFAULT_QEMU_BIN, _wait_for
from osworld_vm_client import OSWorldClient

_LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1
RECORDING_NAME = "recording.json"
FAILURE_NAME = "failure.json"
REPLAY_NAME = "replay.json"
FRAMES_DIR = "frames"

GUEST_FIXTURE_PATH = "/tmp/shortgoal_fixture.py"
GUEST_SPEC_PATH = "/tmp/shortgoal_spec.json"
GUEST_PAGE_PATH = "/tmp/shortgoal_page.html"
GUEST_PROFILE_DIR = "/tmp/shortgoal-chrome"
GUEST_LOG_DIR = "/tmp"

EDITOR_BINARIES = ("gedit", "gnome-text-editor")
BROWSER_BINARIES = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")
TITLE_TOOLS = ("xdotool", "wmctrl")
PAGE_READY_TITLE = fixture.PAGE_READY_TITLE
BROWSER_WINDOW_MARKERS = ("Chrome", "Chromium")

CURSOR_MARGIN = (96, 54)
CURSOR_DRAWS = 64
WIDGET_TOLERANCE_PX = 2
APP_SETTLE_S = 0.8
READY_TIMEOUT_S = 45.0
READY_POLL_S = 0.25
KEY_PROBE_NAME = "shiftleft"
KEY_PROBE_TIMEOUT_S = 4.0
KEY_PROBE_ATTEMPTS = 3


@dataclass(frozen=True)
class Settle:
    """Screenshot settle policy, mirroring freeroll's ``--settle_*`` flags."""

    delay_s: float = 0.3
    stable_timeout_s: float = 2.0
    poll_s: float = 0.1

    def shot(self, client: OSWorldClient) -> Any:
        """One settled screenshot at native VM resolution."""
        return client.screenshot_settled(
            min_delay_s=self.delay_s,
            stability_timeout_s=self.stable_timeout_s,
            poll_s=self.poll_s,
        )


@dataclass(frozen=True)
class StepPlan:
    """One golden turn as dispatchable pixels plus its 0-1000 grid twin."""

    px_action: OrderedAction
    grid_action: OrderedAction
    abs_line: str
    rel_line: str
    zero_deltas: int


_NO_OP_ACTION = OrderedAction(primitives=(), no_op=True)
_PROCS: list[subprocess.Popen] = []


def _rng(key: str) -> random.Random:
    """A ``random.Random`` seeded only by the stable string ``key``."""
    if not isinstance(key, str) or not key:
        raise ValueError(f"seed key must be a nonempty string, got {key!r}")
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))


def cursor_start_px(
    task_id: str,
    screen_wh: tuple[int, int],
    *,
    avoid: tuple[tuple[int, int], ...] = (),
) -> tuple[int, int]:
    """The seeded, grid-snapped pixel the pointer starts a recording at.

    Keyed by ``f"{task_id}:cursor_start"`` — the same task always starts from
    the same pixel — and never equal to a point in ``avoid`` (the first move
    target, whose rel-arm delta would otherwise be the forbidden move(0,0))."""
    rng = _rng(f"{task_id}:cursor_start")
    width, height = screen_wh
    margin_x, margin_y = CURSOR_MARGIN
    if width <= 2 * margin_x or height <= 2 * margin_y:
        raise ValueError(f"screen {screen_wh!r} is too small for a seeded cursor start")
    blocked = {tuple(point) for point in avoid}
    for _ in range(CURSOR_DRAWS):
        point = (
            grammar.snap_point_px(rng.randint(margin_x, width - 1 - margin_x), width),
            grammar.snap_point_px(rng.randint(margin_y, height - 1 - margin_y), height),
        )
        if point not in blocked:
            return point
    raise ValueError(f"no cursor start for {task_id} outside {sorted(blocked)}")


def serialize_primitives(prims: Any) -> list[dict[str, Any]]:
    """Ordered primitives as JSON-ready dicts (``OrderedPrimitive`` fields)."""
    return [asdict(prim) for prim in prims]


def deserialize_primitives(rows: Any) -> tuple[OrderedPrimitive, ...]:
    """The inverse of ``serialize_primitives``, validated field by field."""
    if not isinstance(rows, list):
        raise ValueError(f"primitives must be a list, got {type(rows)!r}")
    fields = set(OrderedPrimitive.__dataclass_fields__)
    out: list[OrderedPrimitive] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - fields or "kind" not in row:
            raise ValueError(f"unusable serialized primitive: {row!r}")
        values = dict(row)
        if values.get("keys") is not None:
            values["keys"] = tuple(values["keys"])
        out.append(OrderedPrimitive(**values))
    return tuple(out)


def _rel_primitives(
    px_prims: Any, cursor_xy: tuple[int, int], screen_wh: tuple[int, int],
) -> tuple[list[OrderedPrimitive], int]:
    """A turn's pixel primitives as rel-arm grid deltas from ``cursor_xy``."""
    width, height = screen_wh
    cursor_x, cursor_y = cursor_xy
    out: list[OrderedPrimitive] = []
    zero_deltas = 0
    for prim in px_prims:
        if prim.kind != "move_to":
            out.append(prim)
            continue
        dx = grammar.norm_delta(prim.x - cursor_x, width)
        dy = grammar.norm_delta(prim.y - cursor_y, height)
        cursor_x, cursor_y = prim.x, prim.y
        if dx == 0 and dy == 0:
            zero_deltas += 1
            continue
        out.append(OrderedPrimitive(kind="move", dx=dx, dy=dy))
    return out, zero_deltas


def convert_step(
    step: golden.GoldenStep, cursor_xy: tuple[int, int], screen_wh: tuple[int, int],
) -> StepPlan:
    """One golden turn -> dispatchable pixel primitives + their grid twin.

    The pixel action is produced the same way the abs arm's model output will
    be: grid primitives are rendered as a v4 line, strictly re-parsed,
    re-rendered byte-identically and denormalized, and the resulting move
    pixels must equal ``snap_point_px`` of the policy's own targets. The rel
    arm's delta line is rendered and re-parsed against ``cursor_xy`` too, so a
    turn no arm can express is rejected here instead of at build time."""
    golden.validate_step(step)
    width, height = screen_wh
    if step[0]["kind"] == "no_op":
        return StepPlan(
            px_action=_NO_OP_ACTION,
            grid_action=_NO_OP_ACTION,
            abs_line=grammar.NO_OP_LINE,
            rel_line=grammar.NO_OP_LINE,
            zero_deltas=0,
        )
    grid_prims: list[OrderedPrimitive] = []
    snapped: list[tuple[int, int]] = []
    for prim in step:
        kind = prim["kind"]
        if kind == "move":
            target_x, target_y = prim["to_xy"]
            grid_prims.append(OrderedPrimitive(
                kind="move_to",
                x=grammar.norm_point(target_x, width),
                y=grammar.norm_point(target_y, height),
            ))
            snapped.append((
                grammar.snap_point_px(target_x, width),
                grammar.snap_point_px(target_y, height),
            ))
        elif kind in ("down", "up"):
            grid_prims.append(OrderedPrimitive(kind=kind, name=prim["name"]))
        elif kind == "type":
            grid_prims.append(OrderedPrimitive(kind="type", text=prim["text"]))
        elif kind == "scroll":
            grid_prims.append(OrderedPrimitive(kind="scroll", dx=0, dy=prim["notches"]))
        else:
            raise ValueError(f"unconvertible golden primitive: {prim!r}")
    abs_line = grammar.render_line(grid_prims, grammar.ARM_ABS)
    grid_action = parse_ordered_v4_action(abs_line, arm=grammar.ARM_ABS)
    if grammar.render_line(grid_action.primitives, grammar.ARM_ABS) != abs_line:
        raise ValueError(f"abs line does not round-trip: {abs_line!r}")
    px_action = grammar.denorm_v4(grid_action, screen_wh)
    px_targets = [(prim.x, prim.y) for prim in px_action.primitives if prim.kind == "move_to"]
    if px_targets != snapped:
        raise ValueError(f"denormalized targets {px_targets!r} != snapped {snapped!r}")
    rel_prims, zero_deltas = _rel_primitives(px_action.primitives, cursor_xy, screen_wh)
    if not rel_prims:
        raise ValueError(f"turn {abs_line!r} is a zero-delta move: the rel arm cannot emit it")
    rel_line = grammar.render_line(rel_prims, grammar.ARM_REL)
    rel_action = parse_ordered_v4_action(rel_line, arm=grammar.ARM_REL)
    if grammar.render_line(rel_action.primitives, grammar.ARM_REL) != rel_line:
        raise ValueError(f"rel line does not round-trip: {rel_line!r}")
    return StepPlan(
        px_action=px_action,
        grid_action=grid_action,
        abs_line=abs_line,
        rel_line=rel_line,
        zero_deltas=zero_deltas,
    )


def advance_cursor(px_prims: Any, cursor_xy: tuple[int, int]) -> tuple[int, int]:
    """Where a turn's pixel primitives leave the pointer, with no VM."""
    cursor = cursor_xy
    for prim in px_prims:
        if prim.kind == "move_to":
            cursor = (prim.x, prim.y)
    return cursor


def golden_plan(
    task: templates.ConcreteTask,
    screen_wh: tuple[int, int],
    *,
    geometry: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], list[golden.GoldenStep]]:
    """The seeded cursor start and golden turns for one task.

    Two passes: the provisional start resolves the policy (which may read the
    cursor), then the published start additionally avoids the first move target
    so the rel arm always has a nonzero opening delta."""
    provisional = cursor_start_px(task.task_id, screen_wh)

    def _steps(start: tuple[int, int]) -> list[golden.GoldenStep]:
        return golden.golden_steps(
            task,
            golden.GoldenCtx(
                cursor_xy=start, screen_wh=screen_wh, geometry=dict(geometry or {}),
            ),
        )

    targets = golden.move_targets(_steps(provisional))
    avoid: tuple[tuple[int, int], ...] = ()
    if targets:
        first_x, first_y = targets[0]
        avoid = ((
            grammar.snap_point_px(first_x, screen_wh[0]),
            grammar.snap_point_px(first_y, screen_wh[1]),
        ),)
    start = cursor_start_px(task.task_id, screen_wh, avoid=avoid)
    return start, _steps(start)


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _guest_python(client: OSWorldClient, code: str, *args: str) -> str:
    """Run a python snippet in the guest and return its stdout."""
    result = client.run_command(["python3", "-c", code, *args])
    return str(result.get("output", ""))


def _guest_json(client: OSWorldClient, code: str, *args: str) -> Any:
    """Run a python snippet in the guest and parse its stdout as JSON."""
    return json.loads(_guest_python(client, code, *args))


_UPLOAD_CODE = (
    "import base64, pathlib, sys\n"
    "target = pathlib.Path(sys.argv[1])\n"
    "target.parent.mkdir(parents=True, exist_ok=True)\n"
    "target.write_bytes(base64.b64decode(sys.argv[2]))\n"
    "print(target.stat().st_size)\n"
)

_FILE_PREP_CODE = (
    "import base64, json, pathlib, sys\n"
    'spec = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))\n'
    "home = pathlib.Path.home()\n"
    'for name in spec["dirs"]:\n'
    "    (home / name).mkdir(parents=True, exist_ok=True)\n"
    'for item in spec["files"]:\n'
    '    path = home / item["path"]\n'
    "    path.parent.mkdir(parents=True, exist_ok=True)\n"
    '    path.write_text(item["content"], encoding="utf-8")\n'
    'print(json.dumps({"home": str(home), "dirs": spec["dirs"], '
    '"files": [item["path"] for item in spec["files"]]}))\n'
)

_PATH_PROBE_CODE = (
    "import json, os, pathlib, sys\n"
    "target = pathlib.Path.home() / sys.argv[1]\n"
    "content = None\n"
    "if target.is_file():\n"
    '    content = target.read_text(encoding="utf-8", errors="replace")\n'
    "print(json.dumps({\n"
    '    "path": str(target),\n'
    '    "exists": target.exists(),\n'
    '    "is_dir": target.is_dir(),\n'
    '    "is_file": target.is_file(),\n'
    '    "executable": target.is_file() and os.access(target, os.X_OK),\n'
    '    "content": content,\n'
    "}))\n"
)

_PROCESS_PROBE_CODE = (
    "import json, pathlib, sys\n"
    "name = sys.argv[1]\n"
    "hits = []\n"
    'for entry in pathlib.Path("/proc").iterdir():\n'
    "    if not entry.name.isdigit():\n"
    "        continue\n"
    "    try:\n"
    '        comm = (entry / "comm").read_text().strip()\n'
    '        raw = (entry / "cmdline").read_bytes().decode("utf-8", "replace")\n'
    "    except OSError:\n"
    "        continue\n"
    '    argv = [part for part in raw.split("\\x00") if part]\n'
    "    argv0 = pathlib.Path(argv[0]).name if argv else \"\"\n"
    "    if comm == name or comm == name[:15] or argv0 == name:\n"
    '        hits.append({"pid": int(entry.name), "comm": comm, "argv0": argv0})\n'
    "print(json.dumps(hits))\n"
)

_PTS_PROBE_CODE = (
    "import json, pathlib\n"
    'print(json.dumps(sorted(entry.name for entry in pathlib.Path("/dev/pts").iterdir() '
    "if entry.name.isdigit())))\n"
)

_TITLE_COMMAND = (
    "if command -v xdotool >/dev/null; then "
    "xdotool getactivewindow getwindowname; "
    "elif command -v wmctrl >/dev/null; then "
    "wmctrl -l; "
    "else exit 127; fi"
)


_READ_TEXT_CODE = (
    "import pathlib, sys\n"
    'print(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))\n'
)


def _upload_bytes(client: OSWorldClient, path: str, payload: bytes) -> None:
    """Write ``payload`` to ``path`` in the guest."""
    _guest_python(client, _UPLOAD_CODE, path, _b64(payload))


def _read_fixture_state(client: OSWorldClient) -> dict[str, Any]:
    """The fixture's published state JSON, read out of the guest."""
    return dict(_guest_json(client, _READ_TEXT_CODE, fixture.STATE_PATH))


def _which(client: OSWorldClient, names: tuple[str, ...]) -> str | None:
    """The first of ``names`` on the guest's PATH, or ``None``."""
    probe = " ".join(f"command -v {name} >/dev/null && echo {name};" for name in names)
    output = str(client.run_command(f"{probe} true", shell=True).get("output", ""))
    found = [line.strip() for line in output.splitlines() if line.strip() in names]
    return found[0] if found else None


def _active_title(client: OSWorldClient) -> str:
    """The active window title (xdotool, else every wmctrl title), or ""."""
    try:
        return str(client.run_command(
            ["bash", "-lc", _TITLE_COMMAND],
        ).get("output", "")).strip()
    except RuntimeError:
        return ""


def _wait_until(
    predicate: Any, *, timeout_s: float = READY_TIMEOUT_S, poll_s: float = READY_POLL_S,
) -> Any:
    """Poll ``predicate`` until it returns something truthy."""
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except (RuntimeError, ValueError, KeyError, OSError, json.JSONDecodeError) as error:
            last = f"{type(error).__name__}: {error}"
        time.sleep(poll_s)
    raise TimeoutError(f"guest condition not met after {timeout_s}s (last={last!r})")


def _wait_for_title(client: OSWorldClient, wanted: str, *, label: str) -> str:
    """Wait until the active window title contains ``wanted``; return it."""
    def _title() -> str | None:
        title = _active_title(client)
        return title if wanted in title else None

    try:
        return str(_wait_until(_title))
    except TimeoutError as error:
        raise RuntimeError(f"{label}: no window titled {wanted!r} ({error})") from error


def _prepare_guest_files(client: OSWorldClient, params: dict[str, Any]) -> dict[str, Any]:
    """Create a task's pre-existing guest files and directories under $HOME."""
    files = [dict(item) for item in params.get("setup_files", ())]
    dirs = [str(name) for name in params.get("setup_dirs", ())]
    workdir = params.get("workdir")
    if workdir and str(workdir) not in dirs:
        dirs.append(str(workdir))
    if not files and not dirs:
        return {"files": [], "dirs": []}
    payload = _b64(json.dumps({"files": files, "dirs": dirs}, sort_keys=True).encode())
    return dict(_guest_json(client, _FILE_PREP_CODE, payload))


def _open_terminal(client: OSWorldClient, params: dict[str, Any]) -> None:
    """Open the guest terminal exactly like freeroll's ``terminal`` setup."""
    client.execute(
        "import subprocess; "
        "subprocess.Popen(['bash', '-lc', "
        "\"(command -v gnome-terminal >/dev/null && gnome-terminal) || "
        "(command -v xfce4-terminal >/dev/null && xfce4-terminal) || "
        "(command -v xterm >/dev/null && xterm)\"]); "
        "time.sleep(2.0); "
        "pyautogui.hotkey('ctrl', 'l'); "
        "time.sleep(0.2)",
    )
    workdir = params.get("workdir")
    if workdir:
        client.execute(
            f"pyautogui.write({f'cd ~/{workdir}'!r}, interval=0); "
            "pyautogui.press('enter'); time.sleep(0.8); "
            "pyautogui.hotkey('ctrl', 'l'); time.sleep(0.2)",
        )


def _activate_window(client: OSWorldClient, title: str) -> bool:
    """Best-effort: ask the WM to raise and focus the window titled ``title``."""
    try:
        client.run_command(
            f"if command -v wmctrl >/dev/null; then wmctrl -a {title}; else exit 127; fi",
            shell=True,
        )
    except RuntimeError as error:
        _LOGGER.info("could not activate the %r window: %s", title, error)
        return False
    return True


def _list_windows(client: OSWorldClient) -> list[tuple[str, str, str]]:
    """``(window_id, desktop, title)`` for every window the WM manages."""
    try:
        output = str(client.run_command(
            "if command -v wmctrl >/dev/null; then wmctrl -l; fi", shell=True,
        ).get("output", ""))
    except RuntimeError as error:
        _LOGGER.info("could not list guest windows: %s", error)
        return []
    rows: list[tuple[str, str, str]] = []
    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            rows.append((parts[0], parts[1], parts[3].strip()))
    return rows


def _close_browser_popups(client: OSWorldClient, keep: str) -> list[str]:
    """Close browser windows other than the page one (Chrome's update nag).

    Such a popup is a separate top-level window drawn OVER the page, so left
    alone it can cover a seeded target and make a template nondeterministic."""
    closed: list[str] = []
    for window_id, desktop, title in _list_windows(client):
        if desktop == "-1" or keep in title:
            continue
        if not any(marker in title for marker in BROWSER_WINDOW_MARKERS):
            continue
        try:
            client.run_command(f"wmctrl -i -c {window_id}", shell=True)
        except RuntimeError as error:
            _LOGGER.info("could not close browser popup %r: %s", title, error)
            continue
        closed.append(title)
    return closed


def _maximize_active_window(client: OSWorldClient) -> None:
    """Best-effort maximize so app frames are geometry-stable."""
    try:
        client.run_command(
            "if command -v wmctrl >/dev/null; then "
            "wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz; fi",
            shell=True,
        )
    except RuntimeError as error:
        _LOGGER.info("could not maximize the active window: %s", error)


def _open_editor(client: OSWorldClient, task: templates.ConcreteTask) -> dict[str, Any]:
    """Open the task's file in the guest text editor, focused."""
    binary = _which(client, EDITOR_BINARIES)
    if binary is None:
        raise RuntimeError(f"guest has none of the editors {EDITOR_BINARIES}")
    if binary != EDITOR_BINARIES[0]:
        _LOGGER.warning("guest editor is %s, not %s", binary, EDITOR_BINARIES[0])
    filename = str(task.params["filename"])
    client.run_command(
        f'nohup env DISPLAY=:0 {binary} "$HOME/{filename}" '
        f">{GUEST_LOG_DIR}/shortgoal_editor.log 2>&1 &",
        shell=True,
    )
    title = _wait_for_title(client, filename, label=f"{task.task_id} editor")
    _maximize_active_window(client)
    time.sleep(APP_SETTLE_S)
    width, height = client.screen_size()
    client.execute(f"pyautogui.click(x={width // 2}, y={height // 2}, button='left')")
    time.sleep(APP_SETTLE_S)
    return {"editor_binary": binary, "editor_title": title}


def _probe_fixture_keyboard(client: OSWorldClient, task_id: str) -> dict[str, Any]:
    """Prove a synthesized keypress reaches the fixture before recording anything.

    Clicks land on an unfocused window but keys do not, and the fixture's commit
    key leaves no other trace, so a focus failure would otherwise surface as a
    silent ``committed: false`` at verification time. The probe key is a bare
    modifier: it bumps ``keys_seen`` and changes nothing else."""
    for attempt in range(1, KEY_PROBE_ATTEMPTS + 1):
        before = int(_read_fixture_state(client).get("keys_seen", 0))
        client.execute(
            f"pyautogui.keyDown({KEY_PROBE_NAME!r}); pyautogui.keyUp({KEY_PROBE_NAME!r})",
        )

        def _seen(before: int = before) -> dict[str, Any] | None:
            state = _read_fixture_state(client)
            return state if int(state.get("keys_seen", 0)) > before else None

        try:
            state = _wait_until(_seen, timeout_s=KEY_PROBE_TIMEOUT_S)
        except TimeoutError:
            _LOGGER.warning(
                "%s fixture saw no probe keypress (attempt %d); activating %r",
                task_id, attempt, fixture.FIXTURE_TITLE,
            )
            _activate_window(client, fixture.FIXTURE_TITLE)
            continue
        return {"key_probe_attempts": attempt, "keys_seen": int(state["keys_seen"])}
    raise RuntimeError(
        f"{task_id}: the fixture window never received a probe keypress after "
        f"{KEY_PROBE_ATTEMPTS} attempts; it does not hold the guest keyboard, so its "
        "commit key would be lost",
    )


def _launch_fixture(
    client: OSWorldClient, task: templates.ConcreteTask, screen_wh: tuple[int, int],
) -> dict[str, Any]:
    """Upload and launch the in-guest Tk fixture for this task's seeded spec."""
    spec = fixture.validate_spec(task.params["fixture_spec"])
    if tuple(spec["screen"]) != tuple(screen_wh):
        raise RuntimeError(f"fixture spec screen {spec['screen']!r} != VM screen {screen_wh!r}")
    source = Path(__file__).with_name("shortgoal_fixture.py")
    _upload_bytes(client, GUEST_FIXTURE_PATH, source.read_bytes())
    _upload_bytes(client, GUEST_SPEC_PATH, fixture.spec_to_json(spec).encode())
    client.run_command(
        f"rm -f {fixture.STATE_PATH}; "
        f"nohup env DISPLAY=:0 python3 {GUEST_FIXTURE_PATH} --spec {GUEST_SPEC_PATH} "
        f"--state {fixture.STATE_PATH} >{GUEST_LOG_DIR}/shortgoal_fixture.log 2>&1 &",
        shell=True,
    )

    def _ready() -> dict[str, Any] | None:
        state = _read_fixture_state(client)
        return state if state.get("ready") and state.get("kind") == spec["kind"] else None

    state = _wait_until(_ready)
    time.sleep(APP_SETTLE_S)
    wanted_window = [0, 0, int(screen_wh[0]), int(screen_wh[1])]
    window = list(state.get("window", []))
    if window != wanted_window:
        raise RuntimeError(
            f"{task.task_id}: the fixture window is {window} instead of {wanted_window}, so "
            "the spec's placed coordinates are not screen coordinates",
        )
    probe = _probe_fixture_keyboard(client, task.task_id)
    state = _read_fixture_state(client)
    offline = fixture.spec_widgets(spec)
    live = dict(state.get("widgets", {}))
    drift = {
        label: {"spec": box, "live": live.get(label)}
        for label, box in offline.items()
        if live.get(label) != box
    }
    if drift:
        _LOGGER.warning("%s fixture widget drift: %s", task.task_id, json.dumps(drift))
    for label, boxes in drift.items():
        live_box = boxes["live"]
        if live_box is None or any(
            abs(int(a) - int(b)) > WIDGET_TOLERANCE_PX
            for a, b in zip(live_box, boxes["spec"], strict=True)
        ):
            raise RuntimeError(f"fixture widget {label!r} is not where the spec places it: {boxes}")
    return {"widgets": live, "widget_drift": drift, "window": window, **probe}


def _launch_page(
    client: OSWorldClient, task: templates.ConcreteTask, screen_wh: tuple[int, int],
) -> dict[str, Any]:
    """Write the task's local page and open it in a kiosk browser.

    The gate is the page's OWN post-paint title (``fixture.PAGE_READY_TITLE``),
    not the static one: Chrome publishes ``<title>`` as soon as the head is
    parsed, and a browser that is still showing a blank white window is
    perfectly stable under screenshot settling — a click dispatched into that
    window hits no element and runs no handler.

    ``--disable-background-networking`` short-circuits the upgrade detector, so
    the "Can't update Chrome" bubble that overlays the page never appears; any
    popup that still does is closed before the first frame is recorded."""
    binary = _which(client, BROWSER_BINARIES)
    if binary is None:
        raise RuntimeError(f"guest has none of the browsers {BROWSER_BINARIES}")
    if tuple(screen_wh) != fixture.PAGE_WH:
        raise RuntimeError(f"browser pages need a {fixture.PAGE_WH} screen, got {screen_wh!r}")
    page = fixture.make_html_page(str(task.params["page_kind"]), task.params)
    _upload_bytes(client, GUEST_PAGE_PATH, page.encode())
    width, height = screen_wh
    client.run_command(
        f"rm -rf {GUEST_PROFILE_DIR}; "
        f"nohup env DISPLAY=:0 {binary} --user-data-dir={GUEST_PROFILE_DIR} "
        "--no-first-run --no-default-browser-check --disable-session-crashed-bubble "
        "--disable-infobars --disable-translate --disable-background-networking "
        "--disable-component-update --disable-default-apps --disable-sync "
        "--force-device-scale-factor=1 --kiosk "
        f"--window-position=0,0 --window-size={width},{height} file://{GUEST_PAGE_PATH} "
        f">{GUEST_LOG_DIR}/shortgoal_browser.log 2>&1 &",
        shell=True,
    )
    title = _wait_for_title(client, PAGE_READY_TITLE, label=f"{task.task_id} page")
    popups = _close_browser_popups(client, PAGE_READY_TITLE)
    if popups:
        _LOGGER.warning("%s closed browser popups: %s", task.task_id, popups)
        title = _wait_for_title(client, PAGE_READY_TITLE, label=f"{task.task_id} page")
    time.sleep(APP_SETTLE_S)
    return {
        "browser_binary": binary,
        "page_title": title,
        "page_ready_title": PAGE_READY_TITLE,
        "popups_closed": popups,
    }


def prepare_task(
    client: OSWorldClient, task: templates.ConcreteTask, screen_wh: tuple[int, int],
) -> dict[str, Any]:
    """Put a fresh VM into ``task``'s start state; returns setup provenance."""
    state: dict[str, Any] = {"setup_id": task.setup_id}
    state["guest_files"] = _prepare_guest_files(client, task.params)
    state["title_tool"] = _which(client, TITLE_TOOLS)
    editor = task.params.get("editor")
    if editor and _which(client, (str(editor),)) is None:
        raise RuntimeError(f"{task.task_id} types {editor!r} but the guest has no such binary")
    if task.setup_id == templates.SETUP_TERMINAL:
        _open_terminal(client, task.params)
    elif task.setup_id == templates.SETUP_EDITOR:
        state.update(_open_editor(client, task))
    elif task.setup_id == templates.SETUP_FIXTURE:
        state.update(_launch_fixture(client, task, screen_wh))
    elif task.setup_id == templates.SETUP_PAGE:
        state.update(_launch_page(client, task, screen_wh))
    else:
        raise ValueError(f"unknown setup id: {task.setup_id!r}")
    expect = task.params.get("expect", {})
    if "process_absent" in expect:
        name = str(expect["process_absent"])
        state["process_before"] = _guest_json(client, _PROCESS_PROBE_CODE, name)
        if not state["process_before"]:
            raise RuntimeError(f"{task.task_id} expects {name!r} to be running before the goal")
    if "min_terminal_tabs" in expect:
        state["pts_before"] = _guest_json(client, _PTS_PROBE_CODE)
    return state


_PATH_EXPECT_KEYS = {
    "guest_path_exists": ("exists",),
    "guest_path_absent": ("exists",),
    "guest_dir_exists": ("is_dir",),
    "guest_file_content": ("content", "content_stripped"),
    "guest_file_executable": ("executable",),
}



def _verify_path(
    client: OSWorldClient, task: templates.ConcreteTask, setup_state: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify a guest path's existence, kind, mode or contents."""
    expect = task.params["expect"]
    wanted = _PATH_EXPECT_KEYS[task.verifier_id]
    if not any(key in expect for key in wanted):
        raise ValueError(f"{task.verifier_id} needs one of {wanted} in {expect!r}")
    probe = dict(_guest_json(client, _PATH_PROBE_CODE, str(expect["path"])))
    checks: dict[str, bool] = {}
    for key, value in expect.items():
        if key == "path":
            continue
        if key == "content_stripped":
            checks[key] = str(probe.get("content") or "").strip() == value
        else:
            checks[key] = probe.get(key) == value
    return all(checks.values()), {"expect": expect, "probe": probe, "checks": checks}


def _verify_window_title(
    client: OSWorldClient, task: templates.ConcreteTask, setup_state: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify the active window title contains the expected text."""
    wanted = str(task.params["expect"]["window_title_contains"])
    title = _active_title(client)
    return wanted in title, {
        "wanted": wanted, "title": title, "title_tool": setup_state.get("title_tool"),
    }


def _verify_browser_title(
    client: OSWorldClient, task: templates.ConcreteTask, setup_state: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify the page's JS-mutated title reached the browser window."""
    wanted = str(task.params["expect"]["title"])
    title = _active_title(client)
    return wanted in title, {
        "wanted": wanted, "title": title, "title_tool": setup_state.get("title_tool"),
        "browser_binary": setup_state.get("browser_binary"),
    }


def _verify_process_absent(
    client: OSWorldClient, task: templates.ConcreteTask, setup_state: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify a process that was running before the goal is now gone."""
    name = str(task.params["expect"]["process_absent"])
    before = setup_state.get("process_before", [])
    after = _guest_json(client, _PROCESS_PROBE_CODE, name)
    return bool(before) and not after, {
        "name": name, "before": before, "after": after,
        "editor_binary": setup_state.get("editor_binary"),
    }


def _verify_terminal_tabs(
    client: OSWorldClient, task: templates.ConcreteTask, setup_state: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify the goal opened new terminal tabs, counted as new guest ptys."""
    wanted = int(task.params["expect"]["min_terminal_tabs"])
    before = list(setup_state.get("pts_before", []))
    after = _guest_json(client, _PTS_PROBE_CODE)
    opened = len(after) - len(before)
    return opened >= wanted - 1, {
        "min_terminal_tabs": wanted, "pts_before": before, "pts_after": after, "opened": opened,
    }


def _verify_fixture_state(
    client: OSWorldClient, task: templates.ConcreteTask, setup_state: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify the fixture's published state, misses included."""
    expect = task.params["expect"]
    state = _read_fixture_state(client)
    checks = {key: state.get(key) == value for key, value in expect.items()}
    checks["misses"] = state.get("misses") == 0
    return all(checks.values()), {"expect": expect, "state": state, "checks": checks}


_VERIFIERS = {
    "guest_path_exists": _verify_path,
    "guest_path_absent": _verify_path,
    "guest_dir_exists": _verify_path,
    "guest_file_content": _verify_path,
    "guest_file_executable": _verify_path,
    "guest_window_title": _verify_window_title,
    "guest_process_absent": _verify_process_absent,
    "guest_terminal_tabs": _verify_terminal_tabs,
    "fixture_state": _verify_fixture_state,
    "browser_title": _verify_browser_title,
}


def verify_task(
    client: OSWorldClient,
    task: templates.ConcreteTask,
    setup_state: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Run ``task``'s verifier, retrying until it passes or ``timeout_s``."""
    checker = _VERIFIERS.get(task.verifier_id)
    if checker is None:
        raise KeyError(f"unknown verifier id: {task.verifier_id!r}")
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        try:
            passed, detail = checker(client, task, setup_state)
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            passed, detail = False, {"error": f"{type(error).__name__}: {error}"}
        if passed or time.time() >= deadline:
            return {"kind": task.verifier_id, "passed": bool(passed), "detail": detail}
        time.sleep(READY_POLL_S)


def _screen_size(client: OSWorldClient) -> tuple[int, int]:
    """The VM screen size, required to be the catalog's resolution."""
    screen = client.screen_size()
    if tuple(screen) != templates.SCREEN_WH:
        raise RuntimeError(
            f"VM screen {tuple(screen)} != the catalog's {templates.SCREEN_WH}; "
            "every fixture and page target is computed for that resolution",
        )
    return screen


def _place_cursor(client: OSWorldClient, xy: tuple[int, int], *, label: str) -> tuple[int, int]:
    """Teleport the pointer to ``xy`` and report where it actually landed."""
    client.execute(f"pyautogui.moveTo({xy[0]}, {xy[1]})")
    actual = client.cursor_position()
    if tuple(actual) != tuple(xy):
        _LOGGER.warning("%s cursor start %s landed at %s", label, list(xy), list(actual))
    return actual


def record_task(
    client: OSWorldClient,
    task: templates.ConcreteTask,
    task_dir: Path,
    *,
    settle: Settle,
    verify_timeout_s: float,
) -> dict[str, Any]:
    """Record one golden trajectory in a freshly booted VM."""
    started = time.time()
    frames_dir = task_dir / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    screen_wh = _screen_size(client)
    setup_state = prepare_task(client, task, screen_wh)
    start, steps = golden_plan(
        task, screen_wh, geometry={"widgets": setup_state.get("widgets", {})},
    )
    cursor = tuple(_place_cursor(client, start, label=task.task_id))
    rows: list[dict[str, Any]] = []
    zero_deltas = 0
    for index, step in enumerate(steps):
        frame_name = f"step_{index:03d}.png"
        settle.shot(client).save(frames_dir / frame_name)
        plan = convert_step(step, cursor, screen_wh)
        zero_deltas += plan.zero_deltas
        _LOGGER.info("%s step %d/%d: %s", task.task_id, index + 1, len(steps), plan.abs_line)
        result = client.dispatch_ordered_action(plan.px_action)
        if tuple(result.cursor_before) != cursor:
            _LOGGER.warning(
                "%s step %d cursor moved outside the trajectory: %s != %s",
                task.task_id, index, list(result.cursor_before), list(cursor),
            )
        rows.append({
            "primitives_px": serialize_primitives(plan.px_action.primitives),
            "primitives_grid": serialize_primitives(plan.grid_action.primitives),
            "cursor_before": list(result.cursor_before),
            "cursor_after": list(result.cursor_after),
            "frame": frame_name,
        })
        cursor = tuple(result.cursor_after)
    final_name = f"step_{len(steps):03d}.png"
    settle.shot(client).save(frames_dir / final_name)
    verifier = verify_task(client, task, setup_state, timeout_s=verify_timeout_s)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "template_id": task.template_id,
        "seed": task.seed,
        "category": task.category,
        "tier_b": task.tier_b,
        "single_action": task.single_action,
        "setup_id": task.setup_id,
        "policy_id": task.policy_id,
        "params": task.params,
        "instruction": task.instruction,
        "screen_size": list(screen_wh),
        "cursor_start": list(start),
        "steps": rows,
        "n_steps": len(rows),
        "n_frames": len(rows) + 1,
        "zero_delta_moves": zero_deltas,
        "setup": setup_state,
        "verifier": verifier,
        "elapsed_s": time.time() - started,
    }


def replay_recording(
    client: OSWorldClient,
    recording: dict[str, Any],
    task_dir: Path,
    *,
    settle: Settle,
    verify_timeout_s: float,
) -> dict[str, Any]:
    """Re-dispatch a recording's own primitives and re-run its verifier."""
    started = time.time()
    task = templates.concrete_task(str(recording["template_id"]), int(recording["seed"]))
    if task.task_id != recording["task_id"]:
        raise ValueError(f"{recording['task_id']!r} does not resolve to {task.task_id!r}")
    if task.instruction != recording["instruction"]:
        raise ValueError(f"{task.task_id} instruction drifted from the recording")
    if json.dumps(task.params, sort_keys=True) != json.dumps(recording["params"], sort_keys=True):
        raise ValueError(f"{task.task_id} seeded param draw drifted from the recording")
    frames_dir = task_dir / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    screen_wh = _screen_size(client)
    if list(screen_wh) != list(recording["screen_size"]):
        raise RuntimeError(f"VM screen {list(screen_wh)} != recorded {recording['screen_size']}")
    setup_state = prepare_task(client, task, screen_wh)
    steps = list(recording["steps"])
    if not steps:
        raise ValueError(f"{task.task_id} recording has no steps")
    start = tuple(int(value) for value in steps[0]["cursor_before"])
    _place_cursor(client, start, label=f"{task.task_id} replay")
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        frame_name = f"step_{index:03d}.png"
        settle.shot(client).save(frames_dir / frame_name)
        prims = deserialize_primitives(step["primitives_px"])
        result = client.dispatch_ordered_action(
            OrderedAction(primitives=prims, no_op=not prims),
        )
        rows.append({
            "frame": frame_name,
            "cursor_before": list(result.cursor_before),
            "cursor_after": list(result.cursor_after),
            "recorded_cursor_before": list(step["cursor_before"]),
            "recorded_cursor_after": list(step["cursor_after"]),
            "cursor_drift_px": [
                int(result.cursor_after[axis]) - int(step["cursor_after"][axis])
                for axis in (0, 1)
            ],
        })
    settle.shot(client).save(frames_dir / f"step_{len(steps):03d}.png")
    verifier = verify_task(client, task, setup_state, timeout_s=verify_timeout_s)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "replay",
        "task_id": task.task_id,
        "template_id": task.template_id,
        "seed": task.seed,
        "category": task.category,
        "screen_size": list(screen_wh),
        "steps": rows,
        "n_steps": len(rows),
        "n_frames": len(rows) + 1,
        "max_cursor_drift_px": max(
            (max(abs(value) for value in row["cursor_drift_px"]) for row in rows), default=0,
        ),
        "setup": setup_state,
        "verifier": verifier,
        "passed": bool(verifier["passed"]),
        "elapsed_s": time.time() - started,
    }


def dry_run_task(task: templates.ConcreteTask) -> dict[str, Any]:
    """Walk one task's policy and step conversion with no VM."""
    screen_wh = templates.SCREEN_WH
    start, steps = golden_plan(task, screen_wh)
    cursor = start
    rows: list[dict[str, Any]] = []
    n_primitives = 0
    zero_deltas = 0
    for index, step in enumerate(steps):
        plan = convert_step(step, cursor, screen_wh)
        n_primitives += len(plan.px_action.primitives)
        zero_deltas += plan.zero_deltas
        rows.append({
            "index": index,
            "abs_line": plan.abs_line,
            "rel_line": plan.rel_line,
            "cursor_before": list(cursor),
            "primitives_px": serialize_primitives(plan.px_action.primitives),
        })
        cursor = advance_cursor(plan.px_action.primitives, cursor)
    return {
        "task_id": task.task_id,
        "template_id": task.template_id,
        "seed": task.seed,
        "category": task.category,
        "tier_b": task.tier_b,
        "single_action": task.single_action,
        "setup_id": task.setup_id,
        "policy_id": task.policy_id,
        "verifier_id": task.verifier_id,
        "instruction": task.instruction,
        "cursor_start": list(start),
        "n_steps": len(steps),
        "n_frames": len(steps) + 1,
        "n_primitives": n_primitives,
        "zero_delta_moves": zero_deltas,
        "steps": rows,
    }


def dry_run(tasks: list[templates.ConcreteTask]) -> dict[str, Any]:
    """Convert every selected task offline; one printed line per task."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for task in tasks:
        try:
            row = dry_run_task(task)
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"task_id": task.task_id, "error": f"{type(error).__name__}: {error}"})
            print(f"{task.task_id:32s} FAILED {type(error).__name__}: {error}")
            continue
        rows.append(row)
        print(
            f"{row['task_id']:32s} {row['category']:8s} "
            f"steps={row['n_steps']} frames={row['n_frames']} prims={row['n_primitives']} "
            f"verifier={row['verifier_id']:22s} first={row['steps'][0]['abs_line'][:52]!r}",
        )
    if not rows:
        raise RuntimeError("no task converted successfully")
    manifest = templates.build_split_manifest()
    by_category: dict[str, int] = {}
    by_steps: dict[str, int] = {}
    for row in rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
        by_steps[str(row["n_steps"])] = by_steps.get(str(row["n_steps"]), 0) + 1
    single = [row for row in rows if row["single_action"]]
    zero = [row["task_id"] for row in rows if row["zero_delta_moves"]]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run",
        "n_tasks": len(rows),
        "n_failed": len(failures),
        "failures": failures,
        "n_steps_total": sum(row["n_steps"] for row in rows),
        "n_frames_total": sum(row["n_frames"] for row in rows),
        "n_primitives_total": sum(row["n_primitives"] for row in rows),
        "max_steps": max(row["n_steps"] for row in rows),
        "steps_histogram": dict(sorted(by_steps.items(), key=lambda item: int(item[0]))),
        "by_category": dict(sorted(by_category.items())),
        "single_action_tasks": len(single),
        "single_action_fraction": len(single) / len(rows),
        "zero_delta_move_tasks": zero,
        "split_counts": manifest["counts"],
        "tasks": rows,
    }
    print(
        f"tasks={summary['n_tasks']} failed={summary['n_failed']} "
        f"steps={summary['n_steps_total']} frames={summary['n_frames_total']} "
        f"primitives={summary['n_primitives_total']} max_steps={summary['max_steps']}",
    )
    print(f"steps_histogram={summary['steps_histogram']} by_category={summary['by_category']}")
    print(
        f"single_action={summary['single_action_tasks']} "
        f"({summary['single_action_fraction']:.2%}) splits={summary['split_counts']} "
        f"zero_delta_move_tasks={zero}",
    )
    return summary


def _boot_vm(
    *, qemu_bin: str, qcow2: str, vm_port: int, vnc_port: int, log_path: Path,
    detach: bool = False,
) -> subprocess.Popen:
    """Boot the OSWorld qcow2 via native qemu+KVM with a throwaway snapshot.

    Mirrors freeroll's boot: ``snapshot=on`` starts every task from the same
    clean disk image and discards its writes, so recordings cannot contaminate
    each other or the shared image.

    ``detach`` puts the VM in its own session with no stdin, so it outlives the
    caller — the one thing ``shortgoal_agent_record`` needs and this recorder
    (which always tears its VM down in a ``finally``) must not have."""
    return subprocess.Popen(
        [qemu_bin,
         "-enable-kvm", "-cpu", "host", "-smp", "4", "-m", "4G",
         "-machine", "type=q35,accel=kvm",
         "-drive", f"file={qcow2},if=virtio,format=qcow2,snapshot=on",
         "-netdev", f"user,id=net0,hostfwd=tcp::{vm_port}-:5000,hostfwd=tcp::{vnc_port}-:5900",
         "-device", "virtio-net-pci,netdev=net0",
         "-display", "none", "-nographic"],
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL if detach else None,
        start_new_session=detach,
    )


def _terminate(proc: subprocess.Popen, *, label: str) -> None:
    """Stop a child process and wait for it to release its ports."""
    if proc.poll() is not None:
        return
    _LOGGER.info("terminating %s (pid=%d)", label, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _in_fresh_vm(
    action: Any,
    *,
    label: str,
    log_path: Path,
    qemu_bin: str,
    qcow2: str,
    vm_port: int,
    vnc_port: int,
    ready_timeout_s: float,
) -> Any:
    """Boot a VM, run ``action(client)`` against it, always tear it down."""
    _LOGGER.info("booting VM for %s (port %d)", label, vm_port)
    vm_proc = _boot_vm(
        qemu_bin=qemu_bin, qcow2=qcow2, vm_port=vm_port, vnc_port=vnc_port, log_path=log_path,
    )
    _PROCS.append(vm_proc)
    try:
        _wait_for(
            f"http://localhost:{vm_port}/screenshot",
            proc=vm_proc,
            poll_s=5,
            max_polls=max(1, int(ready_timeout_s // 5)),
            label=f"VM {label}",
        )
        client = OSWorldClient(f"http://localhost:{vm_port}")
        client.wait_ready(timeout_s=ready_timeout_s)
        return action(client)
    finally:
        _terminate(vm_proc, label=f"VM {label}")
        if vm_proc in _PROCS:
            _PROCS.remove(vm_proc)


def _csv(raw: str) -> list[str]:
    """A comma-separated CLI list, blanks dropped."""
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def select_tasks(
    *, task_ids: list[str], template_ids: list[str], seeds: list[str],
) -> list[templates.ConcreteTask]:
    """The selected subset of the 150 catalog tasks, in catalog order."""
    tasks = templates.concrete_tasks()
    if template_ids:
        unknown = sorted(set(template_ids) - set(templates.TEMPLATES_BY_ID))
        if unknown:
            raise ValueError(f"unknown --template_ids: {unknown}")
        tasks = [task for task in tasks if task.template_id in set(template_ids)]
    if seeds:
        wanted = {int(seed) for seed in seeds}
        tasks = [task for task in tasks if task.seed in wanted]
    if task_ids:
        unknown = sorted(set(task_ids) - {task.task_id for task in templates.concrete_tasks()})
        if unknown:
            raise ValueError(f"unknown --task_ids: {unknown}")
        tasks = [task for task in tasks if task.task_id in set(task_ids)]
    if not tasks:
        raise ValueError("selection matched no tasks")
    return tasks


def resolve_recordings(source: Path, tasks: list[templates.ConcreteTask]) -> list[Path]:
    """Every ``recording.json`` ``--replay_from`` points at, task-filtered."""
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"--replay_from {source} is neither a file nor a directory")
    wanted = {task.task_id for task in tasks}
    paths = sorted(
        path for path in source.glob(f"*/{RECORDING_NAME}") if path.parent.name in wanted
    )
    if not paths:
        raise FileNotFoundError(f"no {RECORDING_NAME} under {source} for the selected tasks")
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact, parents included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _register_cleanup() -> None:
    """Terminate any surviving VM on exit or signal."""
    def _cleanup() -> None:
        for proc in list(_PROCS):
            _terminate(proc, label=f"leftover pid {proc.pid}")

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))


def run_record(
    tasks: list[templates.ConcreteTask],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    settle: Settle,
) -> int:
    """Record every selected task; nonzero unless all verifiers pass."""
    vm_port, vnc_port = _ports(args.vm_port)
    started = time.time()
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_dir = output_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / RECORDING_NAME).unlink(missing_ok=True)
        (task_dir / FAILURE_NAME).unlink(missing_ok=True)
        _LOGGER.info("[%d/%d] recording %s", index + 1, len(tasks), task.task_id)
        reason: str | None = None
        recording: dict[str, Any] | None = None
        try:
            recording = _in_fresh_vm(
                lambda client, task=task, task_dir=task_dir: record_task(
                    client, task, task_dir, settle=settle,
                    verify_timeout_s=args.verify_timeout_s,
                ),
                label=task.task_id,
                log_path=task_dir / "qemu.log",
                qemu_bin=args.qemu_bin,
                qcow2=args.qcow2,
                vm_port=vm_port,
                vnc_port=vnc_port,
                ready_timeout_s=args.vm_ready_timeout_s,
            )
            if not recording["verifier"]["passed"]:
                reason = f"verifier {recording['verifier']['kind']} failed"
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            _LOGGER.exception("recording failed: %s", task.task_id)
            recording = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task.task_id,
                "template_id": task.template_id,
                "seed": task.seed,
                "traceback": traceback.format_exc(),
            }
        passed = reason is None
        _write_json(task_dir / (RECORDING_NAME if passed else FAILURE_NAME), {
            **recording, **({} if passed else {"rejected_reason": reason}),
        })
        rows.append({
            "task_id": task.task_id,
            "template_id": task.template_id,
            "seed": task.seed,
            "category": task.category,
            "passed": passed,
            "reason": reason,
            "n_steps": recording.get("n_steps"),
            "n_frames": recording.get("n_frames"),
            "verifier_kind": recording.get("verifier", {}).get("kind", task.verifier_id),
            "elapsed_s": recording.get("elapsed_s"),
        })
        _write_json(output_dir / "summary.json", {
            "schema_version": SCHEMA_VERSION,
            "mode": "record",
            "n_tasks": len(tasks),
            "n_attempted": len(rows),
            "n_recorded": sum(bool(row["passed"]) for row in rows),
            "n_rejected": sum(not row["passed"] for row in rows),
            "completed": len(rows) == len(tasks),
            "passed": len(rows) == len(tasks) and all(row["passed"] for row in rows),
            "settle": asdict(settle),
            "elapsed_s": time.time() - started,
            "tasks": rows,
        })
        _LOGGER.info(
            "[%d/%d] %s %s", index + 1, len(tasks), task.task_id,
            "recorded" if passed else f"REJECTED ({reason})",
        )
    ok = all(row["passed"] for row in rows)
    _LOGGER.info(
        "done: %d/%d recorded under %s", sum(bool(row["passed"]) for row in rows), len(rows),
        output_dir,
    )
    return 0 if ok else 1


def run_replay(
    tasks: list[templates.ConcreteTask],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    settle: Settle,
) -> int:
    """Rung 0: re-dispatch recordings on fresh VMs and re-run the verifiers."""
    paths = resolve_recordings(Path(args.replay_from), tasks)
    vm_port, vnc_port = _ports(args.vm_port)
    started = time.time()
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        recording = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(recording["task_id"])
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("[%d/%d] replaying %s from %s", index + 1, len(paths), task_id, path)
        try:
            result = _in_fresh_vm(
                lambda client, recording=recording, task_dir=task_dir: replay_recording(
                    client, recording, task_dir, settle=settle,
                    verify_timeout_s=args.verify_timeout_s,
                ),
                label=f"{task_id} replay",
                log_path=task_dir / "qemu.log",
                qemu_bin=args.qemu_bin,
                qcow2=args.qcow2,
                vm_port=vm_port,
                vnc_port=vnc_port,
                ready_timeout_s=args.vm_ready_timeout_s,
            )
        except Exception as error:
            _LOGGER.exception("replay failed: %s", task_id)
            result = {
                "schema_version": SCHEMA_VERSION,
                "mode": "replay",
                "task_id": task_id,
                "passed": False,
                "reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        _write_json(task_dir / REPLAY_NAME, {**result, "source": str(path)})
        rows.append({
            "task_id": task_id,
            "source": str(path),
            "passed": bool(result.get("passed")),
            "reason": result.get("reason"),
            "max_cursor_drift_px": result.get("max_cursor_drift_px"),
            "verifier_kind": result.get("verifier", {}).get("kind"),
        })
        _write_json(output_dir / "replay_summary.json", {
            "schema_version": SCHEMA_VERSION,
            "mode": "replay",
            "n_recordings": len(paths),
            "n_attempted": len(rows),
            "n_passed": sum(bool(row["passed"]) for row in rows),
            "completed": len(rows) == len(paths),
            "passed": len(rows) == len(paths) and all(row["passed"] for row in rows),
            "settle": asdict(settle),
            "elapsed_s": time.time() - started,
            "tasks": rows,
        })
    ok = all(row["passed"] for row in rows)
    _LOGGER.info(
        "done: %d/%d replays passed under %s", sum(bool(row["passed"]) for row in rows), len(rows),
        output_dir,
    )
    return 0 if ok else 1


def _ports(vm_port: int) -> tuple[int, int]:
    """VM/VNC ports, offset from ``SLURM_JOB_ID`` exactly like freeroll."""
    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
    return (vm_port + job_mod if vm_port == 5000 else vm_port), 5900 + job_mod


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(
        description="one-time recorder for the short-goal golden trajectories",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--task_ids", default="",
        help="comma-separated task ids (template__sNN) to record; default: all 150",
    )
    parser.add_argument(
        "--template_ids", default="", help="comma-separated template ids to record",
    )
    parser.add_argument("--seeds", default="", help="comma-separated seeds to record")
    parser.add_argument(
        "--replay_from", default="",
        help="a recording.json, or a recordings root, to re-dispatch and re-verify (rung 0)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="no VM: walk templates, policies and the golden-step conversion only",
    )
    parser.add_argument(
        "--settle_s", type=float, default=0.3,
        help="fixed delay before each screenshot, giving the UI time to repaint",
    )
    parser.add_argument(
        "--settle_stable_timeout_s", type=float, default=2.0,
        help="if >0, poll the framebuffer until two frames are identical or this elapses",
    )
    parser.add_argument(
        "--settle_poll_s", type=float, default=0.1,
        help="poll interval for --settle_stable_timeout_s",
    )
    parser.add_argument(
        "--verify_timeout_s", type=float, default=12.0,
        help="how long a verifier may retry before the recording is rejected",
    )
    parser.add_argument("--vm_ready_timeout_s", type=float, default=300.0)
    parser.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    parser.add_argument("--qemu_bin", default=_DEFAULT_QEMU_BIN)
    parser.add_argument("--vm_port", type=int, default=5000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(output_dir / "shortgoal_record.log"))
    settle = Settle(
        delay_s=args.settle_s,
        stable_timeout_s=args.settle_stable_timeout_s,
        poll_s=args.settle_poll_s,
    )
    try:
        tasks = select_tasks(
            task_ids=_csv(args.task_ids),
            template_ids=_csv(args.template_ids),
            seeds=_csv(args.seeds),
        )
    except ValueError as error:
        parser.error(str(error))
    _LOGGER.info(
        "selected %d/%d tasks; output=%s", len(tasks), templates.N_TEMPLATES * templates.N_SEEDS,
        output_dir,
    )

    if args.dry_run:
        if args.replay_from:
            parser.error("--dry_run and --replay_from are mutually exclusive")
        summary = dry_run(tasks)
        _write_json(output_dir / "dry_run.json", summary)
        return 0 if not summary["n_failed"] else 1

    _register_cleanup()
    if args.replay_from:
        return run_replay(tasks, output_dir, args, settle=settle)
    return run_record(tasks, output_dir, args, settle=settle)


if __name__ == "__main__":
    raise SystemExit(main())
