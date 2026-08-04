"""Deterministic in-guest surfaces for the short-goal golden ladder.

The recorder uploads this single stdlib-only file into the OSWorld guest and
launches it with a seeded JSON spec drawn by ``shortgoal_templates``, so every
widget position is known offline to the pixel. The app renders the widgets at
EXACT screen pixels on a WM-managed fullscreen window (client origin 0,0 and
client size == the spec's screen, both published as ``window`` and hard-checked
by the recorder, so placed coordinates ARE screen coordinates) and atomically
rewrites ``/tmp/shortgoal_state.json`` after every interaction: clicked labels
in order, double/right-click records, slider value, wheel accumulation and the
exact widget bounding boxes the verifiers and the builder's bbox check read back.

The window is deliberately NOT override-redirect: an unmanaged window is
invisible to the window manager, which therefore never gives it the X input
focus, so synthesized clicks land (pointer events go by position) while every
synthesized KEY event is delivered to whatever the WM does consider focused —
the commit key silently never arrives. A managed fullscreen window takes focus
like every other guest app, ``_hold_focus`` re-asserts it every
``_FOCUS_POLL_MS``, ``keys_seen`` lets the recorder PROVE a keypress arrives
before it records anything, and the WM can be told to activate ``FIXTURE_TITLE``
if it does not.

Every handler repaints through ``_repaint`` and flushes Tk's idle queue before it
publishes state, and the bursty wheel path coalesces only the FILE write
(``_write_soon``): a multi-notch scroll arrives as one X event per notch, so
writing to the copy-on-write guest disk inside each handler used to leave the
counter label several notches behind the state, and a settled screenshot could
capture a frame that contradicted the action it was taken after.

All spec/state logic is pure (``validate_spec``, ``spec_widgets``, ``hit_test``,
``record_click``, ``counter_text``, ``apply_*``) and unit-tested headless; the Tk
layer only turns events into those calls. The same file also generates the three local
``file://`` pages used by the browser templates, at fixed pixels for a
1920x1080 kiosk window where page pixels equal screen pixels. Those pages
publish ``PAGE_READY_TITLE`` as their document title from a post-load double
``requestAnimationFrame`` — i.e. only once the page has produced a frame and its
script is live — because the static ``<title>`` is up as soon as the head is
parsed and a still-blank page is perfectly stable under screenshot settling.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from html import escape
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
except Exception:
    tk = None

STATE_PATH = "/tmp/shortgoal_state.json"
COMMIT_KEY = "Return"
FIXTURE_TITLE = "shortgoal-fixture"
DOUBLE_CLICK_S = 1.5
SPEC_KINDS = ("buttons", "colors", "two_buttons", "slider", "scroll_counter", "scroll_list")
PAGE_KINDS = ("link_grid", "input", "below_fold_button")
PAGE_WH = (1920, 1080)
PAGE_READY_TITLE = "shortgoal-page-ready"

_HEX_COLOR_RE = re.compile(r"^#[0-9a-f]{6}$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{6,32}$")

_BG = "#0e1116"
_COMMITTED_BG = "#16351f"
_FOCUS_POLL_MS = 200
_WRITE_COALESCE_MS = 30
_TILE_BG = "#e8edf3"
_TILE_FG = "#131a22"
_PANE_BG = "#f7f9fc"
_TRACK_BG = "#c8d2de"
_ACCENT = "#3584e4"


def _int_pair(value: Any, what: str) -> tuple[int, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(f"{what} must be a pair of ints, got {value!r}")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise ValueError(f"{what} must be a pair of ints, got {value!r}")
    return int(value[0]), int(value[1])


def _bbox(value: Any, what: str) -> tuple[int, int, int, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 4):
        raise ValueError(f"{what} must be [x0,y0,x1,y1], got {value!r}")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise ValueError(f"{what} must be four ints, got {value!r}")
    x0, y0, x1, y1 = (int(v) for v in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{what} must be a nonempty box, got {value!r}")
    return x0, y0, x1, y1


def _in_screen(bbox: tuple[int, int, int, int], screen: tuple[int, int], what: str) -> None:
    x0, y0, x1, y1 = bbox
    if x0 < 0 or y0 < 0 or x1 > screen[0] or y1 > screen[1]:
        raise ValueError(f"{what} {list(bbox)} leaves the {screen[0]}x{screen[1]} screen")


def widget_bbox(center: Any, size: Any) -> list[int]:
    """Screen-pixel ``[x0,y0,x1,y1]`` of a ``size`` widget centred on ``center``."""
    cx, cy = _int_pair(center, "widget center")
    w, h = _int_pair(size, "widget size")
    if w <= 0 or h <= 0:
        raise ValueError(f"widget size must be positive, got {size!r}")
    x0, y0 = cx - w // 2, cy - h // 2
    return [x0, y0, x0 + w, y0 + h]


def bbox_center(bbox: Any) -> list[int]:
    """The centre pixel of a bbox."""
    x0, y0, x1, y1 = _bbox(bbox, "bbox")
    return [(x0 + x1) // 2, (y0 + y1) // 2]


def bbox_contains(bbox: Any, xy: Any) -> bool:
    """Whether pixel ``xy`` lies inside ``bbox`` (right/bottom edge exclusive)."""
    x0, y0, x1, y1 = _bbox(bbox, "bbox")
    x, y = _int_pair(xy, "point")
    return x0 <= x < x1 and y0 <= y < y1


def slider_tick_x(slider: Any, value: int) -> int:
    """Screen x of tick ``value`` on a slider track."""
    x0, _, x1, _ = _bbox(slider["track"], "slider track")
    ticks = int(slider["ticks"])
    if not isinstance(value, int) or not 0 <= value <= ticks:
        raise ValueError(f"slider value must be an int in [0,{ticks}], got {value!r}")
    return x0 + round(value * (x1 - x0) / ticks)


def slider_value_at(slider: Any, x_px: int) -> int:
    """The tick a release at screen x ``x_px`` snaps to."""
    x0, _, x1, _ = _bbox(slider["track"], "slider track")
    ticks = int(slider["ticks"])
    if not isinstance(x_px, int):
        raise ValueError(f"slider release x must be an int, got {x_px!r}")
    return min(ticks, max(0, round((x_px - x0) * ticks / (x1 - x0))))


def slider_handle_bbox(slider: Any, value: int) -> list[int]:
    """Handle bbox when the slider sits at ``value``."""
    _, y0, _, y1 = _bbox(slider["track"], "slider track")
    w, h = _int_pair(slider["handle_size"], "slider handle size")
    cy = (y0 + y1) // 2
    return widget_bbox([slider_tick_x(slider, value), cy], [w, h])


def max_scroll_offset(scroll: Any) -> int:
    """Largest row offset a scrollable list clamps to."""
    return max(0, len(scroll["rows"]) - int(scroll["visible_rows"]))


def row_bbox(scroll: Any, index: int, offset: int) -> list[int] | None:
    """Screen bbox of row ``index`` when the pane is scrolled by ``offset`` rows."""
    x0, y0, x1, _ = _bbox(scroll["pane"], "scroll pane")
    if not isinstance(index, int) or not 0 <= index < len(scroll["rows"]):
        raise ValueError(f"row index out of range: {index!r}")
    if not isinstance(offset, int) or not 0 <= offset <= max_scroll_offset(scroll):
        raise ValueError(f"row offset out of range: {offset!r}")
    slot = index - offset
    if not 0 <= slot < int(scroll["visible_rows"]):
        return None
    height = int(scroll["row_height"])
    top = y0 + slot * height
    return [x0, top, x1, top + height]


def _validate_tiles(spec: dict, key: str, screen: tuple[int, int]) -> list[dict]:
    tiles = spec.get(key)
    if not (isinstance(tiles, list) and tiles):
        raise ValueError(f"fixture spec {key!r} must be a nonempty list, got {tiles!r}")
    labels = [t.get("label") for t in tiles]
    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError(f"every {key} entry needs a nonempty string label: {labels!r}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate {key} labels: {labels!r}")
    boxes = []
    for tile in tiles:
        box = _bbox(widget_bbox(tile["center"], tile["size"]), f"{key} bbox")
        _in_screen(box, screen, f"{key} bbox")
        boxes.append(box)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                raise ValueError(f"{key} bboxes overlap: {list(a)} {list(b)}")
    return tiles


def validate_spec(spec: Any) -> dict:
    """Return ``spec`` after checking every field the app and policies rely on."""
    if not isinstance(spec, dict):
        raise TypeError(f"fixture spec must be a dict, got {type(spec)!r}")
    kind = spec.get("kind")
    if kind not in SPEC_KINDS:
        raise ValueError(f"unknown fixture spec kind: {kind!r}")
    screen = _int_pair(spec.get("screen"), "fixture spec screen")
    if screen[0] <= 0 or screen[1] <= 0:
        raise ValueError(f"fixture spec screen must be positive, got {list(screen)}")
    if spec.get("commit_key") != COMMIT_KEY:
        raise ValueError(f"fixture commit_key must be {COMMIT_KEY!r}, got {spec.get('commit_key')!r}")
    if kind in ("buttons", "two_buttons"):
        tiles = _validate_tiles(spec, "buttons", screen)
        if kind == "two_buttons" and len(tiles) != 2:
            raise ValueError(f"two_buttons needs exactly 2 buttons, got {len(tiles)}")
    elif kind == "colors":
        for tile in _validate_tiles(spec, "squares", screen):
            if not _HEX_COLOR_RE.match(str(tile.get("color"))):
                raise ValueError(f"square colour must be #rrggbb, got {tile.get('color')!r}")
    elif kind == "slider":
        slider = spec.get("slider")
        if not isinstance(slider, dict):
            raise ValueError(f"slider spec must be a dict, got {slider!r}")
        track = _bbox(slider.get("track"), "slider track")
        _in_screen(track, screen, "slider track")
        ticks = slider.get("ticks")
        if not isinstance(ticks, int) or ticks < 2:
            raise ValueError(f"slider needs >=2 ticks, got {ticks!r}")
        if not isinstance(slider.get("value"), int) or not 0 <= slider["value"] <= ticks:
            raise ValueError(f"slider start value out of range: {slider.get('value')!r}")
        _int_pair(slider.get("handle_size"), "slider handle size")
        for value in (0, ticks):
            _in_screen(_bbox(slider_handle_bbox(slider, value), "slider handle"), screen, "slider handle")
    elif kind in ("scroll_counter", "scroll_list"):
        scroll = spec.get("scroll")
        if not isinstance(scroll, dict):
            raise ValueError(f"scroll spec must be a dict, got {scroll!r}")
        pane = _bbox(scroll.get("pane"), "scroll pane")
        _in_screen(pane, screen, "scroll pane")
        if kind == "scroll_list":
            rows = scroll.get("rows")
            if not (isinstance(rows, list) and len(rows) >= 2):
                raise ValueError(f"scroll_list needs >=2 rows, got {rows!r}")
            if not all(isinstance(row, str) and row for row in rows):
                raise ValueError(f"scroll_list rows must be nonempty strings: {rows!r}")
            if len(set(rows)) != len(rows):
                raise ValueError(f"duplicate scroll_list rows: {rows!r}")
            height, visible = scroll.get("row_height"), scroll.get("visible_rows")
            if not isinstance(height, int) or height <= 0:
                raise ValueError(f"row_height must be a positive int, got {height!r}")
            if not isinstance(visible, int) or not 1 <= visible < len(rows):
                raise ValueError(f"visible_rows must be in [1,{len(rows) - 1}], got {visible!r}")
            if visible * height != pane[3] - pane[1]:
                raise ValueError(
                    f"scroll pane height {pane[3] - pane[1]} != {visible} rows of {height}px",
                )
    return spec


def parse_spec(raw: Any) -> dict:
    """A validated spec from JSON text (or bytes) as passed on the command line."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise TypeError(f"spec JSON must be text, got {type(raw)!r}")
    return validate_spec(json.loads(raw))


def spec_to_json(spec: Any) -> str:
    """A validated spec as the exact JSON text handed to the guest process."""
    return json.dumps(validate_spec(spec), sort_keys=True, separators=(",", ":"))


def spec_widgets(spec: Any, *, offset_rows: int = 0, slider_value: int | None = None) -> dict[str, list[int]]:
    """Every named widget's screen bbox, derived purely from the spec."""
    spec = validate_spec(spec)
    kind = spec["kind"]
    boxes: dict[str, list[int]] = {}
    if kind in ("buttons", "two_buttons"):
        for tile in spec["buttons"]:
            boxes[tile["label"]] = widget_bbox(tile["center"], tile["size"])
    elif kind == "colors":
        for tile in spec["squares"]:
            boxes[tile["label"]] = widget_bbox(tile["center"], tile["size"])
    elif kind == "slider":
        slider = spec["slider"]
        boxes["slider_track"] = list(_bbox(slider["track"], "slider track"))
        value = slider["value"] if slider_value is None else slider_value
        boxes["slider_handle"] = slider_handle_bbox(slider, value)
    else:
        scroll = spec["scroll"]
        boxes["scroll_pane"] = list(_bbox(scroll["pane"], "scroll pane"))
        if kind == "scroll_list":
            for index, row in enumerate(scroll["rows"]):
                box = row_bbox(scroll, index, offset_rows)
                if box is not None:
                    boxes[row] = box
    return boxes


def hit_test(spec: Any, xy: Any, *, offset_rows: int = 0) -> str | None:
    """The label a click at ``xy`` lands on, or ``None`` for a miss."""
    for label, box in spec_widgets(spec, offset_rows=offset_rows).items():
        if label in ("slider_track", "slider_handle", "scroll_pane"):
            continue
        if bbox_contains(box, xy):
            return label
    return None


def initial_state(spec: Any) -> dict[str, Any]:
    """The fixture state as written before any interaction.

    ``window`` starts as the geometry the spec REQUIRES (origin 0,0, client size
    == screen); the running app republishes the live one, so the recorder can
    reject a window the WM placed or sized differently instead of dispatching
    pixels that mean something else. ``keys_seen`` counts every keypress Tk
    delivers, which is the recorder's proof that this window has the keyboard."""
    spec = validate_spec(spec)
    slider = spec.get("slider")
    return {
        "ready": False,
        "keyboard": False,
        "keys_seen": 0,
        "window": [0, 0, int(spec["screen"][0]), int(spec["screen"][1])],
        "kind": spec["kind"],
        "clicked": [],
        "double_clicked": [],
        "right_clicked": [],
        "misses": 0,
        "slider_value": int(slider["value"]) if slider else None,
        "wheel_notches": 0,
        "scroll_offset": 0,
        "committed": False,
        "widgets": spec_widgets(spec),
    }


def record_click(state: dict, label: str | None, *, button: str = "left", count: int = 1) -> dict[str, Any]:
    """State after a click that landed on ``label`` (``None`` records a miss)."""
    if button not in ("left", "right"):
        raise ValueError(f"button must be left or right, got {button!r}")
    if not isinstance(count, int) or not 1 <= count <= 3:
        raise ValueError(f"click count must be 1..3, got {count!r}")
    out = dict(state)
    if label is None:
        out["misses"] = int(state["misses"]) + 1
        return out
    if button == "right":
        out["right_clicked"] = [*state["right_clicked"], label]
        return out
    out["clicked"] = [*state["clicked"], label]
    if count >= 2:
        out["double_clicked"] = [*state["double_clicked"], label]
    return out


def counter_text(state: Any) -> str:
    """The scroll pad's displayed value for ``state`` — its cumulative notches.

    The label is the only evidence a policy has that its scroll landed, so it is
    derived from the same accumulator the verifier reads and repainted inside the
    wheel handler; a label that trails the state would tell the model its action
    did nothing."""
    return str(int(state["wheel_notches"]))


def pairs_double_click(
    label: str, *, button: str, count: int, last_label: Any, elapsed_s: float,
) -> bool:
    """Whether this click completes a double click on the same tile.

    Tk's own ``<Double-Button-1>`` needs both presses inside a compiled-in 500 ms
    window, but the harness sends every primitive as its own HTTP ``/execute``
    (~350 ms each), so a dispatched double click lands ~700 ms apart and Tk
    reports two singles. Pairing them under an explicit ``DOUBLE_CLICK_S``
    reproduces exactly the state the native binding would (``clicked`` twice,
    ``double_clicked`` once) instead of silently losing the double."""
    return (
        count == 1
        and button == "left"
        and isinstance(last_label, str)
        and label == last_label
        and 0.0 <= elapsed_s <= DOUBLE_CLICK_S
    )


def apply_click(spec: Any, state: dict, xy: Any, *, button: str = "left", count: int = 1) -> dict[str, Any]:
    """State after a click at screen pixel ``xy``."""
    label = hit_test(spec, xy, offset_rows=int(state["scroll_offset"]))
    return record_click(state, label, button=button, count=count)


def apply_scroll(spec: Any, state: dict, notches: int, *, pointer_xy: Any = None) -> dict[str, Any]:
    """State after ``notches`` wheel notches (positive scrolls up).

    A scroll with the pointer outside the scroll pane is a miss, not a notch, so
    the golden policy's "move into the pane first" turn is load-bearing."""
    spec = validate_spec(spec)
    if not isinstance(notches, int) or notches == 0:
        raise ValueError(f"wheel notches must be a nonzero int, got {notches!r}")
    out = dict(state)
    pane = spec.get("scroll", {}).get("pane")
    if pointer_xy is not None and pane is not None and not bbox_contains(pane, pointer_xy):
        out["misses"] = int(state["misses"]) + 1
        return out
    out["wheel_notches"] = int(state["wheel_notches"]) + notches
    if spec["kind"] == "scroll_list":
        scroll = spec["scroll"]
        out["scroll_offset"] = min(
            max_scroll_offset(scroll), max(0, int(state["scroll_offset"]) - notches),
        )
    return out


def apply_drag(spec: Any, state: dict, x_px: int) -> dict[str, Any]:
    """State after releasing the slider handle at screen x ``x_px``."""
    spec = validate_spec(spec)
    if spec["kind"] != "slider":
        raise ValueError(f"apply_drag needs a slider spec, got {spec['kind']!r}")
    out = dict(state)
    out["slider_value"] = slider_value_at(spec["slider"], x_px)
    return out


def apply_commit(spec: Any, state: dict) -> dict[str, Any]:
    """State after the commit key (Return) confirms the interaction."""
    validate_spec(spec)
    out = dict(state)
    out["committed"] = True
    return out


def write_state(path: Any, state: dict) -> None:
    """Atomically publish the fixture state as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=target.name, dir=str(target.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True)
        Path(tmp).replace(target)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


_PAGE_CSS = (
    "html,body{margin:0;padding:0;background:#ffffff;font-family:sans-serif;color:#131a22;}"
    "#root{position:relative;width:1920px;overflow:hidden;}"
    "#banner{position:absolute;left:60px;top:40px;font-size:28px;color:#4a5563;}"
    ".lnk,.btn,#field{position:absolute;box-sizing:border-box;}"
    ".lnk{display:flex;align-items:center;justify-content:center;font-size:26px;"
    "color:#0b57d0;background:#f2f6fd;border:2px solid #c7d4e8;text-decoration:underline;}"
    ".btn{font-size:28px;background:#0b57d0;color:#ffffff;border:none;}"
    "#field{font-size:30px;padding:8px;border:3px solid #0b57d0;background:#ffffff;}"
)


_PAGE_READY_JS = (
    'window.addEventListener("load",function(){requestAnimationFrame(function(){'
    'requestAnimationFrame(function(){document.title="' + PAGE_READY_TITLE + '";});});});'
)


def _style_box(bbox: Any) -> str:
    x0, y0, x1, y1 = _bbox(bbox, "page element bbox")
    return f"left:{x0}px;top:{y0}px;width:{x1 - x0}px;height:{y1 - y0}px"


def _page_token(params: Any) -> str:
    if not isinstance(params, dict):
        raise TypeError(f"page params must be a dict, got {type(params)!r}")
    token = params.get("token")
    if not (isinstance(token, str) and _TOKEN_RE.match(token)):
        raise ValueError(f"page token must be 6-32 lowercase hex chars, got {token!r}")
    return token


def make_html_page(kind: str, params: Any) -> str:
    """The complete local HTML page for a browser template's seeded params.

    Every page ends with ``_PAGE_READY_JS``: the setup gate waits for
    ``PAGE_READY_TITLE`` to reach the window manager, which happens only after
    ``load`` plus two animation frames, so no policy can click a page that has
    not painted and whose handlers are not installed yet.

    Handlers are attached with ``addEventListener``, never as inline ``onclick``
    attributes: a JS string literal inside an HTML attribute value needs its
    quotes escaped for BOTH parsers at once, and getting that wrong yields a
    page that renders perfectly and silently ignores every click."""
    if kind not in PAGE_KINDS:
        raise ValueError(f"unknown page kind: {kind!r}")
    token = _page_token(params)
    if token in PAGE_READY_TITLE:
        raise ValueError(f"page token {token!r} collides with the ready title")
    height = PAGE_WH[1]
    banner = '<div id="banner">shortgoal local page</div>'
    if kind == "link_grid":
        target = params["label"]
        labels = [link["label"] for link in params["links"]]
        if target not in labels:
            raise ValueError(f"link target {target!r} is not one of {labels!r}")
        parts = []
        for link in params["links"]:
            style = _style_box(widget_bbox(link["center"], link["size"]))
            label = link["label"]
            parts.append(
                '<a class="lnk" href="#" data-label="' + escape(label) + '" style="'
                + style + '">' + escape(label) + "</a>",
            )
        body = banner + "".join(parts)
        script = (
            "var TOKEN=" + json.dumps(token) + ";var TARGET=" + json.dumps(target) + ";"
            'Array.prototype.forEach.call(document.querySelectorAll("a.lnk"),function(node){'
            'node.addEventListener("click",function(event){event.preventDefault();'
            'var label=node.getAttribute("data-label");'
            'document.title=(label===TARGET)?TOKEN:("miss:"+label);});});'
        )
    elif kind == "input":
        style = _style_box(widget_bbox(params["input_xy"], params["input_size"]))
        body = banner + '<input id="field" autofocus autocomplete="off" style="' + style + '">'
        script = (
            "var TOKEN=" + json.dumps(token) + ";var field=document.getElementById(\"field\");"
            "field.focus();"
            "field.addEventListener(\"input\",function(){document.title=TOKEN+\":\"+field.value;});"
            "field.addEventListener(\"keydown\",function(event){"
            "if(event.key===\"Enter\"){event.preventDefault();}});"
        )
    else:
        height = int(params["page_height"])
        if height <= PAGE_WH[1]:
            raise ValueError(f"below_fold page must be taller than {PAGE_WH[1]}px, got {height}")
        box = _bbox(widget_bbox(params["button_page_xy"], params["button_size"]), "page button bbox")
        if box[1] < PAGE_WH[1] or box[3] > height:
            raise ValueError(f"below_fold button {list(box)} is not below the fold of {height}px")
        label = params["label"]
        body = (
            banner + '<button class="btn" id="target" style="' + _style_box(box)
            + '">' + escape(label) + "</button>"
        )
        script = (
            "var TOKEN=" + json.dumps(token) + ";"
            'document.getElementById("target").addEventListener("click",function(){'
            "document.title=TOKEN;});"
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>shortgoal</title>'
        "<style>" + _PAGE_CSS + "#root{height:" + str(height) + "px;}</style></head>"
        '<body><div id="root">' + body + "</div><script>" + script + _PAGE_READY_JS
        + "</script></body></html>"
    )


class Fixture:
    """The in-guest Tk window: spec in, pixel-exact widgets out, state on disk."""

    def __init__(self, spec: dict, state_path: Any = STATE_PATH) -> None:
        if tk is None:
            raise RuntimeError("tkinter is unavailable in this interpreter")
        self.spec = validate_spec(spec)
        self.state = initial_state(self.spec)
        self.state_path = Path(state_path)
        self.widgets: dict[str, Any] = {}
        self.last_click: str | None = None
        self.last_click_at = 0.0
        self.write_scheduled = False
        self.root = tk.Tk()
        width, height = self.spec["screen"]
        self.root.title(FIXTURE_TITLE)
        self.root.geometry(f"{width}x{height}+0+0")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=_BG)
        getattr(self, f"_build_{self.spec['kind']}")()
        self.root.bind("<KeyPress>", self._on_key)
        self.root.bind_all(f"<KeyPress-{COMMIT_KEY}>", self._on_commit)
        self.root.bind_all("<Button-4>", lambda event: self._on_wheel(1))
        self.root.bind_all("<Button-5>", lambda event: self._on_wheel(-1))
        self.root.bind_all("<MouseWheel>", lambda event: self._on_wheel(1 if event.delta > 0 else -1))
        self.root.after(250, self._ready)

    def _tile(self, label: str, bbox: list[int], *, bg: str, fg: str = _TILE_FG, text: str | None = None) -> None:
        x0, y0, x1, y1 = bbox
        widget = tk.Label(
            self.root,
            text=label if text is None else text,
            bg=bg,
            fg=fg,
            font=("Sans", 22, "bold"),
            relief="raised",
            borderwidth=2,
        )
        widget.place(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        widget.bind("<Button-1>", lambda event, name=label: self._on_click(name, 1, "left"))
        widget.bind("<Double-Button-1>", lambda event, name=label: self._on_click(name, 2, "left"))
        widget.bind("<Button-3>", lambda event, name=label: self._on_click(name, 1, "right"))
        self.widgets[label] = widget

    def _build_buttons(self) -> None:
        for tile in self.spec["buttons"]:
            self._tile(tile["label"], widget_bbox(tile["center"], tile["size"]), bg=_TILE_BG)

    def _build_two_buttons(self) -> None:
        self._build_buttons()

    def _build_colors(self) -> None:
        for tile in self.spec["squares"]:
            self._tile(
                tile["label"],
                widget_bbox(tile["center"], tile["size"]),
                bg=tile["color"],
                fg=tile["color"],
                text="",
            )

    def _build_slider(self) -> None:
        slider = self.spec["slider"]
        x0, y0, x1, y1 = slider["track"]
        pad = 70
        self.canvas_origin = (x0 - pad, y0 - pad)
        canvas = tk.Canvas(self.root, bg=_PANE_BG, highlightthickness=0)
        canvas.place(x=x0 - pad, y=y0 - pad, width=(x1 - x0) + 2 * pad, height=(y1 - y0) + 2 * pad)
        canvas.create_rectangle(pad, pad, pad + (x1 - x0), pad + (y1 - y0), fill=_TRACK_BG, outline="")
        for tick in range(slider["ticks"] + 1):
            tx = slider_tick_x(slider, tick) - self.canvas_origin[0]
            canvas.create_line(tx, pad - 22, tx, pad + (y1 - y0) + 22, fill="#7f8b9b")
            canvas.create_text(tx, pad + (y1 - y0) + 44, text=str(tick), fill=_TILE_FG, font=("Sans", 16))
        canvas.bind("<ButtonRelease-1>", self._on_slider_release)
        self.canvas = canvas
        self._draw_handle()

    def _draw_handle(self) -> None:
        self.canvas.delete("handle")
        x0, y0, x1, y1 = slider_handle_bbox(self.spec["slider"], int(self.state["slider_value"]))
        ox, oy = self.canvas_origin
        self.canvas.create_rectangle(
            x0 - ox, y0 - oy, x1 - ox, y1 - oy, fill=_ACCENT, outline="#12356b", width=2, tags="handle",
        )

    def _build_scroll_counter(self) -> None:
        x0, y0, x1, y1 = self.spec["scroll"]["pane"]
        frame = tk.Frame(self.root, bg=_PANE_BG, highlightthickness=3, highlightbackground=_ACCENT)
        frame.place(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        self.counter = tk.Label(
            frame, text=counter_text(self.state), bg=_PANE_BG, fg=_TILE_FG,
            font=("Sans", 72, "bold"),
        )
        self.counter.pack(expand=True)
        self.widgets["scroll_pane"] = frame

    def _build_scroll_list(self) -> None:
        x0, y0, x1, y1 = self.spec["scroll"]["pane"]
        canvas = tk.Canvas(self.root, bg=_PANE_BG, highlightthickness=0)
        canvas.place(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        canvas.bind("<Button-1>", self._on_list_click)
        self.canvas = canvas
        self.canvas_origin = (x0, y0)
        self._draw_rows()

    def _draw_rows(self) -> None:
        scroll = self.spec["scroll"]
        self.canvas.delete("row")
        offset = int(self.state["scroll_offset"])
        for index, row in enumerate(scroll["rows"]):
            box = row_bbox(scroll, index, offset)
            if box is None:
                continue
            ox, oy = self.canvas_origin
            self.canvas.create_rectangle(
                box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy,
                fill="#ffffff" if (index - offset) % 2 == 0 else "#eef2f8",
                outline="#c8d2de",
                tags="row",
            )
            self.canvas.create_text(
                box[0] - ox + 24, (box[1] + box[3]) // 2 - oy,
                text=row, anchor="w", fill=_TILE_FG, font=("Sans", 22), tags="row",
            )

    def _repaint(self) -> None:
        """Push the state's visible effect out before anything slow happens.

        Every handler redraws through here and flushes Tk's idle queue itself, so
        the pixels a screenshot can see always agree with the state the verifier
        will read — the frame is never one input behind."""
        kind = self.spec["kind"]
        if kind == "scroll_list":
            self._draw_rows()
        elif kind == "scroll_counter":
            self.counter.configure(text=counter_text(self.state))
        elif kind == "slider":
            self._draw_handle()
        self.root.update_idletasks()

    def _on_click(self, label: str, count: int, button: str) -> None:
        now = time.monotonic()
        paired = pairs_double_click(
            label,
            button=button,
            count=count,
            last_label=self.last_click,
            elapsed_s=now - self.last_click_at,
        )
        self.last_click = None if paired or button != "left" else label
        self.last_click_at = now
        self.state = record_click(self.state, label, button=button, count=2 if paired else count)
        self.widgets[label].configure(relief="sunken")
        self._repaint()
        self._write()

    def _on_key(self, event: Any) -> None:
        self.state["keys_seen"] = int(self.state["keys_seen"]) + 1
        self._write()

    def _on_list_click(self, event: Any) -> None:
        self.state = apply_click(self.spec, self.state, [int(event.x_root), int(event.y_root)])
        self._repaint()
        self._write()

    def _on_slider_release(self, event: Any) -> None:
        self.state = apply_drag(self.spec, self.state, int(event.x_root))
        self._repaint()
        self._write()

    def _on_wheel(self, notches: int) -> None:
        pointer = self.root.winfo_pointerxy()
        self.state = apply_scroll(
            self.spec, self.state, notches, pointer_xy=[int(pointer[0]), int(pointer[1])],
        )
        self._repaint()
        self._write_soon()

    def _on_commit(self, event: Any) -> None:
        self.state = apply_commit(self.spec, self.state)
        self.root.configure(bg=_COMMITTED_BG)
        self._repaint()
        self._write()

    def _ready(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.update_idletasks()
        self.state["ready"] = True
        self._write()
        self._hold_focus()

    def _hold_focus(self) -> None:
        """Keep the X input focus on this window for as long as it lives.

        A window manager can hand the focus to anything that maps later (or to
        nothing at all right after boot), and a fixture without the focus sees
        clicks but no keys — so re-assert it, and publish whether Tk agrees the
        focus is ours."""
        focused = self.root.focus_displayof() is not None
        if not focused:
            self.root.lift()
            self.root.focus_force()
        if focused != bool(self.state["keyboard"]):
            self.state["keyboard"] = focused
            self._write()
        self.root.after(_FOCUS_POLL_MS, self._hold_focus)

    def _write_soon(self) -> None:
        """Publish the state once the whole input burst has been handled.

        A wheel notch is its own X event and the state file lands on a
        copy-on-write guest disk, so writing inside every notch's handler makes
        the visible counter crawl behind the burst — long enough for a settled
        screenshot to catch two identical frames at a value the action has
        already moved past. The repaint stays synchronous; only the file write
        coalesces, and it is a timer rather than an idle task so the repaint's
        ``update_idletasks`` cannot fire it one notch at a time."""
        if self.write_scheduled:
            return
        self.write_scheduled = True
        self.root.after(_WRITE_COALESCE_MS, self._write_now)

    def _write_now(self) -> None:
        self.write_scheduled = False
        self._write()

    def _write(self) -> None:
        self.root.update_idletasks()
        self.state["window"] = [
            self.root.winfo_rootx(), self.root.winfo_rooty(),
            self.root.winfo_rootx() + self.root.winfo_width(),
            self.root.winfo_rooty() + self.root.winfo_height(),
        ]
        boxes = spec_widgets(
            self.spec,
            offset_rows=int(self.state["scroll_offset"]),
            slider_value=self.state["slider_value"],
        )
        for label, widget in self.widgets.items():
            x0, y0 = widget.winfo_rootx(), widget.winfo_rooty()
            boxes[label] = [x0, y0, x0 + widget.winfo_width(), y0 + widget.winfo_height()]
        self.state["widgets"] = boxes
        write_state(self.state_path, self.state)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="short-goal in-guest fixture")
    parser.add_argument("--spec", required=True, help="fixture spec as JSON text, or a path to a JSON file")
    parser.add_argument("--state", default=STATE_PATH)
    args = parser.parse_args()
    raw = args.spec if args.spec.lstrip().startswith("{") else Path(args.spec).read_text(encoding="utf-8")
    Fixture(parse_spec(raw), Path(args.state)).run()


if __name__ == "__main__":
    main()
