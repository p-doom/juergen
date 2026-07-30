"""Action formatters: (events, windows, dead_zones) -> one label per window.

This is the single place the on-disk keylog becomes assistant-turn text, so
action-format ablations are a stage-04 flag, not a pipeline rerun. A formatter
receives the FULL raw event stream (state layer) plus the dead-zone label
policy's dispositions (label layer) via ``events.apply_label_policy`` — a
stateful format (e.g. cumulative cursor position) folds over every event while
emitting labels only for owned ones.

``CanonicalFormatter`` reproduces the historical format byte-for-byte on
dead-zone-free stretches (``common.format_action`` over per-window bins with
segment-global held-set dedup) — that identity is the regression gate
(tests/test_action_format.py).

``OrderedFormatter`` (``ordered_events_v2``, ported from the yll/action-format
branch) preserves the relative order of movement, scrolling, and key/button
transitions inside each window as one mini-program:
``move(4,-1); down(LMB); move(2,0); up(LMB)``. Continuous motion is
accumulated on an internal ``continuous_action_hz`` motor grid (default 10 Hz
== 100 ms ticks, NOT another frame rate); every press/release is an ordering
barrier at its exact position. The aggregate format cannot represent
``move -> click -> move``; this one can. Held-state anomaly accounting
(redundant press / dangling release / held at end) already lives in the shared
label policy's ``PolicyCounters``.

``OrderedTypingFormatter`` (``ordered_events_v3``) is ordered_events_v2 plus a
``type("...")`` primitive: maximal runs of plain typing collapse into one
quoted string (the typing action the base model natively knows — per-key
down/up typing cost ~8 tokens/char and caused severe truncation downstream).
See ORDERED_EVENTS_V3_GRAMMAR for the exact line grammar eval/ parsers
implement.

``ComputerUseFormatter`` (``computer_use_rel_v1``) emits Qwen3-VL's NATIVE
computer-use format — ``<tool_call>`` JSON blocks against a ``computer_use``
function, extended minimally for a relative mouse (``mouse_move_rel``). The
binding contract is system_prompts/cua_v4_thinking.txt: this formatter's
output is exactly what that tool spec describes (action enum, argument
names, lowercase pyautogui-style key names). No motor grid: continuous
motion/scroll accumulates between discrete barriers only; clicks, chords,
and typing runs collapse into the tool's native high-level actions while
drags and multi-pair modifier scopes decompose exactly.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from realigned_pipeline.lib.common import ActionBin, format_action
from realigned_pipeline.lib.events import (
    DeadZone,
    LabeledEvent,
    PolicyCounters,
    RawEvent,
    Window,
    apply_label_policy,
)
from realigned_pipeline.lib.rel_step_contract import (
    MOVEMENT_SCALES,
    SCROLL_STEPS,
    TYPING_GAP_S,
    TYPING_MAX_CHARS,
    rel_step_delta,
)

DEFAULT_CONTINUOUS_ACTION_HZ = 10.0

# Rendered names must be unambiguous inside the mini-program syntax.
_INPUT_NAME_RE = re.compile(r"^[^\s(),;]+$")

# Grammar of one ``ordered_events_v3`` action line (the assistant-turn payload
# for one window). eval/ parsers implement exactly this. The primitive line
# itself NEVER contains TERMINATE — stage 04 may append TERMINATE as its own
# suffix line, which is stage 04's business, not the formatter's.
ORDERED_EVENTS_V3_GRAMMAR = r'''
line       = "NO_OP" / primitive *("; " primitive)
primitive  = move / scroll / down / up / type
move       = "move(" int "," int ")"         ; accumulated motor-tick mouse delta
scroll     = "scroll(" int "," int ")"       ; accumulated motor-tick scroll delta
down       = "down(" NAME ")"                ; key/button press
up         = "up(" NAME ")"                  ; key/button release
type       = "type(" DQUOTE chars DQUOTE ")" ; run of >=1 typed characters
int        = ["-"] 1*DIGIT                   ; move(0,0)/scroll(0,0) never emitted
NAME       = 1*name-char                     ; name-char = any char except
                                             ; whitespace "(" ")" "," ";"
chars      = 1*char                          ; char = escape / plain
escape     = "\" ("\" / DQUOTE)              ; \\ -> backslash, \" -> double quote
plain      = any printable US-keyboard character except "\" and DQUOTE:
             letters, digits, space, `~!@#$%^&*()-_=+[]{}|;:'<>,./?
                                             ; Return/Tab are down/up, never
                                             ; typed, so chars has no newline/tab
'''


@dataclass
class FormatResult:
    labels: list[str]  # one per window, same order
    counters: PolicyCounters
    # Provenance for primitive-based formats (None for the aggregate format).
    primitive_counts: dict[str, int] | None = None


class ActionFormatter(Protocol):
    name: str
    # ``{what}`` is "the next action" / "the next action toward that goal";
    # stage 04 composes the default system prompts from this.
    reply_contract: str

    def terminate_line(self) -> str:
        """The complete goal-done assistant action line for this format.

        Stage 04t emits this (never a hardcoded literal) as the whole action
        payload of a terminate turn. Text formats return the historical
        ``TERMINATE``; ``computer_use_rel_v1`` returns its native
        ``terminate`` tool_call block."""
        ...

    def is_idle_label(self, label: str) -> bool:
        """True iff ``label`` encodes a do-nothing window in this format
        (``NO_OP`` for the text formats, a lone ``wait`` tool_call for
        ``computer_use_rel_v1``). Stage 04t uses this for the ``n_non_noop``
        metadata count."""
        ...

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
        frame_size: tuple[int, int] | None = None,
    ) -> FormatResult:
        """Per-window action labels.

        ``frame_size`` is the segment's ORIGINAL capture ``(width, height)``
        from the clips manifest. Only formats that normalize move deltas
        against it require it; every other format ignores it."""
        ...


def _ordered_owned(labeled: Sequence[LabeledEvent]) -> list[LabeledEvent]:
    """Owned events in label order: by clamped label time, ties by original
    stream position (so a press clamped onto a tick boundary precedes that
    tick's native events, and a clamped release follows its window's)."""
    return sorted(
        (le for le in labeled if le.window is not None),
        key=lambda le: (le.label_t, le.event.seq),
    )


class CanonicalFormatter:
    """Per-window ``round(dx) round(dy) round(scroll) [; +KEY -KEY]`` / ``NO_OP``."""

    name = "canonical"
    reply_contract = (
        "Reply with {what} as `<dx> <dy> <scroll>` optionally followed by "
        "` ; +KEY -KEY` events, or `NO_OP` if no action."
    )

    def terminate_line(self) -> str:
        return "TERMINATE"

    def is_idle_label(self, label: str) -> bool:
        return label == "NO_OP"

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
        frame_size: tuple[int, int] | None = None,
    ) -> FormatResult:
        labeled, counters = apply_label_policy(
            events, windows, dead_zones, master_fps=master_fps
        )
        bins = [ActionBin() for _ in windows]
        for le in _ordered_owned(labeled):
            e = le.event
            b = bins[le.window]
            if e.kind == "move":
                b.move_dx += e.dx
                b.move_dy += e.dy
            elif e.kind == "scroll":
                b.scroll += e.scroll
            elif e.kind == "press":
                b.events.append(("+", e.name))
            else:
                b.events.append(("-", e.name))
        return FormatResult(
            labels=[format_action(b) for b in bins],
            counters=counters,
        )


# US-layout printable map: key name -> (base char, shifted char). The names are
# the rdev namespace ``events.resolve_key_name`` passes through from the keylog
# (top-row digits are Num0..Num9); the char pairs mirror the viewer's keyToChar
# map (visualize_frame_records.py), the project's existing convention. Keypad
# digits, Return, Tab, Backspace, Escape, arrows, F-keys and modifiers are
# deliberately absent: they render as down()/up() and break typing runs.
_US_PRINTABLE: dict[str, tuple[str, str]] = {
    "Space": (" ", " "),
    **{f"Key{c}": (c.lower(), c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    **{f"Num{d}": (d, s) for d, s in zip("1234567890", "!@#$%^&*()", strict=True)},
    "BackQuote": ("`", "~"),
    "Minus": ("-", "_"),
    "Equal": ("=", "+"),
    "BracketLeft": ("[", "{"),
    "BracketRight": ("]", "}"),
    "BackSlash": ("\\", "|"),
    "SemiColon": (";", ":"),
    "Quote": ("'", '"'),
    "Comma": (",", "<"),
    "Dot": (".", ">"),
    "Slash": ("/", "?"),
}

_SHIFT_KEYS = frozenset({"ShiftLeft", "ShiftRight"})
# A held non-Shift modifier vetoes typing pairs (Ctrl+C is a chord, not typing).
_NON_SHIFT_MODIFIERS = frozenset({
    "ControlLeft", "ControlRight", "Control",
    "Alt", "AltLeft", "AltRight", "AltGr",
    "MetaLeft", "MetaRight", "Meta", "MetaGr",
    "Function",
})
_MODIFIER_KEYS = _SHIFT_KEYS | _NON_SHIFT_MODIFIERS


def _escape_typed_text(text: str) -> str:
    """Escape for the ``type("...")`` payload: only ``\\`` and ``"``."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


@dataclass(frozen=True)
class ActionPrimitive:
    kind: str  # "move" | "scroll" | "down" | "up" | "type"
    dx: int | None = None
    dy: int | None = None
    input_name: str | None = None
    text: str | None = None  # "type" only: raw (unescaped) typed characters
    t_s: float | None = None  # discrete event time; rel-step typing gap policy
    owner: int | None = None  # source window; used for cross-window type bursts

    def render(self) -> str:
        if self.kind in ("move", "scroll"):
            return f"{self.kind}({self.dx},{self.dy})"
        if self.kind == "type":
            return f'type("{_escape_typed_text(self.text)}")'
        return f"{self.kind}({self.input_name})"


class OrderedFormatter:
    """Per-window ordered mini-program: ``primitive(; primitive)*`` / ``NO_OP``
    with primitives ``move(dx,dy)``, ``scroll(dx,dy)``, ``down(NAME)``,
    ``up(NAME)``. A click is ``down(LMB); up(LMB)`` — no click/type/wait/timing
    primitives, no invented transitions.

    Continuous events accumulate per (window, motor tick, kind) in floating
    point and round once at flush; the accumulator flushes when the tick or
    kind changes, a discrete transition occurs, or the window ends. ``move(0,0)``
    and ``scroll(0,0)`` are omitted; an empty window renders ``NO_OP``."""

    name = "ordered_events_v2"
    reply_contract = (
        "Reply with {what} as `; `-separated primitives in the order performed "
        "— `move(<dx>,<dy>)`, `scroll(<dx>,<dy>)`, `down(<KEY>)`, `up(<KEY>)` "
        "— or `NO_OP` if no action."
    )

    def __init__(self, continuous_action_hz: float = DEFAULT_CONTINUOUS_ACTION_HZ):
        if not math.isfinite(continuous_action_hz) or continuous_action_hz <= 0:
            raise ValueError("continuous_action_hz must be finite and positive")
        self.continuous_action_hz = continuous_action_hz

    def terminate_line(self) -> str:
        return "TERMINATE"

    def is_idle_label(self, label: str) -> bool:
        return label == "NO_OP"

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
        frame_size: tuple[int, int] | None = None,
    ) -> FormatResult:
        primitives, counters = self._window_primitives(
            events, windows, dead_zones, master_fps=master_fps
        )
        counts = Counter(p.kind for window in primitives for p in window)
        return FormatResult(
            labels=[
                "; ".join(p.render() for p in window) if window else "NO_OP"
                for window in primitives
            ],
            counters=counters,
            primitive_counts={k: counts.get(k, 0) for k in ("move", "scroll", "down", "up")},
        )

    def _window_primitives(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
    ) -> tuple[list[list[ActionPrimitive]], PolicyCounters]:
        """One ordered primitive list per window (the shared v2/v3 core)."""
        labeled, counters = apply_label_policy(
            events, windows, dead_zones, master_fps=master_fps
        )
        hz = self.continuous_action_hz
        primitives: list[list[ActionPrimitive]] = [[] for _ in windows]
        # (window, motor tick, kind, dx sum, dy sum)
        pending: tuple[int, int, str, float, float] | None = None

        def flush() -> None:
            nonlocal pending
            if pending is None:
                return
            win, _tick, kind, dx, dy = pending
            rdx, rdy = round(dx), round(dy)
            if rdx != 0 or rdy != 0:
                primitives[win].append(ActionPrimitive(kind=kind, dx=rdx, dy=rdy))
            pending = None

        for le in _ordered_owned(labeled):
            e = le.event
            win = le.window
            if e.kind in ("move", "scroll"):
                # Deltas are never clamped, so label_t is the native event time
                # and lies within the owning window's span.
                offset_s = le.label_t - windows[win].start / master_fps
                tick = math.floor(offset_s * hz + 1e-9)
                if pending is not None and pending[:3] == (win, tick, e.kind):
                    pending = (win, tick, e.kind, pending[3] + e.dx, pending[4] + e.dy)
                else:
                    flush()
                    pending = (win, tick, e.kind, e.dx, e.dy)
            else:
                flush()
                if not isinstance(e.name, str) or not _INPUT_NAME_RE.fullmatch(e.name):
                    raise ValueError(f"invalid input name for ordered action: {e.name!r}")
                primitives[win].append(
                    ActionPrimitive(kind="down" if e.kind == "press" else "up",
                                    input_name=e.name)
                )
        flush()
        return primitives, counters


def _collapse_typing(
    prims: Sequence[ActionPrimitive], held_mods: set[str],
    *, max_gap_s: float | None = None,
) -> list[ActionPrimitive]:
    """One window's primitives with typing runs collapsed to ``type("...")``.

    A *typing run* is a maximal contiguous span of ``down``/``up`` primitives
    of printable and Shift keys ONLY, that is BALANCED (every key pressed in
    the span is released in it and vice versa), entered while no non-Shift
    modifier is held. Characters are emitted in ``down`` order — so key
    *rollover* (``down h; down e; up h; up e`` from natural fast typing, where
    releases interleave or reorder) still collapses to ``type("he")``, not a
    string of per-key events. Shift downs/ups inside such a run are absorbed
    (the capital is already in the string); the emitted char honors the Shift
    state at each key's ``down``.

    A span that never balances before a breaker renders explicitly instead:
    a printable key held across a window boundary (its ``up`` is in another
    window) → ``down``/``up``; a Shift enclosing a non-typing primitive (move,
    Tab, …) → the Shift renders while inner printable keys still ``type`` under
    it; a bare Shift tap (zero characters) → ``down``/``up``. Any non-key
    primitive (move, scroll, mouse button, Return/Backspace/arrow, a non-Shift
    modifier) breaks the run. ``held_mods`` is the cross-window physical
    modifier held-set; updated in place for the next window (balanced runs
    net-zero it, so only explicitly-rendered modifier events move it)."""
    n = len(prims)
    out: list[ActionPrimitive] = []
    mods = set(held_mods)

    def _is_typing_key(p: ActionPrimitive) -> bool:
        return p.kind in ("down", "up") and (
            p.input_name in _US_PRINTABLE or p.input_name in _SHIFT_KEYS
        )

    i = 0
    while i < n:
        p = prims[i]
        # A typing run can only OPEN on a printable/Shift key down, and only
        # when no non-Shift modifier is physically held.
        if p.kind == "down" and _is_typing_key(p) and not (mods & _NON_SHIFT_MODIFIERS):
            # Scan the longest BALANCED prefix of the contiguous key-only span
            # starting at i (balanced == every down matched by a later up in
            # the span). ``run_end`` is exclusive; None if it never balances.
            open_counts: dict[str, int] = {}
            run_end: int | None = None
            j = i
            prev_t_s: float | None = None
            while j < n and _is_typing_key(prims[j]):
                q = prims[j]
                if (
                    max_gap_s is not None
                    and prev_t_s is not None
                    and q.t_s is not None
                    and q.t_s - prev_t_s >= max_gap_s
                ):
                    break
                if q.kind == "down":
                    open_counts[q.input_name] = open_counts.get(q.input_name, 0) + 1
                else:  # up
                    c = open_counts.get(q.input_name, 0)
                    if c == 0:
                        break  # up with no matching down in-span → held from before
                    open_counts[q.input_name] = c - 1
                    if open_counts[q.input_name] == 0:
                        del open_counts[q.input_name]
                j += 1
                prev_t_s = q.t_s
                if not open_counts:
                    run_end = j  # balanced here — candidate maximal run end
            if run_end is not None:
                # Emit the run [i, run_end): chars in down order, Shift folded
                # and absorbed. Non-Shift modifiers cannot occur inside (the
                # span is printable/Shift only and we entered with none held).
                shift_held = bool(mods & _SHIFT_KEYS)
                chars: list[str] = []
                for k in range(i, run_end):
                    q = prims[k]
                    if q.input_name in _SHIFT_KEYS:
                        shift_held = q.kind == "down"
                        continue
                    if q.kind == "down":  # printable down → a typed character
                        base, shifted = _US_PRINTABLE[q.input_name]
                        chars.append(shifted if shift_held else base)
                    # printable up → release only, nothing to emit
                if chars:
                    out.append(ActionPrimitive(
                        kind="type", text="".join(chars), t_s=prims[i].t_s,
                        owner=prims[i].owner,
                    ))
                    i = run_end
                    continue
                # Zero characters (e.g. a bare Shift tap) — fall through to
                # render the run's primitives explicitly rather than type("").
        # Not the start of a (character-producing) typing run: render this
        # primitive verbatim and track physical modifier state.
        if p.kind == "down" and p.input_name in _MODIFIER_KEYS:
            mods.add(p.input_name)
        elif p.kind == "up" and p.input_name in _MODIFIER_KEYS:
            mods.discard(p.input_name)
        out.append(p)
        i += 1

    held_mods.clear()
    held_mods.update(mods)
    return out


class OrderedTypingFormatter(OrderedFormatter):
    """``ordered_events_v2`` plus a ``type("...")`` primitive: maximal runs of
    plain typing — press immediately followed by the release of the same
    printable key, in owned-event label order within one window — collapse
    into one quoted string (the typing action the base model natively knows;
    per-key down/up typing cost ~8 tokens/char and truncated downstream).
    ShiftLeft/ShiftRight are absorbed into the run when they enclose only
    typing pairs (exactly one Shift held, no other modifier -> the shifted
    US-layout character); a Shift whose scope includes anything else renders
    as normal down/up and breaks the run, as does every other rendered
    primitive. Escaping inside the quotes: ``\\`` and ``\"`` only. Everything
    else — motor grid, ordering, NO_OP, held-state accounting — is
    byte-identical to v2 (see ORDERED_EVENTS_V3_GRAMMAR)."""

    name = "ordered_events_v3"
    reply_contract = (
        "Reply with {what} as `; `-separated primitives in the order performed "
        "— `move(<dx>,<dy>)`, `scroll(<dx>,<dy>)`, `down(<KEY>)`, `up(<KEY>)`, "
        '`type("<text>")` for typed text — or `NO_OP` if no action.'
    )

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
        frame_size: tuple[int, int] | None = None,
    ) -> FormatResult:
        primitives, counters = self._window_primitives(
            events, windows, dead_zones, master_fps=master_fps
        )
        held_mods: set[str] = set()
        collapsed = [_collapse_typing(window, held_mods) for window in primitives]
        counts = Counter(p.kind for window in collapsed for p in window)
        return FormatResult(
            labels=[
                "; ".join(p.render() for p in window) if window else "NO_OP"
                for window in collapsed
            ],
            counters=counters,
            primitive_counts={
                k: counts.get(k, 0) for k in ("move", "scroll", "down", "up", "type")
            },
        )


# ---------------------------------------------------------------------------
# computer_use_rel_v1 (Qwen3-VL-native tool_call emission, relative mouse)
# ---------------------------------------------------------------------------

# THE canonical rdev/keylog -> computer-use key-name table (lowercase
# pyautogui-style names, the convention of eval/osworld_vm_client.py's
# _RDEV_TO_PYAUTOGUI / _COMPUTER_USE_KEY_ALIASES — eval imports or mirrors
# this table). Left/right modifier variants collapse to the generic name the
# cua_v4 tool spec uses (["ctrl", "c"]); the capture data is macOS, so Meta*
# is the Command key and maps to "command" (eval maps "command" onto the
# target OS's super/win key). Names not in the table pass through lowercased
# and are counted in primitive_counts["unmapped_key_names"] — never a crash.
RDEV_TO_COMPUTER_USE_KEY: dict[str, str] = {
    # letters and digits (rdev top-row digits are Num*; DOM-style Digit* and
    # keypad Kp*/Keypad* spellings appear in sibling tools/macOS resolutions)
    **{f"Key{c}": c.lower() for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    **{f"Num{d}": d for d in "0123456789"},
    **{f"Digit{d}": d for d in "0123456789"},
    **{f"Kp{d}": d for d in "0123456789"},
    **{f"Keypad{d}": d for d in "0123456789"},
    # whitespace / editing
    "Space": "space",
    "Return": "enter",
    "KpReturn": "enter",
    "Tab": "tab",
    "Backspace": "backspace",
    "Delete": "delete",
    "KpDelete": "delete",
    "ForwardDelete": "delete",  # macOS Unknown(117)
    "Insert": "insert",
    "Escape": "esc",
    # navigation
    "Home": "home",
    "End": "end",
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "UpArrow": "up", "ArrowUp": "up",
    "DownArrow": "down", "ArrowDown": "down",
    "LeftArrow": "left", "ArrowLeft": "left",
    "RightArrow": "right", "ArrowRight": "right",
    # modifiers (generic names; macOS Command == Meta*)
    "ShiftLeft": "shift", "ShiftRight": "shift", "Shift": "shift",
    "ControlLeft": "ctrl", "ControlRight": "ctrl", "Control": "ctrl",
    "Alt": "alt", "AltLeft": "alt", "AltRight": "alt", "AltGr": "alt",
    "MetaLeft": "command", "MetaRight": "command", "Meta": "command",
    "MetaGr": "command",
    "Function": "fn",
    # locks / misc
    "CapsLock": "capslock",
    "NumLock": "numlock",
    "ScrollLock": "scrolllock",
    "PrintScreen": "printscreen",
    "Pause": "pause",
    "Help": "help",  # macOS Unknown(114)
    # punctuation — rdev spellings (SemiColon/Dot/BackQuote/BackSlash, the
    # keylog namespace) plus eval's alternate casings, kept in sync
    "Minus": "-", "Equal": "=",
    "BracketLeft": "[", "BracketRight": "]",
    "BackSlash": "\\", "Backslash": "\\",
    "SemiColon": ";", "Semicolon": ";",
    "Quote": "'",
    "Comma": ",",
    "Dot": ".", "Period": ".",
    "Slash": "/",
    "BackQuote": "`", "Backquote": "`",
    # keypad operators
    "KpMinus": "-", "KpPlus": "+", "KpMultiply": "*", "KpDivide": "/",
    # function keys
    **{f"F{i}": f"f{i}" for i in range(1, 25)},
}

_BUTTON_TO_COMPUTER_USE = {"LMB": "left", "RMB": "right", "MMB": "middle"}
_CLICK_ACTION = {"left": "left_click", "right": "right_click",
                 "middle": "middle_click"}
# Only the LEFT button has multi-click actions in the cua_v4 spec; repeated
# right/middle clicks emit repeated single clicks.
_MULTI_CLICK = {1: "left_click", 2: "double_click", 3: "triple_click"}

# Every per-window action kind computer_use_rel_v1 can emit (terminate only
# ever comes from terminate_line(), never from format_segment).
_COMPUTER_USE_ACTIONS = (
    "mouse_move_rel", "scroll", "hscroll",
    "left_click", "right_click", "middle_click", "double_click", "triple_click",
    "button_down", "button_up",
    "key", "key_down", "key_up", "type", "wait",
)


def _is_button(name: str | None) -> bool:
    return name is not None and (
        name in _BUTTON_TO_COMPUTER_USE or name.startswith("M_")
    )


def render_tool_call(arguments: dict[str, Any]) -> str:
    """One computer_use tool call exactly as cua_v4_thinking.txt binds it:
    cookbook-style JSON (default separators, ensure_ascii=False) wrapped in
    ``<tool_call>`` tags on their own lines."""
    body = json.dumps({"name": "computer_use", "arguments": arguments},
                      ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


class ComputerUseFormatter:
    """Per-window Qwen3-VL-native ``<tool_call>`` blocks (one or more, joined
    by newlines) against the ``computer_use`` function of
    system_prompts/cua_v4_thinking.txt, with a relative mouse.

    Collapse rules (order-preserving):
      * Motion/scroll accumulate between discrete barriers (any press/release
        flushes both) — NO motor grid; each accumulator rounds once at flush
        and is omitted when zero. Scroll emits vertical (``scroll``) before
        horizontal (``hscroll``); move vs scroll keep first-seen order.
      * A button press adjacent to its release is a click; two/three
        back-to-back LEFT clicks collapse to double_click/triple_click. An
        unpaired press/release is button_down/button_up, so a drag renders
        button_down, mouse_move_rel, button_up (never a synthesized
        left_click_drag). Non-L/R/M buttons have no click action: an adjacent
        pair decomposes to button_down + button_up.
      * Typing reuses the v3 printable-pair + Shift-folding machinery
        (``_collapse_typing``) -> ``type`` with the raw text (json escapes).
      * Keys: any adjacent press-in-order/release-in-reverse "mountain" whose
        downs are modifiers plus at most one final non-modifier key renders as
        one ``key([...])`` — a plain pair (``key(["enter"])``), a bare
        modifier tap (``key(["shift"])``), or a chord (``key(["ctrl","c"])``).
        Anything else decomposes exactly: key_down for the opening press,
        inner complete pairs as their own key([...]) (physically correct at
        replay — the modifier is held), key_up for the closing release.
      * An empty window renders ``wait`` for the window's span in seconds.
    """

    name = "computer_use_rel_v1"
    normalize_moves = False
    reply_contract = (
        "Reply with {what} as one or more <tool_call> JSON blocks executed in "
        'order, each `{{"name": "computer_use", "arguments": {{"action": ..., '
        "...}}}}` per the computer_use tool spec."
    )

    def terminate_line(self) -> str:
        return render_tool_call({"action": "terminate", "status": "success"})

    def is_idle_label(self, label: str) -> bool:
        stripped = label.strip()
        if stripped.count("<tool_call>") != 1:
            return False
        try:
            body = stripped.split("<tool_call>", 1)[1].rsplit("</tool_call>", 1)[0]
            call = json.loads(body)
        except (IndexError, ValueError):
            return False
        return call.get("arguments", {}).get("action") == "wait"

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
        frame_size: tuple[int, int] | None = None,
    ) -> FormatResult:
        primitives, counters, n_norm_zero = self._window_primitives(
            events, windows, dead_zones, master_fps=master_fps,
            normalize=self.normalize_moves, frame_size=frame_size,
        )
        held_mods: set[str] = set()
        collapsed = [_collapse_typing(window, held_mods) for window in primitives]
        counts: Counter = Counter()
        labels: list[str] = []
        for win, prims in zip(windows, collapsed, strict=True):
            calls = self._window_tool_calls(prims, counts)
            if not calls:
                span_s = (win.end - win.start) / master_fps
                calls = [{"action": "wait", "time": round(span_s, 3)}]
                counts["wait"] += 1
            labels.append("\n".join(render_tool_call(c) for c in calls))
        primitive_counts = {k: counts.get(k, 0) for k in _COMPUTER_USE_ACTIONS}
        primitive_counts["unmapped_key_names"] = counts.get("unmapped_key_names", 0)
        primitive_counts["unmapped_button_names"] = counts.get(
            "unmapped_button_names", 0)
        primitive_counts["moves_normalized_to_zero"] = n_norm_zero
        return FormatResult(
            labels=labels, counters=counters, primitive_counts=primitive_counts
        )

    @staticmethod
    def _window_primitives(
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
        normalize: bool = False,
        frame_size: tuple[int, int] | None = None,
    ) -> tuple[list[list[ActionPrimitive]], PolicyCounters, int]:
        """Ordered primitives per window with BARRIER-level accumulation:
        move and scroll fold independently (floats, first-seen order) until a
        press/release or the window ends flushes both; rounding happens once
        at flush and zero flushes are omitted.

        With ``normalize``, a MOVE flush is expressed per-axis as a 0-1000
        fraction of ``frame_size`` instead of raw device counts; scroll stays
        raw ticks either way. Because 1000/width < 1 for most captures, that
        rounding can annihilate a move the raw path would have kept, so the
        third return value counts the flushes lost that way."""
        if normalize and (frame_size is None
                          or frame_size[0] <= 0 or frame_size[1] <= 0):
            raise ValueError(
                "a normalized action format needs a valid per-segment "
                f"frame_size=(width,height); got {frame_size!r}")
        labeled, counters = apply_label_policy(
            events, windows, dead_zones, master_fps=master_fps
        )
        primitives: list[list[ActionPrimitive]] = [[] for _ in windows]
        pending: dict[str, list[float]] = {}  # kind -> [dx, dy], insertion-ordered
        cur_win: int | None = None
        n_normalized_to_zero = 0

        def flush() -> None:
            nonlocal n_normalized_to_zero
            for kind, (dx, dy) in pending.items():
                if kind == "move" and normalize:
                    w, h = frame_size
                    rdx, rdy = round(dx / w * 1000), round(dy / h * 1000)
                    if rdx == 0 and rdy == 0 and (round(dx) or round(dy)):
                        n_normalized_to_zero += 1
                else:
                    rdx, rdy = round(dx), round(dy)
                if rdx != 0 or rdy != 0:
                    primitives[cur_win].append(
                        ActionPrimitive(kind=kind, dx=rdx, dy=rdy, owner=cur_win))
            pending.clear()

        for le in _ordered_owned(labeled):
            e = le.event
            if le.window != cur_win:
                if cur_win is not None:
                    flush()
                cur_win = le.window
            if e.kind in ("move", "scroll"):
                acc = pending.setdefault(e.kind, [0.0, 0.0])
                acc[0] += e.dx
                acc[1] += e.dy
            else:
                flush()
                primitives[cur_win].append(
                    ActionPrimitive(kind="down" if e.kind == "press" else "up",
                                    input_name=e.name, t_s=le.label_t, owner=cur_win))
        if cur_win is not None:
            flush()
        return primitives, counters, n_normalized_to_zero

    def _window_tool_calls(
        self, prims: Sequence[ActionPrimitive], counts: Counter
    ) -> list[dict[str, Any]]:
        """One window's collapsed primitives -> ordered tool-call arguments."""
        calls: list[dict[str, Any]] = []

        def add(args: dict[str, Any]) -> None:
            counts[args["action"]] += 1
            calls.append(args)

        def key_name(name: str) -> str:
            mapped = RDEV_TO_COMPUTER_USE_KEY.get(name)
            if mapped is None:
                counts["unmapped_key_names"] += 1
                return name.lower()
            return mapped

        def button_name(name: str) -> str:
            mapped = _BUTTON_TO_COMPUTER_USE.get(name)
            if mapped is None:
                counts["unmapped_button_names"] += 1
                return name.lower()
            return mapped

        i = 0
        while i < len(prims):
            p = prims[i]
            if p.kind == "move":
                add({"action": "mouse_move_rel", "delta": [p.dx, p.dy]})
                i += 1
            elif p.kind == "scroll":
                if p.dy != 0:
                    add({"action": "scroll", "pixels": p.dy})
                if p.dx != 0:
                    add({"action": "hscroll", "pixels": p.dx})
                i += 1
            elif p.kind == "type":
                add({"action": "type", "text": p.text})
                i += 1
            elif _is_button(p.input_name):
                i = self._emit_button(prims, i, add, button_name)
            elif p.kind == "down":
                i = self._emit_key_down(prims, i, add, key_name)
            else:  # unmatched key release (its press lies in an earlier window)
                add({"action": "key_up", "key": key_name(p.input_name)})
                i += 1
        return calls

    @staticmethod
    def _emit_button(
        prims: Sequence[ActionPrimitive],
        i: int,
        add: Callable[[dict[str, Any]], None],
        button_name: Callable[[str], str],
    ) -> int:
        p = prims[i]
        btn = button_name(p.input_name)
        if p.kind == "up":
            add({"action": "button_up", "button": btn})
            return i + 1

        def is_pair(j: int) -> bool:
            return (j + 1 < len(prims)
                    and prims[j].kind == "down" and prims[j].input_name == p.input_name
                    and prims[j + 1].kind == "up"
                    and prims[j + 1].input_name == p.input_name)

        if is_pair(i):
            if btn in _CLICK_ACTION:
                reps = 1
                if btn == "left":
                    while reps < 3 and is_pair(i + 2 * reps):
                        reps += 1
                add({"action": _MULTI_CLICK[reps] if btn == "left"
                     else _CLICK_ACTION[btn]})
                return i + 2 * reps
            # non-standard button: the spec has no click action for it
            add({"action": "button_down", "button": btn})
            add({"action": "button_up", "button": btn})
            return i + 2
        add({"action": "button_down", "button": btn})
        return i + 1

    @staticmethod
    def _emit_key_down(
        prims: Sequence[ActionPrimitive],
        i: int,
        add: Callable[[dict[str, Any]], None],
        key_name: Callable[[str], str],
    ) -> int:
        """Match the press-in-order/release-in-reverse mountain at ``i``:
        consecutive modifier downs, at most one final non-modifier down, then
        the ups in exact reverse order — all adjacent. On a match the whole
        scope is one ``key([...])``; otherwise only prims[i] renders (as
        key_down) and scanning resumes, which yields the exact decomposition
        for multi-pair scopes and scopes broken by anything non-keyboard."""
        n = len(prims)
        downs: list[str] = []
        j = i
        while (j < n and prims[j].kind == "down"
               and not _is_button(prims[j].input_name)
               and prims[j].input_name in _MODIFIER_KEYS):
            downs.append(prims[j].input_name)
            j += 1
        if (j < n and prims[j].kind == "down"
                and not _is_button(prims[j].input_name)):
            downs.append(prims[j].input_name)
            j += 1
        k = j
        matched = bool(downs)
        for name in reversed(downs):
            if k < n and prims[k].kind == "up" and prims[k].input_name == name:
                k += 1
            else:
                matched = False
                break
        if matched:
            add({"action": "key", "keys": [key_name(name) for name in downs]})
            return k
        add({"action": "key_down", "key": key_name(prims[i].input_name)})
        return i + 1


class ComputerUseRelNormFormatter(ComputerUseFormatter):
    """``computer_use_rel_v1`` with ``mouse_move_rel`` deltas normalized to a
    resolution-independent 0-1000 scale — per-axis ``round(dx / W * 1000)``,
    ``round(dy / H * 1000)`` against the ORIGINAL capture size, so 1000 is one
    full screen width for dx and one full height for dy. Scroll stays raw
    ticks. Requires a per-segment ``frame_size`` from the clips manifest.

    Only the emitted magnitudes differ from the parent: the collapse rules,
    tool-call vocabulary, terminate block and idle ``wait`` are identical."""

    name = "computer_use_rel_norm_v1"
    normalize_moves = True
    reply_contract = (
        ComputerUseFormatter.reply_contract
        + " mouse_move_rel delta is on a 0-1000 scale (1000 == full screen "
        "width for dx, height for dy), NOT pixels."
    )


class ComputerUseRelStepFormatter(ComputerUseFormatter):
    """Semantic, closed-loop relative actions with a finite movement codebook.

    Unlike ``ordered_events_v2`` this formatter has no motor clock: all raw
    movement in an observation window becomes at most one direction decision.
    Recorded magnitude is discarded.  Consecutive movement windows are
    labelled coarse -> medium -> fine so deployment can take one fixed step,
    observe a fresh screenshot, and correct its aim.

    Printable balanced key runs use the inherited rollover-aware ``type``
    collapse.  Dangling key/button transitions are never emitted: ordinary
    input is atomic within one assistant response, and only a balanced
    button_down/move/button_up drag may contain explicit button state.
    """

    name = "computer_use_rel_step_v1"
    normalize_moves = False
    reply_contract = (
        "Reply with one or more native <tool_call> JSON blocks per the "
        "computer_use_rel_step_v1 tool spec. Normal mouse movement must be "
        "exactly one fixed relative step; ordinary typing uses type(text)."
    )

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
        frame_size: tuple[int, int] | None = None,
    ) -> FormatResult:
        del frame_size  # fixed screen fractions need no source-device scale
        primitives, counters, _ = self._window_primitives(
            events,
            windows,
            dead_zones,
            master_fps=master_fps,
            normalize=False,
            frame_size=None,
        )

        # Collapse a balanced printable burst across decision-window boundaries
        # and assign the atomic type(text) to the window where the burst began.
        # This avoids turning natural typing into 4 Hz one-character actions.
        flat = [primitive for window in primitives for primitive in window]
        collapsed_flat = _collapse_typing(flat, set(), max_gap_s=TYPING_GAP_S)
        collapsed: list[list[ActionPrimitive]] = [[] for _ in windows]
        for primitive in collapsed_flat:
            if primitive.owner is None:
                raise AssertionError(f"relative-step primitive lost window owner: {primitive}")
            collapsed[primitive.owner].append(primitive)

        move_vectors: list[tuple[float, float] | None] = []
        for window in collapsed:
            dx = sum(p.dx for p in window if p.kind == "move")
            dy = sum(p.dy for p in window if p.kind == "move")
            move_vectors.append((dx, dy) if dx != 0 or dy != 0 else None)

        scales: dict[int, int] = {}
        i = 0
        while i < len(move_vectors):
            if move_vectors[i] is None:
                i += 1
                continue
            j = i + 1
            while j < len(move_vectors) and move_vectors[j] is not None:
                j += 1
            run = list(range(i, j))
            if len(run) == 1:
                scales[run[0]] = MOVEMENT_SCALES[1]
            elif len(run) == 2:
                scales[run[0]] = MOVEMENT_SCALES[-1]
                scales[run[1]] = MOVEMENT_SCALES[0]
            else:
                scales[run[0]] = MOVEMENT_SCALES[-1]
                scales[run[-1]] = MOVEMENT_SCALES[0]
                for idx in run[1:-1]:
                    scales[idx] = MOVEMENT_SCALES[1]
            i = j

        labels: list[str] = []
        counts: Counter = Counter()
        dropped_unbalanced = 0
        dropped_competing_move = 0
        for wi, window in enumerate(collapsed):
            rebuilt: list[ActionPrimitive] = []
            inserted_move = False
            for p in window:
                if p.kind != "move":
                    rebuilt.append(p)
                    continue
                if inserted_move:
                    continue
                inserted_move = True
                vector = move_vectors[wi]
                if vector is not None:
                    delta = rel_step_delta(*vector, scales[wi])
                    if delta is not None:
                        rebuilt.append(ActionPrimitive(
                            kind="move", dx=delta[0], dy=delta[1], owner=wi
                        ))

            raw_calls = self._window_tool_calls(rebuilt, counts)

            # This format never exposes keyboard holds.  Button holds are kept
            # only for a complete same-response drag.
            if any(c["action"] in ("key_down", "key_up") for c in raw_calls):
                dropped_unbalanced += sum(
                    c["action"] in ("key_down", "key_up") for c in raw_calls
                )
                raw_calls = [
                    c for c in raw_calls if c["action"] not in ("key_down", "key_up")
                ]

            downs = [c for c in raw_calls if c["action"] == "button_down"]
            ups = [c for c in raw_calls if c["action"] == "button_up"]
            is_drag = (
                len(downs) == len(ups) == 1
                and downs[0].get("button") == ups[0].get("button")
                and raw_calls[0] is downs[0]
                and raw_calls[-1] is ups[0]
                and all(
                    c["action"] == "mouse_move_rel" for c in raw_calls[1:-1]
                )
                and len(raw_calls) >= 3
            )
            if (downs or ups) and not is_drag:
                dropped_unbalanced += len(downs) + len(ups)
                raw_calls = [
                    c for c in raw_calls
                    if c["action"] not in ("button_down", "button_up")
                ]

            if not is_drag:
                discrete = [
                    c for c in raw_calls
                    if c["action"] != "mouse_move_rel"
                ]
                if discrete and any(c["action"] == "mouse_move_rel" for c in raw_calls):
                    dropped_competing_move += 1
                    raw_calls = [c for c in raw_calls if c["action"] != "mouse_move_rel"]

            calls: list[dict[str, Any]] = []
            for call in raw_calls:
                action = call["action"]
                if action in ("scroll", "hscroll"):
                    value = int(call["pixels"])
                    if value:
                        # Direction only; one wheel step is portable across
                        # mouse wheels and trackpads.  The prompt calls this
                        # field ``steps`` to match what the VM executes.
                        step = 1 if value > 0 else -1
                        if step not in SCROLL_STEPS:  # defensive contract check
                            raise AssertionError(step)
                        calls.append({"action": action, "steps": step})
                elif action == "wait":
                    calls.append({"action": "wait"})
                elif action == "type":
                    text = str(call.get("text") or "")
                    calls.extend(
                        {"action": "type", "text": text[k:k + TYPING_MAX_CHARS]}
                        for k in range(0, len(text), TYPING_MAX_CHARS)
                    )
                else:
                    calls.append(call)

            if not calls:
                calls = [{"action": "wait"}]
            labels.append("\n".join(render_tool_call(c) for c in calls))

        primitive_counts = {k: counts.get(k, 0) for k in _COMPUTER_USE_ACTIONS}
        primitive_counts.update({
            "dropped_unbalanced_transitions": dropped_unbalanced,
            "dropped_competing_move": dropped_competing_move,
            "ten_hz_motor_ticks": 0,
        })
        return FormatResult(labels=labels, counters=counters, primitive_counts=primitive_counts)


FORMATTERS: dict[str, Callable[[float], ActionFormatter]] = {
    CanonicalFormatter.name: lambda hz: CanonicalFormatter(),
    OrderedFormatter.name: OrderedFormatter,
    OrderedTypingFormatter.name: OrderedTypingFormatter,
    ComputerUseFormatter.name: lambda hz: ComputerUseFormatter(),
    ComputerUseRelNormFormatter.name: lambda hz: ComputerUseRelNormFormatter(),
    ComputerUseRelStepFormatter.name: lambda hz: ComputerUseRelStepFormatter(),
}


def get_formatter(
    name: str, *, continuous_action_hz: float = DEFAULT_CONTINUOUS_ACTION_HZ
) -> ActionFormatter:
    try:
        factory = FORMATTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown action format {name!r} (available: {sorted(FORMATTERS)})"
        ) from None
    return factory(continuous_action_hz)
