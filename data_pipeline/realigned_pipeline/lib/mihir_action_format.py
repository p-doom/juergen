"""Convert crowd-cast *canonical* assistant-turn action strings to the p-doom
Inverse-Dynamics-Model (IDM) action format authored by Mihir.

Ours (see ``lib.action_format.CanonicalFormatter`` / ``lib.common.format_action``
and ``eval/action_parser.parse_action``) is a per-turn line::

    NO_OP                              no action this turn
    <dx> <dy> <scroll>                 mouse move + scroll, RAW display-point
                                       deltas (dx,dy) + wheel-tick scroll
    <dx> <dy> <scroll> ; +K1 -K2 ...   plus ordered key/button transitions
                                       (LMB/RMB/MMB + rdev key names KeyA, Return,
                                       ShiftLeft, MetaLeft, ...); '+' press '-' release
    TERMINATE                          BC-only episode-end control token

Mihir's IDM (verified against ``inverse-dynamics-model`` @ prepare_data.py,
actions.py, prompts.py, data.py) is a JSON array of action objects, one per
detected event, each ``{"frame": "FNN", "type": T, "details": D}`` with the four
types:

  * MouseMove   details = "dx_n,dy_n"  normalized integers, dx_n = round(dx/W*1000),
                dy_n = round(dy/H*1000) — a signed screen-fraction on 0..1000
                (+dx right, +dy down). (actions.py:norm1000 / extract_mouse_gt)
  * MouseClick  details = "Left"|"Right"|"Middle" — emitted on the press;
                the release is dropped, buttons are NOT tracked as held.
                (prepare_data.py:parse_keylog_events / _parse_button)
  * KeyPress    details = key name with the currently-held modifiers folded in,
                e.g. "A", "Cmd+C", "Shift+A", "Enter"; bare modifier presses emit
                nothing. (prepare_data.py:normalize_key_name / format_key_with_modifiers)
  * MouseScroll details = ONE normalized integer = round(-scroll/H*1000); Mihir
                negates the raw wheel delta so +details = scroll DOWN.
                (actions.py:extract_mouse_gt / prompts.py)

Serialization is COMPACT json (``json.dumps(actions, separators=(",", ":"))``,
object key order frame,type,details) and the empty action set is ``[]``.
(data.py:ProcessedClipDataset._process_clip / build_sft_messages)

FRAME LABEL: Mihir's IDM predicts a whole 10-frame window, so "frame" locates an
event within it. Alfred's BC conversation is ONE frame per assistant turn, so the
frame index is degenerate here — every object gets the constant ``F00``. This
keeps Mihir's exact object *schema* (frame/type/details) while fitting the
per-turn structure; the thing being ablated (the action *representation*) is
faithful.

Held-modifier state is threaded ACROSS the turns of one conversation (a
conversation ~= one of Mihir's clips), so a Shift pressed in an earlier turn and
released in a later one folds into the key pressed in between — matching Mihir's
clip-level ``held_modifiers`` fold.

LOSSY / adapted mappings (documented, and counted at runtime):
  * TERMINATE (and any configured terminal token) is passed through verbatim —
    Mihir's taxonomy has no terminate. Keeping it preserves alfred's terminate
    supervision unchanged; it is intentionally NOT a Mihir action object.
  * scroll magnitude: our on-disk label is already the rounded per-window sum of
    wheel deltas, so the normalized value is derived from that (Mihir sums the
    same raw deltas per frame — same quantity modulo our rounding).
  * mouse-move rounding boundary: a raw sub-pixel move our canonical label
    already rounded to 0 is a NO_OP for us but would be a "0,0" MouseMove for
    Mihir; we can only convert from the rounded label, so those are NO_OP.
  * extra mouse buttons (resolve_button_name "M_*") have no Left/Right/Middle
    slot in Mihir's taxonomy and are dropped.
  * truly-unknown macOS keys (our "KC_<code>", = Mihir's "Unknown(...)" which his
    data.py normalize_actions strips as "no visual cue") are dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

NORM_SCALE = 1000

# The single constant frame label for per-turn (single-frame) conversion.
FRAME_LABEL = "F00"

# --------------------------------------------------------------------------
# Key / button name normalization — ported EXACTLY from Mihir's
# inverse-dynamics-model/prepare_data.py (KEY_NAME_MAP, normalize_key_name,
# format_key_with_modifiers) and data.py (normalize_actions detail remap).
# --------------------------------------------------------------------------

KEY_NAME_MAP = {
    "Return": "Return",
    "Escape": "Escape",
    "Backspace": "Backspace",
    "Space": "Space",
    "Tab": "Tab",
    "ShiftLeft": "Shift",
    "ShiftRight": "Shift",
    "ControlLeft": "Ctrl",
    "ControlRight": "Ctrl",
    "Alt": "Alt",
    "AltLeft": "Alt",
    "AltRight": "Alt",
    "AltGr": "AltGr",
    "MetaLeft": "Cmd",
    "MetaRight": "Cmd",
    "UpArrow": "UpArrow",
    "DownArrow": "DownArrow",
    "LeftArrow": "LeftArrow",
    "RightArrow": "RightArrow",
    "CapsLock": "CapsLock",
    "Delete": "Delete",
    "Home": "Home",
    "End": "End",
    "PageUp": "PageUp",
    "PageDown": "PageDown",
}

# Keys that modify other keys rather than producing their own action.
MODIFIER_KEYS = {"Shift", "Ctrl", "Alt", "AltGr", "Cmd"}

# Combo order used by Mihir's format_key_with_modifiers (note: AltGr, though a
# modifier, is intentionally NOT in this list — replicated verbatim).
_MODIFIER_ORDER = ["Cmd", "Ctrl", "Alt", "Shift"]

# data.py:normalize_actions key-name remap (applied per '+'-split component).
_DETAIL_KEY_NORMALIZE = {
    "SemiColon": "Semicolon",
    "BackSlash": "Backslash",
    "BackQuote": "Backtick",
}

# Our mouse-button names (common.resolve_button_name) -> Mihir click details.
_MOUSE_BUTTON_TO_CLICK = {"LMB": "Left", "RMB": "Right", "MMB": "Middle"}

# Per-frame serialization order (data.py / parse_unified_events type_order).
_TYPE_ORDER = {"KeyPress": 0, "MouseClick": 1, "MouseScroll": 2, "MouseMove": 3}

_EVENT_RE = re.compile(r"^([+-])(\S+)$")


def normalize_key_name(raw: str) -> str:
    """Map raw rdev key names (e.g. 'KeyA', 'MetaLeft') to display names."""
    if raw in KEY_NAME_MAP:
        return KEY_NAME_MAP[raw]
    if raw.startswith("Key") and len(raw) == 4:
        return raw[3].upper()
    if raw.startswith("Digit") and len(raw) == 6:
        return raw[5]
    if raw.startswith("F") and raw[1:].isdigit():
        return raw  # F1, F2, ...
    return raw


def format_key_with_modifiers(key: str, held_modifiers: set[str]) -> str:
    """Combine held modifiers with a key press, e.g. 'Cmd+C'."""
    if key in MODIFIER_KEYS:
        return key
    parts = [mod for mod in _MODIFIER_ORDER if mod in held_modifiers]
    parts.append(key)
    return "+".join(parts)


def _normalize_detail(detail: str) -> str:
    """data.py:normalize_actions per-component remap (SemiColon -> Semicolon...)."""
    return "+".join(_DETAIL_KEY_NORMALIZE.get(p, p) for p in detail.split("+"))


@dataclass
class ConversionCounters:
    n_turns: int = 0
    n_noop: int = 0
    n_terminal: int = 0
    n_empty_array: int = 0
    n_mousemove: int = 0
    n_mouseclick: int = 0
    n_keypress: int = 0
    n_mousescroll: int = 0
    n_dropped_unknown_button: int = 0
    n_dropped_unknown_key: int = 0
    n_parse_errors: int = 0

    def merge(self, other: "ConversionCounters") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))


def _parse_canonical(text: str) -> tuple[int, int, int, list[tuple[str, str]]]:
    """Parse ``<dx> <dy> <scroll> [; +K -K ...]`` into (dx, dy, scroll, events).

    events is an ordered list of (sign, name), sign in {"+","-"}. Raises
    ValueError on anything that isn't the canonical mouse+events grammar.
    """
    mouse_part, _, key_part = text.partition(";")
    mouse_tokens = mouse_part.split()
    if len(mouse_tokens) != 3:
        raise ValueError(f"expected 3 mouse tokens, got {mouse_tokens!r}")
    dx, dy, scroll = (int(t) for t in mouse_tokens)
    events: list[tuple[str, str]] = []
    for tok in key_part.split():
        m = _EVENT_RE.match(tok)
        if not m:
            raise ValueError(f"malformed event token: {tok!r}")
        events.append((m.group(1), m.group(2)))
    return dx, dy, scroll, events


def _convert_action_line(
    line: str,
    held_modifiers: set[str],
    *,
    video_w: int,
    video_h: int,
    counters: ConversionCounters,
) -> str:
    """Convert one canonical action line (no terminal token) to a Mihir JSON
    array string. Mutates ``held_modifiers`` (folded modifier state)."""
    if line == "NO_OP":
        counters.n_noop += 1
        return "[]"

    dx, dy, scroll, events = _parse_canonical(line)

    key_objs: list[dict] = []
    click_objs: list[dict] = []
    seen_keys: set[str] = set()
    seen_clicks: set[str] = set()

    for sign, name in events:
        if name in _MOUSE_BUTTON_TO_CLICK:
            if sign == "+":  # click on press; release ignored, buttons not held
                button = _MOUSE_BUTTON_TO_CLICK[name]
                if button not in seen_clicks:
                    seen_clicks.add(button)
                    click_objs.append(
                        {"frame": FRAME_LABEL, "type": "MouseClick", "details": button}
                    )
            continue
        if name.startswith("M_"):  # extra mouse button, no Mihir slot
            if sign == "+":
                counters.n_dropped_unknown_button += 1
            continue
        # keyboard key
        key = normalize_key_name(name)
        if key in MODIFIER_KEYS:
            if sign == "+":
                held_modifiers.add(key)
            else:
                held_modifiers.discard(key)
            continue
        if sign != "+":  # non-modifier release emits nothing
            continue
        if key.startswith("KC_"):  # unknown macOS key, no visual cue -> drop
            counters.n_dropped_unknown_key += 1
            continue
        detail = _normalize_detail(format_key_with_modifiers(key, held_modifiers))
        if detail not in seen_keys:
            seen_keys.add(detail)
            key_objs.append(
                {"frame": FRAME_LABEL, "type": "KeyPress", "details": detail}
            )

    scroll_objs: list[dict] = []
    if scroll != 0:
        scroll_n = round(-scroll / video_h * NORM_SCALE)
        scroll_objs.append(
            {"frame": FRAME_LABEL, "type": "MouseScroll", "details": str(scroll_n)}
        )

    move_objs: list[dict] = []
    if dx != 0 or dy != 0:
        dx_n = round(dx / video_w * NORM_SCALE)
        dy_n = round(dy / video_h * NORM_SCALE)
        move_objs.append(
            {"frame": FRAME_LABEL, "type": "MouseMove", "details": f"{dx_n},{dy_n}"}
        )

    objs = key_objs + click_objs + scroll_objs + move_objs
    objs.sort(key=lambda o: _TYPE_ORDER[o["type"]])  # stable: KP, MC, MS, MM

    counters.n_keypress += len(key_objs)
    counters.n_mouseclick += len(click_objs)
    counters.n_mousescroll += len(scroll_objs)
    counters.n_mousemove += len(move_objs)
    if not objs:
        counters.n_empty_array += 1
    return json.dumps(objs, separators=(",", ":"))


def convert_turn(
    text: str,
    held_modifiers: set[str],
    *,
    video_w: int,
    video_h: int,
    counters: ConversionCounters,
    terminal_tokens: tuple[str, ...] = ("TERMINATE",),
) -> str:
    """Convert one assistant-turn text to Mihir format. Mutates ``held_modifiers``.

    Handles NO_OP -> "[]", a standalone terminal token (passthrough), the
    appended ``<action>\\n<token>`` form, and the plain action line.
    """
    counters.n_turns += 1
    s = text.strip()
    if not s:
        counters.n_parse_errors += 1
        counters.n_empty_array += 1
        return "[]"
    if s in terminal_tokens:
        counters.n_terminal += 1
        return s
    if "\n" in s:
        head, _, tail = s.rpartition("\n")
        tail = tail.strip()
        if tail in terminal_tokens:
            counters.n_terminal += 1
            conv = _convert_action_line(
                head.strip(),
                held_modifiers,
                video_w=video_w,
                video_h=video_h,
                counters=counters,
            )
            return f"{conv}\n{tail}"
        # An unexpected multi-line turn: convert only the first line (parse the
        # rest would be undefined); this mirrors parse_action cutting at newline.
        s = s.split("\n", 1)[0].strip()
    return _convert_action_line(
        s, held_modifiers, video_w=video_w, video_h=video_h, counters=counters
    )


def convert_conversation(
    messages: list[dict],
    *,
    video_w: int,
    video_h: int,
    counters: ConversionCounters,
    terminal_tokens: tuple[str, ...] = ("TERMINATE",),
) -> None:
    """In-place rewrite of every assistant turn's action text in one
    conversation. Held-modifier state is threaded across the turns."""
    held_modifiers: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not (
            isinstance(content, list)
            and len(content) == 1
            and isinstance(content[0], dict)
            and content[0].get("type") == "text"
        ):
            continue
        content[0]["text"] = convert_turn(
            content[0]["text"],
            held_modifiers,
            video_w=video_w,
            video_h=video_h,
            counters=counters,
            terminal_tokens=terminal_tokens,
        )
