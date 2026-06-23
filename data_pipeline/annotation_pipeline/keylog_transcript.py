#!/usr/bin/env python3
"""Reconstruct a human-readable input transcript from a msgpack keylog.

The old pipeline used the keylog only for per-frame action bins and an
activity pre-gate; the *content* of what the user typed was thrown away and the
VLM had to guess it from sparse pixels. That is the main reason instructions
came out "plainly wrong".

This module recovers the signal the keylog actually carries:

- **typing bursts** -> the literal text the user typed, with Backspace applied
  and Shift handled (best-effort US-layout; flagged ``approx`` because non-US
  layouts and arrow-key edits cannot be reconstructed exactly from physical
  key codes alone). For terminals / code / English chat this is exact and is
  the user's intent verbatim.
- **chords** -> modifier combos (Cmd+S, Ctrl+C, Cmd+Shift+P, ...).
- **mouse bursts** -> click counts + scroll totals/direction with timing.
  Mouse coordinates in this dataset are deltas only (no absolute position), so
  *where* a click lands still requires vision.
- **app switches** -> the focused application from ``ContextChanged`` bundle ids.

Times are seconds from the segment start; with a single-segment stage-00/01
manifest (``segment_offset_s == 0``) they line up 1:1 with the
``global_time_s`` on stage-01 frame records, so a transcript slice and a frame
window refer to the same wall-clock interval.

CLI (spot-check):
    python -m annotation_pipeline.keylog_transcript --keylog PATH [--start 0 --end 60]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from annotation_pipeline.common import load_keylog_entries, resolve_key_name

# ---------------------------------------------------------------------------
# Key maps (rdev-style names, with a few W3C aliases normalized in)
# ---------------------------------------------------------------------------

# Names that, while held, turn a printable key into a shortcut rather than text.
_SUPPRESS_MODIFIERS = {
    "ControlLeft", "ControlRight", "MetaLeft", "MetaRight", "Alt", "AltGr",
}
_SHIFTS = {"ShiftLeft", "ShiftRight"}

# Normalize alternate spellings (W3C KeyboardEvent.code, macOS variants) to the
# rdev names used by the rest of this map.
_NAME_ALIASES = {
    "Period": "Dot", "BracketLeft": "LeftBracket", "BracketRight": "RightBracket",
    "Backslash": "BackSlash", "Semicolon": "SemiColon", "Apostrophe": "Quote",
    "Backquote": "BackQuote", "Enter": "Return", "Esc": "Escape",
    "ControlRight": "ControlLeft",  # treat for chord naming only via _MOD_LABEL
}
for _i in range(10):  # Digit3 -> Num3, Digit0 -> Num0
    _NAME_ALIASES[f"Digit{_i}"] = f"Num{_i}"

_UNSHIFTED = {
    "Minus": "-", "Equal": "=", "LeftBracket": "[", "RightBracket": "]",
    "BackSlash": "\\", "SemiColon": ";", "Quote": "'", "BackQuote": "`",
    "Comma": ",", "Dot": ".", "Slash": "/", "Space": " ",
    "Num1": "1", "Num2": "2", "Num3": "3", "Num4": "4", "Num5": "5",
    "Num6": "6", "Num7": "7", "Num8": "8", "Num9": "9", "Num0": "0",
}
_SHIFTED = {
    "Minus": "_", "Equal": "+", "LeftBracket": "{", "RightBracket": "}",
    "BackSlash": "|", "SemiColon": ":", "Quote": '"', "BackQuote": "~",
    "Comma": "<", "Dot": ">", "Slash": "?", "Space": " ",
    "Num1": "!", "Num2": "@", "Num3": "#", "Num4": "$", "Num5": "%",
    "Num6": "^", "Num7": "&", "Num8": "*", "Num9": "(", "Num0": ")",
}
for _i in range(10):  # keypad digits
    _UNSHIFTED[f"Kp{_i}"] = str(_i)
    _SHIFTED[f"Kp{_i}"] = str(_i)

# Keys that edit/move within a typing burst and make exact reconstruction
# impossible; we apply Backspace, flag the rest as approximate.
_EDIT_KEYS = {"LeftArrow", "RightArrow", "UpArrow", "DownArrow", "Home", "End",
              "PageUp", "PageDown", "Delete", "ForwardDelete"}

_MOD_LABEL = {
    "MetaLeft": "Cmd", "MetaRight": "Cmd", "ControlLeft": "Ctrl",
    "ControlRight": "Ctrl", "Alt": "Opt", "AltGr": "Opt",
    "ShiftLeft": "Shift", "ShiftRight": "Shift",
}

# Best-effort app bundle id -> friendly name. Extend freely; unknown ids pass
# through verbatim so nothing is silently lost.
_BUNDLE_APP = {
    "com.todesktop.230313mzl4w4u92": "Cursor",
    "com.microsoft.VSCode": "VS Code",
    "com.microsoft.VSCodeInsiders": "VS Code Insiders",
    "dev.zed.Zed": "Zed",
    "com.google.Chrome": "Chrome",
    "org.mozilla.firefox": "Firefox",
    "com.apple.Safari": "Safari",
    "company.thebrowser.Browser": "Arc",
    "com.apple.Terminal": "Terminal",
    "com.googlecode.iterm2": "iTerm2",
    "net.kovidgoyal.kitty": "kitty",
    "com.mitchellh.ghostty": "Ghostty",
    "com.openai.chat": "ChatGPT",
    "com.openai.codex": "Codex",
    "com.anthropic.claudefordesktop": "Claude",
    "notion.id": "Notion",
    "com.apple.Preview": "Preview",
    "com.apple.finder": "Finder",
    "org.zotero.zotero": "Zotero",
    "com.google.antigravity": "Antigravity",
    "com.apple.calculator": "Calculator",
    "UNCAPTURED": "(uncaptured window)",
}


def app_name(bundle: str) -> str:
    return _BUNDLE_APP.get(bundle, bundle)


def friendly_chord(held: set[str], key_name: str) -> str:
    order = ["Cmd", "Ctrl", "Opt", "Shift"]
    mods = sorted(
        {_MOD_LABEL[m] for m in held if m in _MOD_LABEL},
        key=lambda m: order.index(m) if m in order else 99,
    )
    key = _key_label(key_name)
    return "+".join(mods + [key])


def _key_label(name: str) -> str:
    name = _NAME_ALIASES.get(name, name)
    if name.startswith("Key") and len(name) == 4:
        return name[3].upper()
    if name in _UNSHIFTED:
        return _UNSHIFTED[name]
    return name  # Return, Tab, Escape, F1, arrows, etc.


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class Event:
    t_s: float
    kind: str  # "app" | "type" | "chord" | "mouse"
    t_end_s: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def overlaps(self, start_s: float, end_s: float) -> bool:
        end = self.t_end_s if self.t_end_s else self.t_s
        return self.t_s <= end_s and end >= start_s


@dataclass
class Transcript:
    events: list[Event]
    segment_offset_s: float = 0.0

    def slice(self, start_s: float, end_s: float) -> list[Event]:
        return [e for e in self.events if e.overlaps(start_s, end_s)]

    def render(
        self,
        start_s: float | None = None,
        end_s: float | None = None,
        max_text_chars: int = 600,
    ) -> str:
        lo = -1e18 if start_s is None else start_s
        hi = 1e18 if end_s is None else end_s
        lines: list[str] = []
        for e in self.events:
            if not e.overlaps(lo, hi):
                continue
            if e.kind == "app":
                lines.append(f"[{e.t_s:7.1f}s] APP -> {e.data['app']}")
            elif e.kind == "chord":
                lines.append(f"[{e.t_s:7.1f}s] KEY  {e.data['chord']}")
            elif e.kind == "type":
                text = e.data["text"]
                shown = text if len(text) <= max_text_chars else text[:max_text_chars] + "…"
                approx = " ~approx" if e.data.get("approx") else ""
                lines.append(
                    f"[{e.t_s:7.1f}–{e.t_end_s:.1f}s] TYPE{approx} "
                    f"({e.data['n_keys']} keys): {shown!r}"
                )
            elif e.kind == "mouse":
                d = e.data
                bits = []
                if d.get("clicks"):
                    bits.append(
                        ", ".join(f"{n}x {btn}" for btn, n in sorted(d["clicks"].items()))
                    )
                if d.get("scroll"):
                    bits.append(f"scroll {d['scroll_dir']} ~{abs(int(d['scroll']))}")
                if not bits:
                    bits.append("move")
                lines.append(
                    f"[{e.t_s:7.1f}–{e.t_end_s:.1f}s] MOUSE {'; '.join(bits)}"
                )
        return "\n".join(lines) if lines else "(no input events in interval)"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _payload_name(event: list[Any]) -> str | None:
    """Key name for KeyPress/KeyRelease, via common.resolve_key_name."""
    payload = event[1] if len(event) > 1 else None
    return resolve_key_name(payload)


def build_transcript(
    keylog_path: Path,
    segment_offset_s: float = 0.0,
    typing_gap_s: float = 2.5,
    mouse_gap_s: float = 1.5,
) -> Transcript:
    entries = load_keylog_entries(Path(keylog_path))
    events: list[Event] = []

    held: set[str] = set()
    caps = False

    # Open typing burst.
    buf: list[str] = []
    buf_keys = 0
    buf_start = 0.0
    buf_last = 0.0
    buf_approx = False

    def flush_typing() -> None:
        nonlocal buf, buf_keys, buf_approx
        text = "".join(buf)
        if text.strip("\n") or buf_keys:
            events.append(Event(
                t_s=buf_start, kind="type", t_end_s=buf_last,
                data={"text": text, "n_keys": buf_keys, "approx": buf_approx},
            ))
        buf = []
        buf_keys = 0
        buf_approx = False

    # Open mouse burst.
    m_start = m_last = 0.0
    m_clicks: dict[str, int] = {}
    m_scroll = 0.0
    m_open = False

    def flush_mouse() -> None:
        nonlocal m_clicks, m_scroll, m_open
        if m_open and (m_clicks or abs(m_scroll) >= 1):
            events.append(Event(
                t_s=m_start, kind="mouse", t_end_s=m_last,
                data={
                    "clicks": dict(m_clicks),
                    "scroll": round(m_scroll, 1),
                    "scroll_dir": "down" if m_scroll < 0 else "up",
                },
            ))
        m_clicks = {}
        m_scroll = 0.0
        m_open = False

    for entry in entries:
        if not (isinstance(entry, list) and len(entry) >= 2):
            continue
        try:
            t = segment_offset_s + int(entry[0]) / 1_000_000
        except (TypeError, ValueError):
            continue
        ev = entry[1]
        if not (isinstance(ev, list) and ev):
            continue
        etype = str(ev[0])

        if etype == "ContextChanged":
            flush_typing(); flush_mouse()
            pl = ev[1] if len(ev) > 1 else None
            bundle = str(pl[0]) if isinstance(pl, list) and pl else str(pl)
            events.append(Event(t_s=t, kind="app", data={"app": app_name(bundle), "bundle": bundle}))
            continue

        if etype in ("MouseMove", "MouseScroll", "MousePress", "MouseRelease"):
            # A mouse action longer than mouse_gap_s after the last one starts a
            # new burst; typing is flushed when the mouse takes over.
            if m_open and t - m_last > mouse_gap_s:
                flush_mouse()
            if not m_open:
                m_open = True
                m_start = t
            m_last = t
            if etype == "MousePress":
                pl = ev[1] if len(ev) > 1 else None
                btn = pl[0] if isinstance(pl, list) and pl else "?"
                btn = {"Left": "LMB", "Right": "RMB", "Middle": "MMB"}.get(str(btn), str(btn))
                m_clicks[btn] = m_clicks.get(btn, 0) + 1
            elif etype == "MouseScroll":
                pl = ev[1] if len(ev) > 1 else None
                if isinstance(pl, list) and len(pl) >= 2:
                    m_scroll += float(pl[1] if pl[1] != 0 else pl[0])
            continue

        name = _payload_name(ev)
        if name is None:
            continue

        if etype == "KeyRelease":
            held.discard(name)
            continue

        if etype != "KeyPress":
            continue

        # KeyPress.
        if name == "CapsLock":
            caps = not caps
            continue
        if name in _SHIFTS or name in _SUPPRESS_MODIFIERS:
            held.add(name)
            continue

        suppress = bool(held & _SUPPRESS_MODIFIERS)
        if suppress:
            # Modifier combo -> chord, not text. Ends any open typing burst.
            flush_typing()
            events.append(Event(t_s=t, kind="chord", data={"chord": friendly_chord(held, name)}))
            continue

        # A printable / editing keystroke -> typing burst.
        if m_open:
            flush_mouse()
        if buf_keys and t - buf_last > typing_gap_s:
            flush_typing()
        if not buf_keys:
            buf_start = t
        buf_last = t
        buf_keys += 1

        shift = bool(held & _SHIFTS)
        norm = _NAME_ALIASES.get(name, name)
        if norm == "Backspace":
            if buf:
                buf.pop()
            else:
                buf_approx = True  # deleted into pre-burst text
        elif norm in ("Return", "KpReturn"):
            buf.append("\n")
        elif norm == "Tab":
            buf.append("\t")
        elif norm in _EDIT_KEYS:
            buf_approx = True  # cursor moved / forward-delete: order unreliable
        elif norm.startswith("Key") and len(norm) == 4:
            ch = norm[3]
            up = shift ^ caps
            buf.append(ch.upper() if up else ch.lower())
        elif norm in _SHIFTED:
            buf.append(_SHIFTED[norm] if shift else _UNSHIFTED[norm])
        else:
            # Function keys, Escape, etc. mid-typing: note but don't corrupt text.
            buf_approx = True

    flush_typing()
    flush_mouse()
    events.sort(key=lambda e: e.t_s)
    return Transcript(events=events, segment_offset_s=segment_offset_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keylog", type=Path, required=True)
    ap.add_argument("--segment-offset-s", type=float, default=0.0)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--max-text-chars", type=int, default=600)
    args = ap.parse_args()
    tr = build_transcript(args.keylog, segment_offset_s=args.segment_offset_s)
    n_type = sum(1 for e in tr.events if e.kind == "type")
    n_chord = sum(1 for e in tr.events if e.kind == "chord")
    n_mouse = sum(1 for e in tr.events if e.kind == "mouse")
    n_app = sum(1 for e in tr.events if e.kind == "app")
    print(f"# {len(tr.events)} events: {n_type} typing, {n_chord} chords, "
          f"{n_mouse} mouse, {n_app} app-switches\n")
    print(tr.render(args.start, args.end, max_text_chars=args.max_text_chars))


if __name__ == "__main__":
    main()
