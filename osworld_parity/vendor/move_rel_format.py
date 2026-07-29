"""v2 native-relative computer_use encoder.

Two grammar corrections over v1 (videocua_nativerel_v1/native_rel_format.py):

  (1) FORMAT FIX (explicit move_rel + coordinate-less click split).
      v1 folded the relative delta INTO the click's `coordinate`, e.g.
          {"action":"left_click","coordinate":[dx,dy]}
      Native computer_use / pyautogui treat a click `coordinate` as an ABSOLUTE
      target, so encoding a *relative* delta there is semantically wrong. v2
      expresses a relative move-and-click as TWO actions -- an EXPLICIT relative
      move then a coordinate-less click at the current position:
          {"action":"move_rel","coordinate":[dx,dy]}   (relative delta)
          {"action":"left_click"}                        (click in place)
      `move_rel` is a DISTINCT action name (not the canonically-absolute
      `mouse_move`) that maps 1:1 to pyautogui.moveRel(dx,dy) in the executor --
      consistent with the other custom actions native_rel already adds
      (mouse_down/up, key_down/up). This removes the absolute/relative semantic
      mismatch entirely rather than moving it to a differently-named action.
      A zero-delta click emits ONLY the coordinate-less click (no redundant move).
      Same split applies to mouse_down when it carries a delta.

  (2) COORD NORMALIZATION (0-999).
      Every coordinate delta is converted from raw native pixels to the
      established 0-999 normalized convention, per-axis:
          norm = round(px / dim * 1000)
      consistent with the RFT cold-start + distillation conventions. `scroll`
      `pixels` is a wheel amount (not a screen coordinate) and is left unchanged.
      The eval executor denormalizes 0-999 -> screen px via --rel_coord_grid 1000
      (px = norm * dim / 1000), the exact inverse.

Alignment (typing causality) is fixed UPSTREAM in build_videocua_chat_v2.py
(emit_typing_text single-interval assignment); this module only rewrites grammar.
"""
from __future__ import annotations

import os
import sys

# Reuse the battle-tested v1 encoder (custom grammar -> native arg dicts, with
# round-trip validation) and only post-process its output into v2 grammar.
_V1_DIR = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/videocua_nativerel_v1"
if _V1_DIR not in sys.path:
    sys.path.insert(0, _V1_DIR)
import native_rel_format as _v1  # noqa: E402

_CLICK_ACTIONS = {"left_click", "right_click", "double_click",
                  "triple_click", "middle_click"}
_MOVE_CARRYING = _CLICK_ACTIONS | {"mouse_down"}


def norm_axis(px, dim):
    """px (raw native pixels) -> 0-999 normalized delta (per-axis, round())."""
    if not dim:
        return 0
    return int(round(float(px) / float(dim) * 1000.0))


def split_and_normalize(arg_dicts, width, height):
    """v1 arg dicts -> v2 arg dicts.

    * split a delta-carrying click/mouse_down into an explicit `move_rel` + the
      coordinate-less op
    * a standalone v1 relative `mouse_move` becomes an explicit `move_rel`
    * normalize every `coordinate` delta to 0-999
    """
    out = []
    for a in arg_dicts:
        act = a.get("action")
        if act in _MOVE_CARRYING and "coordinate" in a:
            dx, dy = a["coordinate"]
            if dx != 0 or dy != 0:
                out.append({"action": "move_rel",
                            "coordinate": [norm_axis(dx, width), norm_axis(dy, height)]})
            out.append({k: v for k, v in a.items() if k != "coordinate"})
        elif act == "mouse_move" and "coordinate" in a:
            dx, dy = a["coordinate"]
            out.append({"action": "move_rel",
                        "coordinate": [norm_axis(dx, width), norm_axis(dy, height)]})
        else:
            out.append(dict(a))
    return out


def convert_turn_v2(custom_text, width, height):
    """custom-grammar turn text -> (assistant_text_v2, arg_dicts_v2).

    Raises native_rel_format.RoundTripError if the v1 conversion is not lossless.
    """
    _text_v1, arg_dicts_v1 = _v1.convert_turn(custom_text)
    v2 = split_and_normalize(arg_dicts_v1, width, height)
    return _v1.render_assistant_text(v2), v2


# Canonical v2 native-relative system prompt (explicit move_rel, normalized,
# coordinate-less click). MUST stay byte-identical to the "native_rel_v2" entry
# in juergen/eval/osworld_system_prompts.py.
SYSTEM_PROMPT = (
    "You operate a desktop computer using the computer_use tool. The first user "
    "turn shows the initial screen and the user's goal; each subsequent user turn "
    "shows the current screen. Reply with one or more computer_use tool calls that "
    "advance toward the goal.\n"
    "\n"
    "Mouse movement is RELATIVE and NORMALIZED. To move the cursor, emit a "
    "`move_rel` action whose `coordinate` is a [dx, dy] offset from the CURRENT "
    "cursor position, expressed in thousandths of the screen (each axis in "
    "[-999, 999]; dx = 1000 spans the full width, dy = 1000 the full height; "
    "positive dx = right, positive dy = down). `move_rel` moves the cursor by that "
    "relative delta (pyautogui.moveRel); it is NOT an absolute screen coordinate. "
    "Look at the visible cursor in the screenshot to judge how far and in which "
    "direction to move. To click a target, FIRST `move_rel` by the relative offset, "
    "THEN issue a click with NO coordinate (the click lands at the current cursor "
    "position).\n"
    "\n"
    "Actions (computer_use `action` field):\n"
    "- move_rel {coordinate:[dx,dy]}: move the cursor by the relative normalized "
    "offset (dx,dy).\n"
    "- left_click / right_click / middle_click: click at the CURRENT cursor "
    "position (no coordinate); move first with move_rel.\n"
    "- double_click / triple_click: double / triple click at the current position.\n"
    "- mouse_down {button} / mouse_up {button}: press / release a mouse button "
    "(button = 'left','right','middle'). A drag is move_rel, mouse_down, one or "
    "more move_rel, then mouse_up.\n"
    "- key {keys:[...]}: press a key or chord, e.g. ['ctrl','a'], ['enter'], ['tab'].\n"
    "- key_down {keys:[...]} / key_up {keys:[...]}: hold / release keys across steps.\n"
    "- type {text}: type a string of text.\n"
    "- scroll {pixels}: scroll the wheel (positive = up, negative = down).\n"
    "- wait {time}: do nothing this step.\n"
    "- terminate {status}: the goal is complete (status = 'success' or 'failure').\n"
    "\n"
    "For each action, return a JSON object within <tool_call></tool_call> tags. To "
    "move the cursor 12 right / 8 up (normalized) and left-click there:\n"
    "<tool_call>\n"
    '{"name": "computer_use", "arguments": {"action": "move_rel", "coordinate": [12, -8]}}\n'
    "</tool_call>\n"
    "<tool_call>\n"
    '{"name": "computer_use", "arguments": {"action": "left_click"}}\n'
    "</tool_call>"
)
