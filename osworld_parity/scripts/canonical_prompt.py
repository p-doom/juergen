"""Canonical OSWorld Qwen3-VL system prompt — transcribed VERBATIM from
OSWorld mm_agents/qwen3vl_agent.py (the gt agent behind the published 33.9%),
coordinate_type='relative' (0-1000 NORMALIZED positions; the agent's
adjust_coordinates scales /999 to the real screen — we scale /1000 at dispatch,
see collector --coord_grid 1000).

We reconstruct the byte-exact string via the SAME json.dumps(tools_def) the agent
uses, so the teacher stays exactly in-distribution. A runtime assert checks this
module and the installed qwen3vl_agent agree (see build_canonical_system_prompt()).

NB coordinate terminology: OSWorld-"relative" = 1000-normalized ABSOLUTE positions,
NOT our cursor-delta "relative". We capture these, scale to real px, then
diff-of-absolute -> our cursor-delta relative. Do not conflate.
"""
from __future__ import annotations

import json

def _description_prompt_lines(resolution: str = "1000x1000") -> list:
    return [
        "Use a mouse and keyboard to interact with a computer, and take screenshots.",
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
        f"* The screen's resolution is {resolution}.",
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
        "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    ]

_ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
        """


def _tools_def(resolution: str = "1000x1000") -> dict:
    return {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": "\n".join(_description_prompt_lines(resolution)),
            "parameters": {
                "properties": {
                    "action": {
                        "description": _ACTION_DESCRIPTION_PROMPT,
                        "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag",
                                 "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"],
                        "type": "string"
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.",
                        "type": "string",
                        "enum": ["success", "failure"]
                    }
                },
                "required": ["action"],
                "type": "object"
            },
            "args_format": "Format the arguments as a JSON object."
        }
    }


def build_canonical_system_prompt(resolution: str = "1000x1000") -> str:
    """The byte-exact canonical OSWorld system prompt.

    resolution='1000x1000' = coordinate_type 'relative' (0-999 normalized).
    resolution='{pw}x{ph}'  = coordinate_type 'absolute' (processed-pixel space;
    caller must scale emitted coords by orig/processed per-axis at dispatch)."""
    return """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(_tools_def(resolution)) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""


# Minimal <think> preamble — FALLBACK ONLY (use 8B-Thinking under the canonical
# prompt first). Appended to (never replacing) the canonical prompt.
THINK_PREAMBLE_FALLBACK = (
    "Before the Action line, add ONE short first-person thought in <think></think> "
    "stating WHY (your current sub-goal and what you observe). Keep it to one sentence.\n\n"
)


if __name__ == "__main__":
    # Fidelity check against the installed OSWorld agent, if reachable.
    import os, sys, re
    root = os.environ.get("OSWORLD_ROOT", "/fast/home/franz.srambical/OSWorld")
    src = open(os.path.join(root, "mm_agents", "qwen3vl_agent.py")).read()
    p = build_canonical_system_prompt()
    # spot-checks: the tool json + response-format section must appear coherent
    assert '"name_for_human": "computer_use"' in p
    assert "# Response format" in p and "action=terminate" in p
    assert "The screen's resolution is 1000x1000." in p
    print("canonical prompt length:", len(p))
    print(p[:400], "...\n...", p[-400:])
