from __future__ import annotations

import base64
import logging
import os
import re
import time
from pathlib import Path

import openai

from action_parser import OrderedPrimitive, parse_ordered_action

_LOGGER = logging.getLogger(__name__)

MAX_RETRY_TIMES = 5

SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data_pipeline"
    / "realigned_pipeline"
    / "system_prompts"
    / "cua_v3_cuagym.txt"
)

INSTRUCTION_TEMPLATE = """
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions}"""

_RDEV_TO_PYAUTOGUI = {
    "Return": "enter",
    "Escape": "esc",
    "Backspace": "backspace",
    "Tab": "tab",
    "Space": "space",
    "ShiftLeft": "shiftleft",
    "ShiftRight": "shiftright",
    "ControlLeft": "ctrlleft",
    "ControlRight": "ctrlright",
    "Alt": "alt",
    "AltGr": "altright",
    "MetaLeft": "winleft",
    "MetaRight": "winright",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "Home": "home",
    "End": "end",
    "Delete": "delete",
    "Insert": "insert",
    "Comma": ",",
    "Period": ".",
    "Slash": "/",
    "Backslash": "\\",
    "Semicolon": ";",
    "Quote": "'",
    "Minus": "-",
    "Equal": "=",
    "Backquote": "`",
    "BracketLeft": "[",
    "BracketRight": "]",
}

_MOUSE_BUTTONS = {"LMB": "left", "MMB": "middle", "RMB": "right"}


def rdev_to_pyautogui(name: str) -> str:
    if name in _RDEV_TO_PYAUTOGUI:
        return _RDEV_TO_PYAUTOGUI[name]
    if name.startswith("Key") and len(name) == 4 and name[3].isalpha():
        return name[3].lower()
    if name.startswith("Num") and len(name) == 4 and name[3].isdigit():
        return name[3]
    if name.startswith("Digit") and len(name) == 6 and name[5].isdigit():
        return name[5]
    return name.lower()


def strip_think(text: str) -> str:
    head, sep, tail = text.partition("</think>")
    if not sep:
        return text.strip()
    return tail.lstrip("\n")


def extract_action_line(response: str) -> str:
    stripped = strip_think(response)
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty response after think strip")
    return lines[-1]


def compile_primitives(
    primitives: list[OrderedPrimitive], screen: tuple[int, int]
) -> str:
    stmts = ["import pyautogui", "pyautogui.FAILSAFE = False"]
    for p in primitives:
        if p.kind == "move":
            dx_px = round(p.dx / 1000 * screen[0])
            dy_px = round(p.dy / 1000 * screen[1])
            stmts.append(f"pyautogui.moveRel({dx_px}, {dy_px})")
        elif p.kind == "scroll":
            if p.dy:
                stmts.append(f"pyautogui.scroll({p.dy})")
            if p.dx:
                stmts.append(f"pyautogui.hscroll({p.dx})")
        elif p.kind == "down":
            if p.mouse_button is not None:
                stmts.append(f"pyautogui.mouseDown(button={_MOUSE_BUTTONS[p.name]!r})")
            else:
                stmts.append(f"pyautogui.keyDown({rdev_to_pyautogui(p.name)!r})")
        elif p.kind == "up":
            if p.mouse_button is not None:
                stmts.append(f"pyautogui.mouseUp(button={_MOUSE_BUTTONS[p.name]!r})")
            else:
                stmts.append(f"pyautogui.keyUp({rdev_to_pyautogui(p.name)!r})")
        elif p.kind == "type":
            stmts.append(f"pyautogui.write({p.text!r}, interval=0.012)")
        else:
            raise ValueError(f"uncompilable primitive kind: {p.kind}")
    return "\n".join(stmts)


class Oev3Agent:
    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "",
        max_tokens: int = 32768,
        top_p: float = 0.95,
        temperature: float = 0.6,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        history_n: int = 4,
        coordinate_type: str = "relative",
        api_backend: str = "openai",
        screen_size: tuple[int, int] = (1920, 1080),
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.history_n = history_n
        self.screen_size = screen_size
        self.system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
        self.screenshots: list[str] = []
        self.stripped_responses: list[str] = []
        self.action_lines: list[str] = []
        self.logger = _LOGGER

    def reset(self, logger=None):
        self.screenshots = []
        self.stripped_responses = []
        self.action_lines = []
        if logger is not None:
            self.logger = logger

    def _instruction_text(self, instruction: str, n_window: int) -> str:
        n_pre = len(self.action_lines) - n_window
        previous = [
            f"Step {i + 1}: {self.action_lines[i]}" for i in range(max(0, n_pre))
        ]
        previous_str = "\n".join(previous) if previous else "None"
        return INSTRUCTION_TEMPLATE.format(
            instruction=instruction, previous_actions=previous_str
        )

    def _build_messages(self, instruction: str, screenshot_b64: str) -> list[dict]:
        n_window = min(self.history_n, len(self.stripped_responses))
        instruction_text = self._instruction_text(instruction, n_window)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        ]
        window_shots = self.screenshots[-n_window:] if n_window else []
        window_resps = self.stripped_responses[-n_window:] if n_window else []
        for idx in range(n_window):
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{window_shots[idx]}"},
                }
            ]
            if idx == 0:
                content.append({"type": "text", "text": instruction_text})
            messages.append({"role": "user", "content": content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": window_resps[idx]}],
                }
            )
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            }
        ]
        if n_window == 0:
            content.append({"type": "text", "text": instruction_text})
        messages.append({"role": "user", "content": content})
        return messages

    def _call_llm(self, messages: list[dict]) -> str:
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        api_key = os.environ.get("OPENAI_API_KEY", "sk-123")
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        for attempt in range(1, MAX_RETRY_TIMES + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                self.logger.error("oev3 llm call failed (attempt %d): %s", attempt, exc)
                if attempt < MAX_RETRY_TIMES:
                    time.sleep(5)
        return ""

    def predict(self, instruction: str, obs: dict) -> tuple[str, list[str]]:
        screenshot_b64 = base64.b64encode(obs["screenshot"]).decode()
        messages = self._build_messages(instruction, screenshot_b64)
        response = self._call_llm(messages)
        if not response:
            return response, []
        try:
            line = extract_action_line(response)
        except ValueError:
            return response, []
        if line == "TERMINATE":
            actions = ["DONE"]
        elif line == "NO_OP":
            actions = ["WAIT"]
        else:
            try:
                parsed = parse_ordered_action(line)
            except (ValueError, TypeError) as exc:
                self.logger.warning("oev3 parse failure: %s on %r", exc, line[:120])
                return response, []
            try:
                actions = [compile_primitives(parsed.primitives, self.screen_size)]
            except ValueError as exc:
                self.logger.warning("oev3 compile failure: %s", exc)
                return response, []
        self.screenshots.append(screenshot_b64)
        self.stripped_responses.append(strip_think(response))
        self.action_lines.append(line)
        return response, actions
