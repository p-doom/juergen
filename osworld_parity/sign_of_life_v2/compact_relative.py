"""Exact Phase-B raw deltatype-v2 contract and VM adapter.

The prompt and grammar below are recovered from the checkpoint's sealed
training provenance.  In particular, model output may retain natural-language
reasoning before the final bare action line; the last non-empty line is the
only action span.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import requests

from ..proper_vm_capability_ladder.rung1.transport import HttpVmTransport, Operation


SYSTEM_PROMPT = """You operate a desktop computer from screenshots.

Return one bare action line after any reasoning. Mouse values are RAW PIXEL deltas from the current cursor:
  dx dy scroll
Optional ordered elements follow ` ; ` and are executed left-to-right. Existing elements are button/key transitions (`+NAME` presses, `-NAME` releases) and `type("JSON string")`.

The only allowed MOVE form is a left-button drag:
  initial_dx initial_dy 0 ; +LMB MOVE(drag_dx,drag_dy) -LMB
The initial delta first moves to the drag start. `+LMB` presses left, MOVE applies the second raw-pixel delta over 0.5 seconds, and `-LMB` releases left. For a drag from the current cursor, use initial_dx=0 and initial_dy=0. Preserve MOVE(0,0) for a real zero-distance drag. MOVE is invalid anywhere else.

Special lines: NO_OP, TERMINATE, FAIL.
"""
SYSTEM_PROMPT_SHA256 = "57f7d0b230974068618b48151b73215d5517d5445a99dbf5abdc05557e3482e6"
PRODUCER_ACTION_V2_SHA256 = "1ded3d5a7e51da71cf3082049fbdd404971ebf72a95d93f333ebb3ee3075ccb7"
PRODUCER_DATASET_MANIFEST_SHA256 = "77085ee3c2ea7d780e96ade76efbffc0746139c0c619a5d9cbcec8562a1a25d5"

_EVENT_RE = re.compile(r"^([+-])([A-Za-z_][A-Za-z0-9_]*)$")
_MOVE_RE = re.compile(r"MOVE\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_BUTTONS = {"LMB": "left", "RMB": "right", "MMB": "middle"}


class CompactRelativeError(ValueError):
    pass


@dataclass(frozen=True)
class CompactRelativeAction:
    dx: int = 0
    dy: int = 0
    scroll: int = 0
    elements: tuple[tuple[str, Any], ...] = ()
    no_op: bool = False
    terminate: bool = False
    fail: bool = False


def _scan_elements(segment: str) -> tuple[tuple[str, Any], ...]:
    elements: list[tuple[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(segment):
        if segment[index].isspace():
            index += 1
            continue
        if segment.startswith("MOVE", index):
            match = _MOVE_RE.match(segment, index)
            if match is None:
                raise CompactRelativeError(
                    f"malformed MOVE element: {segment[index:index + 30]!r}"
                )
            end = match.end()
            if end < len(segment) and not segment[end].isspace():
                raise CompactRelativeError(
                    f"malformed MOVE element: {segment[index:index + 30]!r}"
                )
            elements.append(("move", (int(match[1]), int(match[2]))))
            index = end
            continue
        if segment.startswith("type(", index):
            start = index + 5
            while start < len(segment) and segment[start].isspace():
                start += 1
            if start >= len(segment) or segment[start] != '"':
                raise CompactRelativeError("type(...) must wrap a JSON string")
            try:
                value, end = decoder.raw_decode(segment, start)
            except json.JSONDecodeError as exc:
                raise CompactRelativeError(f"bad type() JSON string: {exc}") from exc
            close = end
            while close < len(segment) and segment[close].isspace():
                close += 1
            if close >= len(segment) or segment[close] != ")":
                raise CompactRelativeError("type(...) missing closing ')'")
            elements.append(("type", value))
            index = close + 1
            continue
        end = index
        while end < len(segment) and not segment[end].isspace():
            end += 1
        token = segment[index:end]
        match = _EVENT_RE.fullmatch(token)
        if match is None:
            raise CompactRelativeError(f"malformed deltatype-v2 element: {token!r}")
        elements.append(
            ("event", ("press" if match[1] == "+" else "release", match[2]))
        )
        index = end
    return tuple(elements)


def _validate_ordered_move(action: CompactRelativeAction) -> CompactRelativeAction:
    moves = [value for kind, value in action.elements if kind == "move"]
    if not moves:
        return action
    expected = (
        ("event", ("press", "LMB")),
        ("move", moves[0]),
        ("event", ("release", "LMB")),
    )
    if len(moves) != 1 or action.scroll != 0 or action.elements != expected:
        raise CompactRelativeError(
            "MOVE is reserved for `initial_dx initial_dy 0 ; "
            "+LMB MOVE(drag_dx,drag_dy) -LMB`"
        )
    return action


def parse_compact_relative(text: str) -> CompactRelativeAction:
    """Parse the exact producer grammar, including its prose extraction rule."""
    if not isinstance(text, str):
        raise TypeError(f"parse_compact_relative expects str, got {type(text)!r}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise CompactRelativeError("empty action text")
    line = lines[-1]
    if line == "NO_OP":
        return CompactRelativeAction(no_op=True)
    if line == "TERMINATE":
        return CompactRelativeAction(terminate=True)
    if line == "FAIL":
        return CompactRelativeAction(fail=True)
    mouse, elements = line.split(";", 1) if ";" in line else (line, "")
    tokens = mouse.strip().split()
    if len(tokens) != 3:
        raise CompactRelativeError(
            f"expected exactly three mouse integers, got {tokens!r}"
        )
    try:
        dx, dy, scroll = (int(token) for token in tokens)
    except ValueError as exc:
        raise CompactRelativeError(
            f"mouse tokens are not integers: {tokens!r}"
        ) from exc
    return _validate_ordered_move(
        CompactRelativeAction(dx, dy, scroll, _scan_elements(elements))
    )


def format_compact_relative(action: CompactRelativeAction) -> str:
    if action.no_op:
        return "NO_OP"
    if action.terminate:
        return "TERMINATE"
    if action.fail:
        return "FAIL"
    rendered: list[str] = []
    for kind, value in action.elements:
        if kind == "move":
            rendered.append(f"MOVE({value[0]},{value[1]})")
        elif kind == "type":
            rendered.append("type(" + json.dumps(value, ensure_ascii=False) + ")")
        elif kind == "event":
            transition, name = value
            rendered.append(("+" if transition == "press" else "-") + name)
        else:  # pragma: no cover - parser fixes the set
            raise CompactRelativeError(f"unknown element kind: {kind!r}")
    label = f"{action.dx} {action.dy} {action.scroll}"
    return label + (" ; " + " ".join(rendered) if rendered else "")


def compile_compact_relative(action: CompactRelativeAction) -> tuple[Operation, ...]:
    """Lower one raw-relative action to the suite's shared guest primitives."""
    if action.no_op or action.terminate or action.fail:
        return ()
    operations: list[Operation] = []
    if action.dx or action.dy:
        operations.append(Operation("move_relative", (action.dx, action.dy)))
    if action.scroll:
        operations.append(Operation("scroll", (action.scroll,)))
    for kind, value in action.elements:
        if kind == "move":
            operations.append(Operation("move_relative", tuple(value)))
            continue
        if kind == "type":
            text = str(value)
            try:
                text.encode("ascii")
            except UnicodeEncodeError as exc:
                raise CompactRelativeError(
                    "sign-of-life compact type must be ASCII"
                ) from exc
            if "\n" in text or "\r" in text:
                raise CompactRelativeError(
                    "type() cannot embed Enter in this suite; emit a Return event"
                )
            operations.append(Operation("ascii_type", (text,)))
            continue
        if kind != "event":  # pragma: no cover - parser fixes the set
            raise CompactRelativeError(f"unknown element kind: {kind!r}")
        transition, name = value
        direction = "down" if transition == "press" else "up"
        if name in _BUTTONS:
            operations.append(Operation(f"mouse_{direction}", (_BUTTONS[name],)))
        else:
            operations.append(Operation(f"key_{direction}", (name,)))
    return tuple(operations)


def execute_compact_relative(
    transport: HttpVmTransport, text: str
) -> dict[str, Any]:
    action = parse_compact_relative(text)
    canonical = format_compact_relative(action)
    operations = compile_compact_relative(action)
    if not operations:
        return {
            "canonical_action": canonical,
            "no_op": action.no_op,
            "terminated": action.terminate,
            "failed": action.fail,
            "operations": [],
            "receipt": None,
        }
    receipt = transport.execute_atomic(operations)
    return {
        "canonical_action": canonical,
        "no_op": False,
        "terminated": False,
        "failed": False,
        "operations": [asdict(operation) for operation in operations],
        "receipt": receipt.as_dict(),
    }


def verify_sealed_contract() -> dict[str, str]:
    observed = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if observed != SYSTEM_PROMPT_SHA256:
        raise RuntimeError(
            f"Phase-B system prompt drift: {observed} != {SYSTEM_PROMPT_SHA256}"
        )
    return {
        "system_prompt_sha256": observed,
        "producer_action_v2_sha256": PRODUCER_ACTION_V2_SHA256,
        "producer_dataset_manifest_sha256": PRODUCER_DATASET_MANIFEST_SHA256,
    }


def _prose_summary(raw_output: str) -> str:
    """Recover the natural-language action description used by Phase-B history."""
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) < 2:
        return lines[0] if lines else "No parseable action description."
    prose = " ".join(lines[:-1]).strip()
    return prose.removeprefix("Action:").strip() or "No action description."


def _pil_to_data_url(frame: Any) -> str:
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=85, optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_phaseb_messages(
    *, instruction: str, frames: list[Any], actions: list[str], history_frames: int = 5
) -> list[dict[str, Any]]:
    """Build the checkpoint's training-shaped rolling conversation.

    Phase-B records contain at most five images (four completed turns plus the
    current image).  Earlier actions are represented as prose in the first
    user turn's ``Previous actions`` block.
    """
    if not frames or len(actions) != len(frames) - 1:
        raise ValueError("Phase-B history requires len(actions) == len(frames) - 1")
    if history_frames < 1:
        raise ValueError("history_frames must be positive")
    first = max(0, len(frames) - history_frames)
    visible_frames = frames[first:]
    visible_actions = actions[first:]
    earlier_actions = actions[:first]
    if earlier_actions:
        previous = "\n".join(
            f"Step {index}: {_prose_summary(raw)}"
            for index, raw in enumerate(earlier_actions, start=1)
        )
    else:
        previous = "None"
    first_text = (
        "\nPlease generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\nPrevious actions:\n{previous}"
    )
    # The sealed teacher-forced evaluator flattens the system text to a string
    # while preserving list-shaped multimodal user/assistant content.
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for index, frame in enumerate(visible_frames):
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": _pil_to_data_url(frame)},
            }
        ]
        if index == 0:
            content.append({"type": "text", "text": first_text})
        messages.append({"role": "user", "content": content})
        if index < len(visible_actions):
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": visible_actions[index]}],
                }
            )
    return messages


def call_phaseb_model(
    *,
    model_url: str,
    api_key: str,
    model: str,
    instruction: str,
    frames: list[Any],
    actions: list[str],
    request_timeout_s: float = 180.0,
) -> str:
    payload = {
        "model": model,
        "messages": build_phaseb_messages(
            instruction=instruction, frames=frames, actions=actions
        ),
        "max_tokens": 256,
        "temperature": 0.0,
    }
    response = requests.post(
        model_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=request_timeout_s,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""
