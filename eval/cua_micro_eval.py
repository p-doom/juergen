"""State-verifiable atomic and short-horizon CUA micro-evaluation suite.

Each attempt starts from a fresh OSWorld VM snapshot. Atomic tasks receive one
goal-conditioned screenshot and emit one strict action. Multi-turn tasks keep
the evolving visual conversation in one VM, require one strict action per
turn, and semantically gate every step. The primary contract is
``computer_use_rel_step_v1``; an explicit native Qwen3-VL computer-use mode
supports off-the-shelf baselines. Four sampled attempts per task produce
empirical pass@1 and pass@4 curves; multi-turn partial credit is the verified
prefix fraction.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import logging
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw

import sampling as sampling_mod
from action_parser import ComputerUseCall
from action_parser import OrderedAction as NativeOrderedAction
from action_parser import parse_ordered_action_tolerant
from cua_micro_action_parser import (
    RelStepAction as OrderedAction,
    RelStepPrimitive as OrderedPrimitive,
    parse_computer_use_rel_step_action,
    parse_qwen3vl_computer_use_action,
)
from osworld_runtime import (
    _DEFAULT_QCOW2,
    _DEFAULT_QEMU_BIN,
    _EVAL_DIR,
    _interleave_messages,
    _pil_to_data_url,
    _wait_for,
    build_loggable_messages,
    evict_history,
    window_frame_labels,
)
from osworld_system_prompts import SYSTEM_PROMPTS
from osworld_vm_client import OSWorldClient
from sampling import SamplingParams

_LOGGER = logging.getLogger(__name__)

# _call_model / _fresh_visual_messages are defined here rather than imported
# from osworld_runtime: this branch's own shared _call_model (used by
# freeroll.py / osworld_grounding_runner.py / osworld_fullbench_runner.py on
# the branch this suite was ported onto -- not present in this worktree,
# which carries only the CUA micro-eval suite) returns a plain str and takes
# max_tokens/temperature directly, while the micro-eval suite (ported from
# yll/cua-micro-evals) needs the (content, finish_reason) tuple contract --
# finish_reason=="length" flags a truncated reply so a half-emitted tool call
# is never dispatched -- plus seed plumbing and the full Qwen sampling tuple
# via sampling.SamplingParams. Changing the shared _call_model's signature/
# return type to match would have broken every existing caller on that
# branch, so the micro-eval suite carries its own copy here instead.


def _fresh_visual_messages(
    system_prompt: str,
    instruction: str | None,
    image_parts: list[Any],
) -> list[dict[str, Any]]:
    """One decision record: system + one user turn containing goal and images.

    This is the runtime twin of stage 04's ``--context-images`` mode. It never
    replays prior assistant actions, so every prediction is conditioned on a
    fresh visual state rather than on model-generated history.
    """
    content: list[Any] = []
    if instruction:
        content.append({"type": "text", "text": instruction})
    content.extend(image_parts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _call_model(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    instruction: str | None,
    recent_frames: list[Image.Image],
    recent_actions: list[str] | None = None,
    fresh_visual_context: bool = False,
    sampling: SamplingParams,
    seed: int | None = None,
    request_timeout_s: float = 120.0,
) -> tuple[str, str | None]:
    """One chat-completion call for the micro-eval suite.

    ``sampling`` is the single source of truth for decoding parameters (see
    ``eval/sampling.py``): the FULL Qwen-recommended tuple (temperature, top_p,
    top_k, repetition_penalty, presence_penalty, max_tokens) is sent to sglang,
    not just temperature -- so the checkpoint's partial ``generation_config``
    can no longer silently fill in the rest (top_p/top_k) or drop
    presence_penalty.

    Returns ``(content, finish_reason)``. ``finish_reason == "length"``
    means the reply was truncated at ``max_tokens`` -- callers MUST NOT
    dispatch a truncated action (a half-emitted ``down(...)`` would leave a
    key held and trigger OS key-repeat).
    """
    image_parts = [
        {"type": "image_url", "image_url": {"url": _pil_to_data_url(f)}} for f in recent_frames
    ]
    messages = (
        _fresh_visual_messages(system_prompt, instruction, image_parts)
        if fresh_visual_context
        else _interleave_messages(system_prompt, instruction, image_parts, recent_actions)
    )
    request_json = {
        "model": model,
        "messages": messages,
        **sampling.as_request_json(),
    }
    if seed is not None:
        request_json["seed"] = int(seed)
    r = requests.post(
        sglang_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_json,
        timeout=request_timeout_s,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return choice["message"]["content"] or "", choice.get("finish_reason")

_GRID = 1000
_MOVEMENT_SCALES = (8, 32, 128)
_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_FIXTURE_GUEST_PATH = "/tmp/cua_micro_fixture.py"
_FIXTURE_STATE_PATH = "/tmp/cua_micro_fixture_state.json"
_NATIVE_TERMINAL_STATE_PATH = "/tmp/cua_native_terminal_state.json"
_NATIVE_EDITOR_PATH = "/tmp/cua_native_editor.txt"
_DEFAULT_SUITE = Path(__file__).with_name("cua_micro_tasks.json")
_REL_STEP_FORMAT = "computer_use_rel_step_v1"
_QWEN3VL_NATIVE_FORMAT = "qwen3vl_native_cua_v1"
_RESULT_SCHEMA_VERSION = 2
# This branch's own ordered_events_v3 format (action_parser.OrderedAction /
# parse_ordered_action_tolerant), added so the suite can evaluate checkpoints
# trained on either lineage via the same --action_format knob.
_NATIVE_ORDERED_FORMAT = "cua_ordered_typing_v1"
_PROMPT_FORMATS = {
    "cua_rel_step_v1_thinking": _REL_STEP_FORMAT,
    "qwen3vl_native_cua_v1": _QWEN3VL_NATIVE_FORMAT,
    "cua_ordered_typing_v1": _NATIVE_ORDERED_FORMAT,
}


@dataclass(frozen=True)
class Turn:
    turn_id: str
    target: dict[str, Any]
    cursor: dict[str, Any]
    expected: dict[str, Any]
    verifier: dict[str, Any]


@dataclass(frozen=True)
class Task:
    task_id: str
    category: str
    instruction: str
    setup: dict[str, Any]
    target: dict[str, Any]
    cursor: dict[str, Any]
    expected: dict[str, Any]
    verifier: dict[str, Any]
    turns: tuple[Turn, ...] = ()
    # "prefix" (default): turns are distinct ordered sub-goals -- the
    # trajectory stops at the first turn whose action+verifier don't match,
    # and success requires every turn to pass in order (see
    # _finalize_multiturn_result). "multiturn": every turn shares the same
    # end goal and is one try out of a budget -- the cursor is NOT reset
    # between turns (it's one continuous trajectory building toward that
    # goal, e.g. clicking several calculator buttons in sequence), the
    # trajectory keeps going after a non-matching turn, and it succeeds as
    # soon as any turn's verifier passes, however it got there. Only
    # meaningful when ``turns`` is set.
    turn_mode: str = "prefix"


def task_turns(task: Task) -> tuple[Turn, ...]:
    if task.turns:
        return task.turns
    return (
        Turn(
            turn_id="action",
            target=task.target,
            cursor=task.cursor,
            expected=task.expected,
            verifier=task.verifier,
        ),
    )


def load_suite(path: Path) -> tuple[dict[str, Any], list[Task]]:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported suite schema: {raw.get('schema_version')!r}")
    if raw.get("coordinate_grid") != _GRID:
        raise ValueError(f"coordinate_grid must be {_GRID}")
    tasks: list[Task] = []
    seen: set[str] = set()
    for index, item in enumerate(raw.get("tasks", [])):
        common = {"id", "category", "instruction", "setup"}
        atomic = {"target", "cursor", "expected", "verifier"}
        has_turns_list = "turns" in item
        # Compact form for turn_mode="multiturn" budgets: one shared turn
        # template repeated ``max_turns`` times, instead of writing out N
        # near-identical dicts by hand.
        has_turn_template = "turn" in item
        if has_turns_list and has_turn_template:
            raise ValueError(
                f"task {index}: specify either 'turns' or 'turn'+'max_turns', not both"
            )
        is_multiturn = has_turns_list or has_turn_template
        if has_turn_template:
            required = common | {"turn", "max_turns", "turn_mode"}
        else:
            required = common | ({"turns"} if is_multiturn else atomic)
        # turn_mode is optional on the explicit 'turns' list (defaults to
        # "prefix"), and always required -- via the required set above --
        # on the compact 'turn'+'max_turns' template. Not allowed at all on
        # atomic tasks.
        allowed = required | ({"turn_mode"} if has_turns_list else set())
        missing = required - set(item)
        extra = set(item) - allowed
        if missing or extra:
            raise ValueError(f"task {index}: missing={sorted(missing)} extra={sorted(extra)}")
        turn_mode = str(item.get("turn_mode", "prefix"))
        if turn_mode not in ("prefix", "multiturn"):
            raise ValueError(
                f"task {index}: turn_mode must be 'prefix' or 'multiturn', got {turn_mode!r}"
            )
        if has_turn_template and turn_mode != "multiturn":
            raise ValueError(
                f"task {index}: 'turn'+'max_turns' only makes sense with turn_mode='multiturn' "
                "(every turn shares one goal) -- use an explicit 'turns' list for a sequence "
                "of distinct sub-goals"
            )
        task_id = item["id"]
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError(f"task {index}: invalid/duplicate id {task_id!r}")
        seen.add(task_id)
        turns: list[Turn] = []
        if has_turn_template:
            template = item["turn"]
            turn_required = {"target", "cursor", "expected", "verifier"}
            if not isinstance(template, dict) or set(template) != turn_required:
                raise ValueError(f"task {index}: 'turn' fields must be {sorted(turn_required)}")
            max_turns = item["max_turns"]
            if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 2:
                raise ValueError(f"task {index}: max_turns must be an integer >= 2")
            turns = [
                Turn(
                    turn_id=f"attempt_{turn_number}",
                    target=dict(template["target"]),
                    cursor=dict(template["cursor"]),
                    expected=dict(template["expected"]),
                    verifier=dict(template["verifier"]),
                )
                for turn_number in range(1, max_turns + 1)
            ]
        elif has_turns_list:
            raw_turns = item["turns"]
            if not isinstance(raw_turns, list) or len(raw_turns) < 2:
                raise ValueError(f"task {index}: multi-turn task needs at least two turns")
            turn_ids: set[str] = set()
            for turn_index, raw_turn in enumerate(raw_turns):
                turn_required = {"id", "target", "cursor", "expected", "verifier"}
                if not isinstance(raw_turn, dict) or set(raw_turn) != turn_required:
                    raise ValueError(
                        f"task {index} turn {turn_index}: fields must be {sorted(turn_required)}"
                    )
                turn_id = raw_turn["id"]
                if not isinstance(turn_id, str) or not turn_id or turn_id in turn_ids:
                    raise ValueError(
                        f"task {index} turn {turn_index}: invalid/duplicate id {turn_id!r}"
                    )
                turn_ids.add(turn_id)
                turns.append(
                    Turn(
                        turn_id=turn_id,
                        target=dict(raw_turn["target"]),
                        cursor=dict(raw_turn["cursor"]),
                        expected=dict(raw_turn["expected"]),
                        verifier=dict(raw_turn["verifier"]),
                    )
                )
        tasks.append(
            Task(
                task_id=task_id,
                category=str(item["category"]),
                instruction=str(item["instruction"]),
                setup=dict(item["setup"]),
                target=dict(item.get("target", {})),
                cursor=dict(item.get("cursor", {})),
                expected=dict(item.get("expected", {})),
                verifier=dict(item.get("verifier", {})),
                turns=tuple(turns),
                turn_mode=turn_mode,
            )
        )
    if not tasks:
        raise ValueError("suite contains no tasks")
    return raw, tasks


def in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x < x2 and y1 <= y < y2


def distance_to_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> float:
    """Euclidean distance to the nearest bbox point; zero anywhere inside."""
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0, x - (x2 - 1))
    dy = max(y1 - y, 0, y - (y2 - 1))
    return math.hypot(dx, dy)


def _clip_point(point: tuple[int, int], screen: tuple[int, int]) -> tuple[int, int]:
    width, height = screen
    return max(0, min(width - 1, point[0])), max(0, min(height - 1, point[1]))


def norm_bbox_to_px(
    bbox: list[int] | tuple[int, int, int, int], screen: tuple[int, int]
) -> tuple[int, int, int, int]:
    if len(bbox) != 4 or any(not isinstance(value, int) for value in bbox):
        raise ValueError(f"invalid normalized bbox {bbox!r}")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= _GRID and 0 <= y1 < y2 <= _GRID):
        raise ValueError(f"normalized bbox outside 0..{_GRID}: {bbox!r}")
    width, height = screen
    return (
        round(x1 * width / _GRID),
        round(y1 * height / _GRID),
        max(round(x1 * width / _GRID) + 1, round(x2 * width / _GRID)),
        max(round(y1 * height / _GRID) + 1, round(y2 * height / _GRID)),
    )


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[0] + bbox[2] - 1) // 2, (bbox[1] + bbox[3] - 1) // 2


def resolve_cursor_start(
    cursor: dict[str, Any],
    bbox: tuple[int, int, int, int],
    screen: tuple[int, int],
) -> tuple[int, int]:
    kind = cursor.get("kind")
    if kind == "target_center":
        return _bbox_center(bbox)
    if kind == "normalized":
        point = cursor.get("point")
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"normalized cursor needs [x,y], got {point!r}")
        return _clip_point(
            (round(point[0] * screen[0] / _GRID), round(point[1] * screen[1] / _GRID)),
            screen,
        )
    if kind == "relative_to_target":
        delta = cursor.get("delta_norm")
        if not isinstance(delta, list) or len(delta) != 2:
            raise ValueError(f"relative cursor needs delta_norm [x,y], got {delta!r}")
        center = _bbox_center(bbox)
        start = (
            center[0] + round(delta[0] * screen[0] / _GRID),
            center[1] + round(delta[1] * screen[1] / _GRID),
        )
        start = _clip_point(start, screen)
        if in_bbox(start, bbox):
            raise ValueError(f"relative cursor start {start} is inside target {bbox}")
        return start
    raise ValueError(f"unknown cursor kind {kind!r}")


def movement_metrics(
    start: tuple[int, int],
    end: tuple[int, int],
    bbox: tuple[int, int, int, int],
    screen: tuple[int, int],
) -> dict[str, float | bool]:
    start_distance = distance_to_bbox(start, bbox)
    end_distance = distance_to_bbox(end, bbox)
    best_distance = start_distance
    for scale in _MOVEMENT_SCALES:
        for dx_sign, dy_sign in _DIRECTIONS:
            candidate = _clip_point(
                (
                    start[0] + round(scale * dx_sign * screen[0] / _GRID),
                    start[1] + round(scale * dy_sign * screen[1] / _GRID),
                ),
                screen,
            )
            best_distance = min(best_distance, distance_to_bbox(candidate, bbox))
    available_gain = start_distance - best_distance
    actual_gain = start_distance - end_distance
    legal_optimality = 1.0 if available_gain <= 0 and end_distance <= start_distance else 0.0
    if available_gain > 0:
        legal_optimality = max(0.0, min(1.0, actual_gain / available_gain))
    distance_gain = (
        1.0 if start_distance == 0 else max(-1.0, min(1.0, actual_gain / start_distance))
    )
    return {
        "start_distance_px": start_distance,
        "end_distance_px": end_distance,
        "best_legal_distance_px": best_distance,
        "distance_gain": distance_gain,
        "legal_step_optimality": legal_optimality,
        "direction_correct": end_distance < start_distance,
        "bbox_hit": end_distance == 0,
    }


def denormalize_action(action: OrderedAction, screen: tuple[int, int]) -> OrderedAction:
    """Convert rel-step 0..1000 deltas to VM pixels; other primitives unchanged."""
    return OrderedAction(
        primitives=tuple(
            replace(
                primitive,
                dx=round(primitive.dx * screen[0] / _GRID),
                dy=round(primitive.dy * screen[1] / _GRID),
            )
            if primitive.kind == "move"
            else primitive
            for primitive in action.primitives
        ),
        no_op=action.no_op,
    )


def native_ordered_to_relstep(action: NativeOrderedAction) -> OrderedAction:
    """Adapt this branch's own ``ordered_events_v3`` (``cua_ordered_typing_v1``)
    parse result into the canonical shape every other action_format here
    normalizes into (see ``qwen3vl_native_to_ordered`` for the native-Qwen3VL
    twin of this function).

    ``action_parser.OrderedPrimitive`` (native) and
    ``cua_micro_action_parser.RelStepPrimitive`` (canonical, aliased to
    ``OrderedPrimitive`` in this module) agree on ``kind``/``dx``/``dy``/
    ``text``/``mouse_button`` for ``move``/``scroll``/``down``/``up``/
    ``type`` -- the only difference is the field holding the key/button name
    (``input_name`` vs ``name``), so this is a rename, not a reinterpretation.
    dx/dy stay in MODEL-resolution pixels here; ``denormalize_native_ordered_action``
    (below) scales them to VM-native pixels right before dispatch, mirroring
    how freeroll.py's own ``client.dispatch_ordered`` scales internally.
    """
    return OrderedAction(
        primitives=tuple(
            OrderedPrimitive(
                kind=p.kind,
                dx=p.dx,
                dy=p.dy,
                name=p.input_name,
                mouse_button=p.mouse_button,
                text=p.text,
            )
            for p in action.primitives
        ),
        no_op=action.no_op,
    )


def denormalize_native_ordered_action(
    action: OrderedAction,
    screen: tuple[int, int],
    model_resolution: tuple[int, int] | None,
) -> OrderedAction:
    """Scale ``cua_ordered_typing_v1`` move deltas from model-resolution
    pixels to VM-native pixels; other primitives unchanged.

    Mirrors ``denormalize_action``'s rel-step-grid-to-pixels job, but the
    source space here is whatever ``--model_resolution`` frame the model was
    actually shown (freeroll.py's convention), not the rel-step format's
    fixed 0..1000 grid. No-op when ``model_resolution`` is unset (the model
    saw native-resolution frames, so deltas are already VM pixels) --
    matches ``OSWorldClient._model_to_screen_scale``'s own identity case.
    """
    if not model_resolution:
        return action
    mw, mh = model_resolution
    return OrderedAction(
        primitives=tuple(
            replace(
                primitive,
                dx=round(primitive.dx * screen[0] / mw),
                dy=round(primitive.dy * screen[1] / mh),
            )
            if primitive.kind == "move"
            else primitive
            for primitive in action.primitives
        ),
        no_op=action.no_op,
    )


def qwen3vl_native_to_ordered(
    calls: tuple[ComputerUseCall, ...],
    screen: tuple[int, int],
    cursor_start: tuple[int, int],
) -> OrderedAction:
    """Adapt official absolute-grid Qwen3-VL calls to VM primitives."""
    primitives: list[OrderedPrimitive] = []
    cursor = cursor_start
    click_map = {
        "left_click": ("left", 1),
        "right_click": ("right", 1),
        "middle_click": ("middle", 1),
        "double_click": ("left", 2),
        "triple_click": ("left", 3),
    }
    for call in calls:
        arguments = call.arguments
        action = str(arguments["action"])
        if action == "mouse_move":
            coordinate = arguments["coordinate"]
            target = (
                max(0, min(screen[0] - 1, round(float(coordinate[0]) * screen[0] / _GRID))),
                max(0, min(screen[1] - 1, round(float(coordinate[1]) * screen[1] / _GRID))),
            )
            primitives.append(
                OrderedPrimitive(kind="move", dx=target[0] - cursor[0], dy=target[1] - cursor[1])
            )
            cursor = target
        elif action in click_map:
            button, count = click_map[action]
            primitives.append(OrderedPrimitive(kind="click", name=button, count=count))
        elif action == "type":
            primitives.append(OrderedPrimitive(kind="type", text=arguments["text"]))
        elif action == "key":
            primitives.append(OrderedPrimitive(kind="key_combo", keys=tuple(arguments["keys"])))
        elif action == "scroll":
            primitives.append(OrderedPrimitive(kind="scroll", dy=round(float(arguments["pixels"]))))
        elif action == "hscroll":
            primitives.append(OrderedPrimitive(kind="scroll", dx=round(float(arguments["pixels"]))))
        elif action == "wait":
            primitives.append(OrderedPrimitive(kind="wait"))
        elif action == "terminate":
            primitives.append(OrderedPrimitive(kind="terminate", status=arguments["status"]))
        elif action == "left_click_drag":
            coordinate = arguments["coordinate"]
            target = (
                max(0, min(screen[0] - 1, round(float(coordinate[0]) * screen[0] / _GRID))),
                max(0, min(screen[1] - 1, round(float(coordinate[1]) * screen[1] / _GRID))),
            )
            primitives.append(
                OrderedPrimitive(kind="drag", dx=target[0] - cursor[0], dy=target[1] - cursor[1])
            )
            cursor = target
        elif action == "answer":
            primitives.append(OrderedPrimitive(kind="answer", text=arguments["text"]))
        else:  # guarded by the strict native parser
            raise AssertionError(f"unhandled Qwen3-VL action {action!r}")
    return OrderedAction(primitives=tuple(primitives), no_op=False)


def serialize_action(action: OrderedAction | None) -> list[dict[str, Any]]:
    if action is None:
        return []
    return [asdict(primitive) for primitive in action.primitives]


_NATIVE_MOUSE_BUTTON_NAME = {"LMB": "left", "MMB": "middle", "RMB": "right"}

# Modifier/named-key spellings that mean the same physical key. Side-suffixed
# forms (ControlLeft/ControlRight) are folded onto the base before lookup.
_KEY_ALIASES = {
    "CONTROL": "CTRL",
    "CTRL": "CTRL",
    "SHIFT": "SHIFT",
    "ALT": "ALT",
    "OPTION": "ALT",
    "META": "META",
    "SUPER": "META",
    "WIN": "META",
    "CMD": "META",
    "COMMAND": "META",
    "RETURN": "ENTER",
    "ENTER": "ENTER",
    "ESCAPE": "ESC",
    "ESC": "ESC",
    "DELETE": "DELETE",
    "DEL": "DELETE",
}


def _canonical_key_name(name: Any) -> str:
    """Fold a key spelling onto one canonical token.

    The model is instructed (see osworld_system_prompts) to emit *rdev* names
    -- ``KeyS``, ``Num7``, ``ControlLeft`` -- while the task suite writes
    ``expected.keys`` in human form -- ``S``, ``7``, ``CTRL``. Comparing those
    with a plain ``.upper()`` never matches, which is why every key task
    scored expected_action_rate 0.0 under cua_ordered_typing_v1 even when the
    verifier confirmed the chord fired. Normalize both sides through here.
    """
    token = str(name).strip().upper()
    if not token:
        return ""
    # rdev prefixes a bare character: KeyS -> S, Num7 -> 7, Digit7 -> 7.
    for prefix in ("KEY", "DIGIT", "NUM"):
        rest = token[len(prefix) :]
        if token.startswith(prefix) and len(rest) == 1 and rest.isalnum():
            return rest
    for suffix in ("LEFT", "RIGHT"):
        if token.endswith(suffix) and token[: -len(suffix)] in _KEY_ALIASES:
            token = token[: -len(suffix)]
            break
    return _KEY_ALIASES.get(token, token)


def _releases_match_presses(downs: Any, ups: Any) -> bool:
    """True when ``ups`` releases exactly the keys ``downs`` pressed.

    Order-insensitive by design. A chord fires when its last key goes down, so
    releasing ``S, Ctrl, Shift`` versus ``S, Shift, Ctrl`` produces the same
    chord and leaves no key stuck. Demanding strict reverse order flagged a
    working Ctrl+Shift+S as wrong. Multiset equality still rejects the real
    fault -- an unbalanced chord that leaves a modifier held.
    """
    return sorted(_canonical_key_name(p.name) for p in downs) == sorted(
        _canonical_key_name(p.name) for p in ups
    )


def _repeated_mouse_click_count(primitives: list[OrderedPrimitive]) -> int:
    """Return N if ``primitives`` is exactly N consecutive down/up pairs on one
    mouse button, else 0.

    A multi-click is *interleaved* -- ``down(LMB); up(LMB); down(LMB); up(LMB)``
    for a double-click -- not the nested ``down; down; up; up`` shape a keyboard
    chord has, so it needs its own recognizer. Single clicks (N=1) go through
    here too; the shapes are the same sequence at different repeat counts.
    """
    if len(primitives) < 2 or len(primitives) % 2 != 0:
        return 0
    button = primitives[0].mouse_button
    if button is None:
        return 0
    for index in range(0, len(primitives), 2):
        down, up = primitives[index], primitives[index + 1]
        if down.kind != "down" or up.kind != "up":
            return 0
        if down.mouse_button != button or up.mouse_button != button:
            return 0
    return len(primitives) // 2


def _canonicalize_native_ordered_action(action: OrderedAction | None) -> OrderedPrimitive | None:
    """cua_ordered_typing_v1 (this branch's own ordered_events_v3) has no
    atomic "click"/"key_combo" primitive the way computer_use_rel_step_v1 /
    qwen3vl_native_cua_v1 do -- a click is a raw ``down(LMB); up(LMB)`` pair
    and a key chord is ``down(k1); down(k2); ...; up(k2); up(k1)``. Collapse
    those into the single atomic primitive action_matches_expected knows how
    to score, so cua_ordered_typing_v1 clicks/chords aren't hard-rejected by
    expected-action matching regardless of correctness (this is exactly what
    made expected_action_rate a flat 0.0 for cua_ordered_typing_v1 runs:
    parsing and dispatch worked, but nothing here ever recognized the result
    as a "click").

    Only called for cua_ordered_typing_v1 -- see action_matches_expected.
    For the other two formats a leading move ahead of a click is a genuine
    "should have been one tool call" violation (they have a dedicated click
    primitive that clicks wherever the cursor already is), so it must stay a
    hard mismatch there; for this format move-then-click-in-one-response is
    simply how "click somewhere other than the current cursor" is expressed,
    so the leading move is stripped here before classifying. Anything that
    doesn't reduce to one of the shapes below (a drag, an unbalanced chord,
    two distinct clicks) returns None, i.e. counts as a non-match -- same as
    an actually-wrong action would.
    """
    if action is None or not action.primitives:
        return None
    primitives = list(action.primitives)
    if len(primitives) > 1 and primitives[0].kind == "move":
        primitives = primitives[1:]
    if len(primitives) == 1:
        return primitives[0]
    # Mouse clicks first: they interleave (down/up, down/up), so the nested
    # chord branch below would split a double-click down the middle, see
    # `[down, up]` where it demands all-downs, and reject it -- which is why a
    # correct double-click scored expected_action_ok=false even with the
    # verifier confirming the folder opened.
    click_count = _repeated_mouse_click_count(primitives)
    if click_count:
        name = primitives[0].name
        return OrderedPrimitive(
            kind="click", name=_NATIVE_MOUSE_BUTTON_NAME.get(name, name), count=click_count
        )
    if (
        len(primitives) >= 2
        and len(primitives) % 2 == 0
        and all(p.kind in ("down", "up") for p in primitives)
    ):
        half = len(primitives) // 2
        downs, ups = primitives[:half], primitives[half:]
        if not (all(p.kind == "down" for p in downs) and all(p.kind == "up" for p in ups)):
            return None
        # A balanced keyboard chord: the ups must release exactly the keys the
        # downs pressed (order-insensitive, see _releases_match_presses).
        if all(p.mouse_button is None for p in primitives) and _releases_match_presses(downs, ups):
            return OrderedPrimitive(
                kind="key_combo", keys=tuple(_canonical_key_name(p.name) for p in downs)
            )
    return None


def _native_ordered_key_or_type_match(
    primitives: tuple[OrderedPrimitive, ...], expected: dict[str, Any]
) -> bool:
    """cua_ordered_typing_v1 responses often bracket the actual key/type
    action with unrelated setup (e.g. a Backspace clear before) or a
    follow-up (e.g. Enter to confirm after) in the same turn. Neither makes
    the response wrong by itself -- only the verifier's view of the final
    state does -- so this scans for the expected primitive anywhere in the
    sequence instead of requiring the whole response to canonicalize to
    exactly one primitive. A single alphanumeric key is also often expressed
    as ``type(char)`` rather than a raw down/up chord; both realize the same
    key press. Only used for expected.kind in ("key", "type") -- click/move/
    scroll still go through the stricter single-primitive canonicalization.
    """
    kind = expected.get("kind")
    if kind == "type":
        text = expected.get("text")
        return any(p.kind == "type" and p.text == text for p in primitives)
    if kind != "key":
        return False
    keys = tuple(_canonical_key_name(k) for k in expected.get("keys", ()))
    if not keys:
        return False
    if len(keys) == 1 and len(keys[0]) == 1:
        if any(
            p.kind == "type" and p.text is not None and p.text.upper() == keys[0]
            for p in primitives
        ):
            return True
    n = len(keys)
    for start in range(len(primitives) - 2 * n + 1):
        downs = primitives[start : start + n]
        ups = primitives[start + n : start + 2 * n]
        if not (all(p.kind == "down" for p in downs) and all(p.kind == "up" for p in ups)):
            continue
        if tuple(_canonical_key_name(p.name) for p in downs) != keys:
            continue
        if not _releases_match_presses(downs, ups):
            continue
        return True
    return False


def action_matches_expected(
    action: OrderedAction | None, expected: dict[str, Any], action_format: str = _REL_STEP_FORMAT
) -> bool:
    if expected.get("kind") == "any":
        # Outcome-only tasks (turn_mode="multiturn") don't prescribe a specific
        # primitive -- any dispatchable, non-no-op action counts, regardless
        # of format or how many primitives it bundles into one reply. Only
        # the verifier decides whether it actually solved the task.
        return action is not None and bool(action.primitives) and not action.no_op
    if (
        action_format == _NATIVE_ORDERED_FORMAT
        and expected.get("kind") in ("key", "type")
        and action is not None
        and _native_ordered_key_or_type_match(action.primitives, expected)
    ):
        return True
    if action_format == _NATIVE_ORDERED_FORMAT:
        primitive = _canonicalize_native_ordered_action(action)
    elif action is not None and len(action.primitives) == 1:
        primitive = action.primitives[0]
    else:
        primitive = None
    if primitive is None:
        return False
    kind = expected.get("kind")
    matches = False
    if kind == "move":
        matches = primitive.kind == "move"
    elif kind == "click":
        matches = (
            primitive.kind == "click"
            and primitive.name == expected.get("button", "left")
            and primitive.count == int(expected.get("count", 1))
        )
    elif kind == "type":
        matches = primitive.kind == "type" and primitive.text == expected.get("text")
    elif kind == "key":
        matches = primitive.kind == "key_combo" and tuple(
            _canonical_key_name(key) for key in primitive.keys
        ) == tuple(_canonical_key_name(key) for key in expected.get("keys", []))
    elif kind == "scroll" and (
        primitive.kind == "scroll" and primitive.dx == 0 and primitive.dy != 0
    ):
        sign = expected.get("sign")
        matches = (sign == "down" and primitive.dy < 0) or (sign == "up" and primitive.dy > 0)
    return matches


def _guest_json(client: OSWorldClient, path: str) -> dict[str, Any]:
    code = f"from pathlib import Path; print(Path({path!r}).read_text())"
    output = client.run_command(["python3", "-c", code]).get("output", "")
    return json.loads(output)


def _upload_bytes(client: OSWorldClient, path: str, payload: bytes) -> None:
    encoded = base64.b64encode(payload).decode("ascii")
    code = (
        "import base64; from pathlib import Path; "
        f"Path({path!r}).write_bytes(base64.b64decode({encoded!r}))"
    )
    client.run_command(["python3", "-c", code])


def _active_title(client: OSWorldClient) -> str:
    """Return a text blob covering every open window's title.

    Every caller treats this as a haystack for substring/regex matching, not
    strictly "the focused window" -- a just-launched app (e.g. LibreOffice
    mid-startup) can have a matching window well before it grabs focus, so
    checking only the active window produces false negatives during launch.
    Prefer tools that list *all* windows; when only xdotool is available, use
    its window search (not ``getactivewindow``) so the same "any window"
    semantics hold regardless of which tool happens to be installed in a
    given VM image.
    """
    return str(
        client.run_command(
            [
                "bash",
                "-lc",
                "if command -v wmctrl >/dev/null; then "
                "wmctrl -l; "
                "elif command -v xdotool >/dev/null; then "
                "xdotool search --name '.' getwindowname %@; "
                "elif command -v gdbus >/dev/null; then "
                "gdbus call --session --dest org.gnome.Shell.Introspect "
                "--object-path /org/gnome/Shell/Introspect "
                "--method org.gnome.Shell.Introspect.GetWindows; "
                "else exit 127; fi",
            ]
        ).get("output", "")
    ).strip()


def _wait_until(predicate: Any, *, timeout_s: float = 12.0, poll_s: float = 0.25) -> Any:
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except (RuntimeError, FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"condition not met after {timeout_s}s (last={last!r})")


def _chrome_html(variant: str) -> dict[str, str]:
    style = """
      html,body{margin:0;font-family:Arial,sans-serif;background:#f7f9fc;color:#172033}
      .hero{padding:8vh 8vw;font-size:42px;font-weight:700}
    """
    if variant == "history":
        body = """
          <div class='hero'>PAGE B — use Chrome Back</div>
          <script>
            history.replaceState({p:'a'},'', '#page-a'); document.title='PAGE_A';
            history.pushState({p:'b'},'', '#page-b'); document.title='PAGE_B';
            onpopstate=()=>{document.title='PAGE_A'};
          </script>
        """
    elif variant == "reload":
        body = """
          <div class='hero'>Reload this deterministic page</div>
          <script>
            let n=Number(sessionStorage.getItem('loads')||0)+1;
            sessionStorage.setItem('loads',String(n)); document.title='LOAD_'+n;
          </script>
        """
    elif variant == "button":
        body = """
          <div class='hero'>Click the target</div>
          <button id='complete' onclick="document.title='PASS_BUTTON'">COMPLETE</button>
          <style>#complete{position:fixed;left:40vw;top:40vh;width:20vw;height:11vh;
          border:0;border-radius:18px;background:#1769e0;color:white;font-size:34px;font-weight:700}</style>
          <script>document.title='BUTTON_READY'</script>
        """
    elif variant == "scroll":
        body = """
          <div class='hero'>Scroll down once</div><div style='height:350vh'></div>
          <script>document.title='SCROLL_READY';onscroll=()=>{if(scrollY>0)document.title='PASS_SCROLL'}</script>
        """
    elif variant == "search":
        # A deterministic local stand-in for a real search engine: any query
        # containing "3blue1brown" (case-insensitive) reveals one result;
        # clicking it "opens" the video. Outcome-verified via title changes,
        # not the exact query text or click path -- real search results are
        # non-deterministic and network-dependent, so this fixture is the
        # reproducible proxy for "did the model actually search and open
        # the right result."
        body = """
          <div id='cua-home'>
            <div class='hero'>Web Search</div>
            <form id='cua-search-form' onsubmit="cuaSubmitSearch(); return false;">
              <input id='cua-search-input' autofocus autocomplete='off' placeholder='Search the web'>
              <button id='cua-search-btn' type='submit'>Search</button>
            </form>
          </div>
          <div id='cua-results' style='display:none'>
            <div class='hero'>Search results</div>
            <button id='cua-first-result' onclick='cuaOpenFirstResult()'>
              <b>3Blue1Brown</b><br>Essence of Linear Algebra — YouTube
            </button>
          </div>
          <style>
            #cua-search-form{position:fixed;left:30vw;top:44vh;width:56vw;height:8vh;display:flex;gap:1vw}
            #cua-search-input{flex:1;font-size:22px;padding:0 16px;border:1px solid #c7ccd6;border-radius:8px}
            #cua-search-btn{width:14vw;font-size:20px;font-weight:700;border:0;border-radius:8px;background:#1769e0;color:#fff}
            #cua-first-result{position:fixed;left:15vw;top:20vh;width:70vw;height:12vh;display:block;
              text-align:left;background:#ffffff;border:1px solid #dbe2ee;border-radius:12px;
              padding:2vh 2vw;cursor:pointer;font-size:20px}
            #cua-first-result b{color:#1a0dab;font-size:24px}
          </style>
          <script>
            document.title='SEARCH_READY';
            function cuaSubmitSearch(){
              var q = document.getElementById('cua-search-input').value || '';
              if (/3blue1brown/i.test(q)) {
                document.getElementById('cua-home').style.display='none';
                document.getElementById('cua-results').style.display='block';
                document.title='RESULTS_3BLUE1BROWN';
              }
            }
            function cuaOpenFirstResult(){
              document.title='PLAYING_3BLUE1BROWN';
            }
          </script>
        """
    elif variant == "wikipedia":
        # Same contract as the "search" variant above, re-skinned for an
        # encyclopedia lookup: a deterministic local stand-in, never the real
        # wikipedia.org. Nothing in this suite has network egress (every
        # fixture is a file:// page and Chrome launches with background
        # networking disabled), and real article titles/rankings drift over
        # time, so a live search could not be verified reproducibly.
        # Two title transitions make the two halves of the goal separately
        # checkable: RESULTS_WIKIPEDIA proves a query was actually submitted,
        # PASS_TRANSFORMERS_ARTICLE proves the result was opened.
        body = """
          <div id='cua-home'>
            <div class='hero'>Web Search</div>
            <form id='cua-search-form' onsubmit="cuaSubmitSearch(); return false;">
              <input id='cua-search-input' autofocus autocomplete='off' placeholder='Search the web'>
              <button id='cua-search-btn' type='submit'>Search</button>
            </form>
          </div>
          <div id='cua-results' style='display:none'>
            <div class='hero'>Search results</div>
            <button id='cua-first-result' onclick='cuaOpenFirstResult()'>
              <b>Transformer (deep learning architecture)</b><br>
              Wikipedia — the free encyclopedia
            </button>
          </div>
          <div id='cua-article' style='display:none'>
            <div class='hero'>Transformer (deep learning architecture) — Wikipedia</div>
          </div>
          <style>
            #cua-search-form{position:fixed;left:30vw;top:44vh;width:56vw;height:8vh;display:flex;gap:1vw}
            #cua-search-input{flex:1;font-size:22px;padding:0 16px;border:1px solid #c7ccd6;border-radius:8px}
            #cua-search-btn{width:14vw;font-size:20px;font-weight:700;border:0;border-radius:8px;background:#1769e0;color:#fff}
            #cua-first-result{position:fixed;left:15vw;top:20vh;width:70vw;height:12vh;display:block;
              text-align:left;background:#ffffff;border:1px solid #dbe2ee;border-radius:12px;
              padding:2vh 2vw;cursor:pointer;font-size:20px}
            #cua-first-result b{color:#1a0dab;font-size:24px}
          </style>
          <script>
            document.title='SEARCH_READY';
            function cuaSubmitSearch(){
              var q = document.getElementById('cua-search-input').value || '';
              if (/wikipedia|transformer/i.test(q)) {
                document.getElementById('cua-home').style.display='none';
                document.getElementById('cua-results').style.display='block';
                document.title='RESULTS_WIKIPEDIA';
              }
            }
            function cuaOpenFirstResult(){
              document.getElementById('cua-results').style.display='none';
              document.getElementById('cua-article').style.display='block';
              document.title='PASS_TRANSFORMERS_ARTICLE';
            }
          </script>
        """
    else:
        body = "<div class='hero'>Chrome micro-eval ready</div><script>document.title='BLANK_READY'</script>"
    return {"/tmp/cua_micro.html": f"<!doctype html><style>{style}</style>{body}"}


def _activate_chrome_target(client: OSWorldClient, title: str) -> dict[str, Any]:
    """Activate the exact CDP page target instead of relying on CLI tab order."""
    code = (
        "import json, urllib.request; "
        "targets=json.load(urllib.request.urlopen('http://127.0.0.1:9222/json')); "
        f"target=next(t for t in targets if t.get('title') == {title!r}); "
        "urllib.request.urlopen("
        "'http://127.0.0.1:9222/json/activate/' + target['id']"
        ").read()"
    )
    return client.run_command(["python3", "-c", code])


def _close_browser_popups(client: OSWorldClient, keep_title: str) -> list[str]:
    """Force-close any top-level window whose title doesn't contain
    ``keep_title`` -- Chrome's "update available"/"Chrome is out of date"
    nag (and similar first-run/extension popups) steals focus and breaks
    every _active_title-based wait below. The launch flags in
    _launch_chrome already suppress the update check at the source
    (--disable-background-networking/--disable-component-update); this is
    the second line of defense for anything that slips through anyway.
    Best-effort: wmctrl absence or a failed close just means nothing gets
    closed, not a crash.
    """
    try:
        listing = client.run_command(
            ["bash", "-lc", "command -v wmctrl >/dev/null && wmctrl -l || true"]
        ).get("output", "")
    except RuntimeError:
        return []
    closed: list[str] = []
    for line in listing.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        win_id, _desktop, _host, title = parts
        if keep_title in title:
            continue
        try:
            client.run_command(["wmctrl", "-ic", win_id])
        except RuntimeError:
            continue
        closed.append(title)
    return closed


_CHROME_STARTUP_INSTALLER = r'''
import os, glob, shutil, subprocess, sys

URL = sys.argv[1]
# Flags mirror _launch_chrome's: without them a user-launched Chrome opens the
# first-run wizard / "make Chrome default" prompt on top of the fixture, which
# the model would have to dismiss before it could do anything.
FLAGS = ("--no-first-run --no-default-browser-check "
         "--disable-session-crashed-bubble --disable-infobars --disable-translate")

candidates = []
for root in ("/usr/share/applications", "/var/lib/snapd/desktop/applications",
             "/usr/local/share/applications"):
    candidates += sorted(glob.glob(os.path.join(root, "*chrom*.desktop")))
if not candidates:
    raise SystemExit("no chrome/chromium .desktop file found")
source = candidates[0]

target_dir = os.path.expanduser("~/.local/share/applications")
os.makedirs(target_dir, exist_ok=True)
target = os.path.join(target_dir, os.path.basename(source))

out = []
for line in open(source, encoding="utf-8", errors="replace").read().splitlines():
    if line.startswith("Exec="):
        exec_line = line[len("Exec="):]
        # Drop field codes (%U/%F/%u/%f) so the desktop entry cannot substitute
        # a caller-supplied file list in place of our URL.
        for code in ("%U", "%F", "%u", "%f"):
            exec_line = exec_line.replace(code, "")
        line = "Exec=" + " ".join(exec_line.split()) + " " + FLAGS + " " + URL
    out.append(line)
open(target, "w", encoding="utf-8").write("\n".join(out) + "\n")
os.chmod(target, 0o755)

# A stale running Chrome would keep its old tabs and ignore the new entry.
subprocess.run(["pkill", "-f", "google-chrome|chromium"], check=False)
if shutil.which("update-desktop-database"):
    subprocess.run(["update-desktop-database", target_dir], check=False)
print("installed " + target + " from " + source)
'''


def _install_chrome_startup_page(client: OSWorldClient, variant: str) -> None:
    """Make a *model-launched* Chrome open the micro-eval fixture.

    Tasks with ``setup.kind == "desktop"`` deliberately leave Chrome closed so
    the model has to open it itself, which means ``_launch_chrome`` never runs
    and the fixture URL can't be passed on the command line. Overriding the
    ``.desktop`` entry in ``~/.local/share/applications`` (which takes
    precedence over ``/usr/share/applications``) makes every launch path --
    dock icon, Activities search, terminal -- land on the fixture, and needs no
    root, unlike a Chrome managed-policy file.
    """
    for path, text in _chrome_html(variant).items():
        _upload_bytes(client, path, text.encode())
    _upload_bytes(
        client, "/tmp/cua_install_chrome_startup.py", _CHROME_STARTUP_INSTALLER.encode()
    )
    client.run_command(
        ["python3", "/tmp/cua_install_chrome_startup.py", "file:///tmp/cua_micro.html"]
    )


def _launch_chrome(client: OSWorldClient, variant: str) -> None:
    files = _chrome_html(variant)
    urls: list[str]
    activate_title: str | None = None
    if variant == "tabs":
        files = {
            "/tmp/cua_alpha.html": "<title>ALPHA</title><h1>ALPHA tab</h1>",
            "/tmp/cua_beta.html": "<title>BETA</title><h1>BETA tab</h1>",
        }
        urls = ["file:///tmp/cua_alpha.html", "file:///tmp/cua_beta.html"]
        expected_title = "BETA"
        activate_title = expected_title
    elif variant == "history":
        files = {
            "/tmp/cua_history_a.html": (
                "<title>PAGE_A</title>"
                "<a href='file:///tmp/cua_history_b.html' "
                "style='position:fixed;inset:0;display:grid;place-items:center;font-size:48px'>"
                "OPEN PAGE B</a>"
            ),
            "/tmp/cua_history_b.html": "<title>PAGE_B</title><h1>PAGE B — use Chrome Back</h1>",
        }
        urls = ["file:///tmp/cua_history_a.html"]
        expected_title = "PAGE_A"
    else:
        urls = ["file:///tmp/cua_micro.html"]
        expected_title = {
            "history": "PAGE_B",
            "reload": "LOAD_1",
            "button": "BUTTON_READY",
            "scroll": "SCROLL_READY",
            "search": "SEARCH_READY",
            "blank": "BLANK_READY",
        }.get(variant, "BLANK_READY")
    for path, text in files.items():
        _upload_bytes(client, path, text.encode())
    quoted_urls = " ".join(f"'{url}'" for url in urls)
    command = (
        "CHROME=$(command -v google-chrome || command -v chromium || command -v chromium-browser); "
        'test -n "$CHROME"; '
        'nohup env DISPLAY=:0 "$CHROME" --user-data-dir=/tmp/cua-micro-chrome '
        "--no-first-run --no-default-browser-check --disable-session-crashed-bubble "
        "--disable-infobars --disable-translate --disable-background-networking "
        "--disable-component-update --disable-default-apps --disable-sync "
        "--start-maximized --remote-debugging-port=9222 "
        f"{quoted_urls} >/tmp/cua_micro_chrome.log 2>&1 &"
    )
    client.run_command(command, shell=True)
    time.sleep(1.0)
    _close_browser_popups(client, keep_title=expected_title)
    if activate_title is not None:
        _wait_until(lambda: _activate_chrome_target(client, activate_title), timeout_s=40)
    _wait_until(lambda: expected_title in _active_title(client), timeout_s=40)
    if variant == "history":
        width, height = client.screen_size()
        client.execute(f"pyautogui.click(x={width // 2}, y={height // 2}, button='left')")
        _wait_until(lambda: "PAGE_B" in _active_title(client), timeout_s=20)
        client.execute("pyautogui.hotkey('alt', 'left')")
        _wait_until(lambda: "PAGE_A" in _active_title(client), timeout_s=20)
        client.execute("pyautogui.hotkey('alt', 'right')")
        _wait_until(lambda: "PAGE_B" in _active_title(client), timeout_s=20)


def _launch_fixture(client: OSWorldClient, mode: str) -> dict[str, Any]:
    fixture_path = Path(__file__).with_name("cua_micro_fixture.py")
    _upload_bytes(client, _FIXTURE_GUEST_PATH, fixture_path.read_bytes())
    command = (
        f"rm -f {_FIXTURE_STATE_PATH}; "
        f"nohup env DISPLAY=:0 python3 {_FIXTURE_GUEST_PATH} --mode {mode} "
        f"--state {_FIXTURE_STATE_PATH} >/tmp/cua_micro_fixture.log 2>&1 &"
    )
    client.run_command(command, shell=True)

    def ready_state() -> dict[str, Any] | None:
        value = _guest_json(client, _FIXTURE_STATE_PATH)
        return value if value.get("ready") and value.get("mode") == mode else None

    state = _wait_until(ready_state, timeout_s=15)
    time.sleep(0.3)
    return state


def _launch_native_app(client: OSWorldClient, app: str) -> dict[str, Any]:
    commands = {
        "files": (
            "rm -rf /tmp/cua_native_files; "
            "mkdir -p /tmp/cua_native_files/EvalTarget; "
            "printf 'native files task\\n' >/tmp/cua_native_files/Alpha.txt; "
            "gsettings set org.gnome.nautilus.preferences default-folder-viewer 'list-view'; "
            "nohup env DISPLAY=:0 nautilus --new-window /tmp/cua_native_files "
            ">/tmp/cua_native_files.log 2>&1 &"
        ),
        "writer": (
            "nohup env DISPLAY=:0 libreoffice --writer --nologo --norestore --nolockcheck "
            ">/tmp/cua_native_writer.log 2>&1 &"
        ),
        "calc": (
            "nohup env DISPLAY=:0 libreoffice --calc --nologo --norestore --nolockcheck "
            ">/tmp/cua_native_calc.log 2>&1 &"
        ),
        "impress": (
            "nohup env DISPLAY=:0 libreoffice --impress --nologo --norestore --nolockcheck "
            ">/tmp/cua_native_impress.log 2>&1 &"
        ),
        "text_editor": (
            f": >{_NATIVE_EDITOR_PATH}; "
            f"nohup env DISPLAY=:0 gnome-text-editor {_NATIVE_EDITOR_PATH} "
            ">/tmp/cua_native_editor.log 2>&1 &"
        ),
        "calculator": (
            "nohup env DISPLAY=:0 gnome-calculator >/tmp/cua_native_calculator.log 2>&1 &"
        ),
        # A plain interactive shell -- unlike native_terminal_capture/_sequence
        # (which pre-compute an exact expected byte count for one strict
        # typed string), this is for open-ended tasks: the model can run
        # whatever commands it wants and we only check the resulting outcome
        # afterward via a separate run_command call, not the terminal's own
        # scrollback.
        "terminal": (
            "rm -f ~/hello_world.py; "
            "nohup env DISPLAY=:0 gnome-terminal --title='CUA Terminal' -- bash -i "
            ">/tmp/cua_native_terminal_open.log 2>&1 &"
        ),
    }
    patterns = {
        "files": r"cua_native_files|Files",
        "writer": r"LibreOffice Writer",
        "calc": r"LibreOffice Calc",
        "impress": r"LibreOffice Impress",
        "text_editor": r"cua_native_editor|Text Editor",
        "calculator": r"Calculator",
        "terminal": r"CUA Terminal",
    }
    if app not in commands:
        raise ValueError(f"unknown native app {app!r}")
    client.run_command(commands[app], shell=True)
    pattern = re.compile(patterns[app], re.IGNORECASE)
    _wait_until(lambda: pattern.search(_active_title(client)), timeout_s=30)
    time.sleep(0.8)
    if app == "text_editor":
        width, height = client.screen_size()
        client.execute(f"pyautogui.click(x={width // 2}, y={height // 2}, button='left')")
    return {"native_app": app}


def _launch_native_terminal_capture(client: OSWorldClient, text: str) -> dict[str, Any]:
    script_path = "/tmp/cua_native_terminal_capture.py"
    script = f"""import json, os, sys, termios, time, tty
from pathlib import Path
state = Path({_NATIVE_TERMINAL_STATE_PATH!r})
state.write_text(json.dumps({{'ready': True, 'value': ''}}))
fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
data = bytearray()
try:
    tty.setraw(fd)
    while len(data) < {len(text.encode("utf-8"))}:
        data.extend(os.read(fd, 1))
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
value = data.decode('utf-8', errors='replace')
state.write_text(json.dumps({{'ready': True, 'value': value}}))
print('\\r\\nCaptured:', value, flush=True)
time.sleep(30)
"""
    _upload_bytes(client, script_path, script.encode())
    command = (
        f"rm -f {_NATIVE_TERMINAL_STATE_PATH}; "
        "nohup env DISPLAY=:0 gnome-terminal --title='CUA Native Terminal' -- "
        f"python3 {script_path} >/tmp/cua_native_terminal.log 2>&1 &"
    )
    client.run_command(command, shell=True)
    _wait_until(lambda: "CUA Native Terminal" in _active_title(client), timeout_s=20)
    state = _wait_until(
        lambda: (
            value
            if (value := _guest_json(client, _NATIVE_TERMINAL_STATE_PATH)).get("ready")
            else None
        ),
        timeout_s=10,
    )
    time.sleep(0.5)
    return state


def _launch_native_terminal_sequence(client: OSWorldClient, task: Task) -> dict[str, Any]:
    chunks: list[str] = []
    for turn in task_turns(task):
        expected = turn.expected
        if expected.get("kind") == "type":
            chunks.append(str(expected["text"]))
        elif expected.get("kind") == "key" and [
            str(key).upper() for key in expected.get("keys", [])
        ] in (["ENTER"], ["RETURN"]):
            chunks.append("\n")
        else:
            raise ValueError("native terminal sequence supports exact type and Enter turns only")
    total_bytes = len("".join(chunks).encode("utf-8"))
    script_path = "/tmp/cua_native_terminal_sequence.py"
    script = f"""import json, os, sys, termios, time
from pathlib import Path
state = Path({_NATIVE_TERMINAL_STATE_PATH!r})
state.write_text(json.dumps({{'ready': True, 'value': ''}}))
fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
new = termios.tcgetattr(fd)
new[3] &= ~termios.ICANON
new[6][termios.VMIN] = 1
new[6][termios.VTIME] = 0
data = bytearray()
try:
    termios.tcsetattr(fd, termios.TCSANOW, new)
    while len(data) < {total_bytes}:
        byte = os.read(fd, 1)
        data.extend(byte)
        value = data.decode('utf-8', errors='replace')
        state.write_text(json.dumps({{'ready': True, 'value': value}}))
        if byte == b'\\n':
            print('next> ', end='', flush=True)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
time.sleep(30)
"""
    _upload_bytes(client, script_path, script.encode())
    command = (
        f"rm -f {_NATIVE_TERMINAL_STATE_PATH}; "
        "nohup env DISPLAY=:0 gnome-terminal --title='CUA Terminal Sequence' -- "
        f"python3 {script_path} >/tmp/cua_native_terminal_sequence.log 2>&1 &"
    )
    client.run_command(command, shell=True)
    _wait_until(lambda: "CUA Terminal Sequence" in _active_title(client), timeout_s=20)
    state = _wait_until(
        lambda: (
            value
            if (value := _guest_json(client, _NATIVE_TERMINAL_STATE_PATH)).get("ready")
            else None
        ),
        timeout_s=10,
    )
    time.sleep(0.5)
    return state


def prepare_task(client: OSWorldClient, task: Task) -> dict[str, Any]:
    kind = task.setup.get("kind")
    if kind == "desktop":
        # Optional: preload a Chrome fixture the model will reach by opening
        # Chrome itself (see _install_chrome_startup_page).
        chrome_startup = task.setup.get("chrome_startup")
        if chrome_startup:
            _install_chrome_startup_page(client, str(chrome_startup))
        try:
            client.run_command(
                [
                    "bash",
                    "-lc",
                    "if command -v wmctrl >/dev/null; then wmctrl -k on; else exit 127; fi",
                ],
            )
        except RuntimeError:
            client.execute("pyautogui.hotkey('win', 'd')")
        time.sleep(0.8)
        return {}
    if kind == "chrome":
        _launch_chrome(client, str(task.setup.get("variant", "blank")))
        return {}
    if kind == "fixture":
        return _launch_fixture(client, str(task.setup["mode"]))
    if kind == "native_app":
        return _launch_native_app(client, str(task.setup["app"]))
    if kind == "native_terminal_capture":
        # task.expected is empty for turn_mode="multiturn" tasks -- the shared
        # expected text lives per-turn (see task_turns); every turn's is
        # identical here since it's the compact 'turn'+'max_turns' template.
        return _launch_native_terminal_capture(
            client, str(task_turns(task)[0].expected["text"])
        )
    if kind == "native_terminal_sequence":
        return _launch_native_terminal_sequence(client, task)
    raise ValueError(f"unknown setup kind {kind!r}")


def resolve_target_bbox(
    client: OSWorldClient,
    task: Task,
    setup_state: dict[str, Any],
    screen: tuple[int, int],
) -> tuple[int, int, int, int]:
    kind = task.target.get("kind")
    if kind == "fixed_norm":
        return norm_bbox_to_px(task.target["bbox"], screen)
    if kind == "fixture_widget":
        state = setup_state or _guest_json(client, _FIXTURE_STATE_PATH)
        bbox = state.get("widgets", {}).get(task.target.get("widget"))
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"fixture did not expose target bbox: {task.target!r}")
        return tuple(int(value) for value in bbox)
    if kind == "window_control":
        state = setup_state or _guest_json(client, _FIXTURE_STATE_PATH)
        content = state.get("widgets", {}).get("__window_content__")
        if not isinstance(content, list) or len(content) != 4:
            raise ValueError("fixture did not expose window content geometry")
        _x1, y1, x2, _y2 = (int(value) for value in content)
        control = task.target.get("control")
        index = {"close": 0, "maximize": 1, "minimize": 2}.get(control)
        if index is None:
            raise ValueError(f"unknown window control {control!r}")
        right = x2 - index * 42
        bbox = (right - 40, max(0, y1 - 38), right, y1)
        return (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(screen[0], bbox[2]),
            min(screen[1], bbox[3]),
        )
    raise ValueError(f"unknown target kind {kind!r}")


def read_verifier_state(  # noqa: PLR0911
    client: OSWorldClient, verifier: dict[str, Any]
) -> Any:
    kind = verifier.get("kind")
    if kind == "bbox_hit":
        return None
    if kind == "active_title_regex":
        return _active_title(client)
    if kind == "fixture_equals":
        return _guest_json(client, _FIXTURE_STATE_PATH).get("values", {}).get(verifier.get("field"))
    if kind == "guest_json_equals":
        return _guest_json(client, str(verifier["path"])).get(verifier.get("field", "value"))
    if kind == "saved_file_equals":
        client.execute("pyautogui.hotkey('ctrl', 's')")
        time.sleep(0.4)
        return str(
            client.run_command(
                [
                    "python3",
                    "-c",
                    f"from pathlib import Path; print(Path({str(verifier['path'])!r}).read_text())",
                ]
            ).get("output", "")
        ).rstrip("\n")
    if kind == "calculator_clipboard_equals":
        client.execute("pyautogui.hotkey('ctrl', 'c')")
        code = (
            "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.update(); "
            "\ntry: print(r.clipboard_get())\nexcept tk.TclError: print('')\nfinally: r.destroy()"
        )
        return str(client.run_command(["python3", "-c", code]).get("output", "")).strip()
    if kind == "clipboard_equals":
        client.execute("pyautogui.hotkey('ctrl', 'c')")
        code = (
            "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.update(); "
            "\ntry: print(r.clipboard_get())\nexcept tk.TclError: print('')\nfinally: r.destroy()"
        )
        return str(client.run_command(["python3", "-c", code]).get("output", "")).rstrip("\n")
    if kind == "guest_command_regex":
        try:
            return str(client.run_command(str(verifier["command"]), shell=True).get("output", ""))
        except RuntimeError:
            return ""
    raise ValueError(f"unknown verifier kind {kind!r}")


def verifier_passed(client: OSWorldClient, verifier: dict[str, Any]) -> tuple[bool, Any]:
    kind = verifier.get("kind")
    if kind == "bbox_hit":
        return True, None

    def check() -> Any:
        state = read_verifier_state(client, verifier)
        if kind == "active_title_regex":
            return state if re.search(str(verifier["pattern"]), str(state), re.IGNORECASE) else None
        if kind == "fixture_equals":
            return {"matched": True, "value": state} if state == verifier.get("value") else None
        if kind in {
            "guest_json_equals",
            "saved_file_equals",
            "calculator_clipboard_equals",
            "clipboard_equals",
        }:
            return {"matched": True, "value": state} if state == verifier.get("value") else None
        if kind == "guest_command_regex":
            return state if re.search(str(verifier["pattern"]), str(state), re.IGNORECASE) else None
        return None

    try:
        # 20s matches the app-launch waits used elsewhere in this file (e.g.
        # _launch_chrome, _launch_native_app) -- a cold LibreOffice launch can
        # take close to 8-10s in the VM, so the old 8s ceiling produced false
        # negatives on a correct click that was simply still loading.
        matched = _wait_until(check, timeout_s=20)
    except TimeoutError:
        state = read_verifier_state(client, verifier)
        return False, state
    if isinstance(matched, dict) and "value" in matched:
        return True, matched["value"]
    return True, matched


def _draw_overlay(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width = max(3, round(image.width / 500))
    draw.rectangle(bbox, outline="#ff304f", width=width)
    draw.line((start, end), fill="#00b8ff", width=width)
    radius = max(5, round(image.width / 250))
    draw.ellipse(
        (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
        fill="#008cff",
    )
    draw.ellipse(
        (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
        fill="#2ac769",
    )
    text = f"target={label}  start={start}  end={end}"
    draw.rectangle((8, 8, 12 + len(text) * 7, 34), fill="#ffffff")
    draw.text((12, 13), text, fill="#111111")
    return overlay


def _finalize_multiturn_result(
    turn_results: list[dict[str, Any]], turns: tuple[Turn, ...], turn_mode: str
) -> tuple[int, bool, bool, float]:
    """Compute (verified_prefix, completed, success, progress) once a
    multi-turn attempt has stopped (naturally or via an early break).

    ``"prefix"``: turns are distinct ordered sub-goals -- ``verified_prefix``
    is the length of the leading run of turn successes, and the attempt only
    succeeds if every turn passed in order.

    ``"multiturn"``: every turn shares the same end goal and is one try out
    of a fixed budget -- the attempt succeeds as soon as ANY turn's verifier
    passes, however many tries it took; ``progress`` is binary since there is
    no single ordered path whose prefix defines partial credit.
    """
    if turn_mode == "multiturn":
        success = any(bool(row["success"]) for row in turn_results)
        verified_prefix = len(turn_results) if success else 0
        completed = success or len(turn_results) == len(turns)
        progress = 1.0 if success else 0.0
        return verified_prefix, completed, success, progress
    verified_prefix = 0
    for row in turn_results:
        if not row["success"]:
            break
        verified_prefix += 1
    completed = len(turn_results) == len(turns)
    success = completed and verified_prefix == len(turns)
    progress = verified_prefix / len(turns)
    return verified_prefix, completed, success, progress


def run_multiturn_attempt(
    *,
    client: OSWorldClient,
    task: Task,
    output_dir: Path,
    sglang_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    action_format: str,
    sampling: SamplingParams,
    seed: int,
    model_resolution: tuple[int, int] | None,
    save_frames: bool,
    settle_s: float,
    n_history_frames: int | None = None,
) -> dict[str, Any]:
    """Run one stateful trajectory, stopping at the first unverified turn.

    ``n_history_frames`` bounds how many past frames the model is shown each
    turn (mirrors freeroll.py's ``append_turn``/StreamingLLM-style block
    eviction, see ``osworld_runtime.evict_history``): once the window would
    exceed it, only the newest ``n_history_frames // 2`` frames are kept.
    ``None`` (the default) means unbounded -- history grows for the whole
    trajectory, matching this function's original behavior. This matters now
    that ``turn_mode="multiturn"`` tasks can run dozens of turns in one attempt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_state = prepare_task(client, task)
    screen = client.screen_size()
    turns = task_turns(task)
    recent_frames: list[Image.Image] = []
    recent_actions: list[str] = []
    turn_results: list[dict[str, Any]] = []
    conversation_turns: list[dict[str, Any]] = []
    started = time.time()
    steps_dir = output_dir / "steps"
    if save_frames:
        steps_dir.mkdir(exist_ok=True)
    # See the matching comment in run_attempt: this is what makes the run
    # browsable in labctl's rollout viewer.
    traj_path = output_dir / "trajectory.jsonl"
    traj_entries: list[dict[str, Any]] = [
        {
            "step_num": 0,
            "action": "<reset>",
            "response": "<reset>",
            "reward": 0.0,
            "done": False,
            "info": {},
        }
    ]
    traj_path.write_text(json.dumps(traj_entries[0]) + "\n")
    instruction = (
        f"GOAL: {task.instruction}" if action_format == _REL_STEP_FORMAT else task.instruction
    )

    for turn_index, turn in enumerate(turns):
        turn_task = replace(
            task,
            target=turn.target,
            cursor=turn.cursor,
            expected=turn.expected,
            verifier=turn.verifier,
            turns=(),
        )
        bbox = resolve_target_bbox(client, turn_task, setup_state, screen)
        # Only warp the cursor once, at the very start of the trajectory.
        # Every later turn -- whether "prefix" (a distinct ordered sub-goal)
        # or "multiturn" (another try at the same end goal) -- is the SAME
        # continuous trajectory as the one before it, so it must inherit
        # wherever the previous turn's dispatched action actually left the
        # cursor, not teleport back to a fixed point.
        if turn_index == 0:
            start = resolve_cursor_start(turn.cursor, bbox, screen)
            client.execute(f"pyautogui.moveTo({start[0]}, {start[1]})")
        actual_start = client.cursor_position()
        before_state = read_verifier_state(client, turn.verifier)
        before = client.screenshot_settled(min_delay_s=0.2, stability_timeout_s=1.0)
        frame_name = f"step_{turn_index:03d}.png"
        if save_frames:
            before.save(steps_dir / frame_name)
        model_frame = before
        if model_resolution and model_resolution != before.size:
            model_frame = before.resize(model_resolution, Image.Resampling.LANCZOS)
        recent_frames.append(model_frame)
        if n_history_frames is not None:
            # Evict now, before the model call this turn, so it never sees
            # more than n_history_frames frames -- matches append_turn's
            # invariant of len(recent_actions) == len(recent_frames) - 1
            # (recent_actions hasn't grown to match the new frame yet).
            evict_history(recent_frames, recent_actions, n_history_frames=n_history_frames)

        frame_labels = window_frame_labels(turn_index + 1, len(recent_frames))
        messages = build_loggable_messages(
            system_prompt=system_prompt,
            instruction=instruction,
            recent_actions=recent_actions,
            frame_labels=frame_labels,
        )
        (output_dir / f"prompt_{turn_index + 1:03d}.json").write_text(
            json.dumps(messages, indent=2)
        )
        turn_seed = seed + turn_index * 100_000
        t0 = time.time()
        response, finish_reason = _call_model(
            sglang_url=sglang_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            instruction=instruction,
            recent_frames=recent_frames,
            recent_actions=recent_actions,
            fresh_visual_context=False,
            sampling=sampling,
            seed=turn_seed,
        )
        parse_error: str | None = None
        dispatch_error: str | None = None
        parsed: OrderedAction | None = None
        dispatched = False
        if finish_reason == "length":
            parse_error = "response truncated at max_tokens; nothing dispatched"
        else:
            try:
                if action_format == _REL_STEP_FORMAT:
                    parsed = parse_computer_use_rel_step_action(response)
                elif action_format == _QWEN3VL_NATIVE_FORMAT:
                    calls = parse_qwen3vl_computer_use_action(response)
                    parsed = qwen3vl_native_to_ordered(calls, screen, actual_start)
                elif action_format == _NATIVE_ORDERED_FORMAT:
                    parsed = native_ordered_to_relstep(parse_ordered_action_tolerant(response))
                else:
                    raise ValueError(f"unknown action format {action_format!r}")
            except (TypeError, ValueError) as error:
                parse_error = str(error)

        # The model believes the trajectory is done (success or give-up) --
        # honor that immediately rather than dispatching it as an action.
        # Without this, turn_mode="multiturn" tasks (no per-turn break-on-failure)
        # kept burning the rest of their turn budget on a task the model had
        # already abandoned, e.g. type.text_editor.native_exact running all
        # 32 turns after the model gave up following an early mismatch.
        terminated = parsed is not None and any(
            primitive.kind == "terminate" for primitive in parsed.primitives
        )
        if parsed is not None and not terminated:
            try:
                if action_format == _REL_STEP_FORMAT:
                    dispatch_action = denormalize_action(parsed, screen)
                elif action_format == _NATIVE_ORDERED_FORMAT:
                    dispatch_action = denormalize_native_ordered_action(
                        parsed, screen, model_resolution
                    )
                else:
                    dispatch_action = parsed
                client.dispatch_ordered_action(dispatch_action)
                dispatched = True
            except (TypeError, ValueError) as error:
                dispatch_error = str(error)
                client.release_all_inputs()

        after = client.screenshot_settled(
            min_delay_s=settle_s,
            stability_timeout_s=1.5,
            poll_s=0.15,
        )
        end = client.cursor_position()
        verifier_ok, after_state = verifier_passed(client, turn.verifier)
        expected_ok = action_matches_expected(parsed, turn.expected, action_format)
        metrics = movement_metrics(actual_start, end, bbox, screen)
        click_in_bbox = in_bbox(actual_start, bbox)
        if turn.expected.get("kind") == "move":
            success = bool(expected_ok and metrics["bbox_hit"])
        else:
            location_ok = click_in_bbox if turn.expected.get("kind") == "click" else True
            success = bool(expected_ok and verifier_ok and location_ok)

        if save_frames:
            # NOT `step_{turn_index+1}.png` -- every turn resets the cursor
            # before capturing its own "before" frame, so that name is the
            # next turn's before-frame and would silently overwrite this
            # turn's real result (see the `_after` convention already used
            # by the synthetic-validation path below).
            after.save(steps_dir / f"step_{turn_index:03d}_after.png")
            _draw_overlay(
                before,
                bbox,
                actual_start,
                end,
                str(turn.target.get("label", turn.turn_id)),
            ).save(output_dir / f"overlay_{turn_index + 1:03d}.png")
        turn_result = {
            "turn": turn_index + 1,
            "turn_id": turn.turn_id,
            "seed": turn_seed,
            "target": {**turn.target, "bbox_px": list(bbox)},
            "cursor_start": list(actual_start),
            "cursor_end": list(end),
            "response": response,
            "finish_reason": finish_reason,
            "parse_valid": parse_error is None,
            "parse_error": parse_error,
            "dispatch_error": dispatch_error,
            "parsed_primitives": serialize_action(parsed),
            "dispatched": dispatched,
            "expected": turn.expected,
            "expected_action_ok": expected_ok,
            "click_in_bbox": click_in_bbox,
            "verifier": turn.verifier,
            "verifier_before": before_state,
            "verifier_after": after_state,
            "verifier_pass": verifier_ok,
            "movement": metrics,
            "success": success,
            "elapsed_s": time.time() - t0,
        }
        turn_results.append(turn_result)
        traj_entries.append(
            {
                "step_num": turn_index + 1,
                "action": response,
                "response": response,
                "reward": 1.0 if success else 0.0,
                "done": (not success) or turn_index + 1 == len(turns),
                "info": turn_result,
            }
        )
        traj_path.write_text("\n".join(json.dumps(e) for e in traj_entries) + "\n")
        conversation_turns.append(
            {
                "turn": turn_index + 1,
                "turn_id": turn.turn_id,
                "messages": messages,
                "response": response,
                "finish_reason": finish_reason,
                "seed": turn_seed,
            }
        )
        recent_actions.append(response)
        if terminated:
            break  # model ended the trajectory itself -- nothing left to try
        if task.turn_mode == "multiturn":
            if success:
                break  # goal reached -- stop spending the turn budget
        elif not success:
            break

    if save_frames and turn_results:
        # The loop only ever writes before-frames under the viewer's
        # `step_{n}.png` name, so the terminal state -- trajectory.jsonl's last
        # entry, step_num == len(turn_results) -- had no frame at all and the
        # rollout viewer showed a blank final step. The last turn's `after`
        # frame IS that state; give it the terminal index too. No collision:
        # the loop has exited, so no later turn claims this name.
        after.save(steps_dir / f"step_{len(turn_results):03d}.png")

    verified_prefix, completed, success, progress = _finalize_multiturn_result(
        turn_results, turns, task.turn_mode
    )
    result = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "validity": "valid",
        "infra_error": None,
        "task_id": task.task_id,
        "category": task.category,
        "instruction": task.instruction,
        "seed": seed,
        "screen_size": list(screen),
        "model_resolution": list(model_resolution) if model_resolution else None,
        "n_history_frames": n_history_frames,
        "action_format": action_format,
        "multi_turn": True,
        "turns_total": len(turns),
        "turns_attempted": len(turn_results),
        "verified_prefix": verified_prefix,
        "turn_completion_rate": len(turn_results) / len(turns),
        "turn_parse_valid_rate": sum(bool(row["parse_valid"]) for row in turn_results) / len(turns),
        "turn_expected_action_rate": sum(bool(row["expected_action_ok"]) for row in turn_results)
        / len(turns),
        "turn_verifier_rate": sum(bool(row["verifier_pass"]) for row in turn_results) / len(turns),
        "turns": turn_results,
        "parse_valid": completed and all(bool(row["parse_valid"]) for row in turn_results),
        "expected_action_ok": completed
        and all(bool(row["expected_action_ok"]) for row in turn_results),
        "verifier_pass": completed and all(bool(row["verifier_pass"]) for row in turn_results),
        "progress": progress,
        "success": success,
        "stop_reason": (
            None
            if success
            else f"model output terminate after turn {len(turn_results)}"
            if terminated
            else f"trajectory stopped after turn {len(turn_results)}"
        ),
        "elapsed_s": time.time() - started,
    }
    (output_dir / "conversation.json").write_text(
        json.dumps({"instruction": instruction, "turns": conversation_turns}, indent=2)
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def run_attempt(
    *,
    client: OSWorldClient,
    task: Task,
    output_dir: Path,
    sglang_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    action_format: str,
    sampling: SamplingParams,
    seed: int,
    model_resolution: tuple[int, int] | None,
    save_frames: bool,
    settle_s: float,
    n_history_frames: int | None = None,
) -> dict[str, Any]:
    if task.turns:
        return run_multiturn_attempt(
            client=client,
            task=task,
            output_dir=output_dir,
            sglang_url=sglang_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            action_format=action_format,
            sampling=sampling,
            seed=seed,
            model_resolution=model_resolution,
            save_frames=save_frames,
            settle_s=settle_s,
            n_history_frames=n_history_frames,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_state = prepare_task(client, task)
    screen = client.screen_size()
    bbox = resolve_target_bbox(client, task, setup_state, screen)
    start = resolve_cursor_start(task.cursor, bbox, screen)
    client.execute(f"pyautogui.moveTo({start[0]}, {start[1]})")
    actual_start = client.cursor_position()
    before_state = read_verifier_state(client, task.verifier)
    before = client.screenshot_settled(min_delay_s=0.2, stability_timeout_s=1.0)
    steps_dir = output_dir / "steps"
    if save_frames:
        steps_dir.mkdir(exist_ok=True)
        before.save(steps_dir / "step_000.png")
    # labctl's rollout viewer (the "screen + action side by side" view freeroll.py
    # supports) reads trajectory.jsonl + steps/step_NNN.png next to it -- see
    # labctl::server::resolve_rollout_paths. Written incrementally like
    # freeroll's own writer so a crash mid-attempt still leaves step 0 viewable.
    traj_path = output_dir / "trajectory.jsonl"
    traj_path.write_text(
        json.dumps(
            {
                "step_num": 0,
                "action": "<reset>",
                "response": "<reset>",
                "reward": 0.0,
                "done": False,
                "info": {},
            }
        )
        + "\n"
    )

    model_frame = before
    if model_resolution and model_resolution != before.size:
        model_frame = before.resize(model_resolution, Image.Resampling.LANCZOS)
    instruction = (
        f"GOAL: {task.instruction}" if action_format == _REL_STEP_FORMAT else task.instruction
    )
    messages = build_loggable_messages(
        system_prompt=system_prompt,
        instruction=instruction,
        recent_actions=None,
        frame_labels=["step_000.png"],
    )
    (output_dir / "prompt.json").write_text(json.dumps(messages, indent=2))

    t0 = time.time()
    response, finish_reason = _call_model(
        sglang_url=sglang_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        instruction=instruction,
        recent_frames=[model_frame],
        recent_actions=None,
        fresh_visual_context=True,
        sampling=sampling,
        seed=seed,
    )
    parse_error: str | None = None
    dispatch_error: str | None = None
    parsed: OrderedAction | None = None
    dispatched = False
    if finish_reason == "length":
        parse_error = "response truncated at max_tokens; nothing dispatched"
    else:
        try:
            if action_format == _REL_STEP_FORMAT:
                parsed = parse_computer_use_rel_step_action(response)
            elif action_format == _QWEN3VL_NATIVE_FORMAT:
                calls = parse_qwen3vl_computer_use_action(response)
                parsed = qwen3vl_native_to_ordered(calls, screen, actual_start)
            elif action_format == _NATIVE_ORDERED_FORMAT:
                parsed = native_ordered_to_relstep(parse_ordered_action_tolerant(response))
            else:
                raise ValueError(f"unknown action format {action_format!r}")
        except (TypeError, ValueError) as error:
            parse_error = str(error)

    if parsed is not None:
        try:
            if any(primitive.kind == "terminate" for primitive in parsed.primitives):
                raise ValueError("terminate is not valid for an atomic micro-task")
            if action_format == _REL_STEP_FORMAT:
                dispatch_action = denormalize_action(parsed, screen)
            elif action_format == _NATIVE_ORDERED_FORMAT:
                dispatch_action = denormalize_native_ordered_action(
                    parsed, screen, model_resolution
                )
            else:
                dispatch_action = parsed
            client.dispatch_ordered_action(dispatch_action)
            dispatched = True
        except (TypeError, ValueError) as error:
            dispatch_error = str(error)
            client.release_all_inputs()

    after = client.screenshot_settled(
        min_delay_s=settle_s,
        stability_timeout_s=1.5,
        poll_s=0.15,
    )
    end = client.cursor_position()
    verifier_ok, after_state = verifier_passed(client, task.verifier)
    expected_ok = action_matches_expected(parsed, task.expected, action_format)
    metrics = movement_metrics(actual_start, end, bbox, screen)
    click_in_bbox = in_bbox(actual_start, bbox)
    if task.expected.get("kind") == "move":
        success = bool(expected_ok and metrics["bbox_hit"])
        progress = max(0.0, float(metrics["legal_step_optimality"]))
    else:
        location_ok = click_in_bbox if task.expected.get("kind") == "click" else True
        success = bool(expected_ok and verifier_ok and location_ok)
        progress = 1.0 if success else 0.0

    if save_frames:
        after.save(steps_dir / "step_001.png")
        _draw_overlay(
            before,
            bbox,
            actual_start,
            end,
            str(task.target.get("label", task.task_id)),
        ).save(output_dir / "overlay.png")
    conversation = {
        "messages": messages,
        "response": response,
        "finish_reason": finish_reason,
        "seed": seed,
    }
    (output_dir / "conversation.json").write_text(json.dumps(conversation, indent=2))
    result = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "validity": "valid",
        "infra_error": None,
        "task_id": task.task_id,
        "category": task.category,
        "instruction": task.instruction,
        "seed": seed,
        "screen_size": list(screen),
        "model_resolution": list(model_resolution) if model_resolution else None,
        "target": {**task.target, "bbox_px": list(bbox)},
        "cursor_start": list(actual_start),
        "cursor_end": list(end),
        "response": response,
        "finish_reason": finish_reason,
        "action_format": action_format,
        "parse_valid": parse_error is None,
        "parse_error": parse_error,
        "dispatch_error": dispatch_error,
        "parsed_primitives": serialize_action(parsed),
        "dispatched": dispatched,
        "expected_action_ok": expected_ok,
        "click_in_bbox": click_in_bbox,
        "verifier": task.verifier,
        "verifier_before": before_state,
        "verifier_after": after_state,
        "verifier_pass": verifier_ok,
        "movement": metrics,
        "progress": progress,
        "success": success,
        "elapsed_s": time.time() - t0,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    with traj_path.open("a") as traj_f:
        traj_f.write(
            json.dumps(
                {
                    "step_num": 1,
                    "action": response,
                    "response": response,
                    "reward": 1.0 if success else 0.0,
                    "done": True,
                    "info": result,
                }
            )
            + "\n"
        )
    return result


def _synthetic_action(
    expected: dict[str, Any],
    *,
    bbox: tuple[int, int, int, int],
    start: tuple[int, int],
) -> OrderedAction:
    kind = expected["kind"]
    if kind == "move":
        center = _bbox_center(bbox)
        primitive = OrderedPrimitive(
            kind="move",
            dx=center[0] - start[0],
            dy=center[1] - start[1],
        )
    elif kind == "click":
        primitive = OrderedPrimitive(
            kind="click",
            name=str(expected.get("button", "left")),
            count=int(expected.get("count", 1)),
        )
    elif kind == "type":
        primitive = OrderedPrimitive(kind="type", text=str(expected["text"]))
    elif kind == "scroll":
        primitive = OrderedPrimitive(
            kind="scroll",
            dx=0,
            dy=-5 if expected.get("sign") == "down" else 5,
        )
    elif kind == "key":
        primitive = OrderedPrimitive(
            kind="key_combo",
            keys=tuple(str(key) for key in expected.get("keys", [])),
        )
    else:
        raise ValueError(f"unsupported synthetic expected action {expected!r}")
    return OrderedAction(primitives=(primitive,), no_op=False)


def validate_multiturn_task_setup(
    *,
    client: OSWorldClient,
    task: Task,
    output_dir: Path,
    save_frames: bool,
    settle_s: float,
    dependency_lines: list[str],
) -> dict[str, Any]:
    """Exercise every turn with a known-correct action in one fresh VM."""
    setup_state = prepare_task(client, task)
    screen = client.screen_size()
    results: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(task_turns(task)):
        turn_task = replace(
            task,
            target=turn.target,
            cursor=turn.cursor,
            expected=turn.expected,
            verifier=turn.verifier,
            turns=(),
        )
        bbox = resolve_target_bbox(client, turn_task, setup_state, screen)
        # See the matching comment in run_multiturn_attempt: only reset once,
        # at the very start of the trajectory (turn 0).
        if turn_index == 0:
            start = resolve_cursor_start(turn.cursor, bbox, screen)
            client.execute(f"pyautogui.moveTo({start[0]}, {start[1]})")
        actual_start = client.cursor_position()
        before_state = read_verifier_state(client, turn.verifier)
        before = client.screenshot_settled(min_delay_s=0.2, stability_timeout_s=1.0)
        if turn.expected.get("kind") == "any":
            # Outcome-only (turn_mode="multiturn") turns don't prescribe a
            # specific primitive, so there is no single "known-correct"
            # synthetic action to replay here -- skip rather than crash.
            # Exercising the underlying verifier/goal for these tasks still
            # needs a real model or a task-specific manual smoke test.
            results.append(
                {
                    "turn": turn_index + 1,
                    "turn_id": turn.turn_id,
                    "success": None,
                    "skipped": "no synthetic action for expected kind 'any'",
                }
            )
            break
        synthetic = _synthetic_action(turn.expected, bbox=bbox, start=actual_start)
        client.dispatch_ordered_action(synthetic)
        after = client.screenshot_settled(
            min_delay_s=settle_s,
            stability_timeout_s=1.5,
            poll_s=0.15,
        )
        end = client.cursor_position()
        verifier_ok, after_state = verifier_passed(client, turn.verifier)
        metrics = movement_metrics(actual_start, end, bbox, screen)
        location_ok = in_bbox(actual_start, bbox) if turn.expected["kind"] == "click" else True
        movement_ok = (
            bool(metrics["bbox_hit"] and metrics["legal_step_optimality"] == 1.0)
            if turn.expected["kind"] == "move"
            else True
        )
        success = bool(
            action_matches_expected(synthetic, turn.expected)
            and verifier_ok
            and location_ok
            and movement_ok
        )
        if save_frames:
            before.save(output_dir / f"step_{turn_index:03d}.png")
            after.save(output_dir / f"turn_{turn_index + 1:03d}_after.png")
            _draw_overlay(
                before,
                bbox,
                actual_start,
                end,
                str(turn.target.get("label", turn.turn_id)),
            ).save(output_dir / f"overlay_{turn_index + 1:03d}.png")
        results.append(
            {
                "turn": turn_index + 1,
                "turn_id": turn.turn_id,
                "success": success,
                "target": {**turn.target, "bbox_px": list(bbox)},
                "cursor_start": list(actual_start),
                "cursor_end": list(end),
                "synthetic_primitives": serialize_action(synthetic),
                "expected_action_ok": action_matches_expected(synthetic, turn.expected),
                "verifier_before": before_state,
                "verifier_after": after_state,
                "verifier_pass": verifier_ok,
                "movement": metrics,
            }
        )
        if not success:
            break

    turns_total = len(task_turns(task))
    verified_prefix = sum(bool(row["success"]) for row in results)
    success = len(results) == turns_total and verified_prefix == turns_total
    result = {
        "schema_version": 1,
        "mode": "validate_setups_only",
        "task_id": task.task_id,
        "category": task.category,
        "multi_turn": True,
        "success": success,
        "screen_size": list(screen),
        "turns_total": turns_total,
        "turns_attempted": len(results),
        "verified_prefix": verified_prefix,
        "progress": verified_prefix / turns_total,
        "turns": results,
        "guest_dependencies": dependency_lines,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def validate_task_setup(
    *,
    client: OSWorldClient,
    task: Task,
    output_dir: Path,
    save_frames: bool,
    settle_s: float,
) -> dict[str, Any]:
    """Apply a known-correct synthetic primitive and assert task semantics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dependencies = client.run_command(
        [
            "bash",
            "-lc",
            "printf 'xdotool=%s\\n' \"$(command -v xdotool || echo MISSING)\"; "
            "printf 'wmctrl=%s\\n' \"$(command -v wmctrl || echo MISSING)\"; "
            "printf 'gdbus=%s\\n' \"$(command -v gdbus || echo MISSING)\"; "
            "printf 'tkinter=%s\\n' "
            "\"$(python3 -c 'import tkinter; print(tkinter.TkVersion)' "
            '2>/dev/null || echo MISSING)"',
        ]
    )
    dependency_lines = str(dependencies.get("output", "")).strip().splitlines()
    required_missing = [line for line in dependency_lines if line == "tkinter=MISSING"]
    if required_missing:
        raise RuntimeError(f"missing required guest dependencies: {required_missing}")
    if task.turns:
        return validate_multiturn_task_setup(
            client=client,
            task=task,
            output_dir=output_dir,
            save_frames=save_frames,
            settle_s=settle_s,
            dependency_lines=dependency_lines,
        )
    setup_state = prepare_task(client, task)
    screen = client.screen_size()
    bbox = resolve_target_bbox(client, task, setup_state, screen)
    start = resolve_cursor_start(task.cursor, bbox, screen)
    client.execute(f"pyautogui.moveTo({start[0]}, {start[1]})")
    actual_start = client.cursor_position()
    before_state = read_verifier_state(client, task.verifier)
    before = client.screenshot_settled(min_delay_s=0.2, stability_timeout_s=1.0)

    expected_kind = task.expected["kind"]
    if expected_kind == "move":
        center = _bbox_center(bbox)
        synthetic = OrderedAction(
            primitives=(
                OrderedPrimitive(
                    kind="move",
                    dx=center[0] - actual_start[0],
                    dy=center[1] - actual_start[1],
                ),
            ),
            no_op=False,
        )
    elif expected_kind == "click":
        synthetic = OrderedAction(
            primitives=(
                OrderedPrimitive(
                    kind="click",
                    name=str(task.expected.get("button", "left")),
                    count=int(task.expected.get("count", 1)),
                ),
            ),
            no_op=False,
        )
    elif expected_kind == "type":
        synthetic = OrderedAction(
            primitives=(OrderedPrimitive(kind="type", text=str(task.expected["text"])),),
            no_op=False,
        )
    elif expected_kind == "scroll":
        direction = -5 if task.expected.get("sign") == "down" else 5
        synthetic = OrderedAction(
            primitives=(OrderedPrimitive(kind="scroll", dx=0, dy=direction),),
            no_op=False,
        )
    elif expected_kind == "key":
        synthetic = OrderedAction(
            primitives=(
                OrderedPrimitive(
                    kind="key_combo",
                    keys=tuple(str(key) for key in task.expected.get("keys", [])),
                ),
            ),
            no_op=False,
        )
    else:
        raise ValueError(f"unsupported synthetic expected action {task.expected!r}")

    client.dispatch_ordered_action(synthetic)
    after = client.screenshot_settled(
        min_delay_s=settle_s,
        stability_timeout_s=1.5,
        poll_s=0.15,
    )
    end = client.cursor_position()
    verifier_ok, after_state = verifier_passed(client, task.verifier)
    metrics = movement_metrics(actual_start, end, bbox, screen)
    location_ok = in_bbox(actual_start, bbox) if expected_kind == "click" else True
    movement_ok = (
        bool(metrics["bbox_hit"] and metrics["legal_step_optimality"] == 1.0)
        if expected_kind == "move"
        else True
    )
    success = bool(
        action_matches_expected(synthetic, task.expected)
        and verifier_ok
        and location_ok
        and movement_ok
    )

    if save_frames:
        before.save(output_dir / "step_000.png")
        after.save(output_dir / "step_001.png")
        _draw_overlay(
            before,
            bbox,
            actual_start,
            end,
            str(task.target.get("label", task.task_id)),
        ).save(output_dir / "overlay.png")
    result = {
        "schema_version": 1,
        "mode": "validate_setups_only",
        "task_id": task.task_id,
        "category": task.category,
        "success": success,
        "screen_size": list(screen),
        "target": {**task.target, "bbox_px": list(bbox)},
        "cursor_start": list(actual_start),
        "cursor_end": list(end),
        "synthetic_primitives": serialize_action(synthetic),
        "expected_action_ok": action_matches_expected(synthetic, task.expected),
        "verifier_before": before_state,
        "verifier_after": after_state,
        "verifier_pass": verifier_ok,
        "movement": metrics,
        "guest_dependencies": dependency_lines,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def aggregate_results(tasks: list[Task], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        by_task[str(attempt["task_id"])].append(attempt)

    per_task: dict[str, dict[str, Any]] = {}
    for task in tasks:
        rows = by_task.get(task.task_id, [])
        valid_rows = [row for row in rows if row["validity"] == "valid"]
        successes = [bool(row.get("success")) for row in rows]
        valid_successes = [bool(row["success"]) for row in valid_rows]
        progress = [float(row.get("progress", 0.0)) for row in rows]
        first_four = successes[:4]
        scheduled_turns = sum(int(row.get("turns_total", 1)) for row in rows)
        attempted_turns = sum(int(row.get("turns_attempted", 1)) for row in rows)
        valid_turns = sum(
            sum(bool(turn.get("parse_valid")) for turn in row.get("turns", []))
            if row.get("multi_turn")
            else bool(row.get("parse_valid"))
            for row in rows
        )
        expected_turns = sum(
            sum(bool(turn.get("expected_action_ok")) for turn in row.get("turns", []))
            if row.get("multi_turn")
            else bool(row.get("expected_action_ok"))
            for row in rows
        )
        verifier_turns = sum(
            sum(bool(turn.get("verifier_pass")) for turn in row.get("turns", []))
            if row.get("multi_turn")
            else bool(row.get("verifier_pass"))
            for row in rows
        )
        verified_turns = sum(
            int(row.get("verified_prefix", bool(row.get("success")))) for row in rows
        )
        per_task[task.task_id] = {
            "category": task.category,
            "n": len(rows),
            "n_attempts_raw": len(rows),
            "n_attempts_valid": len(valid_rows),
            "n_infrastructure_failures": len(rows) - len(valid_rows),
            "successes": sum(successes),
            "pass_at_1": sum(successes) / len(successes) if successes else 0.0,
            "pass_at_1_raw": sum(successes) / len(rows) if rows else 0.0,
            "pass_at_1_valid": (
                sum(valid_successes) / len(valid_rows) if valid_rows else None
            ),
            "pass_at_4": bool(first_four) and any(first_four),
            "all_4_success": len(first_four) == 4 and all(first_four),
            "mean_progress": sum(progress) / len(progress) if progress else 0.0,
            "best_of_4_progress": max(progress[:4], default=0.0),
            "parse_valid_rate": (
                sum(bool(row.get("parse_valid")) for row in rows) / len(rows) if rows else 0.0
            ),
            "expected_action_rate": (
                sum(bool(row.get("expected_action_ok")) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "turn_completion_rate": attempted_turns / scheduled_turns if scheduled_turns else 0.0,
            "turn_parse_valid_rate": valid_turns / scheduled_turns if scheduled_turns else 0.0,
            "turn_expected_action_rate": (
                expected_turns / scheduled_turns if scheduled_turns else 0.0
            ),
            "turn_verifier_rate": verifier_turns / scheduled_turns if scheduled_turns else 0.0,
            "verified_turn_rate": verified_turns / scheduled_turns if scheduled_turns else 0.0,
        }

    def summarize(task_ids: list[str]) -> dict[str, float | int | None]:
        rows = [per_task[task_id] for task_id in task_ids]
        if not rows:
            return {
                "n_attempts_raw": 0,
                "n_attempts_valid": 0,
                "n_infrastructure_failures": 0,
                "pass_at_1": 0.0,
                "pass_at_1_raw": 0.0,
                "pass_at_1_valid": None,
                "pass_at_4": 0.0,
                "all_4_success": 0.0,
                "mean_progress": 0.0,
                "best_of_4_progress": 0.0,
                "parse_valid_rate": 0.0,
                "expected_action_rate": 0.0,
                "turn_completion_rate": 0.0,
                "turn_parse_valid_rate": 0.0,
                "turn_expected_action_rate": 0.0,
                "turn_verifier_rate": 0.0,
                "verified_turn_rate": 0.0,
            }
        keys = (
            "pass_at_1",
            "pass_at_4",
            "all_4_success",
            "mean_progress",
            "best_of_4_progress",
            "parse_valid_rate",
            "expected_action_rate",
            "turn_completion_rate",
            "turn_parse_valid_rate",
            "turn_expected_action_rate",
            "turn_verifier_rate",
            "verified_turn_rate",
        )
        summary: dict[str, float | int | None] = {
            key: sum(float(row[key]) for row in rows) / len(rows) for key in keys
        }
        n_raw = sum(int(row["n_attempts_raw"]) for row in rows)
        n_valid = sum(int(row["n_attempts_valid"]) for row in rows)
        successes = sum(int(row["successes"]) for row in rows)
        summary.update(
            {
                "n_attempts_raw": n_raw,
                "n_attempts_valid": n_valid,
                "n_infrastructure_failures": n_raw - n_valid,
                "pass_at_1_raw": successes / n_raw if n_raw else 0.0,
                "pass_at_1_valid": successes / n_valid if n_valid else None,
            }
        )
        return summary

    categories: dict[str, dict[str, float | int | None]] = {}
    for category in sorted({task.category for task in tasks}):
        categories[category] = summarize(
            [task.task_id for task in tasks if task.category == category]
        )
    overall = summarize([task.task_id for task in tasks])
    scores = {
        f"overall/{key}": value
        for key, value in overall.items()
        if value is not None
    }
    for category, summary in categories.items():
        scores.update(
            {
                f"{category}/{key}": value
                for key, value in summary.items()
                if value is not None
            }
        )
    return {
        "scores": scores,
        # labctl's per-checkpoint dashboard (build_eval_series/first_metric)
        # picks a headline metric via an explicit primary+tasks pin before
        # falling back to name-guessing -- with ~90 flat "category/metric"
        # keys here, name-guessing would land on an arbitrary alphabetically
        # first key instead of the meaningful overall pass@1.
        "primary": "overall/pass_at_1",
        "tasks": scores,
        "overall": overall,
        "categories": categories,
        "per_task": per_task,
    }


def _launch_vm(
    *, qemu_bin: str, qcow2: str, vm_port: int, vnc_port: int, log_path: Path
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            qemu_bin,
            "-enable-kvm",
            "-cpu",
            "host",
            "-smp",
            "4",
            "-m",
            "4G",
            "-machine",
            "type=q35,accel=kvm",
            "-drive",
            f"file={qcow2},if=virtio,format=qcow2,snapshot=on",
            "-netdev",
            f"user,id=net0,hostfwd=tcp::{vm_port}-:5000,hostfwd=tcp::{vnc_port}-:5900",
            "-device",
            "virtio-net-pci,netdev=net0",
            "-display",
            "none",
            "-nographic",
        ],
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )


def _port_free(port: int) -> bool:
    """True if we can bind host ``port`` -- i.e. qemu's hostfwd rule would too.

    Mirrors qemu's bind (INADDR_ANY, no SO_REUSEADDR): if another job's qemu (or
    a leftover of our own) still holds the forwarded port, this returns False,
    which is exactly the condition under which a fresh qemu aborts with 'Could
    not set up host forwarding rule' and exits.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def _wait_ports_free(
    ports: list[int], *, timeout_s: float = 20.0, poll_s: float = 0.25
) -> list[int]:
    """Wait until every port in ``ports`` is bindable; return any still busy."""
    deadline = time.time() + timeout_s
    while True:
        busy = [p for p in ports if not _port_free(p)]
        if not busy or time.time() >= deadline:
            return busy
        time.sleep(poll_s)


def _assert_qemu_alive(proc: subprocess.Popen, log_path: Path, *, what: str) -> None:
    """Fail loudly if qemu died right after spawn (e.g. a fatal hostfwd bind
    failure), instead of letting wait_ready() burn its full 300s timeout -- or,
    worse, silently connect to *another job's* VM on the colliding port.

    qemu prints the reason and exits within milliseconds of such a failure, so a
    short grace period is enough to catch it.
    """
    time.sleep(0.7)
    if proc.poll() is None:
        return
    tail = ""
    try:
        tail = "\n".join(log_path.read_text().strip().splitlines()[-4:])
    except Exception:
        pass
    raise RuntimeError(
        f"{what}: qemu exited immediately (code {proc.returncode}). "
        f"Last lines of {log_path}:\n{tail}"
    )


def _preflight_ports(ports: list[int], *, job_mod: int) -> None:
    """Fail fast at startup if this job's port window is already occupied.

    ``job_mod`` hashes SLURM_JOB_ID into a 10-wide window, so two concurrently
    scheduled jobs on the same node whose ids are congruent mod 200 land on
    identical ports. Without this check the collision only surfaces as every
    attempt timing out in wait_ready() after 300s apiece.
    """
    busy = _wait_ports_free(ports, timeout_s=5.0)
    if not busy:
        return
    raise SystemExit(
        f"port collision: {busy} already bound on this node "
        f"(job_mod={job_mod}, from SLURM_JOB_ID % 200 * 10). Another eval job "
        "with a congruent job id, or a leftover qemu, holds this window -- "
        "resubmit for a different job id or kill the stale qemu."
    )


def _terminate(proc: subprocess.Popen, *, label: str) -> None:
    if proc.poll() is not None:
        return
    _LOGGER.info("terminating %s pid=%d", label, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _task_slug(task_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", task_id.lower()).strip("-")


def _run_slug(index: int, task_id: str) -> str:
    """Flat ``task_NN_<id>`` name, matching freeroll.py's ``_slug`` shape --
    labctl's rollout browser reads a run's displayed label straight off this
    ``task_<digits>_<name>`` pattern in ``subdir``'s basename, so a per-task/
    per-attempt subdirectory nested any deeper than one level (as this file
    used to do: ``tasks/<task_slug>/attempt_NN``) falls outside the pattern
    and labctl shows a generic ``task_0``, ``task_1``, ... fallback instead
    of the real task id.
    """
    return f"task_{index:03d}_{_task_slug(task_id)}"


@dataclass
class _RunContext:
    """Everything about a run that's the same for every (task, attempt) pair.

    Bundled so _run_one_task_attempt/_run_vm_slot don't carry a dozen
    individually-threaded parameters -- one object, passed by every VM slot
    (see main()). ``state_lock`` guards ``attempts``/``runs``: every VM slot
    appends to them and rewrites result.json after each of its own attempts,
    and slots run concurrently (--vms_per_sglang), so without the lock two
    slots finishing around the same moment could interleave list mutation
    with aggregate_results()/json.dumps() reading it -- a data race, not just
    a cosmetic log-ordering issue.
    """

    tasks: list[Task]
    suite_raw: dict[str, Any]
    output_dir: Path
    sglang_url: str
    args: argparse.Namespace
    action_format: str
    system_prompt: str
    sampling: SamplingParams
    model_resolution: tuple[int, int] | None
    total: int
    started: float
    state_lock: threading.Lock
    attempts: list[dict[str, Any]]
    runs: list[dict[str, Any]]


def _run_one_task_attempt(
    ctx: _RunContext,
    *,
    task_index: int,
    task: Task,
    attempt_index: int,
    vm_port: int,
    vnc_port: int,
) -> None:
    """Boot one VM, run one (task, attempt) against the shared sglang, tear
    the VM down, then record the result under ``ctx.state_lock``.

    Called sequentially within a VM slot (see ``_run_vm_slot``) and
    concurrently *across* slots -- ``vm_port``/``vnc_port`` are that slot's
    own, so concurrent qemu instances never collide.
    """
    args = ctx.args
    ordinal = task_index * args.attempts + attempt_index + 1
    seed = args.seed_base + task_index * 100 + attempt_index
    attempt_dir = ctx.output_dir / _run_slug(ordinal - 1, task.task_id)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(
        "[%d/%d] task=%s attempt=%d seed=%d",
        ordinal,
        ctx.total,
        task.task_id,
        attempt_index + 1,
        seed,
    )
    # The previous attempt on this slot has just torn its qemu down; give the
    # kernel a moment to release the forwards before rebinding them.
    busy = _wait_ports_free([vm_port, vnc_port])
    if busy:
        _LOGGER.warning("ports %s still bound before launch -- qemu may abort", busy)
    qemu_log = attempt_dir / "qemu.log"
    vm_proc = _launch_vm(
        qemu_bin=args.qemu_bin,
        qcow2=args.qcow2,
        vm_port=vm_port,
        vnc_port=vnc_port,
        log_path=qemu_log,
    )
    try:
        _assert_qemu_alive(vm_proc, qemu_log, what=f"task={task.task_id} seed={seed}")
        client = OSWorldClient(f"http://localhost:{vm_port}")
        client.wait_ready(timeout_s=300)
        client.patch_xcursor_leak()
        result = run_attempt(
            client=client,
            task=task,
            output_dir=attempt_dir,
            sglang_url=ctx.sglang_url,
            api_key=args.sglang_api_key,
            model=args.model_path,
            system_prompt=ctx.system_prompt,
            action_format=ctx.action_format,
            sampling=ctx.sampling,
            seed=seed,
            model_resolution=ctx.model_resolution,
            save_frames=not args.no_frames,
            settle_s=args.settle_s,
            n_history_frames=args.n_history_frames,
        )
    except Exception as error:
        _LOGGER.exception("attempt failed: task=%s seed=%d", task.task_id, seed)
        result = {
            "schema_version": _RESULT_SCHEMA_VERSION,
            "task_id": task.task_id,
            "category": task.category,
            "seed": seed,
            "validity": "infra_invalid",
            "infra_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "success": None,
            "progress": 0.0,
            "parse_valid": False,
            "expected_action_ok": False,
            "stop_reason": f"exception: {type(error).__name__}: {error}",
        }
        (attempt_dir / "result.json").write_text(json.dumps(result, indent=2))
    finally:
        _terminate(vm_proc, label=f"VM {task.task_id}/{attempt_index + 1}")

    run_entry = {
        "index": ordinal - 1,
        "slug": attempt_dir.name,
        "task_id": task.task_id,
        "instruction": task.instruction,
        "attempt": attempt_index + 1,
        "subdir": str(attempt_dir.relative_to(ctx.output_dir)),
        "validity": result["validity"],
        "infra_error": result["infra_error"],
        "success": result.get("success"),
        "stop_reason": result.get("stop_reason"),
    }
    with ctx.state_lock:
        ctx.attempts.append(result)
        ctx.runs.append(run_entry)
        aggregate = aggregate_results(ctx.tasks, ctx.attempts)
        partial = {
            "schema_version": _RESULT_SCHEMA_VERSION,
            "task": ctx.suite_raw["suite"],
            **aggregate,
            "params": {
                "model_path": args.model_path,
                "attempts": args.attempts,
                "sampling": ctx.sampling.to_dict(),
                "system_prompt_id": args.system_prompt_id,
                "action_format": ctx.action_format,
                "model_resolution": list(ctx.model_resolution) if ctx.model_resolution else None,
            },
            "n_samples": len(ctx.attempts),
            "n_tasks": len(ctx.tasks),
            "elapsed_s": time.time() - ctx.started,
            "completed": len(ctx.attempts) == ctx.total,
            # ctx.runs accumulates in whichever order concurrent VM slots
            # (--vms_per_sglang) finish their attempts, not task order --
            # sort by the run's own "index" so a viewer that (wrongly) trusts
            # array position rather than that field still shows the right
            # screenshots for the right task.
            "runs": sorted(ctx.runs, key=lambda run: run["index"]),
        }
        (ctx.output_dir / "result.json").write_text(json.dumps(partial, indent=2))


def _run_vm_slot(
    slot_id: int,
    assigned: list[tuple[int, Task, int]],
    ctx: _RunContext,
    *,
    vm_port: int,
    vnc_port: int,
) -> None:
    """Run this slot's assigned (task_index, task, attempt_index) triples
    sequentially on its own dedicated vm_port/vnc_port; slots run
    concurrently (see main()'s ThreadPoolExecutor). ``threading.
    current_thread().name`` is set so every _LOGGER call made from here on
    down picks up a ``vmN`` prefix via the ``%(threadName)s`` in the log
    format -- needed once slots interleave, since the log otherwise gives no
    way to tell which VM a given line came from.
    """
    threading.current_thread().name = f"vm{slot_id}"
    for task_index, task, attempt_index in assigned:
        _run_one_task_attempt(
            ctx,
            task_index=task_index,
            task=task,
            attempt_index=attempt_index,
            vm_port=vm_port,
            vnc_port=vnc_port,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--task_ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--system_prompt_id", default="cua_rel_step_v1_thinking")
    parser.add_argument(
        "--action_format",
        choices=(_REL_STEP_FORMAT, _QWEN3VL_NATIVE_FORMAT, _NATIVE_ORDERED_FORMAT),
        default=None,
    )
    parser.add_argument("--validate_setups_only", action="store_true")
    parser.add_argument("--model_resolution", default="1280x720")
    parser.add_argument("--seed_base", type=int, default=41000)
    parser.add_argument("--no_frames", action="store_true")
    parser.add_argument("--settle_s", type=float, default=0.5)
    parser.add_argument(
        "--n_history_frames",
        type=int,
        default=None,
        help="Cap the rolling per-turn frame history for multi_turn tasks "
        "(turn_mode='multiturn' tasks can now run dozens of turns in one "
        "attempt). Mirrors freeroll.py's StreamingLLM-style block eviction "
        "(osworld_runtime.evict_history): once the window would exceed this, "
        "only the newest n_history_frames // 2 frames are kept, preserving "
        "sglang's RadixAttention prefix cache. Unset (default) means "
        "unbounded -- unchanged from this harness's original behavior. "
        "Ignored by atomic (single-turn) tasks.",
    )
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--sglang_api_key", default="osworld")
    parser.add_argument(
        "--sglang_url",
        default=None,
        help="Skip launching a local sglang server and send requests to this "
        "already-running OpenAI-compatible endpoint instead (e.g. "
        "http://some-node:30000/v1). Debugging convenience only -- it is on "
        "you to make sure the server actually has --model_path loaded and "
        "--sglang_api_key matches its --api-key; --sglang_port/"
        "--mem_fraction_static are ignored.",
    )
    parser.add_argument(
        "--vms_per_sglang",
        type=int,
        default=4,
        help="Number of VMs to run concurrently against the single shared "
        "sglang instance, round-robin over the (task, attempt) work list "
        "(slot i gets work[i::vms_per_sglang], each attempt sequential on "
        "that slot's own VM). A single VM's step time is dominated by "
        "sglang prefill and the server sees only one in-flight request at a "
        "time (batch size 1), badly underusing the GPU. Running several VMs "
        "lets sglang batch their requests instead. Must be <= 10 to fit the "
        "per-job port allocation window (see job_mod below); pass 1 for the "
        "historical one-VM-at-a-time behaviour.",
    )
    parser.add_argument("--mem_fraction_static", type=float, default=0.70)
    parser.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    parser.add_argument("--qemu_bin", default=_DEFAULT_QEMU_BIN)
    sampling_mod.add_sampling_cli(parser, default_max_tokens=512)
    args = parser.parse_args()

    if args.attempts < 1:
        parser.error("--attempts must be >= 1")
    if args.n_history_frames is not None and args.n_history_frames < 1:
        parser.error("--n_history_frames must be >= 1")
    if not 1 <= args.vms_per_sglang <= 10:
        parser.error(
            f"--vms_per_sglang must be between 1 and 10 (got {args.vms_per_sglang}); "
            "ports are allocated 1 apart per slot within a 10-wide per-job window, "
            "see job_mod."
        )
    if not args.validate_setups_only and not args.model_path:
        parser.error("--model_path is required unless --validate_setups_only is set")
    if args.system_prompt_id not in _PROMPT_FORMATS:
        parser.error(
            f"unsupported --system_prompt_id {args.system_prompt_id!r}; "
            f"choose one of {sorted(_PROMPT_FORMATS)}"
        )
    inferred_format = _PROMPT_FORMATS[args.system_prompt_id]
    action_format = args.action_format or inferred_format
    if action_format != inferred_format:
        parser.error(
            f"--system_prompt_id {args.system_prompt_id!r} requires "
            f"--action_format {inferred_format!r}"
        )
    width_text, separator, height_text = args.model_resolution.lower().partition("x")
    if not separator:
        parser.error("--model_resolution must be WIDTHxHEIGHT")
    model_resolution = (int(width_text), int(height_text))

    suite_raw, tasks = load_suite(args.suite)
    if args.task_ids:
        selected = set(args.task_ids)
        unknown = selected - {task.task_id for task in tasks}
        if unknown:
            parser.error(f"unknown --task_ids: {sorted(unknown)}")
        tasks = [task for task in tasks if task.task_id in selected]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_marker = output_dir / "completed.json"
    completion_marker.unlink(missing_ok=True)
    # %(threadName)s disambiguates interleaved log lines once --vms_per_sglang
    # > 1 has several VM slots (named vm0, vm1, ...) logging concurrently;
    # the main thread's own lines (sglang startup, final summary) show
    # MainThread. See _run_vm_slot.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s"
    )
    logging.getLogger().addHandler(logging.FileHandler(output_dir / "cua_micro_eval.log"))

    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
    # Every port this job will ever bind, checked once up front so a window
    # collision costs 5s here instead of 300s per attempt in wait_ready().
    _n_slots = 1 if args.validate_setups_only else args.vms_per_sglang
    _preflight_ports(
        [5000 + job_mod + i for i in range(_n_slots)]
        + [5900 + job_mod + i for i in range(_n_slots)],
        job_mod=job_mod,
    )
    if args.validate_setups_only:
        validations: list[dict[str, Any]] = []
        started = time.time()
        for task_index, task in enumerate(tasks):
            attempt_dir = output_dir / "tasks" / _task_slug(task.task_id) / "validation"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            vm_port = 5000 + job_mod
            vnc_port = 5900 + job_mod
            _LOGGER.info(
                "[%d/%d] validating setup task=%s",
                task_index + 1,
                len(tasks),
                task.task_id,
            )
            _wait_ports_free([vm_port, vnc_port])
            qemu_log = attempt_dir / "qemu.log"
            vm_proc = _launch_vm(
                qemu_bin=args.qemu_bin,
                qcow2=args.qcow2,
                vm_port=vm_port,
                vnc_port=vnc_port,
                log_path=qemu_log,
            )
            try:
                _assert_qemu_alive(vm_proc, qemu_log, what=f"validate task={task.task_id}")
                client = OSWorldClient(f"http://localhost:{vm_port}")
                client.wait_ready(timeout_s=300)
                client.patch_xcursor_leak()
                result = validate_task_setup(
                    client=client,
                    task=task,
                    output_dir=attempt_dir,
                    save_frames=not args.no_frames,
                    settle_s=args.settle_s,
                )
            except Exception as error:
                _LOGGER.exception("setup validation failed: task=%s", task.task_id)
                result = {
                    "schema_version": 1,
                    "mode": "validate_setups_only",
                    "task_id": task.task_id,
                    "category": task.category,
                    "success": False,
                    "stop_reason": f"exception: {type(error).__name__}: {error}",
                }
                (attempt_dir / "result.json").write_text(json.dumps(result, indent=2))
            finally:
                _terminate(vm_proc, label=f"validation VM {task.task_id}")
            validations.append(result)
            summary = {
                "schema_version": 1,
                "mode": "validate_setups_only",
                "task": suite_raw["suite"],
                "n_tasks": len(tasks),
                "n_completed": len(validations),
                "n_passed": sum(bool(row.get("success")) for row in validations),
                "completed": len(validations) == len(tasks),
                "success": len(validations) == len(tasks)
                and all(bool(row.get("success")) for row in validations),
                "elapsed_s": time.time() - started,
                "per_task": {str(row["task_id"]): row for row in validations},
            }
            (output_dir / "result.json").write_text(json.dumps(summary, indent=2))
        completion_marker.write_text(json.dumps(summary, indent=2))
        return 0 if summary["success"] else 1

    system_prompt = SYSTEM_PROMPTS[args.system_prompt_id]
    sampling = sampling_mod.from_cli(
        args,
        model_path=args.model_path,
        system_prompt=system_prompt,
    )
    _LOGGER.info(
        "suite=%s tasks=%d attempts=%d model=%s action_format=%s sampling=%s",
        suite_raw["suite"],
        len(tasks),
        args.attempts,
        args.model_path,
        action_format,
        sampling.to_dict(),
    )

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))
    if args.sglang_url:
        sglang_url = args.sglang_url.rstrip("/")
        _LOGGER.info("using existing sglang endpoint (not launching one): %s", sglang_url)
    else:
        sglang_port = args.sglang_port if args.sglang_port != 30000 else 30000 + job_mod
        sglang_log = (output_dir / "sglang.log").open("w")
        sglang_proc = subprocess.Popen(
            [
                "uv",
                "run",
                "--project",
                str(_EVAL_DIR),
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                args.model_path,
                "--host",
                "0.0.0.0",
                "--port",
                str(sglang_port),
                "--api-key",
                args.sglang_api_key,
                "--mem-fraction-static",
                str(args.mem_fraction_static),
                "--chunked-prefill-size",
                "2048",
            ],
            cwd=str(_EVAL_DIR),
            stdout=sglang_log,
            stderr=subprocess.STDOUT,
        )
        atexit.register(_terminate, sglang_proc, label="sglang")
        _wait_for(
            f"http://localhost:{sglang_port}/health_generate",
            headers={"Authorization": f"Bearer {args.sglang_api_key}"},
            proc=sglang_proc,
            poll_s=10,
            max_polls=180,
            label="sglang",
        )
        sglang_url = f"http://localhost:{sglang_port}/v1"

    attempts: list[dict[str, Any]] = []
    # One entry per attempt, each pointing at a subdir with its own
    # trajectory.jsonl + steps/ -- this is labctl's "multi" rollout-viewer
    # shape (see labctl::server::resolve_rollout_paths), the same one
    # freeroll.py's multi-instruction mode uses. Lets `labctl show` browse
    # every attempt's screen+action trace, not just the aggregate numbers.
    runs: list[dict[str, Any]] = []
    started = time.time()
    total = len(tasks) * args.attempts

    # Round-robin (task, attempt) pairs across vms_per_sglang VM slots: slot i
    # handles work[i::n_vms], sequentially on its own dedicated vm_port/
    # vnc_port (offset by slot within the job's 10-wide port window -- see the
    # --vms_per_sglang validation above). Slots run concurrently in a thread
    # pool; every request they fire lands on the one sglang_proc started
    # above, which is the whole point (batched serving instead of one VM --
    # and one in-flight request -- at a time).
    work = [
        (task_index, task, attempt_index)
        for task_index, task in enumerate(tasks)
        for attempt_index in range(args.attempts)
    ]
    n_vms = min(args.vms_per_sglang, len(work)) if work else 1
    slots: list[list[tuple[int, Task, int]]] = [[] for _ in range(n_vms)]
    for i, item in enumerate(work):
        slots[i % n_vms].append(item)
    _LOGGER.info(
        "running %d (task, attempt) pair(s) across %d concurrent VM slot(s)",
        len(work),
        n_vms,
    )

    ctx = _RunContext(
        tasks=tasks,
        suite_raw=suite_raw,
        output_dir=output_dir,
        sglang_url=sglang_url,
        args=args,
        action_format=action_format,
        system_prompt=system_prompt,
        sampling=sampling,
        model_resolution=model_resolution,
        total=total,
        started=started,
        state_lock=threading.Lock(),
        attempts=attempts,
        runs=runs,
    )
    with ThreadPoolExecutor(max_workers=n_vms) as executor:
        futures = [
            executor.submit(
                _run_vm_slot,
                slot_id,
                assigned,
                ctx,
                vm_port=5000 + job_mod + slot_id,
                vnc_port=5900 + job_mod + slot_id,
            )
            for slot_id, assigned in enumerate(slots)
        ]
        # .result() re-raises any exception that escaped a slot (there
        # shouldn't be one -- _run_one_task_attempt already catches
        # per-attempt failures -- but a bug there should still crash main()
        # rather than being silently swallowed).
        for future in futures:
            future.result()

    aggregate = aggregate_results(tasks, attempts)
    partial = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "task": suite_raw["suite"],
        **aggregate,
        "params": {
            "model_path": args.model_path,
            "attempts": args.attempts,
            "sampling": sampling.to_dict(),
            "system_prompt_id": args.system_prompt_id,
            "action_format": action_format,
            "model_resolution": list(model_resolution) if model_resolution else None,
            "n_history_frames": args.n_history_frames,
        },
        "n_samples": len(attempts),
        "n_tasks": len(tasks),
        "elapsed_s": time.time() - started,
        "completed": len(attempts) == total,
        # See the matching comment in _run_one_task_attempt: runs accumulates
        # in VM-slot completion order, not task order.
        "runs": sorted(runs, key=lambda run: run["index"]),
    }
    (output_dir / "result.json").write_text(json.dumps(partial, indent=2))
    completion_marker.write_text(json.dumps(partial, indent=2))
    _LOGGER.info(
        "done: tasks=%d attempts=%d overall=%s output=%s",
        len(tasks),
        len(attempts),
        aggregate_results(tasks, attempts)["overall"],
        output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
