"""One-turn, state-verifiable CUA micro-evaluation suite.

Each attempt starts from a fresh OSWorld VM snapshot, receives exactly one
goal-conditioned screenshot, emits one strict action, and is scored
automatically. The primary contract is ``computer_use_rel_step_v1``; an
explicit native Qwen3-VL computer-use mode supports off-the-shelf baselines.
Move tasks expose continuous distance progress and legal-step optimality;
click/type/scroll tasks require both the correct parsed primitive and a
semantic post-action state change. Four sampled attempts per task produce
empirical pass@1 and pass@4 curves.
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
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

import sampling as sampling_mod
from action_parser import (
    ComputerUseCall,
    OrderedAction,
    OrderedPrimitive,
    parse_computer_use_rel_step_action,
    parse_qwen3vl_computer_use_action,
)
from osworld_runtime import (
    _DEFAULT_QCOW2,
    _DEFAULT_QEMU_BIN,
    _EVAL_DIR,
    _call_model,
    _wait_for,
    build_loggable_messages,
)
from osworld_system_prompts import SYSTEM_PROMPTS
from osworld_vm_client import OSWorldClient
from sampling import SamplingParams

_LOGGER = logging.getLogger(__name__)
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
_DEFAULT_SUITE = Path(__file__).with_name("cua_micro_tasks_v1.json")
_REL_STEP_FORMAT = "computer_use_rel_step_v1"
_QWEN3VL_NATIVE_FORMAT = "qwen3vl_native_cua_v1"
_PROMPT_FORMATS = {
    "cua_rel_step_v1_thinking": _REL_STEP_FORMAT,
    "qwen3vl_native_cua_v1": _QWEN3VL_NATIVE_FORMAT,
}


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


def load_suite(path: Path) -> tuple[dict[str, Any], list[Task]]:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported suite schema: {raw.get('schema_version')!r}")
    if raw.get("coordinate_grid") != _GRID:
        raise ValueError(f"coordinate_grid must be {_GRID}")
    tasks: list[Task] = []
    seen: set[str] = set()
    for index, item in enumerate(raw.get("tasks", [])):
        required = {
            "id",
            "category",
            "instruction",
            "setup",
            "target",
            "cursor",
            "expected",
            "verifier",
        }
        missing = required - set(item)
        extra = set(item) - required
        if missing or extra:
            raise ValueError(f"task {index}: missing={sorted(missing)} extra={sorted(extra)}")
        task_id = item["id"]
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError(f"task {index}: invalid/duplicate id {task_id!r}")
        seen.add(task_id)
        tasks.append(
            Task(
                task_id=task_id,
                category=str(item["category"]),
                instruction=str(item["instruction"]),
                setup=dict(item["setup"]),
                target=dict(item["target"]),
                cursor=dict(item["cursor"]),
                expected=dict(item["expected"]),
                verifier=dict(item["verifier"]),
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


def action_matches_expected(action: OrderedAction | None, expected: dict[str, Any]) -> bool:
    if action is None or len(action.primitives) != 1:
        return False
    primitive = action.primitives[0]
    kind = expected.get("kind")
    matches = False
    if kind == "move":
        matches = primitive.kind == "move"
    elif kind == "click":
        matches = (
            primitive.kind == "click"
            and primitive.name == expected.get("button", "left")
            and primitive.count == 1
        )
    elif kind == "type":
        matches = primitive.kind == "type" and primitive.text == expected.get("text")
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
    try:
        return str(
            client.run_command(
                [
                    "bash",
                    "-lc",
                    "if command -v xdotool >/dev/null; then "
                    "xdotool getactivewindow getwindowname; "
                    "elif command -v wmctrl >/dev/null; then "
                    "wmctrl -l; "
                    "elif command -v gdbus >/dev/null; then "
                    "gdbus call --session --dest org.gnome.Shell.Introspect "
                    "--object-path /org/gnome/Shell/Introspect "
                    "--method org.gnome.Shell.Introspect.GetWindows; "
                    "else exit 127; fi",
                ]
            ).get("output", "")
        ).strip()
    except RuntimeError:
        return ""


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
    else:
        urls = ["file:///tmp/cua_micro.html"]
        expected_title = {
            "history": "PAGE_B",
            "reload": "LOAD_1",
            "button": "BUTTON_READY",
            "scroll": "SCROLL_READY",
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
        "--start-maximized --remote-debugging-port=9222 "
        f"{quoted_urls} >/tmp/cua_micro_chrome.log 2>&1 &"
    )
    client.run_command(command, shell=True)
    if activate_title is not None:
        _wait_until(lambda: _activate_chrome_target(client, activate_title), timeout_s=20)
    _wait_until(lambda: expected_title in _active_title(client), timeout_s=20)


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


def prepare_task(client: OSWorldClient, task: Task) -> dict[str, Any]:
    kind = task.setup.get("kind")
    if kind == "desktop":
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


def read_verifier_state(client: OSWorldClient, verifier: dict[str, Any]) -> Any:
    kind = verifier.get("kind")
    if kind == "bbox_hit":
        return None
    if kind == "active_title_regex":
        return _active_title(client)
    if kind == "fixture_equals":
        return _guest_json(client, _FIXTURE_STATE_PATH).get("values", {}).get(verifier.get("field"))
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
        return None

    try:
        matched = _wait_until(check, timeout_s=8)
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_state = prepare_task(client, task)
    screen = client.screen_size()
    bbox = resolve_target_bbox(client, task, setup_state, screen)
    start = resolve_cursor_start(task.cursor, bbox, screen)
    client.execute(f"pyautogui.moveTo({start[0]}, {start[1]})")
    actual_start = client.cursor_position()
    before_state = read_verifier_state(client, task.verifier)
    before = client.screenshot_settled(min_delay_s=0.2, stability_timeout_s=1.0)
    if save_frames:
        before.save(output_dir / "step_000.png")

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
        fresh_visual_context=True,
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
            else:
                raise ValueError(f"unknown action format {action_format!r}")
        except (TypeError, ValueError) as error:
            parse_error = str(error)

    if parsed is not None:
        try:
            if any(primitive.kind == "terminate" for primitive in parsed.primitives):
                raise ValueError("terminate is not valid for an atomic micro-task")
            dispatch_action = (
                denormalize_action(parsed, screen) if action_format == _REL_STEP_FORMAT else parsed
            )
            client.dispatch_ordered_action(dispatch_action)
            dispatched = True
        except (TypeError, ValueError, RuntimeError) as error:
            dispatch_error = str(error)
            client.release_all_inputs()

    after = client.screenshot_settled(
        min_delay_s=settle_s,
        stability_timeout_s=1.5,
        poll_s=0.15,
    )
    end = client.cursor_position()
    verifier_ok, after_state = verifier_passed(client, task.verifier)
    expected_ok = action_matches_expected(parsed, task.expected)
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
        after.save(output_dir / "step_001.png")
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
        "schema_version": 1,
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
            primitives=(OrderedPrimitive(kind="click", name="left", count=1),),
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
        successes = [bool(row.get("success")) for row in rows]
        progress = [float(row.get("progress", 0.0)) for row in rows]
        first_four = successes[:4]
        per_task[task.task_id] = {
            "category": task.category,
            "n": len(rows),
            "successes": sum(successes),
            "pass_at_1": sum(successes) / len(successes) if successes else 0.0,
            "pass_at_4": bool(first_four) and any(first_four),
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
        }

    def summarize(task_ids: list[str]) -> dict[str, float]:
        rows = [per_task[task_id] for task_id in task_ids]
        if not rows:
            return {
                "pass_at_1": 0.0,
                "pass_at_4": 0.0,
                "mean_progress": 0.0,
                "best_of_4_progress": 0.0,
                "parse_valid_rate": 0.0,
                "expected_action_rate": 0.0,
            }
        keys = (
            "pass_at_1",
            "pass_at_4",
            "mean_progress",
            "best_of_4_progress",
            "parse_valid_rate",
            "expected_action_rate",
        )
        return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}

    categories: dict[str, dict[str, float]] = {}
    for category in sorted({task.category for task in tasks}):
        categories[category] = summarize(
            [task.task_id for task in tasks if task.category == category]
        )
    overall = summarize([task.task_id for task in tasks])
    scores = {f"overall/{key}": value for key, value in overall.items()}
    for category, summary in categories.items():
        scores.update({f"{category}/{key}": value for key, value in summary.items()})
    return {
        "scores": scores,
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
        choices=(_REL_STEP_FORMAT, _QWEN3VL_NATIVE_FORMAT),
        default=None,
    )
    parser.add_argument("--validate_setups_only", action="store_true")
    parser.add_argument("--model_resolution", default="1280x720")
    parser.add_argument("--seed_base", type=int, default=41000)
    parser.add_argument("--no_frames", action="store_true")
    parser.add_argument("--settle_s", type=float, default=0.5)
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--sglang_api_key", default="osworld")
    parser.add_argument("--mem_fraction_static", type=float, default=0.70)
    parser.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    parser.add_argument("--qemu_bin", default=_DEFAULT_QEMU_BIN)
    sampling_mod.add_sampling_cli(parser, default_max_tokens=512)
    args = parser.parse_args()

    if args.attempts < 1:
        parser.error("--attempts must be >= 1")
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().addHandler(logging.FileHandler(output_dir / "cua_micro_eval.log"))

    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
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
            vm_proc = _launch_vm(
                qemu_bin=args.qemu_bin,
                qcow2=args.qcow2,
                vm_port=vm_port,
                vnc_port=vnc_port,
                log_path=attempt_dir / "qemu.log",
            )
            try:
                client = OSWorldClient(f"http://localhost:{vm_port}")
                client.wait_ready(timeout_s=300)
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
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))
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
    started = time.time()
    total = len(tasks) * args.attempts
    for task_index, task in enumerate(tasks):
        for attempt_index in range(args.attempts):
            ordinal = task_index * args.attempts + attempt_index + 1
            seed = args.seed_base + task_index * 100 + attempt_index
            attempt_dir = (
                output_dir / "tasks" / _task_slug(task.task_id) / f"attempt_{attempt_index + 1:02d}"
            )
            attempt_dir.mkdir(parents=True, exist_ok=True)
            _LOGGER.info(
                "[%d/%d] task=%s attempt=%d seed=%d",
                ordinal,
                total,
                task.task_id,
                attempt_index + 1,
                seed,
            )
            vm_port = 5000 + job_mod
            vnc_port = 5900 + job_mod
            vm_proc = _launch_vm(
                qemu_bin=args.qemu_bin,
                qcow2=args.qcow2,
                vm_port=vm_port,
                vnc_port=vnc_port,
                log_path=attempt_dir / "qemu.log",
            )
            try:
                client = OSWorldClient(f"http://localhost:{vm_port}")
                client.wait_ready(timeout_s=300)
                result = run_attempt(
                    client=client,
                    task=task,
                    output_dir=attempt_dir,
                    sglang_url=sglang_url,
                    api_key=args.sglang_api_key,
                    model=args.model_path,
                    system_prompt=system_prompt,
                    action_format=action_format,
                    sampling=sampling,
                    seed=seed,
                    model_resolution=model_resolution,
                    save_frames=not args.no_frames,
                    settle_s=args.settle_s,
                )
            except Exception as error:
                _LOGGER.exception("attempt failed: task=%s seed=%d", task.task_id, seed)
                result = {
                    "schema_version": 1,
                    "task_id": task.task_id,
                    "category": task.category,
                    "seed": seed,
                    "success": False,
                    "progress": 0.0,
                    "parse_valid": False,
                    "expected_action_ok": False,
                    "stop_reason": f"exception: {type(error).__name__}: {error}",
                }
                (attempt_dir / "result.json").write_text(json.dumps(result, indent=2))
            finally:
                _terminate(vm_proc, label=f"VM {task.task_id}/{attempt_index + 1}")
            attempts.append(result)

            aggregate = aggregate_results(tasks, attempts)
            partial = {
                "schema_version": 1,
                "task": suite_raw["suite"],
                **aggregate,
                "params": {
                    "model_path": args.model_path,
                    "attempts": args.attempts,
                    "sampling": sampling.to_dict(),
                    "system_prompt_id": args.system_prompt_id,
                    "action_format": action_format,
                    "model_resolution": list(model_resolution),
                },
                "n_samples": len(attempts),
                "n_tasks": len(tasks),
                "elapsed_s": time.time() - started,
                "completed": len(attempts) == total,
            }
            (output_dir / "result.json").write_text(json.dumps(partial, indent=2))

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
