"""The state-verifiable 18-task CUA action-parity micro-evaluation."""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import requests
from cua_parity_contract import (
    MAX_COMPLETED_TURNS,
    OBSERVATION_CONTRACT,
    OBSERVATION_SIZE,
    PREVIOUS_ACTIONS_MAX_CHARS,
    render_history,
)
from desktop.geometry import DisplayGeometry
from desktop.ir import Operation
from desktop.vm import (
    DesktopPoolConfig,
    DesktopSession,
    acquire_port_range,
    build_desktop_pool,
)
from grammars import split_control
from grammars.ordered_events_v3_relative_1000_grid_v1.codec import (
    CODEC,
    OrderedEventsV3Action,
    OrderedEventsV3Error,
    Primitive,
)
from PIL import Image, ImageDraw

_LOGGER = logging.getLogger(__name__)
_SAMPLING = {
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
}


def _call_model(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    instruction: str,
    history: list[dict[str, Any]],
    seed: int,
) -> tuple[str, str]:
    messages = render_history(
        instruction=instruction,
        steps=history,
        target_index=len(history) - 1,
    )
    request_json = {
        "model": model,
        "messages": messages,
        **_SAMPLING,
        "seed": seed,
    }
    r = requests.post(
        sglang_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_json,
        timeout=120,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict):
        raise RuntimeError("chat completion response must be an object")
    if payload.get("model") != model:
        raise RuntimeError(f"chat completion came from {payload.get('model')!r}, not {model!r}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("chat completion must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise RuntimeError("chat completion choice is malformed")
    content = choice["message"].get("content")
    finish_reason = choice.get("finish_reason")
    if not isinstance(content, str) or finish_reason not in {"stop", "length"}:
        raise RuntimeError("chat completion has an invalid content or finish_reason")
    return content, finish_reason


def _image_part(observation: bytes) -> dict[str, Any]:
    if not observation.startswith(b"\xff\xd8") or not observation.endswith(b"\xff\xd9"):
        raise ValueError("desktop observation is not JPEG")
    encoded = base64.b64encode(observation).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


_GRID = 1000
_FIXTURE_GUEST_PATH = "/tmp/cua_micro_fixture.py"
_FIXTURE_STATE_PATH = "/tmp/cua_micro_fixture_state.json"
_NATIVE_TERMINAL_STATE_PATH = "/tmp/cua_native_terminal_state.json"
_NATIVE_TERMINAL_RC_PATH = "/tmp/cua_terminal_rc"
_NATIVE_EDITOR_PATH = "/tmp/cua_native_editor.txt"
_DEFAULT_SUITE = Path(__file__).with_name("cua_micro_tasks.json")
_ATTEMPTS = 4
_SEED_BASE = 41000
_VM_SLOTS = 4
_XCURSOR_CHECKPOINT = "cua_micro_xcursor_v1"


class EvalSession(Protocol):
    def run_guest(self, argv: list[str], *, timeout_s: float | None = None) -> Any: ...

    def write_guest_file(self, path: str, content: bytes) -> None: ...

    def execute(self, operations: tuple[Operation, ...]) -> Any: ...

    def cursor_position(self) -> tuple[int, int]: ...

    def screen_size(self) -> tuple[int, int]: ...

    def screenshot_settled(
        self,
        *,
        min_delay_s: float = 0.0,
        stability_timeout_s: float = 0.0,
        poll_s: float = 0.1,
    ) -> bytes: ...

    def reset_to_checkpoint(self, name: str, *, setup: Any = None) -> Any: ...


def _run_guest(client: EvalSession, argv: list[str], *, timeout_s: float | None = None) -> str:
    result = client.run_guest(argv, timeout_s=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(f"guest command failed rc={result.returncode}: {result.stderr}")
    return str(result.stdout)


def _run_guest_shell(client: EvalSession, command: str) -> str:
    return _run_guest(client, ["bash", "-lc", command])


def _execute(client: EvalSession, operations: Sequence[Operation]) -> Any:
    if not operations:
        return None
    receipt = client.execute(tuple(operations))
    if not receipt.ok:
        raise RuntimeError(f"desktop action failed: {receipt.error}")
    return receipt


@dataclass(frozen=True)
class Task:
    task_id: str
    category: str
    instruction: str
    setup: dict[str, Any]
    target: dict[str, Any]
    expected: dict[str, Any]
    verifier: dict[str, Any]
    max_turns: int


def load_suite(path: Path) -> tuple[dict[str, Any], list[Task]]:
    raw = json.loads(path.read_text())
    if set(raw) != {"schema_version", "suite", "coordinate_grid", "description", "tasks"}:
        raise ValueError("suite fields do not match the CUA micro-eval contract")
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported suite schema: {raw.get('schema_version')!r}")
    if raw.get("suite") != "cua_micro_tasks":
        raise ValueError(f"unsupported suite: {raw.get('suite')!r}")
    if raw.get("coordinate_grid") != _GRID:
        raise ValueError(f"coordinate_grid must be {_GRID}")
    tasks: list[Task] = []
    seen: set[str] = set()
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 18:
        raise ValueError("the CUA micro-eval requires exactly 18 tasks")
    for index, item in enumerate(raw_tasks):
        common = {"id", "category", "instruction", "setup"}
        multiturn = item.get("category") == "multi_turn"
        required = common | (
            {"turn_mode", "turn", "max_turns"}
            if multiturn
            else {"target", "cursor", "expected", "verifier"}
        )
        if set(item) != required:
            raise ValueError(f"task {index}: fields must be {sorted(required)}, got {sorted(item)}")
        task_id = item["id"]
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError(f"task {index}: invalid/duplicate id {task_id!r}")
        seen.add(task_id)
        if item["category"] not in {
            "chrome_control",
            "multi_turn",
            "native_app",
            "native_launch",
        }:
            raise ValueError(f"task {index}: invalid category")
        if not isinstance(item["instruction"], str) or not item["instruction"].strip():
            raise ValueError(f"task {index}: instruction must be non-empty")
        if multiturn:
            if item["turn_mode"] != "multiturn":
                raise ValueError(f"task {index}: turn_mode must be 'multiturn'")
            template = item["turn"]
            turn_fields = {"target", "cursor", "expected", "verifier"}
            if not isinstance(template, dict) or set(template) != turn_fields:
                raise ValueError(f"task {index}: turn fields must be {sorted(turn_fields)}")
            if template["expected"] != {"kind": "any"}:
                raise ValueError(f"task {index}: multiturn task must be outcome-only")
            max_turns = item["max_turns"]
            if type(max_turns) is not int or max_turns < 2:
                raise ValueError(f"task {index}: max_turns must be an integer >= 2")
        else:
            template = item
            max_turns = 1
        setup = item["setup"]
        setup_kind = setup.get("kind")
        valid_setup = (
            (setup == {"kind": "desktop"})
            or (setup == {"kind": "desktop", "chrome_startup": "wikipedia"})
            or (
                setup_kind == "chrome"
                and set(setup) == {"kind", "variant"}
                and setup["variant"] in {"history", "reload", "button", "scroll", "search"}
            )
            or (setup == {"kind": "fixture", "mode": "calculator"})
            or (
                setup_kind == "native_app"
                and set(setup) == {"kind", "app"}
                and setup["app"]
                in {"files", "writer", "impress", "text_editor", "calculator", "terminal"}
            )
            or (setup == {"kind": "native_terminal_capture"})
        )
        if not valid_setup:
            raise ValueError(f"task {index}: unsupported setup {setup!r}")
        if template["target"].get("kind") != "fixed_norm" or set(template["target"]) != {
            "kind",
            "bbox",
            "label",
        }:
            raise ValueError(f"task {index}: target must be a labeled fixed_norm bbox")
        if template["cursor"] != {"kind": "target_center"}:
            raise ValueError(f"task {index}: cursor must start at target center")
        if template["expected"].get("kind") not in {"any", "click", "key", "scroll", "type"}:
            raise ValueError(f"task {index}: invalid expected action")
        if template["verifier"].get("kind") not in {
            "active_title_regex",
            "calculator_clipboard_equals",
            "fixture_equals",
            "guest_command_regex",
            "guest_json_equals",
            "saved_file_equals",
        }:
            raise ValueError(f"task {index}: invalid verifier")
        tasks.append(
            Task(
                task_id=task_id,
                category=str(item["category"]),
                instruction=str(item["instruction"]),
                setup=dict(item["setup"]),
                target=dict(template["target"]),
                expected=dict(template["expected"]),
                verifier=dict(template["verifier"]),
                max_turns=max_turns,
            )
        )
    return raw, tasks


def in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x < x2 and y1 <= y < y2


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
    token = str(name).strip().upper()
    if not token:
        return ""
    for prefix in ("KEY", "DIGIT", "NUM"):
        rest = token[len(prefix) :]
        if token.startswith(prefix) and len(rest) == 1 and rest.isalnum():
            return rest
    for suffix in ("LEFT", "RIGHT"):
        if token.endswith(suffix) and token[: -len(suffix)] in _KEY_ALIASES:
            token = token[: -len(suffix)]
            break
    return _KEY_ALIASES.get(token, token)


def action_matches_expected(
    action: OrderedEventsV3Action | None,
    expected: dict[str, Any],
) -> bool:
    if expected.get("kind") == "any":
        return action is not None
    if action is None or action.no_op:
        return False
    primitives = list(action.primitives)
    while primitives and primitives[0].kind == "move":
        primitives.pop(0)
    kind = expected.get("kind")
    if kind == "move":
        return len(action.primitives) == 1 and action.primitives[0].kind == "move"
    if kind == "type":
        return len(primitives) == 1 and primitives[0] == Primitive(
            "type", text=str(expected.get("text"))
        )
    if kind == "scroll" and len(primitives) == 1:
        primitive = primitives[0]
        sign = expected.get("sign")
        return (
            primitive.kind == "scroll"
            and primitive.dx == 0
            and ((sign == "down" and primitive.dy < 0) or (sign == "up" and primitive.dy > 0))
        )
    if kind == "click":
        button = {"left": "LMB", "middle": "MMB", "right": "RMB"}[expected.get("button", "left")]
        count = int(expected.get("count", 1))
        return primitives == [
            primitive
            for _ in range(count)
            for primitive in (Primitive("down", name=button), Primitive("up", name=button))
        ]
    if kind == "key":
        keys = tuple(str(key) for key in expected.get("keys", ()))
        if not keys:
            return False
        if len(keys) == 1 and len(keys[0]) == 1:
            if primitives == [Primitive("type", text=keys[0])]:
                return True
        half = len(primitives) // 2
        downs, ups = primitives[:half], primitives[half:]
        return (
            len(primitives) == 2 * len(keys)
            and all(primitive.kind == "down" for primitive in downs)
            and all(primitive.kind == "up" for primitive in ups)
            and tuple(_canonical_key_name(primitive.name) for primitive in downs)
            == tuple(_canonical_key_name(key) for key in keys)
            and tuple(_canonical_key_name(primitive.name) for primitive in ups)
            == tuple(_canonical_key_name(key) for key in reversed(keys))
        )
    raise ValueError(f"unknown expected action kind {kind!r}")


def _guest_json(client: EvalSession, path: str) -> dict[str, Any]:
    code = f"from pathlib import Path; print(Path({path!r}).read_text())"
    return json.loads(_run_guest(client, ["python3", "-c", code]))


def _upload_bytes(client: EvalSession, path: str, payload: bytes) -> None:
    client.write_guest_file(path, payload)


def _active_title(client: EvalSession) -> str:
    return _run_guest(client, ["wmctrl", "-l"]).strip()


class ConditionTimeout(RuntimeError):
    pass


def _wait_until(predicate: Any, *, timeout_s: float = 12.0, poll_s: float = 0.25) -> Any:
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(poll_s)
    raise ConditionTimeout(f"condition not met after {timeout_s}s (last={last!r})")


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
        body = """
          <div id='cua-home'>
            <div class='hero'>Web Search</div>
            <form id='cua-search-form' onsubmit="cuaSubmitSearch(); return false;">
              <input id='cua-search-input' autofocus autocomplete='off'
                placeholder='Search the web'>
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
            #cua-search-input{flex:1;font-size:22px;padding:0 16px;
              border:1px solid #c7ccd6;border-radius:8px}
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
        body = """
          <div id='cua-home'>
            <div class='hero'>Web Search</div>
            <form id='cua-search-form' onsubmit="cuaSubmitSearch(); return false;">
              <input id='cua-search-input' autofocus autocomplete='off'
                placeholder='Search the web'>
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
            #cua-search-input{flex:1;font-size:22px;padding:0 16px;
              border:1px solid #c7ccd6;border-radius:8px}
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
        raise ValueError(f"unknown Chrome fixture {variant!r}")
    return {"/tmp/cua_micro.html": f"<!doctype html><style>{style}</style>{body}"}


_CHROME_STARTUP_INSTALLER = r"""
import os, subprocess, sys

URL = sys.argv[1]
FLAGS = ("--no-first-run --no-default-browser-check "
         "--disable-session-crashed-bubble --disable-infobars --disable-translate")
source = "/usr/share/applications/google-chrome.desktop"
if not os.path.isfile(source):
    raise RuntimeError("google-chrome.desktop is missing")

target_dir = os.path.expanduser("~/.local/share/applications")
os.makedirs(target_dir, exist_ok=True)
target = os.path.join(target_dir, os.path.basename(source))

out = []
for line in open(source, encoding="utf-8", errors="replace").read().splitlines():
    if line.startswith("Exec="):
        exec_line = line[len("Exec="):]
        for code in ("%U", "%F", "%u", "%f"):
            exec_line = exec_line.replace(code, "")
        line = "Exec=" + " ".join(exec_line.split()) + " " + FLAGS + " " + URL
    out.append(line)
open(target, "w", encoding="utf-8").write("\n".join(out) + "\n")
os.chmod(target, 0o755)

subprocess.run(["pkill", "-f", "google-chrome|chromium"], check=False)
subprocess.run(["update-desktop-database", target_dir], check=True)
print("installed " + target + " from " + source)
"""


def _install_chrome_startup_page(client: EvalSession, variant: str) -> None:
    for path, text in _chrome_html(variant).items():
        _upload_bytes(client, path, text.encode())
    _upload_bytes(client, "/tmp/cua_install_chrome_startup.py", _CHROME_STARTUP_INSTALLER.encode())
    _run_guest(
        client,
        ["python3", "/tmp/cua_install_chrome_startup.py", "file:///tmp/cua_micro.html"],
    )


def _launch_chrome(client: EvalSession, variant: str) -> None:
    files = _chrome_html(variant)
    if variant == "history":
        files = {
            "/tmp/cua_history_a.html": (
                "<title>PAGE_A</title>"
                "<a href='file:///tmp/cua_history_b.html' "
                "style='position:fixed;inset:0;display:grid;place-items:center;font-size:48px'>"
                "OPEN PAGE B</a>"
            ),
            "/tmp/cua_history_b.html": "<title>PAGE_B</title><h1>PAGE B — use Chrome Back</h1>",
        }
        url = "file:///tmp/cua_history_a.html"
        expected_title = "PAGE_A"
    else:
        url = "file:///tmp/cua_micro.html"
        expected_title = {
            "reload": "LOAD_1",
            "button": "BUTTON_READY",
            "scroll": "SCROLL_READY",
            "search": "SEARCH_READY",
        }[variant]
    for path, text in files.items():
        _upload_bytes(client, path, text.encode())
    command = (
        "nohup env DISPLAY=:0 google-chrome --user-data-dir=/tmp/cua-micro-chrome "
        "--no-first-run --no-default-browser-check --disable-session-crashed-bubble "
        "--disable-infobars --disable-translate --disable-background-networking "
        "--disable-component-update --disable-default-apps --disable-sync "
        "--start-maximized --remote-debugging-port=9222 "
        f"'{url}' >/tmp/cua_micro_chrome.log 2>&1 &"
    )
    _run_guest_shell(client, command)
    _wait_until(lambda: expected_title in _active_title(client), timeout_s=120)
    if variant == "history":
        width, height = client.screen_size()
        _execute(
            client,
            (
                Operation("move_to", (width // 2, height // 2)),
                Operation("click", ("left",)),
            ),
        )
        _wait_until(lambda: "PAGE_B" in _active_title(client), timeout_s=20)
        _execute(
            client,
            (
                Operation("key_down", ("Alt",)),
                Operation("key_down", ("ArrowLeft",)),
                Operation("key_up", ("ArrowLeft",)),
                Operation("key_up", ("Alt",)),
            ),
        )
        _wait_until(lambda: "PAGE_A" in _active_title(client), timeout_s=20)
        _execute(
            client,
            (
                Operation("key_down", ("Alt",)),
                Operation("key_down", ("ArrowRight",)),
                Operation("key_up", ("ArrowRight",)),
                Operation("key_up", ("Alt",)),
            ),
        )
        _wait_until(lambda: "PAGE_B" in _active_title(client), timeout_s=20)


def _launch_fixture(client: EvalSession) -> None:
    fixture_path = Path(__file__).with_name("cua_micro_fixture.py")
    _upload_bytes(client, _FIXTURE_GUEST_PATH, fixture_path.read_bytes())
    command = (
        f"rm -f {_FIXTURE_STATE_PATH}; "
        f"nohup env DISPLAY=:0 python3 {_FIXTURE_GUEST_PATH} "
        ">/tmp/cua_micro_fixture.log 2>&1 &"
    )
    _run_guest_shell(client, command)

    _run_guest_shell(
        client,
        f"for _ in $(seq 1 60); do test -s {_FIXTURE_STATE_PATH} && exit 0; "
        "sleep 0.25; done; exit 1",
    )
    state = _guest_json(client, _FIXTURE_STATE_PATH)
    if not state.get("ready") or state.get("mode") != "calculator":
        raise RuntimeError(f"fixture did not become ready: {state!r}")
    time.sleep(0.3)


def _launch_native_app(client: EvalSession, app: str) -> None:
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
        # The rcfile keeps Bash from replacing the title after the first prompt.
        "terminal": (
            "rm -f ~/hello_world.py; "
            f"printf '%s\\n' '[ -f ~/.bashrc ] && . ~/.bashrc' "
            f"'PS1=\"$PS1\\[\\e]0;CUA Terminal\\a\\]\"' > {_NATIVE_TERMINAL_RC_PATH}; "
            "nohup env DISPLAY=:0 gnome-terminal --title='CUA Terminal' "
            f"-- bash --rcfile {_NATIVE_TERMINAL_RC_PATH} -i "
            ">/tmp/cua_native_terminal_open.log 2>&1 &"
        ),
    }
    patterns = {
        "files": r"cua_native_files|Files",
        "writer": r"LibreOffice Writer",
        "impress": r"LibreOffice Impress",
        "text_editor": r"cua_native_editor|Text Editor",
        "calculator": r"Calculator",
        "terminal": r"CUA Terminal",
    }
    if app not in commands:
        raise ValueError(f"unknown native app {app!r}")
    _run_guest_shell(client, commands[app])
    pattern = re.compile(patterns[app], re.IGNORECASE)
    _wait_until(lambda: pattern.search(_active_title(client)), timeout_s=30)
    time.sleep(0.8)
    if app == "text_editor":
        width, height = client.screen_size()
        _execute(
            client,
            (
                Operation("move_to", (width // 2, height // 2)),
                Operation("click", ("left",)),
            ),
        )


def _launch_native_terminal_capture(client: EvalSession, text: str) -> None:
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
    _run_guest_shell(client, command)
    _wait_until(lambda: "CUA Native Terminal" in _active_title(client), timeout_s=20)
    _run_guest_shell(
        client,
        f"for _ in $(seq 1 40); do test -s {_NATIVE_TERMINAL_STATE_PATH} && exit 0; "
        "sleep 0.25; done; exit 1",
    )
    state = _guest_json(client, _NATIVE_TERMINAL_STATE_PATH)
    if not state.get("ready"):
        raise RuntimeError(f"terminal capture did not become ready: {state!r}")
    time.sleep(0.5)


def prepare_task(client: EvalSession, task: Task) -> None:
    kind = task.setup.get("kind")
    if kind == "desktop":
        chrome_startup = task.setup.get("chrome_startup")
        if chrome_startup is not None:
            _install_chrome_startup_page(client, str(chrome_startup))
        _run_guest(client, ["wmctrl", "-k", "on"])
        time.sleep(0.8)
        return
    if kind == "chrome":
        _launch_chrome(client, str(task.setup["variant"]))
        return
    if kind == "fixture":
        _launch_fixture(client)
        return
    if kind == "native_app":
        _launch_native_app(client, str(task.setup["app"]))
        return
    if kind == "native_terminal_capture":
        _launch_native_terminal_capture(client, str(task.expected["text"]))
        return
    raise ValueError(f"unknown setup kind {kind!r}")


def read_verifier_state(  # noqa: PLR0911
    client: EvalSession, verifier: dict[str, Any]
) -> Any:
    kind = verifier.get("kind")
    if kind == "active_title_regex":
        return _active_title(client)
    if kind == "fixture_equals":
        return _guest_json(client, _FIXTURE_STATE_PATH).get("values", {}).get(verifier.get("field"))
    if kind == "guest_json_equals":
        return _guest_json(client, str(verifier["path"])).get(verifier.get("field", "value"))
    if kind == "saved_file_equals":
        _execute(
            client,
            (
                Operation("key_down", ("ControlLeft",)),
                Operation("key_down", ("KeyS",)),
                Operation("key_up", ("KeyS",)),
                Operation("key_up", ("ControlLeft",)),
            ),
        )
        time.sleep(0.4)
        return _run_guest(
            client,
            [
                "python3",
                "-c",
                f"from pathlib import Path; print(Path({str(verifier['path'])!r}).read_text())",
            ],
        ).rstrip("\n")
    if kind == "calculator_clipboard_equals":
        _execute(
            client,
            (
                Operation("key_down", ("ControlLeft",)),
                Operation("key_down", ("KeyC",)),
                Operation("key_up", ("KeyC",)),
                Operation("key_up", ("ControlLeft",)),
            ),
        )
        code = (
            "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.update(); "
            "print(r.clipboard_get()); r.destroy()"
        )
        return _run_guest(client, ["python3", "-c", code]).strip()
    if kind == "guest_command_regex":
        return _run_guest_shell(client, str(verifier["command"]))
    raise ValueError(f"unknown verifier kind {kind!r}")


def verifier_passed(client: EvalSession, verifier: dict[str, Any]) -> tuple[bool, Any]:
    kind = verifier.get("kind")

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
        }:
            return {"matched": True, "value": state} if state == verifier.get("value") else None
        if kind == "guest_command_regex":
            return state if re.search(str(verifier["pattern"]), str(state), re.IGNORECASE) else None
        return None

    try:
        matched = _wait_until(check, timeout_s=20)
    except ConditionTimeout:
        state = read_verifier_state(client, verifier)
        return False, state
    if isinstance(matched, dict) and "value" in matched:
        return True, matched["value"]
    return True, matched


def _parse_response(response: str) -> OrderedEventsV3Action:
    control = split_control(response)
    if control.status is None and "TERMINATE" in response:
        raise OrderedEventsV3Error("TERMINATE control must be the exact final line")
    if control.body.strip():
        action = CODEC.parse(control.body)
    elif control.status is not None:
        action = OrderedEventsV3Action(no_op=True, prompt_digest=CODEC.digest)
    else:
        raise OrderedEventsV3Error("empty model response")
    return replace(action, terminate=control.status)


def _click_point(
    operations: Sequence[Operation], cursor: tuple[int, int]
) -> tuple[int, int] | None:
    point = cursor
    for operation in operations:
        if operation.kind == "move_to":
            point = int(operation.args[0]), int(operation.args[1])
        elif operation.kind == "mouse_down" and operation.args == ("left",):
            return point
    return None


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
    draw.text((12, 13), f"target={label} start={start} end={end}", fill="#111111")
    return overlay


def _jpeg_image(value: bytes) -> Image.Image:
    with Image.open(io.BytesIO(value)) as image:
        image.load()
        if image.format != "JPEG" or image.mode != "RGB" or image.size != OBSERVATION_SIZE:
            raise RuntimeError(f"desktop observation violated {OBSERVATION_CONTRACT}")
        return image.copy()


def run_attempt(
    *,
    client: EvalSession,
    task: Task,
    output_dir: Path,
    sglang_url: str,
    api_key: str,
    model: str,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    prepare_task(client, task)
    screen = client.screen_size()
    if screen != OBSERVATION_SIZE:
        raise RuntimeError(f"CUA micro-eval requires {OBSERVATION_SIZE}, got {screen}")

    history: list[dict[str, Any]] = []
    turn_results: list[dict[str, Any]] = []
    steps_dir = output_dir / "steps"
    steps_dir.mkdir()
    started = time.time()
    terminated: str | None = None
    held_keys: tuple[str, ...] = ()
    pointer_button_mask = 0

    for turn_index in range(task.max_turns):
        bbox = norm_bbox_to_px(task.target["bbox"], screen)
        if turn_index == 0:
            start = _bbox_center(bbox)
            _execute(client, (Operation("move_to", start),))
        actual_start = client.cursor_position()
        verifier_before = read_verifier_state(client, task.verifier)
        before = client.screenshot_settled(min_delay_s=0.2, stability_timeout_s=1.0)
        before_path = steps_dir / f"step_{turn_index:03d}.jpg"
        before_path.write_bytes(before)
        history.append(
            {
                "step": turn_index,
                "image": _image_part(before),
            }
        )

        response, finish_reason = _call_model(
            sglang_url=sglang_url,
            api_key=api_key,
            model=model,
            instruction=task.instruction,
            history=history,
            seed=seed,
        )
        parse_error: str | None = None
        parsed: OrderedEventsV3Action | None = None
        operations: tuple[Operation, ...] = ()
        receipt: Any = None
        if finish_reason == "length":
            parse_error = "response truncated at max_tokens"
        else:
            try:
                parsed = _parse_response(response)
            except (TypeError, ValueError) as exc:
                parse_error = str(exc)
        if parsed is not None:
            operations = CODEC.compile_action(
                parsed,
                DisplayGeometry(desktop_width=screen[0], desktop_height=screen[1]),
                actual_start,
            )
            receipt = _execute(client, operations)
            terminated = parsed.terminate
            if receipt is not None:
                held_keys = tuple(receipt.held_keys)
                pointer_button_mask = int(receipt.pointer_button_mask)

        after = client.screenshot_settled(
            min_delay_s=0.5,
            stability_timeout_s=1.5,
            poll_s=0.15,
        )
        end = client.cursor_position()
        verifier_ok, verifier_after = verifier_passed(client, task.verifier)
        expected_ok = action_matches_expected(parsed, task.expected)
        click_point = _click_point(operations, actual_start)
        click_in_bbox = click_point is not None and in_bbox(click_point, bbox)
        location_ok = click_in_bbox if task.expected.get("kind") == "click" else True
        success = bool(expected_ok and verifier_ok and location_ok)

        after_path = steps_dir / f"step_{turn_index:03d}_after.jpg"
        after_path.write_bytes(after)
        _draw_overlay(
            _jpeg_image(before),
            bbox,
            actual_start,
            end,
            str(task.target["label"]),
        ).save(output_dir / f"overlay_{turn_index + 1:03d}.jpg", quality=92)

        canonical_action = CODEC.format(parsed) if parsed is not None else response
        history[-1]["assistant"] = response
        history[-1]["action_text"] = canonical_action
        result = {
            "turn": turn_index + 1,
            "seed": seed,
            "response": response,
            "finish_reason": finish_reason,
            "parse_valid": parse_error is None,
            "parse_error": parse_error,
            "parsed": None if parsed is None else parsed.to_dict(),
            "operations": [operation.as_dict() for operation in operations],
            "execution": None
            if receipt is None
            else {
                "cursor_readback_verified": receipt.cursor_readback_verified,
                "held_keys": list(receipt.held_keys),
                "pointer_button_mask": receipt.pointer_button_mask,
            },
            "target": {**task.target, "bbox_px": list(bbox)},
            "cursor_start": list(actual_start),
            "cursor_end": list(end),
            "expected": task.expected,
            "expected_action_ok": expected_ok,
            "click_point": None if click_point is None else list(click_point),
            "click_in_bbox": click_in_bbox,
            "verifier": task.verifier,
            "verifier_before": verifier_before,
            "verifier_after": verifier_after,
            "verifier_pass": verifier_ok,
            "success": success,
        }
        turn_results.append(result)
        (output_dir / "trajectory.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in turn_results) + "\n",
            encoding="utf-8",
        )
        if success or terminated is not None:
            break

    if not turn_results:
        raise RuntimeError(f"task {task.task_id} produced no turns")
    if terminated is not None and (held_keys or pointer_button_mask):
        raise RuntimeError("model terminated with held desktop inputs")
    success = any(bool(row["success"]) for row in turn_results)
    progress = 1.0 if success else 0.0
    attempt = {
        "schema_version": 1,
        "task_id": task.task_id,
        "category": task.category,
        "instruction": task.instruction,
        "seed": seed,
        "grammar": CODEC.name,
        "system_prompt_sha256": CODEC.digest,
        "turns_total": task.max_turns,
        "turns_attempted": len(turn_results),
        "turns": turn_results,
        "parse_valid": all(bool(row["parse_valid"]) for row in turn_results),
        "expected_action_ok": any(bool(row["expected_action_ok"]) for row in turn_results),
        "verifier_pass": any(bool(row["verifier_pass"]) for row in turn_results),
        "progress": progress,
        "success": success,
        "stop_reason": None
        if success
        else f"model terminated: {terminated}"
        if terminated is not None
        else "turn budget exhausted",
        "elapsed_s": time.time() - started,
    }
    conversation = render_history(
        instruction=task.instruction,
        steps=history,
        target_index=len(history) - 1,
    )
    conversation.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": history[-1]["assistant"]}],
        }
    )
    (output_dir / "conversation.json").write_text(
        json.dumps(
            {
                "instruction": task.instruction,
                "messages": conversation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return attempt


def attempt_plan(tasks: Sequence[Task]) -> list[tuple[int, Task, int, int]]:
    return [
        (task_index, task, attempt_index, _SEED_BASE + attempt_index)
        for task_index, task in enumerate(tasks)
        for attempt_index in range(_ATTEMPTS)
    ]


def _validate_attempts(tasks: Sequence[Task], attempts: Sequence[dict[str, Any]]) -> None:
    expected = {
        (task.task_id, _SEED_BASE + attempt_index)
        for task in tasks
        for attempt_index in range(_ATTEMPTS)
    }
    observed = [(str(row.get("task_id")), row.get("seed")) for row in attempts]
    if (
        len(observed) != len(expected)
        or set(observed) != expected
        or len(set(observed)) != len(observed)
    ):
        raise RuntimeError(
            f"incomplete attempt matrix: expected {len(expected)} unique rows, got "
            f"{len(observed)} rows/{len(set(observed))} unique"
        )


def aggregate_results(tasks: Sequence[Task], attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _validate_attempts(tasks, attempts)
    per_task: dict[str, dict[str, Any]] = {}
    for task in tasks:
        rows = [row for row in attempts if row["task_id"] == task.task_id]
        successes = [bool(row["success"]) for row in rows]
        per_task[task.task_id] = {
            "category": task.category,
            "pass_at_1": sum(successes) / _ATTEMPTS,
            "pass_at_4": any(successes),
            "all_4_success": all(successes),
            "parse_valid_rate": sum(bool(row["parse_valid"]) for row in rows) / _ATTEMPTS,
            "expected_action_rate": sum(bool(row["expected_action_ok"]) for row in rows)
            / _ATTEMPTS,
            "best_of_4_progress": max(float(row["progress"]) for row in rows),
        }
    overall = {
        key: sum(float(row[key]) for row in per_task.values()) / len(per_task)
        for key in ("pass_at_1", "pass_at_4", "all_4_success", "parse_valid_rate")
    }
    scores = {f"overall/{key}": value for key, value in overall.items()}
    return {
        "primary": "overall/pass_at_1",
        "scores": scores,
        "overall": overall,
        "per_task": per_task,
    }


_XCURSOR_REPAIR = """
from pathlib import Path
path = Path('/home/user/server/pyxcursor.py')
old = 'self.display = self.xlib.XOpenDisplay(display)'
new = 'Xcursor.display = self.xlib.XOpenDisplay(display)'
source = path.read_text(encoding='utf-8')
if source.count(new) == 1 and old not in source:
    print('already-patched')
elif source.count(old) == 1 and new not in source:
    path.write_text(source.replace(old, new), encoding='utf-8')
    print('patched')
else:
    raise RuntimeError('unexpected pyxcursor XOpenDisplay contract')
""".strip()


def _require_xcursor_repair(client: DesktopSession) -> None:
    status = _run_guest(client, ["python3", "-c", _XCURSOR_REPAIR]).strip()
    if status == "patched":
        _run_guest_shell(
            client,
            "pid=$(systemctl show -p MainPID --value osworld.service); "
            'test "$pid" -gt 1; '
            'setsid bash -c "sleep 1; kill -9 $pid" >/dev/null 2>&1 &',
        )
        time.sleep(9)
        from desktop.vm import DesktopClient

        probe = DesktopClient(client.base_url)
        probe.wait_ready(timeout_s=120)
        probe.verify_actions_contract()
    elif status != "already-patched":
        raise RuntimeError(f"Xcursor repair returned {status!r}")
    _verify_xcursor_repair(client)


def _verify_xcursor_repair(client: EvalSession) -> None:
    verification = _run_guest(
        client,
        [
            "python3",
            "-c",
            "from pathlib import Path; s=Path('/home/user/server/pyxcursor.py').read_text(); "
            "assert s.count('Xcursor.display = self.xlib.XOpenDisplay(display)') == 1; "
            "assert 'self.display = self.xlib.XOpenDisplay(display)' not in s",
        ],
    )
    if verification:
        raise RuntimeError("Xcursor verification produced unexpected output")


def _wait_for_sglang(proc: subprocess.Popen[Any], port: int, api_key: str) -> None:
    url = f"http://127.0.0.1:{port}/health_generate"
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"owned SGLang process exited with rc={proc.returncode}")
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=3,
            )
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("owned SGLang process did not become ready")


def _assert_serving_model(port: int, api_key: str, model: str) -> None:
    response = requests.get(
        f"http://127.0.0.1:{port}/model_info",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    served = response.json().get("model_path")
    if served != model:
        raise RuntimeError(f"owned SGLang serves {served!r}, not {model!r}")


def _stop_owned_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        raise RuntimeError(f"owned SGLang exited unexpectedly with rc={proc.returncode}")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _init_wandb(output_dir: Path, config: dict[str, Any]) -> Any:
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None
    import wandb

    return wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        dir=str(output_dir),
        config=config,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--desktop-image", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.model_path.is_dir():
        parser.error(f"--model-path is not a directory: {args.model_path}")
    if not args.desktop_image.is_file():
        parser.error(f"--desktop-image is not a file: {args.desktop_image}")
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    model = str(args.model_path.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "cua_micro_eval.log"),
        ],
    )
    suite_raw, tasks = load_suite(_DEFAULT_SUITE)
    plan = attempt_plan(tasks)
    api_key = "cua-micro-owned-sglang"
    attempts: list[dict[str, Any]] = []
    state_lock = threading.Lock()
    port_lease: Any = None
    sglang_log: Any = None
    proc: subprocess.Popen[Any] | None = None
    pool: Any = None
    wandb_run: Any = None
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    final_result: dict[str, Any] | None = None
    started = time.time()
    try:
        port_lease = acquire_port_range(
            count=1,
            purpose="cua-micro-sglang",
            range_start=30_000,
            range_end=32_767,
            lock_dir=Path(f"/tmp/desktop-port-locks-{os.getuid()}"),
        )
        port = port_lease.start
        sglang_log = (output_dir / "sglang.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--api-key",
                api_key,
                "--mem-fraction-static",
                "0.70",
                "--chunked-prefill-size",
                "2048",
            ],
            stdin=subprocess.DEVNULL,
            stdout=sglang_log,
            stderr=subprocess.STDOUT,
        )
        atexit.register(_stop_owned_process, proc)
        pool = build_desktop_pool(
            root_dir=output_dir / "desktop_pool",
            image=args.desktop_image.resolve(),
            config=DesktopPoolConfig(
                min_ready_sessions=_VM_SLOTS,
                max_sessions=_VM_SLOTS,
                max_rollouts_per_session=len(plan) // _VM_SLOTS,
                checkout_timeout_s=1200,
                lease_timeout_s=10_800,
                startup_timeout_s=1200,
            ),
        )
        _wait_for_sglang(proc, port, api_key)
        _assert_serving_model(port, api_key, model)
        pool.start()
        wandb_run = _init_wandb(
            output_dir,
            {
                "suite": suite_raw["suite"],
                "model_path": model,
                "grammar": CODEC.name,
                "attempts": _ATTEMPTS,
                "seeds": [_SEED_BASE + index for index in range(_ATTEMPTS)],
            },
        )

        def run_one(item: tuple[int, Task, int, int]) -> None:
            task_index, task, attempt_index, seed = item
            ordinal = task_index * _ATTEMPTS + attempt_index
            threading.current_thread().name = f"vm{ordinal % _VM_SLOTS}"
            with pool.checkout(timeout_s=1200) as checked:
                client = checked.tracked_env()
                client.reset_to_checkpoint(
                    _XCURSOR_CHECKPOINT,
                    setup=_require_xcursor_repair,
                )
                _verify_xcursor_repair(client)
                result = run_attempt(
                    client=client,
                    task=task,
                    output_dir=output_dir / f"task_{ordinal:03d}_{task.task_id.replace('.', '_')}",
                    sglang_url=f"http://127.0.0.1:{port}/v1",
                    api_key=api_key,
                    model=model,
                    seed=seed,
                )
            with state_lock:
                attempts.append(result)
                partial = {
                    "schema_version": 1,
                    "suite": suite_raw["suite"],
                    "grammar": CODEC.name,
                    "n_samples": len(attempts),
                    "completed": False,
                }
                (output_dir / "result.json").write_text(
                    json.dumps(partial, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

        with ThreadPoolExecutor(max_workers=_VM_SLOTS) as executor:
            futures = [executor.submit(run_one, item) for item in plan]
            for future in futures:
                future.result()
        aggregate = aggregate_results(tasks, attempts)
        final_result = {
            "schema_version": 1,
            "suite": suite_raw["suite"],
            "grammar": CODEC.name,
            **aggregate,
            "params": {
                "model_path": model,
                "attempts": _ATTEMPTS,
                "seeds": [_SEED_BASE + index for index in range(_ATTEMPTS)],
                "sampling": _SAMPLING,
                "observation_contract": OBSERVATION_CONTRACT,
                "max_completed_turns": MAX_COMPLETED_TURNS,
                "previous_actions_max_chars": PREVIOUS_ACTIONS_MAX_CHARS,
            },
            "n_samples": len(attempts),
            "n_tasks": len(tasks),
            "elapsed_s": time.time() - started,
            "completed": True,
        }
        if wandb_run is not None:
            wandb_run.log(final_result["scores"])
    except BaseException as exc:
        primary_error = exc
    finally:
        if pool is not None:
            try:
                pool.close()
                pool_state = pool.snapshot()
                (output_dir / "desktop_pool.json").write_text(
                    json.dumps(pool_state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if pool_state.get("total_failed") != 0:
                    raise RuntimeError(f"desktop pool recorded failures: {pool_state!r}")
            except BaseException as exc:
                cleanup_errors.append(exc)
        if proc is not None:
            try:
                _stop_owned_process(proc)
            except BaseException as exc:
                cleanup_errors.append(exc)
            atexit.unregister(_stop_owned_process)
        if sglang_log is not None:
            sglang_log.close()
        if port_lease is not None:
            try:
                port_lease.release()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1 if primary_error is not None or cleanup_errors else 0)
            except BaseException as exc:
                cleanup_errors.append(exc)

    if primary_error is not None:
        if cleanup_errors:
            raise BaseExceptionGroup(
                "evaluation and cleanup failed", [primary_error, *cleanup_errors]
            )
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_errors:
        raise BaseExceptionGroup("evaluation cleanup failed", cleanup_errors)
    if final_result is None:
        raise RuntimeError("evaluation finished without a result")
    serialized = json.dumps(final_result, indent=2, sort_keys=True) + "\n"
    (output_dir / "result.json").write_text(serialized, encoding="utf-8")
    (output_dir / "completed.json").write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
