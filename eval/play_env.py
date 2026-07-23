"""Interactive OSWorld play environment — drive a running VM + model by hand.

The freeroll eval (``freeroll.py``) boots its own sglang server and qemu VM,
then runs a fully-autonomous loop. That is great for a scored run but terrible
for *iterating on a system prompt*: every tweak pays sglang's ~3 min cold start.

``play_env`` decouples the pieces. You start the slow, stateful servers ONCE and
reuse them across dozens of prompt experiments; this script just *attaches* to
them and drops you into a Python REPL where you can hot-swap the system prompt,
invoke the model on the current screen, dispatch its action, poke the desktop by
hand, and watch the result — all without restarting anything.

It reuses the exact building blocks freeroll uses, so what you see here matches a
real rollout: ``OSWorldClient`` (osworld_vm_client), ``_call_model`` /
``append_turn`` (osworld_runtime), ``parse_action_tolerant`` /
``parse_computer_use_tool_call`` (action_parser), and ``SYSTEM_PROMPTS``.

--------------------------------------------------------------------------------
TYPICAL SESSION (inside a GPU alloc on the compute node — login nodes have no
/dev/kvm and no GPU, so qemu is SIGKILLed and sglang can't load):

    salloc --partition=interactive --gres=gpu:1 --cpus-per-task=8 \
           --mem=32G --time=04:00:00 --qos=low

    # ---- terminal 1: the model server (start once, keep it up) ----
    uv run --project=eval python -m sglang.launch_server \
        --model-path /fast/.../bc_export_hf_artifact_XXXX/036000 \
        --host 0.0.0.0 --port 30000 --api-key osworld \
        --mem-fraction-static 0.80 --chunked-prefill-size 2048

    # ---- terminal 2: the WEB UI (boots + owns the VM so it can reboot it) ----
    uv run --project=eval python eval/play_env.py \
        --boot-vm --ui-port 8080 --desktop-setup terminal \
        --system-prompt computer_use_delta_cot_v1

    # (or drop --ui-port for the plain Python REPL; or attach to a separately
    #  booted VM with --vm-port 5000)

WEB UI (``--ui-port N``): a browser dashboard (tunnel it:
``ssh -L 8080:localhost:8080 <node>.haicore.berlin`` then open http://localhost:8080)
showing the live screen + all session info, with buttons to drive the model
(ask/step/run), edit the goal/decoding, edit + save named system prompts, start a
new conversation (optionally rebooting and/or opening a terminal), and browse past
conversations. The screen is view-only there (use the REPL for manual pokes).

Without --ui-port you get the Python REPL below. To just SEE the screen without the
full UI, pass --view-port N (auto-refreshing single image). Screenshots are also
written to ``<out>/latest.png`` (open it in your IDE; it refreshes on change).

--------------------------------------------------------------------------------
REPL HELPERS (operate on the live session ``S``; ``c`` is the OSWorldClient):

    setprompt(x)   set system prompt — a SYSTEM_PROMPTS id, a file path, or a
                   literal string. Edit a file and call setprompt("my.txt") to
                   reload without touching the servers. Alias: sp.
    goal(t)        set the natural-language instruction (rides the first turn).
    shot()         grab + save the current screen; return the PIL image.
    ask()          call the model on the current history; print raw response +
                   parsed action. Does NOT dispatch. This is "invoke by hand".
    step()         ask(), then dispatch the parsed action, settle, screenshot,
                   and append the turn to history (mirrors one freeroll step).
    run(n=10)      step() up to n times or until the model emits TERMINATE.
    reset(reboot=False, terminal=None)  start a NEW conversation (fresh episode
                   folder + cleared history); optionally reboot the VM (needs
                   --boot-vm) and/or open a focused terminal on it.
    new_conversation(goal=None, prompt=None, reboot=False, terminal=None)
                   set goal/prompt then reset() in one call.
    replay(name, reboot=False, terminal=None)  start a NEW conversation seeded
                   from past conversation ``name`` (same prompt/goal/decoding;
                   reboot restarts the desktop clean) and arm its recorded actions.
    rnext()        copy + dispatch the next recorded action from the replay source
                   (no model call). Or set a message (msg) and step() to DIVERGE —
                   the model takes over from the current, partly-replayed state.
    terminal()     open a focused terminal in the VM (no reboot).
    info()         print the current config.

Each reset()/new_conversation() rolls to its own folder under
``<out>/conversation/ep_NNNN_<time>_<goal-slug>/`` (turn JSONs + frames +
transcript + meta.json), so conversations are self-contained and browsable.

    Manual pokes (bypass the model; refresh the current frame afterwards):
    move(dx,dy)    relative cursor move (same semantics the model emits)
    click(x,y=None) / rclick / dclick   absolute click (uses cursor if x is None)
    scroll(n)      wheel ticks (+up / -down, pyautogui convention)
    typ(s)         type a string      key("ctrl","c")   press a chord
    px("expr")     run a raw pyautogui expression in the VM

    hist_action_only(True/False)  store only the parsed action (not the full CoT
                   prose) as each assistant history turn. Use this to test whether
                   feeding a pure-action BC checkpoint its own reasoning is what
                   destabilizes it. Alias for the --history-action-only flag.

Nothing here launches sglang — that is intentionally separate so the server
survives REPL restarts.
"""

from __future__ import annotations

import argparse
import code
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from action_parser import (
    Action,
    parse_action_tolerant,
    parse_computer_use_tool_call,
)
from osworld_runtime import (
    _DEFAULT_QCOW2,
    _DEFAULT_QEMU_BIN,
    _call_model,
    append_turn,
    build_loggable_messages,
    parse_resolution,
)
from osworld_system_prompts import SYSTEM_PROMPTS
from osworld_vm_client import OSWorldClient

_LOGGER = logging.getLogger("play_env")
_TERMINATE = "TERMINATE"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# Filesystem-safe episode/prompt names (mirrors freeroll._slug's spirit).
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _slug(text: str | None) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (text or "no-goal").lower()).strip("-")
    return base[:40] or "no-goal"


def _count_images(messages: Any) -> int:
    """Number of image parts across a loggable messages list (context size)."""
    n = 0
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            n += sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image")
    return n


def _is_terminate(text: str) -> bool:
    """Match freeroll: TERMINATE only if it's the first line (bare)."""
    return text.strip().split("\n", 1)[0].strip() == _TERMINATE


def _action_to_text(action: Action) -> str:
    """Render a parsed Action back to the canonical BC grammar string.

    Mirrors data_pipeline ``format_action`` so a history turn stored this way is
    byte-identical to a training-time assistant turn.
    """
    if action.no_op:
        return "NO_OP"
    parts = [f"{action.dx} {action.dy} {action.scroll}"]
    ev = " ".join(
        ("+" if e.kind == "press" else "-") + e.what for e in action.events
    )
    if ev:
        parts.append(ev)
    return " ; ".join(parts)


def _resolve_system_prompt(x: str) -> tuple[str, str]:
    """Resolve a --system-prompt value to (text, label).

    Order: a key in SYSTEM_PROMPTS, then an existing file path, else a literal.

    Guard the filesystem probe: a full prompt pasted from the UI is long and/or
    multi-line, and ``Path(huge).is_file()`` raises ``OSError: File name too
    long`` (ENAMETOOLONG). Only stat plausible paths, and swallow OSError.
    """
    if x in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[x], f"id:{x}"
    if "\n" not in x and len(x) <= 255:
        try:
            p = Path(x)
            if p.is_file():
                return p.read_text(), f"file:{p}"
        except OSError:
            pass
    return x, "literal"


def _detect_model(sglang_url: str, api_key: str) -> str:
    """Ask sglang which model id it is serving (GET /v1/models)."""
    r = requests.get(
        sglang_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json().get("data") or []
    if not data:
        raise RuntimeError(f"no models served at {sglang_url}")
    return data[0]["id"]


def _sglang_root(sglang_url: str) -> str:
    """Strip a trailing /v1 so we can hit root endpoints like /health_generate."""
    root = sglang_url.rstrip("/")
    return root[:-3].rstrip("/") if root.endswith("/v1") else root


def _spawn_vm(
    *, qemu_bin: str, qcow2: str, vm_port: int, vnc_port: int, log_path: Path
) -> subprocess.Popen:
    """Boot the OSWorld qcow2 headless (same qemu command as freeroll._boot_vm).

    Crucially, we DETACH qemu from the controlling terminal: ``-nographic`` wires
    the guest serial console + QEMU monitor to stdio and would otherwise put our
    tty into raw mode (no echo), breaking the REPL we drop into afterwards.
    Redirecting stdin to /dev/null (so qemu's stdio chardev sees a non-tty) and
    starting a new session keeps our terminal untouched. freeroll never noticed
    because it has no interactive prompt after boot.
    """
    return subprocess.Popen(
        [
            qemu_bin,
            "-enable-kvm", "-cpu", "host", "-smp", "4", "-m", "4G",
            "-machine", "type=q35,accel=kvm",
            "-drive", f"file={qcow2},if=virtio,format=qcow2,snapshot=on",
            "-netdev",
            f"user,id=net0,hostfwd=tcp::{vm_port}-:5000,hostfwd=tcp::{vnc_port}-:5900",
            "-device", "virtio-net-pci,netdev=net0",
            "-display", "none", "-nographic",
        ],
        stdin=subprocess.DEVNULL,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _port_free(port: int) -> bool:
    """True if we can bind host ``port`` — i.e. qemu's hostfwd rule would too.

    Mirrors qemu's bind (INADDR_ANY, no SO_REUSEADDR): if a leftover qemu still
    holds the forwarded port, this returns False, which is exactly the condition
    under which a fresh qemu would abort with 'Could not set up host forwarding
    rule' and exit.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def _wait_ports_free(ports: list[int], *, timeout_s: float = 20.0, poll_s: float = 0.25) -> list[int]:
    """Wait until every port in ``ports`` is bindable; return any still busy."""
    deadline = time.time() + timeout_s
    while True:
        busy = [p for p in ports if not _port_free(p)]
        if not busy or time.time() >= deadline:
            return busy
        time.sleep(poll_s)


def _assert_qemu_alive(proc: subprocess.Popen, log_path: Any, *, what: str) -> None:
    """Fail loudly if qemu died right after spawn (e.g. a fatal hostfwd bind
    failure), instead of letting wait_ready() silently reconnect to a stale VM.

    qemu prints the reason and exits within milliseconds of such a failure, so a
    short grace period is enough to catch it."""
    time.sleep(0.7)
    if proc.poll() is None:
        return
    tail = ""
    try:
        lines = Path(log_path).read_text().strip().splitlines()
        tail = "\n".join(lines[-4:])
    except Exception:
        pass
    raise RuntimeError(
        f"{what}: qemu exited immediately (code {proc.returncode}). "
        f"Last lines of {log_path}:\n{tail}"
    )


def _restore_tty() -> None:
    """Best-effort undo of raw/-echo left on our terminal (belt-and-suspenders)."""
    try:
        if os.isatty(0):
            os.system("stty sane </dev/tty 2>/dev/null")
    except Exception:
        pass


def _start_view_server(out_dir: Path, port: int) -> None:
    """Serve an auto-refreshing view of <out>/latest.png on 0.0.0.0:port."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    latest = out_dir / "latest.png"
    page = (
        b"<!doctype html><meta http-equiv=refresh content=1>"
        b"<body style='margin:0;background:#111;display:flex;"
        b"justify-content:center'>"
        b"<img src='/latest.png' style='max-width:100%;max-height:100vh'>"
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:  # silence access log
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/latest.png"):
                if not latest.exists():
                    self.send_error(404)
                    return
                body = latest.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(page)

    srv = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _LOGGER.info("view server on :%d (tunnel it, then open in a browser)", port)


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
class Session:
    """Mutable interactive state + the model/VM operations the REPL exposes."""

    def __init__(self, *, client: OSWorldClient, out_dir: Path, args: argparse.Namespace):
        self.c = client
        self.out = out_dir
        self.args = args

        # Model config (None-safe: manual-only mode when --no-model).
        self.sglang_url: str | None = None if args.no_model else args.sglang_url
        self.api_key = args.api_key
        self.model: str | None = None if args.no_model else args.model

        # Prompt / goal / decoding.
        self.system_prompt, self.sys_label = _resolve_system_prompt(args.system_prompt)
        self.instruction: str | None = args.goal
        self.n_history_frames = args.n_history_frames
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.history_action_only = args.history_action_only
        # The client resizes screenshots + rescales dispatched coordinates;
        # mirrored here for info()/state()/config logging.
        self.model_resolution: tuple[int, int] | None = client.model_resolution

        # Where saved prompts live; how a fresh VM should be set up.
        self.prompt_dir = Path(getattr(args, "prompt_dir", "system_prompt")).resolve()
        self.desktop_setup = getattr(args, "desktop_setup", "none")

        # Settle timings for post-action screenshots.
        self.settle_s = args.settle_s
        self.settle_stable = args.settle_stable_timeout_s
        self.settle_poll = args.settle_poll_s

        # VM lifecycle (only set when this process booted the VM).
        self.vm_proc: subprocess.Popen | None = None
        self._vm_boot: dict[str, Any] | None = None

        # Rolling history (freeroll invariant: len(actions) == len(frames) - 1).
        self.recent_frames: list[Image.Image] = []
        self.recent_actions: list[str] = []
        self._gif: list[Image.Image] = []
        self._n = 0
        self.last_response: str | None = None
        self.last_parsed: tuple[str, Any] | None = None
        self.last_cursor: list[int] | None = None
        self.screen_size: list[int] | None = None
        self.last_note: str | None = None
        self.last_frame_name: str | None = None
        # Extra user text sent TO the model on the current turn (alongside the
        # current screenshot). One-shot: consumed (cleared) by the next step().
        self.model_message: str | None = None
        self._last_message_sent: str | None = None  # what a turn actually carried

        # Replay: when armed (by replay_conversation), holds a past conversation's
        # recorded action sequence so replay_next() can re-dispatch them one by one
        # into THIS fresh conversation ("copy" a step) without calling the model.
        # ``replay_i`` is the cursor into ``replay["actions"]``. Cleared by reset().
        self.replay: dict[str, Any] | None = None
        self.replay_i = 0

        # Conversation logging: each conversation is its own EPISODE folder under
        # <out>/conversation/, holding per-event JSON files (config + turns), the
        # frames it produced, a transcript, and a meta.json. reset() rolls to a
        # fresh episode. Counters are per-episode.
        self.conv_root = out_dir / "conversation"
        self.conv_root.mkdir(parents=True, exist_ok=True)
        self._episode_i = 0
        self.conv_dir = self.conv_root
        self.transcript_path = self.conv_root / "transcript.txt"
        self.turn = 0
        self._event_i = 0
        self._last_messages: list[dict[str, Any]] | None = None
        self._new_episode()  # start episode 1 + record initial config

    # ----------------------------------------------------------- framing
    def _grab(self, settle: bool = True) -> Image.Image:
        if settle:
            return self.c.screenshot_settled(
                min_delay_s=self.settle_s,
                stability_timeout_s=self.settle_stable,
                poll_s=self.settle_poll,
            )
        return self.c.screenshot()

    def _set_current(self, frame: Image.Image) -> None:
        """Replace the current (latest) frame without touching action history."""
        if self.recent_frames:
            self.recent_frames[-1] = frame
        else:
            self.recent_frames.append(frame)

    def _save(self, frame: Image.Image) -> Path:
        # Frames live inside the current episode dir; latest.png at the out root
        # is the live-view mirror.
        path = self.conv_dir / f"frame_{self._n:03d}.png"
        frame.save(path)
        frame.save(self.out / "latest.png")
        self._gif.append(frame.copy())
        self.last_frame_name = path.name
        self._n += 1
        return path

    def shot(self) -> Image.Image:
        """Grab, store, and save the current screen."""
        frame = self._grab()
        self._set_current(frame)
        path = self._save(frame)
        try:
            cur = self.c.cursor_position()
            self.last_cursor = list(cur)
        except Exception:
            cur = ("?", "?")
        print(f"screen {frame.size}  cursor {tuple(cur)}  -> {path}")
        return frame

    def _refresh(self) -> Image.Image:
        """Re-grab after a manual poke so latest.png / the model see the change."""
        frame = self._grab()
        self._set_current(frame)
        self._save(frame)
        return frame

    def _ensure_frame(self) -> None:
        if not self.recent_frames:
            self.shot()

    # ----------------------------------------------------------- model
    def _parse(self, resp: str) -> tuple[str, Any]:
        """Classify a model response into ('computer_use'|'action'|'error', value)."""
        try:
            call = parse_computer_use_tool_call(resp)
            return "computer_use", call.arguments
        except (TypeError, ValueError):
            pass
        try:
            return "action", parse_action_tolerant(resp)
        except (TypeError, ValueError) as e:
            return "error", str(e)

    def _infer(self) -> tuple[str, tuple[str, Any]]:
        """Call the model on the current context, print + stash. No dispatch, no log.

        Snapshots the exact context (as loggable messages) BEFORE the call so a
        later _log_turn records what the model actually saw, not the post-step
        window.
        """
        if self.sglang_url is None:
            print("no model configured (started with --no-model)")
            self._last_messages = None
            self._last_message_sent = None
            return "", ("error", "no model")
        self._ensure_frame()
        self._last_message_sent = self.model_message
        labels = [f"img_{i}" for i in range(len(self.recent_frames))]
        self._last_messages = build_loggable_messages(
            system_prompt=self.system_prompt,
            instruction=self.instruction,
            recent_actions=self.recent_actions,
            frame_labels=labels,
            current_message=self.model_message,
        )
        resp = _call_model(
            sglang_url=self.sglang_url,
            api_key=self.api_key,
            model=self.model,
            system_prompt=self.system_prompt,
            instruction=self.instruction,  # persisted every turn in this REPL
            recent_frames=self.recent_frames,
            recent_actions=self.recent_actions,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            current_message=self.model_message,  # user text on the current turn
        )
        self.last_response = resp
        parsed = self._parse(resp)
        self.last_parsed = parsed
        print("=== response " + "=" * 52)
        print(resp)
        print("=== parsed " + "=" * 54)
        kind, val = parsed
        if kind == "action":
            print(f"action: {_action_to_text(val)}   (no_op={val.no_op})")
        elif kind == "computer_use":
            print(f"computer_use: {val}")
        else:
            print(f"PARSE ERROR: {val}")
        return resp, parsed

    def ask(self) -> tuple[str, tuple[str, Any]]:
        """Preview the model's next action on the current history: print raw +
        parsed. Does NOT dispatch, and is NOT recorded as a turn — only step()
        appends to the conversation."""
        return self._infer()

    # ----------------------------------------------------------- logging
    def _log(self, record: dict[str, Any]) -> Path:
        """Write one event as its own pretty JSON file in the conversation dir."""
        if record["event"] == "turn":
            name = f"{self._event_i:04d}_turn_{record['turn']:03d}.json"
        else:
            name = f"{self._event_i:04d}_{record['event']}.json"
        self._event_i += 1
        path = self.conv_dir / name
        with path.open("w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return path

    def _log_config(self) -> None:
        self._log({
            "event": "config", "time": _now(),
            "sys_label": self.sys_label, "system_prompt": self.system_prompt,
            "goal": self.instruction, "model": self.model,
            "n_history_frames": self.n_history_frames,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "history_action_only": self.history_action_only,
            "model_resolution": list(self.model_resolution) if self.model_resolution else None,
        })

    def _log_turn(
        self,
        resp: str,
        parsed: tuple[str, Any],
        *,
        dispatched: bool,
        sr: Any = None,
        result_frame: str | None = None,
        note: str | None = None,
        replay: dict[str, Any] | None = None,
    ) -> None:
        self.turn += 1
        kind, val = parsed
        action_text = _action_to_text(val) if kind == "action" else None
        rec: dict[str, Any] = {
            "event": "turn", "turn": self.turn, "time": _now(),
            "sys_label": self.sys_label, "goal": self.instruction,
            "model_message": self._last_message_sent,
            "messages": self._last_messages, "response": resp,
            "parsed_kind": kind, "action": action_text,
            "computer_use": val if kind == "computer_use" else None,
            "dispatched": dispatched, "note": note,
        }
        if replay is not None:
            rec["replay"] = replay
        if sr is not None:
            rec["cursor_before"] = list(sr.cursor_before)
            rec["cursor_after"] = list(sr.cursor_after)
            rec["delta"] = list(sr.delta)
            rec["scroll"] = sr.scroll
            rec["events_dispatched"] = sr.events_dispatched
        if result_frame:
            rec["result_frame"] = result_frame
        self._log(rec)
        self._transcript(rec)
        self._write_meta()

    def _new_episode(self, goal: str | None = None) -> None:
        """Roll to a fresh conversation folder and reset per-episode counters."""
        if goal is not None:
            self.instruction = goal
        self._episode_i += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.conv_dir = self.conv_root / f"ep_{self._episode_i:04d}_{stamp}_{_slug(self.instruction)}"
        self.conv_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.conv_dir / "transcript.txt"
        self.turn = 0
        self._event_i = 0
        self._n = 0
        self._gif = []
        self._write_meta(started=stamp)
        self._log_config()
        _LOGGER.info("new conversation: %s", self.conv_dir.name)

    def _write_meta(self, *, started: str | None = None) -> None:
        mp = self.conv_dir / "meta.json"
        meta: dict[str, Any] = {}
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                meta = {}
        if started and "started" not in meta:
            meta["started"] = started
        meta.update({
            "episode": self._episode_i, "name": self.conv_dir.name,
            "goal": self.instruction, "sys_label": self.sys_label,
            "model": self.model, "n_history_frames": self.n_history_frames,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "history_action_only": self.history_action_only,
            "n_turns": self.turn, "updated": _now(),
        })
        if self.replay:
            meta["replay_of"] = self.replay["source"]
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def _transcript(self, rec: dict[str, Any]) -> None:
        act = rec.get("action") or (
            f"computer_use {rec['computer_use']}" if rec.get("computer_use") else "-"
        )
        head = (
            f"=== turn {rec['turn']}  [{rec['time']}]  "
            f"{rec['sys_label']}  goal={rec['goal']!r}  "
            f"dispatched={rec['dispatched']}"
            + (f"  ({rec['note']})" if rec.get("note") else "")
        )
        lines = [head, "RESPONSE:", rec["response"], f"ACTION: {act}"]
        if "cursor_before" in rec:
            lines.append(
                f"  cursor {rec['cursor_before']} -> {rec['cursor_after']}  "
                f"events={rec['events_dispatched']}  result_frame={rec.get('result_frame')}"
            )
        with self.transcript_path.open("a") as f:
            f.write("\n".join(lines) + "\n\n")

    def logs(self) -> None:
        """Print where the conversation is being logged."""
        n = len(list(self.conv_dir.glob("*.json")))
        print(f"conversation dir : {self.conv_dir}  ({n} json file(s))")
        print(f"transcript.txt   : {self.transcript_path}")

    def add_note(self, text: str) -> None:
        """Attach a free-text note to the current step (its own note event).

        Recorded as ``NNNN_note.json`` in the episode dir, tagged with the turn it
        follows and the current frame, so the UI can show it beside that step.
        """
        text = (text or "").strip()
        if not text:
            return
        self.last_note = text
        self._log({
            "event": "note", "time": _now(), "text": text,
            "after_turn": self.turn, "frame": self.last_frame_name,
        })
        with self.transcript_path.open("a") as f:
            f.write(f"--- note [after turn {self.turn}] ---\n{text}\n\n")
        self._write_meta()
        print(f"note added (after turn {self.turn})")

    def set_message(self, text: str | None) -> None:
        """Set the user message sent to the model on the current turn (alongside
        the screenshot). Empty/blank clears it. Persists until changed."""
        self.model_message = (text or "").strip() or None
        print(f"model message = {self.model_message!r}")

    def _history_text(self, resp: str, parsed: tuple[str, Any]) -> str:
        """What to store as the assistant turn: full response, or action-only."""
        kind, val = parsed
        if self.history_action_only and kind == "action":
            return _action_to_text(val)
        return resp

    def step(self) -> Any:
        """One full turn: infer -> dispatch -> settle screenshot -> append + log."""
        self.terminated = False
        resp, parsed = self._infer()
        self.model_message = None  # one-shot: consumed by this step
        kind, val = parsed

        if kind == "error" and not resp:  # no model configured
            return None
        if _is_terminate(resp) or (
            kind == "computer_use"
            and str(val.get("action", "")).strip().lower() == "terminate"
        ):
            print(">>> model emitted TERMINATE")
            self.terminated = True
            self._log_turn(resp, parsed, dispatched=False, note="terminate")
            return None
        if kind == "error":
            print(">>> not dispatching (parse error); fix the prompt and ask() again")
            self._log_turn(resp, parsed, dispatched=False, note="parse_error")
            return None

        if kind == "computer_use":
            sr = self.c.dispatch_computer_use(val)
        else:
            sr = self.c.dispatch_action(val)
        print(
            f"dispatched: cursor {sr.cursor_before} -> {sr.cursor_after}  "
            f"delta={sr.delta} scroll={sr.scroll}  events={sr.events_dispatched}"
        )

        self.last_cursor = list(sr.cursor_after)
        new_frame = self._grab()
        append_turn(
            self.recent_frames,
            self.recent_actions,
            new_frame,
            self._history_text(resp, parsed),
            n_history_frames=self.n_history_frames,
        )
        result_path = self._save(new_frame)
        self._log_turn(resp, parsed, dispatched=True, sr=sr, result_frame=result_path.name)
        return sr

    def run(self, n: int = 10) -> None:
        """Autonomous rollout: step() up to n times or until TERMINATE."""
        for i in range(1, n + 1):
            print(f"\n---------- step {i}/{n} " + "-" * 40)
            self.step()
            if getattr(self, "terminated", False):
                print(f">>> stopped after {i} step(s)")
                return
        print(f">>> ran {n} step(s)")

    def reset(self, reboot: bool = False, terminal: bool | None = None) -> str:
        """Start a NEW conversation (fresh episode folder + cleared history).

        reboot   -> also reboot the VM to a clean desktop (needs --boot-vm).
        terminal -> open a focused terminal after reset; None falls back to the
                    session's --desktop-setup default.
        """
        want_term = (self.desktop_setup == "terminal") if terminal is None else bool(terminal)
        self.recent_actions.clear()
        self.model_message = None  # fresh conversation starts with no pending message
        self.replay = None  # abandon any armed replay; replay_conversation re-arms
        self.replay_i = 0
        if reboot:
            self._reboot()
        if want_term:
            self._open_terminal()
        self._new_episode()
        frame = self._grab()
        self.recent_frames[:] = [frame]
        self._save(frame)
        msg = f"new conversation: {self.conv_dir.name}" + (
            " (rebooted)" if reboot else "") + (" (terminal)" if want_term else "")
        print(msg)
        return self.conv_dir.name

    def _reboot(self) -> None:
        if self.vm_proc is None or self._vm_boot is None:
            print(
                "cannot reboot: attached to an external VM. Restart play_env with "
                "--boot-vm to let it own (and reboot) the VM."
            )
            return
        _LOGGER.info("rebooting VM...")
        self.vm_proc.terminate()
        try:
            self.vm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.vm_proc.kill()
            self.vm_proc.wait()
        # qemu treats a hostfwd bind failure as FATAL and exits at once. If the
        # forwarded ports are not yet released, the fresh qemu dies instantly and
        # wait_ready() would silently reconnect to the STALE VM — leaving the old
        # desktop (and its apps) on screen. So wait for the ports to actually free
        # (killing our old qemu frees them; a foreign leftover would not), then
        # confirm the new qemu survived before trusting the agent.
        ports = [self._vm_boot["vm_port"], self._vm_boot["vnc_port"]]
        busy = _wait_ports_free(ports)
        if busy:
            raise RuntimeError(
                f"reboot aborted: host port(s) {busy} still in use, so a fresh "
                f"qemu cannot forward them (it would exit and leave the OLD VM "
                f"answering). A leftover qemu is likely holding them — free it "
                f"and retry, e.g.  fuser -k {busy[0]}/tcp"
            )
        self.vm_proc = _spawn_vm(**self._vm_boot)
        _assert_qemu_alive(self.vm_proc, self._vm_boot["log_path"], what="reboot")
        self.c.wait_ready(timeout_s=180.0)

    def _open_terminal(self) -> None:
        """Open a focused terminal in the guest (reuses freeroll's setup)."""
        try:
            from freeroll import _prepare_desktop  # side-effect-free import
            _prepare_desktop(self.c, "terminal")
        except Exception as e:
            _LOGGER.warning("open terminal via freeroll failed (%s); inline fallback", e)
            self.c.execute(
                "import subprocess; subprocess.Popen(['bash', '-lc', "
                "\"(command -v gnome-terminal >/dev/null && gnome-terminal) || "
                "(command -v xfce4-terminal >/dev/null && xfce4-terminal) || "
                "(command -v xterm >/dev/null && xterm)\"]); "
                "time.sleep(2.0); pyautogui.hotkey('ctrl', 'l'); time.sleep(0.2)"
            )

    def terminal(self) -> Image.Image:
        """Open a focused terminal in the VM without rebooting."""
        self._open_terminal()
        return self._refresh()

    # ----------------------------------------------------------- manual pokes
    def move(self, dx: int, dy: int) -> Image.Image:
        """Relative cursor move — same semantics the model's `dx dy` emits."""
        self.c.dispatch_action(Action(dx=int(dx), dy=int(dy), scroll=0, events=(), no_op=False))
        return self._refresh()

    def click(self, x: int | None = None, y: int | None = None) -> Image.Image:
        expr = "pyautogui.click()" if x is None else f"pyautogui.click({int(x)}, {int(y)})"
        self.c.execute(expr)
        return self._refresh()

    def rclick(self, x: int | None = None, y: int | None = None) -> Image.Image:
        expr = (
            "pyautogui.click(button='right')"
            if x is None
            else f"pyautogui.click({int(x)}, {int(y)}, button='right')"
        )
        self.c.execute(expr)
        return self._refresh()

    def dclick(self, x: int | None = None, y: int | None = None) -> Image.Image:
        expr = (
            "pyautogui.doubleClick()"
            if x is None
            else f"pyautogui.doubleClick({int(x)}, {int(y)})"
        )
        self.c.execute(expr)
        return self._refresh()

    def scroll(self, n: int) -> Image.Image:
        self.c.execute(f"pyautogui.scroll({int(n)})")
        return self._refresh()

    def typ(self, s: str) -> Image.Image:
        self.c.execute(f"pyautogui.write({s!r}, interval=0.02)")
        return self._refresh()

    def key(self, *names: str) -> Image.Image:
        args = ", ".join(repr(n) for n in names)
        self.c.execute(f"pyautogui.hotkey({args})")
        return self._refresh()

    def px(self, expr: str) -> Image.Image:
        """Run a raw pyautogui expression in the VM (e.g. 'pyautogui.press(\"esc\")')."""
        self.c.execute(expr)
        return self._refresh()

    # ----------------------------------------------------------- config
    def setprompt(self, x: str) -> None:
        """Change the system prompt — starts a NEW conversation (fresh episode)."""
        self.system_prompt, self.sys_label = _resolve_system_prompt(x)
        print(f"system prompt = {self.sys_label}  ({len(self.system_prompt)} chars)")
        print("-" * 66)
        preview = self.system_prompt[:400]
        print(preview + ("..." if len(self.system_prompt) > 400 else ""))
        self.reset(reboot=False, terminal=False)  # prompt change -> new conversation

    sp = setprompt  # alias

    def goal(self, t: str | None) -> None:
        """Change the goal — starts a NEW conversation (fresh episode)."""
        self.instruction = t
        print(f"goal = {t!r}  — starting new conversation")
        self.reset(reboot=False, terminal=False)  # goal change -> new conversation

    def hist_action_only(self, on: bool = True) -> None:
        self.history_action_only = bool(on)
        print(f"history_action_only = {self.history_action_only}")
        self._log_config()  # decoding tweak stays within the current conversation

    def save_gif(self, name: str = "session.gif") -> None:
        if len(self._gif) < 2:
            print("need >= 2 frames for a gif")
            return
        path = self.out / name
        small = [
            f.resize((min(960, f.width), int(f.height * min(960, f.width) / f.width)))
            for f in self._gif
        ]
        small[0].save(
            path, save_all=True, append_images=small[1:], duration=400, loop=0, optimize=True
        )
        print(f"wrote {path} ({len(small)} frames)")

    def info(self) -> None:
        print(
            "\n".join(
                [
                    f"  vm            {self.c.base_url}"
                    + ("  (owned: reboot ok)" if self.vm_proc else "  (attached)"),
                    f"  sglang        {self.sglang_url}",
                    f"  model         {self.model}",
                    f"  system prompt {self.sys_label} ({len(self.system_prompt)} chars)",
                    f"  goal          {self.instruction!r}",
                    f"  history       {len(self.recent_frames)} frame(s), "
                    f"{len(self.recent_actions)} action(s); cap={self.n_history_frames}; "
                    f"action_only={self.history_action_only}",
                    f"  resolution    "
                    + ("%dx%d (screenshots downscaled, coords upscaled)"
                       % self.model_resolution if self.model_resolution else "native"),
                    f"  decoding      max_tokens={self.max_tokens} temp={self.temperature}",
                    f"  out           {self.out}",
                ]
            )
        )

    # ----------------------------------------------------------- web accessors
    def state(self) -> dict[str, Any]:
        """A JSON-able snapshot for the web UI (cached cursor; no extra VM calls)."""
        kind = self.last_parsed[0] if self.last_parsed else None
        action = None
        if self.last_parsed and self.last_parsed[0] == "action":
            action = _action_to_text(self.last_parsed[1])
        if self.screen_size is None:
            try:
                self.screen_size = list(self.c.screen_size())
            except Exception:
                pass
        replay = None
        if self.replay:
            acts = self.replay["actions"]
            nxt = acts[self.replay_i] if self.replay_i < len(acts) else None
            replay = {
                "source": self.replay["source"],
                "i": self.replay_i, "total": len(acts),
                "done": self.replay_i >= len(acts),
                "next_kind": nxt["kind"] if nxt else None,
                "next_action": (nxt.get("action") or "computer_use") if nxt else None,
                "next_response": nxt.get("response") if nxt else None,
                "next_message": nxt.get("model_message") if nxt else None,
                "next_source_turn": nxt.get("turn") if nxt else None,
            }
        return {
            "sys_label": self.sys_label, "system_prompt": self.system_prompt,
            "goal": self.instruction, "model": self.model,
            "sglang_url": self.sglang_url, "vm_url": self.c.base_url,
            "vm_owned": self.vm_proc is not None,
            "n_history_frames": self.n_history_frames, "max_tokens": self.max_tokens,
            "temperature": self.temperature, "history_action_only": self.history_action_only,
            "desktop_setup": self.desktop_setup, "episode": self.conv_dir.name,
            "turn": self.turn, "n_frames": len(self.recent_frames),
            "n_actions": len(self.recent_actions), "last_response": self.last_response,
            "last_kind": kind, "last_action": action, "last_note": self.last_note,
            "model_message": self.model_message,
            "cursor": self.last_cursor, "screen_size": self.screen_size,
            "model_resolution": list(self.model_resolution) if self.model_resolution else None,
            "replay": replay,
        }

    def list_prompts(self) -> dict[str, Any]:
        files: list[str] = []
        if self.prompt_dir.is_dir():
            for pat in ("*.txt", "*.md"):
                files += [str(p) for p in sorted(self.prompt_dir.glob(pat))]
        return {"ids": sorted(SYSTEM_PROMPTS), "files": files}

    def list_episodes(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in sorted(self.conv_root.glob("ep_*")):
            if not d.is_dir():
                continue
            meta: dict[str, Any] = {}
            mp = d / "meta.json"
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text())
                except Exception:
                    meta = {}
            n_turns = meta.get("n_turns")
            if n_turns is None:
                n_turns = len(list(d.glob("*_turn_*.json")))
            out.append({
                "name": d.name, "label": meta.get("label"), "goal": meta.get("goal"),
                "started": meta.get("started"), "sys_label": meta.get("sys_label"),
                "n_turns": n_turns, "current": d == self.conv_dir,
            })
        return out

    def rename_episode(self, name: str, label: str) -> None:
        """Set (or clear, if blank) a human label on a conversation's meta.json.

        The directory name is the stable id (used by frames/links), so we never
        move it — the label is purely a display alias for finding it again.
        """
        if not _NAME_RE.match(name):
            raise ValueError("bad episode name")
        d = (self.conv_root / name).resolve()
        if d.parent != self.conv_root.resolve() or not d.is_dir():
            raise ValueError("no such episode")
        mp = d / "meta.json"
        meta: dict[str, Any] = {}
        if mp.exists():
            meta = json.loads(mp.read_text())
        label = (label or "").strip()
        if label:
            meta["label"] = label
        else:
            meta.pop("label", None)  # blank clears it
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"renamed {name} -> {label!r}")

    def read_episode(self, name: str) -> dict[str, Any]:
        if not _NAME_RE.match(name):
            raise ValueError("bad episode name")
        d = (self.conv_root / name).resolve()
        if d.parent != self.conv_root.resolve() or not d.is_dir():
            raise ValueError("no such episode")
        meta: dict[str, Any] = {}
        mp = d / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        # All events in chronological (filename-index) order: turns, user notes,
        # and config changes — so the UI can show notes beside their step.
        items: list[dict[str, Any]] = []
        for f in sorted(d.glob("*.json")):
            if f.name == "meta.json":
                continue
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            ev = rec.get("event")
            if ev == "turn":
                items.append({
                    "event": "turn", "turn": rec.get("turn"), "time": rec.get("time"),
                    "response": rec.get("response"), "action": rec.get("action"),
                    "parsed_kind": rec.get("parsed_kind"), "dispatched": rec.get("dispatched"),
                    "note": rec.get("note"), "result_frame": rec.get("result_frame"),
                    "model_message": rec.get("model_message"),
                    "replay": rec.get("replay"),
                    "n_images": _count_images(rec.get("messages")),
                    "cursor_after": rec.get("cursor_after"),
                    "events_dispatched": rec.get("events_dispatched"),
                })
            elif ev == "note":
                items.append({
                    "event": "note", "time": rec.get("time"), "text": rec.get("text"),
                    "after_turn": rec.get("after_turn"), "frame": rec.get("frame"),
                })
            elif ev == "config":
                items.append({
                    "event": "config", "time": rec.get("time"),
                    "sys_label": rec.get("sys_label"), "goal": rec.get("goal"),
                    "system_prompt": rec.get("system_prompt"),
                })
        turns = [it for it in items if it["event"] == "turn"]
        return {"name": name, "meta": meta, "items": items, "turns": turns}

    def save_prompt(self, name: str, text: str) -> str:
        """Save `text` to the prompt dir under `name`, and make it the current prompt."""
        name = name.strip()
        if not _NAME_RE.match(name):
            raise ValueError("prompt name must match [A-Za-z0-9_.-]+")
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        fn = name if name.endswith((".txt", ".md")) else f"{name}.txt"
        path = self.prompt_dir / fn
        path.write_text(text)
        self.system_prompt = text
        self.sys_label = f"file:{path}"
        print(f"saved prompt -> {path}")
        self.reset(reboot=False, terminal=False)  # apply in a new conversation
        return str(path)

    def new_conversation(
        self,
        goal: str | None = None,
        prompt: str | None = None,
        *,
        reboot: bool = False,
        terminal: bool | None = None,
    ) -> str:
        """Set goal/prompt (directly, not into the old episode) then reset()."""
        if goal is not None:
            self.instruction = goal
        if prompt is not None:
            self.system_prompt, self.sys_label = _resolve_system_prompt(prompt)
        return self.reset(reboot=reboot, terminal=terminal)

    # ----------------------------------------------------------- replay
    def _read_source(self, name: str) -> dict[str, Any]:
        """Read a past episode's STARTING setup + its dispatched action sequence.

        Setup comes from the first ``config`` event (which stores the full system
        prompt text — the only place it survives, since sys_label may be
        ``literal`` or a since-edited file) plus meta.json for decoding params.
        The action list is the ``dispatched`` turns in turn order; each carries the
        original response + model_message so the UI can show what is being copied.
        """
        if not _NAME_RE.match(name):
            raise ValueError("bad episode name")
        d = (self.conv_root / name).resolve()
        if d.parent != self.conv_root.resolve() or not d.is_dir():
            raise ValueError("no such episode")
        meta: dict[str, Any] = {}
        mp = d / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        setup: dict[str, Any] = {
            "goal": meta.get("goal"),
            "sys_label": meta.get("sys_label"),
            "system_prompt": None,
            "n_history_frames": meta.get("n_history_frames", self.n_history_frames),
            "max_tokens": meta.get("max_tokens", self.max_tokens),
            "temperature": meta.get("temperature", self.temperature),
            "history_action_only": meta.get("history_action_only", self.history_action_only),
        }
        actions: list[dict[str, Any]] = []
        for f in sorted(d.glob("*.json")):
            if f.name == "meta.json":
                continue
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            ev = rec.get("event")
            if ev == "config" and setup["system_prompt"] is None:
                setup["system_prompt"] = rec.get("system_prompt")
                if rec.get("sys_label"):
                    setup["sys_label"] = rec["sys_label"]
                if setup.get("goal") is None and rec.get("goal") is not None:
                    setup["goal"] = rec["goal"]
            elif ev == "turn" and rec.get("dispatched"):
                kind = rec.get("parsed_kind")
                if kind == "action" and rec.get("action"):
                    actions.append({
                        "kind": "action", "action": rec["action"], "computer_use": None,
                        "response": rec.get("response"), "model_message": rec.get("model_message"),
                        "turn": rec.get("turn"),
                    })
                elif kind == "computer_use" and rec.get("computer_use"):
                    actions.append({
                        "kind": "computer_use", "action": None, "computer_use": rec["computer_use"],
                        "response": rec.get("response"), "model_message": rec.get("model_message"),
                        "turn": rec.get("turn"),
                    })
        return {"name": name, "setup": setup, "actions": actions}

    def replay_conversation(
        self, name: str, *, reboot: bool = False, terminal: bool | None = None
    ) -> str:
        """Start a NEW conversation seeded from a past one and arm its actions.

        Recovers ``name``'s starting setup (system prompt, goal, decoding), applies
        it to this session, resets to a fresh episode (reboot/terminal restart the
        desktop to the same clean state), then arms the recorded action sequence.
        Advance it with replay_next() to "copy" each step; or type a message and
        step() instead to diverge and let the model take over from there.
        """
        src = self._read_source(name)
        setup = src["setup"]
        if not src["actions"]:
            _LOGGER.warning("replay source %s has no dispatched actions", name)
        if setup.get("system_prompt") is not None:
            self.system_prompt = setup["system_prompt"]
            self.sys_label = setup.get("sys_label") or "literal"
        self.instruction = setup.get("goal")
        self.n_history_frames = int(setup.get("n_history_frames") or self.n_history_frames)
        self.max_tokens = int(setup.get("max_tokens") or self.max_tokens)
        if setup.get("temperature") is not None:
            self.temperature = float(setup["temperature"])
        self.history_action_only = bool(setup.get("history_action_only"))
        # Fresh environment with the same setup (clears any prior replay).
        self.reset(reboot=reboot, terminal=terminal)
        # Arm the plan AFTER reset (reset() clears it), then persist provenance.
        self.replay = {"source": name, "actions": src["actions"]}
        self.replay_i = 0
        self._log({
            "event": "replay", "time": _now(), "source": name,
            "n_actions": len(src["actions"]),
        })
        self._write_meta()
        with self.transcript_path.open("a") as f:
            f.write(f"--- replay of {name} armed ({len(src['actions'])} action(s)) ---\n\n")
        print(f"replay armed: {len(src['actions'])} action(s) from {name} "
              f"-> new conversation {self.conv_dir.name}")
        return self.conv_dir.name

    def replay_next(self) -> Any:
        """Copy + dispatch the next recorded action from the replay source.

        Mirrors step() but substitutes the pre-recorded action for a model call, so
        the trajectory reproduces without inference. The original response is stored
        as the assistant history turn (action-only if history_action_only), keeping
        context byte-identical to how the source turn was recorded. No-op with a
        message once the plan is exhausted."""
        self.terminated = False
        if not self.replay:
            print("no replay armed (use replay_conversation first)")
            return None
        acts = self.replay["actions"]
        if self.replay_i >= len(acts):
            print(">>> replay complete — copy nothing (diverge with a message, or step())")
            self.terminated = True
            return None
        item = acts[self.replay_i]

        # Snapshot the context this turn is grounded in (provenance), exactly as
        # _infer would — but no model is called; the action is copied.
        self._ensure_frame()
        self._last_message_sent = None
        labels = [f"img_{i}" for i in range(len(self.recent_frames))]
        self._last_messages = build_loggable_messages(
            system_prompt=self.system_prompt, instruction=self.instruction,
            recent_actions=self.recent_actions, frame_labels=labels, current_message=None,
        )
        resp = item.get("response") or ""
        if item["kind"] == "computer_use":
            parsed: tuple[str, Any] = ("computer_use", item["computer_use"])
            sr = self.c.dispatch_computer_use(item["computer_use"])
        else:
            act = parse_action_tolerant(item["action"])
            parsed = ("action", act)
            sr = self.c.dispatch_action(act)
        self.last_response = resp
        self.last_parsed = parsed
        self.last_cursor = list(sr.cursor_after)
        print(
            f"replay {self.replay_i + 1}/{len(acts)}: copied "
            f"{item.get('action') or 'computer_use'}  "
            f"cursor {sr.cursor_before} -> {sr.cursor_after}"
        )
        new_frame = self._grab()
        append_turn(
            self.recent_frames, self.recent_actions, new_frame,
            self._history_text(resp, parsed), n_history_frames=self.n_history_frames,
        )
        result_path = self._save(new_frame)
        src_turn = item.get("turn")
        self.replay_i += 1
        self._log_turn(
            resp, parsed, dispatched=True, sr=sr, result_frame=result_path.name,
            note=f"replay: copied turn {src_turn} of {self.replay['source']}",
            replay={"source": self.replay["source"], "source_turn": src_turn,
                    "index": self.replay_i - 1},
        )
        if self.replay_i >= len(acts):
            print(">>> replay complete")
        return sr

    def replay_step(self, op: str = "next") -> Any:
        """UI entry point: 'next' copies+dispatches; 'skip' advances the pointer
        without dispatching (for when you've diverged and the action no longer
        applies); 'stop' disarms replay but keeps the conversation."""
        if op == "stop":
            self.replay = None
            self.replay_i = 0
            print("replay disarmed")
            return None
        if not self.replay:
            print("no replay armed")
            return None
        if op == "skip":
            if self.replay_i < len(self.replay["actions"]):
                self.replay_i += 1
            print(f"replay: skipped to {self.replay_i}/{len(self.replay['actions'])}")
            return None
        return self.replay_next()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _build_namespace(sess: Session) -> dict[str, Any]:
    return {
        "S": sess,
        "c": sess.c,
        "setprompt": sess.setprompt,
        "sp": sess.setprompt,
        "goal": sess.goal,
        "shot": sess.shot,
        "ask": sess.ask,
        "step": sess.step,
        "run": sess.run,
        "reset": sess.reset,
        "move": sess.move,
        "click": sess.click,
        "rclick": sess.rclick,
        "dclick": sess.dclick,
        "scroll": sess.scroll,
        "typ": sess.typ,
        "key": sess.key,
        "px": sess.px,
        "hist_action_only": sess.hist_action_only,
        "save_gif": sess.save_gif,
        "info": sess.info,
        "logs": sess.logs,
        "note": sess.add_note,
        "msg": sess.set_message,
        "rename": sess.rename_episode,
        "terminal": sess.terminal,
        "new_conversation": sess.new_conversation,
        "replay": sess.replay_conversation,
        "rnext": sess.replay_next,
        "Image": Image,
        "Action": Action,
        "SYSTEM_PROMPTS": SYSTEM_PROMPTS,
    }


_BANNER = """\
play_env ready. Helpers: setprompt(x)/sp, goal(t), shot(), ask(), step(),
run(n), reset(reboot=False, terminal=None), terminal(), new_conversation(...),
replay(name, reboot=False)/rnext() (re-run a past conversation's actions here),
move/click/rclick/dclick/scroll/typ/key/px, hist_action_only(), save_gif(),
info(), logs().  `S` = session, `c` = VM client.
Each reset()/new_conversation() starts a fresh episode folder under <out>/conversation/.
Iterate a prompt:  sp("my_prompt.txt"); reset(reboot=True, terminal=True); run(20)
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Interactive OSWorld VM + model REPL.")
    # Model server (attach; never launched here).
    p.add_argument("--sglang-url", default="http://localhost:30000/v1",
                   help="OpenAI-compatible base URL incl. /v1 (default localhost:30000).")
    p.add_argument("--model", default=None,
                   help="Served model id (default: auto-detect via GET /v1/models).")
    p.add_argument("--api-key", default="osworld")
    p.add_argument("--no-model", action="store_true",
                   help="Manual-only mode: don't call the model (poke the VM by hand).")
    # VM: attach or boot.
    p.add_argument("--vm-port", type=int, default=None,
                   help="Attach to a VM whose Flask agent (guest :5000) is forwarded here.")
    p.add_argument("--osworld-url", default=None,
                   help="Full VM agent base URL (overrides --vm-port).")
    p.add_argument("--boot-vm", action="store_true",
                   help="Boot the qemu VM in this process (enables reset(reboot=True)).")
    p.add_argument("--vnc-port", type=int, default=5900)
    p.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    p.add_argument("--qemu-bin", default=_DEFAULT_QEMU_BIN)
    # Prompt / goal / decoding.
    p.add_argument("--system-prompt", default="computer_use_delta_cot_v1",
                   help="A SYSTEM_PROMPTS id, a file path, or a literal string.")
    p.add_argument("--goal", default=None, help="Natural-language instruction.")
    p.add_argument("--n-history-frames", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--history-action-only", action="store_true",
                   help="Store only the parsed action (not CoT prose) in history.")
    p.add_argument("--model-resolution", type=parse_resolution, default=None,
                   help="WIDTHxHEIGHT (e.g. 1280x720) the model sees. Screenshots "
                        "are downscaled to this before entering the history/prompt, "
                        "and model-emitted deltas/coordinates are scaled back up to "
                        "the VM's native screen on dispatch. Default: native. Match "
                        "this to the training resolution.")
    # Screenshot settling (post-action).
    p.add_argument("--settle-s", type=float, default=0.3)
    p.add_argument("--settle-stable-timeout-s", type=float, default=2.0)
    p.add_argument("--settle-poll-s", type=float, default=0.1)
    p.add_argument("--desktop-setup", choices=("none", "terminal"), default="none",
                   help="VM state after a reboot/new conversation. 'terminal' opens a "
                        "focused terminal (for typing tasks).")
    p.add_argument("--prompt-dir", default="system_prompt",
                   help="Directory to load/save named system prompts from.")
    # Output / viewing.
    p.add_argument("--out", default="play_env_out", help="Directory for frames + gif.")
    p.add_argument("--view-port", type=int, default=None,
                   help="Serve an auto-refreshing view of latest.png on this port.")
    p.add_argument("--ui-port", type=int, default=None,
                   help="Serve the interactive web UI on this port (replaces the REPL).")
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve / boot the VM.
    vm_proc: subprocess.Popen | None = None
    vm_boot: dict[str, Any] | None = None
    if args.osworld_url:
        base_url = args.osworld_url
    elif args.boot_vm:
        vm_port = args.vm_port or 5000
        vm_boot = dict(
            qemu_bin=args.qemu_bin, qcow2=args.qcow2, vm_port=vm_port,
            vnc_port=args.vnc_port, log_path=out_dir / "qemu.log",
        )
        # Refuse to boot onto ports a leftover qemu still holds: the new qemu
        # would fail its hostfwd rule, exit, and wait_ready() would attach to the
        # STALE VM (wrong desktop). Fail with an actionable message instead.
        busy = [p for p in (vm_port, args.vnc_port) if not _port_free(p)]
        if busy:
            raise SystemExit(
                f"cannot boot: host port(s) {busy} already in use (a leftover "
                f"qemu?). Free them and retry, e.g.  fuser -k {busy[0]}/tcp"
            )
        _LOGGER.info("booting VM (qemu, snapshot mode)...")
        vm_proc = _spawn_vm(**vm_boot)
        _assert_qemu_alive(vm_proc, vm_boot["log_path"], what="boot")
        base_url = f"http://localhost:{vm_port}"
    else:
        base_url = f"http://localhost:{args.vm_port or 5000}"

    client = OSWorldClient(base_url, model_resolution=args.model_resolution)
    _LOGGER.info("waiting for VM agent at %s ...", base_url)
    client.wait_ready(timeout_s=300.0 if args.boot_vm else 60.0)
    _LOGGER.info("VM ready.")

    # Resolve the served model + sanity-check sglang.
    if not args.no_model:
        root = _sglang_root(args.sglang_url)
        try:
            r = requests.get(
                root + "/health_generate",
                headers={"Authorization": f"Bearer {args.api_key}"},
                timeout=5,
            )
            if r.status_code != 200:
                _LOGGER.warning("sglang /health_generate -> %s (is it up?)", r.status_code)
        except requests.RequestException as e:
            _LOGGER.warning("sglang not reachable at %s (%s). Start it first, or "
                            "use --no-model.", args.sglang_url, e)
        if args.model is None:
            try:
                args.model = _detect_model(args.sglang_url, args.api_key)
                _LOGGER.info("served model: %s", args.model)
            except Exception as e:
                _LOGGER.warning("could not auto-detect model (%s); pass --model.", e)

    sess = Session(client=client, out_dir=out_dir, args=args)
    sess.vm_proc = vm_proc
    sess._vm_boot = vm_boot

    if args.view_port:
        _start_view_server(out_dir, args.view_port)

    sess.shot()  # prime the current frame
    sess.info()

    def _teardown_vm() -> None:
        # Terminate the CURRENTLY-owned qemu, not the closed-over local: _reboot()
        # replaces sess.vm_proc, so the original handle is stale after any reboot.
        # Using the stale handle here leaks the live qemu, which keeps holding the
        # forwarded ports and makes the NEXT boot/reboot attach to a stale VM.
        proc = sess.vm_proc or vm_proc
        if proc is not None and proc.poll() is None:
            _LOGGER.info("terminating VM...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    if args.ui_port:
        import play_env_web  # local import: only needed in UI mode
        node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
        _LOGGER.info("web UI on :%d  —  tunnel: ssh -L %d:localhost:%d %s.haicore.berlin",
                     args.ui_port, args.ui_port, args.ui_port, node)
        try:
            play_env_web.serve(sess, host="0.0.0.0", port=args.ui_port)
        finally:
            _teardown_vm()
        return 0

    _restore_tty()  # in case an earlier qemu (this or a prior run) left -echo
    try:
        code.interact(banner=_BANNER, local=_build_namespace(sess), exitmsg="bye")
    finally:
        _teardown_vm()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
