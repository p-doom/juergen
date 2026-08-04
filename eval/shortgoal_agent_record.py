"""Session-persistent step driver: an interactive agent records short-goal episodes.

The oracle recorder (``shortgoal_record``) owns one VM for the length of one
``for`` loop. A Sonnet agent cannot live inside that loop — it needs to look at
a frame, decide, and come back — so this module turns the same machinery into
four one-shot commands over a session that lives on disk:

    start  --task_id T --session_dir D --slot N   boot, set up, frame 0
    step   --session_dir D --action '<json>'      one decision, next frame
    finish --session_dir D                        final frame, verifier, publish
    abort  --session_dir D                        give up, tear the VM down

Everything that decides what a recording MEANS is imported, not reimplemented:
``prepare_task`` puts the guest in the task's start state, ``cursor_start_px``
draws the seeded opening pixel, ``convert_step`` grid-snaps the turn and pushes
it through render -> strict parse -> byte-identical re-render -> denormalize for
BOTH arms, ``dispatch_ordered_action`` executes the pixels and ``verify_task``
decides whether the episode counts. The published ``recording.json`` is
therefore the exact schema ``record_task`` writes, plus ``source:
"sonnet_agent"`` and ``n_attempt`` — it replays through
``shortgoal_record.py --replay_from`` untouched (rung 0 stays policy-agnostic,
since replay reads only ``template_id``/``seed``/``instruction``/``params``/
``screen_size``/``steps[].primitives_px``/``steps[].cursor_before``).

The action vocabulary is absolute pixels; the driver converts. Two rejections
are deliberate and load-bearing rather than convenience checks: a move onto the
pixel the cursor already holds (the rel arm cannot render ``move(0,0)``, and
dropping it would break the arms' line identity) and any key spelling that is
not in the v4 grammar's NAME set (an unmapped name would reach pyautogui as a
silent no-op). The step cap is hard: at ``--max_steps`` the driver refuses to
dispatch and the episode has to be finished or aborted.

The frame a step captures is settled with ``settle_for``: a scroll turn waits
longer than the rest, because a wheel scroll's repaint can land after the
stability window has already seen two identical frames and the recorded frame
would then contradict the action it follows.

``step`` also CAPTURES a per-turn first-person thought (``--thought``, or
``--thought_b64`` because thoughts travel inside ssh single-quoted commands
where an apostrophe is the failure mode). It is validated
(``grammar.THOUGHT_MAX_CHARS``, no control characters), stored on the step row as
``thought`` and otherwise inert: the rungs are no-think, the builder renders
nothing from it and the replay path never reads it. That keeps the recording
schema at version 1 — ``thought`` is additive and ignorable, exactly like
``source`` and ``n_attempt`` — and leaves a later thinking-render ablation with
data to work from.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import string
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import requests

import shortgoal_golden as golden
import shortgoal_record as sr
import shortgoal_templates as templates
from osworld_runtime import _DEFAULT_QCOW2, _DEFAULT_QEMU_BIN, _wait_for
from osworld_vm_client import OSWorldClient
from shortgoal_grammar import THOUGHT_MAX_CHARS

_LOGGER = logging.getLogger(__name__)

SOURCE = "sonnet_agent"
SESSION_NAME = "session.json"
QEMU_LOG_NAME = "qemu.log"
FRAME_STEM = "step_{:03d}.png"

DEFAULT_MAX_STEPS = 12
PORT_BASE, PORT_STRIDE, PORT_OFFSET = 5000, 10, 3
VNC_BASE = 5900
MAX_SLOT = 49
KVM_DEVICE = "/dev/kvm"
KILL_TIMEOUT_S = 20.0
KILL_POLL_S = 0.2

SCROLL_SETTLE_DELAY_S = 0.8
SCROLL_SETTLE_STABLE_TIMEOUT_S = 3.0

STATUS_RUNNING = "running"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ABORTED = "aborted"
ABORT_REASON = "aborted"

ACTION_FIELDS = {
    "click": ("at", "button", "count"),
    "move": ("to",),
    "drag": ("from", "to", "button"),
    "type": ("text",),
    "key": ("keys",),
    "scroll": ("notches",),
    "no_op": (),
}
ACTION_KINDS = tuple(ACTION_FIELDS)

BUTTON_NAMES = {"left": "LMB", "middle": "MMB", "right": "RMB"}

KEY_NAMES = {
    "ctrl": "ControlLeft",
    "control": "ControlLeft",
    "ctrlleft": "ControlLeft",
    "ctrlright": "ControlRight",
    "controlright": "ControlRight",
    "shift": "ShiftLeft",
    "shiftleft": "ShiftLeft",
    "shiftright": "ShiftRight",
    "alt": "Alt",
    "altgr": "AltGr",
    "altright": "AltGr",
    "meta": "MetaLeft",
    "super": "MetaLeft",
    "win": "MetaLeft",
    "metaright": "MetaRight",
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "space": "Space",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "comma": "Comma",
    "period": "Period",
    "dot": "Period",
    "slash": "Slash",
    "backslash": "Backslash",
    "semicolon": "Semicolon",
    "quote": "Quote",
    "minus": "Minus",
    "equal": "Equal",
    "backquote": "Backquote",
    "bracketleft": "BracketLeft",
    "bracketright": "BracketRight",
}

RDEV_NAMES = frozenset(
    set(KEY_NAMES.values())
    | {f"Key{letter}" for letter in string.ascii_uppercase}
    | {f"Num{digit}" for digit in string.digits},
)


class VmGone(RuntimeError):
    """The session's VM is not running any more."""


class StepCapReached(RuntimeError):
    """The episode already spent its whole step budget."""


def agent_ports(slot: Any) -> tuple[int, int]:
    """The VM and VNC ports of one driver ``slot``.

    Deliberately NOT derived from ``SLURM_JOB_ID``: an interactive session
    outlives any one job step and has to be reachable from the next command.
    The odd ``PORT_OFFSET`` keeps every slot off the recorder's own
    ``5000 + (JOB_ID % 200) * 10`` grid, so a driver and a batch recorder can
    share a node."""
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot <= MAX_SLOT:
        raise ValueError(f"slot must be an int in [0,{MAX_SLOT}], got {slot!r}")
    offset = slot * PORT_STRIDE + PORT_OFFSET
    return PORT_BASE + offset, VNC_BASE + offset


def frame_name(index: int) -> str:
    """The frame file of decision ``index``, named exactly as the recorder does."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"frame index must be a non-negative int, got {index!r}")
    return FRAME_STEM.format(index)


def key_name(key: Any) -> str:
    """One agent key spelling as the v4 grammar's rdev NAME.

    Unknown spellings raise: ``down(<name>)`` reaches the guest through
    ``_rdev_to_pyautogui``, which lowercases anything it does not know, and
    pyautogui's X11 backend turns an unknown key name into a silent no-op — a
    chord that quietly does nothing is the worst possible training turn."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"a key must be a nonempty string, got {key!r}")
    raw = key.strip()
    if raw in RDEV_NAMES:
        return raw
    plain = raw.lower().replace("_", "").replace("-", "")
    if plain in KEY_NAMES:
        return KEY_NAMES[plain]
    if len(plain) == 1 and plain in string.ascii_lowercase:
        return f"Key{plain.upper()}"
    if len(plain) == 1 and plain in string.digits:
        return f"Num{plain}"
    raise ValueError(
        f"unknown key {key!r}; use a letter, a digit, one of {sorted(KEY_NAMES)}, "
        "or an rdev NAME",
    )


def parse_thought(text: Any = None, b64: Any = None) -> str:
    """One step's captured first-person thought as validated plain text.

    ``b64`` is the transport for a driving agent that reaches this CLI through a
    single-quoted shell command, where an apostrophe in the thought is THE
    failure mode; base64 has no shell metacharacters. Either form may be absent
    or empty, which stores ``""``. Control characters are refused so a stored
    thought can never break a JSON line or smuggle a newline into a later
    render."""
    if text is not None and b64 is not None:
        raise ValueError("pass --thought or --thought_b64, not both")
    if b64 is not None:
        if not isinstance(b64, str):
            raise ValueError(f"a base64 thought must be a string, got {b64!r}")
        try:
            raw = base64.b64decode(b64.strip(), validate=True)
        except ValueError as error:
            raise ValueError(f"a base64 thought must be valid base64: {error}") from error
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"a base64 thought must decode as utf-8: {error}") from error
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError(f"a thought must be a string, got {text!r}")
    if len(text) > THOUGHT_MAX_CHARS:
        raise ValueError(f"a thought is at most {THOUGHT_MAX_CHARS} chars, got {len(text)}")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError(f"control character in thought: {text!r}")
    return text


def _pair(value: Any, what: str) -> tuple[Any, Any]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(f"{what} must be an [x,y] pixel pair, got {value!r}")
    return value[0], value[1]


def _button(value: Any) -> str:
    if value is None:
        return BUTTON_NAMES["left"]
    if value not in BUTTON_NAMES:
        raise ValueError(f"button must be one of {sorted(BUTTON_NAMES)}, got {value!r}")
    return BUTTON_NAMES[value]


def _count(value: Any) -> int:
    if value is None:
        return 1
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 3:
        raise ValueError(f"click count must be an int in [1,3], got {value!r}")
    return value


def action_step(action: Any) -> golden.GoldenStep:
    """One agent action JSON as a validated golden turn in absolute pixels.

    The turn shape is the oracle's own (``click_step``, ``drag_step``,
    ``combo_step``, ...), so an agent decision and a golden decision are the
    same object by the time anything is dispatched or recorded."""
    if not isinstance(action, dict):
        raise ValueError(f"an action must be a JSON object, got {action!r}")
    kind = action.get("kind")
    if kind not in ACTION_KINDS:
        raise ValueError(f"unknown action kind {kind!r}, expected one of {ACTION_KINDS}")
    unknown = sorted(set(action) - {"kind", *ACTION_FIELDS[kind]})
    if unknown:
        raise ValueError(f"a {kind} action takes {ACTION_FIELDS[kind]}, not {unknown}")
    if kind == "click":
        return golden.click_step(
            _pair(action.get("at"), "click at"),
            name=_button(action.get("button")),
            count=_count(action.get("count")),
        )
    if kind == "move":
        return [golden.move(*_pair(action.get("to"), "move to"))]
    if kind == "drag":
        return golden.drag_step(
            _pair(action.get("from"), "drag from"),
            _pair(action.get("to"), "drag to"),
            name=_button(action.get("button")),
        )
    if kind == "type":
        return [golden.type_text(action.get("text"))]
    if kind == "key":
        keys = action.get("keys")
        if not (isinstance(keys, (list, tuple)) and keys):
            raise ValueError(f"key needs a nonempty list of keys, got {keys!r}")
        return golden.combo_step([key_name(key) for key in keys])
    if kind == "scroll":
        return [golden.scroll(action.get("notches"))]
    return [golden.no_op()]


def plan_step(
    action: Any, cursor: tuple[int, int], screen_wh: tuple[int, int],
) -> sr.StepPlan:
    """The agent's action as the recorder's own pixel primitives and grid twin."""
    step = golden.validate_step(action_step(action))
    width, height = screen_wh
    for x, y in golden.move_targets([step]):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"the action targets ({x},{y}), off a {width}x{height} screen")
    plan = sr.convert_step(step, cursor, screen_wh)
    if plan.zero_deltas:
        raise ValueError(
            f"the action moves onto {list(cursor)}, the pixel the cursor already holds: "
            "the rel arm cannot render move(0,0). Target a different pixel, or use "
            '{"kind":"click","count":2} for a repeat click.',
        )
    return plan


def settle_for(settle: sr.Settle, prims: Any) -> sr.Settle:
    """The screenshot policy for one dispatched turn — longer after a scroll.

    A wheel scroll's visible effect (a scroll pad's counter, a scrolled list, a
    browser viewport) can repaint after the stability window has already seen two
    identical frames, which records a frame that contradicts the action it
    follows. Scroll turns therefore wait ``SCROLL_SETTLE_DELAY_S`` before the
    first capture and keep polling longer; every other turn is untouched, and an
    explicit no-settle policy (all zeros, as offline self-checks use) stays off."""
    if not any(getattr(prim, "kind", None) == "scroll" for prim in prims):
        return settle
    if settle.delay_s <= 0 and settle.stable_timeout_s <= 0:
        return settle
    return replace(
        settle,
        delay_s=max(settle.delay_s, SCROLL_SETTLE_DELAY_S),
        stable_timeout_s=max(settle.stable_timeout_s, SCROLL_SETTLE_STABLE_TIMEOUT_S),
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _vm_alive(pid: int, vm_port: int) -> bool:
    """Whether ``pid`` is still the qemu process forwarding ``vm_port``.

    The port is matched against the live command line so a recycled pid can
    never be mistaken for (or killed as) this session's VM."""
    if not _pid_alive(pid):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return False
    return f"hostfwd=tcp::{vm_port}-:5000" in cmdline


def _signal_vm(pid: int, sig: int) -> None:
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
            return
    except (ProcessLookupError, PermissionError):
        pass
    os.kill(pid, sig)


def _kill_vm(pid: int, *, port: int, label: str) -> bool:
    """Stop a detached VM by pid; ``True`` if it was running and is now gone."""
    if not _vm_alive(pid, port):
        return False
    _LOGGER.info("terminating VM %s (pid=%d, port=%d)", label, pid, port)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            _signal_vm(pid, sig)
        except ProcessLookupError:
            return True
        deadline = time.time() + KILL_TIMEOUT_S
        while time.time() < deadline:
            if not _pid_alive(pid):
                return True
            time.sleep(KILL_POLL_S)
    raise RuntimeError(f"VM {label} (pid {pid}) survived SIGKILL")


def _require_kvm() -> None:
    if not os.access(KVM_DEVICE, os.R_OK | os.W_OK):
        raise RuntimeError(
            f"{KVM_DEVICE} is not readable and writable here; the driver needs a KVM node",
        )


def _task(task_id: str) -> templates.ConcreteTask:
    template_id, marker, seed = str(task_id).partition("__s")
    if not marker or not seed.isdigit() or template_id not in templates.TEMPLATES_BY_ID:
        raise ValueError(f"unknown task id {task_id!r}, expected <template_id>__sNN")
    task = templates.concrete_task(template_id, int(seed))
    if task.task_id != task_id:
        raise ValueError(f"{task_id!r} does not resolve to {task.task_id!r}")
    return task


@dataclass
class Session:
    """One interactive recording session, mirrored to ``<session_dir>/session.json``."""

    session_dir: Path
    data: dict[str, Any]

    @classmethod
    def create(
        cls,
        session_dir: Path | str,
        task: templates.ConcreteTask,
        *,
        screen_wh: tuple[int, int],
        cursor_start: tuple[int, int],
        cursor: tuple[int, int],
        setup: dict[str, Any],
        settle: sr.Settle,
        max_steps: int = DEFAULT_MAX_STEPS,
        n_attempt: int = 1,
        slot: int = 0,
        qemu_pid: int = 0,
        qcow2: str = "",
        qemu_bin: str = "",
    ) -> Session:
        """A fresh running session for ``task`` against an already prepared VM."""
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f"max_steps must be a positive int, got {max_steps!r}")
        if not isinstance(n_attempt, int) or isinstance(n_attempt, bool) or n_attempt < 1:
            raise ValueError(f"n_attempt must be a positive int, got {n_attempt!r}")
        vm_port, vnc_port = agent_ports(slot)
        return cls(Path(session_dir), {
            "schema_version": sr.SCHEMA_VERSION,
            "source": SOURCE,
            "status": STATUS_RUNNING,
            "task_id": task.task_id,
            "template_id": task.template_id,
            "seed": task.seed,
            "instruction": task.instruction,
            "screen_size": [int(screen_wh[0]), int(screen_wh[1])],
            "cursor_start": [int(cursor_start[0]), int(cursor_start[1])],
            "cursor": [int(cursor[0]), int(cursor[1])],
            "max_steps": max_steps,
            "n_attempt": n_attempt,
            "slot": int(slot),
            "vm_port": vm_port,
            "vnc_port": vnc_port,
            "qemu_pid": int(qemu_pid),
            "qcow2": str(qcow2),
            "qemu_bin": str(qemu_bin),
            "settle": asdict(settle),
            "setup": setup,
            "started_at": time.time(),
            "steps": [],
            "lines": [],
            "actions": [],
        })

    @classmethod
    def load(cls, session_dir: Path | str) -> Session:
        path = Path(session_dir) / SESSION_NAME
        if not path.is_file():
            raise FileNotFoundError(f"no session at {path}; run start first")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("source") != SOURCE:
            raise ValueError(f"{path} is not a {SOURCE} session")
        if data.get("schema_version") != sr.SCHEMA_VERSION:
            raise ValueError(
                f"{path} is schema_version {data.get('schema_version')!r}, "
                f"this driver writes {sr.SCHEMA_VERSION}",
            )
        return cls(Path(session_dir), data)

    def save(self) -> None:
        sr._write_json(self.session_dir / SESSION_NAME, self.data)

    @property
    def task(self) -> templates.ConcreteTask:
        return _task(str(self.data["task_id"]))

    @property
    def frames_dir(self) -> Path:
        return self.session_dir / sr.FRAMES_DIR

    @property
    def screen_wh(self) -> tuple[int, int]:
        return (int(self.data["screen_size"][0]), int(self.data["screen_size"][1]))

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self.data["steps"]

    def frame_path(self, index: int) -> Path:
        return self.frames_dir / frame_name(index)

    def require_running(self) -> None:
        status = self.data.get("status")
        if status != STATUS_RUNNING:
            raise RuntimeError(f"session {self.session_dir} is {status!r}, not {STATUS_RUNNING}")

    def client(self) -> OSWorldClient:
        """A client for this session's VM, or ``VmGone`` if it is not there."""
        pid, port = int(self.data["qemu_pid"]), int(self.data["vm_port"])
        if not _vm_alive(pid, port):
            raise VmGone(f"the qemu process (pid {pid}) forwarding port {port} is gone")
        client = OSWorldClient(f"http://localhost:{port}")
        try:
            client.cursor_position()
        except (requests.RequestException, ValueError, KeyError, IndexError) as error:
            raise VmGone(f"the guest agent on port {port} is unreachable: {error}") from error
        return client

    def kill_vm(self) -> bool:
        return _kill_vm(
            int(self.data["qemu_pid"]),
            port=int(self.data["vm_port"]),
            label=str(self.data["task_id"]),
        )

    def status(self) -> dict[str, Any]:
        """The compact status the driving agent reads after every command."""
        taken = len(self.steps)
        return {
            "task_id": self.data["task_id"],
            "instruction": self.data["instruction"],
            "frame": str(self.frame_path(taken).resolve()),
            "screen_size": list(self.screen_wh),
            "cursor": list(self.data["cursor"]),
            "step": taken,
            "steps_left": int(self.data["max_steps"]) - taken,
            "max_steps": int(self.data["max_steps"]),
        }

    def apply_step(
        self, client: Any, action: Any, *, settle: sr.Settle, thought: str = "",
    ) -> dict[str, Any]:
        """Dispatch one agent decision and append it in the recorder's own schema.

        ``thought`` rides along on the row, validated and never dispatched."""
        self.require_running()
        thought = parse_thought(thought)
        index = len(self.steps)
        max_steps = int(self.data["max_steps"])
        if index >= max_steps:
            raise StepCapReached(
                f"the episode already used all {max_steps} steps; finish or abort it",
            )
        cursor = tuple(client.cursor_position())
        plan = plan_step(action, cursor, self.screen_wh)
        result = client.dispatch_ordered_action(plan.px_action)
        if tuple(result.cursor_before) != cursor:
            _LOGGER.warning(
                "%s step %d: the pointer was at %s, not the planned %s",
                self.data["task_id"], index, list(result.cursor_before), list(cursor),
            )
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        settle_for(settle, plan.px_action.primitives).shot(client).save(
            self.frame_path(index + 1),
        )
        self.steps.append({
            "primitives_px": sr.serialize_primitives(plan.px_action.primitives),
            "primitives_grid": sr.serialize_primitives(plan.grid_action.primitives),
            "cursor_before": list(result.cursor_before),
            "cursor_after": list(result.cursor_after),
            "frame": frame_name(index),
            "thought": thought,
        })
        self.data["cursor"] = list(result.cursor_after)
        self.data["lines"].append(plan.abs_line)
        self.data["actions"].append(action)
        self.save()
        _LOGGER.info("%s step %d/%d: %s", self.data["task_id"], index + 1, max_steps, plan.abs_line)
        return {**self.status(), "line": plan.abs_line, "thought_chars": len(thought)}

    def recording(
        self, verifier: dict[str, Any] | None, *, reason: str | None = None,
    ) -> dict[str, Any]:
        """This episode in the exact schema ``shortgoal_record.record_task`` publishes."""
        task = self.task
        payload = {
            "schema_version": sr.SCHEMA_VERSION,
            "task_id": task.task_id,
            "template_id": task.template_id,
            "seed": task.seed,
            "category": task.category,
            "tier_b": task.tier_b,
            "single_action": task.single_action,
            "setup_id": task.setup_id,
            "policy_id": task.policy_id,
            "params": task.params,
            "instruction": task.instruction,
            "screen_size": list(self.screen_wh),
            "cursor_start": list(self.data["cursor_start"]),
            "steps": list(self.steps),
            "n_steps": len(self.steps),
            "n_frames": len(self.steps) + 1,
            "zero_delta_moves": 0,
            "setup": self.data["setup"],
            "verifier": verifier,
            "elapsed_s": time.time() - float(self.data["started_at"]),
            "source": SOURCE,
            "n_attempt": int(self.data["n_attempt"]),
        }
        return payload if reason is None else {**payload, "rejected_reason": reason}

    def publish(
        self,
        verifier: dict[str, Any] | None,
        *,
        status: str,
        reason: str | None = None,
    ) -> Path:
        """Write ``recording.json`` (passed) or ``failure.json`` (anything else)."""
        path = self.session_dir / (sr.RECORDING_NAME if reason is None else sr.FAILURE_NAME)
        sr._write_json(path, self.recording(verifier, reason=reason))
        self.data["status"] = status
        self.data["finished_at"] = time.time()
        self.save()
        return path


def _settle(args: argparse.Namespace) -> sr.Settle:
    return sr.Settle(
        delay_s=args.settle_s,
        stable_timeout_s=args.settle_stable_timeout_s,
        poll_s=args.settle_poll_s,
    )


def _reject(
    session: Session, reason: str, *, verifier: dict[str, Any] | None,
) -> dict[str, Any]:
    path = session.publish(verifier, status=STATUS_FAILED, reason=reason)
    killed = session.kill_vm()
    _LOGGER.warning("%s REJECTED (%s)", session.data["task_id"], reason)
    return {
        "verifier_passed": False,
        "reason": reason,
        "failure": str(path.resolve()),
        "vm_killed": killed,
        "task_id": session.data["task_id"],
    }


def cmd_start(args: argparse.Namespace) -> dict[str, Any]:
    """Boot a VM, run the template setup, capture frame 0, persist the session."""
    session_dir = Path(args.session_dir)
    task = _task(args.task_id)
    if session_dir.name != task.task_id:
        _LOGGER.warning(
            "%s is not named %s; the builder reads <root>/<task_id>/recording.json",
            session_dir, task.task_id,
        )
    _require_kvm()
    existing = session_dir / SESSION_NAME
    if existing.is_file():
        stale = Session.load(session_dir)
        if stale.data.get("status") == STATUS_RUNNING and _vm_alive(
            int(stale.data["qemu_pid"]), int(stale.data["vm_port"]),
        ):
            raise RuntimeError(
                f"{session_dir} still runs {stale.data['task_id']} (pid "
                f"{stale.data['qemu_pid']}); finish or abort it first",
            )
        _LOGGER.info("replacing the dead session in %s", session_dir)
    vm_port, vnc_port = agent_ports(args.slot)
    settle = _settle(args)
    frames = session_dir / sr.FRAMES_DIR
    frames.mkdir(parents=True, exist_ok=True)
    for stale_frame in sorted(frames.glob("step_*.png")):
        stale_frame.unlink()
    for name in (sr.RECORDING_NAME, sr.FAILURE_NAME, SESSION_NAME):
        (session_dir / name).unlink(missing_ok=True)
    _LOGGER.info("booting VM for %s (slot %d, port %d)", task.task_id, args.slot, vm_port)
    proc = sr._boot_vm(
        qemu_bin=args.qemu_bin,
        qcow2=args.qcow2,
        vm_port=vm_port,
        vnc_port=vnc_port,
        log_path=session_dir / QEMU_LOG_NAME,
        detach=True,
    )
    try:
        _wait_for(
            f"http://localhost:{vm_port}/screenshot",
            proc=proc,
            poll_s=5,
            max_polls=max(1, int(args.vm_ready_timeout_s // 5)),
            label=f"VM {task.task_id}",
        )
        client = OSWorldClient(f"http://localhost:{vm_port}")
        client.wait_ready(timeout_s=args.vm_ready_timeout_s)
        screen_wh = sr._screen_size(client)
        setup_state = sr.prepare_task(client, task, screen_wh)
        start = sr.cursor_start_px(task.task_id, screen_wh)
        cursor = sr._place_cursor(client, start, label=task.task_id)
        settle.shot(client).save(frames / frame_name(0))
    except BaseException:
        _kill_vm(proc.pid, port=vm_port, label=f"{task.task_id} start")
        raise
    session = Session.create(
        session_dir,
        task,
        screen_wh=screen_wh,
        cursor_start=start,
        cursor=cursor,
        setup=setup_state,
        settle=settle,
        max_steps=args.max_steps,
        n_attempt=args.n_attempt,
        slot=args.slot,
        qemu_pid=proc.pid,
        qcow2=args.qcow2,
        qemu_bin=args.qemu_bin,
    )
    session.save()
    return session.status()


def cmd_step(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch one agent decision against the session's live VM."""
    session = Session.load(args.session_dir)
    session.require_running()
    try:
        action = json.loads(args.action)
    except json.JSONDecodeError as error:
        raise ValueError(f"--action is not valid JSON: {error}") from error
    thought = parse_thought(args.thought, args.thought_b64)
    return session.apply_step(
        session.client(), action, settle=_settle(args), thought=thought,
    )


def cmd_finish(args: argparse.Namespace) -> dict[str, Any]:
    """Re-shoot the final frame, run the template verifier, publish or reject."""
    session = Session.load(args.session_dir)
    session.require_running()
    if not session.steps:
        return _reject(session, "no steps were recorded", verifier=None)
    try:
        client = session.client()
    except VmGone as error:
        return _reject(session, f"the VM was gone before verification: {error}", verifier=None)
    settle = _settle(args)
    settle.shot(client).save(session.frame_path(len(session.steps)))
    verifier = sr.verify_task(
        client, session.task, session.data["setup"], timeout_s=args.verify_timeout_s,
    )
    if not verifier["passed"]:
        return _reject(session, f"verifier {verifier['kind']} failed", verifier=verifier)
    path = session.publish(verifier, status=STATUS_PASSED)
    killed = session.kill_vm()
    _LOGGER.info("%s recorded (%d steps)", session.data["task_id"], len(session.steps))
    return {
        "verifier_passed": True,
        "recording": str(path.resolve()),
        "task_id": session.data["task_id"],
        "n_steps": len(session.steps),
        "n_frames": len(session.steps) + 1,
        "n_attempt": int(session.data["n_attempt"]),
        "vm_killed": killed,
    }


def cmd_abort(args: argparse.Namespace) -> dict[str, Any]:
    """Tear the VM down and record why this attempt produced nothing."""
    session = Session.load(args.session_dir)
    if session.data.get("status") == STATUS_PASSED:
        raise RuntimeError(
            f"{session.session_dir} already published a recording; nothing to abort",
        )
    killed = session.kill_vm()
    path = session.publish(None, status=STATUS_ABORTED, reason=ABORT_REASON)
    return {
        "aborted": True,
        "reason": ABORT_REASON,
        "failure": str(path.resolve()),
        "vm_killed": killed,
        "task_id": session.data["task_id"],
        "step": len(session.steps),
    }


_COMMANDS = {
    "start": cmd_start,
    "step": cmd_step,
    "finish": cmd_finish,
    "abort": cmd_abort,
}


def _add_settle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--settle_s", type=float, default=sr.Settle().delay_s)
    parser.add_argument(
        "--settle_stable_timeout_s", type=float, default=sr.Settle().stable_timeout_s,
    )
    parser.add_argument("--settle_poll_s", type=float, default=sr.Settle().poll_s)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="session-persistent short-goal recorder driven one step at a time",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="boot a VM and set one task up")
    start.add_argument("--session_dir", required=True)
    start.add_argument("--task_id", required=True, help="a catalog task id, template__sNN")
    start.add_argument("--slot", type=int, default=0, help=f"port slot in [0,{MAX_SLOT}]")
    start.add_argument("--n_attempt", type=int, default=1)
    start.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    start.add_argument("--vm_ready_timeout_s", type=float, default=300.0)
    start.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    start.add_argument("--qemu_bin", default=_DEFAULT_QEMU_BIN)
    _add_settle_args(start)

    step = subparsers.add_parser("step", help="dispatch one action JSON")
    step.add_argument("--session_dir", required=True)
    step.add_argument(
        "--action", required=True,
        help='one decision, e.g. {"kind":"click","at":[960,540]} or {"kind":"key","keys":["ctrl","s"]}',
    )
    thought = step.add_mutually_exclusive_group()
    thought.add_argument(
        "--thought", default=None,
        help=f"captured first-person thought for this turn, <={THOUGHT_MAX_CHARS} chars",
    )
    thought.add_argument(
        "--thought_b64", default=None,
        help="the same thought as base64 utf-8, for quote-safe transport",
    )
    _add_settle_args(step)

    finish = subparsers.add_parser("finish", help="verify the episode and publish it")
    finish.add_argument("--session_dir", required=True)
    finish.add_argument("--verify_timeout_s", type=float, default=12.0)
    _add_settle_args(finish)

    abort = subparsers.add_parser("abort", help="give up on the episode")
    abort.add_argument("--session_dir", required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr,
    )
    args = _parse_args(argv)
    try:
        payload = _COMMANDS[args.command](args)
    except StepCapReached as error:
        payload = {"error": "step_cap", "detail": str(error)}
    except VmGone as error:
        payload = {"error": "vm_gone", "detail": str(error)}
    except Exception as error:
        _LOGGER.exception("%s failed", args.command)
        payload = {"error": type(error).__name__, "detail": str(error)}
    else:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
