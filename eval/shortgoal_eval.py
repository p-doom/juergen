"""Short-goal ladder evaluator — teacher-forced byte-exactness and closed-loop success.

Two modes over one contract (``eval/shortgoal_grammar.py`` for the line grammar,
the keep-text K=6 window of ``eval/osworld_runtime.py`` for the context):

* ``--mode offline_exact`` — no VM. For every assistant turn of every
  ``shortgoal_build`` chat record, send the EXACT message prefix the builder
  wrote (file-path image refs re-encoded as data URLs the way the runtime sends
  them), decode greedily, and compare the reply to the golden line byte for byte
  after ``strip()``. This is rung 1(a): below 100% on an overfit-1 record the bug
  is plumbing (prompt bytes, chat template, images), not the model.
* ``--mode closed_loop`` — boot the VM the way the recorder does
  (``shortgoal_record._in_fresh_vm``), run the recorder's own ``prepare_task``,
  place its own seeded ``cursor_start_px`` (a recording whose instruction or
  seeded params drifted from the catalog is rejected before the VM boots, since
  training baked the recorded GOAL line in), then step the model through the same
  keep-text assembly training used: settled screenshot -> prompt -> reply -> a
  whole-line ``TERMINATE`` ends the episode, else strict v4 parse (tolerant rescue
  only after the strict failure is logged; unparseable = a failed step,
  re-screenshot and continue) -> ``denorm_v4`` -> ``dispatch_ordered_action``.
  Success is decided by the recorder's own ``verify_task``, never a copy of it.

Everything that defines the training distribution is imported, not restated: the
goal prefix, frame resolution and JPEG quality come from ``shortgoal_build``, the
setup/verifier/cursor/VM plumbing from ``shortgoal_record``, the window from
``osworld_runtime``, the grammar from ``shortgoal_grammar``.

Determinism: pass@1 decodes greedily; for pass@k every attempt samples the Qwen
Instruct tuple under seed ``sha256("<task_id>:<attempt>")`` (``seed_of``). Nothing
here reads the wall clock or the default random state.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

_EVAL_PATH = str(Path(__file__).resolve().parent)
if _EVAL_PATH not in sys.path:
    sys.path.insert(0, _EVAL_PATH)

import sampling as sampling_mod  # noqa: E402
import shortgoal_build as sb  # noqa: E402
import shortgoal_grammar as sg  # noqa: E402
import shortgoal_record as sr  # noqa: E402
import shortgoal_templates as sgt  # noqa: E402
from action_parser import (  # noqa: E402
    OrderedAction, parse_ordered_v4_action, parse_ordered_v4_action_tolerant,
)
from osworld_runtime import (  # noqa: E402
    _DEFAULT_QCOW2, _DEFAULT_QEMU_BIN, _EVAL_DIR, _pil_to_data_url, _wait_for,
    KeepTextWindow, build_keep_text_messages, build_loggable_keep_text_messages,
    call_model_messages, keep_text_messages,
)
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402
from osworld_vm_client import OSWorldClient, StepResult  # noqa: E402
from result import write_result  # noqa: E402
from sampling import SamplingParams  # noqa: E402

_LOGGER = logging.getLogger(__name__)

MODE_OFFLINE = "offline_exact"
MODE_CLOSED = "closed_loop"
MODES = (MODE_OFFLINE, MODE_CLOSED)

DEFAULT_MAX_STEPS = 12
DEFAULT_ATTEMPTS = 1
MAX_ATTEMPTS = 16
DEFAULT_CONTEXT_LENGTH = 16384
DEFAULT_MODEL_RESOLUTION = "{}x{}".format(*sb.DEFAULT_RESOLUTION)
DEFAULT_MAX_EXAMPLES = 20
SEED_MODULUS = 2 ** 31 - 1
UNKNOWN = "unknown"
KIND_TERMINATE = "TERMINATE"
KIND_NO_OP = "NO_OP"
KIND_MISSING = "-"
KIND_UNPARSED = "?"

ARM_BY_PROMPT_ID = {prompt_id: arm for arm, prompt_id in sg.PROMPT_IDS.items()}

_STOP_TERMINATE = "terminate"
_STOP_MAX_STEPS = "max_steps"
_STOP_MODEL_ERROR = "model_error"
_STOP_SCREENSHOT_ERROR = "screenshot_error"

_ModelFn = Callable[..., tuple[str, str | None]]


def seed_of(*parts: str) -> int:
    """A stable seed from string ids alone — sha256 of ``":".join(parts)``."""
    if not parts or any(not isinstance(p, str) or not p for p in parts):
        raise ValueError(f"seed parts must be non-empty strings, got {parts!r}")
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest(), 16) % SEED_MODULUS


def attempt_seed(task_id: str, attempt: int) -> int:
    """The sampling seed of one pass@k attempt (``task_id:attempt``)."""
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise ValueError(f"attempt must be a non-negative int, got {attempt!r}")
    return seed_of(task_id, str(attempt))


def sha256_text(text: str) -> str:
    """Hex sha256 of a string's utf-8 bytes."""
    if not isinstance(text, str):
        raise TypeError(f"sha256_text expects str, got {type(text)!r}")
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path | None) -> str | None:
    """Hex sha256 of a file's bytes, or None when there is no file."""
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_arm(arm: str | None, system_prompt_id: str | None) -> tuple[str, str]:
    """``(arm, system_prompt_id)`` — each implies the other; a mismatch is an error."""
    if arm is None and system_prompt_id is None:
        raise ValueError("pass --arm or --system_prompt_id")
    if arm is None:
        arm = ARM_BY_PROMPT_ID.get(system_prompt_id)
        if arm is None:
            raise ValueError(
                f"--system_prompt_id {system_prompt_id!r} is not a short-goal arm prompt "
                f"(available: {sorted(ARM_BY_PROMPT_ID)})"
            )
    if arm not in sg.ARMS:
        raise ValueError(f"unknown arm {arm!r} (available: {sg.ARMS})")
    resolved = sg.PROMPT_IDS[arm]
    if system_prompt_id is not None and system_prompt_id != resolved:
        raise ValueError(
            f"--system_prompt_id {system_prompt_id!r} does not match --arm {arm} ({resolved})"
        )
    if resolved not in SYSTEM_PROMPTS:
        raise KeyError(f"system prompt {resolved!r} is not registered in osworld_system_prompts")
    return arm, resolved


def task_of(task_id: str) -> sgt.ConcreteTask:
    """The concrete task behind a ``<template_id>__sNN`` id."""
    template_id, sep, seed_text = str(task_id).partition("__s")
    if not sep or not seed_text.isdigit() or template_id not in sgt.TEMPLATES_BY_ID:
        raise ValueError(
            f"not a short-goal task id: {task_id!r} (want <known template_id>__sNN)"
        )
    task = sgt.concrete_task(template_id, int(seed_text))
    if task.task_id != task_id:
        raise ValueError(f"task id {task_id!r} does not round-trip ({task.task_id})")
    return task


def resolve_task_ids(
    *,
    task_ids: str | None,
    split: str | None,
    subset: str | None,
    splits: dict[str, Any],
    limit: int = 0,
) -> list[str]:
    """The ordered task ids to run: ``--task_ids`` win over ``--subset`` over ``--split``."""
    if task_ids:
        ids = [t for t in (s.strip() for s in task_ids.replace(",", " ").split()) if t]
    elif subset:
        ids = list(sb.subset_task_ids(subset, splits))
    elif split:
        ids = list(splits[split])
    else:
        raise ValueError("pass --task_ids, --subset or --split")
    if not ids:
        raise ValueError(
            f"no tasks selected (task_ids={task_ids!r} subset={subset!r} split={split!r})"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"repeated task id in the selection: {sorted(ids)}")
    for task_id in ids:
        task_of(task_id)
        sb.split_of(task_id, splits)
    return ids[:limit] if limit and limit > 0 else ids


def parse_model_resolution(text: str | None) -> tuple[int, int] | None:
    """``"WxH"`` as a pixel pair; empty means "leave frames at native size"."""
    return sb.parse_resolution(text) if text else None


def to_model_frame(img: Image.Image, model_resolution: tuple[int, int] | None) -> Image.Image:
    """The frame as the model sees it: resized to the training frame scale."""
    if model_resolution and img.size != tuple(model_resolution):
        return img.resize(tuple(model_resolution), Image.LANCZOS)
    return img


def _image_part(img: Image.Image, *, quality: int) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": _pil_to_data_url(img, quality=quality)}}


def hydrate_messages(
    messages: Sequence[dict[str, Any]],
    *,
    quality: int,
    model_resolution: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    """A builder record's messages as a request payload: url image refs -> data URLs."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for block in content:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif kind == "image":
                path = Path(block["url"])
                if not path.is_file():
                    raise FileNotFoundError(f"chat record image ref does not exist: {path}")
                with Image.open(path) as raw:
                    frame = to_model_frame(raw.convert("RGB"), model_resolution)
                parts.append(_image_part(frame, quality=quality))
            elif kind == "image_url":
                parts.append({"type": "image_url", "image_url": dict(block["image_url"])})
            else:
                raise ValueError(f"unknown content block {block!r} in a {role} turn")
        out.append({"role": role, "content": parts})
    return out


def kind_seq(text: str, arm: str) -> tuple[str, ...] | None:
    """The primitive-kind sequence of a v4 reply; None when it does not parse."""
    if text == sg.TERMINATE_LINE:
        return (KIND_TERMINATE,)
    try:
        action = parse_ordered_v4_action(text, arm=arm)
    except (TypeError, ValueError):
        return None
    return (KIND_NO_OP,) if action.no_op else tuple(p.kind for p in action.primitives)


def kind_confusion(golden: Sequence[str], predicted: Sequence[str] | None) -> list[str]:
    """Positional ``golden->predicted`` kind pairs (``?`` when the reply won't parse)."""
    if predicted is None:
        return [f"{kind}->{KIND_UNPARSED}" for kind in golden]
    width = max(len(golden), len(predicted))
    return [f"{_at(golden, i)}->{_at(predicted, i)}" for i in range(width)]


def _at(seq: Sequence[str], i: int) -> str:
    return seq[i] if i < len(seq) else KIND_MISSING


def parse_reply(text: str, *, arm: str) -> tuple[OrderedAction | None, bool, bool, str | None]:
    """``(action, strict_ok, tolerant_rescue, error)`` — strict first, rescue after."""
    try:
        return parse_ordered_v4_action(text, arm=arm), True, False, None
    except (TypeError, ValueError) as strict_error:
        error = f"strict {arm} parse failed: {strict_error}"
    try:
        return parse_ordered_v4_action_tolerant(text, arm=arm), False, True, error
    except (TypeError, ValueError) as tolerant_error:
        return None, False, False, f"{error}; tolerant rescue failed: {tolerant_error}"


@dataclass
class TurnScore:
    """One teacher-forced assistant turn: the golden line versus what the model said."""

    task_id: str
    template_id: str
    category: str
    record_index: int
    turn_index: int
    golden: str
    predicted: str
    finish_reason: str | None
    exact: bool
    parse_ok: bool
    parse_error: str | None
    golden_kinds: list[str]
    predicted_kinds: list[str] | None
    elapsed_s: float


@dataclass
class _TurnTally:
    n: int = 0
    exact: int = 0
    parse_ok: int = 0

    def add(self, turn: TurnScore) -> None:
        self.n += 1
        self.exact += int(turn.exact)
        self.parse_ok += int(turn.parse_ok)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "exact": self.exact,
            "exact_line_rate": _rate(self.exact, self.n),
            "parse_ok": self.parse_ok,
            "parse_valid_rate": _rate(self.parse_ok, self.n),
        }


def _rate(hits: int, n: int) -> float:
    return (hits / n) if n else 0.0


def _tallies(groups: dict[str, _TurnTally]) -> dict[str, Any]:
    return {key: groups[key].as_dict() for key in sorted(groups)}


def record_meta(record: dict[str, Any], index: int) -> tuple[str, str, str]:
    """``(task_id, template_id, category)`` of a builder record, derived from its task id."""
    task_id = str(record.get("task_id") or f"record_{index:04d}")
    template_id = str(record.get("template_id") or task_id.partition("__s")[0])
    template = sgt.TEMPLATES_BY_ID.get(template_id)
    category = str(record.get("category") or (template.category if template else UNKNOWN))
    return task_id, template_id, category


def record_messages(record: Any, index: int) -> list[dict[str, Any]]:
    """The record's validated chat messages (system first, at least one assistant turn)."""
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        raise ValueError(f"chat record {index} has no messages list")
    messages = record["messages"]
    roles = [m.get("role") for m in messages]
    if not messages or roles[0] != "system" or "assistant" not in roles:
        raise ValueError(f"chat record {index} is not a system/user/assistant record: {roles!r}")
    return messages


def check_record_contract(
    record: dict[str, Any], messages: Sequence[dict[str, Any]], *,
    index: int, arm: str, prompt: str, prompt_id: str,
) -> None:
    """The same-bytes train/eval gate: registered prompt, matching arm."""
    system = messages[0]["content"]
    if not isinstance(system, str) or system.strip() != prompt:
        raise ValueError(
            f"chat record {index} system prompt differs from {prompt_id}: record sha "
            f"{sha256_text(system if isinstance(system, str) else '')} vs registered sha "
            f"{sha256_text(prompt)}"
        )
    record_arm = record.get("arm") or record.get("action_format")
    if record_arm is not None and str(record_arm) != arm:
        raise ValueError(f"chat record {index} was built for {record_arm!r}, not {arm!r}")


def score_record(
    record: dict[str, Any],
    *,
    index: int,
    arm: str,
    prompt: str,
    prompt_id: str,
    call: _ModelFn,
    quality: int,
    model_resolution: tuple[int, int] | None,
) -> list[TurnScore]:
    """Teacher-forced scores for every assistant turn of one builder record."""
    messages = record_messages(record, index)
    check_record_contract(
        record, messages, index=index, arm=arm, prompt=prompt, prompt_id=prompt_id,
    )
    task_id, template_id, category = record_meta(record, index)
    hydrated = hydrate_messages(messages, quality=quality, model_resolution=model_resolution)
    scores: list[TurnScore] = []
    for turn_index, position in enumerate(
        [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    ):
        if messages[position - 1]["role"] != "user":
            raise ValueError(
                f"chat record {index} assistant turn {turn_index} does not follow a user turn"
            )
        golden = str(messages[position]["content"]).strip()
        golden_kinds = kind_seq(golden, arm)
        if golden_kinds is None:
            raise ValueError(
                f"chat record {index} turn {turn_index} golden line is not a legal {arm} "
                f"line: {golden!r}"
            )
        t0 = time.time()
        response, finish_reason = call(hydrated[:position], seed=None)
        predicted = response.strip()
        predicted_kinds = kind_seq(predicted, arm)
        truncated = finish_reason == "length"
        if truncated:
            parse_error = f"response truncated at max_tokens: {predicted!r}"
        elif predicted_kinds is None:
            parse_error = f"unparseable {arm} reply"
        else:
            parse_error = None
        scores.append(TurnScore(
            task_id=task_id,
            template_id=template_id,
            category=category,
            record_index=index,
            turn_index=turn_index,
            golden=golden,
            predicted=predicted,
            finish_reason=finish_reason,
            exact=not truncated and predicted == golden,
            parse_ok=parse_error is None,
            parse_error=parse_error,
            golden_kinds=list(golden_kinds),
            predicted_kinds=None if predicted_kinds is None else list(predicted_kinds),
            elapsed_s=time.time() - t0,
        ))
    return scores


def aggregate_offline(
    turns: Sequence[TurnScore], *, n_records: int, max_examples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(scores, extra)`` for the teacher-forced pass: rates, confusion, worst examples."""
    overall = _TurnTally()
    by_template: dict[str, _TurnTally] = {}
    by_category: dict[str, _TurnTally] = {}
    by_turn_index: dict[str, _TurnTally] = {}
    by_task: dict[str, _TurnTally] = {}
    confusion: Counter[str] = Counter()
    golden_kinds: Counter[str] = Counter()
    for turn in turns:
        overall.add(turn)
        by_template.setdefault(turn.template_id, _TurnTally()).add(turn)
        by_category.setdefault(turn.category, _TurnTally()).add(turn)
        by_turn_index.setdefault(str(turn.turn_index), _TurnTally()).add(turn)
        by_task.setdefault(turn.task_id, _TurnTally()).add(turn)
        confusion.update(kind_confusion(turn.golden_kinds, turn.predicted_kinds))
        golden_kinds.update(turn.golden_kinds)
    worst = sorted(
        (t for t in turns if not t.exact),
        key=lambda t: (t.parse_ok, -len(t.golden), t.task_id, t.turn_index),
    )[:max_examples]
    scores = {
        "exact_line_rate": _rate(overall.exact, overall.n),
        "parse_valid_rate": _rate(overall.parse_ok, overall.n),
        "n_turns": overall.n,
        "n_records": n_records,
        "n_exact": overall.exact,
        "by_template": _tallies(by_template),
        "by_category": _tallies(by_category),
        "by_turn_index": _tallies(by_turn_index),
        "by_task": _tallies(by_task),
    }
    extra = {
        "kind_confusion": dict(sorted(confusion.items())),
        "golden_kind_totals": dict(sorted(golden_kinds.items())),
        "worst_examples": [asdict(t) for t in worst],
    }
    return scores, extra


@dataclass
class StepRecord:
    """One closed-loop model turn: what came back and what was dispatched."""

    step: int
    response: str
    action_line: str
    finish_reason: str | None
    terminate: bool
    strict_ok: bool
    tolerant_rescue: bool
    dispatched: bool
    parse_error: str | None
    events_dispatched: list[str]
    elapsed_s: float


@dataclass
class Episode:
    """One attempt at one task: the verifier's verdict plus the format diagnostics.

    ``blind_history_steps`` counts the decisions taken under a context that already
    holds an evicted frame. Training records never carry one (the builder refuses the
    multi-record shape), so a nonzero count marks a rollout that ran past the trained
    window and is scored on a context shape it was never taught.
    """

    task_id: str
    template_id: str
    category: str
    tier: str
    arm: str
    attempt: int
    seed: int | None
    success: bool
    stop_reason: str
    steps_used: int
    strict_ok: int
    tolerant_rescue: int
    failed_steps: int
    blind_history_steps: int
    parse_valid_rate: float
    tolerant_rescue_rate: float
    spurious_terminate: bool
    never_terminate: bool
    verifier_detail: Any
    verifier_error: str | None
    artifact_dir: str


def load_recording(recordings_root: Path | None, task_id: str) -> dict[str, Any] | None:
    """The task's ``recording.json`` — the recorded cursor start and setup provenance."""
    if recordings_root is None:
        return None
    path = Path(recordings_root) / task_id / sr.RECORDING_NAME
    if not path.is_file():
        _LOGGER.warning("no %s for %s under %s", sr.RECORDING_NAME, task_id, recordings_root)
        return None
    return json.loads(path.read_text())


def check_recording(recording: dict[str, Any] | None, task: sgt.ConcreteTask) -> None:
    """The recorded goal and params must still be the ones this run prompts with.

    The closed loop conditions on ``GOAL: <catalog instruction>`` and sets the task up
    from the catalog draw, while training baked the RECORDED instruction into every
    record; ``shortgoal_build.check_catalog_task`` is the same gate the builder ran."""
    if recording is None:
        return
    sb.check_catalog_task(recording, what=f"{task.task_id} recording")
    if recording.get("task_id") != task.task_id:
        raise ValueError(
            f"recording claims task_id {recording.get('task_id')!r}, not {task.task_id!r}"
        )


def cursor_start(
    recording: dict[str, Any] | None, task_id: str, screen_wh: tuple[int, int],
) -> tuple[int, int]:
    """The pixel the pointer starts at: the recorded start, else the recorder's own draw."""
    value = (recording or {}).get("cursor_start")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return sr.cursor_start_px(task_id, screen_wh)


def run_episode(
    *,
    task: sgt.ConcreteTask,
    tier: str,
    arm: str,
    attempt: int,
    seed: int | None,
    client: Any,
    call: _ModelFn,
    system_prompt: str,
    out_dir: Path,
    max_steps: int,
    settle: sr.Settle,
    model_resolution: tuple[int, int] | None,
    jpeg_quality: int,
    save_frames: bool,
    setup: Callable[..., dict[str, Any]],
    verify: Callable[..., dict[str, Any]],
    verify_timeout_s: float,
    recording: dict[str, Any] | None = None,
) -> tuple[Episode, list[StepRecord]]:
    """One closed-loop attempt: recorder setup, keep-text model loop, recorder verifier.

    The model's own reply text is what enters the history (an empty reply falls
    back to ``NO_OP`` so the window keeps its turn positions), and a reply that
    neither parses strictly nor survives the tolerant rescue is a failed step:
    nothing is dispatched, the screen is re-captured and the loop continues.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = out_dir / "steps"
    if save_frames:
        steps_dir.mkdir(exist_ok=True)
    screen_wh = sr._screen_size(client)
    (out_dir / "task.json").write_text(json.dumps(asdict(task), indent=2))
    setup_state = setup(client, task, screen_wh)
    cursor = sr._place_cursor(
        client, cursor_start(recording, task.task_id, screen_wh), label=task.task_id,
    )
    goal = sb.GOAL_PREFIX + task.instruction
    _LOGGER.info(
        "%s attempt %d | screen %dx%d | cursor start %s | goal %r",
        task.task_id, attempt, screen_wh[0], screen_wh[1], list(cursor), goal,
    )

    frame = settle.shot(client)
    frames_for_gif = [frame.copy()]
    if save_frames:
        frame.save(steps_dir / "step_000.png")
    window = KeepTextWindow(to_model_frame(frame, model_resolution))

    steps: list[StepRecord] = []
    stop_reason = _STOP_MAX_STEPS
    blind_history_steps = 0
    with (out_dir / "conversation.jsonl").open("w") as conv_f:
        for step in range(1, max_steps + 1):
            t0 = time.time()
            messages = build_keep_text_messages(
                system_prompt=system_prompt, goal=goal,
                frames=window.frames, actions=window.actions, quality=jpeg_quality,
            )
            blind_history_steps += int(window.live_count() < len(window.frames))
            loggable = build_loggable_keep_text_messages(
                system_prompt=system_prompt, goal=goal, actions=window.actions,
                frame_labels=window.frame_labels(), liveness=window.liveness(),
            )
            if save_frames:
                (steps_dir / f"prompt_{step:03d}.json").write_text(
                    json.dumps(loggable, indent=2))
            try:
                response, finish_reason = call(messages, seed=seed)
            except Exception as e:
                _LOGGER.error("%s step %d: model call failed: %s", task.task_id, step, e)
                stop_reason = _STOP_MODEL_ERROR
                break
            conv_f.write(json.dumps({
                "step": step, "messages": loggable,
                "response": response, "finish_reason": finish_reason,
            }) + "\n")
            conv_f.flush()
            text = response.strip()
            _LOGGER.info(
                "%s step %d | live frames %d | finish_reason=%s | response=%r",
                task.task_id, step, window.live_count(), finish_reason, response,
            )

            terminate = finish_reason != "length" and text == sg.TERMINATE_LINE
            strict_ok = terminate
            rescued = False
            dispatched = False
            parse_error: str | None = None
            events: list[str] = []
            if finish_reason == "length":
                parse_error = (
                    f"response truncated at max_tokens (finish_reason='length'); "
                    f"nothing dispatched: {response!r}"
                )
            elif not terminate:
                action, strict_ok, rescued, parse_error = parse_reply(text, arm=arm)
                if action is not None:
                    result = client.dispatch_ordered_action(sg.denorm_v4(action, screen_wh))
                    dispatched = True
                    events = list(result.events_dispatched)
            if parse_error is not None:
                log = _LOGGER.warning if dispatched else _LOGGER.error
                log("%s step %d: %s", task.task_id, step, parse_error)

            steps.append(StepRecord(
                step=step, response=response, action_line=text, finish_reason=finish_reason,
                terminate=terminate, strict_ok=strict_ok, tolerant_rescue=rescued,
                dispatched=dispatched, parse_error=parse_error, events_dispatched=events,
                elapsed_s=time.time() - t0,
            ))
            if terminate:
                stop_reason = _STOP_TERMINATE
                _LOGGER.info("%s step %d: model terminated", task.task_id, step)
                break

            try:
                frame = settle.shot(client)
            except Exception as e:
                _LOGGER.error("%s step %d: screenshot failed: %s", task.task_id, step, e)
                stop_reason = _STOP_SCREENSHOT_ERROR
                break
            frames_for_gif.append(frame.copy())
            if save_frames:
                frame.save(steps_dir / f"step_{step:03d}.png")
            window.append_turn(text or response or sg.NO_OP_LINE,
                               to_model_frame(frame, model_resolution))

    if save_frames and len(frames_for_gif) > 1:
        small = [
            f.resize((min(960, f.width), int(f.height * min(960, f.width) / f.width)))
            for f in frames_for_gif
        ]
        small[0].save(out_dir / "rollout.gif", save_all=True, append_images=small[1:],
                      duration=300, loop=0, optimize=True)

    success = False
    detail: Any = None
    verifier_error: str | None = None
    try:
        detail = verify(client, task, setup_state, timeout_s=verify_timeout_s)
        success = bool(detail["passed"])
    except Exception as e:
        verifier_error = f"{type(e).__name__}: {e}"
        _LOGGER.error("%s attempt %d: verifier failed: %s", task.task_id, attempt, e)
    n_turns = len(steps)
    episode = Episode(
        task_id=task.task_id, template_id=task.template_id, category=task.category,
        tier=tier, arm=arm, attempt=attempt, seed=seed, success=success,
        stop_reason=stop_reason, steps_used=n_turns,
        strict_ok=sum(s.strict_ok for s in steps),
        tolerant_rescue=sum(s.tolerant_rescue for s in steps),
        failed_steps=sum(not s.strict_ok and not s.tolerant_rescue for s in steps),
        blind_history_steps=blind_history_steps,
        parse_valid_rate=_rate(sum(s.strict_ok for s in steps), n_turns),
        tolerant_rescue_rate=_rate(sum(s.tolerant_rescue for s in steps), n_turns),
        spurious_terminate=stop_reason == _STOP_TERMINATE and not success,
        never_terminate=stop_reason != _STOP_TERMINATE,
        verifier_detail=detail, verifier_error=verifier_error,
        artifact_dir=str(out_dir),
    )
    (out_dir / "episode.json").write_text(json.dumps(
        {"episode": asdict(episode), "steps": [asdict(s) for s in steps]}, indent=2))
    _LOGGER.info(
        "%s attempt %d | success=%s stop=%s steps=%d strict=%d rescue=%d failed=%d blind=%d",
        task.task_id, attempt, success, stop_reason, n_turns,
        episode.strict_ok, episode.tolerant_rescue, episode.failed_steps,
        episode.blind_history_steps,
    )
    return episode, steps


@dataclass
class _TaskTally:
    n_tasks: int = 0
    pass1: int = 0
    passk: int = 0
    attempts: int = 0
    steps_sum: int = 0
    strict_ok: int = 0
    rescued: int = 0
    spurious: int = 0
    never_term: int = 0

    def add(self, episodes: Sequence[Episode]) -> None:
        self.n_tasks += 1
        self.attempts += len(episodes)
        self.pass1 += int(bool(episodes) and episodes[0].success)
        self.passk += int(any(e.success for e in episodes))
        for e in episodes:
            self.steps_sum += e.steps_used
            self.strict_ok += e.strict_ok
            self.rescued += e.tolerant_rescue
            self.spurious += int(e.spurious_terminate)
            self.never_term += int(e.never_terminate)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "n_attempts": self.attempts,
            "pass_at_1": _rate(self.pass1, self.n_tasks),
            "pass_at_k": _rate(self.passk, self.n_tasks),
            "mean_steps": _rate(self.steps_sum, self.attempts),
            "parse_valid_rate": _rate(self.strict_ok, self.steps_sum),
            "tolerant_rescue_rate": _rate(self.rescued, self.steps_sum),
            "spurious_terminate_rate": _rate(self.spurious, self.attempts),
            "never_terminate_rate": _rate(self.never_term, self.attempts),
        }


def aggregate_closed(
    episodes: Sequence[Episode], *, attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(scores, extra)`` for the closed-loop pass: pass@1/pass@k per group."""
    by_task: dict[str, list[Episode]] = {}
    for e in episodes:
        by_task.setdefault(e.task_id, []).append(e)
    for runs in by_task.values():
        runs.sort(key=lambda e: e.attempt)
    overall = _TaskTally()
    groups: dict[str, dict[str, _TaskTally]] = {
        "by_template": {}, "by_category": {}, "by_tier": {}, "by_arm": {},
    }
    for task_id in sorted(by_task):
        runs = by_task[task_id]
        head = runs[0]
        overall.add(runs)
        groups["by_template"].setdefault(head.template_id, _TaskTally()).add(runs)
        groups["by_category"].setdefault(head.category, _TaskTally()).add(runs)
        groups["by_tier"].setdefault(head.tier, _TaskTally()).add(runs)
        groups["by_arm"].setdefault(head.arm, _TaskTally()).add(runs)
    scores = {"k": attempts, **overall.as_dict()}
    for name, tallies in groups.items():
        scores[name] = {key: tallies[key].as_dict() for key in sorted(tallies)}
    extra = {
        "failed_tasks": sorted(
            task_id for task_id, runs in by_task.items() if not any(e.success for e in runs)
        ),
        "blind_history_steps": sum(e.blind_history_steps for e in episodes),
        "blind_history_tasks": sorted(
            {e.task_id for e in episodes if e.blind_history_steps},
        ),
        "stop_reasons": dict(sorted(Counter(e.stop_reason for e in episodes).items())),
        "verifier_errors": sorted(
            {e.task_id for e in episodes if e.verifier_error is not None}
        ),
    }
    return scores, extra


def sglang_caller(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    sampling: SamplingParams,
    request_timeout_s: float,
) -> _ModelFn:
    """A model callable over an already-assembled message list (the only network path)."""

    def call(messages: list[dict[str, Any]], *, seed: int | None = None) -> tuple[str, str | None]:
        return call_model_messages(
            sglang_url=sglang_url, api_key=api_key, model=model, messages=messages,
            sampling=sampling, seed=seed, request_timeout_s=request_timeout_s,
        )

    return call


def spawn_sglang(
    *,
    model_path: str,
    port: int,
    api_key: str,
    mem_fraction_static: float,
    context_length: int,
    log_path: Path,
) -> subprocess.Popen:
    """Serve ``model_path`` with sglang, freeroll's spawn pattern plus a context floor."""
    return subprocess.Popen(
        ["uv", "run", "--project", str(_EVAL_DIR), "python", "-m", "sglang.launch_server",
         "--model-path", model_path,
         "--host", "0.0.0.0",
         "--port", str(port),
         "--api-key", api_key,
         "--mem-fraction-static", str(mem_fraction_static),
         "--context-length", str(context_length),
         "--chunked-prefill-size", "2048"],
        cwd=str(_EVAL_DIR),
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )


def run_offline(
    *,
    chat_path: Path,
    arm: str,
    prompt: str,
    prompt_id: str,
    call: _ModelFn,
    out_dir: Path,
    quality: int,
    model_resolution: tuple[int, int] | None,
    limit: int,
    max_examples: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Teacher-forced pass over a builder chat.jsonl; writes ``records.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    turns: list[TurnScore] = []
    n_records = 0
    with chat_path.open() as chat_f, (out_dir / "records.jsonl").open("w") as out_f:
        for index, line in enumerate(chat_f):
            if not line.strip():
                continue
            if limit and n_records >= limit:
                break
            record = json.loads(line)
            scores = score_record(
                record, index=index, arm=arm, prompt=prompt, prompt_id=prompt_id,
                call=call, quality=quality, model_resolution=model_resolution,
            )
            n_records += 1
            turns.extend(scores)
            task_id, template_id, category = record_meta(record, index)
            out_f.write(json.dumps({
                "record_index": index, "task_id": task_id, "template_id": template_id,
                "category": category, "n_turns": len(scores),
                "n_exact": sum(t.exact for t in scores),
                "n_parse_ok": sum(t.parse_ok for t in scores),
                "turns": [asdict(t) for t in scores],
            }) + "\n")
            out_f.flush()
            _LOGGER.info(
                "record %d (%s) | %d/%d exact | %d/%d parse-valid",
                index, task_id, sum(t.exact for t in scores), len(scores),
                sum(t.parse_ok for t in scores), len(scores),
            )
    if not turns:
        raise ValueError(f"no assistant turns scored from {chat_path}")
    scores, extra = aggregate_offline(turns, n_records=n_records, max_examples=max_examples)
    return scores, extra, len(turns)


def run_closed_loop(
    *,
    task_ids: Sequence[str],
    splits: dict[str, Any],
    arm: str,
    system_prompt: str,
    call: _ModelFn,
    out_dir: Path,
    attempts: int,
    max_steps: int,
    settle: sr.Settle,
    model_resolution: tuple[int, int] | None,
    jpeg_quality: int,
    save_frames: bool,
    in_vm: Callable[..., Any],
    setup: Callable[..., dict[str, Any]],
    verify: Callable[..., dict[str, Any]],
    verify_timeout_s: float,
    recordings_root: Path | None,
    greedy: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[int | None]]]:
    """Closed-loop pass over the tasks, one fresh VM per attempt; writes ``tasks.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[Episode] = []
    seeds: dict[str, list[int | None]] = {}
    with (out_dir / "tasks.jsonl").open("w") as out_f:
        for index, task_id in enumerate(task_ids):
            task = task_of(task_id)
            tier = sb.split_of(task_id, splits)
            recording = load_recording(recordings_root, task_id)
            check_recording(recording, task)
            seeds[task_id] = []
            for attempt in range(attempts):
                seed = None if greedy else attempt_seed(task_id, attempt)
                seeds[task_id].append(seed)
                attempt_dir = out_dir / task_id / f"attempt_{attempt:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                _LOGGER.info("[%d/%d] %s attempt %d/%d", index + 1, len(task_ids),
                             task_id, attempt + 1, attempts)
                episode, _ = in_vm(
                    lambda client, task=task, tier=tier, attempt=attempt, seed=seed,
                    attempt_dir=attempt_dir, recording=recording: run_episode(
                        task=task, tier=tier, arm=arm, attempt=attempt, seed=seed,
                        client=client, call=call, system_prompt=system_prompt,
                        out_dir=attempt_dir, max_steps=max_steps, settle=settle,
                        model_resolution=model_resolution, jpeg_quality=jpeg_quality,
                        save_frames=save_frames,
                        setup=setup, verify=verify, verify_timeout_s=verify_timeout_s,
                        recording=recording,
                    ),
                    label=f"{task_id} attempt {attempt}",
                    log_path=attempt_dir / "qemu.log",
                )
                episodes.append(episode)
                out_f.write(json.dumps(asdict(episode)) + "\n")
                out_f.flush()
    if not episodes:
        raise ValueError("no episodes ran")
    scores, extra = aggregate_closed(episodes, attempts=attempts)
    return scores, extra, seeds


def build_params(
    args: argparse.Namespace,
    *,
    arm: str,
    prompt_id: str,
    prompt: str,
    sampling: SamplingParams,
    sampling_source: dict[str, str],
    model_resolution: tuple[int, int] | None,
    settle: sr.Settle,
    splits_sha: str | None,
    splits_source: str,
    seeds: dict[str, Any],
) -> dict[str, Any]:
    """The run's fully-resolved parameters, for the result.json audit trail."""
    return {
        "mode": args.mode,
        "arm": arm,
        "model_path": args.model_path,
        "model": args.model or args.model_path,
        "system_prompt_id": prompt_id,
        "system_prompt_sha256": sha256_text(prompt),
        "sampling": sampling.to_dict(),
        "sampling_source": sampling_source,
        "seeds": seeds,
        "attempts": args.attempts,
        "max_steps": args.max_steps,
        "model_resolution": list(model_resolution) if model_resolution else None,
        "jpeg_quality": args.jpeg_quality,
        "k_images": sg.K_IMAGES,
        "keep_images": sg.KEEP_IMAGES,
        "split": args.split,
        "subset": args.subset,
        "splits_sha256": splits_sha,
        "splits_source": splits_source,
        "recipe": sb.RECIPE,
        "goal_prefix": sb.GOAL_PREFIX,
        "settle": asdict(settle),
        "verify_timeout_s": args.verify_timeout_s,
        "context_length": args.context_length,
    }


class _StubModel:
    """A scripted stand-in for the served model: reply per assistant turn already in context."""

    def __init__(self, replies: Sequence[str]) -> None:
        if not replies:
            raise ValueError("stub model needs at least one reply")
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, messages: list[dict[str, Any]], *, seed: int | None = None,
    ) -> tuple[str, str | None]:
        turn = sum(1 for m in messages if m["role"] == "assistant")
        self.calls.append({"turn": turn, "seed": seed, "n_messages": len(messages)})
        return self.replies[min(turn, len(self.replies) - 1)], "stop"


class _StubClient:
    """An in-process stand-in for ``OSWorldClient``: no VM, no network, distinct frames."""

    def __init__(self, *, screen_wh: tuple[int, int], frame_wh: tuple[int, int]) -> None:
        self.screen_wh = screen_wh
        self.frame_wh = frame_wh
        self.commands: list[str] = []
        self.dispatched: list[OrderedAction] = []
        self.cursor = (0, 0)
        self.shots = 0

    def wait_ready(self, **_kw: Any) -> None:
        return None

    def screen_size(self) -> tuple[int, int]:
        return self.screen_wh

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def execute(self, command: str) -> None:
        self.commands.append(command)
        if command.startswith("pyautogui.moveTo("):
            x, y = command[len("pyautogui.moveTo("):-1].split(", ")
            self.cursor = (int(x), int(y))

    def screenshot(self) -> Image.Image:
        self.shots += 1
        level = (self.shots * 37) % 256
        return Image.new("RGB", self.frame_wh, (level, level, level))

    def screenshot_settled(self, **_kw: Any) -> Image.Image:
        return self.screenshot()

    def dispatch_ordered_action(self, action: OrderedAction) -> StepResult:
        self.dispatched.append(action)
        return StepResult(
            cursor_before=(0, 0), cursor_after=(0, 0), intended_target=(0, 0),
            delta=(0, 0), scroll=0, events_dispatched=["stub"], parse_ok=True,
            action_text="stub",
        )

    def run_command(self, command: list[str] | str, *, shell: bool = False) -> dict:
        return {"status": "success", "returncode": 0, "output": "", "command": command,
                "shell": shell}


_SELF_CHECK_TASK_ID = "fx_click_button__s00"
_SELF_CHECK_CLICKS = {
    sg.ARM_REL: "move(120,-40); down(LMB); up(LMB)",
    sg.ARM_ABS: "move_to(520,470); down(LMB); up(LMB)",
}
_SELF_CHECK_COMMIT = "down(Return); up(Return)"
_SELF_CHECK_PROSE = "Sure, I will confirm now."
_SELF_CHECK_JUNK = "I am not sure what to do here."
_SELF_CHECK_FRAME_WH = (96, 54)
_SELF_CHECK_MODEL_WH = (64, 36)


def _expect(failures: list[str], label: str, got: Any, want: Any) -> None:
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def _self_check_record(
    *, arm: str, prompt: str, frames_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """A synthetic 3-turn builder record (frame refs on disk) plus its golden lines."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    task = task_of(_SELF_CHECK_TASK_ID)
    actions = [_SELF_CHECK_CLICKS[arm], _SELF_CHECK_COMMIT, sg.TERMINATE_LINE]
    parts: list[dict[str, Any]] = []
    for i in range(len(actions)):
        path = frames_dir / f"step_{i:03d}.png"
        Image.new("RGB", _SELF_CHECK_FRAME_WH, (16 * (i + 1),) * 3).save(path)
        parts.append({"type": "image", "url": str(path)})
    messages = keep_text_messages(prompt, sb.GOAL_PREFIX + task.instruction, parts, actions)
    return {"messages": messages, "task_id": task.task_id, "arm": arm}, actions


def _self_check_offline(out_dir: Path, *, arm: str, prompt_id: str) -> list[str]:
    """The offline_exact plumbing end to end on a stub model: 2 of 3 turns exact."""
    prompt = SYSTEM_PROMPTS[prompt_id]
    out_dir.mkdir(parents=True, exist_ok=True)
    record, actions = _self_check_record(arm=arm, prompt=prompt, frames_dir=out_dir / "frames")
    chat_path = out_dir / "chat.jsonl"
    chat_path.write_text(json.dumps(record) + "\n")
    model = _StubModel([actions[0], _SELF_CHECK_JUNK, sg.TERMINATE_LINE])
    scores, extra, n_turns = run_offline(
        chat_path=chat_path, arm=arm, prompt=prompt, prompt_id=prompt_id, call=model,
        out_dir=out_dir, quality=sb.DEFAULT_JPEG_QUALITY,
        model_resolution=_SELF_CHECK_MODEL_WH, limit=0, max_examples=DEFAULT_MAX_EXAMPLES,
    )
    write_result(
        out_dir / "result.json", task=f"shortgoal_{MODE_OFFLINE}", scores=scores,
        params={"arm": arm, "system_prompt_id": prompt_id, "self_check": True},
        inputs={"chat": str(chat_path)}, n_samples=n_turns, elapsed_s=0, extra=extra,
    )
    failures: list[str] = []
    _expect(failures, f"offline[{arm}] n_turns", n_turns, 3)
    _expect(failures, f"offline[{arm}] exact_line_rate", round(scores["exact_line_rate"], 6),
            round(2 / 3, 6))
    _expect(failures, f"offline[{arm}] parse_valid_rate",
            round(scores["parse_valid_rate"], 6), round(2 / 3, 6))
    _expect(failures, f"offline[{arm}] turn indices",
            sorted(scores["by_turn_index"]), ["0", "1", "2"])
    _expect(failures, f"offline[{arm}] template",
            sorted(scores["by_template"]), ["fx_click_button"])
    _expect(failures, f"offline[{arm}] worst examples", len(extra["worst_examples"]), 1)
    _expect(failures, f"offline[{arm}] prompt reached the model",
            model.calls[0]["n_messages"], 2)
    _expect(failures, f"offline[{arm}] confusion has an unparsed pair",
            any(pair.endswith(f"->{KIND_UNPARSED}") for pair in extra["kind_confusion"]), True)
    _expect(failures, f"offline[{arm}] result.json", (out_dir / "result.json").is_file(), True)
    return failures


def _self_check_closed(out_dir: Path, *, arm: str, prompt_id: str) -> list[str]:
    """The closed_loop plumbing end to end on a stub model and stub VM client."""
    prompt = SYSTEM_PROMPTS[prompt_id]
    out_dir.mkdir(parents=True, exist_ok=True)
    replies = [
        _SELF_CHECK_CLICKS[arm],
        f"{_SELF_CHECK_PROSE}\n{_SELF_CHECK_COMMIT}",
        _SELF_CHECK_JUNK,
        sg.TERMINATE_LINE,
    ]
    model = _StubModel(replies)
    client = _StubClient(screen_wh=sgt.SCREEN_WH, frame_wh=_SELF_CHECK_FRAME_WH)
    setup_calls: list[str] = []

    def setup(_client: Any, task: sgt.ConcreteTask, screen_wh: tuple[int, int]) -> dict[str, Any]:
        setup_calls.append(task.task_id)
        return {"setup_id": task.setup_id, "screen_wh": list(screen_wh)}

    def verify(
        _client: Any, task: sgt.ConcreteTask, setup_state: dict[str, Any], *, timeout_s: float,
    ) -> dict[str, Any]:
        return {"kind": task.verifier_id, "passed": True,
                "detail": {"setup_id": setup_state["setup_id"], "timeout_s": timeout_s}}

    scores, extra, seeds = run_closed_loop(
        task_ids=[_SELF_CHECK_TASK_ID], splits=sgt.build_split_manifest(), arm=arm,
        system_prompt=prompt, call=model, out_dir=out_dir, attempts=2,
        max_steps=DEFAULT_MAX_STEPS,
        settle=sr.Settle(delay_s=0.0, stable_timeout_s=0.0, poll_s=0.0),
        model_resolution=_SELF_CHECK_MODEL_WH, jpeg_quality=sb.DEFAULT_JPEG_QUALITY,
        save_frames=True,
        in_vm=lambda action, **_kw: action(client),
        setup=setup, verify=verify, verify_timeout_s=1.0,
        recordings_root=None, greedy=False,
    )
    write_result(
        out_dir / "result.json", task=f"shortgoal_{MODE_CLOSED}", scores=scores,
        params={"arm": arm, "system_prompt_id": prompt_id, "seeds": seeds, "self_check": True},
        inputs={"task_ids": [_SELF_CHECK_TASK_ID]}, n_samples=2, elapsed_s=0, extra=extra,
    )
    episode = json.loads((out_dir / "tasks.jsonl").read_text().splitlines()[0])
    failures: list[str] = []
    _expect(failures, f"closed[{arm}] pass_at_1", scores["pass_at_1"], 1.0)
    _expect(failures, f"closed[{arm}] pass_at_k", scores["pass_at_k"], 1.0)
    _expect(failures, f"closed[{arm}] mean_steps", scores["mean_steps"], 4.0)
    _expect(failures, f"closed[{arm}] stop_reason", episode["stop_reason"], _STOP_TERMINATE)
    _expect(failures, f"closed[{arm}] strict_ok", episode["strict_ok"], 2)
    _expect(failures, f"closed[{arm}] tolerant_rescue", episode["tolerant_rescue"], 1)
    _expect(failures, f"closed[{arm}] failed_steps", episode["failed_steps"], 1)
    _expect(failures, f"closed[{arm}] blind_history_steps", episode["blind_history_steps"], 0)
    _expect(failures, f"closed[{arm}] spurious_terminate", episode["spurious_terminate"], False)
    _expect(failures, f"closed[{arm}] never_terminate", episode["never_terminate"], False)
    _expect(failures, f"closed[{arm}] setup ran per attempt", setup_calls,
            [_SELF_CHECK_TASK_ID] * 2)
    _expect(failures, f"closed[{arm}] dispatched actions", len(client.dispatched), 4)
    _expect(failures, f"closed[{arm}] tier came from the split manifest",
            episode["tier"], "train")
    _expect(failures, f"closed[{arm}] seeds are keyed by task and attempt",
            seeds[_SELF_CHECK_TASK_ID],
            [attempt_seed(_SELF_CHECK_TASK_ID, 0), attempt_seed(_SELF_CHECK_TASK_ID, 1)])
    cx, cy = sr.cursor_start_px(_SELF_CHECK_TASK_ID, sgt.SCREEN_WH)
    _expect(failures, f"closed[{arm}] cursor start is seeded",
            client.commands[0], f"pyautogui.moveTo({cx}, {cy})")
    steps_dir = out_dir / _SELF_CHECK_TASK_ID / "attempt_00" / "steps"
    _expect(failures, f"closed[{arm}] step pngs", len(list(steps_dir.glob("step_*.png"))), 4)
    _expect(failures, f"closed[{arm}] prompt sidecars",
            len(list(steps_dir.glob("prompt_*.json"))), 4)
    _expect(failures, f"closed[{arm}] gif",
            (out_dir / _SELF_CHECK_TASK_ID / "attempt_00" / "rollout.gif").is_file(), True)
    conv = (out_dir / _SELF_CHECK_TASK_ID / "attempt_00" / "conversation.jsonl").read_text()
    _expect(failures, f"closed[{arm}] conversation lines", len(conv.splitlines()), 4)
    return failures


def self_check(out_dir: Path) -> int:
    """Both modes, both arms, no network: returns 0 when every expectation holds."""
    root = out_dir / "self_check"
    failures: list[str] = []
    for arm in sg.ARMS:
        prompt_id = sg.PROMPT_IDS[arm]
        failures += _self_check_offline(root / f"offline_{arm}", arm=arm, prompt_id=prompt_id)
        failures += _self_check_closed(root / f"closed_{arm}", arm=arm, prompt_id=prompt_id)
    for failure in failures:
        _LOGGER.error("self_check FAIL %s", failure)
    _LOGGER.info("self_check %s (%d checks failed); artifacts under %s",
                 "PASS" if not failures else "FAIL", len(failures), root)
    return 0 if not failures else 1


def _add_cli(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--mode", choices=MODES, default=None,
                   help="offline_exact = teacher-forced byte-exactness (no VM); "
                        "closed_loop = boot the VM and run the model to the verifier. "
                        "Required unless --self_check (which runs both).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--arm", choices=sg.ARMS, default=None,
                   help="Action format under test. Implies --system_prompt_id.")
    p.add_argument("--system_prompt_id", default=None,
                   help="Registered prompt id. Implies --arm; a mismatch is an error.")
    p.add_argument("--chat", default=None,
                   help="offline_exact: the chat.jsonl written by shortgoal_build.")
    p.add_argument("--splits", default=None,
                   help="splits.json from shortgoal_templates.build_split_manifest. "
                        "Omitted = recomputed from the catalog (shortgoal_build.load_splits).")
    p.add_argument("--split", choices=sgt.SPLIT_NAMES, default=None,
                   help="closed_loop: run every task id of this split.")
    p.add_argument("--subset", choices=sb.SUBSETS, default=None,
                   help="closed_loop: run the builder's subset of the ladder rung "
                        "(overfit1/overfit32/full/tiera_val/tierb_val); wins over --split.")
    p.add_argument("--task_ids", default=None,
                   help="closed_loop: explicit task ids (comma/space separated); wins "
                        "over --subset and --split.")
    p.add_argument("--recordings_root", default=None,
                   help="Root of the golden recordings; <root>/<task_id>/recording.json "
                        "supplies setup params and the recorded cursor start.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only the first N records / tasks (smoke runs).")
    p.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS,
                   help="closed_loop attempts per task. 1 decodes greedily (pass@1); >1 "
                        "samples the Qwen tuple with seed sha256(task_id:attempt) for pass@k.")
    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS,
                   help="closed_loop step cap; steps past trained length are failure territory.")
    p.add_argument("--max_examples", type=int, default=DEFAULT_MAX_EXAMPLES,
                   help="offline_exact: how many worst (non-exact) turns to keep in result.json.")
    p.add_argument("--model_resolution", default=DEFAULT_MODEL_RESOLUTION,
                   help='"WxH" view served to the model (the builder\'s frame scale). '
                        "Empty = leave frames at native size.")
    p.add_argument("--jpeg_quality", type=int, default=sb.DEFAULT_JPEG_QUALITY,
                   help="JPEG quality of the data URLs sent to the model (the builder's).")
    p.add_argument("--no_frames", action="store_true",
                   help="Skip per-step PNGs, prompt sidecars and the rollout gif.")
    p.add_argument("--self_check", action="store_true",
                   help="Run both modes and both arms against a stub model and stub VM "
                        "client (no network, no GPU) and exit.")
    p.add_argument("--sglang_url", default=None,
                   help="An already-serving OpenAI-compatible base url (…/v1). Without it "
                        "sglang is spawned from --model_path.")
    p.add_argument("--model_path", default=None,
                   help="Checkpoint to serve (required unless --sglang_url is given).")
    p.add_argument("--model", default=None,
                   help="Served model name sent in the request. Defaults to --model_path.")
    p.add_argument("--sglang_port", type=int, default=30000)
    p.add_argument("--sglang_api_key", default="osworld")
    p.add_argument("--mem_fraction_static", type=float, default=0.40)
    p.add_argument("--context_length", type=int, default=DEFAULT_CONTEXT_LENGTH,
                   help="sglang context length; keep >=16k so a long keep-text prompt "
                        "cannot crash the server.")
    p.add_argument("--request_timeout_s", type=float, default=180.0)
    p.add_argument("--settle_s", type=float, default=sr.Settle().delay_s)
    p.add_argument("--settle_stable_timeout_s", type=float, default=sr.Settle().stable_timeout_s)
    p.add_argument("--settle_poll_s", type=float, default=sr.Settle().poll_s)
    p.add_argument("--verify_timeout_s", type=float, default=12.0,
                   help="How long the recorder's verifier may retry before failing the task.")
    p.add_argument("--vm_ready_timeout_s", type=float, default=300.0)
    p.add_argument("--vm_port", type=int, default=5000)
    p.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    p.add_argument("--qemu_bin", default=_DEFAULT_QEMU_BIN)
    sampling_mod.add_sampling_cli(p, default_max_tokens=256)
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _add_cli(argparse.ArgumentParser()).parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(output_dir / "shortgoal_eval.log"))

    if args.self_check:
        return self_check(output_dir)
    if args.mode is None:
        print(f"--mode is required (one of {MODES}), or pass --self_check", file=sys.stderr)
        return 2
    if not isinstance(args.attempts, int) or not 1 <= args.attempts <= MAX_ATTEMPTS:
        print(f"--attempts must be 1..{MAX_ATTEMPTS}, got {args.attempts}", file=sys.stderr)
        return 2
    try:
        arm, prompt_id = resolve_arm(args.arm, args.system_prompt_id)
        model_resolution = parse_model_resolution(args.model_resolution)
    except (KeyError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
    prompt = SYSTEM_PROMPTS[prompt_id]

    greedy = args.greedy or args.mode == MODE_OFFLINE or args.attempts == 1
    sampling = replace(
        sampling_mod.from_cli(args, model_path=args.model_path, system_prompt=prompt),
        greedy=greedy,
    )
    sampling_source = sampling_mod.source_map(args, sampling)
    settle = sr.Settle(
        delay_s=args.settle_s,
        stable_timeout_s=args.settle_stable_timeout_s,
        poll_s=args.settle_poll_s,
    )
    model_name = args.model or args.model_path
    if not model_name:
        print("pass --model_path (to serve) or --model (name of an external server)",
              file=sys.stderr)
        return 2
    if args.sglang_url is None and not args.model_path:
        print("pass --sglang_url or --model_path", file=sys.stderr)
        return 2

    try:
        splits, splits_source = sb.load_splits(args.splits)
        task_ids = (
            resolve_task_ids(
                task_ids=args.task_ids, split=args.split, subset=args.subset,
                splits=splits, limit=args.limit,
            )
            if args.mode == MODE_CLOSED
            else []
        )
    except (OSError, KeyError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.mode == MODE_OFFLINE and not args.chat:
        print("--mode offline_exact needs --chat <chat.jsonl>", file=sys.stderr)
        return 2
    splits_sha = sha256_file(Path(args.splits)) if args.splits else None

    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
    vm_port, vnc_port = sr._ports(args.vm_port)
    sglang_port = (30000 + job_mod) if args.sglang_port == 30000 else args.sglang_port

    procs: list[subprocess.Popen] = []

    def cleanup() -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    atexit.register(cleanup)
    sr._register_cleanup()

    sglang_url = args.sglang_url
    if sglang_url is None:
        _LOGGER.info("starting sglang for %s on port %d", args.model_path, sglang_port)
        sglang_proc = spawn_sglang(
            model_path=args.model_path, port=sglang_port, api_key=args.sglang_api_key,
            mem_fraction_static=args.mem_fraction_static, context_length=args.context_length,
            log_path=output_dir / "sglang.log",
        )
        procs.append(sglang_proc)
        _wait_for(f"http://localhost:{sglang_port}/health_generate",
                  headers={"Authorization": f"Bearer {args.sglang_api_key}"},
                  proc=sglang_proc, poll_s=10, max_polls=120, label="sglang")
        sglang_url = f"http://localhost:{sglang_port}/v1"
    _LOGGER.info(
        "mode=%s arm=%s prompt=%s model=%s sglang=%s sampling=%s (source %s)",
        args.mode, arm, prompt_id, model_name, sglang_url, sampling.to_dict(), sampling_source,
    )

    call = sglang_caller(
        sglang_url=sglang_url, api_key=args.sglang_api_key, model=model_name,
        sampling=sampling, request_timeout_s=args.request_timeout_s,
    )
    t_start = time.time()
    seeds: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "chat": args.chat, "splits": args.splits, "recordings_root": args.recordings_root,
        "sglang_url": sglang_url,
    }

    if args.mode == MODE_OFFLINE:
        scores, extra, n_samples = run_offline(
            chat_path=Path(args.chat), arm=arm, prompt=prompt, prompt_id=prompt_id,
            call=call, out_dir=output_dir, quality=args.jpeg_quality,
            model_resolution=model_resolution, limit=args.limit,
            max_examples=args.max_examples,
        )
        inputs["records_path"] = str(output_dir / "records.jsonl")
    else:

        def in_vm(action: Callable[[OSWorldClient], Any], *, label: str, log_path: Path) -> Any:
            return sr._in_fresh_vm(
                action, label=label, log_path=log_path, qemu_bin=args.qemu_bin,
                qcow2=args.qcow2, vm_port=vm_port, vnc_port=vnc_port,
                ready_timeout_s=args.vm_ready_timeout_s,
            )

        scores, extra, seeds = run_closed_loop(
            task_ids=task_ids, splits=splits, arm=arm, system_prompt=prompt, call=call,
            out_dir=output_dir, attempts=args.attempts, max_steps=args.max_steps,
            settle=settle, model_resolution=model_resolution,
            jpeg_quality=args.jpeg_quality,
            save_frames=not args.no_frames, in_vm=in_vm,
            setup=sr.prepare_task, verify=sr.verify_task,
            verify_timeout_s=args.verify_timeout_s,
            recordings_root=Path(args.recordings_root) if args.recordings_root else None,
            greedy=greedy,
        )
        inputs["task_ids"] = list(task_ids)
        inputs["tasks_path"] = str(output_dir / "tasks.jsonl")
        n_samples = len(task_ids) * args.attempts

    write_result(
        output_dir / "result.json",
        task=f"shortgoal_{args.mode}",
        scores=scores,
        params=build_params(
            args, arm=arm, prompt_id=prompt_id, prompt=prompt, sampling=sampling,
            sampling_source=sampling_source, model_resolution=model_resolution,
            settle=settle, splits_sha=splits_sha, splits_source=splits_source, seeds=seeds,
        ),
        inputs=inputs,
        n_samples=n_samples,
        elapsed_s=int(time.time() - t_start),
        extra=extra,
    )
    _LOGGER.info("done. %s -> %s", args.mode, output_dir / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
