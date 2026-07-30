"""OSWorld freeroll — single entry point for labctl freeroll recipes.

Boots the VM via native qemu+KVM, starts sglang, runs a closed-loop
rollout, writes result.json for labctl.

Replaces the former run_freeroll_recipe.sh + rollout.py split.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

# Eval primitives now live in juergen/eval/ (see osworld_grounding_runner.py
# for the rationale: the shared modules are pre-existing crowd-cast BC
# infrastructure, not experiment-specific glue). Add juergen/eval to
# sys.path so freeroll runs from its slurm/dev/franz/.../crowd-cast-bc/
# location can still pull these in. Labctl recipes that point at this
# script already use `uv run --project=/fast/.../juergen/eval`, so the
# venv has the same Python; only the import path needs hinting.
_JUERGEN_EVAL = str(Path(__file__).resolve().parent)
if _JUERGEN_EVAL not in sys.path:
    sys.path.insert(0, _JUERGEN_EVAL)

from action_parser import (  # noqa: E402
    OrderedAction, parse_action_tolerant, parse_computer_use_action_tolerant,
    parse_computer_use_tool_call, parse_ordered_action_tolerant,
)
from osworld_vm_client import OSWorldClient  # noqa: E402
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402
from osworld_runtime import (  # noqa: E402
    _DEFAULT_QCOW2, _DEFAULT_QEMU_BIN, _EVAL_DIR,
    _call_model, _pil_to_data_url, _wait_for, append_turn,
    build_loggable_messages, window_frame_labels,
)
import sampling as sampling_mod  # noqa: E402
from sampling import SamplingParams  # noqa: E402

_LOGGER = logging.getLogger(__name__)
_TERMINATE = "TERMINATE"

# Action-line grammars (names mirror the data pipeline's formatter names in
# data_pipeline/realigned_pipeline/lib/action_format.py). "canonical" is the
# legacy `<dx> <dy> <scroll> [; +KEY -KEY]` aggregate format (which also
# tolerates computer_use tool calls); the ordered formats are `; `-joined
# primitives — ordered_events_v3 adds type("...") on top of v2 and both are
# handled by the same parser (v2 lines are a strict subset of v3).
# "computer_use_rel_v1" is the Qwen-native tool-call format with a RELATIVE
# mouse (cua_v4_thinking contract): one or more <tool_call>{json}</tool_call>
# blocks executed strictly in order; terminate is a tool call, not a
# TERMINATE line.
# "computer_use_rel_norm_v1" is the same grammar with mouse_move_rel deltas on
# a per-axis 0-1000 scale (1000 == one full screen width for dx, height for
# dy) instead of raw device counts; it is denormalized to VM pixels at
# dispatch. Everything else about it — parsing, terminate, wait — is identical.
_ACTION_FORMATS = (
    "canonical", "ordered_events_v2", "ordered_events_v3", "computer_use_rel_v1",
    "computer_use_rel_norm_v1",
)
_ORDERED_FORMATS = frozenset({"ordered_events_v2", "ordered_events_v3"})
_NATIVE_FORMAT = "computer_use_rel_v1"
_NATIVE_NORM_FORMAT = "computer_use_rel_norm_v1"
# Formats parsed by the Qwen-native tool-call grammar.
_NATIVE_FORMATS = frozenset({_NATIVE_FORMAT, _NATIVE_NORM_FORMAT})
# Formats whose move deltas are a 0-1000 screen fraction and must be scaled to
# VM pixels before dispatch.
_NORMALIZED_FORMATS = frozenset({_NATIVE_NORM_FORMAT})
_NORMALIZED_SCALE = 1000.0
# Default action format per system prompt; anything unlisted is canonical.
_PROMPT_ACTION_FORMATS = {
    "cua_v3_thinking": "ordered_events_v3",
    "cua_v4_thinking": _NATIVE_FORMAT,
    "cua_v4_thinking_v2": _NATIVE_FORMAT,
    "cua_v4_thinking_norm": _NATIVE_NORM_FORMAT,
    "cua_oev2_thinking": "ordered_events_v2",
}
# Prompts whose training data conditions on "GOAL: {goal}" as the first
# user-turn text (stage_04t goal conditioning).
_GOAL_CONDITIONED_PROMPT_IDS = frozenset(
    {"cua_v3_thinking", "cua_v4_thinking", "cua_v4_thinking_v2",
     "cua_v4_thinking_norm", "cua_oev2_thinking"})


def _resolve_action_format(explicit: str | None, system_prompt_id: str) -> str:
    """--action_format wins; otherwise infer from the system prompt id."""
    if explicit:
        if explicit not in _ACTION_FORMATS:
            raise ValueError(
                f"unknown action format {explicit!r} (available: {_ACTION_FORMATS})"
            )
        return explicit
    return _PROMPT_ACTION_FORMATS.get(system_prompt_id, "canonical")


def _instruction_text(instruction: str | None, *, goal_conditioned: bool) -> str | None:
    """The first-user-turn text for the goal.

    Goal-conditioned prompts (cua_v3_thinking) were trained with the first
    user turn reading exactly ``GOAL: {goal_text}`` (stage 04t's
    goal_memory_block; no ``So far:`` at episode start — freeroll episodes
    begin with empty memory). Every other prompt passes the instruction
    through verbatim.
    """
    if instruction is None:
        return None
    return f"GOAL: {instruction}" if goal_conditioned else instruction


def _instruction_for_step(
    instr_text: str | None, step: int, persist_instruction: bool
) -> str | None:
    """Which instruction text rides this step's earliest in-window user turn.

    With ``persist_instruction`` the SAME formatted text (incl. the GOAL
    prefix) is re-anchored every step; otherwise it appears on step 1 only.
    """
    return instr_text if (step == 1 or persist_instruction) else None


@dataclass
class StepLog:
    step: int
    action_text: str
    parsed: dict | None
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    intended_target: tuple[int, int] | None
    events_dispatched: list[str]
    parse_error: str | None
    elapsed_s: float


# _pil_to_data_url / _call_model / _wait_for moved to
# juergen/eval/osworld_runtime.py; imported above.


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_think(text: str) -> str:
    """Drop the model's thought, leaving only the action content.

    Handles all three shapes the serving stack can produce:
    - Response starts with ``<think>``: strip through the FIRST matching
      ``</think>`` (plus following whitespace/newline). The verbatim chat
      template injects nothing, so thinking-SFT models emit the opener
      themselves.
    - Dangling ``</think>`` with no opener (legacy checkpoints whose
      template injected ``<think>`` into the prompt): strip through the
      first ``</think>`` likewise.
    - An UNTERMINATED ``<think>`` (no closer — e.g. truncated mid-thought):
      the whole response is thought; there is no action content, so return
      "" and let the parser reject it loudly.
    - No think markers: return the text unchanged.
    """
    s = text.lstrip()
    if s.startswith(_THINK_OPEN):
        end = s.find(_THINK_CLOSE)
        if end == -1:
            return ""
        return s[end + len(_THINK_CLOSE):].lstrip()
    end = s.find(_THINK_CLOSE)
    if end != -1 and _THINK_OPEN not in s[:end]:
        return s[end + len(_THINK_CLOSE):].lstrip()
    return text


def _is_terminate(text: str) -> bool:
    # After think-stripping: first line == TERMINATE (classic), or last line
    # == TERMINATE (thinking models are trained with the terminal token
    # appended after the final action line). A preceding thought never hides
    # the token — "<think>done</think>\nTERMINATE" terminates.
    lines = [ln.strip() for ln in _strip_think(text).splitlines() if ln.strip()]
    return bool(lines) and (lines[0] == _TERMINATE or lines[-1] == _TERMINATE)


def _dispatch_plan(action_text: str, finish_reason: str | None) -> tuple[str | None, str | None]:
    """Truncation guard + think-strip, ahead of terminate detection and parsing.

    Returns ``(clean_text, error)``. When ``finish_reason == "length"`` the
    reply was cut at max_tokens: ``clean_text`` is None and ``error`` set —
    the step is recorded as a parse error and NOTHING is dispatched (a
    half-emitted press would leave a key held down and flood the VM with OS
    key-repeat). Otherwise ``clean_text`` is the think-stripped response.
    """
    if finish_reason == "length":
        return None, (
            "response truncated at max_tokens (finish_reason='length'); "
            f"nothing dispatched: {action_text!r}"
        )
    return _strip_think(action_text), None


def _scale_ordered_moves(action: OrderedAction, ax: float, ay: float) -> OrderedAction:
    """Scale move() deltas by an explicit per-axis factor.

    Only move() carries cursor deltas — scroll ticks and key/type primitives
    are unaffected.

    No caller applies a factor today: every trained action format emits move
    deltas straight from the crowd-cast keylog, which records raw device counts
    (pre-acceleration evdev/CGEvent HID deltas), NOT pixels of the frame the
    model was shown. Re-extracting a segment at a different frame height yields
    byte-identical deltas, so no frame-derived factor belongs here. This exists
    for a normalized action format, whose denormalization is a genuine per-axis
    scale: a 0-1000 screen-fraction delta becomes pixels via sw/1000, sh/1000.
    """
    return OrderedAction(
        primitives=tuple(
            replace(p, dx=round(p.dx * ax), dy=round(p.dy * ay))
            if p.kind == "move" else p
            for p in action.primitives
        ),
        no_op=action.no_op,
    )


def _computer_use_terminate_status(text: str) -> str | None:
    """Return the computer_use terminate status, or None for non-terminate."""
    try:
        call = parse_computer_use_tool_call(text)
    except (TypeError, ValueError):
        return None
    if str(call.arguments.get("action", "")).strip().lower() != "terminate":
        return None
    return str(call.arguments.get("status", "success")).strip().lower() or "success"


def _computer_use_rel_terminate_status(text: str) -> str | None:
    """computer_use_rel_v1 terminate detection: a parsed terminate tool call
    ANYWHERE in the reply stops the rollout. Returns its status, or None
    when the reply parses to no terminate call (or doesn't parse at all —
    that's the dispatch path's parse error to report, not a terminate)."""
    try:
        action = parse_computer_use_action_tolerant(text)
    except (TypeError, ValueError):
        return None
    for p in action.primitives:
        if p.kind == "terminate":
            return p.status
    return None


def _is_left_click(parsed: dict | None) -> bool:
    if not parsed:
        return False
    computer_use_action = str(
        parsed.get("computer_use", {}).get("action", "")
    ).strip().lower()
    if computer_use_action in {
        "left_click",
        "double_click",
        "triple_click",
        "left_click_drag",
    }:
        return True
    if any(
        (p.get("kind") == "down" and p.get("name") == "LMB")
        or (p.get("kind") in ("click", "button_down") and p.get("name") == "left")
        for p in parsed.get("primitives", [])
    ):
        return True
    return any(
        e["what"] == "LMB" and e["kind"] == "press"
        for e in parsed.get("events", [])
    )


def _prepare_desktop(client: OSWorldClient, setup: str) -> None:
    """Put the freshly booted VM into a recipe-selected start state."""
    if setup == "none":
        return
    if setup == "terminal":
        _LOGGER.info("desktop setup: opening terminal")
        client.execute(
            "import subprocess; "
            "subprocess.Popen(['bash', '-lc', "
            "\"(command -v gnome-terminal >/dev/null && gnome-terminal) || "
            "(command -v xfce4-terminal >/dev/null && xfce4-terminal) || "
            "(command -v xterm >/dev/null && xterm)\"]); "
            "time.sleep(2.0); "
            "pyautogui.hotkey('ctrl', 'l'); "
            "time.sleep(0.2)"
        )
        return
    raise ValueError(f"unknown desktop setup: {setup!r}")


def _run_rollout(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    osworld_url: str,
    output_dir: Path,
    max_steps: int,
    instruction: str | None,
    system_prompt: str,
    system_prompt_id: str,
    action_format: str,
    n_history_frames: int,
    persist_instruction: bool,
    sampling: SamplingParams,
    sampling_source: dict[str, str],
    save_frames: bool,
    stop_on_click: bool,
    desktop_setup: str,
    settle_s: float,
    settle_stable_timeout_s: float,
    settle_poll_s: float,
    model_resolution: tuple[int, int] | None = None,
) -> dict:
    steps_dir = output_dir / "steps"
    if save_frames:
        steps_dir.mkdir(exist_ok=True)
    traj_path = output_dir / "trajectory.jsonl"
    conv_path = output_dir / "conversation.jsonl"
    gif_path = output_dir / "rollout.gif"

    client = OSWorldClient(osworld_url)
    client.wait_ready()
    sw, sh = client.screen_size()
    _LOGGER.info("VM screen %dx%d; max_steps=%d; instruction=%r", sw, sh, max_steps, instruction)
    _prepare_desktop(client, desktop_setup)

    # Goal-conditioned prompts (cua_v3_thinking) train with the first user
    # turn reading exactly "GOAL: {goal}"; persist_instruction re-anchors the
    # SAME formatted text on the earliest in-window user turn every step.
    instr_text = _instruction_text(
        instruction,
        goal_conditioned=system_prompt_id in _GOAL_CONDITIONED_PROMPT_IDS,
    )
    use_ordered = action_format in _ORDERED_FORMATS
    use_native = action_format in _NATIVE_FORMATS

    # The ONLY delta scaling in the dispatch path: a 0-1000 screen fraction
    # back to VM pixels. Depends on the VM screen, never on the model view.
    denorm = ((sw / _NORMALIZED_SCALE, sh / _NORMALIZED_SCALE)
              if action_format in _NORMALIZED_FORMATS else None)
    if denorm:
        _LOGGER.info("%s: denormalizing move deltas x%.4f/x%.4f to VM %dx%d",
                     action_format, denorm[0], denorm[1], sw, sh)

    # The model view: frames are resized to model_resolution before they enter
    # the conversation (matching the training frame scale). Move deltas are NOT
    # rescaled with them — a raw format's deltas are device counts invariant to
    # the frame scale, and a normalized format's are a fraction of the VM
    # screen. GIF/saved frames stay native.
    mres = model_resolution if (model_resolution and tuple(model_resolution) != (sw, sh)) else None
    if mres:
        _LOGGER.info("model view %dx%d; VM %dx%d; move deltas not scaled by "
                     "the view ratio", mres[0], mres[1], sw, sh)

    def _to_model(img: Image.Image) -> Image.Image:
        return img.resize(mres, Image.LANCZOS) if mres else img

    frame = client.screenshot()
    frames_for_gif: list[Image.Image] = [frame.copy()]
    if save_frames:
        frame.save(steps_dir / "step_000.png")
    recent_frames: list[Image.Image] = [_to_model(frame)]
    recent_actions: list[str] = []

    t_start = time.time()
    steps: list[StepLog] = []
    stop_reason = "max_steps"
    parse_errors = 0

    with traj_path.open("w") as traj_f, conv_path.open("w") as conv_f:
        traj_f.write(json.dumps({
            "step_num": 0, "action": "<reset>", "response": "<reset>",
            "reward": 0.0, "done": False, "info": {},
        }) + "\n")
        traj_f.flush()

        for step in range(1, max_steps + 1):
            t0 = time.time()
            instr_used = _instruction_for_step(instr_text, step, persist_instruction)
            # The exact interleaved message list sent to the model this step,
            # with each frame replaced by a <image step_NNN.png> placeholder
            # (see build_loggable_messages). Built once and reused for both the
            # per-step prompt sidecar and the conversation.jsonl transcript.
            frame_labels = window_frame_labels(step, len(recent_frames))
            loggable_messages = build_loggable_messages(
                system_prompt=system_prompt, instruction=instr_used,
                recent_actions=recent_actions,
                frame_labels=frame_labels,
            )
            if save_frames:
                (steps_dir / f"prompt_{step:03d}.json").write_text(
                    json.dumps(loggable_messages, indent=2))
            try:
                action_text, finish_reason = _call_model(
                    sglang_url=sglang_url, api_key=api_key, model=model,
                    system_prompt=system_prompt,
                    instruction=instr_used,
                    recent_frames=recent_frames,
                    recent_actions=recent_actions,
                    sampling=sampling,
                )
            except Exception as e:
                _LOGGER.error("step %d: model call failed: %s", step, e)
                stop_reason = "model_error"
                break

            # Log the full conversation as sent (prompt + this step's reply),
            # unconditionally — independent of --no_frames. Covers the terminate,
            # normal-action, and stop-on-click paths below with a single write.
            conv_f.write(json.dumps({
                "step": step,
                "messages": loggable_messages,
                "response": action_text,
                "finish_reason": finish_reason,
            }) + "\n")
            conv_f.flush()

            # Surface the model's per-step reply in stdout (the .lab log), keyed
            # by step and the current (latest in-window) frame it acted on.
            _LOGGER.info(
                "step %d | current frame %s | finish_reason=%s | response=%r",
                step, frame_labels[-1], finish_reason, action_text,
            )

            # Truncation guard + think-strip. A truncated reply is never
            # terminate-checked or dispatched (recorded as a parse error below).
            clean_text, trunc_err = _dispatch_plan(action_text, finish_reason)

            # Terminate detection. computer_use_rel_v1 terminates on a parsed
            # terminate TOOL CALL anywhere in the reply — never on a TERMINATE
            # line; the line detection stays for every other format.
            if clean_text is None:
                computer_use_status = None
            elif use_native:
                computer_use_status = _computer_use_rel_terminate_status(clean_text)
            else:
                computer_use_status = _computer_use_terminate_status(clean_text)
            if trunc_err is None and (
                (not use_native and _is_terminate(action_text))
                or computer_use_status is not None
            ):
                if computer_use_status and computer_use_status != "success":
                    stop_reason = f"terminate_{computer_use_status}"
                else:
                    stop_reason = "terminate"
                _LOGGER.info("step %d: model emitted terminate", step)
                try:
                    cursor = client.cursor_position()
                    frame = client.screenshot()
                except Exception as e:
                    _LOGGER.error("step %d: terminate snapshot failed: %s", step, e)
                    stop_reason = "screenshot_error"
                    break

                frames_for_gif.append(frame.copy())
                if save_frames:
                    frame.save(steps_dir / f"step_{step:03d}.png")
                append_turn(recent_frames, recent_actions, _to_model(frame), action_text,
                            n_history_frames=n_history_frames)

                step_log = StepLog(
                    step=step,
                    action_text=action_text,
                    parsed={
                        "terminate": True,
                        "computer_use_status": computer_use_status,
                    },
                    cursor_before=cursor,
                    cursor_after=cursor,
                    intended_target=cursor,
                    events_dispatched=[],
                    parse_error=None,
                    elapsed_s=time.time() - t0,
                )
                steps.append(step_log)
                traj_f.write(json.dumps({
                    "step_num": step, "action": action_text, "response": action_text,
                    "reward": 0.0, "done": True, "info": asdict(step_log),
                }) + "\n")
                traj_f.flush()
                break

            parse_err: str | None = None
            parsed = None
            sr_dict: dict | None = None
            if trunc_err is not None:
                parse_err = trunc_err
                parse_errors += 1
                _LOGGER.warning("step %d: %s", step, trunc_err)
            else:
                try:
                    sr = None
                    if use_native or use_ordered:
                        if use_native:
                            action = parse_computer_use_action_tolerant(clean_text)
                            # parsed/logs keep the model's own frame of reference.
                            parsed = {
                                "no_op": all(
                                    p.kind == "wait" for p in action.primitives
                                ),
                                "primitives": [
                                    {
                                        "kind": p.kind, "dx": p.dx, "dy": p.dy,
                                        "name": p.name,
                                        "keys": list(p.keys) if p.keys else None,
                                        "count": p.count, "text": p.text,
                                        "status": p.status,
                                    }
                                    for p in action.primitives
                                ],
                            }
                        else:
                            action = parse_ordered_action_tolerant(clean_text)
                            # parsed/logs keep the model's own frame of reference.
                            parsed = {
                                "no_op": action.no_op,
                                "primitives": [
                                    {
                                        "kind": p.kind, "dx": p.dx, "dy": p.dy,
                                        "name": p.name, "text": p.text,
                                    }
                                    for p in action.primitives
                                ],
                            }
                        if denorm:
                            action = _scale_ordered_moves(action, *denorm)
                        sr = client.dispatch_ordered_action(action)
                    else:
                        try:
                            computer_call = parse_computer_use_tool_call(clean_text)
                        except (TypeError, ValueError):
                            computer_call = None
                        if computer_call is not None:
                            parsed = {
                                "computer_use": computer_call.arguments,
                                "no_op": str(
                                    computer_call.arguments.get("action", "")
                                ).strip().lower() == "answer",
                            }
                            sr = client.dispatch_computer_use(computer_call.arguments)
                        else:
                            action = parse_action_tolerant(clean_text)
                            parsed = {
                                "dx": action.dx, "dy": action.dy,
                                "scroll": action.scroll, "no_op": action.no_op,
                                "events": [
                                    {
                                        "kind": e.kind,
                                        "what": e.what,
                                        "mouse_button": e.mouse_button,
                                    }
                                    for e in action.events
                                ],
                            }
                            sr = client.dispatch_action(action)
                    sr_dict = {
                        "cursor_before": list(sr.cursor_before),
                        "cursor_after": list(sr.cursor_after),
                        "intended_target": list(sr.intended_target),
                        "events_dispatched": sr.events_dispatched,
                    }
                except (TypeError, ValueError) as e:
                    parse_err = str(e)
                    parse_errors += 1
                    _LOGGER.warning("step %d: parse error %s on %r", step, e, action_text)

            try:
                frame = client.screenshot_settled(
                    min_delay_s=settle_s,
                    stability_timeout_s=settle_stable_timeout_s,
                    poll_s=settle_poll_s,
                )
            except Exception as e:
                _LOGGER.error("step %d: screenshot failed: %s", step, e)
                stop_reason = "screenshot_error"
                break

            frames_for_gif.append(frame.copy())
            if save_frames:
                frame.save(steps_dir / f"step_{step:03d}.png")
            append_turn(recent_frames, recent_actions, _to_model(frame), action_text,
                        n_history_frames=n_history_frames)

            step_log = StepLog(
                step=step, action_text=action_text, parsed=parsed,
                cursor_before=tuple(sr_dict["cursor_before"]) if sr_dict else (-1, -1),
                cursor_after=tuple(sr_dict["cursor_after"]) if sr_dict else (-1, -1),
                intended_target=tuple(sr_dict["intended_target"]) if sr_dict else None,
                events_dispatched=sr_dict["events_dispatched"] if sr_dict else [],
                parse_error=parse_err, elapsed_s=time.time() - t0,
            )
            steps.append(step_log)
            traj_f.write(json.dumps({
                "step_num": step, "action": action_text, "response": action_text,
                "reward": 0.0, "done": False, "info": asdict(step_log),
            }) + "\n")
            traj_f.flush()

            if stop_on_click and parsed and not parsed["no_op"] and _is_left_click(parsed):
                stop_reason = "click"
                _LOGGER.info("step %d: model clicked, stopping", step)
                break

    elapsed_s = time.time() - t_start

    if len(frames_for_gif) > 1:
        small = [f.resize((min(960, f.width), int(f.height * min(960, f.width) / f.width)))
                 for f in frames_for_gif]
        small[0].save(gif_path, save_all=True, append_images=small[1:],
                      duration=300, loop=0, optimize=True)
        _LOGGER.info("wrote GIF %s (%d frames)", gif_path, len(small))

    return {
        "schema_version": 1,
        "model": model,
        "osworld_url": osworld_url,
        "sglang_url": sglang_url,
        "screen_size": [sw, sh],
        "n_steps": len(steps),
        "max_steps": max_steps,
        "stop_reason": stop_reason,
        "parse_errors": parse_errors,
        "click": stop_reason == "click",
        "elapsed_s": elapsed_s,
        "instruction": instruction,
        "system_prompt": system_prompt,
        "system_prompt_id": system_prompt_id,
        "action_format": action_format,
        "n_history_frames": n_history_frames,
        "persist_instruction": persist_instruction,
        "sampling": sampling.to_dict(),
        # per-field provenance: 'flag' | 'qwen:<mode>' | 'greedy' (see
        # sampling.source_map) — tells an explicit override from a regime default
        "sampling_source": sampling_source,
        # back-compat scalar keys for existing result.json readers
        "max_tokens": sampling.max_tokens,
        "temperature": 0.0 if sampling.greedy else sampling.temperature,
        "desktop_setup": desktop_setup,
        "settle_s": settle_s,
        "settle_stable_timeout_s": settle_stable_timeout_s,
        "settle_poll_s": settle_poll_s,
        "traj_path": str(traj_path),
        "conversation_path": str(conv_path),
        "gif_path": str(gif_path),
    }


def _parse_instructions(raw: str | None) -> list[str | None]:
    """Split a --instruction value into one-or-more instructions.

    Instructions are newline-separated (use a TOML multiline string to pass
    several). Blank lines and ``#``-comment lines are dropped. A single-line
    value yields a one-element list, so the legacy single-instruction usage
    is unchanged. An empty/absent value yields ``[None]`` — one rollout with
    no natural-language goal.
    """
    if raw is None:
        return [None]
    out: list[str | None] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out or [None]


def _slug(text: str | None, idx: int) -> str:
    """Filesystem-safe per-instruction subdir name, prefixed by index."""
    base = re.sub(r"[^a-z0-9]+", "-", (text or "no-instruction").lower()).strip("-")
    return f"task_{idx:02d}_{base[:40] or 'task'}"


def _boot_vm(
    *, qemu_bin: str, qcow2: str, vm_port: int, vnc_port: int, log_path: Path,
) -> subprocess.Popen:
    """Boot the OSWorld qcow2 via native qemu+KVM with a throwaway snapshot.

    ``snapshot=on`` means every boot starts from the same clean disk image and
    discards writes on shutdown — so rebooting between instructions gives each
    one a pristine desktop without mutating the shared qcow2.
    """
    return subprocess.Popen(
        [qemu_bin,
         "-enable-kvm", "-cpu", "host", "-smp", "4", "-m", "4G",
         "-machine", "type=q35,accel=kvm",
         "-drive", f"file={qcow2},if=virtio,format=qcow2,snapshot=on",
         "-netdev", f"user,id=net0,hostfwd=tcp::{vm_port}-:5000,hostfwd=tcp::{vnc_port}-:5900",
         "-device", "virtio-net-pci,netdev=net0",
         "-display", "none", "-nographic"],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )


def _terminate(proc: subprocess.Popen, *, label: str) -> None:
    """Stop a child process and wait for it to release its ports."""
    if proc.poll() is not None:
        return
    _LOGGER.info("terminating %s (pid=%d)", label, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument(
        "--instruction", default=None,
        help="Natural-language goal. Pass several newline-separated "
             "instructions (TOML multiline string) to run them sequentially, "
             "each on a freshly rebooted VM but the same sglang server.",
    )
    p.add_argument("--system_prompt_id", default="training_v1")
    p.add_argument(
        "--action_format", choices=_ACTION_FORMATS, default=None,
        help="Action-line grammar the model emits (names mirror the data "
             "pipeline's formatter names). Default: inferred from "
             "--system_prompt_id (cua_v3_thinking -> ordered_events_v3, "
             "cua_v4_thinking -> computer_use_rel_v1, everything else -> "
             "canonical).",
    )
    # Sampling flags (--temperature/--top_p/--top_k/--repetition_penalty/
    # --presence_penalty/--max_tokens/--sampling_mode/--greedy). Default to the
    # Qwen-recommended tuple for the detected regime — NOT greedy. max_tokens
    # defaults to 256 (was 64, which truncated native tool-calls).
    sampling_mod.add_sampling_cli(p, default_max_tokens=256)
    p.add_argument("--n_history_frames", type=int, default=16)
    p.add_argument(
        "--persist_instruction", action=argparse.BooleanOptionalAction, default=True,
        help="Re-anchor the natural-language goal on the earliest in-window user "
             "turn every step so it stays in context after the first frame is "
             "evicted. On by default; --no-persist_instruction reverts to "
             "goal-on-step-1 (the training/legacy behaviour).",
    )
    p.add_argument("--no_frames", action="store_true")
    p.add_argument("--stop_on_click", action="store_true")
    p.add_argument(
        "--settle_s", type=float, default=0.3,
        help="Fixed delay (seconds) after dispatching an action before the "
             "post-action screenshot, giving the UI time to repaint. 0 keeps "
             "the legacy zero-wait behaviour.",
    )
    p.add_argument(
        "--settle_stable_timeout_s", type=float, default=2.0,
        help="If >0, after --settle_s poll the framebuffer (every "
             "--settle_poll_s) until two consecutive frames are identical or "
             "this timeout elapses, then use the last frame. Adapts the wait "
             "to how long the UI actually takes to settle.",
    )
    p.add_argument(
        "--settle_poll_s", type=float, default=0.1,
        help="Poll interval (seconds) for --settle_stable_timeout_s.",
    )
    p.add_argument(
        "--desktop_setup",
        choices=("none", "terminal"),
        default="none",
        help="Optional VM setup before the first model turn. Use 'terminal' "
             "for typing evals that should start with a focused terminal.",
    )
    p.add_argument(
        "--model_resolution", default="",
        help='"WxH" view served to the model (frames resized, dx/dy scaled '
             'back to VM pixels). Empty = native VM resolution.')
    p.add_argument("--sglang_port", type=int, default=30000)
    p.add_argument("--sglang_api_key", default="osworld")
    p.add_argument("--mem_fraction_static", type=float, default=0.40)
    p.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    p.add_argument("--qemu_bin", default=_DEFAULT_QEMU_BIN)
    args = p.parse_args()

    if args.system_prompt_id not in SYSTEM_PROMPTS:
        print(f"Unknown --system_prompt_id {args.system_prompt_id!r}. "
              f"Available: {list(SYSTEM_PROMPTS)}", file=sys.stderr)
        return 1
    action_format = _resolve_action_format(args.action_format, args.system_prompt_id)

    # Resolve the Qwen-recommended sampling tuple once (Instruct vs Thinking is
    # auto-detected from the checkpoint / system prompt unless --sampling_mode
    # forces it). Constant across instructions, so build it before the loop.
    sampling = sampling_mod.from_cli(
        args,
        model_path=args.model_path,
        system_prompt=SYSTEM_PROMPTS[args.system_prompt_id],
    )
    sampling_source = sampling_mod.source_map(args, sampling)
    _LOGGER.info("sampling: %s (source: %s)", sampling.to_dict(), sampling_source)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger().addHandler(
        logging.FileHandler(output_dir / "freeroll.log")
    )

    instructions = _parse_instructions(args.instruction)
    mres: tuple[int, int] | None = None
    if args.model_resolution:
        try:
            w_s, h_s = args.model_resolution.lower().split("x")
            mres = (int(w_s), int(h_s))
        except ValueError:
            print(f"bad --model_resolution {args.model_resolution!r} (want WxH)", file=sys.stderr)
            return 2

    # Port isolation: shift from SLURM_JOB_ID so concurrent jobs on the same
    # node don't collide. Range: 5000 + (job_id % 200) * 10 → 5000..6990.
    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
    vm_port = 5000 + job_mod
    vnc_port = 5900 + job_mod
    sglang_port = (30000 + job_mod) if args.sglang_port == 30000 else args.sglang_port

    _LOGGER.info("model=%s n_instructions=%d max_steps=%d vm_port=%d sglang_port=%d "
                 "system_prompt_id=%s action_format=%s",
                 args.model_path, len(instructions), args.max_steps,
                 vm_port, sglang_port, args.system_prompt_id, action_format)
    _LOGGER.info("desktop_setup=%s", args.desktop_setup)

    # Cleanup: terminate VM and sglang on exit.
    _procs: list[subprocess.Popen] = []

    def _cleanup() -> None:
        _LOGGER.info("cleanup...")
        for proc in _procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))

    # Start sglang ONCE and keep it alive across every instruction — loading
    # the model onto the GPU is the most expensive step and is fully
    # instruction-independent. Only the VM is rebooted per instruction (below).
    _LOGGER.info("starting sglang...")
    sglang_proc = subprocess.Popen(
        ["uv", "run", "--project", str(_EVAL_DIR), "python", "-m", "sglang.launch_server",
         "--model-path", args.model_path,
         "--host", "0.0.0.0",
         "--port", str(sglang_port),
         "--api-key", args.sglang_api_key,
         "--mem-fraction-static", str(args.mem_fraction_static),
         "--chunked-prefill-size", "2048"],
        cwd=str(_EVAL_DIR),
        stdout=open(output_dir / "sglang.log", "w"),
        stderr=subprocess.STDOUT,
    )
    _procs.append(sglang_proc)
    _LOGGER.info("sglang pid=%d", sglang_proc.pid)

    runs: list[dict] = []
    for idx, instruction in enumerate(instructions):
        slug = _slug(instruction, idx)
        run_dir = output_dir / slug if len(instructions) > 1 else output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("=== instruction %d/%d (%s): %r ===",
                     idx + 1, len(instructions), slug, instruction)

        # Reboot the VM for a clean desktop. snapshot=on discards the prior
        # instruction's writes, so each run starts from the pristine image.
        _LOGGER.info("starting VM...")
        vm_proc = _boot_vm(
            qemu_bin=args.qemu_bin, qcow2=args.qcow2,
            vm_port=vm_port, vnc_port=vnc_port,
            log_path=run_dir / "qemu.log",
        )
        _procs.append(vm_proc)
        _LOGGER.info("qemu pid=%d", vm_proc.pid)

        try:
            # Wait for VM (up to 5 min).
            _wait_for(f"http://localhost:{vm_port}/screenshot",
                      proc=vm_proc, poll_s=5, max_polls=60, label="VM")

            # Wait for sglang only on the first iteration; afterwards it is
            # already serving (the first VM boot overlaps the model load).
            if idx == 0:
                _wait_for(f"http://localhost:{sglang_port}/health_generate",
                          headers={"Authorization": f"Bearer {args.sglang_api_key}"},
                          proc=sglang_proc, poll_s=10, max_polls=120, label="sglang")

            result = _run_rollout(
                sglang_url=f"http://localhost:{sglang_port}/v1",
                api_key=args.sglang_api_key,
                model=args.model_path,
                osworld_url=f"http://localhost:{vm_port}",
                output_dir=run_dir,
                max_steps=args.max_steps,
                instruction=instruction,
                system_prompt=SYSTEM_PROMPTS[args.system_prompt_id],
                system_prompt_id=args.system_prompt_id,
                action_format=action_format,
                n_history_frames=args.n_history_frames,
                persist_instruction=args.persist_instruction,
                sampling=sampling,
                sampling_source=sampling_source,
                save_frames=not args.no_frames,
                stop_on_click=args.stop_on_click,
                desktop_setup=args.desktop_setup,
                model_resolution=mres,
                settle_s=args.settle_s,
                settle_stable_timeout_s=args.settle_stable_timeout_s,
                settle_poll_s=args.settle_poll_s,
            )
            with (run_dir / "result.json").open("w") as f:
                json.dump(result, f, indent=2)
            runs.append({
                "index": idx, "slug": slug, "instruction": instruction,
                "subdir": slug if len(instructions) > 1 else ".",
                "stop_reason": result["stop_reason"], "n_steps": result["n_steps"],
                "click": result["click"], "parse_errors": result["parse_errors"],
                "desktop_setup": result["desktop_setup"],
            })
        finally:
            _terminate(vm_proc, label="VM")
            if vm_proc in _procs:
                _procs.remove(vm_proc)

    # With multiple instructions each run owns a subdir; write a top-level
    # result.json so labctl's `marker = "result.json"` resolves and the
    # aggregate is browsable. For a single instruction the per-run write above
    # already populated output_dir/result.json directly.
    if len(instructions) > 1:
        with (output_dir / "result.json").open("w") as f:
            json.dump({
                "schema_version": 1,
                "model": args.model_path,
                "system_prompt_id": args.system_prompt_id,
                "action_format": action_format,
                "desktop_setup": args.desktop_setup,
                "n_instructions": len(instructions),
                "runs": runs,
            }, f, indent=2)

    _LOGGER.info("done. %d instruction(s); outputs under %s", len(runs), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
