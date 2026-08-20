#!/usr/bin/env python3
"""One converter: recorded rollouts -> training records in any target grammar.

Replaces the five ``convert_abs_to_{absolute,relative,moverel,diffabs,deltatype}.py``
scripts (1,389 LOC). Rollout walking, task-success / slug filtering, per-step frame
pairing, parse-error accounting, terminate handling, message assembly, the
deterministic split and the manifest live here once; the encoding is the target
grammar's job, selected with ``--codec``:

    codec  = grammars.load("deltatype_v2")     # grammars/deltatype_v2/
    action = codec.action_from_operations(ops, geometry=geom, cursor=cursor_before)
    text   = codec.format(action)              # the assistant training target
    prompt = codec.describe()                  # the matching system prompt

The seam between this converter and a grammar is the absolute-Operation
vocabulary documented in ``grammars/_support.py`` (``move_to``, ``glide_to``,
``mouse_down/up``, ``scroll``, ``key_down/up``, ``coalesced_type``, ``wait`` — every
coordinate an absolute, clamped screen pixel). It is the same seam
``Codec.compile_action`` uses in the other direction: a source reader resolves one
recorded step into Operations, and the target codec lifts Operations into its own
Action in its own coordinate convention. No coordinate space is named here — the
convention is a property of the grammar you pick, so there is no ``--coord_space``
flag.

Two producers write the rollout layout this reads, and the artifact DECLARES which
(see :func:`source_reader`). Our own harness records the grammar it ran, so a
rollout carrying ``result.json["codec"]`` is read through :func:`rollout_step`,
which takes the Operation stream the harness dispatched; the external
absolute-teacher collections carry no such field and go through
:func:`teacher_step`. There is still exactly one action vocabulary in the estate —
Operations — and no grammar is asked to emit ``computer_use``.

The geometry the teacher lift needs comes from the freeroll log, not from the
model's text: ``cursor_before`` (the real VM cursor before the action) and
``intended_target`` (the absolute pixel the teacher's coordinate resolved to,
post-scale, post-clip). Because ``cursor_before[t] == intended_target[t-1]``, a
relative codec's diff telescopes exactly — the cursor motion is identical to the
teacher's, only the encoding changes.

Prose is preserved by default (``--keep_prose``, on unless ``--no_keep_prose``).
Measured over ``/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/
onpolicy_distill/converted/*/_normalized/*/chat.jsonl`` (2026-08-05), all three
arms built from ``rollouts/teacher_8b_osworld_train_v1``:

    osworld_train_absolute        10721 / 10721 assistant turns prose-bearing
    osworld_train_deltatype_raw       0 / 11337
    osworld_train_diffabs             0 / 11102

Under a uniform loss mask the relative arms carried zero reasoning supervision
while the absolute control carried full coverage, which invalidates every trained
absolute-vs-relative comparison built from them. So prose is applied uniformly to
every codec, the thinking system prompt is selected whenever prose is kept, prose
coverage is a manifest field, and the run aborts at zero coverage
(``_assert_prose_coverage``).

The zero-abort is a floor, not a comparison: ``100% vs 5%`` passes it. A second
check measures the asymmetry between arms directly (``prose_divergence``).
``--codec`` is single-valued, so one invocation is one arm and sibling arms of a
sweep are sibling ``--out_dir``s, each with a ``convert_manifest.json``. After
building, this arm's prose coverage is compared against every sibling manifest
under ``--out_dir``'s parent that records the same rollout selection, and a
relative divergence above ``--prose_divergence_tol`` is reported
(``--prose_divergence warn``, the default; ``abort`` refuses to write, ``off``
skips) and recorded in the manifest under ``prose_divergence``.

The gated statistic is coverage over the pre-codec turn population,
``n_turns_with_prose / (n_turns + n_turns_dropped_by_codec)``: that denominator is
codec-invariant in practice (measured identical for all seven codecs: 734 on
``teacher_8b_v1``, 1053 on ``v2``, 2011 on ``v3``) while raw ``prose_turn_frac``
is not — ``diffabs`` cannot spell ``type()``, so on ``teacher_8b_v1`` it drops 74
of 733 turns, none of them prose-bearing, and its raw frac reads 9.96% above
``deltatype_v2``'s from expressiveness alone. Both divergences are recorded; only
the normalized one gates.

Gaps neither check closes. (1) The divergence check is order-dependent: it only
sees arms that have already finished, so the first arm of a sweep is compared
against nothing and an absent or mid-build sibling is skipped, never aborts — run
the last arm with ``--prose_divergence abort`` or diff the manifests when the
sweep is done. (2) It cannot tell a stale sibling manifest from a current one.
(3) A failure that zeroes prose in every arm diverges nowhere and is caught only
by the zero-abort. (4) The normalization forgives the supervision-density
difference the trainer sees (``diffabs`` gets ~10% more prose per turn than
``deltatype_v2`` on ``teacher_8b_v1``); that shows only in the ungated raw figure.
(5) At small prose counts the statistic is coarse: on ``teacher_8b_v1``'s 32 prose
turns one lost turn is 3.1% and passes, two is 6.25% and fires. So compare
``n_turns``, ``n_turns_with_prose`` and ``n_turns_dropped_by_codec`` across arms
before believing any format comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``grammars/`` is a sibling of this file at the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import grammars  # noqa: E402
from grammars import _support  # noqa: E402

# Duration used when lowering a teacher drag into a timed stroke.
DRAG_SECONDS = 0.5

_CLICK_BUTTON = {
    "left_click": "left",
    "right_click": "right",
    "middle_click": "middle",
    "double_click": "left",
    "triple_click": "left",
}
_CLICK_N = {"double_click": 2, "triple_click": 3}
_COORD_ACTIONS = frozenset(
    {
        "mouse_move",
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "left_click_drag",
        "mouse_down",
    }
)


@dataclass(frozen=True)
class Step:
    """One source step at the converter's seam, before any target grammar is applied.

    ``operations`` is the absolute-Operation vocabulary documented in
    ``grammars/_support.py`` — the one action vocabulary the estate has, and what
    every source reader below resolves to. ``None`` means the SOURCE could not
    express the step at all (an off-grammar teacher action), which is a dropped
    turn rather than a lift that raises.

    ``terminate`` is ``None`` for a normal turn, else ``"success"`` / ``"failure"``.
    It stays a status here because every grammar spells termination differently (a
    ``terminate{status}`` tool call vs a bare ``TERMINATE`` / ``FAIL`` token), and
    which one the record gets is the target codec's decision, not this file's.

    ``screen`` has no default. Every relative grammar clamps against it and
    ``move_rel`` normalizes against it, so an invented size is a silently wrong
    label, not a missing one.
    """

    screen: tuple[int, int]
    operations: tuple[Any, ...] | None = ()
    cursor_before: tuple[int, int] | None = None
    terminate: str | None = None


def teacher_operations(
    action: dict[str, Any],
    *,
    target: tuple[int, int] | None,
    cursor: tuple[int, int] | None,
) -> tuple[Any, ...] | None:
    """Lower one teacher ``computer_use`` call into absolute Operations.

    ``None`` if off-grammar. Coordinates come from ``target``
    (``info["intended_target"]``), which is already the post-scale, post-clip
    absolute pixel the VM acted on — so this never re-resolves a coordinate
    convention, it only re-expresses an action the VM already executed. The call's
    own ``coordinate`` is NOT a pixel: every collection in scope was sampled on a
    normalized grid (``result.json["coord_grid"] == 1000``).
    """
    a = action
    kind = str(a.get("action", "")).strip().lower()
    ops: list[Any] = []

    if kind == "hscroll":
        amount = _scroll_amount(a)
        if amount is None:
            return None
        return (_support.scroll(amount, 0),)

    if kind in _COORD_ACTIONS and target is not None and kind != "left_click_drag":
        ops.append(_support.move_to(target))

    if kind in _CLICK_BUTTON:
        button = _CLICK_BUTTON[kind]
        for _ in range(_CLICK_N.get(kind, 1)):
            ops.append(_support.mouse_down(button))
            ops.append(_support.mouse_up(button))
    elif kind == "left_click_drag":
        # A press at the current cursor, a timed stroke to the target, a release.
        # The two bare-token originals degraded this to a stationary `+LMB -LMB`,
        # losing the stroke; routing through Operations keeps it, and each codec
        # renders as much of it as its grammar can express.
        if cursor is not None:
            ops.append(_support.move_to(cursor))
        ops.append(_support.mouse_down("left"))
        if target is not None:
            ops.append(_support.glide_to(target, DRAG_SECONDS))
        ops.append(_support.mouse_up("left"))
    elif kind == "mouse_move":
        if target is None:
            return None
    elif kind == "mouse_down":
        ops.append(_support.mouse_down(str(a.get("button", "left")).strip().lower()))
    elif kind == "mouse_up":
        ops.append(_support.mouse_up(str(a.get("button", "left")).strip().lower()))
    elif kind in {"key", "key_down", "key_up"}:
        keys = a.get("keys", a.get("key"))
        names = [str(k) for k in keys] if isinstance(keys, list) else [str(keys)]
        try:
            names = [_support.normalize_key(n) for n in names if n and n != "None"]
        except ValueError:
            return None
        if not names:
            return None
        if kind in {"key", "key_down"}:
            ops += [_support.key_down(n) for n in names]
        if kind in {"key", "key_up"}:
            ops += [_support.key_up(n) for n in reversed(names)]
    elif kind == "type":
        text = str(a.get("text") or "")
        if not text:
            return None
        ops.append(_support.coalesced_type(text))
    elif kind == "scroll":
        amount = _scroll_amount(a)
        if amount is None:
            return None
        if str(a.get("scroll_direction") or "").strip().lower() == "down" and amount > 0:
            amount = -amount
        ops.append(_support.scroll(0, amount))
    elif kind == "wait":
        ops.append(_support.wait(_number(a.get("time", 1)) or 1.0))
    else:
        return None  # off-grammar (incl. `answer`, which has no action channel)

    return tuple(ops)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


#: The keys the teacher collections in scope have used for a scroll magnitude.
#: The teacher is an external, unversioned producer, so this reads all three —
#: but only these three, and a step carrying none of them is off-grammar rather
#: than a zero scroll. A zero scroll lowers to `scroll(0, 0)`, which every
#: bare-token grammar renders as its idle line: a real scroll silently labelled
#: as doing nothing.
_SCROLL_KEYS = ("pixels", "scroll_amount", "amount")


def _scroll_amount(action: dict[str, Any]) -> int | None:
    for key in _SCROLL_KEYS:
        if key in action:
            try:
                return int(round(float(action[key])))
            except (TypeError, ValueError):
                return None
    return None


def _geometry(screen: tuple[int, int]):
    """A ``DisplayGeometry`` for the rollout's screen size.

    Field names are ``desktop_*`` (``desktop.geometry.DisplayGeometry`` is a
    verbatim copy of Harbor's dataclass). The window is set to the full desktop:
    freeroll's ``intended_target`` is already a desktop-space pixel, so there is
    no inner window to offset against.

    ``grammars/_support.screen_size`` reads ``geometry.desktop_width`` /
    ``geometry.desktop_height``, so it consumes this dataclass as-is.
    """
    from desktop.geometry import DisplayGeometry  # noqa: PLC0415

    width, height = screen
    return DisplayGeometry(
        desktop_width=int(width),
        desktop_height=int(height),
        window_width=int(width),
        window_height=int(height),
    )


def _lift(codec: Any, step: Step) -> Any:
    """Absolute Operations -> this grammar's Action (the inverse of compile_action).

    The codec owns its coordinate convention and how it spells termination, so the
    same call takes ``step.terminate`` (``None`` for a normal turn, else
    ``"success"`` / ``"failure"``). A terminal turn is spelled two ways across the
    seven grammars — a flag on the Action
    (``deltatype_v2.fail``, ``diffabs.terminate``, ``ordered_events_v3.terminate``)
    or a ``terminate`` call inside it (``native_absolute``, ``move_rel``) — while
    ``compact_raw`` and ``compact_absolute`` cannot express one and raise.
    The converter must not reconstruct any of this by introspecting Action
    dataclasses.

    A grammar that has not implemented this raises here, naming the method it
    must add, rather than emitting a differently-encoded dataset.
    """
    lift = getattr(codec, "action_from_operations", None)
    if not callable(lift):
        raise NotImplementedError(
            f"grammar codec {type(codec).__name__} has no action_from_operations. "
            "Training-target construction needs the inverse of Codec.compile_action:\n\n"
            "    def action_from_operations(self, operations, *, geometry, cursor, "
            "terminate=None) -> Action\n\n"
            "taking the absolute-Operation vocabulary documented in grammars/_support.py "
            "and resolving it into this grammar's own coordinate convention and its own "
            "terminal spelling. It cannot live in the converter: the converter must not "
            "know a coordinate space, and must not know how each grammar encodes "
            "TERMINATE / FAIL."
        )
    return lift(
        step.operations,
        geometry=_geometry(step.screen),
        cursor=step.cursor_before or (0, 0),
        terminate=step.terminate,
    )


def format_step(codec: Any, step: Step) -> str | None:
    """One :class:`Step` -> this grammar's assistant text (``None`` = drop the turn)."""
    if step.operations is None:
        return None
    try:
        return codec.format(_lift(codec, step))
    except ValueError:
        # An expressiveness ceiling, and the only thing a dropped turn may mean.
        # Every codec's error type subclasses ValueError; an AssertionError or a
        # KeyError out of a lift is a broken codec invariant, and swallowing one
        # here booked a bug as a per-arm `n_turns_dropped_by_codec`.
        return None


def teacher_step(info: dict[str, Any], screen: tuple[int, int]) -> Step | None:
    """One step of an external absolute-teacher collection -> the seam.

    ``info["parsed"]`` is the teacher's own payload: the absolute ``computer_use``
    arguments (private ``_``-prefixed keys stripped), or a ``terminate`` flag with
    ``computer_use_status`` beside it. ``None`` when the row does not speak that
    vocabulary at all, which is a source parse error, not a codec drop — including
    when there is no payload, which is why the caller need not pre-check one.
    """
    parsed = info.get("parsed")
    if not parsed:
        return None
    if parsed.get("terminate"):
        status = str(parsed.get("computer_use_status") or "success").strip().lower()
        return Step(screen=screen, terminate=status)
    arguments = parsed.get("computer_use")
    if not isinstance(arguments, dict):
        return None
    cursor = _pair(info.get("cursor_before"))
    return Step(
        screen=screen,
        operations=teacher_operations(
            {k: v for k, v in arguments.items() if not str(k).startswith("_")},
            target=_pair(info.get("intended_target")),
            cursor=cursor,
        ),
        cursor_before=cursor,
    )


#: The harness's grammar-independent control verdict -> the lift's terminate status.
#: `no_op` is deliberately absent: idling is not a termination, and it arrives as
#: the empty operation stream every target grammar already spells its own way.
_TERMINAL_CONTROL = {"terminate": "success", "fail": "failure"}


def rollout_step(info: dict[str, Any], screen: tuple[int, int]) -> Step | None:
    """One step of one of OUR OWN rollouts -> the seam.

    The harness records the absolute Operation stream it dispatched, so the seam is
    read back rather than re-derived. That is the point: ``info["parsed"]`` is the
    SOURCE grammar's own ``to_dict()`` — seven different vocabularies, none of them
    the teacher's ``computer_use`` — while Operations are the one vocabulary every
    grammar resolves to, and they are what the guest actually executed. Reading
    them needs no source codec, so a dataset built from an old rollout does not
    change meaning when that grammar is next edited.

    ``info["control"]`` is the grammar-independent terminal verdict the harness
    already resolved (``agent.agent._control_of``), so termination is not recovered
    by introspecting an Action dataclass either.
    """
    operations = info.get("operations")
    if not isinstance(operations, list):
        return None
    return Step(
        screen=screen,
        # `Operation.from_dict` raises on a malformed row rather than shrinking the
        # dataset by one turn: a corrupt artifact is not an expressiveness ceiling.
        operations=tuple(_support.Operation.from_dict(item) for item in operations),
        cursor_before=_pair(info.get("cursor_before")),
        terminate=_TERMINAL_CONTROL.get(str(info.get("control") or "")),
    )


def source_stop_reason(result: dict[str, Any]) -> str | None:
    """Why the source rollout ended, in the producer's own spelling.

    Keyed on the same declared field :func:`source_reader` is: our harness
    publishes `outcome` (`evals/harness.py`'s result mirror) and the external
    teacher collections publish `stop_reason`. Reading only one of the two spellings
    left this null for every rollout of the other producer.
    """
    return result.get("outcome") if result.get("codec") else result.get("stop_reason")


def source_reader(result: dict[str, Any]):
    """Which source vocabulary this rollout speaks, from what it DECLARES.

    Two producers write this layout. Our own harness records the grammar it ran
    (``evals/harness.py`` -> ``result.json["codec"]``), and a rollout that declares
    one carries that grammar's action dict and the Operation stream it dispatched.
    The external absolute-teacher collections have no such field and carry the
    teacher's ``computer_use`` arguments plus a post-scale ``intended_target``.

    Keyed on the declared field, never on the shape of the payload. Assuming one
    vocabulary is what marked every step of our own rollouts a source parse error —
    which silently blocked the whole rollouts-to-training path — and sniffing the
    other way round is how a 0..999 grid gets read as pixels.
    """
    return rollout_step if result.get("codec") else teacher_step


_TOOLCALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def extract_prose(raw: str) -> str:
    """The teacher's reasoning prose: everything OUTSIDE ``<tool_call>`` blocks.

    Keeps any ``<think>...</think>`` wrapper the teacher emitted. Empty for a bare
    tool call (a 1-2 char remainder is fence debris, not reasoning).
    """
    if not isinstance(raw, str):
        return ""
    prose = _TOOLCALL_RE.sub("", raw).strip()
    return prose if len(prose) > 2 else ""


def _pair(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(round(float(value[0]))), int(round(float(value[1]))))
        except (TypeError, ValueError):
            return None
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _read_slugs(path: str | None, *, flag: str) -> set[str] | None:
    """The slug set from a caller-named file. ``None`` only when no file was named.

    A named-but-absent file raises. Reading it as "no filter" would silently
    disable the OSWorld eval-leak drop (``--exclude_slugs``) or the quality keep
    (``--keep_slugs``) and produce a dataset the caller believes is filtered.
    """
    if path is None:
        return None
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"{flag}: not a file: {file}")
    slugs = {s.strip() for s in file.read_text().splitlines() if s.strip()}
    if not slugs:
        raise SystemExit(f"{flag}: {file} lists no slugs")
    return slugs


@dataclass
class ConvertStats:
    n_seen: int = 0
    n_kept: int = 0
    n_dropped_by_filter: int = 0
    n_dropped_leak: int = 0
    n_unusable: int = 0
    n_turns: int = 0
    n_turns_with_prose: int = 0
    n_records_with_prose: int = 0
    #: Turns the TEACHER lost (no parsed payload / no computer_use dict). Same
    #: for every codec, so it never skews a cross-arm comparison.
    n_turns_teacher_parse_error: int = 0
    #: Turns this codec could not express, so they are absent from this arm and
    #: this arm only. Kept separate from the teacher errors because it is the
    #: number that differs between arms built from one collection (e.g. diffabs
    #: cannot spell ``type()`` and compact_raw / compact_absolute have no
    #: TERMINATE); a format comparison is only meaningful if it matches, exactly
    #: like ``n_turns_with_prose``.
    n_turns_dropped_by_codec: int = 0
    #: Rollouts carrying a non-null ``task_reward``, counted only under
    #: ``--min_task_success``. Zero means the flag filtered on a score no rollout in
    #: the collection has.
    n_scored: int = 0


def convert_rollout(
    run_dir: Path,
    codec: Any,
    *,
    keep_prose: bool = True,
    min_valid_actions: int = 2,
    max_parse_error_frac: float = 0.5,
    min_task_success: float | None = None,
    stats: ConvertStats | None = None,
) -> dict[str, Any] | None:
    """One rollout dir -> one chat record in ``codec``'s grammar (or None if unusable)."""
    result_path = run_dir / "result.json"
    traj_path = run_dir / "trajectory.jsonl"
    if not result_path.is_file() or not traj_path.is_file():
        return None
    result = json.loads(result_path.read_text())

    # Deterministic-task-success filter (gold): keep only traces the OSWorld
    # evaluator scored >= threshold. The key is `task_reward` — what
    # `evals/harness.py` publishes from `evaluate_on_finish`; no producer has ever
    # written `task_success`, so reading that dropped every rollout on every
    # collection.
    #
    # `task_reward` is present-but-null on every rollout our harness did not score
    # (it is written unconditionally, `evals/harness.py:953`), so a key-presence
    # check cannot tell "scored 0.0" from "never scored" and a null cannot be
    # distinguished from a genuine miss here, one rollout at a time. Dropping a null
    # is right — an unscored rollout cannot be filtered by a score — but dropping
    # EVERY rollout means the flag cannot filter this collection at all, which
    # `main` refuses via `n_scored`.
    if min_task_success is not None:
        reward = result.get("task_reward")
        if reward is not None and stats is not None:
            stats.n_scored += 1
        if reward is None or float(reward) < min_task_success:
            return None

    instruction = result.get("instruction")
    screen = _pair(result.get("screen_size"))
    if screen is None:
        raise SystemExit(
            f"{result_path}: no usable `screen_size`. Every grammar resolves its "
            "coordinates against it — a relative delta is clamped to it and "
            "move_rel normalizes by it — so a default would emit silently wrong "
            "labels for the whole rollout."
        )
    steps_dir = run_dir / "steps"
    read_step = source_reader(result)

    turns: list[tuple[str, str, str]] = []  # (frame_path, assistant_text, prose)
    n_steps = n_parse_err = n_codec_drop = 0
    for entry in _read_jsonl(traj_path):
        step_num = entry.get("step_num", 0)
        if step_num == 0:
            continue
        n_steps += 1
        info = entry.get("info", {}) or {}
        seen_frame = steps_dir / f"step_{step_num - 1:03d}.png"
        if not seen_frame.is_file():
            continue

        prose = extract_prose(entry.get("action") or entry.get("response") or "")
        # `parsed` is NOT read here. It is the source grammar's own action dict,
        # which only `teacher_step` speaks; `rollout_step` reads `operations`, so
        # requiring a truthy `parsed` discarded one of our own rollouts as a parse
        # error while its dispatched Operation stream sat right there.
        #
        # `truncated` is read, and has to be: a turn `max_tokens` cut off records
        # an EMPTY operation stream, which lifts to a legitimate idle action, so
        # it enters the dataset as an unterminated `<think>` block labelled NO_OP.
        # The truthy-`parsed` check used to reject it as a side effect.
        if info.get("parse_error") or info.get("truncated"):
            n_parse_err += 1
            continue

        step = read_step(info, screen)
        if step is None:
            n_parse_err += 1
            continue

        text = format_step(codec, step)
        if text is None:
            # Off-grammar for this grammar only. Counted into ``n_parse_err`` so
            # the max_parse_error_frac gate behaves as before, and counted
            # separately as the per-arm number: it is why two arms built from one
            # collection can end up with different turn sets (see ConvertStats).
            n_parse_err += 1
            n_codec_drop += 1
            continue
        turns.append((str(seen_frame), text, prose))

    if len(turns) < min_valid_actions:
        return None
    if n_steps and (n_parse_err / n_steps) > max_parse_error_frac:
        return None

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": grammars.system_prompt(codec, thinking=bool(keep_prose))}],
        }
    ]
    n_prose_turns = 0
    for i, (frame_path, text, prose) in enumerate(turns):
        user_content: list[dict[str, Any]] = [{"type": "image", "image": frame_path}]
        if i == 0 and instruction:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        if keep_prose and prose:
            text = f"{prose}\n{text}"
            n_prose_turns += 1
        messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

    if stats is not None:
        stats.n_turns += len(turns)
        stats.n_turns_with_prose += n_prose_turns
        stats.n_records_with_prose += 1 if n_prose_turns else 0
        stats.n_turns_teacher_parse_error += n_parse_err - n_codec_drop
        stats.n_turns_dropped_by_codec += n_codec_drop

    slug = result.get("slug") or run_dir.name
    return {
        "sample_id": f"onpol_{slug}",
        "recording_id": slug,
        "app": "osworld_onpolicy",
        "platform": "UBUNTU",
        "instruction": instruction,
        "n_frames": len(turns),
        "subrecord_idx": 0,
        "n_subrecords": 1,
        "source_stop_reason": source_stop_reason(result),
        "source_parse_errors": n_parse_err,
        "messages": messages,
    }


def discover_run_dirs(roots: str, *, recursive: bool = False) -> list[Path]:
    """Rollout dirs under one or more COMMA-separated collection roots.

    ``recursive`` walks nested layouts (what ``convert_abs_to_deltatype.py`` did
    with ``rglob``); the default is the flat one-level layout the other four used.

    A root that is not a directory is fatal: skipping it silently would halve a
    two-collection build on one typo, with the manifest still recording the
    string that was asked for.
    """
    out: list[Path] = []
    for raw in roots.split(","):
        root = Path(raw.strip())
        if not root.is_dir():
            raise SystemExit(f"--rollouts_dir: not a directory: {root}")
        candidates = root.rglob("*") if recursive else root.iterdir()
        out += [
            d
            for d in candidates
            if d.is_dir() and (d / "result.json").is_file() and (d / "trajectory.jsonl").is_file()
        ]
    return sorted(set(out))


def split_by_recording(
    records: list[dict[str, Any]], *, train_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic train/val split grouped by ``recording_id`` (no leakage)."""
    rng = random.Random(seed)
    ids = sorted({r["recording_id"] for r in records})
    rng.shuffle(ids)
    n_train = max(1, round(len(ids) * train_ratio)) if ids else 0
    train_ids = set(ids[:n_train])
    train = [r for r in records if r["recording_id"] in train_ids]
    val = [r for r in records if r["recording_id"] not in train_ids]
    return train, val


def _assert_prose_coverage(stats: ConvertStats, keep_prose: bool) -> None:
    """Refuse to write an empty dataset, or a zero-prose one under ``keep_prose``.

    Zero prose under ``keep_prose`` is never a legitimate outcome for teacher
    rollouts (the teacher reasons before acting); it means the prose channel was
    dropped — the defect measured on the three ``teacher_8b_osworld_train_v1``
    arms in the module docstring (absolute 10721/10721 vs deltatype_raw 0/11337
    and diffabs 0/11102). ``onpolicy_distill/scripts/action_span_conversion.py``,
    outside this repo, is the predecessor's fix for the same problem.

    A zero floor only — it cannot see ``100% vs 5%``. The asymmetry itself is
    measured by :func:`prose_divergence`.
    """
    # An empty result is also how the prose guard gets bypassed if
    # ``n_turns == 0`` returns early: pointing the converter at a rollout
    # collection whose trajectory file / result schema it cannot read (e.g. the
    # ``traj.jsonl`` + nested-``params`` layout of
    # rollouts/teacher_8b_osworld_train_v1, which is where all three arms above
    # came from) exited 0 with a zero-record dataset and a manifest reading
    # prose_turn_frac 0.0.
    if stats.n_turns == 0:
        raise SystemExit(
            f"REFUSING TO WRITE: 0 usable assistant turns from {stats.n_seen} "
            f"discovered rollout dir(s) ({stats.n_unusable} unusable, "
            f"{stats.n_dropped_by_filter} filtered, {stats.n_dropped_leak} "
            "leak-dropped). An empty dataset is never the intent, and it is also "
            "how the prose-coverage check below gets bypassed. Check that "
            "--rollouts_dir points at dirs holding `result.json` + "
            "`trajectory.jsonl` (add --recursive for nested layouts), and that "
            "result.json carries the fields this converter reads "
            "(`instruction`, `screen_size`) and each trajectory entry the payload "
            "its producer's reader wants (`info.operations` for one of ours, "
            "`info.parsed` for a teacher collection)."
        )
    if not keep_prose:
        return
    if stats.n_turns_with_prose == 0:
        raise SystemExit(
            "REFUSING TO WRITE: --keep_prose is on but 0 of "
            f"{stats.n_turns} assistant turns carry prose "
            f"({stats.n_turns_dropped_by_codec} turn(s) were dropped because this "
            "grammar cannot express them, which also drops their prose). This is the "
            "defect that made the deltatype_raw / diffabs arms 0-of-11k prose-bearing "
            "against absolute's 10721-of-10721 and invalidated every trained "
            "absolute-vs-relative comparison built on them. Check that the rollout "
            "trajectory entries carry `action` / `response` text outside their "
            "<tool_call> blocks, that this grammar can express the turns those entries "
            "sit on, or pass --no_keep_prose deliberately to build a tool-call-only "
            "arm."
        )


#: Default relative tolerance for the cross-arm prose-coverage check, on the
#: pre-codec statistic (see ``_prose_frac_pre_codec``). Chosen from measurement:
#: over real collections the legitimate spread of that statistic between codecs
#: is 0.00% (teacher_8b_v1/v2/v3, all seven codecs, prose counts 32/34/4 against
#: an identical pre-codec denominator 734/1053/2011); the smallest real defect
#: instance is compact_raw and compact_absolute on teacher_8b_v1 losing 2
#: of 32 prose turns because they have no TERMINATE token (6.25%); the historical
#: defect this guard exists for was 100%. 0.05 sits above the observed legitimate
#: spread and below the smallest observed real defect. Its cost: on 32 prose
#: turns a single lost turn is 3.1% and passes.
PROSE_DIVERGENCE_TOL = 0.05


def _abs_or_none(value: Any) -> str | None:
    return str(Path(str(value)).resolve()) if value else None


def _source_selection(m: dict[str, Any]) -> tuple[Any, ...] | None:
    """The rollout selection a manifest was built from — its comparability key.

    Two arms are comparable iff they were offered the same rollouts under the same
    admission filters. The codec may differ, and so may the per-arm quality gates
    (``min_valid_actions`` / ``max_parse_error_frac``), whose effect on prose
    coverage is part of what the check should see. Paths are resolved so ``dir``
    and ``dir/`` are one selection. ``None`` when the manifest does not say (a
    pre-``n_turns`` manifest, which carries no prose accounting either).
    """
    raw = m.get("rollouts_dir")
    if not raw:
        return None
    roots = tuple(sorted(str(Path(r.strip()).resolve()) for r in str(raw).split(",") if r.strip()))
    if not roots:
        return None
    return (
        roots,
        bool(m.get("recursive")),
        _abs_or_none(m.get("keep_slugs")),
        _abs_or_none(m.get("exclude_slugs")),
        m.get("min_task_success"),
    )


def _pre_codec_turns(m: dict[str, Any]) -> int | None:
    """Turns this arm was offered: kept turns plus the ones its codec could not spell."""
    n, dropped = m.get("n_turns"), m.get("n_turns_dropped_by_codec")
    if not isinstance(n, int) or not isinstance(dropped, int) or (n + dropped) <= 0:
        return None
    return n + dropped


def _prose_frac_pre_codec(m: dict[str, Any]) -> float | None:
    """Prose coverage over the pre-codec turn population (the codec-invariant one).

    ``prose_turn_frac`` divides by the turns that survived the codec, so it moves
    when a codec is merely less expressive: on ``teacher_8b_v1`` ``diffabs`` drops
    74 of 733 turns, none prose-bearing, and reads 9.96% higher than
    ``deltatype_v2`` for no difference in supervision. Dividing by
    ``n_turns + n_turns_dropped_by_codec`` removes that confound — measured
    identical (734) for all seven codecs there — so what is left moves only when a
    codec drops a prose-bearing turn, which is never benign under a uniform loss
    mask. The invariance is empirical, not structural: a whole record lost to
    ``max_parse_error_frac`` takes its turns out of both terms, so the denominators
    are compared and a mismatch is reported rather than assumed away.
    """
    prose, pre = m.get("n_turns_with_prose"), _pre_codec_turns(m)
    if not isinstance(prose, int) or pre is None:
        return None
    return prose / pre


def _prose_frac_kept(m: dict[str, Any]) -> float:
    """Recorded ``prose_turn_frac``, re-derived if a manifest omits it. Reported only."""
    recorded = m.get("prose_turn_frac")
    if isinstance(recorded, (int, float)):
        return float(recorded)
    prose, n = m.get("n_turns_with_prose"), m.get("n_turns")
    if isinstance(prose, int) and isinstance(n, int) and n > 0:
        return prose / n
    return 0.0


def _rel_divergence(a: float, b: float) -> float:
    """Relative gap between two coverages, normalized by the LARGER one (0 if both 0)."""
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


def prose_divergence(
    out_dir: Path, self_manifest: dict[str, Any], *, tol: float
) -> dict[str, Any]:
    """Compare this arm's prose coverage against sibling arms from the same rollouts.

    ``--codec`` is single-valued, so an arm IS an invocation and a sweep IS a set
    of sibling ``--out_dir``s; each one leaves a ``convert_manifest.json``. This
    walks ``out_dir.parent``, keeps the manifests whose ``_source_selection``
    matches this arm's, and reports the relative divergence of
    ``_prose_frac_pre_codec``.

    Non-fatal by itself; the caller decides. A sibling arm may be mid-build,
    absent, or a legitimately prose-free ``--no_keep_prose`` arm, so missing,
    unreadable and prose-free siblings are skipped, never faults.
    """
    me = out_dir.resolve()
    key = _source_selection(self_manifest)
    self_frac = _prose_frac_pre_codec(self_manifest)
    report: dict[str, Any] = {
        "tol": tol,
        "statistic": "n_turns_with_prose / (n_turns + n_turns_dropped_by_codec)",
        "sibling_root": str(me.parent),
        "self_prose_frac_pre_codec": self_frac,
        "self_prose_turn_frac": _prose_frac_kept(self_manifest),
        "self_pre_codec_turns": _pre_codec_turns(self_manifest),
        "n_siblings_compared": 0,
        "max_rel_divergence": None,
        "diverged": False,
        "skipped": None,
        "siblings": [],
    }
    if key is None or self_frac is None:
        report["skipped"] = "this arm records no comparable source selection"
        return report
    if not me.parent.is_dir():
        report["skipped"] = f"no sibling root on disk: {me.parent}"
        return report

    rows: list[dict[str, Any]] = []
    for cand in sorted(me.parent.iterdir()):
        if not cand.is_dir() or cand.resolve() == me:
            continue
        mpath = cand / "convert_manifest.json"
        if not mpath.is_file():
            continue  # not an arm, or an arm still mid-build
        try:
            other = json.loads(mpath.read_text())
        except (OSError, ValueError):
            continue  # truncated / half-written manifest: treat as absent
        if not isinstance(other, dict) or not other.get("keep_prose"):
            continue  # --no_keep_prose arms are intentionally prose-free
        if _source_selection(other) != key:
            continue  # different rollouts or filters: not a comparison at all
        other_frac = _prose_frac_pre_codec(other)
        if other_frac is None:
            continue  # too old to carry prose accounting
        rel = _rel_divergence(self_frac, other_frac)
        rows.append(
            {
                "out_dir": str(cand),
                "codec": other.get("codec"),
                "n_turns": other.get("n_turns"),
                "n_turns_with_prose": other.get("n_turns_with_prose"),
                "n_turns_dropped_by_codec": other.get("n_turns_dropped_by_codec"),
                "pre_codec_turns": _pre_codec_turns(other),
                "prose_frac_pre_codec": other_frac,
                "prose_turn_frac": _prose_frac_kept(other),
                "rel_divergence": rel,
                "rel_divergence_raw_frac": _rel_divergence(
                    _prose_frac_kept(self_manifest), _prose_frac_kept(other)
                ),
                # The normalization above assumes this matches; say so when it does not.
                "pre_codec_turns_match": _pre_codec_turns(other)
                == _pre_codec_turns(self_manifest),
                "diverged": rel > tol,
            }
        )

    report["siblings"] = rows
    report["n_siblings_compared"] = len(rows)
    if not rows:
        report["skipped"] = (
            "no sibling arm from the same rollout selection has finished yet "
            "(nothing to compare against — NOT a pass)"
        )
        return report
    report["max_rel_divergence"] = max(r["rel_divergence"] for r in rows)
    report["diverged"] = any(r["diverged"] for r in rows)
    return report


def format_prose_divergence(report: dict[str, Any], codec: str) -> str:
    """One-paragraph human rendering of a diverging :func:`prose_divergence` report."""
    lines = [
        f"cross-arm PROSE COVERAGE DIVERGENCE: arm `{codec}` carries "
        f"{report['self_prose_frac_pre_codec']:.4%} prose over its "
        f"{report['self_pre_codec_turns']} pre-codec turns, which differs by more than "
        f"{report['tol']:.1%} (relative) from a sibling arm built from the SAME rollouts:"
    ]
    for r in sorted(report["siblings"], key=lambda r: -r["rel_divergence"]):
        if not r["diverged"]:
            continue
        note = "" if r["pre_codec_turns_match"] else "  [!] pre-codec turn totals DIFFER too"
        lines.append(
            f"  {r['rel_divergence']:+.2%} vs {r['codec']}: "
            f"{r['n_turns_with_prose']}/{r['pre_codec_turns']} = "
            f"{r['prose_frac_pre_codec']:.4%} pre-codec "
            f"(raw prose_turn_frac {r['prose_turn_frac']:.4%}, "
            f"{r['n_turns_dropped_by_codec']} turn(s) dropped by that codec)"
            f"  [{r['out_dir']}]{note}"
        )
    lines.append(
        "Arms with unequal reasoning supervision are NOT a format comparison — this is "
        "the same defect class as the 0-of-11k vs 10721-of-10721 asymmetry, at lower "
        "severity. Either the codecs drop different prose-bearing turns (check "
        "n_turns_dropped_by_codec), or a sibling manifest is stale. Pass "
        "--prose_divergence off to silence, or --no_keep_prose if this arm is "
        "meant to be prose-free."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--codec", required=True,
                   help="Target grammar under grammars/ (see `python -c \"import grammars; "
                        "print(grammars.available())\"`). Its Codec does the encoding, and its "
                        "coordinate convention IS the choice — there is no --coord_space.")
    p.add_argument("--rollouts_dir", required=True,
                   help="Collection output dir(s) with per-rollout subdirs. COMMA-separated to "
                        "combine multiple collect rounds (e.g. scale_v1,scale_v2).")
    p.add_argument("--recursive", action="store_true",
                   help="Walk nested rollout layouts instead of one level down.")
    p.add_argument("--out_dir", required=True,
                   help="Dataset root; writes _normalized/{train,val}/chat.jsonl.")
    p.add_argument("--keep_slugs", default=None,
                   help="Text file of run slugs to KEEP (quality-filter output), one per line.")
    p.add_argument("--exclude_slugs", default=None,
                   help="Text file of slugs / recording_ids to DROP (OSWorld eval-leak filter).")
    p.add_argument("--min_task_success", type=float, default=None,
                   help="Keep only traces the OSWorld evaluator scored >= this (e.g. 1.0). "
                        "Requires collection to have run with --evaluate.")
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--min_valid_actions", type=int, default=2)
    p.add_argument("--max_parse_error_frac", type=float, default=0.5)
    p.add_argument("--no_keep_prose", dest="keep_prose", action="store_false",
                   help="Build a tool-call-only arm (drop the teacher's reasoning). The default "
                        "keeps prose for every codec — see this module's docstring. Such an "
                        "arm is intentionally prose-free, so it is exempt from the cross-arm "
                        "divergence check below and invisible to other arms' checks.")
    p.set_defaults(keep_prose=True)
    p.add_argument("--prose_divergence", choices=("warn", "abort", "off"), default=None,
                   help="What a cross-arm prose-coverage divergence does: `warn` (the default) "
                        "reports it, `abort` refuses to write — use it for the LAST arm of a "
                        "sweep, or in CI — and `off` skips the comparison. The zero-coverage "
                        "abort applies regardless. Forced to `off` by --no_keep_prose; naming "
                        "both is an error.")
    p.add_argument("--prose_divergence_tol", type=float, default=None,
                   help="Relative tolerance for this arm's prose coverage against sibling arms "
                        "in sibling --out_dirs built from the same rollout selection, measured "
                        f"over the PRE-CODEC turn population. Default {PROSE_DIVERGENCE_TOL}: "
                        "the measured legitimate spread between codecs is 0.00%% and the "
                        "smallest real defect instance is 6.25%% (see PROSE_DIVERGENCE_TOL).")
    args = p.parse_args(argv)

    if not args.keep_prose and args.prose_divergence not in (None, "off"):
        p.error(
            f"--prose_divergence {args.prose_divergence} cannot run under --no_keep_prose: "
            "that arm is intentionally prose-free and is exempt from the comparison"
        )
    divergence = args.prose_divergence or ("warn" if args.keep_prose else "off")
    if divergence == "off" and args.prose_divergence_tol is not None:
        p.error("--prose_divergence_tol has no effect while the comparison is off")
    tol = (
        PROSE_DIVERGENCE_TOL
        if args.prose_divergence_tol is None
        else args.prose_divergence_tol
    )

    codec = grammars.load(args.codec)
    keep = _read_slugs(args.keep_slugs, flag="--keep_slugs")
    drop = _read_slugs(args.exclude_slugs, flag="--exclude_slugs") or set()

    stats = ConvertStats()
    records: list[dict[str, Any]] = []
    for d in discover_run_dirs(args.rollouts_dir, recursive=args.recursive):
        stats.n_seen += 1
        if keep is not None and d.name not in keep:
            stats.n_dropped_by_filter += 1
            continue
        if d.name in drop:
            stats.n_dropped_leak += 1
            continue
        rec = convert_rollout(
            d, codec,
            keep_prose=args.keep_prose,
            min_valid_actions=args.min_valid_actions,
            max_parse_error_frac=args.max_parse_error_frac,
            min_task_success=args.min_task_success,
            stats=stats,
        )
        if rec is None:
            stats.n_unusable += 1
            continue
        # Leak-drop by recording_id too (belt-and-suspenders vs the dir name).
        if rec["recording_id"] in drop:
            stats.n_dropped_leak += 1
            continue
        records.append(rec)
        stats.n_kept += 1

    # Before the prose guard: an all-dropped filter also trips that one, and its
    # message blames the rollout layout, which sends the reader after the wrong bug.
    # The slug counts are quoted because they are the other way to reach zero
    # scored: rollouts the slug filters removed never reached the score filter.
    if args.min_task_success is not None and not stats.n_scored:
        raise SystemExit(
            f"--min_task_success {args.min_task_success} filtered on a score not one "
            f"of the {stats.n_seen} discovered rollout dir(s) carries "
            f"({stats.n_dropped_by_filter} never reached it, dropped by --keep_slugs; "
            f"{stats.n_dropped_leak} by --exclude_slugs): no `task_reward`, or a null "
            "one. Our harness writes that field only under `evaluate_on_finish` "
            "(`evals/harness.py`) and the external teacher collections carry no "
            "per-rollout score at all, so this flag cannot filter such a collection — "
            "dropping all of it is not the filter working."
        )
    _assert_prose_coverage(stats, args.keep_prose)

    train, val = split_by_recording(records, train_ratio=args.train_ratio, seed=args.split_seed)
    out = Path(args.out_dir)

    manifest = {
        "codec": args.codec,
        # The grammar's own spec digest, so a dataset is traceable to the exact
        # grammar revision that produced it.
        "codec_digest": codec.digest,
        # The digest of the prompt actually written into every record, which the
        # eval harness compares against the prompt it renders. Under
        # ``keep_prose`` that is THINKING_PREAMBLE + describe(), so it differs
        # from ``codec_digest`` and only this field identifies the artifact.
        "system_prompt_sha256": hashlib.sha256(
            grammars.system_prompt(codec, thinking=bool(args.keep_prose)).encode()
        ).hexdigest(),
        "rollouts_dir": args.rollouts_dir,
        "recursive": args.recursive,
        "keep_slugs": args.keep_slugs,
        "exclude_slugs": args.exclude_slugs,
        "min_task_success": args.min_task_success,
        "keep_prose": args.keep_prose,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "min_valid_actions": args.min_valid_actions,
        "max_parse_error_frac": args.max_parse_error_frac,
        "n_seen": stats.n_seen,
        "n_kept": stats.n_kept,
        "n_dropped_by_filter": stats.n_dropped_by_filter,
        "n_dropped_leak": stats.n_dropped_leak,
        "n_unusable": stats.n_unusable,
        "n_scored": stats.n_scored,
        "n_train": len(train),
        "n_val": len(val),
        # Prose coverage has to match across arms for a format comparison to mean
        # anything, so it is recorded here.
        "n_turns": stats.n_turns,
        "n_turns_with_prose": stats.n_turns_with_prose,
        "n_records_with_prose": stats.n_records_with_prose,
        "prose_turn_frac": (stats.n_turns_with_prose / stats.n_turns) if stats.n_turns else 0.0,
        # Turn accounting split by where the loss came from. The teacher figure is
        # codec-independent; the codec figure is this arm's expressiveness gap,
        # and two arms are only comparable if it matches (diffabs cannot spell
        # `type()`, compact_raw / compact_absolute have no TERMINATE).
        "n_turns_teacher_parse_error": stats.n_turns_teacher_parse_error,
        "n_turns_dropped_by_codec": stats.n_turns_dropped_by_codec,
    }

    # The zero-abort above is a floor; this is the asymmetry check. Run it before
    # anything is written so `abort` can still refuse.
    if divergence != "off":
        report = prose_divergence(out, manifest, tol=tol)
        manifest["prose_divergence"] = report
        if report["diverged"]:
            message = format_prose_divergence(report, args.codec)
            if divergence == "abort":
                raise SystemExit(f"REFUSING TO WRITE: {message}")
            print(f"WARNING: {message}", file=sys.stderr)

    for split, recs in (("train", train), ("val", val)):
        d = out / "_normalized" / split
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chat.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out.mkdir(parents=True, exist_ok=True)
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
