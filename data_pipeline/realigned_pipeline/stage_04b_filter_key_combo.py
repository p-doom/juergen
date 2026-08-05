#!/usr/bin/env python3
"""Stage 04b (key-combo windows): re-cut stage-04 conversations into SHORT
windows ANCHORED at a key combination.

Stage 04's application filter selects by WHICH APP is in the foreground, which is
the finest label the recorder gives (``ContextChanged`` carries a bundle id and
nothing else -- no window title, no URL). This stage adds an orthogonal, purely
BEHAVIOURAL cut: keep only the stretch of a conversation that BEGINS with a
specific chord and runs for at most N turns after it.

    --key-combo Meta+KeyT --max-frames-after 15
      -> every window is "the user opened a new tab, then the next <=15 turns"

The output is the same ``chat.jsonl`` schema stage 04 writes, so stages 05/06
consume it unchanged -- this is a filter between 04 and 05, not a new lineage.

WHAT COUNTS AS A MATCH
  A window opens at the turn whose action program presses the combo's TRIGGER key
  (the last token of the spec) while every MODIFIER group in the spec is held.
  ``Meta`` / ``Ctrl`` / ``Shift`` / ``Alt`` are side-agnostic groups -- ``Meta``
  matches ``MetaLeft`` or ``MetaRight`` -- and any raw input name from the action
  vocabulary (``KeyT``, ``Return``, ``Tab``, ``LMB``, ``PageDown``, ...) works
  verbatim. A bare spec with no ``+`` (``--key-combo Return``) matches an
  unmodified press.

  ``--combo-scope turn`` (the default) additionally requires the modifiers to go
  down in the SAME turn as the trigger, so the emitted window is SELF-CONTAINED:
  the chord is fully visible inside the first assistant turn. ``conversation``
  scope lets a modifier be held from an earlier turn -- higher recall, but then
  the window's first turn shows ``down(KeyT)`` whose ``down(MetaLeft)`` was cut
  away, and the sequence no longer explains itself. Held state is tracked across
  turns either way (v3 programs really do carry a key across a turn boundary).

  ``--strict-modifiers`` additionally rejects a match carrying modifiers the spec
  did not ask for, so ``Meta+KeyT`` stops matching a ``Meta+Shift+KeyT`` press.

WINDOWS
  A window is the trigger turn plus up to ``--max-frames-after`` turns (so at most
  ``1 + N`` turns), truncated at the end of the source conversation. Windows
  shorter than ``--min-frames-after`` turns of follow-through are dropped as
  stubs. By default a trigger that lands INSIDE an already-open window does not
  start another one (``--allow-overlap`` to emit those too, which duplicates
  frames across rows).

ACTION FORMATS
  Reads the source's ``action_format`` from its summary and picks the matching
  parser: ordered mini-programs (``ordered_events_v2``/``v3``) via
  ``eval/action_parser.parse_ordered_action_tolerant``, the aggregate
  ``"<dx> <dy> <scroll> ; +KEY -KEY"`` forms (``sampled``/``canonical``) via
  ``parse_action_tolerant``. ``computer_use_rel_v1`` (JSON tool calls) is NOT
  supported -- there is no key-transition stream to match on.

OUTPUTS (drop-in for stages 05/06)
  conversations.jsonl / chat.jsonl   one row per WINDOW, source provenance kept,
                                     plus source_conversation_id, key_combo,
                                     combo_turn_index, combo_window_idx.
  conversations_summary.json         aggregate stats + the filter's settings.
  manifest.json                      artifact marker.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

# The strict grammar parsers live on the eval side (stdlib-only). Appended, not
# prepended, so ``eval/``'s module names can never shadow a stdlib import.
EVAL_DIR = DATA_PIPELINE_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from realigned_pipeline.lib.common import write_json, write_jsonl  # noqa: E402

try:
    from action_parser import (  # noqa: E402
        parse_action_tolerant,
        parse_ordered_action_tolerant,
    )
except ImportError as exc:  # pragma: no cover -- only if the repo is split up
    raise SystemExit(
        f"cannot import the action parsers from {EVAL_DIR} ({exc}); this stage "
        "matches chords with eval/action_parser.py so the grammar has one "
        "implementation"
    ) from exc

ORDERED_FORMATS = {"ordered_events_v2", "ordered_events_v3"}
AGGREGATE_FORMATS = {"sampled", "canonical"}

# Side-agnostic modifier groups. A spec token resolves to a SET of raw input
# names and matches if ANY member is held -- the recorder reports the physical
# key (MetaLeft/MetaRight), but nobody means "the left command key" when they say
# Cmd+T. Names observed in this corpus: MetaLeft/MetaRight, ControlLeft/
# ControlRight, ShiftLeft/ShiftRight, Alt/AltGr (macOS reports a bare ``Alt``).
_GROUPS: dict[str, frozenset[str]] = {
    "meta": frozenset({"MetaLeft", "MetaRight"}),
    "cmd": frozenset({"MetaLeft", "MetaRight"}),
    "command": frozenset({"MetaLeft", "MetaRight"}),
    "super": frozenset({"MetaLeft", "MetaRight"}),
    "win": frozenset({"MetaLeft", "MetaRight"}),
    "ctrl": frozenset({"ControlLeft", "ControlRight"}),
    "control": frozenset({"ControlLeft", "ControlRight"}),
    "shift": frozenset({"ShiftLeft", "ShiftRight"}),
    "alt": frozenset({"Alt", "AltLeft", "AltRight", "AltGr"}),
    "option": frozenset({"Alt", "AltLeft", "AltRight", "AltGr"}),
    "opt": frozenset({"Alt", "AltLeft", "AltRight", "AltGr"}),
}

# Every raw name that is a modifier, for --strict-modifiers ("no OTHER modifier
# may be held"). Union of the groups above.
ALL_MODIFIERS: frozenset[str] = frozenset().union(*_GROUPS.values())


def resolve_token(token: str) -> frozenset[str]:
    """One spec token -> the set of raw input names that satisfy it. A friendly
    group name (``Meta``, ``Ctrl``) expands side-agnostically; anything else is a
    raw name passed through verbatim, so the full action vocabulary works."""
    key = token.strip()
    if not key:
        raise ValueError("empty key token")
    return _GROUPS.get(key.lower(), frozenset({key}))


class KeyCombo:
    """A parsed ``Mod+Mod+Trigger`` spec.

    The LAST token is the trigger (the key whose PRESS opens the window); every
    earlier token is a modifier that must already be held at that moment.
    """

    def __init__(self, spec: str) -> None:
        tokens = [t for t in str(spec).replace(" ", "").split("+") if t]
        if not tokens:
            raise ValueError(f"empty key combo: {spec!r}")
        self.spec = "+".join(tokens)
        self.trigger: frozenset[str] = resolve_token(tokens[-1])
        self.modifiers: list[frozenset[str]] = [resolve_token(t) for t in tokens[:-1]]
        self.modifier_names: frozenset[str] = (
            frozenset().union(*self.modifiers) if self.modifiers else frozenset()
        )

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        return f"KeyCombo({self.spec!r})"

    def matches(
        self,
        pressed: str,
        held: dict[str, int],
        turn_idx: int,
        *,
        same_turn: bool,
        strict: bool,
    ) -> bool:
        """Is ``pressed`` (going down at ``turn_idx``) this combo?

        ``held`` maps a currently-held raw input name to the turn it went down on,
        which is what makes ``--combo-scope turn`` expressible: the modifier must
        have gone down on THIS turn for the window to be self-contained.
        """
        if pressed not in self.trigger:
            return False
        for group in self.modifiers:
            hits = [n for n in group if n in held]
            if not hits:
                return False
            if same_turn and not any(held[n] == turn_idx for n in hits):
                return False
        if strict:
            extra = (set(held) & ALL_MODIFIERS) - self.modifier_names
            if extra:
                return False
        return True


def split_combo_specs(values: Iterable[str] | None) -> list[str]:
    """``--key-combo`` values -> individual specs. Repeatable OR comma-separated,
    because a labctl ``[args]`` table can only render each key once."""
    out: list[str] = []
    for value in values or []:
        out.extend(part for part in str(value).split(",") if part.strip())
    return out


def _str2bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    v = str(s).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


# --------------------------------------------------------------------------
# turn -> ordered key transitions
# --------------------------------------------------------------------------

Transitions = list[tuple[str, str]]  # [("down"|"up", raw_input_name), ...]


def ordered_transitions(text: str) -> Transitions:
    """Key transitions of an ordered mini-program, order preserved."""
    return [
        ("down" if p.kind == "down" else "up", p.input_name)
        for p in parse_ordered_action_tolerant(text).primitives
        if p.kind in ("down", "up") and p.input_name
    ]


def aggregate_transitions(text: str) -> Transitions:
    """Key transitions of an aggregate ``+KEY -KEY`` action, order preserved."""
    return [
        ("down" if e.kind == "press" else "up", e.what)
        for e in parse_action_tolerant(text).events
        if e.what
    ]


def pick_transition_parser(action_format: str | None) -> Callable[[str], Transitions]:
    fmt = str(action_format or "").strip()
    if fmt in ORDERED_FORMATS:
        return ordered_transitions
    if fmt in AGGREGATE_FORMATS:
        return aggregate_transitions
    raise SystemExit(
        f"action_format {fmt!r} carries no key-transition stream this stage can "
        f"match on (supported: {sorted(ORDERED_FORMATS | AGGREGATE_FORMATS)}). "
        "Rebuild stage 04 with --action-format ordered_events_v3."
    )


# --------------------------------------------------------------------------
# message <-> turn mapping
# --------------------------------------------------------------------------


def split_messages(messages: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]
]:
    """``messages`` -> (leading system messages, [(user, assistant), ...]).

    Stage 04 always emits strict user/assistant alternation after the system
    turn, so an unpaired trailing user message is a malformed row, not a case to
    absorb silently.
    """
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    i = 0
    while i + 1 < len(rest):
        user, assistant = rest[i], rest[i + 1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError(
                f"expected a user/assistant pair at message {i}, got "
                f"{user.get('role')}/{assistant.get('role')}"
            )
        pairs.append((user, assistant))
        i += 2
    if i != len(rest):
        raise ValueError(f"trailing unpaired {rest[i].get('role')!r} message")
    return system, pairs


def leading_text_blocks(user_msg: dict[str, Any]) -> list[dict[str, Any]]:
    """The text blocks that precede the image on a first user turn -- stage 04's
    instruction (and any ``[CONTEXT]`` fused after it). Carried onto a window's
    own first turn so the schema contract still holds mid-conversation."""
    content = user_msg.get("content")
    if not isinstance(content, list):
        return []
    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            break
        if block.get("type") == "text":
            out.append(block)
        else:
            break
    return out


def with_leading_text(
    user_msg: dict[str, Any], blocks: list[dict[str, Any]]
) -> dict[str, Any]:
    """A copy of ``user_msg`` with ``blocks`` prepended to its content."""
    if not blocks:
        return user_msg
    content = user_msg.get("content")
    if not isinstance(content, list):
        return user_msg
    return {**user_msg, "content": [*blocks, *content]}


def assistant_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text", ""))
        return ""
    return str(content or "")


# --------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------


def find_windows(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    combos: list[KeyCombo],
    parse: Callable[[str], Transitions],
    *,
    max_frames_after: int,
    min_frames_after: int,
    same_turn: bool,
    strict: bool,
    allow_overlap: bool,
) -> list[dict[str, Any]]:
    """Anchor points in one conversation -> window descriptors.

    Held state is carried ACROSS turns (a v3 program can hold a key over a turn
    boundary), so ``held`` is seeded once per conversation and never per turn.
    """
    held: dict[str, int] = {}  # raw input name -> the turn it went down on
    windows: list[dict[str, Any]] = []
    open_until = -1  # last turn index covered by the previously emitted window

    for turn_idx, (_user, assistant) in enumerate(pairs):
        for kind, name in parse(assistant_text(assistant)):
            if kind == "up":
                held.pop(name, None)
                continue
            # A press is matched BEFORE it joins `held`, so a combo can never be
            # satisfied by its own trigger key (Shift+ShiftLeft is not a chord).
            if not (allow_overlap or turn_idx > open_until):
                held[name] = turn_idx
                continue
            for combo in combos:
                if not combo.matches(
                    name, held, turn_idx, same_turn=same_turn, strict=strict
                ):
                    continue
                end = min(turn_idx + max_frames_after, len(pairs) - 1)
                frames_after = end - turn_idx
                if frames_after < min_frames_after:
                    break  # a stub at the tail; no other combo can be longer
                windows.append({
                    "start": turn_idx,
                    "end": end,
                    "frames_after": frames_after,
                    "key_combo": combo.spec,
                    "trigger_key": name,
                })
                open_until = end
                break
            held[name] = turn_idx
    return windows


def build_window_row(
    row: dict[str, Any],
    system: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    window: dict[str, Any],
    window_idx: int,
    parse: Callable[[str], Transitions],
    *,
    carry_instruction: bool,
) -> dict[str, Any]:
    """One window descriptor -> a stage-04-shaped conversation row."""
    start, end = window["start"], window["end"]
    selected = pairs[start : end + 1]
    blocks = (
        leading_text_blocks(pairs[0][0]) if carry_instruction and start > 0 else []
    )

    messages: list[dict[str, Any]] = list(system)
    for i, (user, assistant) in enumerate(selected):
        messages.append(with_leading_text(user, blocks) if i == 0 else user)
        messages.append(assistant)

    n_turns = len(selected)
    n_non_noop = sum(1 for _u, a in selected if _is_non_noop(a))
    source_id = row.get("conversation_id")
    return {
        **row,
        "conversation_id": f"{source_id}_kc{window_idx:03d}",
        "source_conversation_id": source_id,
        "key_combo": window["key_combo"],
        "trigger_key": window["trigger_key"],
        "combo_turn_index": start,          # index in the SOURCE conversation
        "combo_window_idx": window_idx,
        "combo_frames_after": window["frames_after"],
        "source_n_turns": row.get("n_turns"),
        "n_frames": n_turns,
        "n_turns": n_turns,
        "n_non_noop": n_non_noop,
        "messages": messages,
    }


def _is_non_noop(assistant: dict[str, Any]) -> bool:
    """Did this turn do anything? ``NO_OP`` is the idle token in BOTH the ordered
    and the aggregate grammars, so one string test covers every supported
    format -- and it also catches move/scroll-only turns, which carry no key
    transitions at all."""
    text = assistant_text(assistant).strip()
    return bool(text) and text != "NO_OP"


def load_source_summary(source_dir: Path) -> dict[str, Any]:
    for name in ("conversations_summary.json", "manifest.json"):
        path = source_dir / name
        if path.is_file():
            return json.loads(path.read_text())
    raise SystemExit(
        f"no conversations_summary.json/manifest.json under {source_dir} -- "
        "point --source-dir at a stage-04 conversations artifact"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source-dir", type=Path, required=True,
                   help="A stage-04 conversations artifact (reads its chat.jsonl).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--key-combo", action="append", default=None, metavar="COMBO",
                   required=True,
                   help="Chord that OPENS a window, e.g. Meta+KeyT (new tab), "
                        "Meta+KeyL (address bar), Meta+Shift+KeyT, Return. The last "
                        "token is the trigger key; earlier tokens are modifiers that "
                        "must be held. Meta/Cmd/Ctrl/Shift/Alt are side-agnostic; any "
                        "raw input name (KeyT, Tab, LMB, PageDown, ...) works "
                        "verbatim. Repeatable OR comma-separated "
                        "(--key-combo=Meta+KeyT,Meta+KeyL), since a labctl recipe "
                        "renders each arg once and cannot repeat a flag.")
    p.add_argument("--max-frames-after", type=int, default=15,
                   help="Keep at most this many turns AFTER the trigger turn, so a "
                        "window is at most 1+N turns (at --target-fps 1, N seconds). "
                        "Truncated at the end of the source conversation.")
    p.add_argument("--min-frames-after", type=int, default=0,
                   help="Drop windows with fewer than this many follow-through turns "
                        "(stubs at the tail of a conversation).")
    p.add_argument("--combo-scope", choices=("turn", "conversation"), default="turn",
                   help="'turn' (default): the modifiers must go down in the SAME turn "
                        "as the trigger, so the chord is fully visible inside the "
                        "window's first assistant turn. 'conversation': a modifier held "
                        "from an earlier turn also counts -- more matches, but the "
                        "window no longer contains the whole chord.")
    p.add_argument("--strict-modifiers", nargs="?", const=True, type=_str2bool,
                   default=False, metavar="BOOL",
                   help="Reject a match that carries modifiers the spec did not ask "
                        "for, so Meta+KeyT stops matching Meta+Shift+KeyT.")
    p.add_argument("--allow-overlap", nargs="?", const=True, type=_str2bool,
                   default=False, metavar="BOOL",
                   help="Emit a window for a trigger that fires INSIDE an already-open "
                        "window. Off by default -- overlapping windows duplicate frames "
                        "across rows.")
    p.add_argument("--carry-instruction", nargs="?", const=True, type=_str2bool,
                   default=True, metavar="BOOL",
                   help="Copy the source's first-user-turn text (instruction, fused "
                        "[CONTEXT]) onto a window that does not start at turn 0, so the "
                        "chat.jsonl contract still holds. No-op on goal-free data.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N source conversations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    chat_path = source_dir / "chat.jsonl"
    if not chat_path.is_file():
        raise SystemExit(f"no chat.jsonl under {source_dir}")

    src_summary = load_source_summary(source_dir)
    parse = pick_transition_parser(src_summary.get("action_format"))

    specs = split_combo_specs(args.key_combo)
    if not specs:
        raise SystemExit("--key-combo resolved to nothing")
    combos = [KeyCombo(s) for s in specs]
    print(
        f"[stage_04b] combos: {', '.join(c.spec for c in combos)} | "
        f"scope={args.combo_scope} strict_modifiers={bool(args.strict_modifiers)} | "
        f"window = trigger + <={args.max_frames_after} turns "
        f"(>= {args.min_frames_after} after)",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    combo_counts: Counter = Counter()
    trigger_counts: Counter = Counter()
    app_counts: Counter = Counter()
    n_source = 0
    n_source_with_hit = 0
    n_turns_total = 0
    n_failed = 0
    frames_after_hist: Counter = Counter()

    with chat_path.open() as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            n_source += 1
            row = json.loads(line)
            try:
                system, pairs = split_messages(row.get("messages") or [])
            except ValueError as exc:
                n_failed += 1
                print(f"  FAIL {row.get('conversation_id')}: {exc}", flush=True)
                continue
            if not pairs:
                continue
            windows = find_windows(
                pairs, combos, parse,
                max_frames_after=int(args.max_frames_after),
                min_frames_after=int(args.min_frames_after),
                same_turn=args.combo_scope == "turn",
                strict=bool(args.strict_modifiers),
                allow_overlap=bool(args.allow_overlap),
            )
            if windows:
                n_source_with_hit += 1
            for k, window in enumerate(windows):
                out = build_window_row(
                    row, system, pairs, window, k, parse,
                    carry_instruction=bool(args.carry_instruction),
                )
                records.append(out)
                n_turns_total += out["n_turns"]
                combo_counts[window["key_combo"]] += 1
                trigger_counts[window["trigger_key"]] += 1
                frames_after_hist[window["frames_after"]] += 1
                if out.get("app"):
                    app_counts[str(out["app"])] += 1
            if n_source % 2000 == 0:
                print(
                    f"  {n_source} source conversations | {len(records)} windows",
                    flush=True,
                )

    if not records:
        raise SystemExit(
            "no windows matched -- check --key-combo against the source's input "
            "vocabulary, or relax --combo-scope to 'conversation'"
        )

    out_dir = args.output_dir
    write_jsonl(out_dir / "conversations.jsonl", records)
    write_jsonl(out_dir / "chat.jsonl", records)

    summary = {
        "n_conversations": len(records),
        "n_source_conversations": n_source,
        "n_source_conversations_with_match": n_source_with_hit,
        "n_failed": n_failed,
        "n_frames_total": n_turns_total,
        "n_turns_total": n_turns_total,
        "mode": "key_combo_window",
        # --- key-combo filter ---------------------------------------------
        "key_combos": [c.spec for c in combos],
        "max_frames_after": int(args.max_frames_after),
        "min_frames_after": int(args.min_frames_after),
        "combo_scope": args.combo_scope,
        "strict_modifiers": bool(args.strict_modifiers),
        "allow_overlap": bool(args.allow_overlap),
        "carry_instruction": bool(args.carry_instruction),
        "combo_window_counts": dict(combo_counts.most_common()),
        "trigger_key_counts": dict(trigger_counts.most_common()),
        "frames_after_histogram": dict(sorted(frames_after_hist.items())),
        "app_conversation_counts": dict(app_counts.most_common()) or None,
        # --- inherited from the source ------------------------------------
        "action_format": src_summary.get("action_format"),
        "source_dir": str(source_dir),
        "source_action_format": src_summary.get("action_format"),
        "source_include_app": src_summary.get("include_app"),
        "source_n_conversations": src_summary.get("n_conversations"),
        "has_system_prompt": bool(src_summary.get("has_system_prompt")),
        "system_prompt_id": src_summary.get("system_prompt_id"),
        "target_fps": src_summary.get("target_fps"),
    }
    write_json(out_dir / "conversations_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "juergen_annotation_conversations",
        "schema_version": 1,
        "conversations": "conversations.jsonl",
        "chat": "chat.jsonl",  # split-agnostic drop-in source_path for stages 05/06
        **summary,
    })

    med = sorted(r["n_turns"] for r in records)[len(records) // 2]
    print(
        f"[stage_04b] {len(records)} windows from {n_source_with_hit}/{n_source} "
        f"conversations | {n_turns_total} turns, median {med}/window | "
        + ", ".join(f"{k}={v}" for k, v in combo_counts.most_common(6))
        + f" -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
