#!/usr/bin/env python3
"""Action-distribution viewer for the **ordered_events_v3** mini-program format.

The sibling of ``visualize_action_distribution.py``, which aggregates the
*aggregate* action grammar (``<dx> <dy> <scroll> ; +Key -Key``). That grammar
cannot express order, so that viewer answers "how much movement / which keys";
it cannot answer the questions v3 exists to make answerable. This one is built
for the ordered grammar directly (``lib/action_format.ORDERED_EVENTS_V3_GRAMMAR``,
served in-UI so it cannot drift)::

    line       = "NO_OP" / primitive *("; " primitive)
    primitive  = move(dx,dy) / scroll(dx,dy) / down(NAME) / up(NAME)
                 / type("chars")

so the aggregate questions it answers are v3-shaped:

  * **Program shape** — what do turns actually look like as mini-programs?
    ``move+; down(LMB); up(LMB)``, ``type+``, ``scroll+; down; up``… ranked, and
    clickable to see verbatim example turns.
  * **Order** — how often does a turn really use the ordering v2/v3 bought:
    move→click→move, click-after-motion, drags (``down(LMB); move…; up(LMB)``),
    segmented motion, typing interleaved with motion.
  * **Typing collapse** — how much text arrives as ``type("…")`` vs as bare
    printable ``down``/``up`` pairs the formatter could not fold (key rollover /
    Shift spans that never balanced inside one window). This is THE metric for
    judging stage 04's ``--coalesce-typing``, plus the typed strings, words and
    character classes themselves.
  * **Motion granularity** — deltas per *primitive* (motor-tick quanta, not
    per-frame sums), and per-turn primitive counts against the
    ``continuous_action_hz / target_fps`` tick budget.
  * **Held state across turns** — chords (a modifier held into a later turn),
    dangling ``up``, redundant ``down``, keys still held at segment end. The
    per-window formatter cannot see these; only a cross-turn pass can.
  * **Grammar conformance** — every turn is parsed by the *eval* parser
    (``eval/action_parser.parse_ordered_action``, the same code freeroll
    dispatches model replies with), STRICTLY: no tolerant fallback, because a
    training label — unlike a model reply — has no excuse for needing one. A turn
    counted here as failing is a turn whose text does not match the grammar it
    teaches. Failures are shown verbatim.
  * **Drill-down** — every aggregate above is also a way in. Clicking a bar
    filters the turns by its token (``down(LMB)``, ``move(-100,10)``), and the
    per-turn count filter selects on how much a turn *does*: actions, mouse
    (move + scroll + buttons), move, scroll, clicks, keys, ``type()`` — each an
    inclusive min/max, ANDed with each other and with the token. So "turns with
    ≥10 actions", "turns that scroll but never move" (``scroll`` min 1,
    ``move`` max 0) or "clicks with no motion at all" are two numbers away, and
    every per-turn histogram bucket sets the same filter when clicked. The
    result is the matched turn/segment counts, the primitive mix *inside* the
    selection, the segments it concentrates in, and verbatim example turns.
  * **Manifest cross-check** — a stage-04 manifest carries the formatter's own
    ``primitive_counts`` / dead-zone counters; the recomputed counts are shown
    beside them, so a mismatch (wrong dataset, partial write, re-derived labels)
    is visible instead of implied.

It reads the same datasets the frame viewer browses — stage-04
``conversations.jsonl``, stage-06 inline SFT records (ArrayRecord shards), and
anything else ``visualize_frame_records.detect_mode`` recognizes — by importing
that module's loaders, so a segment's numbers here match the trajectory you open
there. Datasets in another action format (canonical, ``computer_use_rel_v1``)
load fine and are reported as such: the parse-failure rate is the detector, and
the UI says so instead of showing zeros. ``ordered_events_v2`` is the v3 grammar
minus ``type()``, so v2 datasets aggregate correctly too (their typing cards are
simply empty).

Run::

    cd .../data_pipeline
    uv run python realigned_pipeline/visualize_ordered_action_distribution.py \
        --dataset <dir_or_file> [<dir_or_file> ...] \
        --limit 200 --port 8790
    # then SSH-forward the port and open http://127.0.0.1:8790/
    #   ssh -L 8790:127.0.0.1:8790 <host>

Pass several datasets and switch between them in the UI's "dataset" dropdown
(e.g. a coalesced build next to its un-coalesced source); each is aggregated
lazily on first selection and cached.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Reuse the frame viewer's dataset loaders (single source of truth for the four
# dataset shapes + ``ar://`` handling) and the formatter's own tables for what
# counts as a printable key / a modifier — the same constants
# ``OrderedTypingFormatter`` renders with, so "collapsible typing" here means
# exactly what it means on the write side.
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))
# The strict grammar parser lives on the eval side (stdlib-only). Appended, not
# prepended, so ``eval/``'s module names can never shadow a stdlib import.
EVAL_DIR = DATA_PIPELINE_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from realigned_pipeline import visualize_frame_records as V  # noqa: E402
from realigned_pipeline.lib.action_format import (  # noqa: E402
    _NON_SHIFT_MODIFIERS,
    _SHIFT_KEYS,
    _US_PRINTABLE,
    ORDERED_EVENTS_V3_GRAMMAR,
    OrderedFormatter,
    OrderedTypingFormatter,
)

try:  # the eval-side parser freeroll dispatches with (see module docstring)
    from action_parser import parse_ordered_action as parse_ordered_line  # noqa: E402
except ImportError as exc:  # pragma: no cover — only if the repo is split up
    raise SystemExit(
        f"cannot import the ordered-action parser from {EVAL_DIR} ({exc}); this "
        "viewer parses turns with eval/action_parser.py so the grammar has one "
        "implementation"
    ) from exc

V3_FORMAT = OrderedTypingFormatter.name          # "ordered_events_v3"
V2_FORMAT = OrderedFormatter.name                # "ordered_events_v2"
ORDERED_FORMATS = (V3_FORMAT, V2_FORMAT)

# Dataset registry: display-name -> {"path", "mode", "obj", "dist"}; built lazily
# on first selection (mirrors the sibling viewers).
DATASETS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
DATASET_SAMPLE_LIMIT: "int | None" = None
TOP_N = 60                    # rows per ranked list
MAX_EXAMPLES_PER_SHAPE = 3    # verbatim turns kept per program shape
MAX_SHAPES_WITH_EXAMPLES = 4000   # bound the example store on huge datasets
MAX_ERRORS = 60               # verbatim parse failures kept

_BUTTONS = ("LMB", "RMB", "MMB")
_BUTTON_SET = frozenset(_BUTTONS)

# Left/right modifier variants fold into one chord label; the per-key card keeps
# the raw rdev names.
_MOD_CANON = {
    "ControlLeft": "Ctrl", "ControlRight": "Ctrl", "Control": "Ctrl",
    "ShiftLeft": "Shift", "ShiftRight": "Shift", "Shift": "Shift",
    "Alt": "Alt", "AltLeft": "Alt", "AltRight": "Alt", "AltGr": "AltGr",
    "MetaLeft": "Meta", "MetaRight": "Meta", "Meta": "Meta", "MetaGr": "Meta",
    "Function": "Fn",
}
_MODIFIERS = frozenset(_SHIFT_KEYS | _NON_SHIFT_MODIFIERS)

# Keys the v3 formatter deliberately never folds into ``type()`` — they are the
# actions worth supervising per turn (submit, backtrack, navigate).
_EDIT_KEYS = (
    "Backspace", "Delete", "Return", "Enter", "NumpadEnter", "Tab", "Escape",
    "UpArrow", "DownArrow", "LeftArrow", "RightArrow",
    "Home", "End", "PageUp", "PageDown", "Insert", "CapsLock",
)

# Per-PRIMITIVE delta bins. A v3 move is one motor tick (default 10 Hz), not a
# whole frame's travel, so the interesting resolution is much finer than the
# aggregate viewer's ±100 quantum — but ±100 stays an edge (and is highlighted)
# because the bounds-clip corner-pinning failure mode lives there.
_DELTA_EDGES = [-100000, -250, -100, -60, -40, -25, -15, -10, -6, -3, -1,
                1, 3, 6, 10, 15, 25, 40, 60, 100, 250, 100000]
_MAG_EDGES = [0, 2, 5, 10, 15, 25, 40, 60, 100, 150, 250, 100000]
_DIR_LABELS = ["E →", "SE ↘", "S ↓", "SW ↙", "W ←", "NW ↖", "N ↑", "NE ↗"]

# Integer-count buckets (primitives per turn, characters per type(), …).
_COUNT_BUCKETS: list[tuple[int, "int | None"]] = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10),
    (11, 15), (16, 20), (21, 30), (31, 50), (51, None),
]
_LEN_BUCKETS: list[tuple[int, "int | None"]] = [
    (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40),
    (41, 80), (81, None),
]


# --------------------------------------------------------------------------- #
# Small histogram / bucket helpers (shared shape with the sibling viewer so the
# front-end renderers are interchangeable).
# --------------------------------------------------------------------------- #
def _bin_index(value: float, edges: list[int]) -> int:
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges) - 2
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def _bin_labels(edges: list[int]) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= -100000:
            labels.append(f"<{hi}")
        elif hi >= 100000:
            labels.append(f"≥{lo}")
        else:
            labels.append(f"{lo}..{hi}")
    return labels


def _hist_spec(edges: list[int], counts: list[int],
               highlight_at: "list[int]") -> dict[str, Any]:
    hi_idx = [i for i in range(len(edges) - 1) if edges[i] in highlight_at]
    return {"labels": _bin_labels(edges), "counts": counts, "highlight": hi_idx}


def _bucket_label(lo: int, hi: "int | None") -> str:
    if hi is None:
        return f"{lo}+"
    return str(lo) if lo == hi else f"{lo}–{hi}"


def _int_hist(counter: Counter, buckets: list[tuple[int, "int | None"]],
              highlight: "list[int]" = (),
              field: "str | None" = None) -> dict[str, Any]:
    """Bucket an integer-valued Counter into a renderable histogram.

    ``field`` names the per-turn filter field the buckets belong to (see
    ``_TURN_FIELDS``); the UI turns such a histogram's bars into clickable
    filters — the bucket's own bounds ARE the min/max — so the distribution and
    the drill-down into it are the same control."""
    counts = [0] * len(buckets)
    for value, n in counter.items():
        for i, (lo, hi) in enumerate(buckets):
            if value >= lo and (hi is None or value <= hi):
                counts[i] += n
                break
    return {
        "labels": [_bucket_label(lo, hi) for lo, hi in buckets],
        "counts": counts,
        "highlight": [i for i, (lo, _hi) in enumerate(buckets) if lo in highlight],
        "field": field,
        "bounds": [[lo, hi] for lo, hi in buckets],
    }


def _dir8(dx: float, dy: float) -> int:
    """8-way compass bucket of a (dx, dy) move (screen coords: +dy is DOWN)."""
    return round(math.degrees(math.atan2(dy, dx)) / 45.0) % 8


def _mean(total: float, n: int) -> float:
    return round(total / n, 2) if n else 0.0


# --------------------------------------------------------------------------- #
# One assistant turn -> primitives (+ the anomalies visible from the text alone)
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    """One assistant turn of an ordered_events dataset, as the viewer sees it.

    ``body`` is the turn with any ``<think>…</think>`` block and the stage-04
    ``TERMINATE`` suffix removed — i.e. the action line the eval parser sees.
    ``prims`` is empty for a ``NO_OP`` / terminate-only / failed turn; ``error``
    is set only for genuine grammar violations."""

    body: str
    prims: tuple = ()
    think_chars: int = 0
    terminate: bool = False
    noop: bool = False
    error: "str | None" = None
    n_lines: int = 0


def parse_turn(raw: Any) -> Turn:
    """Parse one assistant turn's text into a ``Turn``.

    The ordered formats put exactly ONE action line in a turn; a thinking-SFT
    turn prefixes it with ``<think>…</think>``, and stage 04 may overwrite or
    suffix the turn with its ``TERMINATE`` token. Everything else is a grammar
    violation and is reported as one (with the offending text), because the eval
    parser would reject it at rollout time too."""
    text = "" if raw is None else str(raw)
    think_chars = sum(len(m.group(0)) for m in V._THINK_RE.finditer(text))
    body = V._THINK_RE.sub("", text).strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    terminate = any(ln == "TERMINATE" for ln in lines)
    lines = [ln for ln in lines if ln != "TERMINATE"]
    turn = Turn(body="; ".join(lines), think_chars=think_chars,
                terminate=terminate, n_lines=len(lines))
    if not lines:
        # A terminate-only turn is stage 04 doing its job; an empty assistant
        # turn is not.
        if not terminate:
            turn.error = "empty action text"
        return turn
    prims: list[Any] = []
    for line in lines:
        try:
            prims.extend(parse_ordered_line(line).primitives)
        except (ValueError, TypeError) as exc:
            turn.error = str(exc)
            return turn
    turn.prims = tuple(prims)
    turn.noop = not prims
    return turn


def _shape(kinds: list[str]) -> str:
    """A turn's program shape: kinds in order, consecutive repeats folded into a
    ``+`` suffix (``move,move,down,up`` -> ``move+; down; up``). Folding is what
    makes shapes group — the interesting distinction is "moved then clicked",
    not "moved 3 times vs 4 times" (the per-turn count histograms cover that)."""
    out: list[str] = []
    i = 0
    while i < len(kinds):
        j = i
        while j < len(kinds) and kinds[j] == kinds[i]:
            j += 1
        out.append(kinds[i] + ("+" if j - i > 1 else ""))
        i = j
    return "; ".join(out)


# --------------------------------------------------------------------------- #
# Per-turn counts + the numeric turn filter
# --------------------------------------------------------------------------- #
# The quantities a turn can be selected on. ``key`` is BOTH the query-parameter
# stem (``min_<key>`` / ``max_<key>``) and the histogram ``field``, and the UI
# renders one min/max pair per entry from this list — so the two sides cannot
# drift on what "mouse" or "click" counts.
_TURN_FIELDS: list[tuple[str, str, str]] = [
    ("actions", "actions", "every primitive in the turn"),
    ("mouse", "mouse", "move + scroll + button down/up"),
    ("move", "move", "move() primitives"),
    ("scroll", "scroll", "scroll() primitives"),
    ("click", "clicks", "button presses — down(LMB/RMB/MMB)"),
    ("key", "keys", "key down/up, buttons excluded"),
    ("type", "type()", 'type("…") primitives'),
]
_TURN_FIELD_KEYS = tuple(k for k, _lab, _desc in _TURN_FIELDS)


def turn_counts(turn: Turn) -> dict[str, int]:
    """The per-turn quantities the histograms bucket and the filter selects on.

    A NO_OP / terminate-only turn counts as zero of everything (so ``actions``
    max 0 IS the way to select the idle turns); a turn that fails the grammar
    has no trustworthy counts at all and is excluded by the filter instead."""
    kinds = Counter(p.kind for p in turn.prims)
    buttons = presses = keys = 0
    for p in turn.prims:
        if p.kind not in ("down", "up"):
            continue
        if p.input_name in _BUTTON_SET:
            buttons += 1
            if p.kind == "down":
                presses += 1
        else:
            keys += 1
    move, scroll = kinds.get("move", 0), kinds.get("scroll", 0)
    return {
        "actions": len(turn.prims),
        "mouse": move + scroll + buttons,
        "move": move,
        "scroll": scroll,
        "click": presses,
        "key": keys,
        "type": kinds.get("type", 0),
    }


def _int_param(q: dict[str, list[str]], name: str) -> "int | None":
    vals = q.get(name)
    if not vals or not str(vals[0]).strip():
        return None
    try:
        return int(float(vals[0]))
    except ValueError:
        return None


@dataclass(frozen=True)
class TurnFilter:
    """Inclusive per-turn count bounds — ``actions ≥ 10``, ``scroll = 0``, …

    An empty filter matches every turn, so "no filter" is the same code path as
    a filter; combined with the substring query it is an AND (the turn must
    contain the token *and* satisfy every bound)."""

    bounds: tuple[tuple[str, "int | None", "int | None"], ...] = ()

    @classmethod
    def from_query(cls, q: dict[str, list[str]]) -> "TurnFilter":
        out = []
        for key in _TURN_FIELD_KEYS:
            lo, hi = _int_param(q, "min_" + key), _int_param(q, "max_" + key)
            if lo is not None or hi is not None:
                out.append((key, lo, hi))
        return cls(tuple(out))

    def __bool__(self) -> bool:
        return bool(self.bounds)

    def accepts(self, counts: dict[str, int]) -> bool:
        return all(
            (lo is None or counts[key] >= lo) and (hi is None or counts[key] <= hi)
            for key, lo, hi in self.bounds
        )

    def label(self) -> str:
        parts = []
        for key, lo, hi in self.bounds:
            if lo is not None and hi is not None:
                parts.append(f"{key}/turn {lo}" if lo == hi else f"{key}/turn {lo}–{hi}")
            elif lo is not None:
                parts.append(f"{key}/turn ≥{lo}")
            else:
                parts.append(f"{key}/turn ≤{hi}")
        return " · ".join(parts)


NO_TURN_FILTER = TurnFilter()   # the "every turn" singleton (frozen, shareable)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
# Ordering facts, in the order they are displayed. Each is a per-turn boolean —
# the counts are "turns in which this happened", never primitive counts, so they
# read as fractions of active turns.
_STRUCTURE_LABELS: list[tuple[str, str]] = [
    ("move_click_move", "move → click → move (one turn)"),
    ("motion_then_click", "motion before a click"),
    ("click_then_motion", "motion after a click"),
    ("drag", "drag (button held across motion)"),
    ("segmented_motion", "≥2 separate motion runs"),
    ("multi_click", "≥2 button presses"),
    ("move_and_scroll", "move + scroll mixed"),
    ("type_and_motion", "typing + motion mixed"),
    ("type_and_click", "typing + click mixed"),
    ("type_and_edit", "typing + editing key (Backspace/Return/…)"),
    ("motion_only", "motion only (no key/button/typing)"),
    ("keys_only", "keys/typing only (no motion)"),
]


@dataclass
class Aggregator:
    """Single-pass accumulator over a dataset's turns.

    Two levels of state: per-turn structure (shapes, ordering, click pairing)
    and per-SEGMENT carry-over (the physical held-key set, the reconstructed
    typed text). The carry-over is the part no per-window formatter can see —
    a Ctrl held into the next turn is a chord, and a printable key whose release
    landed in the next window is typing the formatter had to spell out."""

    hz: float = 10.0
    fps: float = 0.0

    n_segments: int = 0
    n_turns: int = 0
    n_active: int = 0
    n_noop: int = 0
    n_terminate: int = 0
    n_thinking: int = 0
    n_error: int = 0
    n_multiline: int = 0
    think_chars: int = 0
    prims_total: int = 0

    kind_counts: Counter = field(default_factory=Counter)
    kind_turns: Counter = field(default_factory=Counter)
    shapes: Counter = field(default_factory=Counter)
    shape_examples: dict = field(default_factory=dict)
    structure: Counter = field(default_factory=Counter)
    prims_per_turn: Counter = field(default_factory=Counter)
    mouse_per_turn: Counter = field(default_factory=Counter)
    moves_per_turn: Counter = field(default_factory=Counter)
    scrolls_per_turn: Counter = field(default_factory=Counter)
    clicks_per_turn: Counter = field(default_factory=Counter)
    types_per_turn: Counter = field(default_factory=Counter)
    over_grid: int = 0

    # movement / scrolling, per primitive
    move_top: Counter = field(default_factory=Counter)
    scroll_top: Counter = field(default_factory=Counter)
    dx_counts: list = field(default_factory=lambda: [0] * (len(_DELTA_EDGES) - 1))
    dy_counts: list = field(default_factory=lambda: [0] * (len(_DELTA_EDGES) - 1))
    mag_counts: list = field(default_factory=lambda: [0] * (len(_MAG_EDGES) - 1))
    dirs: Counter = field(default_factory=Counter)
    travel_px: float = 0.0
    scroll_v_mag: int = 0
    scroll_h_mag: int = 0
    scroll_axes: Counter = field(default_factory=Counter)

    # keys / buttons
    key_down: Counter = field(default_factory=Counter)
    key_up: Counter = field(default_factory=Counter)
    key_segments: Counter = field(default_factory=Counter)
    btn_down: Counter = field(default_factory=Counter)
    btn_up: Counter = field(default_factory=Counter)
    clicks: Counter = field(default_factory=Counter)      # per button, adjacent pairs
    dbl_clicks: int = 0
    tri_clicks: int = 0
    drags: Counter = field(default_factory=Counter)
    unpaired_down: Counter = field(default_factory=Counter)
    unpaired_up: Counter = field(default_factory=Counter)
    chords: Counter = field(default_factory=Counter)

    # cross-turn held-state anomalies
    dangling_up: int = 0
    redundant_down: int = 0
    held_across_turns: int = 0
    held_at_end: Counter = field(default_factory=Counter)
    typing_under_mod: int = 0

    # typing
    n_type: int = 0
    type_chars: int = 0
    explicit_chars: int = 0
    max_type_len: int = 0
    shift_transitions: int = 0
    type_len: Counter = field(default_factory=Counter)
    typed_strings: Counter = field(default_factory=Counter)
    char_classes: Counter = field(default_factory=Counter)
    typed: dict = field(default_factory=dict)   # segment_id -> reconstructed text

    errors: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def add_segment(self, sid: str, raw_actions: list[Any]) -> None:
        self.n_segments += 1
        held: dict[str, int] = {}      # physical held key/button -> turn index
        chars: list[str] = []          # reconstructed typed text, this segment
        seg_keys: set[str] = set()
        for idx, raw in enumerate(raw_actions):
            turn = parse_turn(raw)
            self.n_turns += 1
            if turn.think_chars:
                self.n_thinking += 1
                self.think_chars += turn.think_chars
            if turn.terminate:
                self.n_terminate += 1
            if turn.n_lines > 1:
                self.n_multiline += 1
            if turn.error is not None:
                self.n_error += 1
                if len(self.errors) < MAX_ERRORS:
                    body = turn.body
                    self.errors.append({
                        "segment_id": sid, "turn": idx, "error": turn.error,
                        "text": body if len(body) <= 300 else body[:297] + "…",
                    })
                continue
            if turn.noop:
                self.n_noop += 1
                continue
            if not turn.prims:
                # A turn stage 04 overwrote with TERMINATE alone: already
                # counted above, and it carries no primitives to aggregate.
                continue
            self._add_turn(sid, idx, turn, held, chars, seg_keys)
        for name in held:               # still down when the segment ended
            self.held_at_end[name] += 1
        for name in seg_keys:
            self.key_segments[name] += 1
        text = "".join(chars)
        if text:
            self.typed[sid] = text

    # ------------------------------------------------------------------ #
    def _add_turn(self, sid: str, idx: int, turn: Turn, held: dict[str, int],
                  chars: list[str], seg_keys: set[str]) -> None:
        prims = turn.prims
        self.n_active += 1
        self.prims_total += len(prims)
        kinds = [p.kind for p in prims]
        counts = Counter(kinds)
        for kind, n in counts.items():
            self.kind_counts[kind] += n
            self.kind_turns[kind] += 1
        # Per-turn counts come from the same helper the numeric turn filter
        # selects on, so a histogram bucket and the filter a click on it sets
        # can never disagree about what "mouse actions in this turn" means.
        per_turn = turn_counts(turn)
        self.prims_per_turn[per_turn["actions"]] += 1
        self.mouse_per_turn[per_turn["mouse"]] += 1
        self.moves_per_turn[per_turn["move"]] += 1
        self.scrolls_per_turn[per_turn["scroll"]] += 1
        self.clicks_per_turn[per_turn["click"]] += 1
        self.types_per_turn[per_turn["type"]] += 1
        shape = _shape(kinds)
        self.shapes[shape] += 1
        examples = self.shape_examples.get(shape)
        if examples is None and len(self.shape_examples) < MAX_SHAPES_WITH_EXAMPLES:
            examples = self.shape_examples[shape] = []
        if examples is not None and len(examples) < MAX_EXAMPLES_PER_SHAPE:
            body = turn.body
            examples.append({
                "segment_id": sid, "turn": idx,
                "text": body if len(body) <= 400 else body[:397] + "…",
            })
        self._add_structure(prims, kinds, counts)
        self._add_motion(prims, counts)
        self._add_inputs(prims, held, chars, seg_keys, idx)

    # ------------------------------------------------------------------ #
    def _add_structure(self, prims: tuple, kinds: list[str],
                       counts: Counter) -> None:
        """Per-turn ordering facts — the questions only an ordered format can
        answer. A "click" here is any button PRESS: what matters for ordering is
        where the discrete barrier sits relative to the motion."""
        motion_at = [i for i, p in enumerate(prims) if p.kind in ("move", "scroll")]
        press_at = [i for i, p in enumerate(prims)
                    if p.kind == "down" and p.input_name in _BUTTON_SET]
        flags: set[str] = set()
        if motion_at and press_at:
            if motion_at[0] < press_at[-1]:
                flags.add("motion_then_click")
            if motion_at[-1] > press_at[0]:
                flags.add("click_then_motion")
            if any(motion_at[0] < p < motion_at[-1] for p in press_at):
                flags.add("move_click_move")
        if len(press_at) >= 2:
            flags.add("multi_click")
        # A drag is a button whose press and release straddle motion.
        for i in press_at:
            name = prims[i].input_name
            for j in range(i + 1, len(prims)):
                if prims[j].kind == "up" and prims[j].input_name == name:
                    if any(prims[k].kind in ("move", "scroll") for k in range(i + 1, j)):
                        flags.add("drag")
                        self.drags[name] += 1
                    break
        # Separate motion runs: motion, something else, motion again.
        runs = 0
        prev_motion = False
        for kind in kinds:
            is_motion = kind in ("move", "scroll")
            if is_motion and not prev_motion:
                runs += 1
            prev_motion = is_motion
        if runs >= 2:
            flags.add("segmented_motion")
        if counts.get("move") and counts.get("scroll"):
            flags.add("move_and_scroll")
        has_type = bool(counts.get("type"))
        if has_type and motion_at:
            flags.add("type_and_motion")
        if has_type and press_at:
            flags.add("type_and_click")
        if has_type and any(p.kind in ("down", "up") and p.input_name in _EDIT_KEYS
                            for p in prims):
            flags.add("type_and_edit")
        if motion_at and len(motion_at) == len(prims):
            flags.add("motion_only")
        if not motion_at:
            flags.add("keys_only")
        for flag in flags:
            self.structure[flag] += 1
        if self._over_grid_bound(counts, len(prims)):
            self.over_grid += 1

    def _over_grid_bound(self, counts: Counter, n_prims: int) -> bool:
        """Whether a turn emits more continuous primitives than its motor grid
        can explain.

        One window spans ``1/fps`` seconds == ``hz/fps`` motor ticks, and a
        move/scroll accumulator flushes on a tick change, a kind switch, or a
        discrete barrier (down/up/type). So per kind the ceiling is
        ``ticks + barriers + other-kind flushes`` — a LOOSE bound on purpose:
        exceeding it means the labels were not built on the grid the manifest
        claims (a different ``--continuous-action-hz``, or a re-derivation at
        another target fps), which is worth flagging rather than averaging away."""
        if self.fps <= 0:
            return False
        ticks = self.hz / self.fps
        n_move, n_scroll = counts.get("move", 0), counts.get("scroll", 0)
        barriers = n_prims - n_move - n_scroll
        return (n_move > ticks + barriers + n_scroll
                or n_scroll > ticks + barriers + n_move)

    # ------------------------------------------------------------------ #
    def _add_motion(self, prims: tuple, counts: Counter) -> None:
        for p in prims:
            if p.kind == "move":
                dx, dy = int(p.dx), int(p.dy)
                self.move_top[f"move({dx},{dy})"] += 1
                self.dx_counts[_bin_index(dx, _DELTA_EDGES)] += 1
                self.dy_counts[_bin_index(dy, _DELTA_EDGES)] += 1
                self.mag_counts[_bin_index(math.hypot(dx, dy), _MAG_EDGES)] += 1
                self.dirs[_dir8(dx, dy)] += 1
                self.travel_px += math.hypot(dx, dy)
            elif p.kind == "scroll":
                dx, dy = int(p.dx), int(p.dy)
                self.scroll_top[f"scroll({dx},{dy})"] += 1
                self.scroll_h_mag += abs(dx)
                self.scroll_v_mag += abs(dy)
                # v3 scroll is a 2-vector (the aggregate format only had a
                # scalar), so which axis is actually used is a real question.
                if dx and dy:
                    self.scroll_axes["both axes"] += 1
                elif dy:
                    self.scroll_axes["vertical only"] += 1
                elif dx:
                    self.scroll_axes["horizontal only"] += 1

    # ------------------------------------------------------------------ #
    def _add_inputs(self, prims: tuple, held: dict[str, int],
                    chars: list[str], seg_keys: set[str], turn_idx: int) -> None:
        """Key/button transitions, chords, click pairing and the typed-text
        reconstruction — all with the segment's held-set carried across turns."""
        for i, p in enumerate(prims):
            if p.kind == "type":
                self.n_type += 1
                text = p.text or ""
                self.type_chars += len(text)
                self.type_len[len(text)] += 1
                self.max_type_len = max(self.max_type_len, len(text))
                self.typed_strings[text] += 1
                self._count_chars(text)
                chars.extend(text)
                if any(n in _NON_SHIFT_MODIFIERS for n in held):
                    # v3 never folds typing under a non-Shift modifier, so this
                    # is either a stale held key or a hand-edited label.
                    self.typing_under_mod += 1
                continue
            if p.kind not in ("down", "up"):
                continue
            name = p.input_name or "?"
            is_button = name in _BUTTON_SET
            if p.kind == "down":
                (self.btn_down if is_button else self.key_down)[name] += 1
                if not is_button:
                    seg_keys.add(name)
                if name in held:
                    self.redundant_down += 1
                mods = sorted({_MOD_CANON[m] for m in held if m in _MOD_CANON})
                if mods and name not in _MODIFIERS:
                    self.chords["+".join([*mods, name])] += 1
                if name in _SHIFT_KEYS:
                    self.shift_transitions += 1
                self._type_char(name, held, chars)
                held[name] = turn_idx
                if is_button and not self._pairs_at(prims, i, name):
                    self.unpaired_down[name] += 1
                elif is_button:
                    self.clicks[name] += 1
            else:
                (self.btn_up if is_button else self.key_up)[name] += 1
                start = held.pop(name, None)
                if start is None:
                    self.dangling_up += 1
                    if is_button:
                        self.unpaired_up[name] += 1
                elif start != turn_idx:
                    self.held_across_turns += 1
        # Adjacent LMB pair runs -> double / triple click (the formatter emits
        # them as repeated pairs; only their adjacency makes them a multi-click).
        i = 0
        while i < len(prims):
            reps = 0
            while self._pairs_at(prims, i + 2 * reps, "LMB"):
                reps += 1
                if reps == 3:
                    break
            if reps >= 2:
                if reps == 2:
                    self.dbl_clicks += 1
                else:
                    self.tri_clicks += 1
                i += 2 * reps
            else:
                i += 1

    @staticmethod
    def _pairs_at(prims: tuple, i: int, name: str) -> bool:
        """True iff ``prims[i:i+2]`` is exactly ``down(name); up(name)``."""
        return (i >= 0 and i + 1 < len(prims)
                and prims[i].kind == "down" and prims[i].input_name == name
                and prims[i + 1].kind == "up" and prims[i + 1].input_name == name)

    def _type_char(self, name: str, held: dict[str, int],
                   chars: list[str]) -> None:
        """Fold one ``down(name)`` into the reconstructed typed text.

        A printable key that reaches this path is typing the formatter could NOT
        collapse (its release fell in another window, or a Shift span never
        balanced) — counting those characters is what makes the collapse ratio
        meaningful. Backspace deletes, Return/Tab are whitespace; every other
        key contributes nothing to the text."""
        if name == "Backspace":
            if chars:
                chars.pop()
            return
        if name in ("Return", "Enter", "NumpadEnter"):
            chars.append("\n")
            return
        if name == "Tab":
            chars.append("\t")
            return
        if name not in _US_PRINTABLE:
            return
        if any(n in _NON_SHIFT_MODIFIERS for n in held):
            return  # Ctrl+C is a chord, not a character
        base, shifted = _US_PRINTABLE[name]
        ch = shifted if any(n in _SHIFT_KEYS for n in held) else base
        self.explicit_chars += 1
        self._count_chars(ch)
        chars.append(ch)

    def _count_chars(self, text: str) -> None:
        for ch in text:
            if ch.isalpha():
                self.char_classes["letters"] += 1
                if ch.isupper():
                    self.char_classes["uppercase"] += 1
            elif ch.isdigit():
                self.char_classes["digits"] += 1
            elif ch == " ":
                self.char_classes["space"] += 1
            elif ch in "\n\t":
                self.char_classes["newline/tab"] += 1
            else:
                self.char_classes["punctuation"] += 1

    # ------------------------------------------------------------------ #
    def result(self, *, mode: str, action_format: "str | None",
               manifest: "dict[str, Any] | None", partial: bool = False) -> dict[str, Any]:
        """Serialize the aggregate for the UI (all ranked lists already cut).

        Two ``_``-prefixed keys carry server-side state the UI never needs (the
        per-segment typed text for search, the per-shape example turns); the
        registry strips them out of the payload on first build."""
        kinds = ["move", "scroll", "down", "up", "type"]
        typed_all = "".join(self.typed.values())
        words = Counter(
            w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'’-]+", typed_all)
            if len(w) > 1
        )
        total_chars = self.type_chars + self.explicit_chars

        manifest_check = None
        if manifest and isinstance(manifest.get("primitive_counts"), dict):
            # Only an EXHAUSTIVE pass is comparable to the builder's own totals;
            # under --limit the rows are still shown but flagged, so a partial
            # sample never reads as a mismatch.
            manifest_check = {
                "partial": partial,
                "rows": [
                    {
                        "kind": k,
                        "recomputed": self.kind_counts.get(k, 0),
                        "manifest": manifest["primitive_counts"].get(k),
                    }
                    for k in kinds
                    if k in manifest["primitive_counts"] or self.kind_counts.get(k)
                ],
            }

        return {
            "mode": mode,
            "action_format": action_format,
            "partial": partial,
            "hz": self.hz,
            "fps": self.fps,
            "grid_budget": round(self.hz / self.fps, 2) if self.fps else None,
            "manifest": manifest,
            "manifest_check": manifest_check,
            "n_segments": self.n_segments,
            "n_turns": self.n_turns,
            "n_active": self.n_active,
            "n_noop": self.n_noop,
            "n_terminate": self.n_terminate,
            "n_thinking": self.n_thinking,
            "n_error": self.n_error,
            "n_multiline": self.n_multiline,
            "think_chars_mean": _mean(self.think_chars, self.n_thinking),
            "prims_total": self.prims_total,
            "prims_per_turn_mean": _mean(self.prims_total, self.n_active),
            "over_grid": self.over_grid,
            "kinds": [
                {
                    "kind": k,
                    "count": self.kind_counts.get(k, 0),
                    "turns": self.kind_turns.get(k, 0),
                    "per_turn": _mean(self.kind_counts.get(k, 0), self.n_active),
                }
                for k in kinds
            ],
            "shapes": [
                {"shape": s, "count": n} for s, n in self.shapes.most_common(TOP_N)
            ],
            "n_shapes": len(self.shapes),
            "structure": [
                {"label": label, "count": self.structure.get(key, 0)}
                for key, label in _STRUCTURE_LABELS
            ],
            "buttons": [
                {
                    "name": b,
                    "down": self.btn_down.get(b, 0),
                    "up": self.btn_up.get(b, 0),
                    "clicks": self.clicks.get(b, 0),
                    "drags": self.drags.get(b, 0),
                    "unpaired_down": self.unpaired_down.get(b, 0),
                    "unpaired_up": self.unpaired_up.get(b, 0),
                }
                for b in _BUTTONS
            ],
            "dbl_clicks": self.dbl_clicks,
            "tri_clicks": self.tri_clicks,
            "keys": [
                {
                    "name": k,
                    "down": n,
                    "up": self.key_up.get(k, 0),
                    "segments": self.key_segments.get(k, 0),
                }
                for k, n in self.key_down.most_common(TOP_N)
            ],
            "n_keys": len(self.key_down),
            "chords": [
                {"combo": c, "count": n} for c, n in self.chords.most_common(TOP_N)
            ],
            "edit_keys": [
                {"name": k, "count": self.key_down.get(k, 0)}
                for k in _EDIT_KEYS if self.key_down.get(k)
            ],
            "anomalies": [
                {"label": "dangling up (no matching down)", "count": self.dangling_up},
                {"label": "redundant down (already held)", "count": self.redundant_down},
                {"label": "key held across a turn boundary", "count": self.held_across_turns},
                {"label": "still held at segment end", "count": sum(self.held_at_end.values())},
                {"label": "type() under a non-Shift modifier", "count": self.typing_under_mod},
                {"label": "turns with >1 action line", "count": self.n_multiline},
                {"label": "turns over the motor-tick bound", "count": self.over_grid},
                {"label": "turns that fail the grammar", "count": self.n_error},
            ],
            "held_at_end": [
                {"name": k, "count": n} for k, n in self.held_at_end.most_common(30)
            ],
            "typing": {
                "n_type": self.n_type,
                "type_chars": self.type_chars,
                "explicit_chars": self.explicit_chars,
                "total_chars": total_chars,
                "collapse_pct": (round(100.0 * self.type_chars / total_chars, 1)
                                 if total_chars else 0.0),
                "chars_per_type": _mean(self.type_chars, self.n_type),
                "max_type_len": self.max_type_len,
                "shift_transitions": self.shift_transitions,
                "n_typing_segments": len(self.typed),
            },
            "typed_strings": [
                {"text": t, "count": n}
                for t, n in self.typed_strings.most_common(TOP_N)
            ],
            "typed_words": [
                {"word": w, "count": n} for w, n in words.most_common(TOP_N)
            ],
            "char_classes": [
                {"label": k, "count": n} for k, n in self.char_classes.most_common()
            ],
            "moves": {
                "count": self.kind_counts.get("move", 0),
                "travel_px": round(self.travel_px),
                "top": [
                    {"prim": p, "count": n} for p, n in self.move_top.most_common(TOP_N)
                ],
                "dirs": [
                    {"label": _DIR_LABELS[i], "count": self.dirs.get(i, 0)}
                    for i in range(8)
                ],
                "dx_hist": _hist_spec(_DELTA_EDGES, self.dx_counts, highlight_at=[-100, 100]),
                "dy_hist": _hist_spec(_DELTA_EDGES, self.dy_counts, highlight_at=[-100, 100]),
                "mag_hist": _hist_spec(_MAG_EDGES, self.mag_counts, highlight_at=[100]),
            },
            "scrolls": {
                "count": self.kind_counts.get("scroll", 0),
                "v_mag": self.scroll_v_mag,
                "h_mag": self.scroll_h_mag,
                "top": [
                    {"prim": p, "count": n} for p, n in self.scroll_top.most_common(TOP_N)
                ],
                "axes": [
                    {"label": k, "count": n}
                    for k, n in self.scroll_axes.most_common()
                ],
            },
            "prims_per_turn_hist": _int_hist(self.prims_per_turn, _COUNT_BUCKETS,
                                             field="actions"),
            "mouse_per_turn_hist": _int_hist(self.mouse_per_turn, _COUNT_BUCKETS,
                                             field="mouse"),
            "moves_per_turn_hist": _int_hist(self.moves_per_turn, _COUNT_BUCKETS,
                                             field="move"),
            "scrolls_per_turn_hist": _int_hist(self.scrolls_per_turn, _COUNT_BUCKETS,
                                               field="scroll"),
            "clicks_per_turn_hist": _int_hist(self.clicks_per_turn, _COUNT_BUCKETS,
                                              field="click"),
            "types_per_turn_hist": _int_hist(self.types_per_turn, _COUNT_BUCKETS,
                                             field="type"),
            "type_len_hist": _int_hist(self.type_len, _LEN_BUCKETS),
            "errors": self.errors,
            "top_typing_segments": [
                {"segment_id": sid, "count": len(t)}
                for sid, t in sorted(self.typed.items(),
                                     key=lambda kv: len(kv[1]), reverse=True)[:25]
            ],
            "_typed": self.typed,
            "_shape_examples": self.shape_examples,
        }


# --------------------------------------------------------------------------- #
# Dataset access
# --------------------------------------------------------------------------- #
def _collect_segments(ds: Any) -> list[tuple[str, list[Any]]]:
    """``[(segment_id, [assistant_turn_text, ...]), ...]`` in dataset order.

    A frames-master store is keylog-free (no actions at all), so it yields
    nothing; every other loader exposes ``.segments`` whose frames carry the
    verbatim assistant text in ``action``."""
    if isinstance(ds, V.FramesMasterDataset):
        return []
    return [
        (sid, [f.get("action") for f in seg.frames])
        for sid, seg in ds.segments.items()
    ]


def _cache_segments(ds: Any) -> list[tuple[str, list[Any]]]:
    segs = getattr(ds, "_ordered_segments", None)
    if segs is None:
        segs = _collect_segments(ds)
        ds._ordered_segments = segs  # type: ignore[attr-defined]
    return segs


def _load_manifest(path: Path) -> "dict[str, Any] | None":
    """The dataset's ``manifest.json`` (stage 04 / 06), or None.

    Worth reading even though the aggregate does not need it: stage 04 records
    the formatter's OWN ``primitive_counts`` and the label policy's dead-zone
    counters, which the UI shows next to the recomputed numbers."""
    p = path.expanduser()
    candidates = [p / "manifest.json"] if p.is_dir() else [
        p.parent / "manifest.json", p.parent.parent / "manifest.json"
    ]
    for cand in candidates:
        try:
            if cand.is_file():
                data = json.loads(cand.read_text())
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _dataset_format(ds: Any, manifest: "dict[str, Any] | None") -> "str | None":
    """The dataset's declared action format: the conversation rows' own field
    first (authoritative per row), else the manifest's."""
    for seg in getattr(ds, "segments", {}).values():
        fmt = getattr(seg, "action_format", None)
        if fmt:
            return str(fmt)
    if manifest and manifest.get("action_format"):
        return str(manifest["action_format"])
    return None


def _dataset_fps(ds: Any, manifest: "dict[str, Any] | None") -> float:
    for seg in getattr(ds, "segments", {}).values():
        fps = getattr(seg, "target_fps", None)
        if fps:
            try:
                return float(fps)
            except (TypeError, ValueError):
                break
    try:
        return float((manifest or {}).get("target_fps") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_distribution(ds: Any, path: Path) -> dict[str, Any]:
    """Aggregate one dataset in a single streaming pass over its turns."""
    manifest = _load_manifest(path)
    fmt = _dataset_format(ds, manifest)
    try:
        hz = float((manifest or {}).get("continuous_action_hz") or 10.0)
    except (TypeError, ValueError):
        hz = 10.0
    agg = Aggregator(hz=hz, fps=_dataset_fps(ds, manifest))
    for sid, actions in _cache_segments(ds):
        agg.add_segment(sid, actions)
    partial = bool(getattr(ds, "_limited", False)) or DATASET_SAMPLE_LIMIT is not None
    dist = agg.result(mode=getattr(ds, "mode", "frame_records"),
                      action_format=fmt, manifest=manifest, partial=partial)
    dist["note"] = _format_note(dist, fmt)
    if not dist["n_active"]:
        # Nothing parsed as an ordered program (a canonical / tool-call dataset):
        # the motor grid is meaningless here, so don't advertise a tick budget.
        dist["grid_budget"] = None
    return dist


def _format_note(dist: dict[str, Any], fmt: "str | None") -> "str | None":
    """A banner when the dataset is not the format this viewer is for.

    The parse-failure rate is the ground truth (a canonical or computer_use
    dataset fails the ordered grammar on nearly every turn); the declared
    format only sharpens the wording."""
    turns = dist["n_turns"]
    if not turns:
        return ("no assistant turns in this dataset — a stage-01a frames-master "
                "store is keylog-free; open a stage-04 conversations or stage-06 "
                "inline-records dataset instead")
    fail = dist["n_error"] / turns
    if fail > 0.5:
        return (f"{fail:.0%} of turns do not parse as ordered_events primitives"
                + (f" — this dataset declares action_format={fmt!r}" if fmt else "")
                + ". For the aggregate `<dx> <dy> <scroll> ; +Key -Key` grammar use "
                  "visualize_action_distribution.py; for Qwen tool-call SFT use the "
                  "frame viewer.")
    if fmt and fmt not in ORDERED_FORMATS:
        return (f"dataset declares action_format={fmt!r} but its turns parse as "
                "ordered primitives — reporting them as such")
    n_type = next((k["count"] for k in dist["kinds"] if k["kind"] == "type"), 0)
    if fmt == V2_FORMAT or (n_type == 0 and dist["n_active"]):
        return (f"{fmt or 'ordered_events_v2'}: no `type(\"…\")` primitives — the "
                "typing cards below are the v3-only view and stay empty")
    if fail:
        return f"{dist['n_error']} of {turns} turns fail the grammar (see the parse-errors card)"
    return None


def search_turns(ds: Any, typed: dict[str, str], query: str,
                 tf: TurnFilter = NO_TURN_FILTER) -> dict[str, Any]:
    """Which turns match ``query`` (case-insensitive substring) and ``tf``.

    The substring reaches two independent coverages, because in v3 they answer
    different questions: the ACTION LINE (so ``move(-100,`` / ``down(LMB)`` /
    ``type("`` reach the primitives verbatim, thinking blocks excluded) and the
    reconstructed TYPED TEXT per segment (so ``gmail`` finds what was typed,
    however the formatter happened to spell it). ``tf`` adds the per-turn count
    bounds — "turns with ≥10 actions", "turns that scroll but never move" — and
    the two AND together. Examples are verbatim turns, ready to look up in
    visualize_frame_records.py.

    The count bounds need the turn PARSED, so an active ``tf`` costs a parse per
    candidate turn (the same work the initial aggregate did once); the
    substring-only path stays a plain scan. The typed-text coverage is a
    per-SEGMENT reconstruction and therefore ignores ``tf`` — it is reported
    only for the substring, and the UI labels it as such."""
    q = query.strip().lower()
    segs = _cache_segments(ds)
    n_turns_total = sum(len(a) for _, a in segs)
    out: dict[str, Any] = {
        "query": query, "filter": tf.label(), "n_turns": 0,
        "turns_total": n_turns_total,
        "n_segments": 0, "segments_total": len(segs),
        "typed_segments": 0, "typed_hits": 0,
        "top_segments": [], "examples": [], "mix": [],
    }
    if not q and not tf:
        return out
    n_turns = 0
    per_seg: list[tuple[str, int]] = []
    mix: Counter = Counter()
    n_unparsed = 0
    for sid, actions in segs:
        hits = 0
        for idx, raw in enumerate(actions):
            body = V._THINK_RE.sub("", "" if raw is None else str(raw)).strip()
            if q and q not in body.lower():
                continue
            if tf:
                turn = parse_turn(raw)
                if turn.error is not None:
                    n_unparsed += 1
                    continue
                counts = turn_counts(turn)
                if not tf.accepts(counts):
                    continue
                for key, n in counts.items():
                    mix[key] += n
            hits += 1
            if len(out["examples"]) < 25:
                out["examples"].append({
                    "segment_id": sid, "turn": idx,
                    "text": body if len(body) <= 400 else body[:397] + "…",
                })
        if hits:
            n_turns += hits
            per_seg.append((sid, hits))
    per_seg.sort(key=lambda kv: kv[1], reverse=True)
    typed_hits = 0
    typed_segments = 0
    if q:
        for text in typed.values():
            c = text.lower().count(q)
            if c:
                typed_hits += c
                typed_segments += 1
    out.update({
        "n_turns": n_turns,
        "n_segments": len(per_seg),
        "typed_hits": typed_hits,
        "typed_segments": typed_segments,
        "top_segments": [{"segment_id": s, "count": c} for s, c in per_seg[:25]],
        # What the matched turns are made of — the mix inside the selection,
        # which is the question a count filter is usually asked in service of.
        "mix": [
            {"key": key, "count": mix.get(key, 0),
             "per_turn": _mean(mix.get(key, 0), n_turns)}
            for key, _lab, _desc in _TURN_FIELDS if mix.get(key)
        ] if tf else [],
        "unparsed": n_unparsed,
    })
    return out


# --------------------------------------------------------------------------- #
# Registry / lazy build
# --------------------------------------------------------------------------- #
def register_datasets(paths: list[str]) -> None:
    for raw in paths:
        p = Path(raw).expanduser()
        name = p.name or str(p)
        base, k = name, 2
        while name in DATASETS:
            name, k = f"{base}#{k}", k + 1
        DATASETS[name] = {"path": p, "mode": V.detect_mode(p), "obj": None,
                          "dist": None}


def get_distribution(name: str) -> "dict[str, Any] | None":
    """Build (or return cached) aggregate for a registered dataset."""
    entry = DATASETS.get(name)
    if entry is None:
        return None
    if entry["dist"] is None:
        # The frame viewer's loaders take WHICH samples to read as a Sampling
        # object (it also offers a random draw); this viewer only ever aggregates
        # the first --limit samples, so pass that mode explicitly.
        sampling = V.Sampling("first", DATASET_SAMPLE_LIMIT)
        try:
            ds = V._build_dataset(entry["path"], sampling)
        except SystemExit as exc:  # empty / not-yet-generated dataset dir
            raise RuntimeError(str(exc)) from exc
        entry["obj"] = ds
        dist = build_distribution(ds, entry["path"])
        # Search/example state stays on the server: the typed text and the kept
        # example turns would multiply the payload for no UI benefit (examples
        # are fetched per shape on click).
        entry["typed"] = dist.pop("_typed", None) or {}
        entry["shape_examples"] = dist.pop("_shape_examples", None) or {}
        entry["dist"] = dist
    return entry["dist"]


def get_dataset_obj(name: str) -> "Any | None":
    entry = DATASETS.get(name)
    if entry is None:
        return None
    if entry["obj"] is None:
        get_distribution(name)
    return entry["obj"]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a: Any) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _dsname(self, q: dict[str, list[str]]) -> str:
        vals = q.get("ds")
        if vals and vals[0]:
            return vals[0]
        return next(iter(DATASETS), "")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route, q = parsed.path, parse_qs(parsed.query)
        try:
            if route == "/":
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif route == "/api/datasets":
                names = list(DATASETS)
                self._send_json({
                    "datasets": [{"name": n, "mode": DATASETS[n]["mode"]} for n in names],
                    "default": names[0] if names else None,
                    "grammar": ORDERED_EVENTS_V3_GRAMMAR.strip("\n"),
                    "format": V3_FORMAT,
                    "turn_fields": [
                        {"key": k, "label": lab, "desc": desc}
                        for k, lab, desc in _TURN_FIELDS
                    ],
                })
            elif route == "/api/dist":
                name = self._dsname(q)
                if name not in DATASETS:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                    return
                try:
                    self._send_json(get_distribution(name))
                except Exception as exc:  # noqa: BLE001 — report, keep UI alive
                    self._send_json({"error": f"failed to load {name!r}: {exc}"}, 500)
            elif route == "/api/search":
                name = self._dsname(q)
                if name not in DATASETS:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                    return
                ds = get_dataset_obj(name)   # builds + caches the aggregate too
                typed = DATASETS[name].get("typed") or {}
                self._send_json(search_turns(ds, typed, (q.get("q") or [""])[0],
                                             TurnFilter.from_query(q)))
            elif route == "/api/examples":
                name = self._dsname(q)
                if name not in DATASETS:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                    return
                get_distribution(name)
                shape = (q.get("shape") or [""])[0]
                examples = DATASETS[name].get("shape_examples") or {}
                self._send_json({"shape": shape, "examples": examples.get(shape, [])})
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001
            self._send(500, f"{type(exc).__name__}: {exc}".encode(), "text/plain")


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ordered_events_v3 action distribution</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.45 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
         background:#14161a; color:#d7dae0; }
  header { position:sticky; top:0; z-index:10; padding:8px 14px; border-bottom:1px solid #2a2e36;
           display:flex; gap:10px; align-items:center; flex-wrap:wrap; background:#191c21; }
  header .title { font-weight:700; color:#e8eef7; }
  header .title small { color:#7fd6a2; font-weight:400; }
  select,input,button { background:#22262e; color:#d7dae0; border:1px solid #343a44;
                  border-radius:4px; padding:4px 8px; font:inherit; }
  select,button { cursor:pointer; }
  button:hover { border-color:#5b9dd9; }
  button.on { border-color:#5b9dd9; color:#e8eef7; }
  .hint { color:#6b7280; font-size:12px; }
  #mode { margin-left:auto; }
  main { padding:14px; max-width:1560px; margin:0 auto; }
  #err { color:#f7a6a6; padding:6px 0; }
  #loading { color:#8b93a1; padding:10px 0; }
  #note { display:none; margin-bottom:12px; padding:8px 12px; border-radius:6px;
          background:#241f18; border:1px solid #5a4526; color:#f0c98a; }
  #note.show { display:block; }
  #grammar { display:none; margin-bottom:14px; background:#171b22; border:1px solid #2a2e36;
             border-radius:6px; padding:10px 12px; white-space:pre; overflow:auto; color:#9fb4cc; font-size:12px; }
  #grammar.show { display:block; }

  .tiles { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
  .tile { background:#191c21; border:1px solid #2a2e36; border-radius:6px; padding:8px 12px; min-width:118px; }
  .tile .k { color:#8b93a1; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .tile .v { color:#e8eef7; font-size:19px; font-weight:700; margin-top:2px; }
  .tile .s { color:#7fd6a2; font-size:11px; }
  .tile.bad .v { color:#f0a0a0; }

  #searchbar { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  #q { min-width:320px; flex:1; }
  #numbar { display:flex; gap:8px; align-items:center; margin-bottom:10px; flex-wrap:wrap; }
  #numbar .cap { color:#8b93a1; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .nf { display:inline-flex; align-items:center; gap:3px; background:#171b22; border:1px solid #2a2e36;
        border-radius:5px; padding:2px 6px; }
  .nf.on { border-color:#5b9dd9; }
  .nf b { color:#aeb6c2; font-weight:400; font-size:12px; }
  .nf input { width:46px; padding:1px 4px; text-align:center; font-size:12px; }
  .nf input::-webkit-outer-spin-button, .nf input::-webkit-inner-spin-button { -webkit-appearance:none; margin:0; }
  .nf input[type=number] { -moz-appearance:textfield; }
  #searchres { background:#171b22; border:1px solid #2a2e36; border-radius:6px; padding:10px 12px;
               margin-bottom:16px; display:none; }
  #searchres.show { display:block; }
  #searchres .big { font-size:15px; color:#e8eef7; }
  #searchres .big b.hl { color:#f5b544; }
  .seglist { margin-top:8px; display:flex; flex-direction:column; gap:2px; max-height:230px; overflow:auto; }
  .segrow { display:grid; grid-template-columns:1fr 70px; gap:8px; }
  .segrow .bar { background:#20242b; border-radius:3px; position:relative; overflow:hidden; }
  .segrow .bar > i { position:absolute; inset:0 auto 0 0; background:#2d4a75; }
  .segrow .lab { position:relative; padding:1px 6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#cbd3df; }
  .segrow .cnt { text-align:right; color:#8fc4f2; }
  .exlist { margin-top:8px; display:flex; flex-direction:column; gap:4px; max-height:320px; overflow:auto; }
  .ex { background:#14181e; border:1px solid #262b33; border-radius:4px; padding:5px 8px; }
  .ex .who { color:#6b7280; font-size:11px; }
  .ex .txt { color:#cfe0f2; white-space:pre-wrap; word-break:break-word; }

  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(430px, 1fr)); gap:14px; }
  .card { background:#191c21; border:1px solid #2a2e36; border-radius:8px; padding:10px 12px; }
  .card.wide { grid-column:span 2; }
  .card h3 { margin:0 0 8px; font-size:12px; color:#aeb6c2; text-transform:uppercase; letter-spacing:.05em;
             display:flex; align-items:baseline; gap:8px; }
  .card h3 .sub { color:#6b7280; font-size:11px; text-transform:none; letter-spacing:0; font-weight:400; }

  .blist { display:flex; flex-direction:column; gap:3px; max-height:340px; overflow:auto; }
  .brow { display:grid; grid-template-columns:1fr 96px; gap:8px; align-items:center; }
  .brow.clickable { cursor:pointer; }
  .brow.clickable:hover .lab { color:#fff; }
  .brow .bar { height:18px; background:#20242b; border-radius:3px; position:relative; overflow:hidden; }
  .brow .bar > i { position:absolute; inset:0 auto 0 0; background:#3564a0; }
  .brow.hl .bar > i { background:#c08a2a; }
  .brow.good .bar > i { background:#2d6a45; }
  .brow.bad .bar > i { background:#8d3b3b; }
  .brow .lab { position:relative; padding:0 7px; line-height:18px; white-space:nowrap; overflow:hidden;
               text-overflow:ellipsis; color:#cbd3df; font-size:12px; }
  .brow .cnt { text-align:right; color:#8fc4f2; font-size:12px; }
  .brow .cnt small { color:#6b7280; }

  table.kv { width:100%; border-collapse:collapse; font-size:12px; }
  table.kv td { padding:2px 4px; border-bottom:1px solid #22262e; }
  table.kv td.k { color:#8b93a1; }
  table.kv td.v { text-align:right; color:#cfe0f2; }
  table.kv td.v.mm { color:#f0a0a0; }
  table.kv td.v.ok { color:#7fd6a2; }

  .hist { width:100%; height:150px; }
  .hist .bar { fill:#3564a0; }
  .hist .bar.hl { fill:#c08a2a; }
  .histx { display:flex; justify-content:space-between; color:#6b7280; font-size:10px; margin-top:2px; }
  .empty { color:#6b7280; font-style:italic; }
</style>
</head><body>
<header>
  <span class="title">ordered action distribution <small id="fmt"></small></span>
  <select id="ds"></select>
  <button id="gbtn">grammar</button>
  <span id="mode" class="hint"></span>
</header>
<main>
  <div id="err"></div>
  <div id="loading">select a dataset…</div>
  <div id="note"></div>
  <pre id="grammar"></pre>
  <div id="content" style="display:none">
    <div class="tiles" id="tiles"></div>
    <div id="searchbar">
      <input id="q" placeholder='filter turns: down(LMB) · move(-100, · type(" · KeyEnter — substring over the action line + typed text'>
      <button id="qgo">count</button>
      <button id="qclear">clear</button>
      <span class="hint">click a bar to filter by it · click a program shape for examples</span>
    </div>
    <div id="numbar"></div>
    <div id="searchres"></div>
    <div class="grid" id="grid"></div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
let CUR=null, DIST=null, GRAMMAR='', FIELDS=[];
function fmt(n){ return (n==null?0:n).toLocaleString(); }
function pct(a,b){ return b? (100*a/b).toFixed(1)+'%' : '0%'; }
function esc(s){ return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

// ---- per-turn count filter (min/max per field, ANDed with the substring) ----
function renderNumBar(){
  const chips = FIELDS.map(f=>
    `<span class="nf" data-f="${f.key}" title="${esc(f.label)} per turn — ${esc(f.desc)}">`+
    `<b>${esc(f.label)}</b>/turn`+
    `<input type="number" min="0" step="1" data-kind="min" data-f="${f.key}" placeholder="min">`+
    `<input type="number" min="0" step="1" data-kind="max" data-f="${f.key}" placeholder="max">`+
    `</span>`).join('');
  $('#numbar').innerHTML = `<span class="cap">per turn</span>${chips}`+
    `<span class="hint">inclusive; blank = unbounded · e.g. actions min 10, or scroll min 1 + move max 0</span>`;
  $('#numbar').querySelectorAll('input').forEach(el=>{
    el.onchange = ()=>{ markNumBar(); runSearch($('#q').value); };
    el.addEventListener('keydown', e=>{ if(e.key==='Enter'){ markNumBar(); runSearch($('#q').value); } });
  });
}
function markNumBar(){
  $('#numbar').querySelectorAll('.nf').forEach(chip=>{
    const on=[...chip.querySelectorAll('input')].some(i=>i.value.trim()!=='');
    chip.classList.toggle('on',on);
  });
}
// "min_actions=10&max_move=0" for the fields that carry a bound.
function numParams(){
  const out=[];
  $('#numbar').querySelectorAll('input').forEach(el=>{
    const v=el.value.trim();
    if(v!=='') out.push(`${el.dataset.kind}_${el.dataset.f}=${encodeURIComponent(v)}`);
  });
  return out;
}
function anyNum(){ return numParams().length>0; }
function clearNum(){ $('#numbar').querySelectorAll('input').forEach(el=>el.value=''); markNumBar(); }
// Set one field's bounds (a histogram bucket click) and leave the others alone.
function setNum(field, lo, hi){
  const chip=$(`#numbar .nf[data-f="${field}"]`);
  if(!chip) return;
  chip.querySelector('input[data-kind=min]').value = (lo==null? '' : lo);
  chip.querySelector('input[data-kind=max]').value = (hi==null? '' : hi);
  markNumBar();
}

async function loadDatasets(){
  const d = await (await fetch('/api/datasets')).json();
  GRAMMAR = d.grammar||''; $('#grammar').textContent = GRAMMAR;
  $('#fmt').textContent = d.format||'';
  FIELDS = d.turn_fields||[]; renderNumBar();
  const sel = $('#ds'); sel.innerHTML='';
  for(const ds of d.datasets){
    const o=document.createElement('option'); o.value=ds.name; o.textContent=`${ds.name}  [${ds.mode}]`;
    sel.appendChild(o);
  }
  sel.onchange = ()=> selectDataset(sel.value);
  if(d.default) selectDataset(d.default);
}

async function selectDataset(name){
  CUR=name; DIST=null;
  $('#err').textContent=''; $('#content').style.display='none'; $('#note').className='';
  $('#loading').style.display=''; $('#loading').textContent=`aggregating ${name} … (first load builds the dataset)`;
  $('#mode').textContent=''; $('#searchres').className=''; $('#q').value=''; clearNum();
  try{
    const d = await (await fetch('/api/dist?ds='+encodeURIComponent(name))).json();
    if(d.error){ $('#loading').style.display='none'; $('#err').textContent=d.error; return; }
    DIST=d; render(d);
  }catch(e){ $('#loading').style.display='none'; $('#err').textContent=String(e); }
}

function render(d){
  $('#loading').style.display='none'; $('#content').style.display='';
  const mf = d.action_format? ` · ${d.action_format}` : '';
  const hz = d.grid_budget!=null? ` · ${d.hz}Hz / ${d.fps}fps = ${d.grid_budget} ticks/turn` : '';
  $('#mode').textContent = `${d.mode}${mf} · ${fmt(d.n_segments)} segments · ${fmt(d.n_turns)} turns${hz}`
    + (d.partial? ' · SAMPLE (--limit)' : '');
  if(d.note){ $('#note').className='show'; $('#note').textContent=d.note; }
  if(!d.n_turns){ $('#grid').innerHTML=''; $('#tiles').innerHTML=''; return; }
  renderTiles(d); renderGrid(d);
}

function renderTiles(d){
  const t=d.typing;
  const tiles=[
    ['segments', fmt(d.n_segments), ''],
    ['turns', fmt(d.n_turns), ''],
    ['active', fmt(d.n_active), pct(d.n_active,d.n_turns)+' of turns'],
    ['NO_OP', fmt(d.n_noop), pct(d.n_noop,d.n_turns)],
    ['primitives', fmt(d.prims_total), d.prims_per_turn_mean+' per active turn'],
    ['type()', fmt(t.n_type), fmt(t.type_chars)+' chars'],
    ['typing collapsed', t.collapse_pct+'%', fmt(t.explicit_chars)+' chars still explicit'],
    ['TERMINATE', fmt(d.n_terminate), pct(d.n_terminate,d.n_turns)],
    ['thinking', fmt(d.n_thinking), d.n_thinking? d.think_chars_mean+' chars avg':''],
    ['parse errors', fmt(d.n_error), pct(d.n_error,d.n_turns), d.n_error?'bad':''],
  ];
  $('#tiles').innerHTML = tiles.map(([k,v,s,cls])=>
    `<div class="tile ${cls||''}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s||''}</div></div>`).join('');
}

// items: [{lab, cnt, sub?, cls?, token?, shape?}]
function barCard(title, sub, items, opts){
  opts=opts||{};
  const max = items.reduce((m,it)=>Math.max(m,it.cnt),0) || 1;
  const rows = items.map(it=>{
    const w=(100*it.cnt/max).toFixed(1);
    const cls=(it.cls?' '+it.cls:'')+((it.token!=null||it.shape!=null)?' clickable':'');
    const attr=(it.token!=null? ` data-token="${encodeURIComponent(it.token)}"`:'')
             + (it.shape!=null? ` data-shape="${encodeURIComponent(it.shape)}"`:'');
    const s=it.sub? ` <small>${esc(it.sub)}</small>`:'';
    return `<div class="brow${cls}"${attr}><div class="bar"><i style="width:${w}%"></i>`+
           `<span class="lab">${esc(it.lab)}</span></div><div class="cnt">${fmt(it.cnt)}${s}</div></div>`;
  }).join('');
  const body = items.length? `<div class="blist">${rows}</div>` : `<div class="empty">${opts.empty||'none'}</div>`;
  return `<div class="card${opts.wide?' wide':''}"><h3>${title}${sub?` <span class="sub">${esc(sub)}</span>`:''}</h3>${body}</div>`;
}

function histCard(title, sub, hist){
  const n=hist.counts.length, max=Math.max(1,...hist.counts), W=100/n;
  // A per-turn-count histogram (hist.field) doubles as a filter control: the
  // bucket's own bounds are the min/max a click sets.
  const f=hist.field||null;
  const bars=hist.counts.map((c,i)=>{
    const h=100*c/max, hl=hist.highlight.includes(i)?' hl':'';
    const b=(hist.bounds&&hist.bounds[i])||null;
    const attr=f&&b? ` data-field="${f}" data-lo="${b[0]==null?'':b[0]}" data-hi="${b[1]==null?'':b[1]}" style="cursor:pointer"`:'';
    const tip=`${esc(hist.labels[i])}: ${fmt(c)}`+(f? ` — click to filter ${esc(f)}/turn`:'');
    return `<rect class="bar${hl}" x="${(i*W).toFixed(2)}%" y="${(100-h).toFixed(2)}%" `+
           `width="${(W*0.86).toFixed(2)}%" height="${h.toFixed(2)}%"${attr}>`+
           `<title>${tip}</title></rect>`;
  }).join('');
  const marks=new Set([0,Math.floor(n/2),n-1,...hist.highlight]);
  const xl=hist.labels.map((l,i)=> marks.has(i)? `<span>${esc(l)}</span>`:'').join('');
  return `<div class="card"><h3>${title}${sub?` <span class="sub">${esc(sub)}</span>`:''}</h3>`+
         `<svg class="hist" preserveAspectRatio="none">${bars}</svg><div class="histx">${xl}</div></div>`;
}

// rows: [[key, value, cls?]]
function kvCard(title, sub, rows, wide){
  const body = rows.length
    ? `<table class="kv">${rows.map(([k,v,c])=>`<tr><td class="k">${esc(k)}</td><td class="v ${c||''}">${esc(String(v))}</td></tr>`).join('')}</table>`
    : `<div class="empty">none</div>`;
  return `<div class="card${wide?' wide':''}"><h3>${title}${sub?` <span class="sub">${esc(sub)}</span>`:''}</h3>${body}</div>`;
}

function exCard(title, sub, items){
  const body = items.length
    ? `<div class="exlist">${items.map(e=>`<div class="ex"><div class="who">${esc(e.segment_id)} · turn ${e.turn}${e.error?` · <span style="color:#f0a0a0">${esc(e.error)}</span>`:''}</div><div class="txt">${esc(e.text)}</div></div>`).join('')}</div>`
    : `<div class="empty">none</div>`;
  return `<div class="card wide"><h3>${title}${sub?` <span class="sub">${esc(sub)}</span>`:''}</h3>${body}</div>`;
}

function renderGrid(d){
  const c=[], t=d.typing, mv=d.moves, sc=d.scrolls, A=d.n_active;
  // --- what the programs look like -----------------------------------------
  c.push(barCard('primitive mix','count · turns containing · per active turn',
    d.kinds.map(k=>({lab:k.kind, cnt:k.count, sub:`${pct(k.turns,A)} turns`,
                     token:k.kind+'(', cls:k.kind==='type'?'good':''}))));
  c.push(barCard('program shapes',`${fmt(d.n_shapes)} distinct · consecutive repeats folded (move+)`,
    d.shapes.map(s=>({lab:s.shape, cnt:s.count, sub:pct(s.count,A), shape:s.shape})),
    {wide:true}));
  c.push(barCard('ordering structure','share of active turns — what the aggregate format could not express',
    d.structure.map(s=>({lab:s.label, cnt:s.count, sub:pct(s.count,A)}))));
  c.push(histCard('primitives per turn','active turns · click a bucket to filter turns by it', d.prims_per_turn_hist));
  // --- typing (v3's reason for existing) -----------------------------------
  c.push(kvCard('typing collapse','type("…") vs printable down/up the formatter could not fold', [
    ['type() primitives', fmt(t.n_type)],
    ['chars inside type()', fmt(t.type_chars)],
    ['chars as explicit key pairs', fmt(t.explicit_chars)],
    ['collapsed share', t.collapse_pct+'%', t.collapse_pct>=80?'ok':(t.collapse_pct<50?'mm':'')],
    ['chars per type()', t.chars_per_type],
    ['longest type()', fmt(t.max_type_len)+' chars'],
    ['Shift transitions rendered', fmt(t.shift_transitions)],
    ['segments with typed text', fmt(t.n_typing_segments)+' / '+fmt(d.n_segments)],
  ]));
  c.push(histCard('characters per type()','type() primitives', d.type_len_hist));
  c.push(barCard('top typed strings','exact type() payloads',
    d.typed_strings.map(x=>({lab:JSON.stringify(x.text), cnt:x.count,
                             token:'type("'+x.text+'")'})),
    {empty:'no type() primitives (v2 dataset?)'}));
  c.push(barCard('typed words','from the reconstructed per-segment text',
    d.typed_words.map(x=>({lab:x.word, cnt:x.count}))));
  c.push(barCard('typed character classes','type() payloads + explicit printable keys',
    d.char_classes.map(x=>({lab:x.label, cnt:x.count, sub:pct(x.count,t.total_chars)}))));
  // --- keys / buttons ------------------------------------------------------
  c.push(barCard('mouse buttons','down() · clk = adjacent down/up pair · drag = motion in between',
    d.buttons.map(b=>({lab:b.name, cnt:b.down, cls:'good',
      sub:`${fmt(b.clicks)} clk`+(b.drags?` ${fmt(b.drags)} drag`:'')+(b.unpaired_down?` ${fmt(b.unpaired_down)} held`:''),
      token:'down('+b.name+')'}))
      .concat([{lab:'double clicks', cnt:d.dbl_clicks},{lab:'triple clicks', cnt:d.tri_clicks}])));
  c.push(barCard('keys pressed',`down() · ${fmt(d.n_keys)} distinct`,
    d.keys.map(k=>({lab:k.name, cnt:k.down, sub:`${fmt(k.segments)} seg`, token:'down('+k.name+')'}))));
  c.push(barCard('chords','a key/button pressed while a modifier is held (tracked across turns)',
    d.chords.map(x=>({lab:x.combo, cnt:x.count}))));
  c.push(barCard('editing keys','never folded into type() — backtracking and submitting stay their own turns',
    d.edit_keys.map(x=>({lab:x.name, cnt:x.count, token:'down('+x.name+')'}))));
  // --- motion --------------------------------------------------------------
  c.push(barCard('top move primitives',`${fmt(mv.count)} moves · ${fmt(mv.travel_px)} px travelled`,
    mv.top.map(x=>({lab:x.prim, cnt:x.count, token:x.prim}))));
  c.push(barCard('top scroll primitives',`${fmt(sc.count)} scrolls · v ${fmt(sc.v_mag)} · h ${fmt(sc.h_mag)}`,
    sc.top.map(x=>({lab:x.prim, cnt:x.count, token:x.prim}))));
  c.push(barCard('scroll axes','v3 scroll is a 2-vector',
    sc.axes.map(x=>({lab:x.label, cnt:x.count, sub:pct(x.count,sc.count)}))));
  c.push(barCard('move direction','8-way, per move primitive',
    mv.dirs.map(x=>({lab:x.label, cnt:x.count, sub:pct(x.count,mv.count)}))));
  c.push(histCard('move dx','per primitive (motor tick); ±100 highlighted', mv.dx_hist));
  c.push(histCard('move dy','per primitive (motor tick); ±100 highlighted', mv.dy_hist));
  c.push(histCard('|move| magnitude','px per primitive; 100 highlighted', mv.mag_hist));
  c.push(histCard('mouse actions per turn','move + scroll + button down/up', d.mouse_per_turn_hist));
  c.push(histCard('moves per turn', d.grid_budget!=null? `motor budget ≈ ${d.grid_budget} ticks`:'active turns', d.moves_per_turn_hist));
  c.push(histCard('scrolls per turn','active turns', d.scrolls_per_turn_hist));
  c.push(histCard('clicks per turn','button presses — down(LMB/RMB/MMB)', d.clicks_per_turn_hist));
  c.push(histCard('type() per turn','active turns', d.types_per_turn_hist));
  // --- integrity -----------------------------------------------------------
  c.push(barCard('cross-turn held state','press/release bookkeeping no per-window formatter can see',
    d.anomalies.map(x=>({lab:x.label, cnt:x.count, cls:x.count?'bad':''}))));
  if(d.held_at_end.length)
    c.push(barCard('still held at segment end','a press whose release never arrived',
      d.held_at_end.map(x=>({lab:x.name, cnt:x.count, cls:'bad'}))));
  if(d.manifest_check)
    c.push(kvCard('manifest cross-check',
      d.manifest_check.partial
        ? 'recomputed (this SAMPLE) vs the builder\'s totals for the WHOLE dataset — not comparable under --limit'
        : 'recomputed vs the formatter\'s own primitive_counts',
      d.manifest_check.rows.map(r=>[r.kind, `${fmt(r.recomputed)} / ${r.manifest==null?'–':fmt(r.manifest)}`,
        (d.manifest_check.partial||r.manifest==null)? '' : (r.manifest===r.recomputed? 'ok':'mm')])));
  if(d.manifest) c.push(manifestCard(d.manifest));
  if(d.errors.length) c.push(exCard('turns that fail the grammar',
    `first ${d.errors.length} of ${fmt(d.n_error)} · strict per-line parse, no tolerant fallback`,
    d.errors));
  c.push(barCard('segments by typed characters','reconstructed per segment',
    d.top_typing_segments.map(x=>({lab:x.segment_id, cnt:x.count}))));
  $('#grid').innerHTML=c.join('');
  document.querySelectorAll('.brow[data-token]').forEach(el=>{
    el.onclick=()=>{ const tok=decodeURIComponent(el.dataset.token); $('#q').value=tok; runSearch(tok); };
  });
  document.querySelectorAll('.brow[data-shape]').forEach(el=>{
    el.onclick=()=> loadExamples(decodeURIComponent(el.dataset.shape));
  });
  document.querySelectorAll('.hist rect[data-field]').forEach(el=>{
    el.onclick=()=>{
      const lo=el.dataset.lo===''? null : +el.dataset.lo;
      const hi=el.dataset.hi===''? null : +el.dataset.hi;
      setNum(el.dataset.field, lo, hi);
      runSearch($('#q').value);
      $('#searchres').scrollIntoView({block:'nearest'});
    };
  });
}

function manifestCard(m){
  const rows=[];
  const add=(k,v)=>{ if(v!==undefined&&v!==null) rows.push([k, typeof v==='number'? fmt(v):String(v)]); };
  add('artifact_type', m.artifact_type); add('action_format', m.action_format);
  add('continuous_action_hz', m.continuous_action_hz);
  add('system_prompt_id', m.system_prompt_id); add('terminate_token', m.terminate_token);
  add('coalesce_typing', m.coalesce_typing); add('max_coalesce_frames', m.max_coalesce_frames);
  add('n_coalesced_turns', m.n_coalesced_turns); add('n_frames_coalesced_away', m.n_frames_coalesced_away);
  add('n_coalesce_forced_idle_turns', m.n_coalesce_forced_idle_turns);
  add('n_conversations', m.n_conversations); add('n_turns_total', m.n_turns_total);
  add('n_dead_zone_flagged', m.n_dead_zone_flagged);
  const dz=m.dead_zone_totals||{};
  for(const k of Object.keys(dz)) rows.push(['dead_zone.'+k, fmt(dz[k])]);
  return kvCard('dataset manifest','what the builder recorded', rows);
}

async function loadExamples(shape){
  const r = await (await fetch(`/api/examples?ds=${encodeURIComponent(CUR)}&shape=${encodeURIComponent(shape)}`)).json();
  $('#searchres').className='show';
  $('#searchres').innerHTML = `<div class="big">program shape <b class="hl">${esc(shape)}</b> — `+
    `${fmt((DIST.shapes.find(s=>s.shape===shape)||{}).count)} turns</div>`+
    (r.examples&&r.examples.length? `<div class="exlist">${r.examples.map(e=>
      `<div class="ex"><div class="who">${esc(e.segment_id)} · turn ${e.turn}</div><div class="txt">${esc(e.text)}</div></div>`).join('')}</div>`
      : '<div class="empty">no examples kept</div>');
  $('#searchres').scrollIntoView({block:'nearest'});
}

async function runSearch(query){
  const nums=numParams();
  const q=(query||'').trim();
  if(!q && !nums.length){ $('#searchres').className=''; return; }
  const url=`/api/search?ds=${encodeURIComponent(CUR)}&q=${encodeURIComponent(q)}`
          + (nums.length? '&'+nums.join('&') : '');
  const s = await (await fetch(url)).json();
  $('#searchres').className='show';
  if(s.error){ $('#searchres').innerHTML=`<span style="color:#f7a6a6">${esc(s.error)}</span>`; return; }
  const max=s.top_segments.reduce((m,x)=>Math.max(m,x.count),0)||1;
  const segs=s.top_segments.map(x=>{
    const w=(100*x.count/max).toFixed(1);
    return `<div class="segrow"><div class="bar"><i style="width:${w}%"></i><span class="lab">${esc(x.segment_id)}</span></div><div class="cnt">${fmt(x.count)}</div></div>`;
  }).join('');
  const ex=s.examples.map(e=>`<div class="ex"><div class="who">${esc(e.segment_id)} · turn ${e.turn}</div><div class="txt">${esc(e.text)}</div></div>`).join('');
  const crit=[]; if(s.query) crit.push(`<b class="hl">${esc(s.query)}</b>`);
  if(s.filter) crit.push(`<b class="hl">${esc(s.filter)}</b>`);
  const mix=(s.mix||[]).map(m=>`${esc(m.key)} <b>${fmt(m.count)}</b> <small>(${m.per_turn}/turn)</small>`).join(' · ');
  $('#searchres').innerHTML =
    `<div class="big">${crit.join(' AND ')} → <b>${fmt(s.n_turns)}</b> / ${fmt(s.turns_total)} action lines `+
    `(${pct(s.n_turns,s.turns_total)}) · <b>${fmt(s.n_segments)}</b> / ${fmt(s.segments_total)} segments `+
    `(${pct(s.n_segments,s.segments_total)})`+
    (s.typed_hits? ` · <b>${fmt(s.typed_hits)}</b> occurrences of the text in the reconstructed typed text of ${fmt(s.typed_segments)} segments (whole segments, count bounds not applied)`:'')+
    `</div>`+
    (mix? `<div class="hint" style="margin-top:6px">inside the matched turns</div><div>${mix}</div>`:'')+
    (s.unparsed? `<div class="hint" style="margin-top:4px">${fmt(s.unparsed)} candidate turns skipped: they fail the grammar, so they have no counts</div>`:'')+
    (s.top_segments.length? `<div class="hint" style="margin-top:6px">top segments by count</div><div class="seglist">${segs}</div>`:'')+
    (ex? `<div class="hint" style="margin-top:6px">example turns</div><div class="exlist">${ex}</div>`:'');
}

$('#qgo').onclick=()=>runSearch($('#q').value);
$('#q').addEventListener('keydown',e=>{ if(e.key==='Enter') runSearch($('#q').value); });
$('#qclear').onclick=()=>{ $('#q').value=''; clearNum(); $('#searchres').className=''; };
$('#gbtn').onclick=()=>{ const g=$('#grammar'); const on=!g.classList.contains('show');
  g.className=on?'show':''; $('#gbtn').classList.toggle('on',on); };

loadDatasets();
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset", required=True, nargs="+", metavar="PATH",
        help="one or more ordered_events_v3 datasets (same shapes as "
             "visualize_frame_records.py: a stage-04 conversations dir/file or a "
             "stage-06 inline-records dir) — auto-detected; choose in the UI",
    )
    p.add_argument(
        "--limit", "--limit-samples", dest="limit", type=_positive_int, default=None,
        help="aggregate at most the first K samples per dataset (a full stage-04 "
             "build is ~1.6M turns; start with a few hundred)",
    )
    p.add_argument("--top", type=_positive_int, default=TOP_N,
                   help=f"rows per ranked list (default {TOP_N})")
    p.add_argument("--port", type=int, default=8790, help="HTTP port (default 8790)")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    return p.parse_args()


def main() -> None:
    global DATASET_SAMPLE_LIMIT, TOP_N
    args = parse_args()
    DATASET_SAMPLE_LIMIT = args.limit
    TOP_N = args.top
    register_datasets(args.dataset)
    if not DATASETS:
        raise SystemExit("no datasets given")
    print(f"registered {len(DATASETS)} dataset(s) for {V3_FORMAT}:", flush=True)
    for name, entry in DATASETS.items():
        print(f"  {name}  [{entry['mode']}]  {entry['path']}", flush=True)
    if DATASET_SAMPLE_LIMIT is not None:
        print(f"sample limit: first {DATASET_SAMPLE_LIMIT} samples per dataset", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"serving on http://{args.host}:{args.port}/  "
        f"(datasets aggregate on first selection; Ctrl-C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
