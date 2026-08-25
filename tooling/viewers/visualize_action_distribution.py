#!/usr/bin/env python3
"""Action-distribution viewer — aggregate stats over a dataset's action tokens.

The companion to ``visualize_frame_records.py``. That tool browses ONE trajectory
at a time (frame + action per step); this one answers the *aggregate* questions
across the whole dataset:

  * How often is a mouse click present? Which button?
  * What exact mouse movements dominate (e.g. is ``-100 10 0`` over-represented)?
    And do those moves carry a scroll? The movement panels have an all / with-scroll
    / without-scroll toggle, so a cursor-move distribution can be read separately
    from the moves that happen while the wheel turns. All of them — exact triples,
    direction rose, dx / dy / |move| — cover the same population: frames that
    actually moved. Scroll-only and key-only frames are NOT movements; they show up
    in the full-action, scroll-amount and key cards instead.
  * Which keys are pressed the most? Which key *combinations* (Ctrl+C, Shift+…)?
  * What does the dx / dy / scroll magnitude distribution look like?
  * For any token you type (``LMB``, ``+KeyEnter``, ``-100 10 0``): what fraction
    of frames / segments contain it, and which segments contain it the most?
  * What does ONE segment (or one recording's segments) look like? The header's
    "segment id contains" box restricts the whole aggregate — every chart and the
    token filter — to segments whose id matches, and lists the matches so a partial
    id can be clicked down to the exact one. Cached per filter, so flipping back to
    the full view is instant.

The dataset's ``action_format`` is detected (from the records, else the artifact
manifest) and shown in the header, together with a line on how to read the movement
panels for it — an ``ordered_events_*`` turn keeps an intra-window trajectory the
charts can only show as its net sum, where a sampled head has nothing more to give.
The binning itself is shared: every format charts net displacement over one frame's
window, and measured on the same clip the sampled and ordered_events distributions
agree closely, so one wide edge set serves both (see ``_DXDY_EDGES``).

It reads the SAME datasets ``visualize_frame_records.py`` does — stage-stage 03
``frame_records.jsonl``, stage-04 ``conversations.jsonl``, stage-06 inline SFT
records (ArrayRecord ``train``/``val`` shards), all auto-detected — by importing
that module's loaders and its ``format_action`` parser, so a segment's stats here
match exactly what the frame viewer reconstructs when you open it. (A stage-01a
frames-master store is keylog-free, so it has no actions to aggregate; it's
reported as empty.)

The action grammar it aggregates over
(``pipeline.crowdcast.lib.common.format_action``): ``NO_OP``, or ``"<dx> <dy>
<scroll>"`` optionally followed by ``" ; "`` and space-separated ``+Name`` /
``-Name`` press/release tokens (rdev key names + mouse buttons ``LMB``/``RMB``/
``MMB``). Conversation/inline rows whose assistant turn prefixes the action with a
natural-language plan are handled by the shared tolerant parser.

For an ``ordered_events_v2``/``v3`` dataset (``move(dx,dy); down(LMB); up(LMB)``
mini-programs) use ``visualize_ordered_action_distribution.py`` instead: this
viewer's grammar has no notion of order or of ``type("…")``, so it can only
report a v3 turn's summed movement, while that one aggregates program shapes,
ordering, typing collapse and cross-turn held state.

The wheel has TWO axes. rdev reports scrolling as ``MouseScroll [delta_x,
delta_y]``, and ``format_action`` keeps only one number — ``lib.common`` takes the
vertical delta, or the horizontal one when vertical is 0 — so in a 3-number head a
sideways scroll is indistinguishable from a downward one. The native formats do
keep both (``computer_use`` ``scroll``, ordered-events ``scroll(dx,dy)``), and the
frame viewer's translation preserves them as an extended 4-number head ``<dx> <dy>
<scroll> <hscroll>`` emitted only when horizontal is nonzero. So here a frame counts
as a scroll if EITHER axis turns, the magnitude total sums both, and the two
amount-distributions are charted separately (the horizontal card is hidden for plain
3-number datasets, which can never populate it).

Run::

    cd .../data_pipeline
    uv run python tooling/viewers/visualize_action_distribution.py \
        --dataset <dir_or_file> [<dir_or_file> ...] \
        --port 8780
    # then SSH-forward the port and open http://127.0.0.1:8780/
    #   ssh -L 8780:127.0.0.1:8780 <host>

Pass several datasets and switch between them in the UI's "dataset" dropdown; each
is aggregated lazily on first selection and cached.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, OrderedDict
from http import HTTPStatus  # noqa: F401  (kept for parity / future use)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Make the ``pipeline.crowdcast`` package importable when run directly, then reuse
# the frame-records viewer's dataset loaders + action parser wholesale (single
# source of truth for the action grammar and the 4 dataset formats).
# Run as a file path, so put the repo root (for `pipeline`/`grammars`) and this
# viewers directory (for the sibling viewer imported below) on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWERS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, VIEWERS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import visualize_frame_records as V  # noqa: E402

# Dataset registry: display-name -> {"path": Path, "mode": str, "obj": built|None,
# "dist": aggregated|None}. Built lazily on first selection.
DATASETS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
DATASET_SAMPLE_LIMIT: "int | None" = None

# rdev modifier keys, canonicalized so "ControlLeft"/"ControlRight" fold into one
# "Ctrl" label in chords (the per-key chart keeps the raw left/right names).
_MOD_CANON = {
    "ControlLeft": "Ctrl", "ControlRight": "Ctrl",
    "ShiftLeft": "Shift", "ShiftRight": "Shift",
    "Alt": "Alt", "AltGr": "AltGr",
    "MetaLeft": "Meta", "MetaRight": "Meta",
}
_MODIFIERS = set(_MOD_CANON)
_BUTTONS = ("LMB", "RMB", "MMB")

_DIR_LABELS = ["E →", "SE ↘", "S ↓", "SW ↙", "W ←", "NW ↖", "N ↑", "NE ↗"]

# Histogram bin edges (non-uniform; rendered as equal-width categorical bars, with
# ±1e5 outer overflow catch-alls). ±100 is an edge AND highlighted so a pile-up at
# the "cursor moves 100px" quantum — the freeroll corner-pinning signature — is
# visible at a glance.
#
# These are deliberately NOT per-action-format. Every format charts the same
# quantity: net displacement over ONE frame's label window. An ordered_events turn
# holds a run of primitives and the hud sums them, but the sampled 3-number head
# sums the same window's MouseMove deltas — measured over the same clip at fps 1 the
# two distributions agree closely (|dx| p50 58 vs 57, p75 191 vs 194, p90 413 vs
# 436). So the tails must be wide for all of them: ±250 leaves ~20% of axis values
# in the overflow bin regardless of format, ±800 leaves 2-3%.
_DXDY_EDGES = [-100000, -800, -500, -350, -250, -160, -120, -100, -60, -30, -10, -1,
               1, 10, 30, 60, 100, 120, 160, 250, 350, 500, 800, 100000]
_MAG_EDGES = [0, 5, 10, 25, 50, 100, 175, 275, 400, 600, 900, 1400, 100000]

# What the movement charts MEAN for a given action_format. The format doesn't change
# the binning, but it does change how much detail the raw turn holds behind each
# charted number — worth saying out loud next to the panels.
_FORMAT_NOTES = {
    "ordered_events": "one turn is a run of primitives; the charted move/scroll is "
                      "their NET sum over the frame's window (the raw text keeps the "
                      "intra-window trajectory the chart cannot show)",
    "computer_use": "translated from the turn's tool calls; per-frame net move/scroll",
    "pyautogui": "translated from the turn's primitives; per-frame net move/scroll",
    "sampled": "per-frame net move/scroll, straight from the format_action head",
}


def _format_note(action_format: "str | None") -> str:
    """One line on how to read the movement panels for this dataset's format."""
    fmt = (action_format or "").strip()
    for prefix, note in _FORMAT_NOTES.items():
        if fmt.startswith(prefix):
            return note
    return _FORMAT_NOTES["sampled"]


def _detect_action_format(ds: Any, path: Path) -> "str | None":
    """The dataset's ``action_format``, preferring the value the records carry (the
    loader lifts it off each conversation/inline row) and falling back to the
    artifact manifest. ``None`` for a stage-stage 03 frame_records or frames-master
    store, which have no format field — those are the sampled grammar by
    construction."""
    for seg in getattr(ds, "segments", {}).values():
        fmt = getattr(seg, "action_format", None)
        if fmt:
            return str(fmt)
    return _manifest_action_format(path) or _sniff_action_format(ds)


def _sniff_action_format(ds: Any) -> "str | None":
    """Last resort: infer the format from the turns themselves.

    Artifacts built before stage 04 grew ``--action-format`` declare nothing —
    neither in the records nor anywhere up the manifest chain — yet their turns are
    plainly native. Reporting those as the sampled grammar would be wrong, so ask
    the same two parsers the loader used which one matches. Marked ``(inferred)`` so
    a declared format is never confused with a guessed one; ``None`` means the turns
    really are plain ``format_action`` strings."""
    for seg in getattr(ds, "segments", {}).values():
        for frame in seg.frames[:20]:
            if frame.get("hud") is None:   # loader found no native turn here
                continue
            raw = str(frame.get("action") or "")
            if V.parse_native_action(raw) is not None:
                return "computer_use_rel_v1 (inferred)"
            if V.parse_ordered_action(raw) is not None:
                return "ordered_events (inferred)"
        return None  # first segment had no native turn: plain format_action
    return None


def _manifest_action_format(path: Path, _depth: int = 0) -> "str | None":
    """``action_format`` read from the artifact manifest alone — no dataset build,
    so this is cheap enough to run at registration time (before anything is loaded,
    when only the MODE is known from the filenames).

    A stage-06 pack's own manifest records the packing params, not the action
    grammar, so when the key is absent follow ``inputs.source`` up the artifact
    chain to the stage-04 conversations set that produced it."""
    if _depth > 3:
        return None
    root = path if path.is_dir() else path.parent
    for name in ("manifest.json", "conversations_summary.json", "sample_summary.json"):
        f = root / name
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        fmt = data.get("action_format")
        if fmt:
            return str(fmt)
        inputs = data.get("inputs")
        src = inputs.get("source") if isinstance(inputs, dict) else None
        if isinstance(src, str) and src:
            upstream = _manifest_action_format(Path(src), _depth + 1)
            if upstream:
                return upstream
    return None


# The mouse-movement panels (top movements, direction, dx/dy/|move| histograms) are
# aggregated three times over disjoint-ish frame populations so the UI can toggle
# between them: every frame, only frames that ALSO carry a scroll (vertical or
# horizontal), and only frames with no scroll at all. "all" reproduces the numbers
# you get without the toggle; "scroll" + "noscroll" partition it.
_MOVE_VARIANTS = ("all", "scroll", "noscroll")


def _bin_index(value: float, edges: list[int]) -> int:
    """Index of the ``[edges[i], edges[i+1])`` bin containing ``value`` (clamped)."""
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


def _hist_spec(edges: list[int], counts: list[int], highlight_at: "list[int]") -> dict[str, Any]:
    """A renderable histogram: bin labels, counts, and which bins to highlight
    (the bins whose LEFT edge is one of ``highlight_at``, e.g. ±100)."""
    hi_idx = [i for i in range(len(edges) - 1) if edges[i] in highlight_at]
    return {"labels": _bin_labels(edges), "counts": counts, "highlight": hi_idx}


def _dir8(dx: float, dy: float) -> int:
    """8-way compass bucket of a (dx, dy) move (screen coords: +dy is DOWN)."""
    ang = math.degrees(math.atan2(dy, dx))  # 0=right, 90=down, -90=up, 180=left
    return int(round(ang / 45.0)) % 8


def _new_move_acc() -> dict[str, Any]:
    """Fresh accumulator for one mouse-movement variant (see ``_MOVE_VARIANTS``)."""
    return {
        "n_move": 0,       # frames with dx|dy != 0 — also the exact-triple population
        "triples": Counter(),
        "directions": Counter(),
        "dx": [0] * (len(_DXDY_EDGES) - 1),
        "dy": [0] * (len(_DXDY_EDGES) - 1),
        "mag": [0] * (len(_MAG_EDGES) - 1),
    }


def _move_payload(acc: dict[str, Any]) -> dict[str, Any]:
    """Render one movement accumulator into the JSON the UI's movement cards read."""
    return {
        "n_move": acc["n_move"],
        "triples": [{"move": t, "count": n} for t, n in acc["triples"].most_common(60)],
        "directions": [
            {"label": _DIR_LABELS[i], "count": acc["directions"].get(i, 0)}
            for i in range(8)
        ],
        "dx_hist": _hist_spec(_DXDY_EDGES, acc["dx"], highlight_at=[-100, 100]),
        "dy_hist": _hist_spec(_DXDY_EDGES, acc["dy"], highlight_at=[-100, 100]),
        "mag_hist": _hist_spec(_MAG_EDGES, acc["mag"], highlight_at=[100]),
    }


def _canonical_action(dx: int, dy: int, scroll: int, hscroll: int,
                      events: list[tuple[str, str]], noop: bool) -> str:
    """Rebuild the clean ``format_action`` string from parsed parts — strips any
    natural-language plan prefix so identical actions collapse to one bucket.

    A nonzero horizontal scroll keeps the extended 4-number head the native
    translations emit, so a tilt-wheel frame doesn't collapse to a bare "0 0 0"
    that reads as a NO_OP."""
    if noop:
        return "NO_OP"
    head = f"{dx} {dy} {scroll}" + (f" {hscroll}" if hscroll else "")
    if events:
        return head + " ; " + " ".join(f"{s}{n}" for s, n in events)
    return head


def _collect_segments(ds: Any) -> list[tuple[str, list[str]]]:
    """``[(segment_id, [action_str, ...]), ...]`` in dataset order.

    Frames-master stores are keylog-free (no per-frame actions), so they yield an
    empty list; every other loader exposes ``.segments`` of ``Segment`` objects
    whose ``.frames`` carry the action strings."""
    if isinstance(ds, V.FramesMasterDataset):
        return []
    out: list[tuple[str, list[str]]] = []
    for sid, seg in ds.segments.items():
        # ``hud`` is the format_action translation of a native computer_use /
        # ordered-events turn (aggregate that, not the raw tool-call or
        # primitive text); plain datasets have no hud.
        out.append((sid, [str(f.get("hud") or f.get("action") or "")
                          for f in seg.frames]))
    return out


def _match_segments(segs: list[tuple[str, list[str]]],
                    query: str) -> list[tuple[str, list[str]]]:
    """The subset whose segment_id contains ``query`` (case-insensitive). Empty
    query -> everything, so the unfiltered aggregate is the same code path."""
    q = query.strip().lower()
    if not q:
        return segs
    return [(sid, actions) for sid, actions in segs if q in sid.lower()]


def build_distribution(ds: Any,
                       segs: "list[tuple[str, list[str]]] | None" = None) -> dict[str, Any]:
    """Aggregate the full action distribution over a dataset in a single pass.

    Every count is frame-level unless named ``*_frames`` (frames CONTAINING at
    least one such event) or ``*_segments``. Chords are press-triggered: a chord is
    emitted when a non-modifier key or a mouse button is pressed while ≥1 modifier
    is held (tracked across frames within a segment, mirroring the frame viewer's
    press/release bookkeeping), so Ctrl+C spanning two turns still counts once.

    ``segs`` restricts the aggregate to a subset (the segment-id filter)."""
    if segs is None:
        segs = _collect_segments(ds)
    n_segments = len(segs)
    n_frames = n_noop = 0
    n_move = n_scroll = n_click = n_key = 0

    key_press: Counter = Counter()          # +Key presses, per raw rdev name
    key_frames: Counter = Counter()         # frames containing ≥1 press of a key
    key_segments: Counter = Counter()       # segments containing ≥1 press of a key
    btn_press = {b: 0 for b in _BUTTONS}
    btn_frames = {b: 0 for b in _BUTTONS}
    chords: Counter = Counter()
    full_actions: Counter = Counter()       # exact canonical action over active frames
    scroll_vals: Counter = Counter()        # exact nonzero vertical scroll amounts
    hscroll_vals: Counter = Counter()       # ditto horizontal (native formats only)
    # movement stats, once per scroll-presence variant (all / with-scroll / no-scroll)
    move_acc = {v: _new_move_acc() for v in _MOVE_VARIANTS}
    n_move_scroll = 0                       # frames that both move AND scroll
    total_scroll_mag = 0

    for sid, actions in segs:
        held: set[str] = set()
        seg_keys: set[str] = set()
        for a in actions:
            n_frames += 1
            dx_f, dy_f, scroll_f, hscroll_f, events = V._parse_action_str(a)
            dx, dy, scroll = int(round(dx_f)), int(round(dy_f)), int(round(scroll_f))
            hscroll = int(round(hscroll_f))
            noop = (dx == 0 and dy == 0 and scroll == 0 and hscroll == 0
                    and not events)
            if noop:
                n_noop += 1
            moved = dx != 0 or dy != 0
            has_scroll = scroll != 0 or hscroll != 0
            if moved:
                n_move += 1
                if has_scroll:
                    n_move_scroll += 1
            # A scroll frame is one with either wheel axis turning: the 3-number
            # format_action head carries only the vertical wheel, but the native
            # (computer_use / ordered-events) translations keep horizontal in a 4th
            # slot, so a tilt-wheel-only frame is a scroll too. Amounts stay split
            # per axis — mixing them in one histogram would be meaningless.
            if has_scroll:
                n_scroll += 1
                total_scroll_mag += abs(scroll) + abs(hscroll)
                if scroll:
                    scroll_vals[scroll] += 1
                if hscroll:
                    hscroll_vals[hscroll] += 1
            # feed the "all" accumulator plus exactly one of the scroll variants.
            # ALL movement panels (triples, direction, dx/dy/|move|) share one
            # population — frames that actually moved. A scroll-only or key-only
            # frame is not a mouse movement and would otherwise dominate the
            # exact-triple list as a meaningless "0 0 0"; its exact action is
            # already covered by the full-actions and scroll-amount cards.
            if moved:
                for acc in (move_acc["all"],
                            move_acc["scroll" if has_scroll else "noscroll"]):
                    acc["n_move"] += 1
                    acc["dx"][_bin_index(dx, _DXDY_EDGES)] += 1
                    acc["dy"][_bin_index(dy, _DXDY_EDGES)] += 1
                    acc["mag"][_bin_index(math.hypot(dx, dy), _MAG_EDGES)] += 1
                    acc["directions"][_dir8(dx, dy)] += 1
                    # the scroll column rides along so "did this move carry a
                    # scroll" stays readable; hscroll in the extended 4-number form
                    acc["triples"][f"{dx} {dy} {scroll}"
                                   + (f" {hscroll}" if hscroll else "")] += 1
            if moved or has_scroll or events:
                full_actions[_canonical_action(dx, dy, scroll, hscroll, events, False)] += 1

            frame_buttons: set[str] = set()
            has_key = False
            for sign, name in events:
                if name in _BUTTONS:
                    if sign == "+":
                        btn_press[name] += 1
                        frame_buttons.add(name)
                        mods = sorted({_MOD_CANON[m] for m in held if m in _MODIFIERS})
                        if mods:
                            chords["+".join(mods + [name])] += 1
                        held.add(name)
                    else:
                        held.discard(name)
                else:
                    if sign == "+":
                        key_press[name] += 1
                        has_key = True
                        key_frames[name] += 1
                        seg_keys.add(name)
                        if name not in _MODIFIERS:
                            mods = sorted({_MOD_CANON[m] for m in held if m in _MODIFIERS})
                            if mods:
                                chords["+".join(mods + [name])] += 1
                        held.add(name)
                    else:
                        held.discard(name)
            if frame_buttons:
                n_click += 1
                for b in frame_buttons:
                    btn_frames[b] += 1
            if has_key:
                n_key += 1
        for k in seg_keys:
            key_segments[k] += 1

    n_active = n_frames - n_noop
    keys_sorted = [
        {"name": k, "presses": key_press[k], "frames": key_frames[k],
         "segments": key_segments[k]}
        for k, _ in key_press.most_common()
    ]
    return {
        "n_segments": n_segments,
        "n_frames": n_frames,
        "n_noop": n_noop,
        "n_active": n_active,
        "present": {  # frames CONTAINING each (non-exclusive)
            "move": n_move, "scroll": n_scroll, "click": n_click, "key": n_key,
            "noop": n_noop, "move_scroll": n_move_scroll,
            "move_noscroll": n_move - n_move_scroll,
        },
        "totals": {
            "clicks": sum(btn_press.values()),
            "keypresses": sum(key_press.values()),
            "scroll_mag": total_scroll_mag,
        },
        "buttons": [
            {"name": b, "presses": btn_press[b], "frames": btn_frames[b]}
            for b in _BUTTONS
        ],
        "keys": keys_sorted,
        "chords": [{"combo": c, "count": n} for c, n in chords.most_common(60)],
        "full_actions": [
            {"action": a, "count": n} for a, n in full_actions.most_common(60)
        ],
        "scroll_values": [
            {"amount": v, "count": n} for v, n in scroll_vals.most_common(40)
        ],
        # horizontal wheel; only the native translations carry it, so this list is
        # empty for plain 3-number format_action datasets and its card is hidden
        "hscroll_values": [
            {"amount": v, "count": n} for v, n in hscroll_vals.most_common(40)
        ],
        # mouse-movement panels per scroll-presence variant; the UI toggles between
        # them ("all" == the unfiltered numbers, "scroll" + "noscroll" partition it)
        "movement": {v: _move_payload(move_acc[v]) for v in _MOVE_VARIANTS},
    }


def search_token(ds: Any, query: str, seg_query: str = "") -> dict[str, Any]:
    """How often ``query`` (case-insensitive substring) appears across the dataset.

    Matches against the aggregated action strings (the ``format_action`` grammar:
    raw for plain datasets, the ``hud`` translation for native computer_use /
    ordered-events turns), so it reaches tokens verbatim — ``LMB``, ``+KeyEnter``,
    ``-100 10 0`` — even when a conversation turn wraps the action in a
    natural-language plan. Returns frame/segment coverage and the segments where
    it occurs most.

    ``seg_query`` narrows the population to segments whose id matches, so the token
    coverage is reported against the same subset the charts are showing."""
    q = query.strip().lower()
    segs = _match_segments(_cache_segments(ds), seg_query)
    n_frames_total = sum(len(a) for _, a in segs)
    n_seg_total = len(segs)
    if not q:
        return {"query": query, "n_frames": 0, "frames_total": n_frames_total,
                "n_segments": 0, "segments_total": n_seg_total, "top_segments": []}
    n_frames = 0
    per_seg: list[tuple[str, int]] = []
    for sid, actions in segs:
        c = sum(1 for a in actions if q in a.lower())
        if c:
            n_frames += c
            per_seg.append((sid, c))
    per_seg.sort(key=lambda x: x[1], reverse=True)
    return {
        "query": query,
        "seg_query": seg_query,
        "n_frames": n_frames,
        "frames_total": n_frames_total,
        "n_segments": len(per_seg),
        "segments_total": n_seg_total,
        "top_segments": [{"segment_id": s, "count": c} for s, c in per_seg[:25]],
    }


def list_segments(ds: Any, seg_query: str, limit: int = 200) -> dict[str, Any]:
    """Segment ids matching ``seg_query``, with per-segment frame/active counts —
    the "which segment am I looking for" lookup that pairs with the token filter."""
    all_segs = _cache_segments(ds)
    hits = _match_segments(all_segs, seg_query)
    rows = []
    for sid, actions in hits[:limit]:
        active = 0
        for a in actions:
            dx, dy, scroll, hscroll, events = V._parse_action_str(a)
            if dx or dy or scroll or hscroll or events:
                active += 1
        rows.append({"segment_id": sid, "n_frames": len(actions), "n_active": active})
    return {
        "seg_query": seg_query,
        "n_matched": len(hits),
        "segments_total": len(all_segs),
        "frames_matched": sum(len(a) for _, a in hits),
        "truncated": len(hits) > limit,
        "segments": rows,
    }


def _cache_segments(ds: Any) -> list[tuple[str, list[str]]]:
    segs = getattr(ds, "_dist_segments", None)
    if segs is None:
        segs = _collect_segments(ds)
        ds._dist_segments = segs  # type: ignore[attr-defined]
    return segs


# --------------------------------------------------------------------------- #
# Dataset registration / lazy build (delegates format detection to the sibling).
# --------------------------------------------------------------------------- #
def register_datasets(paths: list[str]) -> None:
    for raw in paths:
        p = Path(raw).expanduser()
        name = p.name or str(p)
        base, k = name, 2
        while name in DATASETS:
            name, k = f"{base}#{k}", k + 1
        # ``mode`` is the container shape (which loader to use); ``action_format`` is
        # the grammar inside the turns. Both are named up front — the format from the
        # manifest chain only, since nothing is loaded yet; the record-derived value
        # replaces it in ``_ensure_built``.
        DATASETS[name] = {"path": p, "mode": V.detect_mode(p), "obj": None,
                          "action_format": _manifest_action_format(p), "dist": {}}


def _ensure_built(entry: dict[str, Any]) -> Any:
    """Build the dataset object once, detecting its action format on the way."""
    if entry["obj"] is None:
        # The frame viewer's loaders take WHICH samples to read as a Sampling
        # object (it also offers a random draw); this viewer only ever aggregates
        # the first --limit samples, so pass that mode explicitly.
        sampling = V.Sampling("first", DATASET_SAMPLE_LIMIT)
        try:
            ds = V._build_dataset(entry["path"], sampling)
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
        entry["obj"] = ds
        entry["action_format"] = (_detect_action_format(ds, entry["path"])
                                  or entry["action_format"])
        _cache_segments(ds)  # warm the segment/search cache
    return entry["obj"]


def get_distribution(name: str, seg_query: str = "") -> dict[str, Any] | None:
    """Build (or return cached) aggregate distribution for a registered dataset,
    optionally restricted to segments whose id matches ``seg_query``. Cached per
    (dataset, segment filter) so flipping back to the full view is instant."""
    entry = DATASETS.get(name)
    if entry is None:
        return None
    key = seg_query.strip().lower()
    if key not in entry["dist"]:
        ds = _ensure_built(entry)
        action_format = entry["action_format"]
        all_segs = _cache_segments(ds)
        segs = _match_segments(all_segs, key)
        dist = build_distribution(ds, segs)
        dist["mode"] = getattr(ds, "mode", entry["mode"])
        # None for a frame_records / frames-master store, which is the sampled
        # grammar by construction; the note says how to read the movement panels
        dist["action_format"] = action_format
        dist["movement_note"] = _format_note(action_format)
        dist["seg_query"] = seg_query
        dist["n_segments_total"] = len(all_segs)
        entry["dist"][key] = dist
    return entry["dist"][key]


def get_dataset_obj(name: str) -> Any | None:
    entry = DATASETS.get(name)
    if entry is None:
        return None
    return _ensure_built(entry)


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
        route = parsed.path
        q = parse_qs(parsed.query)
        try:
            if route == "/":
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif route == "/api/datasets":
                names = list(DATASETS.keys())
                self._send_json({
                    "datasets": [{"name": n, "mode": DATASETS[n]["mode"],
                                  "action_format": DATASETS[n]["action_format"]}
                                 for n in names],
                    "default": names[0] if names else None,
                })
            elif route == "/api/dist":
                name = self._dsname(q)
                if name not in DATASETS:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                    return
                seg = (q.get("seg") or [""])[0]
                try:
                    dist = get_distribution(name, seg)
                    if dist is not None and seg.strip() and dist["n_segments"] == 0:
                        self._send_json({"error": f"no segment id contains {seg!r}"}, 404)
                        return
                    self._send_json(dist)
                except Exception as exc:  # noqa: BLE001 — report, keep UI alive
                    self._send_json({"error": f"failed to load {name!r}: {exc}"}, 500)
            elif route == "/api/search":
                name = self._dsname(q)
                query = (q.get("q") or [""])[0]
                seg = (q.get("seg") or [""])[0]
                ds = get_dataset_obj(name) if name in DATASETS else None
                if ds is None:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                else:
                    self._send_json(search_token(ds, query, seg))
            elif route == "/api/segments":
                name = self._dsname(q)
                seg = (q.get("seg") or [""])[0]
                ds = get_dataset_obj(name) if name in DATASETS else None
                if ds is None:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                else:
                    self._send_json(list_segments(ds, seg))
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
<title>action-distribution viewer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.45 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
         background:#14161a; color:#d7dae0; }
  header { position:sticky; top:0; z-index:10; padding:8px 14px; border-bottom:1px solid #2a2e36;
           display:flex; gap:10px; align-items:center; flex-wrap:wrap; background:#191c21; }
  header .title { font-weight:700; color:#e8eef7; }
  header #segq { min-width:270px; }
  header #segq.on { border-color:#c08a2a; }
  select,input,button { background:#22262e; color:#d7dae0; border:1px solid #343a44;
                  border-radius:4px; padding:4px 8px; font:inherit; }
  select,button { cursor:pointer; }
  button:hover { border-color:#5b9dd9; }
  .hint { margin-left:auto; color:#6b7280; font-size:12px; }
  main { padding:14px; max-width:1500px; margin:0 auto; }
  #err { color:#f7a6a6; padding:6px 0; }
  #loading { color:#8b93a1; padding:10px 0; }

  .tiles { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
  .tile { background:#191c21; border:1px solid #2a2e36; border-radius:6px; padding:8px 12px; min-width:120px; }
  .tile .k { color:#8b93a1; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .tile .v { color:#e8eef7; font-size:19px; font-weight:700; margin-top:2px; }
  .tile .s { color:#7fd6a2; font-size:11px; }

  /* search */
  #searchbar { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  #q { min-width:280px; flex:1; }
  #searchres { background:#171b22; border:1px solid #2a2e36; border-radius:6px; padding:10px 12px;
               margin-bottom:16px; display:none; }
  #searchres.show { display:block; }
  #searchres .big { font-size:15px; color:#e8eef7; }
  #searchres .big b.hl { color:#f5b544; }
  #searchres .seglist { margin-top:8px; display:flex; flex-direction:column; gap:2px; max-height:230px; overflow:auto; }
  #searchres .segrow { display:grid; grid-template-columns:1fr 88px; gap:8px; }
  #searchres .segrow[data-seg] { cursor:pointer; }
  #searchres .segrow[data-seg]:hover .lab { color:#fff; }
  #searchres .segrow .cnt small { color:#6b7280; }
  #searchres .segrow .bar { background:#20242b; border-radius:3px; position:relative; overflow:hidden; }
  #searchres .segrow .bar > i { position:absolute; inset:0 auto 0 0; background:#2d4a75; }
  #searchres .segrow .lab { position:relative; padding:1px 6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#cbd3df; }
  #searchres .segrow .cnt { text-align:right; color:#8fc4f2; }

  /* mouse-movement scroll-presence toggle */
  #movbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
  #movbar .lbl { color:#8b93a1; }
  /* how to read the movement panels for this dataset's action format */
  #fmtnote { color:#8fc4f2; font-size:12px; margin:-4px 0 10px; }
  #fmtnote:empty { display:none; }
  .seg { display:inline-flex; border:1px solid #343a44; border-radius:5px; overflow:hidden; }
  .seg button { border:0; border-radius:0; background:#1d2128; color:#8b93a1; padding:4px 10px; }
  .seg button + button { border-left:1px solid #343a44; }
  .seg button:hover { color:#d7dae0; background:#262b34; }
  .seg button.on { background:#2d4a75; color:#eaf1fb; }
  .seg button small { color:#7f8a99; }
  .seg button.on small { color:#b7cbe6; }

  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(430px, 1fr)); gap:14px; }
  .card { background:#191c21; border:1px solid #2a2e36; border-radius:8px; padding:10px 12px; }
  .card.filt { border-color:#4a5f86; }
  .card h3 { margin:0 0 8px; font-size:12px; color:#aeb6c2; text-transform:uppercase; letter-spacing:.05em;
             display:flex; align-items:baseline; gap:8px; }
  .card h3 .sub { color:#6b7280; font-size:11px; text-transform:none; letter-spacing:0; font-weight:400; }

  /* horizontal bar list */
  .blist { display:flex; flex-direction:column; gap:3px; max-height:340px; overflow:auto; }
  .brow { display:grid; grid-template-columns:1fr 74px; gap:8px; align-items:center; cursor:pointer; }
  .brow:hover .lab { color:#fff; }
  .brow .bar { height:18px; background:#20242b; border-radius:3px; position:relative; overflow:hidden; }
  .brow .bar > i { position:absolute; inset:0 auto 0 0; background:#3564a0; }
  .brow.hl .bar > i { background:#c08a2a; }
  .brow.click .bar > i { background:#2d6a45; }
  .brow .lab { position:relative; padding:0 7px; line-height:18px; white-space:nowrap; overflow:hidden;
               text-overflow:ellipsis; color:#cbd3df; font-size:12px; }
  .brow .cnt { text-align:right; color:#8fc4f2; font-size:12px; }
  .brow .cnt small { color:#6b7280; }

  /* svg histogram */
  .hist { width:100%; height:150px; }
  .hist .bar { fill:#3564a0; }
  .hist .bar.hl { fill:#c08a2a; }
  .hist text { fill:#7a828e; font-size:9px; }
  .histx { display:flex; justify-content:space-between; color:#6b7280; font-size:10px; margin-top:2px; }
  .empty { color:#6b7280; font-style:italic; }
</style>
</head><body>
<header>
  <span class="title">action distribution</span>
  <select id="ds"></select>
  <input id="segq" type="search" placeholder="segment id contains… (aggregate this subset)">
  <button id="seggo">apply</button>
  <button id="segclear">all</button>
  <span id="mode" class="hint"></span>
</header>
<main>
  <div id="err"></div>
  <div id="loading">select a dataset…</div>
  <div id="content" style="display:none">
    <div class="tiles" id="tiles"></div>
    <div id="searchbar">
      <input id="q" placeholder="filter: type a token — LMB, +KeyEnter, ControlLeft, -100 10 0 — substring match">
      <button id="qgo">count</button>
      <button id="qclear">clear</button>
      <span class="hint">click any bar to filter by it</span>
    </div>
    <div id="searchres"></div>
    <div id="movbar">
      <span class="lbl">mouse movement:</span>
      <span class="seg" id="movseg"></span>
      <span class="hint" id="movhint"></span>
    </div>
    <div id="fmtnote"></div>
    <div class="grid" id="grid"></div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
let CUR = null;      // current dataset name
let DIST = null;     // current distribution
let SEGQ = '';       // segment-id substring the aggregate is restricted to ('' = all)
let MOVEVAR = 'all'; // scroll-presence filter for the mouse-movement cards

// scroll-presence variants of the movement panels (server aggregates all three)
const MOVE_VARIANTS = {
  all:      {label:'all',             sub:'any scroll',    count:'move'},
  scroll:   {label:'with scroll',     sub:'scroll ≠ 0', count:'move_scroll'},
  noscroll: {label:'without scroll',  sub:'scroll = 0',    count:'move_noscroll'},
};

function fmt(n){ return (n==null?0:n).toLocaleString(); }
function pct(a,b){ return b? (100*a/b).toFixed(1)+'%' : '0%'; }

async function loadDatasets(){
  const r = await fetch('/api/datasets'); const d = await r.json();
  const sel = $('#ds'); sel.innerHTML='';
  for(const ds of d.datasets){
    const o=document.createElement('option'); o.value=ds.name;
    // mode = container shape, action_format = grammar inside the turns
    o.textContent=`${ds.name}  [${ds.mode}${ds.action_format?' · '+ds.action_format:''}]`;
    sel.appendChild(o);
  }
  sel.onchange = ()=> selectDataset(sel.value);
  if(d.default) selectDataset(d.default);
}

async function selectDataset(name){
  CUR = name;
  await loadDist();
}

// (Re)fetch the aggregate for CUR under the current segment-id filter.
async function loadDist(){
  DIST = null;
  const scope = SEGQ ? ` · segments matching "${SEGQ}"` : '';
  $('#err').textContent=''; $('#content').style.display='none';
  $('#loading').style.display='';
  $('#loading').textContent=`aggregating ${CUR}${scope} … (first load builds the dataset)`;
  $('#mode').textContent='';
  $('#segq').className = SEGQ ? 'on' : '';
  try{
    const r = await fetch(`/api/dist?ds=${encodeURIComponent(CUR)}&seg=${encodeURIComponent(SEGQ)}`);
    const d = await r.json();
    if(d.error){ $('#loading').style.display='none'; $('#err').textContent=d.error; return; }
    DIST = d; render(d);
  }catch(e){ $('#loading').style.display='none'; $('#err').textContent=String(e); }
}

function render(d){
  $('#loading').style.display='none'; $('#content').style.display='';
  // the action format is what picked the movement binning — show it, don't hide it
  const segs = (d.seg_query && d.n_segments_total!=null)
    ? `${fmt(d.n_segments)} / ${fmt(d.n_segments_total)} segments` : `${fmt(d.n_segments)} segments`;
  $('#mode').textContent = `${d.mode} · ${d.action_format || 'sampled (no action_format field)'}`
    + ` · ${segs} · ${fmt(d.n_frames)} frames`;
  if(d.n_frames===0){ $('#grid').innerHTML='<div class="empty">no action tokens in this dataset (a frames-master store is keylog-free — run stage 03 / 04 to get actions).</div>'; $('#tiles').innerHTML=''; $('#movbar').style.display='none'; $('#fmtnote').textContent=''; return; }
  $('#movbar').style.display='';
  renderTiles(d); renderMoveSeg(d); renderGrid(d);
  $('#searchres').className=''; $('#searchres').innerHTML=''; $('#q').value='';
  if(SEGQ) showMatchedSegments();
}

// Which segment ids the current filter selected (so a partial id can be narrowed
// to the one you meant — click a row to pin the aggregate to exactly that segment).
async function showMatchedSegments(){
  const r = await fetch(`/api/segments?ds=${encodeURIComponent(CUR)}&seg=${encodeURIComponent(SEGQ)}`);
  const s = await r.json();
  if(s.error) return;
  const max = s.segments.reduce((m,x)=>Math.max(m,x.n_frames),0)||1;
  const rows = s.segments.map(x=>{
    const w=(100*x.n_frames/max).toFixed(1);
    return `<div class="segrow" data-seg="${encodeURIComponent(x.segment_id)}" title="click to aggregate only this segment">`
      + `<div class="bar"><i style="width:${w}%"></i><span class="lab">${esc(x.segment_id)}</span></div>`
      + `<div class="cnt">${fmt(x.n_active)}<small>/${fmt(x.n_frames)}</small></div></div>`;
  }).join('');
  $('#searchres').className='show';
  $('#searchres').innerHTML =
    `<div class="big">segment filter <b class="hl">${esc(SEGQ)}</b> → `
    + `<b>${fmt(s.n_matched)}</b> / ${fmt(s.segments_total)} segments · `
    + `<b>${fmt(s.frames_matched)}</b> frames — every chart below covers this subset</div>`
    + `<div class="hint" style="margin-top:6px">matching segments (active/frames)`
    + `${s.truncated?` — first ${s.segments.length} shown`:''}</div>`
    + `<div class="seglist">${rows}</div>`;
  document.querySelectorAll('#searchres .segrow[data-seg]').forEach(el=>{
    el.onclick = ()=>{ const sid=decodeURIComponent(el.dataset.seg);
                       $('#segq').value=sid; SEGQ=sid; loadDist(); };
  });
}

function renderTiles(d){
  const t = [
    ['segments', fmt(d.n_segments), ''],
    ['frames', fmt(d.n_frames), ''],
    ['active', fmt(d.n_active), pct(d.n_active,d.n_frames)+' of frames'],
    ['NO_OP', fmt(d.n_noop), pct(d.n_noop,d.n_frames)+' of frames'],
    ['move frames', fmt(d.present.move), pct(d.present.move,d.n_frames)],
    ['clicks', fmt(d.totals.clicks), pct(d.present.click,d.n_frames)+' of frames'],
    ['key presses', fmt(d.totals.keypresses), pct(d.present.key,d.n_frames)+' of frames'],
    ['scroll frames', fmt(d.present.scroll), pct(d.present.scroll,d.n_frames)],
  ];
  $('#tiles').innerHTML = t.map(([k,v,s])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');
}

// the scroll-presence segmented control; counts are move frames per variant
function renderMoveSeg(d){
  $('#movseg').innerHTML = Object.entries(MOVE_VARIANTS).map(([k,v])=>
    `<button data-mv="${k}" class="${k===MOVEVAR?'on':''}">${v.label} <small>${fmt(d.present[v.count])}</small></button>`).join('');
  $('#movhint').textContent =
    `filters the movement cards (top movements · direction · dx · dy · |move|) to frames `+
    (MOVEVAR==='all' ? 'regardless of scroll' : `whose scroll is ${MOVEVAR==='scroll'?'nonzero':'zero'}`);
  $('#fmtnote').textContent = d.movement_note ? `${d.action_format||'sampled'}: ${d.movement_note}` : '';
  document.querySelectorAll('#movseg button').forEach(b=>{
    b.onclick = ()=>{ MOVEVAR = b.dataset.mv; renderMoveSeg(DIST); renderGrid(DIST); };
  });
}

// one horizontal bar-list card. items:[{lab, cnt, sub?, cls?, token?}]
function barCard(title, sub, items, cardCls){
  const max = items.reduce((m,it)=>Math.max(m,it.cnt),0) || 1;
  const rows = items.map(it=>{
    const w = (100*it.cnt/max).toFixed(1);
    const cls = it.cls? ' '+it.cls : '';
    const tok = it.token!=null? ` data-token="${encodeURIComponent(it.token)}"` : '';
    const sub = it.sub? ` <small>${it.sub}</small>` : '';
    return `<div class="brow${cls}"${tok}><div class="bar"><i style="width:${w}%"></i><span class="lab">${esc(it.lab)}</span></div><div class="cnt">${fmt(it.cnt)}${sub}</div></div>`;
  }).join('');
  const body = items.length? `<div class="blist">${rows}</div>` : `<div class="empty">none</div>`;
  return `<div class="card${cardCls?' '+cardCls:''}"><h3>${title}${sub?` <span class="sub">${sub}</span>`:''}</h3>${body}</div>`;
}

function histCard(title, sub, hist, cardCls){
  const n = hist.counts.length, max = Math.max(1,...hist.counts);
  const W=100/n;
  const bars = hist.counts.map((c,i)=>{
    const h = 100*c/max, hl = hist.highlight.includes(i)?' hl':'';
    return `<rect class="bar${hl}" x="${(i*W).toFixed(2)}%" y="${(100-h).toFixed(2)}%" width="${(W*0.86).toFixed(2)}%" height="${h.toFixed(2)}%"><title>${esc(hist.labels[i])}: ${fmt(c)}</title></rect>`;
  }).join('');
  // sparse x labels (first, ~mid, last, and highlighted)
  const marks = new Set([0, Math.floor(n/2), n-1, ...hist.highlight]);
  const xl = hist.labels.map((l,i)=> marks.has(i)? `<span>${esc(l)}</span>`:'').join('');
  return `<div class="card${cardCls?' '+cardCls:''}"><h3>${title}${sub?` <span class="sub">${sub}</span>`:''}</h3>`+
         `<svg class="hist" preserveAspectRatio="none">${bars}</svg><div class="histx">${xl}</div></div>`;
}

function renderGrid(d){
  const cards = [];
  // the movement cards read the currently toggled scroll-presence variant
  const mv = d.movement[MOVEVAR] || d.movement.all;
  const vsub = MOVE_VARIANTS[MOVEVAR].sub;
  const mcls = MOVEVAR==='all' ? '' : 'filt';
  // exact actions & movements — the "why is X so common" panels
  cards.push(barCard('top full actions','exact canonical string',
     d.full_actions.map(x=>({lab:x.action, cnt:x.count, token:x.action})) ));
  cards.push(barCard('top mouse movements',
     `dx dy scroll[ hscroll] · of ${fmt(mv.n_move)} move frames (keys ignored) · ${vsub}`,
     mv.triples.map(x=>({lab:x.move, cnt:x.count, token:x.move})), mcls ));
  // buttons
  cards.push(barCard('mouse buttons','presses · frames-with',
     d.buttons.map(b=>({lab:b.name, cnt:b.presses, sub:pct(b.frames,d.n_frames), cls:'click', token:b.name})) ));
  // keys
  cards.push(barCard('keys pressed','presses · '+d.keys.length+' distinct',
     d.keys.map(k=>({lab:k.name, cnt:k.presses, sub:`${fmt(k.segments)} seg`, token:'+'+k.name})) ));
  // chords
  cards.push(barCard('key combinations','modifier + key/button chords',
     d.chords.map(c=>({lab:c.combo, cnt:c.count, token:null})) ));
  // directions
  cards.push(barCard('move direction',`8-way, of ${fmt(mv.n_move)} move frames · ${vsub}`,
     mv.directions.map(x=>({lab:x.label, cnt:x.count})), mcls ));
  // scroll amounts — vertical wheel, plus horizontal when the format carries it
  cards.push(barCard('scroll amounts','vertical wheel · exact nonzero values',
     d.scroll_values.map(x=>({lab:String(x.amount), cnt:x.count, token:null})) ));
  if((d.hscroll_values||[]).length)
    cards.push(barCard('horizontal scroll amounts','4th head slot · exact nonzero values',
       d.hscroll_values.map(x=>({lab:String(x.amount), cnt:x.count, token:null})) ));
  // histograms
  cards.push(histCard('dx distribution',`over move frames · ${vsub}; ±100 highlighted`, mv.dx_hist, mcls));
  cards.push(histCard('dy distribution',`over move frames · ${vsub}; ±100 highlighted`, mv.dy_hist, mcls));
  cards.push(histCard('|move| magnitude',`px per frame · ${vsub}; 100 highlighted`, mv.mag_hist, mcls));
  $('#grid').innerHTML = cards.join('');
  // clicking a bar with a token filters
  document.querySelectorAll('.brow[data-token]').forEach(el=>{
    el.onclick = ()=>{ const tok=decodeURIComponent(el.dataset.token); $('#q').value=tok; runSearch(tok); };
  });
}

async function runSearch(query){
  if(!query || !query.trim()){ SEGQ ? showMatchedSegments() : ($('#searchres').className=''); return; }
  const r = await fetch(`/api/search?ds=${encodeURIComponent(CUR)}`
    + `&q=${encodeURIComponent(query)}&seg=${encodeURIComponent(SEGQ)}`);
  const s = await r.json();
  if(s.error){ $('#searchres').className='show'; $('#searchres').innerHTML=`<span style="color:#f7a6a6">${s.error}</span>`; return; }
  const max = s.top_segments.reduce((m,x)=>Math.max(m,x.count),0)||1;
  const segs = s.top_segments.map(x=>{
    const w=(100*x.count/max).toFixed(1);
    return `<div class="segrow"><div class="bar"><i style="width:${w}%"></i><span class="lab">${esc(x.segment_id)}</span></div><div class="cnt">${fmt(x.count)}</div></div>`;
  }).join('');
  $('#searchres').className='show';
  $('#searchres').innerHTML =
    `<div class="big"><b class="hl">${esc(s.query)}</b> appears in `+
    `<b>${fmt(s.n_frames)}</b> / ${fmt(s.frames_total)} frames (${pct(s.n_frames,s.frames_total)}) · `+
    `<b>${fmt(s.n_segments)}</b> / ${fmt(s.segments_total)} segments (${pct(s.n_segments,s.segments_total)})</div>`+
    (s.top_segments.length? `<div class="hint" style="margin-top:6px">top segments by count</div><div class="seglist">${segs}</div>`:'');
}

function esc(s){ return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

$('#qgo').onclick = ()=> runSearch($('#q').value);
$('#q').addEventListener('keydown', e=>{ if(e.key==='Enter') runSearch($('#q').value); });
$('#qclear').onclick = ()=>{ $('#q').value=''; SEGQ ? showMatchedSegments() : ($('#searchres').className=''); };

// segment-id filter: restricts the WHOLE aggregate (every chart + the token filter)
function applySegFilter(){
  const v = $('#segq').value.trim();
  if(v === SEGQ) return;
  SEGQ = v; loadDist();
}
$('#seggo').onclick = applySegFilter;
$('#segq').addEventListener('keydown', e=>{ if(e.key==='Enter') applySegFilter(); });
$('#segclear').onclick = ()=>{ $('#segq').value=''; if(SEGQ){ SEGQ=''; loadDist(); } };

loadDatasets();
</script>
</body></html>
"""


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
        help="one or more datasets (same shapes as visualize_frame_records.py: a "
             "stage-stage 03 frame_records dir/file, a stage-04 conversations dir/file, "
             "or a stage-06 inline-records dir) — auto-detected; choose in the UI",
    )
    p.add_argument(
        "--limit", "--limit-samples", dest="limit", type=_positive_int, default=None,
        help="aggregate at most the first K samples per dataset",
    )
    p.add_argument("--port", type=int, default=8780, help="HTTP port (default 8780)")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    return p.parse_args()


def main() -> None:
    global DATASET_SAMPLE_LIMIT
    args = parse_args()
    DATASET_SAMPLE_LIMIT = args.limit
    register_datasets(args.dataset)
    if not DATASETS:
        raise SystemExit("no datasets given")
    print(f"registered {len(DATASETS)} dataset(s)  [mode · action_format]:", flush=True)
    for name, entry in DATASETS.items():
        fmt = entry["action_format"] or "sampled?"
        print(f"  {name}  [{entry['mode']} · {fmt}]  {entry['path']}", flush=True)
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
