#!/usr/bin/env python3
"""Viewer for stage 01b (frame-records) datasets — frames + binned actions.

The spiritual successor to ``ylli_visualizer`` (the realignment inspector), but
pointed at the **01b sampler output** instead of raw mp4 + keylogs. 01b has
already done the work the old inspector re-derived on the fly: target-fps
sampling, NO_OP head/tail thinning, and per-bin action strings, with each kept
frame referenced as an ``ar:///…/images.array_record#idx`` URI into the stage
01a frames-master store. So this viewer is *much* thinner: no ffmpeg, no video
transcode, no raw-vs-realigned dual clock — just browse the trajectory 01b
produced, one frame + one action per step.

Input contract (one JSON object per line, the ``frame_records.jsonl`` schema
that stage 01 emits and 01b is expected to reproduce)::

    {
      "recording_id":    str,
      "segment_id":      str,          # trajectory / grouping key
      "segment_idx":     int,
      "local_bin_idx":   int,          # bin index within the segment
      "local_time_s":    float,        # seconds from segment start
      "source_frame_idx": int,         # provenance: frame in the original video
      "image_path":      "ar:///…/images.array_record#idx",   # or "image"
      "action":          "NO_OP" | "<dx> <dy> <scroll>"
                         | "<dx> <dy> <scroll> ; +KeyW -KeyW +LMB"   # format_action
    }

The ``action`` grammar (``realigned_pipeline.lib.common.format_action``): ``NO_OP``,
or ``"<dx> <dy> <scroll>"`` (summed pixel deltas + scroll), optionally followed
by ``" ; "`` and space-separated ``+Name``/``-Name`` press/release tokens in
temporal order. Names are rdev keys (``KeyA``, ``Num1``, ``Space``, ``Backspace``,
``ShiftLeft`` …) and mouse buttons (``LMB``, ``RMB``, ``MMB``). The front-end
parses this to light a keyboard, draw a mouse arrow, and reconstruct typed text.

Records for one segment must be contiguous and in trajectory order (both stage
01 and 01b stream ``chat.jsonl``/segments in order, so this holds). Any extra
top-level keys are ignored, so the viewer keeps working if 01b enriches rows.

Frames are resolved through ``realigned_pipeline.lib.image_store`` (the same
``ar://`` resolver stage 02 uses), so it also renders plain image-file paths.
The browser only ever asks for ``(segment, frame_index)``; the server maps that
to the image ref held in its own index, so there is no client-supplied path.

Run::

    cd .../data_pipeline
    uv run python realigned_pipeline/visualize_frame_records.py \
        --dataset <dir_or_file> [<dir_or_file> ...] \
        --port 8770
    # then SSH-forward the port and open http://127.0.0.1:8770/
    #   ssh -L 8770:127.0.0.1:8770 <host>

Pass several datasets and switch between them in the UI's "dataset" dropdown;
each is built lazily on first selection (a build failure — e.g. an empty or
not-yet-generated dir — is reported inline, the others keep working).

A store too big to browse whole is sampled: ``--limit N`` loads N samples (default
100, blank N in the UI loads every one), and the header's **samples** control picks
WHICH N — ``first N`` in store order
(cheap: the loader stops at N+1) or ``random N`` under a **seed**. A random draw
is deterministic in ``(seed, N, store)``: the same seed always yields the same N
samples, so a finding stays reachable and shareable ("dataset X, random 500, seed
7"). Only the membership is random — the list stays in store order. Mode, N and
seed are switchable per dataset in the UI without a restart (``--sample-mode`` /
``--seed`` just set the initial state); each sampling is one cached build, so
toggling back to one you already looked at doesn't re-read the store.

``--dataset`` may point at the 01b output directory (a ``frame_records.jsonl``
at its root or one level down is discovered), or directly at a
``frame_records.jsonl`` file.

It also opens a stage-01a **frames-master** store directly (auto-detected by its
``segment_index.jsonl`` + ``frames/`` layout / ``juergen_annotation_frames_master``
marker): the raw decoded frames are browsable with no sampler run, but since 01a
is keylog-free the action HUD stays empty — run 01b to get actions.

And it opens a stage-04 **conversations** dataset (auto-detected by a
``conversations.jsonl`` / ``juergen_annotation_conversations`` marker): each
segment's interleaved screenshot→action chat is browsed as a trajectory — the
user-turn screenshots are the frames, the following assistant turn is that frame's
action — so the same frame/action/HUD/timeline UI applies, plus a banner with the
system prompt and (if goal-conditioned) the instruction. The images are ``ar://``
refs into the same stage-01a master, so frames and the black-frame flag resolve
just as in the 01b view.

Conversations may also carry **native computer_use actions** (manifest
``action_format: computer_use_rel_v1`` — Qwen tool-call SFT): each assistant turn
is an optional ``<think>…</think>`` block plus one or more
``<tool_call>{"name": "computer_use", "arguments": {…}}</tool_call>`` blocks.
Detected per turn and translated once at load time into (a) an equivalent
``format_action`` string (``hud``) that drives the keyboard/mouse HUD, typed-text
reconstruction and filter metrics unchanged, and (b) a compact summary (``disp``,
e.g. ``💭 move(205,-105) · click · type("hi")``) shown in the action rows /
status line / timeline tooltips. The verbatim native text stays in ``action``
(full-chat window + "conversation contains" search).

They may instead carry **ordered-events actions** (manifest ``action_format:
ordered_events_v2`` / ``…_v3`` — the thinking SFT): each assistant turn is an
optional ``<think>…</think>`` block plus one action line — ``NO_OP``,
``TERMINATE``, or ``"; "``-separated primitives in performed order:
``move(dx,dy)``, ``scroll(dx,dy)`` (horizontal, vertical), ``down(EV)``,
``up(EV)`` and (v3) ``type("…")``. Same treatment, translated per turn at load:
``hud`` passes the events through verbatim (EV names are already rdev, the
HUD's namespace) with movement/scroll summed into the ``format_action`` head —
the radar shows one pointer state per bin, and that sum exists for it alone.
``disp`` is the action line AS IS: every primitive, in performed order, nothing
coalesced or abbreviated, because in this format the sequence IS the label.
The raw text (thinking included) stays in ``action``, one hover away.

Any turn can be found by substring — "action contains" in the filter panel
searches every turn of every segment (``down(LMB)``, ``move(-100,``, ``type("``),
narrows the segment list to those that have it, marks the matching turns in the
timeline and the rows list, and ``n``/``N`` step through them.

Finally it opens a stage-04 → stage-06 **inline SFT records** store (auto-detected
by a ``manifest.json`` marking stage ``inline_records`` / a ``train``+``val`` layout
of ArrayRecord shards): each record is one tokenized training example — a
``<= max_length`` chunk of a conversation, still carrying its ``ar://`` frame refs —
so it browses exactly like the stage-04 view, plus a banner showing the train/val
split and how full the token budget is. Since only the KEPT records survive stage
06's overflow drop/truncate, this is literally the dataset fed to the model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Make the ``realigned_pipeline`` package importable when run directly
# (mirrors build_frames_master.py's PYTHONPATH setup).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib import config  # noqa: E402
from realigned_pipeline.lib.image_store import (  # noqa: E402
    is_arrayrecord_image_uri,
    parse_arrayrecord_image_uri,
    read_jpeg_bytes,
)
from realigned_pipeline.lib.common import (  # noqa: E402
    aggregate_actions,
    format_action,
    load_keylog_entries,
    resolve_button_name,
    resolve_key_name,
)
from realigned_pipeline.lib import realign_lib as R  # noqa: E402  (keylog_to_video)

# Image ref field, tried in order (stage 01 rewrites ``image_path`` to ar://;
# the grain manifests / stage 02 use ``image``).
_IMAGE_KEYS = ("image_path", "image", "image_uri")

# Dataset registry: display-name -> {"path": Path, "mode": str, "objs": {samp_key: built}}.
# Datasets are built lazily on first access — a frames-master build is cheap, but a
# frame_records build eager-loads every row, so we defer it until the dataset is
# actually selected in the UI. One dataset can be built under several samplings
# (first N vs random N with a seed); each build is cached under its sampling key.
DATASETS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

# How many builds of ONE dataset (distinct samplings) stay in memory. Toggling
# first/random or trying seeds shouldn't re-read the store every time, but each
# build holds its own segments, so the per-dataset cache is small and LRU.
_SAMPLING_CACHE_CAP = 4

# Fallback keylog source for overlaying raw actions on frames-master datasets (a
# stage-00 clips_manifest.jsonl mapping segment_id -> keylog_path). Normally each
# master is auto-linked via its manifest.json "source_clips_manifest"; this only
# covers masters that don't record it.
CLIPS_MANIFEST_OVERRIDE: "str | None" = None

# Optional stage-00 alignment source (an alignment.jsonl file or the realign
# clip-manifest dir containing it). When set/discovered, master datasets get the
# dual-clock event table + the "aligned + trims" timeline. Applies to master only.
ALIGNMENT_OVERRIDE: "str | None" = None

# Optional per-dataset load cap. For eager trajectory stores this means the first
# K segments/conversations/records; for frames-master it caps the listed segments.
DATASET_SAMPLE_LIMIT: "int | None" = None

# Default sampling the UI starts with ("first" K in file order, or K drawn at
# random under DATASET_SAMPLE_SEED). Both are overridden per request, so the UI
# can switch without a restart; these only seed the controls.
DATASET_SAMPLE_MODE = "first"
DATASET_SAMPLE_SEED = 0

_COALESCE_MOVES = 4


class Sampling:
    """WHICH K samples of a dataset to load — the first K in file order, or K drawn
    at random.

    Random draws are fully deterministic: ``(seed, n, population size)`` decides the
    index set, so the same seed on the same store always yields the same K samples
    (and the same UI list, in file order — only the *membership* is random, not the
    ordering). ``mode="first"`` keeps the old behaviour, including the cheap early
    break that never reads past sample K; a random draw has to see the whole
    population first, so it costs one extra enumeration pass — free for inline
    records (ArrayRecord footers) and a frames-master (its segment index), a line
    count for a conversations file, a full scan for a multi-file frame_records store
    (~100s over 30k per-clip files). Each build is cached per (mode, n, seed), so
    that pass is paid once."""

    __slots__ = ("mode", "n", "seed")

    def __init__(self, mode: str = "first", n: "int | None" = None, seed: int = 0) -> None:
        # A random draw without a size is just "everything" — fall back to first/all
        # rather than silently sampling nothing.
        self.mode = "random" if (mode == "random" and n is not None) else "first"
        self.n = n
        self.seed = seed

    @property
    def is_random(self) -> bool:
        return self.mode == "random"

    def key(self) -> str:
        """Cache key: the seed only matters for a random draw."""
        return f"{self.mode}:{self.n}:{self.seed if self.is_random else 0}"

    def select(self, total: int) -> list[int]:
        """The sample indices to keep out of ``total``, ascending (file order)."""
        if self.n is None or self.n >= total:
            return list(range(total))
        if self.is_random:
            return sorted(random.Random(self.seed).sample(range(total), self.n))
        return list(range(self.n))

    def public(self) -> dict[str, Any]:
        """The sampling as the UI sees it (echoed back so the controls show what the
        server actually did, not what was asked for)."""
        return {"mode": self.mode, "n": self.n, "seed": self.seed}

    def note(self, *, kept: int, total: "int | None", limited: bool, noun: str) -> str:
        """The load-banner suffix describing what this sampling kept."""
        if self.is_random:
            return (
                f" (random {kept} of {total if total is not None else '?'} {noun}"
                f", seed {self.seed})"
            )
        if limited and self.n is not None:
            return f" (limited to first {self.n} {noun})"
        return ""


def _resolve_alignment_path(clips_manifest: "str | Path | None") -> "Path | None":
    """Locate an ``alignment.jsonl``: ``--alignment`` (a file or a dir holding one),
    else a ``*realign*`` sibling of the (raw) clips_manifest dir that belongs to the
    SAME dataset family (matched by the manifest dir's distinctive prefix, so e.g. an
    ``eval`` master never picks up a ``subset100`` alignment)."""
    if ALIGNMENT_OVERRIDE:
        p = Path(ALIGNMENT_OVERRIDE).expanduser()
        if p.is_dir():
            p = p / "alignment.jsonl"
        return p if p.exists() else None
    if not clips_manifest:
        return None
    manifest_dir = Path(clips_manifest).expanduser().resolve().parent
    prefix = manifest_dir.name
    for tok in ("_rerun_clip_manifest", "_clip_manifest_rerun", "_clip_manifest", "_manifest"):
        if prefix.endswith(tok):
            prefix = prefix[: -len(tok)]
            break
    for sibling in sorted(manifest_dir.parent.glob(f"{prefix}*realign*")):
        candidate = sibling / "alignment.jsonl"
        if candidate.exists():
            return candidate
    return None


def _raw_events(
    keylog_path: Path,
    splices: "list[dict] | None" = None,
    video_end: "float | None" = None,
) -> list[list[Any]]:
    """Keylog events as table rows.

    Without ``splices``: ``[t_raw, type, detail]`` on the raw keylog clock.
    With ``splices`` (the stage-00 alignment map): **dual-clock**
    ``[t_raw, t_aln, type, detail, trimmed]`` where ``t_aln = keylog_to_video`` and
    ``trimmed`` marks events inside a collapsed span (kept content folded to one
    instant) or past ``video_end`` (overhang dropped off the video).

    Individual events (not per-frame bins); consecutive MouseMoves are coalesced
    into one row (summed dx/dy + count). Times are seconds."""
    dual = splices is not None
    rows: list[list[Any]] = []
    mm_n = 0
    mm_dx = mm_dy = 0.0
    mm_t0: float | None = None

    def emit(t_raw: float, etype: str, detail: str) -> None:
        if dual:
            t_aln = R.keylog_to_video(t_raw, splices)
            trimmed = (video_end is not None and t_aln >= video_end) or any(
                s["kp"] <= t_raw < s["kp"] + s["collapse"] for s in splices
            )
            rows.append([round(t_raw, 3), round(t_aln, 3), etype, detail, trimmed])
        else:
            rows.append([round(t_raw, 3), etype, detail])

    def flush_mm() -> None:
        nonlocal mm_n, mm_dx, mm_dy, mm_t0
        if mm_n:
            emit(mm_t0, "MouseMove", f"{round(mm_dx)} {round(mm_dy)} (x{mm_n})")
            mm_n, mm_dx, mm_dy, mm_t0 = 0, 0.0, 0.0, None

    for entry in load_keylog_entries(keylog_path):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        ts, event = entry[0], entry[1]
        if not isinstance(event, list) or not event:
            continue
        try:
            t = int(ts) / 1e6
        except (TypeError, ValueError):
            continue
        et = str(event[0])
        payload = event[1] if len(event) > 1 else None
        if et == "ContextChanged":
            continue
        if et == "MouseMove":
            if isinstance(payload, list) and len(payload) >= 2:
                if mm_n == 0:
                    mm_t0 = t
                mm_dx += float(payload[0])
                mm_dy += float(payload[1])
                mm_n += 1
                if mm_n >= _COALESCE_MOVES:
                    flush_mm()
            continue
        flush_mm()
        if et in ("KeyPress", "KeyRelease"):
            detail = resolve_key_name(payload) or "?"
        elif et in ("MousePress", "MouseRelease"):
            detail = resolve_button_name(payload) or "?"
        elif et == "MouseScroll":
            detail = " ".join(str(x) for x in payload[:2]) if isinstance(payload, list) else "?"
        else:
            detail = ""
        emit(t, et, detail)
    flush_mm()
    # Dual: order by the aligned clock (what the viewer navigates by); raw: by t_raw.
    rows.sort(key=(lambda r: (r[1], r[0])) if dual else (lambda r: r[0]))
    return rows


def _image_ref(rec: dict[str, Any]) -> str | None:
    for k in _IMAGE_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    return None


_SID_RES = (
    re.compile(r'"segment_id"\s*:\s*"([^"\\]*)"'),
    re.compile(r'"clip_id"\s*:\s*"([^"\\]*)"'),
)


def _peek_segment_id(line: str) -> str:
    """One frame_records line's grouping key, WITHOUT parsing the whole record — a
    random draw has to see every line, and these files run to millions of them.

    Mirrors the loader's own ``segment_id or clip_id or "unknown"`` precedence
    (each key tried in turn, so field order in the JSON doesn't matter), falling
    back to a real parse when the cheap scan misses (escaped or non-string id)."""
    for rx in _SID_RES:
        m = rx.search(line)
        if m and m.group(1):
            return m.group(1)
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return "unknown"
    return str(rec.get("segment_id") or rec.get("clip_id") or "unknown")


def _is_noop(action: Any) -> bool:
    return not action or str(action).strip().upper() == "NO_OP"


def _is_black(mean_luma: Any, frac_dark: Any) -> bool:
    """Black-frame flag from the stage-01a luma metrics, using the sampler's
    default thresholds (``realigned_pipeline.lib.config``). Mirrors
    ``sample_frames_actions._is_black`` so both views flag exactly the frames 01b
    would drop under default settings — the master reads its own manifest, the
    01b sample cross-references the master's. Frames without metrics (older
    masters, decode failures) are never flagged (absence of evidence != black)."""
    return (mean_luma is not None and mean_luma <= config.DEFAULT_BLACK_LUMA_MAX) or (
        frac_dark is not None and frac_dark >= config.DEFAULT_BLACK_DARK_FRAC_MIN
    )


# --------------------------------------------------------------------------- #
# Per-trajectory metrics (mouse travel, clicks, scroll, typed-text stats, and the
# per-TURN action counts — how many primitives / tool calls one turn performs).
# These power the left filter panel (alongside the always-present "segment id
# contains" box, the way to reach a known segment id without scrolling the
# dropdown): the client filters the segment list on them,
# so they must be computed for EVERY segment up front. The action-parse and
# typed-text reconstruction mirror the front-end's parseAction/stateAt/keyToChar
# exactly, so a segment's metrics match what the HUD reconstructs when you open it.
# --------------------------------------------------------------------------- #
_PUNCT = {
    "BackQuote": ("`", "~"), "Minus": ("-", "_"), "Equal": ("=", "+"),
    "BracketLeft": ("[", "{"), "BracketRight": ("]", "}"), "BackSlash": ("\\", "|"),
    "SemiColon": (";", ":"), "Quote": ("'", '"'), "Comma": (",", "<"),
    "Dot": (".", ">"), "Slash": ("/", "?"),
}
_SHIFTNUM = {"Num1": "!", "Num2": "@", "Num3": "#", "Num4": "$", "Num5": "%",
             "Num6": "^", "Num7": "&", "Num8": "*", "Num9": "(", "Num0": ")"}


def _key_to_char(name: str, shift: bool) -> str | None:
    """One rdev key name -> the character it types, or None for modifiers / arrows /
    F-keys (Backspace is handled by the caller). Mirrors the front-end keyToChar."""
    if len(name) == 4 and name.startswith("Key") and name[3].isalpha():
        c = name[3].upper()
        return c if shift else c.lower()
    if len(name) == 4 and name.startswith("Num") and name[3].isdigit():
        return _SHIFTNUM[name] if (shift and name in _SHIFTNUM) else name[3]
    if name == "Space":
        return " "
    if name in ("Return", "Enter", "NumpadEnter"):
        return "\n"
    if name == "Tab":
        return "\t"
    if name in _PUNCT:
        return _PUNCT[name][1 if shift else 0]
    return None


def _parse_action_str(
    action: Any,
) -> tuple[float, float, float, float, list[tuple[str, str]]]:
    """Parse a ``format_action`` string into
    ``(dx, dy, scroll, hscroll, [(sign, name), ...])``.

    Tolerates a conversation assistant turn that prefixes the action with a
    natural-language plan (``"plan text\\n<dx> <dy> <scroll> ; +Key... -Key..."``):
    the movement is read as the trailing numeric run of the last line before the
    ``" ; "`` separator, and the press/release tokens from after it. A movement
    line of EXACTLY four numbers is the extended ``<dx> <dy> <scroll> <hscroll>``
    form emitted by the native computer_use translation; otherwise hscroll is 0."""
    if not action:
        return (0.0, 0.0, 0.0, 0.0, [])
    s = str(action)
    if s.strip().upper() == "NO_OP":
        return (0.0, 0.0, 0.0, 0.0, [])
    parts = s.split(" ; ")
    last_line = parts[0].splitlines()[-1] if parts[0] else ""
    toks = last_line.split()
    hscroll = 0.0
    vals: list[float] = []
    try:
        vals = [float(t) for t in toks]
    except ValueError:
        vals = []
    if len(vals) == 4:
        dx, dy, scroll, hscroll = vals
    else:
        nums: list[float] = []
        for tok in reversed(toks):
            try:
                nums.append(float(tok))
            except ValueError:
                break
            if len(nums) == 3:
                break
        nums.reverse()
        dx = nums[0] if len(nums) >= 1 else 0.0
        dy = nums[1] if len(nums) >= 2 else 0.0
        scroll = nums[2] if len(nums) >= 3 else 0.0
    events: list[tuple[str, str]] = []
    if len(parts) > 1:
        for tok in " ; ".join(parts[1:]).split():
            if len(tok) > 1 and tok[0] in "+-":
                events.append((tok[0], tok[1:]))
    return (dx, dy, scroll, hscroll, events)


# --------------------------------------------------------------------------- #
# Native computer_use (Qwen tool-call) actions — ``action_format:
# computer_use_rel_v1`` conversations, whose assistant turns are an optional
# ``<think>…</think>`` block plus ``<tool_call>{JSON}</tool_call>`` blocks
# instead of format_action strings. Each turn is translated once at load time:
#   hud  — an equivalent format_action string ("<dx> <dy> <scroll>[ <hscroll>]
#          ; +Key -Key …") consumed unchanged by compute_metrics and the
#          front-end HUD/typed-text parsers;
#   disp — the calls spelled out one-for-one for the rows / status line /
#          tooltips (the JSON is what is unreadable, not the gestures), each
#          rendered in full — no elision, no merging of calls.
# The verbatim native text stays in the frame's ``action``.
# --------------------------------------------------------------------------- #
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

# Qwen lowercase key names -> the rdev names the keyboard HUD / typed-text
# reconstruction know. Unknown names pass through and render as chips.
_QWEN_KEY_TO_RDEV = {
    "shift": "ShiftLeft", "ctrl": "ControlLeft", "control": "ControlLeft",
    "alt": "Alt", "option": "Alt", "altgr": "AltGr",
    "command": "MetaLeft", "cmd": "MetaLeft", "meta": "MetaLeft",
    "win": "MetaLeft", "super": "MetaLeft",
    "enter": "Return", "return": "Return", "space": "Space",
    "backspace": "Backspace", "tab": "Tab", "esc": "Escape", "escape": "Escape",
    "capslock": "CapsLock", "delete": "Delete", "insert": "Insert",
    "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
    "left": "LeftArrow", "right": "RightArrow", "up": "UpArrow", "down": "DownArrow",
    "leftbracket": "BracketLeft", "rightbracket": "BracketRight",
    "-": "Minus", "=": "Equal", "\\": "BackSlash", ";": "SemiColon",
    "'": "Quote", ",": "Comma", ".": "Dot", "/": "Slash", "`": "BackQuote",
}


def _qwen_key(name: Any) -> str:
    s = str(name)
    low = s.lower()
    if len(low) == 1 and "a" <= low <= "z":
        return f"Key{low.upper()}"
    if len(low) == 1 and low.isdigit():
        return f"Num{low}"
    if low in _QWEN_KEY_TO_RDEV:
        return _QWEN_KEY_TO_RDEV[low]
    if re.fullmatch(r"f[0-9]{1,2}", low):
        return low.upper()
    return s


# char -> (rdev key name, needs shift); the inverse of _key_to_char, so a
# ``type`` action reconstructs to exactly its text in the typed-text panel.
_CHAR_TO_KEY: dict[str, tuple[str, bool]] = {}
for _name, (_plain, _shifted) in _PUNCT.items():
    _CHAR_TO_KEY[_plain] = (_name, False)
    _CHAR_TO_KEY[_shifted] = (_name, True)
for _name, _ch in _SHIFTNUM.items():
    _CHAR_TO_KEY[_ch] = (_name, True)
_CHAR_TO_KEY.update({" ": ("Space", False), "\n": ("Return", False), "\t": ("Tab", False)})


def _char_key(ch: str) -> "tuple[str, bool] | None":
    low = ch.lower()
    if len(low) == 1 and "a" <= low <= "z":
        return (f"Key{low.upper()}", ch.isupper())
    if ch.isdigit() and ord(ch) < 128:
        return (f"Num{ch}", False)
    return _CHAR_TO_KEY.get(ch)


def _type_events(text: str) -> list[tuple[str, str]]:
    """A ``type`` action as press/release events, with ShiftLeft held around
    shifted runs. Characters with no key mapping (unicode, …) are skipped."""
    events: list[tuple[str, str]] = []
    shifted = False
    for ch in text:
        ck = _char_key(ch)
        if ck is None:
            continue
        name, need_shift = ck
        if need_shift != shifted:
            events.append(("+", "ShiftLeft") if need_shift else ("-", "ShiftLeft"))
            shifted = need_shift
        events.append(("+", name))
        events.append(("-", name))
    if shifted:
        events.append(("-", "ShiftLeft"))
    return events


_CLICK_EVENTS = {
    "left_click": ("LMB", 1, "click"), "right_click": ("RMB", 1, "rclick"),
    "middle_click": ("MMB", 1, "mclick"), "double_click": ("LMB", 2, "dblclick"),
    "triple_click": ("LMB", 3, "triclick"),
}
_QWEN_BTN = {"left": "LMB", "right": "RMB", "middle": "MMB"}


def _fmt_num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "0"
    return str(int(f)) if f == int(f) else f"{f:g}"


def _signed(v: float) -> str:
    return ("+" if v > 0 else "") + _fmt_num(v)


def parse_native_action(text: str) -> "tuple[str, str] | None":
    """Translate one native computer_use assistant turn into ``(hud, disp)``,
    or None when ``text`` carries no ``<tool_call>`` block (not native format).

    Mirrors the binning semantics of ``format_action``: movement/scroll summed
    over the turn, press/release events in call order (clicks become +/- pairs,
    ``key`` presses in order then releases in reverse, ``button_down``/``key_down``
    stay held across turns until their up). ``wait`` contributes nothing (a
    wait-only turn is a NO_OP); ``terminate`` becomes a +Terminate -Terminate
    chip so the HUD marks the turn.

    Covers the ``computer_use_rel_v1`` vocabulary and the ``pyautogui_gestures``/
    ``pyautogui_primitives`` variants: ``move``/``mouse_move`` (arg ``offset``),
    ``scroll`` with an ``offset`` [h, v] vector, clicks with an optional pre-move
    ``offset``, ``left_click_drag``, and ``down``/``up`` (whose ``key`` may be an
    rdev name like ``KeyC``/``LMB`` — passed through — or a lowercase Qwen name)."""
    calls = _TOOL_CALL_RE.findall(text)
    if not calls:
        return None
    dx = dy = scroll = hscroll = 0.0
    events: list[tuple[str, str]] = []
    disp: list[str] = []
    for raw in calls:
        try:
            args = json.loads(raw).get("arguments") or {}
        except json.JSONDecodeError:
            disp.append("?")
            continue
        act = str(args.get("action") or "")

        def _vec2(keys: tuple[str, ...] = ("delta", "offset")) -> tuple[float, float]:
            for k in keys:
                v = args.get(k)  # noqa: B023 — consumed within this iteration
                if isinstance(v, (list, tuple)) and v:
                    return (
                        float(v[0]) if len(v) >= 1 else 0.0,
                        float(v[1]) if len(v) >= 2 else 0.0,
                    )
            return (0.0, 0.0)

        if act in ("mouse_move_rel", "mouse_move", "move"):
            mdx, mdy = _vec2()
            dx += mdx
            dy += mdy
            disp.append(f"move({_fmt_num(mdx)},{_fmt_num(mdy)})")
        elif act == "scroll":
            if args.get("offset") is not None:  # pyautogui variants: [h, v]
                h, v = _vec2(("offset",))
                hscroll += h
                scroll += v
                disp.append(f"scroll({_fmt_num(h)},{_fmt_num(v)})")
            else:
                v = float(args.get("pixels") or 0.0)
                scroll += v
                disp.append(f"scroll({_signed(v)})")
        elif act == "hscroll":
            v = float(args.get("pixels") or 0.0)
            hscroll += v
            disp.append(f"hscroll({_signed(v)})")
        elif act in _CLICK_EVENTS:
            mdx, mdy = _vec2(("offset",))  # optional pre-move (pyautogui gestures)
            dx += mdx
            dy += mdy
            btn, n, label = _CLICK_EVENTS[act]
            events.extend([("+", btn), ("-", btn)] * n)
            disp.append(label if mdx == mdy == 0 else f"{label}({_fmt_num(mdx)},{_fmt_num(mdy)})")
        elif act == "left_click_drag":
            mdx, mdy = _vec2(("offset",))
            dx += mdx
            dy += mdy
            events.extend([("+", "LMB"), ("-", "LMB")])
            disp.append(f"drag({_fmt_num(mdx)},{_fmt_num(mdy)})")
        elif act in ("button_down", "button_up"):
            raw_btn = str(args.get("button") or "left")
            btn = _QWEN_BTN.get(raw_btn.lower(), "LMB")
            if act == "button_down":
                events.append(("+", btn))
                disp.append(f"hold({raw_btn})")
            else:
                events.append(("-", btn))
                disp.append(f"release({raw_btn})")
        elif act == "key":
            names = [_qwen_key(k) for k in (args.get("keys") or [])]
            events.extend(("+", k) for k in names)
            events.extend(("-", k) for k in reversed(names))
            disp.append("key(" + "+".join(str(k) for k in (args.get("keys") or [])) + ")")
        elif act in ("key_down", "down"):
            events.append(("+", _qwen_key(args.get("key") or "")))
            disp.append(f"down({args.get('key')})")
        elif act in ("key_up", "up"):
            events.append(("-", _qwen_key(args.get("key") or "")))
            disp.append(f"up({args.get('key')})")
        elif act == "type":
            t = str(args.get("text") or "")
            events.extend(_type_events(t))
            disp.append("type(" + json.dumps(t) + ")")   # verbatim, never elided
        elif act == "wait":
            disp.append(f"wait({_fmt_num(args.get('time') or 0)}s)")
        elif act == "terminate":
            events.extend([("+", "Terminate"), ("-", "Terminate")])
            disp.append(f"terminate({args.get('status') or '?'})")
        else:
            disp.append(act or "?")
    if dx == 0 and dy == 0 and scroll == 0 and hscroll == 0 and not events:
        hud = "NO_OP"
    else:
        head = f"{round(dx)} {round(dy)} {round(scroll)}"
        if round(hscroll):
            head += f" {round(hscroll)}"
        hud = head if not events else head + " ; " + " ".join(f"{s}{n}" for s, n in events)
    d = " · ".join(disp) or "∅"
    if "<think>" in text:
        d = "💭 " + d
    return hud, d


# --------------------------------------------------------------------------- #
# Ordered-events actions — ``action_format: ordered_events_v2`` / ``…_v3``
# (thinking SFT, lib/action_format.py of the thinking pipeline): each assistant
# turn is an optional ``<think>…</think>`` block plus one action line —
# ``NO_OP``, ``TERMINATE``, or ``"; "``-separated primitives in performed
# order: ``move(dx,dy)``, ``scroll(dx,dy)`` (horizontal, vertical),
# ``down(EV)``, ``up(EV)`` and (v3) ``type("…")``. EV names are already rdev
# (``KeyA``, ``ShiftLeft``, ``LMB`` …) — the HUD's native namespace — so the
# ``hud`` translation passes events through verbatim and only sums the
# movement/scroll into the ``format_action`` head (that sum is the radar's
# semantics, nothing more); ``disp`` is the action line AS IS, primitive for
# primitive, since that sequence is what the dataset teaches.
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_ORDERED_PRIM_RE = re.compile(
    r"(?P<mk>move|scroll)\((?P<mdx>-?\d+),(?P<mdy>-?\d+)\)"
    r"|(?P<ud>down|up)\((?P<name>[^\s(),;]+)\)"
    r'|type\("(?P<text>(?:[^"\\]|\\.)*)"\)'
)
_ORDERED_BUTTONS = frozenset({"LMB", "RMB", "MMB"})


def _ordered_primitives(payload: str) -> "list[tuple[Any, ...]] | None":
    """Parse the (think-stripped) payload of an ordered_events_v2/v3 turn into
    ``("move"|"scroll", dx, dy) | ("down"|"up", name) | ("type", text) |
    ("terminate",)`` tuples, or None when any line deviates from the grammar
    (not this format — e.g. a ``format_action`` string or free prose)."""
    prims: list[tuple[Any, ...]] = []
    saw_line = False
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        saw_line = True
        if line == "NO_OP":
            continue
        if line == "TERMINATE":  # stage 04's goal-done turn / suffix line
            prims.append(("terminate",))
            continue
        pos = 0
        while True:
            m = _ORDERED_PRIM_RE.match(line, pos)
            if m is None:
                return None
            if m.group("mk") is not None:
                prims.append((m.group("mk"), int(m.group("mdx")), int(m.group("mdy"))))
            elif m.group("ud") is not None:
                prims.append((m.group("ud"), m.group("name")))
            else:
                prims.append(("type", re.sub(r'\\(["\\])', r"\1", m.group("text"))))
            pos = m.end()
            if pos == len(line):
                break
            if not line.startswith("; ", pos):
                return None
            pos += 2
    return prims if saw_line else None


def parse_ordered_action(text: str) -> "tuple[str, str] | None":
    """Translate one ordered_events_v2/v3 assistant turn into ``(hud, disp)``,
    or None when ``text`` is not that format (tried after
    ``parse_native_action``; a plain ``format_action`` string or prose never
    matches the primitive grammar).

    hud mirrors ``format_action``: movement and scroll summed over the turn
    (``scroll(dx,dy)`` splits into vertical ``<scroll>`` + horizontal
    ``<hscroll>``), every ``down``/``up`` a ``+Name``/``-Name`` event verbatim
    (the names are already rdev) — so held keys, clicks and typed text
    reconstruct exactly in the HUD and metrics. ``type("…")`` (v3) expands to
    per-char press/release via ``_type_events``; ``TERMINATE`` becomes the same
    ``+Terminate -Terminate`` chip as the native translation. This summing is
    the HUD's own semantics (one radar vector + one keyboard state per bin) and
    goes no further than that.

    disp is the action line VERBATIM — the primitives in performed order,
    exactly as the label spells them. Nothing is coalesced, folded or
    abbreviated: an eleven-move turn reads as eleven moves, because the whole
    point of the ordered format is that the sequence is the label. Only the
    ``<think>`` block is lifted out (marked 💭; the full raw text stays one hover
    away in the row title and in the chat window)."""
    body = _THINK_RE.sub("", text).strip()
    if not body:
        return None
    prims = _ordered_primitives(body)
    if prims is None:
        return None
    dx = dy = scroll = hscroll = 0
    events: list[tuple[str, str]] = []
    for p in prims:
        if p[0] == "move":
            dx += p[1]
            dy += p[2]
        elif p[0] == "scroll":
            hscroll += p[1]
            scroll += p[2]
        elif p[0] == "down":
            events.append(("+", p[1]))
        elif p[0] == "up":
            events.append(("-", p[1]))
        elif p[0] == "type":
            events.extend(_type_events(p[1]))
        else:  # terminate
            events.extend([("+", "Terminate"), ("-", "Terminate")])
    if dx == 0 and dy == 0 and scroll == 0 and hscroll == 0 and not events:
        hud = "NO_OP"
    else:
        head = f"{dx} {dy} {scroll}"
        if hscroll:
            head += f" {hscroll}"
        hud = head if not events else head + " ; " + " ".join(f"{s}{n}" for s, n in events)
    # Verbatim, single-line (a turn is one action line; a stage-04 TERMINATE
    # suffix is a second one, joined with the same "; " the grammar uses).
    d = "; ".join(ln.strip() for ln in body.splitlines() if ln.strip()) or "NO_OP"
    if "<think>" in text:
        d = "💭 " + d
    return hud, d


# Tool-call actions that drive the pointer (the ``mouse`` half of the per-turn
# counts); everything else in the vocabulary is typing / keys / wait / terminate.
_MOUSE_TOOL_ACTIONS = frozenset({
    "mouse_move_rel", "mouse_move", "move", "scroll", "hscroll",
    "left_click_drag", "button_down", "button_up", *_CLICK_EVENTS,
})


def turn_action_counts(text: str, hud: "str | None" = None) -> tuple[int, int]:
    """``(actions, mouse actions)`` performed in ONE assistant turn.

    What counts as "an action" follows the turn's own format, because that is
    what the label teaches the model:

      * **ordered_events_v2/v3** — one per PRIMITIVE (``move``/``scroll``/
        ``down``/``up``/``type``). This is the format where a turn is a
        mini-program, so ``move(4,-1); move(6,0); down(LMB); up(LMB)`` is 4
        actions — the line is rendered verbatim, and this count is what makes
        its length filterable. ``TERMINATE`` is not an action.
      * **native computer_use** — one per tool call (one gesture each).
      * **plain ``format_action``** — a turn IS one binned action, so this
        counts what that bin does: its movement, its scroll, and each
        press/release event.

    ``mouse`` is the subset that drives the pointer — movement, scrolling and
    button transitions; typing and other keys are excluded."""
    body = _THINK_RE.sub("", text or "").strip()
    prims = _ordered_primitives(body) if body else None
    if prims is not None:
        prims = [p for p in prims if p[0] != "terminate"]
        mouse = sum(
            1 for p in prims
            if p[0] in ("move", "scroll")
            or (p[0] in ("down", "up") and p[1] in _ORDERED_BUTTONS)
        )
        return len(prims), mouse
    calls = _TOOL_CALL_RE.findall(text or "")
    if calls:
        mouse = 0
        for raw in calls:
            try:
                act = str((json.loads(raw).get("arguments") or {}).get("action") or "")
            except json.JSONDecodeError:
                continue
            if act in _MOUSE_TOOL_ACTIONS:
                mouse += 1
        return len(calls), mouse
    dx, dy, scr, hscr, events = _parse_action_str(hud or text or "")
    moved = 1 if (dx or dy) else 0
    scrolled = 1 if (scr or hscr) else 0
    buttons = sum(1 for _sign, name in events if name in ("LMB", "RMB", "MMB"))
    return moved + scrolled + len(events), moved + scrolled + buttons


def find_actions(ds: Any, query: str) -> dict[str, Any]:
    """Segments whose turns contain ``query`` (case-insensitive substring).

    Searching happens HERE rather than over the segment list the client already
    holds, because the per-segment action text is exactly what that list must
    not carry: tens of thousands of segments × a full trajectory of primitives
    is a payload, not a filter. What comes back is only ``{segment_id: hits}``,
    which the sidebar intersects with its other predicates; the client then
    re-finds the matching turns inside the segment it opens (it has those
    frames) to highlight and step through them.

    Both the raw turn text and its ``disp`` rendering are searched, so
    ``down(LMB)`` reaches an ordered turn's primitives and ``click`` reaches a
    native tool call's readable form. A turn counts once however many times it
    matches."""
    q = (query or "").strip().lower()
    out: dict[str, Any] = {"query": query, "segments": {}, "n_turns": 0,
                           "n_segments": 0}
    if not q:
        return out
    hits: dict[str, int] = {}
    n_turns = 0
    for sid, seg in (getattr(ds, "segments", {}) or {}).items():
        # A frames-master store keeps its frames on disk (loaded per segment on
        # demand) and carries no actions of its own — nothing to search.
        frames = getattr(seg, "frames", None)
        if not frames:
            continue
        n = 0
        for f in frames:
            if q in str(f.get("action") or "").lower() or q in str(f.get("disp") or "").lower():
                n += 1
        if n:
            hits[str(sid)] = n
            n_turns += n
    out.update({"segments": hits, "n_segments": len(hits), "n_turns": n_turns})
    return out


def compute_metrics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mouse travel / scroll / clicks / key presses and typed-text stats
    over a trajectory's per-frame action strings. ``typed`` is the reconstructed
    typed string (for the "typed text contains" filter); the counts are derived
    from it so letters+digits+special+whitespace add up to ``chars``.

    Also annotates every frame with its per-turn action counts (``n_act`` /
    ``n_mouse``, see ``turn_action_counts``) and reduces them to the four
    filter-panel metrics ``acts_max`` / ``acts_mean`` / ``mouse_max`` /
    ``mouse_mean``. The means are per ACTING turn (a segment of mostly NO_OPs
    would otherwise read as sparse for the wrong reason), the maxima are over
    the whole segment — so "acts_max ≥ 20" finds the segments that CONTAIN a
    dense turn, and the annotation then says which turn it is.

    ``stuck_key_frames`` is the longest keyboard-only press->release gap in frame
    units. If a key never releases, the gap is measured through the segment's last
    frame. The UI applies the requested strict ``> T`` threshold against this value."""
    mouse = scroll_total = 0.0
    clicks = keys = 0
    acts_total = acts_max = mouse_total = mouse_max = n_acting = 0
    down: set[str] = set()
    down_since: dict[str, int] = {}
    chars: list[str] = []
    max_hold = 0
    max_hold_key: str | None = None
    max_hold_start: int | None = None
    max_hold_end: int | None = None
    max_hold_unreleased = False
    key_hold_ranges: list[dict[str, Any]] = []

    def note_hold(name: str, start: int, end: int, unreleased: bool) -> None:
        nonlocal max_hold, max_hold_key, max_hold_start, max_hold_end, max_hold_unreleased
        held = max(0, end - start)
        key_hold_ranges.append({
            "key": name,
            "start": start,
            "end": end,
            "frames": held,
            "unreleased": unreleased,
        })
        if held > max_hold:
            max_hold = held
            max_hold_key = name
            max_hold_start = start
            max_hold_end = end
            max_hold_unreleased = unreleased

    for frame_idx, f in enumerate(frames):
        # ``hud`` is the format_action translation of a native computer_use turn
        # (parse that, not the raw tool-call text); plain datasets have no hud.
        dx, dy, scr, hscr, events = _parse_action_str(f.get("hud") or f.get("action", ""))
        # The turn's own length, BEFORE the hud/disp collapse (an 11-move turn
        # sums to one vector above but is 11 actions here).
        n_act, n_mouse = turn_action_counts(f.get("action") or "", f.get("hud"))
        f["n_act"], f["n_mouse"] = n_act, n_mouse
        acts_total += n_act
        mouse_total += n_mouse
        acts_max = max(acts_max, n_act)
        mouse_max = max(mouse_max, n_mouse)
        if n_act:
            n_acting += 1
        mouse += math.hypot(dx, dy)
        scroll_total += abs(scr) + abs(hscr)
        for sign, name in events:
            if name in ("LMB", "RMB", "MMB"):
                if sign == "+":
                    clicks += 1
                continue
            if sign == "+":
                keys += 1
                down.add(name)
                down_since.setdefault(name, frame_idx)
                shift = "ShiftLeft" in down or "ShiftRight" in down
                if name == "Backspace":
                    if chars:
                        chars.pop()
                else:
                    ch = _key_to_char(name, shift)
                    if ch is not None:
                        chars.append(ch)
            else:
                start = down_since.pop(name, None)
                if start is not None:
                    note_hold(name, start, frame_idx, False)
                down.discard(name)
    last_frame = len(frames) - 1
    for name, start in down_since.items():
        note_hold(name, start, last_frame, True)
    text = "".join(chars)
    return {
        "mouse_px": round(mouse),
        "scroll": round(scroll_total),
        "clicks": clicks,
        "keys": keys,
        "acts_max": acts_max,
        "acts_mean": round(acts_total / n_acting, 2) if n_acting else 0.0,
        "mouse_max": mouse_max,
        "mouse_mean": round(mouse_total / n_acting, 2) if n_acting else 0.0,
        "chars": len(text),
        "letters": sum(c.isalpha() for c in text),
        "digits": sum(c.isdigit() for c in text),
        "special": sum((not c.isalnum()) and (not c.isspace()) for c in text),
        "typed": text,
        "stuck_key_frames": max_hold,
        "stuck_key": max_hold_key,
        "stuck_key_start": max_hold_start,
        "stuck_key_end": max_hold_end,
        "stuck_key_unreleased": max_hold_unreleased,
        "key_hold_ranges": key_hold_ranges,
    }


class Segment:
    """One trajectory: the ordered kept frames of a single ``segment_id``."""

    def __init__(self, segment_id: str, recording_id: str | None) -> None:
        self.segment_id = segment_id
        self.recording_id = recording_id
        self.frames: list[dict[str, Any]] = []  # normalized per-frame rows

    def summary(self) -> dict[str, Any]:
        n = len(self.frames)
        n_non_noop = sum(1 for f in self.frames if not f["is_noop"])
        dur = self.frames[-1]["t"] if self.frames else 0.0
        metrics = {
            k: v for k, v in self.metrics().items()
            if k != "key_hold_ranges"
        }
        return {
            "segment_id": self.segment_id,
            "recording_id": self.recording_id,
            "n_frames": n,
            "n_non_noop": n_non_noop,
            "duration_s": round(dur, 2),
            **metrics,
        }

    def metrics(self) -> dict[str, Any]:
        """Filter-panel metrics (mouse travel, clicks, scroll, key presses, typed
        text + char class counts), computed once from the action strings and cached
        so repeated ``info()`` builds are cheap."""
        m = getattr(self, "_metrics", None)
        if m is None:
            m = compute_metrics(self.frames)
            self._metrics = m  # type: ignore[attr-defined]
        return m

    def detail(self) -> dict[str, Any]:
        metrics = self.metrics()
        return {
            "segment_id": self.segment_id,
            "recording_id": self.recording_id,
            "n_frames": len(self.frames),
            "n_non_noop": sum(1 for f in self.frames if not f["is_noop"]),
            # ``is_black`` is filled in lazily by the dataset (cross-referenced
            # from the master frame_manifest); absent -> 0.
            "n_black": sum(1 for f in self.frames if f.get("is_black")),
            "n_black_act": sum(1 for f in self.frames if f.get("is_black") and not f["is_noop"]),
            "stuck_key_frames": metrics.get("stuck_key_frames"),
            "stuck_key": metrics.get("stuck_key"),
            "stuck_key_start": metrics.get("stuck_key_start"),
            "stuck_key_end": metrics.get("stuck_key_end"),
            "stuck_key_unreleased": metrics.get("stuck_key_unreleased"),
            "key_hold_ranges": metrics.get("key_hold_ranges"),
            # Drop the internal image ref from the payload; the client fetches
            # frames by (segment, index) via /frame instead.
            "frames": [{k: v for k, v in f.items() if k != "ref"} for f in self.frames],
        }


class FrameRecordsDataset:
    """All segments loaded from one or more ``frame_records.jsonl`` files."""

    def __init__(
        self, jsonl_paths: list[Path], sampling: "Sampling | None" = None
    ) -> None:
        self.segments: "OrderedDict[str, Segment]" = OrderedDict()
        self.sampling = sampling or Sampling()
        self.limit = self.sampling.n
        self._limited = False
        self._missing_ref = 0
        # Segment ids a random draw keeps (None = take them in file order under
        # ``limit``). Chosen up front from the full population, so the loader can
        # skip a record by its id alone.
        self._keep_ids: "set[str] | None" = None
        self.total_available: "int | None" = None
        # Per-master-shard {record_index -> is_black}, read lazily from the shard's
        # sibling frame_manifest.jsonl the first time a segment on it is viewed.
        self._black_luts: "dict[str, dict[int, bool]]" = {}
        if self.sampling.is_random:
            self._keep_ids = self._draw_segment_ids(jsonl_paths)
        total = 0
        for p in jsonl_paths:
            if self._keep_ids is None and self._limit_reached():
                self._limited = True
                break
            total += self._load_file(p)
        if not self.segments:
            raise SystemExit(
                f"No frame records found in {', '.join(str(p) for p in jsonl_paths)}"
            )
        print(
            f"loaded {total} frame records across {len(self.segments)} segments"
            + self.sampling.note(
                kept=len(self.segments), total=self.total_available,
                limited=self._limited, noun="segments",
            )
            + (f" ({self._missing_ref} without an image ref)" if self._missing_ref else ""),
            flush=True,
        )

    def _draw_segment_ids(self, jsonl_paths: list[Path]) -> set[str]:
        """The segment ids of a random draw: enumerate every segment in file order
        (one cheap pass — ids are scanned out of the raw line, no full parse), then
        keep the sampled positions."""
        ids: list[str] = []
        seen: set[str] = set()
        for p in jsonl_paths:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sid = _peek_segment_id(line)
                    if sid not in seen:
                        seen.add(sid)
                        ids.append(sid)
        self.total_available = len(ids)
        keep = self.sampling.select(len(ids))
        self._limited = len(keep) < len(ids)
        return {ids[i] for i in keep}

    def _limit_reached(self) -> bool:
        return self.limit is not None and len(self.segments) >= self.limit

    def _load_file(self, path: Path) -> int:
        n = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if self._keep_ids is not None and _peek_segment_id(line) not in self._keep_ids:
                    continue   # not in this draw — skip without parsing the record
                rec = json.loads(line)
                sid = str(rec.get("segment_id") or rec.get("clip_id") or "unknown")
                seg = self.segments.get(sid)
                if seg is None:
                    if self._keep_ids is None and self._limit_reached():
                        self._limited = True
                        break
                    seg = Segment(sid, rec.get("recording_id"))
                    self.segments[sid] = seg
                ref = _image_ref(rec)
                if ref is None:
                    self._missing_ref += 1
                t = rec.get("local_time_s")
                if not isinstance(t, (int, float)):
                    t = seg.frames[-1]["t"] if seg.frames else 0.0
                action = rec.get("action", "")
                seg.frames.append(
                    {
                        "i": len(seg.frames),
                        "bin": rec.get("local_bin_idx"),
                        "t": round(float(t), 3),
                        "src": rec.get("source_frame_idx"),
                        "action": "" if action is None else str(action),
                        "is_noop": _is_noop(action),
                        "ref": ref,  # internal; stripped before it hits the client
                    }
                )
                n += 1
        return n

    def frame_ref(self, segment_id: str, index: int) -> str | None:
        seg = self.segments.get(segment_id)
        if seg is None or index < 0 or index >= len(seg.frames):
            return None
        return seg.frames[index]["ref"]

    def _black_lut(self, shard: Path) -> dict[int, bool]:
        """``{record_index -> is_black}`` for one master shard, read once from its
        sibling ``frame_manifest.jsonl`` (same dir as the ``images.array_record``)
        and cached. A missing/unreadable manifest yields an empty map — nothing is
        flagged rather than crashing the segment view."""
        key = str(shard)
        lut = self._black_luts.get(key)
        if lut is not None:
            return lut
        lut = {}
        manifest = shard.with_name("frame_manifest.jsonl")
        try:
            with manifest.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    ri = r.get("record_index")
                    if ri is not None:
                        lut[int(ri)] = _is_black(r.get("mean_luma"), r.get("frac_dark"))
        except (OSError, json.JSONDecodeError):
            lut = {}
        self._black_luts[key] = lut
        return lut

    def _black_flag(self, ref: str | None) -> bool:
        """Whether one kept frame is (near-)black, cross-referenced from the master
        frame_manifest via the frame's ``ar://…#idx`` shard URI. Plain-file refs
        (no master manifest) and non-ar refs are treated as not-black."""
        if not ref or not is_arrayrecord_image_uri(str(ref)):
            return False
        try:
            shard, idx = parse_arrayrecord_image_uri(str(ref))
        except ValueError:
            return False
        return self._black_lut(shard).get(idx, False)

    def _ensure_black(self, seg: Segment) -> None:
        """Tag ``seg``'s frames with ``is_black`` once (idempotent)."""
        if getattr(seg, "_black_done", False):
            return
        for f in seg.frames:
            f["is_black"] = self._black_flag(f.get("ref"))
        seg._black_done = True  # type: ignore[attr-defined]

    def segment_detail(self, segment_id: str) -> dict[str, Any] | None:
        seg = self.segments.get(segment_id)
        if seg is None:
            return None
        self._ensure_black(seg)
        return seg.detail()

    def info(self) -> dict[str, Any]:
        return {
            "n_segments": len(self.segments),
            "mode": "frame_records",
            "limit": self.limit,
            "limited": self._limited,
            "sampling": self.sampling.public(),
            "total_available": self.total_available,
            "segments": [s.summary() for s in self.segments.values()],
        }


def _first_text(content: Any) -> str | None:
    """First ``{"type":"text","text":...}`` block's text in a message ``content`` list."""
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                return c["text"]
    return None


def _first_image(content: Any) -> str | None:
    """First image ref in a message ``content`` list (``image``/``image_url``/``url``)."""
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "image":
                v = c.get("image") or c.get("image_url") or c.get("url")
                if isinstance(v, str) and v:
                    return v
    return None


class ConversationsDataset(FrameRecordsDataset):
    """A stage-04 ``conversations.jsonl`` browsed as trajectories.

    Each conversation (one segment's interleaved screenshot->action chat) is one
    Segment: every ``user`` turn's screenshot is a frame, and the ``assistant``
    turn that follows carries that frame's action string. The images are ``ar://``
    refs into the SAME stage-01a master store the 01b sample used, so frame
    fetching and the black-frame cross-reference are inherited unchanged from
    ``FrameRecordsDataset``. Conversation-level context (system prompt, per-segment
    instruction, target fps, alignment) is surfaced in the segment detail.

    No per-turn timestamps exist in the chat, so each frame's time is synthesized
    from its turn index and the conversation's ``target_fps`` (t = i / fps)."""

    mode = "conversations"

    def __init__(
        self, conversations_path: Path, sampling: "Sampling | None" = None
    ) -> None:
        self.segments: "OrderedDict[str, Segment]" = OrderedDict()
        self.sampling = sampling or Sampling()
        self.limit = self.sampling.n
        self._limited = False
        self._missing_ref = 0
        self._black_luts: "dict[str, dict[int, bool]]" = {}
        # Row numbers a random draw keeps (None = first ``limit`` rows in file order).
        self._keep_rows: "set[int] | None" = None
        self.total_available: "int | None" = None
        if self.sampling.is_random:
            self._keep_rows = self._draw_rows(conversations_path)
        n = self._load_file(conversations_path)
        if not self.segments:
            raise SystemExit(f"No conversations found in {conversations_path}")
        print(
            f"loaded {n} conversations across {len(self.segments)} segments"
            + self.sampling.note(
                kept=n, total=self.total_available,
                limited=self._limited, noun="conversations",
            )
            + (f" ({self._missing_ref} turns without an image)" if self._missing_ref else ""),
            flush=True,
        )

    def _draw_rows(self, path: Path) -> set[int]:
        """The row numbers of a random draw: one conversation per line, so counting
        the non-empty lines (no JSON parse) is the whole population pass."""
        total = 0
        with path.open() as f:
            for line in f:
                if line.strip():
                    total += 1
        self.total_available = total
        keep = self.sampling.select(total)
        self._limited = len(keep) < total
        return set(keep)

    def _load_file(self, path: Path) -> int:  # type: ignore[override]
        n = 0
        row = -1
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row += 1
                if self._keep_rows is not None:
                    if row not in self._keep_rows:
                        continue   # not in this draw — skip without parsing the row
                elif self._limit_reached():
                    self._limited = True
                    break
                seg = self._parse_conversation(json.loads(line))
                if seg is not None:
                    self.segments[seg.segment_id] = seg
                    n += 1
        return n

    def _parse_conversation(self, rec: dict[str, Any]) -> "Segment | None":
        # external goal-SFT rows carry no segment_id/conversation_id — fall back to
        # sample_id (e.g. u02171b00_20260417_g003), a meaningful per-goal label.
        sid = str(rec.get("segment_id") or rec.get("conversation_id")
                  or rec.get("sample_id") or f"conv{len(self.segments)}")
        seg = Segment(sid, rec.get("recording_id"))
        # Conversation-level metadata, read back in segment_detail()/info().
        seg.conversation_id = rec.get("conversation_id")  # type: ignore[attr-defined]
        # goal-window builders (e.g. the native computer_use SFT) carry the goal
        # as ``goal_text`` instead of ``instruction``
        seg.instruction = rec.get("instruction") or rec.get("goal_text")  # type: ignore[attr-defined]
        # Self-compaction ``[CONTEXT]…[/CONTEXT]`` rolling summary, present on non-first
        # chunks of a split goal (carried as a top-level field by stage 04). Surfaced as
        # its own banner note; also fused into the first user turn by the builder.
        seg.context = rec.get("context")  # type: ignore[attr-defined]
        seg.goal_conditioned = bool(rec.get("goal_conditioned"))  # type: ignore[attr-defined]
        seg.action_format = rec.get("action_format")  # type: ignore[attr-defined]
        seg.target_fps = rec.get("target_fps")  # type: ignore[attr-defined]
        seg.alignment_status = rec.get("alignment_status")  # type: ignore[attr-defined]
        seg.split = rec.get("split")  # type: ignore[attr-defined]
        seg.system_prompt = None  # type: ignore[attr-defined]
        # Verbatim text of the FIRST user turn -- what the model actually reads before
        # the first screenshot. For a goal-conditioned segment this is the goal AND
        # (selfcompact sets) the fused ``[CONTEXT]`` block, which the top-level
        # ``instruction`` field alone doesn't show; read straight from ``messages`` so
        # it stays faithful even where stage 06 has dropped that field.
        seg.first_user_text = None  # type: ignore[attr-defined]
        fps = float(rec.get("target_fps") or 0.0)
        pending_ref: str | None = None
        for m in rec.get("messages") or []:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                seg.system_prompt = _first_text(content)  # type: ignore[attr-defined]
            elif role == "user":
                # A goal-conditioned first turn puts the instruction (+ any [CONTEXT])
                # text before the image; keep the FIRST user turn's text verbatim for
                # the banner, and take the image as this frame's ref.
                if seg.first_user_text is None:  # type: ignore[attr-defined]
                    seg.first_user_text = _first_text(content)  # type: ignore[attr-defined]
                pending_ref = _first_image(content)
                if pending_ref is None:
                    self._missing_ref += 1
            elif role == "assistant":
                action = _first_text(content) or ""
                # Native computer_use tool-call turn, else an ordered-events
                # (thinking SFT) turn -> hud (format_action translation, drives
                # HUD + metrics) and disp (compact summary); the raw text stays
                # in ``action``. None for plain-format turns.
                native = parse_native_action(action)
                if native is None:
                    native = parse_ordered_action(action)
                i = len(seg.frames)
                t = (i / fps) if fps > 0 else float(i)
                frame = {
                    "i": i,
                    "bin": i,
                    "t": round(t, 3),
                    "src": None,  # source_frame_idx isn't carried into the conversation
                    "action": action,
                    "is_noop": (native[0] == "NO_OP") if native else _is_noop(action),
                    "ref": pending_ref,  # internal; stripped before it hits the client
                }
                if native is not None:
                    frame["hud"], frame["disp"] = native
                seg.frames.append(frame)
                pending_ref = None
        return seg

    @staticmethod
    def _conv_text(seg: Segment) -> str:
        """Full searchable text of one conversation: the first user turn (goal + any
        ``[CONTEXT]``) followed by every assistant turn's action/plan string. Powers
        the left panel's 'conversation contains' filter (so search reaches the plans
        and actions, not just the goal), and mirrors the front-end's 'full
        conversation' note. One string per segment, sent with the segment list."""
        parts: list[str] = []
        fut = getattr(seg, "first_user_text", None)
        if fut:
            parts.append(str(fut))
        for f in seg.frames:
            parts.append(str(f.get("action") or ""))
            # native turns: also search the readable summary ("click",
            # "terminate(success)", …), not just the raw tool-call JSON
            if f.get("disp"):
                parts.append(str(f["disp"]))
        return "\n".join(parts)

    def segment_detail(self, segment_id: str) -> dict[str, Any] | None:
        d = super().segment_detail(segment_id)  # runs _ensure_black + Segment.detail()
        if d is None:
            return None
        seg = self.segments[segment_id]
        d.update({
            "mode": "conversations",
            "conversation_id": getattr(seg, "conversation_id", None),
            "instruction": getattr(seg, "instruction", None),
            "context": getattr(seg, "context", None),
            "first_user_text": getattr(seg, "first_user_text", None),
            "goal_conditioned": getattr(seg, "goal_conditioned", False),
            "action_format": getattr(seg, "action_format", None),
            "system_prompt": getattr(seg, "system_prompt", None),
            "target_fps": getattr(seg, "target_fps", None),
            "alignment_status": getattr(seg, "alignment_status", None),
            "split": getattr(seg, "split", None),
            "n_turns": len(seg.frames),
        })
        return d

    def info(self) -> dict[str, Any]:
        segs = list(self.segments.values())
        return {
            "n_segments": len(segs),
            "mode": "conversations",
            "limit": self.limit,
            "limited": self._limited,
            "sampling": self.sampling.public(),
            "total_available": self.total_available,
            "goal_conditioned": any(getattr(s, "goal_conditioned", False) for s in segs),
            "action_format": next(
                (getattr(s, "action_format", None) for s in segs
                 if getattr(s, "action_format", None)), None
            ),
            "has_system_prompt": any(getattr(s, "system_prompt", None) for s in segs),
            "target_fps": next(
                (getattr(s, "target_fps", None) for s in segs if getattr(s, "target_fps", None)), None
            ),
            "segments": [
                {
                    **s.summary(),
                    "instruction": getattr(s, "instruction", None),
                    "conv_text": self._conv_text(s),
                }
                for s in segs
            ],
        }


class InlineRecordsDataset(ConversationsDataset):
    """A stage-06 *inline SFT records* store — the tokenized training examples,
    browsed as trajectories.

    Payload-free records built by omegalax ``build_records_from_chat``: each
    ArrayRecord entry across the ``train/`` and ``val/`` splits IS one training
    example — a ``<= max_length`` token chunk of a stage-04 conversation with its
    ``ar://`` image refs into the SAME stage-01a master preserved (message slices,
    not pre-encoded pixels). So every record is shaped exactly like a stage-04
    conversation row, and the frame / action / HUD / system-prompt rendering is
    inherited verbatim from ``ConversationsDataset`` — only the *source* differs
    (ArrayRecord shards instead of a ``conversations.jsonl``).

    Only KEPT records are present (overflow-dropped / -truncated conversations are
    already gone), so browsing them is literally the model's training input. On top
    of the conversation view it surfaces what stage 06 adds: the ``train``/``val``
    split, the measured token length vs the ``max_length`` budget, the overflow
    mode, and the tokenizer/model the lengths were measured against.

    ``--dataset`` may point at the stage-06 root (``train/`` + ``val/`` subdirs) or
    straight at a single split dir. Each record becomes one browsable chunk, keyed
    ``<split>/<segment_id>`` (a ``#n`` suffix disambiguates a segment that ``split``
    overflow-mode broke into several chunks)."""

    mode = "inline_records"

    def __init__(self, root: Path, sampling: "Sampling | None" = None) -> None:
        self.segments: "OrderedDict[str, Segment]" = OrderedDict()
        self.sampling = sampling or Sampling()
        self.limit = self.sampling.n
        self._limited = False
        self._missing_ref = 0
        self._black_luts: "dict[str, dict[int, bool]]" = {}
        self.root = root
        self.max_length: int | None = None
        self.model_id: str | None = None
        self.overflow_mode: str | None = None
        self.total_available: "int | None" = None
        total = 0
        for split_name, split_dir, shard_name, idxs in self._plan(root):
            total += self._load_shard(split_name, split_dir, shard_name, idxs)
        if not self.segments:
            raise SystemExit(f"No inline records found under {root}")
        print(
            f"loaded {total} inline records across {len(self.segments)} chunks "
            f"(max_length={self.max_length}, model={self.model_id})"
            + self.sampling.note(
                kept=total, total=self.total_available,
                limited=self._limited, noun="records",
            )
            + (f" ({self._missing_ref} turns without an image)" if self._missing_ref else ""),
            flush=True,
        )

    @staticmethod
    def _discover_splits(root: Path) -> "list[tuple[str, Path]]":
        """The split dirs to read: ``train``/``val`` subdirs of a stage-06 root, or
        the root itself when it is already a single split dir (has ``metadata.json``)."""
        splits = [
            (name, root / name)
            for name in ("train", "val")
            if (root / name / "metadata.json").exists()
        ]
        if splits:
            return splits
        if (root / "metadata.json").exists():
            return [(root.name, root)]
        raise SystemExit(f"no train/ or val/ split (metadata.json) found under {root}")

    @staticmethod
    def _reader(shard: Path):
        from array_record.python.array_record_module import (  # noqa: PLC0415
            ArrayRecordReader,
        )

        return ArrayRecordReader(str(shard))

    def _split_shards(self, split_dir: Path) -> list[str]:
        """One split's shard names, reading its ``metadata.json`` (and picking up the
        store-level max_length / overflow mode / tokenizer on the way)."""
        meta = json.loads((split_dir / "metadata.json").read_text())
        if self.max_length is None:
            self.max_length = meta.get("max_length")
        if self.overflow_mode is None:
            self.overflow_mode = meta.get("overflow_mode")
        if self.model_id is None:
            self.model_id = (meta.get("profile_metadata") or {}).get("model_id")
        return meta.get("shard_paths") or sorted(
            p.name for p in split_dir.glob("*.array_record")
        )

    def _plan(self, root: Path) -> "list[tuple[str, Path, str, list[int]]]":
        """Which records to read, as ``(split, split_dir, shard, record_idxs)`` in
        store order.

        Shard sizes come from the ArrayRecord footers — metadata, no payload read —
        so the whole population is known before a single record is parsed: "first"
        takes the leading K, a random draw picks K positions out of every record in
        every shard of every split."""
        shards: "list[tuple[str, Path, str, int]]" = []
        for split_name, split_dir in self._discover_splits(root):
            for shard_name in self._split_shards(split_dir):
                num = self._reader(split_dir / shard_name).num_records()
                shards.append((split_name, split_dir, shard_name, num))
        total = sum(num for *_, num in shards)
        self.total_available = total
        picked = self.sampling.select(total)   # ascending global record indices
        self._limited = len(picked) < total
        plan: "list[tuple[str, Path, str, list[int]]]" = []
        base, pos = 0, 0
        for split_name, split_dir, shard_name, num in shards:
            idxs: list[int] = []
            while pos < len(picked) and picked[pos] < base + num:
                idxs.append(picked[pos] - base)
                pos += 1
            if idxs:
                plan.append((split_name, split_dir, shard_name, idxs))
            base += num
        return plan

    def _load_shard(
        self, split_name: str, split_dir: Path, shard_name: str, idxs: list[int]
    ) -> int:
        reader = self._reader(split_dir / shard_name)
        n = 0
        # Records are small (message slices with ar:// refs, no pixels), but read
        # in bounded batches so a big shard doesn't materialize all at once.
        for start in range(0, len(idxs), 512):
            batch = idxs[start:start + 512]
            for payload in reader.read(batch):
                rec = json.loads(payload)
                seg = self._parse_conversation(rec)
                seg.split = split_name  # type: ignore[attr-defined]
                seg.measured_length = rec.get("_omegalax_measured_length")  # type: ignore[attr-defined]
                seg.omega_session_id = rec.get("_omegalax_session_id")  # type: ignore[attr-defined]
                seg.max_length = self.max_length  # type: ignore[attr-defined]
                seg.shard = shard_name  # type: ignore[attr-defined]
                key = self._unique_key(f"{split_name}/{seg.segment_id}")
                seg.segment_id = key
                self.segments[key] = seg
                n += 1
        return n

    def _unique_key(self, base: str) -> str:
        key, k = base, 2
        while key in self.segments:
            key, k = f"{base}#{k}", k + 1
        return key

    def segment_detail(self, segment_id: str) -> dict[str, Any] | None:
        d = super().segment_detail(segment_id)  # Conversations detail + _ensure_black
        if d is None:
            return None
        seg = self.segments[segment_id]
        d.update({
            "mode": "inline_records",
            "split": getattr(seg, "split", None),
            "measured_length": getattr(seg, "measured_length", None),
            "max_length": self.max_length,
            "overflow_mode": self.overflow_mode,
            "model_id": self.model_id,
            "omega_session_id": getattr(seg, "omega_session_id", None),
        })
        return d

    def info(self) -> dict[str, Any]:
        d = super().info()  # conversation-level goal/system/fps rollup
        segs = list(self.segments.values())
        d.update({
            "mode": "inline_records",
            "limit": self.limit,
            "limited": self._limited,
            "max_length": self.max_length,
            "model_id": self.model_id,
            "overflow_mode": self.overflow_mode,
            "splits": sorted(
                {getattr(s, "split", None) for s in segs if getattr(s, "split", None)}
            ),
            "segments": [
                {
                    **s.summary(),
                    "instruction": getattr(s, "instruction", None),
                    "conv_text": self._conv_text(s),
                    "split": getattr(s, "split", None),
                    "measured_length": getattr(s, "measured_length", None),
                }
                for s in segs
            ],
        })
        return d


class FramesMasterDataset:
    """A stage-01a *frames-master* store browsed directly — raw frames, no actions.

    The frames-master is keylog-free by design: it holds decoded JPEG frames at
    ``master_fps`` and nothing about the keylog (actions are added later by the
    01b sampler). So this view shows frames + timing only; the action HUD stays
    empty. Segments are listed from ``segment_index.jsonl`` up front (cheap), and
    each segment's per-frame rows are read lazily from its ``frame_manifest.jsonl``
    on first access — the store can be hundreds of thousands of frames, so nothing
    is loaded eagerly. A small LRU keeps the last few browsed segments in memory.
    """

    mode = "frames_master"

    def __init__(self, root: Path, sampling: "Sampling | None" = None) -> None:
        self.root = root
        self.sampling = sampling or Sampling()
        self.limit = self.sampling.n
        self._limited = False
        self.total_available: "int | None" = None
        self.master_fps: float | None = None
        self.segments: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        index_path = root / "segment_index.jsonl"
        # The index is one row per segment and the per-frame rows are read lazily,
        # so a random draw can read the whole index (cheap) and keep the sampled
        # rows; "first" still stops at row K.
        rows: "list[dict[str, Any]]" = []
        with index_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not row.get("num_records"):
                    continue  # skip empty/failed segments (no usable frame store)
                if not self.sampling.is_random and self.limit is not None \
                        and len(rows) >= self.limit:
                    self._limited = True
                    break
                rows.append(row)
        if self.sampling.is_random:
            self.total_available = len(rows)
            keep = self.sampling.select(len(rows))
            self._limited = len(keep) < len(rows)
            rows = [rows[i] for i in keep]
        for row in rows:
            sid = str(row.get("segment_id"))
            self.master_fps = row.get("master_fps", self.master_fps)
            self.segments[sid] = {
                "recording_id": row.get("recording_id"),
                "num_records": int(row["num_records"]),
                "duration_s": float(row.get("video_duration_s") or 0.0),
                "manifest": row.get("frame_manifest"),
            }
        if not self.segments:
            raise SystemExit(f"no usable segments in {index_path}")
        # Optional keylog overlay: raw events placed by timestamp onto the master
        # frames. Discovered from this master's manifest ("source_clips_manifest",
        # a stage-00 clips_manifest mapping segment_id -> keylog_path), falling back
        # to --clips-manifest. The frames-master store itself is keylog-free.
        self.keylogs: dict[str, str] = {}
        cm = None
        manifest_json = root / "manifest.json"
        if manifest_json.exists():
            try:
                cm = json.loads(manifest_json.read_text()).get("source_clips_manifest")
            except (OSError, json.JSONDecodeError):
                cm = None
        if cm is None:
            cm = CLIPS_MANIFEST_OVERRIDE
        if cm and Path(cm).exists():
            with Path(cm).open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    sid = str(row.get("segment_id") or "")
                    kp = row.get("keylog_path")
                    if sid and kp and row.get("keylog_exists", True):
                        self.keylogs[sid] = kp
            print(f"  keylog overlay: {len(self.keylogs)} segments (from {cm})", flush=True)
        # Optional stage-00 alignment overlay (raw<->video time map): enables the
        # dual-clock event table + the "aligned + trims" timeline. Keyed by segment_id.
        self.align: dict[str, dict[str, Any]] = {}
        align_path = _resolve_alignment_path(cm)
        if align_path is not None:
            with align_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    sid = str(row.get("segment_id") or "")
                    if sid:
                        self.align[sid] = {
                            "status": row.get("status"),
                            "total_collapse_s": float(row.get("total_collapse_s") or 0.0),
                            "residual_s": float(row.get("residual_s") or 0.0),
                            "overhang_s": float(row.get("overhang_s") or 0.0),
                            "video_dur_s": float(row.get("video_dur_s") or 0.0),
                            "splices": row.get("splices") or [],
                        }
            print(f"  alignment overlay: {len(self.align)} segments (from {align_path})", flush=True)
        self._cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._cache_cap = 6
        print(
            f"frames-master: {len(self.segments)} segments, master_fps={self.master_fps}"
            + self.sampling.note(
                kept=len(self.segments), total=self.total_available,
                limited=self._limited, noun="segments",
            ),
            flush=True,
        )

    def _load(self, segment_id: str) -> dict[str, Any] | None:
        """Frames (+ optional keylog-derived per-frame actions and a raw event
        list), read and cached on first access."""
        meta = self.segments.get(segment_id)
        if meta is None:
            return None
        cached = self._cache.get(segment_id)
        if cached is not None:
            self._cache.move_to_end(segment_id)
            return cached
        manifest = Path(meta["manifest"]) if meta.get("manifest") else (
            self.root / "frames" / segment_id / "frame_manifest.jsonl"
        )
        frames: list[dict[str, Any]] = []
        with manifest.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                frames.append({
                    "i": len(frames),
                    "bin": r.get("record_index", len(frames)),
                    "t": round(float(r.get("source_time_s") or 0.0), 3),
                    "src": r.get("source_frame_idx"),
                    "action": "",          # filled from the keylog overlay if present
                    "is_noop": True,
                    # (near-)black per stage-01a luma metrics + default thresholds;
                    # what the 01b sampler drops with --drop-black-frames on.
                    "is_black": _is_black(r.get("mean_luma"), r.get("frac_dark")),
                    "ref": r.get("image") or r.get("image_path"),
                })
        events: list[list[Any]] = []
        align_info: dict[str, Any] | None = None
        keylog = self.keylogs.get(segment_id)
        if keylog and frames:
            fps = float(meta.get("master_fps") or self.master_fps or 1.0)
            try:
                # Per-frame actions: bin the raw keylog at master_fps (bin i == the
                # master frame at [i/fps, (i+1)/fps)) so the HUD lights up. Raw,
                # un-realigned — the frames are on the raw video clock too.
                bins, _ = aggregate_actions(Path(keylog), len(frames), fps)
                for i, b in enumerate(bins):
                    action = format_action(b)
                    frames[i]["action"] = action
                    frames[i]["is_noop"] = action == "NO_OP"
                al = self.align.get(segment_id)
                if al is not None:
                    # Dual-clock events + the "aligned + trims" overlay. video_end is
                    # the end of the last master frame window (≈ video duration).
                    splices = al["splices"]
                    video_end = len(frames) / fps
                    events = _raw_events(Path(keylog), splices=splices, video_end=video_end)
                    # Per-frame actions on the ALIGNED clock (the raw/aligned HUD
                    # toggle), binned with the same keylog_to_video map used above.
                    abins, _ = aggregate_actions(
                        Path(keylog), len(frames), fps,
                        timemap=lambda t: R.keylog_to_video(t, splices),
                    )
                    active: list[int] = []
                    for i, b in enumerate(abins):
                        a = format_action(b)
                        frames[i]["action_aln"] = a
                        if a != "NO_OP":
                            active.append(i)
                    # Frames "cut" by each collapse: the raw-keylog span [kp, kp+collapse]
                    # (in frame units) that realignment folds to the single frame at vp.
                    n = len(frames)
                    cut_ranges: list[list[int]] = []
                    for sp in splices:
                        lo = max(0, int(sp["kp"] * fps))
                        hi = min(n, int((sp["kp"] + sp["collapse"]) * fps) + 1)
                        if hi > lo:
                            cut_ranges.append([lo, hi])
                    align_info = {
                        "status": al["status"],
                        "total_collapse_s": round(al["total_collapse_s"], 2),
                        "residual_s": round(al["residual_s"], 2),
                        "overhang_s": round(al["overhang_s"], 2),
                        "video_dur_s": al["video_dur_s"],
                        "aligned_active": active,
                        "collapse_frames": [
                            {"frame": int(sp["vp"] * fps),
                             "collapse": round(sp["collapse"], 2),
                             "kp": round(sp["kp"], 2)}
                            for sp in splices
                        ],
                        "cut_ranges": cut_ranges,
                        "n_overhang_events": sum(1 for e in events if e[1] >= video_end),
                    }
                else:
                    events = _raw_events(Path(keylog))
            except Exception as exc:  # noqa: BLE001 — keep frames if keylog unreadable
                print(f"  keylog read failed for {segment_id}: {exc}", flush=True)
                keylog = None
        entry = {"frames": frames, "events": events, "has_keylog": bool(keylog),
                 "align": align_info}
        self._cache[segment_id] = entry
        self._cache.move_to_end(segment_id)
        while len(self._cache) > self._cache_cap:
            self._cache.popitem(last=False)
        return entry

    def _frames(self, segment_id: str) -> list[dict[str, Any]] | None:
        entry = self._load(segment_id)
        return entry["frames"] if entry else None

    def segment_detail(self, segment_id: str) -> dict[str, Any] | None:
        entry = self._load(segment_id)
        if entry is None:
            return None
        frames = entry["frames"]
        meta = self.segments[segment_id]
        return {
            "segment_id": segment_id,
            "recording_id": meta.get("recording_id"),
            "n_frames": len(frames),
            "n_non_noop": sum(1 for f in frames if not f["is_noop"]),
            "n_black": sum(1 for f in frames if f.get("is_black")),
            "n_black_act": sum(1 for f in frames if f.get("is_black") and not f["is_noop"]),
            "has_actions": entry["has_keylog"] and any(not f["is_noop"] for f in frames),
            "master_fps": self.master_fps,
            "has_alignment": entry.get("align") is not None,
            "dual_clock": entry.get("align") is not None,
            "align": entry.get("align"),
            "events": entry["events"],
            "frames": [{k: v for k, v in fr.items() if k != "ref"} for fr in frames],
        }

    def frame_ref(self, segment_id: str, index: int) -> str | None:
        frames = self._frames(segment_id)
        if frames is None or index < 0 or index >= len(frames):
            return None
        return frames[index]["ref"]

    def info(self) -> dict[str, Any]:
        return {
            "n_segments": len(self.segments),
            "mode": self.mode,
            "master_fps": self.master_fps,
            "limit": self.limit,
            "limited": self._limited,
            "sampling": self.sampling.public(),
            "total_available": self.total_available,
            "has_keylog": bool(self.keylogs),
            "has_alignment": bool(self.align),
            "segments": [
                {
                    "segment_id": sid,
                    "recording_id": m.get("recording_id"),
                    "n_frames": m.get("num_records"),
                    "n_non_noop": 0,
                    "duration_s": round(m.get("duration_s", 0.0), 2),
                    "align_status": (self.align.get(sid) or {}).get("status"),
                }
                for sid, m in self.segments.items()
            ],
        }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a: Any) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _dsname(self, q: dict[str, list[str]]) -> str:
        """The requested dataset name, defaulting to the first registered one."""
        vals = q.get("ds")
        if vals and vals[0]:
            return vals[0]
        return next(iter(DATASETS), "")

    @staticmethod
    def _sampling(q: dict[str, list[str]]) -> Sampling:
        """The sampling the client asked for: ``sm`` (first|random), ``n`` samples
        (``n=0`` = every sample, i.e. the UI's blank N), ``seed``. Every route
        carries it, so one dataset can be browsed under several samplings at once;
        anything missing or unparseable falls back to the CLI defaults rather than
        failing the request."""
        def _int(key: str, default: "int | None") -> "int | None":
            raw = (q.get(key) or [""])[0].strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        mode = (q.get("sm") or [DATASET_SAMPLE_MODE])[0]
        if mode not in ("first", "random"):
            mode = DATASET_SAMPLE_MODE
        n = _int("n", DATASET_SAMPLE_LIMIT)
        if n is not None and n <= 0:
            n = None   # explicit "no cap" — overrides --limit for this request
        return Sampling(mode, n, _int("seed", DATASET_SAMPLE_SEED) or 0)

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
                    "datasets": [{"name": n, "mode": DATASETS[n]["mode"]} for n in names],
                    "default": names[0] if names else None,
                    # initial state of the sampling controls (--limit/--sample-mode/--seed)
                    "sampling": Sampling(
                        DATASET_SAMPLE_MODE, DATASET_SAMPLE_LIMIT, DATASET_SAMPLE_SEED
                    ).public(),
                })
            elif route == "/api/segments":
                name = self._dsname(q)
                if name not in DATASETS:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                else:
                    try:
                        self._send_json(get_dataset(name, self._sampling(q)).info())
                    except Exception as exc:  # noqa: BLE001 — report, keep UI alive
                        self._send_json({"error": f"failed to load {name!r}: {exc}"}, 500)
            elif route == "/api/segment":
                name = self._dsname(q)
                sid = (q.get("id") or [""])[0]
                ds = get_dataset(name, self._sampling(q)) if name in DATASETS else None
                detail = ds.segment_detail(sid) if ds is not None else None
                if detail is None:
                    self._send_json({"error": f"unknown segment {sid!r}"}, 404)
                else:
                    self._send_json(detail)
            elif route == "/api/marks":
                name = self._dsname(q)
                mid = (q.get("mid") or [""])[0]
                if name not in DATASETS:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                else:
                    marks = _load_marks(name, mid)
                    self._send_json({
                        "marks": marks, "n_marked": len(marks), "path": str(_marks_path(name, mid)),
                    })
            elif route == "/api/find":
                name = self._dsname(q)
                if name not in DATASETS:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                else:
                    try:
                        self._send_json(find_actions(
                            get_dataset(name, self._sampling(q)),
                            (q.get("q") or [""])[0],
                        ))
                    except Exception as exc:  # noqa: BLE001 — report, keep UI alive
                        self._send_json({"error": f"search failed: {exc}"}, 500)
            elif route == "/frame":
                self._serve_frame(q)
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 — surface as 500, keep server up
            self._send(500, f"{type(exc).__name__}: {exc}".encode(), "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON body"}, 400)
                return
            if route == "/api/mark":
                self._set_mark(body)
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 — surface as 500, keep server up
            self._send(500, f"{type(exc).__name__}: {exc}".encode(), "text/plain")

    def _set_mark(self, body: dict[str, Any]) -> None:
        """Toggle one segment's golden-trace mark on/off and persist immediately —
        marking happens one segment at a time while browsing, so there is no
        "unsaved" state to lose track of. Re-reads the file fresh (see
        ``_load_marks``) so a second server process / browser tab open on the same
        dataset+marks_id doesn't clobber marks made from here, or vice versa."""
        name = str(body.get("ds") or "")
        sid = str(body.get("id") or "")
        mid = str(body.get("mid") or "")
        if name not in DATASETS or not sid:
            self._send_json({"error": f"unknown dataset/segment: {name!r} / {sid!r}"}, 404)
            return
        marks = _load_marks(name, mid)
        marked = bool(body.get("marked"))
        if marked:
            marks[sid] = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        else:
            marks.pop(sid, None)
        _save_marks(name, marks, mid)
        self._send_json({
            "ok": True, "id": sid, "marked": marked,
            "n_marked": len(marks), "path": str(_marks_path(name, mid)),
        })

    def _serve_frame(self, q: dict[str, list[str]]) -> None:
        name = self._dsname(q)
        if name not in DATASETS:
            self._send(404, b"unknown dataset", "text/plain")
            return
        sid = (q.get("seg") or [""])[0]
        try:
            idx = int((q.get("i") or ["-1"])[0])
        except ValueError:
            idx = -1
        ds = get_dataset(name, self._sampling(q))
        ref = ds.frame_ref(sid, idx) if ds is not None else None
        if ref is None:
            self._send(404, b"no such frame", "text/plain")
            return
        try:
            jpeg = read_jpeg_bytes(ref)
        except Exception as exc:  # noqa: BLE001
            self._send(502, f"frame read failed: {exc}".encode(), "text/plain")
            return
        self._send(200, jpeg, "image/jpeg", cache=True)


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>frame-records viewer — stage 01b</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.45 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
         background:#14161a; color:#d7dae0; height:100vh; display:flex; flex-direction:column; }
  header { padding:7px 12px; border-bottom:1px solid #2a2e36; display:flex; gap:10px;
           align-items:center; flex-wrap:wrap; background:#191c21; }
  select,button { background:#22262e; color:#d7dae0; border:1px solid #343a44;
                  border-radius:4px; padding:3px 8px; font:inherit; cursor:pointer; }
  button:hover { border-color:#5b9dd9; }
  button.on { background:#2d4a75; border-color:#5b9dd9; }
  button#markgood.on { background:#5a4a1a; border-color:#d9b95b; color:#ffd580; }
  /* sampling controls (first N / random N + seed) */
  header input { background:#22262e; color:#d7dae0; border:1px solid #343a44;
                 border-radius:4px; padding:2px 6px; font:inherit; }
  header input:focus { outline:none; border-color:#5b9dd9; }
  .snum { width:76px; }
  #sampinfo.busy { color:#e8c877; }
  .hint { margin-left:auto; color:#6b7280; font-size:12px; }
  kbd { background:#22262e; border:1px solid #343a44; border-radius:3px; padding:0 4px; }
  main { flex:1; display:flex; min-height:0; }
  #resizer { width:6px; flex:none; cursor:col-resize; background:#20242b;
             border-left:1px solid #2a2e36; border-right:1px solid #2a2e36; }
  #resizer:hover, #resizer.drag { background:#5b9dd9; }
  #screen { flex:1; min-width:0; padding:10px; display:flex; flex-direction:column; gap:6px; overflow:hidden; }
  #frameimg { flex:1 1 auto; min-height:0; width:100%; object-fit:contain;
              background:#000; border-radius:4px; }
  #status { display:flex; gap:14px; color:#aeb6c2; flex-wrap:wrap; align-items:center; }
  #status b { color:#fff; }
  /* the verbatim action line — wraps rather than eliding, so a long primitive
     sequence is readable in full instead of ending in "…" */
  #rawaction { color:#8b93a1; font-size:11px; max-width:100%; white-space:pre-wrap;
               word-break:break-word; }
  .badge { display:inline-block; padding:0 6px; border-radius:3px; font-size:11px; }
  .badge.noop { background:#26292f; color:#8b93a1; }
  .badge.act  { background:#1e3a2a; color:#7fd6a2; }
  .badge.black { background:#2b2440; color:#c3b3f5; }

  /* HUD: mouse radar + keyboard (top of the right sidebar) */
  #hud { display:flex; flex-wrap:wrap; gap:12px 14px; align-items:flex-start; justify-content:center;
         flex:none; padding:8px 10px; border-bottom:1px solid #2a2e36; }
  #mousebox { width:150px; flex:none; display:flex; flex-direction:column; align-items:center; gap:3px; }
  #radar { width:150px; height:150px; }
  #radar .ring { fill:#181b20; stroke:#343a44; stroke-width:1.5; }
  #radar .hub  { fill:#5b6270; }
  #radar .mvline { stroke:#5b9dd9; stroke-width:3; stroke-linecap:round; }
  #radar .mvhead { fill:#8fc4f2; }
  #btns { display:flex; gap:6px; }
  .btn { width:24px; height:20px; border-radius:4px; background:#22262e; border:1px solid #343a44;
         display:flex; align-items:center; justify-content:center; font-size:11px; color:#8b93a1; }
  .btn.press { background:#2d6a45; border-color:#7fd6a2; color:#eafff2; }
  .btn.held  { background:#274536; border-color:#4f8f6b; color:#bfe8cf; }
  #scrollind { height:14px; color:#c9a227; font-size:11px; }
  #dxy { color:#8b93a1; font-size:11px; }

  #kbwrap { flex:1 1 360px; min-width:0; overflow-x:auto; }
  #kbd { display:inline-flex; flex-direction:column; gap:3px; }
  .krow { display:flex; gap:3px; }
  .krow.arrows { justify-content:flex-start; }
  .key { height:26px; border-radius:4px; background:#20242b; border:1px solid #30353f;
         display:flex; align-items:center; justify-content:center; font-size:11px; color:#9aa2af; flex:none; }
  .key.press { background:#2d6a45; border-color:#7fd6a2; color:#eafff2; box-shadow:0 0 7px rgba(127,214,162,.5); }
  .key.held  { background:#274536; border-color:#4f8f6b; color:#cdeeda; }
  #otherkeys { margin-top:4px; display:flex; gap:4px; flex-wrap:wrap; }
  .chip { padding:1px 6px; border-radius:3px; font-size:11px; background:#274536; color:#cdeeda; }
  .chip.press { background:#2d6a45; color:#eafff2; }
  .kbcap { color:#6b7280; font-size:10px; margin-top:3px; }

  #strip { display:flex; gap:2px; overflow-x:auto; padding:4px 0 1px; flex:none; }
  .cell { flex:none; width:9px; height:24px; border-radius:2px; background:#2a2e36; cursor:pointer; }
  .cell.act { background:#3f7d5b; }
  .cell.cut { background:#6e4a1c; box-shadow:inset 0 0 0 1px rgba(233,200,119,.55); }
  /* black-frame flag wins the fill (defined after act/cut) — the action string stays in the tooltip. */
  .cell.black { background:#3b2f5e; box-shadow:inset 0 0 0 1px rgba(167,139,250,.7); }
  /* black frame that ALSO has an action: keep the violet fill, swap the border to a green glow. */
  .cell.black.act { box-shadow:inset 0 0 0 1.5px rgba(127,214,162,.95), 0 0 5px rgba(127,214,162,.5); }
  .cell.stuck { box-shadow:inset 0 0 0 2px #f59e0b, 0 0 5px rgba(245,158,11,.45); }
  .cell.match { background:#c08a2a; box-shadow:0 0 5px rgba(245,181,68,.55); }
  .cell.cur { outline:2px solid #5b9dd9; outline-offset:1px; }
  #strip2wrap { flex:none; }
  #striplabels { display:flex; justify-content:space-between; align-items:baseline; }
  .striplab { color:#6b7280; font-size:10px; }
  #alignstat { color:#c9a227; font-size:11px; }
  #strip2 { display:flex; gap:2px; overflow-x:auto; padding:1px 0 4px; flex:none; }
  #strip2 .cell { height:16px; }
  .cell.col { background:#7a2633; box-shadow:0 0 5px rgba(255,120,140,.55); }

  /* left filter sidebar — hidden until toggled */
  #filters { width:262px; flex:none; display:none; flex-direction:column; gap:8px;
             min-height:0; overflow-y:auto; padding:9px 11px;
             background:#191c21; border-right:1px solid #2a2e36; }
  #filters.show { display:flex; }
  #filters .fhead { color:#8b93a1; font-size:11px; text-transform:uppercase; letter-spacing:.06em;
                    display:flex; align-items:center; gap:8px; margin-top:4px; }
  #filters .fhead:first-child { margin-top:0; }
  #filters .fhead button { margin-left:auto; padding:1px 7px; font-size:11px; text-transform:none; letter-spacing:0; }
  #fcount { color:#7fd6a2; font-size:12px; }
  #filters input, #filters select { background:#22262e; color:#d7dae0; border:1px solid #343a44;
                                     border-radius:4px; padding:2px 6px; font:inherit; min-width:0; }
  #filters input:focus, #filters select:focus { outline:none; border-color:#5b9dd9; }
  .frow2 { display:flex; flex-direction:column; gap:3px; color:#aeb6c2; font-size:11px; }
  .frow2 input { width:100%; }
  .frow { display:grid; grid-template-columns:1fr 58px 58px; gap:5px; align-items:center; }
  .flab { color:#aeb6c2; font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .fnum { width:100%; }
  #fsort { display:flex; gap:5px; }
  #fsort select { flex:1; }
  .fnote { color:#6b7280; font-size:11px; }

  #panel { width:430px; flex:none; display:flex; flex-direction:column; min-height:0; overflow:hidden; }
  #modenote { display:none; padding:6px 10px; background:#3a2f14; color:#e8c877;
              font-size:11px; border-bottom:1px solid #2a2e36; }
  #modenote.show { display:block; }
  #modenote .cnote-goal { color:#7fd6a2; font-weight:600; }
  #modenote .cnote-context { color:#c9a0e8; font-weight:600; }
  #modenote .cnote-sys { color:#b9a06a; }
  #modenote details { margin-top:4px; }
  #modenote summary { cursor:pointer; user-select:none; }
  #modenote .cnote-fixed { margin-top:4px; }
  #modenote .cnote-text { margin-top:3px; white-space:pre-wrap; color:#e4d7a6; }
  #chatbtn { margin-top:6px; display:inline-block; cursor:pointer; background:#2a3550;
             color:#cfe0ff; border:1px solid #3a4a6a; border-radius:4px; padding:3px 9px;
             font-size:11px; font-weight:600; }
  #chatbtn:hover { background:#33436a; }
  /* Dedicated, collapsible full-chat window (overlay docked to the right edge). */
  #chatwin { position:fixed; top:0; right:0; height:100vh; width:min(680px,62vw);
             background:#12151b; border-left:1px solid #2a2e36; box-shadow:-8px 0 24px rgba(0,0,0,.45);
             z-index:50; display:none; flex-direction:column; }
  #chatwin.show { display:flex; }
  #chatwin-head { flex:none; display:flex; align-items:center; gap:10px; padding:10px 14px;
                  border-bottom:1px solid #2a2e36; background:#171b22; }
  #chatwin-head .ttl { font-weight:600; color:#e8eef7; font-size:13px; }
  #chatwin-head .sub { color:#8b93a1; font-size:11px; }
  #chatclose { cursor:pointer; background:#2a2e36; color:#cbd3df; border:none; border-radius:4px;
               padding:4px 10px; font-size:12px; }
  #chatclose:hover { background:#39404c; }
  #chatbody { flex:1; overflow:auto; padding:14px 16px; }
  .turn { margin-bottom:9px; border-radius:8px; padding:8px 11px; }
  .turn .role { font-size:10px; text-transform:uppercase; letter-spacing:.08em; font-weight:700; margin-bottom:4px; }
  .turn.sys  { background:#1c1a12; }         .turn.sys  .role { color:#b9a06a; }
  .turn.user { background:#141c26; }         .turn.user .role { color:#8fc8e8; }
  .turn.asst { background:#131f18; }         .turn.asst .role { color:#7fd6a2; }
  .turn .body { white-space:pre-wrap; word-break:break-word; color:#d7dde6; font-size:12.5px; line-height:1.45; }
  .turn .act  { font-family:ui-monospace,Menlo,Consolas,monospace; color:#e6d39a; }
  .turn .shot { color:#7f8794; font-style:italic; font-size:11.5px; }
  .turn .goal { color:#9be3b6; font-weight:600; margin-bottom:6px; }
  .turn .ctx  { margin-top:6px; padding:7px 9px; background:#0e1520; border:1px solid #24314a;
                border-radius:6px; color:#c6b8e6; font-size:11.5px; white-space:pre-wrap; word-break:break-word; }
  #chatbody details.syswrap > summary { cursor:pointer; color:#b9a06a; font-size:11px; }
  .phead { padding:6px 12px; border-bottom:1px solid #2a2e36; border-top:1px solid #2a2e36;
           color:#8b93a1; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  #typed { flex:1 1 40%; min-height:70px; overflow-y:auto; padding:10px 12px; white-space:pre-wrap; word-break:break-word;
           font-size:14px; line-height:1.5; color:#c4b58a; }
  #typed .now { background:#2d6a45; color:#eafff2; border-radius:2px; }
  #typed .empty { color:#5b6270; }
  #typed .caret { border-left:2px solid #5b9dd9; margin-left:1px; animation:blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  #rows { overflow-y:auto; flex:1 1 45%; min-height:80px; }
  .row { display:grid; grid-template-columns:46px 54px 34px 1fr; gap:8px; padding:2px 12px; cursor:pointer; border-left:2px solid transparent; }
  .row .n { color:#7fa8d6; text-align:right; }
  .row .n.hi { color:#f5b544; }
  .row:hover { background:#1c2027; }
  .row.cur { background:#232a35; border-left-color:#5b9dd9; }
  .row.noop { color:#6b7280; }
  .row .t { color:#8b93a1; }
  .row .a { white-space:pre-wrap; word-break:break-word; }
  .row .a mark { background:#5a4a1a; color:#ffd580; border-radius:2px; padding:0 1px; }
  .row.match { background:#1e2530; }
  .row.match.cur { background:#2a3444; }
  .erow { display:grid; grid-template-columns:58px 92px 1fr; gap:8px; padding:2px 12px; cursor:pointer; border-left:2px solid transparent; }
  .erow:hover { background:#1c2027; }
  .erow.now { background:#243a2c; border-left-color:#7fd6a2; }
  .erow .t { color:#8b93a1; }
  .erow .ty { color:#8fb0d0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .erow .a { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .erow.dual { grid-template-columns:48px 48px 78px 1fr; }
  .erow .raw { color:#8b93a1; }
  .erow .aln { color:#c4b58a; }
  .erow.trim { color:#ff9db0; }
  .erow.trim .a { text-decoration:line-through; }
  .seginfo { color:#8b93a1; }
</style></head><body>
<header>
  <button id="filttoggle" title="show/hide filters (f)">⚙ filters</button>
  <label>dataset <select id="ds"></select></label>
  <label title="which N samples of the dataset to load: the first N in store order, or N drawn at random (deterministic for a given seed)">samples
    <select id="smode"><option value="first">first N</option><option value="random">random N</option></select></label>
  <label title="how many samples to load (blank = all)">N <input id="sn" class="snum" type="number" min="1" step="1" placeholder="all"></label>
  <label id="sseedwrap" title="the same seed + N always draws the same samples from this dataset">seed <input id="sseed" class="snum" type="number" step="1" value="0"></label>
  <span id="sampinfo" class="seginfo"></span>
  <label>segment <select id="seg"></select></label>
  <button id="prev" title="←">◀</button>
  <button id="play">▶ play</button>
  <button id="next" title="→">▶</button>
  <button id="clocktoggle" title="raw vs realigned keylog for the HUD" style="display:none">keys: raw</button>
  <button id="prevmatch" title="previous matching turn (N)" style="display:none">◂ match</button>
  <button id="nextmatch" title="next matching turn (n)" style="display:none">match ▸</button>
  <span id="matchinfo" class="seginfo"></span>
  <span id="seginfo" class="seginfo"></span>
  <button id="markgood" title="mark this segment a golden/good trace (m)">☆ mark good</button>
  <label title="optional: give this review pass its own marks file (golden_marks_&lt;id&gt;.json) instead of the dataset's shared golden_marks.json — so independent sessions never overwrite each other's marks. Remembered per-browser; blank = shared file.">marks id <input id="marksid" type="text" placeholder="(shared)" size="8"></label>
  <span id="markinfo" class="seginfo"></span>
  <span class="hint"><kbd>←</kbd>/<kbd>→</kbd>/<kbd>a</kbd>/<kbd>d</kbd> step · <kbd>↑</kbd>/<kbd>↓</kbd>/<kbd>w</kbd>/<kbd>s</kbd> prev/next segment · <kbd>space</kbd> play · <kbd>n</kbd>/<kbd>N</kbd> prev/next action match · <kbd>,</kbd>/<kbd>.</kbd> prev/next black · <kbd>⇧,</kbd>/<kbd>⇧.</kbd> black w/ action · <kbd>m</kbd> mark good · <kbd>c</kbd> chat</span>
</header>
<main>
  <aside id="filters">
    <div class="fhead">filters <button id="fclear" title="clear all filters">reset</button></div>
    <div id="fcount"></div>
    <div id="ftext"></div>
    <div class="fhead">ranges (min / max)</div>
    <div id="fnums"></div>
    <div class="fhead">sort</div>
    <div id="fsort"></div>
  </aside>
  <div id="screen">
    <img id="frameimg" alt="frame">
    <div id="status">
      <span>frame <b id="fi">–</b>/<b id="fn">–</b></span>
      <span>t=<b id="ft">–</b>s</span>
      <span>bin <b id="fbin">–</b></span>
      <span>src <b id="fsrc">–</b></span>
      <span id="fbadge"></span>
      <span id="rawaction"></span>
    </div>
    <div id="strip"></div>
    <div id="strip2wrap" style="display:none">
      <div id="striplabels">
        <span class="striplab">aligned keys + trims (■ = collapsed idle)</span>
        <span id="alignstat"></span>
      </div>
      <div id="strip2"></div>
    </div>
  </div>
  <div id="resizer" title="drag to resize the sidebar"></div>
  <div id="panel">
    <div id="modenote"></div>
    <div id="hud">
      <div id="mousebox">
        <svg id="radar" viewBox="0 0 100 100">
          <circle class="ring" cx="50" cy="50" r="45"/>
          <line id="mvline" class="mvline" x1="50" y1="50" x2="50" y2="50"/>
          <polygon id="mvhead" class="mvhead" points="0,0 -8,-4.5 -8,4.5"/>
          <circle class="hub" cx="50" cy="50" r="3"/>
        </svg>
        <div id="btns">
          <span class="btn" data-b="LMB">L</span>
          <span class="btn" data-b="MMB">M</span>
          <span class="btn" data-b="RMB">R</span>
        </div>
        <div id="scrollind"></div>
        <div id="dxy">Δ 0, 0</div>
      </div>
      <div id="kbwrap">
        <div id="kbd"></div>
        <div id="otherkeys"></div>
        <div class="kbcap">bright = pressed this bin · dim = held from earlier</div>
      </div>
    </div>
    <div class="phead">typed text <span id="typedcap" style="text-transform:none;letter-spacing:0"></span></div>
    <div id="typed"></div>
    <div class="phead" id="rowshead">actions</div>
    <div id="rows"></div>
  </div>
</main>
<div id="chatwin">
  <div id="chatwin-head">
    <span class="ttl">full chat</span>
    <span class="sub" id="chatsub"></span>
    <span style="flex:1"></span>
    <button id="chatclose" title="collapse (Esc)">✕ collapse</button>
  </div>
  <div id="chatbody"></div>
</div>
<script>
const $ = s => document.querySelector(s);
let DS=null, SEG=null, FR=[], PARSED=[], cur=0, playing=false, timer=null;
let MODE='frame_records'; const EMPTYSET=new Set(); let _lastCell=null, _lastRow=null, _lastCell2=null;
let MASTER_FPS=15, EV=null, EVT=[], HAS_ACTIONS=false, _evLo=-1, _evHi=-1;
let ALN=null, DUAL=false, ALNACT=null;   // alignment overlay (master + --alignment)
let CLOCK='raw';                         // which keylog drives the HUD: 'raw' | 'aligned'
let ALL_SEGMENTS=[];                      // full segment list for DS (pre-filter)
let MARKS={}, MARKS_PATH='';              // golden-trace marks for DS: segment_id -> {ts}
// Which N samples of the dataset the server loads: the first N in store order, or
// N drawn at random. A random draw is deterministic in (seed, N, dataset) — the
// same seed always yields the same N samples — and it is the *membership* that is
// random, not the order (the list stays in store order). Every request carries the
// sampling, because it decides WHICH build of the dataset answers it.
let SAMP={mode:'first', n:null, seed:0};
let SAMP_DEFKEY='';                       // the launch defaults this session's override belongs to
// N is always sent — 0 means "no cap, every sample", which is what a blank N box
// asks for and must override the server's --limit.
function sampQS(){
  const s='&sm='+SAMP.mode+'&n='+(SAMP.n||0);
  return SAMP.mode==='random' ? s+'&seed='+SAMP.seed : s;
}
let activeNumMetrics=[];                  // numeric filter rows rendered for this dataset
// Numeric filter metrics (min/max). Only those present in the loaded dataset's
// segment summaries are rendered, so the panel adapts to master/sample/conv/records.
const NUM_METRICS=[
  {key:'duration_s',  label:'duration (s)'},
  {key:'n_frames',    label:'frames / turns'},
  {key:'n_non_noop',  label:'action frames'},
  // Per-TURN action counts (ordered_events: one per primitive; computer_use: one
  // per tool call; plain: movement + scroll + each key event). max finds the
  // segments containing a dense turn, mean the uniformly dense ones.
  {key:'acts_max',    label:'actions/turn (max)'},
  {key:'acts_mean',   label:'actions/turn (mean)'},
  {key:'mouse_max',   label:'mouse+scroll/turn (max)'},
  {key:'mouse_mean',  label:'mouse+scroll/turn (mean)'},
  {key:'mouse_px',    label:'mouse travel (px)'},
  {key:'clicks',      label:'clicks'},
  {key:'scroll',      label:'scroll'},
  {key:'keys',        label:'key presses'},
  {key:'chars',       label:'chars typed'},
  {key:'letters',     label:'letters'},
  {key:'digits',      label:'digits'},
  {key:'special',     label:'special chars'},
  {key:'measured_length', label:'tokens'},
];
// Per-frame action string the HUD parses: aligned-clock binning when toggled,
// else the hud translation of a native computer_use turn, else the action itself.
function hudAction(f){ return (CLOCK==='aligned' && f.action_aln!=null) ? f.action_aln : (f.hud!=null ? f.hud : f.action); }
// Compact display string for rows / status / tooltips (native turns), else the action.
function dispAction(f){ return f.disp!=null ? f.disp : hudAction(f); }
function curActions(){ return FR.map(hudAction); }
function lb(a,x){ let lo=0,hi=a.length; while(lo<hi){ const m=(lo+hi)>>1; if(a[m]<x) lo=m+1; else hi=m; } return lo; }
// Keep `el` horizontally centered in scroll container `c` (browser clamps at the ends).
function centerX(c, el){ if(!c||!el) return; const cr=c.getBoundingClientRect(), er=el.getBoundingClientRect(); c.scrollLeft += (er.left - cr.left) - (c.clientWidth - er.width)/2; }
// Page vertically: only when `el` leaves the viewport, jump to the page grid holding it.
function pageY(c, el){ if(!c||!el) return; const cr=c.getBoundingClientRect(), er=el.getBoundingClientRect(); if(er.top < cr.top || er.bottom > cr.bottom){ const rel=(er.top - cr.top) + c.scrollTop; c.scrollTop = Math.floor(rel / Math.max(1,c.clientHeight)) * c.clientHeight; } }

// ---- action parsing --------------------------------------------------------
function parseAction(s){
  if(!s || s.trim()==='NO_OP') return {noop:true,dx:0,dy:0,scroll:0,hscroll:0,events:[]};
  const parts=s.split(' ; ');
  const mv=(parts[0]||'').trim().split(/\s+/).map(Number);
  // exactly four numbers = the extended "<dx> <dy> <scroll> <hscroll>" form
  const four = mv.length===4 && mv.every(Number.isFinite);
  const events=[];
  if(parts[1]){
    for(const tok of parts[1].trim().split(/\s+/)){
      if(!tok) continue;
      const sign=tok[0], name=tok.slice(1);
      if((sign==='+'||sign==='-') && name) events.push({sign,name});
    }
  }
  return {noop:false, dx:mv[0]||0, dy:mv[1]||0, scroll:mv[2]||0, hscroll:four?(mv[3]||0):0, events};
}

// ---- typed-text reconstruction --------------------------------------------
const BACK='\b';
const PUNCT={ BackQuote:['`','~'],Minus:['-','_'],Equal:['=','+'],
  BracketLeft:['[','{'],BracketRight:[']','}'],BackSlash:['\\','|'],
  SemiColon:[';',':'],Quote:["'",'"'],Comma:[',','<'],Dot:['.','>'],Slash:['/','?'] };
const SHIFTNUM={Num1:'!',Num2:'@',Num3:'#',Num4:'$',Num5:'%',Num6:'^',Num7:'&',Num8:'*',Num9:'(',Num0:')'};
function keyToChar(name, shift){
  if(/^Key[A-Z]$/.test(name)){ const c=name.slice(3); return shift?c:c.toLowerCase(); }
  if(/^Num[0-9]$/.test(name)){ const d=name.slice(3); return shift?(SHIFTNUM[name]||d):d; }
  if(name==='Space') return ' ';
  if(name==='Return'||name==='Enter'||name==='NumpadEnter'||name==='Numpad Enter') return '\n';
  if(name==='Tab') return '\t';
  if(name==='Backspace') return BACK;
  if(PUNCT[name]) return PUNCT[name][shift?1:0];
  return null;  // modifiers, arrows, F-keys, etc.
}
function isBtn(n){ return n==='LMB'||n==='RMB'||n==='MMB'; }

// Replay events 0..cur -> materialized text (with per-char frame provenance)
// plus the set of keys/buttons still held at the end of `cur`.
function stateAt(cur){
  const chars=[]; const down=new Set();
  for(let i=0;i<=cur;i++){
    for(const ev of PARSED[i].events){
      if(ev.sign==='+'){
        down.add(ev.name);
        const shift=down.has('ShiftLeft')||down.has('ShiftRight');
        const ch=keyToChar(ev.name, shift);
        if(ch===BACK){ if(chars.length) chars.pop(); }
        else if(ch!==null){ chars.push({ch, frame:i}); }
      } else { down.delete(ev.name); }
    }
  }
  const pressed=new Set(PARSED[cur].events.filter(e=>e.sign==='+').map(e=>e.name));
  return {chars, held:down, pressed};
}

// ---- keyboard --------------------------------------------------------------
const KEY_ROWS = [
  [{n:'Escape',l:'esc',w:1.5},{n:'BackQuote',l:'`'},{n:'Num1',l:'1'},{n:'Num2',l:'2'},{n:'Num3',l:'3'},
   {n:'Num4',l:'4'},{n:'Num5',l:'5'},{n:'Num6',l:'6'},{n:'Num7',l:'7'},{n:'Num8',l:'8'},{n:'Num9',l:'9'},
   {n:'Num0',l:'0'},{n:'Minus',l:'-'},{n:'Equal',l:'='},{n:'Backspace',l:'⌫',w:2}],
  [{n:'Tab',l:'tab',w:1.7},{n:'KeyQ',l:'Q'},{n:'KeyW',l:'W'},{n:'KeyE',l:'E'},{n:'KeyR',l:'R'},{n:'KeyT',l:'T'},
   {n:'KeyY',l:'Y'},{n:'KeyU',l:'U'},{n:'KeyI',l:'I'},{n:'KeyO',l:'O'},{n:'KeyP',l:'P'},
   {n:'BracketLeft',l:'['},{n:'BracketRight',l:']'},{n:'BackSlash',l:'\\',w:1.4}],
  [{n:'CapsLock',l:'caps',w:2},{n:'KeyA',l:'A'},{n:'KeyS',l:'S'},{n:'KeyD',l:'D'},{n:'KeyF',l:'F'},{n:'KeyG',l:'G'},
   {n:'KeyH',l:'H'},{n:'KeyJ',l:'J'},{n:'KeyK',l:'K'},{n:'KeyL',l:'L'},{n:'SemiColon',l:';'},{n:'Quote',l:"'"},
   {n:'Return',l:'⏎',w:2.1}],
  [{n:'ShiftLeft',l:'shift',w:2.5},{n:'KeyZ',l:'Z'},{n:'KeyX',l:'X'},{n:'KeyC',l:'C'},{n:'KeyV',l:'V'},{n:'KeyB',l:'B'},
   {n:'KeyN',l:'N'},{n:'KeyM',l:'M'},{n:'Comma',l:','},{n:'Dot',l:'.'},{n:'Slash',l:'/'},{n:'ShiftRight',l:'shift',w:2.6}],
  [{n:['ControlLeft','Control'],l:'ctrl',w:1.7},{n:['MetaLeft','Meta','MetaGr'],l:'meta',w:1.4},{n:'Alt',l:'alt',w:1.4},
   {n:'Space',l:'space',w:6.7},{n:['AltGr','AltRight'],l:'alt',w:1.4},{n:'MetaRight',l:'meta',w:1.4},
   {n:'ControlRight',l:'ctrl',w:1.7}],
];
const ARROWS=[{n:'LeftArrow',l:'←'},{n:'UpArrow',l:'↑'},{n:'DownArrow',l:'↓'},{n:'RightArrow',l:'→'}];
let UNIT=26;  // key size in px; auto-fitted to the sidebar width (see fitKeyboard)
let keyEls=[]; const allKeyNames=new Set();
function buildKeyboard(){
  const kbd=$('#kbd'); kbd.innerHTML=''; keyEls=[]; allKeyNames.clear();
  const addKey=(row,k)=>{
    const el=document.createElement('div'); el.className='key';
    el.style.width=((k.w||1)*UNIT)+'px'; el.textContent=k.l;
    const names=Array.isArray(k.n)?k.n:[k.n];
    names.forEach(n=>allKeyNames.add(n));
    keyEls.push({names:new Set(names), el}); row.appendChild(el);
  };
  for(const row of KEY_ROWS){ const r=document.createElement('div'); r.className='krow'; for(const k of row) addKey(r,k); kbd.appendChild(r); }
  const ar=document.createElement('div'); ar.className='krow arrows'; for(const k of ARROWS) addKey(ar,k); kbd.appendChild(ar);
}
function lightKeyboard(pressed, held){
  for(const {names,el} of keyEls){
    let p=false,h=false;
    for(const n of names){ if(pressed.has(n)) p=true; else if(held.has(n)) h=true; }
    el.classList.toggle('press', p);
    el.classList.toggle('held', !p && h);
  }
  const others=[];
  for(const n of pressed) if(!allKeyNames.has(n) && !isBtn(n)) others.push([n,'press']);
  for(const n of held) if(!pressed.has(n) && !allKeyNames.has(n) && !isBtn(n)) others.push([n,'held']);
  $('#otherkeys').innerHTML = others.map(([n,c])=>`<span class="chip ${c}">${esc(n)}</span>`).join('');
}

// ---- mouse radar -----------------------------------------------------------
const btnEls={};
function initRadar(){ document.querySelectorAll('.btn').forEach(b=>btnEls[b.dataset.b]=b); }
function updateRadar(p, pressed, held){
  const R=42, cx=50, cy=50;
  const mag=Math.hypot(p.dx,p.dy);
  const len = mag>0 ? R*Math.tanh(mag/250) : 0;
  const ang = Math.atan2(p.dy,p.dx);       // screen coords: +y is down
  const x2 = cx+len*Math.cos(ang), y2 = cy+len*Math.sin(ang);
  const line=$('#mvline'), head=$('#mvhead');
  line.setAttribute('x2',x2.toFixed(2)); line.setAttribute('y2',y2.toFixed(2));
  head.setAttribute('transform',`translate(${x2.toFixed(2)},${y2.toFixed(2)}) rotate(${(ang*180/Math.PI).toFixed(1)})`);
  const vis = mag>0 ? 1 : 0; line.style.opacity=vis; head.style.opacity=vis;
  for(const [b,el] of Object.entries(btnEls)){
    el.classList.toggle('press', pressed.has(b));
    el.classList.toggle('held', !pressed.has(b) && held.has(b));
  }
  let sc = p.scroll ? (p.scroll>0 ? `▲ scroll ${p.scroll}` : `▼ scroll ${Math.abs(p.scroll)}`) : '';
  if(p.hscroll) sc += (sc?' · ':'') + (p.hscroll>0 ? `▶ ${p.hscroll}` : `◀ ${Math.abs(p.hscroll)}`);
  $('#scrollind').textContent = sc;
  $('#dxy').textContent = `Δ ${p.dx}, ${p.dy}`;
}

// ---- typed text render -----------------------------------------------------
function esc(s){ return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function collapsibleNote(cls, label, text){
  if(!text) return '';
  return `<details><summary><span class="${cls}">${label}</span></summary><div class="cnote-text">${esc(String(text))}</div></details>`;
}
// The goal is the one thing you need on screen at all times to judge whether a
// trace is doing the right thing — unlike context/system prompt, it's never
// tucked behind a click.
function goalNote(text){
  if(!text) return '';
  return `<div class="cnote-fixed"><span class="cnote-goal">goal</span><div class="cnote-text">${esc(String(text))}</div></div>`;
}
// Dedicated full-chat window. Reassembled client-side from what the segment detail
// already carries (system prompt + goal/context + per-frame actions) — no extra
// payload. Images are stripped from the client, so each user turn shows a screenshot
// placeholder (index + time). Toggled from the banner button; collapse with ✕ / Esc.
function renderChat(){
  const body=$('#chatbody'), sub=$('#chatsub');
  if(!SEG || !FR.length){ if(body) body.innerHTML='<div class="fnote" style="color:#8b93a1">no turns</div>'; return; }
  if(sub) sub.textContent=`${SEG.segment_id||''} · ${FR.length} turns`;
  const h=[];
  if(SEG.system_prompt){
    h.push(`<div class="turn sys"><div class="role">system</div>`
      +`<details class="syswrap"><summary>show system prompt</summary>`
      +`<div class="body" style="margin-top:5px">${esc(String(SEG.system_prompt))}</div></details></div>`);
  }
  for(let i=0;i<FR.length;i++){
    let u=`<div class="turn user"><div class="role">user · turn ${i}</div>`;
    if(i===0){
      if(SEG.instruction) u+=`<div class="goal">🎯 ${esc(String(SEG.instruction))}</div>`;
      if(SEG.context)     u+=`<div class="ctx">${esc(String(SEG.context))}</div>`;
      if(!SEG.instruction && !SEG.context && SEG.first_user_text)
        u+=`<div class="body">${esc(String(SEG.first_user_text))}</div>`;
    }
    u+=`<div class="shot">🖼 screenshot #${i}${FR[i].t!=null?` · t=${FR[i].t}s`:''}${FR[i].is_black?' · ⬛ black':''}</div></div>`;
    h.push(u);
    h.push(`<div class="turn asst"><div class="role">assistant · turn ${i}</div>`
      +`<div class="body act">${esc(String(FR[i].action||'')) || '∅'}</div></div>`);
  }
  body.innerHTML=h.join('');
  body.scrollTop=0;
}
function chatBtn(){ return `<span id="chatbtn">💬 full chat (${FR.length} turns)</span>`; }
function openChat(){ renderChat(); $('#chatwin').classList.add('show'); }
function closeChat(){ $('#chatwin').classList.remove('show'); }
function toggleChat(){ $('#chatwin').classList.contains('show') ? closeChat() : openChat(); }
function renderTyped(chars, cur){
  const el=$('#typed');
  if(!chars.length){ el.innerHTML='<span class="empty">— no text typed yet —</span>'; return; }
  let html=''; let i=0;
  while(i<chars.length){
    const hi = chars[i].frame===cur;
    let j=i, run='';
    while(j<chars.length && (chars[j].frame===cur)===hi){ run+=chars[j].ch; j++; }
    html += hi ? `<span class="now">${esc(run)}</span>` : esc(run);
    i=j;
  }
  html+='<span class="caret"></span>';
  el.innerHTML=html;
  const now=el.querySelector('.now');
  if(now) now.scrollIntoView({block:'nearest'}); else el.scrollTop=el.scrollHeight;
}

// ---- data loading ----------------------------------------------------------
async function jget(u){ const r=await fetch(u); if(!r.ok) throw new Error(await r.text()); return r.json(); }

async function loadDatasets(){
  const info=await jget('/api/datasets');
  const sel=$('#ds'); sel.innerHTML='';
  for(const d of info.datasets){
    const o=document.createElement('option'); o.value=d.name;
    const ml = d.mode==='frames_master'?'master/raw':d.mode==='conversations'?'conversation':d.mode==='inline_records'?'stage-06 records':'sample';
    o.textContent=`${d.name}  (${ml})`;
    sel.appendChild(o);
  }
  initSampling(info.sampling||{});
  DS = info.default || (info.datasets[0] && info.datasets[0].name) || null;
  if(DS){ sel.value=DS; await loadSegments(); }
}
// ---- sampling (first N / random N, seeded) ----------------------------------
// Prime the controls from the server's CLI defaults (--limit/--sample-mode/--seed),
// overridden by whatever this browser was last left on — but only while the launch
// defaults are the SAME ones that override was made against. Relaunching with a
// different --limit/--sample-mode/--seed is an explicit instruction, so it wins over
// a remembered choice instead of silently doing nothing.
function initSampling(def){
  let s={mode:def.mode||'first', n:(def.n!=null?def.n:null), seed:def.seed||0};
  SAMP_DEFKEY=JSON.stringify([s.mode, s.n, s.seed]);
  try{
    const saved=JSON.parse(localStorage.getItem('fr_sampling')||'null');
    if(saved&&saved.mode&&saved.defKey===SAMP_DEFKEY)
      s={mode:saved.mode, n:(saved.n!=null?saved.n:s.n), seed:saved.seed||0};
  }catch(e){}
  SAMP=s;
  $('#smode').value=SAMP.mode; $('#sn').value=(SAMP.n!=null?SAMP.n:''); $('#sseed').value=SAMP.seed;
  syncSeedVis();
}
// The seed only means something for a random draw.
function syncSeedVis(){ $('#sseedwrap').style.display = SAMP.mode==='random' ? '' : 'none'; }
// Re-read the controls and, if the sampling really changed, rebuild the dataset
// under it (a rebuild re-reads the store, so an unchanged sampling does nothing).
async function onSamplingChange(){
  const nRaw=String($('#sn').value||'').trim();
  const next={
    mode: $('#smode').value,
    n: nRaw===''?null:Math.max(1, parseInt(nRaw,10)||1),
    seed: parseInt($('#sseed').value,10)||0,
  };
  const same = next.mode===SAMP.mode && next.n===SAMP.n && next.seed===SAMP.seed;
  SAMP=next; syncSeedVis();
  $('#sn').value=(SAMP.n!=null?SAMP.n:'');
  localStorage.setItem('fr_sampling', JSON.stringify({...SAMP, defKey:SAMP_DEFKEY}));
  if(same) return;
  await loadSegments();
}
// What the server actually loaded — reported from its own answer, not from the
// controls, so "random N" with no N shows up as the "all" it fell back to.
function showSampling(info){
  const el=$('#sampinfo'); el.classList.remove('busy');
  const sp=info.sampling||{}, tot=info.total_available, n=info.n_segments;
  // "of <total>" only where the loader knows the population: always for a random
  // draw, and for "first N" on the stores whose size is metadata (inline records).
  const of = tot!=null ? ` of ${tot}` : '';
  el.textContent = sp.mode==='random'
    ? `random ${n}${tot!=null?of:' of ?'} · seed ${sp.seed}`
    : (info.limited ? `first ${n}${of}` : `all ${n}`);
}
async function loadSegments(){
  $('#sampinfo').textContent='sampling…'; $('#sampinfo').classList.add('busy');
  const info=await jget('/api/segments?ds='+encodeURIComponent(DS)+sampQS());
  if(info.error){ $('#seginfo').textContent='⚠ '+info.error; $('#sampinfo').textContent=''; $('#sampinfo').classList.remove('busy'); $('#seg').innerHTML=''; $('#strip').innerHTML=''; $('#rows').innerHTML=''; ALL_SEGMENTS=[]; buildFilters([]); return; }
  showSampling(info);
  MODE=info.mode||'frame_records';
  MASTER_FPS=info.master_fps||MASTER_FPS;
  const note=$('#modenote');
  if(MODE==='frames_master' && info.has_alignment){
    note.textContent=`master + realignment overlay — event table is dual-clock (raw vs aligned); the lower "aligned + trims" timeline shows collapsed idle spans. The frames themselves are raw (unrealigned).`;
    note.classList.add('show');
  } else if(MODE==='frames_master' && info.has_keylog){
    note.textContent=`raw frames-master (01a) + keylog overlay — raw events by timestamp at master_fps=${info.master_fps||'?'} (no realignment). Pass --alignment for the dual-clock/trims view.`;
    note.classList.add('show');
  } else if(MODE==='frames_master'){
    note.textContent=`raw frames-master (01a), master_fps=${info.master_fps||'?'} — no keylog linked, so no actions. Pass --clips-manifest, or run the 01b sampler.`;
    note.classList.add('show');
  } else if(MODE==='conversations'){
    note.textContent=`stage-04 conversations · ${info.goal_conditioned?'goal-conditioned':'goal-free'}${info.action_format?` · ${info.action_format} actions`:''}${info.target_fps?` · ${info.target_fps} fps`:''} — each step is one screenshot→action turn (assistant reply = the frame's action). Frames resolve from the linked stage-01a master.`;
    note.classList.add('show');
  } else if(MODE==='inline_records'){
    note.textContent=`stage-06 inline records · ${info.model_id||'?'} · max_length ${info.max_length||'?'}${info.overflow_mode?` · overflow=${info.overflow_mode}`:''}${info.splits&&info.splits.length?` · splits: ${info.splits.join('/')}`:''} — each entry is ONE tokenized training example (a ≤max_length conversation chunk; ar:// frames into the stage-01a master). Only kept records are shown, so this IS the model's training input.`;
    note.classList.add('show');
  } else note.classList.remove('show');
  ALL_SEGMENTS = info.segments || [];
  // The hit map belongs to the dataset it was searched in.
  clearTimeout(_actTimer); ACTQ=''; ACTHITS=null; MATCHES=[];
  await loadMarks();       // marks belong to DS too — load before options render
  buildFilters(ALL_SEGMENTS);
  await applyFilters();   // populates #seg from the (filtered) list and loads one
}

// ---- golden-trace marks -----------------------------------------------------
// Marking is a per-segment boolean (this trace is 100% correct), persisted
// server-side to golden_marks.json (or golden_marks_<id>.json, see MARKS_ID)
// next to the dataset so it survives restarts and browser reloads; the client
// just mirrors that file in MARKS.
// MARKS_ID namespaces the marks file so independent review passes/sessions
// don't share (and overwrite) one another's marks: blank -> golden_marks.json,
// set -> golden_marks_<id>.json. Remembered per-browser via localStorage, since
// it's a "who/which pass is reviewing" choice, not a dataset property.
let MARKS_ID = localStorage.getItem('fr_marks_id') || '';
$('#marksid').value = MARKS_ID;
function marksQS(){ return '&mid='+encodeURIComponent(MARKS_ID); }
async function loadMarks(){
  try{
    const info=await jget('/api/marks?ds='+encodeURIComponent(DS)+marksQS());
    MARKS = info.marks || {};
    MARKS_PATH = info.path || '';
  }catch(e){ MARKS = {}; MARKS_PATH=''; }
  updateMarkCount();
}
function updateMarkCount(){
  const el=$('#markinfo'); if(!el) return;
  el.textContent = Object.keys(MARKS).length+' marked';
  el.title = MARKS_PATH || '';
}
function updateMarkUI(){
  const on = !!(SEG && MARKS[SEG.segment_id]);
  const btn=$('#markgood'); if(!btn) return;
  btn.classList.toggle('on', on);
  btn.textContent = on ? '★ marked good' : '☆ mark good';
}
async function toggleMark(){
  if(!SEG || !SEG.segment_id) return;
  const id=SEG.segment_id, marked=!MARKS[id];
  const r=await fetch('/api/mark', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ds:DS, id, marked, mid:MARKS_ID}),
  });
  const j=await r.json();
  if(j.error){ console.error(j.error); return; }
  if(marked) MARKS[id]={ts:Date.now()}; else delete MARKS[id];
  MARKS_PATH = j.path || MARKS_PATH;
  updateMarkUI(); updateMarkCount();
  const sel=$('#seg'); const opt=sel && sel.querySelector(`option[value="${CSS.escape(id)}"]`);
  const s=ALL_SEGMENTS.find(s=>s.segment_id===id);
  if(opt && s) opt.textContent = segOptionLabel(s);
  // "marked only" filters on MARKS membership, so a toggle can drop the current
  // segment out of the visible list — re-run the filter to reflect that.
  if($('#fmarked') && $('#fmarked').checked) await applyFilters();
}
// Switching marks id mid-session re-reads that file's marks for DS and refreshes
// every place MARKS shows up (button state, counter, dropdown stars, "marked
// only" filter) without touching the loaded segment/frame data.
async function onMarksIdChange(){
  MARKS_ID = $('#marksid').value.trim();
  localStorage.setItem('fr_marks_id', MARKS_ID);
  await loadMarks();
  updateMarkUI();
  if(ALL_SEGMENTS.length) await applyFilters();
}

// ---- filtering (left sidebar) ----------------------------------------------
// The dropdown option label for one segment, per dataset mode.
function stuckOptionLabel(s){
  if(!(s.stuck_key_frames>0)) return '';
  return `, stuck ${s.stuck_key||'key'} ${s.stuck_key_frames}f${s.stuck_key_unreleased?' open':''}`;
}
function segOptionLabel(s){
  const star = MARKS[s.segment_id] ? '★ ' : '';
  return star + (MODE==='frames_master'
    ? `${s.segment_id}  (${s.n_frames} frames, ${s.duration_s}s${s.align_status?', '+s.align_status:''})`
    : MODE==='inline_records'
    ? `${s.segment_id}  (${s.n_non_noop}/${s.n_frames} act${s.measured_length?', '+s.measured_length+' tok':''}${stuckOptionLabel(s)})`
    : `${s.segment_id}  (${s.n_non_noop}/${s.n_frames} act, ${s.duration_s}s${stuckOptionLabel(s)})`);
}
const _has = (segs,k)=>segs.some(s=>s[k]!=null);
// Build the filter controls to match the fields this dataset actually carries.
function buildFilters(segs){
  const hasGoal=_has(segs,'instruction'), hasConv=_has(segs,'conv_text'), hasTyped=_has(segs,'typed'), hasStuck=_has(segs,'stuck_key_frames');
  // action search: every mode whose turns carry actions (acts_max comes from
  // compute_metrics, which is exactly those modes)
  const hasActs=_has(segs,'acts_max');
  // segment id is always present — with tens of thousands of segments this is the
  // only practical way to reach a known one without scrolling the dropdown
  let th=`<label class="frow2"><input id="fmarked" type="checkbox"> ★ marked good only</label>`
        + `<label class="frow2">segment id contains<input id="fseg" type="search" placeholder="uuid / _seg0000 / substring…"></label>`;
  if(hasActs) th+=`<label class="frow2">action contains<input id="fact" type="search" placeholder='down(LMB) · move(-100, · type(" · KeyEnter'></label>`
                + `<div class="fnote"><span id="factstat"></span>every turn searched verbatim · `
                + `<kbd>n</kbd>/<kbd>N</kbd> step through the matches in the open segment</div>`;
  if(hasGoal)  th+=`<label class="frow2">goal contains<input id="fgoal" type="search" placeholder="substring…"></label>`;
  if(hasConv)  th+=`<label class="frow2">conversation contains<input id="fconv" type="search" placeholder="goal / context / any action…"></label>`;
  if(hasTyped) th+=`<label class="frow2">typed text contains<input id="ftyped" type="search" placeholder="substring…"></label>`;
  if(hasStuck) th+=`<label class="frow2">stuck key held &gt; frames<input id="fstuck" type="number" min="0" step="1" placeholder="T"></label>`;
  $('#ftext').innerHTML = th;
  activeNumMetrics = NUM_METRICS.filter(m=>_has(segs,m.key));
  $('#fnums').innerHTML = activeNumMetrics.map(m=>
    `<div class="frow"><span class="flab" title="${m.label}">${m.label}</span>`
    + `<input id="min_${m.key}" class="fnum" type="number" step="any" placeholder="min">`
    + `<input id="max_${m.key}" class="fnum" type="number" step="any" placeholder="max"></div>`
  ).join('') || '<div class="fnote">no numeric metrics</div>';
  const sopts=['<option value="">— none —</option>']
    .concat(activeNumMetrics.map(m=>`<option value="${m.key}">${m.label}</option>`));
  if(hasGoal) sopts.push('<option value="instruction">goal (A→Z)</option>');
  $('#fsort').innerHTML = `<select id="fsortkey">${sopts.join('')}</select>`
    + `<select id="fsortdir"><option value="desc">high → low</option><option value="asc">low → high</option></select>`;
  // Re-filter live as any control changes (elements are recreated per dataset).
  $('#filters').querySelectorAll('input,select').forEach(el=>{
    if(el.id==='fact'){   // needs the server: debounce, then re-filter
      el.addEventListener('input', queueActionSearch);
      el.addEventListener('change', queueActionSearch);
      el.addEventListener('keydown', e=>{ if(e.key==='Enter'){ clearTimeout(_actTimer); runActionSearch(); } });
      return;
    }
    el.addEventListener('input', applyFilters);
    el.addEventListener('change', applyFilters);
  });
}
// The current filtered + sorted segment list.
function filteredSegments(){
  const sg=($('#fseg')&&$('#fseg').value||'').trim().toLowerCase();
  const g=($('#fgoal')&&$('#fgoal').value||'').trim().toLowerCase();
  const cv=($('#fconv')&&$('#fconv').value||'').trim().toLowerCase();
  const tp=($('#ftyped')&&$('#ftyped').value||'').trim().toLowerCase();
  const stuckRaw=($('#fstuck')&&$('#fstuck').value||'').trim();
  const stuckT=stuckRaw===''?null:Number(stuckRaw);
  const markedOnly=!!($('#fmarked')&&$('#fmarked').checked);
  let list=ALL_SEGMENTS.filter(s=>{
    if(markedOnly && !MARKS[s.segment_id]) return false;
    if(sg && !String(s.segment_id||'').toLowerCase().includes(sg)) return false;
    if(g && !String(s.instruction||'').toLowerCase().includes(g)) return false;
    if(cv && !String(s.conv_text||'').toLowerCase().includes(cv)) return false;
    if(tp && !String(s.typed||'').toLowerCase().includes(tp)) return false;
    // action search: only once the server has answered for the CURRENT query,
    // so a half-typed token never silently empties the list
    if(ACTHITS && ACTQ===actQuery() && !(ACTHITS[s.segment_id]>0)) return false;
    if(stuckT!=null && Number.isFinite(stuckT) && !((s.stuck_key_frames||0)>stuckT)) return false;
    for(const m of activeNumMetrics){
      const loEl=$('#min_'+m.key), hiEl=$('#max_'+m.key);
      const lo=loEl?loEl.value:'', hi=hiEl?hiEl.value:'';
      const v=s[m.key];
      if(lo!=='' && (v==null || v< +lo)) return false;
      if(hi!=='' && (v==null || v> +hi)) return false;
    }
    return true;
  });
  const skEl=$('#fsortkey'), sk=skEl?skEl.value:'';
  if(sk){
    const dir=($('#fsortdir')&&$('#fsortdir').value==='asc')?1:-1;
    list=list.slice().sort((a,b)=> sk==='instruction'
      ? dir*String(a.instruction||'').localeCompare(String(b.instruction||''))
      : dir*(((a[sk]!=null?a[sk]:0)-(b[sk]!=null?b[sk]:0))||0));
  }
  return list;
}
// Repopulate the segment dropdown from the filtered list; keep the current
// selection if it still matches, else load the first match.
async function applyFilters(){
  const list=filteredSegments();
  const prev=SEG&&SEG.segment_id;
  const sel=$('#seg'); sel.innerHTML='';
  for(const s of list){ const o=document.createElement('option'); o.value=s.segment_id; o.textContent=segOptionLabel(s); sel.appendChild(o); }
  const cnt=$('#fcount'); if(cnt) cnt.textContent=`${list.length} / ${ALL_SEGMENTS.length} match`;
  if(!list.length){ $('#seginfo').textContent='⚠ no segments match the filters'; return; }
  if(prev && list.some(s=>s.segment_id===prev)){
    sel.value=prev;   // keep current view
    if(FR.length){
      refreshMatches(); buildStrip(); buildRows();
      // a fresh search moves to its first hit; without one, stay where you were
      show(MATCHES.length && !MATCHES.includes(cur) ? MATCHES[0] : cur);
    }
  } else await loadSegment(list[0].segment_id);
}
async function loadSegment(id){
  SEG=await jget('/api/segment?ds='+encodeURIComponent(DS)+'&id='+encodeURIComponent(id)+sampQS());
  updateMarkUI();
  FR=SEG.frames;
  DUAL = !!SEG.dual_clock;
  ALN = SEG.align || null;
  ALNACT = ALN ? new Set(ALN.aligned_active) : null;
  EV = (SEG.events && SEG.events.length) ? SEG.events : null;
  // Sync/highlight by the aligned clock when dual (frames are video-clock), else raw.
  EVT = EV ? EV.map(e=> DUAL ? e[1] : e[0]) : [];
  HAS_ACTIONS = !!SEG.has_actions || FR.some(f=>!f.is_noop);
  MASTER_FPS = SEG.master_fps || MASTER_FPS;
  _evLo=_evHi=-1; _lastCell=_lastCell2=_lastRow=null;
  const canAlign = FR.some(f=>f.action_aln!=null);
  if(!canAlign) CLOCK='raw';
  const tg=$('#clocktoggle');
  tg.style.display = canAlign ? '' : 'none';
  tg.textContent = 'keys: '+CLOCK; tg.classList.toggle('on', CLOCK==='aligned');
  PARSED=curActions().map(parseAction);
  const stuckInfo = SEG.stuck_key_frames
    ? ` · stuck ${SEG.stuck_key||'key'} ${SEG.stuck_key_frames}f${SEG.stuck_key_unreleased?' unreleased':''}`
    : '';
  $('#seginfo').textContent = ((MODE==='frames_master' && !HAS_ACTIONS)
    ? `${SEG.recording_id||'?'} · ${SEG.n_frames} raw frames`
    : `${SEG.recording_id||'?'} · ${SEG.n_non_noop}/${SEG.n_frames} active frames`)
    + (SEG.n_black ? ` · ${SEG.n_black} black${SEG.n_black_act?` (${SEG.n_black_act} w/ action)`:''}` : '')
    + stuckInfo;
  if(MODE==='conversations'){
    // Per-segment banner: the system prompt + (optional) goal instruction that
    // frame this conversation, plus its turn count / fps / alignment.
    const note=$('#modenote');
    let h=`<b>stage-04 conversation</b> · ${SEG.goal_conditioned?'goal-conditioned':'goal-free'}`;
    if(SEG.action_format) h+=` · ${esc(String(SEG.action_format))}`;
    if(SEG.target_fps) h+=` · ${SEG.target_fps} fps`;
    if(SEG.alignment_status) h+=` · ${esc(String(SEG.alignment_status))}`;
    h+=` · ${SEG.n_turns||FR.length} turns`;
    h+=goalNote(SEG.instruction);
    h+=collapsibleNote('cnote-context', 'context', SEG.context);
    h+=collapsibleNote('cnote-sys', 'system prompt', SEG.system_prompt);
    h+='<div>'+chatBtn()+'</div>';
    note.innerHTML=h; note.classList.add('show');
  }
  if(MODE==='inline_records'){
    // Per-record banner: which split, how full the token budget is, and the
    // system prompt / goal that frame this training example.
    const note=$('#modenote');
    const ml=SEG.measured_length, mx=SEG.max_length;
    const pct=(ml&&mx)?Math.round(100*ml/mx):null;
    let h=`<b>stage-06 record</b> · <span class="cnote-goal">${esc(String(SEG.split||'?'))}</span> split · ${SEG.n_turns||FR.length} turns`;
    if(ml) h+=` · ${ml}${mx?`/${mx}`:''} tok${pct!=null?` (${pct}% of budget)`:''}`;
    if(SEG.model_id) h+=` · ${esc(String(SEG.model_id))}`;
    if(SEG.overflow_mode) h+=` · overflow=${esc(String(SEG.overflow_mode))}`;
    if(SEG.action_format) h+=` · ${esc(String(SEG.action_format))}`;
    if(SEG.target_fps) h+=` · ${SEG.target_fps} fps`;
    h+=goalNote(SEG.instruction);
    h+=collapsibleNote('cnote-context', 'context', SEG.context);
    h+=collapsibleNote('cnote-sys', 'system prompt', SEG.system_prompt);
    h+='<div>'+chatBtn()+'</div>';
    note.innerHTML=h; note.classList.add('show');
  }
  $('#fn').textContent=FR.length;
  refreshMatches();
  buildStrip(); buildStrip2(); buildRows();
  // Land on the first matching turn when a search is active — the segment was
  // selected BECAUSE it contains one.
  show(MATCHES.length ? MATCHES[0] : 0);
  if($('#chatwin').classList.contains('show')) renderChat();  // keep the open window in sync
}
// ---- action search ---------------------------------------------------------
// Two halves of one query: the server says WHICH segments contain it (the
// segment list must not carry every trajectory's action text), and the client
// finds the matching turns inside the segment it has open.
let ACTQ='', ACTHITS=null, MATCHES=[], _actTimer=null;
function actQuery(){ const el=$('#fact'); return el ? el.value.trim() : ''; }
function actStat(t){ const el=$('#factstat'); if(el) el.textContent = t ? t+' · ' : ''; }
async function runActionSearch(){
  const q=actQuery();
  if(!q){ ACTQ=''; ACTHITS=null; actStat(''); refreshMatches(); await applyFilters(); return; }
  actStat('searching…');
  try{
    const r=await jget('/api/find?ds='+encodeURIComponent(DS)+'&q='+encodeURIComponent(q)+sampQS());
    if(r.error){ actStat(r.error); return; }
    ACTQ=q; ACTHITS=r.segments||{};
    actStat(`${r.n_turns} turns in ${r.n_segments} segments`);
  }catch(e){ actStat(String(e)); return; }
  refreshMatches();
  await applyFilters();
}
function queueActionSearch(){ clearTimeout(_actTimer); _actTimer=setTimeout(runActionSearch, 250); }
// Does this turn match? Raw text AND the rendered line, so "down(LMB)" reaches
// an ordered turn's primitives and "click" reaches a native tool call.
function turnMatches(f, q){
  return String(f.action||'').toLowerCase().includes(q)
      || String(dispAction(f)||'').toLowerCase().includes(q);
}
function refreshMatches(){
  const q=actQuery().toLowerCase();
  MATCHES = q ? FR.map((f,i)=>turnMatches(f,q)?i:-1).filter(i=>i>=0) : [];
  const on=MATCHES.length>0;
  $('#prevmatch').style.display=on?'':'none';
  $('#nextmatch').style.display=on?'':'none';
  updateMatchInfo();
}
function updateMatchInfo(){
  const el=$('#matchinfo');
  if(!MATCHES.length){ el.textContent = actQuery()? '0 matches here' : ''; return; }
  const k=MATCHES.indexOf(cur);
  el.textContent = `match ${k>=0?k+1:'–'}/${MATCHES.length}`;
}
// Next/previous matching turn, wrapping at the ends.
function nextMatch(d){
  if(!MATCHES.length) return;
  let i = d>0 ? MATCHES.find(x=>x>cur) : [...MATCHES].reverse().find(x=>x<cur);
  if(i===undefined) i = d>0 ? MATCHES[0] : MATCHES[MATCHES.length-1];
  show(i);
}
// Escape for HTML, then wrap the query's occurrences in <mark>.
function hlText(s, q){
  const e=esc(s);
  if(!q) return e;
  const needle=esc(q).replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  return e.split(new RegExp('('+needle+')','ig'))
          .map((p,i)=> i%2 ? '<mark>'+p+'</mark>' : p).join('');
}

// Built via innerHTML (+ delegated click) so multi-thousand-frame segments
// (e.g. a 15fps frames-master: ~6k frames) render fast.
function stuckThreshold(){
  const el=$('#fstuck');
  if(!el || el.value.trim()==='') return null;
  const t=Number(el.value);
  return Number.isFinite(t) ? t : null;
}
function stuckTimelineMap(){
  const t=stuckThreshold();
  const map=new Map();
  if(t==null || !SEG || !SEG.key_hold_ranges) return map;
  for(const r of SEG.key_hold_ranges){
    if(!((r.frames||0)>t)) continue;
    const lo=Math.max(0, Math.min(FR.length-1, r.start||0));
    const hi=Math.max(lo, Math.min(FR.length-1, r.end||0));
    for(let i=lo;i<=hi;i++){
      if(!map.has(i)) map.set(i, []);
      map.get(i).push(r);
    }
  }
  return map;
}
function buildStrip(){
  const cuts=(ALN&&ALN.cut_ranges)?ALN.cut_ranges:[];
  const inCut=i=>{ for(const r of cuts) if(i>=r[0]&&i<r[1]) return true; return false; };
  const stuck=stuckTimelineMap();
  const mset=new Set(MATCHES);
  let h='';
  for(let i=0;i<FR.length;i++){ const f=FR[i]; const cut=inCut(i); const blk=!!f.is_black; const blkAct=blk&&!f.is_noop;
    const stuckHere=stuck.get(i)||[];
    const stuckTitle=stuckHere.length
      ? '['+stuckHere.map(r=>`${r.key||'key'} held ${r.frames||0}f${r.unreleased?' unreleased':''}`).join(', ')+'] '
      : '';
    h+=`<div class="cell${f.is_noop?'':' act'}${cut?' cut':''}${blk?' black':''}${stuckHere.length?' stuck':''}${mset.has(i)?' match':''}" data-i="${i}" title="#${i} t=${f.t}s ${stuckTitle}${blkAct?'[black + action] ':(blk?'[black frame] ':'')}${cut?'[cut — collapsed idle] ':''}${esc(dispAction(f)||'NO_OP')}"></div>`; }
  $('#strip').innerHTML=h;
}
// Second timeline (master + alignment only): aligned keys on the video clock,
// with collapse markers where realignment folded idle spans.
function buildStrip2(){
  const wrap=$('#strip2wrap');
  if(!ALN){ wrap.style.display='none'; $('#strip2').innerHTML=''; return; }
  wrap.style.display='';
  const col=new Map(); for(const c of (ALN.collapse_frames||[])) col.set(c.frame, c);
  let h='';
  for(let i=0;i<FR.length;i++){
    const c=col.get(i);
    const cls='cell'+(ALNACT&&ALNACT.has(i)?' act':'')+(c?' col':'');
    const title=c ? `${c.collapse}s of idle collapsed here (keylog ${c.kp}s → video ${(i/MASTER_FPS).toFixed(1)}s)`
                  : `aligned #${i} t=${(i/MASTER_FPS).toFixed(2)}s`;
    h+=`<div class="${cls}" data-i="${i}" title="${title}"></div>`;
  }
  $('#strip2').innerHTML=h;
  const ov=ALN.n_overhang_events||0;
  $('#alignstat').textContent =
    `${ALN.status||'?'} · collapse ${(ALN.total_collapse_s||0).toFixed(1)}s · residual ${(ALN.residual_s||0).toFixed(1)}s`
    + (ov?` · ${ov} events trimmed past video`:'');
}
function buildRows(){
  const head=$('#rowshead');
  if(EV && DUAL){                             // dual-clock events (raw vs aligned)
    if(head) head.textContent=`events — raw ▸ aligned (${EV.length})`;
    let h='';
    for(let k=0;k<EV.length;k++){ const e=EV[k];   // [t_raw, t_aln, type, detail, trimmed]
      const fi=Math.floor(e[1]*MASTER_FPS);
      h+=`<div class="erow dual${e[4]?' trim':''}" id="ev${k}" data-fi="${fi}"><span class="raw">${e[0].toFixed(2)}</span><span class="aln">${e[1].toFixed(2)}</span><span class="ty">${esc(String(e[2]))}</span><span class="a">${esc(String(e[3]))}</span></div>`; }
    $('#rows').innerHTML=h;
  } else if(EV){                              // single-clock raw events
    if(head) head.textContent=`raw events (${EV.length})`;
    let h='';
    for(let k=0;k<EV.length;k++){ const e=EV[k];
      h+=`<div class="erow" id="ev${k}" data-fi="${Math.floor(e[0]*MASTER_FPS)}"><span class="t">${e[0].toFixed(2)}s</span><span class="ty">${esc(String(e[1]))}</span><span class="a">${esc(String(e[2]))}</span></div>`; }
    $('#rows').innerHTML=h;
  } else {                                    // per-frame binned actions
    if(head) head.textContent='actions  (n = actions in the turn)';
    const q=actQuery().toLowerCase();
    let h='';
    for(let i=0;i<FR.length;i++){ const f=FR[i];
      // the turn as the label spells it; the FULL raw text (thinking included,
      // tool-call JSON for native turns) stays on hover
      const title=f.disp!=null?` title="${esc(f.action||'')}"`:'';
      const n=f.n_act||0;
      const nc=`<span class="n${n>=10?' hi':''}" title="${n} actions, ${f.n_mouse||0} mouse/scroll">${n||''}</span>`;
      const m=q&&turnMatches(f,q)?' match':'';
      h+=`<div class="row${f.is_noop?' noop':''}${m}" id="row${i}" data-i="${i}"${title}><span class="t">#${i}</span><span class="t">${f.t}s</span>${nc}<span class="a">${hlText(dispAction(f)||'NO_OP', q)}</span></div>`; }
    $('#rows').innerHTML=h;
  }
}

// ---- main render -----------------------------------------------------------
function show(i){
  if(!FR.length) return;
  cur=Math.max(0,Math.min(FR.length-1,i));
  const f=FR[cur], p=PARSED[cur];
  $('#frameimg').src=`/frame?ds=${encodeURIComponent(DS)}&seg=${encodeURIComponent(SEG.segment_id)}&i=${cur}`+sampQS();
  $('#fi').textContent=cur; $('#ft').textContent=f.t;
  $('#fbin').textContent=(f.bin??'–'); $('#fsrc').textContent=(f.src??'–');
  const act=hudAction(f);
  const noop=!act||act==='NO_OP';
  $('#fbadge').innerHTML=(noop?'<span class="badge noop">NO_OP</span>'
      : `<span class="badge act">ACTION${f.n_act>1?' ×'+f.n_act:''}</span>`)
    + (f.is_black?' <span class="badge black">BLACK</span>':'');
  $('#rawaction').textContent=dispAction(f)||'NO_OP';
  if(HAS_ACTIONS){
    const st=stateAt(cur);
    lightKeyboard(st.pressed, st.held); updateRadar(p, st.pressed, st.held); renderTyped(st.chars, cur);
  } else {
    lightKeyboard(EMPTYSET, EMPTYSET); updateRadar(p, EMPTYSET, EMPTYSET);
    $('#typed').innerHTML='<span class="empty">no actions — run 01b, or link a keylog via --clips-manifest</span>';
  }
  if(_lastCell) _lastCell.classList.remove('cur');
  _lastCell=$('#strip').children[cur];
  if(_lastCell){ _lastCell.classList.add('cur'); centerX($('#strip'), _lastCell); }
  if(_lastCell2) _lastCell2.classList.remove('cur');
  if(ALN){ _lastCell2=$('#strip2').children[cur];
    if(_lastCell2){ _lastCell2.classList.add('cur'); centerX($('#strip2'), _lastCell2); } }
  if(EV){                       // highlight the events inside this frame's time window
    const t0=f.t, t1=t0+1/MASTER_FPS, lo=lb(EVT,t0), hi=lb(EVT,t1);
    if(lo!==_evLo || hi!==_evHi){
      for(let k=_evLo;k>=0 && k<_evHi;k++){ const el=document.getElementById('ev'+k); if(el) el.classList.remove('now'); }
      for(let k=lo;k<hi;k++){ const el=document.getElementById('ev'+k); if(el) el.classList.add('now'); }
      pageY($('#rows'), document.getElementById('ev'+Math.min(lo,EV.length-1)));
      _evLo=lo; _evHi=hi;
    }
  } else {
    if(_lastRow) _lastRow.classList.remove('cur');
    _lastRow=document.getElementById('row'+cur);
    if(_lastRow){ _lastRow.classList.add('cur'); pageY($('#rows'), _lastRow); }
  }
  updateMatchInfo();
}
function step(d){ show(cur+d); }
// Jump to the prev/next (near-)black frame. Both modes carry is_black (master:
// from its own luma metrics; 01b sample: cross-referenced from the master).
function nextBlack(d){ let i=cur+d; while(i>=0&&i<FR.length){ if(FR[i].is_black){show(i);return;} i+=d; } }
// Prev/next black frame that ALSO has an action (is_noop matches the strip's
// action marking — raw binning). No-op when nothing black carries an action.
function nextBlackAct(d){ let i=cur+d; while(i>=0&&i<FR.length){ if(FR[i].is_black&&!FR[i].is_noop){show(i);return;} i+=d; } }
function togglePlay(){
  playing=!playing;
  $('#play').textContent=playing?'⏸ pause':'▶ play';
  $('#play').classList.toggle('on',playing);
  if(playing) timer=setInterval(()=>{ if(cur>=FR.length-1){togglePlay();return;} step(1); },150);
  else clearInterval(timer);
}

$('#ds').onchange=e=>{ DS=e.target.value; loadSegments(); };
$('#seg').onchange=e=>loadSegment(e.target.value);
// number inputs fire `change` on blur/Enter — a rebuild is too expensive to run
// on every keystroke
$('#smode').onchange=onSamplingChange;
$('#sn').onchange=onSamplingChange;
$('#sseed').onchange=onSamplingChange;
for(const id of ['#sn','#sseed'])
  $(id).addEventListener('keydown', e=>{ if(e.key==='Enter') onSamplingChange(); });
// Move to the prev/next segment in the (filtered, sorted) dropdown; clamps at the ends.
function stepSegment(delta){
  const sel=$('#seg'); if(!sel || !sel.options.length) return;
  const i=Math.max(0, Math.min(sel.options.length-1, (sel.selectedIndex<0?0:sel.selectedIndex)+delta));
  if(i===sel.selectedIndex) return;
  sel.selectedIndex=i; loadSegment(sel.value);
}
$('#prev').onclick=()=>step(-1);
$('#next').onclick=()=>step(1);
$('#prevmatch').onclick=()=>nextMatch(-1);
$('#nextmatch').onclick=()=>nextMatch(1);
$('#play').onclick=togglePlay;
$('#markgood').onclick=toggleMark;
$('#marksid').onchange=onMarksIdChange;
$('#marksid').addEventListener('keydown', e=>{ if(e.key==='Enter') onMarksIdChange(); });
$('#strip').onclick=e=>{ const c=e.target.closest('.cell'); if(c) show(+c.dataset.i); };
$('#strip2').onclick=e=>{ const c=e.target.closest('.cell'); if(c) show(+c.dataset.i); };
$('#clocktoggle').onclick=()=>{
  CLOCK = (CLOCK==='aligned') ? 'raw' : 'aligned';
  const tg=$('#clocktoggle'); tg.textContent='keys: '+CLOCK; tg.classList.toggle('on', CLOCK==='aligned');
  PARSED=curActions().map(parseAction); show(cur);
};
$('#rows').onclick=e=>{ const er=e.target.closest('.erow'); if(er){ show(+er.dataset.fi); return; } const r=e.target.closest('.row'); if(r) show(+r.dataset.i); };
// Full-chat window: the button lives inside #modenote (rebuilt per segment via
// innerHTML), so bind it by delegation; ✕ / Esc collapse the window.
$('#modenote').addEventListener('click', e=>{ if(e.target.closest('#chatbtn')) toggleChat(); });
$('#chatclose').onclick=closeChat;
// --- filter panel toggle / reset ---
function setFilters(on){ $('#filters').classList.toggle('show',on); $('#filttoggle').classList.toggle('on',on);
  localStorage.setItem('fr_filters_open', on?'1':''); }
$('#filttoggle').onclick=()=>setFilters(!$('#filters').classList.contains('show'));
$('#fclear').onclick=()=>{ $('#filters').querySelectorAll('input').forEach(i=>i.value=''); const sk=$('#fsortkey'); if(sk) sk.value='';
  clearTimeout(_actTimer); ACTQ=''; ACTHITS=null; actStat(''); refreshMatches();
  if(FR.length){ buildStrip(); buildRows(); }
  applyFilters(); };
document.addEventListener('keydown',e=>{
  if(e.key==='Escape' && $('#chatwin').classList.contains('show')){ closeChat(); e.preventDefault(); return; }
  if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT') return;
  if(e.key==='c'){ toggleChat(); return; }
  if(e.key==='f'){ setFilters(!$('#filters').classList.contains('show')); return; }
  if(e.key==='m'){ toggleMark(); return; }
  // wasd mirrors the arrows: w/s step segments (up/down), a/d step frames (left/right).
  if(e.key==='ArrowUp'||e.key==='w'){stepSegment(-1);e.preventDefault();}
  else if(e.key==='ArrowDown'||e.key==='s'){stepSegment(1);e.preventDefault();}
  else if(e.key==='ArrowLeft'||e.key==='a'){step(-1);e.preventDefault();}
  else if(e.key==='ArrowRight'||e.key==='d'){step(1);e.preventDefault();}
  else if(e.key===' '){togglePlay();e.preventDefault();}
  else if(e.key==='n'){nextMatch(1);}
  else if(e.key==='N'){nextMatch(-1);}   // shift+n
  else if(e.key===','){nextBlack(-1);}
  else if(e.key==='.'){nextBlack(1);}
  else if(e.key==='<'){nextBlackAct(-1);}   // shift+,
  else if(e.key==='>'){nextBlackAct(1);}    // shift+.
  else if(e.key==='Home'){show(0);}
  else if(e.key==='End'){show(FR.length-1);}
});
// --- resizable sidebar + keyboard autofit ---
const panel=$('#panel'), resizer=$('#resizer');
function setPanelWidth(w){ w=Math.max(280, Math.min(window.innerWidth-320, w)); panel.style.width=w+'px'; }
function fitKeyboard(){
  const wrap=$('#kbwrap'); if(!wrap) return;
  const w=wrap.clientWidth||400;
  const u=Math.max(15, Math.min(34, Math.floor((w-20)/16.8)));  // widest row ~16.8 units
  if(u!==UNIT){ UNIT=u; buildKeyboard(); if(FR.length) show(cur); }
}
let rdrag=false;
resizer.addEventListener('mousedown', e=>{ rdrag=true; resizer.classList.add('drag');
  document.body.style.userSelect='none'; e.preventDefault(); });
document.addEventListener('mousemove', e=>{ if(rdrag) setPanelWidth(window.innerWidth - e.clientX); });
document.addEventListener('mouseup', ()=>{ if(!rdrag) return; rdrag=false; resizer.classList.remove('drag');
  document.body.style.userSelect=''; localStorage.setItem('fr_panelw', parseInt(panel.style.width)||430); fitKeyboard(); });
window.addEventListener('resize', ()=>{ setPanelWidth(parseInt(panel.style.width)||430); fitKeyboard(); });
(function(){ const s=parseInt(localStorage.getItem('fr_panelw')); if(s) setPanelWidth(s); })();
if(localStorage.getItem('fr_filters_open')) setFilters(true);

buildKeyboard(); initRadar();
setTimeout(fitKeyboard, 0);
loadDatasets().catch(err=>{ document.body.innerHTML='<pre style="padding:20px;color:#ff9db0">'+err+'</pre>'; });
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
def resolve_jsonl_paths(dataset: Path) -> list[Path]:
    """Find the ``frame_records.jsonl`` file(s) under ``dataset``.

    Handles both shapes:
      * a single combined file (stage-01 style: ``<dir>/frame_records.jsonl``),
      * the stage-01b annotate layout — one file per segment at
        ``<dir>/clips/<segment_id>/stage_01/frame_records.jsonl`` (a
        ``run_dataset --phase annotate`` drop-in).
    A path pointing straight at a ``frame_records.jsonl`` is used as-is.
    """
    dataset = dataset.expanduser().resolve()
    if dataset.is_file():
        return [dataset]
    if not dataset.is_dir():
        raise SystemExit(f"--dataset not found: {dataset}")
    # A single combined file at the root wins if present.
    root = dataset / "frame_records.jsonl"
    if root.exists():
        return [root]
    # Otherwise gather per-segment files: shallow layout, then the 01b annotate
    # layout, then a bounded recursive sweep. First non-empty match wins.
    for pattern in (
        "*/frame_records.jsonl",
        "clips/*/stage_01/frame_records.jsonl",
        "**/frame_records.jsonl",
    ):
        found = sorted(dataset.glob(pattern))
        if found:
            return found
    raise SystemExit(
        f"no frame_records.jsonl found under {dataset} "
        f"(looked at the root, */, clips/*/stage_01/, and recursively); "
        f"pass the file directly with --dataset <path>/frame_records.jsonl"
    )


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
        help="one or more datasets, each a 01b output dir (or frame_records.jsonl "
             "file), a 01a frames-master store dir (segment_index.jsonl + frames/), "
             "a stage-04 conversations dir (or conversations.jsonl file), or a "
             "stage-06 inline-records dir (train/ + val/ ArrayRecord shards) — "
             "auto-detected; choose between them in the UI",
    )
    p.add_argument(
        "--limit", "--limit-samples", dest="limit", type=_positive_int, default=100,
        help="load at most K samples per dataset (default 100 — a store of any size "
             "opens fast; raise it here or in the UI, blank N there = every sample). "
             "For frame_records a sample is a segment_id, for conversations/inline "
             "records a row/chunk, for frames-master a listed segment. WHICH K is "
             "--sample-mode; both are switchable per dataset in the UI ('samples' in "
             "the header).",
    )
    p.add_argument(
        "--sample-mode", choices=("first", "random"), default="first",
        help="which --limit samples to load: the 'first' K in store order (default, "
             "stops reading at K+1) or 'random' K drawn with --seed. Sets the UI's "
             "initial state only — toggle it there without a restart.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="seed for --sample-mode random (default 0). The same seed + N on the "
             "same store always draws the same K samples.",
    )
    p.add_argument(
        "--clips-manifest", default=None,
        help="fallback stage-00 clips_manifest.jsonl (segment_id -> keylog_path) for "
             "overlaying raw actions on frames-master datasets whose manifest.json "
             "doesn't already record source_clips_manifest",
    )
    p.add_argument(
        "--alignment", default=None,
        help="stage-00 alignment.jsonl (or the realign clip-manifest dir holding it) "
             "for the master dual-clock event table + 'aligned + trims' timeline; "
             "auto-discovered from a *realign* sibling of the clips_manifest if omitted",
    )
    p.add_argument("--port", type=int, default=8770, help="HTTP port (default 8770)")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    return p.parse_args()


def looks_like_frames_master(root: Path) -> bool:
    """A stage-01a store: segment_index.jsonl + frames/ (or a frames_master marker)."""
    root = root.expanduser()
    if not root.is_dir():
        return False
    if (root / "segment_index.jsonl").exists() and (root / "frames").is_dir():
        return True
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            return "frames_master" in json.loads(manifest.read_text()).get("artifact_type", "")
        except (OSError, json.JSONDecodeError):
            return False
    return False


def looks_like_conversations(root: Path) -> bool:
    """A conversations-shaped artifact: our stage-04 ``conversations.jsonl`` OR an
    external goal-SFT store (e.g. hindsight_fold canonical SFT: a ``chat.jsonl`` at
    the root or under ``train``/``val``). Same interleaved screenshot->action
    ``messages`` schema; the goal shows as the per-conversation instruction."""
    root = root.expanduser()
    if root.is_file():
        name = root.name.lower()
        return name.endswith(".jsonl") and ("conversation" in name or "chat" in name)
    if not root.is_dir():
        return False
    if any((root / p).exists() for p in (
        "conversations.jsonl", "chat.jsonl", "val/chat.jsonl", "train/chat.jsonl"
    )):
        return True
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            at = json.loads(manifest.read_text()).get("artifact_type", "")
            return "conversations" in at or "canonical_sft" in at
        except (OSError, json.JSONDecodeError):
            return False
    return False


def looks_like_inline_records(root: Path) -> bool:
    """A stage-06 inline SFT records store: a dir whose ``manifest.json`` marks the
    stage ``inline_records``, or one with ``train``/``val`` split subdirs (or a split
    dir itself) holding a ``metadata.json`` with ``inline_records: true``."""
    root = root.expanduser()
    if not root.is_dir():
        return False
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            if json.loads(manifest.read_text()).get("stage") == "inline_records":
                return True
        except (OSError, json.JSONDecodeError):
            pass
    for cand in (root, root / "train", root / "val"):
        meta = cand / "metadata.json"
        if meta.exists():
            try:
                if json.loads(meta.read_text()).get("inline_records"):
                    return True
            except (OSError, json.JSONDecodeError):
                pass
    return False


def resolve_conversations_path(dataset: Path) -> Path:
    """Locate the conversations file for a dataset: the file itself, or -- for a dir
    -- ``conversations.jsonl`` (our stage-04), then ``chat.jsonl`` / ``val/chat.jsonl``
    / ``train/chat.jsonl`` (external goal-SFT). A big combined ``chat.jsonl`` loads
    eagerly, so pass a split file directly (e.g. ``.../val/chat.jsonl``) to browse fast."""
    dataset = dataset.expanduser().resolve()
    if dataset.is_file():
        return dataset
    for candidate in (
        dataset / "conversations.jsonl",
        dataset / "chat.jsonl",
        dataset / "val" / "chat.jsonl",
        dataset / "train" / "chat.jsonl",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit(f"no conversations.jsonl / chat.jsonl found under {dataset}")


def detect_mode(path: Path) -> str:
    """Cheap mode detection shared by registration and building (no full load).
    Frames-master, conversations and inline-records are checked before the
    frame_records default."""
    if looks_like_frames_master(path):
        return "frames_master"
    if looks_like_conversations(path):
        return "conversations"
    if looks_like_inline_records(path):
        return "inline_records"
    return "frame_records"


def _build_dataset(path: Path, sampling: Sampling):
    """Build the right dataset object for a path (frames-master / conversations /
    inline-records / frame_records), loading the samples ``sampling`` selects."""
    mode = detect_mode(path)
    if mode == "frames_master":
        return FramesMasterDataset(path.expanduser().resolve(), sampling)
    if mode == "conversations":
        return ConversationsDataset(resolve_conversations_path(path), sampling)
    if mode == "inline_records":
        return InlineRecordsDataset(path.expanduser().resolve(), sampling)
    return FrameRecordsDataset(resolve_jsonl_paths(path), sampling)


def get_dataset(name: str, sampling: "Sampling | None" = None):
    """Return the built dataset for a registered name + sampling (build + cache on
    first use).

    One name can be built under several samplings — first N, random N seed 0,
    random N seed 1 — so switching in the UI doesn't re-read a store you already
    looked at; the per-name cache keeps the last ``_SAMPLING_CACHE_CAP`` builds and
    drops the least-recently-used (an evicted sampling just costs a rebuild).

    Build failures (e.g. an empty / not-yet-generated dataset dir) are re-raised as
    RuntimeError so the request handler reports them as JSON instead of a bare
    SystemExit tearing down the handler thread. The failure isn't cached, so
    re-selecting the dataset retries the build."""
    entry = DATASETS.get(name)
    if entry is None:
        return None
    samp = sampling or Sampling(DATASET_SAMPLE_MODE, DATASET_SAMPLE_LIMIT, DATASET_SAMPLE_SEED)
    objs: "OrderedDict[str, Any]" = entry["objs"]
    key = samp.key()
    obj = objs.get(key)
    if obj is None:
        try:
            obj = _build_dataset(entry["path"], samp)
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
        objs[key] = obj
        while len(objs) > _SAMPLING_CACHE_CAP:
            objs.popitem(last=False)
    else:
        objs.move_to_end(key)
    return obj


_MARKS_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_marks_id(raw: str) -> str:
    """A client-chosen marks-file identifier, defanged to a safe filename
    fragment (it lands directly in a path segment): anything outside
    ``[A-Za-z0-9._-]`` collapses to ``_``, capped at 64 chars. Blank stays blank
    -- that's the "no id" case, i.e. the dataset's shared ``golden_marks.json``."""
    return _MARKS_ID_RE.sub("_", (raw or "").strip())[:64]


def _marks_path(name: str, marks_id: str = "") -> Path:
    """Where a dataset's golden-trace marks live: ``golden_marks.json`` next to the
    dataset root (or next to the file itself, if ``--dataset`` pointed at a file
    directly) — so the marks travel with the dataset rather than piling up
    somewhere central. A non-blank ``marks_id`` gets its own sibling file
    (``golden_marks_<id>.json``) instead of the shared default, so independent
    review passes / sessions never step on each other's marks."""
    root = DATASETS[name]["path"].expanduser()
    root = root if root.is_dir() else root.parent
    mid = _sanitize_marks_id(marks_id)
    return root / (f"golden_marks_{mid}.json" if mid else "golden_marks.json")


def _load_marks(name: str, marks_id: str = "") -> dict[str, Any]:
    """The mark dict for ``name``/``marks_id`` (segment_id -> {"ts": iso str}), read
    fresh from disk on every call — no in-memory cache. Several server processes
    (or browser tabs against the same dataset+marks_id) can be open at once; a
    cached copy would go stale the moment ANOTHER one writes, and the next save
    from here would blindly overwrite it and lose those marks. Re-reading is cheap
    (the file is a small dict), so there's no reason to risk that. A
    missing/corrupt file just means no marks yet."""
    path = _marks_path(name, marks_id)
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text()).get("marks") or {})
    except (OSError, json.JSONDecodeError):
        return {}


def _save_marks(name: str, marks: dict[str, Any], marks_id: str = "") -> None:
    """Persist ``marks`` for ``name``/``marks_id``, atomically (write to a temp file
    then rename) so a crash mid-write can't corrupt the marks file. Callers must
    have just re-read the current on-disk state via ``_load_marks`` and applied
    only their own change to it, so a concurrent writer's marks survive."""
    path = _marks_path(name, marks_id)
    payload = {
        "dataset": name,
        "dataset_path": str(DATASETS[name]["path"]),
        "marks_id": marks_id or None,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "marks": marks,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def register_datasets(paths: list[str]) -> None:
    """Register each path under its basename (disambiguating collisions); detect the
    mode cheaply (no full load) so the UI can label master vs sample up front."""
    for raw in paths:
        p = Path(raw).expanduser()
        name = p.name or str(p)
        base, k = name, 2
        while name in DATASETS:
            name, k = f"{base}#{k}", k + 1
        DATASETS[name] = {"path": p, "mode": detect_mode(p), "objs": OrderedDict()}


def main() -> None:
    global CLIPS_MANIFEST_OVERRIDE, ALIGNMENT_OVERRIDE, DATASET_SAMPLE_LIMIT
    global DATASET_SAMPLE_MODE, DATASET_SAMPLE_SEED
    args = parse_args()
    CLIPS_MANIFEST_OVERRIDE = args.clips_manifest
    ALIGNMENT_OVERRIDE = args.alignment
    DATASET_SAMPLE_LIMIT = args.limit
    DATASET_SAMPLE_MODE = args.sample_mode
    DATASET_SAMPLE_SEED = args.seed
    register_datasets(args.dataset)
    if not DATASETS:
        raise SystemExit("no datasets given")
    print(f"registered {len(DATASETS)} dataset(s):", flush=True)
    for name, entry in DATASETS.items():
        print(f"  {name}  [{entry['mode']}]  {entry['path']}", flush=True)
    if DATASET_SAMPLE_LIMIT is None:
        print("sample limit: none (every sample per dataset)", flush=True)
    elif DATASET_SAMPLE_MODE == "random":
        print(
            f"sample limit: {DATASET_SAMPLE_LIMIT} random samples per dataset "
            f"(seed {DATASET_SAMPLE_SEED}) — switch mode/N/seed in the UI",
            flush=True,
        )
    else:
        print(
            f"sample limit: first {DATASET_SAMPLE_LIMIT} samples per dataset "
            "— switch to random N (seeded) in the UI",
            flush=True,
        )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"serving on http://{args.host}:{args.port}/  "
        f"(datasets build on first selection; Ctrl-C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
