#!/usr/bin/env python3
"""Stage 04 (conversations): turn a stage-03 frame-sampling dataset into a
training ``conversations.jsonl`` -- NO VLM annotation, NO re-decoding.

Stage 03 (``sample_frames_actions``) emitted, per segment, a
``frame_records.jsonl`` of ordered ``(image_path, action)`` rows (frames already
fps-sampled, NO_OP-thinned, and black-filtered). This stage assembles each
segment into ONE interleaved screenshot->action conversation:

    [ system,
      user(screenshot_0 [+ instruction]), assistant(action_0),
      user(screenshot_1),                 assistant(action_1),
      ... ,
      user(screenshot_N),                 assistant(action_N) ]

i.e. the user shows a screenshot, the model replies with the action taken from
it, and that action is what produced the NEXT screenshot -- exactly the shape the
eval-side OSWorld runtime prompts with, but materialized for SFT. ``action_i`` is
the raw recorded action string from the frame record (``"<dx> <dy> <scroll>"``
optionally ``" ; +KEY -KEY"``, or ``"NO_OP"``); the image is the ``ar://`` grain
ref, passed through verbatim (already portable/absolute).

The message/content schema matches the annotation pipeline's canonical
``chat.jsonl`` (``stage_04_build_canonical_sft``): content is a list of
``{"type":"image","image":...}`` / ``{"type":"text","text":...}`` blocks, and on
the first user turn the instruction TEXT precedes the image. Unlike that builder,
this one does NOT require an instruction -- goal-free (system-prompt-only) is the
default -- so it runs straight off the sampled dataset with no labeling step.

Instruction (first user turn) is configurable:
  * default: goal-free (image only).
  * ``--instruction TEXT``: a fixed instruction on every segment's first turn.
  * ``--instruction-field KEY``: a PER-SEGMENT instruction read from KEY on the
    sample_index row (falling back to the first frame record), for when goals are
    joined in upstream (e.g. OSWorld task text). Falls back to --instruction, then
    goal-free, when the field is absent/empty.

GOAL-CONDITIONED mode (``--goal-index``): instead of one conversation per segment,
build one conversation PER GOAL from a stage-03b ``goal_frame_index.jsonl``. Each
goal's OUR frames are its ``--sample-dir`` frames windowed to the goal's
source-frame span ``[coll_source_frame_idx_lo, coll_source_frame_idx_hi]``, with
the goal ``instruction`` on the first user turn. When the goal index carries a
``context`` (a self-compaction ``[CONTEXT]`` rolling summary, present on non-first
chunks of a split goal), it is fused after the instruction on that first turn -- so a
resumed chunk sees its progress summary, the format the selfcompact set was built
with. The span is colleague-derived and
fps-independent, so ONE goal index goal-conditions ANY ``--sample-dir`` fps. Goals
whose window is all-idle (dropped by ``noop_mode=none``) fall below ``--min-frames``
and are skipped. Without ``--goal-index`` the per-segment behavior below is unchanged.
Optionally (``--terminate-token TERMINATE``) the final assistant turn's action is
overwritten with a terminate token, marking goal completion at the window's end (the
token the eval side's ``freeroll._is_terminate`` recognizes). This applies in
goal-conditioned mode ONLY -- a per-segment end is not a task completion -- and pairs
with a system prompt that describes the contract (e.g. ``--system-prompt-id yll_v1``).

ACTION FORMAT (``--action-format``): by default (``sampled``) the assistant turns
carry the stage-03 frame records' ``action`` strings verbatim (the canonical
``"<dx> <dy> <scroll> [; +KEY -KEY]"`` aggregate). Any registered formatter name
(``lib/action_format.FORMATTERS``: ``canonical``, ``ordered_events_v2``,
``ordered_events_v3``, ``computer_use_rel_v1``) instead RE-DERIVES every label
from the segment's REALIGNED keylog at build time, so the action format is a
stage-04 ablation flag, not a pipeline rerun. The re-derivation reconstructs the
lib/events view from the sample artifact: the kept frames' ``master_record_index``
ticks become label windows ``[tick_i, tick_{i+1})`` (the last runs to the end of
master coverage), and dead zones are the pre-first-frame span plus every
(near-)black master tick (from the master store's ``frame_manifest.jsonl``, using
the thresholds the sample was built with -- only when it was built with
``drop_black_frames``). Labels then follow the shared dead-zone policy
(``lib/events.apply_label_policy``): deltas in dead zones are discarded, straddling
press/release pairs are clamped to zone boundaries. NOTE this differs near black
gaps from the ``sampled`` strings (stage 03 dropped whole black bins); ``canonical``
via the formatter is byte-identical to ``sampled`` only on dead-zone-free
stretches. Per-segment dead-zone counters land on every row
(``dead_zone_counters`` / ``dead_zone_flagged``) as a realignment health signal.

TYPING COALESCING (``--coalesce-typing``, ``ordered_events_v3`` only): v3 collapses
typing into ``type("...")`` PER FRAME; this fuses it ACROSS frames. A maximal run of
typing-only frames becomes ONE turn -- the run's first frame (its screenshot) carrying
the whole run's text as a single ``type()`` -- and the run's other frames are DROPPED
from the conversation. This is the only place in the pipeline that drops frames after
stage 03; the stage-03 artifact is untouched (dropped frames' images simply stop being
referenced), so it is an ablation flag, not a re-sample.

Implementation: labels are derived TWICE from the one keylog/manifest read. Pass 1
labels the per-frame windows and only CLASSIFIES them (``lib/action_format
.plan_typing_coalesce``); pass 2 re-derives every label over the MERGED windows, which
is what yields one ``type("abcd")`` instead of ``type("ab"); type("cd")`` and what
repairs typing no single window could balance (key rollover across a frame boundary,
a Shift held across frames). What breaks a run: any non-typing primitive (mouse,
scroll, Return/Backspace/Tab/arrows -- backtracking is supervision worth keeping per
frame -- chords, non-Shift modifiers), a trailing idle frame (idle frames INSIDE a run
are absorbed), a dead zone in the gap between two frames (its keystrokes are discarded
by the label policy, so a merged label would have a silent hole), a goal-window
boundary (``--goal-index``), and ``--max-coalesce-frames`` (the staleness bound: the
turn shows the run's first screenshot). A run never ends a conversation: on reaching a
goal's last frame or the segment's last frame, that frame is kept as a trailing
``NO_OP`` turn (so ``--terminate-token`` overwrites an idle turn, never a merged typing
run). ``--min-frames`` gates on the PRE-coalesce frame count. Provenance:
``n_coalesced_turns`` / ``n_frames_pre_coalesce`` / ``n_frames_coalesced_away`` per row,
totals in the summary.

One conversation per segment (no windowing): a long, high-fps segment becomes a
long conversation -- watch the trainee's context window at high --target-fps.

APPLICATION FILTER (``--include-app`` / ``--exclude-app`` / ``--split-by-app``):
selects conversations by the FOREGROUND APP, read from the per-frame ``app``
labels stage 03 writes (the recorder's ``ContextChanged`` events off the same
realigned keylog the actions come from -- exact join, not a heuristic; back-filled
here when the sample predates the labels). Two modes, because a segment is not
app-homogeneous (only ~31% of ccast0618d segments touch a single app):
``--include-app firefox`` GATES whole segments on their dominant app (optionally
with a purity floor, ``--app-min-frac``), keeping trajectories intact;
``--split-by-app`` instead cuts each segment into ONE CONVERSATION PER maximal
same-app run, which is how you get pure per-app data. Runs are labeled before any
cut, so every surviving turn's action string is byte-identical to the unfiltered
build. ``--app-drop-seam-turns`` (default on) drops the boundary turn whose action
window straddles the switch -- the same argument as the black-frame dead zones:
one label must not aggregate input from two applications.

The train/val split is NOT applied here: this stage emits a single
split-agnostic ``chat.jsonl`` and the recording-level split is deferred to the
records stage (stage 06, via ``--val_fraction``). That keeps this stage -- and
the measure cache (stage 05) -- independent of the split, so changing the val
fraction re-runs only stage 06 and never re-tokenizes.

Input  (--sample-dir): a stage-03 output (``sample_index.jsonl`` +
        ``clips/<seg>/stage_01/frame_records.jsonl``).
Output (--output-dir):
  conversations.jsonl          one row per segment: {messages, + provenance}.
  chat.jsonl                   the canonical layout (same rows, same schema as
                               conversations.jsonl) -- a single split-agnostic
                               drop-in source_path for the measure/records stages
                               (stage 05 reads <source>/chat.jsonl, stage 06 reads
                               it and applies the split). Carries recording_id per
                               row so the downstream split can group by recording.
  conversations_summary.json   aggregate stats.
  manifest.json                artifact marker.

Run::

    cd data_pipeline
    uv run python realigned_pipeline/stage_04_build_conversations.py \
        --sample-dir  <stage-03 --output-dir> \
        --output-dir  <dest> \
        [--instruction "..."]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Make the ``realigned_pipeline`` package importable when run directly
# (mirrors build_frames_master.py / sample_frames_actions.py).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib import config  # noqa: E402
from realigned_pipeline.lib.action_format import (  # noqa: E402
    DEFAULT_CONTINUOUS_ACTION_HZ,
    FORMATTERS,
    ORDERED_IDLE_LABEL,
    TYPING_COALESCE_FORMATS,
    ActionFormatter,
    get_formatter,
    plan_typing_coalesce,
)
from realigned_pipeline.lib.app_context import (  # noqa: E402
    UNRESOLVED_APPS,
    frame_app_stats,
    iter_app_spans,
    load_app_track,
    resolve_app_selector,
)
from realigned_pipeline.lib.common import ensure_dir, read_jsonl, write_json, write_jsonl  # noqa: E402
from realigned_pipeline.lib.events import DeadZone, Window, load_events  # noqa: E402
from realigned_pipeline.lib.image_store import (  # noqa: E402
    is_arrayrecord_image_uri,
    parse_arrayrecord_image_uri,
)

# Named system prompts are shared with the eval side (single source of truth):
# the OSWorld runners select from this same SYSTEM_PROMPTS dict by id, so a model
# can be trained and evaluated under an identical system message. ``eval/`` is a
# sibling of ``data_pipeline/`` (repo root) with no package init, so add it to
# sys.path and import the module directly -- exactly as the eval runners do. The
# module is dependency-free (just the dict), so this is cheap and safe.
EVAL_DIR = DATA_PIPELINE_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402

# Stage-03 statuses that carry a usable frame_records.jsonl.
USABLE_STATUSES = {"ok", "cached"}

# --action-format value meaning "no formatter": pass the stage-03 frame records'
# canonical action strings through verbatim (the historical behavior).
SAMPLED_FORMAT = "sampled"

# Default system prompts (``sampled`` mode). Goal-free reuses the verbatim
# training-time prompt ("training_v1") from the shared eval dict; goal-conditioned
# reuses the canonical one (it names a goal) but keeps the action-format contract.
GOAL_FREE_SYSTEM_PROMPT = SYSTEM_PROMPTS["yll_v1"]
GOAL_SYSTEM_PROMPT = config.SYSTEM_PROMPT

# Formatter-mode default system prompts: a fixed framing prefix + the formatter's
# own reply contract, so the default prompt always describes the SELECTED action
# format (ported from main's injection-point stage 04; for ``canonical`` the
# composition is byte-identical to the historical prompts -- the regression gate
# in tests/test_action_format.py).
GOAL_FREE_PROMPT_PREFIX = (
    "You operate a desktop computer. Each user turn shows the current screen. "
)
GOAL_PROMPT_PREFIX = (
    "You operate a desktop computer. The first user turn shows the initial "
    "screen and the user's goal; subsequent user turns show the current screen. "
)


def default_system_prompt(formatter: ActionFormatter, *, goal_conditioned: bool) -> str:
    if goal_conditioned:
        return GOAL_PROMPT_PREFIX + formatter.reply_contract.format(
            what="the next action toward that goal"
        )
    return GOAL_FREE_PROMPT_PREFIX + formatter.reply_contract.format(
        what="the next action"
    )


def _str2bool(s: str | bool) -> bool:
    """Parse a boolean CLI value. Accepts the labctl ``--flag=value`` form (labctl
    renders every arg as ``--key=value``, so a valueless flag can't be expressed);
    truthy = 1/true/yes/on, everything else False. Same helper as stage 03."""
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(image: str) -> dict[str, Any]:
    return {"type": "image", "image": image}


def _join_instruction_context(instruction: str | None, context: str | None) -> str | None:
    """Fuse the goal instruction with its self-compaction ``[CONTEXT]`` block into a
    single first-turn text, reproducing the external (selfcompact) builder's layout
    ``"<instruction>\n\n[CONTEXT]…[/CONTEXT]"``. ``context`` is present only on
    non-first chunks of a split goal (carried through stage-03b's goal index); either
    argument may be absent (goal-free -> None; first/single chunk -> instruction only)."""
    parts = [p.strip() for p in (instruction, context) if p and str(p).strip()]
    return "\n\n".join(parts) if parts else None


def build_messages(
    frames: list[dict[str, Any]],
    *,
    instruction: str | None,
    system_prompt: str | None,
    context: str | None = None,
    terminate_token: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the interleaved conversation for one segment. Matches the canonical
    chat.jsonl schema: instruction TEXT before the image on the first user turn,
    image-only on later turns, one assistant turn per frame carrying its action.

    ``context`` (goal-conditioned self-compaction only): the goal's ``[CONTEXT]`` block,
    fused after the instruction on the first user turn (see ``_join_instruction_context``)
    so a non-first chunk resumes from its rolling progress summary -- the format the
    selfcompact set was built with. None for goal-free / first-or-single chunks.

    ``terminate_token`` (goal-conditioned mode only): OVERWRITE the final assistant
    turn's action with this token (e.g. ``"TERMINATE"``), marking goal completion at
    the window's end -- the eval side (``freeroll._is_terminate``) treats the first
    stripped line ``== "TERMINATE"`` as end-of-episode. The last frame's real action
    label is dropped in exchange (the yll pilot's convention)."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [_text_block(system_prompt)]})
    first_text = _join_instruction_context(instruction, context)
    for idx, frame in enumerate(frames):
        content: list[dict[str, Any]] = []
        if idx == 0 and first_text:
            content.append(_text_block(first_text))
        content.append(_image_block(str(frame["image_path"])))
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [_text_block(str(frame["action"]))]})
    if terminate_token and messages and messages[-1]["role"] == "assistant":
        messages[-1]["content"] = [_text_block(terminate_token)]
    return messages


def _load_segment_frames(index_row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """One segment's kept frame records, in conversation order (or None when the
    stage-03 artifact has no records for it)."""
    fr_path = index_row.get("frame_records")
    if not fr_path or not Path(fr_path).exists():
        return None
    frames = read_jsonl(Path(fr_path))
    frames.sort(key=lambda r: int(r.get("global_frame_idx") or 0))
    return frames


def _is_black_record(mrec: dict[str, Any], luma_max: float, dark_frac_min: float) -> bool:
    """Same predicate as stage 03: (near-)black per the stage-01a luma metrics;
    records without metrics are never black (absence of evidence isn't blackness)."""
    ml, fd = mrec.get("mean_luma"), mrec.get("frac_dark")
    return (ml is not None and ml <= luma_max) or (fd is not None and fd >= dark_frac_min)


def _load_sample_config(sample_dir: Path) -> dict[str, Any]:
    """The stage-03 artifact's build provenance (black-filter flag + thresholds,
    frames_master_dir, master_fps) -- what the formatter path needs to reconstruct
    dead zones exactly as the sample was built."""
    for name in ("sample_summary.json", "manifest.json"):
        path = sample_dir / name
        if path.is_file():
            return json.loads(path.read_text())
    raise SystemExit(
        f"no sample_summary.json/manifest.json under {sample_dir} -- "
        "--action-format needs the sample artifact's build provenance"
    )


def _master_manifest_path(
    frames: list[dict[str, Any]], sample_cfg: dict[str, Any], segment_id: str
) -> Path:
    """Locate the segment's master ``frame_manifest.jsonl``: prefer the shard the
    frames actually reference (``ar://<shard>#idx`` -- robust to a moved store),
    falling back to the artifact's recorded ``frames_master_dir`` layout."""
    for f in frames:
        ref = str(f.get("image_path") or "")
        if is_arrayrecord_image_uri(ref):
            shard, _ = parse_arrayrecord_image_uri(ref)
            return Path(shard).parent / "frame_manifest.jsonl"
    root = sample_cfg.get("frames_master_dir")
    if root:
        return Path(root) / "frames" / segment_id / "frame_manifest.jsonl"
    raise FileNotFoundError(
        f"{segment_id}: cannot locate the master frame_manifest.jsonl "
        "(no ar:// image refs and no frames_master_dir in the sample summary)"
    )


def _tile_windows(ticks: list[int], axis_end: int) -> list[Window]:
    """Label-ownership windows for ``ticks``: each tick owns up to the next one,
    the last runs to the end of master coverage. Contiguous by construction --
    ``lib/events._Locator`` requires the tiling to have no holes."""
    return [
        Window(t, t, ticks[i + 1] if i + 1 < len(ticks) else axis_end)
        for i, t in enumerate(ticks)
    ]


def _goal_coalesce_bounds(
    frames: list[dict[str, Any]], goal_spans: list[tuple[int, int]]
) -> tuple[set[int], set[int]]:
    """(barrier_start, terminal) FRAME-INDEX sets for a coalesce plan, from the
    goal windows' source-frame spans (stage-03b ``coll_source_frame_idx_lo/hi``).

    A goal's first frame must start a fresh run and its last frame must never be
    a run's interior: coalescing is computed once per SEGMENT (the frame list all
    of its goals slice), so without these a run could straddle a goal boundary --
    moving one goal's opening keystrokes into a turn that goal doesn't contain, or
    leaking post-goal keystrokes into its final turn. Spans may overlap; both sets
    only ever REDUCE merging, so conservatism is safe."""
    barriers: set[int] = set()
    terminals: set[int] = set()
    src = [f.get("source_frame_idx") for f in frames]
    for lo, hi in goal_spans:
        first = next(
            (i for i, s in enumerate(src) if s is not None and lo <= int(s) <= hi), None
        )
        if first is None:  # goal window holds none of our frames
            continue
        last = next(
            i for i in range(len(src) - 1, -1, -1)
            if src[i] is not None and lo <= int(src[i]) <= hi
        )
        barriers.add(first)
        terminals.add(last)
    return barriers, terminals


def _dead_zone_breaks(ticks: list[int], dead_zones: list[DeadZone]) -> set[int]:
    """Frame indices whose gap from the previous frame contains a dead zone.

    Stage 03 drops black frames, so no KEPT frame sits inside a zone -- but two
    consecutive kept frames can straddle one, and the label policy discards or
    clamps the keystrokes in it. Coalescing across that would concatenate text
    with a silent hole in the middle, so it is a hard run breaker."""
    zones = sorted(dead_zones, key=lambda z: z.start)
    breaks: set[int] = set()
    zi = 0
    for i in range(1, len(ticks)):
        prev, cur = ticks[i - 1], ticks[i]
        while zi < len(zones) and zones[zi].end <= prev:
            zi += 1  # ticks ascend, so zones behind us stay behind
        if zi < len(zones) and zones[zi].start < cur:
            breaks.add(i)
    return breaks


def reformat_segment_actions(
    frames: list[dict[str, Any]],
    index_row: dict[str, Any],
    *,
    formatter: ActionFormatter,
    sample_cfg: dict[str, Any],
    dead_zone_flag_frac: float,
    coalesce_typing: bool = False,
    max_coalesce_frames: int = 0,
    goal_spans: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Replace every kept frame's ``action`` IN PLACE with ``formatter``'s label,
    re-derived from the segment's realigned keylog under the shared dead-zone
    policy (see the module docstring). Windows are the kept frames' master ticks
    tiled to the end of master coverage; dead zones are the pre-first-frame span
    plus black master ticks (when the sample was built with drop_black_frames).

    ``coalesce_typing`` additionally DROPS frames from ``frames`` (in place):
    runs of typing-only windows fuse into their first frame's turn, and every
    label is then re-derived over the merged windows -- so the surviving turn
    carries one ``type()`` for the whole run (see ``plan_typing_coalesce``; the
    two passes share the keylog/manifest read, the second is a pure in-memory
    fold). ``goal_spans`` are the segment's goal windows, whose boundaries clamp
    the runs. Coalesced frames gain ``coalesced_n_frames`` provenance.

    Returns segment-level accounting: ``action_format`` / ``dead_zone_counters`` /
    ``dead_zone_flagged`` (row fields) + ``primitive_counts`` (summary totals)."""
    seg = str(index_row.get("segment_id"))
    ticks = [int(f["master_record_index"]) for f in frames]
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        raise ValueError(f"{seg}: master_record_index not strictly increasing")

    manifest_path = _master_manifest_path(frames, sample_cfg, seg)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{seg}: no master frame_manifest at {manifest_path}")
    master_manifest = read_jsonl(manifest_path)
    if not master_manifest:
        raise ValueError(f"{seg}: empty master frame_manifest {manifest_path}")
    n_records = len(master_manifest)

    # Windows tile from each kept tick to the next; the last runs to the end of
    # master coverage (events past it resolve to the implicit no_coverage zone).
    axis_end = max(n_records, ticks[-1] + 1)
    windows = _tile_windows(ticks, axis_end)

    first_tick = ticks[0]
    dead_zones: list[DeadZone] = []
    if first_tick > 0:
        dead_zones.append(DeadZone(0, first_tick, "pre_first_frame"))
    # Black zones only when the sample was built black-filtered (same thresholds),
    # clipped to the visible region -- everything before the first kept frame is
    # already one pre_first_frame zone. Kept ticks are non-black by construction
    # (stage 03 dropped black candidate bins), so no window start is swallowed.
    if sample_cfg.get("drop_black_frames"):
        luma_max = float(sample_cfg.get("black_luma_max", config.DEFAULT_BLACK_LUMA_MAX))
        dark_min = float(sample_cfg.get("black_dark_frac_min", config.DEFAULT_BLACK_DARK_FRAC_MIN))
        run_start: int | None = None
        for tick, mrec in enumerate(master_manifest):
            black = tick >= first_tick and _is_black_record(mrec, luma_max, dark_min)
            if black and run_start is None:
                run_start = tick
            elif not black and run_start is not None:
                dead_zones.append(DeadZone(run_start, tick, "black"))
                run_start = None
        if run_start is not None:
            dead_zones.append(DeadZone(run_start, n_records, "black"))

    keylog = index_row.get("keylog_path")
    if keylog and not Path(keylog).exists():
        # load_events would silently yield no events -> an all-NO_OP segment.
        raise FileNotFoundError(f"{seg}: realigned keylog missing: {keylog}")
    events, _ = load_events(Path(keylog)) if keylog else ([], None)

    master_fps = float(index_row.get("master_fps") or sample_cfg["master_fps"])
    result = formatter.format_segment(events, windows, dead_zones, master_fps=master_fps)

    coalesce_fields: dict[str, Any] = {}
    plan = None
    if coalesce_typing:
        if result.primitives is None:
            raise ValueError(
                f"{seg}: --coalesce-typing needs a primitive-emitting formatter, "
                f"{formatter.name} renders none"
            )
        barriers, terminals = _goal_coalesce_bounds(frames, goal_spans or [])
        terminals.add(len(frames) - 1)  # a segment end is a terminal window too
        plan = plan_typing_coalesce(
            result.primitives,
            barrier_start=barriers,
            terminal=terminals,
            break_before=_dead_zone_breaks(ticks, dead_zones),
            max_frames=max_coalesce_frames,
        )

    if plan is not None and plan.spans:
        # PASS 2: re-derive every label over the MERGED windows. forced-idle
        # frames are left out of the tiling so the run's window extends across
        # them (no keystroke is lost -- the text lands on the run's first turn).
        forced = set(plan.forced_idle)
        win_frames = [i for i in plan.keep if i not in forced]
        result = formatter.format_segment(
            events, _tile_windows([ticks[i] for i in win_frames], axis_end),
            dead_zones, master_fps=master_fps,
        )
        label_of = dict(zip(win_frames, result.labels, strict=True))
        for i in plan.keep:
            f = frames[i]
            f["action"] = ORDERED_IDLE_LABEL if i in forced else label_of[i]
            end = plan.spans.get(i)
            if end is not None:
                f["coalesced_n_frames"] = end - i + 1
                f["coalesced_master_record_index_end"] = ticks[end]
                f["coalesced_source_frame_idx_end"] = frames[end].get("source_frame_idx")
            if i in forced:
                f["coalesce_forced_idle"] = True
        coalesce_fields = {
            "coalesced_turns": len(plan.spans),
            "coalesced_frames_dropped": plan.n_dropped,
            "coalesce_forced_idle_turns": len(plan.forced_idle),
        }
        frames[:] = [frames[i] for i in plan.keep]
    else:
        for f, label in zip(frames, result.labels, strict=True):
            f["action"] = label
        if coalesce_typing:
            coalesce_fields = {
                "coalesced_turns": 0,
                "coalesced_frames_dropped": 0,
                "coalesce_forced_idle_turns": 0,
            }

    counters = result.counters
    n_discarded = (
        counters.n_discarded_black
        + counters.n_discarded_no_coverage
        + counters.n_discarded_pre_first_frame
        + 2 * counters.n_pairs_dropped_dead_zone
        + counters.n_unreleased_press_dropped
    )
    return {
        "action_format": formatter.name,
        "dead_zone_counters": asdict(counters),
        "dead_zone_flagged": len(events) > 0 and (n_discarded / len(events)) > dead_zone_flag_frac,
        **coalesce_fields,
        "primitive_counts": result.primitive_counts,
    }


def _coalesce_counts(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-CONVERSATION coalescing provenance (the segment-level counters can't
    serve goal mode, where one segment's frames feed many goals). Empty when
    nothing in this conversation was coalesced."""
    spans = [int(f["coalesced_n_frames"]) for f in frames if f.get("coalesced_n_frames")]
    if not spans:
        return {}
    # A forced-idle tail is inside its run's span AND kept as its own turn, so
    # it would otherwise be counted twice.
    n_forced = sum(1 for f in frames if f.get("coalesce_forced_idle"))
    n_pre = sum(spans) + sum(1 for f in frames if not f.get("coalesced_n_frames")) - n_forced
    return {
        "n_coalesced_turns": len(spans),
        "n_frames_pre_coalesce": n_pre,
        "n_frames_coalesced_away": n_pre - len(frames),
    }


# --------------------------------------------------------------------------- #
# Application filtering (--include-app / --exclude-app / --split-by-app)
#
# The foreground app is a per-FRAME label written by stage 03 (``app``, plus
# ``app_window_switches`` marking turns whose action window straddles a switch).
# It is read here, never re-derived, EXCEPT when the sample predates the labels:
# then ``_ensure_app_labels`` fills them from the same realigned keylog the
# formatter already uses, so an app-filtered set can be built off an existing
# stage-03 artifact without re-sampling.
#
# Two modes, because a segment is NOT app-homogeneous (measured on
# ccast0618d: only ~31% of segments touch a single app, median 3 same-app runs):
#   * gate  -- keep or drop the WHOLE segment on its dominant app. Trajectories
#              stay intact; a kept conversation still carries the minority apps.
#   * split -- one conversation per maximal same-app run (``--split-by-app``).
#              Pure per-app data at the cost of cutting segments.
#
# Splitting inherits goal mode's windowing caveat: a formatter's cross-turn state
# (a key held across the cut) can leave an ``up(X)`` whose ``down(X)`` sits in
# another conversation. Dropping the seam turn (``--app-drop-seam-turns``, on by
# default) removes the boundary turn itself, whose action window mixes both apps.
# --------------------------------------------------------------------------- #
APP_UNKNOWN_MODES = ("keep", "drop")


@dataclass(frozen=True)
class AppFilter:
    """Resolved ``--include-app``/``--exclude-app``/``--split-by-app`` policy."""

    include: frozenset[str] = frozenset()
    exclude: frozenset[str] = frozenset()
    min_frac: float = 0.0
    unknown: str = "keep"
    split: bool = False
    min_run_frames: int = 1
    drop_seam_turns: bool = True

    @property
    def active(self) -> bool:
        return bool(
            self.include or self.exclude or self.split
            or self.min_frac > 0.0 or self.unknown == "drop"
        )

    def accepts(self, app: str | None, frac: float) -> bool:
        """Does a conversation whose dominant app is ``app`` (holding ``frac`` of its
        labeled frames) survive? An unresolved/absent label can only pass when no
        include list is set and unknowns are kept -- never claim a segment is
        Firefox because nothing said otherwise."""
        if app is None or app in UNRESOLVED_APPS:
            return not self.include and self.unknown == "keep"
        if self.include and app not in self.include:
            return False
        if app in self.exclude:
            return False
        return frac >= self.min_frac


def split_app_selectors(values: list[str] | None) -> list[str]:
    """Flatten ``--include-app`` / ``--exclude-app`` values: repeatable AND
    comma-separated, because labctl renders every recipe arg as a single
    ``--key=value`` and so cannot repeat a flag."""
    out: list[str] = []
    for value in values or []:
        out.extend(part for part in str(value).replace(";", ",").split(",") if part.strip())
    return out


def app_stats(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-conversation app scoring over the frames this conversation actually
    contains. Same frame-weighted function stage 03 writes into its index row, so a
    prefilter on that row and this gate can never disagree."""
    return frame_app_stats(frames)


def _ensure_app_labels(
    frames: list[dict[str, Any]],
    index_row: dict[str, Any],
    sample_cfg: dict[str, Any] | None,
) -> bool:
    """Backfill ``app``/``app_window_switches`` on a stage-03 sample built before
    app labeling existed. Same source, same clock as stage 03 does it (the
    realigned keylog off the index row). Returns True if labels are present after
    this call."""
    if not frames:
        return False
    if frames[0].get("app") is not None:
        return True
    keylog = index_row.get("keylog_path")
    if not keylog or not Path(keylog).exists():
        return False
    ticks = [int(f["master_record_index"]) for f in frames]
    master_fps = float(
        index_row.get("master_fps") or (sample_cfg or {}).get("master_fps") or 0.0
    )
    if master_fps <= 0:
        raise ValueError(
            f"{index_row.get('segment_id')}: no master_fps on the sample index row -- "
            "cannot place app switches on the master axis"
        )
    axis_end = ticks[-1] + 1
    track = load_app_track(keylog, n_ticks=axis_end, master_fps=master_fps)
    for i, f in enumerate(frames):
        win_end = ticks[i + 1] if i + 1 < len(ticks) else axis_end
        f["app"] = track.at(ticks[i])
        f["app_window_switches"] = len(track.switches_in(ticks[i], win_end))
    return True


@dataclass(frozen=True)
class AppSpan:
    """One ``[lo, hi)`` frame span of a segment that becomes one conversation."""

    app: str | None
    lo: int
    hi: int
    seam_trimmed: bool = False


def plan_app_spans(
    frames: list[dict[str, Any]],
    cfg: AppFilter,
    *,
    min_frames: int,
) -> list[AppSpan]:
    """Which frame spans of one segment become conversations.

    Gate mode returns at most one span (the whole segment, or nothing). Split mode
    returns one span per maximal same-app run that passes the filter and is long
    enough; ``--app-drop-seam-turns`` first trims the boundary turn whose action
    window straddles the switch."""
    if not frames:
        return []
    if not cfg.split:
        stats = app_stats(frames)
        if not cfg.accepts(stats["app"], float(stats["app_frac"])):
            return []
        return [AppSpan(stats["app"], 0, len(frames))]

    out: list[AppSpan] = []
    for app, lo, hi in iter_app_spans(frames):
        end, trimmed = hi, False
        if cfg.drop_seam_turns and end > lo and frames[end - 1].get("app_window_switches"):
            end -= 1  # that turn's action label mixes this app with the next one
            trimmed = True
        if end - lo < max(cfg.min_run_frames, min_frames, 1):
            continue
        if not cfg.accepts(app, 1.0):
            continue
        out.append(AppSpan(app, lo, end, trimmed))
    return out


def _resolve_instruction(
    index_row: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    instruction: str | None,
    instruction_field: str | None,
) -> str | None:
    """Per-segment instruction from --instruction-field (sample_index row, then
    first frame record), else the fixed --instruction, else None (goal-free)."""
    if instruction_field:
        for src in (index_row, frames[0] if frames else {}):
            val = src.get(instruction_field)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return instruction


def build_conversation(
    index_row: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    instruction: str | None,
    instruction_field: str | None,
    system_prompt: str | None,
    idle_fn: Callable[[str], bool],
    fmt_fields: dict[str, Any] | None = None,
    app_fields: dict[str, Any] | None = None,
    id_suffix: str = "",
) -> dict[str, Any]:
    """One segment (or one same-app run of it) -> one conversation record. Frames are
    pre-loaded and pre-filtered by the caller; ``fmt_fields`` carries the formatter
    provenance when the actions were re-derived (action_format / dead-zone
    accounting) and ``app_fields`` the app-filter provenance. ``id_suffix``
    distinguishes several conversations cut from one segment (``--split-by-app``)."""
    seg_instruction = _resolve_instruction(
        index_row, frames, instruction=instruction, instruction_field=instruction_field
    )
    messages = build_messages(frames, instruction=seg_instruction, system_prompt=system_prompt)
    return {
        "conversation_id": f"{index_row.get('segment_id')}{id_suffix}",
        "recording_id": index_row.get("recording_id"),
        "segment_id": index_row.get("segment_id"),
        "segment_idx": index_row.get("segment_idx"),
        "instruction": seg_instruction,
        "goal_conditioned": seg_instruction is not None,
        **(fmt_fields or {}),
        **(app_fields or {}),
        **_coalesce_counts(frames),
        "n_frames": len(frames),
        "n_turns": len(frames),  # one user+assistant pair per frame
        "n_non_noop": sum(1 for f in frames if not idle_fn(str(f.get("action")))),
        "target_fps": index_row.get("target_fps"),
        "alignment_status": index_row.get("alignment_status"),
        "messages": messages,
    }


def build_goal_conversation(
    goal: dict[str, Any],
    seg_frames: list[dict[str, Any]],
    index_row: dict[str, Any],
    *,
    system_prompt: str | None,
    min_frames: int,
    idle_fn: Callable[[str], bool],
    terminate_token: str | None = None,
    fmt_fields: dict[str, Any] | None = None,
    precoalesce_source_idx: list[int] | None = None,
    app_filter: AppFilter | None = None,
) -> dict[str, Any] | None:
    """One goal-conditioned conversation: OUR ``--sample-dir`` frames for the goal's
    segment, windowed to its source-frame span ``[lo, hi]`` (from stage-03b), with the
    goal text -- plus its self-compaction ``[CONTEXT]`` summary when the goal index
    carries one (non-first chunks of a split goal) -- as the first user turn. Returns
    None if the window holds fewer than ``min_frames`` frames (e.g. an all-idle goal
    ``noop_mode=none`` dropped).

    ``precoalesce_source_idx`` (``--coalesce-typing``): the segment's source-frame
    indices BEFORE frames were coalesced away, so ``min_frames`` gates on the goal's
    original frame count -- coalescing must not decide which goals exist.

    ``terminate_token`` marks the window's end as goal-complete: the final assistant
    turn's action is overwritten with the token (see ``build_messages``)."""
    lo, hi = goal.get("coll_source_frame_idx_lo"), goal.get("coll_source_frame_idx_hi")
    if lo is None or hi is None:
        return None
    lo, hi = int(lo), int(hi)
    frames = [
        f for f in seg_frames
        if f.get("source_frame_idx") is not None and lo <= int(f["source_frame_idx"]) <= hi
    ]
    n_gate = (
        sum(1 for s in precoalesce_source_idx if lo <= s <= hi)
        if precoalesce_source_idx is not None else len(frames)
    )
    if n_gate < min_frames or not frames:
        return None
    frames.sort(key=lambda r: int(r["source_frame_idx"]))
    # App gate over the goal's own window (goal windows already define the units, so
    # --split-by-app is rejected in goal mode; this only keeps or drops).
    app_info = app_stats(frames)
    if app_filter is not None and not app_filter.accepts(
        app_info["app"], float(app_info["app_frac"])
    ):
        return None
    instruction = goal.get("instruction")
    context = goal.get("context")  # self-compaction rolling summary (non-first chunks)
    messages = build_messages(
        frames, instruction=instruction, system_prompt=system_prompt,
        context=context, terminate_token=terminate_token,
    )
    return {
        "conversation_id": goal.get("sample_id") or f"{goal.get('segment_id')}_g{goal.get('goal_id')}",
        "recording_id": goal.get("recording_id") or index_row.get("recording_id"),
        "segment_id": goal.get("segment_id"),
        "segment_idx": index_row.get("segment_idx"),
        "goal_id": goal.get("goal_id"),
        "sample_id": goal.get("sample_id"),
        "instruction": instruction,
        # self-compaction provenance (goal-conditioned selfcompact set); context is
        # also fused into messages[0], these mirror it as clean metadata.
        "context": context,
        "chunk_idx": goal.get("chunk_idx"),
        "n_chunks": goal.get("n_chunks"),
        "parent_sample_id": goal.get("parent_sample_id"),
        "context_tokens": goal.get("context_tokens"),
        "goal_conditioned": True,
        "long_ref": goal.get("long_ref"),
        "long_text": goal.get("long_text"),
        "kind": goal.get("kind"),
        "status": goal.get("status"),
        "split": goal.get("split"),
        **app_info,
        **(fmt_fields or {}),  # segment-level formatter provenance (dead-zone counters)
        **_coalesce_counts(frames),
        "n_frames": len(frames),
        "n_turns": len(frames),  # one user+assistant pair per frame
        "n_non_noop": sum(1 for f in frames if not idle_fn(str(f.get("action")))),
        "target_fps": index_row.get("target_fps"),
        "alignment_status": index_row.get("alignment_status"),
        "messages": messages,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-dir", type=Path, required=True,
                   help="A stage-03 (sample_frames_actions) --output-dir: must contain "
                        "sample_index.jsonl and clips/<seg>/stage_01/frame_records.jsonl.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--action-format", type=str, default=SAMPLED_FORMAT,
                   choices=[SAMPLED_FORMAT, *sorted(FORMATTERS)],
                   help="Assistant-turn action format. 'sampled' (default): pass the "
                        "stage-03 frame records' canonical action strings through "
                        "verbatim. Any registered formatter (lib/action_format) instead "
                        "re-derives every label from the realigned keylog at build time "
                        "under the shared dead-zone policy: 'canonical' (aggregate "
                        "'<dx> <dy> <scroll> ; +KEY -KEY'), 'ordered_events_v2' (ordered "
                        "'move(dx,dy); down(LMB); ...' mini-programs), 'ordered_events_v3' "
                        "(v2 + type(\"...\") typing runs), 'computer_use_rel_v1' (Qwen "
                        "native <tool_call> JSON, relative mouse).")
    p.add_argument("--continuous-action-hz", type=float,
                   default=DEFAULT_CONTINUOUS_ACTION_HZ,
                   help="ordered_events_* only: internal motor-grid rate for accumulating "
                        "move/scroll deltas within a window (NOT a frame rate; recorded as "
                        "null for formats that ignore it).")
    p.add_argument("--jitter-deadband-px", type=int, default=0,
                   help="ordered_events_* only: drop a move(dx,dy) primitive when it sits "
                        "BETWEEN two other primitives (a click, scroll, or typed text) and "
                        "both |dx| and |dy| are within this threshold -- incidental cursor "
                        "jitter from hand tension while operating a button/wheel/keyboard, "
                        "not intentional pointer control. A move at either end of a window "
                        "(nothing before or after it) is never dropped. Default 0 = off.")
    p.add_argument("--coalesce-typing", nargs="?", const=True, type=_str2bool,
                   default=False, metavar="BOOL",
                   help="ordered_events_v3 ONLY: fuse consecutive typing-only frames "
                        "into ONE turn -- the first frame's screenshot keeps the whole "
                        "run's typing as a single type(\"...\"), the rest are DROPPED "
                        "from the conversation. Idle frames inside a run are absorbed; "
                        "trailing idle frames, keypresses (Return/Backspace/Tab/arrows), "
                        "mouse actions, chords, dead zones and goal boundaries all break "
                        "a run, and a run never ends a conversation (its last frame is "
                        "kept as a NO_OP turn). Bare --coalesce-typing = on; "
                        "--coalesce-typing=false is off (labctl's --key=value arg form). "
                        "See --max-coalesce-frames.")
    p.add_argument("--max-coalesce-frames", type=int, default=8,
                   help="--coalesce-typing: how many original frames ONE turn may span "
                        "(0 = unlimited). The turn shows the run's FIRST screenshot, so "
                        "this bounds how stale that screenshot may be relative to the "
                        "typing it is labeled with; longer runs split into consecutive "
                        "chunks, each keeping its own screenshot.")
    p.add_argument("--dead-zone-flag-frac", type=float, default=0.05,
                   help="Formatter modes: flag a segment (dead_zone_flagged) when more than "
                        "this fraction of its keylog events were discarded by the dead-zone "
                        "policy (realignment health).")
    p.add_argument("--instruction", type=str, default=None,
                   help="Fixed instruction placed on each segment's first user turn "
                        "(goal-conditioned). Omit for goal-free (system-prompt only).")
    p.add_argument("--instruction-field", type=str, default=None,
                   help="Per-segment instruction: read this key from the sample_index row "
                        "(then the first frame record). Falls back to --instruction, then goal-free.")
    sp = p.add_mutually_exclusive_group()
    sp.add_argument("--system-prompt", type=str, default=None,
                    help="Raw system message text. Default: a goal-free prompt, or the canonical "
                         "goal-conditioned prompt when an instruction is set.")
    sp.add_argument("--system-prompt-id", type=str, default=None,
                    help="Select a named system prompt from eval/osworld_system_prompts.py "
                         "(shared with the OSWorld eval runners). One of: "
                         f"{', '.join(SYSTEM_PROMPTS)}.")
    sp.add_argument("--no-system-prompt", action="store_true", help="Emit no system message.")
    p.add_argument("--goal-index", type=Path, default=None,
                   help="A stage-03b goal_frame_index.jsonl. Switches to GOAL-CONDITIONED "
                        "mode: one conversation per goal, --sample-dir frames windowed to "
                        "the goal's source-frame span, goal text as the first-turn "
                        "instruction. Ignores --instruction/--instruction-field.")
    p.add_argument("--terminate-token", type=str, default="TERMINATE",
                   help="GOAL-CONDITIONED (--goal-index) ONLY: overwrite the final assistant "
                        "turn's action with this token (canonical: \"TERMINATE\") to mark goal "
                        "completion at the window's end. Off by default; ignored (with a warning) "
                        "in per-segment mode, where a segment end is not a task completion. Pair "
                        "with a system prompt that describes the contract, e.g. "
                        "--system-prompt-id yll_v1.")
    ap = p.add_argument_group(
        "application filter",
        "Select conversations by the FOREGROUND APP recorded in the keylog "
        "(stage-03 'app' labels; back-filled from the realigned keylog when the "
        "sample predates them). App ids accept friendly names (firefox, cursor, "
        "vscode, ghostty, safari, arc, ...) or raw bundle ids / process names.",
    )
    ap.add_argument("--include-app", action="append", default=None, metavar="APP",
                    help="Keep only conversations whose app is APP. Repeatable "
                         "(--include-app firefox --include-app safari) OR comma-separated "
                         "(--include-app=firefox,safari), since a labctl recipe renders "
                         "each arg once as --key=value and cannot repeat a flag.")
    ap.add_argument("--exclude-app", action="append", default=None, metavar="APP",
                    help="Drop conversations whose app is APP. Repeatable or "
                         "comma-separated, like --include-app. Applied after it.")
    ap.add_argument("--app-min-frac", type=float, default=0.0,
                    help="Gate mode only: require the dominant app to hold at least this "
                         "fraction of the conversation's LABELED frames (0 = no purity "
                         "requirement). Measured on ccast0618d: 0.8 keeps ~65%% of "
                         "segments, 0.95 keeps ~43%%. Ignored with --split-by-app, whose "
                         "runs are pure by construction.")
    ap.add_argument("--app-unknown", choices=APP_UNKNOWN_MODES, default="keep",
                    help="What to do with conversations that carry NO app label -- "
                         "recorder versions before 0.1.1 emit no ContextChanged, and "
                         "UNCAPTURED is the privacy blackout. 'keep' (default) passes them "
                         "when no --include-app is set; 'drop' removes every unlabeled "
                         "conversation, which is what you want for a clean per-app "
                         "comparison (~23%% of ccast0618d segments are unlabeled).")
    ap.add_argument("--split-by-app", nargs="?", const=True, type=_str2bool,
                    default=False, metavar="BOOL",
                    help="Cut each segment into ONE CONVERSATION PER maximal same-app run "
                         "instead of gating whole segments. A segment is not "
                         "app-homogeneous (only ~31%% touch a single app), so this is the "
                         "way to get pure per-app data. An UNCAPTURED gap does not break a "
                         "run when the same app resumes (those frames are the black ones "
                         "already dropped). Rejected in --goal-index mode, where the goal "
                         "windows already define the conversation units.")
    ap.add_argument("--app-min-run-frames", type=int, default=1,
                    help="--split-by-app: skip runs shorter than this many frames (== turns; "
                         "at --target-fps 1 it is also seconds). Runs >= 30 frames @1 fps "
                         "hold ~91%% of the corpus's captured foreground time.")
    ap.add_argument("--app-drop-seam-turns", nargs="?", const=True, type=_str2bool,
                    default=True, metavar="BOOL",
                    help="--split-by-app: drop each run's final turn when its action window "
                         "straddles the app switch (stage-03 'app_window_switches'), since "
                         "that one label aggregates input from BOTH apps -- the same "
                         "argument as the black-frame dead zones. ~2.5%% of turns @1 fps. "
                         "Pass =false to keep them.")
    p.add_argument("--min-frames", type=int, default=1,
                   help="Skip segments (or goals, in --goal-index mode) with fewer than "
                        "this many frames.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N segments (or goals, in --goal-index mode).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    index_path = args.sample_dir / "sample_index.jsonl"
    if not index_path.is_file():
        raise SystemExit(f"no sample_index.jsonl under {args.sample_dir} (is it a stage-03 sample artifact?)")
    index_rows = read_jsonl(index_path)
    usable = [r for r in index_rows if r.get("status") in USABLE_STATUSES]
    if args.limit is not None:
        usable = usable[: args.limit]
    if not usable:
        raise SystemExit(f"no usable segments (status in {sorted(USABLE_STATUSES)}) in {index_path}")

    # Action format: 'sampled' keeps the frame records' strings; a registered
    # formatter re-derives every label from the realigned keylog (fails fast on an
    # unknown name / invalid hz). The idle predicate feeds n_non_noop either way.
    formatter: ActionFormatter | None = None
    sample_cfg: dict[str, Any] | None = None
    if args.action_format != SAMPLED_FORMAT:
        formatter = get_formatter(
            args.action_format,
            continuous_action_hz=args.continuous_action_hz,
            jitter_deadband_px=args.jitter_deadband_px,
        )
        sample_cfg = _load_sample_config(args.sample_dir)
    # Coalescing needs a format where "this window is typing and nothing else" is
    # a decidable question -- i.e. one with a type() primitive.
    if args.coalesce_typing and args.action_format not in TYPING_COALESCE_FORMATS:
        raise SystemExit(
            f"--coalesce-typing requires --action-format "
            f"{'/'.join(sorted(TYPING_COALESCE_FORMATS))} (got {args.action_format!r}): "
            "the other formats spell typing as bare key transitions, indistinguishable "
            "from a chord."
        )
    idle_fn: Callable[[str], bool] = (
        formatter.is_idle_label if formatter is not None else (lambda a: a == "NO_OP")
    )

    # Application filter. Selectors are resolved once (friendly name -> canonical
    # id) so an unknown spelling fails here rather than silently matching nothing.
    try:
        app_filter = AppFilter(
            include=frozenset(
                resolve_app_selector(a) for a in split_app_selectors(args.include_app)
            ),
            exclude=frozenset(
                resolve_app_selector(a) for a in split_app_selectors(args.exclude_app)
            ),
            min_frac=float(args.app_min_frac),
            unknown=args.app_unknown,
            split=bool(args.split_by_app),
            min_run_frames=int(args.app_min_run_frames),
            drop_seam_turns=bool(args.app_drop_seam_turns),
        )
    except ValueError as exc:
        raise SystemExit(f"--include-app/--exclude-app: {exc}") from exc
    if app_filter.split and args.goal_index:
        raise SystemExit(
            "--split-by-app is not compatible with --goal-index: a goal window already "
            "defines the conversation unit. Use --include-app/--exclude-app to gate goals "
            "on their dominant app instead."
        )
    if app_filter.active and sample_cfg is None:
        sample_cfg = _load_sample_config(args.sample_dir)
    if app_filter.active and not sample_cfg.get("app_context", False):
        print("[conversations] NOTE: this stage-03 sample carries no app labels; "
              "back-filling them from the realigned keylogs.", flush=True)

    # System prompt: explicit override wins; else default by goal-conditioning. A
    # segment is goal-conditioned iff it resolves an instruction, but the system
    # prompt is chosen once for the run from whether ANY instruction source is set
    # (a --goal-index run is always goal-conditioned). With a formatter active the
    # default is COMPOSED from its reply contract, so it describes the format.
    goal_conditioned = bool(args.goal_index or args.instruction or args.instruction_field)
    system_prompt_id = None
    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    elif args.system_prompt_id is not None:
        if args.system_prompt_id not in SYSTEM_PROMPTS:
            raise SystemExit(
                f"unknown --system-prompt-id {args.system_prompt_id!r}; "
                f"available: {', '.join(SYSTEM_PROMPTS)}"
            )
        system_prompt_id = args.system_prompt_id
        system_prompt = SYSTEM_PROMPTS[system_prompt_id]
    elif formatter is not None:
        system_prompt = default_system_prompt(formatter, goal_conditioned=goal_conditioned)
    else:
        system_prompt_id = "training_v1" if not goal_conditioned else None
        system_prompt = GOAL_SYSTEM_PROMPT if goal_conditioned else GOAL_FREE_SYSTEM_PROMPT

    # TERMINATE is goal-conditioned-only (a segment end is not a task completion).
    # With a formatter active, the default token is the formatter's own terminate
    # line ("TERMINATE" for the text formats, the native terminate tool_call for
    # computer_use_rel_v1); an explicitly different --terminate-token still wins.
    terminate_token = args.terminate_token
    if terminate_token and not args.goal_index:
        print("[conversations] WARNING: --terminate-token is ignored without --goal-index "
              "(per-segment ends are not goal completions).", flush=True)
        terminate_token = None
    elif terminate_token == "TERMINATE" and formatter is not None:
        terminate_token = formatter.terminate_line()

    out_dir = ensure_dir(args.output_dir)
    records: list[dict[str, Any]] = []
    n_skipped = 0
    n_failed = 0
    n_frames_total = 0
    n_turns_total = 0
    dz_totals: Counter = Counter()
    prim_totals: Counter = Counter()
    n_dz_flagged = 0
    n_coalesced_turns = 0
    n_coalesced_dropped = 0
    n_forced_idle_turns = 0
    n_skipped_app = 0
    n_app_unlabeled = 0
    n_seam_turns_dropped = 0
    app_conv_counts: Counter = Counter()

    def _reformat(
        frames: list[dict[str, Any]],
        row: dict[str, Any],
        goal_spans: list[tuple[int, int]] | None = None,
    ) -> dict[str, Any] | None:
        """Re-derive one segment's labels (formatter modes) and fold its
        accounting into the run totals; returns the per-row provenance fields.
        With --coalesce-typing this also DROPS coalesced frames from ``frames``
        in place, so the caller's list is the post-coalesce conversation."""
        nonlocal n_dz_flagged, n_coalesced_turns, n_coalesced_dropped, n_forced_idle_turns
        if formatter is None or not frames:
            return None
        info = reformat_segment_actions(
            frames, row, formatter=formatter, sample_cfg=sample_cfg,
            dead_zone_flag_frac=args.dead_zone_flag_frac,
            coalesce_typing=args.coalesce_typing,
            max_coalesce_frames=args.max_coalesce_frames,
            goal_spans=goal_spans,
        )
        for k, v in info["dead_zone_counters"].items():
            if k != "max_simultaneous_keys":
                dz_totals[k] += int(v)
        for k, v in (info.pop("primitive_counts", None) or {}).items():
            prim_totals[k] += int(v)
        if info["dead_zone_flagged"]:
            n_dz_flagged += 1
        n_coalesced_turns += int(info.get("coalesced_turns") or 0)
        n_coalesced_dropped += int(info.get("coalesced_frames_dropped") or 0)
        n_forced_idle_turns += int(info.get("coalesce_forced_idle_turns") or 0)
        return info

    if args.goal_index:
        # GOAL-CONDITIONED: one conversation per goal, OUR sampled frames windowed to
        # the goal's source-frame span. The span is colleague-derived (fps-independent),
        # so this goal index goal-conditions whatever fps --sample-dir holds. Group goals
        # by segment so each segment's frame_records is read once.
        index_by_seg = {str(r["segment_id"]): r for r in index_rows}
        goals = [g for g in read_jsonl(args.goal_index)
                 if g.get("match_status") == "ok" and g.get("segment_id")]
        if args.limit is not None:
            goals = goals[: args.limit]
        goals_by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for g in goals:
            goals_by_seg[str(g["segment_id"])].append(g)
        print(f"[conversations] goal-conditioned: {len(goals)} goals across "
              f"{len(goals_by_seg)} segments", flush=True)
        for j, (segment_id, seg_goals) in enumerate(goals_by_seg.items(), 1):
            row = index_by_seg.get(segment_id)
            if row is None or row.get("status") not in USABLE_STATUSES:
                n_skipped += len(seg_goals)  # segment not in --sample-dir (or unusable)
                continue
            try:
                seg_frames = _load_segment_frames(row) or []
                # --min-frames gates on the ORIGINAL frame count, so capture the
                # source-frame indices before coalescing drops any.
                pre_src = [int(f["source_frame_idx"]) for f in seg_frames
                           if f.get("source_frame_idx") is not None]
                # Goal boundaries clamp the coalesce runs (a run must not straddle
                # a goal window); computed once per segment, shared by its goals.
                spans = [(int(g["coll_source_frame_idx_lo"]), int(g["coll_source_frame_idx_hi"]))
                         for g in seg_goals
                         if g.get("coll_source_frame_idx_lo") is not None
                         and g.get("coll_source_frame_idx_hi") is not None]
                fmt_fields = _reformat(seg_frames, row, goal_spans=spans)
                if app_filter.active and not _ensure_app_labels(seg_frames, row, sample_cfg):
                    n_app_unlabeled += 1
            except Exception as exc:  # noqa: BLE001 - one bad segment must not abort the run
                n_failed += len(seg_goals)
                print(f"  FAIL {segment_id}: {exc}", flush=True)
                continue
            for g in seg_goals:
                conv = build_goal_conversation(
                    g, seg_frames, row, system_prompt=system_prompt,
                    min_frames=args.min_frames, idle_fn=idle_fn,
                    terminate_token=terminate_token, fmt_fields=fmt_fields,
                    precoalesce_source_idx=pre_src if args.coalesce_typing else None,
                    app_filter=app_filter if app_filter.active else None,
                )
                if conv is None:
                    n_skipped += 1
                    continue
                records.append(conv)
                n_frames_total += conv["n_frames"]
                n_turns_total += conv["n_turns"]
                if conv.get("app"):
                    app_conv_counts[str(conv["app"])] += 1
            if j % 500 == 0:
                print(f"  {j}/{len(goals_by_seg)} segments | {len(records)} goal conversations", flush=True)
    else:
        for i, row in enumerate(usable, 1):
            try:
                frames = _load_segment_frames(row)
                if frames is None or len(frames) < args.min_frames:
                    n_skipped += 1
                    continue
                # Labels are re-derived (and typing coalesced) over the WHOLE segment
                # first, so every kept turn's action is byte-identical to the unsplit
                # dataset; the app filter then only decides which turns survive.
                fmt_fields = _reformat(frames, row)
                if app_filter.active and not _ensure_app_labels(frames, row, sample_cfg):
                    n_app_unlabeled += 1
                spans = (
                    plan_app_spans(frames, app_filter, min_frames=args.min_frames)
                    if app_filter.active
                    else [AppSpan(None, 0, len(frames))]
                )
                if not spans:
                    n_skipped += 1
                    n_skipped_app += 1
                    continue
                convs = []
                for run_idx, span in enumerate(spans):
                    span_frames = frames[span.lo:span.hi]
                    info = app_stats(span_frames) if app_filter.active else {}
                    if app_filter.split:
                        if span.seam_trimmed:
                            n_seam_turns_dropped += 1
                        info = {
                            **info,
                            "app_run_idx": run_idx,
                            "n_app_runs": len(spans),
                            "app_seam_turn_dropped": span.seam_trimmed,
                        }
                    convs.append(build_conversation(
                        row,
                        span_frames,
                        instruction=args.instruction,
                        instruction_field=args.instruction_field,
                        system_prompt=system_prompt,
                        idle_fn=idle_fn,
                        fmt_fields=fmt_fields,
                        app_fields=info or None,
                        # one segment can now yield several conversations
                        id_suffix=f"_app{run_idx:02d}" if app_filter.split else "",
                    ))
            except Exception as exc:  # noqa: BLE001 - one bad segment must not abort the run
                n_failed += 1
                print(f"  FAIL {row.get('segment_id')}: {exc}", flush=True)
                continue
            for conv in convs:
                records.append(conv)
                n_frames_total += conv["n_frames"]
                n_turns_total += conv["n_turns"]
                if conv.get("app"):
                    app_conv_counts[str(conv["app"])] += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(usable)} segments | {len(records)} conversations", flush=True)

    if not records:
        raise SystemExit(
            "no conversations built (all segments empty or below --min-frames"
            + (", or rejected by the app filter" if app_filter.active else "")
            + ")"
        )

    write_jsonl(out_dir / "conversations.jsonl", records)
    write_jsonl(out_dir / "chat.jsonl", records)

    summary = {
        "n_conversations": len(records),
        "n_skipped": n_skipped,  # goals (goal-index mode) or segments below --min-frames
        "n_failed": n_failed,
        "n_frames_total": n_frames_total,
        "n_turns_total": n_turns_total,
        "mode": "goal" if args.goal_index else "segment",
        "goal_conditioned": goal_conditioned,
        "goal_index": str(args.goal_index) if args.goal_index else None,
        "action_format": args.action_format,
        "continuous_action_hz": getattr(formatter, "continuous_action_hz", None),
        "jitter_deadband_px": getattr(formatter, "jitter_deadband_px", None),
        "coalesce_typing": bool(args.coalesce_typing),
        "max_coalesce_frames": args.max_coalesce_frames if args.coalesce_typing else None,
        "n_coalesced_turns": n_coalesced_turns if args.coalesce_typing else None,
        "n_frames_coalesced_away": n_coalesced_dropped if args.coalesce_typing else None,
        "n_coalesce_forced_idle_turns": n_forced_idle_turns if args.coalesce_typing else None,
        "primitive_counts": dict(prim_totals) if prim_totals else None,
        "dead_zone_totals": dict(dz_totals) if formatter is not None else None,
        "n_dead_zone_flagged": n_dz_flagged if formatter is not None else None,
        # --- application filter -------------------------------------------------
        "app_filter_active": app_filter.active,
        "include_app": sorted(app_filter.include) or None,
        "exclude_app": sorted(app_filter.exclude) or None,
        "app_min_frac": app_filter.min_frac if not app_filter.split else None,
        "app_unknown": app_filter.unknown,
        "split_by_app": app_filter.split,
        "app_min_run_frames": app_filter.min_run_frames if app_filter.split else None,
        "app_drop_seam_turns": app_filter.drop_seam_turns if app_filter.split else None,
        "n_app_seam_turns_dropped": n_seam_turns_dropped if app_filter.split else None,
        # segment mode only -- goal mode folds app rejections into n_skipped
        "n_skipped_app_filter": n_skipped_app if not args.goal_index else None,
        "n_segments_without_app_labels": n_app_unlabeled if app_filter.active else None,
        "app_conversation_counts": dict(app_conv_counts.most_common()) or None,
        "terminate_token": terminate_token,
        "instruction": args.instruction,
        "instruction_field": args.instruction_field,
        "has_system_prompt": system_prompt is not None,
        "system_prompt_id": system_prompt_id,
        "sample_dir": str(args.sample_dir),
    }
    write_json(out_dir / "conversations_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "juergen_annotation_conversations",
        "schema_version": 1,
        "conversations": "conversations.jsonl",
        "chat": "chat.jsonl",  # split-agnostic drop-in source_path for stages 05/06
        **summary,
    })
    coalesce_note = (
        f" | coalesced {n_coalesced_turns} turns (-{n_coalesced_dropped} frames)"
        if args.coalesce_typing else ""
    )
    app_note = ""
    if app_filter.active:
        top = ", ".join(f"{a}={n}" for a, n in app_conv_counts.most_common(5))
        app_note = (
            f" | apps: {'split' if app_filter.split else 'gate'}, "
            f"{len(app_conv_counts)} distinct"
            + (f", {n_seam_turns_dropped} seam turns dropped" if app_filter.split else "")
            + (f", {n_app_unlabeled} segments unlabeled" if n_app_unlabeled else "")
            + (f" (top: {top})" if top else "")
        )
    print(
        f"[conversations] {len(records)} conversations, {n_turns_total} turns, "
        f"{n_frames_total} frames, {n_skipped} skipped, {n_failed} failed "
        f"| format={args.action_format}{coalesce_note}{app_note} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
