#!/usr/bin/env python3
"""Functional QA over a PUBLISHED sequential_goal_memory artifact.

Quality for this recipe is functional, not stylistic: instead of eyeballing
prose we re-derive the properties training depends on straight from the
published jsonl files (never from the annotator's in-memory state, so a bug in
the annotator cannot vouch for itself), and we probe whether a checkpoint is
actually a resumable state.

Deterministic checks (always):
  parser         every ``assistant_action`` parses as an action reply and every
                 checkpoint ``text`` as a checkpoint reply, through the
                 evaluator's own ``parse_sequential_reply``.
  memory_chain   one snapshot per semantic event, contiguous anchors, and
                 ``memory_before[i] == memory_after[i-1]`` (re-run, not trusted).
  references     no snapshot/decision/checkpoint reference resolves to a LATER
                 event.
  leaks          future-only strings (typed text tokens, key combos) that a text
                 could not causally know, found as substrings in memory /
                 thoughts / checkpoints.
  thoughts       density overall and on motor events (must be 0), word budget vs
                 THOUGHT_MAX_WORDS, divergence rate, thought/action n-gram
                 overlap ("parrot score").
  checkpoints    seven non-empty fields, word budget vs CHECKPOINT_MAX_WORDS,
                 and ``Completed`` folding regression across chained anchors.
  stage04        with ``--stage04-chat``: every assistant turn round-trips, the
                 checkpoint handoff is byte-identical, and n_images respects the
                 packing capacity.

LLM checks (skipped under ``--no-llm``): a resumability probe on ``--sample``
boundary checkpoints — a fresh context of system prompt + goal/checkpoint + the
boundary screenshot ONLY, judged against the true next three action packets.

Writes ``qa_report.json`` (every metric) and ``review_draft.json`` (the six
human-review gate keys, auto-drafted where a metric can decide them) into the
artifact dir. The draft deliberately never satisfies the full-run gate:
``reviewed_by`` stays null and the human-judgement gates stay null.

Run::

    cd data_pipeline
    uv run python realigned_pipeline/annotation/qa_sequential_goal_memory.py \
        --artifact <stage-03b artifact dir> --stage04-chat <stage-04>/chat.jsonl \
        --sample 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.annotation.lib.labeler import (  # noqa: E402
    Labeler,
    LabelerConfig,
    labeler_model,
)
from realigned_pipeline.annotation.lib.units import frames_to_data_urls  # noqa: E402
from realigned_pipeline.lib.common import normalize_dashed_argv, write_json  # noqa: E402
from realigned_pipeline.lib.sequential_goal_memory_contract import (  # noqa: E402
    CHECKPOINT_FIELDS,
    CHECKPOINT_MAX_WORDS,
    METHOD,
    PROACTIVE_GOAL_TEXT,
    THOUGHT_MAX_WORDS,
    goal_conditioning,
    system_prompt,
)

QA_PROMPT_VERSIONS = {
    "resumability_probe": "resumability_probe_v1",
    "resumability_judge": "resumability_judge_v1",
}
# The six keys stage_annotate._require_pilot_review demands (that function is
# the authority; this mirror only exists so the writer can refuse to emit a
# draft that would unblock a full run).
FULL_RUN_GATES = (
    "goal_grounding", "causal_thoughts", "cross_day_links", "checkpoints",
    "action_provenance", "parser_validity",
)
# Auto-draftable gate thresholds.
PROBE_PASS_RATE_MIN = 0.8
# Leak candidates are strings a text could only know from the future: typed-text
# tokens this long, plus key-combo names.
LEAK_MIN_CHARS = 4
_LEAK_TOKEN_RE = re.compile(r"[^\W_][\w@.\-/']{3,}")
_WORD_RE = re.compile(r"[a-z0-9]+")
# "Completed" must FOLD, so a chain whose Completed only ever grows — and ends
# this much longer than it started — is a folding regression, as is restating
# the previous Completed verbatim.
FOLDING_GROWTH_RATIO = 1.5
FOLDING_MIN_CHAIN = 3
N_WORST_EXAMPLES = 10
PROBE_NEXT_PACKETS = 3
# Parroting the action back is not a causal thought; these carry no content.
_STOPWORDS = frozenset(
    "a an and are as at be been but by can for from has have i if in into is it its "
    "me my need needs of on or should so that the then there this to up will with".split()
)
# Schema fields the redesign adds. Absent -> unknown, reported, never fatal.
NEW_FIELDS = {
    "memory_snapshots": ("is_decision_boundary", "agreed"),
    "checkpoints": ("packing_config_hash", "is_day_final", "source_memory_snapshot_id"),
    "decisions": ("gate",),
}

PROBE_REQUEST = (
    "You are resuming this task after a context compaction. All you have is the "
    "GOAL and checkpoint above plus the single screenshot below. Describe the "
    "situation in one or two sentences, then state the ONE next step you would "
    "take and where on the screen you would take it. Do not emit tool calls."
)
JUDGE_SYSTEM = (
    "You grade whether an agent resuming from a compaction checkpoint proposed a "
    "continuation compatible with what the human operator actually did next. "
    "Reply with JSON only."
)
JUDGE_TEMPLATE = (
    "GOAL\n{goal}\n\n"
    "CHECKPOINT THE RESUMING AGENT WAS GIVEN\n{checkpoint}\n\n"
    "WHAT THE RESUMING AGENT SAID\n{answer}\n\n"
    "WHAT THE HUMAN OPERATOR ACTUALLY DID NEXT (ordered computer_use packets)\n"
    "{packets}\n\n"
    "Compatible means the proposed next step advances the same sub-task, in the "
    "same region of the same application, as the recorded continuation. It need "
    "NOT match pixel deltas, wording, or the exact input primitive. Reply with "
    'exactly {{"compatible": true|false, "reason": "<one sentence>"}}.'
)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _rows(path: Path) -> list[dict[str, Any]]:
    """Tolerant jsonl read: a missing or empty file is simply no rows."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _words(text: str) -> list[str]:
    return str(text or "").split()


@dataclass
class DayView:
    """One day's published rows, joined by semantic-event index."""

    day_tag: str
    events: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)
    _position: dict[str, int] | None = None

    @property
    def position(self) -> dict[str, int]:
        """Semantic-event id -> day-local index; built once the day is loaded."""
        if self._position is None:
            self._position = {str(row["semantic_event_id"]): index
                              for index, row in enumerate(self.events)}
        return self._position

    def anchor_index(self, row: dict[str, Any]) -> int:
        """Anchor position, preferring the resolved event id over the stored
        index so a stale index cannot hide a future reference."""
        resolved = self.position.get(str(row.get("anchor_semantic_event_id") or ""))
        return resolved if resolved is not None else _int(row.get("anchor_event_index"), -1)

    def covering_goals(self, index: int) -> list[dict[str, Any]]:
        return [node for node in self.goals
                if _int(node.get("start_event_index"), 0) <= index
                <= _int(node.get("end_event_index"), -1)]


@dataclass
class Artifact:
    dir: Path
    manifest: dict[str, Any]
    days: list[DayView]
    mission_links: list[dict[str, Any]]
    orphans: dict[str, int]

    @property
    def all_checkpoints(self) -> list[tuple[DayView, dict[str, Any]]]:
        return [(day, row) for day in self.days for row in day.checkpoints]


def load_artifact(artifact_dir: Path) -> Artifact:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no manifest.json under {artifact_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("method") != METHOD:
        raise SystemExit(f"artifact method is {manifest.get('method')!r}, not {METHOD!r}")
    events = _rows(artifact_dir / "semantic_events.jsonl")
    if not events:
        raise SystemExit(f"{artifact_dir} has no semantic_events.jsonl rows")
    days: dict[str, DayView] = {}
    for row in sorted(events, key=lambda r: (str(r.get("day_tag")),
                                             _int(r.get("day_event_index"), 0))):
        days.setdefault(str(row.get("day_tag")), DayView(str(row.get("day_tag")))).events.append(row)
    orphans = {"memory_snapshots": 0, "checkpoints": 0, "decisions": 0, "goal_nodes": 0}
    for name, key, attribute in (
        ("memory_snapshots.jsonl", "memory_snapshots", "snapshots"),
        ("checkpoints.jsonl", "checkpoints", "checkpoints"),
        ("decision_thoughts.jsonl", "decisions", "decisions"),
        ("goal_nodes.jsonl", "goal_nodes", "goals"),
    ):
        for row in _rows(artifact_dir / name):
            day = days.get(str(row.get("day_tag")))
            if day is None:
                # goal_nodes carry no day_tag in schema v2; fall back to the
                # single day their events belong to.
                day = _day_for_row(days, row)
            if day is None:
                orphans[key] += 1
                continue
            getattr(day, attribute).append(row)
    for day in days.values():
        day.snapshots.sort(key=lambda row: day.anchor_index(row))
        day.checkpoints.sort(key=lambda row: day.anchor_index(row))
    return Artifact(
        dir=artifact_dir, manifest=manifest,
        days=[days[tag] for tag in sorted(days)],
        mission_links=_rows(artifact_dir / "mission_links.jsonl"),
        orphans=orphans,
    )


def _day_for_row(days: dict[str, DayView], row: dict[str, Any]) -> DayView | None:
    anchor = str(row.get("anchor_semantic_event_id")
                 or row.get("start_semantic_event_id") or "")
    if not anchor:
        return None
    for day in days.values():
        if anchor in day.position:
            return day
    return None


def schema_tolerance(artifact: Artifact) -> dict[str, Any]:
    """Which redesign fields are absent, so every 'unknown' in the report has a
    stated cause rather than looking like a zero."""
    rows_by_kind = {
        "memory_snapshots": [row for day in artifact.days for row in day.snapshots],
        "checkpoints": [row for day in artifact.days for row in day.checkpoints],
        "decisions": [row for day in artifact.days for row in day.decisions],
    }
    missing: dict[str, dict[str, int]] = {}
    for kind, fields in NEW_FIELDS.items():
        rows = rows_by_kind[kind]
        absent = {name: sum(1 for row in rows if name not in row) for name in fields}
        if any(absent.values()):
            missing[kind] = absent
    return {
        "method_schema_version": artifact.manifest.get("method_schema_version"),
        "prompt_versions": artifact.manifest.get("prompt_versions"),
        "packing_config": artifact.manifest.get("packing_config"),
        "rows_missing_new_fields": missing,
        "orphan_rows": {key: count for key, count in artifact.orphans.items() if count},
    }


# ---------------------------------------------------------------------------
# deterministic checks
# ---------------------------------------------------------------------------


def load_parse_reply() -> tuple[Callable[..., Any] | None, str | None]:
    """The evaluator's parser, imported the way stage_04_conversations and the
    eval tests do (eval/ on sys.path — it is stdlib-only). Returns the reason
    instead of raising so QA still reports every other metric."""
    eval_dir = DATA_PIPELINE_DIR.parent / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    try:
        from action_parser import parse_sequential_reply  # noqa: PLC0415 - optional
    except Exception as exc:  # pragma: no cover - importability is asserted in tests
        return None, f"{type(exc).__name__}: {exc}"
    return parse_sequential_reply, None


def _parse_one(parse_reply: Callable[..., Any], text: str) -> str | None:
    expected = "checkpoint" if text.strip().startswith("<checkpoint>") else "action"
    try:
        parse_reply(text, expected=expected)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def check_parser(artifact: Artifact, parse_reply: Callable[..., Any] | None,
                 reason: str | None) -> dict[str, Any]:
    """Round-trip every assistant-visible string the artifact publishes."""
    if parse_reply is None:
        return {"status": "skipped", "reason": f"eval/action_parser unimportable ({reason})"}
    failures: list[dict[str, Any]] = []
    n_actions = 0
    n_checkpoints = 0
    for day in artifact.days:
        for index, event in enumerate(day.events):
            text = str(event.get("assistant_action") or "")
            if not text:
                failures.append({"day_tag": day.day_tag, "event_index": index,
                                 "kind": "action", "error": "no assistant_action"})
                continue
            n_actions += 1
            error = _parse_one(parse_reply, text)
            if error is not None:
                failures.append({"day_tag": day.day_tag, "event_index": index,
                                 "kind": "action", "error": error})
        for row in day.checkpoints:
            n_checkpoints += 1
            error = _parse_one(parse_reply, str(row.get("text") or ""))
            if error is not None:
                failures.append({"day_tag": day.day_tag,
                                 "checkpoint_id": row.get("checkpoint_id"),
                                 "kind": "checkpoint", "error": error})
    return {
        "status": "ran", "n_action_texts": n_actions, "n_checkpoint_texts": n_checkpoints,
        "n_failures": len(failures), "failures": failures[:N_WORST_EXAMPLES],
    }


def check_memory_chain(artifact: Artifact) -> dict[str, Any]:
    breaks: list[dict[str, Any]] = []
    n_links = 0
    n_snapshots = 0
    for day in artifact.days:
        position = day.position
        anchors = [str(row.get("anchor_semantic_event_id") or "") for row in day.snapshots]
        covered = {anchor for anchor in anchors if anchor in position}
        for event_id in position:
            if event_id not in covered:
                breaks.append({"day_tag": day.day_tag, "kind": "event_without_snapshot",
                               "semantic_event_id": event_id})
        for anchor in anchors:
            if anchor not in position:
                breaks.append({"day_tag": day.day_tag, "kind": "snapshot_without_event",
                               "semantic_event_id": anchor})
        previous: str | None = None
        for order, row in enumerate(day.snapshots):
            n_snapshots += 1
            index = day.anchor_index(row)
            if index != order:
                breaks.append({"day_tag": day.day_tag, "kind": "non_contiguous_anchor",
                               "expected_index": order, "anchor_event_index": index})
            memory_after = str(row.get("memory_after") or "")
            if not memory_after.strip():
                breaks.append({"day_tag": day.day_tag, "kind": "empty_memory_after",
                               "event_index": index})
            if previous is not None:
                n_links += 1
                if str(row.get("memory_before") or "") != previous:
                    breaks.append({"day_tag": day.day_tag, "kind": "chain_break",
                                   "event_index": index,
                                   "memory_before_head": str(row.get("memory_before") or "")[:120],
                                   "previous_memory_after_head": previous[:120]})
            previous = memory_after
    return {
        "status": "ran", "n_snapshots": n_snapshots, "n_links": n_links,
        "n_breaks": len(breaks), "breaks": breaks[:N_WORST_EXAMPLES],
    }


def check_references(artifact: Artifact) -> dict[str, Any]:
    future: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    n_refs = 0
    for day in artifact.days:
        position = day.position
        for kind, rows in (("memory_snapshot", day.snapshots),
                           ("decision", day.decisions),
                           ("checkpoint", day.checkpoints)):
            for row in rows:
                anchor = day.anchor_index(row)
                for ref in row.get("references") or []:
                    n_refs += 1
                    at = position.get(str(ref))
                    where = {"day_tag": day.day_tag, "kind": kind,
                             "anchor_event_index": anchor, "reference": str(ref)}
                    if at is None:
                        unresolved.append(where)
                    elif at > anchor:
                        future.append({**where, "reference_event_index": at})
    return {
        "status": "ran", "n_references": n_refs,
        "n_future": len(future), "n_unresolved": len(unresolved),
        "future": future[:N_WORST_EXAMPLES], "unresolved": unresolved[:N_WORST_EXAMPLES],
    }


def _arguments(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {}
    inner = call.get("arguments")
    return inner if isinstance(inner, dict) else call


def _keys(arguments: dict[str, Any]) -> list[str]:
    raw = arguments.get("keys")
    if isinstance(raw, str):
        raw = [raw]
    elif not isinstance(raw, (list, tuple)):
        raw = []
    return [str(key).strip().casefold() for key in raw if str(key).strip()]


def leak_candidates(calls: Any) -> set[str]:
    """Strings only this action packet reveals: typed-text tokens and key combos."""
    out: set[str] = set()
    for call in calls if isinstance(calls, (list, tuple)) else []:
        arguments = _arguments(call)
        action = str(arguments.get("action") or "")
        if action == "type":
            out.update(match.group(0).casefold()
                       for match in _LEAK_TOKEN_RE.finditer(str(arguments.get("text") or "")))
        elif action == "key":
            keys = _keys(arguments)
            if len(keys) > 1:
                out.add("+".join(keys))
            out.update(key for key in keys if len(key) >= LEAK_MIN_CHARS)
        elif action in ("key_down", "key_up"):
            key = str(arguments.get("key") or "").strip().casefold()
            if len(key) >= LEAK_MIN_CHARS:
                out.add(key)
    return out


def check_leaks(artifact: Artifact) -> dict[str, Any]:
    """A text leaks if it contains a string that only a LATER action packet
    reveals — one that never appears in an earlier-or-current packet nor in the
    goal texts active there (goal text is hindsight the recipe grants on purpose).

    ``memory_after``/``thought`` at event i may know packet i (the action is
    recorded as intended, not completed), so the known prefix includes i; a
    checkpoint projected at anchor i has exactly the same horizon.
    """
    hits: list[dict[str, Any]] = []
    by_kind: dict[str, int] = {"memory_after": 0, "thought": 0, "checkpoint": 0}
    n_texts = 0
    n_candidates = 0
    for day in artifact.days:
        n = len(day.events)
        packets = [leak_candidates(row.get("tool_calls")) for row in day.events]
        n_candidates += len(set().union(*packets)) if packets else 0
        known: list[set[str]] = []
        seen: set[str] = set()
        for candidates in packets:
            seen = seen | candidates
            known.append(seen)
        later: list[set[str]] = [set()] * n
        tail: set[str] = set()
        for index in range(n - 1, -1, -1):
            later[index] = tail
            tail = tail | packets[index]
        goal_blob = [
            " ".join(str(node.get("text") or "") for node in day.covering_goals(index)).casefold()
            for index in range(n)
        ]
        snapshot_by_index = {day.anchor_index(row): row for row in day.snapshots}
        texts: list[tuple[int, str, str, str]] = []
        for index in range(n):
            snapshot = snapshot_by_index.get(index)
            if snapshot is None:
                continue
            texts.append((index, "memory_after", str(snapshot.get("memory_after") or ""), ""))
            thought = str(snapshot.get("thought") or "").strip()
            if thought:
                texts.append((index, "thought", thought, ""))
        for row in day.checkpoints:
            texts.append((day.anchor_index(row), "checkpoint", str(row.get("text") or ""),
                          str(row.get("checkpoint_id") or "")))
        for index, kind, text, row_id in texts:
            if not text or not 0 <= index < n:
                continue
            n_texts += 1
            unknown = {candidate for candidate in later[index] - known[index]
                       if candidate not in goal_blob[index]}
            lowered = text.casefold()
            leaked = sorted(candidate for candidate in unknown if candidate in lowered)
            if leaked:
                by_kind[kind] += 1
                hits.append({"day_tag": day.day_tag, "event_index": index, "kind": kind,
                             "row_id": row_id, "n_leaked": len(leaked),
                             "leaked": leaked[:N_WORST_EXAMPLES], "excerpt": text[:240]})
    hits.sort(key=lambda row: -int(row["n_leaked"]))
    return {
        "status": "ran", "n_texts": n_texts, "n_candidate_strings": n_candidates,
        "n_texts_with_leak": len(hits),
        "leak_rate": round(len(hits) / n_texts, 6) if n_texts else 0.0,
        "n_leaks_by_kind": by_kind, "worst": hits[:N_WORST_EXAMPLES],
    }


def _content_words(text: str) -> list[str]:
    return [word for word in _WORD_RE.findall(str(text or "").casefold())
            if word not in _STOPWORDS]


def _bigrams(words: list[str]) -> set[tuple[str, str]]:
    return set(zip(words, words[1:]))


def _action_phrase(calls: Any) -> str:
    parts: list[str] = []
    for call in calls if isinstance(calls, (list, tuple)) else []:
        arguments = _arguments(call)
        parts.append(str(arguments.get("action") or "").replace("_", " "))
        parts.extend(str(arguments.get(name) or "") for name in ("button", "key", "text"))
        parts.extend(_keys(arguments))
    return " ".join(parts)


def parrot_score(thought: str, calls: Any) -> dict[str, float]:
    """How much of the thought is just the action restated. 1.0 = pure parrot."""
    words = _content_words(thought)
    action = _content_words(_action_phrase(calls))
    unigram = len(set(words) & set(action)) / len(set(words)) if words else 0.0
    thought_bigrams = _bigrams(words)
    bigram = (len(thought_bigrams & _bigrams(action)) / len(thought_bigrams)
              if thought_bigrams else 0.0)
    return {"unigram": round(unigram, 4), "bigram": round(bigram, 4)}


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered), "min": ordered[0], "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1) + 0.5))],
    }


def check_thoughts(artifact: Artifact) -> dict[str, Any]:
    n_events = 0
    n_thoughts = 0
    n_motor = 0
    n_decision = 0
    n_unknown_boundary = 0
    n_agreed_known = 0
    n_divergent = 0
    motor_thoughts: list[dict[str, Any]] = []
    over_budget: list[dict[str, Any]] = []
    word_counts: list[int] = []
    parrots: list[dict[str, Any]] = []
    gates: dict[str, int] = {}
    for day in artifact.days:
        for row in day.snapshots:
            n_events += 1
            index = day.anchor_index(row)
            thought = str(row.get("thought") or "").strip()
            boundary = row.get("is_decision_boundary")
            if boundary is None:
                n_unknown_boundary += 1
            elif bool(boundary):
                n_decision += 1
            else:
                n_motor += 1
                if thought:
                    motor_thoughts.append({"day_tag": day.day_tag, "event_index": index,
                                           "thought": thought[:240]})
            agreed = row.get("agreed")
            if agreed is not None:
                n_agreed_known += 1
                n_divergent += int(not bool(agreed))
            if not thought:
                continue
            n_thoughts += 1
            words = len(_words(thought))
            word_counts.append(words)
            if words > THOUGHT_MAX_WORDS:
                over_budget.append({"day_tag": day.day_tag, "event_index": index,
                                    "n_words": words, "thought": thought[:240]})
            score = parrot_score(thought, row.get("upcoming_tool_calls"))
            parrots.append({"day_tag": day.day_tag, "event_index": index, **score})
        for row in day.decisions:
            gates[str(row.get("gate") or "unknown")] = gates.get(
                str(row.get("gate") or "unknown"), 0) + 1
    parrots.sort(key=lambda row: (-float(row["unigram"]), -float(row["bigram"])))
    return {
        "status": "ran", "n_events": n_events, "n_thoughts": n_thoughts,
        "density": round(n_thoughts / n_events, 6) if n_events else 0.0,
        "n_decision_boundaries": n_decision, "n_motor_events": n_motor,
        "n_unknown_boundary": n_unknown_boundary,
        "motor_thought_density": (round(len(motor_thoughts) / n_motor, 6)
                                 if n_motor else None),
        "n_motor_thoughts": len(motor_thoughts),
        "motor_thoughts": motor_thoughts[:N_WORST_EXAMPLES],
        "word_counts": _distribution(word_counts),
        "word_budget": THOUGHT_MAX_WORDS, "n_over_budget": len(over_budget),
        "over_budget": over_budget[:N_WORST_EXAMPLES],
        "n_agreement_gated": n_agreed_known, "n_divergent": n_divergent,
        "divergence_rate": (round(n_divergent / n_agreed_known, 6)
                            if n_agreed_known else None),
        "gate_counts": gates,
        "parrot_mean_unigram": (round(statistics.fmean([row["unigram"] for row in parrots]), 4)
                                if parrots else None),
        "parrot_mean_bigram": (round(statistics.fmean([row["bigram"] for row in parrots]), 4)
                               if parrots else None),
        "parrot_worst": parrots[:N_WORST_EXAMPLES],
    }


def _checkpoint_values(row: dict[str, Any]) -> dict[str, str]:
    """The seven field bodies, from ``values`` when published and otherwise
    recovered from the rendered ``text``."""
    values = row.get("values")
    if isinstance(values, dict) and values:
        return {name: " ".join(str(values.get(name) or "").split())
                for name in CHECKPOINT_FIELDS}
    text = str(row.get("text") or "")
    out: dict[str, str] = {}
    for index, name in enumerate(CHECKPOINT_FIELDS):
        start = text.find(f"## {name}")
        if start < 0:
            out[name] = ""
            continue
        start += len(f"## {name}")
        rest = text[start:]
        stops = [rest.find(f"## {later}") for later in CHECKPOINT_FIELDS[index + 1:]]
        stops.append(rest.find("</checkpoint>"))
        cuts = [stop for stop in stops if stop >= 0]
        out[name] = " ".join(rest[: min(cuts)].split()) if cuts else " ".join(rest.split())
    return out


def check_checkpoints(artifact: Artifact) -> dict[str, Any]:
    total_words: list[int] = []
    over_budget: list[dict[str, Any]] = []
    empty_fields: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    n_day_final = 0
    n_unknown_day_final = 0
    for day in artifact.days:
        by_chain: dict[str, list[dict[str, Any]]] = {}
        for row in day.checkpoints:
            values = _checkpoint_values(row)
            words = sum(len(_words(values[name])) for name in CHECKPOINT_FIELDS)
            total_words.append(words)
            missing = [name for name in CHECKPOINT_FIELDS if not values[name].strip()]
            if missing:
                empty_fields.append({"day_tag": day.day_tag,
                                     "checkpoint_id": row.get("checkpoint_id"),
                                     "empty": missing})
            if words > CHECKPOINT_MAX_WORDS:
                over_budget.append({"day_tag": day.day_tag,
                                    "checkpoint_id": row.get("checkpoint_id"),
                                    "n_words": words})
            if "is_day_final" not in row:
                n_unknown_day_final += 1
            elif bool(row.get("is_day_final")):
                n_day_final += 1
            key = str(row.get("packing_config_hash") or "unknown")
            by_chain.setdefault(key, []).append(
                {"checkpoint_id": row.get("checkpoint_id"),
                 "event_index": day.anchor_index(row),
                 "completed": values["Completed"]})
        for key, anchors in by_chain.items():
            anchors.sort(key=lambda row: int(row["event_index"]))
            counts = [len(_words(row["completed"])) for row in anchors]
            restated = [
                {"day_tag": day.day_tag, "kind": "restates_previous",
                 "from": anchors[index - 1]["checkpoint_id"],
                 "to": anchors[index]["checkpoint_id"]}
                for index in range(1, len(anchors))
                if anchors[index - 1]["completed"]
                and anchors[index - 1]["completed"] in anchors[index]["completed"]
            ]
            monotonic = len(counts) >= FOLDING_MIN_CHAIN and all(
                later > earlier for earlier, later in zip(counts, counts[1:]))
            ratio = (counts[-1] / counts[0]) if counts and counts[0] else None
            regressions.extend(restated)
            if monotonic and ratio is not None and ratio > FOLDING_GROWTH_RATIO:
                regressions.append({
                    "day_tag": day.day_tag, "kind": "monotonic_growth",
                    "packing_config_hash": key, "completed_word_counts": counts,
                    "growth_ratio": round(ratio, 3)})
            chains.append({
                "day_tag": day.day_tag, "packing_config_hash": key,
                "n_anchors": len(anchors), "completed_word_counts": counts,
                "monotonic_growth": monotonic,
                "growth_ratio": round(ratio, 3) if ratio is not None else None,
                "n_restatements": len(restated)})
    return {
        "status": "ran", "n_checkpoints": len(total_words),
        "n_day_final": n_day_final, "n_unknown_day_final": n_unknown_day_final,
        "word_budget": CHECKPOINT_MAX_WORDS, "total_words": _distribution(total_words),
        "n_over_budget": len(over_budget), "over_budget": over_budget[:N_WORST_EXAMPLES],
        "n_incomplete_fields": len(empty_fields),
        "incomplete_fields": empty_fields[:N_WORST_EXAMPLES],
        "n_chains": len(chains), "chains": chains[:N_WORST_EXAMPLES],
        "n_folding_regressions": len(regressions),
        "folding_regressions": regressions[:N_WORST_EXAMPLES],
    }


# ---------------------------------------------------------------------------
# stage 04 chat.jsonl
# ---------------------------------------------------------------------------


def _assistant_texts(record: dict[str, Any]) -> list[tuple[int, str]]:
    return [(index, str((message.get("content") or [{}])[0].get("text") or ""))
            for index, message in enumerate(record.get("messages") or [])
            if message.get("role") == "assistant"]


def _first_user_text(record: dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        if message.get("role") != "user":
            continue
        for block in message.get("content") or []:
            if block.get("type") == "text":
                return str(block.get("text") or "")
    return ""


def _embedded_checkpoint(text: str) -> str | None:
    start = text.find("<checkpoint>")
    end = text.find("</checkpoint>")
    if start < 0 or end < start:
        return None
    return text[start: end + len("</checkpoint>")]


def _n_images(record: dict[str, Any]) -> int:
    return sum(1 for message in record.get("messages") or []
               for block in message.get("content") or []
               if block.get("type") == "image")


def _capacity_for(chat_path: Path) -> int | None:
    for name in ("manifest.json", "conversations_summary.json"):
        path = chat_path.parent / name
        if not path.is_file():
            continue
        config = json.loads(path.read_text()).get("packing_config")
        if isinstance(config, dict) and config.get("capacity") is not None:
            return int(config["capacity"])
    return None


def check_stage04(chat_path: Path, parse_reply: Callable[..., Any] | None,
                  reason: str | None) -> dict[str, Any]:
    if not chat_path.is_file():
        raise SystemExit(f"--stage04-chat {chat_path} is not a file")
    records = _rows(chat_path)
    if not records:
        raise SystemExit(f"--stage04-chat {chat_path} has no records")
    capacity = _capacity_for(chat_path)
    parse_failures: list[dict[str, Any]] = []
    image_mismatches: list[dict[str, Any]] = []
    over_capacity: list[dict[str, Any]] = []
    n_assistant = 0
    n_undeclared_images = 0
    episodes: dict[str, list[dict[str, Any]]] = {}
    n_cross_day = 0
    for record in records:
        cid = str(record.get("conversation_id") or "")
        if parse_reply is not None:
            for turn, text in _assistant_texts(record):
                n_assistant += 1
                error = _parse_one(parse_reply, text)
                if error is not None:
                    parse_failures.append({"conversation_id": cid, "turn": turn,
                                           "error": error})
        images = _n_images(record)
        # Only ``n_images`` is a claim about images; the legacy ``n_frames``
        # counted semantic events, so its absence is tolerated, not flagged.
        declared = record.get("n_images")
        if declared is None:
            n_undeclared_images += 1
        elif int(declared) != images:
            image_mismatches.append({"conversation_id": cid, "declared": int(declared),
                                     "actual": images})
        if capacity is not None and images > capacity:
            over_capacity.append({"conversation_id": cid, "n_images": images,
                                  "capacity": capacity})
        if record.get("cross_day"):
            n_cross_day += 1
            continue
        episode = str(record.get("episode_id") or cid.rsplit("_s", 1)[0])
        episodes.setdefault(episode, []).append(record)
    handoff_breaks: list[dict[str, Any]] = []
    n_handoffs = 0
    for episode, rows in sorted(episodes.items()):
        rows.sort(key=lambda row: (_int(row.get("segment_index"), 0),
                                  str(row.get("conversation_id") or "")))
        for earlier, later in zip(rows, rows[1:]):
            out_text = next((text for _turn, text in reversed(_assistant_texts(earlier))
                             if text.strip().startswith("<checkpoint>")), None)
            in_text = _embedded_checkpoint(_first_user_text(later))
            if out_text is None and in_text is None:
                continue
            n_handoffs += 1
            if out_text != in_text:
                handoff_breaks.append({
                    "episode_id": episode,
                    "from": earlier.get("conversation_id"),
                    "to": later.get("conversation_id"),
                    "checkpoint_out_head": (out_text or "")[:120],
                    "checkpoint_in_head": (in_text or "")[:120]})
    return {
        "status": "ran", "chat": str(chat_path), "n_records": len(records),
        "n_cross_day_records": n_cross_day, "capacity": capacity,
        "parse": ({"status": "skipped",
                   "reason": f"eval/action_parser unimportable ({reason})"}
                  if parse_reply is None else
                  {"status": "ran", "n_assistant_turns": n_assistant,
                   "n_failures": len(parse_failures),
                   "failures": parse_failures[:N_WORST_EXAMPLES]}),
        "n_handoffs": n_handoffs, "n_handoff_breaks": len(handoff_breaks),
        "handoff_breaks": handoff_breaks[:N_WORST_EXAMPLES],
        "n_image_count_mismatches": len(image_mismatches),
        "image_count_mismatches": image_mismatches[:N_WORST_EXAMPLES],
        "n_records_without_n_images": n_undeclared_images,
        "max_images_per_record": max((_n_images(row) for row in records), default=0),
        "n_over_capacity": len(over_capacity),
        "over_capacity": over_capacity[:N_WORST_EXAMPLES],
    }


# ---------------------------------------------------------------------------
# LLM check: resumability probe
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "env"))


def _goal_text(day: DayView, index: int) -> str:
    """The GOAL a resumed record would carry: the covering mid node, else the
    covering long node, else the proactive framing (Stage 04's own fallback)."""
    for level in ("mid", "long"):
        covering = [node for node in day.covering_goals(index)
                    if str(node.get("level")) == level]
        if covering and str(covering[0].get("text") or "").strip():
            return str(covering[0]["text"])
    return PROACTIVE_GOAL_TEXT


def sample_boundary_checkpoints(artifact: Artifact,
                               sample: int) -> list[tuple[DayView, dict[str, Any]]]:
    """Deterministic hash-ordered subset of the non-day-final checkpoints (a
    day-final checkpoint has no continuation in the same day to resume into)."""
    rows = [(day, row) for day, row in artifact.all_checkpoints
            if not bool(row.get("is_day_final"))]
    rows.sort(key=lambda pair: hashlib.sha256(
        str(pair[1].get("checkpoint_id") or "").encode()).hexdigest())
    return rows[: max(0, sample)]


def run_resumability(artifact: Artifact, labeler: Any, *, sample: int, calls_dir: Path,
                     no_cache: bool, frame_height: int, jpeg_quality: int) -> dict[str, Any]:
    selected = sample_boundary_checkpoints(artifact, sample)
    results: list[dict[str, Any]] = []
    n_pass = 0
    n_error = 0
    prompt = system_prompt()
    for day, row in selected:
        index = day.anchor_index(row)
        checkpoint = str(row.get("text") or "")
        goal = _goal_text(day, index)
        packets = [event.get("tool_calls")
                   for event in day.events[index: index + PROBE_NEXT_PACKETS]]
        record: dict[str, Any] = {
            "day_tag": day.day_tag, "checkpoint_id": row.get("checkpoint_id"),
            "anchor_event_index": index, "n_next_packets": len(packets),
        }
        try:
            if not 0 <= index < len(day.events):
                raise ValueError(f"checkpoint anchor {index} is outside the day")
            key = hashlib.sha256(json.dumps(
                [QA_PROMPT_VERSIONS, checkpoint, goal, packets],
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
            images = frames_to_data_urls(
                [str(day.events[index]["image"])], frame_height, jpeg_quality)
            answer = labeler.call_text(
                prompt,
                goal_conditioning(goal, checkpoint) + "\n\n" + PROBE_REQUEST,
                images=images,
                cache_path=calls_dir / f"probe_{_slug(row.get('checkpoint_id'))}_{key}.txt",
                no_cache=no_cache,
            )
            verdict = labeler.call_json(
                JUDGE_SYSTEM,
                JUDGE_TEMPLATE.format(
                    goal=goal, checkpoint=checkpoint, answer=answer,
                    packets=json.dumps(packets, ensure_ascii=False, indent=2)),
                cache_path=calls_dir / f"judge_{_slug(row.get('checkpoint_id'))}_{key}.txt",
                no_cache=no_cache,
            )
            compatible = bool(verdict.get("compatible"))
            n_pass += int(compatible)
            record.update({"compatible": compatible,
                           "reason": " ".join(str(verdict.get("reason") or "").split()),
                           "answer": answer[:600]})
        except Exception as exc:
            n_error += 1
            record.update({"compatible": False, "error": f"{type(exc).__name__}: {exc}"})
        results.append(record)
    n_probed = len(results)
    return {
        "status": "ran", "prompt_versions": QA_PROMPT_VERSIONS,
        "n_eligible": len([1 for _day, row in artifact.all_checkpoints
                           if not bool(row.get("is_day_final"))]),
        "n_probed": n_probed, "n_pass": n_pass, "n_fail": n_probed - n_pass,
        "n_error": n_error,
        "pass_rate": round(n_pass / n_probed, 6) if n_probed else None,
        "pass_rate_threshold": PROBE_PASS_RATE_MIN,
        "probes": results,
    }


# ---------------------------------------------------------------------------
# report + review draft
# ---------------------------------------------------------------------------


def _clean(check: dict[str, Any], *keys: str) -> bool:
    return check.get("status") == "ran" and not any(int(check.get(key) or 0) for key in keys)


def build_report(artifact: Artifact, checks: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []

    def flag(name: str, count: Any, detail: str) -> None:
        if int(count or 0):
            violations.append({"check": name, "count": int(count), "detail": detail})

    flag("parser", checks["parser"].get("n_failures"), "assistant text failed parse_sequential_reply")
    flag("memory_chain", checks["memory_chain"].get("n_breaks"), "rolling memory chain is broken")
    flag("references", checks["references"].get("n_future"), "reference points at a future event")
    flag("references", checks["references"].get("n_unresolved"), "reference does not resolve")
    flag("leaks", checks["leaks"].get("n_texts_with_leak"), "text contains future-only strings")
    flag("thoughts", checks["thoughts"].get("n_motor_thoughts"), "motor event carries a thought")
    flag("thoughts", checks["thoughts"].get("n_over_budget"),
         f"thought exceeds THOUGHT_MAX_WORDS={THOUGHT_MAX_WORDS}")
    flag("checkpoints", checks["checkpoints"].get("n_over_budget"),
         f"checkpoint exceeds CHECKPOINT_MAX_WORDS={CHECKPOINT_MAX_WORDS}")
    flag("checkpoints", checks["checkpoints"].get("n_incomplete_fields"),
         "checkpoint has an empty field")
    flag("checkpoints", checks["checkpoints"].get("n_folding_regressions"),
         "Completed grows instead of folding")
    stage04 = checks["stage04"]
    if stage04.get("status") == "ran":
        flag("stage04", (stage04.get("parse") or {}).get("n_failures"),
             "stage-04 assistant turn failed parse_sequential_reply")
        flag("stage04", stage04.get("n_handoff_breaks"), "checkpoint handoff is not byte-identical")
        flag("stage04", stage04.get("n_over_capacity"), "record exceeds the packing capacity")
        flag("stage04", stage04.get("n_image_count_mismatches"),
             "record n_images disagrees with its messages")
    return {
        "artifact_dir": str(artifact.dir.resolve()),
        "method": artifact.manifest.get("method"),
        "schema": schema_tolerance(artifact),
        "counts": {
            "n_days": len(artifact.days),
            "n_semantic_events": sum(len(day.events) for day in artifact.days),
            "n_memory_snapshots": sum(len(day.snapshots) for day in artifact.days),
            "n_checkpoints": sum(len(day.checkpoints) for day in artifact.days),
            "n_decisions": sum(len(day.decisions) for day in artifact.days),
            "n_goal_nodes": sum(len(day.goals) for day in artifact.days),
            "n_mission_links": len(artifact.mission_links),
        },
        "checks": checks,
        "violations": violations,
        "ok": not violations,
    }


def draft_review(report: dict[str, Any]) -> dict[str, Any]:
    """Auto-draft what a metric can decide; leave human judgement null.

    NEVER fills ``reviewed_by`` and never sets the three judgement gates, so the
    emitted file cannot satisfy stage_annotate._require_pilot_review.
    """
    checks = report["checks"]
    leaks, chain, refs = checks["leaks"], checks["memory_chain"], checks["references"]
    thoughts, checkpoints = checks["thoughts"], checks["checkpoints"]
    parser, stage04, resume = checks["parser"], checks["stage04"], checks["resumability"]

    provenance_ok = (_clean(chain, "n_breaks")
                     and _clean(refs, "n_future", "n_unresolved")
                     and _clean(leaks, "n_texts_with_leak"))
    parser_ok: bool | None = None
    if parser.get("status") == "ran":
        parser_ok = not int(parser.get("n_failures") or 0)
        if stage04.get("status") == "ran":
            parser_ok = bool(parser_ok
                             and _clean(stage04.get("parse") or {}, "n_failures")
                             and not int(stage04.get("n_handoff_breaks") or 0)
                             and not int(stage04.get("n_over_capacity") or 0))
    pass_rate = resume.get("pass_rate") if resume.get("status") == "ran" else None
    checkpoints_ok = None if pass_rate is None else bool(pass_rate >= PROBE_PASS_RATE_MIN)

    draft = {
        "reviewed_by": None,
        "artifact_dir": report["artifact_dir"],
        "qa_report": "qa_report.json",
        "note": ("Auto-drafted by qa_sequential_goal_memory.py. A human must read "
                 "the metrics, set the null gates, and fill reviewed_by; until then "
                 "this file cannot unblock a full run."),
        "goal_grounding": None,
        "causal_thoughts": None,
        "cross_day_links": None,
        "checkpoints": checkpoints_ok,
        "action_provenance": provenance_ok,
        "parser_validity": parser_ok,
        "basis": {
            "goal_grounding": (
                f"human: {report['counts']['n_goal_nodes']} goal nodes over "
                f"{report['counts']['n_semantic_events']} events; no metric decides "
                "whether goal text is grounded and eval-instruction shaped"),
            "causal_thoughts": (
                f"human: density {thoughts.get('density')}, motor density "
                f"{thoughts.get('motor_thought_density')}, divergence rate "
                f"{thoughts.get('divergence_rate')}, mean parrot unigram "
                f"{thoughts.get('parrot_mean_unigram')}, "
                f"{thoughts.get('n_over_budget')} over the word budget"),
            "cross_day_links": (
                f"human: {report['counts']['n_mission_links']} mission link(s) over "
                f"{report['counts']['n_days']} day(s); relations need reading"),
            "checkpoints": (
                f"probe pass rate {pass_rate} vs threshold {PROBE_PASS_RATE_MIN} on "
                f"{resume.get('n_probed')} boundary checkpoint(s), "
                f"{resume.get('n_error')} error(s); "
                f"{checkpoints.get('n_over_budget')} over budget, "
                f"{checkpoints.get('n_folding_regressions')} folding regression(s)"
                if pass_rate is not None else
                f"null: probe skipped ({resume.get('reason')})"),
            "action_provenance": (
                f"{chain.get('n_breaks')} memory-chain break(s), "
                f"{refs.get('n_future')} future reference(s), "
                f"{refs.get('n_unresolved')} unresolved reference(s), "
                f"leak rate {leaks.get('leak_rate')} over {leaks.get('n_texts')} text(s)"),
            "parser_validity": (
                f"{parser.get('n_failures')} artifact parse failure(s) over "
                f"{parser.get('n_action_texts')} action + "
                f"{parser.get('n_checkpoint_texts')} checkpoint text(s); stage-04: "
                f"{stage04.get('status')}"
                if parser.get("status") == "ran" else
                f"null: {parser.get('reason')}"),
        },
    }
    if would_pass_full_run_gate(draft):
        raise RuntimeError("refusing to emit a review_draft.json that unblocks a full run")
    return draft


def would_pass_full_run_gate(review: dict[str, Any]) -> bool:
    """Mirror of stage_annotate._require_pilot_review's acceptance condition."""
    return (all(review.get(gate) is True for gate in FULL_RUN_GATES)
            and bool(str(review.get("reviewed_by") or "").strip()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    normalize_dashed_argv()  # accept pmanager's --foo_bar=value arg form
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True,
                   help="A published sequential_goal_memory artifact dir (stage 03b).")
    p.add_argument("--stage04-chat", type=Path, default=None,
                   help="Optional stage-04 chat.jsonl to round-trip and check handoffs.")
    p.add_argument("--sample", type=int, default=20,
                   help="Boundary checkpoints to run the resumability probe on.")
    p.add_argument("--no-llm", action="store_true",
                   help="Deterministic checks only; the probe/judge gate stays null.")
    # Labeler config, mirroring stage_annotate's flags.
    p.add_argument("--model", default=None,
                   help="Labeler model for the probe/judge. Unset: env LABELER_MODEL.")
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--vlm-frame-height", type=int, default=720)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def run_qa(artifact_dir: Path, *, stage04_chat: Path | None, sample: int, no_llm: bool,
           labeler: Any | None, frame_height: int = 720, jpeg_quality: int = 80,
           no_cache: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every check, then the report and the review draft, written to the artifact."""
    artifact = load_artifact(artifact_dir)
    parse_reply, reason = load_parse_reply()
    checks = {
        "parser": check_parser(artifact, parse_reply, reason),
        "memory_chain": check_memory_chain(artifact),
        "references": check_references(artifact),
        "leaks": check_leaks(artifact),
        "thoughts": check_thoughts(artifact),
        "checkpoints": check_checkpoints(artifact),
        "stage04": ({"status": "skipped", "reason": "no --stage04-chat"}
                    if stage04_chat is None else
                    check_stage04(stage04_chat, parse_reply, reason)),
        "resumability": (
            {"status": "skipped",
             "reason": "--no-llm" if no_llm else "no labeler configured"}
            if no_llm or labeler is None else
            run_resumability(artifact, labeler, sample=sample,
                             calls_dir=artifact.dir / "qa_calls", no_cache=no_cache,
                             frame_height=frame_height, jpeg_quality=jpeg_quality)),
    }
    report = build_report(artifact, checks)
    draft = draft_review(report)
    write_json(artifact.dir / "qa_report.json", report)
    write_json(artifact.dir / "review_draft.json", draft)
    return report, draft


def main() -> None:
    args = parse_args()
    labeler = None
    if not args.no_llm:
        labeler = Labeler(LabelerConfig.from_env(
            model=args.model or labeler_model(),
            reasoning_effort=args.reasoning_effort, temperature=args.temperature,
        ))
    report, draft = run_qa(
        args.artifact, stage04_chat=args.stage04_chat, sample=args.sample,
        no_llm=args.no_llm, labeler=labeler, frame_height=args.vlm_frame_height,
        jpeg_quality=args.jpeg_quality, no_cache=args.no_cache,
    )
    counts = report["counts"]
    print(f"[qa] {counts['n_days']} day(s), {counts['n_semantic_events']} events, "
          f"{counts['n_checkpoints']} checkpoints -> {args.artifact / 'qa_report.json'}",
          flush=True)
    for violation in report["violations"]:
        print(f"[qa] VIOLATION {violation['check']}: {violation['detail']} "
              f"(x{violation['count']})", flush=True)
    gates = " ".join(f"{gate}={draft[gate]}" for gate in FULL_RUN_GATES)
    print(f"[qa] draft gates: {gates} (reviewed_by=None) -> "
          f"{args.artifact / 'review_draft.json'}", flush=True)
    if not report["ok"]:
        print(f"[qa] {len(report['violations'])} violation kind(s); see qa_report.json",
              flush=True)


if __name__ == "__main__":
    main()
