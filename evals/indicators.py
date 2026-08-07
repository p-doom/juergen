"""Diagnostics that ride the trace.

A cell can flip for the wrong reason, so pass/fail is not sufficient; the four
failure-mode indicators below say whether a defect is actually gone. As
`@vf.metric` they are computed in-process from the same step records the harness
already writes, so offline `traces.jsonl` re-scoring gets them too (a metric
declares no `runtime`, so it is never skipped).

  A  literal `\\n` / `\\r` / `\\t` inside a typed string — the s900 Return defect;
     the model writes an escape sequence instead of a key transition.
  B  same-action type + submit — the target composition (`type("...") +Return -Return`).
  C  premature TERMINATE — the model declared done before the state changed.
  D  submission in a cell whose success *requires* not submitting — the
     over-generalisation the A-fix risks.

`MouseIndicators` covers the relative-move diagnostics: on-lattice rate,
delta-magnitude histogram, in-bbox rate, terminate rate.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import verifiers.v1 as vf

from evals.tasks import RESULT_KEY

__all__ = [
    "DIGIT_LATTICE",
    "SUBMIT_KEYS",
    "FailureModeIndicators",
    "MouseIndicators",
    "delta_histogram",
    "on_lattice",
    "step_records",
    "typed_texts",
]

SUBMIT_KEYS = frozenset({"Return", "Enter", "KpEnter", "NumpadEnter", "ENTER", "RETURN"})

DIGIT_LATTICE = frozenset({0, 1, 10, 100})
"""The observed output support of the collapsed relative-delta checkpoints:
`{0, ±1, ±10, ±100}` per axis, 400/400 samples, mode literally `(±10, ±10)` —
14.1 px = hypot(10, 10). On-lattice rate is therefore a collapse detector: a
healthy relative policy emits arbitrary integers, a collapsed one emits digits."""

_LITERAL_ESCAPES = ("\\n", "\\r", "\\t")


def step_records(trace: vf.Trace) -> list[dict[str, Any]]:
    result = trace.info.get(RESULT_KEY) or {}
    steps = result.get("steps_detail")
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def _elements(parsed: Any) -> list[dict[str, Any]]:
    """The bare-token grammars' ordered element list, as dicts.

    `deltatype_v2` / `compact_raw` serialise an element to
    `{"kind": "event", "name": ..., "pressed": ...}`,
    `{"kind": "type", "text": ...}` or `{"kind": "move", "delta": [dx, dy]}`. Cached
    traces from before the grammar consolidation carry the older `(kind, value)`
    pair form, so both are accepted — a metric that silently returned 0 on old
    traces would look like a fixed defect.
    """
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("elements")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append(_legacy_element(str(item[0]), item[1]))
    return out


def _legacy_element(kind: str, value: Any) -> dict[str, Any]:
    if kind == "type":
        return {"kind": "type", "text": str(value)}
    if kind == "move":
        return {"kind": "move", "delta": list(value or ())}
    if kind == "event" and isinstance(value, (list, tuple)) and len(value) == 2:
        return {"kind": "event", "name": str(value[1]), "pressed": value[0] == "press"}
    return {"kind": kind}


def _calls(parsed: Any) -> list[dict[str, Any]]:
    """The tool-call grammars' calls (`native_absolute`, `move_rel`).

    Each call serialises to its `computer_use` arguments dict, so `action`, `text`,
    `keys` and `coordinate` read the same way whether the turn carried one call or
    several.
    """
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("calls")
    if isinstance(raw, (list, tuple)):
        return [call for call in raw if isinstance(call, dict)]
    return [parsed] if "action" in parsed else []


def typed_texts(parsed: Any) -> list[str]:
    """Every string this action types, under either grammar family."""
    out = [
        str(element.get("text", ""))
        for element in _elements(parsed)
        if element.get("kind") == "type"
    ]
    for call in _calls(parsed):
        if str(call.get("action", "")).strip().lower() == "type":
            text = call.get("text")
            if isinstance(text, str):
                out.append(text)
    return out


def submit_keys(parsed: Any) -> list[str]:
    """Every Return-equivalent key transition this action performs.

    Presses only: a release without a press is not a submission, and counting both
    halves of `+Return -Return` would double every B and D reading.
    """
    out: list[str] = []
    for element in _elements(parsed):
        if element.get("kind") != "event" or not element.get("pressed"):
            continue
        name = str(element.get("name", ""))
        if name in SUBMIT_KEYS:
            out.append(name)
    for call in _calls(parsed):
        if str(call.get("action", "")).strip().lower() not in {"key", "key_down"}:
            continue
        keys = call.get("keys")
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(keys, (list, tuple)):
            out.extend(str(k) for k in keys if str(k) in SUBMIT_KEYS)
    return out


def deltas(parsed: Any) -> list[tuple[int, int]]:
    """Mouse deltas this action requests, in the grammar's own unit.

    Only meaningful for relative grammars: `deltatype_v2` / `compact_raw` in raw
    pixels, `move_rel` in normalized 0-999. `native_absolute` carries a target
    rather than a delta and reports nothing; differencing consecutive targets to
    invent one would fabricate a distribution.
    """
    if not isinstance(parsed, dict):
        return []
    out: list[tuple[int, int]] = []
    if "dx" in parsed and "dy" in parsed:
        try:
            out.append((int(parsed["dx"]), int(parsed["dy"])))
        except (TypeError, ValueError):
            pass
    for element in _elements(parsed):
        delta = element.get("delta") if element.get("kind") == "move" else None
        if isinstance(delta, (list, tuple)) and len(delta) == 2:
            try:
                out.append((int(delta[0]), int(delta[1])))
            except (TypeError, ValueError):
                pass
    for call in _calls(parsed):
        if str(call.get("action", "")).strip().lower() != "move_rel":
            continue
        coord = call.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            try:
                out.append((int(coord[0]), int(coord[1])))
            except (TypeError, ValueError):
                pass
    return out


def on_lattice(delta: tuple[int, int]) -> bool:
    return abs(delta[0]) in DIGIT_LATTICE and abs(delta[1]) in DIGIT_LATTICE


def delta_histogram(values: Iterable[tuple[int, int]]) -> dict[str, float]:
    """log2 magnitude bins of |(dx, dy)|, plus the summary statistics.

    Bins rather than a mean because the failure mode is bimodal: a collapsed policy
    piles up at hypot(10,10) = 14.1 px while a working one spreads over hundreds of
    pixels, and a mean sits between the two describing neither.
    """
    magnitudes = [math.hypot(dx, dy) for dx, dy in values]
    bins: dict[str, float] = {f"delta_bin_{2 ** k}": 0.0 for k in range(0, 12, 2)}
    bins["delta_bin_0"] = 0.0
    for magnitude in magnitudes:
        if magnitude == 0:
            bins["delta_bin_0"] += 1.0
            continue
        exponent = min(10, max(0, int(math.log2(magnitude)) // 2 * 2))
        bins[f"delta_bin_{2 ** exponent}"] += 1.0
    if magnitudes:
        ordered = sorted(magnitudes)
        bins["delta_mean_px"] = sum(magnitudes) / len(magnitudes)
        bins["delta_median_px"] = ordered[len(ordered) // 2]
        bins["delta_max_px"] = ordered[-1]
    return bins


class FailureModeIndicators:
    """A, B, C, D — the four indicators the calibrated gate must keep reporting."""

    @vf.metric
    async def failure_modes(self, trace: vf.Trace) -> dict[str, float]:
        task = trace.task.data
        result = trace.info.get(RESULT_KEY) or {}
        success = bool(result.get("success"))
        no_submit_cell = bool(getattr(task, "no_submit", False))

        literal = 0
        same_action = 0
        submitted_in_no_submit = 0
        offending_literal: list[str] = []
        offending_same: list[str] = []
        offending_submit: list[str] = []

        for step in step_records(trace):
            parsed = step.get("parsed_action")
            raw = step.get("raw_model_output") or ""
            last_line = raw.splitlines()[-1] if raw.splitlines() else ""
            texts = typed_texts(parsed)
            submits = submit_keys(parsed)
            if any(escape in text for text in texts for escape in _LITERAL_ESCAPES):
                literal += 1
                offending_literal.append(last_line)
            if texts and submits:
                same_action += 1
                offending_same.append(last_line)
            if submits and no_submit_cell:
                submitted_in_no_submit += 1
                offending_submit.append(last_line)

        terminated = bool(result.get("control_terminate"))
        premature = bool(terminated and not success)
        trace.info.setdefault("indicators", {})["failure_modes"] = {
            "A_offending": offending_literal,
            "B_examples": offending_same,
            "D_submitted_in_no_submit_cell": offending_submit,
            "C_termination_raw": result.get("control_terminate"),
            "stop_reason": result.get("outcome"),
        }
        return {
            "A_literal_escape_actions": float(literal),
            "A_cell_has_literal_escape": 1.0 if literal else 0.0,
            "B_same_action_submit_actions": float(same_action),
            "B_cell_has_same_action_submit": 1.0 if same_action else 0.0,
            "C_terminated": 1.0 if terminated else 0.0,
            "C_terminated_before_success": 1.0 if premature else 0.0,
            "D_submitted_in_no_submit_cell": 1.0 if submitted_in_no_submit else 0.0,
            "parse_errors": float(result.get("parse_errors", 0)),
            "action_errors": float(result.get("action_errors", 0)),
            "executor_errors": float(result.get("executor_errors", 0)),
        }


class MouseIndicators:
    """Relative-move diagnostics: lattice collapse, magnitude spread, reach."""

    @vf.metric
    async def mouse(self, trace: vf.Trace) -> dict[str, float]:
        steps = step_records(trace)
        result = trace.info.get(RESULT_KEY) or {}
        all_deltas: list[tuple[int, int]] = []
        for step in steps:
            all_deltas.extend(deltas(step.get("parsed_action")))
        nonzero = [d for d in all_deltas if d != (0, 0)]
        lattice = [d for d in all_deltas if on_lattice(d)]
        out: dict[str, float] = {
            "n_deltas": float(len(all_deltas)),
            "on_lattice_rate": (len(lattice) / len(all_deltas)) if all_deltas else 0.0,
            "zero_delta_rate": (
                (len(all_deltas) - len(nonzero)) / len(all_deltas) if all_deltas else 0.0
            ),
            "no_op_rate": _rate(steps, lambda s: s.get("control") == "no_op"),
            "parse_error_rate": _rate(steps, lambda s: bool(s.get("parse_error"))),
            "terminate_rate": 1.0 if result.get("control_terminate") else 0.0,
            "in_bbox_rate": _rate(steps, lambda s: bool((s.get("probe") or {}).get("in_bbox"))),
        }
        out.update(delta_histogram(all_deltas))
        return out


class SamplingProvenance:
    """Record on the trace what sampling actually reached the wire.

    `Dialect.apply_overrides` discards a program-set temperature whenever the eval
    set one, so both the resolved value and its source are recorded.
    """

    @vf.metric
    async def sampling(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        sampling = result.get("sampling") or {}
        temperature = sampling.get("temperature")
        return {
            "temperature": float(temperature) if isinstance(temperature, (int, float)) else -1.0,
            "temperature_from_ctx_sampling": (
                1.0 if sampling.get("temperature_source") == "ctx.sampling" else 0.0
            ),
            "max_tokens": float(sampling.get("max_tokens") or 0),
        }


def _rate(steps: Sequence[dict[str, Any]], predicate: Any) -> float:
    if not steps:
        return 0.0
    return sum(1.0 for step in steps if predicate(step)) / len(steps)
