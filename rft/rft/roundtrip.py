"""Mandatory round-trip audit for every action-format conversion.

A conversion is only correct if the **exact parser the evaluation harness uses**
recovers the original action from the converted text. Not an equivalent parser, not
a spot check on three examples — every record, through
``juergen/eval/action_parser.py`` (imported via :mod:`rft.evalparser`), with a
report that names the first mismatches.

This is the audit that showed a rebuilt pipeline was clean (0/2028 conversion
mismatches). It is exposed as a function that :mod:`rft.records` calls
unconditionally, not as a script someone remembers to run.

It also carries the ``native_rel`` lesson: a *relative* delta emitted inside a
schema whose field is documented as absolute round-trips fine syntactically and is
still wrong. :func:`assert_convention_declared` forces every converted record to
carry the grammar name it was written in, so the convention travels with the data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rft.conversion import split_response
from rft.errors import MissingFieldError, RoundTripError
from rft.evalparser import ACTION_PARSER_PATH
from rft.grammars import ParsedCompletion, get_grammar, parse_completion


@dataclass(frozen=True)
class RoundTripMismatch:
    index: int
    sample_id: str
    original: str
    reparsed: str | None
    error: str | None


@dataclass
class RoundTripReport:
    """Audit outcome. ``n_mismatched == 0`` is the only acceptable result."""

    grammar: str
    parser_path: str
    n_checked: int = 0
    n_unparseable: int = 0
    mismatches: list[RoundTripMismatch] = field(default_factory=list)
    anomaly_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_mismatched(self) -> int:
        return len(self.mismatches)

    @property
    def clean(self) -> bool:
        return self.n_mismatched == 0 and self.n_unparseable == 0

    def describe(self) -> str:
        anomalies = (
            "\n  tolerated-but-counted anomalies: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.anomaly_counts.items()))
            if self.anomaly_counts
            else ""
        )
        return (
            f"roundtrip[{self.grammar}] via {self.parser_path}: "
            f"{self.n_checked - self.n_mismatched - self.n_unparseable}/{self.n_checked} clean, "
            f"{self.n_mismatched} mismatched, {self.n_unparseable} unparseable{anomalies}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "grammar": self.grammar,
            "parser_path": self.parser_path,
            "n_checked": self.n_checked,
            "n_mismatched": self.n_mismatched,
            "n_unparseable": self.n_unparseable,
            "clean": self.clean,
            "anomaly_counts": self.anomaly_counts,
            "mismatches": [m.__dict__ for m in self.mismatches[:50]],
        }


def audit_roundtrip(
    items: Iterable[tuple[str, str]],
    *,
    grammar: str,
    canonicalise: Callable[[ParsedCompletion], str] | None = None,
) -> RoundTripReport:
    """Re-parse every converted action and compare canonical forms.

    Args:
        items: ``(sample_id, converted_text)`` pairs.
        grammar: the grammar the text was written in. Required — a round trip
            against a guessed grammar proves nothing (defect #5).
        canonicalise: how to reduce a parse to a comparable string. Defaults to
            :attr:`ParsedCompletion.canonical` when the grammar provides one, else a
            structural tuple.

    Returns:
        A :class:`RoundTripReport`. Unparseable items are counted, never skipped.
    """
    get_grammar(grammar)  # fail fast on an unknown name
    report = RoundTripReport(grammar=grammar, parser_path=str(ACTION_PARSER_PATH))
    for i, (sample_id, full_text) in enumerate(items):
        report.n_checked += 1
        # Audit the ACTION SPAN, not the whole assistant turn: a correctly-built
        # record keeps its format-independent reasoning preamble (see
        # rft.conversion), and prose is not the parser's business. Auditing the
        # whole turn would make every prose-bearing record look unparseable and
        # would push builders towards stripping the prose - which is the very
        # defect rft.conversion exists to prevent.
        text = split_response(full_text).action_span or full_text
        try:
            parsed = parse_completion(text, grammar=grammar)
        except Exception as exc:  # noqa: BLE001 - counted, and surfaced in the report
            report.n_unparseable += 1
            report.mismatches.append(
                RoundTripMismatch(
                    index=i,
                    sample_id=sample_id,
                    original=text[:200],
                    reparsed=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        for a in parsed.anomalies:
            report.anomaly_counts[a] = report.anomaly_counts.get(a, 0) + 1
        canon = canonicalise(parsed) if canonicalise else _default_canonical(parsed)
        try:
            reparsed = parse_completion(canon, grammar=grammar)
        except Exception as exc:  # noqa: BLE001
            report.mismatches.append(
                RoundTripMismatch(
                    index=i,
                    sample_id=sample_id,
                    original=text[:200],
                    reparsed=canon[:200],
                    error=f"canonical form does not re-parse: {type(exc).__name__}: {exc}",
                )
            )
            continue
        recanon = canonicalise(reparsed) if canonicalise else _default_canonical(reparsed)
        if recanon != canon:
            report.mismatches.append(
                RoundTripMismatch(
                    index=i,
                    sample_id=sample_id,
                    original=canon[:200],
                    reparsed=recanon[:200],
                    error="canonical form is not a fixed point of parse->format",
                )
            )
    return report


def _default_canonical(parsed: ParsedCompletion) -> str:
    if parsed.canonical is not None:
        return parsed.canonical
    return repr(
        (
            parsed.grammar,
            tuple((o.kind, o.dx, o.dy, o.absolute) for o in parsed.mouse_ops),
            parsed.terminate,
            parsed.no_op,
            parsed.typed_text,
        )
    )


def assert_roundtrip_clean(report: RoundTripReport) -> RoundTripReport:
    """Raise unless the audit is perfectly clean."""
    if not report.clean:
        first = report.mismatches[:5]
        raise RoundTripError(
            f"{report.describe()}\nfirst failures: "
            + "\n".join(
                f"  [{m.index}] {m.sample_id}: {m.error}\n    orig={m.original!r}\n"
                f"    reparsed={m.reparsed!r}"
                for m in first
            )
        )
    return report


def assert_convention_declared(
    records: Sequence[dict[str, Any]], *, key: str = "grammar"
) -> None:
    """Every record must name the grammar it is written in.

    The ``native_rel`` defect was a *relative* delta stored in a field whose schema
    documents an absolute coordinate. Nothing about the number itself reveals the
    error; only a declared convention does. A record without one is rejected.
    """
    missing = [i for i, r in enumerate(records) if not r.get(key)]
    if missing:
        raise MissingFieldError(
            f"records[{missing[0]}].{key} (and {len(missing) - 1} more): every "
            "converted record must declare its action grammar, so a relative delta "
            "can never be read as an absolute coordinate"
        )
    grammars = {str(r[key]) for r in records}
    for g in grammars:
        get_grammar(g)
