"""Within-record information leakage: auxiliary text must not leak the label.

This is a **different failure mode** from dataset overlap. :mod:`rft.splits` stops the
same *task* appearing in train and eval. This module stops a single record from
handing the model its own answer through a channel that is not the input it is
supposed to reason from.

The instance that motivated it: a reasoning preamble that happened to contain the
target coordinates. Had it, the experiment would have silently degenerated into a
text-arithmetic task — read the numbers out of the prose, do arithmetic, emit them —
and produced a spectacular false positive that looked exactly like success. So:
**preamble digit-leak must be 0.**

Generalised here as :func:`assert_no_label_leak`, which checks that no numeric value
from the label appears in the auxiliary text. Two refinements matter:

* it compares **numbers**, not substrings, so ``127`` in the label is not "found" in
  the word ``1279`` and *is* found in ``x=127``;
* it is applied to prose, thinking blocks and the user turn — every channel that
  travels with the record.

:func:`assert_no_geometry_overlap` is the companion for eval-set construction, in the
form the ladder owner's guard uses: dedupe on **exact ``(cursor, bbox)`` geometry**
rather than on a scene identifier, because two differently-named scenes can carry
identical geometry and a name-based check passes them straight through.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rft.errors import LeakError, SchemaError


class LabelLeakError(LeakError):
    """Auxiliary text contained values from the label."""


#: A signed integer or decimal, as a standalone number (not part of a longer number).
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: Numbers too common to be evidence of leakage: coordinates of 0/1 and small counts
#: appear in ordinary prose ("the first tab", "step 2"). Configurable per call.
DEFAULT_IGNORED_VALUES: frozenset[float] = frozenset({0.0, 1.0, 2.0, 3.0})


def numbers_in(text: str) -> list[float]:
    """Every standalone number in ``text``, as floats."""
    if not isinstance(text, str):
        raise TypeError(f"numbers_in expects str, got {type(text).__name__}")
    return [float(m.group(0)) for m in _NUMBER.finditer(text)]


@dataclass
class LeakReport:
    """Per-record leak accounting. ``n_records_leaking`` must be 0."""

    n_records: int = 0
    n_records_leaking: int = 0
    n_values_leaked: int = 0
    examples: list[str] = field(default_factory=list)
    ignored_values: tuple[float, ...] = ()

    @property
    def clean(self) -> bool:
        return self.n_records_leaking == 0

    @property
    def leak_rate(self) -> float:
        if not self.n_records:
            raise SchemaError("no records: leak rate is undefined, not 0.0")
        return self.n_records_leaking / self.n_records

    def describe(self) -> str:
        return (
            f"label leak: {self.n_records_leaking}/{self.n_records} records leak label "
            f"values into auxiliary text ({self.n_values_leaked} values total); "
            f"ignored common values {sorted(self.ignored_values)}"
            + ("" if self.clean else "\n  " + "\n  ".join(self.examples[:5]))
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_records_leaking": self.n_records_leaking,
            "n_values_leaked": self.n_values_leaked,
            "clean": self.clean,
            "ignored_values": sorted(self.ignored_values),
            "examples": self.examples[:50],
        }


def check_label_leak(
    records: Iterable[tuple[str, str, str]],
    *,
    ignored_values: Iterable[float] = DEFAULT_IGNORED_VALUES,
) -> LeakReport:
    """Check ``(record_id, auxiliary_text, label_text)`` triples for numeric leakage.

    A label value counts as leaked when the same number appears as a standalone number
    in the auxiliary text.
    """
    ignored = frozenset(float(v) for v in ignored_values)
    report = LeakReport(ignored_values=tuple(sorted(ignored)))
    for record_id, aux, label in records:
        report.n_records += 1
        label_values = {v for v in numbers_in(label) if v not in ignored}
        if not label_values:
            continue
        aux_values = set(numbers_in(aux))
        leaked = sorted(label_values & aux_values)
        if leaked:
            report.n_records_leaking += 1
            report.n_values_leaked += len(leaked)
            if len(report.examples) < 50:
                report.examples.append(
                    f"{record_id}: label values {leaked} also appear in auxiliary text "
                    f"{aux.strip()[:120]!r}"
                )
    return report


def assert_no_label_leak(
    records: Iterable[tuple[str, str, str]],
    *,
    ignored_values: Iterable[float] = DEFAULT_IGNORED_VALUES,
    context: str = "",
) -> LeakReport:
    """Raise unless auxiliary text leaks no label values.

    Raises:
        LabelLeakError: any record leaks. The message explains the consequence, not
            just the fact: a leaked coordinate turns a perception task into text
            arithmetic and the resulting "success" means nothing.
    """
    report = check_label_leak(records, ignored_values=ignored_values)
    if not report.clean:
        where = f" ({context})" if context else ""
        raise LabelLeakError(
            f"LABEL LEAK{where}: {report.n_records_leaking} of {report.n_records} records "
            "carry label values in their auxiliary text. A preamble that contains the "
            "target coordinates turns the task into text arithmetic - read the number "
            "out of the prose, emit it - so any accuracy measured on it is a false "
            f"positive that looks exactly like success.\n{report.describe()}"
        )
    return report


def prose_digit_leak(
    records: Iterable[Mapping[str, Any]],
    *,
    id_key: str = "sample_id",
    ignored_values: Iterable[float] = DEFAULT_IGNORED_VALUES,
) -> LeakReport:
    """Apply :func:`check_label_leak` to chat records, splitting each assistant turn.

    Auxiliary text = the record's prose (prefix + suffix around the action span) plus
    every user turn. Label = the action span. This is the concrete "preamble digit-leak
    = 0" check.
    """
    from rft.conversion import split_response
    from rft.records import assistant_target

    triples: list[tuple[str, str, str]] = []
    for i, record in enumerate(records):
        target = assistant_target(record)
        parts = split_response(target)
        user_text = _user_text(record)
        aux = f"{parts.prefix}\n{parts.suffix}\n{user_text}"
        triples.append((str(record.get(id_key, f"records[{i}]")), aux, parts.action_span))
    return check_label_leak(triples, ignored_values=ignored_values)


def _user_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(
                part["text"]
                for part in content
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            )
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# geometry-level duplicate detection
# ---------------------------------------------------------------------------

Geometry = tuple[tuple[float, float], tuple[float, float, float, float]]


def geometry_key(
    cursor: Sequence[float], bbox: Sequence[float]
) -> Geometry:
    """Canonical ``(cursor, bbox)`` key for duplicate detection.

    Deliberately geometry, not an identifier: two scenes with different ids can carry
    the same cursor start and the same target box, in which case an id-based dedupe
    passes duplicates through and an "unseen" eval instance is not unseen.
    """
    if len(cursor) != 2:
        raise SchemaError(f"cursor must be (x, y), got {cursor!r}")
    if len(bbox) != 4:
        raise SchemaError(f"bbox must be (x0, y0, x1, y1), got {bbox!r}")
    return (
        (float(cursor[0]), float(cursor[1])),
        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
    )


def assert_no_geometry_overlap(
    train: Iterable[Geometry], heldout: Iterable[Geometry], *, context: str = ""
) -> None:
    """Raise if any exact ``(cursor, bbox)`` geometry appears in both splits.

    Raises:
        LeakError: geometries overlap.
    """
    train_set = set(train)
    held_set = set(heldout)
    if not held_set:
        raise SchemaError("held-out geometry set is empty; the check would be vacuous")
    overlap = train_set & held_set
    if overlap:
        where = f" ({context})" if context else ""
        raise LeakError(
            f"GEOMETRY LEAK{where}: {len(overlap)} exact (cursor, bbox) geometries appear "
            f"in both train and held-out. Scene IDENTIFIERS can differ while the "
            f"geometry is duplicated, so an id-based check would have passed these. "
            f"First: {sorted(overlap)[:3]!r}"
        )


def duplicate_geometries(items: Iterable[Geometry]) -> dict[Geometry, int]:
    """Geometries occurring more than once, with their counts."""
    counts: dict[Geometry, int] = {}
    for g in items:
        counts[g] = counts.get(g, 0) + 1
    return {g: n for g, n in counts.items() if n > 1}
