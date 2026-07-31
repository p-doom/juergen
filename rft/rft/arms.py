""""The arms differ in exactly one thing" as a machine-checkable property.

A controlled comparison requires that its arms differ in the dimension under study
and in **nothing else**. That has repeatedly been a reviewer's responsibility, and it
has repeatedly failed:

* the absolute arm kept its reasoning preamble and ``<tools>`` schema while every
  relative arm had both deleted (2383/2383 vs 0/2441) — so "absolute vs relative"
  was really "absolute-with-scratchpad vs relative-without", and the entire research
  programme reasoned from it for weeks;
* a checkpoint pair that looked like a clean goals-vs-no-goals comparison by name
  turned out to differ in action format, window size **and** sequence length.

So it becomes a build-stage assertion. :func:`assert_arms_differ_only_in` takes the
arms and the single dimension that is allowed to vary, and raises with per-arm counts
on any other difference. :func:`arm_parity_report` produces the same information as a
reportable object, and the build stage records it in the dataset manifest — including
an explicit, named opt-out if someone deliberately wants uncontrolled arms.

The dimensions are deliberately open (:data:`CONTROLLED_DIMENSIONS`): grammar,
encoding, label content, goal conditioning, window, sequence length. Those are the
contrasts the science depends on, and each is a thing we have got wrong.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rft.conversion import ContextFeatures
from rft.errors import SchemaError


class UncontrolledComparisonError(SchemaError):
    """Two arms differ in more than the dimension under study."""


#: Dimensions a controlled comparison may vary. Anything not named here that differs
#: between arms is a confound.
CONTROLLED_DIMENSIONS: tuple[str, ...] = (
    "grammar",
    "action_format",
    "encoding",
    "label_content",
    "goal_conditioned",
    "window",
    "max_length",
    "model",
)

#: Context properties that must match across arms no matter which dimension varies.
#: These are all format-INDEPENDENT: nothing about changing an action grammar
#: justifies changing any of them.
_INVARIANT_FEATURES: tuple[str, ...] = (
    "has_reasoning_preamble",
    "has_tools_schema",
    "has_action_marker",
)


@dataclass
class ArmProfile:
    """Aggregate context profile of one format arm."""

    name: str
    n_records: int = 0
    n_with_preamble: int = 0
    n_with_tools_schema: int = 0
    n_with_action_marker: int = 0
    action_kinds: dict[str, int] = field(default_factory=dict)

    def add(self, features: ContextFeatures) -> None:
        self.n_records += 1
        self.n_with_preamble += int(features.has_reasoning_preamble)
        self.n_with_tools_schema += int(features.has_tools_schema)
        self.n_with_action_marker += int(features.has_action_marker)
        self.action_kinds[features.action_kind] = (
            self.action_kinds.get(features.action_kind, 0) + 1
        )

    def fraction(self, feature: str) -> float:
        if not self.n_records:
            raise SchemaError(f"arm {self.name!r} has no records; fractions are undefined")
        return {
            "has_reasoning_preamble": self.n_with_preamble,
            "has_tools_schema": self.n_with_tools_schema,
            "has_action_marker": self.n_with_action_marker,
        }[feature] / self.n_records

    def describe(self) -> str:
        return (
            f"{self.name:<28} n={self.n_records:>6} "
            f"preamble={self.n_with_preamble:>6}/{self.n_records} "
            f"tools_schema={self.n_with_tools_schema:>6}/{self.n_records} "
            f"action_marker={self.n_with_action_marker:>6}/{self.n_records} "
            f"action_kinds={self.action_kinds}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_records": self.n_records,
            "n_with_preamble": self.n_with_preamble,
            "n_with_tools_schema": self.n_with_tools_schema,
            "n_with_action_marker": self.n_with_action_marker,
            "action_kinds": self.action_kinds,
        }


def profile_arm(name: str, targets: Iterable[str]) -> ArmProfile:
    """Build an :class:`ArmProfile` from one arm's assistant-target texts."""
    profile = ArmProfile(name=name)
    for text in targets:
        profile.add(ContextFeatures.of(text))
    if not profile.n_records:
        raise SchemaError(f"arm {name!r} contained no records")
    return profile


@dataclass
class ArmParityReport:
    """Cross-arm parity outcome. Written into every multi-arm dataset manifest."""

    dimension: str
    profiles: list[ArmProfile]
    violations: list[str] = field(default_factory=list)
    opt_out_reason: str | None = None

    @property
    def controlled(self) -> bool:
        return not self.violations

    def describe(self) -> str:
        head = (
            f"arm parity (varying only {self.dimension!r}): "
            f"{'CONTROLLED' if self.controlled else 'UNCONTROLLED'}"
        )
        lines = [head, *[p.describe() for p in self.profiles]]
        if self.violations:
            lines.append("violations:")
            lines.extend(f"  - {v}" for v in self.violations)
        if self.opt_out_reason:
            lines.append(f"OPT-OUT RECORDED: {self.opt_out_reason}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "controlled": self.controlled,
            "profiles": [p.as_dict() for p in self.profiles],
            "violations": self.violations,
            "opt_out_reason": self.opt_out_reason,
        }


def arm_parity_report(
    arms: Mapping[str, Sequence[str]],
    *,
    dimension: str,
    tolerance: float = 0.0,
) -> ArmParityReport:
    """Compare arms on every format-independent context feature.

    Args:
        arms: ``{arm_name: [assistant_target_text, ...]}``. All arms must be built
            from the SAME source records, in the same order, for the record-count
            check to be meaningful.
        dimension: the one dimension under study (see :data:`CONTROLLED_DIMENSIONS`).
        tolerance: allowed absolute difference in a feature's *fraction* between
            arms. Default 0.0 — the historical failure was 1.0 vs 0.0, and there is
            no legitimate reason for a grammar change to move any of these at all.
    """
    if dimension not in CONTROLLED_DIMENSIONS:
        raise SchemaError(
            f"unknown comparison dimension {dimension!r}; known: "
            f"{list(CONTROLLED_DIMENSIONS)!r}. Name the dimension explicitly - "
            "an unnamed comparison cannot be checked."
        )
    if len(arms) < 2:
        raise SchemaError(f"arm parity needs at least 2 arms, got {list(arms)!r}")

    profiles = [profile_arm(name, targets) for name, targets in sorted(arms.items())]
    report = ArmParityReport(dimension=dimension, profiles=profiles)

    counts = {p.n_records for p in profiles}
    if len(counts) != 1:
        report.violations.append(
            "arms have different record counts "
            + ", ".join(f"{p.name}={p.n_records}" for p in profiles)
            + " - they were not built from the same source records, so no per-record "
            "comparison is valid"
        )

    for feature in _INVARIANT_FEATURES:
        fractions = {p.name: p.fraction(feature) for p in profiles}
        spread = max(fractions.values()) - min(fractions.values())
        if spread > tolerance:
            detail = ", ".join(
                f"{p.name}={getattr(p, _FEATURE_COUNT_ATTR[feature])}/{p.n_records}"
                f" ({fractions[p.name]:.3f})"
                for p in profiles
            )
            report.violations.append(
                f"{feature} differs across arms (spread {spread:.3f} > {tolerance}): "
                f"{detail}. This is format-INDEPENDENT content; a grammar change must "
                "not move it."
            )
    return report


_FEATURE_COUNT_ATTR = {
    "has_reasoning_preamble": "n_with_preamble",
    "has_tools_schema": "n_with_tools_schema",
    "has_action_marker": "n_with_action_marker",
}


def assert_arms_differ_only_in(
    arms: Mapping[str, Sequence[str]],
    *,
    dimension: str,
    tolerance: float = 0.0,
    opt_out_reason: str | None = None,
) -> ArmParityReport:
    """Raise unless the arms differ only in ``dimension``.

    Args:
        opt_out_reason: if given, an uncontrolled comparison is *permitted* but the
            reason is recorded on the report (and hence in the dataset manifest). It
            is deliberately a free-text reason rather than a boolean: someone has to
            write down why, and it travels with the data.

    Raises:
        UncontrolledComparisonError: a confound was found and no opt-out was given.
    """
    report = arm_parity_report(arms, dimension=dimension, tolerance=tolerance)
    if report.controlled:
        return report
    if opt_out_reason:
        report.opt_out_reason = opt_out_reason
        return report
    raise UncontrolledComparisonError(
        "arms are NOT a controlled comparison:\n"
        + report.describe()
        + "\n\nFix the builder so the conversion touches only the action span "
        "(rft.conversion.convert_action_span), or pass opt_out_reason=... to record "
        "deliberately-uncontrolled arms in the manifest."
    )


# ---------------------------------------------------------------------------
# checking already-written datasets
# ---------------------------------------------------------------------------


def load_arm_targets(chat_jsonl: str | Path) -> list[str]:
    """Read the assistant-target text of every record in a ``chat.jsonl``."""
    from rft.records import assistant_target

    path = Path(chat_jsonl)
    if not path.is_file():
        raise SchemaError(f"chat.jsonl not found: {path}")
    out: list[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        out.append(assistant_target(record))
    if not out:
        raise SchemaError(f"{path} contained no records")
    return out


def audit_written_arms(
    arm_dirs: Mapping[str, str | Path],
    *,
    dimension: str,
    split: str = "train",
    tolerance: float = 0.0,
) -> ArmParityReport:
    """Audit already-written per-format datasets for parity.

    Points at the ``converted/<arm>/`` directory of each arm; reads
    ``_normalized/<split>/chat.jsonl``. Use this to check datasets that were built
    before the guard existed — it is how the absolute-vs-relative confound is
    detected in the shipped data.
    """
    arms = {
        name: load_arm_targets(Path(d) / "_normalized" / split / "chat.jsonl")
        for name, d in arm_dirs.items()
    }
    return arm_parity_report(arms, dimension=dimension, tolerance=tolerance)
