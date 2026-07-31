"""Distributional gates — because greedy decoding is not reproducible here.

**Defect #19.** The reference implementation disagrees with **itself** on 43.8% of
greedy 8-step trajectories. Batched attention kernels, non-deterministic reduction
order and continuous batching make token-level output a function of what else is in
flight. Consequences:

* an exact-match regression test on a multi-step rollout is **meaningless** — it
  fails ~44% of the time for a bit-identical model and codebase;
* a gate must therefore compare *distributions*: "is this run's outcome inside the
  band that the reference itself occupies?"

This module provides :func:`assert_within_band` (a proportion gate with an explicit
tolerance) and :func:`assert_distribution_close` (a two-sample check over
per-item scores), plus :class:`ReproducibilityProbe` which measures a harness's
self-agreement so a gate's tolerance is derived from data rather than guessed.

:func:`forbid_exact_match_gate` exists purely to be loud: it raises with an
explanation if someone tries to add an exact-match multi-step gate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rft.errors import AnchorMismatch, SchemaError


@dataclass(frozen=True)
class ProportionBand:
    """An acceptance band for a proportion, with the reasoning attached."""

    low: float
    high: float
    reason: str

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def describe(self) -> str:
        return f"[{self.low:.4f}, {self.high:.4f}] ({self.reason})"


def wilson_band(
    successes: int, n: int, *, z: float = 1.96, slack: float = 0.0, reason: str = ""
) -> ProportionBand:
    """Wilson score interval for an observed proportion, optionally widened.

    Used to turn a reference *reading* (e.g. "31 of 100 scored tasks at 1.0") into
    a band a re-run must land in, instead of demanding it reproduce 0.3085 exactly.
    """
    if n < 1:
        raise SchemaError("cannot build a band from zero observations")
    if not 0 <= successes <= n:
        raise SchemaError(f"successes {successes} out of range for n={n}")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return ProportionBand(
        low=max(0.0, centre - margin - slack),
        high=min(1.0, centre + margin + slack),
        reason=reason or f"Wilson {z:g}-sigma on {successes}/{n}, slack {slack:g}",
    )


def assert_within_band(
    value: float, band: ProportionBand, *, what: str
) -> None:
    """Raise :class:`~rft.errors.AnchorMismatch` if ``value`` is outside ``band``."""
    if not band.contains(value):
        raise AnchorMismatch(
            f"{what}: observed {value:.4f} outside the acceptance band {band.describe()}. "
            "Either the instrument changed or the model did - find out which before "
            "reporting either."
        )


@dataclass(frozen=True)
class ReproducibilityReport:
    """How much a harness agrees with itself on repeated identical runs."""

    n_items: int
    n_agreeing: int
    label: str

    @property
    def agreement(self) -> float:
        if not self.n_items:
            raise SchemaError("no items compared; self-agreement is undefined")
        return self.n_agreeing / self.n_items

    @property
    def disagreement(self) -> float:
        return 1.0 - self.agreement

    def describe(self) -> str:
        return (
            f"{self.label}: self-agreement {self.n_agreeing}/{self.n_items} "
            f"({self.agreement:.1%}); disagreement {self.disagreement:.1%}"
        )

    def suggested_slack(self) -> float:
        """Slack a gate should add to absorb the harness's own noise."""
        return self.disagreement / 2.0


class ReproducibilityProbe:
    """Measure a harness's self-agreement over repeated identical invocations.

    Run the *same* input twice through the *same* harness and compare. If agreement
    is below 1.0, an exact-match gate on that harness is invalid, and
    :meth:`ReproducibilityReport.suggested_slack` gives a defensible tolerance.
    """

    def __init__(self, label: str = "harness") -> None:
        self.label = label
        self._pairs: list[tuple[object, object]] = []

    def add(self, run_a: object, run_b: object) -> None:
        self._pairs.append((run_a, run_b))

    def report(self) -> ReproducibilityReport:
        n_agree = sum(1 for a, b in self._pairs if a == b)
        return ReproducibilityReport(
            n_items=len(self._pairs), n_agreeing=n_agree, label=self.label
        )


def forbid_exact_match_gate(n_steps: int, *, context: str = "") -> None:
    """Raise if a caller is about to gate on exact-match over a multi-step rollout.

    Kept as an explicit, callable guard so the reason travels with the code: the
    reference harness disagrees with itself on 43.8% of greedy 8-step trajectories,
    so an exact-match gate is a coin flip dressed as a regression test.
    """
    if n_steps > 1:
        where = f" ({context})" if context else ""
        raise SchemaError(
            f"refusing an exact-match gate over {n_steps} steps{where}: greedy decoding "
            "is not reproducible in this stack - the reference disagrees with itself on "
            "43.8% of greedy 8-step trajectories (defect #19). Gate distributionally: "
            "rft.gates.assert_within_band / assert_distribution_close."
        )


def assert_distribution_close(
    observed: Sequence[float],
    reference: Sequence[float],
    *,
    what: str,
    max_mean_delta: float,
    max_ks: float | None = None,
) -> None:
    """Two-sample gate on per-item scores: mean shift and (optionally) KS distance.

    No scipy dependency; the KS statistic is computed directly.
    """
    if not observed or not reference:
        raise SchemaError(
            f"{what}: cannot compare distributions with an empty sample "
            f"({len(observed)} observed, {len(reference)} reference)"
        )
    mean_o = sum(observed) / len(observed)
    mean_r = sum(reference) / len(reference)
    if abs(mean_o - mean_r) > max_mean_delta:
        raise AnchorMismatch(
            f"{what}: mean {mean_o:.4f} vs reference {mean_r:.4f} "
            f"(delta {abs(mean_o - mean_r):.4f} > {max_mean_delta})"
        )
    if max_ks is not None:
        ks = _ks_statistic(observed, reference)
        if ks > max_ks:
            raise AnchorMismatch(
                f"{what}: KS distance {ks:.4f} > {max_ks} between observed and reference "
                "score distributions"
            )


def _ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    sa, sb = sorted(a), sorted(b)
    grid = sorted(set(sa) | set(sb))
    worst = 0.0
    for x in grid:
        fa = sum(1 for v in sa if v <= x) / len(sa)
        fb = sum(1 for v in sb if v <= x) / len(sb)
        worst = max(worst, abs(fa - fb))
    return worst
