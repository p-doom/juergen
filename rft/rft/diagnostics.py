"""First-class prediction diagnostics — emitted by every eval, always.

Three numbers exposed the single biggest finding of the project (a
magnitude-encoding collapse that aggregate accuracy hid completely):

1. the count of **distinct predicted deltas**,
2. the fraction of predictions inside the ``{0, ±1, ±10, ±100}`` **lattice**,
3. the **median |pred| / |gold|** magnitude ratio.

They are therefore part of the eval's output contract, not something a reader
has to reconstruct with a one-off script. :class:`DeltaDiagnostics` is returned
by :func:`delta_diagnostics` and rendered by
:meth:`DeltaDiagnostics.describe`; :mod:`rft.evaluation` writes it into every
result document.

Aggregate accuracy is ~85% insensitive to mouse control on the held-out set, so
a run that moves only the aggregate has not been shown to move mouse control.
These diagnostics are what makes that visible.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from rft.errors import SchemaError

#: The magnitude lattice a collapsed policy falls into: it emits only a handful
#: of round magnitudes instead of continuously-valued deltas.
LATTICE_VALUES: Final[frozenset[int]] = frozenset({0, 1, -1, 10, -10, 100, -100})

Delta = tuple[float, float]


@dataclass(frozen=True)
class DeltaDiagnostics:
    """The mandatory three numbers, plus the counts they were derived from."""

    n_predictions: int
    n_distinct_deltas: int
    lattice_fraction: float
    median_magnitude_ratio: float | None
    n_gold_pairs: int
    n_zero_gold_skipped: int
    most_common: tuple[tuple[Delta, int], ...]
    #: Predictions that were a zero delta against a non-zero gold. A rising count
    #: here IS the collapse; excluding them from the ratio hides it.
    n_zero_pred: int = 0
    zero_pred_excluded: bool = False
    median_kind: str = "average"

    def describe(self) -> str:
        ratio = (
            "n/a (no non-zero gold deltas)"
            if self.median_magnitude_ratio is None
            else f"{self.median_magnitude_ratio:.3f}"
        )
        top = ", ".join(f"({dx:g},{dy:g})x{n}" for (dx, dy), n in self.most_common)
        zero_note = (
            f"{self.n_zero_pred} zero-pred "
            f"{'EXCLUDED (hides collapse)' if self.zero_pred_excluded else 'included'}"
        )
        return (
            f"distinct_deltas={self.n_distinct_deltas}/{self.n_predictions}  "
            f"lattice_fraction={self.lattice_fraction:.3f}  "
            f"median|pred|/|gold|={ratio} [{self.median_kind}] (over "
            f"{self.n_gold_pairs} pairs, {self.n_zero_gold_skipped} zero-gold skipped, "
            f"{zero_note})\n"
            f"  most common predicted deltas: {top}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "n_predictions": self.n_predictions,
            "n_distinct_deltas": self.n_distinct_deltas,
            "lattice_fraction": self.lattice_fraction,
            "median_magnitude_ratio": self.median_magnitude_ratio,
            "n_gold_pairs": self.n_gold_pairs,
            "n_zero_gold_skipped": self.n_zero_gold_skipped,
            "n_zero_pred": self.n_zero_pred,
            "zero_pred_excluded": self.zero_pred_excluded,
            "median_kind": self.median_kind,
            "most_common_deltas": [
                {"delta": list(d), "count": n} for d, n in self.most_common
            ],
        }


def _magnitude(d: Delta) -> float:
    return (d[0] ** 2 + d[1] ** 2) ** 0.5


def _in_lattice(d: Delta) -> bool:
    return all(float(c).is_integer() and int(c) in LATTICE_VALUES for c in d)


def _median(values: Sequence[float], kind: str) -> float:
    """Median with an explicit convention.

    ``"average"`` is :func:`statistics.median` (mean of the two middle values on an
    even-length list). ``"upper"`` is ``sorted(x)[len(x)//2]``, which is what the
    historical ``tf_decomp_iso.py`` reference used. The two differ on even-length
    samples, so reproducing a reference reading requires naming the convention —
    silently picking one is how two "same" metrics disagree.
    """
    if kind == "average":
        return statistics.median(values)
    if kind == "upper":
        return sorted(values)[len(values) // 2]
    raise SchemaError(f"unknown median kind {kind!r}; use 'average' or 'upper'")


def delta_diagnostics(
    predictions: Sequence[Delta],
    golds: Sequence[Delta] | None = None,
    *,
    top_k: int = 5,
    exclude_zero_pred: bool = False,
    median_kind: str = "average",
) -> DeltaDiagnostics:
    """Compute the three mandatory diagnostics.

    Args:
        predictions: predicted ``(dx, dy)`` deltas, one per scored step.
        golds: matching gold deltas, same length, or ``None`` when no reference
            deltas exist (closed-loop rollouts). The magnitude ratio is then
            ``None`` — never 0.0, never silently omitted.
        top_k: how many most-common predicted deltas to surface. A collapsed
            policy shows up here immediately (one delta with almost all the
            mass).
        exclude_zero_pred: drop pairs whose *prediction* is a zero delta from the
            magnitude ratio. **Default False, deliberately.** The historical
            reference (``tf_decomp_iso.py``: ``if gn>0 and pn>0``) excluded them,
            which hides exactly the failure mode we care about: a policy that has
            collapsed to emitting ``(0,0)`` disappears from its own magnitude
            ratio. Set True only to reproduce that reference reading, and say so.
        median_kind: ``"average"`` (default) or ``"upper"``; see :func:`_median`.

    Raises:
        SchemaError: ``predictions`` is empty, or ``golds`` has a different
            length. An empty prediction set has no diagnostics; reporting zeros
            for it is the defect-#9 pattern.
    """
    if not predictions:
        raise SchemaError(
            "no predictions: diagnostics are undefined for an empty set "
            "(do not report 0 distinct deltas / 0.0 lattice fraction)"
        )
    if golds is not None and len(golds) != len(predictions):
        raise SchemaError(
            f"predictions/golds length mismatch: {len(predictions)} vs {len(golds)}"
        )

    preds = [(float(dx), float(dy)) for dx, dy in predictions]
    counts: dict[Delta, int] = {}
    for d in preds:
        counts[d] = counts.get(d, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    lattice_fraction = sum(1 for d in preds if _in_lattice(d)) / len(preds)

    ratio: float | None = None
    n_pairs = 0
    n_zero_gold = 0
    n_zero_pred = 0
    if golds is not None:
        ratios: list[float] = []
        for p, g in zip(preds, [(float(a), float(b)) for a, b in golds], strict=True):
            gm = _magnitude(g)
            if gm == 0.0:
                # |pred|/0 is not a number. Skip and *report* the skip; a
                # zero-gold step cannot contribute to a magnitude ratio.
                n_zero_gold += 1
                continue
            if _magnitude(p) == 0.0:
                n_zero_pred += 1
                if exclude_zero_pred:
                    continue
            ratios.append(_magnitude(p) / gm)
        n_pairs = len(ratios)
        ratio = _median(ratios, median_kind) if ratios else None

    return DeltaDiagnostics(
        n_predictions=len(preds),
        n_distinct_deltas=len(counts),
        lattice_fraction=lattice_fraction,
        median_magnitude_ratio=ratio,
        n_gold_pairs=n_pairs,
        n_zero_gold_skipped=n_zero_gold,
        n_zero_pred=n_zero_pred,
        zero_pred_excluded=exclude_zero_pred,
        median_kind=median_kind,
        most_common=tuple(ranked),
    )


def deltas_from_completions(
    completions: Iterable[str], *, grammar: str, skip_unparseable: bool = False
) -> tuple[list[Delta], list[str]]:
    """Extract net predicted deltas from raw completions under one grammar.

    Returns ``(deltas, errors)``. Unparseable completions are **counted and
    returned**, never dropped on the floor: if ``skip_unparseable`` is False
    (the default) the first parse failure propagates.
    """
    from rft.grammars import parse_completion  # local import: avoids cycle

    deltas: list[Delta] = []
    errors: list[str] = []
    for i, text in enumerate(completions):
        try:
            parsed = parse_completion(text, grammar=grammar)
        except Exception as exc:  # noqa: BLE001 - re-raised or accounted below
            if not skip_unparseable:
                raise
            errors.append(f"completion[{i}]: {type(exc).__name__}: {exc}")
            continue
        net = parsed.net_delta
        if net is None:
            if not skip_unparseable:
                raise SchemaError(
                    f"completion[{i}] under grammar {grammar!r} has no relative delta; "
                    "an absolute grammar cannot contribute to delta diagnostics"
                )
            errors.append(f"completion[{i}]: no relative delta")
            continue
        deltas.append((float(net[0]), float(net[1])))
    return deltas, errors
