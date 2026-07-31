"""Read an offline wandb datastore correctly.

**Defect #10.** In an offline ``.wandb`` run file, a history item's key lives at
``item.nested_key`` (a repeated path field), not at ``item.key``. Reading ``.key``
returns an empty string for every item, so a scan for ``val/loss`` finds nothing
and the conclusion becomes "no val rows exist" — when in fact every row was there.

:func:`iter_history` reads both fields and **raises if every item yields an empty
key**, which is the exact signature of the defect. There is no code path in which
"found nothing" is reported as a fact without that check having passed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rft.errors import SchemaError


def _key_of(item: Any) -> str:
    """Resolve a history item's key from ``nested_key`` first, then ``key``.

    ``nested_key`` is a repeated string field: ``["val", "loss"]`` means
    ``val/loss``. protobuf returns an empty list (not None) when it is unset, so
    the emptiness check has to be explicit.
    """
    nested = getattr(item, "nested_key", None)
    if nested:
        return "/".join(str(part) for part in nested)
    key = getattr(item, "key", None)
    if key:
        return str(key)
    if isinstance(item, dict):
        nested = item.get("nested_key")
        if nested:
            return "/".join(str(p) for p in nested)
        return str(item.get("key") or "")
    return ""


def _value_of(item: Any) -> Any:
    raw = getattr(item, "value_json", None)
    if raw is None and isinstance(item, dict):
        raw = item.get("value_json")
    if raw is None:
        raise SchemaError(f"history item has no value_json: {item!r}")
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"history item value_json is not JSON: {raw!r}") from exc


@dataclass(frozen=True)
class HistoryRow:
    """One logged step: ``{metric_name: value}`` plus the step if present."""

    step: int | None
    values: dict[str, Any]


def rows_from_history_items(item_batches: Sequence[Sequence[Any]]) -> list[HistoryRow]:
    """Decode batches of wandb history items into rows.

    Args:
        item_batches: one sequence of items per logged step.

    Raises:
        SchemaError: every item resolved to an empty key. That means the reader is
            looking at the wrong field (defect #10) — not that the run logged
            nothing.
    """
    rows: list[HistoryRow] = []
    n_items = 0
    n_empty_keys = 0
    for batch in item_batches:
        values: dict[str, Any] = {}
        step: int | None = None
        for item in batch:
            n_items += 1
            key = _key_of(item)
            if not key:
                n_empty_keys += 1
                continue
            value = _value_of(item)
            if key in ("_step", "step"):
                step = int(value)
            values[key] = value
        rows.append(HistoryRow(step=step, values=values))
    if n_items and n_empty_keys == n_items:
        raise SchemaError(
            f"all {n_items} wandb history items resolved to an EMPTY key. The keys live "
            "at item.nested_key (a repeated path field), not item.key (defect #10). "
            "This is a reader bug, not an empty run."
        )
    return rows


def iter_history(run_path: str | Path) -> Iterator[HistoryRow]:
    """Stream history rows from an offline ``.wandb`` file.

    Uses wandb's own datastore reader, so no format is re-implemented here.
    """
    from wandb.sdk.internal import datastore  # noqa: PLC0415 - optional heavy dep
    from wandb.proto import wandb_internal_pb2  # noqa: PLC0415

    p = Path(run_path)
    if not p.is_file():
        raise SchemaError(f"offline wandb run file not found: {p}")
    ds = datastore.DataStore()
    ds.open_for_scan(str(p))
    batches: list[list[Any]] = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        batches.append(list(record.history.item))
    yield from rows_from_history_items(batches)


def metric_series(run_path: str | Path, metric: str) -> list[tuple[int | None, float]]:
    """All ``(step, value)`` points for one metric.

    Raises:
        SchemaError: the metric is absent. The available metric names are listed in
            the message, so "no val rows exist" can never again be the conclusion
            when the real answer is "the name is ``val/loss`` and you asked for
            ``val_loss``".
    """
    rows = list(iter_history(run_path))
    out: list[tuple[int | None, float]] = []
    seen: set[str] = set()
    for row in rows:
        seen.update(row.values)
        if metric in row.values:
            out.append((row.step, float(row.values[metric])))
    if not out:
        raise SchemaError(
            f"metric {metric!r} has no rows in {run_path}. Metrics present: "
            f"{sorted(seen)!r}"
        )
    return out
