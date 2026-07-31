"""Task-level splits, collision-proof sample ids, and the eval-leak guard.

Three defects live here:

* **Defect #16** — a *record*-level train/val split scattered one task's ``k``
  sibling trajectories across both sides. Rejection sampling produces ``k``
  correlated rollouts per task, so a record-level split is a leak by
  construction. Splits must be **task-level**: every rollout of a task lands on
  exactly one side.
* **Defect #17** — ``slug = f"{app}__{task_id}"`` collides across sample roots,
  producing duplicate ``sample_id``s when two sampling runs are merged.
  :func:`make_sample_id` includes the sample root and the rollout index.
* **The eval-leak rule** — 259 OSWorld tasks are training-adjacent; the 110
  held-out tasks are evaluation-only. :func:`assert_no_leak` is called by the
  record builder unconditionally, and checkpoint selection never sees held-out
  scores.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rft.errors import LeakError, MissingFieldError, SchemaError

# ---------------------------------------------------------------------------
# sample ids
# ---------------------------------------------------------------------------


def _slugify_root(sample_root: str | Path) -> str:
    """Short, stable, filesystem-safe tag for a sample root directory.

    A hash of the *absolute* path, so two roots that share a basename (the
    defect-#17 case: ``.../runA/samples`` and ``.../runB/samples``) get
    different tags.
    """
    resolved = str(Path(sample_root).resolve())
    return hashlib.sha256(resolved.encode()).hexdigest()[:8]


def make_sample_id(
    *,
    sample_root: str | Path,
    task_id: str,
    rollout_index: int,
    app: str | None = None,
    step: int | None = None,
) -> str:
    """Globally-unique id for one rollout, or one step of one rollout.

    Shape: ``<root-tag>/<app>/<task_id>#<rollout_index>[@step<NNN>]``.

    Each component earns its place:

    * ``<root-tag>`` — a hash of the *absolute* sample root, so two sampling runs
      whose sample dirs share a basename (the k=8 case: every collector names its
      subdirs ``sample_0..7``) cannot collide;
    * ``<rollout_index>`` — unique within a task, so the ``k`` siblings differ;
    * ``@step<NNN>`` — unique within a rollout, for per-step training records. A
      multi-step rollout expands to several records and they must not share an id.

    ``app`` is included for readability only. It is *not* what provides uniqueness,
    which is precisely the mistake defect #17 encodes.
    """
    if not task_id:
        raise SchemaError("task_id must be non-empty")
    if rollout_index < 0:
        raise SchemaError(f"rollout_index must be >= 0, got {rollout_index}")
    if step is not None and step < 0:
        raise SchemaError(f"step must be >= 0, got {step}")
    parts = [_slugify_root(sample_root)]
    if app:
        parts.append(app)
    leaf = f"{task_id}#{rollout_index}"
    if step is not None:
        leaf += f"@step{step:03d}"
    parts.append(leaf)
    return "/".join(parts)


def assert_unique_sample_ids(records: Sequence[Mapping[str, object]]) -> None:
    """Raise if any ``sample_id`` repeats. Called on every merge."""
    seen: dict[str, int] = {}
    dupes: dict[str, int] = {}
    for i, rec in enumerate(records):
        if "sample_id" not in rec:
            raise MissingFieldError(f"records[{i}].sample_id", available=list(rec.keys()))
        sid = str(rec["sample_id"])
        if sid in seen:
            dupes[sid] = dupes.get(sid, 1) + 1
        seen[sid] = i
    if dupes:
        preview = sorted(dupes.items(), key=lambda kv: -kv[1])[:10]
        raise SchemaError(
            f"{len(dupes)} duplicate sample_id(s) across {len(records)} records "
            f"(defect #17 - slug collision across sample roots). Worst: {preview!r}"
        )


# ---------------------------------------------------------------------------
# task-level split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSplit:
    """A task-level partition. ``train`` and ``val`` are disjoint task-id sets."""

    train: frozenset[str]
    val: frozenset[str]

    def __post_init__(self) -> None:
        overlap = self.train & self.val
        if overlap:
            raise LeakError(
                f"{len(overlap)} task(s) in both train and val: {sorted(overlap)[:10]!r}"
            )

    @property
    def n_tasks(self) -> int:
        return len(self.train) + len(self.val)


def _stable_task_hash(task_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}\x00{task_id}".encode()).hexdigest()[:16], 16)


def task_level_split(
    task_ids: Iterable[str], *, val_fraction: float, salt: str = "rft-v1"
) -> TaskSplit:
    """Deterministically split TASK IDS (never records) into train/val.

    The hash is over the task id alone, so every rollout of a task is assigned
    identically no matter how many rollouts exist or what order they arrive in.
    Re-running with the same ``salt`` reproduces the split exactly, which is what
    makes val numbers comparable across runs.
    """
    if not 0.0 < val_fraction < 1.0:
        raise SchemaError(f"val_fraction must be in (0,1), got {val_fraction}")
    unique = sorted(set(task_ids))
    if not unique:
        raise SchemaError("no task ids to split")
    threshold = int(val_fraction * (1 << 64))
    val = {t for t in unique if _stable_task_hash(t, salt) % (1 << 64) < threshold}
    train = set(unique) - val
    if not train or not val:
        raise SchemaError(
            f"degenerate split at val_fraction={val_fraction} over {len(unique)} tasks: "
            f"{len(train)} train / {len(val)} val"
        )
    return TaskSplit(train=frozenset(train), val=frozenset(val))


def partition_records(
    records: Sequence[Mapping[str, object]], split: TaskSplit, *, task_key: str = "task_id"
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    """Route records to (train, val) by their task id.

    Raises:
        MissingFieldError: a record has no task id. A record whose task is
            unknown cannot be split safely, so it is an error, not a default.
        LeakError: a record's task is in neither side of the split.
    """
    train_out: list[Mapping[str, object]] = []
    val_out: list[Mapping[str, object]] = []
    for i, rec in enumerate(records):
        if task_key not in rec:
            raise MissingFieldError(f"records[{i}].{task_key}", available=list(rec.keys()))
        tid = str(rec[task_key])
        if tid in split.val:
            val_out.append(rec)
        elif tid in split.train:
            train_out.append(rec)
        else:
            raise LeakError(
                f"records[{i}] has task_id {tid!r} which is in neither side of the split; "
                "an unassigned task must not be silently dropped or trained on"
            )
    return train_out, val_out


# ---------------------------------------------------------------------------
# held-out eval-leak guard
# ---------------------------------------------------------------------------


def load_task_ids(path: str | Path) -> frozenset[str]:
    """Load a task-id list from ``.json`` (list or ``{"task_ids": [...]}``),
    ``.jsonl`` (objects with ``task_id``/``id``), or ``.txt`` (one per line)."""
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"task-id list not found: {p}")
    text = p.read_text()
    if p.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            for key in ("task_ids", "tasks", "ids"):
                if key in payload:
                    items = payload[key]
                    break
            else:
                # {app: [task_id, ...]} - the OSWorld test_all.json shape.
                items = [t for v in payload.values() if isinstance(v, list) for t in v]
        else:
            raise SchemaError(f"unsupported JSON shape in {p}: {type(payload).__name__}")
        out = {str(it["task_id"] if isinstance(it, dict) else it) for it in items}
    elif p.suffix == ".jsonl":
        out = set()
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            for key in ("task_id", "id"):
                if key in obj:
                    out.add(str(obj[key]))
                    break
            else:
                raise MissingFieldError(f"{p}:{line_no}.task_id", available=list(obj.keys()))
    else:
        out = {ln.strip() for ln in text.splitlines() if ln.strip()}
    if not out:
        raise SchemaError(f"{p} contained no task ids")
    return frozenset(out)


def assert_no_leak(
    train_task_ids: Iterable[str], heldout_task_ids: Iterable[str], *, context: str = ""
) -> None:
    """Raise :class:`LeakError` if any training task is in the held-out set.

    Called unconditionally by the record builder. There is no flag to skip it.
    """
    train = set(map(str, train_task_ids))
    held = set(map(str, heldout_task_ids))
    if not held:
        raise SchemaError("held-out set is empty; a leak check against nothing proves nothing")
    overlap = train & held
    if overlap:
        where = f" ({context})" if context else ""
        raise LeakError(
            f"EVAL LEAK{where}: {len(overlap)} of {len(train)} training tasks are in the "
            f"{len(held)}-task held-out set: {sorted(overlap)[:10]!r}"
        )
