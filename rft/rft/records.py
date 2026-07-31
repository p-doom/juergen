"""Stage 3 - Build records: convert accepted rollouts into SFT chat records.

The *conversion* is domain-specific and stays where it is (the per-format
converters). What this module owns is the plumbing that kept going wrong:

* a **mandatory round-trip audit** of every converted assistant target through the
  exact eval parser (:mod:`rft.roundtrip`) — not a spot check, and not skippable;
* a **mandatory leak check** against the held-out task list (:func:`rft.splits.assert_no_leak`);
* **TASK-level** splitting (defect #16 — a record-level split scattered one task's
  ``k`` sibling trajectories across both sides);
* collision-proof ``sample_id``s (defect #17 — ``slug = app__task_id`` collides
  across sample roots, so merging two sampling runs produced duplicates);
* a declared grammar on every record (:func:`rft.roundtrip.assert_convention_declared`),
  so a relative delta can never be read as an absolute coordinate.

:func:`build_records` runs all of it and refuses to write anything if any check
fails. There is no ``--skip-audit`` flag; adding one would defeat the module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rft.arms import ArmProfile
from rft.conversion import (
    ContextAlteredError,
    ContextFeatures,
    assert_only_action_span_changed,
)
from rft.errors import MissingFieldError, SchemaError
from rft.grammars import get_grammar
from rft.label_leak import LabelLeakError, prose_digit_leak
from rft.roundtrip import (
    RoundTripReport,
    assert_convention_declared,
    assert_roundtrip_clean,
    audit_roundtrip,
)
from rft.splits import (
    TaskSplit,
    assert_no_leak,
    assert_unique_sample_ids,
    load_task_ids,
    make_sample_id,
    partition_records,
    task_level_split,
)

#: A converter turns one accepted rollout into zero or more chat records. It gets
#: the rollout payload and must return records whose final message is the assistant
#: target written in ``grammar``. Injected so the per-format conversion logic stays
#: in its own module and this stage stays generic.
#:
#: A converter should be written with :func:`rft.conversion.convert_action_span`, so
#: it can only rewrite the action span and physically cannot delete the reasoning
#: preamble. :func:`build_records` checks that with ``source_text_key``.
ConverterFn = Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]


@dataclass
class BuildReport:
    """Stage-3 outcome, written beside the records as ``build_manifest.json``."""

    grammar: str
    n_rollouts_in: int = 0
    n_records: int = 0
    n_tasks: int = 0
    n_train_records: int = 0
    n_val_records: int = 0
    n_train_tasks: int = 0
    n_val_tasks: int = 0
    roundtrip: dict[str, Any] = field(default_factory=dict)
    leak_check: dict[str, Any] = field(default_factory=dict)
    converter_skips: list[str] = field(default_factory=list)
    #: Aggregate context profile of this arm (reasoning-preamble / tools-schema /
    #: action-marker counts). Recorded so a later cross-arm parity audit needs only
    #: the manifests, not the raw data.
    context_profile: dict[str, Any] = field(default_factory=dict)
    #: Per-record verification that the conversion touched ONLY the action span.
    context_audit: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"stage=build_records grammar={self.grammar} "
            f"rollouts_in={self.n_rollouts_in} records={self.n_records} "
            f"tasks={self.n_tasks} -> train {self.n_train_records} rec/"
            f"{self.n_train_tasks} tasks, val {self.n_val_records} rec/"
            f"{self.n_val_tasks} tasks\n  {self.roundtrip.get('summary', 'no roundtrip')}\n"
            f"  {self.leak_check.get('summary', 'no leak check')}\n"
            f"  {self.context_audit.get('summary', 'no context audit')}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "grammar": self.grammar,
            "n_rollouts_in": self.n_rollouts_in,
            "n_records": self.n_records,
            "n_tasks": self.n_tasks,
            "n_train_records": self.n_train_records,
            "n_val_records": self.n_val_records,
            "n_train_tasks": self.n_train_tasks,
            "n_val_tasks": self.n_val_tasks,
            "roundtrip": self.roundtrip,
            "leak_check": self.leak_check,
            "converter_skips": self.converter_skips[:100],
            "context_profile": self.context_profile,
            "context_audit": self.context_audit,
        }


def assistant_target(record: Mapping[str, Any]) -> str:
    """Extract the final assistant turn's text from a chat record.

    Raises:
        MissingFieldError / SchemaError: the record has no final assistant turn, or
            its content is not text. A record whose target cannot be read cannot be
            audited, so it is an error rather than a skip.
    """
    if "messages" not in record:
        raise MissingFieldError("record.messages", available=list(record.keys()))
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise SchemaError(f"record.messages is {type(messages).__name__} or empty")
    last = messages[-1]
    if not isinstance(last, Mapping) or last.get("role") != "assistant":
        raise SchemaError(
            f"final message role is {last.get('role') if isinstance(last, Mapping) else '?'!r}, "
            "expected 'assistant' - the training target must be the last turn"
        )
    content = last.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            part["text"]
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        if len(texts) == 1:
            return texts[0]
        if not texts:
            raise SchemaError("final assistant turn has no text part")
        raise SchemaError(
            f"final assistant turn has {len(texts)} text parts; a training target must "
            "be a single text block"
        )
    raise SchemaError(f"final assistant content is {type(content).__name__}")


def build_records(
    rollouts: Iterable[Mapping[str, Any]],
    *,
    grammar: str,
    convert: ConverterFn,
    out_dir: str | Path,
    heldout_tasks_path: str | Path,
    sample_root: str | Path | None = None,
    val_fraction: float = 0.1,
    split_salt: str = "rft-v1",
    task_key: str = "task_id",
    write: bool = True,
    source_text_key: str | None = "source_response",
    context_opt_out_reason: str | None = None,
) -> tuple[BuildReport, TaskSplit]:
    """Convert, audit, leak-check, split, and write.

    Order matters and is enforced: the round-trip audit and the leak check both run
    **before** anything is written, so a failed audit cannot leave a half-written
    dataset that a later stage picks up.

    Args:
        rollouts: accepted rollouts from stage 2.
        grammar: the action grammar the converter emits. Validated against the
            registry, and stamped on every record.
        convert: per-rollout converter (see :data:`ConverterFn`).
        out_dir: destination. ``_normalized/{train,val}/chat.jsonl`` plus
            ``build_manifest.json``, matching the layout the tokenizer expects.
        heldout_tasks_path: the 110-task held-out list. Not optional.
        sample_root: root used for ``sample_id`` uniqueness; defaults to ``out_dir``.
        val_fraction: TASK-level val fraction.
        source_text_key: key on each produced record holding the ORIGINAL (pre-
            conversion) response text. When present, every record is checked with
            :func:`rft.conversion.assert_only_action_span_changed`, which is the gate
            against the absolute-vs-relative confound: the converter kept the
            reasoning preamble in the absolute arm and deleted it from every relative
            arm (2383/2383 vs 0/2441), silently invalidating the comparison. Set to
            ``None`` only when no source text exists to compare against.
        context_opt_out_reason: permit records whose conversion did alter context, at
            the cost of writing the reason into the manifest. Free text on purpose:
            somebody has to justify it and the justification travels with the data.

    Raises:
        RoundTripError: any converted target fails to survive the eval parser.
        LeakError: any training task is in the held-out set.
        ContextAlteredError: a conversion changed content outside the action span.
        SchemaError: no records were produced, or ids collide, or a record has no
            declared grammar.
    """
    get_grammar(grammar)
    out = Path(out_dir)
    root = Path(sample_root) if sample_root is not None else out
    report = BuildReport(grammar=grammar)

    records: list[dict[str, Any]] = []
    for rollout in rollouts:
        report.n_rollouts_in += 1
        if task_key not in rollout:
            raise MissingFieldError(f"rollout.{task_key}", available=list(rollout.keys()))
        task_id = str(rollout[task_key])
        produced = convert(rollout)
        if not isinstance(produced, Sequence):
            raise SchemaError(
                f"convert() returned {type(produced).__name__} for task {task_id}, "
                "expected a sequence of records"
            )
        if not produced:
            report.converter_skips.append(
                f"{rollout.get('sample_id', task_id)}: converter produced 0 records"
            )
            continue
        rollout_index = int(rollout.get("rollout_index", 0))
        for i, rec in enumerate(produced):
            item = dict(rec)
            item["task_id"] = task_id
            item["grammar"] = grammar
            item.setdefault("app", rollout.get("app"))
            item.setdefault("source_sample_id", rollout.get("sample_id"))
            item.setdefault("rollout_index", rollout_index)
            # A record id must be unique across (root, task, ROLLOUT, step). Keying it
            # on step alone collapses the k siblings of a task onto one id - defect
            # #17 all over again, one level down.
            item["sample_id"] = make_sample_id(
                sample_root=root,
                task_id=task_id,
                rollout_index=rollout_index,
                app=item.get("app"),
                step=int(item.get("step", i)),
            )
            records.append(item)

    if not records:
        raise SchemaError(
            f"0 records produced from {report.n_rollouts_in} rollouts; refusing to "
            f"write an empty dataset. Converter skips: {report.converter_skips[:5]!r}"
        )
    report.n_records = len(records)

    assert_unique_sample_ids(records)
    assert_convention_declared(records)

    # --- action-span isolation: the conversion may touch NOTHING else ---------
    profile = ArmProfile(name=grammar)
    n_context_checked = 0
    context_violations: list[str] = []
    for rec in records:
        target = assistant_target(rec)
        profile.add(ContextFeatures.of(target))
        source = rec.get(source_text_key) if source_text_key else None
        if not isinstance(source, str):
            continue
        n_context_checked += 1
        try:
            assert_only_action_span_changed(
                source, target, context=f"{rec['sample_id']} -> {grammar}"
            )
        except ContextAlteredError as exc:
            context_violations.append(str(exc))
    report.context_profile = profile.as_dict()
    report.context_audit = {
        "source_text_key": source_text_key,
        "n_records_checked": n_context_checked,
        "n_violations": len(context_violations),
        "opt_out_reason": context_opt_out_reason,
        "violations": context_violations[:5],
        "summary": (
            f"action-span isolation: {n_context_checked}/{len(records)} records had a "
            f"source response to compare; {len(context_violations)} altered context "
            f"outside the action span"
        ),
    }
    # --- within-record label leakage: auxiliary text must not carry the label --
    leak = prose_digit_leak(records)
    report.context_audit["label_leak"] = leak.as_dict()
    report.context_audit["label_leak_summary"] = leak.describe()
    if not leak.clean:
        raise LabelLeakError(
            f"{leak.n_records_leaking} of {leak.n_records} records leak label values "
            "into their prose or user turn. That turns the task into text arithmetic "
            "and any accuracy measured on it is a false positive.\n" + leak.describe()
        )

    if context_violations and not context_opt_out_reason:
        raise ContextAlteredError(
            f"{len(context_violations)} of {n_context_checked} conversions to "
            f"{grammar!r} changed content OUTSIDE the action span. Express the "
            "converter through rft.conversion.convert_action_span so it cannot, or "
            "pass context_opt_out_reason=... to record the deviation in the manifest."
            "\n\nFirst violation:\n" + context_violations[0]
        )

    # --- mandatory round-trip audit through the EXACT eval parser -------------
    rt: RoundTripReport = audit_roundtrip(
        ((str(r["sample_id"]), assistant_target(r)) for r in records), grammar=grammar
    )
    report.roundtrip = rt.as_dict()
    report.roundtrip["summary"] = rt.describe()
    assert_roundtrip_clean(rt)

    # --- mandatory leak check ------------------------------------------------
    heldout = load_task_ids(heldout_tasks_path)
    task_ids = {str(r["task_id"]) for r in records}
    report.n_tasks = len(task_ids)
    assert_no_leak(task_ids, heldout, context=f"stage3 build_records grammar={grammar}")
    report.leak_check = {
        "heldout_path": str(heldout_tasks_path),
        "n_heldout": len(heldout),
        "n_train_adjacent_tasks": len(task_ids),
        "overlap": 0,
        "summary": (
            f"no leak: 0 of {len(task_ids)} record tasks appear in the "
            f"{len(heldout)}-task held-out set ({heldout_tasks_path})"
        ),
    }

    # --- TASK-level split ----------------------------------------------------
    split = task_level_split(task_ids, val_fraction=val_fraction, salt=split_salt)
    train, val = partition_records(records, split, task_key="task_id")
    report.n_train_records = len(train)
    report.n_val_records = len(val)
    report.n_train_tasks = len(split.train)
    report.n_val_tasks = len(split.val)

    if write:
        for name, recs in (("train", train), ("val", val)):
            d = out / "_normalized" / name
            d.mkdir(parents=True, exist_ok=True)
            with (d / "chat.jsonl").open("w") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.mkdir(parents=True, exist_ok=True)
        (out / "build_manifest.json").write_text(json.dumps(report.as_dict(), indent=2))
        (out / "split_task_ids.json").write_text(
            json.dumps(
                {"train": sorted(split.train), "val": sorted(split.val), "salt": split_salt},
                indent=2,
            )
        )
    return report, split


def verify_written_split(out_dir: str | Path, heldout_tasks_path: str | Path) -> dict[str, Any]:
    """Re-verify an already-written dataset: task-level split + no held-out leak.

    Cheap, so it runs at the start of stage 4 as well. A dataset that passed the
    check at build time can still be the wrong dataset by the time training reads it.
    """
    out = Path(out_dir)
    per_split: dict[str, set[str]] = {}
    for name in ("train", "val"):
        path = out / "_normalized" / name / "chat.jsonl"
        if not path.is_file():
            raise SchemaError(f"missing {path}")
        tasks: set[str] = set()
        n = 0
        with path.open() as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "task_id" not in obj:
                    raise MissingFieldError(f"{path}:{line_no}.task_id")
                tasks.add(str(obj["task_id"]))
                n += 1
        per_split[name] = tasks
        per_split[f"_n_{name}"] = n  # type: ignore[assignment]
    overlap = per_split["train"] & per_split["val"]
    if overlap:
        raise SchemaError(
            f"TASK-LEVEL SPLIT VIOLATION: {len(overlap)} task(s) appear in both train and "
            f"val chat.jsonl: {sorted(overlap)[:10]!r} (defect #16)"
        )
    heldout = load_task_ids(heldout_tasks_path)
    assert_no_leak(per_split["train"] | per_split["val"], heldout, context=str(out))
    return {
        "n_train_records": per_split["_n_train"],
        "n_val_records": per_split["_n_val"],
        "n_train_tasks": len(per_split["train"]),
        "n_val_tasks": len(per_split["val"]),
        "task_overlap": 0,
        "heldout_leak": 0,
    }
