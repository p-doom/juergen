#!/usr/bin/env python3
"""Prove the committed deltatype-v2 codec reproduces the sealed Phase-B dataset.

The s900 checkpoint's behaviour depends on the exact model-facing bytes of the
Phase-B dataset: the system prompt and every assistant action span. This script
re-derives those bytes from the *committed* codec and asserts identity against
the sealed artifact, so a refactor that changed behaviour cannot pass silently.

Checks, over ``{train,val}/chat.jsonl`` of the sealed dataset:

1. file SHA-256 identity against the pinned digests (or ``dataset_manifest.json``);
2. ``SYSTEM_PROMPT`` byte identity with every record's system message, and its
   SHA-256 identity with ``system_prompt_sha256`` in the manifest;
3. for every assistant action span: ``format_deltatype_v2(parse_deltatype_v2(s)) == s``
   (byte round-trip through the committed codec);
4. the span equals the ``label`` recorded in that record's ``raw_deltatype_v2_audit``;
5. ``ordered_plan`` invariance: replaying each task's cursor from the screen
   centre reproduces every recorded ``command_plan`` exactly.

Defaults point at the sealed artifact on the hai cluster; override with
``--dataset`` to run elsewhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from action_v2 import format_deltatype_v2, ordered_plan, parse_deltatype_v2
from prompt import SYSTEM_PROMPT

DEFAULT_DATASET = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/franz.srambical"
    "/phaseb_raw_deltatype_v2_build_audit_v1_run_019fb5a5564e7a71b3ad6e55426af463"
)
DEFAULT_VENDOR = Path(__file__).with_name("vendor")

SEALED_SHA256 = {
    "train/chat.jsonl": (
        "5f449f3d57b368e55cfe2ba486bcdd9953aa6f9bad343948e0b8653b2ab4de99"
    ),
    "val/chat.jsonl": (
        "a819011d5f8524cad1980d720fcdbc98a838a37b33de499c46eb4c13c94acadd"
    ),
}
SEALED_SYSTEM_PROMPT_SHA256 = (
    "57f7d0b230974068618b48151b73215d5517d5445a99dbf5abdc05557e3482e6"
)
SEALED_ASSISTANT_SPANS = 10721
SW, SH = 1920, 1080


class VerifyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_assistant_turn(vendor: Path) -> Any:
    """Load the audited action-span splitter (vendored byte-identically)."""
    import importlib.util

    path = vendor / "action_span_conversion.py"
    spec = importlib.util.spec_from_file_location("sealed_action_span_conversion", path)
    if spec is None or spec.loader is None:
        raise VerifyError(f"cannot load vendored converter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.split_assistant_turn


def text_of(message: dict[str, Any]) -> str:
    parts = [
        part
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    if len(parts) != 1 or not isinstance(parts[0].get("text"), str):
        raise VerifyError("message must carry exactly one text part")
    return parts[0]["text"]


def advance(cursor: tuple[int, int], action: Any) -> tuple[int, int]:
    """Cursor after ``action``, using the same clipping rule as ordered_plan."""
    if action.no_op or action.terminate or action.fail:
        return cursor
    nxt = (
        max(0, min(SW - 1, cursor[0] + action.dx)),
        max(0, min(SH - 1, cursor[1] + action.dy)),
    )
    for kind, value in action.elements:
        if kind == "move":
            nxt = (
                max(0, min(SW - 1, nxt[0] + value[0])),
                max(0, min(SH - 1, nxt[1] + value[1])),
            )
    return nxt


def verify(dataset: Path, vendor: Path) -> dict[str, Any]:
    split_assistant_turn = load_split_assistant_turn(vendor)

    manifest_path = dataset / "dataset_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_file_sha256") != SEALED_SHA256:
            raise VerifyError("manifest output digests differ from the pinned digests")
        if manifest.get("system_prompt_sha256") != SEALED_SYSTEM_PROMPT_SHA256:
            raise VerifyError("manifest system-prompt digest differs from the pin")

    prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if prompt_digest != SEALED_SYSTEM_PROMPT_SHA256:
        raise VerifyError(
            f"committed SYSTEM_PROMPT digest {prompt_digest} "
            f"!= sealed {SEALED_SYSTEM_PROMPT_SHA256}"
        )

    file_digests: dict[str, str] = {}
    spans = plans_checked = records = 0
    # (app, task_id) -> {mapped_step: (span, command_plan)} merged over prefixes
    tasks: dict[tuple[str, str], dict[int, tuple[str, list[list[Any]]]]] = {}

    for split in ("train", "val"):
        relative = f"{split}/chat.jsonl"
        path = dataset / relative
        digest = sha256_file(path)
        file_digests[relative] = digest
        if digest != SEALED_SHA256[relative]:
            raise VerifyError(
                f"{relative}: sha256 {digest} != sealed {SEALED_SHA256[relative]}"
            )
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                records += 1
                where = f"{relative}:{line_number}"
                if text_of(record["messages"][0]) != SYSTEM_PROMPT:
                    raise VerifyError(f"{where}: system prompt is not byte-identical")
                assistants = [
                    message
                    for message in record["messages"]
                    if message.get("role") == "assistant"
                ]
                audit = record["raw_deltatype_v2_audit"]
                if len(audit) != len(assistants):
                    raise VerifyError(f"{where}: audit/assistant turn count mismatch")
                key = (record["app"], record["task_id"])
                per_step = tasks.setdefault(key, {})
                for message, entry in zip(assistants, audit, strict=True):
                    _before, span, _after = split_assistant_turn(text_of(message))
                    parsed = parse_deltatype_v2(span)
                    if format_deltatype_v2(parsed) != span:
                        raise VerifyError(
                            f"{where}: codec round-trip is not byte-exact: {span!r}"
                        )
                    if entry["label"] != span:
                        raise VerifyError(
                            f"{where}: audit label {entry['label']!r} != span {span!r}"
                        )
                    spans += 1
                    step = entry["mapped_step"]
                    seen = per_step.get(step)
                    if seen is None:
                        per_step[step] = (span, entry["command_plan"])
                    elif seen != (span, entry["command_plan"]):
                        raise VerifyError(
                            f"{where}: prefix expansion disagrees at step {step}"
                        )

    for (app, task_id), per_step in sorted(tasks.items()):
        cursor = (SW // 2, SH // 2)
        for step in sorted(per_step):
            span, recorded = per_step[step]
            parsed = parse_deltatype_v2(span)
            derived = [list(command) for command in ordered_plan(parsed, cursor, (SW, SH))]
            if derived != recorded:
                raise VerifyError(
                    f"{app}/{task_id}:{step}: ordered_plan {derived!r} != "
                    f"recorded {recorded!r}"
                )
            plans_checked += 1
            cursor = advance(cursor, parsed)

    if spans != SEALED_ASSISTANT_SPANS:
        raise VerifyError(f"{spans} assistant spans != sealed {SEALED_ASSISTANT_SPANS}")

    return {
        "records": records,
        "assistant_spans_roundtripped": spans,
        "unique_decisions_plan_checked": plans_checked,
        "tasks": len(tasks),
        "file_sha256": file_digests,
        "system_prompt_sha256": prompt_digest,
        "manifest_cross_checked": bool(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    args = parser.parse_args()
    summary = verify(args.dataset, args.vendor)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("sealed deltatype-v2 dataset round-trip: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
