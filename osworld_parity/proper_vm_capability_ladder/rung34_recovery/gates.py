from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REQUIRED_GATES = ("roadmap_3_1", "roadmap_3_2", "roadmap_3_3")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EarlierGateError(RuntimeError):
    pass


def require_earlier_gate_evidence(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EarlierGateError(f"cannot read earlier-gate evidence: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise EarlierGateError("earlier-gate evidence schema mismatch")
    if raw.get("base_commit") != "48a54e8585eb9d6abff31e2ba6ea857c946a7d3d":
        raise EarlierGateError("earlier-gate base commit mismatch")
    gates = raw.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATES):
        raise EarlierGateError("earlier-gate evidence set mismatch")
    for name in REQUIRED_GATES:
        evidence = gates[name]
        if not isinstance(evidence, dict) or evidence.get("status") != "passed":
            raise EarlierGateError(f"earlier gate has not passed: {name}")
        if not SHA256.fullmatch(str(evidence.get("artifact_sha256", ""))):
            raise EarlierGateError(f"earlier gate artifact commitment missing: {name}")
    return raw
