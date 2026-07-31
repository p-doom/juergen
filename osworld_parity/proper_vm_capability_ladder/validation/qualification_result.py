from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifact_index import PINNED_SUBSTRATE_SHA256, sha256_file


class QualificationResultError(RuntimeError):
    pass


def validate_result(*, kind: str, path: Path, provider: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "passed":
        raise QualificationResultError(f"{kind} result is not a passed JSON object")
    expected_counts = {"preflight": 10, "rung1a": 16, "rung1b": 12, "sameapp": 8}
    if kind == "preflight":
        observed = value.get("trial_count")
    elif kind in {"rung1a", "rung1b"}:
        observed = len(value.get("cells", []))
    elif kind == "sameapp":
        observed = len(value.get("rows", []))
        if value.get("mode") != "vm" or value.get("split") != "development":
            raise QualificationResultError("same-app result is not a development VM replay")
        if value.get("sealed_eval_executed") is not False:
            raise QualificationResultError("same-app result opened sealed evaluation")
    else:
        raise QualificationResultError(f"unsupported qualification result kind: {kind}")
    if observed != expected_counts[kind]:
        raise QualificationResultError(
            f"{kind} result count {observed!r} != {expected_counts[kind]}"
        )
    provider_sha = sha256_file(provider)
    if provider_sha != PINNED_SUBSTRATE_SHA256["provider"]:
        raise QualificationResultError("qualification provider hash differs from the pin")
    exact = {
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }
    for key, wanted in exact.items():
        if value.get(key) != wanted:
            raise QualificationResultError(
                f"{kind} producer must emit {key}={wanted!r} explicitly"
            )
    recorded_provider = value.get("provider_sha256")
    if recorded_provider is None and isinstance(value.get("provider"), dict):
        recorded_provider = value["provider"].get("sha256")
    if recorded_provider != provider_sha:
        raise QualificationResultError(
            f"{kind} producer did not bind the pinned provider hash"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("preflight", "rung1a", "rung1b", "sameapp"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate_result(
        kind=args.kind, path=args.result.resolve(), provider=args.provider.resolve()
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
