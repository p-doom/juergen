"""CPU-only gate verifier for a future ROADMAP 3.5 pilot release."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .gates import GateBundle, GateError, SignedGatePaths, verify_gate_bundle
from .io import atomic_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prerequisites-gate", type=Path, required=True)
    parser.add_argument("--prerequisites-signature", type=Path, required=True)
    parser.add_argument("--pilot-release-gate", type=Path, required=True)
    parser.add_argument("--pilot-release-signature", type=Path, required=True)
    parser.add_argument("--allowed-signers", type=Path, required=True)
    parser.add_argument("--signer-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = GateBundle(
        prerequisites=SignedGatePaths(
            args.prerequisites_gate, args.prerequisites_signature
        ),
        pilot_release=SignedGatePaths(
            args.pilot_release_gate, args.pilot_release_signature
        ),
        allowed_signers=args.allowed_signers,
        signer_identity=args.signer_identity,
    )
    try:
        authorization = verify_gate_bundle(bundle)
        atomic_json(args.output, authorization.as_dict())
    except GateError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
