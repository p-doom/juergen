"""Suite loading, bundle verification, and one-time bundle extraction.

The suite pins every bundle file by sha256 against dataset revision
`suite.json:dataset_revision`. Verification is the revision pin: the HF
snapshot does not need to be present at eval time, but a bundle whose bytes
drifted from the pinned revision refuses to load rather than silently scoring
a different task.

Extraction (`python -m evals.cuagym.bundles --dataset-root ... --out ...`) is
the only step that touches the snapshot. It streams the zstd tar once and
writes only the suite's members; AppleDouble `._*` entries are dropped because
they are archive cruft, not task content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

__all__ = ["extract_bundles", "load_suite", "verify_bundle"]

_SUITE_PATH = Path(__file__).with_name("suite.json")


def load_suite(path: str | Path | None = None) -> dict[str, Any]:
    suite = json.loads(Path(path or _SUITE_PATH).read_text(encoding="utf-8"))
    for key in ("suite", "dataset_revision", "tasks"):
        if key not in suite:
            raise ValueError(f"suite file is missing {key!r}")
    for index, task in enumerate(suite["tasks"]):
        for key in ("id", "instruction", "sha256"):
            if key not in task:
                raise ValueError(f"suite task {index} is missing {key!r}")
    return suite


def verify_bundle(bundle_dir: Path, pins: dict[str, str], task_id: str) -> None:
    """Raise unless the bundle holds exactly the pinned bytes."""

    for name, expected in pins.items():
        path = bundle_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"bundle for {task_id} is missing {name!r} under {bundle_dir}"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"bundle file {task_id}/{name} does not match the suite's pin "
                f"(expected {expected[:12]}…, found {actual[:12]}…); the bundle "
                "was extracted from a different dataset revision or was edited"
            )


def extract_bundles(dataset_root: Path, out_dir: Path, suite: dict[str, Any]) -> int:
    """Extract the suite's bundles from the pinned snapshot archive."""

    import pyarrow as pa  # heavy import, extraction-only

    archive = (
        dataset_root
        / "snapshots"
        / str(suite["dataset_revision"])
        / str(suite["archive"])
    )
    if not archive.is_file():
        raise FileNotFoundError(
            f"pinned archive not found: {archive} — is --dataset-root the "
            "HF hub directory (datasets--xlangai--CUA-Gym)?"
        )
    wanted = {str(task["id"]) for task in suite["tasks"]}
    written = 0
    with pa.input_stream(str(archive), compression="zstd") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as tar:
            for member in tar:
                top, _, rest = member.name.partition("/")
                if top not in wanted or not member.isfile():
                    continue
                if Path(rest).name.startswith("._"):
                    continue
                target = out_dir / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(member)
                assert extracted is not None
                target.write_bytes(extracted.read())
                written += 1
    for task in suite["tasks"]:
        verify_bundle(out_dir / str(task["id"]), dict(task["sha256"]), str(task["id"]))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=None)
    args = parser.parse_args()
    suite = load_suite(args.suite)
    written = extract_bundles(args.dataset_root, args.out, suite)
    print(f"extracted {written} files for {len(suite['tasks'])} tasks into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
