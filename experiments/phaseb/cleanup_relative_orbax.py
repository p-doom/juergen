#!/usr/bin/env python3
"""Delete only the sealed Phase-B relative source Orbax tree."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from cleanup_orbax import CleanupError, completed, load, sha256, validate_relative


def allocated_bytes(root: Path) -> int:
    return sum((Path(directory) / name).stat().st_blocks * 512
               for directory, _subdirs, files in os.walk(root)
               for name in files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        entry = validate_relative(
            source=args.source, model=args.model, evaluation=args.evaluation
        )
        analysis_path = args.analysis.resolve() / "matched_report.json"
        analysis = load(analysis_path)
        sources = analysis.get("sources", {})
        expected = {
            "relative_training_job_id": entry["source_job_id"],
            "relative_export_job_id": entry["export_job_id"],
            "relative_eval_job_id": entry["eval_job_id"],
            "relative_train_manifest_sha256": entry["train_manifest_sha256"],
            "relative_export_manifest_sha256": entry["export_manifest_sha256"],
            "relative_eval_manifest_sha256": entry["eval_manifest_sha256"],
        }
        bad = {key: (sources.get(key), value) for key, value in expected.items()
               if sources.get(key) != value}
        if (analysis.get("artifact_type") != "phaseb_matched_natural_prose_analysis"
                or analysis.get("status") != "complete" or bad):
            raise CleanupError(f"matched analysis lineage mismatch: {bad}")
        analysis_job = "135580"
        completed(analysis_job)
        target = Path(entry["target"])
        logical_before = entry["logical_bytes_before"]
        allocated_before = allocated_bytes(target)
        retained = {
            Path(entry["train_manifest"]): entry["train_manifest_sha256"],
            Path(entry["export_manifest"]): entry["export_manifest_sha256"],
            Path(entry["eval_manifest"]): entry["eval_manifest_sha256"],
            analysis_path: sha256(analysis_path),
        }
        shutil.rmtree(target)
        target.mkdir()
        if any(target.iterdir()):
            raise CleanupError("relative Orbax tombstone is not empty")
        for path, digest in retained.items():
            if not path.is_file() or sha256(path) != digest:
                raise CleanupError(f"retained artifact changed: {path}")
        payload = {
            "artifact_type": "phaseb_relative_source_orbax_cleanup",
            "schema_version": 1, "status": "complete",
            "execution": "CPU-only Slurm job",
            "target": str(target), "tombstone_empty": True,
            "logical_bytes_removed": logical_before,
            "allocated_bytes_removed": allocated_before,
            "source_job_id": entry["source_job_id"],
            "export_job_id": entry["export_job_id"],
            "eval_job_id": entry["eval_job_id"],
            "analysis_job_id": analysis_job,
            "retained": {str(path): digest for path, digest in retained.items()},
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "cleanup_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    except CleanupError as exc:
        raise SystemExit(f"FATAL relative-only Orbax cleanup: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
