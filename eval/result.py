"""Helper for writing the eval result.json that pmanager ingests as a metric.

Per pipeline_task contract for ``cfg.kind = "eval"``: the dispatcher detects
completion via ``<output_dir>/result.json`` and copies its contents into
``runs.result_json``. Schema is intentionally rich (params + inputs +
elapsed) so that registry queries can correlate metrics with the config that
produced them without joining additional tables.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def write_result(
    result_path: Path,
    *,
    task: str,
    scores: dict,
    params: dict,
    inputs: dict,
    n_samples: int | None = None,
    elapsed_s: int | None = None,
    extra: dict | None = None,
) -> None:
    """Atomic write of result.json (write + rename)."""
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "task": task,
        "scores": scores,
        "params": params,
        "inputs": inputs,
        "n_samples": n_samples,
        "elapsed_s": elapsed_s,
        "completed_at": int(time.time()),
        "pmanager_run_id": os.environ.get("PMANAGER_RUN_ID", ""),
        "pmanager_parent_run_id": os.environ.get("PMANAGER_PARENT_RUN_ID", ""),
        "pmanager_parent_step": os.environ.get("PMANAGER_PARENT_STEP", ""),
        **(extra or {}),
    }
    tmp = result_path.with_suffix(result_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(result_path)
