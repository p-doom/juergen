"""inspect_ai runner. Invokes the ``inspect`` binary from this same uv venv
as a subprocess, then parses the produced eval log for scores."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_inspect_eval(
    *,
    task: str,
    model: str,
    server_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    log_dir: Path,
    limit: int | None = None,
) -> tuple[dict, int, int]:
    """Run ``inspect eval <task>`` against the running SGLang server.

    Returns ``(scores, n_samples, elapsed_s)``.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    inspect_bin = str(Path(sys.prefix) / "bin" / "inspect")
    if not Path(inspect_bin).is_file():
        raise FileNotFoundError(
            f"inspect binary not found at {inspect_bin} — is the eval venv missing inspect-ai?"
        )
    env = {
        **os.environ,
        "SGLANG_BASE_URL": server_url,
        "SGLANG_API_KEY": api_key,
        "INSPECT_LOG_FORMAT": "json",
    }
    cmd = [
        inspect_bin,
        "eval",
        task,
        "--model",
        f"sglang/{model}",
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--seed",
        str(seed),
        "--log-dir",
        str(log_dir),
        "--display",
        "plain",
        # inspect-ai defaults to 10 concurrent requests; bumping lets sglang
        # batch generations and dramatically shortens wall-clock on verbose
        # models (post-OPD checkpoints). Override via INSPECT_MAX_CONNECTIONS.
        "--max-connections",
        os.environ.get("INSPECT_MAX_CONNECTIONS", "64"),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    print(f"[inspect] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, check=False).returncode
    elapsed_s = int(time.time() - t0)
    if rc != 0:
        raise RuntimeError(f"inspect eval failed (rc={rc})")
    scores, n_samples = parse_eval_log(log_dir)
    return scores, n_samples, elapsed_s


def parse_eval_log(log_dir: Path) -> tuple[dict, int]:
    """Find the most recent eval log under ``log_dir`` and extract scores.

    inspect_ai's JSON eval log has ``results.scores`` as a list of
    ``{"name": ..., "metrics": {<metric_name>: {"value": ..., ...}}}``
    entries. Flatten to ``{<score_name>/<metric_name>: value}``.
    """
    candidates = sorted(log_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime)
    candidates = [p for p in candidates if not p.name.startswith(".")]
    if not candidates:
        raise RuntimeError(f"no inspect logs found under {log_dir}")
    log_file = candidates[-1]
    print(f"[inspect] parsing {log_file}", flush=True)
    log = json.loads(log_file.read_text())

    scores: dict = {}
    results = log.get("results") or {}
    for s in results.get("scores") or []:
        name = s.get("name", "?")
        metrics = s.get("metrics") or {}
        for metric_name, metric_val in metrics.items():
            v = metric_val.get("value") if isinstance(metric_val, dict) else metric_val
            scores[f"{name}/{metric_name}"] = v
    n_samples = len(log.get("samples") or [])
    return scores, n_samples
