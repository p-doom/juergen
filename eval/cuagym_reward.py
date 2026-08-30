"""Post-episode CUA-Gym reward: upload reward.py into the guest, run it with
the guest python3 via the OSWorld /setup/execute endpoint, and parse the last
``REWARD: <float>`` line from stdout. Any failure returns reward=None
(quarantine); stdout/stderr are always copied into the episode dir.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests

REWARD_RE = re.compile(r"REWARD:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

DEFAULT_GUEST_PATH = "/home/user/.cuagym_reward.py"
DEFAULT_TIMEOUT_S = 300.0
GUEST_WHEELS_DIR = "/home/user/.cuagym_wheels"


@dataclass(frozen=True)
class RewardOutcome:
    reward: float | None
    stdout: str
    stderr: str
    error: str | None


def parse_reward(stdout: str) -> float | None:
    matches = REWARD_RE.findall(stdout or "")
    if not matches:
        return None
    return float(matches[-1])


def _upload_reward_script(setup_controller, reward_script: Path, guest_path: str) -> None:
    setup_controller._upload_file_setup(
        files=[{"local_path": str(reward_script), "path": guest_path}]
    )


def _execute_reward_script(
    http_server: str, guest_path: str, timeout_s: float
) -> tuple[str, str]:
    payload = json.dumps({"command": ["python3", guest_path], "shell": False})
    response = requests.post(
        http_server + "/setup/execute",
        headers={"Content-Type": "application/json"},
        data=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    results = response.json()
    return results.get("output") or "", results.get("error") or ""


def _run_guest_command(http_server: str, command: list[str], timeout_s: float) -> tuple[str, str]:
    payload = json.dumps({"command": command, "shell": False})
    response = requests.post(
        http_server + "/setup/execute",
        headers={"Content-Type": "application/json"},
        data=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    results = response.json()
    return results.get("output") or "", results.get("error") or ""


def bootstrap_wheels(env, wheels_dir: Path | str, timeout_s: float = 600.0) -> tuple[str, str]:
    wheels = sorted(Path(wheels_dir).glob("*.whl"))
    if not wheels:
        return "", "no wheels found"
    env.setup_controller._upload_file_setup(
        files=[{"local_path": str(w), "path": f"{GUEST_WHEELS_DIR}/{w.name}"} for w in wheels]
    )
    return _run_guest_command(
        env.setup_controller.http_server,
        [
            "python3",
            "-m",
            "pip",
            "install",
            "--user",
            "--no-index",
            f"--find-links={GUEST_WHEELS_DIR}",
            "openpyxl",
            "python-docx",
            "python-pptx",
            "pymupdf",
            "PyPDF2",
            "pypdf",
            "pandas",
            "odfpy",
        ],
        timeout_s,
    )


def _dump_outcome(outcome: RewardOutcome, episode_dir: Path | str) -> None:
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "reward_stdout.txt").write_text(outcome.stdout)
    (episode_dir / "reward_stderr.txt").write_text(outcome.stderr)
    if outcome.error:
        (episode_dir / "reward_error.txt").write_text(outcome.error)


def compute_reward(
    env,
    reward_script: Path | str,
    episode_dir: Path | str,
    guest_path: str = DEFAULT_GUEST_PATH,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> RewardOutcome:
    stdout = ""
    stderr = ""
    try:
        _upload_reward_script(env.setup_controller, Path(reward_script), guest_path)
        stdout, stderr = _execute_reward_script(
            env.setup_controller.http_server, guest_path, timeout_s
        )
    except Exception as exc:
        outcome = RewardOutcome(
            reward=None,
            stdout=stdout,
            stderr=stderr,
            error=f"{type(exc).__name__}: {exc}",
        )
        _dump_outcome(outcome, episode_dir)
        return outcome
    reward = parse_reward(stdout)
    error = None if reward is not None else "no REWARD: line in reward script stdout"
    outcome = RewardOutcome(reward=reward, stdout=stdout, stderr=stderr, error=error)
    _dump_outcome(outcome, episode_dir)
    return outcome
