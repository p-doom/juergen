"""Strict parsing for CUA-Gym reward-process stdout."""

from __future__ import annotations

import math
import re

from .errors import RewardParseError
from .manifest import CompatibilityManifest
from .models import RewardOutputFormat, TaskCompatibility, TaskId

_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
_PREFIXED_RE = re.compile(rf"^\s*REWARD:\s*({_NUMBER})\s*$", re.IGNORECASE)
_ZERO_DIAGNOSTIC_RE = re.compile(
    rf"^\s*REWARD:\s*({_NUMBER})\s+—\s+(\S(?:.*\S)?)\s*$",
    re.IGNORECASE,
)
_BARE_RE = re.compile(rf"^\s*({_NUMBER})\s*$", re.IGNORECASE)
_TOTAL_SCORE_RE = re.compile(rf"^\s*Total score:\s*({_NUMBER})\s*$")


def parse_reward_stdout(
    task_id: str | TaskId,
    stdout: str,
    manifest: CompatibilityManifest,
) -> float:
    """Parse one finite reward in ``[0, 1]`` using manifest compatibility policy."""

    return parse_reward_output(task_id, stdout, manifest.task(task_id))


def parse_reward_output(
    task_id: str | TaskId,
    stdout: str,
    compatibility: TaskCompatibility,
) -> float:
    """Parse one finite reward in ``[0, 1]`` under an explicit task policy."""

    if not isinstance(stdout, str):
        raise RewardParseError("Reward stdout must be text")
    lines = [line for line in stdout.splitlines() if line.strip()]
    candidates: list[tuple[RewardOutputFormat, str, bool]] = []
    malformed_prefixed = False
    malformed_total_score = False
    for line in lines:
        prefixed_match = _PREFIXED_RE.fullmatch(line)
        if prefixed_match is not None:
            candidates.append(
                (RewardOutputFormat.REWARD_PREFIX, prefixed_match.group(1), False)
            )
        elif diagnostic_match := _ZERO_DIAGNOSTIC_RE.fullmatch(line):
            candidates.append(
                (RewardOutputFormat.REWARD_PREFIX, diagnostic_match.group(1), True)
            )
        elif "REWARD:" in line.upper():
            malformed_prefixed = True
        else:
            bare_match = _BARE_RE.fullmatch(line)
            if bare_match is not None:
                candidates.append(
                    (RewardOutputFormat.BARE_NUMBER, bare_match.group(1), False)
                )
                continue
            if compatibility.reward_output_format is RewardOutputFormat.TOTAL_SCORE:
                total_score_match = _TOTAL_SCORE_RE.fullmatch(line)
                if total_score_match is not None:
                    candidates.append(
                        (
                            RewardOutputFormat.TOTAL_SCORE,
                            total_score_match.group(1),
                            False,
                        )
                    )
                elif "TOTAL SCORE:" in line.upper():
                    malformed_total_score = True

    if malformed_prefixed:
        raise RewardParseError("Malformed REWARD line in reward stdout")
    if malformed_total_score:
        raise RewardParseError("Malformed Total score line in reward stdout")
    if len(set(candidates)) > 1:
        raise RewardParseError("Ambiguous reward stdout contains multiple candidates")
    if not candidates:
        raise RewardParseError("Reward stdout contains no numeric reward candidate")

    output_format, raw_value, is_zero_diagnostic = candidates[0]
    if output_format is not compatibility.reward_output_format:
        if output_format is RewardOutputFormat.BARE_NUMBER:
            raise RewardParseError(
                f"Task {task_id} is not approved for a bare reward value"
            )
        raise RewardParseError(
            f"Task {task_id} expects {compatibility.reward_output_format.value}, "
            f"not {output_format.value}"
        )
    reward = _validate_reward(raw_value)
    if is_zero_diagnostic:
        if not compatibility.allow_zero_reward_diagnostic:
            raise RewardParseError(
                f"Task {task_id} is not approved for a zero-reward diagnostic"
            )
        if reward != 0.0:
            raise RewardParseError("A reward diagnostic must report zero reward")
    return reward


def _validate_reward(raw_value: str) -> float:
    try:
        reward = float(raw_value)
    except ValueError as error:
        raise RewardParseError(f"Invalid reward value: {raw_value!r}") from error
    if not math.isfinite(reward):
        raise RewardParseError("Reward must be finite")
    if not 0.0 <= reward <= 1.0:
        raise RewardParseError("Reward must be in the inclusive range [0, 1]")
    return reward
