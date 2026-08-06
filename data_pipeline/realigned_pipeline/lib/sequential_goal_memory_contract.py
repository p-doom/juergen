"""Shared sequential goal-memory prompt/checkpoint contract."""

from __future__ import annotations

from pathlib import Path

METHOD = "sequential_goal_memory"
RECIPE = "sequential_goal_memory_v1"
ACTION_SPEC = "computer_use_rel_norm_v1"
CHECKPOINT_SCHEMA_VERSION = 1
SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "system_prompts" / "sequential_goal_memory_v1.txt"
)
CHECKPOINT_FIELDS = (
    "Long-term goal", "Mid-term objective", "Short-term objective",
    "Completed", "Current state", "Next step", "Critical details",
)
CHECKPOINT_CONTROL_REQUEST = (
    "CHECKPOINT CONTROL: summarize only what is causally known through the current "
    "screenshot. Reply with exactly one checkpoint block in the system-specified format."
)
PROACTIVE_GOAL_TEXT = (
    "Continue the user's work on this computer. Infer what they are doing from "
    "the screen and prior context, and advance it."
)
THOUGHT_MAX_WORDS = 60
CHECKPOINT_MAX_WORDS = 180   # total across the seven field bodies
RESUME_UPWEIGHT_TURNS = 3    # first assistant action turns after a checkpoint_in


def system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def goal_conditioning(goal: str, checkpoint: str | None = None) -> str:
    text = f"GOAL: {goal.strip()}"
    if checkpoint:
        text += "\n\n" + checkpoint.strip()
    return text


def render_checkpoint(values: dict[str, object]) -> str:
    lines = ["<checkpoint>"]
    for field in CHECKPOINT_FIELDS:
        value = " ".join(str(values.get(field) or "").split()) or "None."
        lines.extend([f"## {field}", value, ""])
    lines[-1] = "</checkpoint>"
    return "\n".join(lines)
