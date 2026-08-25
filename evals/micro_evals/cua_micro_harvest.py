"""Turn oracle-driven micro-eval trajectories into trainable ``chat.jsonl``.

``cua_micro_eval.py --mode harvest`` runs the ordinary multiturn loop with
``cua_micro_oracle`` in place of the model (see ``run_multiturn_attempt``'s
``action_source``). This module supplies the two halves that are specific to
harvesting: the action source itself, and the conversion of a finished
trajectory into one training record.

The output row matches the canonical schema built by
``pipeline/crowdcast/stage_04_build_conversations.py``
(``build_messages``): instruction text before the image on the first user turn,
image-only on later turns, one assistant turn per frame carrying its action, and
every content field a list of typed blocks. That is what stage-05/stage-06 and
the payload-free inline-record path already read, so harvested data drops
straight into the existing training pipeline.

Note the two shape differences from the eval's own ``build_loggable_messages``,
both handled here: the eval uses plain strings for system/assistant content
where stage-04 uses ``[{"type": "text", ...}]`` blocks, and the eval writes
``<image step_000.png>`` placeholders where training needs the real path.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.micro_evals import cua_micro_oracle
from evals.micro_evals.cua_micro_oracle import OracleEnv, OracleError, OracleRuntime

# Legal terminal label for the frame that follows the last action. NOT
# "TERMINATE": that token belongs to the computer_use_rel_step_v1 tool-call
# format and is explicitly not part of the ordered_events_v3 grammar (see
# action_parser's note), so parse_ordered_action would reject it and the eval
# would score it as a parse error rather than a stop. NO_OP is in the grammar
# and means "do nothing / wait", which is the right thing to teach after the
# goal is already reached.
_TERMINAL_ACTION = "NO_OP"


def make_oracle_action_source(
    *,
    plan: tuple[dict[str, Any], ...],
    active_title: Callable[[], str],
    model_resolution: tuple[int, int] | None,
    seed: int,
) -> Callable[..., tuple[str, str | None]]:
    """An ``action_source`` for ``run_multiturn_attempt`` backed by the oracle.

    ``seed`` drives the per-trajectory jitter (approach step count and residual
    offsets), so the same (task, attempt) reproduces exactly while different
    attempts of one task produce genuinely different approach shapes.
    """
    runtime = OracleRuntime(
        env=OracleEnv(active_title=active_title),
        screen=(0, 0),
        model_resolution=model_resolution,
        rng=random.Random(seed),
    )
    state: dict[str, Any] = {"gen": None}

    def action_source(
        *,
        turn_index: int,
        cursor: tuple[int, int],
        bbox: tuple[int, int, int, int],
        screen: tuple[int, int],
    ) -> tuple[str, str | None]:
        runtime.turn_index = turn_index
        runtime.cursor = cursor
        runtime.bbox = bbox
        runtime.screen = screen
        if state["gen"] is None:
            state["gen"] = cua_micro_oracle.run_plan(runtime, list(plan))
        try:
            line = next(state["gen"])
        except StopIteration:
            # The loop only pulls again when the verifier has NOT passed, so an
            # exhausted plan means the script genuinely failed. Fail loudly and
            # free the VM instead of burning the remaining turn budget on
            # NO_OPs that cannot rescue it.
            raise OracleError(
                f"oracle plan exhausted after {turn_index} turn(s) without passing the verifier"
            ) from None
        # "stop" mirrors _call_model's finish_reason for a complete reply --
        # anything else (notably "length") makes the loop refuse to dispatch.
        return line, "stop"

    return action_source


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(image: str) -> dict[str, Any]:
    return {"type": "image", "image": image}


def build_chat_record(
    *,
    result: dict[str, Any],
    attempt_dir: Path,
    suite_name: str,
    system_prompt: str,
    system_prompt_id: str,
    plan: tuple[dict[str, Any], ...],
    terminal_action: str | None = _TERMINAL_ACTION,
) -> dict[str, Any] | None:
    """One finished trajectory -> one ``chat.jsonl`` row, or None if unusable.

    Returns None for any trajectory whose verifier never fired, or whose frames
    are missing -- an unverified oracle run is a bug in the plan, not a training
    example, and silently keeping it would poison the set with confidently
    wrong labels.

    ``terminal_action`` labels the final (post-success) frame; pass None to drop
    that frame instead, leaving the trajectory ending on its last real action.
    """
    if not result.get("success"):
        return None
    turns = result.get("turns") or []
    if not turns:
        return None
    steps_dir = attempt_dir / "steps"

    frames: list[Path] = []
    actions: list[str] = []
    for index, turn in enumerate(turns):
        # The before-frame of turn i is what the model saw when it chose turn
        # i's action -- NOT step_{i}_after.png, which is the post-dispatch debug
        # capture taken on a different settle schedule.
        frame = steps_dir / f"step_{index:03d}.png"
        if not frame.exists():
            return None
        frames.append(frame)
        actions.append(str(turn.get("response", "")))
    if terminal_action is not None:
        # run_multiturn_attempt saves the terminal state under the index just
        # past the last turn (see its save_frames block after the loop).
        terminal = steps_dir / f"step_{len(turns):03d}.png"
        if not terminal.exists():
            return None
        frames.append(terminal)
        actions.append(terminal_action)

    instruction = str(result.get("instruction") or "")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [_text_block(system_prompt)]}
    ]
    for index, (frame, action) in enumerate(zip(frames, actions, strict=True)):
        content: list[dict[str, Any]] = []
        if index == 0 and instruction:
            content.append(_text_block(instruction))
        content.append(_image_block(str(frame.resolve())))
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [_text_block(action)]})

    task_id = str(result.get("task_id"))
    return {
        "conversation_id": f"{suite_name}.{task_id}.seed{result.get('seed')}",
        "task_id": task_id,
        "suite": suite_name,
        "category": result.get("category"),
        "instruction": instruction,
        "goal_conditioned": bool(instruction),
        "source": "cua_micro_oracle",
        "action_format": result.get("action_format"),
        "system_prompt_id": system_prompt_id,
        "model_resolution": result.get("model_resolution"),
        "screen_size": result.get("screen_size"),
        "seed": result.get("seed"),
        "oracle_plan": [str(op["op"]) for op in plan],
        "verifier_passed": True,
        "verifier_turn": len(turns),
        "n_frames": len(frames),
        "n_turns": len(frames),
        "n_non_noop": sum(1 for action in actions if action.strip() != "NO_OP"),
        "messages": messages,
    }


def write_chat_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def harvest_summary(
    records: list[dict[str, Any]],
    *,
    attempted: int,
    suite_name: str,
    task_ids: list[str],
) -> dict[str, Any]:
    per_task: dict[str, int] = dict.fromkeys(task_ids, 0)
    for record in records:
        per_task[record["task_id"]] = per_task.get(record["task_id"], 0) + 1
    return {
        "schema_version": 1,
        "mode": "harvest",
        "suite": suite_name,
        "n_attempted": attempted,
        "n_kept": len(records),
        "n_dropped": attempted - len(records),
        "n_frames": sum(record["n_frames"] for record in records),
        "n_tasks_with_data": len(per_task),
        "per_task_kept": dict(sorted(per_task.items())),
        # A task at 0 is a broken plan, not bad luck: the oracle is
        # deterministic modulo approach jitter, so it either works or doesn't.
        "tasks_with_no_data": sorted(
            task for task, count in per_task.items() if count == 0
        ),
    }
