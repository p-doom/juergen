"""Pure rejection-SFT conversion for synthetic relative-mouse rollouts.

Only genuinely successful, fully parseable on-policy trajectories are retained.
No corrective action, terminate action, reward shaping, or failed prefix is
synthesized.  The sparse verifier is independently replayed from the recorded
cursor and normalized deltas before any record is emitted.
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from stage5_rft.util import ContractError, atomic_write_json, atomic_write_jsonl, sha256_bytes


SCREEN_W = 1920
SCREEN_H = 1080
SYSTEM_PROMPT = """You operate a desktop computer using the computer_use tool. The first user turn shows the initial screen and the user's goal; each subsequent user turn shows the current screen. Reply with one or more computer_use tool calls that advance toward the goal.

Mouse movement is RELATIVE and NORMALIZED. To move the cursor, emit a `move_rel` action whose `coordinate` is a [dx, dy] offset from the CURRENT cursor position, expressed in thousandths of the screen (each axis in [-999, 999]; dx = 1000 spans the full width, dy = 1000 the full height; positive dx = right, positive dy = down). `move_rel` moves the cursor by that relative delta (pyautogui.moveRel); it is NOT an absolute screen coordinate. Look at the visible cursor in the screenshot to judge how far and in which direction to move. To click a target, FIRST `move_rel` by the relative offset, THEN issue a click with NO coordinate (the click lands at the current cursor position).

Actions (computer_use `action` field):
- move_rel {coordinate:[dx,dy]}: move the cursor by the relative normalized offset (dx,dy).
- left_click / right_click / middle_click: click at the CURRENT cursor position (no coordinate); move first with move_rel.
- double_click / triple_click: double / triple click at the current position.
- mouse_down {button} / mouse_up {button}: press / release a mouse button (button = 'left','right','middle'). A drag is move_rel, mouse_down, one or more move_rel, then mouse_up.
- key {keys:[...]}: press a key or chord, e.g. ['ctrl','a'], ['enter'], ['tab'].
- key_down {keys:[...]} / key_up {keys:[...]}: hold / release keys across steps.
- type {text}: type a string of text.
- scroll {pixels}: scroll the wheel (positive = up, negative = down).
- wait {time}: do nothing this step.
- terminate {status}: the goal is complete (status = 'success' or 'failure').

For each action, return a JSON object within <tool_call></tool_call> tags. To move the cursor 12 right / 8 up (normalized) and left-click there:
<tool_call>
{"name": "computer_use", "arguments": {"action": "move_rel", "coordinate": [12, -8]}}
</tool_call>
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click"}}
</tool_call>"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _in_box(cursor: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    return box[0] <= cursor[0] < box[2] and box[1] <= cursor[1] < box[3]


def _apply_delta(cursor: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    dx = round(delta[0] / 1000.0 * SCREEN_W)
    dy = round(delta[1] / 1000.0 * SCREEN_H)
    return (
        max(0, min(SCREEN_W - 1, cursor[0] + dx)),
        max(0, min(SCREEN_H - 1, cursor[1] + dy)),
    )


def _parse_exact_move(raw: Any, *, allow_zero: bool = False) -> tuple[int, int]:
    if not isinstance(raw, str):
        raise ContractError("assistant output is not a string")
    stripped = raw.strip()
    opening, closing = "<tool_call>", "</tool_call>"
    if not stripped.startswith(opening) or not stripped.endswith(closing):
        raise ContractError("assistant output is not one exact tool_call")
    payload_text = stripped[len(opening) : -len(closing)].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"assistant tool_call JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("name") != "computer_use":
        raise ContractError("assistant tool_call name is not computer_use")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or arguments.get("action") != "move_rel":
        raise ContractError("assistant action is not move_rel")
    coordinate = arguments.get("coordinate")
    if (
        not isinstance(coordinate, list)
        or len(coordinate) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in coordinate)
    ):
        raise ContractError("move_rel coordinate must be exactly two integers")
    delta = (coordinate[0], coordinate[1])
    if any(value < -999 or value > 999 for value in delta):
        raise ContractError("move_rel coordinate lies outside [-999,999]")
    if delta == (0, 0) and not allow_zero:
        raise ContractError("zero-delta action is not learner-eligible")
    return delta


def _normalized(cursor: tuple[int, int]) -> tuple[int, int]:
    return (
        max(0, min(999, round(cursor[0] / SCREEN_W * 1000))),
        max(0, min(999, round(cursor[1] / SCREEN_H * 1000))),
    )


def _user_text(cursor: tuple[int, int]) -> str:
    nx, ny = _normalized(cursor)
    return (
        "Move the mouse cursor INTO the green highlighted box.\n"
        f"Screen resolution: {SCREEN_W}x{SCREEN_H}. Coordinates are NORMALIZED "
        "0-999, NOT pixels.\n"
        "The cursor is at the red crosshair marker, normalized position "
        f"({nx}, {ny}). Emit a `move_rel` action whose [dx, dy] moves the cursor "
        "toward the green box (dx>0 = right, dy>0 = down). When the cursor is "
        "inside the box, you are done."
    )


def _render(
    *, background: Path, box: tuple[int, int, int, int], cursor: tuple[int, int]
) -> bytes:
    with Image.open(background) as source:
        image = source.convert("RGB")
    if image.size != (SCREEN_W, SCREEN_H):
        image = image.resize((SCREEN_W, SCREEN_H))
    draw = ImageDraw.Draw(image)
    draw.rectangle([box[0], box[1], box[2], box[3]], outline=(0, 255, 0), width=5)
    x, y = cursor
    radius = 10
    draw.line([(x - radius, y), (x + radius, y)], fill=(255, 0, 0), width=3)
    draw.line([(x, y - radius), (x, y + radius)], fill=(255, 0, 0), width=3)
    draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 0, 0), width=3)
    import io

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _task_split(task_id: str, *, salt: str, val_fraction: float) -> str:
    digest = hashlib.sha256(f"{salt}:{task_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if value < val_fraction else "train"


def _validate_task(raw: Mapping[str, Any], approved_background_root: Path) -> dict[str, Any]:
    if raw.get("kind") != "train":
        raise ContractError("relative-mouse task kind is not train")
    try:
        task_id = str(int(raw["idx"]))
        box = tuple(int(v) for v in raw["box"])
        cursor = tuple(int(v) for v in raw["cursor_start"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid relative-mouse task geometry: {exc}") from exc
    if len(box) != 4 or len(cursor) != 2:
        raise ContractError("invalid box or cursor arity")
    if not (0 <= box[0] < box[2] < SCREEN_W and 0 <= box[1] < box[3] < SCREEN_H):
        raise ContractError("box lies outside screen or is empty")
    if not (0 <= cursor[0] < SCREEN_W and 0 <= cursor[1] < SCREEN_H):
        raise ContractError("cursor lies outside screen")
    background = Path(str(raw.get("background_path", ""))).resolve()
    approved = approved_background_root.resolve()
    try:
        background.relative_to(approved)
    except ValueError as exc:
        raise ContractError("background is outside the approved train-only root") from exc
    if not background.is_file():
        raise ContractError(f"background is missing: {background}")
    return {"task_id": task_id, "box": box, "cursor": cursor, "background": background}


def build_pure_relative_mouse_records(
    *,
    rollout_glob: str,
    output_dir: str | Path,
    approved_background_root: str | Path,
    val_fraction: float = 0.05,
    split_salt: str = "stage5-relative-mouse-pure-v1",
    maximum_trajectories_per_task: int = 4,
) -> dict[str, Any]:
    """Build exact-action single-turn records from complete successful trajectories."""

    if not 0 < val_fraction < 1:
        raise ContractError("val_fraction must be in (0,1)")
    if maximum_trajectories_per_task < 1:
        raise ContractError("maximum_trajectories_per_task must be positive")
    files = [Path(path) for path in sorted(glob.glob(rollout_glob))]
    if not files:
        raise ContractError(f"no rollout files matched {rollout_glob!r}")

    approved = Path(approved_background_root)
    if not approved.is_dir():
        raise ContractError(f"approved background root is missing: {approved}")
    out = Path(output_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    rejection_reasons: Counter[str] = Counter()
    kept_per_task: Counter[str] = Counter()
    counts = Counter()
    source_files = []

    for path in files:
        source_files.append(
            {"path": str(path.resolve()), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
        )
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                counts["rollouts_seen"] += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ContractError("rollout is not an object")
                    task = _validate_task(row.get("task", {}), approved)
                    accepted = row.get("accepted")
                    reward = row.get("reward")
                    if not isinstance(accepted, bool) or isinstance(reward, bool) or not isinstance(
                        reward, (int, float)
                    ):
                        raise ContractError("accepted/reward fields have invalid types")
                    if float(reward) != (1.0 if accepted else 0.0):
                        raise ContractError("sparse reward disagrees with accepted flag")
                    if not accepted:
                        rejection_reasons["task_not_successful"] += 1
                        continue
                    counts["accepted_by_sampler"] += 1
                    trajectory = row.get("traj")
                    if not isinstance(trajectory, list) or not trajectory:
                        raise ContractError("accepted rollout has no trajectory")
                    if int(row.get("steps", -1)) != len(trajectory):
                        raise ContractError("trajectory length disagrees with steps")
                    if kept_per_task[task["task_id"]] >= maximum_trajectories_per_task:
                        rejection_reasons["per_task_cap"] += 1
                        continue

                    cursor = task["cursor"]
                    step_rows: list[tuple[tuple[int, int], str]] = []
                    reached = False
                    preserved_no_ops = 0
                    for step_index, step in enumerate(trajectory):
                        if not isinstance(step, Mapping):
                            raise ContractError("trajectory step is not an object")
                        recorded_cursor = tuple(int(v) for v in step.get("cursor", []))
                        if recorded_cursor != cursor:
                            raise ContractError("recorded cursor disagrees with replay state")
                        assistant = step.get("assistant")
                        if not isinstance(assistant, str):
                            raise ContractError("assistant output is not a string")
                        raw_delta = step.get("delta")
                        if raw_delta is None:
                            # Terminate, wait, click, or unparsed output: preserve the
                            # exact on-policy target and replay it as the sampler did.
                            preserved_no_ops += 1
                        else:
                            recorded_delta = tuple(int(v) for v in raw_delta)
                            if len(recorded_delta) != 2:
                                raise ContractError("recorded delta has invalid arity")
                            parsed_delta = _parse_exact_move(
                                assistant, allow_zero=recorded_delta == (0, 0)
                            )
                            if recorded_delta != parsed_delta:
                                raise ContractError(
                                    "parsed assistant delta disagrees with recorded delta"
                                )
                            if recorded_delta == (0, 0):
                                preserved_no_ops += 1
                            else:
                                cursor = _apply_delta(cursor, recorded_delta)
                        step_rows.append((cursor if raw_delta is None else recorded_cursor, assistant))
                        if _in_box(cursor, task["box"]):
                            if step_index != len(trajectory) - 1:
                                raise ContractError("accepted trajectory contains steps after success")
                            reached = True
                    if not reached:
                        raise ContractError("independent sparse verifier did not accept trajectory")

                    split = _task_split(task["task_id"], salt=split_salt, val_fraction=val_fraction)
                    trajectory_id = f"movebox-{task['task_id']}-{kept_per_task[task['task_id']]}"
                    for step_index, (cursor_before, assistant) in enumerate(step_rows):
                        png = _render(
                            background=task["background"], box=task["box"], cursor=cursor_before
                        )
                        image_sha = sha256_bytes(png)
                        image_path = image_dir / f"{image_sha}.png"
                        if not image_path.exists():
                            image_path.write_bytes(png)
                        records[split].append(
                            {
                                "schema_version": "stage5.relative_mouse_sft.v1",
                                "recording_id": trajectory_id,
                                "task_id": task["task_id"],
                                "trajectory_step": step_index,
                                "source": "pure_on_policy_success",
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                                    },
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": str(image_path.resolve())},
                                            {"type": "text", "text": _user_text(cursor_before)},
                                        ],
                                    },
                                    {
                                        "role": "assistant",
                                        "content": [{"type": "text", "text": assistant}],
                                    },
                                ],
                            }
                        )
                    kept_per_task[task["task_id"]] += 1
                    counts["trajectories_kept"] += 1
                    counts["no_op_actions_preserved"] += preserved_no_ops
                except (ContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    rejection_reasons[f"malformed_or_unverifiable:{type(exc).__name__}"] += 1
                    counts["malformed_or_unverifiable"] += 1
                    if counts["malformed_or_unverifiable"] <= 10:
                        rejection_reasons[f"example:{path.name}:{line_number}:{str(exc)[:120]}"] += 0

    if counts["malformed_or_unverifiable"]:
        raise ContractError(
            f"{counts['malformed_or_unverifiable']} rollout rows were malformed or unverifiable; "
            "pure conversion fails closed"
        )
    if not records["train"] or not records["val"]:
        raise ContractError("pure conversion produced an empty train or validation split")
    train_tasks = {row["task_id"] for row in records["train"]}
    val_tasks = {row["task_id"] for row in records["val"]}
    if train_tasks & val_tasks:
        raise ContractError("task-level train/validation split leaked")

    normalized = out / "_normalized"
    atomic_write_jsonl(normalized / "train" / "chat.jsonl", records["train"])
    atomic_write_jsonl(normalized / "val" / "chat.jsonl", records["val"])
    manifest = {
        "schema_version": "stage5.relative_mouse_dataset.v1",
        "status": "complete",
        "method": "pure_rejection_sft",
        "source_files": source_files,
        "approved_background_root": str(approved.resolve()),
        "contains_official_heldout": False,
        "contains_real_vm_eval": False,
        "contains_crowd_cast": False,
        "synthetic_actions_added": 0,
        "synthetic_terminate_added": False,
        "no_op_actions_dropped": 0,
        "no_op_actions_preserved": counts["no_op_actions_preserved"],
        "reward_verifier": "independent normalized-delta replay then cursor-in-box",
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
        "rollouts_seen": counts["rollouts_seen"],
        "accepted_by_sampler": counts["accepted_by_sampler"],
        "trajectories_kept": counts["trajectories_kept"],
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "train_records": len(records["train"]),
        "val_records": len(records["val"]),
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "maximum_trajectories_per_task": maximum_trajectories_per_task,
        "val_fraction": val_fraction,
        "split_salt": split_salt,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_write_json(out / "manifest.json", manifest)
    return manifest
