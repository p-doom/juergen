from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .executor import CompactRawExecutor, NativeAbsoluteExecutor, parse_compact_raw
from .fixtures import MANIFEST_PATH, load_manifest
from .transport import RecordingTransport, pointer_mask_for_buttons


SPEC_PATH = Path(__file__).with_name("curriculum_spec.json")
FORMATS = ("native_absolute_control", "compact_raw_phaseb")
SPLIT_NAMES = {"train": "train", "validation": "val"}
CANVAS = (1000, 700)

NATIVE_SYSTEM_PROMPT = (
    "You operate a desktop computer. The first user turn contains a goal and "
    "the current screenshot; later turns contain the current screenshot. Reply "
    "with exactly one computer_use tool call using absolute pixel coordinates. "
    "Available actions are mouse_move, left_click, mouse_down, mouse_up, "
    "left_click_drag, scroll, key, and type."
)
COMPACT_SYSTEM_PROMPT = (
    "You operate a desktop computer. Reply with exactly one action per turn as "
    "dx dy scroll, optionally followed by ordered events after ';'. dx and dy "
    "are relative screen pixels; scroll is signed wheel clicks. +LMB/-LMB press "
    "and release the left button. Keyboard events use rdev names. Exact Unicode "
    "text is type(\"...\"). A click is +LMB -LMB; a drag keeps press, move, and "
    "release on separate turns."
)


class CurriculumError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curriculum_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    seal = value.pop("spec_payload_sha256", None)
    if not isinstance(seal, str) or _sha(value) != seal:
        raise CurriculumError("curriculum spec seal mismatch")
    value["spec_payload_sha256"] = seal
    if value.get("schema_version") != 1:
        raise CurriculumError("unsupported curriculum schema")
    if value.get("status") != "preregistered_not_launched":
        raise CurriculumError("curriculum launch status drifted")
    if value.get("official_osworld_reuse") is not False:
        raise CurriculumError("official OSWorld reuse must remain false")
    if tuple(value.get("formats", ())) != FORMATS:
        raise CurriculumError("matched format matrix drifted")
    matrix = value.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        raise CurriculumError("curriculum matrix missing")
    for split in ("train", "validation"):
        declared = int(value["splits"][split]["records_per_format"])
        observed = sum(int(row[split]) for row in matrix)
        if declared != observed:
            raise CurriculumError(
                f"{split} count mismatch: declared={declared} matrix={observed}"
            )
    return value


def iter_cells(spec: dict[str, Any], split: str) -> Iterator[tuple[int, str]]:
    seed = int(spec["splits"][split]["seed_start"])
    for row in spec["matrix"]:
        for _ in range(int(row[split])):
            yield seed, str(row["capability"])
            seed += 1


def _skill_sequence(seed: int, capability: str) -> list[str]:
    if capability in {"click", "focus_type", "scroll", "drag"}:
        return [capability]
    count = int(capability.rsplit("_", 1)[1])
    skills = ["click", "focus_type", "scroll", "drag"]
    random.Random(seed ^ 0x51A7).shuffle(skills)
    return skills[:count]


def make_scene(seed: int, capability: str, split: str) -> dict[str, Any]:
    rng = random.Random(seed)
    skills = _skill_sequence(seed, capability)
    slots = [
        (470 + rng.randint(-45, 45), 165 + rng.randint(-30, 30)),
        (755 + rng.randint(-45, 45), 220 + rng.randint(-30, 30)),
        (485 + rng.randint(-45, 45), 415 + rng.randint(-30, 30)),
        (750 + rng.randint(-45, 45), 520 + rng.randint(-30, 30)),
    ]
    rng.shuffle(slots)
    controls: dict[str, Any] = {}
    for skill, (x, y) in zip(skills, slots[: len(skills)], strict=True):
        if skill == "click":
            controls[skill] = {"center": [x, y], "label": f"Select {seed % 997:03d}"}
        elif skill == "focus_type":
            controls[skill] = {
                "center": [x, y],
                "label": f"Field {seed % 991:03d}",
                "initial_text": f"draft-{seed % 89}",
                "target_text": f"Zürich μ-{seed}",
            }
        elif skill == "scroll":
            clicks = rng.choice((-8, -6, -4, 4, 6, 8))
            controls[skill] = {
                "center": [x, y],
                "label": f"Panel {seed % 983:03d}",
                "clicks": clicks,
            }
        elif skill == "drag":
            direction = rng.choice(("left", "right"))
            start_x = x - 110 if direction == "right" else x + 110
            end_x = x + 110 if direction == "right" else x - 110
            controls[skill] = {
                "center": [x, y],
                "label": f"Level {seed % 977:03d}",
                "start": [start_x, y],
                "end": [end_x, y],
                "initial_value": 20 if direction == "right" else 80,
                "target_value": 100 if direction == "right" else 0,
            }
    cursor = [rng.randint(80, 920), rng.randint(100, 640)]
    instructions = []
    for skill in skills:
        control = controls[skill]
        if skill == "click":
            instructions.append(f"click {control['label']}")
        elif skill == "focus_type":
            instructions.append(
                f"replace {control['label']} with {control['target_text']!r}"
            )
        elif skill == "scroll":
            direction = "down" if control["clicks"] < 0 else "up"
            instructions.append(f"scroll {control['label']} {direction}")
        else:
            direction = "right" if control["target_value"] == 100 else "left"
            instructions.append(f"drag {control['label']} fully {direction}")
    return {
        "schema_version": 1,
        "seed": seed,
        "split": split,
        "capability": capability,
        "skills": skills,
        "instruction": "Complete in order: " + "; then ".join(instructions) + ".",
        "cursor": cursor,
        "controls": controls,
    }


def _native(arguments: dict[str, Any]) -> str:
    return (
        "<tool_call>\n"
        + json.dumps(
            {"name": "computer_use", "arguments": arguments},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n</tool_call>"
    )


@dataclass(frozen=True)
class PlannedTurn:
    state: dict[str, Any]
    action: str
    native_arguments: dict[str, Any] | None


@dataclass(frozen=True)
class PlannedTrajectory:
    turns: tuple[PlannedTurn, ...]
    final_state: dict[str, Any]


def plan_trajectory(scene: dict[str, Any], arm: str) -> PlannedTrajectory:
    if arm not in FORMATS:
        raise CurriculumError(f"unknown curriculum format: {arm}")
    state: dict[str, Any] = {
        "cursor": list(scene["cursor"]),
        "completed": [],
        "focused": False,
        "selected": False,
        "input_text": scene["controls"].get("focus_type", {}).get(
            "initial_text", ""
        ),
        "scroll_offset": 0,
        "slider_value": scene["controls"].get("drag", {}).get(
            "initial_value", 0
        ),
        "held_buttons": [],
    }
    turns: list[PlannedTurn] = []

    def emit(raw: str, arguments: dict[str, Any] | None = None) -> None:
        turns.append(PlannedTurn(copy.deepcopy(state), raw, arguments))

    def move_click(target: list[int]) -> None:
        if arm == "native_absolute_control":
            arguments = {"action": "left_click", "coordinate": target}
            emit(_native(arguments), arguments)
        else:
            dx, dy = target[0] - state["cursor"][0], target[1] - state["cursor"][1]
            emit(f"{dx} {dy} 0 ; +LMB -LMB")
        state["cursor"] = list(target)

    for skill in scene["skills"]:
        control = scene["controls"][skill]
        if skill == "click":
            move_click(control["center"])
            state["completed"].append(skill)
        elif skill == "focus_type":
            move_click(control["center"])
            state["focused"] = True
            if arm == "native_absolute_control":
                arguments = {"action": "key", "keys": ["CTRL", "A"]}
                emit(_native(arguments), arguments)
            else:
                emit("0 0 0 ; +ControlLeft +KeyA -KeyA -ControlLeft")
            state["selected"] = True
            text = str(control["target_text"])
            if arm == "native_absolute_control":
                arguments = {"action": "type", "text": text}
                emit(_native(arguments), arguments)
            else:
                emit("0 0 0 ; type(" + json.dumps(text, ensure_ascii=False) + ")")
            state["input_text"] = text
            state["selected"] = False
            state["completed"].append(skill)
        elif skill == "scroll":
            clicks = int(control["clicks"])
            if arm == "native_absolute_control":
                arguments = {"action": "scroll", "clicks": clicks}
                emit(_native(arguments), arguments)
            else:
                emit(f"0 0 {clicks}")
            state["scroll_offset"] += clicks
            state["completed"].append(skill)
        elif skill == "drag":
            start, end = control["start"], control["end"]
            if arm == "native_absolute_control":
                arguments = {"action": "mouse_move", "coordinate": start}
                emit(_native(arguments), arguments)
                state["cursor"] = list(start)
                arguments = {"action": "left_click_drag", "coordinate": end}
                emit(_native(arguments), arguments)
                state["cursor"] = list(end)
                state["slider_value"] = int(control["target_value"])
            else:
                dx, dy = start[0] - state["cursor"][0], start[1] - state["cursor"][1]
                emit(f"{dx} {dy} 0 ; +LMB")
                state["cursor"] = list(start)
                state["held_buttons"] = ["left"]
                dx, dy = end[0] - start[0], end[1] - start[1]
                emit(f"{dx} {dy} 0")
                state["cursor"] = list(end)
                state["slider_value"] = int(control["target_value"])
                emit("0 0 0 ; -LMB")
                state["held_buttons"] = []
            state["completed"].append(skill)
        else:  # pragma: no cover - spec validation owns this boundary
            raise CurriculumError(f"unsupported capability {skill}")
    return PlannedTrajectory(tuple(turns), copy.deepcopy(state))


def oracle_accepts(scene: dict[str, Any], state: dict[str, Any]) -> bool:
    if state.get("held_buttons"):
        return False
    if state.get("completed") != scene.get("skills"):
        return False
    controls = scene["controls"]
    if "focus_type" in controls and state.get("input_text") != controls["focus_type"]["target_text"]:
        return False
    if "scroll" in controls:
        expected = int(controls["scroll"]["clicks"])
        if int(state.get("scroll_offset", 0)) * expected <= 0:
            return False
    if "drag" in controls and int(state.get("slider_value", -1)) != int(
        controls["drag"]["target_value"]
    ):
        return False
    return True


def validate_trajectory(scene: dict[str, Any], arm: str) -> dict[str, Any]:
    trajectory = plan_trajectory(scene, arm)
    transport = RecordingTransport(cursor=tuple(scene["cursor"]), screen=CANVAS)
    executor: NativeAbsoluteExecutor | CompactRawExecutor
    executor = (
        NativeAbsoluteExecutor(transport)
        if arm == "native_absolute_control"
        else CompactRawExecutor(transport)
    )
    round_trips = 0
    for turn in trajectory.turns:
        if arm == "native_absolute_control":
            if turn.native_arguments is None:
                raise CurriculumError("native turn lacks canonical arguments")
            from eval.action_parser import parse_computer_use_tool_call

            parsed = parse_computer_use_tool_call(turn.action)
            if parsed.arguments != turn.native_arguments:
                raise CurriculumError("native action failed exact round trip")
            executor.execute(parsed.arguments)
        else:
            parse_compact_raw(turn.action)
            executor.execute(turn.action)
        round_trips += 1
    final = trajectory.final_state
    if not oracle_accepts(scene, final):
        raise CurriculumError("positive curriculum oracle rejected planned final state")
    near_miss = copy.deepcopy(final)
    near_miss["completed"] = near_miss["completed"][:-1]
    if oracle_accepts(scene, near_miss):
        raise CurriculumError("near-miss curriculum oracle accepted")
    final_mask = pointer_mask_for_buttons(transport.audit.held_buttons)
    if final_mask != 0:
        raise CurriculumError(f"trajectory left pointer mask {final_mask}")
    if transport.cursor_position() != tuple(final["cursor"]):
        raise CurriculumError("executor cursor disagrees with planned cursor")
    expected_text = scene["controls"].get("focus_type", {}).get("target_text")
    if expected_text is not None and transport.audit.typed_texts != [expected_text]:
        raise CurriculumError("Unicode typing round trip failed")
    return {
        "turn_count": len(trajectory.turns),
        "round_trip_count": round_trips,
        "final_pointer_mask": final_mask,
        "oracle_positive": True,
        "near_miss_rejected": True,
    }


def _font(size: int):
    from PIL import ImageFont

    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render_state(scene: dict[str, Any], state: dict[str, Any], path: Path) -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", CANVAS, "#eef1f5")
    draw = ImageDraw.Draw(image)
    title, body, small = _font(22), _font(17), _font(14)
    draw.rectangle((0, 0, CANVAS[0], 62), fill="#26364a")
    draw.text((24, 18), "Capability Workshop", fill="white", font=title)
    draw.rounded_rectangle((24, 84, 292, 650), 14, fill="white", outline="#c7ced8", width=2)
    draw.text((44, 108), "Ordered goal", fill="#26364a", font=body)
    y = 148
    for index, skill in enumerate(scene["skills"], 1):
        done = skill in state["completed"]
        draw.text(
            (44, y),
            f"{'✓' if done else '○'} {index}. {skill.replace('_', ' ')}",
            fill="#198754" if done else "#445268",
            font=small,
        )
        y += 34
    draw.text((44, 610), "synthetic practice", fill="#7c8798", font=small)
    for skill in scene["skills"]:
        control = scene["controls"][skill]
        x, y = control.get("center", control.get("start"))
        done = skill in state["completed"]
        if skill == "click":
            draw.rounded_rectangle((x - 90, y - 34, x + 90, y + 34), 10, fill="#d9e7ff", outline="#315f9d", width=3)
            draw.text((x - 68, y - 10), control["label"], fill="#17345f", font=small)
            if done:
                draw.text((x + 60, y - 10), "✓", fill="#198754", font=body)
        elif skill == "focus_type":
            outline = "#1d6fe8" if state["focused"] else "#7e8b9e"
            draw.text((x - 115, y - 48), control["label"], fill="#334155", font=small)
            draw.rounded_rectangle((x - 125, y - 26, x + 125, y + 26), 6, fill="white", outline=outline, width=3)
            if state["selected"]:
                draw.rectangle((x - 111, y - 15, x + 75, y + 15), fill="#b9d7ff")
            draw.text((x - 110, y - 10), state["input_text"], fill="#1f2937", font=small)
        elif skill == "scroll":
            draw.rounded_rectangle((x - 115, y - 75, x + 115, y + 75), 8, fill="#fcfcfd", outline="#8793a3", width=2)
            draw.text((x - 95, y - 57), control["label"], fill="#334155", font=small)
            offset = int(state["scroll_offset"])
            for row in range(4):
                yy = y - 24 + row * 24 + max(-12, min(12, offset))
                draw.line((x - 92, yy, x + 72, yy), fill="#c7ced8", width=5)
            thumb_y = y - 32 - max(-25, min(25, offset * 3))
            draw.rounded_rectangle((x + 91, thumb_y, x + 101, thumb_y + 42), 5, fill="#62748a")
        elif skill == "drag":
            start_x, sy = control["start"]
            end_x, _ = control["end"]
            left, right = sorted((start_x, end_x))
            draw.text((left, sy - 48), control["label"], fill="#334155", font=small)
            draw.line((left, sy, right, sy), fill="#8997aa", width=8)
            fraction = int(state["slider_value"]) / 100.0
            knob_x = int(round(left + (right - left) * fraction))
            draw.ellipse((knob_x - 15, sy - 15, knob_x + 15, sy + 15), fill="#315fbd", outline="white", width=3)
    cx, cy = state["cursor"]
    draw.polygon(((cx, cy), (cx + 4, cy + 20), (cx + 10, cy + 13), (cx + 18, cy + 17)), fill="black", outline="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return _file_sha(path)


def _messages(
    scene: dict[str, Any], arm: str, turns: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    system = NATIVE_SYSTEM_PROMPT if arm == "native_absolute_control" else COMPACT_SYSTEM_PROMPT
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system}]}
    ]
    for index, (image, action) in enumerate(turns):
        content: list[dict[str, Any]] = []
        if index == 0:
            content.append({"type": "text", "text": scene["instruction"]})
        content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": content})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": action}]}
        )
    return messages


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def validate_seed_and_parameter_isolation(
    spec: dict[str, Any], scenes: list[dict[str, Any]]
) -> dict[str, int]:
    manifest = load_manifest(MANIFEST_PATH)
    sealed_seeds = {fixture.parameter_seed for fixture in manifest.fixtures}
    curriculum_seeds = {int(scene["seed"]) for scene in scenes}
    seed_overlap = sealed_seeds & curriculum_seeds
    if seed_overlap:
        raise CurriculumError(f"sealed fixture seed overlap: {sorted(seed_overlap)}")
    fixture_parameters = {_sha(fixture.params) for fixture in manifest.fixtures}
    scene_parameters = {_sha(scene["controls"]) for scene in scenes}
    parameter_overlap = fixture_parameters & scene_parameters
    if parameter_overlap:
        raise CurriculumError("sealed fixture parameter fingerprint overlap")
    train = {_sha(scene["controls"]) for scene in scenes if scene["split"] == "train"}
    validation = {
        _sha(scene["controls"])
        for scene in scenes
        if scene["split"] == "validation"
    }
    if train & validation:
        raise CurriculumError("train/validation scene fingerprint overlap")
    return {
        "sealed_fixture_seed_overlap": 0,
        "sealed_fixture_parameter_fingerprint_overlap": 0,
        "train_validation_scene_fingerprint_overlap": 0,
    }


def build_curriculum(
    output: Path, *, spec_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec = load_curriculum_spec() if spec_override is None else spec_override
    if output.exists() and any(output.iterdir()):
        raise CurriculumError(f"refusing to overwrite non-empty output: {output}")
    stage = output.with_name(f".{output.name}.building-{os.getpid()}-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    scenes = [
        make_scene(seed, capability, split)
        for split in ("train", "validation")
        for seed, capability in iter_cells(spec, split)
    ]
    isolation = validate_seed_and_parameter_isolation(spec, scenes)
    reports: dict[str, dict[str, int]] = {
        arm: {"records": 0, "turns": 0, "round_trips": 0}
        for arm in FORMATS
    }
    initial_hashes: dict[tuple[str, int], str] = {}
    rows_by_cell: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    try:
        for arm in FORMATS:
            for split in ("train", "validation"):
                rows: list[dict[str, Any]] = []
                for scene in (item for item in scenes if item["split"] == split):
                    validated = validate_trajectory(scene, arm)
                    trajectory = plan_trajectory(scene, arm)
                    turns: list[tuple[str, str]] = []
                    first_hash = ""
                    for index, turn in enumerate(trajectory.turns):
                        relative_image = (
                            Path("images")
                            / arm
                            / SPLIT_NAMES[split]
                            / f"scene-{scene['seed']}-turn-{index:02d}.png"
                        )
                        image_path = stage / relative_image
                        image_hash = render_state(scene, turn.state, image_path)
                        if index == 0:
                            first_hash = image_hash
                        turns.append((str((output / relative_image).resolve()), turn.action))
                    key = (split, int(scene["seed"]))
                    previous = initial_hashes.setdefault(key, first_hash)
                    if previous != first_hash:
                        raise CurriculumError(f"format twin initial pixels differ: {key}")
                    row = {
                        "sample_id": f"r1c-{arm}-{split}-{scene['seed']}",
                        "recording_id": f"r1c-{split}-{scene['seed']}",
                        "scene_id": f"r1c-{split}-{scene['seed']}",
                        "kind": scene["capability"],
                        "format": arm,
                        "seed": scene["seed"],
                        "scene_fingerprint": _sha(scene["controls"]),
                        "instruction_sha256": _sha(scene["instruction"]),
                        "initial_image_sha256": first_hash,
                        "final_oracle": "accepted",
                        "messages": _messages(scene, arm, turns),
                    }
                    rows.append(row)
                    rows_by_cell.setdefault(key, {})[arm] = row
                    reports[arm]["records"] += 1
                    reports[arm]["turns"] += int(validated["turn_count"])
                    reports[arm]["round_trips"] += int(validated["round_trip_count"])
                _write_jsonl(
                    stage / arm / "_normalized" / SPLIT_NAMES[split] / "chat.jsonl",
                    rows,
                )
        for key, twins in rows_by_cell.items():
            if set(twins) != set(FORMATS):
                raise CurriculumError(f"missing matched format twin: {key}")
            native, compact = (twins[arm] for arm in FORMATS)
            for field in (
                "recording_id",
                "scene_id",
                "kind",
                "seed",
                "scene_fingerprint",
                "instruction_sha256",
                "initial_image_sha256",
                "final_oracle",
            ):
                if native[field] != compact[field]:
                    raise CurriculumError(f"format twin mismatch {key} field={field}")
        report = {
            "status": "pass",
            "schema_version": 1,
            "curriculum_spec_sha256": _file_sha(SPEC_PATH),
            "sealed_fixture_manifest_sha256": load_manifest().manifest_payload_sha256,
            "fixture_manifest_file_sha256": _file_sha(MANIFEST_PATH),
            "isolation": isolation,
            "format_twin_identity": {
                "passing": len(rows_by_cell),
                "total": len(rows_by_cell),
            },
            "positive_oracle": {"passing": 2 * len(scenes), "total": 2 * len(scenes)},
            "near_miss_oracle_rejection": {
                "passing": 2 * len(scenes),
                "total": 2 * len(scenes),
            },
            "final_pointer_mask_zero": {
                "passing": 2 * len(scenes),
                "total": 2 * len(scenes),
            },
            "formats": reports,
        }
        (stage / "invariant_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "build_manifest.json").write_text(
            json.dumps(
                {
                    "artifact_type": "rung1_synthetic_capability_curriculum",
                    "schema_version": 1,
                    "status": "complete",
                    "invariant_report": "invariant_report.json",
                    "records_per_format": {
                        split: int(spec["splits"][split]["records_per_format"])
                        for split in ("train", "validation")
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output.mkdir(parents=True, exist_ok=True)
        children = sorted(
            (child for child in stage.iterdir() if child.name != "build_manifest.json"),
            key=lambda child: child.name,
        )
        children.append(stage / "build_manifest.json")
        for child in children:
            os.replace(child, output / child.name)
        stage.rmdir()
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_curriculum(args.output)
    except CurriculumError as exc:
        print(f"FATAL curriculum invariant: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
