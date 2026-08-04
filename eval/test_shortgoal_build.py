"""Builder tests: synthetic recordings in, keep-text training records out.

No VM and no golden policy run — a hand-built ``recording.json`` in
``shortgoal_record``'s exact schema plus tiny PNGs per task exercise all of
``shortgoal_build``: the finevision message shape, the registered prompt bytes,
GOAL pinning, the closing ``TERMINATE``, the eviction cuts that decide record
boundaries and placeholder positions, the live-image budget, the blank-image
control, the cross-arm masked identity and the strict-parse gate every emitted
line must survive.

``write_recording`` / ``synthetic_steps`` are public because
``test_shortgoal_contract.py`` builds its episode from the same fixture: one
synthetic-episode definition feeds both the shape tests here and the
serialization-identity contract there. The fixture plays the recorder's role
exactly — grid primitives rendered as an abs line, strictly re-parsed, then
denormalized to the pixels the VM would have been sent — so the rows it writes
are the rows ``record_task`` writes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shortgoal_build as build  # noqa: E402
import shortgoal_grammar as grammar  # noqa: E402
import shortgoal_templates as templates  # noqa: E402
from action_parser import (  # noqa: E402
    OrderedAction,
    OrderedPrimitive,
    parse_ordered_v4_action,
)
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402
from shortgoal_grammar import (  # noqa: E402
    ARM_ABS,
    ARM_REL,
    ARMS,
    IMAGE_PLACEHOLDER,
    K_IMAGES,
    NO_OP_LINE,
    PROMPT_IDS,
    TERMINATE_LINE,
)

TASK_ID = templates.OVERFIT1_TASK_ID
SCREEN = templates.SCREEN_WH
RESOLUTION = (64, 36)
SPLITS = build.load_splits("")[0]
SOURCE_FRAME_WH = (160, 90)
START_CURSOR_GRID = (60, 40)


def grid_to_px(grid_xy: tuple[int, int]) -> tuple[int, int]:
    """The pixel of a 0-1000 grid point, through the shipped denormalizer only."""
    action = grammar.denorm_v4(
        OrderedAction(
            primitives=(OrderedPrimitive(kind="move_to", x=grid_xy[0], y=grid_xy[1]),),
            no_op=False,
        ),
        SCREEN,
    )
    return action.primitives[0].x, action.primitives[0].y


def _grid_action(prims: list[OrderedPrimitive]) -> OrderedAction:
    if not prims:
        return OrderedAction(primitives=(), no_op=True)
    line = grammar.render_line(prims, ARM_ABS)
    action = parse_ordered_v4_action(line, arm=ARM_ABS)
    if grammar.render_line(action.primitives, ARM_ABS) != line:
        raise AssertionError(f"fixture line does not round-trip: {line!r}")
    return action


def step_row(
    index: int, grid_prims: list[OrderedPrimitive], cursor_px: tuple[int, int],
) -> tuple[dict[str, Any], tuple[int, int]]:
    """One recorder step row for a turn, plus the pixel the turn leaves the cursor on."""
    action = _grid_action(grid_prims)
    pixels = grammar.denorm_v4(action, SCREEN).primitives
    after = cursor_px
    for prim in pixels:
        if prim.kind == "move_to":
            after = (prim.x, prim.y)
    row = {
        "frame": f"step_{index:03d}.png",
        "cursor_before": list(cursor_px),
        "cursor_after": list(after),
        "primitives_grid": [asdict(prim) for prim in action.primitives],
        "primitives_px": [asdict(prim) for prim in pixels],
    }
    return row, after


def _move_to(grid_xy: tuple[int, int]) -> OrderedPrimitive:
    return OrderedPrimitive(kind="move_to", x=grid_xy[0], y=grid_xy[1])


def _grid_target(index: int) -> tuple[int, int]:
    return (100 + (index * 97) % 800, 100 + (index * 61) % 800)


def synthetic_steps(
    n_steps: int, *, start_cursor_grid: tuple[int, int] = START_CURSOR_GRID,
) -> list[dict]:
    """A deterministic golden trajectory whose turns cover every primitive kind."""
    cursor = grid_to_px(start_cursor_grid)
    steps: list[dict] = []
    for index in range(n_steps):
        kind = index % 6
        if kind == 0:
            prims = [
                _move_to(_grid_target(index)),
                OrderedPrimitive(kind="down", name="LMB"),
                OrderedPrimitive(kind="up", name="LMB"),
            ]
        elif kind == 1:
            prims = [
                OrderedPrimitive(kind="type", text=f'echo "step{index}" > out_{index}.txt'),
                OrderedPrimitive(kind="down", name="Return"),
                OrderedPrimitive(kind="up", name="Return"),
            ]
        elif kind == 2:
            prims = [OrderedPrimitive(kind="scroll", dx=0, dy=-3 if index % 4 else 4)]
        elif kind == 3:
            start = _grid_target(index)
            prims = [
                _move_to(start),
                OrderedPrimitive(kind="down", name="LMB"),
                _move_to((start[0] + 60, start[1] + 40)),
                OrderedPrimitive(kind="up", name="LMB"),
            ]
        elif kind == 4:
            prims = []
        else:
            prims = [
                OrderedPrimitive(kind="down", name="ControlLeft"),
                OrderedPrimitive(kind="down", name="KeyS"),
                OrderedPrimitive(kind="up", name="KeyS"),
                OrderedPrimitive(kind="up", name="ControlLeft"),
            ]
        row, cursor = step_row(index, prims, cursor)
        steps.append(row)
    return steps


def _frame_image(index: int) -> Image.Image:
    width, height = SOURCE_FRAME_WH
    image = Image.new("RGB", SOURCE_FRAME_WH, ((index * 29) % 256, (index * 71) % 256, 40))
    box = Image.new("RGB", (width // 4, height // 4), (255, 255, 255))
    image.paste(box, ((index * 7) % (width - width // 4), (index * 5) % (height - height // 4)))
    return image


def write_recording(
    recordings_root: Path | str,
    task_id: str = TASK_ID,
    *,
    n_steps: int = 1,
    steps: list[dict] | None = None,
    verified: bool = True,
    instruction: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write one synthetic ``<root>/<task_id>/{recording.json, frames/*.png}``.

    The goal line and the seeded params are the catalog's own — the builder checks
    both against ``shortgoal_templates`` — so only the trajectory is synthetic."""
    template_id, _, seed = task_id.partition("__s")
    task = templates.concrete_task(template_id, int(seed))
    directory = Path(recordings_root) / task_id
    (directory / build.FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    steps = synthetic_steps(n_steps) if steps is None else steps
    for index in range(len(steps) + 1):
        _frame_image(index).save(directory / build.FRAMES_DIR / f"step_{index:03d}.png")
    data: dict[str, Any] = {
        "schema_version": build.RECORDING_SCHEMA_VERSION,
        "task_id": task_id,
        "template_id": template_id,
        "seed": int(seed),
        "category": templates.TEMPLATES_BY_ID[template_id].category,
        "instruction": instruction or task.instruction,
        "params": task.params,
        "screen_size": list(SCREEN),
        "cursor_start": list(grid_to_px(START_CURSOR_GRID)),
        "steps": steps,
        "n_steps": len(steps),
        "n_frames": len(steps) + 1,
        "verifier": {
            "kind": templates.TEMPLATES_BY_ID[template_id].verifier_id,
            "passed": verified,
            "detail": {},
        },
    }
    data.update(overrides or {})
    (directory / build.RECORDING_NAME).write_text(json.dumps(data, indent=2))
    return directory


def build_records(
    recordings_root: Path | str,
    output_root: Path | str,
    arm: str = ARM_REL,
    *,
    subset: str = "overfit1",
    resolution: tuple[int, int] = RESOLUTION,
    blank_images: bool = False,
    quality: int = build.DEFAULT_JPEG_QUALITY,
    allow_resupervision: bool = True,
) -> tuple[list[dict], dict]:
    """Build one arm and return ``(records, manifest)`` read back off disk.

    ``allow_resupervision`` is on by default because most cases here are the 7-9
    frame shape the builder only emits on request; ``EvictionCutTests`` is what pins
    the refusal."""
    manifest = build.build_arm(
        recordings_root=recordings_root,
        output_root=output_root,
        arm=arm,
        subset=subset,
        splits=SPLITS,
        resolution=resolution,
        quality=quality,
        blank_images=blank_images,
        allow_resupervision=allow_resupervision,
    )
    return build.read_records(Path(output_root) / build.CHAT_RELPATH), manifest


def _assistant(record: dict) -> list[str]:
    return [m["content"] for m in record["messages"] if m["role"] == "assistant"]


def _user(record: dict) -> list[list[dict]]:
    return [m["content"] for m in record["messages"] if m["role"] == "user"]


def _neutralized(record: dict, root: Path) -> str:
    return json.dumps(record).replace(str(Path(root).resolve()), "ROOT")


def _placeholder_indices(record: dict) -> list[int]:
    return [
        index for index, content in enumerate(_user(record))
        if content[-1] == {"type": "text", "text": IMAGE_PLACEHOLDER}
    ]


class _Built(unittest.TestCase):
    """Shared setUp: one synthetic 9-frame episode built for one arm."""

    n_steps = 8
    arm = ARM_REL

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.recordings = self.root / "recordings"
        self.output = self.root / "out"
        write_recording(self.recordings, n_steps=self.n_steps)
        self.records, self.manifest = build_records(self.recordings, self.output, self.arm)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class MessageShapeTests(_Built):
    def test_block_vocabulary_matches_prep_finevision(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
        import prep_finevision  # noqa: PLC0415

        reference = prep_finevision._row_to_messages(
            {"texts": [{"user": "u0", "assistant": "a0"}, {"user": "u1", "assistant": "a1"}]},
            ["/abs/path/img_00.jpg"],
        )
        schema = {
            block["type"]: frozenset(block)
            for message in reference
            for block in (message["content"] if isinstance(message["content"], list) else [])
        }
        self.assertEqual(
            schema,
            {"image": frozenset({"type", "url"}), "text": frozenset({"type", "text"})},
        )
        self.assertTrue(all(
            isinstance(message["content"], str)
            for message in reference if message["role"] == "assistant"
        ))
        for record in self.records:
            for message in record["messages"]:
                content = message["content"]
                if isinstance(content, str):
                    self.assertIn(message["role"], ("system", "assistant"))
                    continue
                for block in content:
                    self.assertEqual(frozenset(block), schema[block["type"]])

    def test_roles_alternate_user_then_assistant(self) -> None:
        for record in self.records:
            roles = [m["role"] for m in record["messages"]]
            self.assertEqual(roles[0], "system")
            self.assertEqual(roles[1:], ["user", "assistant"] * record["n_frames"])
            self.assertEqual(len(_user(record)), record["n_frames"])

    def test_image_urls_are_absolute_existing_jpegs_at_model_resolution(self) -> None:
        seen: set[str] = set()
        for record in self.records:
            for content in _user(record):
                for block in content:
                    if block["type"] != "image":
                        continue
                    path = Path(block["url"])
                    self.assertTrue(path.is_absolute())
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.suffix, ".jpg")
                    self.assertEqual(path.parent.name, TASK_ID)
                    self.assertEqual(
                        path.parent.parent, (self.output / build.IMAGES_RELPATH).resolve(),
                    )
                    with Image.open(path) as image:
                        self.assertEqual(image.size, RESOLUTION)
                    seen.add(path.name)
        self.assertEqual(sorted(seen), [f"step_{i:03d}.jpg" for i in range(self.n_steps + 1)])

    def test_system_turn_is_the_registered_prompt_bytes(self) -> None:
        prompt_file = (
            Path(__file__).resolve().parents[1]
            / "data_pipeline/realigned_pipeline/system_prompts"
            / f"{PROMPT_IDS[self.arm]}.txt"
        )
        expected = prompt_file.read_text().strip()
        self.assertEqual(SYSTEM_PROMPTS[PROMPT_IDS[self.arm]], expected)
        for record in self.records:
            self.assertEqual(record["messages"][0], {"role": "system", "content": expected})

    def test_goal_rides_only_the_first_user_turn_of_every_record(self) -> None:
        goal = build.GOAL_PREFIX + self.records[0]["instruction"]
        for record in self.records:
            turns = _user(record)
            self.assertEqual(turns[0][0], {"type": "text", "text": goal})
            self.assertEqual(len(turns[0]), 2)
            for content in turns[1:]:
                self.assertEqual(len(content), 1)
                self.assertNotIn(goal, json.dumps(content))

    def test_last_assistant_turn_is_terminate_after_the_final_frame(self) -> None:
        last = self.records[-1]
        self.assertEqual(last["messages"][-1], {"role": "assistant", "content": TERMINATE_LINE})
        self.assertEqual(last["n_frames"], self.n_steps + 1)
        self.assertEqual(_assistant(last).count(TERMINATE_LINE), 1)
        for record in self.records[:-1]:
            self.assertNotIn(TERMINATE_LINE, _assistant(record))

    def test_top_level_record_keys(self) -> None:
        for index, record in enumerate(self.records):
            self.assertEqual(record["conversation_id"], f"{TASK_ID}__r{index:02d}")
            self.assertEqual(record["task_id"], TASK_ID)
            self.assertEqual(record["template_id"], TASK_ID.split("__s")[0])
            self.assertEqual(record["seed"], 0)
            self.assertEqual(record["split"], "train")
            self.assertEqual(record["arm"], self.arm)
            self.assertEqual(record["action_format"], self.arm)
            self.assertEqual(record["recipe"], build.RECIPE)
            self.assertEqual(record["record_index"], index)
            self.assertEqual(record["n_records_in_episode"], len(self.records))
            self.assertEqual(record["n_frames"], len(_user(record)))
            self.assertEqual(
                record["n_live_images"], record["n_frames"] - len(_placeholder_indices(record)),
            )
            self.assertLessEqual(record["n_live_images"], K_IMAGES)

    def test_assistant_turns_are_the_validated_episode_lines(self) -> None:
        rec = build.load_recording(self.recordings, TASK_ID)
        lines = build.episode_lines(rec, self.arm)
        self.assertEqual(len(lines), self.n_steps + 1)
        for record in self.records:
            self.assertEqual(_assistant(record), list(lines[: record["n_frames"]]))
        self.assertIn(NO_OP_LINE, lines)


class EvictionCutTests(unittest.TestCase):
    def _build(self, n_steps: int, arm: str = ARM_REL) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recording(root / "recordings", n_steps=n_steps)
            records, _ = build_records(root / "recordings", root / "out", arm)
            return records

    def test_record_counts_and_placeholders_per_episode_length(self) -> None:
        expected = {
            1: [(2, 2, [])],
            5: [(6, 6, [])],
            6: [(6, 6, []), (7, 3, [0, 1, 2, 3])],
            8: [(6, 6, []), (9, 5, [0, 1, 2, 3])],
        }
        for n_steps, rows in expected.items():
            with self.subTest(n_steps=n_steps):
                records = self._build(n_steps)
                self.assertEqual(len(records), len(rows))
                for record, (n_frames, n_live, placeholders) in zip(records, rows, strict=True):
                    self.assertEqual(record["n_frames"], n_frames)
                    self.assertEqual(record["n_live_images"], n_live)
                    self.assertEqual(_placeholder_indices(record), placeholders)
                    self.assertEqual(record["n_records_in_episode"], len(rows))

    def test_cuts_land_exactly_on_the_runtime_eviction_points(self) -> None:
        for n_steps in range(1, 12):
            with self.subTest(n_steps=n_steps):
                points = build.keep_text_eviction_points(n_steps + 1)
                records = self._build(n_steps)
                self.assertEqual(len(records), len(points) + 1)
                self.assertEqual([r["n_frames"] for r in records[:-1]], points)

    def test_live_images_never_exceed_k(self) -> None:
        for n_steps in range(1, 12):
            with self.subTest(n_steps=n_steps):
                for record in self._build(n_steps):
                    self.assertLessEqual(record["n_live_images"], K_IMAGES)
                    self.assertGreaterEqual(record["n_live_images"], 1)

    def test_a_multi_record_episode_is_refused_without_the_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recording(root / "recordings", n_steps=8)
            with self.assertRaises(ValueError):
                build_records(
                    root / "recordings", root / "refused", allow_resupervision=False,
                )
            records, manifest = build_records(
                root / "recordings", root / "opted_in", allow_resupervision=True,
            )
            self.assertEqual(len(records), 2)
            self.assertTrue(manifest["allow_resupervision"])
            self.assertEqual(
                [record["first_supervised_turn"] for record in records], [0, K_IMAGES],
            )
            self.assertEqual(manifest["counts"]["n_resupervised_turns"], K_IMAGES)

    def test_a_single_record_episode_needs_no_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recording(root / "recordings", n_steps=5)
            records, manifest = build_records(
                root / "recordings", root / "out", allow_resupervision=False,
            )
            self.assertEqual(len(records), 1)
            self.assertFalse(manifest["allow_resupervision"])
            self.assertEqual(records[0]["first_supervised_turn"], 0)
            self.assertEqual(manifest["counts"]["n_resupervised_turns"], 0)

    def test_second_record_repeats_the_whole_text_history(self) -> None:
        first, second = self._build(8)
        self.assertEqual(_assistant(second)[: len(_assistant(first))], _assistant(first))
        self.assertEqual(len(_assistant(second)), 9)
        self.assertEqual(
            [content[-1]["type"] for content in _user(second)],
            ["text"] * 4 + ["image"] * 5,
        )


class LineRenderTests(unittest.TestCase):
    def _drag_step(self) -> dict:
        row, _ = step_row(
            0,
            [
                _move_to((500, 500)),
                OrderedPrimitive(kind="down", name="LMB"),
                _move_to((750, 750)),
                OrderedPrimitive(kind="up", name="LMB"),
            ],
            grid_to_px((100, 100)),
        )
        return row

    def test_rel_deltas_come_from_the_recorded_cursor(self) -> None:
        step = self._drag_step()
        self.assertEqual(step["cursor_before"], [192, 108])
        self.assertEqual(step["cursor_after"], [1440, 810])
        self.assertEqual(
            build.step_line(step, ARM_REL, screen_size=SCREEN),
            "move(400,400); down(LMB); move(250,250); up(LMB)",
        )
        self.assertEqual(
            build.step_line(step, ARM_ABS, screen_size=SCREEN),
            "move_to(500,500); down(LMB); move_to(750,750); up(LMB)",
        )

    def test_arm_invariant_kinds_render_identically(self) -> None:
        steps = synthetic_steps(6)
        for index, step in enumerate(steps):
            rel = build.step_line(step, ARM_REL, screen_size=SCREEN)
            abs_ = build.step_line(step, ARM_ABS, screen_size=SCREEN)
            with self.subTest(index=index):
                self.assertEqual(build.masked_line(rel, ARM_REL), build.masked_line(abs_, ARM_ABS))
                if index % 6 in (0, 3):
                    self.assertNotEqual(rel, abs_)
                else:
                    self.assertEqual(rel, abs_)
        self.assertEqual(build.step_line(steps[4], ARM_REL, screen_size=SCREEN), NO_OP_LINE)
        self.assertEqual(build.masked_line(NO_OP_LINE, ARM_ABS), NO_OP_LINE)
        self.assertEqual(build.masked_line(TERMINATE_LINE, ARM_REL), TERMINATE_LINE)

    def test_masked_line_keeps_type_payloads_verbatim(self) -> None:
        row, _ = step_row(
            0,
            [
                _move_to((10, 20)),
                OrderedPrimitive(kind="type", text='move(1,2) move_to(3,4) \\ "quoted"'),
            ],
            grid_to_px((100, 100)),
        )
        line = build.step_line(row, ARM_REL, screen_size=SCREEN)
        self.assertEqual(
            build.masked_line(line, ARM_REL),
            f'{build.MOVE_MASK}; type("move(1,2) move_to(3,4) \\\\ \\"quoted\\"")',
        )
        self.assertEqual(build.validate_line(line, ARM_REL), line)

    def test_rejects_degenerate_and_malformed_steps(self) -> None:
        cursor = grid_to_px((100, 100))
        base, _ = step_row(0, [_move_to((500, 500))], cursor)
        grid_row, px_row = base["primitives_grid"][0], base["primitives_px"][0]
        zero_delta, _ = step_row(0, [_move_to((100, 100))], cursor)
        scroll, _ = step_row(0, [OrderedPrimitive(kind="scroll", dx=0, dy=4)], cursor)
        cases = {
            "zero delta": zero_delta,
            "no cursor": {**base, "cursor_before": None},
            "cursor off screen": {**base, "cursor_before": [SCREEN[0], 0]},
            "tampered px": {**base, "primitives_px": [{**px_row, "x": 961}]},
            "missing px": {k: v for k, v in base.items() if k != "primitives_px"},
            "unknown field": {**base, "primitives_grid": [{**grid_row, "nonsense": 1}]},
            "out of grid": {**base, "primitives_grid": [{**grid_row, "y": 1001}]},
            "unknown kind": {**base, "primitives_grid": [{**grid_row, "kind": "fly"}]},
            "zero scroll": {
                **scroll,
                "primitives_grid": [{**scroll["primitives_grid"][0], "dy": 0}],
                "primitives_px": [{**scroll["primitives_px"][0], "dy": 0}],
            },
            "no primitives": {"cursor_before": list(cursor)},
        }
        for name, step in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    build.step_line(step, ARM_REL, screen_size=SCREEN)

    def test_strict_parse_rejects_corrupted_lines(self) -> None:
        for line in (
            "move(1,2); junk",
            "move_to(10,20)",
            "move(1,2)x",
            "move(0,0)",
            "scroll(1,2)",
            'type("unterminated',
            "",
        ):
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    build.validate_line(line, ARM_REL)
        with self.assertRaises(ValueError):
            build.validate_line("move(1,2)", ARM_ABS)


class ArmIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_recording(self.root / "recordings", n_steps=8)
        self.assertEqual(build.main([
            "--recordings_root", str(self.root / "recordings"),
            "--output_dir", str(self.root / "out"),
            "--arm", "both",
            "--subset", "overfit1",
            "--model_resolution", f"{RESOLUTION[0]}x{RESOLUTION[1]}",
            "--allow_resupervision",
            "--check_arms",
        ]), 0)
        self.chats = {
            arm: build.arm_root(self.root / "out", arm, per_arm=True) / build.CHAT_RELPATH
            for arm in ARMS
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_both_arms_build_into_per_arm_roots(self) -> None:
        for arm, chat in self.chats.items():
            self.assertTrue(chat.is_file())
            records = build.read_records(chat)
            self.assertEqual({r["arm"] for r in records}, {arm})
            self.assertEqual(len(records), 2)

    def test_arms_are_line_identical_after_masking_moves(self) -> None:
        stats = build.check_arms_identity(self.chats[ARM_REL], self.chats[ARM_ABS])
        self.assertEqual(stats["n_records"], 2)
        self.assertGreater(stats["n_masked_moves"], 0)
        self.assertEqual(
            stats["n_turns"],
            sum(len(r["messages"]) for r in build.read_records(self.chats[ARM_REL])),
        )

    def test_check_only_reruns_the_comparison_without_building(self) -> None:
        mtimes = {arm: chat.stat().st_mtime_ns for arm, chat in self.chats.items()}
        self.assertEqual(build.main([
            "--recordings_root", str(self.root / "recordings"),
            "--output_dir", str(self.root / "out"),
            "--arm", "both",
            "--check_only",
        ]), 0)
        self.assertEqual(
            mtimes, {arm: chat.stat().st_mtime_ns for arm, chat in self.chats.items()},
        )

    def test_a_corrupted_arm_line_breaks_the_check(self) -> None:
        records = build.read_records(self.chats[ARM_REL])
        for message in records[0]["messages"]:
            if message["role"] == "assistant":
                message["content"] = 'type("tampered")'
                break
        with self.chats[ARM_REL].open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.assertRaises(ValueError):
            build.check_arms_identity(self.chats[ARM_REL], self.chats[ARM_ABS])

    def test_an_unparseable_arm_line_breaks_the_check(self) -> None:
        records = build.read_records(self.chats[ARM_ABS])
        records[-1]["messages"][2]["content"] = "move_to(10,20) oops"
        with self.chats[ARM_ABS].open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.assertRaises(ValueError):
            build.check_arms_identity(self.chats[ARM_REL], self.chats[ARM_ABS])

    def test_check_arms_needs_both_arms(self) -> None:
        with self.assertRaises(ValueError):
            build.main([
                "--recordings_root", str(self.root / "recordings"),
                "--output_dir", str(self.root / "out2"),
                "--arm", ARM_REL,
                "--check_arms",
            ])


class BlankImageTests(unittest.TestCase):
    def test_blank_mode_is_structure_identical_with_gray_jpegs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recording(root / "recordings", n_steps=6)
            sighted, sighted_manifest = build_records(root / "recordings", root / "sighted")
            blanked, blank_manifest = build_records(
                root / "recordings", root / "blank", blank_images=True,
            )
            self.assertFalse(sighted_manifest["blank_images"])
            self.assertTrue(blank_manifest["blank_images"])
            self.assertEqual(len(sighted), len(blanked))
            for left, right in zip(sighted, blanked, strict=True):
                self.assertEqual(
                    _neutralized(left, root / "sighted"), _neutralized(right, root / "blank"),
                )
            urls = [
                block["url"]
                for record in blanked for content in _user(record) for block in content
                if block["type"] == "image"
            ]
            self.assertTrue(urls)
            for url in urls:
                with Image.open(url) as image:
                    self.assertEqual(image.size, RESOLUTION)
                    colors = image.convert("RGB").getcolors(maxcolors=1 << 16)
                    self.assertEqual(len(colors), 1)
                    self.assertEqual(colors[0][1], (build.BLANK_LEVEL,) * 3)
            for record in sighted:
                for content in _user(record):
                    for block in content:
                        if block["type"] == "image":
                            with Image.open(block["url"]) as image:
                                self.assertGreater(
                                    len(image.convert("RGB").getcolors(maxcolors=1 << 16)), 1,
                                )


class ManifestAndSubsetTests(unittest.TestCase):
    def test_manifest_counts_and_provenance_over_overfit32(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recordings = root / "recordings"
            for index, task in enumerate(templates.OVERFIT32_TASK_IDS):
                write_recording(recordings, task, n_steps=1 + index % 3)
            records, manifest = build_records(
                recordings, root / "out", ARM_ABS, subset="overfit32",
            )
            self.assertEqual(manifest["stage"], build.STAGE)
            self.assertEqual(manifest["arm"], ARM_ABS)
            self.assertEqual(manifest["subset"], "overfit32")
            self.assertEqual(manifest["prompt_id"], PROMPT_IDS[ARM_ABS])
            self.assertEqual(
                manifest["prompt_sha256"],
                hashlib.sha256(SYSTEM_PROMPTS[PROMPT_IDS[ARM_ABS]].encode()).hexdigest(),
            )
            self.assertEqual((manifest["k_images"], manifest["keep_images"]), (K_IMAGES, 3))
            self.assertEqual(manifest["model_resolution"], "64x36")
            self.assertEqual(manifest["splits_source"], build.RECOMPUTED_SPLITS)
            self.assertEqual(manifest["counts"]["n_tasks"], 32)
            self.assertEqual(manifest["counts"]["n_records"], len(records))
            self.assertEqual(
                manifest["counts"]["n_frames_written"],
                sum(record["n_frames"] for record in records),
            )
            self.assertEqual(manifest["counts"]["records_by_split"], {"train": len(records)})
            self.assertEqual(manifest["counts"]["tasks_by_split"], {"train": 32})
            self.assertEqual(
                set(manifest["counts"]["tasks_by_category"]), set(templates.CATEGORIES),
            )
            self.assertEqual(sum(manifest["counts"]["tasks_by_category"].values()), 32)
            self.assertLessEqual(manifest["counts"]["max_live_images"], K_IMAGES)
            self.assertEqual(
                json.loads((root / "out" / build.MANIFEST_NAME).read_text()), manifest,
            )

    def test_subsets_resolve_inside_their_split(self) -> None:
        self.assertEqual(build.subset_task_ids("overfit1", SPLITS), (TASK_ID,))
        self.assertEqual(len(build.subset_task_ids("overfit32", SPLITS)), 32)
        self.assertEqual(build.subset_task_ids("full", SPLITS), tuple(SPLITS["train"]))
        self.assertEqual(build.subset_task_ids("tiera_val", SPLITS), tuple(SPLITS["tier_a"]))
        self.assertEqual(build.subset_task_ids("tierb_val", SPLITS), tuple(SPLITS["tier_b"]))
        for subset in build.SUBSETS:
            with self.subTest(subset=subset):
                ids = build.subset_task_ids(subset, SPLITS)
                self.assertTrue(set(ids) <= set(SPLITS[build.SUBSET_SPLITS[subset]]))
        with self.assertRaises(ValueError):
            build.subset_task_ids("nope", SPLITS)

    def test_splits_file_and_resolution_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            path.write_text(json.dumps(templates.build_split_manifest()))
            splits, source = build.load_splits(str(path))
            self.assertEqual(splits["counts"], SPLITS["counts"])
            self.assertEqual(source, str(path.resolve()))
            broken = dict(templates.build_split_manifest())
            broken["tier_a"] = [*broken["tier_a"], broken["train"][0]]
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                build.load_splits(str(path))
        self.assertEqual(build.parse_resolution(" 1280x720 "), (1280, 720))
        for text in ("1280", "1280*720", "x720", "1280x"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    build.parse_resolution(text)

    def test_default_model_resolution_is_1280x720(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recording(root / "recordings", n_steps=1)
            records, manifest = build_records(
                root / "recordings", root / "out", resolution=build.DEFAULT_RESOLUTION,
            )
            self.assertEqual(manifest["model_resolution"], "1280x720")
            self.assertEqual(manifest["jpeg_quality"], 90)
            for content in _user(records[0]):
                for block in content:
                    if block["type"] == "image":
                        with Image.open(block["url"]) as image:
                            self.assertEqual(image.size, (1280, 720))
                            self.assertEqual(image.format, "JPEG")


class RecordingValidationTests(unittest.TestCase):
    def _root(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_unverified_recording_is_rejected(self) -> None:
        root = self._root()
        write_recording(root / "recordings", n_steps=2, verified=False)
        with self.assertRaises(ValueError):
            build.load_recording(root / "recordings", TASK_ID)

    def test_declared_counts_must_agree_with_the_steps(self) -> None:
        for override in ({"n_frames": 2}, {"n_steps": 5}, {"steps": []}):
            with self.subTest(override=override):
                root = self._root()
                write_recording(root / "recordings", n_steps=2, overrides=override)
                with self.assertRaises(ValueError):
                    build.load_recording(root / "recordings", TASK_ID)

    def test_catalog_and_schema_mismatches_are_rejected(self) -> None:
        for override in (
            {"schema_version": 2},
            {"task_id": "term_mkdir__s00"},
            {"template_id": "term_mkdir"},
            {"seed": 1},
            {"category": "browser"},
            {"instruction": "  "},
            {"screen_size": [0, 1080]},
            {"verifier": {"kind": "guest_path_exists"}},
        ):
            with self.subTest(override=override):
                root = self._root()
                write_recording(root / "recordings", n_steps=1, overrides=override)
                with self.assertRaises(ValueError):
                    build.load_recording(root / "recordings", TASK_ID)

    def test_a_goal_or_param_draw_that_drifted_from_the_catalog_is_rejected(self) -> None:
        task = templates.concrete_task(TASK_ID.partition("__s")[0], 0)
        for override in (
            {"instruction": task.instruction + " Then stop."},
            {"params": {}},
            {"params": {**task.params, "extra": 1}},
        ):
            with self.subTest(override=sorted(override)):
                root = self._root()
                write_recording(root / "recordings", n_steps=1, overrides=override)
                with self.assertRaises(ValueError):
                    build.load_recording(root / "recordings", TASK_ID)

    def test_step_frame_names_must_be_positional(self) -> None:
        root = self._root()
        steps = synthetic_steps(2)
        steps[1]["frame"] = "step_007.png"
        write_recording(root / "recordings", steps=steps)
        with self.assertRaises(ValueError):
            build.load_recording(root / "recordings", TASK_ID)

    def test_missing_recording_and_frames(self) -> None:
        root = self._root()
        with self.assertRaises(FileNotFoundError):
            build.load_recording(root / "recordings", TASK_ID)
        directory = write_recording(root / "recordings", n_steps=1)
        (directory / build.FRAMES_DIR / "step_001.png").unlink()
        with self.assertRaises(FileNotFoundError):
            build.load_recording(root / "recordings", TASK_ID)

    def test_a_widget_bbox_that_is_not_a_pixel_box_is_rejected(self) -> None:
        for widgets in ({}, {"Alpha": [10, 20, 10, 40]}, {"Alpha": [10, 20, 30]}):
            with self.subTest(widgets=widgets):
                root = self._root()
                write_recording(
                    root / "recordings", n_steps=1, overrides={"setup": {"widgets": widgets}},
                )
                with self.assertRaises(ValueError):
                    build.load_recording(root / "recordings", TASK_ID)

    def test_a_captured_thought_is_accepted_only_as_a_short_string(self) -> None:
        root = self._root() / "recordings"
        good = synthetic_steps(1)
        good[0]["thought"] = "x" * grammar.THOUGHT_MAX_CHARS
        write_recording(root, steps=good)
        self.assertEqual(build.load_recording(root, TASK_ID).n_frames, 2)
        for thought in ("x" * (grammar.THOUGHT_MAX_CHARS + 1), 5, ["a thought"], {}):
            with self.subTest(thought=thought):
                bad = synthetic_steps(1)
                bad[0]["thought"] = thought
                write_recording(root, steps=bad)
                with self.assertRaises(ValueError):
                    build.load_recording(root, TASK_ID)

    def test_every_built_line_survives_a_second_strict_parse(self) -> None:
        root = self._root()
        write_recording(root / "recordings", n_steps=8)
        for arm in ARMS:
            records, _ = build_records(root / "recordings", root / arm, arm)
            for record in records:
                for line in _assistant(record):
                    self.assertEqual(build.validate_line(line, arm), line)


class WidgetContainmentTests(unittest.TestCase):
    """The bbox gate, on the pixel each arm's OWN line presses — not the intended one."""

    def _click_recording(
        self,
        *,
        cursor_grid: tuple[int, int],
        target_grid: tuple[int, int],
        widgets: dict[str, list[int]],
        source: str | None = None,
    ) -> build.Recording:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        step, _ = step_row(
            0,
            [
                _move_to(target_grid),
                OrderedPrimitive(kind="down", name="LMB"),
                OrderedPrimitive(kind="up", name="LMB"),
            ],
            grid_to_px(cursor_grid),
        )
        root = Path(tmp.name) / "recordings"
        overrides: dict[str, Any] = {"setup": {"widgets": widgets}}
        if source is not None:
            overrides["source"] = source
            overrides["n_attempt"] = 2
        write_recording(root, steps=[step], overrides=overrides)
        return build.load_recording(root, TASK_ID)

    def test_a_press_inside_a_recorded_widget_passes_and_outside_fails(self) -> None:
        x, y = grid_to_px((500, 500))
        inside = self._click_recording(
            cursor_grid=(100, 100),
            target_grid=(500, 500),
            widgets={"Alpha": [x - 8, y - 8, x + 8, y + 8]},
        )
        outside = self._click_recording(
            cursor_grid=(100, 100),
            target_grid=(500, 500),
            widgets={"Alpha": [0, 0, 8, 8], "Beta": [x + 40, y + 40, x + 60, y + 60]},
        )
        for arm in ARMS:
            with self.subTest(arm=arm):
                self.assertEqual(len(build.episode_lines(inside, arm)), 2)
                with self.assertRaises(ValueError):
                    build.episode_lines(outside, arm)

    def test_an_agent_sourced_episode_keeps_the_parser_gate_and_drops_the_bbox_gate(self) -> None:
        x, y = grid_to_px((500, 500))
        widgets = {"Alpha": [0, 0, 8, 8], "Beta": [x + 40, y + 40, x + 60, y + 60]}
        oracle = self._click_recording(
            cursor_grid=(100, 100), target_grid=(500, 500), widgets=widgets,
        )
        agent = self._click_recording(
            cursor_grid=(100, 100), target_grid=(500, 500), widgets=widgets,
            source=build.SOURCE_AGENT,
        )
        self.assertEqual(oracle.source, build.SOURCE_ORACLE)
        self.assertEqual(agent.source, build.SOURCE_AGENT)
        for arm in ARMS:
            with self.subTest(arm=arm):
                with self.assertRaises(ValueError):
                    build.episode_lines(oracle, arm)
                lines = build.episode_lines(agent, arm)
                self.assertEqual(len(lines), 2)
                for line in lines:
                    self.assertEqual(build.validate_line(line, arm), line)

    def test_an_unknown_source_is_rejected_instead_of_skipping_the_bbox_gate(self) -> None:
        with self.assertRaises(ValueError):
            self._click_recording(
                cursor_grid=(100, 100),
                target_grid=(500, 500),
                widgets={"Alpha": [0, 0, 8, 8]},
                source="hand_written",
            )

    def test_the_rel_arms_reconstructed_pixel_is_what_gets_checked(self) -> None:
        target = grid_to_px((7, 7))
        rec = self._click_recording(
            cursor_grid=(1, 1),
            target_grid=(7, 7),
            widgets={"Alpha": [target[0], target[1], target[0] + 1, target[1] + 1]},
        )
        step = rec.steps[0]
        pressed = {
            arm: build.dispatched_clicks(
                build.step_line(step, arm, screen_size=SCREEN), arm, step, screen_size=SCREEN,
            )
            for arm in ARMS
        }
        self.assertEqual(pressed[ARM_ABS], (target,))
        self.assertNotEqual(pressed[ARM_REL], (target,))
        self.assertEqual(len(build.episode_lines(rec, ARM_ABS)), 2)
        with self.assertRaises(ValueError):
            build.episode_lines(rec, ARM_REL)


if __name__ == "__main__":
    unittest.main()
