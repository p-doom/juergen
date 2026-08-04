"""Per-arm ``chat.jsonl`` builder: golden recordings -> keep-text training records.

One recording is one golden episode of ``N`` frames and ``N-1`` semantic turns, so
its assistant turns are the ``N-1`` action lines plus the closing ``TERMINATE``
that follows the post-success frame. Records are assembled through the runtime's
OWN keep-text helpers (``osworld_runtime.KeepTextWindow``,
``keep_text_eviction_points``, ``keep_text_messages``) and cut at the runtime's
eviction points, so every assistant turn trains under the exact context the
closed-loop evaluator builds for it — ``test_shortgoal_contract.py`` is that
identity, turn by turn. Episodes of <=6 frames are one record; 7-9 frames are
two, the second opening with the pinned ``GOAL:`` turn, the whole text history,
the ``IMAGE_PLACEHOLDER`` literal on evicted turns and live images after — and
because the trainer's collator supervises EVERY assistant span, that second
record would also train its pre-cut turns a second time under blinded frames, so
a multi-record episode is refused unless ``--allow_resupervision`` asks for it.

``<recordings_root>/<task_id>/recording.json`` is exactly what
``shortgoal_record.record_task`` publishes — and it only publishes when the
verifier passed, which the builder re-checks. Read strictly (schema_version 1)::

    {"schema_version": 1,
     "task_id": "fx_click_button__s00",
     "template_id": "fx_click_button",
     "seed": 0,
     "category": "fixture",
     "instruction": "In the fixture window, click the ...",
     "params": {"label": "Alpha", "target_xy": [960, 540], ...},
     "screen_size": [1920, 1080],
     "setup": {"widgets": {"Alpha": [870, 505, 1050, 575]}, ...},
     "verifier": {"kind": "fixture_state", "passed": true, "detail": {...}},
     "n_steps": 2,
     "n_frames": 3,
     "steps": [{"frame": "step_000.png",
                "cursor_before": [192, 108],
                "cursor_after": [960, 540],
                "primitives_grid": [{"kind": "move_to", "x": 500, "y": 500, ...}, ...],
                "primitives_px": [{"kind": "move_to", "x": 960, "y": 540, ...}, ...]},
               ...]}

Frames live in ``<task_id>/frames/step_NNN.png``: one per step plus the final
post-success frame, so ``n_frames == n_steps + 1``. Step primitives are
``OrderedPrimitive`` field dicts; a whole-line ``NO_OP`` turn is the empty list.

Both arms are rendered from ``primitives_grid`` — the abs arm's ``move_to(x,y)``
is the grid point verbatim, and the rel arm's ``move(dx,dy)`` is
``norm_delta`` of the recorded ``cursor_before`` to the SAME grid point pushed
back through ``denorm_v4``, which is how the recorder derived (and validated) the
pixels it dispatched. Nothing is measured off the screenshots, the derived pixels
are asserted equal to the recorded ``primitives_px``, and every line is
strict-parsed and re-rendered byte-identically: a divergence is a hard error, not
a dropped sample. ``--check_arms`` then proves the two arms' records are
line-identical once the move token is masked.

Two further gates: the recording's ``instruction`` and seeded ``params`` must
still be the ones ``shortgoal_templates`` draws today (a template edited after the
recordings exist would otherwise train on a GOAL line the evaluator no longer
prompts with), and — for an ORACLE recording — every mouse press the arm's OWN
line dispatches must land inside one of the widget bboxes the fixture published
at setup, re-parsed and denormalized exactly as the closed loop would, so the rel
arm's reconstructed pixel is what gets checked, not the pixel it meant.

``source`` (absent means ``oracle``) is what that bbox gate keys on. An oracle
policy aims at a widget centre by construction, so a press outside every widget
is a bug in the geometry and has to fail the build. A ``sonnet_agent`` episode is
a real trajectory: it may miss, correct itself, click empty background to focus a
window, or land its winning click last — and it only exists at all because the
template's own verifier passed on the live VM, which is the stronger statement.
Agent recordings therefore keep every other gate (schema, catalog identity,
verifier-passed, grid/pixel agreement, strict parse -> byte-identical re-render
per arm, arms line-identical modulo the move token) and drop only the bbox check.

An agent driver may also capture a per-step first-person ``thought``. These rungs
are NO-THINK: a thought is validated here (a string of at most
``THOUGHT_MAX_CHARS``, so a malformed capture cannot reach a later render) and
then ignored — it never enters a message, so a chat built from recordings with
thoughts is byte-identical to one built from the same recordings without them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

import shortgoal_templates as templates
from action_parser import OrderedAction, OrderedPrimitive, parse_ordered_v4_action
from osworld_runtime import (
    KeepTextWindow,
    keep_text_eviction_points,
    keep_text_messages,
)
from osworld_system_prompts import SYSTEM_PROMPTS
from shortgoal_fixture import bbox_contains
from shortgoal_golden import MOUSE_NAMES
from shortgoal_grammar import (
    ARM_ABS,
    ARM_REL,
    ARMS,
    FRAME_JPEG_QUALITY,
    K_IMAGES,
    KEEP_IMAGES,
    NO_OP_LINE,
    PROMPT_IDS,
    TERMINATE_LINE,
    THOUGHT_MAX_CHARS,
    denorm_v4,
    norm_delta,
    render_line,
    render_primitive,
)

RECIPE = "shortgoal_oev4_v1"
STAGE = "shortgoal_stage_04_chat"
GOAL_PREFIX = "GOAL: "
CHAT_RELPATH = "train/chat.jsonl"
IMAGES_RELPATH = "train/images"
MANIFEST_NAME = "manifest.json"
RECORDING_NAME = "recording.json"
FRAMES_DIR = "frames"
RECORDING_SCHEMA_VERSION = 1
SOURCE_ORACLE = "oracle"
SOURCE_AGENT = "sonnet_agent"
SOURCES = (SOURCE_ORACLE, SOURCE_AGENT)

ARM_SLUGS = {ARM_REL: "oev4rel", ARM_ABS: "oev4abs"}
SUBSETS = ("overfit1", "overfit32", "full", "tiera_val", "tierb_val")
SUBSET_SPLITS = {
    "overfit1": "train",
    "overfit32": "train",
    "full": "train",
    "tiera_val": "tier_a",
    "tierb_val": "tier_b",
}

DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_JPEG_QUALITY = FRAME_JPEG_QUALITY
BLANK_LEVEL = 128
MOVE_MASK = "MOVE"
SYSTEM_MASK = "REGISTERED_SYSTEM_PROMPT"
FRAME_STEM = "step_{:03d}"
RECOMPUTED_SPLITS = "shortgoal_templates.build_split_manifest"

_PRIMITIVE_FIELDS = frozenset(OrderedPrimitive.__dataclass_fields__)
_RESOLUTION_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
_SENTINEL_FRAME = Image.new("RGB", (1, 1))
_SENTINEL_ACTION = "x"


def _require(mapping: dict[str, Any], key: str, kind: type, what: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{what} is missing {key!r}")
    value = mapping[key]
    if not isinstance(value, kind) or isinstance(value, bool) != (kind is bool):
        raise ValueError(f"{what}[{key!r}] must be {kind.__name__}, got {value!r}")
    return value


def _int_pair(
    value: Any, *, what: str, bounds: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(f"{what} must be a pair, got {value!r}")
    for coordinate, (lo, hi) in zip(value, bounds, strict=True):
        if (
            not isinstance(coordinate, int)
            or isinstance(coordinate, bool)
            or not lo <= coordinate <= hi
        ):
            raise ValueError(f"{what} must be ints inside {bounds!r}, got {value!r}")
    return int(value[0]), int(value[1])


def _pixel_pair(value: Any, screen_size: tuple[int, int], *, what: str) -> tuple[int, int]:
    return _int_pair(
        value, what=what, bounds=((0, screen_size[0] - 1), (0, screen_size[1] - 1)),
    )


@dataclass(frozen=True)
class Recording:
    """One validated golden episode: ``N`` frame paths and ``N-1`` recorded turns."""

    task_id: str
    template_id: str
    seed: int
    category: str
    instruction: str
    screen_size: tuple[int, int]
    frames: tuple[Path, ...]
    steps: tuple[dict[str, Any], ...]
    widgets: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()
    source: str = SOURCE_ORACLE

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def check_catalog_task(data: dict[str, Any], *, what: str) -> templates.ConcreteTask:
    """The catalog task a recording claims to be, with its goal and params re-drawn.

    Same guard as ``shortgoal_record.replay_recording``: an instruction string or a
    seeded param draw edited after the recordings exist must not silently split the
    training GOAL line from the one the closed loop prompts with."""
    task = templates.concrete_task(
        _require(data, "template_id", str, what), _require(data, "seed", int, what),
    )
    instruction = data.get("instruction")
    if instruction != task.instruction:
        raise ValueError(
            f"{what} instruction drifted from the catalog: {instruction!r} != "
            f"{task.instruction!r}"
        )
    params = json.dumps(data.get("params"), sort_keys=True)
    if params != json.dumps(task.params, sort_keys=True):
        raise ValueError(f"{what} seeded param draw drifted from the catalog: {params}")
    return task


def _widget_boxes(
    setup: Any, *, what: str,
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """The widget bboxes the setup published, as ``(label, bbox)`` pairs."""
    boxes = setup.get("widgets") if isinstance(setup, dict) else None
    if boxes is None:
        return ()
    if not isinstance(boxes, dict) or not boxes:
        raise ValueError(f"{what} setup.widgets must be a nonempty object, got {boxes!r}")
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for label in sorted(boxes):
        box = boxes[label]
        if not (isinstance(box, (list, tuple)) and len(box) == 4) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in box
        ):
            raise ValueError(f"{what} widget {label!r} is not a pixel bbox: {box!r}")
        x0, y0, x1, y1 = (int(value) for value in box)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{what} widget {label!r} is an empty bbox: {box!r}")
        out.append((str(label), (x0, y0, x1, y1)))
    return tuple(out)


def load_recording(recordings_root: Path | str, task_id: str) -> Recording:
    """Read and fully validate ``<recordings_root>/<task_id>/recording.json``."""
    directory = Path(recordings_root) / task_id
    path = directory / RECORDING_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no recording for {task_id}: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object, got {type(data)!r}")
    what = f"{task_id} recording"
    if _require(data, "schema_version", int, what) != RECORDING_SCHEMA_VERSION:
        raise ValueError(
            f"{path} is schema_version {data['schema_version']}, "
            f"this builder reads {RECORDING_SCHEMA_VERSION}"
        )
    if _require(data, "task_id", str, what) != task_id:
        raise ValueError(f"{path} declares task_id {data['task_id']!r}, not {task_id!r}")
    verifier = _require(data, "verifier", dict, what)
    if not _require(verifier, "passed", bool, f"{what} verifier"):
        raise ValueError(f"{task_id} failed verifier {verifier.get('kind')!r}; recording rejected")
    template_id = _require(data, "template_id", str, what)
    seed = _require(data, "seed", int, what)
    if templates.task_id(template_id, seed) != task_id:
        raise ValueError(f"{task_id} is not {template_id!r} at seed {seed}")
    category = _require(data, "category", str, what)
    expected = templates.TEMPLATES_BY_ID[template_id].category
    if category != expected:
        raise ValueError(f"{task_id} claims category {category!r}, catalog says {expected!r}")
    instruction = _require(data, "instruction", str, what)
    if not instruction.strip():
        raise ValueError(f"{task_id} has a blank instruction")
    check_catalog_task(data, what=what)
    source = data.get("source", SOURCE_ORACLE)
    if source not in SOURCES:
        raise ValueError(f"{task_id} recording claims source {source!r}, expected one of {SOURCES}")
    widgets = _widget_boxes(data.get("setup"), what=what)
    screen_size = _int_pair(
        data.get("screen_size"),
        what=f"{what} screen_size",
        bounds=((1, 1 << 16), (1, 1 << 16)),
    )
    n_steps = _require(data, "n_steps", int, what)
    steps = _require(data, "steps", list, what)
    if not steps or n_steps != len(steps):
        raise ValueError(f"{task_id} declares {n_steps} steps but holds {len(steps)}")
    if _require(data, "n_frames", int, what) != n_steps + 1:
        raise ValueError(
            f"{task_id} declares {data['n_frames']} frames for {n_steps} steps; a golden "
            "episode has one post-action frame per step plus the initial one"
        )
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"{task_id} step {index} is not an object: {step!r}")
        if step.get("frame") != f"{FRAME_STEM.format(index)}.png":
            raise ValueError(f"{task_id} step {index} names frame {step.get('frame')!r}")
        thought = step.get("thought")
        if thought is not None and (
            not isinstance(thought, str) or len(thought) > THOUGHT_MAX_CHARS
        ):
            raise ValueError(
                f"{task_id} step {index} carries a thought that is not a string of at most "
                f"{THOUGHT_MAX_CHARS} chars: {thought!r}"
            )
    frames: list[Path] = []
    for index in range(len(steps) + 1):
        frame_path = directory / FRAMES_DIR / f"{FRAME_STEM.format(index)}.png"
        if not frame_path.is_file():
            raise FileNotFoundError(f"{task_id} frame {index} is missing: {frame_path}")
        frames.append(frame_path)
    return Recording(
        task_id=task_id,
        template_id=template_id,
        seed=seed,
        category=category,
        instruction=instruction,
        screen_size=screen_size,
        frames=tuple(frames),
        steps=tuple(steps),
        widgets=widgets,
        source=str(source),
    )


def load_primitives(rows: Any, *, what: str) -> tuple[OrderedPrimitive, ...]:
    """Serialized ``OrderedPrimitive`` field dicts back into primitives, field by field."""
    if not isinstance(rows, list):
        raise ValueError(f"{what} must be a list of primitives, got {rows!r}")
    out: list[OrderedPrimitive] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - _PRIMITIVE_FIELDS or "kind" not in row:
            raise ValueError(f"{what} holds an unusable primitive: {row!r}")
        values = dict(row)
        if values.get("keys") is not None:
            values["keys"] = tuple(values["keys"])
        out.append(OrderedPrimitive(**values))
    return tuple(out)


def _check_dispatched_pixels(
    prims: tuple[OrderedPrimitive, ...], step: dict[str, Any], what: str,
) -> None:
    recorded = step.get("primitives_px")
    if recorded is None:
        raise ValueError(f"{what} has no primitives_px to check the grid twin against")
    derived = json.dumps([asdict(prim) for prim in prims], sort_keys=True)
    if derived != json.dumps(recorded, sort_keys=True):
        raise ValueError(
            f"{what}: the grid primitives denormalize to {derived}, but the VM was sent "
            f"{json.dumps(recorded, sort_keys=True)}"
        )


def step_line(
    step: dict[str, Any],
    arm: str,
    *,
    screen_size: tuple[int, int],
    what: str = "step",
) -> str:
    """The arm's action line for one recorded turn, rendered from its grid primitives.

    The abs arm emits the recorded grid points; the rel arm emits ``norm_delta`` of
    the recorded ``cursor_before`` to those same points denormalized by
    ``denorm_v4`` — the recorder's own derivation, re-checked here against the
    ``primitives_px`` it actually dispatched. A zero delta is a hard error: the rel
    arm cannot express it and dropping it would break the arms' line identity.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown ordered_events_v4 arm: {arm!r}")
    grid = load_primitives(step.get("primitives_grid"), what=f"{what} primitives_grid")
    pixels = denorm_v4(
        OrderedAction(primitives=grid, no_op=not grid), screen_size,
    ).primitives
    _check_dispatched_pixels(pixels, step, what)
    if not grid:
        return NO_OP_LINE
    if arm == ARM_ABS:
        return render_line(grid, ARM_ABS)
    cursor: tuple[int, int] | None = None
    rendered: list[OrderedPrimitive] = []
    for prim in pixels:
        if prim.kind != "move_to":
            rendered.append(prim)
            continue
        if cursor is None:
            cursor = _pixel_pair(
                step.get("cursor_before"), screen_size, what=f"{what} cursor_before",
            )
        delta = (
            norm_delta(prim.x - cursor[0], screen_size[0]),
            norm_delta(prim.y - cursor[1], screen_size[1]),
        )
        cursor = (prim.x, prim.y)
        if delta == (0, 0):
            raise ValueError(f"{what} moves onto the cursor it already holds: {cursor!r}")
        rendered.append(OrderedPrimitive(kind="move", dx=delta[0], dy=delta[1]))
    return render_line(rendered, ARM_REL)


def validate_line(line: str, arm: str, *, what: str = "line") -> str:
    """Strict-parse one assistant line and require its re-render to be byte-identical."""
    if line == TERMINATE_LINE:
        return line
    action = parse_ordered_v4_action(line, arm=arm)
    again = render_line(action.primitives, arm)
    if again != line:
        raise ValueError(f"{what} does not round-trip under {arm}: {line!r} -> {again!r}")
    return line


def dispatched_clicks(
    line: str,
    arm: str,
    step: dict[str, Any],
    *,
    screen_size: tuple[int, int],
    what: str = "step",
) -> tuple[tuple[int, int], ...]:
    """Every pixel the arm's own line presses a mouse button at.

    The line goes back through ``parse_ordered_v4_action`` -> ``denorm_v4`` and is
    walked from the recorded ``cursor_before`` exactly as the closed loop dispatches
    it, so the rel arm is checked on the pixel its delta RECONSTRUCTS (up to a grid
    unit away from the intended target) rather than on the target it meant."""
    if line in (TERMINATE_LINE, NO_OP_LINE):
        return ()
    action = denorm_v4(parse_ordered_v4_action(line, arm=arm), screen_size)
    cursor = _pixel_pair(step.get("cursor_before"), screen_size, what=f"{what} cursor_before")
    presses: list[tuple[int, int]] = []
    for prim in action.primitives:
        if prim.kind == "move_to":
            cursor = (prim.x, prim.y)
        elif prim.kind == "move":
            cursor = (cursor[0] + prim.dx, cursor[1] + prim.dy)
        elif prim.kind == "down" and prim.name in MOUSE_NAMES:
            presses.append(cursor)
    return tuple(presses)


def check_clicks_in_widgets(
    clicks: tuple[tuple[int, int], ...],
    widgets: tuple[tuple[str, tuple[int, int, int, int]], ...],
    *,
    what: str = "step",
) -> None:
    """Require every press to land inside one of the recorded widget bboxes."""
    for xy in clicks:
        if not any(bbox_contains(box, xy) for _, box in widgets):
            raise ValueError(
                f"{what} presses at {list(xy)}, outside every recorded widget bbox: "
                + ", ".join(f"{label}={list(box)}" for label, box in widgets)
            )


def masked_line(line: str, arm: str) -> str:
    """``line`` with its move tokens masked — the arm-invariant projection of a turn."""
    if line in (TERMINATE_LINE, NO_OP_LINE):
        return line
    action = parse_ordered_v4_action(line, arm=arm)
    return "; ".join(
        MOVE_MASK if prim.kind in ("move", "move_to") else render_primitive(prim, arm)
        for prim in action.primitives
    )


def episode_lines(rec: Recording, arm: str) -> tuple[str, ...]:
    """Every assistant turn of an episode: one validated line per step, then TERMINATE.

    Lines are strict-parsed and re-rendered byte-identically, whatever drove the
    episode. An ORACLE episode whose setup published widget bboxes is additionally
    dispatch-checked: a press that lands outside every recorded widget fails here
    instead of surfacing as a closed-loop verifier miss. An agent episode is not —
    its misses and corrections are the trajectory, and its verifier already passed."""
    lines: list[str] = []
    for index, step in enumerate(rec.steps):
        what = f"{rec.task_id} step {index}"
        line = step_line(step, arm, screen_size=rec.screen_size, what=what)
        lines.append(validate_line(line, arm, what=what))
        if rec.widgets and rec.source == SOURCE_ORACLE:
            check_clicks_in_widgets(
                dispatched_clicks(line, arm, step, screen_size=rec.screen_size, what=what),
                rec.widgets,
                what=f"{what} ({arm})",
            )
    lines.append(TERMINATE_LINE)
    if len(lines) != rec.n_frames:
        raise ValueError(f"{rec.task_id}: {len(lines)} lines for {rec.n_frames} frames")
    return tuple(lines)


def _resized(source: Path, resolution: tuple[int, int]) -> Image.Image:
    with Image.open(source) as image:
        return image.convert("RGB").resize(resolution, Image.LANCZOS)


def write_episode_images(
    rec: Recording,
    images_dir: Path | str,
    *,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    quality: int = DEFAULT_JPEG_QUALITY,
    blank: bool = False,
) -> tuple[str, ...]:
    """Materialize every frame under ``<images_dir>/<task_id>/step_NNN.jpg``.

    ``blank`` writes a uniform mid-gray image of the same resolution in place of
    each frame — the sighted/blind control keeps byte-identical text and turn
    structure, so only the pixels differ.
    """
    out_dir = Path(images_dir) / rec.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    for index, source in enumerate(rec.frames):
        dest = out_dir / f"{FRAME_STEM.format(index)}.jpg"
        image = (
            Image.new("RGB", resolution, (BLANK_LEVEL,) * 3)
            if blank
            else _resized(source, resolution)
        )
        image.save(dest, format="JPEG", quality=quality, optimize=False)
        urls.append(str(dest.resolve()))
    return tuple(urls)


def episode_windows(n_frames: int) -> tuple[list[list[bool]], list[int]]:
    """Per-decision liveness snapshots plus eviction points, from the runtime window.

    ``states[i]`` is the liveness of frames ``0..i`` at the decision taken on
    frame ``i`` — the same list the closed-loop evaluator holds at that step,
    because it comes from ``KeepTextWindow`` itself rather than a reimplementation.
    """
    window = KeepTextWindow(_SENTINEL_FRAME)
    states = [window.liveness()]
    for _ in range(1, n_frames):
        window.append_turn(_SENTINEL_ACTION, _SENTINEL_FRAME)
        states.append(window.liveness())
    points = keep_text_eviction_points(n_frames)
    if window.evicted_at != points:
        raise ValueError(
            f"keep-text window ({window.evicted_at!r}) and eviction points "
            f"({points!r}) disagree for {n_frames} frames"
        )
    return states, points


def record_bounds(n_frames: int, points: list[int]) -> tuple[tuple[int, int], ...]:
    """``(first, last)`` decision index of every record — one extra cut per eviction."""
    starts = [0, *points]
    bounds = tuple(
        (start, starts[index + 1] - 1 if index + 1 < len(starts) else n_frames - 1)
        for index, start in enumerate(starts)
    )
    if any(last < first for first, last in bounds) or bounds[-1][1] != n_frames - 1:
        raise ValueError(f"{n_frames} frames with eviction points {points!r} cut badly")
    return bounds


def build_records(
    rec: Recording,
    arm: str,
    *,
    prompt: str,
    lines: tuple[str, ...],
    image_urls: tuple[str, ...],
    split: str,
    allow_resupervision: bool = False,
) -> list[dict[str, Any]]:
    """Every training record of one episode, cut at the runtime's eviction points.

    ``n_frames`` is the record's own user-turn count and ``n_live_images`` how many
    of those still carry pixels; the rest render the ``IMAGE_PLACEHOLDER`` literal
    at their original position.

    ``first_supervised_turn`` is the decision this record is the owning context for.
    It is 0 for every single-record episode, which is the only shape the catalog
    produces (test_shortgoal_templates pins that). A record after an eviction has to
    reopen with the whole text history, and the trainer's collator supervises every
    assistant span with no per-turn opt-out, so those earlier turns would be trained
    a second time under blinded frames they never saw at inference: that shape is a
    hard error unless ``allow_resupervision`` asks for it, and
    ``manifest["counts"]["n_resupervised_turns"]`` is its audit total.
    """
    if not (len(lines) == len(image_urls) == rec.n_frames):
        raise ValueError(
            f"{rec.task_id}: {len(lines)} lines, {len(image_urls)} images, "
            f"{rec.n_frames} frames must agree"
        )
    states, points = episode_windows(rec.n_frames)
    bounds = record_bounds(rec.n_frames, points)
    if len(bounds) > 1 and not allow_resupervision:
        raise ValueError(
            f"{rec.task_id} has {rec.n_frames} frames, so it cuts into {len(bounds)} records "
            f"and record 1 would re-supervise its first {bounds[1][0]} turns under evicted "
            "frames; the collator supervises every assistant span, so shorten the episode "
            "or pass --allow_resupervision"
        )
    goal = GOAL_PREFIX + rec.instruction
    records: list[dict[str, Any]] = []
    for index, (first, last) in enumerate(bounds):
        liveness = states[last]
        for decision in range(first, last + 1):
            if states[decision] != liveness[: decision + 1]:
                raise ValueError(
                    f"{rec.task_id} record {index} spans an eviction at frame {decision}"
                )
        n_live = sum(liveness)
        if n_live > K_IMAGES:
            raise ValueError(f"{rec.task_id} record {index} carries {n_live} live images")
        parts = [
            {"type": "image", "url": url} if live else None
            for url, live in zip(image_urls[: last + 1], liveness, strict=True)
        ]
        records.append({
            "conversation_id": f"{rec.task_id}__r{index:02d}",
            "task_id": rec.task_id,
            "template_id": rec.template_id,
            "category": rec.category,
            "seed": rec.seed,
            "split": split,
            "arm": arm,
            "recipe": RECIPE,
            "action_format": arm,
            "instruction": rec.instruction,
            "n_frames": last + 1,
            "n_live_images": n_live,
            "record_index": index,
            "first_supervised_turn": first,
            "n_records_in_episode": len(bounds),
            "messages": keep_text_messages(prompt, goal, parts, list(lines[: last + 1])),
        })
    return records


def validate_splits(splits: dict[str, Any]) -> dict[str, Any]:
    """Check a split manifest: known names, known task ids, disjoint at (template,seed)."""
    seen: set[str] = set()
    for name in templates.SPLIT_NAMES:
        ids = splits.get(name)
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValueError(f"split {name!r} must be a list of task ids, got {ids!r}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"split {name!r} repeats a task id")
        if seen & set(ids):
            raise ValueError(f"split {name!r} overlaps an earlier split")
        for task in ids:
            template_id, marker, seed = task.partition("__s")
            if (
                not marker
                or not seed.isdigit()
                or template_id not in templates.TEMPLATES_BY_ID
                or templates.task_id(template_id, int(seed)) != task
            ):
                raise ValueError(f"split {name!r} holds an unknown task id: {task!r}")
        seen |= set(ids)
    return splits


def load_splits(path: str | None) -> tuple[dict[str, Any], str]:
    """The split manifest and its provenance — recomputed from the catalog by default."""
    if not path:
        return validate_splits(templates.build_split_manifest()), RECOMPUTED_SPLITS
    resolved = Path(path)
    return validate_splits(json.loads(resolved.read_text())), str(resolved.resolve())


def split_of(task: str, splits: dict[str, Any]) -> str:
    """The one split a task id belongs to."""
    hits = [name for name in templates.SPLIT_NAMES if task in splits[name]]
    if len(hits) != 1:
        raise ValueError(f"{task} appears in {hits!r}, expected exactly one split")
    return hits[0]


def subset_task_ids(subset: str, splits: dict[str, Any]) -> tuple[str, ...]:
    """The ordered task ids of one subset, all of them inside the subset's split."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}, expected one of {SUBSETS}")
    pool = tuple(splits[SUBSET_SPLITS[subset]])
    if subset == "overfit1":
        ids: tuple[str, ...] = (templates.OVERFIT1_TASK_ID,)
    elif subset == "overfit32":
        ids = tuple(templates.OVERFIT32_TASK_IDS)
    else:
        ids = pool
    outside = [task for task in ids if task not in pool]
    if outside:
        raise ValueError(f"subset {subset!r} wants ids outside {SUBSET_SPLITS[subset]!r}: {outside}")
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError(f"subset {subset!r} resolved to {len(ids)} ids ({len(set(ids))} distinct)")
    return ids


def parse_resolution(text: str) -> tuple[int, int]:
    """``"1280x720"`` -> ``(1280, 720)``."""
    match = _RESOLUTION_RE.match(str(text).strip())
    if not match:
        raise ValueError(f"model resolution must look like 1280x720, got {text!r}")
    return int(match.group(1)), int(match.group(2))


def arm_root(output_dir: Path | str, arm: str, *, per_arm: bool) -> Path:
    """Where one arm's dataset lives — a per-arm subdirectory only when building both."""
    if arm not in ARMS:
        raise ValueError(f"unknown ordered_events_v4 arm: {arm!r}")
    root = Path(output_dir)
    return root / ARM_SLUGS[arm] if per_arm else root


def _tally(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[str(record[key])] = counts.get(str(record[key]), 0) + 1
    return dict(sorted(counts.items()))


def _task_tally(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    tasks: dict[str, set[str]] = {}
    for record in records:
        tasks.setdefault(str(record[key]), set()).add(record["task_id"])
    return {name: len(ids) for name, ids in sorted(tasks.items())}


def build_arm(
    *,
    recordings_root: Path | str,
    output_root: Path | str,
    arm: str,
    subset: str,
    splits: dict[str, Any],
    splits_source: str = RECOMPUTED_SPLITS,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    quality: int = DEFAULT_JPEG_QUALITY,
    blank_images: bool = False,
    allow_resupervision: bool = False,
    replicas: int = 1,
) -> dict[str, Any]:
    """Build one arm's ``train/chat.jsonl`` plus its ``manifest.json``; returns the manifest."""
    if arm not in ARMS:
        raise ValueError(f"unknown ordered_events_v4 arm: {arm!r}")
    if not isinstance(replicas, int) or replicas < 1:
        raise ValueError(f"replicas must be a positive int, got {replicas!r}")
    prompt_id = PROMPT_IDS[arm]
    prompt = SYSTEM_PROMPTS[prompt_id]
    task_ids = subset_task_ids(subset, splits)
    root = Path(output_root)
    chat_path = root / CHAT_RELPATH
    images_dir = root / IMAGES_RELPATH
    chat_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    n_frames_written = 0
    sources: dict[str, int] = {}
    for task in task_ids:
        rec = load_recording(recordings_root, task)
        sources[rec.source] = sources.get(rec.source, 0) + 1
        lines = episode_lines(rec, arm)
        urls = write_episode_images(
            rec, images_dir, resolution=resolution, quality=quality, blank=blank_images,
        )
        n_frames_written += len(urls)
        records.extend(build_records(
            rec, arm,
            prompt=prompt,
            lines=lines,
            image_urls=urls,
            split=split_of(task, splits),
            allow_resupervision=allow_resupervision,
        ))
    if replicas > 1:
        records = [
            record if copy == 0
            else {**record, "conversation_id": f"{record['conversation_id']}__c{copy:02d}"}
            for record in records
            for copy in range(replicas)
        ]
    with chat_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    manifest = {
        "stage": STAGE,
        "recipe": RECIPE,
        "arm": arm,
        "action_format": arm,
        "subset": subset,
        "prompt_id": prompt_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "k_images": K_IMAGES,
        "keep_images": KEEP_IMAGES,
        "blank_images": bool(blank_images),
        "allow_resupervision": bool(allow_resupervision),
        "replicas": int(replicas),
        "model_resolution": f"{resolution[0]}x{resolution[1]}",
        "jpeg_quality": int(quality),
        "recordings_root": str(Path(recordings_root).resolve()),
        "splits_source": splits_source,
        "chat_relpath": CHAT_RELPATH,
        "images_relpath": IMAGES_RELPATH,
        "counts": {
            "n_tasks": len(task_ids),
            "n_records": len(records),
            "n_frames_written": n_frames_written,
            "n_live_images": sum(record["n_live_images"] for record in records),
            "max_live_images": max((record["n_live_images"] for record in records), default=0),
            "n_resupervised_turns": sum(record["first_supervised_turn"] for record in records),
            "tasks_by_source": dict(sorted(sources.items())),
            "records_by_split": _tally(records, "split"),
            "records_by_category": _tally(records, "category"),
            "tasks_by_split": _task_tally(records, "split"),
            "tasks_by_category": _task_tally(records, "category"),
        },
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_records(chat_path: Path | str) -> list[dict[str, Any]]:
    """Every record of a built ``chat.jsonl``, in file order."""
    with Path(chat_path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _image_key(url: str) -> str:
    path = Path(url)
    return f"{path.parent.name}/{path.name}"


def canonical_turn(turn: dict[str, Any], arm: str) -> tuple[str, Any]:
    """A turn reduced to what BOTH arms must share: role, text bytes, frame identity.

    The arm-divergent mouse paragraph makes the system prompts differ by design, so
    a system turn canonicalizes to a constant AFTER its bytes are checked against
    the arm's registered prompt.
    """
    role = turn["role"]
    content = turn["content"]
    if role == "system":
        if content != SYSTEM_PROMPTS[PROMPT_IDS[arm]]:
            raise ValueError(f"the system turn is not the registered {PROMPT_IDS[arm]} prompt")
        return (role, SYSTEM_MASK)
    if role == "assistant":
        return (role, masked_line(content, arm))
    return (role, tuple(
        ("text", block["text"]) if block["type"] == "text" else ("image", _image_key(block["url"]))
        for block in content
    ))


def check_arms_identity(rel_chat: Path | str, abs_chat: Path | str) -> dict[str, int]:
    """Assert both arms' records differ ONLY in their move tokens and system prompt."""
    rel_records = read_records(rel_chat)
    abs_records = read_records(abs_chat)
    if not rel_records or len(rel_records) != len(abs_records):
        raise ValueError(f"arms hold {len(rel_records)} and {len(abs_records)} records")
    n_turns = 0
    n_moves = 0
    for left, right in zip(rel_records, abs_records, strict=True):
        for key in sorted(set(left) | set(right)):
            if key in ("messages", "arm", "action_format"):
                continue
            if left.get(key) != right.get(key):
                raise ValueError(f"{left['conversation_id']} arms differ in {key!r}")
        if (left["arm"], right["arm"]) != (ARM_REL, ARM_ABS):
            raise ValueError(f"{left['conversation_id']} is not a (rel, abs) pair")
        if len(left["messages"]) != len(right["messages"]):
            raise ValueError(f"{left['conversation_id']} arms differ in turn count")
        for turn_left, turn_right in zip(left["messages"], right["messages"], strict=True):
            canonical = canonical_turn(turn_left, ARM_REL)
            if canonical != canonical_turn(turn_right, ARM_ABS):
                raise ValueError(
                    f"{left['conversation_id']} diverges beyond the move token at a "
                    f"{turn_left['role']} turn"
                )
            n_turns += 1
            if turn_left["role"] == "assistant":
                n_moves += canonical[1].count(MOVE_MASK)
    return {
        "n_records": len(rel_records),
        "n_turns": n_turns,
        "n_masked_moves": n_moves,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build short-goal ordered_events_v4 chat records.")
    p.add_argument("--recordings_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--arm", default="both", choices=(*ARMS, "both"))
    p.add_argument("--subset", default="full", choices=SUBSETS)
    p.add_argument("--splits", default="")
    p.add_argument("--model_resolution", default="1280x720")
    p.add_argument("--jpeg_quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument("--blank_images", action="store_true")
    p.add_argument("--allow_resupervision", action="store_true")
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--check_arms", action="store_true")
    p.add_argument("--check_only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    arms = ARMS if args.arm == "both" else (args.arm,)
    if (args.check_arms or args.check_only) and len(arms) != 2:
        raise ValueError("--check_arms/--check_only compare both arms; pass --arm both")
    resolution = parse_resolution(args.model_resolution)
    splits, splits_source = load_splits(args.splits)
    roots = {arm: arm_root(args.output_dir, arm, per_arm=len(arms) > 1) for arm in arms}
    for arm in arms:
        if args.check_only:
            continue
        manifest = build_arm(
            recordings_root=args.recordings_root,
            output_root=roots[arm],
            arm=arm,
            subset=args.subset,
            splits=splits,
            splits_source=splits_source,
            resolution=resolution,
            quality=args.jpeg_quality,
            blank_images=args.blank_images,
            allow_resupervision=args.allow_resupervision,
            replicas=args.replicas,
        )
        counts = manifest["counts"]
        print(
            f"[shortgoal_build] {arm} {args.subset}: {counts['n_records']} records from "
            f"{counts['n_tasks']} tasks -> {roots[arm] / CHAT_RELPATH}"
        )
    if args.check_arms or args.check_only:
        stats = check_arms_identity(
            roots[ARM_REL] / CHAT_RELPATH, roots[ARM_ABS] / CHAT_RELPATH,
        )
        print(f"[shortgoal_build] arms identical after masking moves: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
