#!/usr/bin/env python3
"""Pinned rung-2 canvas/action contract plus parity-loop geometry.

The shared harness files are imported read-only.  Their hashes are frozen in
``frozen_manifest.json`` so a later edit fails loudly rather than silently
changing this preregistered experiment.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
FROZEN_PATH = HERE / "frozen_manifest.json"
DEFAULT_AUDIT_DIR = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/audit_operand"
)
DEFAULT_PARITY_DIR = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/parity_harness"
)
DEFAULT_GROUNDING_DIR = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/"
    "reinforcement-learning/rl/grounding"
)
Semantic = Literal["absolute_toolcall", "move_rel", "deltatype_raw"]
SEMANTICS: tuple[Semantic, ...] = ("absolute_toolcall", "move_rel")
SPACE = {
    "absolute_toolcall": "abs_norm",
    "move_rel": "rel_norm",
    "deltatype_raw": "rel_px",
}
EXPECTED_ACTION = {
    "absolute_toolcall": "left_click",
    "move_rel": "move_rel",
    "deltatype_raw": "delta",
}
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_frozen() -> dict[str, Any]:
    value = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if value.get("status") != "preregistered_no_results":
        raise ContractError("frozen manifest is not in preregistered state")
    return value


def _source_paths(audit_dir: Path, parity_dir: Path, grounding_dir: Path) -> dict[str, Path]:
    return {
        "rung2_scene.py": audit_dir / "rung2_scene.py",
        "audit_render.py": audit_dir / "render.py",
        "audit_arms.py": audit_dir / "arms.py",
        "parity_run_parity.py": parity_dir / "run_parity.py",
        "parity_render.py": parity_dir / "render.py",
        "parity_arms.py": parity_dir / "arms.py",
        "grounding_dataset.py": grounding_dir / "dataset.py",
        "grounding_parsing.py": grounding_dir / "parsing.py",
        "heldout_scenes.jsonl": audit_dir / "runs/rung2_offshelf/px/scenes.jsonl",
        "train_scenes.jsonl": audit_dir / "r3data_2k/scenes_train.jsonl",
        "val_scenes.jsonl": audit_dir / "r3data_2k/scenes_val.jsonl",
    }


def verify_frozen_sources(
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    parity_dir: Path = DEFAULT_PARITY_DIR,
    grounding_dir: Path = DEFAULT_GROUNDING_DIR,
) -> dict[str, str]:
    expected = load_frozen()["sources"]
    actual: dict[str, str] = {}
    for name, path in _source_paths(audit_dir, parity_dir, grounding_dir).items():
        if not path.is_file():
            raise ContractError(f"missing frozen source {name}: {path}")
        actual[name] = sha256_file(path)
        if actual[name] != expected[name]:
            raise ContractError(
                f"frozen source drift for {name}: {actual[name]} != {expected[name]}"
            )
    return actual


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ContractError(f"blank JSONL line {path}:{line_no}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"non-object JSONL row {path}:{line_no}")
        rows.append(value)
    return rows


def heldout_image_aggregate(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        data = Path(row["image_path"]).read_bytes()
        h.update(row["scene_id"].encode("utf-8") + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()


def _load_rung2(audit_dir: Path):
    """Load audit_operand/rung2_scene.py without retaining generic module aliases.

    The audited file uses sibling imports named ``arms`` and ``render``.  Tests in
    this monorepo may already have modules under those names, so temporarily
    isolate the aliases and restore the caller's module table afterward.
    """
    path = (audit_dir / "rung2_scene.py").resolve()
    saved = {name: sys.modules.get(name) for name in ("arms", "render", "_rlmods")}
    for name in saved:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(audit_dir.resolve()))
    name = f"_synthetic_multistep_rung2_{hash(path)}"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ContractError(f"cannot load rung-2 contract from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        rlmods = sys.modules.get("_rlmods")
        if rlmods is None:
            raise ContractError("rung-2 import did not load audited _rlmods")
    finally:
        sys.path.pop(0)
        for alias, old in saved.items():
            if old is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = old
    if Path(module.__file__).resolve() != path:
        raise ContractError(f"loaded wrong rung-2 module: {module.__file__}")
    return module, rlmods


@dataclass(frozen=True)
class ParsedMove:
    coord: tuple[int, int] | None
    action: str | None
    terminate: bool
    parse_ok: bool


class Contract:
    def __init__(self, audit_dir: Path = DEFAULT_AUDIT_DIR, *, verify: bool = True) -> None:
        self.audit_dir = audit_dir.resolve()
        if verify:
            verify_frozen_sources(self.audit_dir)
        self.rung2, self.rlmods = _load_rung2(self.audit_dir)
        frozen = load_frozen()["episode_contract"]
        for actual, expected, label in (
            (self.rung2.SW, frozen["screen"][0], "screen width"),
            (self.rung2.SH, frozen["screen"][1], "screen height"),
            (self.rung2.BOX, frozen["box_side_px"], "box side"),
        ):
            if actual != expected:
                raise ContractError(f"rung-2 {label} drift: {actual} != {expected}")

    @property
    def screen(self) -> tuple[int, int]:
        return self.rung2.SW, self.rung2.SH

    def render_png(self, bbox: list[int] | tuple[int, ...], cursor: tuple[int, int]) -> bytes:
        """Render exactly as rung2_scene.build_scenes (including PNG defaults)."""
        bx, by, x2, y2 = (int(v) for v in bbox)
        if x2 - bx != self.rung2.BOX or y2 - by != self.rung2.BOX:
            raise ContractError(f"bbox violates rung-2 BOX={self.rung2.BOX}: {bbox}")
        image = Image.new("RGB", self.screen, self.rung2.BG)
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [bx, by, x2, y2],
            fill=self.rung2.BOX_FILL,
            outline=self.rung2.BOX_EDGE,
            width=4,
        )
        self.rung2.draw_arrow(image, cursor)
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    def to_norm(self, point: tuple[int, int]) -> tuple[int, int]:
        return tuple(self.rlmods.to_norm(point[0], point[1], *self.screen))

    def apply_coord(
        self, semantic: Semantic, cursor: tuple[int, int], coord: tuple[int, int]
    ) -> tuple[int, int]:
        """Parity ``run_parity.apply_coord`` for the two preregistered spaces."""
        sw, sh = self.screen
        if semantic == "absolute_toolcall":
            return tuple(self.rlmods.from_norm(coord[0], coord[1], sw, sh))
        if semantic == "move_rel":
            cnx, cny = self.rlmods.to_norm(cursor[0], cursor[1], sw, sh)
            return tuple(self.rlmods.from_norm(cnx + coord[0], cny + coord[1], sw, sh))
        if semantic == "deltatype_raw":
            return (
                max(0, min(sw - 1, cursor[0] + coord[0])),
                max(0, min(sh - 1, cursor[1] + coord[1])),
            )
        raise ContractError(f"unknown semantic {semantic!r}")

    def ideal_coord(
        self, semantic: Semantic, cursor: tuple[int, int], target: tuple[int, int]
    ) -> tuple[int, int]:
        target_norm = self.to_norm(target)
        if semantic == "absolute_toolcall":
            return target_norm
        if semantic == "deltatype_raw":
            return target[0] - cursor[0], target[1] - cursor[1]
        cursor_norm = self.to_norm(cursor)
        return target_norm[0] - cursor_norm[0], target_norm[1] - cursor_norm[1]

    def in_bbox(self, point: tuple[int, int], bbox: list[int] | tuple[int, ...]) -> bool:
        return bool(self.rlmods.in_bbox(point, tuple(int(v) for v in bbox)))

    def distance_to_box(
        self, point: tuple[int, int], bbox: list[int] | tuple[int, ...]
    ) -> float:
        return float(self.rlmods.distance_to_box(point, tuple(int(v) for v in bbox)))

    def parse(self, semantic: Semantic, text: str, tool_calls: Any = None) -> ParsedMove:
        move = self.rung2.GRAMMARS[semantic]["parse"](text, tool_calls)
        coord = tuple(move.coord) if move.coord is not None else None
        return ParsedMove(coord, move.action, bool(move.terminate), bool(move.parse_ok))

    def system_prompt(self, semantic: Semantic) -> str:
        return str(self.rung2.GRAMMARS[semantic]["system"])

    def preamble_text(self, cursor: tuple[int, int], target: tuple[int, int]) -> str:
        return self.rung2.preamble_text({"cursor": list(cursor), "target_center": list(target)})

    def user_text(
        self,
        semantic: Semantic,
        cursor: tuple[int, int],
        target: tuple[int, int],
        *,
        target_index: int,
        target_count: int,
        preamble: bool,
        prior: list[str] | None = None,
    ) -> str:
        scene = {"cursor": list(cursor), "target_center": list(target)}
        base = self.rung2.build_user_text(
            self.rung2.GRAMMARS[semantic], scene, False, preamble
        )
        old = (
            "This is a SINGLE-STEP targeting task. The screenshot is the FINAL state -- "
            "do NOT wait and do NOT terminate."
        )
        new = (
            "This is a MULTI-STEP targeting task. After a correct action a new green box "
            "appears; after a miss the same box remains. Keep acting until every box is hit.\n"
            f"Current target: {target_index + 1} of {target_count}."
        )
        if base.count(old) != 1:
            raise ContractError("rung-2 single-step clause changed")
        text = base.replace(old, new)
        if prior:
            # Full outputs: no strip, clipping, last-line extraction, or prose deletion.
            text += "\nYour previous complete outputs:\n" + "\n---\n".join(prior)
        return text


def serialize_action(
    semantic: Semantic,
    coord: tuple[int, int],
    *,
    prose: str | None = None,
) -> str:
    if semantic == "deltatype_raw":
        action_span = f"{coord[0]} {coord[1]} 0 ; +LMB -LMB"
        return action_span if prose is None else prose + "\n" + action_span
    action = EXPECTED_ACTION[semantic]
    action_span = (
        '<tool_call>\n{"name": "computer_use", "arguments": '
        f'{{"action": "{action}", "coordinate": [{coord[0]}, {coord[1]}]}}}}\n'
        "</tool_call>"
    )
    return action_span if prose is None else prose + "\n" + action_span


def strict_schema_ok(semantic: Semantic, raw: str, coord: tuple[int, int] | None) -> bool:
    if coord is None:
        return False
    if semantic == "deltatype_raw":
        if "<tool_call>" in (raw or "") or " | tool_calls=" in (raw or ""):
            return False
        lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
        if not lines:
            return False
        return bool(re.fullmatch(r"-?\d+\s+-?\d+\s+0\s*;\s*\+LMB\s+-LMB", lines[-1]))
    matches = list(_TOOL_RE.finditer(raw or ""))
    if len(matches) != 1:
        return False
    try:
        payload = json.loads(matches[0].group(1))
        arguments = payload["arguments"]
        parsed = tuple(int(round(float(v))) for v in arguments["coordinate"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        payload.get("name") == "computer_use"
        and arguments.get("action") == EXPECTED_ACTION[semantic]
        and len(arguments.get("coordinate", [])) == 2
        and parsed == coord
    )


def unit_range_ok(semantic: Semantic, coord: tuple[int, int] | None) -> bool:
    if coord is None:
        return False
    if semantic == "absolute_toolcall":
        return all(0 <= value <= 999 for value in coord)
    if semantic == "deltatype_raw":
        return -1919 <= coord[0] <= 1919 and -1079 <= coord[1] <= 1079
    return all(-999 <= value <= 999 for value in coord)


def request_seed(episode_id: str, k: int, target_index: int, attempt: int) -> int:
    key = f"synthetic-multistep-v1|{episode_id}|{k}|{target_index}|{attempt}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def cosine(a: tuple[int, int], b: tuple[int, int]) -> float | None:
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return None
    return (a[0] * b[0] + a[1] * b[1]) / (na * nb)


def oscillates(previous: tuple[int, int] | None, current: tuple[int, int]) -> bool:
    """Preregistered reversal: >=3 px moves with direction cosine <= -0.8."""
    if previous is None or math.hypot(*previous) < 3 or math.hypot(*current) < 3:
        return False
    value = cosine(previous, current)
    return value is not None and value <= -0.8
