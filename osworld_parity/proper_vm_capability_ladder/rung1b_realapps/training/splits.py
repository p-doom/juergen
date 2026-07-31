from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..fixtures import Fixture, canonical_bytes, load_manifest, sha256_value


SPLITS_PATH = Path(__file__).with_name("splits.json")
TEMPLATES = ("vscode_focus_type", "local_document_scroll", "files_drag")


class SplitContractError(RuntimeError):
    pass


class SealedEvaluationError(SplitContractError):
    pass


@dataclass(frozen=True)
class SeedCell:
    template: str
    seed: int


@dataclass(frozen=True)
class SplitManifest:
    splits: dict[str, tuple[SeedCell, ...]]
    split_sha256: dict[str, str]
    manifest_payload_sha256: str


def load_split_manifest(path: Path = SPLITS_PATH) -> SplitManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    seal = raw.pop("manifest_payload_sha256", None)
    if not isinstance(seal, str) or sha256_value(raw) != seal:
        raise SplitContractError("training split manifest seal mismatch")
    if raw.get("schema_version") != 1 or raw.get("generator_version") != "r1b-realapps-v1":
        raise SplitContractError("unsupported training split schema")
    if raw.get("official_osworld_reuse") is not False:
        raise SplitContractError("benchmark reuse policy drift")
    if raw.get("eval_policy") != "sealed_ids_only_never_materialize_in_build_or_collection":
        raise SplitContractError("sealed evaluation policy drift")
    raw_splits = raw.get("splits")
    hashes = raw.get("split_sha256")
    if not isinstance(raw_splits, dict) or not isinstance(hashes, dict):
        raise SplitContractError("split rows/hashes missing")
    splits: dict[str, tuple[SeedCell, ...]] = {}
    seen: set[tuple[str, int]] = set()
    for split in ("train", "development", "evaluation_sealed"):
        rows = raw_splits.get(split)
        if not isinstance(rows, list) or sha256_value(rows) != hashes.get(split):
            raise SplitContractError(f"split hash mismatch: {split}")
        cells = tuple(SeedCell(**row) for row in rows)
        for cell in cells:
            if cell.template not in TEMPLATES or cell.seed <= 0:
                raise SplitContractError(f"invalid seed cell: {cell}")
            key = (cell.template, cell.seed)
            if key in seen:
                raise SplitContractError(f"seed cell crosses splits: {key}")
            seen.add(key)
        splits[split] = cells
    return SplitManifest(splits, dict(hashes), seal)


_UNICODE_TEXTS = (
    "Crème brûlée — Αθήνα — 🛰️",
    "mañana • Zürich • 雪 • 🧩",
    "Smørrebrød — Łódź — 🐋",
    "São Paulo • Київ • 🌿",
    "İstanbul — résumé — 🧪",
    "Québec • Ελληνικά • 🪁",
)


def _make_fixture(cell: SeedCell, split: str) -> Fixture:
    rng = random.Random(cell.seed)
    seed = cell.seed
    if cell.template == "vscode_focus_type":
        expected = _UNICODE_TEXTS[seed % len(_UNICODE_TEXTS)] + f" #{seed}"
        params: dict[str, Any] = {
            "file_name": f"training-note-{seed}.txt",
            "initial_text": f"replace deterministic training seed {seed}\n",
        }
        expected_value = {"text": expected}
        near = {"text": expected.encode("ascii", "ignore").decode("ascii")}
        horizon = 4
        instruction = (
            f"In the open VS Code document, focus the editor, replace all text with "
            f"“{expected}”, and save the file."
        )
    elif cell.template == "local_document_scroll":
        direction = "down" if seed % 2 else "up"
        initial_y = 0 if direction == "down" else 2600 + 100 * rng.randrange(4)
        params = {
            "direction": direction,
            "initial_y": initial_y,
            "port": 19000 + seed % 700,
            "document_lines": 160 + 10 * rng.randrange(5),
        }
        expected_value = {"min_delta": 500}
        near = {"direction": "up" if direction == "down" else "down"}
        horizon = 2
        instruction = f"Scroll {direction} in the open local training document by at least one screen."
    else:
        params = {
            "source_name": f"training-parcel-{seed}.txt",
            "destination_name": f"Delivered-{seed}",
            "decoy_name": f"Archive-{seed}",
            "content": f"deterministic training parcel {seed}\n",
        }
        expected_value = {"destination": params["destination_name"]}
        near = {"destination": params["decoy_name"]}
        horizon = 3
        instruction = (
            f"In Files, drag “{params['source_name']}” into the folder "
            f"“{params['destination_name']}”."
        )
    unsigned = {
        "id": f"r1b-{split}-{cell.template}-{seed}",
        "template": cell.template,
        "split": split,
        "parameter_seed": seed,
        "horizon": horizon,
        "instruction": instruction,
        "params": params,
        "expected": expected_value,
        "near_miss": near,
    }
    return Fixture(**unsigned, fixture_sha256=sha256_value(unsigned))


def materialize_tasks(split: str) -> tuple[Fixture, ...]:
    manifest = load_split_manifest()
    if split == "evaluation_sealed":
        raise SealedEvaluationError(
            "sealed evaluation IDs may not be materialized by training/build tooling"
        )
    if split == "development":
        fixtures = load_manifest().fixtures
        expected = {(cell.template, cell.seed) for cell in manifest.splits[split]}
        observed = {(fixture.template, fixture.parameter_seed) for fixture in fixtures}
        if observed != expected:
            raise SplitContractError("development fixture/split manifest mismatch")
        return fixtures
    if split != "train":
        raise SplitContractError(f"unknown/unopen split: {split}")
    return tuple(_make_fixture(cell, split) for cell in manifest.splits[split])


def proposed_train_extension(template: str, *, first_seed: int, count: int) -> tuple[SeedCell, ...]:
    if template not in TEMPLATES or first_seed <= 0 or count <= 0:
        raise ValueError("invalid train extension request")
    existing = {
        (cell.template, cell.seed)
        for rows in load_split_manifest().splits.values()
        for cell in rows
    }
    cells = tuple(SeedCell(template, first_seed + offset) for offset in range(count))
    collisions = [(cell.template, cell.seed) for cell in cells if (cell.template, cell.seed) in existing]
    if collisions:
        raise SplitContractError(f"proposed train seeds collide with sealed splits: {collisions}")
    return cells
