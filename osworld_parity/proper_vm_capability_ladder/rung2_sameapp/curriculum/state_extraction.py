"""Independent artifact/state extraction for curriculum verifier inputs.

Extractors read application artifacts. They do not read ``task.expected`` or
``task.near_miss``; only task identity and setup parameters locate the artifact.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Any

from .schema import SemanticTask


def _base(task: SemanticTask, held_inputs: Iterable[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "held_inputs": sorted(held_inputs),
    }


def extract_state(
    task: SemanticTask, artifact_root: Path, *, held_inputs: Iterable[str] = ()
) -> dict[str, Any]:
    """Extract hidden verifier state without consulting the expected object."""

    root = artifact_root.resolve(strict=True)
    state = _base(task, held_inputs)
    if task.app == "writer":
        path = root / str(task.params["file_name"])
        with zipfile.ZipFile(path) as archive:
            content = archive.read("content.xml")
        xml = ET.fromstring(content)
        namespace = {"text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
        text = "".join(
            "".join(node.itertext()) for node in xml.findall(".//text:p", namespace)
        )
        state.update(
            {
                "app": "writer",
                "text": text,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bold": b'font-weight="bold"' in content,
                "saved": text != str(task.params["initial_text"]),
            }
        )
        return state
    if task.app == "calc":
        path = root / str(task.params["file_name"])
        with zipfile.ZipFile(path) as archive:
            xml = ET.fromstring(archive.read("content.xml"))
        namespace = {
            "t": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
            "o": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
        }
        target = str(task.params["cell"])
        target_col, target_row = ord(target[0]) - 65, int(target[1:]) - 1
        logical_row = 0
        selected = None
        for row_node in xml.findall(".//t:table-row", namespace):
            repeat = int(
                row_node.get(
                    "{" + namespace["t"] + "}number-rows-repeated", "1"
                )
            )
            if logical_row <= target_row < logical_row + repeat:
                logical_col = 0
                for cell_node in row_node.findall("t:table-cell", namespace):
                    column_repeat = int(
                        cell_node.get(
                            "{" + namespace["t"] + "}number-columns-repeated", "1"
                        )
                    )
                    if logical_col <= target_col < logical_col + column_repeat:
                        selected = cell_node
                        break
                    logical_col += column_repeat
                break
            logical_row += repeat
        if selected is None:
            raise ValueError(f"{task.task_id}: target Calc cell is absent")
        formula = selected.get("{" + namespace["t"] + "}formula")
        value = selected.get("{" + namespace["o"] + "}value") or "".join(
            selected.itertext()
        )
        state.update(
            {
                "app": "calc",
                "cell": target,
                "formula": formula,
                "display_value": value,
                "saved": formula is not None,
            }
        )
        return state
    if task.app == "files":
        source = root / str(task.params["source_name"])
        found: list[tuple[str, Path]] = []
        for name in (task.params["destination_name"], task.params["decoy_name"]):
            destination = root / str(name)
            if destination.is_dir():
                found.extend((str(name), path) for path in destination.iterdir() if path.is_file())
        name, path = found[0] if len(found) == 1 else (None, source)
        data = path.read_bytes() if path.is_file() else b""
        state.update(
            {
                "app": "files",
                "source_exists": source.is_file(),
                "destination": name,
                "final_name": path.name,
                "content_sha256": hashlib.sha256(data).hexdigest(),
                "saved": not source.exists(),
            }
        )
        return state
    if task.app == "chrome":
        raw = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if raw.get("ready") is not True:
            raise ValueError(f"{task.task_id}: Chrome fixture is not ready")
        state.update(
            {
                "app": "chrome",
                "section": raw["section"],
                "scroll_y": int(raw["scroll_y"]),
                "setting_enabled": bool(raw["setting_enabled"]),
                "saved": bool(raw["setting_enabled"]),
            }
        )
        return state
    if task.app == "vscode":
        path = root / str(task.params["file_name"])
        data = path.read_bytes()
        data.decode("utf-8")
        state.update(
            {
                "application": "vscode",
                "file_name": path.name,
                "content_b64": base64.b64encode(data).decode("ascii"),
                "content_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        return state
    raise ValueError(f"unsupported curriculum app: {task.app}")
