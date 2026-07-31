from __future__ import annotations

import base64
import hashlib
from typing import Any

from .fixtures import Fixture


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base_state(fixture: Fixture) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
    }


def focus_state(fixture: Fixture, text: str) -> dict[str, Any]:
    return {
        **base_state(fixture),
        "application": "vscode",
        "file_name": fixture.params["file_name"],
        "content_b64": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "content_sha256": _sha(text),
    }


def scroll_state(fixture: Fixture, y: int) -> dict[str, Any]:
    return {
        **base_state(fixture),
        "application": "chrome",
        "document_kind": "guest_local_development_document",
        "scroll_y": int(y),
    }


def drag_state(fixture: Fixture, location: str) -> dict[str, Any]:
    content_sha = _sha(str(fixture.params["content"]))
    return {
        **base_state(fixture),
        "application": "files",
        "drag_backend": "filesystem",
        "source_exists": location == "source",
        "destination_sha256": content_sha if location == "destination" else None,
        "decoy_sha256": content_sha if location == "decoy" else None,
    }


def reset_state(fixture: Fixture) -> dict[str, Any]:
    if fixture.template == "vscode_focus_type":
        return focus_state(fixture, str(fixture.params["initial_text"]))
    if fixture.template == "local_document_scroll":
        return scroll_state(fixture, int(fixture.params["initial_y"]))
    return drag_state(fixture, "source")


def gold_state(fixture: Fixture) -> dict[str, Any]:
    if fixture.template == "vscode_focus_type":
        return focus_state(fixture, str(fixture.expected["text"]))
    if fixture.template == "local_document_scroll":
        delta = int(fixture.expected["min_delta"]) + 100
        y = int(fixture.params["initial_y"]) + (
            delta if fixture.params["direction"] == "down" else -delta
        )
        return scroll_state(fixture, y)
    return drag_state(fixture, "destination")


def near_miss_state(fixture: Fixture) -> dict[str, Any]:
    if fixture.template == "vscode_focus_type":
        return focus_state(fixture, str(fixture.near_miss["text"]))
    if fixture.template == "local_document_scroll":
        delta = int(fixture.expected["min_delta"]) + 100
        y = int(fixture.params["initial_y"]) + (
            -delta if fixture.params["direction"] == "down" else delta
        )
        return scroll_state(fixture, max(0, y))
    return drag_state(fixture, "decoy")
