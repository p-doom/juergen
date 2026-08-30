from __future__ import annotations

from importlib.resources import files
from typing import Any

from harness_render import HarnessRenderer, HarnessRenderSpec

SPEC_ID = "stream-cuagym-qwen35-render-q92"
SPEC_SHA256 = "33a72e145aa48ea1f5851b47cc170cb8b1ffaf3a71e10ea6e0d5433c05a934bc"
SYSTEM_PROMPT_SHA256 = (
    "5377682e1af25754a3b982d2dc4521690f41b59b25bac04e690e8ddc9445700e"
)
ACTION_CONTRACT = "ordered_events_v3_relative_1000_grid_v1"
OBSERVATION_CONTRACT = "osworld_cursor_jpeg_q92_420_1920x1080_v1"
OBSERVATION_METADATA = {
    "media_type": "image/jpeg",
    "jpeg_quality": 92,
    "color_mode": "RGB",
    "chroma_subsampling": "4:2:0",
    "width": 1920,
    "height": 1080,
}


def system_prompt() -> str:
    return (
        files(__package__)
        .joinpath("system_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


def renderer() -> HarnessRenderer[Any]:
    raw = files(__package__).joinpath("render_spec.json").read_bytes()
    spec = HarnessRenderSpec.from_bytes(raw, expected_sha256=SPEC_SHA256)
    return HarnessRenderer(
        spec,
        spec_sha256=SPEC_SHA256,
        system_prompt=system_prompt(),
        action_contract=ACTION_CONTRACT,
        observation_contract=OBSERVATION_CONTRACT,
    )


def metadata() -> dict[str, Any]:
    bound = renderer()
    return {
        "render_spec_id": bound.spec.spec_id,
        "render_spec_sha256": bound.spec.sha256,
        "system_prompt_sha256": bound.spec.system_prompt_sha256,
        "action_contract": bound.spec.action_contract,
        "observation_contract": bound.spec.observation_contract,
        "max_completed_turns": bound.spec.max_completed_turns,
        **OBSERVATION_METADATA,
    }


__all__ = [
    "ACTION_CONTRACT",
    "OBSERVATION_CONTRACT",
    "OBSERVATION_METADATA",
    "SPEC_ID",
    "SPEC_SHA256",
    "SYSTEM_PROMPT_SHA256",
    "metadata",
    "renderer",
    "system_prompt",
]
