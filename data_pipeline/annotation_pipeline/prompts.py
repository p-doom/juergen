"""Load annotation/judge prompts from prompts.yaml.

Keeps all prompt text out of the Python so it can be iterated on directly.
``${name}`` placeholders are filled with string.Template.safe_substitute (so the
literal ``{ }`` of JSON examples pass through untouched, and a missing field is
left as ``${name}`` rather than raising). Callers pre-format numbers/repr.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

import yaml

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.yaml"


@lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    data = yaml.safe_load(PROMPTS_PATH.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{PROMPTS_PATH} must be a mapping of prompt-name -> text")
    return data


def get(key: str) -> str:
    """Return a prompt with no placeholders (e.g. 'system')."""
    return _prompts()[key]


def render(key: str, **fields: Any) -> str:
    """Return prompt ``key`` with ${...} placeholders substituted."""
    return Template(_prompts()[key]).safe_substitute(**fields)
