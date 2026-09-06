"""Load and attest the describe-extract prompt pack."""

from __future__ import annotations

import hashlib
from pathlib import Path
from string import Template
from typing import Any

import yaml


class PromptPack:
    def __init__(self, path: Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha = hashlib.sha256(raw).hexdigest()
        data = yaml.safe_load(raw)
        expected = {"system", "describe_prose", "extract_system", "extract"}
        if (
            not isinstance(data, dict)
            or set(data) != expected
            or any(
                not isinstance(value, str) or not value.strip()
                for value in data.values()
            )
        ):
            raise TypeError(f"{self.path} must contain the canonical prompt pack")
        self._prompts: dict[str, str] = data

    def get(self, key: str) -> str:
        return self._prompts[key]

    def render(self, key: str, **fields: Any) -> str:
        return Template(self._prompts[key]).substitute(**fields)

    def snapshot_to(self, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "prompts.yaml"
        dest.write_bytes(self.path.read_bytes())
        return dest
