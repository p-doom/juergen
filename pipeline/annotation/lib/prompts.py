"""Per-method prompt packs.

Each annotation method owns a ``prompts.yaml`` next to its ``annotator.py``;
prompt text stays out of the Python so it can be iterated on directly. The
pack's SHA (over the raw yaml bytes) is stamped on every goal row
(``prompt_pack_sha``) and into the artifact manifest, and the stage snapshots
the yaml into the output dir — so an artifact is always traceable to the exact
prompts that produced it.

``${name}`` placeholders are filled with string.Template.safe_substitute (the
literal ``{ }`` of JSON examples pass through untouched; a missing field stays
``${name}`` rather than raising). Callers pre-format numbers/repr.
"""

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
        self.sha = hashlib.sha256(raw).hexdigest()[:16]
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{self.path} must be a mapping of prompt-name -> text")
        self._prompts: dict[str, str] = data

    def get(self, key: str) -> str:
        """A prompt with no placeholders (e.g. 'system')."""
        return self._prompts[key]

    def render(self, key: str, **fields: Any) -> str:
        """Prompt ``key`` with ${...} placeholders substituted."""
        return Template(self._prompts[key]).safe_substitute(**fields)

    def snapshot_to(self, dest_dir: Path) -> Path:
        """Copy the yaml into the artifact (audit trail)."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "prompts.yaml"
        dest.write_bytes(self.path.read_bytes())
        return dest
