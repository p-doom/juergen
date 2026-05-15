"""Complete an omegalax-exported HF dir.

omegalax/scripts/export_to_hf.py emits config.json + safetensors but omits
tokenizer sidecars + a few config keys SGLang/transformers need. We patch
both by referencing the cached HF snapshot for the architecture's model_id.

This is a stopgap — the right fix is to make omegalax's exporter complete.
Until then, every roundtrip eval calls ``complete_export_dir`` post-export.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Tokenizer + processor sidecars to copy from the HF snapshot.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

# config.json keys missing from omegalax's export.
_CONFIG_PATCH_KEYS = ("architectures", "transformers_version", "vision_end_token_id")


def find_hf_snapshot(model_id: str, hf_home: Path) -> Path:
    """Locate the cached HF snapshot dir for ``model_id`` under ``hf_home``."""
    repo_dir_name = "models--" + model_id.replace("/", "--")
    snapshots_root = Path(hf_home) / "hub" / repo_dir_name / "snapshots"
    snapshots = sorted(p for p in snapshots_root.glob("*") if p.is_dir())
    if not snapshots:
        raise FileNotFoundError(
            f"no HF snapshot for {model_id} at {snapshots_root}; populate the cache first"
        )
    return snapshots[-1]


def complete_export_dir(export_dir: Path, snapshot_dir: Path) -> dict:
    """Copy missing tokenizer sidecars + patch missing config.json keys.

    Returns ``{"copied": [...], "patched": [...]}`` for the manifest.
    """
    export_dir = Path(export_dir)
    snapshot_dir = Path(snapshot_dir)
    copied: list[str] = []
    for fname in _TOKENIZER_FILES:
        src = snapshot_dir / fname
        dst = export_dir / fname
        if src.is_file() and not dst.exists():
            shutil.copy(src, dst)
            copied.append(fname)

    cfg_path = export_dir / "config.json"
    snap_cfg_path = snapshot_dir / "config.json"
    patched: list[str] = []
    if cfg_path.is_file() and snap_cfg_path.is_file():
        export_cfg = json.loads(cfg_path.read_text())
        snap_cfg = json.loads(snap_cfg_path.read_text())
        for k in _CONFIG_PATCH_KEYS:
            if k in snap_cfg and k not in export_cfg:
                export_cfg[k] = snap_cfg[k]
                patched.append(k)
        if patched:
            cfg_path.write_text(json.dumps(export_cfg, indent=2))

    return {"copied": copied, "patched": patched, "snapshot_dir": str(snapshot_dir)}
