"""Annotation-method registry.

A method is a package directory ``annotation/methods/<name>/`` containing
``annotator.py`` (+ its ``prompts.yaml``). The module must declare:

  INPUT_KIND: "frames"  — consumes AnnotationUnits of a segment view; its
                          ``run_unit(unit, ctx)`` returns view-local goal
                          spans (converted to master intervals by the stage).
              "goals"   — an enrichment pass; consumes an existing goals
                          artifact (``--input-goals-dir``); its
                          ``run_unit(item, ctx)`` returns the enriched rows.
              "days"    — sequential-watching methods; consumes a whole
                          DayStream (lib/days; needs ``--clips-manifest`` for
                          wall-clock day grouping); its ``run_unit(item, ctx)``
                          walks the day IN ORDER (many cached calls) and
                          returns thought rows in (segment_id, master_idx)
                          coordinates.

  run_unit(unit_or_item, ctx) -> dict   — cached labeler round-trip(s).

Optionally: ``LABELER_DEFAULTS`` — {"temperature": .., "reasoning_effort": ..}
applied by the stage when the corresponding CLI flag is unset (a method's
model discipline travels with the method).

``ctx`` is the stage-provided MethodContext (labeler, prompt pack, per-unit
cache dir, render params, and ``params`` — method knobs from ``--param`` plus
mode-specific plumbing the stage injects).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from realigned_pipeline.annotation.lib.labeler import Labeler
from realigned_pipeline.annotation.lib.prompts import PromptPack

METHODS_DIR = Path(__file__).resolve().parents[1] / "methods"
INPUT_KINDS = ("frames", "goals", "days")


@dataclass
class Method:
    name: str
    input_kind: str
    run_unit: Callable[..., dict[str, Any]]
    prompts: PromptPack
    dir: Path
    labeler_defaults: dict[str, Any]


@dataclass
class MethodContext:
    """Everything a method needs for one unit, provided by the stage."""

    labeler: Labeler
    prompts: PromptPack
    cache_dir: Path  # per-unit (and per-model) response cache
    vlm_frame_height: int
    jpeg_quality: int
    no_cache: bool = False
    params: dict[str, Any] = field(default_factory=dict)


def discover_methods() -> dict[str, Path]:
    """{name -> method dir} for every methods/<name>/annotator.py."""
    out: dict[str, Path] = {}
    if not METHODS_DIR.is_dir():
        return out
    for d in sorted(METHODS_DIR.iterdir()):
        if d.is_dir() and (d / "annotator.py").is_file():
            out[d.name] = d
    return out


def load_method(name: str) -> Method:
    available = discover_methods()
    if name not in available:
        raise KeyError(f"unknown annotation method {name!r} (available: {sorted(available)})")
    module = importlib.import_module(f"realigned_pipeline.annotation.methods.{name}.annotator")
    input_kind = getattr(module, "INPUT_KIND", None)
    if input_kind not in INPUT_KINDS:
        raise ValueError(f"method {name!r} declares INPUT_KIND={input_kind!r}; must be one of {INPUT_KINDS}")
    run_unit = getattr(module, "run_unit", None)
    if not callable(run_unit):
        raise ValueError(f"method {name!r} does not define run_unit(unit, ctx)")
    prompts_path = available[name] / "prompts.yaml"
    if not prompts_path.is_file():
        raise FileNotFoundError(f"method {name!r} has no prompts.yaml at {prompts_path}")
    labeler_defaults = getattr(module, "LABELER_DEFAULTS", {}) or {}
    if not isinstance(labeler_defaults, dict):
        raise ValueError(f"method {name!r}: LABELER_DEFAULTS must be a dict")
    return Method(
        name=name,
        input_kind=input_kind,
        run_unit=run_unit,
        prompts=PromptPack(prompts_path),
        dir=available[name],
        labeler_defaults=labeler_defaults,
    )
