"""Pre-gate ROADMAP 3.3 mixed-action VM training/evaluation infrastructure."""

from .manifest import (
    DEVELOPMENT_MANIFEST,
    SEALED_EVALUATION_MANIFEST,
    TRAIN_MANIFEST,
    ManifestError,
    TaskDefinition,
    load_manifest,
    materialize_tasks,
)
from .runtime import Episode, StepResult

__all__ = [
    "DEVELOPMENT_MANIFEST",
    "SEALED_EVALUATION_MANIFEST",
    "TRAIN_MANIFEST",
    "Episode",
    "ManifestError",
    "StepResult",
    "TaskDefinition",
    "load_manifest",
    "materialize_tasks",
]
