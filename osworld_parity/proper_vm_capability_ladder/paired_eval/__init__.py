"""Development-only paired evaluation for the proper-VM capability ladder.

The package deliberately contains no model server or VM startup code.  A real
runtime is injected only after :func:`consume_executor_ready` has validated and
consumed the executor integration marker.
"""

from .aggregate import aggregate_results
from .curriculum_adapter import load_curriculum_evaluation_manifest
from .manifest import EvaluationManifest, ManifestError, load_evaluation_manifest
from .planning import TrialSpec, build_plan
from .readiness import ConsumedReadiness, ReadinessError, consume_executor_ready
from .runner import PairedEvaluationRunner

__all__ = [
    "ConsumedReadiness",
    "EvaluationManifest",
    "ManifestError",
    "PairedEvaluationRunner",
    "ReadinessError",
    "TrialSpec",
    "aggregate_results",
    "build_plan",
    "consume_executor_ready",
    "load_curriculum_evaluation_manifest",
    "load_evaluation_manifest",
]
