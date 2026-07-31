"""ROADMAP 3.5 coarse official-pilot contracts.

This package deliberately contains no official task-source adapter.  Source
access is injected only after the signed ROADMAP 3.1--3.4 and pilot-release
gates have both been verified.
"""

from .contract import (
    ARMS,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    COMPACT_RAW_ARM,
    NATIVE_ABSOLUTE_ARM,
    PAIRED_SEEDS,
    PILOT_TASK_COUNT,
)

__all__ = [
    "ARMS",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "COMPACT_RAW_ARM",
    "NATIVE_ABSOLUTE_ARM",
    "PAIRED_SEEDS",
    "PILOT_TASK_COUNT",
]
