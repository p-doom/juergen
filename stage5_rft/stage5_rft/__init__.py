"""Stage-5 task-level on-policy rollout and RFT infrastructure."""

from stage5_rft.schema import (
    ActionTrace,
    ArtifactRef,
    EpisodeTrace,
    FailureKind,
    PolicyProvenance,
    ResetSpec,
    StateRef,
    StepTrace,
    TaskSpec,
)

__all__ = [
    "ActionTrace",
    "ArtifactRef",
    "EpisodeTrace",
    "FailureKind",
    "PolicyProvenance",
    "ResetSpec",
    "StateRef",
    "StepTrace",
    "TaskSpec",
]

__version__ = "0.1.0"
