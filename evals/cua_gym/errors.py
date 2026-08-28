"""CUA-Gym dataset-layer exceptions."""

from __future__ import annotations


class CuaGymError(Exception):
    """Base exception for CUA-Gym dataset operations."""


class SnapshotValidationError(CuaGymError):
    """The configured dataset snapshot does not match its pinned manifest."""


class BundleValidationError(CuaGymError):
    """A raw task bundle does not match its catalog metadata."""


class MaterializationError(CuaGymError):
    """Endpoint materialization could not produce a safe episode bundle."""


class RewardParseError(CuaGymError, ValueError):
    """Reward stdout does not contain one valid, unambiguous score."""
