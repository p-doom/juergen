"""deltatype_v2: compact `dx dy scroll ; ordered elements`, raw pixel deltas."""

from .codec import CODEC, DeltatypeV2Action, DeltatypeV2Codec, DeltatypeV2Error

__all__ = ["CODEC", "DeltatypeV2Action", "DeltatypeV2Codec", "DeltatypeV2Error"]
