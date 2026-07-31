"""Post-gate interfaces for an external official-task broker.

No implementation is shipped here.  In particular, this module cannot resolve
or inspect a held-out filesystem.  A release operator must inject a broker only
through :func:`gates.with_authorized_source` after both signed gates pass.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from .gates import LaunchAuthorization


@dataclass(frozen=True)
class OpaqueTaskLease:
    """A broker-owned task lease whose contents never enter aggregation."""

    cluster_key: str
    pair_seed: int
    reset_ordinal: int
    private_payload: Any


class OfficialPilotBroker(Protocol):
    """Capability interface implemented outside this repository after release."""

    def lease_episode(
        self,
        authorization: LaunchAuthorization,
        *,
        cluster_index: int,
        pair_seed: int,
        arm: str,
        reset_ordinal: int,
    ) -> AbstractContextManager[OpaqueTaskLease]: ...
