"""The agent layer: what the model sees, how it is sampled, how its text becomes
operations, and how an episode gets a desktop.

Two concerns, one per module:

  * `agent` — one canonical render and sampling path. Sampling settings come from `ModelContext`
    (`ctx.sampling`), never from a per-harness constant, so there is one
    temperature source and the body we log is the body `Dialect.apply_overrides`
    puts on the wire.
  * `desktop` — leased desktop sessions with a bounded lifetime: verifiers'
    worker pool is upscale-only, and a worker that holds a VM never gives it back.
"""

from agent.agent import (
    Agent,
    Decision,
    EffectiveSampling,
    ModelCallError,
    resolve_sampling,
)
from agent.desktop import (
    DesktopLease,
    LeasedDesktopPool,
    LeaseRegistry,
    lease_for_trace,
)

__all__ = [
    "Agent",
    "Decision",
    "DesktopLease",
    "EffectiveSampling",
    "LeaseRegistry",
    "LeasedDesktopPool",
    "ModelCallError",
    "lease_for_trace",
    "resolve_sampling",
]
