"""The agent layer: what the model sees, how it is sampled, how its text becomes
operations, and how an episode gets a desktop.

Two concerns, one per module:

  * `agent` — one canonical render and sampling path. Sampling settings come from `ModelContext`
    (`ctx.sampling`), never from a per-harness constant, so there is one
    temperature source and the body we log is the body `Dialect.apply_overrides`
    puts on the wire.
  * `desktop` — process-global prewarmed desktop sessions under a node-wide slot
    budget: verifiers' worker pool is upscale-only.
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
)

__all__ = [
    "Agent",
    "Decision",
    "DesktopLease",
    "EffectiveSampling",
    "LeasedDesktopPool",
    "ModelCallError",
    "resolve_sampling",
]
