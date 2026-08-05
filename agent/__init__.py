"""The agent layer: what the model sees, how it is sampled, how its text becomes
operations, and how an episode gets a desktop.

Three concerns, one per module, all injected rather than inherited:

  * `history` — the history policy. Previously an emergent property of whichever
    episode driver you forked (freeroll's interleaved window, Phase-B's
    prose-summarised five-image window, target_box's latest-image-only
    accumulation, movebox's stateless single turn). Naming it turns an A/B into a
    constructor argument.
  * `agent` — one sampling path. Sampling settings come from `ModelContext`
    (`ctx.sampling`), never from a per-harness constant, so there is exactly one
    temperature source and the body we log is the body `Dialect.apply_overrides`
    puts on the wire.
  * `desktop` — leased desktop sessions with a bounded lifetime, because
    verifiers' worker pool is upscale-only and a worker that holds a VM never
    gives it back.
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
    LeaseRegistry,
    LeasedDesktopPool,
    lease_for_trace,
)
from agent.history import (
    History,
    HistoryPolicy,
    ImageBudget,
    InterleavedFrames,
    LatestImageOnly,
    ProseSummarisedWindow,
    StatelessSingleTurn,
    Turn,
    history_policy,
)

__all__ = [
    "Agent",
    "Decision",
    "DesktopLease",
    "EffectiveSampling",
    "History",
    "HistoryPolicy",
    "ImageBudget",
    "InterleavedFrames",
    "LatestImageOnly",
    "LeaseRegistry",
    "LeasedDesktopPool",
    "ModelCallError",
    "ProseSummarisedWindow",
    "StatelessSingleTurn",
    "Turn",
    "history_policy",
    "lease_for_trace",
    "resolve_sampling",
]
