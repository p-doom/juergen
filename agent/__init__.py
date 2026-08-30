"""The agent layer: what the model sees, how it is sampled, how its text becomes
operations, and how an episode gets a desktop.

Three concerns, one per module, all injected rather than inherited:

  * `history` — the history policy: freeroll's interleaved window, Phase-B's
    prose-summarised five-image window, target_box's latest-image-only
    accumulation, movebox's stateless single turn. Selecting one is a constructor
    argument.
  * `agent` — one sampling path. Sampling settings come from `ModelContext`
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
    "LeasedDesktopPool",
    "ModelCallError",
    "ProseSummarisedWindow",
    "StatelessSingleTurn",
    "Turn",
    "history_policy",
    "resolve_sampling",
]
