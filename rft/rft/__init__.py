"""``rft`` — production RFT (rejection fine-tuning) infrastructure.

Five stages, each an importable module with a labctl recipe:

==== ================== ==========================================================
1    :mod:`rft.sampling`   draw k completions per task from a served checkpoint
2    :mod:`rft.scoring`    accept-reject predicate over schema-validated rewards
3    :mod:`rft.records`    action-format conversion + round-trip + leak check
4    :mod:`rft.training`   validated omegalax invocation (omegalax itself untouched)
5    :mod:`rft.evaluation` the validated parity harness + bucketed reporting
==== ================== ==========================================================

Cross-cutting modules: :mod:`rft.rewards` (reward semantics), :mod:`rft.grammars`
(action formats), :mod:`rft.diagnostics` (the three mandatory numbers),
:mod:`rft.anchors` (reference readings every metric is validated against),
:mod:`rft.gates` (distributional gates), :mod:`rft.cursor`, :mod:`rft.session`,
:mod:`rft.serving`, :mod:`rft.splits`, :mod:`rft.primerl_metrics`,
:mod:`rft.wandb_offline`.

Two rules the whole package is built around:

1. **Fail loudly, never silently degrade.** A missing field raises; an unscored
   task is not a zero; an empty sample has no mean.
2. **No metric is trusted until it reproduces a known reference reading**, and
   that reproduction is an automated test (:mod:`rft.anchors`, ``tests/test_anchors.py``).
"""

from __future__ import annotations

__all__ = [
    "anchors",
    "cursor",
    "diagnostics",
    "errors",
    "evaluation",
    "gates",
    "grammars",
    "primerl_metrics",
    "records",
    "rewards",
    "roundtrip",
    "sampling",
    "scoring",
    "serving",
    "session",
    "splits",
    "training",
    "wandb_offline",
]

__version__ = "0.1.0"
