"""Exception hierarchy for the RFT pipeline.

Design rule: **fail loudly, never silently degrade.** Every module in this
package raises one of these rather than returning a sentinel that a caller
might mistake for a measurement. In particular:

* a *missing* field raises :class:`MissingFieldError` — it never defaults to 0;
* a field of the wrong shape raises :class:`SchemaError`;
* a task that was never scored is :data:`rft.rewards.UNSCORED`, and any attempt
  to fold it into an aggregate raises :class:`UnscoredRewardError` unless the
  caller has explicitly declared how unscored tasks are handled.

Callers that legitimately need to tolerate per-item failures (the sampler, for
instance) must *count and report* them through
:class:`rft.sampling.ErrorLedger`, which aborts once a failure-rate ceiling is
crossed. Swallowing an exception without counting it is a bug, and
``tests/test_no_silent_swallow.py`` greps this package to prove it does not
happen.
"""

from __future__ import annotations


class RftError(Exception):
    """Base class for every error raised by this package."""


class SchemaError(RftError):
    """A payload did not match the schema it was validated against."""


class MissingFieldError(SchemaError):
    """A required field was absent.

    Raised instead of defaulting to a neutral value. Defect #1 (reward lives at
    ``scores.reward``, not top level) and defect #2 (a ``success`` field that is
    absent from 100% of result files) were both silent zeros produced by
    ``payload.get(key, 0)``.
    """

    def __init__(self, path: str, *, available: object = None) -> None:
        msg = f"required field {path!r} is absent"
        if available is not None:
            msg += f"; available keys at that level: {sorted(available)!r}"  # type: ignore[arg-type]
        super().__init__(msg)
        self.path = path


class UnscoredRewardError(RftError):
    """An unscored (NaN) reward reached code that requires a real number.

    Defect #3: ``final_reward`` initialises to NaN and stays NaN when
    ``evaluate()`` throws. NaN means *the task was never scored*, which is not
    the same as scoring 0. Coercing it to 0 turns an instrument failure into a
    reported capability failure.
    """


class PreflightError(RftError):
    """An inference endpoint failed its real chat-completion preflight."""


class FailureRateExceeded(RftError):
    """Per-rollout failure rate crossed the configured ceiling; run aborted."""


class LeakError(RftError):
    """Training records overlapped the held-out evaluation split."""


class RoundTripError(RftError):
    """An action-format conversion did not survive a round trip through the
    exact parser the evaluation harness uses."""


class AnchorMismatch(RftError):
    """A metric disagreed with a reference reading whose answer is known.

    Every metric in this package is validated against such an anchor before it
    is trusted; see :mod:`rft.anchors`.
    """


class RetentionError(RftError):
    """A checkpoint-retention configuration would delete checkpoints that
    later checkpoint selection needs (defect #12)."""


class ValCoverageError(RftError):
    """A validation configuration would score only part of the val split, so
    val numbers are not comparable across runs (defect #11)."""


class DeploymentConfigError(RftError):
    """An inference-deployment config is internally inconsistent (defect #7)."""


class ExportConfigError(RftError):
    """An exported HF checkpoint would not serve as a generative model
    (defect #8)."""
