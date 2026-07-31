"""Schema-validated reward reads with explicit NaN (= unscored) semantics.

This module exists because three separate confidently-wrong conclusions were
produced by careless reward handling:

* **Defect #1** — the reward lives at ``scores.reward``, *not* at the top
  level. Aggregates keyed on a top-level ``"reward"`` silently returned zeros,
  because ``payload.get("reward", 0.0)`` cannot tell "absent" from "zero".
* **Defect #2** — a ``success`` field is absent from 100% of result files. A
  "zero full completions" verdict was counting a key that does not exist.
  The reference check that would have caught it instantly (score the
  off-the-shelf model, which is known to succeed on ~31% of tasks, through the
  same reader) was never run — see :mod:`rft.anchors`.
* **Defect #3** — ``final_reward`` initialises to NaN and stays NaN when
  ``evaluate()`` throws. NaN means *never scored*; coercing it to 0 counts an
  unscored task as a failure.

The API therefore has three properties:

1. Reads are **path-explicit and schema-validated**: :func:`read_reward` is
   told where the reward lives and raises :class:`~rft.errors.MissingFieldError`
   if it is not there. There is no fallback search and no default.
2. Unscored is a **first-class value**, :data:`UNSCORED`, distinct from 0.0.
3. Aggregation **requires an explicit policy** for unscored entries
   (:class:`UnscoredPolicy`). There is no default that quietly picks one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from rft.errors import MissingFieldError, SchemaError, UnscoredRewardError

#: Canonical location of the scalar reward inside an OSWorld result payload.
#: Discovered the hard way (defect #1): it is nested under ``scores``.
CANONICAL_REWARD_PATH: Final[str] = "scores.reward"

#: Sentinel for "this task was never scored". Deliberately *not* a float, so
#: that arithmetic on it raises a TypeError rather than propagating NaN or, far
#: worse, being coerced to 0.0.
class _Unscored:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "UNSCORED"

    def __bool__(self) -> bool:
        raise UnscoredRewardError(
            "truth-testing an UNSCORED reward: an unscored task is neither a "
            "success nor a failure; handle it explicitly"
        )


UNSCORED: Final[_Unscored] = _Unscored()

Reward = float | _Unscored


class UnscoredPolicy(Enum):
    """How an aggregate treats tasks that were never scored.

    There is no default. The caller must say which of these it means, because
    the three give materially different numbers and the difference is exactly
    what defect #3 hid.
    """

    #: Refuse to aggregate at all if any entry is unscored (strictest; use for
    #: gates and regression tests).
    RAISE = "raise"
    #: Drop unscored entries and report how many were dropped. The denominator
    #: shrinks. This is what the off-shelf anchor readings did (31 tasks at
    #: exactly 1.0 out of 100 scored, one NaN excluded).
    EXCLUDE = "exclude"
    #: Count unscored entries as failures. Almost always wrong; available only
    #: so that a caller which really means it has to name it.
    COUNT_AS_ZERO = "count_as_zero"


def _walk(payload: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted ``path`` in ``payload``, raising on any missing level."""
    node: Any = payload
    walked: list[str] = []
    for part in path.split("."):
        if not isinstance(node, Mapping):
            raise SchemaError(
                f"cannot descend into {'.'.join(walked) or '<root>'} while "
                f"resolving {path!r}: expected a mapping, got {type(node).__name__}"
            )
        if part not in node:
            raise MissingFieldError(
                ".".join([*walked, part]), available=list(node.keys())
            )
        node = node[part]
        walked.append(part)
    return node


def read_reward(
    payload: Mapping[str, Any], *, path: str = CANONICAL_REWARD_PATH
) -> Reward:
    """Read the scalar reward out of one result payload.

    Args:
        payload: a parsed result JSON document.
        path: dotted path to the reward. Defaults to the canonical
            ``scores.reward``. Pass an explicit path when reading a foreign
            schema — but never a *list* of candidate paths: a reader that
            searches several locations cannot distinguish "found the right
            field" from "found a differently-meaning field with the same name".

    Returns:
        A float, or :data:`UNSCORED` if the stored value is NaN or null.

    Raises:
        MissingFieldError: the path does not exist. This is the defect-#1 guard:
            reading a top-level ``"reward"`` from a payload that nests it raises
            instead of yielding 0.0.
        SchemaError: the value exists but is not a number or null.
    """
    value = _walk(payload, path)
    if value is None:
        return UNSCORED
    if isinstance(value, bool):
        # bools are ints in Python; a bool reward is a schema smell (someone
        # stored `success` where a reward belongs).
        raise SchemaError(f"{path} is a bool ({value!r}); expected a number")
    if not isinstance(value, (int, float)):
        raise SchemaError(f"{path} is {type(value).__name__} ({value!r}); expected a number")
    fvalue = float(value)
    if math.isnan(fvalue):
        return UNSCORED
    if math.isinf(fvalue):
        raise SchemaError(f"{path} is infinite ({fvalue!r})")
    return fvalue


def require_field(payload: Mapping[str, Any], path: str) -> Any:
    """Read ``path`` or raise. Use for any field a verdict depends on.

    Defect #2 in one line: if your verdict is "zero tasks reported success",
    the field named ``success`` had better exist. ``require_field`` makes its
    absence an error instead of a finding.
    """
    return _walk(payload, path)


def has_field(payload: Mapping[str, Any], path: str) -> bool:
    """Whether ``path`` resolves. Use to *report* schema coverage, never to
    silently branch to a default."""
    try:
        _walk(payload, path)
    except (MissingFieldError, SchemaError):
        return False
    return True


@dataclass(frozen=True)
class RewardAggregate:
    """Result of aggregating rewards, with the unscored bookkeeping exposed.

    ``n_total`` is every entry seen; ``n_scored`` is the ones with a real
    number; ``n_unscored`` is the difference. ``mean`` is over the denominator
    implied by the chosen :class:`UnscoredPolicy`, and ``denominator`` records
    which it was so a reader never has to guess. ``n_at_one`` is populated when
    the scored rewards are 0/1-valued, because the historical anchor readings
    are quoted that way ("31 tasks at exactly 1.0, one NaN to exclude").
    """

    mean: float
    n_total: int
    n_scored: int
    n_unscored: int
    denominator: int
    policy: UnscoredPolicy
    n_at_one: int | None = None


def aggregate_rewards(
    rewards: Iterable[Reward], *, unscored: UnscoredPolicy
) -> RewardAggregate:
    """Mean reward with explicit unscored handling.

    Args:
        rewards: values from :func:`read_reward`.
        unscored: mandatory policy for :data:`UNSCORED` entries. There is no
            default; see :class:`UnscoredPolicy`.

    Raises:
        UnscoredRewardError: ``unscored`` is :attr:`UnscoredPolicy.RAISE` and at
            least one entry is unscored, or the effective denominator is zero.
    """
    if not isinstance(unscored, UnscoredPolicy):
        raise TypeError("`unscored` must be an UnscoredPolicy; pick one explicitly")
    values = list(rewards)
    scored = [v for v in values if not isinstance(v, _Unscored)]
    n_unscored = len(values) - len(scored)

    if n_unscored and unscored is UnscoredPolicy.RAISE:
        raise UnscoredRewardError(
            f"{n_unscored}/{len(values)} entries were never scored; an unscored "
            "task is not a zero. Re-run them, or pass "
            "UnscoredPolicy.EXCLUDE / COUNT_AS_ZERO to say what you mean."
        )

    if unscored is UnscoredPolicy.COUNT_AS_ZERO:
        pool: Sequence[float] = [*scored, *([0.0] * n_unscored)]
    else:
        pool = scored

    if not pool:
        raise UnscoredRewardError(
            f"no scored entries among {len(values)}; refusing to report a mean "
            "of an empty set (a 0/0 probe that writes 0.0 is defect #9)"
        )

    n_at_one = None
    if all(v in (0.0, 1.0) for v in scored):
        n_at_one = sum(1 for v in scored if v == 1.0)

    return RewardAggregate(
        mean=sum(pool) / len(pool),
        n_total=len(values),
        n_scored=len(scored),
        n_unscored=n_unscored,
        denominator=len(pool),
        policy=unscored,
        n_at_one=n_at_one,
    )
