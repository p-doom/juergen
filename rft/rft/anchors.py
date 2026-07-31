"""Reference readings every metric is validated against, before it is trusted.

The rule this module enforces: **no metric is believed until it reproduces a
reference whose answer is already known, and that reproduction is an automated
test.** Defect #2 is the cautionary case — a "zero full completions" verdict came
from counting a field that does not exist, and scoring the off-the-shelf model
through the same reader would have shown 0 too, instantly exposing the reader
rather than the model. That check was never run.

So the anchors live in code, with their provenance, and ``tests/test_anchors.py``
drives the real readers over fixtures built from them. An anchor is a *band*, not a
point: greedy decoding is not reproducible in this stack (defect #19), so a gate
that demands an exact figure is a coin flip. :func:`band_for` converts an anchor
into an acceptance band via :mod:`rft.gates`.
"""

from __future__ import annotations

from dataclasses import dataclass

from rft.errors import AnchorMismatch, SchemaError
from rft.gates import ProportionBand, assert_within_band, wilson_band


@dataclass(frozen=True)
class Anchor:
    """One reference reading.

    Attributes:
        name: stable identifier used by tests and reports.
        value: the reading, as a proportion in [0, 1].
        n_scored: denominator the reading was taken over. Needed to build a band.
        n_successes: numerator, when the metric is 0/1-valued.
        n_unscored: entries excluded because they were never scored (NaN). Recorded
            because excluding them is what makes ``value`` correct, and folding them
            in as zeros is defect #3.
        provenance: where the number comes from, in enough detail to re-derive it.
        slack: extra tolerance beyond the Wilson interval, to absorb harness
            non-determinism (defect #19).
    """

    name: str
    value: float
    n_scored: int
    provenance: str
    n_successes: int | None = None
    n_unscored: int = 0
    slack: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise SchemaError(f"anchor {self.name} value {self.value} is not a proportion")
        if self.n_scored < 1:
            raise SchemaError(f"anchor {self.name} has no denominator")
        if self.n_successes is not None and not 0 <= self.n_successes <= self.n_scored:
            raise SchemaError(f"anchor {self.name} successes out of range")

    def band(self) -> ProportionBand:
        successes = (
            self.n_successes
            if self.n_successes is not None
            else round(self.value * self.n_scored)
        )
        return wilson_band(
            successes,
            self.n_scored,
            slack=self.slack,
            reason=f"anchor {self.name} ({self.provenance})",
        )

    def check(self, observed: float) -> None:
        assert_within_band(observed, self.band(), what=f"anchor {self.name}")

    def describe(self) -> str:
        b = self.band()
        return (
            f"{self.name}: {self.value:.4f} over n={self.n_scored} "
            f"(unscored excluded: {self.n_unscored}) band={b.describe()}"
        )


#: THE anchor for closed-loop OSWorld: off-the-shelf Qwen3-VL-8B-Instruct over the
#: 110-task held-out split, minus the 3 gdrive-unscorable tasks => denominator 107.
#:
#: The invariant a reused harness must reproduce is not a percentage but the
#: **reward sum**: SUM(reward) = 33.00378 over 107 written result files, of which 31
#: are >= 0.999, exactly 1 is NaN, and 3 are partial credit
#: (libreoffice_writer 0.9977, vlc 0.8949, multi_apps 0.1111). The often-quoted
#: "30.85% vs 31.14%" is that one run read two ways: 33.00378/107 vs 33.00378/106.
#: Pinning the sum makes the ambiguity impossible to reintroduce.
OFFSHELF_8B_HELDOUT_REWARD_SUM: float = 33.00378
OFFSHELF_8B_HELDOUT_N_WRITTEN: int = 107
OFFSHELF_8B_HELDOUT_N_SOLVED: int = 31
OFFSHELF_8B_HELDOUT_N_NAN: int = 1
#: Partial-credit rewards in that run, kept so a "0/1 only" assumption fails loudly.
OFFSHELF_8B_HELDOUT_PARTIALS: tuple[float, ...] = (0.9977, 0.8949, 0.1111)

OFFSHELF_8B_CLOSED_LOOP = Anchor(
    name="offshelf_8b_closed_loop",
    value=OFFSHELF_8B_HELDOUT_REWARD_SUM / OFFSHELF_8B_HELDOUT_N_WRITTEN,
    n_scored=OFFSHELF_8B_HELDOUT_N_WRITTEN,
    n_successes=OFFSHELF_8B_HELDOUT_N_SOLVED,
    n_unscored=OFFSHELF_8B_HELDOUT_N_NAN,
    provenance=(
        "Qwen3-VL-8B-Instruct, 110-task held-out split minus 3 gdrive-unscorable => "
        "107 written results; SUM(scores.reward)=33.00378 => 30.845% (/107) or 31.136% "
        "(/106). 31 tasks >= 0.999, 1 NaN, 3 partial. Source: "
        "onpolicy_distill/ckpt_evals/baseline_8b_instruct"
    ),
    slack=0.02,
)

#: Off-the-shelf 4B on the FULL 369-task OSWorld-Verified set. This is the
#: harness-validation anchor: it is what established that the local harness
#: reproduces the published reference at all.
#: Recorded reading: overall_success_rate = 0.26847732969374355 over n_done = 361
#: (n_total 369, n_missing 8, n_nan 0) => 26.85% over the scored, or 26.27% if the 8
#: missing count as zero. Published: 26.2%. The two denominators are why this is
#: quoted as "26.0 vs 26.2" - the anchor stores both so nobody has to guess.
OFFSHELF_4B_N_TOTAL: int = 369
OFFSHELF_4B_N_DONE: int = 361
OFFSHELF_4B_N_MISSING: int = 8
OFFSHELF_4B_RATE_OVER_DONE: float = 0.26847732969374355
OFFSHELF_4B_PUBLISHED: float = 0.262

OFFSHELF_4B_OSWORLD = Anchor(
    name="offshelf_4b_osworld",
    value=OFFSHELF_4B_RATE_OVER_DONE * OFFSHELF_4B_N_DONE / OFFSHELF_4B_N_TOTAL,
    n_scored=OFFSHELF_4B_N_TOTAL,
    provenance=(
        "Qwen3-VL-4B local harness: overall_success_rate=0.2684773 over n_done=361 of "
        "n_total=369 (8 missing, 0 NaN) => 26.27% counting missing as 0; published "
        "26.2%. Source: labctl eval_logs/osworld_fullbench_offshelf_qwen3vl4b/score.json"
    ),
    slack=0.015,
)

#: Single-step grounding for the off-the-shelf model in the ABSOLUTE convention,
#: measured by the grounding parity harness on the bbox29 validation anchor
#: (29 targets x 3 cursor-start regimes = 87 instances, k=4 => n=348 per arm).
#: The historical "~1% grounding wall" was this same metric with an
#: absolute-convention model scored through a relative-convention harness.
#: The 90.5% figure is the external reference; this harness reads 0.9713
#: (crosshair render) / 0.9828 (arrow render) for `absolute_native`.
GROUNDING_BBOX29_N_PER_ARM: int = 348
GROUNDING_BBOX29_N_INSTANCES: int = 87

OFFSHELF_MATCHED_SINGLE_STEP = Anchor(
    name="offshelf_matched_single_step_grounding",
    value=0.9253,
    n_scored=GROUNDING_BBOX29_N_PER_ARM,
    provenance=(
        "grounding parity harness, bbox29 crosshair, arm=absolute_matched: 0.9253 "
        "(absolute_native 0.9713; external reference 90.5%). n=348 = 29 targets x 3 "
        "regimes x k=4"
    ),
    slack=0.05,
)

#: Absolute-convention single-step grounding, native prompt. Near-ceiling: absolute
#: pointing is solved, which is why the open problem is relative/closed-loop control.
ABSOLUTE_SINGLE_STEP = Anchor(
    name="absolute_single_step_grounding",
    value=0.9713,
    n_scored=GROUNDING_BBOX29_N_PER_ARM,
    provenance=(
        "grounding parity harness, bbox29 crosshair, arm=absolute_native: 0.9713 "
        "(arrow render 0.9828; train-only 22-cluster read 0.9924)"
    ),
    slack=0.04,
)

#: The relative arms on the SAME instances, same run. These are the numbers that
#: make the absolute-vs-relative gap the finding rather than the harness: an
#: absolute arm at 0.97 and a relative arm at 0.06 in one run cannot both be a
#: harness artifact.
RELATIVE_MOVEREL_SINGLE_STEP = Anchor(
    name="relative_moverel_single_step_grounding",
    value=0.0575,
    n_scored=GROUNDING_BBOX29_N_PER_ARM,
    provenance=(
        "grounding parity harness, bbox29 crosshair, arm=move_rel: 0.0575 "
        "(arrow 0.0374; deltatype_raw 0.0201; deltatype_norm 0.0086) - measured in the "
        "SAME run as absolute_native 0.9713"
    ),
    slack=0.03,
)

#: Mouse-op rate of the bare-line grammar. A detector that matches only
#: computer_use op names scores this 0.0 (defect #5); the true rate is 80.1%.
#: Any mouse-op detector must reproduce this on the bare-line fixture.
BARE_LINE_MOUSE_OP_RATE = Anchor(
    name="bare_line_mouse_op_rate",
    value=0.801,
    n_scored=1000,
    provenance=(
        "bare-line grammar true mouse-op rate 80.1%; a computer_use-only detector "
        "reports 0.0% for the same data (defect #5)"
    ),
    slack=0.03,
)

#: Fraction of ``deltatype`` completions that omit the scroll token. Strict
#: three-token parsing counts these as no-moves, a penalty that lands only on this
#: grammar (defect #14).
DELTATYPE_MISSING_SCROLL_RATE = Anchor(
    name="deltatype_missing_scroll_rate",
    value=0.12,
    n_scored=1000,
    provenance="11-13% of deltatype completions omit the scroll token (defect #14)",
    slack=0.03,
)

#: Self-disagreement of the reference harness on repeated greedy 8-step rollouts.
#: The reason every gate in this package is distributional (defect #19).
GREEDY_SELF_DISAGREEMENT = Anchor(
    name="greedy_self_disagreement_8step",
    value=0.438,
    n_scored=200,
    provenance=(
        "the reference implementation disagrees with ITSELF on 43.8% of greedy 8-step "
        "trajectories; exact-match multi-step gates are therefore meaningless"
    ),
    slack=0.05,
)

#: Staleness rate of ``info.cursor_after`` in the computer_use/move_rel harness.
#: Independently corroborated: the grounding harness's own
#: ``cursor_visibility_evidence.json`` records ``telemetry_staleness.stale_frac``
#: = 0.983, and ``analysis/bucket_partition.py`` carries the same caveat.
CURSOR_AFTER_STALE_RATE = Anchor(
    name="cursor_after_stale_rate",
    value=0.974,
    n_scored=17090,
    provenance=(
        "info.cursor_after == cursor_before on 97.4% of 17,090 computer_use/move_rel "
        "steps (grounding harness telemetry_staleness.stale_frac=0.983 corroborates); "
        "cursor motion must be measured BETWEEN steps (defect #4)"
    ),
    slack=0.02,
)

#: prime-rl's printed ``Reward`` relative to the true unfiltered mean.
PRIMERL_REWARD_BIAS = Anchor(
    name="primerl_reward_bias_factor",
    value=1.0,
    n_scored=1,
    provenance=(
        "prime-rl's per-step Reward is the mean over the POST-zero_advantage batch and "
        "was measured ~2.7x biased high; it structurally cannot show a climb (defect #6)"
    ),
    slack=1.0,
)

#: Every anchor, by name.
ANCHORS: dict[str, Anchor] = {
    a.name: a
    for a in (
        OFFSHELF_8B_CLOSED_LOOP,
        OFFSHELF_4B_OSWORLD,
        OFFSHELF_MATCHED_SINGLE_STEP,
        ABSOLUTE_SINGLE_STEP,
        RELATIVE_MOVEREL_SINGLE_STEP,
        BARE_LINE_MOUSE_OP_RATE,
        DELTATYPE_MISSING_SCROLL_RATE,
        GREEDY_SELF_DISAGREEMENT,
        CURSOR_AFTER_STALE_RATE,
        PRIMERL_REWARD_BIAS,
    )
}

#: The prime-rl reward bias is a ratio, not a proportion, so it is kept separately.
PRIMERL_REWARD_BIAS_FACTOR: float = 2.7


def get_anchor(name: str) -> Anchor:
    try:
        return ANCHORS[name]
    except KeyError:
        raise SchemaError(
            f"unknown anchor {name!r}; known: {sorted(ANCHORS)!r}"
        ) from None


def band_for(name: str) -> ProportionBand:
    return get_anchor(name).band()


def check_anchor(name: str, observed: float) -> None:
    """Validate an observed reading against a named anchor.

    Raises:
        AnchorMismatch: the reading is outside the anchor's band.
    """
    get_anchor(name).check(observed)


def assert_reference_check_ran(
    *, metric_name: str, reference_reading: float | None, anchor_name: str
) -> None:
    """Refuse to report a metric that has not been validated against a reference.

    Call this from any reporting path. ``reference_reading`` is the value the metric
    produced *on the reference input*; ``None`` means the check was skipped, which is
    exactly what happened in defect #2.
    """
    if reference_reading is None:
        raise AnchorMismatch(
            f"metric {metric_name!r} is being reported without ever having been run "
            f"against the {anchor_name!r} reference. Score the known-good reference "
            "through this exact reader first - if the reader is broken it will read "
            "wrong there too, which is far cheaper to discover."
        )
    check_anchor(anchor_name, reference_reading)


def describe_all() -> str:
    return "\n".join(a.describe() for a in ANCHORS.values())
