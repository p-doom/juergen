"""Stage-03 judgment parameters.

These knobs decide which frames survive, and they are read from two places: the CLI
builds them from `argparse`, and the pool worker reads them off its task dict. Two
assemblies is what let the two disagree, so `filter_params` is the only one — it
normalises and validates, and neither caller may supply a default of its own.

The worker's read goes through `filter_params_from_task`, which only checks that the
task dict carries the whole set before handing it over. It resolves nothing.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pipeline.lib import config
from pipeline.stage_03_filter import (
    FILTER_PARAM_KEYS,
    filter_params,
    filter_params_from_task,
    parse_args,
)


def _args(*extra: str):
    import sys

    argv = [
        "stage_03_filter",
        "--frames-master-dir", "/nonexistent/master",
        "--clips-manifest", "/nonexistent/clips.jsonl",
        "--output-dir", "/nonexistent/out",
        *extra,
    ]
    original, sys.argv = sys.argv, argv
    try:
        return parse_args()
    finally:
        sys.argv = original


def _from_cli(*extra: str) -> dict:
    args = _args(*extra)
    return filter_params(**{key: getattr(args, key) for key in FILTER_PARAM_KEYS})


def _from_task(task: dict) -> dict:
    """The worker's own reader, so these exercise the path a pool takes."""
    return filter_params_from_task(task)


def test_filter_params_declares_no_default_of_its_own() -> None:
    """The one place defaults are applied is the CLI, from `lib.config`.

    A default here would be a second one, and a task dict that omitted the key
    would then be judged differently from an identical CLI invocation.
    """
    signature = inspect.signature(filter_params)
    assert set(signature.parameters) == set(FILTER_PARAM_KEYS)
    for name, parameter in signature.parameters.items():
        assert parameter.default is inspect.Parameter.empty, f"{name} has a default"
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, f"{name} is positional"


def test_a_task_dict_missing_a_knob_fails_loudly() -> None:
    """Silently substituting anything here changes which frames survive.

    And the failure has to name the contract, not just whichever key the reader
    reached first: this aborts a pool, so the message is what the operator gets.
    """
    task = dict.fromkeys(FILTER_PARAM_KEYS, 0)
    task["idle_activity"] = "raw"
    del task["idle_judgment_bin_s"]
    del task["idle_keep_tail_s"]
    with pytest.raises(KeyError) as raised:
        _from_task(task)
    message = str(raised.value)
    assert "idle_judgment_bin_s" in message, "the message names only some of the missing knobs"
    assert "idle_keep_tail_s" in message, "the message names only some of the missing knobs"
    for key in FILTER_PARAM_KEYS:
        assert key in message, f"the message does not state that {key} is required"


def test_both_entry_points_resolve_the_same_parameters() -> None:
    """The CLI path and the task-dict path agree on every knob, defaults included."""
    args = _args()
    task = {key: getattr(args, key) for key in FILTER_PARAM_KEYS}
    assert _from_task(task) == _from_cli()


def test_the_cli_default_activity_is_the_shared_constant() -> None:
    assert _from_cli()["idle_activity"] == config.DEFAULT_IDLE_ACTIVITY
    assert config.DEFAULT_IDLE_ACTIVITY in config.IDLE_ACTIVITIES


@pytest.mark.parametrize("bin_flag", ["0", "0.0"])
def test_a_zero_judgment_bin_normalises_to_none(bin_flag: str) -> None:
    """`0` means "no binning". It must not be recorded as `0.0` in one artifact
    and `None` in another: the per-segment file and the run manifest carry the
    same dict, and a resume compares them."""
    params = _from_cli("--idle-activity", "raw", "--idle-judgment-bin-s", bin_flag)
    assert params["idle_judgment_bin_s"] is None


def test_rounded_without_a_bin_is_refused_before_any_work() -> None:
    """`_rounded_activity_mask` multiplies the bin by the master fps, so `None`
    would surface as a TypeError inside a pool worker, where failures are
    captured and reported as a per-segment status rather than raised."""
    with pytest.raises(ValueError, match="idle_judgment_bin_s"):
        _from_cli("--idle-activity", "rounded", "--idle-judgment-bin-s", "0")


def test_an_unknown_activity_is_refused() -> None:
    args = _args()
    task = {key: getattr(args, key) for key in FILTER_PARAM_KEYS}
    task["idle_activity"] = "sometimes"
    with pytest.raises(ValueError, match="idle_activity"):
        _from_task(task)


def test_the_recorded_set_is_the_judgment_set_plus_only_the_qc_knob() -> None:
    """`qc_view_fps` only affects the diagnostic view. It is recorded in the
    manifest but deliberately excluded from the judgment set, because adding it
    would invalidate the resume cache of every artifact already on disk."""
    params = _from_cli()
    assert "qc_view_fps" not in params
    recorded = {**params, "qc_view_fps": None}
    assert set(recorded) - set(params) == {"qc_view_fps"}


def test_values_are_coerced_so_a_string_argv_compares_equal_to_a_float() -> None:
    """labctl renders every arg as `--key=value`, so a task dict can carry
    strings where a fresh parse carries floats. The resume comparison is `==` on
    the dicts, so the coercion has to happen in one place."""
    args = _args()
    task = {key: getattr(args, key) for key in FILTER_PARAM_KEYS}
    stringly = {**task, "idle_min_duration_s": str(task["idle_min_duration_s"])}
    assert _from_task(stringly) == _from_task(task)


def test_the_module_defines_the_activity_vocabulary_once() -> None:
    """The CLI's `choices` and the validator read the same tuple."""
    source = Path(__file__).resolve().parents[1] / "pipeline" / "stage_03_filter.py"
    text = source.read_text()
    assert 'choices=config.IDLE_ACTIVITIES' in text
    assert '"raw", "rounded"' not in text, "a second activity vocabulary appeared"
