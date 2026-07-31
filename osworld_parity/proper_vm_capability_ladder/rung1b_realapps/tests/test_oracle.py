import os

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.fixtures import load_manifest
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.oracle import (
    evaluate_in_fresh_process,
    evaluate_state,
)
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.states import (
    gold_state,
    near_miss_state,
    reset_state,
)


@pytest.mark.parametrize("fixture", load_manifest().fixtures, ids=lambda f: f.id)
def test_gold_passes_reset_and_near_miss_fail(fixture):
    assert evaluate_state(fixture, gold_state(fixture)).MOUSE_SOLVED is True
    assert evaluate_state(fixture, reset_state(fixture)).MOUSE_SOLVED is False
    assert evaluate_state(fixture, near_miss_state(fixture)).MOUSE_SOLVED is False


@pytest.mark.parametrize("fixture", load_manifest().fixtures, ids=lambda f: f.id)
def test_oracle_runs_in_fresh_process(fixture):
    result = evaluate_in_fresh_process(fixture, gold_state(fixture))
    assert result.MOUSE_SOLVED is True
    assert result.oracle_pid != os.getpid()


def test_hash_and_application_provenance_fail_closed():
    fixture = load_manifest().fixtures[0]
    state = gold_state(fixture)
    state["fixture_sha256"] = "0" * 64
    assert evaluate_state(fixture, state).oracle_status == "error"
    state = gold_state(fixture)
    state["application"] = "browser-form"
    assert evaluate_state(fixture, state).MOUSE_SOLVED is False


def test_exact_unicode_not_normalized_or_transliterated():
    fixture = load_manifest().by_id("r1b-vscode-type-dev-3102")
    assert evaluate_state(fixture, near_miss_state(fixture)).MOUSE_SOLVED is False


def test_files_near_miss_in_decoy_is_rejected():
    fixture = load_manifest().by_id("r1b-files-drag-dev-3301")
    state = near_miss_state(fixture)
    assert state["decoy_sha256"] is not None
    assert evaluate_state(fixture, state).MOUSE_SOLVED is False
