from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import load_manifest
from osworld_parity.proper_vm_capability_ladder.rung1.oracle import (
    evaluate_in_fresh_process,
    evaluate_state,
)
from osworld_parity.proper_vm_capability_ladder.rung1.server import (
    FixtureHttpServer,
    FixtureStateStore,
)


def _ready_event(fixture, generation: int) -> dict:
    if fixture.template == "click":
        value = False
    elif fixture.template == "focus_type":
        value = fixture.params["initial_text"]
    elif fixture.template == "scroll":
        value = fixture.params["initial_y"]
    else:
        value = fixture.params["initial_value"]
    return {
        "kind": "ready",
        "generation": generation,
        "geometry": {"target": {"center_x": 100, "center_y": 100}},
        "value": value,
    }


def test_oracle_process_is_fresh_and_exact_unicode() -> None:
    fixture = next(
        item
        for item in load_manifest().select(split="development")
        if item.template == "focus_type"
    )
    state = {
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "ready": True,
        "current": {"text": fixture.expected["text"]},
    }
    result = evaluate_in_fresh_process(fixture, state)
    assert result.MOUSE_SOLVED is True
    assert result.oracle_pid != os.getpid()
    state["current"]["text"] += "x"
    assert evaluate_in_fresh_process(fixture, state).MOUSE_SOLVED is False


def test_store_reset_removes_all_state_classes_and_button_leakage() -> None:
    manifest = load_manifest()
    store = FixtureStateStore(manifest)
    for fixture in manifest.select(split="development"):
        generation = store.reset(fixture)
        store.apply_event(fixture, _ready_event(fixture, generation))
        store.apply_event(
            fixture,
            {
                "kind": "pointer",
                "generation": generation,
                "event": "pointerdown",
                "button": 0,
                "buttons": 1,
            },
        )
        if fixture.template == "click":
            mutation = {"kind": "click", "checked": True, "decoy_checked": True}
        elif fixture.template == "focus_type":
            mutation = {"kind": "text", "text": "leaked typing"}
        elif fixture.template == "scroll":
            mutation = {"kind": "scroll", "scroll_y": 3999}
        else:
            mutation = {"kind": "drag", "value": 99}
        mutation["generation"] = generation
        store.apply_event(fixture, mutation)
        assert store.snapshot(fixture.id)["last_pointer_buttons"] == 1
        store.reset(fixture)
        clean = store.snapshot(fixture.id)
        assert clean["last_pointer_buttons"] == 0
        assert clean["ready"] is False
        assert clean["events"] == []
        assert evaluate_state(fixture, {**clean, "ready": True}).MOUSE_SOLVED is False


def test_two_consecutive_fixture_resets_are_equivalent() -> None:
    manifest = load_manifest()
    store = FixtureStateStore(manifest)
    for fixture in manifest.select(split="development"):
        first_generation = store.reset(fixture)
        store.apply_event(fixture, _ready_event(fixture, first_generation))
        first = store.snapshot(fixture.id)
        second_generation = store.reset(fixture)
        store.apply_event(fixture, _ready_event(fixture, second_generation))
        second = store.snapshot(fixture.id)
        for state in (first, second):
            state.pop("generation")
        assert first == second


def test_http_surface_exposes_fixture_but_no_oracle_state() -> None:
    manifest = load_manifest()
    fixture = manifest.select(split="development")[0]
    with FixtureHttpServer(manifest) as server:
        base = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(base + f"/fixture/{fixture.id}") as response:
            body = response.read().decode("utf-8")
        assert fixture.instruction in body
        assert fixture.fixture_sha256 not in body
        for forbidden in ("/state", "/oracle", f"/event/{fixture.id}"):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + forbidden)
            assert caught.value.code == 404
