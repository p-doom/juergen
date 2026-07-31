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
    FixtureServerError,
    render_fixture_html,
)


def _ready_event(
    fixture,
    generation: int,
    *,
    client_sequence: int,
    geometry: dict | None = None,
) -> dict:
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
        "client_sequence": client_sequence,
        "client_monotonic_ms": float(client_sequence),
        "fonts_ready": True,
        "geometry": geometry or {"target": {"center_x": 100, "center_y": 100}},
        "value": value,
    }


def _apply_stable_ready(
    store: FixtureStateStore,
    fixture,
    generation: int,
    *,
    geometries: list[dict] | None = None,
) -> int:
    geometry = {"target": {"center_x": 100, "center_y": 100}}
    observations = geometries or [geometry, geometry, geometry]
    for index, observed in enumerate(observations, start=1):
        store.apply_event(
            fixture,
            {
                "kind": "geometry_observation",
                "generation": generation,
                "client_sequence": index,
                "client_monotonic_ms": float(index),
                "animation_frame": index,
                "fonts_ready": True,
                "geometry": observed,
            },
        )
    ready_sequence = len(observations) + 1
    store.apply_event(
        fixture,
        _ready_event(
            fixture,
            generation,
            client_sequence=ready_sequence,
            geometry=observations[-1],
        ),
    )
    return ready_sequence


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
        _apply_stable_ready(store, fixture, generation)
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
        _apply_stable_ready(store, fixture, first_generation)
        first = store.snapshot(fixture.id)
        second_generation = store.reset(fixture)
        _apply_stable_ready(store, fixture, second_generation)
        second = store.snapshot(fixture.id)
        for state in (first, second):
            state.pop("generation")
            for event in state["events"]:
                event.pop("host_monotonic_ns")
        assert first == second


def test_pointer_diagnostics_are_ordered_and_record_exact_hit_coordinates() -> None:
    manifest = load_manifest()
    fixture = next(
        item
        for item in manifest.select(split="development")
        if item.template == "click"
    )
    store = FixtureStateStore(manifest)
    generation = store.reset(fixture)
    ready_sequence = _apply_stable_ready(store, fixture, generation)
    pointer = {
        "kind": "pointer",
        "generation": generation,
        "client_sequence": ready_sequence + 1,
        "client_monotonic_ms": 123.75,
        "event": "pointerdown",
        "button": 0,
        "buttons": 1,
        "client_x": 295,
        "client_y": 313,
        "screen_x": 365,
        "screen_y": 345,
        "hit_id": "target",
        "hit_tag": "input",
    }
    store.apply_event(fixture, pointer)
    event = store.snapshot(fixture.id)["events"][-1]
    assert event["client_sequence"] == ready_sequence + 1
    assert event["client_monotonic_ms"] == 123.75
    assert (event["screen_x"], event["screen_y"]) == (365, 345)
    assert (event["hit_id"], event["hit_tag"]) == ("target", "input")
    assert isinstance(event["host_monotonic_ns"], int)
    with pytest.raises(FixtureServerError, match="non-monotonic"):
        store.apply_event(fixture, pointer)

    html = render_fixture_html(fixture, generation)
    assert "postQueue = postQueue.then(send, send)" in html
    assert "client_sequence" in html
    assert "client_monotonic_ms" in html
    assert "document.elementFromPoint" in html
    assert "screen_x:" in html and "screen_y:" in html


def test_ready_is_rejected_before_fonts_and_exact_geometry_stabilization() -> None:
    manifest = load_manifest()
    fixture = manifest.select(split="development")[0]
    store = FixtureStateStore(manifest)
    generation = store.reset(fixture)
    with pytest.raises(FixtureServerError, match="before exact geometry"):
        store.apply_event(
            fixture,
            _ready_event(fixture, generation, client_sequence=1),
        )
    assert store.snapshot(fixture.id)["ready"] is False

    store = FixtureStateStore(manifest)
    generation = store.reset(fixture)
    geometry = {"target": {"left": 351, "right": 379}}
    for sequence in (1, 2):
        store.apply_event(
            fixture,
            {
                "kind": "geometry_observation",
                "generation": generation,
                "client_sequence": sequence,
                "animation_frame": sequence,
                "fonts_ready": True,
                "geometry": geometry,
            },
        )
    with pytest.raises(FixtureServerError, match="before exact geometry"):
        store.apply_event(
            fixture,
            _ready_event(
                fixture,
                generation,
                client_sequence=3,
                geometry=geometry,
            ),
        )
    assert store.snapshot(fixture.id)["ready"] is False


def test_delayed_layout_requires_three_consecutive_exact_frame_observations() -> None:
    manifest = load_manifest()
    fixture = manifest.select(split="development")[0]
    store = FixtureStateStore(manifest)
    generation = store.reset(fixture)
    early = {
        "window": {"inner_width": 1849, "inner_height": 966},
        "target": {"left": 350, "top": 331, "right": 378, "bottom": 359},
    }
    stable = {
        "window": {"inner_width": 1850, "inner_height": 966},
        "target": {"left": 351, "top": 331, "right": 379, "bottom": 359},
    }
    ready_sequence = _apply_stable_ready(
        store,
        fixture,
        generation,
        geometries=[early, stable, stable, stable],
    )
    state = store.snapshot(fixture.id)
    assert state["ready"] is True
    assert state["geometry"] == stable
    assert state["geometry_stabilization"] == {
        "fonts_ready": True,
        "observation_count": 4,
        "stable_observation_count": 3,
        "first_stable_client_sequence": 2,
        "last_stable_client_sequence": 4,
        "ready_client_sequence": ready_sequence,
        "first_stable_animation_frame": 2,
        "last_stable_animation_frame": 4,
    }
    html = render_fixture_html(fixture, generation)
    assert "await document.fonts.ready" in html
    assert "await new Promise(resolve => requestAnimationFrame(resolve))" in html
    assert "consecutiveIdentical >= 3" in html
    assert "animationFrame <= 120" in html


def test_browser_quiescence_requires_causal_release_and_click_ack() -> None:
    manifest = load_manifest()
    fixture = next(
        item
        for item in manifest.select(split="development")
        if item.template == "click"
    )
    store = FixtureStateStore(manifest)
    generation = store.reset(fixture)
    ready_sequence = _apply_stable_ready(store, fixture, generation)
    store.apply_event(
        fixture,
        {
            "kind": "pointer",
            "generation": generation,
            "client_sequence": ready_sequence + 1,
            "event": "pointerdown",
            "button": 0,
            "buttons": 1,
        },
    )
    store.apply_event(
        fixture,
        {
            "kind": "pointer",
            "generation": generation,
            "client_sequence": ready_sequence + 2,
            "event": "pointerup",
            "button": 0,
            "buttons": 0,
        },
    )
    store.apply_event(
        fixture,
        {
            "kind": "click",
            "generation": generation,
            "client_sequence": ready_sequence + 3,
            "checked": True,
            "decoy_checked": False,
        },
    )
    ack = store.wait_for_browser_quiescence(
        fixture.id,
        after_sequence=ready_sequence,
        required_kinds=("click",),
        require_pointer_up=True,
        expected_pointer_buttons=0,
        timeout_s=0.05,
        quiet_s=0,
    )
    assert ack["last_sequence"] == ready_sequence + 3
    assert ack["pointer_up_acknowledged"] is True
    assert ack["pointer_buttons"] == 0

    # Already-consumed events are stale and cannot acknowledge another action.
    with pytest.raises(TimeoutError, match="acknowledgement timeout"):
        store.wait_for_browser_quiescence(
            fixture.id,
            after_sequence=ready_sequence + 3,
            required_kinds=("click",),
            require_pointer_up=True,
            expected_pointer_buttons=0,
            timeout_s=0.01,
            quiet_s=0,
        )


@pytest.mark.parametrize("include_pointer_up", [False, True])
def test_browser_quiescence_rejects_held_or_unacknowledged_click(
    include_pointer_up: bool,
) -> None:
    manifest = load_manifest()
    fixture = next(
        item
        for item in manifest.select(split="development")
        if item.template == "click"
    )
    store = FixtureStateStore(manifest)
    generation = store.reset(fixture)
    ready_sequence = _apply_stable_ready(store, fixture, generation)
    store.apply_event(
        fixture,
        {
            "kind": "pointer",
            "generation": generation,
            "client_sequence": ready_sequence + 1,
            "event": "pointerdown",
            "button": 0,
            "buttons": 1,
        },
    )
    if include_pointer_up:
        store.apply_event(
            fixture,
            {
                "kind": "pointer",
                "generation": generation,
                "client_sequence": ready_sequence + 2,
                "event": "pointerup",
                "button": 0,
                "buttons": 0,
            },
        )
    with pytest.raises(TimeoutError, match="acknowledgement timeout"):
        store.wait_for_browser_quiescence(
            fixture.id,
            after_sequence=ready_sequence,
            required_kinds=("click",),
            require_pointer_up=True,
            expected_pointer_buttons=0,
            timeout_s=0.01,
            quiet_s=0,
        )


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
