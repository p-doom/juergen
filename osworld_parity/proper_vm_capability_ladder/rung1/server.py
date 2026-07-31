from __future__ import annotations

import copy
import html
import json
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .fixtures import Fixture, FixtureManifest


class FixtureServerError(RuntimeError):
    pass


STABLE_GEOMETRY_OBSERVATIONS = 3
MAX_GEOMETRY_ANIMATION_FRAMES = 120
HOST_DIAGNOSTIC_JOURNAL_LIMIT = 1024
BROWSER_AUDIT_EVENT_LIMIT = 512
BROWSER_AUDIT_SCHEMA_VERSION = 2
BROWSER_AUDIT_EVENTS = {
    "audit_heartbeat",
    "audit_ready",
    "pointerdown",
    "pointerup",
    "pointermove",
    "mousedown",
    "mouseup",
    "mousemove",
    "click",
    "input",
    "change",
    "focus",
    "blur",
}


def _initial_current(fixture: Fixture) -> dict[str, Any]:
    if fixture.template == "click":
        return {"checked": False, "decoy_checked": False}
    if fixture.template == "focus_type":
        return {"text": fixture.params["initial_text"]}
    if fixture.template == "scroll":
        return {"scroll_y": int(fixture.params["initial_y"])}
    if fixture.template == "drag":
        return {"value": int(fixture.params["initial_value"])}
    raise FixtureServerError(f"unknown template {fixture.template!r}")


class FixtureStateStore:
    """Host-only oracle state. No HTTP read route is registered for this store."""

    def __init__(self, manifest: FixtureManifest) -> None:
        self._manifest = manifest
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._states: dict[str, dict[str, Any]] = {}
        for fixture in manifest.fixtures:
            self.reset(fixture)

    def reset(self, fixture: Fixture) -> int:
        with self._condition:
            generation = int(self._states.get(fixture.id, {}).get("generation", 0)) + 1
            self._states[fixture.id] = {
                "fixture_id": fixture.id,
                "fixture_sha256": fixture.fixture_sha256,
                "generation": generation,
                "ready": False,
                "geometry": {},
                "current": _initial_current(fixture),
                "events": [],
                "last_pointer_buttons": 0,
                "last_client_sequence": 0,
                "geometry_observations": [],
                "geometry_stabilization": None,
                "geometry_stabilization_error": None,
                "diagnostic_journal": [],
                "diagnostic_journal_dropped": 0,
                "diagnostic_journal_next_sequence": 1,
                "diagnostic_http_request_next_id": 1,
                "browser_audit_events": [],
                "browser_audit_dropped": 0,
                "browser_audit_request_next_id": 1,
            }
            self._condition.notify_all()
            return generation

    def apply_browser_audit(
        self, fixture: Fixture, payload: dict[str, Any]
    ) -> dict[str, int]:
        """Persist the arm-neutral DOM listener trace outside ``postQueue``.

        Audit reports use their own HTTP path, sequence, request identity, and
        bounded store.  They therefore remain available when the serialized
        semantic reporter or CDP diagnostic connection fails.  Arrival order
        is not treated as browser order: ``sendBeacon`` requests may be served
        concurrently, so the browser sequence and timestamps are preserved.
        """
        with self._condition:
            state = self._states[fixture.id]
            if (
                type(payload.get("schema_version")) is not int
                or payload["schema_version"] != BROWSER_AUDIT_SCHEMA_VERSION
            ):
                raise FixtureServerError("browser audit schema mismatch")
            if (
                type(payload.get("generation")) is not int
                or payload["generation"] != state["generation"]
            ):
                raise FixtureServerError("stale browser audit generation")
            event_name = payload.get("event")
            if event_name not in BROWSER_AUDIT_EVENTS:
                raise FixtureServerError("unsupported browser audit event")
            audit_sequence = payload.get("audit_sequence")
            if (
                not isinstance(audit_sequence, int)
                or isinstance(audit_sequence, bool)
                or audit_sequence < 1
            ):
                raise FixtureServerError("invalid browser audit sequence")
            request_id = int(state["browser_audit_request_next_id"])
            state["browser_audit_request_next_id"] = request_id + 1
            event = copy.deepcopy(payload)
            event.update(
                {
                    "host_audit_request_id": request_id,
                    "host_monotonic_ns": time.monotonic_ns(),
                    "host_wall_time_ns": time.time_ns(),
                }
            )
            if len(state["browser_audit_events"]) >= BROWSER_AUDIT_EVENT_LIMIT:
                state["browser_audit_dropped"] += 1
            else:
                state["browser_audit_events"].append(event)
            self._condition.notify_all()
            return {
                "audit_sequence": int(event["audit_sequence"]),
                "host_audit_request_id": request_id,
                "host_monotonic_ns": int(event["host_monotonic_ns"]),
            }

    @staticmethod
    def _causal_heartbeat_summaries(
        state: dict[str, Any], observation_deadline_host_monotonic_ns: int
    ) -> list[dict[str, Any]]:
        summaries = []
        for event in state["browser_audit_events"]:
            if event.get("event") != "audit_heartbeat":
                continue
            host_ns = event.get("host_monotonic_ns")
            acknowledged_host_ns = event.get("acknowledged_host_monotonic_ns")
            summaries.append(
                {
                    "audit_sequence": event.get("audit_sequence"),
                    "client_monotonic_ms": event.get("client_monotonic_ms"),
                    "host_monotonic_ns": host_ns,
                    "host_arrival_lag_ns": (
                        host_ns - observation_deadline_host_monotonic_ns
                        if isinstance(host_ns, int)
                        and not isinstance(host_ns, bool)
                        else None
                    ),
                    "acknowledged_heartbeat_audit_sequence": event.get(
                        "acknowledged_heartbeat_audit_sequence"
                    ),
                    "acknowledged_host_audit_request_id": event.get(
                        "acknowledged_host_audit_request_id"
                    ),
                    "acknowledged_host_monotonic_ns": acknowledged_host_ns,
                    "acknowledged_receipt_lag_ns": (
                        acknowledged_host_ns
                        - observation_deadline_host_monotonic_ns
                        if isinstance(acknowledged_host_ns, int)
                        and not isinstance(acknowledged_host_ns, bool)
                        else None
                    ),
                }
            )
        return sorted(
            summaries,
            key=lambda item: (
                item["audit_sequence"]
                if isinstance(item["audit_sequence"], int)
                else -1
            ),
        )

    @staticmethod
    def _causal_post_window_marker_sequences(
        state: dict[str, Any], observation_deadline_host_monotonic_ns: int
    ) -> list[int]:
        events = state["browser_audit_events"]
        by_sequence: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            sequence = event.get("audit_sequence")
            if isinstance(sequence, int) and not isinstance(sequence, bool):
                by_sequence.setdefault(sequence, []).append(event)
        candidates = []
        for event in events:
            sequence = event.get("audit_sequence")
            acknowledged_sequence = event.get(
                "acknowledged_heartbeat_audit_sequence"
            )
            acknowledged_request_id = event.get(
                "acknowledged_host_audit_request_id"
            )
            acknowledged_host_ns = event.get("acknowledged_host_monotonic_ns")
            if (
                event.get("event") != "audit_heartbeat"
                or not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or not isinstance(acknowledged_sequence, int)
                or isinstance(acknowledged_sequence, bool)
                or not isinstance(acknowledged_request_id, int)
                or isinstance(acknowledged_request_id, bool)
                or not isinstance(acknowledged_host_ns, int)
                or isinstance(acknowledged_host_ns, bool)
                or acknowledged_host_ns
                <= observation_deadline_host_monotonic_ns
                or any(
                    len(by_sequence.get(index, [])) != 1
                    for index in range(1, sequence + 1)
                )
            ):
                continue
            acknowledged = by_sequence.get(acknowledged_sequence, [])
            if (
                acknowledged_sequence >= sequence
                or len(acknowledged) != 1
                or acknowledged[0].get("event") != "audit_heartbeat"
                or acknowledged[0].get("host_audit_request_id")
                != acknowledged_request_id
                or acknowledged[0].get("host_monotonic_ns")
                != acknowledged_host_ns
            ):
                continue
            candidates.append(sequence)
        return sorted(candidates)

    def wait_for_causal_post_window_heartbeat(
        self,
        fixture_id: str,
        *,
        generation: int,
        observation_deadline_host_monotonic_ns: int,
        timeout_s: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Wait for a contiguous marker causally generated after the deadline."""
        wait_started_ns = time.monotonic_ns()
        wait_deadline_ns = wait_started_ns + int(timeout_s * 1_000_000_000)
        with self._condition:
            state = self._states[fixture_id]
            self._append_diagnostic_locked(
                state,
                "causal_audit_wait_started",
                {
                    "generation": generation,
                    "observation_deadline_host_monotonic_ns": (
                        observation_deadline_host_monotonic_ns
                    ),
                    "wait_started_host_monotonic_ns": wait_started_ns,
                    "wait_timeout_s": timeout_s,
                    "wait_deadline_host_monotonic_ns": wait_deadline_ns,
                },
            )
            candidate_sequences: list[int] = []
            while True:
                state = self._states[fixture_id]
                if state["generation"] != generation:
                    raise FixtureServerError(
                        f"{fixture_id}: fixture reset while awaiting causal audit heartbeat"
                    )
                candidate_sequences = self._causal_post_window_marker_sequences(
                    state, observation_deadline_host_monotonic_ns
                )
                now_ns = time.monotonic_ns()
                if candidate_sequences or now_ns >= wait_deadline_ns:
                    break
                self._condition.wait(
                    timeout=max(0.001, min(0.05, (wait_deadline_ns - now_ns) / 1e9))
                )
            wait_completed_ns = time.monotonic_ns()
            timed_out = not candidate_sequences
            heartbeat_summaries = self._causal_heartbeat_summaries(
                state, observation_deadline_host_monotonic_ns
            )
            self._append_diagnostic_locked(
                state,
                "causal_audit_wait_completed",
                {
                    "generation": generation,
                    "timed_out": timed_out,
                    "candidate_audit_sequences": candidate_sequences,
                    "wait_completed_host_monotonic_ns": wait_completed_ns,
                    "wait_duration_ns": wait_completed_ns - wait_started_ns,
                    "heartbeat_summaries": heartbeat_summaries,
                },
            )
            snapshot = copy.deepcopy(state)
        return snapshot, {
            "schema_version": 1,
            "generation": generation,
            "observation_deadline_host_monotonic_ns": (
                observation_deadline_host_monotonic_ns
            ),
            "wait_started_host_monotonic_ns": wait_started_ns,
            "wait_completed_host_monotonic_ns": wait_completed_ns,
            "wait_duration_ns": wait_completed_ns - wait_started_ns,
            "wait_timeout_s": timeout_s,
            "wait_deadline_host_monotonic_ns": wait_deadline_ns,
            "timed_out": timed_out,
            "candidate_audit_sequences": candidate_sequences,
            "heartbeat_summaries": heartbeat_summaries,
        }

    @staticmethod
    def _append_diagnostic_locked(
        state: dict[str, Any], stage: str, details: dict[str, Any]
    ) -> None:
        if len(state["diagnostic_journal"]) >= HOST_DIAGNOSTIC_JOURNAL_LIMIT:
            state["diagnostic_journal_dropped"] += 1
            return
        sequence = int(state["diagnostic_journal_next_sequence"])
        state["diagnostic_journal_next_sequence"] = sequence + 1
        state["diagnostic_journal"].append(
            {
                "journal_sequence": sequence,
                "stage": stage,
                "host_monotonic_ns": time.monotonic_ns(),
                "host_wall_time_ns": time.time_ns(),
                "details": copy.deepcopy(details),
            }
        )

    def record_diagnostic(
        self, fixture: Fixture, stage: str, details: dict[str, Any]
    ) -> None:
        with self._condition:
            state = self._states[fixture.id]
            self._append_diagnostic_locked(state, stage, details)
            self._condition.notify_all()

    def begin_http_request(
        self, fixture: Fixture, details: dict[str, Any]
    ) -> int:
        with self._condition:
            state = self._states[fixture.id]
            request_id = int(state["diagnostic_http_request_next_id"])
            state["diagnostic_http_request_next_id"] = request_id + 1
            self._append_diagnostic_locked(
                state,
                "http_ingress",
                {**details, "host_request_id": request_id},
            )
            self._condition.notify_all()
            return request_id

    def apply_event(
        self,
        fixture: Fixture,
        payload: dict[str, Any],
        *,
        host_request_id: int | None = None,
    ) -> None:
        with self._condition:
            state = self._states[fixture.id]
            self._append_diagnostic_locked(
                state,
                "store_apply_started",
                {
                    "kind": payload.get("kind"),
                    "client_sequence": payload.get("client_sequence"),
                    "generation": payload.get("generation"),
                    "host_request_id": host_request_id,
                },
            )
            if payload.get("generation") != state["generation"]:
                raise FixtureServerError("stale fixture generation")
            kind = payload.get("kind")
            if not isinstance(kind, str):
                raise FixtureServerError("event kind missing")
            client_sequence = int(payload.get("client_sequence", -1))
            if client_sequence >= 0:
                if client_sequence <= state["last_client_sequence"]:
                    raise FixtureServerError(
                        f"non-monotonic client event sequence {client_sequence}"
                    )
                state["last_client_sequence"] = client_sequence
            event: dict[str, Any] = {
                "kind": kind,
                "client_sequence": client_sequence,
                "host_request_id": host_request_id,
                "client_monotonic_ms": float(
                    payload.get("client_monotonic_ms", -1.0)
                ),
                "host_monotonic_ns": time.monotonic_ns(),
            }
            if kind == "ready":
                geometry = payload.get("geometry")
                if not isinstance(geometry, dict):
                    raise FixtureServerError("ready geometry missing")
                observations = state["geometry_observations"]
                required = observations[-STABLE_GEOMETRY_OBSERVATIONS:]
                if (
                    payload.get("fonts_ready") is not True
                    or len(required) != STABLE_GEOMETRY_OBSERVATIONS
                    or any(item["fonts_ready"] is not True for item in required)
                    or any(item["geometry"] != geometry for item in required)
                    or any(
                        required[index]["animation_frame"] + 1
                        != required[index + 1]["animation_frame"]
                        for index in range(len(required) - 1)
                    )
                    or any(
                        required[index]["client_sequence"]
                        >= required[index + 1]["client_sequence"]
                        for index in range(len(required) - 1)
                    )
                    or required[-1]["client_sequence"] >= client_sequence
                ):
                    raise FixtureServerError(
                        "ready rejected before exact geometry stabilization"
                    )
                state["geometry"] = copy.deepcopy(geometry)
                state["ready"] = True
                state["geometry_stabilization"] = {
                    "fonts_ready": True,
                    "observation_count": len(observations),
                    "stable_observation_count": STABLE_GEOMETRY_OBSERVATIONS,
                    "first_stable_client_sequence": required[0]["client_sequence"],
                    "last_stable_client_sequence": required[-1]["client_sequence"],
                    "ready_client_sequence": client_sequence,
                    "first_stable_animation_frame": required[0]["animation_frame"],
                    "last_stable_animation_frame": required[-1]["animation_frame"],
                }
                browser_value = payload.get("value")
                if fixture.template == "click":
                    state["current"]["checked"] = bool(browser_value)
                elif fixture.template == "focus_type":
                    state["current"]["text"] = str(browser_value)
                elif fixture.template == "scroll":
                    state["current"]["scroll_y"] = int(browser_value)
                elif fixture.template == "drag":
                    state["current"]["value"] = int(browser_value)
                event["geometry_stabilization"] = copy.deepcopy(
                    state["geometry_stabilization"]
                )
            elif kind == "geometry_observation":
                geometry = payload.get("geometry")
                if not isinstance(geometry, dict):
                    raise FixtureServerError("geometry observation missing geometry")
                observation = {
                    "client_sequence": client_sequence,
                    "host_request_id": host_request_id,
                    "client_monotonic_ms": event["client_monotonic_ms"],
                    "animation_frame": int(payload.get("animation_frame", -1)),
                    "fonts_ready": payload.get("fonts_ready") is True,
                    "geometry": copy.deepcopy(geometry),
                }
                if observation["animation_frame"] < 1 or not observation["fonts_ready"]:
                    raise FixtureServerError("invalid geometry observation")
                state["geometry_observations"].append(observation)
                event.update(copy.deepcopy(observation))
            elif kind == "geometry_failure":
                state["geometry_stabilization_error"] = str(
                    payload.get("error", "geometry did not stabilize")
                )
                event["error"] = state["geometry_stabilization_error"]
            elif kind == "pointer":
                buttons = int(payload.get("buttons", 0))
                state["last_pointer_buttons"] = buttons
                event.update(
                    {
                        "event": str(payload.get("event", "")),
                        "button": int(payload.get("button", -1)),
                        "buttons": buttons,
                        "client_x": int(payload.get("client_x", -1)),
                        "client_y": int(payload.get("client_y", -1)),
                        "screen_x": int(payload.get("screen_x", -1)),
                        "screen_y": int(payload.get("screen_y", -1)),
                        "hit_id": str(payload.get("hit_id", "")),
                        "hit_tag": str(payload.get("hit_tag", "")),
                    }
                )
            elif kind == "click":
                state["current"]["checked"] = bool(payload.get("checked"))
                state["current"]["decoy_checked"] = bool(payload.get("decoy_checked"))
            elif kind == "text":
                state["current"]["text"] = str(payload.get("text", ""))
            elif kind == "scroll":
                state["current"]["scroll_y"] = int(payload.get("scroll_y", 0))
            elif kind == "drag":
                state["current"]["value"] = int(payload.get("value", -1))
            else:
                raise FixtureServerError(f"unsupported browser event {kind!r}")
            state["events"].append(event)
            self._append_diagnostic_locked(
                state,
                "store_apply_committed",
                {
                    "host_request_id": host_request_id,
                    "kind": kind,
                    "event": event.get("event"),
                    "buttons": event.get("buttons"),
                    "client_sequence": client_sequence,
                    "last_client_sequence": state["last_client_sequence"],
                    "last_pointer_buttons": state["last_pointer_buttons"],
                    "current": state["current"],
                },
            )
            self._condition.notify_all()

    def snapshot(self, fixture_id: str) -> dict[str, Any]:
        with self._lock:
            if fixture_id not in self._states:
                raise FixtureServerError(f"unknown fixture {fixture_id!r}")
            return copy.deepcopy(self._states[fixture_id])

    def wait_ready(self, fixture_id: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while time.monotonic() < deadline:
                state = self._states[fixture_id]
                if state["geometry_stabilization_error"] is not None:
                    raise FixtureServerError(
                        f"{fixture_id}: {state['geometry_stabilization_error']}"
                    )
                if state["ready"]:
                    return copy.deepcopy(state)
                self._condition.wait(timeout=min(0.2, deadline - time.monotonic()))
        raise TimeoutError(f"fixture {fixture_id} did not report ready in {timeout_s}s")

    def wait_for_browser_quiescence(
        self,
        fixture_id: str,
        *,
        after_sequence: int,
        required_kinds: tuple[str, ...] = (),
        require_pointer_up: bool = False,
        require_pointer_down: bool = False,
        expected_pointer_buttons: int,
        timeout_s: float = 3.0,
        quiet_s: float = 0.1,
    ) -> dict[str, Any]:
        """Wait for causal acknowledgements and a stable client sequence."""
        deadline = time.monotonic() + timeout_s
        deadline_host_monotonic_ns = int(deadline * 1_000_000_000)
        quiet_sequence: int | None = None
        quiet_since: float | None = None
        prior_observation: tuple[Any, ...] | None = None
        with self._condition:
            generation = self._states[fixture_id]["generation"]
            state = self._states[fixture_id]
            self._append_diagnostic_locked(
                state,
                "waiter_started",
                {
                    "after_sequence": after_sequence,
                    "required_kinds": list(required_kinds),
                    "require_pointer_up": require_pointer_up,
                    "require_pointer_down": require_pointer_down,
                    "expected_pointer_buttons": expected_pointer_buttons,
                    "timeout_s": timeout_s,
                    "quiet_s": quiet_s,
                    "deadline_host_monotonic_ns": deadline_host_monotonic_ns,
                },
            )
            while True:
                now = time.monotonic()
                state = self._states[fixture_id]
                if state["generation"] != generation:
                    raise FixtureServerError(
                        f"{fixture_id}: fixture reset while awaiting browser acknowledgement"
                    )
                relevant = [
                    event
                    for event in state["events"]
                    if int(event.get("client_sequence", -1)) > after_sequence
                ]
                observed_kinds = {str(event.get("kind")) for event in relevant}
                has_up = any(
                    event.get("kind") == "pointer"
                    and event.get("event") == "pointerup"
                    and int(event.get("buttons", -1)) == 0
                    for event in relevant
                )
                has_down = any(
                    event.get("kind") == "pointer"
                    and event.get("event") == "pointerdown"
                    and int(event.get("buttons", -1)) != 0
                    for event in relevant
                )
                acknowledged = (
                    bool(relevant)
                    and set(required_kinds).issubset(observed_kinds)
                    and (not require_pointer_up or has_up)
                    and (not require_pointer_down or has_down)
                    and state["last_pointer_buttons"] == expected_pointer_buttons
                )
                current_sequence = int(state["last_client_sequence"])
                observation = (
                    current_sequence,
                    tuple(sorted(observed_kinds)),
                    has_down,
                    has_up,
                    int(state["last_pointer_buttons"]),
                    acknowledged,
                )
                if observation != prior_observation:
                    self._append_diagnostic_locked(
                        state,
                        "waiter_observation",
                        {
                            "last_client_sequence": current_sequence,
                            "relevant_client_sequences": [
                                event.get("client_sequence") for event in relevant
                            ],
                            "relevant_host_request_ids": [
                                event.get("host_request_id") for event in relevant
                            ],
                            "observed_kinds": sorted(observed_kinds),
                            "pointer_down_observed": has_down,
                            "pointer_up_observed": has_up,
                            "pointer_buttons": state["last_pointer_buttons"],
                            "acknowledged": acknowledged,
                            "quiet_s": quiet_s,
                            "quiet_window_started_host_monotonic_ns": (
                                int(now * 1_000_000_000) if acknowledged else None
                            ),
                            "deadline_host_monotonic_ns": (
                                deadline_host_monotonic_ns
                            ),
                        },
                    )
                    prior_observation = observation
                if acknowledged:
                    if quiet_sequence != current_sequence:
                        quiet_sequence = current_sequence
                        quiet_since = now
                    elif quiet_since is not None and now - quiet_since >= quiet_s:
                        self._append_diagnostic_locked(
                            state,
                            "waiter_decision",
                            {
                                "decision": "acknowledged",
                                "last_client_sequence": current_sequence,
                                "relevant_client_sequences": [
                                    event.get("client_sequence") for event in relevant
                                ],
                                "relevant_host_request_ids": [
                                    event.get("host_request_id") for event in relevant
                                ],
                                "pointer_buttons": state["last_pointer_buttons"],
                                "quiet_s": quiet_s,
                                "deadline_host_monotonic_ns": (
                                    deadline_host_monotonic_ns
                                ),
                            },
                        )
                        return {
                            "after_sequence": after_sequence,
                            "last_sequence": current_sequence,
                            "event_count": len(relevant),
                            "observed_kinds": sorted(observed_kinds),
                            "pointer_up_acknowledged": has_up,
                            "pointer_down_acknowledged": has_down,
                            "pointer_buttons": state["last_pointer_buttons"],
                            "quiet_s": quiet_s,
                            "events": copy.deepcopy(relevant),
                        }
                else:
                    quiet_sequence = None
                    quiet_since = None
                if now >= deadline:
                    self._append_diagnostic_locked(
                        state,
                        "waiter_decision",
                        {
                            "decision": "timeout",
                            "last_client_sequence": current_sequence,
                            "relevant_client_sequences": [
                                event.get("client_sequence") for event in relevant
                            ],
                            "relevant_host_request_ids": [
                                event.get("host_request_id") for event in relevant
                            ],
                            "observed_kinds": sorted(observed_kinds),
                            "pointer_down_observed": has_down,
                            "pointer_up_observed": has_up,
                            "pointer_buttons": state["last_pointer_buttons"],
                            "after_sequence": after_sequence,
                            "required_kinds": list(required_kinds),
                            "require_pointer_up": require_pointer_up,
                            "require_pointer_down": require_pointer_down,
                            "expected_pointer_buttons": expected_pointer_buttons,
                            "quiet_s": quiet_s,
                            "deadline_host_monotonic_ns": (
                                deadline_host_monotonic_ns
                            ),
                        },
                    )
                    raise TimeoutError(
                        f"{fixture_id}: browser acknowledgement timeout after sequence "
                        f"{after_sequence}; required_kinds={required_kinds}, "
                        f"require_pointer_up={require_pointer_up}, "
                        f"require_pointer_down={require_pointer_down}, "
                        f"expected_pointer_buttons={expected_pointer_buttons}, "
                        f"last_sequence={current_sequence}, events={relevant}"
                    )
                wait_for = min(0.05, deadline - now)
                if acknowledged and quiet_since is not None:
                    wait_for = min(wait_for, max(0.0, quiet_s - (now - quiet_since)))
                self._condition.wait(timeout=max(0.001, wait_for))


class FixtureHttpServer:
    def __init__(
        self,
        manifest: FixtureManifest,
        *,
        host: str = "0.0.0.0",
        enable_browser_audit: bool = False,
    ) -> None:
        self.manifest = manifest
        self.store = FixtureStateStore(manifest)
        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                if path == "/health":
                    self._send(200, b'{"status":"ok"}', "application/json")
                    return
                prefix = "/fixture/"
                if not path.startswith(prefix):
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                fixture_id = urllib.parse.unquote(path[len(prefix) :])
                try:
                    fixture = manifest.by_id(fixture_id)
                    generation = store.snapshot(fixture.id)["generation"]
                    body = render_fixture_html(
                        fixture,
                        generation,
                        enable_browser_audit=enable_browser_audit,
                    ).encode("utf-8")
                except Exception as exc:  # fail closed; no state is returned
                    self._send(404, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(200, body, "text/html; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                audit_prefix = "/audit/"
                if path.startswith(audit_prefix):
                    if not enable_browser_audit:
                        self._send(404, b"not found", "text/plain; charset=utf-8")
                        return
                    try:
                        fixture = manifest.by_id(
                            urllib.parse.unquote(path[len(audit_prefix) :])
                        )
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 65536:
                            raise FixtureServerError("invalid browser audit body length")
                        payload = json.loads(self.rfile.read(length))
                        if not isinstance(payload, dict):
                            raise FixtureServerError(
                                "browser audit body must be an object"
                            )
                        acknowledgement = store.apply_browser_audit(fixture, payload)
                    except Exception as exc:
                        self._send(
                            HTTPStatus.BAD_REQUEST,
                            json.dumps({"error": str(exc)}).encode("utf-8"),
                            "application/json",
                        )
                        return
                    self._send(
                        200,
                        json.dumps(
                            {"status": "accepted", **acknowledgement},
                            separators=(",", ":"),
                        ).encode("utf-8"),
                        "application/json",
                    )
                    return
                prefix = "/event/"
                if not path.startswith(prefix):
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                fixture: Fixture | None = None
                host_request_id: int | None = None
                client_sequence: int | None = None
                apply_started = False
                try:
                    fixture = manifest.by_id(urllib.parse.unquote(path[len(prefix) :]))
                    length = int(self.headers.get("Content-Length", "0"))
                    host_request_id = store.begin_http_request(
                        fixture,
                        {"method": "POST", "path": path, "content_length": length},
                    )
                    if not 0 < length <= 65536:
                        raise FixtureServerError("invalid event body length")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise FixtureServerError("event body must be an object")
                    raw_client_sequence = payload.get("client_sequence")
                    client_sequence = (
                        int(raw_client_sequence)
                        if raw_client_sequence is not None
                        else None
                    )
                    store.record_diagnostic(
                        fixture,
                        "http_body_received",
                        {
                            "content_length": length,
                            "kind": payload.get("kind"),
                            "client_sequence": payload.get("client_sequence"),
                            "host_request_id": host_request_id,
                            "payload": payload,
                        },
                    )
                    apply_started = True
                    store.apply_event(
                        fixture, payload, host_request_id=host_request_id
                    )
                except Exception as exc:
                    if fixture is not None:
                        if apply_started:
                            store.record_diagnostic(
                                fixture,
                                "store_apply_rejected",
                                {
                                    "host_request_id": host_request_id,
                                    "client_sequence": client_sequence,
                                    "error": str(exc),
                                },
                            )
                        store.record_diagnostic(
                            fixture,
                            "http_response_started",
                            {
                                "status": int(HTTPStatus.BAD_REQUEST),
                                "error": str(exc),
                                "host_request_id": host_request_id,
                                "client_sequence": client_sequence,
                            },
                        )
                    self._send(
                        HTTPStatus.BAD_REQUEST,
                        json.dumps({"error": str(exc)}).encode("utf-8"),
                        "application/json",
                    )
                    if fixture is not None:
                        store.record_diagnostic(
                            fixture,
                            "http_response_completed",
                            {
                                "status": int(HTTPStatus.BAD_REQUEST),
                                "host_request_id": host_request_id,
                                "client_sequence": client_sequence,
                            },
                        )
                    return
                store.record_diagnostic(
                    fixture,
                    "http_response_started",
                    {
                        "status": int(HTTPStatus.OK),
                        "host_request_id": host_request_id,
                        "client_sequence": client_sequence,
                    },
                )
                self._send(200, b'{"status":"accepted"}', "application/json")
                store.record_diagnostic(
                    fixture,
                    "http_response_completed",
                    {
                        "status": int(HTTPStatus.OK),
                        "host_request_id": host_request_id,
                        "client_sequence": client_sequence,
                    },
                )

        self._server = ThreadingHTTPServer((host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def guest_url(self, fixture: Fixture) -> str:
        # qemu user-mode networking exposes the host as 10.0.2.2.
        return f"http://10.0.2.2:{self.port}/fixture/{urllib.parse.quote(fixture.id)}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "FixtureHttpServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _browser_audit_script(fixture: Fixture) -> str:
    endpoint = json.dumps(f"/audit/{urllib.parse.quote(fixture.id)}")
    script = r"""
const auditEndpoint = __AUDIT_ENDPOINT__;
let browserAuditSequence = 0;
let browserHeartbeatAcknowledgement = null;
function auditElement(element) {
  if (!element) return null;
  return {
    id: element.id || '',
    tag: element.tagName ? element.tagName.toLowerCase() : '',
    checked: typeof element.checked === 'boolean' ? element.checked : null
  };
}
function auditPageState() {
  const target = document.getElementById('target');
  const decoy = document.getElementById('decoy');
  return {
    checkbox_state: {
      target: target && typeof target.checked === 'boolean' ? target.checked : null,
      decoy: decoy && typeof decoy.checked === 'boolean' ? decoy.checked : null
    },
    active_element: auditElement(
      document.activeElement instanceof Element ? document.activeElement : null
    ),
    document_has_focus: document.hasFocus(),
    visibility_state: document.visibilityState
  };
}
function prepareBrowserAudit(payload) {
  payload.schema_version = __AUDIT_SCHEMA_VERSION__;
  payload.generation = generation;
  const priorAuditSequence = browserAuditSequence;
  payload.audit_sequence = ++browserAuditSequence;
  if (payload.event === 'audit_heartbeat') {
    payload.expected_previous_audit_sequence = priorAuditSequence;
    payload.expected_audit_count_through_marker = browserAuditSequence;
  }
  payload.browser_wall_time_ms = Date.now();
  payload.client_monotonic_ms = Math.round(performance.now() * 1000) / 1000;
  Object.assign(payload, auditPageState());
  return payload;
}
function sendBrowserAudit(payload) {
  payload = prepareBrowserAudit(payload);
  navigator.sendBeacon(auditEndpoint, JSON.stringify(payload));
}
async function sendBrowserHeartbeat() {
  const acknowledged = browserHeartbeatAcknowledgement;
  const payload = prepareBrowserAudit({
    event: 'audit_heartbeat',
    event_time_stamp_ms: null,
    is_trusted: null,
    default_prevented: null,
    target: null,
    target_checked: null,
    button: null,
    buttons: null,
    pointer_type: null,
    client_x: null,
    client_y: null,
    screen_x: null,
    screen_y: null,
    acknowledged_heartbeat_audit_sequence:
      acknowledged ? acknowledged.audit_sequence : null,
    acknowledged_host_audit_request_id:
      acknowledged ? acknowledged.host_audit_request_id : null,
    acknowledged_host_monotonic_ns:
      acknowledged ? acknowledged.host_monotonic_ns : null
  });
  try {
    const response = await fetch(auditEndpoint, {
      method: 'POST',
      body: JSON.stringify(payload),
      headers: {'Content-Type': 'text/plain;charset=UTF-8'},
      cache: 'no-store',
      keepalive: true
    });
    if (!response.ok) return;
    const candidate = await response.json();
    if (
      candidate.audit_sequence === payload.audit_sequence &&
      Number.isInteger(candidate.host_audit_request_id) &&
      Number.isInteger(candidate.host_monotonic_ns) &&
      (
        browserHeartbeatAcknowledgement === null ||
        candidate.audit_sequence > browserHeartbeatAcknowledgement.audit_sequence
      )
    ) {
      browserHeartbeatAcknowledgement = candidate;
    }
  } catch (_error) {
    // The host-side bounded causal wait owns timeout and raw failure evidence.
  }
}
function emitBrowserAudit(event) {
  const target = event.target instanceof Element ? event.target : null;
  sendBrowserAudit({
    event: event.type,
    event_time_stamp_ms: Math.round(Number(event.timeStamp) * 1000) / 1000,
    is_trusted: event.isTrusted === true,
    default_prevented: event.defaultPrevented === true,
    target: auditElement(target),
    target_checked: target && typeof target.checked === 'boolean'
      ? target.checked : null,
    button: typeof event.button === 'number' ? event.button : null,
    buttons: typeof event.buttons === 'number' ? event.buttons : null,
    pointer_type: typeof event.pointerType === 'string' ? event.pointerType : null,
    client_x: typeof event.clientX === 'number' ? Math.round(event.clientX) : null,
    client_y: typeof event.clientY === 'number' ? Math.round(event.clientY) : null,
    screen_x: typeof event.screenX === 'number' ? Math.round(event.screenX) : null,
    screen_y: typeof event.screenY === 'number' ? Math.round(event.screenY) : null
  });
}
for (const auditName of [
  'pointerdown', 'pointerup', 'pointermove',
  'mousedown', 'mouseup', 'mousemove',
  'click', 'input', 'change', 'focus', 'blur'
]) {
  document.addEventListener(auditName, emitBrowserAudit, true);
}
sendBrowserAudit({
  event: 'audit_ready',
  event_time_stamp_ms: null,
  is_trusted: null,
  default_prevented: null,
  target: null,
  target_checked: null,
  button: null,
  buttons: null,
  pointer_type: null,
  client_x: null,
  client_y: null,
  screen_x: null,
  screen_y: null,
  page_time_origin_ms: performance.timeOrigin,
  url: location.href
});
setInterval(() => { void sendBrowserHeartbeat(); }, 500);
""".strip()
    return script.replace("__AUDIT_ENDPOINT__", endpoint).replace(
        "__AUDIT_SCHEMA_VERSION__", str(BROWSER_AUDIT_SCHEMA_VERSION)
    )


def _common_script(
    fixture: Fixture,
    generation: int,
    ready_payload_js: str,
    *,
    setup_js: str = "",
    enable_browser_audit: bool = False,
) -> str:
    endpoint = f"/event/{urllib.parse.quote(fixture.id)}"
    audit_script = _browser_audit_script(fixture) if enable_browser_audit else ""
    return f"""
<script>
const generation = {generation};
const endpoint = {json.dumps(endpoint)};
let clientSequence = 0;
let postQueue = Promise.resolve();
const diagnosticRingLimit = 256;
const rung1aDiagnostics = {{
  schema_version: 1,
  generation,
  page_time_origin_ms: performance.timeOrigin,
  page_events: [],
  report_queue: {{
    enqueued: 0,
    send_started: 0,
    response_received: 0,
    acknowledged: 0,
    failed: 0,
    pending: 0,
    last_enqueued_sequence: null,
    last_fetch_started_sequence: null,
    last_resolved_sequence: null,
    last_rejected_sequence: null,
    records: [],
    transitions: []
  }}
}};
window.__RUNG1A_DIAGNOSTICS__ = rung1aDiagnostics;
{audit_script}
function boundedPush(items, value) {{
  items.push(value);
  if (items.length > diagnosticRingLimit) items.splice(0, items.length - diagnosticRingLimit);
}}
function queueTransition(record, state) {{
  const transition = {{
    client_sequence: record.client_sequence,
    state,
    client_monotonic_ms: Math.round(performance.now() * 1000) / 1000,
    pending: rung1aDiagnostics.report_queue.pending
  }};
  record.state = state;
  boundedPush(record.state_transitions, transition);
  boundedPush(rung1aDiagnostics.report_queue.transitions, transition);
}}
function post(payload) {{
  payload.generation = generation;
  payload.client_sequence = ++clientSequence;
  payload.client_monotonic_ms = Math.round(performance.now() * 1000) / 1000;
  const body = JSON.stringify(payload);
  const queue = rung1aDiagnostics.report_queue;
  const record = {{
    client_sequence: payload.client_sequence,
    predecessor_client_sequence: payload.client_sequence > 1
      ? payload.client_sequence - 1 : null,
    kind: payload.kind,
    event: payload.event || null,
    enqueue_client_monotonic_ms: payload.client_monotonic_ms,
    send_started_client_monotonic_ms: null,
    response_client_monotonic_ms: null,
    acknowledged_client_monotonic_ms: null,
    failed_client_monotonic_ms: null,
    http_status: null,
    error: null,
    state: null,
    predecessor_settlement_at_fetch: null,
    state_transitions: []
  }};
  boundedPush(rung1aDiagnostics.page_events, JSON.parse(body));
  boundedPush(queue.records, record);
  queue.enqueued += 1;
  queue.pending += 1;
  queue.last_enqueued_sequence = payload.client_sequence;
  queueTransition(record, 'enqueued');
  const send = (predecessorSettlement) => {{
    queue.send_started += 1;
    queue.last_fetch_started_sequence = payload.client_sequence;
    record.predecessor_settlement_at_fetch = payload.client_sequence === 1
      ? 'root_resolved' : predecessorSettlement;
    record.send_started_client_monotonic_ms = Math.round(performance.now() * 1000) / 1000;
    queueTransition(record, 'fetch_started');
    return fetch(endpoint, {{method: 'POST',
      headers: {{'Content-Type':'application/json'}}, body, cache: 'no-store'}})
      .then(response => {{
        queue.response_received += 1;
        record.response_client_monotonic_ms = Math.round(performance.now() * 1000) / 1000;
        record.http_status = response.status;
        if (!response.ok) throw new Error(`event POST ${{response.status}}`);
        queue.acknowledged += 1;
        queue.pending -= 1;
        queue.last_resolved_sequence = payload.client_sequence;
        record.acknowledged_client_monotonic_ms = Math.round(performance.now() * 1000) / 1000;
        queueTransition(record, 'resolved');
        return response;
      }})
      .catch(error => {{
        queue.failed += 1;
        queue.pending -= 1;
        queue.last_rejected_sequence = payload.client_sequence;
        record.failed_client_monotonic_ms = Math.round(performance.now() * 1000) / 1000;
        record.error = String(error);
        queueTransition(record, 'rejected');
        throw error;
      }});
  }};
  // Preserve browser dispatch order at the host oracle. Independent fetches can
  // otherwise arrive out of order and make pointer traces non-causal.
  postQueue = postQueue.then(
    () => send('resolved'),
    () => send('rejected')
  );
  return postQueue;
}}
function screenRect(element) {{
  const r = element.getBoundingClientRect();
  const topChrome = Math.max(0, window.outerHeight - window.innerHeight);
  return {{left: Math.round(window.screenX + r.left),
           top: Math.round(window.screenY + topChrome + r.top),
           right: Math.round(window.screenX + r.right),
           bottom: Math.round(window.screenY + topChrome + r.bottom),
           width: Math.round(r.width), height: Math.round(r.height),
           center_x: Math.round(window.screenX + r.left + r.width / 2),
           center_y: Math.round(window.screenY + topChrome + r.top + r.height / 2)}};
}}
function measuredGeometry(parts) {{
  parts.window = {{screen_x: Math.round(window.screenX), screen_y: Math.round(window.screenY),
    screen_width: Math.round(window.screen.width), screen_height: Math.round(window.screen.height),
    outer_width: Math.round(window.outerWidth), outer_height: Math.round(window.outerHeight),
    inner_width: Math.round(window.innerWidth), inner_height: Math.round(window.innerHeight),
    chrome_top: Math.max(0, Math.round(window.outerHeight - window.innerHeight))}};
  return parts;
}}
for (const name of ['pointerdown', 'pointerup', 'pointermove']) {{
  document.addEventListener(name, (e) => {{
    if (name !== 'pointermove' || e.buttons) {{
      const hit = document.elementFromPoint(e.clientX, e.clientY);
      const topChrome = Math.max(0, window.outerHeight - window.innerHeight);
      post({{kind:'pointer', event:name, button:e.button, buttons:e.buttons,
        client_x:Math.round(e.clientX), client_y:Math.round(e.clientY),
        screen_x:Math.round(window.screenX + e.clientX),
        screen_y:Math.round(window.screenY + topChrome + e.clientY),
        hit_id:hit && hit.id ? hit.id : '',
        hit_tag:hit && hit.tagName ? hit.tagName.toLowerCase() : ''}});
    }}
  }}, true);
}}
async function stabilizeGeometry() {{
  await document.fonts.ready;
  {setup_js}
  let priorGeometry = null;
  let consecutiveIdentical = 0;
  for (let animationFrame = 1; animationFrame <= {MAX_GEOMETRY_ANIMATION_FRAMES}; animationFrame++) {{
    await new Promise(resolve => requestAnimationFrame(resolve));
    const sample = {ready_payload_js};
    const serialized = JSON.stringify(sample.geometry);
    consecutiveIdentical = serialized === priorGeometry ? consecutiveIdentical + 1 : 1;
    priorGeometry = serialized;
    await post({{kind:'geometry_observation', geometry:sample.geometry,
      animation_frame:animationFrame, fonts_ready:true,
      consecutive_identical:consecutiveIdentical}});
    if (consecutiveIdentical >= {STABLE_GEOMETRY_OBSERVATIONS}) {{
      await post({{kind:'ready', geometry:sample.geometry, value:sample.value,
        animation_frame:animationFrame, fonts_ready:true,
        stable_observation_count:consecutiveIdentical}});
      return;
    }}
  }}
  throw new Error('geometry did not stabilize within animation-frame bound');
}}
window.addEventListener('load', () => stabilizeGeometry().catch(error =>
  post({{kind:'geometry_failure', error:String(error)}})));
</script>
"""


def render_fixture_html(
    fixture: Fixture, generation: int, *, enable_browser_audit: bool = False
) -> str:
    p = fixture.params
    accent = html.escape(str(p["accent"]), quote=True)
    base = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Capability fixture</title>
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin:0; width:100%; min-height:100%; font-family:Ubuntu,Arial,sans-serif;
  color:#17202a; background:#f5f7fa; }}
.banner {{ position:fixed; z-index:10; left:0; top:0; right:0; height:86px;
  background:#fff; border-bottom:4px solid {accent}; padding:16px 28px; }}
.banner h1 {{ margin:0 0 5px; font-size:24px; }} .banner p {{ margin:0; font-size:17px; }}
.card {{ position:absolute; background:#fff; border:2px solid #d5dbe3; border-radius:12px;
  padding:24px; box-shadow:0 4px 14px rgba(0,0,0,.10); }}
label {{ font-size:21px; font-weight:600; }}
</style></head><body>
<div class="banner"><h1>Browser control</h1><p>{html.escape(fixture.instruction)}</p></div>
"""
    if fixture.template == "click":
        position = _card_position(p, width=360, height=180)
        content = f"""
<div class="card" style="{position};width:360px">
 <label><input id="target" type="checkbox" style="width:28px;height:28px;vertical-align:middle">
 {html.escape(str(p['label']))}</label>
 <hr><label style="font-weight:400"><input id="decoy" type="checkbox"> Preview only</label>
</div>
<script>
target.addEventListener('change', () => post({{kind:'click', checked:target.checked,
 decoy_checked:decoy.checked}}));
decoy.addEventListener('change', () => post({{kind:'click', checked:target.checked,
 decoy_checked:decoy.checked}}));
</script>
"""
        ready = "({geometry:measuredGeometry({target:screenRect(target), decoy:screenRect(decoy)}), value:target.checked})"
        setup = ""
    elif fixture.template == "focus_type":
        position = _card_position(p, width=520, height=160)
        content = f"""
<div class="card" style="{position};width:520px">
 <label for="target">{html.escape(str(p['label']))}</label><br>
 <input id="target" value="{html.escape(str(p['initial_text']), quote=True)}"
  style="margin-top:14px;width:100%;height:52px;font-size:22px;padding:8px">
</div>
<script>target.addEventListener('input', () => post({{kind:'text', text:target.value}}));</script>
"""
        ready = "({geometry:measuredGeometry({target:screenRect(target)}), value:target.value})"
        setup = ""
    elif fixture.template == "drag":
        card_width = int(p["width"]) + 70
        position = _card_position(p, width=card_width, height=180)
        content = f"""
<div class="card" style="{position};width:{card_width}px">
 <label for="target">{html.escape(str(p['label']))}</label><br>
 <input id="target" type="range" min="0" max="100" step="1" value="{int(p['initial_value'])}"
  style="margin-top:22px;width:{int(p['width'])}px;height:42px;accent-color:{accent}">
 <output id="readout">{int(p['initial_value'])}</output>
</div>
<script>target.addEventListener('input', () => {{readout.value=target.value;
 post({{kind:'drag', value:Number(target.value)}});}});</script>
"""
        ready = "({geometry:measuredGeometry({target:screenRect(target)}), value:Number(target.value)})"
        setup = ""
    elif fixture.template == "scroll":
        blocks = "".join(
            f'<section style="height:420px;padding:120px 80px;font-size:28px;background:{"#fff" if i % 2 else "#edf2f7"}">'
            f'{html.escape(str(p["label"]))} checkpoint {i}</section>'
            for i in range(1, 13)
        )
        content = f"""
<main style="padding-top:86px">{blocks}</main>
<script>
let scrollTimer;
window.addEventListener('scroll', () => {{ clearTimeout(scrollTimer); scrollTimer=setTimeout(() =>
 post({{kind:'scroll', scroll_y:Math.round(window.scrollY)}}), 40); }});
</script>
"""
        ready = (
            "({geometry:measuredGeometry({viewport:{width:window.innerWidth,"
            "height:window.innerHeight}}), value:Math.round(window.scrollY)})"
        )
        setup = f"window.scrollTo(0, {int(p['initial_y'])});"
    else:
        raise FixtureServerError(f"unknown template {fixture.template!r}")
    return (
        base
        + content
        + _common_script(
            fixture,
            generation,
            ready,
            setup_js=setup,
            enable_browser_audit=enable_browser_audit,
        )
        + "</body></html>"
    )


def _card_position(params: dict[str, Any], *, width: int, height: int) -> str:
    """Map sealed 1920x1080 design coordinates into the measured viewport.

    The final CSS clamp keeps the whole card visible even when Chrome's actual
    inner viewport is smaller than the QCOW's 1920x1080 desktop.
    """
    left_percent = 100.0 * int(params["left"]) / 1920.0
    top_percent = 100.0 * int(params["top"]) / 1080.0
    return (
        f"left:clamp(24px,{left_percent:.6f}vw,calc(100vw - {width + 24}px));"
        f"top:clamp(104px,{top_percent:.6f}vh,calc(100vh - {height + 24}px))"
    )
