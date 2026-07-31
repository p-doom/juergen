"""Independent structural and live replay validation for VM episode traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from stage5_rft.collector import EnvObservation, EnvTransition, EpisodeStore
from stage5_rft.schema import EpisodeTrace
from stage5_rft.util import ContractError, read_json, sha256_bytes, sha256_json


@dataclass(frozen=True)
class ReplayDivergence:
    episode_id: str
    step_index: int | None
    field: str
    expected: Any
    observed: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "field": self.field,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class ReplayReport:
    episodes_checked: int
    steps_checked: int
    divergences: tuple[ReplayDivergence, ...]
    live_replay: bool

    @property
    def passed(self) -> bool:
        return not self.divergences

    @property
    def pass_rate(self) -> float:
        if any(d.step_index is None for d in self.divergences):
            return 0.0
        if self.steps_checked == 0:
            return 1.0 if self.passed else 0.0
        failed_steps = len({(d.episode_id, d.step_index) for d in self.divergences})
        return max(0.0, (self.steps_checked - failed_steps) / self.steps_checked)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes_checked": self.episodes_checked,
            "steps_checked": self.steps_checked,
            "divergences": [d.as_dict() for d in self.divergences],
            "live_replay": self.live_replay,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
        }


class ReplayEnvironment(Protocol):
    def reset(self, spec: Any) -> EnvObservation: ...

    def step(self, action: Mapping[str, Any]) -> EnvTransition: ...

    def close(self) -> None: ...


def _artifact_digest(root: Path, uri: str) -> tuple[str, int]:
    path = (root / uri).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ContractError(f"artifact URI escapes rollout root: {uri}")
    if not path.is_file():
        return "missing", -1
    payload = path.read_bytes()
    return sha256_bytes(payload), len(payload)


def validate_collection(root: str | Path) -> ReplayReport:
    store = EpisodeStore(root)
    divergences: list[ReplayDivergence] = []
    episodes = store.load_all()
    for episode in episodes:
        episode.validate()
        for step in episode.steps:
            for field, ref in (
                ("screenshot_before", step.screenshot_before),
                ("screenshot_after", step.screenshot_after),
            ):
                digest, size = _artifact_digest(store.root, ref.uri)
                if digest != ref.sha256:
                    divergences.append(
                        ReplayDivergence(
                            episode.episode_id, step.step_index, f"{field}.sha256", ref.sha256, digest
                        )
                    )
                if size != ref.size_bytes:
                    divergences.append(
                        ReplayDivergence(
                            episode.episode_id, step.step_index, f"{field}.size_bytes", ref.size_bytes, size
                        )
                    )

    manifest_path = store.root / "collection_manifest.json"
    if not manifest_path.is_file():
        divergences.append(
            ReplayDivergence("<collection>", None, "collection_manifest", "present", "missing")
        )
    else:
        manifest = read_json(manifest_path)
        recorded_digest = manifest.pop("manifest_sha256", None)
        observed_digest = sha256_json(manifest)
        if recorded_digest != observed_digest:
            divergences.append(
                ReplayDivergence(
                    "<collection>", None, "manifest_sha256", recorded_digest, observed_digest
                )
            )
        expected = manifest.get("episodes", {})
        observed = {episode.episode_id: episode.trace_sha256 for episode in episodes}
        if expected != observed:
            divergences.append(
                ReplayDivergence("<collection>", None, "episodes", expected, observed)
            )
        fingerprints = {episode.policy.fingerprint for episode in episodes}
        if fingerprints and fingerprints != {manifest.get("actor_policy_fingerprint")}:
            divergences.append(
                ReplayDivergence(
                    "<collection>",
                    None,
                    "actor_policy_fingerprint",
                    manifest.get("actor_policy_fingerprint"),
                    sorted(fingerprints),
                )
            )
    return ReplayReport(
        episodes_checked=len(episodes),
        steps_checked=sum(len(episode.steps) for episode in episodes),
        divergences=tuple(divergences),
        live_replay=False,
    )


def _compare_observation(
    episode_id: str,
    step_index: int | None,
    observed: EnvObservation,
    *,
    screenshot_sha256: str,
    state_sha256: str,
) -> list[ReplayDivergence]:
    out: list[ReplayDivergence] = []
    image = sha256_bytes(observed.screenshot_png)
    state = sha256_json(observed.state)
    if image != screenshot_sha256:
        out.append(
            ReplayDivergence(episode_id, step_index, "screenshot", screenshot_sha256, image)
        )
    if state != state_sha256:
        out.append(ReplayDivergence(episode_id, step_index, "state", state_sha256, state))
    return out


def replay_episodes(
    episodes: Sequence[EpisodeTrace], environment: ReplayEnvironment
) -> ReplayReport:
    divergences: list[ReplayDivergence] = []
    steps_checked = 0
    try:
        for episode in episodes:
            episode.validate()
            observation = environment.reset(episode.reset)
            first = episode.steps[0]
            divergences.extend(
                _compare_observation(
                    episode.episode_id,
                    None,
                    observation,
                    screenshot_sha256=first.screenshot_before.sha256,
                    state_sha256=first.state_before.sha256,
                )
            )
            for step in episode.steps:
                steps_checked += 1
                if step.action.dispatched:
                    transition = environment.step(step.action.parsed_action or {})
                    observation = transition.observation
                    if not math.isclose(transition.reward, step.reward, abs_tol=1e-9):
                        divergences.append(
                            ReplayDivergence(
                                episode.episode_id,
                                step.step_index,
                                "reward",
                                step.reward,
                                transition.reward,
                            )
                        )
                    expected_done = bool(step.info.get("environment_done", step.done))
                    if transition.done != expected_done:
                        divergences.append(
                            ReplayDivergence(
                                episode.episode_id,
                                step.step_index,
                                "done",
                                expected_done,
                                transition.done,
                            )
                        )
                    if transition.task_success != step.task_success:
                        divergences.append(
                            ReplayDivergence(
                                episode.episode_id,
                                step.step_index,
                                "task_success",
                                step.task_success,
                                transition.task_success,
                            )
                        )
                divergences.extend(
                    _compare_observation(
                        episode.episode_id,
                        step.step_index,
                        observation,
                        screenshot_sha256=step.screenshot_after.sha256,
                        state_sha256=step.state_after.sha256,
                    )
                )
    finally:
        environment.close()
    return ReplayReport(
        episodes_checked=len(episodes),
        steps_checked=steps_checked,
        divergences=tuple(divergences),
        live_replay=True,
    )


def validate_deterministic_reset(
    episode: EpisodeTrace, environment: ReplayEnvironment, *, repeats: int = 2
) -> ReplayReport:
    if repeats < 2:
        raise ContractError("deterministic reset validation needs at least two resets")
    divergences: list[ReplayDivergence] = []
    expected_image = episode.reset.expected_initial_screenshot_sha256
    expected_state = episode.reset.expected_initial_state_sha256
    try:
        for repeat in range(repeats):
            observation = environment.reset(episode.reset)
            divergences.extend(
                _compare_observation(
                    episode.episode_id,
                    repeat,
                    observation,
                    screenshot_sha256=expected_image,
                    state_sha256=expected_state,
                )
            )
    finally:
        environment.close()
    return ReplayReport(1, repeats, tuple(divergences), live_replay=True)
