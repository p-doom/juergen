"""Task-level VM rollout collector with atomic episode resumption.

The collector is deliberately adapter-based: Stage 5 owns the scientific trace
contract while the VM provider and model server own actuation and inference.  A
batch pins one immutable actor fingerprint.  Learner outputs cannot be hot-loaded
into an in-flight batch.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from stage5_rft.contamination import (
    ContaminationBlocklist,
    assert_clean,
    audit_tasks,
)
from stage5_rft.schema import (
    SCHEMA_VERSION,
    ActionTrace,
    ArtifactRef,
    EpisodeTrace,
    FailureKind,
    PolicyProvenance,
    ResetSpec,
    StateRef,
    StepTrace,
    TaskSpec,
)
from stage5_rft.util import (
    ContractError,
    atomic_write_json,
    read_json,
    sha256_bytes,
    sha256_json,
)


@dataclass(frozen=True)
class EnvObservation:
    screenshot_png: bytes
    state: dict[str, Any]


@dataclass(frozen=True)
class EnvTransition:
    observation: EnvObservation
    reward: float
    done: bool
    task_success: bool
    failure_kind: FailureKind = FailureKind.NONE
    info: dict[str, Any] | None = None


@dataclass(frozen=True)
class ActorRequest:
    request_id: str
    episode_id: str
    instruction: str
    step_index: int
    screenshot: ArtifactRef
    state: StateRef
    history: tuple[dict[str, Any], ...]
    seed: int


@dataclass(frozen=True)
class PolicyOutput:
    raw_output: str
    parsed_action: dict[str, Any] | None
    parser: str
    served_policy_fingerprint: str
    logprob: float | None = None
    failure_kind: FailureKind = FailureKind.NONE


class VMEnvironment(Protocol):
    def reset(self, spec: ResetSpec) -> EnvObservation: ...

    def step(self, action: Mapping[str, Any]) -> EnvTransition: ...

    def close(self) -> None: ...


class PolicyActor(Protocol):
    @property
    def provenance(self) -> PolicyProvenance: ...

    def sample(self, request: ActorRequest) -> PolicyOutput: ...


class EpisodeStore:
    """Content-address screenshots and atomically commit complete episodes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.episodes = self.root / "episodes"
        self.partials = self.root / "partials"
        for path in (self.artifacts, self.episodes, self.partials):
            path.mkdir(parents=True, exist_ok=True)

    def put_screenshot(self, payload: bytes) -> ArtifactRef:
        digest = sha256_bytes(payload)
        path = self.artifacts / f"{digest}.png"
        if path.exists():
            if sha256_bytes(path.read_bytes()) != digest:
                raise ContractError(f"corrupt content-addressed artifact: {path}")
        else:
            # Bytes are immutable by digest.  A same-directory replace makes the
            # artifact visible only after the complete payload is synced.
            # latin1 is not safe for PNG, so use a local binary equivalent here.
            import os
            import tempfile

            fd, raw_tmp = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=path.parent)
            tmp = Path(raw_tmp)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        return ArtifactRef(
            uri=str(path.relative_to(self.root)),
            sha256=digest,
            size_bytes=len(payload),
        )

    def episode_path(self, episode_id: str) -> Path:
        return self.episodes / f"{episode_id}.json"

    def partial_path(self, episode_id: str) -> Path:
        return self.partials / f"{episode_id}.json"

    def load_complete(self, episode_id: str) -> EpisodeTrace | None:
        path = self.episode_path(episode_id)
        if not path.is_file():
            return None
        return EpisodeTrace.from_dict(read_json(path))

    def next_attempt(self, episode_id: str) -> int:
        path = self.partial_path(episode_id)
        if not path.is_file():
            return 1
        value = read_json(path)
        return int(value.get("collection_attempt", 0)) + 1

    def save_partial(
        self,
        *,
        task: TaskSpec,
        policy: PolicyProvenance,
        actor_id: str,
        collection_attempt: int,
        steps: Sequence[StepTrace],
    ) -> None:
        atomic_write_json(
            self.partial_path(task.episode_id),
            {
                "status": "partial_restart_from_reset",
                "episode_id": task.episode_id,
                "task_fingerprint": sha256_json(asdict(task)),
                "policy_fingerprint": policy.fingerprint,
                "actor_id": actor_id,
                "collection_attempt": collection_attempt,
                "n_durable_steps": len(steps),
                "steps": [asdict(step) for step in steps],
            },
        )

    def commit(self, episode: EpisodeTrace) -> None:
        episode.validate()
        existing = self.load_complete(episode.episode_id)
        if existing is not None:
            if existing.trace_sha256 != episode.trace_sha256:
                raise ContractError(
                    f"episode {episode.episode_id} already committed with different content"
                )
            return
        atomic_write_json(self.episode_path(episode.episode_id), episode.as_dict())
        # Keep a tiny tombstone instead of deleting provenance for an interrupted
        # attempt.  Resume always trusts the atomic complete record first.
        atomic_write_json(
            self.partial_path(episode.episode_id),
            {
                "status": "committed",
                "episode_id": episode.episode_id,
                "collection_attempt": episode.collection_attempt,
                "trace_sha256": episode.trace_sha256,
            },
        )

    def load_all(self) -> list[EpisodeTrace]:
        return [
            EpisodeTrace.from_dict(read_json(path))
            for path in sorted(self.episodes.glob("*.json"))
        ]


def _assert_initial_observation(reset: ResetSpec, observation: EnvObservation) -> None:
    screenshot_digest = sha256_bytes(observation.screenshot_png)
    state_digest = sha256_json(observation.state)
    if screenshot_digest != reset.expected_initial_screenshot_sha256:
        raise ContractError(
            "deterministic reset screenshot mismatch: "
            f"expected {reset.expected_initial_screenshot_sha256}, got {screenshot_digest}"
        )
    if state_digest != reset.expected_initial_state_sha256:
        raise ContractError(
            "deterministic reset state mismatch: "
            f"expected {reset.expected_initial_state_sha256}, got {state_digest}"
        )


class EpisodeCollector:
    def __init__(
        self,
        *,
        store: EpisodeStore,
        environment: VMEnvironment,
        actor: PolicyActor,
        actor_id: str,
        contamination_blocklist: ContaminationBlocklist,
    ) -> None:
        self.store = store
        self.environment = environment
        self.actor = actor
        self.actor_id = actor_id
        self.blocklist = contamination_blocklist
        self.actor.provenance.validate()

    def collect(self, task: TaskSpec) -> EpisodeTrace:
        task.validate()
        assert_clean(audit_tasks([task], self.blocklist))
        complete = self.store.load_complete(task.episode_id)
        if complete is not None:
            if complete.policy.fingerprint != self.actor.provenance.fingerprint:
                raise ContractError("resume found episode from a different actor policy")
            if complete.reset.fingerprint != task.reset.fingerprint:
                raise ContractError("resume found episode with a different reset contract")
            if (
                complete.instruction_sha256 != task.instruction_sha256
                or complete.condition != task.condition
                or complete.max_steps != task.max_steps
                or complete.reward_schema != task.reward_schema
                or complete.reward_config_sha256 != task.reward_config_sha256
            ):
                raise ContractError("resume found episode from a different task contract")
            return complete

        attempt = self.store.next_attempt(task.episode_id)
        steps: list[StepTrace] = []
        observation = self.environment.reset(task.reset)
        _assert_initial_observation(task.reset, observation)

        for step_index in range(task.max_steps):
            before_image = self.store.put_screenshot(observation.screenshot_png)
            before_state = StateRef.capture(observation.state)
            request_id = sha256_json(
                {
                    "episode_id": task.episode_id,
                    "step_index": step_index,
                    "attempt": attempt,
                    "policy": self.actor.provenance.fingerprint,
                }
            )[:24]
            request = ActorRequest(
                request_id=request_id,
                episode_id=task.episode_id,
                instruction=task.instruction,
                step_index=step_index,
                screenshot=before_image,
                state=before_state,
                history=tuple(
                    {
                        "raw_output": step.action.raw_output,
                        "action": step.action.parsed_action,
                        "reward": step.reward,
                        "done": step.done,
                    }
                    for step in steps
                ),
                seed=task.reset.seed + step_index,
            )
            started = time.monotonic()
            actor_error = ""
            try:
                output = self.actor.sample(request)
            except TimeoutError:
                output = PolicyOutput(
                    raw_output="",
                    parsed_action=None,
                    parser="actor_exception",
                    served_policy_fingerprint=self.actor.provenance.fingerprint,
                    failure_kind=FailureKind.POLICY_TIMEOUT,
                )
            except Exception as exc:
                output = PolicyOutput(
                    raw_output="",
                    parsed_action=None,
                    parser="actor_exception",
                    served_policy_fingerprint=self.actor.provenance.fingerprint,
                    failure_kind=FailureKind.POLICY_ERROR,
                )
                actor_error = f"{type(exc).__name__}: {exc}"
            else:
                actor_error = ""

            if output.served_policy_fingerprint != self.actor.provenance.fingerprint:
                raise ContractError(
                    "actor served a different checkpoint than the batch-pinned policy"
                )

            valid = output.parsed_action is not None
            if not valid:
                failure = (
                    output.failure_kind
                    if output.failure_kind != FailureKind.NONE
                    else FailureKind.PARSE_ERROR
                )
                action = ActionTrace(
                    raw_output=output.raw_output,
                    parsed_action=None,
                    parser=output.parser,
                    schema=self.actor.provenance.action_schema,
                    served_policy_fingerprint=output.served_policy_fingerprint,
                    valid=False,
                    dispatched=False,
                    logprob=output.logprob,
                )
                transition = EnvTransition(
                    observation=observation,
                    reward=0.0,
                    done=True,
                    task_success=False,
                    failure_kind=failure,
                    info={"actor_error": actor_error} if actor_error else {},
                )
            else:
                try:
                    transition = self.environment.step(output.parsed_action or {})
                except Exception as exc:
                    transition = EnvTransition(
                        observation=observation,
                        reward=0.0,
                        done=True,
                        task_success=False,
                        failure_kind=FailureKind.DISPATCH_ERROR,
                        info={"dispatch_error": f"{type(exc).__name__}: {exc}"},
                    )
                    dispatched = False
                else:
                    dispatched = True
                action = ActionTrace(
                    raw_output=output.raw_output,
                    parsed_action=dict(output.parsed_action or {}),
                    parser=output.parser,
                    schema=self.actor.provenance.action_schema,
                    served_policy_fingerprint=output.served_policy_fingerprint,
                    valid=True,
                    dispatched=dispatched,
                    logprob=output.logprob,
                )

            terminal = transition.done or step_index + 1 == task.max_steps
            failure_kind = transition.failure_kind
            info = dict(transition.info or {})
            if terminal and not transition.done and not transition.task_success:
                failure_kind = FailureKind.MAX_STEPS
                info["environment_done"] = False
            if terminal and not transition.task_success and failure_kind == FailureKind.NONE:
                failure_kind = FailureKind.TASK_FAILURE
            after_image = self.store.put_screenshot(transition.observation.screenshot_png)
            after_state = StateRef.capture(transition.observation.state)
            step = StepTrace(
                step_index=step_index,
                request_id=request_id,
                sampling_seed=request.seed,
                screenshot_before=before_image,
                state_before=before_state,
                action=action,
                screenshot_after=after_image,
                state_after=after_state,
                reward=float(transition.reward),
                done=terminal,
                task_success=bool(transition.task_success),
                failure_kind=failure_kind,
                elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
                info=info,
            )
            steps.append(step)
            self.store.save_partial(
                task=task,
                policy=self.actor.provenance,
                actor_id=self.actor_id,
                collection_attempt=attempt,
                steps=steps,
            )
            observation = transition.observation
            if terminal:
                break

        episode = EpisodeTrace(
            schema_version=SCHEMA_VERSION,
            episode_id=task.episode_id,
            instruction=task.instruction,
            instruction_sha256=task.instruction_sha256,
            condition=task.condition,
            max_steps=task.max_steps,
            reward_schema=task.reward_schema,
            reward_config_sha256=task.reward_config_sha256,
            policy=self.actor.provenance,
            actor_id=self.actor_id,
            reset=task.reset,
            collection_attempt=attempt,
            steps=tuple(steps),
            total_reward=sum(step.reward for step in steps),
            success=steps[-1].task_success,
            terminal_failure=(FailureKind.NONE if steps[-1].task_success else steps[-1].failure_kind),
        )
        self.store.commit(episode)
        return episode

    def collect_many(self, tasks: Sequence[TaskSpec]) -> dict[str, Any]:
        if len({task.episode_id for task in tasks}) != len(tasks):
            raise ContractError("task manifest contains duplicate episode_id values")
        report = audit_tasks(tasks, self.blocklist)
        assert_clean(report)
        episodes = [self.collect(task) for task in tasks]
        fingerprints = {episode.policy.fingerprint for episode in episodes}
        if fingerprints != {self.actor.provenance.fingerprint}:
            raise ContractError("collection is not on-policy for one immutable actor")
        manifest = {
            "schema_version": "stage5.collection.v1",
            "status": "complete",
            "on_policy": True,
            "actor_id": self.actor_id,
            "actor_policy": asdict(self.actor.provenance),
            "actor_policy_fingerprint": self.actor.provenance.fingerprint,
            "episode_count": len(episodes),
            "episodes": {
                episode.episode_id: episode.trace_sha256 for episode in sorted(episodes, key=lambda x: x.episode_id)
            },
            "contamination": report.as_dict(),
            "resume_unit": "complete_episode_restart_incomplete_from_reset",
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        atomic_write_json(self.store.root / "collection_manifest.json", manifest)
        return manifest

    def close(self) -> None:
        self.environment.close()
