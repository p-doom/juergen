"""Actor/learner boundary and resumable learner handoff contract.

This module plans and validates training; it intentionally does not import a
trainer or allocate accelerators.  The consuming trainer must write checkpoints
outside both the actor checkpoint and rollout collection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stage5_rft.schema import PolicyProvenance
from stage5_rft.util import ContractError, atomic_write_json, read_json, sha256_json


@dataclass(frozen=True)
class LearnerPlan:
    schema_version: str
    learner_run_id: str
    method: str
    dataset_dir: str
    dataset_manifest_sha256: str
    parent_actor_policy: PolicyProvenance
    parent_actor_policy_fingerprint: str
    output_checkpoint_dir: str
    trainer_adapter: str
    resume_mode: str
    seed: int
    actor_frozen_for_collection: bool = True
    output_eligible_only_next_iteration: bool = True
    launch_authorized: bool = False

    def validate(self) -> None:
        if self.schema_version != "stage5.learner_plan.v1":
            raise ContractError("unsupported learner plan schema")
        if self.method not in {"rejection", "reward_weighted"}:
            raise ContractError("unsupported learner method")
        self.parent_actor_policy.validate()
        if self.parent_actor_policy.fingerprint != self.parent_actor_policy_fingerprint:
            raise ContractError("learner parent fingerprint mismatch")
        if self.resume_mode != "exact_manifest_and_parent":
            raise ContractError("learner resume must pin both manifest and parent policy")
        actor_path = Path(self.parent_actor_policy.checkpoint_uri).resolve()
        output_path = Path(self.output_checkpoint_dir).resolve()
        dataset_path = Path(self.dataset_dir).resolve()
        if output_path == actor_path or output_path.is_relative_to(actor_path):
            raise ContractError("learner must never overwrite the rollout actor checkpoint")
        if output_path == dataset_path or output_path.is_relative_to(dataset_path):
            raise ContractError("learner checkpoint cannot be written into learner data")
        if not self.actor_frozen_for_collection or not self.output_eligible_only_next_iteration:
            raise ContractError("actor/learner separation flags must remain true")
        if self.launch_authorized:
            raise ContractError(
                "construction plan cannot authorize training; promotion authorization is external"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_learner_plan(
    *,
    dataset_dir: str | Path,
    output_checkpoint_dir: str | Path,
    learner_run_id: str,
    trainer_adapter: str,
    seed: int = 0,
) -> LearnerPlan:
    dataset = Path(dataset_dir)
    manifest = read_json(dataset / "manifest.json")
    if manifest.get("status") != "complete":
        raise ContractError("learner dataset is not atomically complete")
    recorded = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if recorded != sha256_json(unsigned):
        raise ContractError("learner dataset manifest digest mismatch")
    if not manifest.get("contamination", {}).get("clean"):
        raise ContractError("learner dataset did not pass contamination guard")
    policy = PolicyProvenance.from_dict(manifest["actor_policy"])
    if policy.fingerprint != manifest.get("actor_policy_fingerprint"):
        raise ContractError("dataset actor policy provenance is inconsistent")
    plan = LearnerPlan(
        schema_version="stage5.learner_plan.v1",
        learner_run_id=learner_run_id,
        method=str(manifest["method"]),
        dataset_dir=str(dataset.resolve()),
        dataset_manifest_sha256=str(recorded),
        parent_actor_policy=policy,
        parent_actor_policy_fingerprint=policy.fingerprint,
        output_checkpoint_dir=str(Path(output_checkpoint_dir).resolve()),
        trainer_adapter=trainer_adapter,
        resume_mode="exact_manifest_and_parent",
        seed=seed,
    )
    plan.validate()
    return plan


def write_learner_plan(plan: LearnerPlan, path: str | Path) -> None:
    plan.validate()
    payload = plan.as_dict()
    payload["plan_sha256"] = sha256_json(payload)
    atomic_write_json(path, payload)


def validate_resume_state(plan: LearnerPlan, state_path: str | Path) -> dict[str, Any]:
    state = read_json(state_path)
    if state.get("dataset_manifest_sha256") != plan.dataset_manifest_sha256:
        raise ContractError("learner resume dataset changed")
    if state.get("parent_actor_policy_fingerprint") != plan.parent_actor_policy_fingerprint:
        raise ContractError("learner resume parent checkpoint changed")
    return state
