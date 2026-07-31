"""Build task-level learner inputs from on-policy VM episodes.

The default and promotion-eligible method is deliberately simple rejection SFT:
retain complete successful episodes and use unit weight.  Reward weighting is a
separate, explicit experimental mode and can never be enabled by a missing flag.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from stage5_rft.contamination import (
    ContaminationBlocklist,
    assert_clean,
    audit_episodes,
)
from stage5_rft.replay import validate_collection
from stage5_rft.schema import EpisodeTrace
from stage5_rft.util import (
    ContractError,
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    sha256_json,
)


@dataclass(frozen=True)
class RFTConfig:
    mode: str = "rejection"
    minimum_return: float = 0.0
    val_fraction: float = 0.1
    split_salt: str = "stage5-rft-v1"
    reward_temperature: float = 1.0
    maximum_weight: float = 4.0
    enable_reward_weighting_experiment: bool = False

    def validate(self) -> None:
        if self.mode not in {"rejection", "reward_weighted"}:
            raise ContractError("RFT mode must be rejection or reward_weighted")
        if self.mode == "reward_weighted" and not self.enable_reward_weighting_experiment:
            raise ContractError(
                "reward_weighted mode requires enable_reward_weighting_experiment=true; "
                "rejection RFT is the preregistered first method"
            )
        if not 0.0 < self.val_fraction < 1.0:
            raise ContractError("val_fraction must be strictly between 0 and 1")
        if self.reward_temperature <= 0 or self.maximum_weight <= 0:
            raise ContractError("reward weighting parameters must be positive")


def _split(task_id: str, config: RFTConfig) -> str:
    bucket = int(sha256_json({"salt": config.split_salt, "task_id": task_id})[:12], 16)
    return "val" if bucket / float(16**12) < config.val_fraction else "train"


def _weight(episode: EpisodeTrace, config: RFTConfig) -> float:
    if config.mode == "rejection":
        return 1.0
    raw = math.exp((episode.total_reward - config.minimum_return) / config.reward_temperature)
    return min(config.maximum_weight, raw)


def _training_row(episode: EpisodeTrace, config: RFTConfig) -> dict[str, Any]:
    return {
        "schema_version": "stage5.rft_trajectory.v1",
        "sample_id": episode.trace_sha256,
        "episode_id": episode.episode_id,
        "task_id": episode.reset.task_id,
        "task_content_sha256": episode.reset.task_content_sha256,
        "condition": episode.condition,
        "reward_schema": episode.reward_schema,
        "reward_config_sha256": episode.reward_config_sha256,
        "instruction": episode.instruction,
        "actor_policy_fingerprint": episode.policy.fingerprint,
        "source_trace_sha256": episode.trace_sha256,
        "weight": _weight(episode, config),
        "trajectory": [
            {
                "step_index": step.step_index,
                "sampling_seed": step.sampling_seed,
                "screenshot_before": asdict(step.screenshot_before),
                "state_before_sha256": step.state_before.sha256,
                "assistant_target": step.action.raw_output,
                "parsed_action": step.action.parsed_action,
                "action_schema": step.action.schema,
                "reward": step.reward,
                "done": step.done,
                "screenshot_after": asdict(step.screenshot_after),
                "state_after_sha256": step.state_after.sha256,
            }
            for step in episode.steps
        ],
    }


def build_rft_dataset(
    *,
    rollout_root: str | Path,
    output_dir: str | Path,
    blocklist: ContaminationBlocklist,
    config: RFTConfig = RFTConfig(),
) -> dict[str, Any]:
    config.validate()
    root = Path(rollout_root)
    replay = validate_collection(root)
    if not replay.passed:
        raise ContractError(
            f"cannot build learner data from replay-invalid collection: {replay.as_dict()}"
        )
    from stage5_rft.collector import EpisodeStore

    episodes = EpisodeStore(root).load_all()
    if not episodes:
        raise ContractError("rollout collection contains no complete episodes")
    contamination = audit_episodes(episodes, blocklist)
    assert_clean(contamination)
    policy_fingerprints = {episode.policy.fingerprint for episode in episodes}
    if len(policy_fingerprints) != 1:
        raise ContractError("learner input mixes actor policy versions and is not on-policy")

    rejected: Counter[str] = Counter()
    accepted: list[EpisodeTrace] = []
    for episode in episodes:
        if episode.total_reward < config.minimum_return:
            rejected["return_below_threshold"] += 1
            continue
        if config.mode == "rejection" and not episode.success:
            rejected["task_not_successful"] += 1
            continue
        if any(not step.action.valid or not step.action.dispatched for step in episode.steps):
            rejected["invalid_or_undispatched_action"] += 1
            continue
        accepted.append(episode)
    if not accepted:
        raise ContractError("RFT filter accepted zero complete task-level episodes")

    task_splits: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for episode in sorted(accepted, key=lambda x: x.episode_id):
        task_id = episode.reset.task_id
        split = task_splits.setdefault(task_id, _split(task_id, config))
        rows_by_split[split].append(_training_row(episode, config))
    if set(r["task_id"] for r in rows_by_split["train"]) & set(
        r["task_id"] for r in rows_by_split["val"]
    ):
        raise ContractError("task-level split invariant failed")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(out / "train.jsonl", rows_by_split["train"])
    atomic_write_jsonl(out / "val.jsonl", rows_by_split["val"])
    collection_manifest = read_json(root / "collection_manifest.json")
    manifest = {
        "schema_version": "stage5.rft_dataset.v1",
        "status": "complete",
        "method": config.mode,
        "promotion_eligible_method": config.mode == "rejection",
        "actor_policy": collection_manifest["actor_policy"],
        "actor_policy_fingerprint": next(iter(policy_fingerprints)),
        "source_collection_manifest_sha256": collection_manifest["manifest_sha256"],
        "source_rollout_root": str(root.resolve()),
        "source_episode_count": len(episodes),
        "accepted_episode_count": len(accepted),
        "rejected_episode_count": len(episodes) - len(accepted),
        "rejection_reasons": dict(sorted(rejected.items())),
        "train_records": len(rows_by_split["train"]),
        "val_records": len(rows_by_split["val"]),
        "task_splits": dict(sorted(task_splits.items())),
        "contamination": contamination.as_dict(),
        "config": asdict(config),
        "weighting": "unit" if config.mode == "rejection" else "exp_clipped",
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    atomic_write_json(out / "manifest.json", manifest)
    return manifest
