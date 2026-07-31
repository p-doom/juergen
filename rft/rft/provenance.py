"""Answer "what actually IS this checkpoint" without Postgres.

labctl's registry is the normal answer, but its Postgres lives on hai-login2 and has
been down. More importantly, a comparison was nearly run today on a checkpoint pair
that looked like a clean goals-vs-no-goals pair **by name** and in fact differed in
action format, window size *and* sequence length. Names are not provenance.

Everything needed is on disk, reachable by a **4-hop walk**:

1. ``<checkpoint_dir>/.meta.json`` -> ``producer_run_id``, ``metadata.producer_recipe``,
   ``user``, ``metadata.step``;
2. ``labctl_runs/runs/<owner>/<run_id>/.lab/context.json`` -> ``inputs[]`` with
   ``role`` and ``resolved_path`` (plus ``provenance.git_head`` / ``repo_path``);
3. if that run was an *export* (its input role is ``checkpoint``), hop again to the
   **training** run that produced the orbax stream;
4. the training run's ``dataset`` input -> ``<dataset>/manifest.json``, which is the
   authoritative record of ``goal_conditioned`` / ``mode`` / ``action_format`` etc.

:func:`resolve_checkpoint` performs the walk and returns a
:class:`CheckpointProvenance`. It **hard-fails** rather than proceeding with unknowns:
an unresolvable checkpoint must never silently enter a controlled comparison. Two
checkpoints are known-unresolvable and are blacklisted by path
(:data:`UNRESOLVABLE_CHECKPOINTS`) — they have no ``.meta.json``, no run claims their
output, and their training runs record ``"inputs": []``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rft.errors import MissingFieldError, SchemaError

#: Default labctl root on this cluster.
LABCTL_ROOT = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl")

#: Checkpoints whose provenance is unrecoverable: no ``.meta.json``, no run claims
#: their output, and their training runs record ``"inputs": []``. A provenance stage
#: must hard-fail on these rather than proceed with unknowns.
UNRESOLVABLE_CHECKPOINTS: frozenset[str] = frozenset(
    {
        "labctl/checkpoints/mihir.mahajan/abl_hf_oe2/003000",
        "labctl/checkpoints/mihir.mahajan/abl_hf_v7u/003000",
    }
)

#: Dataset-manifest fields that determine what a checkpoint actually learned. These
#: are the ones whose silent divergence made a "clean" comparison uncontrolled.
DECISIVE_DATASET_FIELDS: tuple[str, ...] = (
    "action_format",
    "format",
    "goal_conditioned",
    "goals",
    "mode",
    "max_length",
    "window",
    "n_history_frames",
    "model_id",
    "overflow_mode",
)


class ProvenanceError(SchemaError):
    """A checkpoint's provenance could not be established."""


class BlacklistedCheckpointError(ProvenanceError):
    """A known-unresolvable checkpoint was offered to a controlled comparison."""


@dataclass
class Hop:
    """One resolved step of the walk, kept so the chain is auditable."""

    kind: str  # "checkpoint_meta" | "run_context" | "dataset_manifest"
    path: Path
    detail: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.kind}: {self.path}"


@dataclass
class CheckpointProvenance:
    """What a checkpoint actually is, with the chain that established it."""

    checkpoint_dir: Path
    artifact_id: str | None
    owner: str | None
    producer_recipe: str | None
    producer_run_id: str | None
    step: int | None
    training_run_id: str | None
    dataset_path: Path | None
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    repo_head: str | None = None
    repo_path: str | None = None
    train_args: dict[str, Any] = field(default_factory=dict)
    hops: list[Hop] = field(default_factory=list)

    def decisive(self) -> dict[str, Any]:
        """The fields a controlled comparison must match on.

        Read from the dataset manifest first (authoritative), then from the training
        run's args as a fallback. Missing keys are simply absent — never guessed.
        """
        out: dict[str, Any] = {}
        params = self.dataset_manifest.get("params")
        sources: list[Mapping[str, Any]] = []
        if isinstance(params, Mapping):
            sources.append(params)
        sources.append(self.dataset_manifest)
        sources.append(self.train_args)
        for field_name in DECISIVE_DATASET_FIELDS:
            for source in sources:
                if field_name in source:
                    out[field_name] = source[field_name]
                    break
        return out

    def describe(self) -> str:
        lines = [
            f"checkpoint: {self.checkpoint_dir}",
            f"  artifact_id={self.artifact_id} owner={self.owner} step={self.step}",
            f"  producer_recipe={self.producer_recipe} producer_run={self.producer_run_id}",
            f"  training_run={self.training_run_id}",
            f"  dataset={self.dataset_path}",
            f"  repo={self.repo_path}@{self.repo_head}",
            f"  decisive fields: {self.decisive()!r}",
            "  chain:",
            *[f"    {i + 1}. {h.describe()}" for i, h in enumerate(self.hops)],
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "artifact_id": self.artifact_id,
            "owner": self.owner,
            "producer_recipe": self.producer_recipe,
            "producer_run_id": self.producer_run_id,
            "step": self.step,
            "training_run_id": self.training_run_id,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "decisive": self.decisive(),
            "repo_head": self.repo_head,
            "repo_path": self.repo_path,
            "chain": [{"kind": h.kind, "path": str(h.path)} for h in self.hops],
        }


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise ProvenanceError(f"{what} not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"{what} is not valid JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ProvenanceError(f"{what} is not a JSON object: {path}")
    return payload


def assert_not_blacklisted(checkpoint_dir: str | Path) -> None:
    """Raise if this checkpoint is known to have unrecoverable provenance."""
    resolved = str(Path(checkpoint_dir).resolve())
    for suffix in UNRESOLVABLE_CHECKPOINTS:
        if resolved.endswith(suffix):
            raise BlacklistedCheckpointError(
                f"{checkpoint_dir} is on the unresolvable-provenance blacklist: no "
                ".meta.json, no run claims its output, and its training run records "
                '"inputs": []. Its action format, window and sequence length cannot be '
                "established, so it must not enter a controlled comparison."
            )


def _run_context_path(root: Path, owner: str, run_id: str) -> Path:
    return root / "labctl_runs" / "runs" / owner / run_id / ".lab" / "context.json"


def resolve_checkpoint(
    checkpoint_dir: str | Path,
    *,
    labctl_root: str | Path = LABCTL_ROOT,
    max_hops: int = 6,
) -> CheckpointProvenance:
    """Walk a checkpoint back to the dataset manifest that defines what it learned.

    Raises:
        BlacklistedCheckpointError: the checkpoint is known-unresolvable.
        ProvenanceError: any hop is missing. This is deliberate — an unknown is
            worse than an error, because an unknown gets compared anyway.
    """
    root = Path(labctl_root)
    ckpt = Path(checkpoint_dir)
    assert_not_blacklisted(ckpt)
    if not ckpt.is_dir():
        raise ProvenanceError(f"checkpoint dir not found: {ckpt}")

    # --- hop 1: the checkpoint's own artifact metadata ----------------------
    meta_path = ckpt / ".meta.json"
    if not meta_path.is_file():
        raise ProvenanceError(
            f"{ckpt} has no .meta.json, so nothing records which run produced it. "
            "Provenance is unrecoverable from the filesystem; do not use this "
            "checkpoint in a controlled comparison (add it to "
            "rft.provenance.UNRESOLVABLE_CHECKPOINTS if it is permanent)."
        )
    meta = _read_json(meta_path, "checkpoint .meta.json")
    metadata = meta.get("metadata") or {}
    prov = CheckpointProvenance(
        checkpoint_dir=ckpt,
        artifact_id=meta.get("id"),
        owner=meta.get("user"),
        producer_recipe=metadata.get("producer_recipe"),
        producer_run_id=meta.get("producer_run_id"),
        step=metadata.get("step"),
        training_run_id=None,
        dataset_path=None,
    )
    prov.hops.append(Hop("checkpoint_meta", meta_path, detail=meta))
    if not prov.producer_run_id:
        raise MissingFieldError(f"{meta_path}.producer_run_id")
    if not prov.owner:
        raise MissingFieldError(f"{meta_path}.user")

    # --- hops 2-3: follow run contexts until a dataset input appears --------
    run_id: str | None = prov.producer_run_id
    for _ in range(max_hops):
        ctx_path = _run_context_path(root, prov.owner, run_id)
        ctx = _read_json(ctx_path, f"run context for {run_id}")
        prov.hops.append(Hop("run_context", ctx_path, detail={"run_id": run_id}))
        inputs = ctx.get("inputs")
        if inputs is None:
            raise MissingFieldError(f"{ctx_path}.inputs")
        if not isinstance(inputs, list):
            raise ProvenanceError(f"{ctx_path}.inputs is not a list")
        if not inputs:
            raise ProvenanceError(
                f'{ctx_path} records "inputs": [] - the run declares no inputs, so the '
                "dataset it trained on cannot be identified. Provenance stops here; "
                "this checkpoint must not enter a controlled comparison."
            )
        by_role = {str(i.get("role")): i for i in inputs if isinstance(i, dict)}
        prov.repo_head = (ctx.get("provenance") or {}).get("git_head") or prov.repo_head
        prov.repo_path = (ctx.get("provenance") or {}).get("repo_path") or prov.repo_path
        if isinstance(ctx.get("args"), dict):
            prov.train_args = {**ctx["args"], **prov.train_args}

        if "dataset" in by_role:
            prov.training_run_id = run_id
            prov.dataset_path = Path(str(by_role["dataset"].get("resolved_path")))
            break
        if "checkpoint" in by_role:
            # This run was an export/eval fed by another run's checkpoint. Hop to the
            # run that owns that checkpoint.
            upstream = Path(str(by_role["checkpoint"].get("resolved_path")))
            up_meta = upstream / ".meta.json"
            if not up_meta.is_file():
                # stream root rather than a step dir - try the parent
                up_meta = upstream.parent / ".meta.json"
            if not up_meta.is_file():
                raise ProvenanceError(
                    f"cannot hop past {ctx_path}: upstream checkpoint {upstream} has no "
                    ".meta.json"
                )
            up = _read_json(up_meta, "upstream checkpoint .meta.json")
            run_id = up.get("producer_run_id")
            prov.hops.append(Hop("checkpoint_meta", up_meta, detail=up))
            if not run_id:
                raise MissingFieldError(f"{up_meta}.producer_run_id")
            continue
        raise ProvenanceError(
            f"{ctx_path} has inputs with roles {sorted(by_role)!r} - neither `dataset` "
            "nor `checkpoint`, so the walk cannot continue"
        )
    else:
        raise ProvenanceError(f"provenance walk exceeded {max_hops} hops from {ckpt}")

    # --- hop 4: the dataset manifest ----------------------------------------
    if prov.dataset_path is None:
        raise ProvenanceError(f"no dataset input found walking back from {ckpt}")
    manifest_path = prov.dataset_path / "manifest.json"
    if not manifest_path.is_file():
        for alt in ("build_manifest.json", "tokenize_manifest.json"):
            candidate = prov.dataset_path / alt
            if candidate.is_file():
                manifest_path = candidate
                break
    prov.dataset_manifest = _read_json(manifest_path, "dataset manifest")
    prov.hops.append(Hop("dataset_manifest", manifest_path))
    return prov


def assert_comparable(
    provenances: Sequence[CheckpointProvenance], *, dimension: str
) -> dict[str, Any]:
    """Raise unless the checkpoints differ only in ``dimension``.

    The checkpoint-level twin of :func:`rft.arms.assert_arms_differ_only_in`. This is
    the guard that would have stopped a "goals vs no-goals" comparison whose members
    also differed in action format, window and sequence length.

    Returns the per-field value map, so the caller can record what actually varied.
    """
    from rft.arms import UncontrolledComparisonError

    if len(provenances) < 2:
        raise SchemaError("need at least 2 checkpoints to compare")
    fields: dict[str, dict[str, Any]] = {}
    for prov in provenances:
        for key, value in prov.decisive().items():
            fields.setdefault(key, {})[str(prov.checkpoint_dir)] = value

    confounds: list[str] = []
    for key, per_ckpt in fields.items():
        values = {json.dumps(v, sort_keys=True, default=str) for v in per_ckpt.values()}
        if len(values) > 1 and key != dimension:
            confounds.append(f"{key}: {per_ckpt!r}")
        if len(per_ckpt) != len(provenances):
            confounds.append(
                f"{key}: only {len(per_ckpt)}/{len(provenances)} checkpoints declare it "
                f"({per_ckpt!r}) - an undeclared field is an unknown, not a match"
            )
    if confounds:
        raise UncontrolledComparisonError(
            f"checkpoints are NOT comparable on {dimension!r} alone; these also differ "
            "or are undeclared:\n  - " + "\n  - ".join(confounds)
        )
    return fields
