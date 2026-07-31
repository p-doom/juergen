# Stage 4: native-absolute teacher → compact-relative SFT

This package is the train-only supply chain for collecting successful native
absolute-action demonstrations and converting them into compact raw-pixel
relative supervision. It is backend-neutral across OSWorld and CUA-Gym and
produces hash-addressed inputs for Omegalax SFT plus train-derived replay/eval.

No teacher model, GPU training, official heldout task, or official evaluation
was run while building this package. The labctl recipes are prepared launch
infrastructure, not execution records.

## Pipeline and gates

1. `build-tasks` reads explicit `source_split="train"` indexes. OSWorld needs a
   train allowlist; it never defaults to `test_all.json`. CUA-Gym needs the
   public `tasks/train` index and extracted task bundles.
2. `collect` assigns each candidate to an isolated newline-JSON VM worker. The
   teacher must be deterministic (`temperature=0`) and emit native absolute
   actions. Every executed primitive records cursor-before, resolved target,
   cursor-after, viewport, screenshots, and hashes.
3. `reject` requires a finite reward at or above threshold, environment
   success, successful termination, zero parse/runtime errors, and membership
   in the canonical task manifest. It deterministically keeps the best N.
4. `convert` checks every teacher coordinate against VM telemetry. It rejects
   clipping, missing cursor state, target disagreement, unbalanced input state,
   unknown actions, and non-canonical output. It symbolically replays the
   converted program and emits no completion manifest if any accepted rollout
   is ambiguous.
5. `replay` runs converted programs in fresh train-derived environments and
   requires the original programmatic reward. This is a construction gate, not
   official evaluation.
6. `build-sft` re-applies the heldout hash denylist, enforces task-disjoint
   `train` / `train_validation`, and writes chat rows plus row/image/source
   hashes. Reward, success, setup code, paths, and provenance never enter model
   visible text.
7. The prepared labctl tail tokenizes, trains, exports HF weights, and evaluates
   only `train_validation`. There is deliberately no heldout stage.

All stages write `manifest.json` last. A quarantine or partial directory without
that marker is not a valid artifact.

## Compact action contract

Each line is the established raw-relative grammar:

```text
NO_OP | TERMINATE | FAIL
dx dy scroll
dx dy scroll ; +KEY -KEY type("JSON escaped text")
```

A model turn may contain multiple lines executed in order. This is the minimum
backward-compatible extension needed for a real drag:

```text
0 0 0 ; +LMB
120 -30 0
0 0 0 ; -LMB
```

Collapsing that sequence to `120 -30 0 ; +LMB -LMB` would click at the target;
it would not drag. Clicks remain one line, scroll remains the third integer, and
teacher `type` stays one coalesced `type("...")` element. The shared eval parser
and `OSWorldClient.dispatch_deltatype` execute the same contract.

## Source spec

Paths may be absolute or relative to `source_spec.json`:

```json
{
  "schema_version": 1,
  "split_seed": "stage4-v1",
  "validation_fraction": 0.1,
  "sources": [
    {
      "kind": "osworld",
      "source_split": "train",
      "source_revision": "<git sha>",
      "task_index": "osworld_train.json",
      "task_root": "OSWorld/evaluation_examples/examples"
    },
    {
      "kind": "cua_gym",
      "source_split": "train",
      "source_revision": "<dataset revision>",
      "task_index": "cua_tasks_train.jsonl",
      "bundle_root": "cua_gym_tasks"
    }
  ]
}
```

The heldout denylist is mandatory, even when empty. It contains only opaque
`task_keys`, `source_task_ids`, `instruction_sha256`, and `asset_sha256`; Stage 4
does not inspect heldout definitions to construct it.

## Commands

```bash
python -m experiments.teacher_sft.cli build-tasks \
  --source-spec source_spec.json --heldout-denylist denylist.json --output tasks
python -m experiments.teacher_sft.cli collect \
  --tasks tasks --teacher-spec teacher.json --env-command 'python adapter.py' --output rollouts
python -m experiments.teacher_sft.cli reject \
  --tasks tasks --rollouts rollouts --output rejection
python -m experiments.teacher_sft.cli convert --rejection rejection --output converted
python -m experiments.teacher_sft.cli replay \
  --converted converted --env-command 'python adapter.py' --output replay
python -m experiments.teacher_sft.cli build-sft \
  --converted converted --heldout-denylist denylist.json --output sft
```

The backend protocol is in `VM_ADAPTER_PROTOCOL.md`; source-interface findings
are in `SOURCE_INTERFACES.md`. Run the CPU suite with:

```bash
uvx --with pytest pytest -q experiments/teacher_sft/tests eval/test_deltatype_sequence.py
```

The full prepared DAG is `labctl/pipeline.toml`. Register the source spec,
denylist, teacher spec, and VM adapter aliases named by the recipes before any
submission. Review all cluster paths and model pins; the examples intentionally
contain no credentials.
