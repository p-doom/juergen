# Training and evaluation reproduction bundle (2026-07-31)

This branch collects source, labctl recipes, tests, preregistration material, and
small operational contracts from the relative-action training/evaluation effort.
It deliberately contains no model checkpoints, HF exports, datasets, run logs,
W&B state, caches, virtual environments, or generated Python metadata.

## Provenance

The committed files were reconstructed from the Juergen working tree and the
immutable labctl source snapshot for run
`run_019fb71771ee7e30b5259c3735b00587` (Slurm job 135676). Later finalized deltas
were checked against the recipes and run records for:

- relative-factorial capacity summary: job 135450;
- synthetic typing-factorial comparison: job 135667;
- proper-VM closed-loop pilot: job 135679;
- normalized Phase-B comparison: job 135685.

The first four branch-parent commits (`638be96` through `860bb66`) are the earlier
relative-factorial build/train/eval pipeline. The new commits keep independently
reviewable families separate: the RFT toolkit, relative-factorial analyses,
Phase-B pipelines, synthetic multistep/typing evaluation, and proper-task probes.

## Intentionally excluded

- `experiments/phaseb_deltatype_raw_v2/`: still an input to the active raw
  continuation/export chain when this bundle was cut;
- `experiments/phaseb/tests/test_final_export_eval_readiness.py`: a cross-arm
  contract test that pins files in that active raw subtree, to be added with the
  frozen raw handoff rather than left failing in this branch;
- `experiments/compact_scale_ablation/` and
  `experiments/typing_prose_factorial/`: still actively authored and tokenizing;
- `experiments/phaseb_deltatype_raw_inactive/`: superseded recovery material,
  not a reproducible final endpoint;
- caches, `*.pyc`, `*.egg-info`, `.pytest_cache`, artifacts, checkpoints, exports,
  datasets, and logs.

Those active trees should only be added after their owners hand off a frozen run
source snapshot and focused test results.
