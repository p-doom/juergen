# Relative-format LoRA capacity follow-up: preregistration

Frozen before submission on 2026-07-30. This follow-up changes only LoRA
capacity for the four relative-format cells. It does not retrain any absolute
cell.

## Design

- Cells: `relraw_act`, `relraw_pre`, `reltool_act`, and `reltool_pre`.
- Capacity levels: rank/alpha 32/32 (completed reference), 64/64, and 256/256.
- New runs: one independent rank-64 run and one independent rank-256 run per
  cell (eight runs total).
- Held fixed: Qwen3-VL-8B-Instruct base, the registered 2,000/200-record
  train/validation artifact, seed 0, 750 optimizer steps, maximum length 4096,
  batch size 1, gradient accumulation 8, learning rate 1e-4, WSD schedule,
  warmup 30, stable fraction 0.7, weight decay 0.01, and all other optimizer,
  data, and evaluation settings.
- Evaluation: the same 80 audited scenes and the grammar/preamble condition
  matching each cell. Overall and long-scene accuracy are paired by scene
  against the completed rank-32 reference.

## Primary decision rules

For each new capacity level, average the four cell accuracies with equal cell
weight. A **capacity response** is present if either the mean overall accuracy
improves by at least 5 percentage points over rank 32 or the mean long-scene
accuracy improves by at least 10 percentage points over rank 32. Report both
criteria independently; no significance gate is substituted for these
predeclared effect-size gates.

**Practical parity** is reached if mean overall accuracy is at least 95% and
mean long-scene accuracy is at least 90% at a new capacity level. Both gates
must pass at the same capacity level.

## Required reporting

Report overall, short, and long accuracy for every cell at every rank; paired
changes from rank 32; equal-cell means; the two capacity-response gates; and
the practical-parity gates. Also report capacity-by-grammar,
capacity-by-preamble, and capacity-by-grammar-by-preamble interactions using
the same effect coding as the original 2x2 relative factorial analysis.

No threshold, cell weighting, scene subset, or reference rank will be changed
after observing the new evaluations.
