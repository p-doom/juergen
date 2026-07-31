# Rung 1 synthetic capability curriculum preregistration

Status: prepared, validation-gated, and **not launched**. This curriculum is not
part of the rung1 mechanics result and does not authorize opening the sealed
evaluation pages or running a model against them.

## Frozen matrix

`curriculum_spec.json` is the controlling matrix. For each deterministic scene,
it creates an exact pair: `native_absolute_control` and
`compact_raw_phaseb`. The instruction, initial pixels, control parameters,
seed, and final oracle are identical; only the system/action grammar and the
necessary low-level drag turn count may differ.
The spec's `validation` partition is the curriculum development split; it is
never the sealed rung1 evaluation split.

| capability | train scenes/format | validation scenes/format | high-level steps |
|---|---:|---:|---:|
| click | 256 | 32 | 1 |
| focus + exact Unicode type | 256 | 32 | 1 |
| signed scroll | 128 | 16 | 1 |
| explicit hold/move/release drag | 256 | 32 | 1 |
| composition | 256 | 32 | 2 |
| composition | 256 | 32 | 3 |
| composition | 256 | 32 | 4 |
| **total per format** | **1,664** | **208** | |

The two seed namespaces begin at 310000 and 410000. The builder compares every
generated seed and control-parameter fingerprint against the sealed rung1
manifest and aborts on overlap. It also requires train/validation scene
fingerprint overlap to be zero. It reads sealed metadata only for this negative
set-membership test; no evaluation page is rendered and no evaluation content
enters a message or image.

Compositions are deterministic permutations/subsets of the four primitive
capabilities. Every screenshot visibly marks prior completed steps, so each
next-action target is grounded in the current state rather than hidden history.
Compact click events use adjacent `+LMB -LMB`; compact drags must retain separate
press, motion, and release turns. Unicode is carried only through `type(...)`.

## Non-negotiable prelaunch gates

`curriculum.py` renders newly authored raster controls and writes canonical
Omegalax `chat.jsonl` records. Before publishing `build_manifest.json`, it
requires:

- zero sealed-fixture seed and parameter-fingerprint overlap;
- zero train/validation seed and scene-fingerprint overlap;
- exact native-tool and compact-raw parser round trips for every action;
- identical seed/instruction/initial pixels/parameters/oracle across twins;
- 100% positive final-oracle acceptance and 100% one-step near-miss rejection;
- final pointer mask zero for every trajectory, exact Unicode typing, and
  explicit non-coalesced drag holds;
- exact matrix counts.

The CPU build/tokenize stage then requires exact 1,664/208 metadata counts per
format, max length 16,384, and zero split/truncation/drop statistics. It writes
`build_tokenize_manifest.json` last. GPU stages consume only that marker-bearing
stage artifact; they cannot consume the source build directly.

## Reused training path

The prepared labctl pipeline uses the existing Omegalax
`build_sft_records_from_chat.py` and `train_vlm_sft.py` paths. Each of the two
matched format cells trains Qwen3-VL-8B with LoRA r32/alpha32 for 750 steps,
seed 0, the same optimizer schedule, and the same dataset counts. This is a
capability warm-start, not a scientific format comparison: no sealed model
evaluation or proper-VM ladder run is chained after training.

No CPU build, tokenization, GPU training, export, or evaluation job is submitted
by these files. Launch requires explicit owner authorization after the local
unit suite passes and after reviewing the materialized invariant and
tokenization reports.
