# OEV3 action-space distillation experiments

## Suggestion 2 / Experiment A (now): native proposals, whole-action transport

Both models use the format they already understand.  Qwen3.5 samples native
absolute actions and Qwen3.8 scores those same native actions.  Only afterward
does the deterministic adapter convert each complete action to OEV3:

```
a_i ~ p_native_Qwen3.5(. | state, fixed reasoning)
t_i  = convert_native_to_OEV3(a_i, cursor)
w_i  = q_native_Qwen3.8(a_i) / p_native_Qwen3.5(a_i)
loss = -sum_i normalize(w_i) * log p_OEV3_Qwen3.5(t_i)
```

Sample at least four native actions after the **same fixed reasoning prefix**,
merge probability when several native candidates convert to the same OEV3
program, and apply loss only to converted OEV3 action tokens.  The implementation
probe is `eval/probe_native_rollout_transport.py`.

This is a self-normalized importance-sampling estimate of forward distillation
from the teacher's native action distribution into the OEV3 policy.  It is
online and sampled from the current Qwen3.5, but it is not strict "on-policy
reverse-KL" for the OEV3 prompt because the proposals were sampled under the
native prompt.  That distinction should remain explicit in experiment names.
Strict OEV3 reverse-KL would instead sample OEV3 actions from the OEV3 student
and map them backward for native teacher scoring.

### Stage-0 result (2026-08-31)

Slurm job 147294 was an initial **reverse-direction preflight**: a trained OEV3
Qwen3.5 sampled OEV3 actions which were mapped backward for Qwen3.8.  All 12
programs mapped losslessly despite unequal token lengths (student mean 18.3
action tokens, teacher mean 35.5).  This validated the converter and scoring
machinery, but it was not the native/native proposal experiment specified
above and must not be used as its result.

The corrected probe samples four native actions per state under one fixed
Qwen3.5 reasoning prefix, scores both native models, converts the candidates,
and reports the normalized transported weights and effective sample size.
It is packaged in the normal labctl harness as
`recipes/eval/native_oev3_transport_qwen38.toml` (in the labctl repository),
with declared model/data inputs and a registered `eval_result` output.

The optional recorded-action diagnostic is not a clean teacher-accuracy test:
Qwen3.8 scored both candidates after the student's sampled reasoning.  For
example, after the student explicitly reasoned that it should click the Insert
menu, the teacher quite reasonably preferred that click to a recorded action
from a different trajectory/history.  It must not be interpreted as evidence
that the mapping failed.  Real training must retain the actual rollout history;
the small stage-0 probe intentionally used only the current screenshot and
instruction.

Important implementation constraints:

- include the action terminator/EOS on both sides;
- never align native and OEV3 token positions;
- reject actions that cannot be losslessly converted;
- compute `q_native / p_native_old` from the complete native action sequence,
  in log space, and detach it before the update;
- use more than one proposal per fixed state/reasoning prefix; with one proposal
  the normalized weight is always 1 and the teacher contributes nothing;
- merge duplicate converted OEV3 programs before constructing the target;
- reduce per sequence, because native and OEV3 action token counts differ.

For the first training ablation, use self-normalized weights over four or eight
native proposals, clip the log importance ratio before exponentiation, average
**sequences**, not tokens, and mix in native/replay examples or a reference KL
to protect general capabilities.  Log effective sample size; if it repeatedly
collapses near 1, use more proposals or switch to Suggestion 1.

## Suggestion 1 / Experiment B (try later): transported top-K complete actions

This is the earlier "suggestion 1."  At each state, obtain roughly 16 likely
**complete native actions** from the teacher, convert every action to OEV3, sum
probability when several native strings map to the same OEV3 action, and train
the student on that soft distribution.  Build a prefix trie over converted
OEV3 token sequences to recover token-level targets without pretending native
and OEV3 digit positions align.

Do not substitute ordinary top-16 *tokens* at each native position: a digit's
meaning changes after absolute-to-relative arithmetic, and converted actions
can have different token lengths.  The existing finite-candidate proof is
`eval/probe_opd_action_transport.py`.

## Framework decision (2026-08-31)

Use the current upstream **Miles** as the intended continuous-online training
base, while keeping the first signal probe framework-independent.  Miles already has Qwen3.5/VLM
support, SGLang teacher serving, computer-use/agent environment connectors,
custom rollout and loss hooks, and ordinary OPD.  Our required extension is
small but fundamental: score a separately tokenized mapped native action and
attach its one scalar sequence score plus an OEV3 action-token mask.  Stock
Miles and stock slime both assume that teacher and student score the same token
positions.

The clean implementation is a custom weighted-sequence distillation loss:
native Qwen3.5 proposals and frozen native log-probabilities, native Qwen3.8
scores, deterministic conversion, then OEV3 teacher-forcing with detached
self-normalized importance weights.  Leave stock `--use-opd` off.  Miles'
built-in OPD and slime's built-in OPD assume aligned teacher/student token
positions and therefore cannot perform this semantic transport unchanged.

One important current limitation: Miles' convenient TITO *session* path cannot
yet carry images.  CUA-Gym therefore has to use Miles' lower-level multimodal
`/generate` plus a custom generate/rollout function (the same integration layer
used by its experimental HUD computer-use connector).  Miles still owns model
training, weight updates, batching, and advantages; our callback owns exact
token capture, screenshots, cursor state, conversion, and teacher scoring.

Other options:

- **KDFlow** is the strongest dedicated KD codebase and supports Qwen3.5 VLM,
  online KD, EMA self-teachers, and cross-tokenizer KD.  Its cross-tokenizer
  method still aligns two tokenizations of the same text, not two semantically
  equivalent action languages, and it has less mature agent-environment glue.
- **slime** is the shortest route from the local prototype and supports custom
  agent rollouts, but its built-in OPD stores position-aligned teacher logps.
  Miles now carries the more useful extension points for this experiment.
- **Relax** is attractive if large asynchronous omni-modal throughput becomes
  the bottleneck.  It is a larger Ray/Megatron/SGLang system than this initial
  experiment needs, and its built-in OPD has the same alignment assumption.
- **Omegalax** is useful for the existing SFT pipeline, but the local checkout
  lacks an online rollout/teacher/OPD loop.  It is nevertheless the shortest
  path to a **round-based first training experiment** in the existing harness:
  collect a frozen native proposal batch, write converted OEV3 records carrying
  sequence weights, add sample-weight support to its already available
  per-sample cross-entropy path, train a short LoRA round, then recollect.  This
  is transformed online distillation in rounds, not synchronous OPD.

Closest research references:

- GUI-SD (GUI grounding): https://arxiv.org/abs/2605.00642
- VLA-OPD (action models and forgetting): https://arxiv.org/abs/2603.26666
- GNDPO (batch-global stabilization): https://arxiv.org/abs/2606.09091
- TrOPD (teacher/student mismatch): https://arxiv.org/abs/2606.01249
- Cross-tokenizer OPD: https://arxiv.org/abs/2606.09456
- Miles OPD: https://miles.radixark.com/docs/advanced/on-policy-distillation
- KDFlow: https://github.com/songmzhang/KDFlow
- Relax: https://github.com/redai-infra/Relax
