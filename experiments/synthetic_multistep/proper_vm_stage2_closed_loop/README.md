# Roadmap stage 2 design draft: free-running synthetic proper-VM loop

This directory is design-only and is not launch-authorized. It is scientifically
separate from roadmap stage 1.5 endpoint-actuation conformance.

The primary condition uses all 80 heldout episodes and four ordered targets.
Every target receives at most three attempts. A valid model action moves and
clicks the real guest cursor; a miss retains the target and changes the next
cursor plus rendered cursor pixels, while a verified hit alone advances the
target. The next request therefore sees the actual on-policy state. Complete
model outputs—not oracle actions—form the bounded history.

The clustered primary is episode success: all four targets reached within the
three-attempt budget for each target. First-attempt target success and cumulative
success by attempts two and three are mandatory diagnostics, reported with both
risk-set and unconditional denominators. Transitions after terminal failure are
not silently dropped.

Finite-benchmark parity and inferential support are separate. The paired finite
difference must exceed -5pp; at 80 episodes, three unoffset absolute-only harms
pass and four fail. The separate conservative one-sided 95% exact support gate
passes only with zero absolute-only harms and is underpowered. If finite parity
passes while exact support fails, the report must say “finite parity but
inferentially unresolved”; the exact gate cannot silently redefine the 5pp
estimand.

A 320-cell, one-attempt single-step sentinel is run and reported separately. It
is never pooled with the free-running primary. This separation distinguishes
single-step grounding from retry recovery and cross-target state propagation.

The absolute preamble checkpoint is the matched evaluation control: it gets the
same VM dynamics, episode/target order, retry policy, seed slots, history length,
sampling tuple, and infrastructure gates as normalized and raw relative arms.
Its existing rank-32 versus relative rank-256 capacity difference remains an
explicit limitation, so the study is not framed as a pure action-format causal
effect. The contrast is best-pipeline parity, not causal format identification.

Roadmap stage-1.5 evidence may support only executor/provider/button mechanics.
Its fixed-cell pixels cannot stand in for the new single-step sentinel: dynamic
cursor pixels must be rendered and verified anew.

The launch-disabled implementation now includes a dynamic guest app, arm
runner, fail-closed paired aggregator, CPU/KVM dynamic smoke, and three prepared
one-GPU recipes. The host runner uses the pinned OSWorld venv while the model
server and readiness probe remain in the prime-rl environment, matching the
environment split proven during stage 1.5. Every guest app teardown validates
exact argv identity, TERM-polls, KILL-polls only if necessary, and proves no
same-source process remains.

Resumption is atomic at one fully terminated sentinel cell or one fully
terminated multi-step episode. A new job may explicitly import matching
complete units from an incomplete artifact; mid-episode state and loose rows
are never reusable. The aggregator independently replays all transitions,
model parses, endpoints, seeds, dynamic pixel hashes, guest target revisions,
button counts, terminal summaries, unit grids, and the merged rows hash before
emitting a paired report.

The strict maximum is 1,280 requests per arm: 320 sentinel requests plus 80 × 4
targets × 3 attempts. Planning at seven seconds per request plus 15 minutes for
model/VM startup gives roughly 165 minutes per arm, so each prepared recipe has
a three-hour wall. Early hits and terminal failures shorten this. The three arm
jobs can run in parallel after authorization.

This remains `design_draft_not_launch_authorized`. A live dynamic KVM smoke,
final source/hash/readiness report, decision-rule acceptance, and separate
roadmap-stage-2 GPU authorization are still required. None of the prepared GPU
recipes has been submitted.
