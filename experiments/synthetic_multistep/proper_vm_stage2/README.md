# Roadmap stage 1.5: proper-VM endpoint-actuation conformance

This is explicitly **not user roadmap stage 2** and **not a free-running
multi-step closed loop**. The legacy `proper_vm_stage2` filesystem path predates
the corrected roadmap classification and is not a scientific stage label.
Every model request uses an independently reset frozen cell; its action cannot
update the next observation, cursor, seed, or oracle-prefix context.

This roadmap stage-1.5 bridge preserves the preregistered Phase-A
geometry population: all 80 episodes and all four target transitions, for 320
paired cells. An 80-cell step-1 subset is deliberately not used because the
normalized result is close enough to a 5 percentage-point margin that the
subset would be dominated by threshold noise.

The three frozen arms are the primary absolute Phase-A action-only checkpoint,
the normalized Phase-A r256 preamble checkpoint, and the canonical original
raw A-to-B curriculum checkpoint. The later 5e-5 A-to-B run remains a
sensitivity and does not replace the registered original model.

For each cell the guest shows the exact frozen 1920x1080 PNG in a borderless
full-screen window with the real cursor hidden. The physical guest cursor is
placed at the frozen start coordinate. Before inference, decoded screenshot
pixels must hash exactly to the frozen image. The three arms receive the same
cell order and the existing semantic-independent request seed.

One parsed model endpoint is reset-and-replayed twice through real guest
`pyautogui`: first as a move plus click, then as mouse-down, a nonzero-duration
move, and mouse-up. Absolute uses `moveTo`, normalized uses `moveRel` after the
registered 0--999 conversion, and raw uses its clipped pixel endpoint. Guest
event state, final cursor, released-button state, and geometry are read back.
Any infrastructure mismatch invalidates the arm; parsing/schema/unit errors and
geometric misses are policy failures.

The primary cell endpoint is compound click-and-drag success. Both normalized
minus absolute and raw-A-to-B minus absolute must exceed the -5pp margin. The
confirmatory sensitivity is deliberately conservative: the one-sided 95%
Clopper--Pearson upper bound on absolute-success/treatment-failure discordance
must be below 5%; treatment-only successes are ignored. Both contrasts must
pass (an intersection-union test, so no alpha split is needed). The inference
scope is the finite frozen geometry benchmark. Episode-cluster and target-index
summaries remain secondary generalization diagnostics.

The drag replay is an endpoint-transfer test, not a claim that these models
natively generated a drag grammar: their matched synthetic supervision taught
target endpoints. This limitation is explicit so that a passing result cannot
be overread as full OSWorld action-policy parity.

Minimum execution plan: one H100-80GB, 32 CPU, 128GB RAM, one persistent KVM VM,
and three serial model-server loads. That is 960 model requests and 1,920 VM
action replays, expected 45--75 minutes with a two-hour hard wall. No checkpoint
is copied and output is capped at 5GB. The faster alternative is three parallel
one-GPU arm jobs plus a CPU aggregator.

Run the CPU-only preparation gate with:

```bash
python experiments/synthetic_multistep/proper_vm_stage2/gate.py
```

This does not launch a VM or GPU. `live_smoke.py` and its CPU-only prepared
labctl recipe provided the next no-model gate. Run
`run_019fb64ec53771f18a6a6caface915cd` / job `135599` passed all six real
KVM replays (three action semantics by click/drag), with exact decoded screenshot
equality and released-button/final-cursor state readback. Its frozen manifest is
recorded in `protocol.json`.

The three one-GPU arm recipes validate but are launch-disabled pending recovery
proof and fresh authorization. Their recipe names and output aliases identify
roadmap stage 1.5, although the legacy filenames remain stable:
`proper_vm_stage2_absolute_prepared.toml`,
`proper_vm_stage2_normalized_prepared.toml`, and
`proper_vm_stage2_raw_a_to_b_prepared.toml`. Every exported 35.07 GB weight file
is byte-hash sealed in the protocol. The wrapper performs a read-only protocol,
checkpoint, no-leak, VM, smoke, and explicit-authorization preflight before
querying `nvidia-smi` or starting vLLM.

The first environment-split recovery reached exactly 115 rows in all three arms
and then exhausted the guest X server's client slots. Each cell had created two
detached Tk scenes; teardown sent SIGTERM but did not wait for exit. Those three
115-row files are quarantined and cannot be resumed or aggregated. The
execution-only amendment now validates exact argv identity, sends TERM, polls,
uses KILL only as a fallback, polls again, and asserts that no exact-source guest
process remains before a new scene starts. A CPU/KVM lifecycle stress must pass
at least 131 cells / 262 scenes, beyond the failed 115 / 230 boundary, with a
bounded `xlsclients` inventory before any new GPU authorization. Fixed fresh-VM
chunks of at most 80 cells remain a predeclared fallback only if hardened
teardown cannot pass that stress. The estimand, frozen cells, seeds, model
checkpoints, thresholds, and aggregator are unchanged.

The protocol is currently `prepared_not_launched` with
`launch_gate.authorized=false`. This does not authorize another stage-1.5 arm
retry or any true roadmap stage-2 study. All three eventual arms must use the
identical final reauthorized protocol hash.

After three complete arm artifacts exist, `aggregate.py` revalidates every row,
recomputes geometry and compound success, checks paired cell/seed/pixel identity,
and applies both prespecified 5pp contrasts. It refuses partial, drifted, or
unauthorized inputs and writes `paired_report.json` only after the complete
intersection-union result is available.
