# ROADMAP 3.5 coarse official-pilot preregistration

Status: infrastructure only. The official held-out source has not been opened,
enumerated, materialized, inspected, hashed, or executed while preparing this
package. No result exists. This contract becomes eligible only after signed
PASS artifacts for ROADMAP 3.1, 3.2, 3.3, and 3.4.

## Purpose and source separation

This is a small diagnostic of transfer to the untouched official task source;
it is not a training gate and cannot authorize a broad benchmark run. A broker
outside this repository owns source access. This repository ships only a Python
protocol for that broker and intentionally ships no filesystem adapter, split
reader, task selector, task manifest, source path, task identifier, or official
task hash.

After both release gates validate, the broker selects eight distinct task
clusters with the frozen `broker-sealed-stratified-without-replacement-v1`
policy. Build and development processes receive neither selection contents nor
a source capability. The sanitized analysis rows use only zero-based opaque
cluster indices.

## Arms, seeds, resets, and ordering

Every selected cluster is run under the same two arms:

1. `native_absolute_control`: the pinned native absolute-action control.
2. `compact_raw_phaseb`: the pinned compact raw-delta Phase-B export.

Seeds 3501 and 3511 are paired across arms for every cluster. They are rollout
and deterministic setup seeds, not task-selection seeds. Each episode restores
the same pinned `osworld_ready` snapshot before setup. Arm order is
counterbalanced by cluster index and seed index; reset ordinal 1 is the first
arm and reset ordinal 2 the second. Any missing reset, setup, oracle evaluation,
seed, arm, or pair invalidates the entire aggregate rather than becoming a task
failure or an exclusion. The frozen maximum is 8 clusters x 2 seeds x 2 arms =
32 episodes. A larger run requires a new signed contract.

## Sanitized row trace

One row is required per episode. It records pilot ID, opaque cluster index,
paired seed/key, arm and arm order, reset protocol/ordinal/status, setup status,
oracle-evaluated status, episode parse/executor/task success, termination class,
and a bounded step trace. Each step contains only its index, parse success,
executor success, registered action class, ineffective-action flag, and a
registered error code.

The schema rejects unknown keys. Consequently instructions, official task IDs,
paths, expected states, screenshots, raw model text, raw actions, exception
messages, task hashes, and source metadata cannot enter aggregation artifacts.
`task_success` is the final official oracle result; termination text cannot set
it. Parse success, executor success, and task success remain separate metrics.

## Estimand and analysis

The primary estimand is the paired difference
`compact_raw_phaseb - native_absolute_control` in task success. Arm success
rates receive Wilson 95% intervals. The paired difference receives a
preregistered hierarchical percentile bootstrap: resample eight task clusters,
then resample two paired seeds within each sampled cluster, with 10,000 draws
and PRNG seed 20260731.

The diagnostic passes the noninferiority gate only when all three criteria hold:

- the bootstrap 95% lower bound for the paired difference is strictly above
  -0.15;
- compact-raw task success is at least 0.60; and
- compact-raw parse-or-executor failure is at most 0.05.

No alternative confidence interval, seed, margin, success floor, failure
ceiling, exclusion, arm, or sample-size expansion may replace these values
after results are seen. A pass authorizes reporting this coarse diagnostic only.

## Signed release gates

Two canonical JSON payloads, each with an OpenSSH detached signature in the
`juergen-proper-vm-release-gate-v1` namespace, are mandatory:

- the prerequisites gate records PASS for exactly ROADMAP 3.1--3.4;
- the pilot-release gate names its parent gate and repeats this complete frozen
  design, including source protocol, selection policy, arms, seeds, episode
  cap, bootstrap settings, and noninferiority thresholds.

Both signatures, identities, timestamps, scopes, and cross-references are
verified before any injected source factory or sanitized row loader is called.
The OpenSSH allowed-signers trust anchor is a separate read-only labctl input,
not a file supplied alongside the release gates.
Gate payloads reject official-detail keys, absolute paths, file URIs, and
SHA-256-looking values. Missing, expired, noncanonical, tampered, symlinked, or
contract-mismatched artifacts fail closed.
