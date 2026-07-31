# Task-source interface audit

The adapters were designed against these read-only source revisions on
2026-07-31:

- OSWorld `xlang-ai/OSWorld@b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`
- CUA-Gym `xlang-ai/CUA-Gym@1e50b797200f8afd6f11ca8e3ee04412de97b0f2`
- CUA-Gym HF viewer schema for dataset `xlangai/CUA-Gym`, config `tasks`, split
  `train`

## OSWorld

The upstream index is an app-to-task-id mapping. Each task file is
`evaluation_examples/examples/<app>/<uuid>.json` and carries the instruction,
setup config, and evaluator config. `desktop_env.DesktopEnv` exposes
`reset(task_config)`, `step(action)`, and `evaluate()`; the in-guest server also
exposes screenshot, cursor position, screen size, and execution endpoints.

Stage 4 intentionally does not treat upstream `test_all.json` as training data.
The OSWorld source adapter requires a separate explicitly named train allowlist,
rejects eval/heldout-looking index names, hashes the allowlist and every selected
task file, and checks an externally supplied heldout identity/hash denylist.
The VM adapter is responsible for calling upstream setup/evaluate and returning
native cursor telemetry after every primitive.

## CUA-Gym

The HF train row includes `id`, `instruction`, `app_type`, `app_family`,
`platform`, `setup_kind`, `setup_files`, archive/member references, and reward
member. An extracted task bundle contains:

```text
<task-id>/task.json
<task-id>/reward.py
<task-id>/initial_setup.py|sh|document
```

The CUA-Gym environment utility exposes VM screenshot and setup execution/upload
operations. Web mocks use session-scoped state APIs; `reward.py` is the
task-specific programmatic verifier. Stage 4 hashes the index row's bundle,
requires the index instruction to equal the bundle instruction, and records the
setup/reward hashes without placing their content in prompts. The VM adapter
runs setup in an isolated session and returns the parsed numeric reward and an
explicit success boolean.

## Common boundary

Source adapters stop at canonical immutable task rows. VM lifecycle, teacher
transport, action execution, and reward evaluation are behind the same JSONL
RPC boundary. This keeps setup/evaluator semantics native to each backend while
making rejection, conversion, leakage checks, replay, and SFT packaging exactly
the same.
