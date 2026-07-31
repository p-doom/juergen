# VM adapter protocol

The collector starts one adapter process per rollout and exchanges one compact
JSON object per line on stdin/stdout. Every reply is:

```json
{"id": 1, "ok": true, "result": {}}
```

Errors use `ok:false` and a non-sensitive `error` string. Logs belong on stderr.

Required methods:

- `reset {task, work_dir?}`: restore a fresh VM/session, verify the task row's
  artifact hashes, run native setup, and return an observation.
- `step_native {action}`: execute one absolute native action and return
  `{observation, trace}`. Trace must contain integer `cursor_before`,
  `cursor_after`, and `resolved_target_px` (null only for coordinate-less
  actions). Do not reconstruct a missing cursor from prior model coordinates.
- `step_compact {sequence}`: execute each compact line in order and return the
  final observation.
- `reward {}`: run the native programmatic evaluator and return
  `{reward: 0..1, success: bool}`. A crashed/malformed reward is an adapter error,
  never implicit success.
- `close {}`: destroy the throwaway VM/session and release ports.

An observation is:

```json
{"image_path":"/absolute/local/screenshot.png","cursor":[x,y],"screen_size":[w,h]}
```

The path must remain readable until the caller copies it. The adapter must use
fresh snapshots/session ids and must never mutate the registered task-manifest
artifact. OSWorld adapters call native reset/evaluate; CUA-Gym adapters run the
bundle setup/reward scripts and isolate mock-web session ids. Construction
adapters must reject any task whose `source_split` is not exactly `train`.
