# Crowd-Cast sign-of-life VM gate v2

This is one fixed development-gate suite, not a final benchmark and not a
train/development/test split. It replaces qualitative freerolls with four
deterministic cells whose success is decided from realized guest state:

- execute `ls` in an already-focused terminal and observe both exact shell
  history and a unique directory-listing marker in terminal output;
- enter one exact supplied paragraph into a focused terminal capture;
- click Chrome in the desktop dock and verify a Chrome process is the active
  foreground window;
- focus a visible-but-unfocused terminal, enter an exact command, and verify
  the resulting file bytes and shell history.

The runner stops as soon as a postcondition is reached, while recording model
termination separately. Every task stores before/after and per-step screenshots,
raw model output, parsed actions, atomic executor receipts, state evidence, and
errors. Missing or ambiguous evidence fails closed.

Three labctl recipes exercise the same suite:

- `sign_of_life_v2_oracle_cpu_kvm.toml`: scripted native-absolute gold actions;
- `sign_of_life_v2_negative_cpu_kvm.toml`: wrong text/no-op/wrong-click controls;
- `sign_of_life_v2_offshelf_native_absolute_gpu_kvm.toml`: off-the-shelf
  Qwen3-VL-4B-Instruct using the native absolute computer-use tool format. Run
  this recipe with `labctl run-sweep`; its four jobs each own one GPU and one
  reset-isolated VM while preserving the single fixed suite definition.

The scripted and model arms both compile into the same `Operation` stream and
execute through `HttpVmTransport.execute_atomic`; transport acknowledgement is
never used as task success.
