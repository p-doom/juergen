# Juergen

Juergen contains Crowd-Cast data preparation, evaluation harnesses, and the
shared prompt renderer used by online and SFT computer-use runs. Model training
and desktop lifecycle are separate packages.

## Harness rendering

`HarnessRenderSpec` loads canonical JSON with an expected SHA-256 digest. A
renderer also verifies the system-prompt digest and the action and observation
contract names before accepting a frame.

`HarnessRenderer` retains `max_completed_turns` completed image/action pairs and
always appends the current image. Once the window is full, evicted action lines
move into the instruction's previous-actions text. Historical assistant text
has its completed `<think>` block removed.

`render_sft_records` uses the same renderer as online inference. It marks
historical assistant messages with `loss = false` and leaves only the current
assistant target trainable.

## Checks

```bash
uv run --with pytest pytest tests/test_harness_render.py
```
