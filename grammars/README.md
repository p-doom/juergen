# Action grammars

Each peer directory exposes one module-level `CODEC` from `codec.py` and one
conformance-vector file under `vectors/`. The directory name is the grammar id;
there is no entry-point or alias registry.

- `deltatype_v2` is the canonical Crowd-Cast training grammar.
- `ordered_events_v3` is the normalized relative grammar used by the CUA-Gym
  action-format parity stream.

Both codecs parse, format, compile to `desktop.ir.Operation`, lift operations
back into training labels, and derive their system prompt from the same object.
Episode termination is a separate final `TERMINATE: success|failure` control
line handled by `grammars._support.split_control`.

Run the full conformance gate with:

```bash
uv run pytest -q grammars
```
