# Action grammars

`grammars.__init__` registers exactly two module-level codecs. Each has one
conformance-vector file under `vectors/`; checkout-local directories cannot
become runtime grammars.

- `deltatype_v2` is the canonical Crowd-Cast training grammar.
- `ordered_events_v3_relative_1000_grid_v1` is the normalized relative grammar
  used by the CUA-Gym action-format parity stream. The former pixel-relative
  `ordered_events_v3` identity is unsupported.

Both codecs parse, format, compile to `desktop.ir.Operation`, and derive their
system prompt from the same object. Episode termination is a separate final
`TERMINATE: success|failure` control line handled by
`grammars._support.split_control`.

Run the full conformance gate with:

```bash
uv run --locked --extra dev pytest -q tests/test_grammar_vectors.py
```
