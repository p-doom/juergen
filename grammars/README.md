# grammars

Seven action grammars, one directory each: `compact_absolute`, `compact_raw`,
`deltatype_v2`, `diffabs`, `move_rel`, `native_absolute`, `ordered_events_v3`.

## Adding a grammar

Create `grammars/<name>/` holding `codec.py` — a class with `parse` · `format` ·
`compile` · `describe` and a `stop_sequences` tuple, exported as `CODEC` — and
`vectors/<name>.json` pinning both directions and the lowering. `CODEC` must
satisfy `desktop.codec_protocol.Codec`.

The grammar's spec lives as docstrings on the codec: the class docstring is the
prompt preamble, each `@_support.production("syntax")` member's docstring is that
production's only specification, and `describe()` renders the system prompt from
them.

Add one line to `[project.entry-points."juergen.grammars"]` in the root
`pyproject.toml`. Nothing else in the repo enumerates grammars, and until the
package is reinstalled the directory is discovered by scanning anyway.

## What a codec owns

`compile(text, geometry, cursor)` is the only place a coordinate convention is
resolved, and it always returns absolute screen pixels clamped to the display.
There is no coordinate-space enum: the convention is an open record inside each
codec, and resolution context arrives as data through `compile`. A grammar that
needs richer context extends its own context struct, never a shared one.

`parse` serves eval and RL rollouts, `format` builds training targets, and the
vectors assert the round trip between them per grammar.

A codec's job ends at `compile`. It contributes no handler or dispatch table:
lowering an Operation is an `if kind ==` chain in
`desktop.execute.guest_program`, over a set no grammar extends.

The prompt digest is reported, never raised. `codec.digest` and `codec.report()`
return the rendered prompt's sha256, whatever producer provenance the grammar
recorded, and a `matches_producer` boolean. A mismatch is information.

## Matched pairs

`compact_absolute` and `compact_raw` differ only in whether the two leading
integers name a position or an offset. Change one and you must change the other,
or the comparison measures the change instead. Their shared prose therefore
lives once, in `_support.MATCHED_ARM_PREAMBLE` and friends, applied by
`_support.apply_matched_arm_prose(...)` after each class body;
`test_matched_arms_share_their_prose_byte_for_byte` pins it byte for byte and
pins that the only differing productions are the two mouse-triple ones. Each arm
declares the other via `PAIRED_WITH`.

## The Operation vocabulary

Every codec lowers to the same absolute-pixel IR, documented in `_support.py`:
`move_to`, `glide_to`, `mouse_down`, `mouse_up`, `scroll`, `key_down`, `key_up`,
`coalesced_type`, `wait`. The lift in the other direction accepts every kind in
`desktop.ir.CANONICAL_KINDS`: `group_operations` decomposes `drag` into
move · press · stroke · release, so a grammar with no stroke primitive raises on
one, and `ascii_type` flattens into the same `type` group as `coalesced_type` —
the one flattening in the lift, pinned as a documented-lossy vector.

## Registry

```python
import grammars

grammars.available()            # every registered grammar name
codec = grammars.load("diffabs")
codec.describe()                # the system prompt, from docstrings
codec.compile(text, geometry, cursor)
grammars.codecs()               # every codec, keyed by grammar name
```

## Running the vectors

```bash
uv sync --extra dev
uv run pytest grammars/
```

`grammars/test_vectors.py` is the gate: every case in every `vectors/*.json`,
all three directions plus the lift triangle, and it refuses to let a section
exist that no test runs.

The vectors use deltas that round-trip exactly, so the lossy region is pinned by
a separate test instead. Relative coordinates ride a thousandths grid: above
1000 px on an axis some deltas come back on the wrong pixel, and at 2000 px and
above a one-pixel delta normalises to zero and must raise. It must raise per
axis — a whole-delta check let a `(1, 100)` px move through as `[0, 50]`,
silently dropping the horizontal component.
