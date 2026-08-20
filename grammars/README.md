# grammars

## Adding a grammar

Create a peer directory `grammars/<name>/` holding `codec.py` — a class with
`parse` · `format` · `compile` · `describe` and a `stop_sequences` tuple,
exported as `CODEC` — and `vectors/<name>.json` pinning both directions and the
lowering. `CODEC` must satisfy `desktop.codec_protocol.Codec`;
`grammars/test_vectors.py` asserts `isinstance` for every registered grammar.
Write the grammar's spec as docstrings on the codec: the class docstring is the
prompt's preamble, each `@_support.production("syntax")` member's docstring is
that production's only specification, and `describe()` renders the system prompt
from them. Add one line to `[project.entry-points."juergen.grammars"]` in the
root `pyproject.toml`; nothing else in the repo needs to change, and until the
package is reinstalled the directory is discovered by scanning anyway.

## What a codec owns

`compile(text, geometry, cursor)` is the only place a coordinate convention is
resolved, and it always returns absolute screen pixels clamped to the display.
There is no coordinate-space enum: the convention is an open record inside each
codec, and resolution context arrives as data through `compile`. A grammar that
needs richer context — relative to the last click, relative to a detected
element — extends its own context struct rather than a shared one.

`parse` and `format` are members of the same object. `parse` serves eval and RL
rollouts, `format` builds training targets, and the conformance vectors assert
the round trip between them per grammar.

## Matched pairs

`compact_absolute` and `compact_raw` are a matched pair: identical prose
preamble, element vocabulary, line-extraction rule and canonical separator,
differing only in whether the two leading integers name a position or an offset.
Anything changed in one must be changed in the other, or the comparison measures
the change instead.

The shared prose is therefore not written twice. It lives in
`_support.MATCHED_ARM_PREAMBLE` and friends, and each arm calls
`_support.apply_matched_arm_prose(...)` after its class body. When each arm held
its own copy they drifted: the same sentence wrapped at a different column, a
different token sequence in the two arms.
`test_matched_arms_share_their_prose_byte_for_byte` pins it, and also pins that
the only differing productions are the two mouse-triple ones.

Each arm declares the other via `PAIRED_WITH` and `report()["paired_with"]`, and
a `matched_pair` section in `compact_absolute`'s vectors asserts that one
intent lowers to one operation sequence through both.

## What is never enforced

A prompt digest is reported, never raised. `codec.digest` and `codec.report()`
return the rendered prompt's sha256 alongside whatever producer provenance the
grammar recorded, plus a `matches_producer` boolean; a mismatch is information.
Raising `RuntimeError` on digest drift was tried; it made editing a grammar in
place more expensive than forking a worktree.

## The Operation vocabulary

Every codec lowers to the same absolute-pixel IR, documented in
`_support.py`: `move_to`, `glide_to`, `mouse_down`, `mouse_up`, `scroll`,
`key_down`, `key_up`, `coalesced_type`, `wait`. `_support.py` also holds the
scanners and helpers the bare-token and tool-call families share — a new
grammar may use them or ignore them, but adding one never edits them.

Codecs emit that subset; the lift accepts every kind in
`desktop.ir.CANONICAL_KINDS`. `drag`, `click` and `ascii_type` are desktop's
— it synthesises `click` itself — and while they fell through to "unknown
Operation kind" no recorded trajectory containing one could be lifted in any
grammar. `group_operations` decomposes `drag` into move · press · stroke ·
release, so the three grammars that can express a drag do, and the three with no
stroke primitive still raise. `ascii_type` flattens into the same `type` group as
`coalesced_type`, the single flattening in the lift, since no grammar here has
two typing primitives; it is pinned as a documented-lossy vector.

## No dispatch tables

A grammar contributes no handler table, and there is no place to put one. Each
grammar used to ship a `handlers.py` exporting
`HANDLERS: dict[str, Handler]`, described as its contribution to desktop's
dispatch engine. No such engine existed — nothing in desktop ever read those
tables — and the `Handler` they were annotated with runs in the opposite
direction (a parsed call to Operations, not a backend plus args to `None`).

A shared `match` over grammar-specific action names would be wrong, because that
set is open per grammar. A codec's job ends at `compile`, and the Operation
vocabulary on the far side is closed: a pointer moves, a button transitions, a
wheel turns, text arrives. Lowering it is an `if kind ==` chain in
`desktop.execute.guest_program`, over a set no grammar extends.

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
pip install -e '.[dev]'
pytest grammars/                # every vector, one test per case
```

`grammars/test_vectors.py` is the gate. It executes every case in every
`vectors/*.json` — all three directions plus the lift triangle — and refuses to
let a section exist that no test runs. It also asserts the four invariants that
are not vector-shaped: `isinstance(codec, Codec)` for all seven, the matched
pair's byte-identical prose, that every canonical Operation kind can be grouped,
and `move_rel`'s quantisation ceiling across five screen sizes.

The vectors use deltas that round-trip exactly, so the lossy region — where a
relative label is quietly wrong — is pinned by that last test instead. Above
1000 px on an axis the thousandths grid is coarser than pixels and some deltas
come back on the wrong one; at 2000 px and above a one-pixel delta normalises to
zero and must raise. It must raise per axis: testing only the whole delta let a
`(1, 100)` px move through as `[0, 50]`, silently dropping the horizontal
component.
