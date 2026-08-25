"""`pipeline.finevision` — FineVision/GroundUI → canonical chat.jsonl.

A replay SOURCE, not a stage chain: `prep.py` is the whole corpus. It emits the
same `chat.jsonl` shape `pipeline.crowdcast`'s stage 04 does, which is what lets
`stage_06_training_records` build records from either without knowing which
produced them.

Its own package because it shares no code with the crowd-cast stages — no
keylog, no realignment, no master frame store, no grammar. What the two share is
the output contract, and that is deliberate.
"""
