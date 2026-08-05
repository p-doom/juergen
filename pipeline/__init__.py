"""`pipeline` — crowd-cast screen recordings → computer-use SFT records.

The current generation, moved to the repo root out of
``data_pipeline/realigned_pipeline`` (and ``lib/realign_lib.py`` → ``lib/realign.py``).
Flat and stage-numbered successor to the annotation_pipeline / realignment_fix
split: each ``stage_NN_*.py`` is one runnable stage (00 clip manifest → 06
training records), shared code lives in ``pipeline.lib``, and the per-method
annotation subsystem lives in ``pipeline.annotation``.

The alignment logic (``lib/realign.py``, ``lib/events.py``, ``lib/frames_actions.py``,
``lib/common.py``) is welded to the capture format and is correct — the restructure
moved it, it did not touch it.
"""
