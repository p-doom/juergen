"""`pipeline.crowdcast` — crowd-cast screen recordings → computer-use SFT records.

Each ``stage_NN_*.py`` is one runnable stage (00 clip manifest → 06 training
records), shared code lives in ``pipeline.crowdcast.lib``, and the per-method
annotation subsystem lives in ``pipeline.crowdcast.annotation``.

Stage 03 has two independent readings of the master frame axis, and stage 04
accepts either:

  * ``stage_03_filter``         — a keep/drop mask at master resolution, fps
    decided downstream (``lib/views.build_view``). Required by
    ``annotation/stage_annotate`` (stage 03b), which annotates a filter view.
  * ``stage_03_sample_frames``  — samples to a target fps up front and emits
    ``frame_records.jsonl`` with per-frame labels, NO_OP thinning
    (``--noop-mode``) and foreground-app tags (``lib/app_context``).
"""
