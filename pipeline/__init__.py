"""`pipeline` — the data-preparation sources, one package per corpus.

  * ``pipeline.crowdcast``  — crowd-cast screen recordings → computer-use SFT.
  * ``pipeline.finevision`` — FineVision/GroundUI prep.

Nothing imports this package: labctl invokes the stages as file paths in a
checkout, and each stage entrypoint puts the repo root on ``sys.path`` itself.
It exists so the two corpora are siblings rather than one shadowing the other.
"""
