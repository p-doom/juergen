"""Round-trip the sealed Phase-B dataset through the committed codec.

Equivalence proof for the model-facing contract: every one of the 10,721
assistant action spans in the sealed artifact must parse and re-format to the
identical bytes, the committed ``SYSTEM_PROMPT`` must be byte-identical to the
one in all 2,616 records, and ``ordered_plan`` must reproduce every recorded
command plan when each task's cursor is replayed from the screen centre.

Skips when the sealed dataset is not mounted (it is ~10 MB of JSONL plus image
references and is not committed). Override the location with
``PHASEB_SEALED_DATASET``.
"""

from __future__ import annotations

import pytest

from conftest import ROOT, external_root
from verify_sealed_dataset import DEFAULT_DATASET, verify

DATASET = external_root("PHASEB_SEALED_DATASET", str(DEFAULT_DATASET))


@pytest.mark.skipif(
    not (DATASET / "train" / "chat.jsonl").exists(),
    reason=f"sealed dataset not mounted at {DATASET}",
)
def test_sealed_dataset_roundtrips_byte_exactly():
    summary = verify(DATASET, ROOT / "vendor")
    assert summary["records"] == 2616
    assert summary["assistant_spans_roundtripped"] == 10721
    assert summary["unique_decisions_plan_checked"] == 2616
    assert summary["tasks"] == 239
    assert summary["manifest_cross_checked"] is True
