from pathlib import Path

import pytest

from pipeline.lib.common import read_jsonl


def test_read_jsonl_requires_the_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "missing.jsonl")


def test_read_jsonl_rejects_blank_rows(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"row": 1}\n\n')
    with pytest.raises(ValueError, match=r"Blank JSONL row at .*:2"):
        read_jsonl(path)
