"""Small idempotent stage journal used inside resumable labctl stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage5_rft.util import ContractError, atomic_write_json, read_json, sha256_json


@dataclass(frozen=True)
class StageReceipt:
    stage: str
    input_sha256: str
    output_sha256: str
    status: str = "complete"

    def as_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


class StageJournal:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        return self.root / f"{stage}.receipt.json"

    def reusable(self, stage: str, inputs: Any) -> bool:
        path = self.path(stage)
        if not path.is_file():
            return False
        receipt = read_json(path)
        if receipt.get("status") != "complete":
            return False
        expected = sha256_json(inputs)
        if receipt.get("input_sha256") != expected:
            raise ContractError(
                f"stage {stage!r} has a receipt for different inputs; use a new run/output"
            )
        return True

    def complete(self, stage: str, *, inputs: Any, outputs: Any) -> StageReceipt:
        receipt = StageReceipt(
            stage=stage,
            input_sha256=sha256_json(inputs),
            output_sha256=sha256_json(outputs),
        )
        atomic_write_json(self.path(stage), receipt.as_dict())
        return receipt
