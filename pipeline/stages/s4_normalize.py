from __future__ import annotations

"""Stage 4: normalize Stage 3 rows into curator `Sample` records.

Input:
- `stage3.jsonl`

Output (`stage4.jsonl`):
- normalized Sample-shaped rows with `en_text`, `ko_text`, `metadata`,
  and `provenance`
"""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s0_generate import GenerateStage


class NormalizeStage(Stage):
    name = "S4_normalize"

    def __init__(self, cfg: DictConfig, out_jsonl: str | Path):
        self.inner = GenerateStage(cfg, out_jsonl)

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
