from __future__ import annotations

"""Stage 10: near-duplicate removal.

Input:
- quality-thresholded rows from Stage 9

Output (`stage10.jsonl`):
- same rows with duplicate candidates rejected
"""

from omegaconf import DictConfig

from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s5_filter import FuzzyDedupStage


class DedupStage(Stage):
    name = "S10_dedup"

    def __init__(self, dedup_cfg: DictConfig):
        self.inner = FuzzyDedupStage(dedup_cfg)

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
