from __future__ import annotations

"""Stage 9: aggregate quality thresholding.

Input:
- semantically judged rows from Stage 8

Output (`stage9.jsonl`):
- same rows with `quality.aggregate`
- `valid` / `reject_reasons` updated on quality threshold failures
"""

from omegaconf import DictConfig

from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s5_filter import QualityFilterStage


class QualityStage(Stage):
    name = "S9_quality"

    def __init__(self, thresholds: DictConfig, weights: DictConfig):
        self.inner = QualityFilterStage(thresholds, weights)

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
