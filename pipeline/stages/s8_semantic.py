from __future__ import annotations

"""Stage 8: semantic and property judging.

Input:
- safety-cleared rows from Stage 7

Output (`stage8.jsonl`):
- same rows with semantic/property quality fields populated
- `valid` / `reject_reasons` updated on semantic failures
"""

from omegaconf import DictConfig

from pipeline.judges.protocol import Judge
from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s4_semantic import SemanticJudgeStage


class SemanticStage(Stage):
    name = "S8_semantic"

    def __init__(self, judge: Judge, thresholds: DictConfig):
        self.inner = SemanticJudgeStage(judge, thresholds)

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
