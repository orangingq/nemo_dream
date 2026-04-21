from __future__ import annotations

"""Stage 11: reward scoring.

Input:
- deduplicated rows from Stage 10

Output (`stage11.jsonl`):
- same rows with `quality.reward`
"""

from pipeline.judges.protocol import Judge
from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s6_reward import RewardStage


class RewardScoringStage(Stage):
    name = "S11_reward"

    def __init__(self, judge: Judge):
        self.inner = RewardStage(judge)

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
