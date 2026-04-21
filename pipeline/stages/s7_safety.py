from __future__ import annotations

"""Stage 7: PII and content safety checks.

Input:
- rule-validated rows from Stage 6

Output (`stage7.jsonl`):
- same rows with `quality.pii_pass` and `quality.safety_pass`
- `valid` / `reject_reasons` updated on safety failures
"""

from pipeline.judges.protocol import Judge
from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s3_safety import SafetyStage


class SafetyValidationStage(Stage):
    name = "S7_safety"

    def __init__(self, judge: Judge):
        self.inner = SafetyStage(judge)

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
