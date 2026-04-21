from __future__ import annotations

"""Stage 6: rule-based validation.

Input:
- schema-validated rows from Stage 5

Output (`stage6.jsonl`):
- same rows with heuristic validation applied
- `valid` / `reject_reasons` updated on rule failures
"""

from omegaconf import DictConfig

from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s2_rules import RuleValidatorStage


class RulesStage(Stage):
    name = "S6_rules"

    def __init__(self, thresholds: DictConfig):
        self.inner = RuleValidatorStage(thresholds)

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
