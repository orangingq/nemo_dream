from __future__ import annotations

"""Stage 5: schema validation.

Input:
- normalized Sample-shaped rows from Stage 4

Output (`stage5.jsonl`):
- same rows with `sample` validation applied
- `valid` / `reject_reasons` updated on schema failures
"""

from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s1_schema import SchemaValidationStage


class SchemaStage(Stage):
    name = "S5_schema"

    def __init__(self):
        self.inner = SchemaValidationStage()

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
