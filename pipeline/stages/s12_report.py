from __future__ import annotations

"""Stage 12: final reporting.

Input:
- reward-scored rows from Stage 11

Output (`stage12.jsonl`):
- compact batch summary row
- detailed report artifacts under `validation/`
"""

from pathlib import Path

from pipeline.schema import Record
from pipeline.stages.base import Stage
from pipeline.stages.s7_report import ReportStage


class ReportWriterStage(Stage):
    name = "S12_report"

    def __init__(self, out_dir: str | Path, chart: bool = True):
        self.inner = ReportStage(out_dir, chart=chart)

    def run(self, records: list[Record]) -> list[Record]:
        return self.inner.run(records)
