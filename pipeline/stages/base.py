from __future__ import annotations
from abc import ABC, abstractmethod
from pipeline.schema import Record


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def run(self, records: list[Record]) -> list[Record]:
        ...


class RecordStage(Stage):
    """Per-record stage. Already-rejected records skip processing."""

    def run(self, records: list[Record]) -> list[Record]:
        return [self.process(r) if r.valid else r for r in records]

    @abstractmethod
    def process(self, record: Record) -> Record:
        ...


class BatchStage(Stage):
    """Stage that needs the whole batch (dedup, threshold, report)."""
