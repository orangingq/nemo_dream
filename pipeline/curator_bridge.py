from __future__ import annotations
import pandas as pd
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch
from pipeline.schema import Record
from pipeline.stages.base import Stage


RECORD_COL = "_record_state"


class StageAdapter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Bridge our in-house Stage to Curator's ProcessingStage contract.

    Serializes Record state through a single JSON column so arbitrary
    pipeline state (quality scores, reject reasons, parsed Sample) survives
    DocumentBatch round-trips across Curator's executor backends.
    """

    def __init__(self, stage: Stage):
        self.stage = stage
        self.name = stage.name

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        records = _df_to_records(df)
        records = self.stage.run(records)
        out_df = _records_to_df(records)
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=out_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def _df_to_records(df: pd.DataFrame) -> list[Record]:
    if RECORD_COL in df.columns:
        return [Record.model_validate_json(s) for s in df[RECORD_COL]]
    return [Record(raw=_row_to_raw(row)) for _, row in df.iterrows()]


def _records_to_df(records: list[Record]) -> pd.DataFrame:
    return pd.DataFrame({RECORD_COL: [r.model_dump_json() for r in records]})


def _row_to_raw(row) -> dict:
    raw = {}
    for k, v in row.items():
        if k == RECORD_COL:
            continue
        if hasattr(v, "tolist"):
            v = v.tolist()
        if _is_missing(v):
            continue
        raw[k] = v
    return raw


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        import math
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False
