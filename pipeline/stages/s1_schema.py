from __future__ import annotations
from pydantic import ValidationError
from pipeline.schema import Record, Sample
from pipeline.stages.base import RecordStage


class SchemaValidationStage(RecordStage):
    name = "S1_schema"

    def process(self, record: Record) -> Record:
        try:
            record.sample = Sample.model_validate(record.raw)
        except ValidationError as e:
            record.reject("S1", detail=_summarize_errors(e))
        return record


def _summarize_errors(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Nemo / NeMo Curator equivalent (parity reference, not wired)
# ---------------------------------------------------------------------------
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


class NemoSchemaValidationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native schema validator that mirrors SchemaValidationStage.

    Reads each row of the DocumentBatch as a dict, attempts Sample.model_validate,
    and writes a `_schema_valid` column plus `_schema_error` so a downstream
    Curator Filter can drop invalid rows. Pure ProcessingStage — runs on
    Curator executors (Ray, Xenna) without an outer adapter.
    """

    name = "S1_schema_nemo"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_schema_valid", "_schema_error"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        valids: list[bool] = []
        errors: list[str | None] = []
        for raw in df.to_dict(orient="records"):
            try:
                Sample.model_validate(raw)
                valids.append(True)
                errors.append(None)
            except ValidationError as e:
                valids.append(False)
                errors.append(_summarize_errors(e))
        df["_schema_valid"] = valids
        df["_schema_error"] = errors
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
