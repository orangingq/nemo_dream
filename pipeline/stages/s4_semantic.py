from __future__ import annotations
from omegaconf import DictConfig
from pipeline.judges.protocol import Judge
from pipeline.schema import Record
from pipeline.stages.base import RecordStage


class SemanticJudgeStage(RecordStage):
    name = "S4_semantic"

    def __init__(self, judge: Judge, thresholds: DictConfig):
        self.judge = judge
        self.t = thresholds

    def process(self, record: Record) -> Record:
        sample = record.sample
        assert sample is not None
        q = record.quality

        q.semantic_cosine = self.judge.semantic_cosine(sample.en_text, sample.ko_text)
        axes = self.judge.property_judge(sample)
        q.property_preservation = int(axes["property_preservation"])
        q.naturalness = int(axes["naturalness"])
        q.cultural_appropriateness = int(axes["cultural_appropriateness"])
        q.register_consistency = int(axes["register_consistency"])

        if q.semantic_cosine < self.t.semantic_cosine_floor:
            record.reject(
                "S4",
                detail=f"semantic_cosine={q.semantic_cosine:.2f} below floor",
                rule="semantic_drift",
            )
            return record

        floor = min(
            q.property_preservation,
            q.naturalness,
            q.cultural_appropriateness,
            q.register_consistency,
        )
        if floor <= self.t.axis_floor:
            record.reject(
                "S4",
                detail=f"axis floor breached: {axes}",
                rule="axis_floor",
            )
        return record


# ---------------------------------------------------------------------------
# Nemo / NeMo Curator equivalent (parity reference, not wired)
# ---------------------------------------------------------------------------
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.tasks import DocumentBatch


class NemoSemanticEmbeddingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native cosine via EmbeddingCreatorStage.

    Embeds `en_text` and `ko_text` separately with a multilingual SentenceTransformer
    and writes a `_semantic_cosine` column. Decomposes to Curator's CompositeStage
    so it benefits from Curator's batching / GPU autoscale when available.
    """

    name = "S4_semantic_nemo_embed"

    def __init__(
        self,
        model_identifier: str = "intfloat/multilingual-e5-base",
        en_field: str = "en_text",
        ko_field: str = "ko_text",
    ):
        self.en_field = en_field
        self.ko_field = ko_field
        self.en_embedder = EmbeddingCreatorStage(
            model_identifier=model_identifier,
            text_field=en_field,
            embedding_field="_en_emb",
        )
        self.ko_embedder = EmbeddingCreatorStage(
            model_identifier=model_identifier,
            text_field=ko_field,
            embedding_field="_ko_emb",
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.en_field, self.ko_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.en_field, self.ko_field, "_semantic_cosine"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        import numpy as np
        batch = self.en_embedder.process(batch)
        batch = self.ko_embedder.process(batch)
        df = batch.to_pandas().copy()
        en = np.vstack(df["_en_emb"].to_list())
        ko = np.vstack(df["_ko_emb"].to_list())
        en /= np.linalg.norm(en, axis=1, keepdims=True) + 1e-12
        ko /= np.linalg.norm(ko, axis=1, keepdims=True) + 1e-12
        df["_semantic_cosine"] = (en * ko).sum(axis=1)
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class NemoNimJudgeStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native property judge via NIM Nemotron-3 super.

    Calls the hosted NIM endpoint per row using ChatNVIDIA. Writes raw JSON
    verdicts to `_property_judge` column.
    """

    name = "S4_property_nemo_nim"

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-super-120b-a12b",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        en_field: str = "en_text",
        ko_field: str = "ko_text",
    ):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        import os
        self.en_field = en_field
        self.ko_field = ko_field
        self.llm = ChatNVIDIA(model=model, api_key=os.environ["NVIDIA_API_KEY"], base_url=base_url)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.en_field, self.ko_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.en_field, self.ko_field, "_property_judge"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        from langchain_core.messages import HumanMessage, SystemMessage
        df = batch.to_pandas().copy()
        verdicts = []
        for en, ko in zip(df[self.en_field].astype(str), df[self.ko_field].astype(str)):
            msg = self.llm.invoke([
                SystemMessage(content="Score property/naturalness/cultural/register 1-5 as JSON."),
                HumanMessage(content=f"EN:\n{en}\nKO:\n{ko}"),
            ])
            verdicts.append(msg.content)
        df["_property_judge"] = verdicts
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
