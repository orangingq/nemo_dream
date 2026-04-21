from __future__ import annotations
from pipeline.judges.protocol import Judge
from pipeline.schema import Record
from pipeline.stages.base import RecordStage


class RewardStage(RecordStage):
    name = "S6_reward"

    def __init__(self, judge: Judge):
        self.judge = judge

    def process(self, record: Record) -> Record:
        sample = record.sample
        assert sample is not None
        record.quality.reward = self.judge.reward(sample.en_text, sample.ko_text)
        return record


# ---------------------------------------------------------------------------
# Nemo / NeMo Curator equivalent (parity reference, not wired)
# ---------------------------------------------------------------------------
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


class NemoRewardStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native reward scoring via NIM `nemotron-4-340b-reward`.

    Calls the hosted Nemotron reward endpoint per row using ChatNVIDIA. The
    reward model returns per-axis logprobs in `response.additional_kwargs`
    which are written to a `_reward` JSON column.
    """

    name = "S6_reward_nemo_nim"

    def __init__(
        self,
        model: str = "nvidia/nemotron-4-340b-reward",
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
        return ["data"], [self.en_field, self.ko_field, "_reward"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        from langchain_core.messages import HumanMessage, AIMessage
        import json
        df = batch.to_pandas().copy()
        rewards = []
        for en, ko in zip(df[self.en_field].astype(str), df[self.ko_field].astype(str)):
            resp = self.llm.invoke([
                HumanMessage(content=en),
                AIMessage(content=ko),
            ])
            scores = resp.response_metadata.get("logprobs", {}) or resp.additional_kwargs.get("logprobs", {})
            rewards.append(json.dumps(scores, ensure_ascii=False))
        df["_reward"] = rewards
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
