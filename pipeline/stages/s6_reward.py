from __future__ import annotations

"""Stage 6: reward-model evaluation.

Input record shape:
`stage5.jsonl` rows that already include `metadata`, `quality.aggregate`,
`valid`, and `reject_reasons`.

Output record shape (`stage6.jsonl`):
- same row contract as Stage 5
- `quality.reward` is added for rows that remain valid
"""

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

    Calls the OpenAI-compatible Nemotron reward endpoint per row. The reward
    model returns per-axis logprobs, which are written to a `_reward` JSON
    column.
    """

    name = "S6_reward_nemo_nim"

    def __init__(
        self,
        model: str = "nvidia/nemotron-4-340b-reward",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        en_field: str = "en_text",
        ko_field: str = "ko_text",
    ):
        import os
        from openai import OpenAI
        self.en_field = en_field
        self.ko_field = ko_field
        self.model = model
        api_key = "no-key" if base_url.startswith(("http://localhost", "http://127.0.0.1")) else os.environ["NVIDIA_API_KEY"]
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.en_field, self.ko_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.en_field, self.ko_field, "_reward"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        import json
        df = batch.to_pandas().copy()
        rewards = []
        for en, ko in zip(df[self.en_field].astype(str), df[self.ko_field].astype(str)):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": en},
                    {"role": "assistant", "content": ko},
                ],
                logprobs=True,
            )
            logprobs = getattr(resp.choices[0], "logprobs", None)
            content = getattr(logprobs, "content", None) if logprobs else None
            scores = {tok.token: tok.logprob for tok in content} if content else {}
            rewards.append(json.dumps(scores, ensure_ascii=False))
        df["_reward"] = rewards
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
