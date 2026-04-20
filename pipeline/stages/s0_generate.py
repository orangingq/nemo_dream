from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pydantic import BaseModel

from pipeline.schema import Record
from pipeline.stages.base import Stage


class _GeneratedPair(BaseModel):
    en_text: str
    ko_text: str


class GenerateStage(Stage):
    """Data Designer-backed seed generator.

    Synthesizes EN->KO localization candidates via NVIDIA Data Designer
    and writes raw JSONL. Downstream S1..S7 consume the returned records.
    """

    name = "S0_generate"

    def __init__(self, cfg: DictConfig, out_jsonl: str | Path):
        self.cfg = cfg
        self.out_jsonl = Path(out_jsonl)

    def run(self, _: list[Record]) -> list[Record]:
        df = self._invoke_data_designer()
        rows = [_to_sample_raw(row, idx) for idx, row in enumerate(df.to_dict(orient="records"))]
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.out_jsonl.open("w", encoding="utf-8") as f:
            for raw in rows:
                f.write(json.dumps(raw, ensure_ascii=False) + "\n")
        return [Record(raw=raw) for raw in rows]

    def _invoke_data_designer(self):
        dd = importlib.import_module("data_designer.config")
        interface_mod = importlib.import_module("data_designer.interface")
        DataDesigner = interface_mod.DataDesigner

        api_key_env = self.cfg.api_key_env
        if not os.environ.get(api_key_env):
            raise RuntimeError(f"Missing API key; set ${api_key_env}.")

        model_configs = [
            dd.ModelConfig(
                alias=self.cfg.model_alias,
                model=self.cfg.model_id,
                provider=self.cfg.model_provider,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_tokens,
                    max_parallel_requests=self.cfg.max_parallel_requests,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ),
            )
        ]

        builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
        for col_name in ("speech_act", "register", "platform"):
            builder.add_column(
                dd.SamplerColumnConfig(
                    name=col_name,
                    sampler_type=dd.SamplerType.CATEGORY,
                    params=dd.CategorySamplerParams(values=list(self.cfg.samplers[col_name])),
                )
            )
        builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="pair",
                model_alias=self.cfg.model_alias,
                output_format=_GeneratedPair,
                prompt=(
                    "Produce one English source snippet and a natural Korean localization of it. "
                    "Constraints: speech_act={{ speech_act }}, register={{ register }}, "
                    "platform={{ platform }}. Return JSON {en_text, ko_text}."
                ),
            )
        )

        designer = DataDesigner(artifact_path=str(self.cfg.artifact_path))
        pd = importlib.import_module("pandas")

        total = int(self.cfg.num_samples)
        batch_size = max(1, int(self.cfg.batch_size))
        frames = []
        remaining = total
        idx = 0
        while remaining > 0:
            n = min(batch_size, remaining)
            result = designer.create(
                builder,
                num_records=n,
                dataset_name=f"{self.cfg.dataset_name}_part_{idx:03d}",
            )
            frames.append(result.load_dataset())
            remaining -= n
            idx += 1
        return pd.concat(frames, ignore_index=True)


def _to_sample_raw(row: dict[str, Any], idx: int) -> dict[str, Any]:
    pair = row.get("pair") or {}
    if isinstance(pair, str):
        pair = json.loads(pair)
    return {
        "id": row.get("id") or f"gen_{idx:06d}",
        "en_text": pair.get("en_text", ""),
        "ko_text": pair.get("ko_text", ""),
        "metadata": {
            "speech_act": row["speech_act"],
            "register": row["register"],
            "emotion_type": "neutral",
            "emotion_intensity": 3,
            "estimated_age_group": "20s",
            "platform_fit": [row["platform"]],
            "target_platform": row["platform"],
            "cultural_refs": [],
            "internet_markers": {"laughter": "none", "emphasis": [], "sarcasm_marker": False},
        },
        "provenance": {"source": "data_designer"},
    }


# ---------------------------------------------------------------------------
# Nemo / NeMo Curator equivalent (parity reference, not wired)
# ---------------------------------------------------------------------------
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


class NemoGenerateStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native wrapper around GenerateStage for Curator pipelines.

    Ignores input batch contents and emits a DocumentBatch of freshly
    generated raw rows so a JsonlWriter or StageAdapter chain can take over.
    """

    name = "S0_generate_nemo"

    def __init__(self, cfg: DictConfig, out_jsonl: str | Path):
        self.inner = GenerateStage(cfg, out_jsonl)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        import pandas as pd
        records = self.inner.run([])
        df = pd.DataFrame([r.raw for r in records])
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
