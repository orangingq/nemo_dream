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
    """Seed generator / adapter.

    When `cfg.source_jsonl` is set, this stage normalizes externally curated
    stage data (stage1..stage4 JSONL) into the Sample-shaped records expected by
    S1..S7. Otherwise it synthesizes EN->KO localization candidates via NVIDIA
    Data Designer and writes raw JSONL. Downstream S1..S7 consume the returned
    records.
    """

    name = "S0_generate"

    def __init__(self, cfg: DictConfig, out_jsonl: str | Path):
        self.cfg = cfg
        self.out_jsonl = Path(out_jsonl)

    def run(self, _: list[Record]) -> list[Record]:
        rows = self._load_curated_source()
        if rows is None:
            df = self._invoke_data_designer()
            rows = [
                _to_sample_raw(row, idx)
                for idx, row in enumerate(df.to_dict(orient="records"))
            ]
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.out_jsonl.open("w", encoding="utf-8") as f:
            for raw in rows:
                f.write(json.dumps(raw, ensure_ascii=False) + "\n")
        return [Record(raw=raw) for raw in rows]

    def _load_curated_source(self) -> list[dict[str, Any]] | None:
        source_jsonl = self.cfg.get("source_jsonl")
        if not source_jsonl:
            return None

        source_path = Path(source_jsonl)
        rows: list[dict[str, Any]] = []
        for idx, line in enumerate(source_path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            rows.append(_normalize_curated_raw(json.loads(line), idx, source_path))
        return rows

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
    platform = row["platform"]
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
            "platform_fit": [platform],
            "target_platform": platform,
            "cultural_refs": [],
            "internet_markers": {"laughter": "none", "emphasis": [], "sarcasm_marker": False},
        },
        "provenance": {"source": "data_designer"},
    }


def _normalize_curated_raw(
    raw: dict[str, Any],
    idx: int,
    source_path: Path,
) -> dict[str, Any]:
    """Convert external curated stage rows into the pipeline Sample contract.

    Supported inputs:
    - stage1/stage2 rows: `text`/`source_text` + `decomposed`
    - stage3 rows: `source_text` + `decomposed` + `generation.rewritten_text`
    - stage4 rows: `en_text` + `ko_text` + `metadata`
    """

    metadata_in = raw.get("metadata") or {}
    decomposed = raw.get("decomposed") or {}
    generation = raw.get("generation") or {}

    source_text = (
        raw.get("en_text")
        or raw.get("source_text")
        or raw.get("text")
        or decomposed.get("source_text")
        or ""
    )
    ko_text = raw.get("ko_text") or generation.get("rewritten_text") or ""
    platform_fit = _as_list(
        metadata_in.get("platform_fit")
        or decomposed.get("platform_fit")
        or generation.get("target_platform")
        or metadata_in.get("target_platform")
    )
    target_platform = (
        metadata_in.get("target_platform")
        or generation.get("target_platform")
        or (platform_fit[0] if platform_fit else "unknown")
    )
    if not platform_fit:
        platform_fit = [target_platform]

    emotion = decomposed.get("emotion") or {}
    metadata = {
        "speech_act": _normalize_speech_act(
            metadata_in.get("speech_act") or decomposed.get("speech_act")
        ),
        "register": metadata_in.get("register") or decomposed.get("register") or "casual",
        "emotion_type": (
            metadata_in.get("emotion_type")
            or emotion.get("type")
            or "neutral"
        ),
        "emotion_intensity": int(
            metadata_in.get("emotion_intensity")
            or emotion.get("intensity")
            or 3
        ),
        "estimated_age_group": _normalize_age_group(
            metadata_in.get("estimated_age_group")
            or decomposed.get("estimated_age_group")
            or generation.get("target_age_group")
        ),
        "platform_fit": platform_fit,
        "target_platform": target_platform,
        "cultural_refs": _normalize_cultural_refs(raw),
        "internet_markers": _normalize_internet_markers(
            metadata_in.get("internet_markers") or decomposed.get("internet_markers") or {}
        ),
    }

    out = {
        "id": raw.get("id") or f"curated_{idx:06d}",
        "en_text": source_text,
        "ko_text": ko_text,
        "ko_text_pre_marker": raw.get("ko_text_pre_marker"),
        "metadata": metadata,
        "provenance": _normalize_provenance(raw, source_path),
    }

    # stage4 rows may already carry judge scores; keep them available for
    # consumers while S4 remains free to recompute or overwrite them.
    if "quality" in raw:
        out["quality"] = raw["quality"]
    if "valid" in raw:
        out["valid"] = raw["valid"]
    if "reject_reasons" in raw:
        out["reject_reasons"] = raw["reject_reasons"]
    return out


def _normalize_cultural_refs(raw: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_refs = (raw.get("metadata") or {}).get("cultural_refs")
    if metadata_refs is not None:
        return [
            {
                "type": _normalize_ref_type(ref.get("type")),
                "en_term": ref.get("en_term") or ref.get("term") or "",
                "ko_term": ref.get("ko_term") or ref.get("ko") or "",
                "source": _normalize_ref_source(ref.get("source")),
                "retrieved_doc_id": ref.get("retrieved_doc_id"),
            }
            for ref in metadata_refs
        ]

    mapped_refs = raw.get("mapped_refs") or []
    return [
        {
            "type": _normalize_ref_type(ref.get("type")),
            "en_term": ref.get("term") or ref.get("en_term") or "",
            "ko_term": ref.get("ko") or ref.get("ko_term") or "",
            "source": _normalize_ref_source(ref.get("source"), ref.get("retrieved")),
            "retrieved_doc_id": ref.get("retrieved_doc_id"),
        }
        for ref in mapped_refs
        if ref.get("term") or ref.get("en_term")
    ]


def _normalize_internet_markers(markers: dict[str, Any]) -> dict[str, Any]:
    emphasis_map = {
        "punctuation": "exclaim",
        "emoji": "exclaim",
        "caps": "CAPS",
    }
    emphasis = []
    for value in _as_list(markers.get("emphasis")):
        normalized = emphasis_map.get(str(value).lower(), value)
        if normalized in {"CAPS", "repetition", "exclaim", "ellipsis"}:
            emphasis.append(normalized)
    return {
        "laughter": markers.get("laughter") or "none",
        "emphasis": emphasis,
        "sarcasm_marker": bool(markers.get("sarcasm_marker", False)),
    }


def _normalize_provenance(raw: dict[str, Any], source_path: Path) -> dict[str, Any]:
    provenance = dict(raw.get("provenance") or {})
    provenance.setdefault("source", "curated_jsonl")
    provenance.setdefault("source_path", str(source_path))
    if raw.get("generation"):
        provenance["generation"] = raw["generation"]
    return provenance


def _normalize_speech_act(value: Any) -> str:
    mapping = {
        "statement": "announce",
        "greeting": "announce",
    }
    value = str(value or "other")
    return mapping.get(value, value)


def _normalize_age_group(value: Any) -> str:
    value = str(value or "20s")
    if value in {"teen", "20s", "30s", "40plus"}:
        return value
    if value in {"40s", "50s", "60s", "adult"}:
        return "40plus"
    return "20s"


def _normalize_ref_type(value: Any) -> str:
    mapping = {
        "pop_culture": "meme",
        "slang": "meme",
        "other": "event",
    }
    value = str(value or "event")
    return mapping.get(value, value)


def _normalize_ref_source(value: Any, retrieved: Any = None) -> str:
    value = str(value or "")
    if value == "dict":
        return "map"
    if retrieved or "web" in value:
        return "retrieved"
    if value in {"map", "retrieved", "llm"}:
        return value
    return "llm"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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
