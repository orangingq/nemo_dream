from __future__ import annotations
from pipeline.judges.protocol import Judge
from pipeline.schema import Record
from pipeline.stages.base import RecordStage

try:
    from presidio_analyzer import Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpArtifacts
except ModuleNotFoundError:
    Pattern = None
    PatternRecognizer = None
    NlpArtifacts = None

KR_RRN_REGEX = r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)"
KR_PHONE_REGEX = r"(?<!\d)01[016789]-?\d{3,4}-?\d{4}(?!\d)"

if Pattern is not None:
    KR_RRN_PATTERN = Pattern(name="kr_rrn", regex=KR_RRN_REGEX, score=0.9)
    KR_PHONE_PATTERN = Pattern(name="kr_phone", regex=KR_PHONE_REGEX, score=0.9)
    _EMPTY_NLP = NlpArtifacts(
        entities=[], tokens=[], tokens_indices=[], lemmas=[], nlp_engine=None, language="en"
    )
else:
    KR_RRN_PATTERN = None
    KR_PHONE_PATTERN = None
    _EMPTY_NLP = None


class SafetyStage(RecordStage):
    name = "S3_safety"

    def __init__(self, judge: Judge):
        self.judge = judge
        if PatternRecognizer is None:
            self.rrn = None
            self.phone = None
        else:
            self.rrn = PatternRecognizer(
                supported_entity="KR_RRN", patterns=[KR_RRN_PATTERN]
            )
            self.phone = PatternRecognizer(
                supported_entity="KR_PHONE", patterns=[KR_PHONE_PATTERN]
            )

    def process(self, record: Record) -> Record:
        sample = record.sample
        assert sample is not None
        text = sample.ko_text

        if self._detect(self.rrn, text, "KR_RRN"):
            record.quality.pii_pass = False
            record.reject("S3", detail="KR_RRN detected", rule="pii_rrn")
            return record
        if self._detect(self.phone, text, "KR_PHONE"):
            record.quality.pii_pass = False
            record.reject("S3", detail="KR_PHONE detected", rule="pii_phone")
            return record
        record.quality.pii_pass = True

        verdict = self.judge.safety(text)
        if verdict["toxic"]:
            record.quality.safety_pass = False
            record.reject(
                "S3",
                detail=f"content_safety: {verdict.get('category')}",
                rule="content_safety",
            )
            return record
        record.quality.safety_pass = True
        return record

    @staticmethod
    def _detect(recognizer: PatternRecognizer, text: str, entity: str) -> bool:
        if recognizer is None:
            import re
            pattern = KR_RRN_REGEX if entity == "KR_RRN" else KR_PHONE_REGEX
            return bool(re.search(pattern, text))
        results = recognizer.analyze(text=text, entities=[entity], nlp_artifacts=_EMPTY_NLP)
        return bool(results)


# ---------------------------------------------------------------------------
# Nemo / NeMo Curator equivalent (parity reference, not wired)
# ---------------------------------------------------------------------------
try:
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.text.classifiers import AegisClassifier
    from nemo_curator.tasks import DocumentBatch
except Exception:
    class ProcessingStage:
        def __class_getitem__(cls, item):
            return cls

    class DocumentBatch:
        pass

    AegisClassifier = None


class NemoAegisSafetyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native safety screen via NVIDIA Aegis LlamaGuard.

    Decomposes to AegisClassifier (CompositeStage) which loads
    `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0` from HF and
    writes an `aegis_pred` column ('safe' / 'unsafe' / category labels).
    Downstream a Curator `Filter` keeps rows whose `aegis_pred == 'safe'`.
    """

    name = "S3_safety_nemo_aegis"

    def __init__(self, hf_token: str | None = None, text_field: str = "ko_text"):
        if AegisClassifier is None:
            raise ImportError("AegisClassifier requires compatible nemo-curator/transformers deps.")
        self.classifier = AegisClassifier(
            aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
            hf_token=hf_token,
            text_field=text_field,
            label_field="aegis_pred",
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.classifier.inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.classifier.outputs()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        return self.classifier.process(batch)


class NemoNemoguardSafetyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native safety screen via NIM `llama-3.1-nemoguard-8b-content-safety`.

    Calls the OpenAI-compatible NIM endpoint per row. Writes an
    `_nemoguard_verdict` column with raw JSON from the safety model.
    """

    name = "S3_safety_nemo_nim"

    def __init__(
        self,
        model: str = "nvidia/llama-3.1-nemoguard-8b-content-safety",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        text_field: str = "ko_text",
    ):
        import os
        from openai import OpenAI
        self.text_field = text_field
        self.model = model
        api_key = "no-key" if base_url.startswith(("http://localhost", "http://127.0.0.1")) else os.environ["NVIDIA_API_KEY"]
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field, "_nemoguard_verdict"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        verdicts = []
        for text in df[self.text_field].astype(str):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
            verdicts.append(resp.choices[0].message.content)
        df["_nemoguard_verdict"] = verdicts
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
