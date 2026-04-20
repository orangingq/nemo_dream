from __future__ import annotations
import re
from typing import Optional
from omegaconf import DictConfig
from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
from pipeline.schema import Record, Sample
from pipeline.stages.base import RecordStage


HONORIFIC = re.compile(r"(습니다|입니다|합니다|됩니다|세요|에요|예요|어요|아요|네요|하나요|가요)")
LAUGH_KO = re.compile(r"[ㅋㅎ]+")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")

LAUGH_LEN_MAP: dict[str, tuple[int, int]] = {
    "none": (0, 0),
    "lol":  (1, 3),
    "lmao": (3, 6),
    "rofl": (5, 10),
}

Violation = tuple[str, str]


class SampleFilter(DocumentFilter):
    """Curator DocumentFilter adapted to operate on full Sample objects.

    score_document returns a violation tuple (rule, detail) or None.
    keep_document returns True when the sample passes (no violation).
    """

    def score_document(self, sample: Sample) -> Optional[Violation]:  # type: ignore[override]
        raise NotImplementedError

    def keep_document(self, scores: Optional[Violation]) -> bool:  # type: ignore[override]
        return scores is None


class LaughterFilter(SampleFilter):
    def score_document(self, sample: Sample) -> Optional[Violation]:
        laugh = sample.metadata.internet_markers.laughter
        lo, hi = LAUGH_LEN_MAP[laugh]
        longest = max((len(m.group()) for m in LAUGH_KO.finditer(sample.ko_text)), default=0)
        if lo <= longest <= hi:
            return None
        return ("R2_laughter", f"laughter={laugh} expects {lo}..{hi} but got {longest}")


class RegisterFilter(SampleFilter):
    def __init__(self, thresholds: DictConfig):
        super().__init__()
        self.t = thresholds

    def score_document(self, sample: Sample) -> Optional[Violation]:
        ratio = _honorific_ratio(sample.ko_text)
        reg = sample.metadata.register
        if reg == "intimate" and ratio > self.t.honorific_ratio.intimate_max:
            return ("R3_register_honorific", f"intimate but honorific ratio={ratio:.2f}")
        if reg == "formal" and ratio < self.t.honorific_ratio.formal_min:
            return ("R3_register_plain", f"formal but honorific ratio={ratio:.2f}")
        return None


class AsciiFilter(SampleFilter):
    def __init__(self, thresholds: DictConfig):
        super().__init__()
        self.t = thresholds

    def score_document(self, sample: Sample) -> Optional[Violation]:
        txt = sample.ko_text
        if not txt:
            return None
        letters = sum(1 for c in txt if c.isascii() and c.isalpha())
        ratio = letters / len(txt)
        if ratio > self.t.ascii_ratio_max:
            return ("R4_ascii", f"ascii letter ratio={ratio:.2f}")
        return None


class CulturalRefFilter(SampleFilter):
    def score_document(self, sample: Sample) -> Optional[Violation]:
        for ref in sample.metadata.cultural_refs:
            if ref.ko_term not in sample.ko_text:
                return ("R5_cultural_missing", f"ko_term={ref.ko_term!r} missing from ko_text")
        return None


class LengthFilter(SampleFilter):
    def __init__(self, thresholds: DictConfig):
        super().__init__()
        self.t = thresholds

    def score_document(self, sample: Sample) -> Optional[Violation]:
        ko_n = len(sample.ko_text)
        en_n = len(sample.en_text)
        if ko_n < self.t.length.min_chars or ko_n > self.t.length.max_chars:
            return ("R6_length", f"ko length={ko_n} outside [{self.t.length.min_chars}, {self.t.length.max_chars}]")
        if en_n > 0:
            ratio = ko_n / en_n
            if ratio < self.t.length.ratio_min or ratio > self.t.length.ratio_max:
                return ("R6_length_ratio", f"ko/en={ratio:.2f} outside [{self.t.length.ratio_min}, {self.t.length.ratio_max}]")
        return None


class EmojiFilter(SampleFilter):
    def __init__(self, thresholds: DictConfig):
        super().__init__()
        self.t = thresholds

    def score_document(self, sample: Sample) -> Optional[Violation]:
        n = len(EMOJI.findall(sample.ko_text))
        reg = sample.metadata.register
        if reg == "formal" and n > self.t.emoji.formal_max:
            return ("R7_emoji", f"formal but emoji count={n}")
        if reg == "casual" and n > self.t.emoji.casual_max:
            return ("R7_emoji", f"casual but emoji count={n}")
        return None


class RuleValidatorStage(RecordStage):
    name = "S2_rules"

    def __init__(self, thresholds: DictConfig):
        self.filters: list[SampleFilter] = [
            LaughterFilter(),
            RegisterFilter(thresholds),
            AsciiFilter(thresholds),
            CulturalRefFilter(),
            LengthFilter(thresholds),
            EmojiFilter(thresholds),
        ]

    def process(self, record: Record) -> Record:
        sample = record.sample
        assert sample is not None
        for f in self.filters:
            violation = f.score_document(sample)
            if not f.keep_document(violation):
                rule, detail = violation
                record.reject("S2", detail=detail, rule=rule)
        return record


def _honorific_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(HONORIFIC.findall(text)) / len(words)


# ---------------------------------------------------------------------------
# Nemo / NeMo Curator equivalent (parity reference, not wired)
# ---------------------------------------------------------------------------
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch
import pandas as pd


class NemoRuleValidatorStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native counterpart to RuleValidatorStage.

    For each row, materializes Sample, runs every SampleFilter (DocumentFilter
    subclass), and writes per-rule violation columns plus an aggregate
    `_rule_violations` JSON column. Downstream Curator `Filter` stages can
    drop rows where `_rule_violations` is non-empty.
    """

    name = "S2_rules_nemo"

    def __init__(self, thresholds):
        self.filters: list[SampleFilter] = [
            LaughterFilter(),
            RegisterFilter(thresholds),
            AsciiFilter(thresholds),
            CulturalRefFilter(),
            LengthFilter(thresholds),
            EmojiFilter(thresholds),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_rule_violations"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        all_violations: list[list[dict]] = []
        for raw in df.to_dict(orient="records"):
            sample = Sample.model_validate(raw)
            row_v: list[dict] = []
            for f in self.filters:
                v = f.score_document(sample)
                if not f.keep_document(v):
                    rule, detail = v
                    row_v.append({"rule": rule, "detail": detail})
            all_violations.append(row_v)
        df["_rule_violations"] = all_violations
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
