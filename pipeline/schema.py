from __future__ import annotations
import warnings
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

warnings.filterwarnings(
    "ignore",
    message=r'Field name "register" .*shadows an attribute in parent .*',
)


class CulturalRef(BaseModel):
    type: Literal["holiday", "brand", "event", "person", "meme", "service", "place", "food"]
    en_term: str
    ko_term: str
    source: Literal["map", "retrieved", "llm"]
    retrieved_doc_id: Optional[str] = None


class InternetMarkers(BaseModel):
    laughter: Literal["lol", "lmao", "rofl", "none"]
    emphasis: list[Literal["CAPS", "repetition", "exclaim", "ellipsis"]] = []
    sarcasm_marker: bool = False


class Metadata(BaseModel):
    speech_act: Literal[
        "complaint", "brag", "question", "empathy_seeking",
        "sarcasm", "joke", "announce", "advice", "other",
    ]
    register: Literal["intimate", "casual", "formal", "public"]
    emotion_type: str
    emotion_intensity: int = Field(ge=1, le=5)
    estimated_age_group: Literal["teen", "20s", "30s", "40plus"]
    platform_fit: list[str]
    target_platform: str
    cultural_refs: list[CulturalRef] = []
    internet_markers: InternetMarkers


class Sample(BaseModel):
    id: str
    en_text: str
    ko_text: str
    ko_text_pre_marker: Optional[str] = None
    metadata: Metadata
    provenance: dict[str, Any]


class QualityScores(BaseModel):
    semantic_cosine: Optional[float] = None
    property_preservation: Optional[int] = None
    naturalness: Optional[int] = None
    cultural_appropriateness: Optional[int] = None
    register_consistency: Optional[int] = None
    safety_pass: Optional[bool] = None
    pii_pass: Optional[bool] = None
    aggregate: Optional[float] = None
    reward: Optional[dict[str, float]] = None


class RejectReason(BaseModel):
    stage: str
    rule: Optional[str] = None
    detail: str
    extra: dict[str, Any] = {}


class Record(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw: dict[str, Any]
    sample: Optional[Sample] = None
    valid: bool = True
    quality: QualityScores = Field(default_factory=QualityScores)
    reject_reasons: list[RejectReason] = []

    @property
    def id(self) -> str:
        return self.sample.id if self.sample else self.raw.get("id", "<unknown>")

    def reject(self, stage: str, detail: str, *, rule: Optional[str] = None, **extra: Any) -> None:
        self.valid = False
        self.reject_reasons.append(
            RejectReason(stage=stage, rule=rule, detail=detail, extra=extra)
        )

    def to_output(self) -> dict[str, Any]:
        if self.sample is None:
            base = dict(self.raw)
        else:
            base = self.sample.model_dump()
        base["quality"] = self.quality.model_dump(exclude_none=True)
        if self.reject_reasons:
            base["reject_reasons"] = [r.model_dump(exclude_none=True) for r in self.reject_reasons]
        return base
