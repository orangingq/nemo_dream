from __future__ import annotations
from typing import Protocol, runtime_checkable
from pipeline.schema import Sample


@runtime_checkable
class Judge(Protocol):
    def safety(self, text: str) -> dict:
        """Return {"toxic": bool, "category": str|None}."""

    def semantic_cosine(self, en: str, ko: str) -> float:
        """Back-translate ko to English then cosine-compare with en."""

    def property_judge(self, sample: Sample) -> dict:
        """Return ints 1..5 for property_preservation, naturalness,
        cultural_appropriateness, register_consistency."""

    def reward(self, en: str, ko: str) -> dict:
        """Return floats keyed by helpfulness/correctness/coherence/..."""
