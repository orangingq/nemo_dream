from __future__ import annotations
import re
from pipeline.schema import Sample


EN_TOPICS: dict[str, list[str]] = {
    "tech":    ["wifi", "router", "computer", "internet", "network", "laptop", "phone", "app", "code"],
    "food":    ["lunch", "dinner", "coffee", "latte", "eat", "food", "pizza", "drink", "spice"],
    "work":    ["boss", "work", "meeting", "email", "office", "manager", "deadline", "project"],
    "emotion": ["furious", "angry", "sad", "depress", "happy", "excited", "anxious", "stressed", "down", "dread"],
    "health":  ["doctor", "appointment", "medical", "sick", "hospital", "patient"],
    "holiday": ["thanksgiving", "christmas", "halloween", "easter", "in-law", "in-laws"],
    "drive":   ["driving", "license", "car", "test"],
    "event":   ["prom", "date", "concert", "party", "wedding"],
    "school":  ["school", "homework", "exam", "class"],
}

KO_TOPICS: dict[str, list[str]] = {
    "tech":    ["와이파이", "공유기", "네트워크", "컴퓨터", "노트북", "핸드폰", "스마트폰"],
    "food":    ["점심", "저녁", "아침", "커피", "라떼", "먹", "음식", "메뉴", "피자", "스파이스"],
    "work":    ["팀장", "회의", "이메일", "사무실", "마감", "프로젝트", "일하", "직장", "출근"],
    "emotion": ["분노", "화나", "슬프", "우울", "기쁘", "짜증", "스트레스", "기분", "숨막"],
    "health":  ["병원", "예약", "의사", "진료", "환자", "약"],
    "holiday": ["추석", "설날", "명절", "시댁", "친정"],
    "drive":   ["운전", "면허", "시험", "자동차"],
    "event":   ["데이트", "콘서트", "파티", "결혼식", "수학여행", "졸업"],
    "school":  ["학교", "숙제", "수업"],
}


LITERAL_PATTERNS: list[str] = [
    "이것은", "그것은", "그것이", "오 마이 갓", "뜨겁다", "그러나",
]

KNOWN_BAD_CULTURAL: set[tuple[str, str]] = {
    ("prom", "수학여행"),
}

INTENSITY_MARKERS_KO: list[str] = [
    "진짜", "완전", "너무", "대박", "레전드", "미쳤", "쩔",
    "정말", "진심", "엄청", "ㅠㅠ",
]

REGION_TOXIC = re.compile(r"지역.*(싫|혐오|차별|똑같)")

HONORIFIC = re.compile(r"(습니다|입니다|합니다|됩니다|세요|에요|예요|어요|아요|네요|하나요|가요)")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


def _categorize(text: str, topics: dict[str, list[str]]) -> set[str]:
    low = text.lower()
    return {cat for cat, words in topics.items() if any(w in low for w in words)}


def _has_intensity_marker(ko: str) -> bool:
    if any(m in ko for m in INTENSITY_MARKERS_KO):
        return True
    if re.search(r"[!?]{2,}", ko):
        return True
    if re.search(r"(.)\1{2,}", ko):
        return True
    if EMOJI.search(ko):
        return True
    return False


def _honorific_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(HONORIFIC.findall(text)) / len(words)


class MockJudge:
    """Deterministic heuristic judge for offline testing.

    Designed so the 22 mockup samples produce expected reject/accept decisions
    without any LLM calls. Real runs use NimJudge.
    """

    def safety(self, text: str) -> dict:
        if REGION_TOXIC.search(text):
            return {"toxic": True, "category": "discrimination"}
        return {"toxic": False, "category": None}

    def semantic_cosine(self, en: str, ko: str) -> float:
        en_cats = _categorize(en, EN_TOPICS)
        ko_cats = _categorize(ko, KO_TOPICS)
        if en_cats and ko_cats:
            return 0.85 if (en_cats & ko_cats) else 0.30
        return 0.70

    def property_judge(self, sample: Sample) -> dict:
        ko = sample.ko_text
        meta = sample.metadata
        prop = self._property_preservation(meta.emotion_intensity, ko)
        nat = self._naturalness(ko)
        cult = self._cultural_appropriateness(sample)
        reg = self._register_consistency(meta.register, ko)
        return {
            "property_preservation": prop,
            "naturalness": nat,
            "cultural_appropriateness": cult,
            "register_consistency": reg,
        }

    def reward(self, en: str, ko: str) -> dict:
        return {
            "helpfulness": 4.0,
            "correctness": 4.0,
            "coherence": 4.0,
            "complexity": 3.0,
            "verbosity": 3.0,
        }

    @staticmethod
    def _property_preservation(intensity: int, ko: str) -> int:
        if intensity >= 4:
            return 5 if _has_intensity_marker(ko) else 1
        return 4

    @staticmethod
    def _naturalness(ko: str) -> int:
        hits = sum(1 for p in LITERAL_PATTERNS if p in ko)
        if hits >= 2:
            return 1
        if hits == 1:
            return 3
        return 5

    @staticmethod
    def _cultural_appropriateness(sample: Sample) -> int:
        for ref in sample.metadata.cultural_refs:
            if (ref.en_term.lower(), ref.ko_term) in KNOWN_BAD_CULTURAL:
                return 1
        return 5

    @staticmethod
    def _register_consistency(register: str, ko: str) -> int:
        ratio = _honorific_ratio(ko)
        if register == "intimate":
            return 5 if ratio < 0.10 else 2
        if register == "casual":
            return 5 if ratio < 0.30 else 3
        if register == "formal":
            return 5 if ratio >= 0.10 else 2
        if register == "public":
            return 5 if 0.03 <= ratio < 0.50 else 3
        return 4
