from __future__ import annotations
import json
import os
import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from pipeline.schema import Sample


JUDGE_PROMPT = """당신은 한국어 SNS 텍스트 품질 검수 전문가입니다.
아래 샘플을 평가하고 JSON으로만 응답하세요.

[원문 영어] {en}
[메타데이터] speech_act={sa}, register={reg}, emotion={emo}(강도 {ei}/5), age={age}, platform={pf}
[한국어 재작성] {ko}
[문화 치환 내역] {refs}

평가 축 (각 1~5점, 정수):
- property_preservation: 화행/공식성/감정 강도가 재작성본에서 유지되는가
- naturalness: 실제 한국인이 해당 플랫폼/연령대에서 쓸 법한 자연스러운 표현인가
- cultural_appropriateness: 영어 문화 요소의 한국 치환이 맥락상 어색하지 않은가
- register_consistency: 존댓말/반말/인터넷체 일관성

필수 출력:
{{"property_preservation":X, "naturalness":X, "cultural_appropriateness":X,
  "register_consistency":X, "issues":["..."], "suggested_fix":"..."}}
"""

BACK_TRANSLATE_SYS = (
    "Translate the following Korean text into natural English. "
    "Preserve meaning only. Return only the translation."
)


class NimJudge:
    def __init__(
        self,
        base_url: str,
        models: dict,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        api_key = os.environ["NVIDIA_API_KEY"]
        self.models = models
        self.max_tokens = max_tokens
        self.temperature = temperature

        common = {"api_key": api_key, "base_url": base_url}
        self.safety_llm = ChatNVIDIA(model=models["safety"], temperature=temperature, **common)
        self.translate_llm = ChatNVIDIA(
            model=models["translate"], temperature=temperature,
            max_completion_tokens=max_tokens, **common,
        )
        self.judge_llm = ChatNVIDIA(
            model=models["judge"], temperature=temperature,
            max_completion_tokens=max_tokens, **common,
        )
        self.reward_llm = ChatNVIDIA(model=models["reward"], **common)
        self.embed = NVIDIAEmbeddings(model=models["embed"], **common)

    def safety(self, text: str) -> dict:
        msg = self.safety_llm.invoke([HumanMessage(content=text)])
        payload = json.loads(msg.content)
        toxic = any(payload.get(k, "no") == "yes" for k in payload if k.startswith("S"))
        return {"toxic": toxic, "category": payload.get("violations")}

    def semantic_cosine(self, en: str, ko: str) -> float:
        bt = self._back_translate(ko)
        v1 = np.asarray(self.embed.embed_query(en))
        v2 = np.asarray(self.embed.embed_query(bt))
        return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def property_judge(self, sample: Sample) -> dict:
        meta = sample.metadata
        refs = ", ".join(
            f"{r.en_term}→{r.ko_term}({r.source})" for r in meta.cultural_refs
        ) or "(없음)"
        prompt = JUDGE_PROMPT.format(
            en=sample.en_text,
            sa=meta.speech_act, reg=meta.register,
            emo=meta.emotion_type, ei=meta.emotion_intensity,
            age=meta.estimated_age_group, pf=meta.target_platform,
            ko=sample.ko_text, refs=refs,
        )
        msg = self.judge_llm.invoke(
            [HumanMessage(content=prompt)],
            response_format={"type": "json_object"},
        )
        return json.loads(msg.content)

    def reward(self, en: str, ko: str) -> dict:
        msg = self.reward_llm.invoke([
            HumanMessage(content=en),
            AIMessage(content=ko),
        ])
        return _parse_reward(msg)

    def _back_translate(self, ko: str) -> str:
        msg = self.translate_llm.invoke([
            SystemMessage(content=BACK_TRANSLATE_SYS),
            HumanMessage(content=ko),
        ])
        return msg.content


def _parse_reward(msg) -> dict:
    meta = msg.response_metadata or {}
    logprobs = meta.get("logprobs") or {}
    content = logprobs.get("content") or []
    if not content:
        raise RuntimeError("reward response missing logprobs.content")
    return {tok["token"]: tok["logprob"] for tok in content}
