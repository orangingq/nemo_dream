from __future__ import annotations
import json
import os
import numpy as np
from openai import OpenAI
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
        api_key = _api_key_for(base_url)
        self.models = models
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def safety(self, text: str) -> dict:
        content = self._chat(self.models["safety"], [{"role": "user", "content": text}])
        payload = json.loads(content)
        toxic = any(payload.get(k, "no") == "yes" for k in payload if k.startswith("S"))
        return {"toxic": toxic, "category": payload.get("violations")}

    def semantic_cosine(self, en: str, ko: str) -> float:
        bt = self._back_translate(ko)
        v1 = np.asarray(self._embed(en))
        v2 = np.asarray(self._embed(bt))
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
        content = self._chat(
            self.models["judge"],
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(content)

    def reward(self, en: str, ko: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.models["reward"],
            messages=[
                {"role": "user", "content": en},
                {"role": "assistant", "content": ko},
            ],
            logprobs=True,
        )
        return _parse_reward(resp)

    def _back_translate(self, ko: str) -> str:
        return self._chat(
            self.models["translate"],
            [
                {"role": "system", "content": BACK_TRANSLATE_SYS},
                {"role": "user", "content": ko},
            ],
        )

    def _chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> str:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.models["embed"], input=text)
        return resp.data[0].embedding


def _api_key_for(base_url: str) -> str:
    if base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1"):
        return os.environ.get("NVIDIA_API_KEY", "no-key")
    return os.environ["NVIDIA_API_KEY"]


def _parse_reward(resp) -> dict:
    choice = resp.choices[0]
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs else None
    if not content:
        raise RuntimeError("reward response missing logprobs.content")
    return {tok.token: tok.logprob for tok in content}
