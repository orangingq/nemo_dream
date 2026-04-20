import json
import os
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
from urllib import error, request

from config import load_env_from_key_json
from util import load_jsonl
from openai import OpenAI

POSITIVE_CULTURAL_HINTS = [
    "병원",
    "예약",
    "토스",
    "더치페이",
    "명절",
    "지하철",
    "룸메",
    "야근",
    "배달 앱",
    "현관",
    "공유 폴더",
    "복원",
    "공항버스",
    "회의",
    "회의록",
    "구독",
    "모바일 신분증",
    "카드 명세서",
    "스터디룸",
    "어깨",
    "지도 앱",
    "콘서트 티켓",
    "고등학생",
    "과적합",
    "물",
    "쿠폰",
    "가격",
    "파이썬",
    "택시",
    "사이즈표",
    "후드티",
    "약속 시간",
    "자리 지키",
    "git 커밋",
    "변경 내용",
]
INFORMAL_MARKERS = ["ㅋㅋ", "ㅠ", "ㅎㅎ", "진짜", "또", "와"]
FORMAL_ENDINGS = ["니다", "까요", "요?"]
RESIDUAL_ENGLISH_PATTERN = re.compile(r"[A-Za-z]{4,}")


@dataclass
class JudgeConfig:
    mode: str
    judge_api_key: str | None
    judge_model: str | None
    judge_timeout: int
    llm_weight: float
    safety_weight: float
    fail_on_guardrail: bool


class JudgeRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, details: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, round(value, 4)))


def semantic_fidelity(sample: dict[str, Any]) -> float:
    source = sample["source_text"].lower()
    target = sample["rewritten_text"]
    score = 0.55
    for mapped in sample.get("cultural_mapping", {}).values():
        mapped_str = str(mapped)
        if mapped_str in target:
            score += 0.12
        elif any(token.strip() and token.strip() in target for token in mapped_str.split("/")):
            score += 0.08
    if "?" in sample["source_text"] and "?" in target:
        score += 0.08
    if len(target) >= 12:
        score += 0.06
    if "doctor" in source and "병원" in target:
        score += 0.08
    if "train" in source and "지하철" in target:
        score += 0.08
    return bounded(score)


def attribute_preservation(sample: dict[str, Any]) -> float:
    meta = sample["metadata"]
    text = sample["rewritten_text"]
    score = 0.45

    speech_act = meta["speech_act"]
    if speech_act == "question" and "?" in text:
        score += 0.22
    elif speech_act in {"complaint", "sarcasm"} and any(
        tok in text for tok in ["또", "너무", "진짜", "어렵", "믿기 어렵", "대단하다"]
    ):
        score += 0.22

    intensity = meta["emotion"]["intensity"]
    exclam = text.count("!") + text.count("ㅠ") + text.count("ㅋ")
    if intensity >= 4 and exclam >= 1:
        score += 0.18
    elif intensity <= 2 and exclam <= 1:
        score += 0.18

    register = meta["register"]
    if register == "formal" and any(text.endswith(end) for end in FORMAL_ENDINGS):
        score += 0.15
    elif register in {"casual", "intimate"} and any(marker in text for marker in INFORMAL_MARKERS):
        score += 0.15
    elif register == "public" and ("네요" in text or "합니다" in text):
        score += 0.15

    return bounded(score)


def cultural_alignment(sample: dict[str, Any]) -> float:
    text = sample["rewritten_text"]
    score = 0.4
    matched = sum(1 for hint in POSITIVE_CULTURAL_HINTS if hint in text)
    score += min(0.35, matched * 0.09)

    if any(eng in text.lower() for eng in ["appointment", "leftovers", "roommate", "split bill"]):
        score -= 0.12

    if sample["metadata"]["platform"] == "instagram_story" and len(text) < 40:
        score += 0.12
    if sample["metadata"]["platform"] == "linkedin" and "구성원" in text:
        score += 0.1
    return bounded(score)


def fluency(sample: dict[str, Any]) -> float:
    text = sample["rewritten_text"].strip()
    score = 0.62
    if 10 <= len(text) <= 80:
        score += 0.15
    if "  " not in text:
        score += 0.08
    if not RESIDUAL_ENGLISH_PATTERN.search(text):
        score += 0.08
    if text and text[-1] in ".?!요다네ㅠㅋ":
        score += 0.07
    return bounded(score)


def style_fit(sample: dict[str, Any]) -> float:
    platform = sample["metadata"]["platform"]
    text = sample["rewritten_text"]
    score = 0.45
    if platform in {"twitter", "instagram_story"} and any(marker in text for marker in ["ㅠ", "ㅋㅋ", "…"]):
        score += 0.28
    if platform == "faq" and ("가능한가요" in text or "해야 하나요" in text):
        score += 0.28
    if platform == "linkedin" and ("구성원" in text or "배려" in text):
        score += 0.28
    if platform == "community" and ("뭐예요?" in text or "되나요?" in text):
        score += 0.28
    return bounded(score)


def heuristic_safety(sample: dict[str, Any]) -> float:
    text = sample["rewritten_text"]
    score = 0.9
    if RESIDUAL_ENGLISH_PATTERN.search(text):
        score -= 0.1
    if re.search(r"(죽여|폭탄|마약|테러)", text):
        score -= 0.5
    return bounded(score)


def heuristic_evaluate(sample: dict[str, Any]) -> dict[str, Any]:
    semantic = semantic_fidelity(sample)
    attribute = attribute_preservation(sample)
    cultural = cultural_alignment(sample)
    fluency_score = fluency(sample)
    style = style_fit(sample)
    safety = heuristic_safety(sample)

    consistency = semantic
    naturalness = bounded(0.5 * fluency_score + 0.3 * cultural + 0.2 * style)
    relevance = bounded(0.6 * attribute + 0.4 * semantic)
    coherence = bounded(0.55 * attribute + 0.45 * fluency_score)

    final = bounded((consistency + naturalness + relevance + coherence + fluency_score) / 5)

    reasons: list[str] = []
    if consistency < 0.75:
        reasons.append("consistency with the source is relatively low")
    if naturalness < 0.75:
        reasons.append("naturalness could be improved")
    if coherence < 0.75:
        reasons.append("the response flow is only partially coherent")
    if safety < 0.75:
        reasons.append("safety or residual English needs review")
    if not reasons:
        reasons.append("balanced quality across the main evaluation axes")

    return {
        "backend": "heuristic",
        "consistency": consistency,
        "naturalness": naturalness,
        "relevance": relevance,
        "coherence": coherence,
        "fluency": fluency_score,
        "semantic": semantic,
        "attribute": attribute,
        "cultural": cultural,
        "style": style,
        "safety": safety,
        "final": final,
        "judge_reason": "; ".join(reasons),
        "guardrail_pass": safety >= 0.7,
        "flags": {
            "unsafe_content": safety < 0.5,
            "residual_english": bool(RESIDUAL_ENGLISH_PATTERN.search(sample["rewritten_text"])),
            "culturally_inappropriate": cultural < 0.45,
        },
    }


def _judge_system_prompt() -> str:
    return (
        "You are an expert evaluator for English-to-Korean localization quality. "
        "Score the Korean output rigorously and return only valid JSON. "
        "Use 0.0 to 1.0 scores. Evaluate consistency, naturalness, relevance, coherence, and fluency. "
        "Penalize omissions, mistranslations, off-topic content, contradictions, awkward Korean, "
        "residual English, unsafe content, and culturally awkward phrasing."
    )


def _judge_user_prompt(sample: dict[str, Any]) -> str:
    payload = {
        "task": "Evaluate a localized Korean output.",
        "rubric": {
            "consistency": "How faithfully the Korean output preserves the source meaning and intended facts.",
            "naturalness": "How native, idiomatic, and culturally natural the Korean output feels.",
            "relevance": "How directly the output addresses the source intent, metadata, and task constraints without drifting.",
            "coherence": "How logically well-structured, self-consistent, and easy to follow the output is.",
            "fluency": "How grammatically smooth, readable, and polished the Korean output is.",
        },
        "instructions": [
            "Use the metadata and cultural_mapping as constraints.",
            "If the output is partially untranslated, reduce consistency, naturalness, and fluency.",
            "If the output drifts from the source intent, reduce relevance.",
            "If the output contradicts itself or is hard to follow, reduce coherence.",
            "If the output is safe but low quality, do not mark unsafe.",
            "Return concise rationale and actionable improvements.",
        ],
        "output_schema": {
            "consistency": "float 0..1",
            "naturalness": "float 0..1",
            "relevance": "float 0..1",
            "coherence": "float 0..1",
            "fluency": "float 0..1",
            "flags": {
                "unsafe_content": "bool",
                "residual_english": "bool",
                "culturally_inappropriate": "bool",
                "register_mismatch": "bool",
                "off_topic": "bool",
                "self_contradiction": "bool",
            },
            "rationale": "short string",
            "improvements": ["short string"],
        },
        "sample": sample,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Judge response is not valid JSON.")


def _post_chat_completion(config: JudgeConfig, messages: list[dict[str, str]]) -> str:
    if not config.judge_api_key or not config.judge_model:
        raise ValueError("LLM judge requires api base, api key, and model.")
    
    client = OpenAI(api_key=config.judge_api_key)
    
    try:
        assert len(messages) == 2, "Expected exactly 2 messages: system and user."
        assert messages[0]["role"] == "system", "First message must be system role."
        assert messages[1]["role"] == "user", "Second message must be user role."
        instructions = messages[0]["content"]
        input_content = messages[1]["content"]
        response = client.responses.create(
            model=config.judge_model,
            # messages=messages,
            instructions=instructions,
            input=input_content,
            # temperature=0.0,
            # timeout=config.judge_timeout,
        )
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise JudgeRequestError(
            f"Judge request failed: {exc.code} {details}",
            status_code=exc.code,
            details=details,
        ) from exc
    except error.URLError as exc:
        raise JudgeRequestError(f"Cannot reach judge endpoint: {exc.reason}") from exc

    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Judge response did not include any choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise RuntimeError("Judge response content is missing.")


def llm_evaluate(sample: dict[str, Any], config: JudgeConfig) -> dict[str, Any]:
    content = _post_chat_completion(
        config,
        [
            {"role": "system", "content": _judge_system_prompt()},
            {"role": "user", "content": _judge_user_prompt(sample)},
        ],
    )
    raw = _extract_json_object(content)

    improvements = raw.get("improvements") or []
    if not isinstance(improvements, list):
        improvements = [str(improvements)]

    flags = raw.get("flags") or {}
    if not isinstance(flags, dict):
        flags = {}

    consistency = bounded(float(raw["consistency"]))
    naturalness = bounded(float(raw["naturalness"]))
    relevance = bounded(float(raw["relevance"]))
    coherence = bounded(float(raw["coherence"]))
    fluency_score = bounded(float(raw["fluency"]))
    safety = bounded(
        1.0
        - 0.5 * float(bool(flags.get("unsafe_content")))
        - 0.2 * float(bool(flags.get("residual_english")))
        - 0.15 * float(bool(flags.get("culturally_inappropriate")))
    )
    final = bounded((consistency + naturalness + relevance + coherence + fluency_score) / 5)

    return {
        "backend": "llm",
        "consistency": consistency,
        "naturalness": naturalness,
        "relevance": relevance,
        "coherence": coherence,
        "fluency": fluency_score,
        "safety": safety,
        "final": final,
        "judge_reason": str(raw.get("rationale", "")).strip() or "scored by LLM judge",
        "guardrail_pass": not any(
            bool(flags.get(key))
            for key in ["unsafe_content", "culturally_inappropriate", "off_topic", "self_contradiction"]
        ) and safety >= 0.6,
        "flags": {
            "unsafe_content": bool(flags.get("unsafe_content")),
            "residual_english": bool(flags.get("residual_english")),
            "culturally_inappropriate": bool(flags.get("culturally_inappropriate")),
            "register_mismatch": bool(flags.get("register_mismatch")),
            "off_topic": bool(flags.get("off_topic")),
            "self_contradiction": bool(flags.get("self_contradiction")),
        },
        "improvements": [str(item) for item in improvements[:3]],
    }


def merge_evaluations(
    sample: dict[str, Any],
    heuristic_result: dict[str, Any],
    llm_result: dict[str, Any] | None,
    config: JudgeConfig,
    judge_error: str | None = None,
) -> dict[str, Any]:
    result = heuristic_result
    backend = "heuristic"

    if llm_result is not None and config.mode == "llm":
        result = llm_result
        backend = "llm"
    elif llm_result is not None and config.mode == "hybrid":
        backend = "hybrid"
        result = {
            "consistency": bounded(
                (1 - config.llm_weight) * heuristic_result["consistency"] + config.llm_weight * llm_result["consistency"]
            ),
            "naturalness": bounded(
                (1 - config.llm_weight) * heuristic_result["naturalness"] + config.llm_weight * llm_result["naturalness"]
            ),
            "relevance": bounded(
                (1 - config.llm_weight) * heuristic_result["relevance"] + config.llm_weight * llm_result["relevance"]
            ),
            "coherence": bounded(
                (1 - config.llm_weight) * heuristic_result["coherence"] + config.llm_weight * llm_result["coherence"]
            ),
            "fluency": bounded(
                (1 - config.llm_weight) * heuristic_result["fluency"] + config.llm_weight * llm_result["fluency"]
            ),
            "safety": bounded(
                (1 - config.safety_weight) * heuristic_result["safety"] + config.safety_weight * llm_result["safety"]
            ),
        }
        result["final"] = bounded(
            (result["consistency"] + result["naturalness"] + result["relevance"] + result["coherence"] + result["fluency"]) / 5
        )
        result["guardrail_pass"] = bool(heuristic_result["guardrail_pass"]) and bool(llm_result["guardrail_pass"])
        result["flags"] = {
            key: bool(heuristic_result["flags"].get(key)) or bool(llm_result["flags"].get(key))
            for key in set(heuristic_result["flags"]) | set(llm_result["flags"])
        }
        result["judge_reason"] = (
            f"hybrid judge; heuristic={heuristic_result['judge_reason']}; "
            f"llm={llm_result['judge_reason']}"
        )
        result["improvements"] = llm_result.get("improvements", [])

    if config.fail_on_guardrail and not result["guardrail_pass"]:
        result["final"] = min(result["final"], 0.35)

    if judge_error:
        result["judge_reason"] = f"{result['judge_reason']}; llm fallback: {judge_error}"

    return {
        "sample_id": sample["sample_id"],
        "consistency": result["consistency"],
        "naturalness": result["naturalness"],
        "relevance": result["relevance"],
        "coherence": result["coherence"],
        "fluency": result["fluency"],
        "safety": result["safety"],
        "final": result["final"],
        "judge_reason": result["judge_reason"],
        "judge_backend": backend,
        "guardrail_pass": result["guardrail_pass"],
        "flags": result["flags"],
        "improvements": result.get("improvements", []),
        "rewritten_text": sample["rewritten_text"],
        "platform": sample["metadata"]["platform"],
        "speech_act": sample["metadata"]["speech_act"],
        "judge_error": judge_error,
        "heuristic_breakdown": heuristic_result,
        "llm_breakdown": llm_result,
    }

def evaluate_sample(sample: dict[str, Any], config: JudgeConfig) -> dict[str, Any]:
    heuristic_result = heuristic_evaluate(sample)
    llm_result: dict[str, Any] | None = None
    judge_error: str | None = None

    if config.mode in {"llm", "hybrid"}:
        try:
            llm_result = llm_evaluate(sample, config)
        except JudgeRequestError as exc:
            judge_error = str(exc)

    return merge_evaluations(sample, heuristic_result, llm_result, config, judge_error)

def build_judge_config(args: Any) -> JudgeConfig:
    return JudgeConfig(
        mode=args.mode,
        judge_api_key=os.environ.get("OPENAI_API_KEY"),
        judge_model=args.judge_model,
        judge_timeout=args.judge_timeout,
        llm_weight=bounded(args.llm_weight),
        safety_weight=bounded(args.safety_weight),
        fail_on_guardrail=args.fail_on_guardrail,
    )

def argparser() -> argparse.Namespace:
    '''Defines command-line arguments for the evaluation script.'''
    parser = argparse.ArgumentParser(description="Evaluate generated Korean localization data.")
    parser.add_argument("--input", default="data/example_generated_data.jsonl", help="Path to generated JSONL file")
    parser.add_argument("--output", default="data/evaluated_data.jsonl", help="Path to save evaluation results")
    parser.add_argument("--mode", choices=["heuristic", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--key-json-path", default="key.json", help="Path to key.json for API key loading")
    parser.add_argument("--judge-model", default="gpt-5.2")
    parser.add_argument("--judge-timeout", type=int, default=120)
    parser.add_argument("--llm-weight", type=float, default=0.7)
    parser.add_argument("--safety-weight", type=float, default=0.8)
    parser.add_argument("--fail-on-guardrail", action="store_true")
    args = parser.parse_args()
    return args


def main() -> None:
    args = argparser()
    load_env_from_key_json(Path(args.key_json_path))
    config = build_judge_config(args)
    samples = load_jsonl(args.input)

    if config.mode in {"llm", "hybrid"} and (not config.judge_api_key or not config.judge_model):
        raise ValueError(
            "LLM or hybrid mode requires OPENAI_API_KEY in key.json and a judge model."
        )

    results = [evaluate_sample(sample, config) for sample in samples]

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg_final = mean(row["final"] for row in results)
    pass_rate = sum(1 for row in results if row["guardrail_pass"]) / max(len(results), 1)
    fallback_count = sum(1 for row in results if row.get("judge_error"))
    print(f"Evaluated {len(results)} samples")
    print(f"Average final score: {avg_final:.3f}")
    print(f"Guardrail pass rate: {pass_rate:.3f}")
    if fallback_count:
        print(f"LLM judge fallback count: {fallback_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
