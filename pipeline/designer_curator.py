from __future__ import annotations

"""Shared helpers for Stage 1-3 designer stages.

This module contains the NIM prompts, normalization helpers, and JSONL I/O
used by `pipeline/stages/s1_decompose.py`, `s2_map.py`, and `s3_rewrite.py`.
The stage-by-stage pipeline entrypoints live in `pipeline/stages/`.
"""

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import OpenAI


SPEECH_ACTS = {
    "complaint", "brag", "question", "empathy_seeking", "sarcasm",
    "joke", "statement", "greeting", "request",
}
REGISTERS = {"intimate", "casual", "formal", "public"}
EMOTIONS = {"joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"}
REF_TYPES = {"holiday", "brand", "service", "event", "food", "pop_culture", "slang", "other"}
PLATFORMS = {"twitter", "reddit", "instagram", "tiktok", "discord", "sms"}

DECOMPOSE_SYSTEM_PROMPT = """You are a sociolinguistic annotator. Given an English social media post,
return JSON only.

Allowed values:
- speech_act: complaint | brag | question | empathy_seeking | sarcasm | joke | statement | greeting | request
- register: intimate | casual | formal | public
- emotion.type: joy | anger | sadness | fear | surprise | disgust | neutral
- emotion.intensity: integer 1..5
- cultural_refs[i].type: holiday | brand | service | event | food | pop_culture | slang | other
- internet_markers.laughter: lol | lmao | rofl | haha | none
- internet_markers.emphasis: subset of [CAPS, repetition, punctuation, emoji]
- estimated_age_group: teen | 20s | 30s | 40plus | unknown
- platform_fit: subset of [twitter, reddit, instagram, tiktok, discord, sms]

Return this shape:
{"speech_act": "...", "register": "...", "emotion": {"type": "...", "intensity": 3},
 "cultural_refs": [{"type": "...", "term": "..."}],
 "internet_markers": {"laughter": "none", "emphasis": [], "sarcasm_marker": false},
 "estimated_age_group": "unknown", "platform_fit": ["twitter"]}
"""

REWRITE_SYSTEM_PROMPT = """You localize English social media text into natural Korean.
Preserve meaning, intent, register, emotion, internet style, and important named entities.
Use mapped cultural references only when they improve Korean naturalness without distorting the source.
Return JSON only with:
{"target_platform": "...", "target_age_group": "...", "target_community": "...",
 "target_gender_style": "neutral", "rewritten_text": "...", "generation_notes": "..."}"""


@dataclass(slots=True)
class DesignerCuratorConfig:
    model: str
    base_url: str
    cultural_map: str


def load_seed_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        text = raw.get("text") or raw.get("source_text") or raw.get("en_text") or raw.get("context") or ""
        rows.append(
            {
                "id": str(raw.get("id") or f"seed-{idx:06d}"),
                "text": str(text),
                "metadata": raw.get("metadata") or {},
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_term(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9가-힣\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=4)
def cultural_index(path: str) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    index = {}
    for category, entries in raw.items():
        for entry in entries:
            key = normalize_term(entry["en"])
            index[key] = {
                "term": entry["en"],
                "ko": entry["ko"],
                "type": entry.get("type", category.rstrip("s")),
                "source": "dict" if category != "retrieved" else "retriever",
                "retrieved": category == "retrieved",
                "notes": entry.get("notes", ""),
            }
    return index


def normalize_decomposition(data: dict[str, Any], text: str, cultural_map: str) -> dict[str, Any]:
    speech_act = str(data.get("speech_act") or "statement").lower().replace(" ", "_")
    if speech_act not in SPEECH_ACTS:
        speech_act = "statement"
    register = str(data.get("register") or "casual").lower()
    if register not in REGISTERS:
        register = "casual"
    emotion = data.get("emotion") if isinstance(data.get("emotion"), dict) else {}
    emotion_type = str(emotion.get("type") or "neutral").lower()
    if emotion_type not in EMOTIONS:
        emotion_type = "neutral"
    try:
        intensity = int(round(float(emotion.get("intensity", 3))))
    except (TypeError, ValueError):
        intensity = 3
    markers = data.get("internet_markers") if isinstance(data.get("internet_markers"), dict) else {}
    laughter = str(markers.get("laughter") or "none").lower()
    if laughter not in {"lol", "lmao", "rofl", "haha", "none"}:
        laughter = "none"
    emphasis = markers.get("emphasis") or []
    if isinstance(emphasis, str):
        emphasis = [part.strip() for part in emphasis.split(",")]
    emphasis = [("CAPS" if str(item).lower() == "caps" else str(item)) for item in emphasis]
    emphasis = [item for item in emphasis if item in {"CAPS", "repetition", "punctuation", "emoji"}]
    age = str(data.get("estimated_age_group") or "unknown").lower()
    if age not in {"teen", "20s", "30s", "40plus", "unknown"}:
        age = "unknown"
    platform_fit = data.get("platform_fit") or []
    if isinstance(platform_fit, str):
        platform_fit = [part.strip() for part in platform_fit.split(",")]
    platform_fit = [str(item).lower() for item in platform_fit if str(item).lower() in PLATFORMS]
    if not platform_fit:
        platform_fit = ["reddit" if "?" in text else "twitter"]

    refs = []
    for ref in data.get("cultural_refs") or []:
        if isinstance(ref, str):
            term = normalize_term(ref)
            ref_type = "other"
        elif isinstance(ref, dict):
            term = normalize_term(str(ref.get("term") or ""))
            ref_type = str(ref.get("type") or "other")
        else:
            continue
        if not term:
            continue
        hit = cultural_index(cultural_map).get(term)
        if hit:
            ref_type = hit["type"]
        if ref_type not in REF_TYPES:
            ref_type = "other"
        refs.append({"type": ref_type, "term": term})

    return {
        "source_text": text,
        "speech_act": speech_act,
        "register": register,
        "emotion": {"type": emotion_type, "intensity": max(1, min(5, intensity))},
        "cultural_refs": refs,
        "internet_markers": {
            "laughter": laughter,
            "emphasis": emphasis,
            "sarcasm_marker": bool(markers.get("sarcasm_marker", False)),
        },
        "estimated_age_group": age,
        "platform_fit": platform_fit,
    }


def client(base_url: str) -> OpenAI:
    if base_url.startswith(("http://localhost", "http://127.0.0.1")):
        api_key = os.environ.get("NVIDIA_API_KEY", "no-key")
    else:
        api_key = os.environ["NVIDIA_API_KEY"]
    return OpenAI(base_url=base_url, api_key=api_key)


def nim_decompose(text: str, cfg: DesignerCuratorConfig) -> dict[str, Any]:
    resp = client(cfg.base_url).chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f'Annotate this post:\n"""{text}"""'},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    payload = json.loads(resp.choices[0].message.content or "{}")
    return normalize_decomposition(payload, text, cfg.cultural_map)


def nim_map_refs(row: dict[str, Any], cfg: DesignerCuratorConfig) -> list[dict[str, Any]]:
    refs = row["decomposed"].get("cultural_refs", [])
    if not refs:
        return []
    candidates = []
    for ref in refs:
        hit = cultural_index(cfg.cultural_map).get(normalize_term(ref["term"]))
        if hit:
            candidates.append(hit)
    prompt = {
        "source_text": row["source_text"],
        "decomposed": row["decomposed"],
        "cultural_refs": refs,
        "candidate_mappings": candidates,
        "instructions": (
            "Map each English cultural reference to the best Korean equivalent. "
            "Use candidate_mappings when they preserve meaning; otherwise keep the named entity "
            "or produce a better Korean equivalent. Return mapped_refs only."
        ),
    }
    resp = client(cfg.base_url).chat.completions.create(
        model=cfg.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You map English cultural references for Korean localization. "
                    "Return JSON only: {\"mapped_refs\":[{\"term\":\"...\",\"ko\":\"...\","
                    "\"type\":\"holiday|brand|service|event|food|pop_culture|slang|other\","
                    "\"source\":\"dict|retriever|web+llm\",\"retrieved\":false,\"notes\":\"...\"}]}"
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    payload = json.loads(resp.choices[0].message.content or "{}")
    mapped = payload.get("mapped_refs")
    if not isinstance(mapped, list):
        raise RuntimeError("NIM Stage2 response missing mapped_refs list")

    normalized = []
    for ref, item in zip(refs, mapped, strict=False):
        item = item if isinstance(item, dict) else {}
        term = normalize_term(str(item.get("term") or ref.get("term") or ""))
        ko = str(item.get("ko") or "").strip()
        if not term or not ko:
            raise RuntimeError(f"NIM Stage2 returned invalid mapping: {item}")
        ref_type = str(item.get("type") or ref.get("type") or "other")
        if ref_type not in REF_TYPES:
            ref_type = "other"
        source = str(item.get("source") or "web+llm")
        if source not in {"dict", "retriever", "web+llm"}:
            source = "web+llm"
        normalized.append(
            {
                "term": term,
                "ko": ko,
                "type": ref_type,
                "source": source,
                "retrieved": bool(item.get("retrieved", source != "dict")),
                "notes": str(item.get("notes") or ""),
            }
        )
    if len(normalized) != len(refs):
        raise RuntimeError("NIM Stage2 mapped_refs length does not match cultural_refs length")
    return normalized


def nim_rewrite(row: dict[str, Any], cfg: DesignerCuratorConfig) -> dict[str, Any]:
    prompt = {
        "source_text": row["source_text"],
        "decomposed": row["decomposed"],
        "mapped_refs": row.get("mapped_refs", []),
    }
    resp = client(cfg.base_url).chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    payload = json.loads(resp.choices[0].message.content or "{}")
    if not payload.get("rewritten_text"):
        raise RuntimeError("NIM Stage3 response missing rewritten_text")
    return {
        "target_platform": payload.get("target_platform") or (row["decomposed"].get("platform_fit") or ["twitter"])[0],
        "target_age_group": payload.get("target_age_group") or row["decomposed"].get("estimated_age_group", "unknown"),
        "target_community": payload.get("target_community") or "general",
        "target_gender_style": payload.get("target_gender_style") or "neutral",
        "rewritten_text": payload["rewritten_text"],
        "generation_notes": payload.get("generation_notes", ""),
    }
