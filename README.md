# nemo_dream — EN→KO Localization (Mockup) Pipeline

## Stages

| # | Stage | Purpose |
| - | ----- | ------- |
| S0 | `GenerateStage` | Data Designer-backed seed synthesis (EN/KO pairs with sampler-driven metadata) |
| S1 | `SchemaValidationStage` | Pydantic `Sample` validation |
| S2 | `RuleValidatorStage` | Heuristic rule gates (laughter, register, length, emoji, cultural refs) |
| S3 | `SafetyStage` | PII + content-safety guardrails |
| S4 | `SemanticJudgeStage` | Cosine + LLM judge axes (property / naturalness / cultural / register) |
| S5 | `QualityFilterStage` + `FuzzyDedupStage` | Weighted quality threshold and MinHash dedup |
| S6 | `RewardStage` | Nemotron Reward scoring |
| S7 | `ReportStage` | JSON report + score distribution chart |

## Run

```bash
pip install -r requirements.txt

# Validate mockup candidates (S0 off)
python -m pipeline.run

# End-to-end with Data Designer seed generation (S0 on)
export NVIDIA_API_KEY=nvapi-...
python -m pipeline.run generate.enabled=true generate.num_samples=50

# NeMo Curator execution path (Pipeline + RayActorPoolExecutor)
python -m pipeline.curator_run

# Verify outputs against mockup-data/expected-outcomes.json
python -m tests.verify_mockup
```

Outputs land under `val_out/`:

- `accepted.jsonl` / `rejected.jsonl`
- `report.json` — totals, reject reasons, quality summary
- `score_distribution.png` — aggregate histogram + per-axis means

## Layout

```
conf/                Hydra config (judge=mock|nim, generate, thresholds, report)
pipeline/
├── schema.py
├── orchestrator.py
├── curator_bridge.py
├── run.py            @hydra.main — lightweight runner
├── curator_run.py    @hydra.main — NeMo Curator + Ray executor
├── judges/           Judge protocol + MockJudge + NimJudge
└── stages/           S0 generate → S7 report (S2 uses Curator DocumentFilter)
mockup-data/          Fixture samples + expected outcomes
tests/verify_mockup.py
```

## Pipeline

1. **Schema Validation**
    - Curator DocumentDataset (Dask/cuDF) :  컬럼 타입 일치 보장, parquet/jsonl IO
        - 컬럼-레벨 타입만, 중첩 객체 검증 안 됨
    - example code
        
        ```python
        # stages/s1_schema.py
        from nemo_curator.stages.base import Stage
        from pydantic import ValidationError
        from schema import Sample
        
        class SchemaValidationStage(Stage):
            name = "schema_validation"
            def process(self, record: dict) -> dict:
                try:
                    s = Sample.model_validate(record)
                    record["_valid"] = True
                    return record
                except ValidationError as e:
                    record["_valid"] = False
                    record.setdefault("reject_reasons", []).append(
                        {"stage": "schema", "detail": str(e)})
                    return record
        ```
        
2. **Rule-based Validator**
    
    
    | 규칙 | 로직 |
    | --- | --- |
    | R1 Laughter 일관성 | `internet_markers.laughter == "none"` ⇒ `ko_text`에 `ㅋ+` / `ㅎ+` 금지 |
    | R2 Laughter 강도 | intensity ↔ `ㅋ`의 길이 매핑 테이블 위반 탐지 |
    | R3 Register ↔ 존칭 | `register="intimate"` ⇒ `-요/-습니다` 사용률 < 10%. `register="formal"` ⇒ ≥ 80% |
    | R4 영문 잔여 | `ko_text`의 ASCII 비율 < 20% (브랜드명 등은 예외) |
    | R5 Cultural 치환 적용 | metadata의 각 CulturalRef의 `ko_term`이 `ko_text`에 실제로 등장 |
    | R6 길이 | 5 ≤ chars ≤ 1000, 원본 대비 0.4~2.5배 |
    | R7 이모지 상한 | `register="formal"` ⇒ 이모지 0. `"casual"` ⇒ ≤ 5 |
    - Curator DocumentFilter
3. **Safety & PII rails**
    - Nemo gaurdrails + Nemoguard + NIM batch
        - guardrails 대화형응답 필터가 목적이지만 검증 전용 모드로 사용 가능 : output rails만 활성화
        - LLMRails.generate_async는 여러 LLM round trip. 3-7회 LLM 호출이 필요하고, embedding based intent classifiy도 필요함.
        - `nvidia/llama-3.1-nemoguard-8b-content-safety` 으로 직접 호출 예시
            
            ```python
            # stages/s3_safety.py
            from openai import AsyncOpenAI
            import asyncio
            
            nim = AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1",
                              api_key=os.environ["NVIDIA_API_KEY"])
            
            async def content_safety(text: str) -> dict:
                r = await nim.chat.completions.create(
                    model="nvidia/llama-3.1-nemoguard-8b-content-safety",
                    messages=[{"role": "user", "content": text}],
                    temperature=0.0,
                )
                # 모델 카드 스펙: JSON {"S1":..,"S2":..,"violations":[...]}
                import json
                return json.loads(r.choices[0].message.content)
            ```
            
    - rail 목록
        - KR_RNN(주민번호), PHONE(핸드폰번호), CARD PII(카드번호)
        - jailbreak-detect (원본 seed로)
            - LLM에게 가드레일 우회하도록 유도하는 seed 프롬프트 분류
                - NeMo Curator : InstructionDataGuardClassifier
                - NVIDIA NIM : nemoguard-jailbreak-detect
4. **Semantic Validator**
    - LLM judge 예시
        
        ```python
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
        
        async def llm_judge(sample: dict) -> dict:
            meta = sample["metadata"]
            refs = ", ".join(f"{r['en_term']}→{r['ko_term']}({r['source']})"
                             for r in meta["cultural_refs"]) or "(없음)"
            r = await nim.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    en=sample["en_text"], sa=meta["speech_act"], reg=meta["register"],
                    emo=meta["emotion_type"], ei=meta["emotion_intensity"],
                    age=meta["estimated_age_group"], pf=meta["target_platform"],
                    ko=sample["ko_text"], refs=refs,
                )}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=512,
            )
            import json
            return json.loads(r.choices[0].message.content)
        ```
        
    - 생성 모델 ≠ judge 모델, self-consistency
    - LLM/Embedding
        - ( Back-translation → NV-Embed consine )
        - property preservation LLM classify
        - Naturalness LLM judge (Korean rubric)
        - Cultural appropriateness LLM judge
5. **Curator Filter + Dedup**
    - curator dedup 예시
        
        ```python
        from nemo_curator.modules import ExactDuplicates, FuzzyDuplicates, SemanticDedup
        
        pipeline.add_stage(ExactDuplicates(id_field="id", text_field="ko_text"))
        pipeline.add_stage(FuzzyDuplicates(
            id_field="id", text_field="ko_text",
            num_hashes=260, num_buckets=20, hashes_per_bucket=13,
            jaccard_threshold=0.8,
        ))
        pipeline.add_stage(SemanticDedup(
            id_field="id", text_field="ko_text",
            embedding_model_name_or_path="nvidia/nv-embedqa-e5-v5",
            eps_to_extract=0.08,      # cosine 0.92 이상 중복 처리
        ))
        ```
        
    - nemo_curator.modules.ExactDuplicates
    - nemo_curator.modules.FuzzyDuplicates (내부적으로 MinHashLSH 사용)
        - fuzzy dedup : 유사도 임계값 이상
        - 글자 공백 제거 후 k-gram으로 쪼개고 permutation으로 해싱 → MinHash signiture
        - MinHash는 집합을 n차원 hash vector로 압축 후 벡터 비교로 jaccard를 근사, 집합 연산을 안해도 됨.
        - LSH는 유사한 MinHash끼리 같은 버킷에 떨어지도록 해싱 O(N)
        - Jaccard 유사도 0.8이상이면 match
    - nemo_curator.modules.SemanticDedup (NV-Embed 기반 cosine > 0.92)
        - weighted sum quality score threshold
        - MinHash fuzzy dedup
        - NV-embed semantic dedup (cosine > 0.92)
6. **Reward Model Scoring**
    - 최종 랭킹 & 선택적 top-N curation용
    - reward는 영어→한국어 재작성 특성상 correctness와 coherence만 가중 사용, helpfulness는 context-free 판정이라 편향 위험이 있음.
    - NEmotron-4-340B-Reward
        - helpfulness / correctness / coherence .. etc
7. **Evaluation Report**
8. **(Downstream Finetue + Benchmark)**
