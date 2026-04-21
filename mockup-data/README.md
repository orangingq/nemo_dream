# Mockup Data - 검증·필터 파이프라인 테스트용

## 파일

| 파일 | 용도 |
|---|---|
| `candidates.jsonl` | 21개 샘플 (good 6 + bad 15) — 파이프라인 입력 |
| `expected-outcomes.json` | 각 샘플의 예상 탈락 stage + 이유 |

## 샘플 구성

### Good samples (6개, 전부 pass 기대)

| ID | 시나리오 | 특징 |
|---|---|---|
| `good-001` | 20s twitter 불평 | ㅋㅋ(lol) 강도 일치, casual 반말 |
| `good-002` | teen instagram 자랑 | `pumpkin spice latte → 스타벅스 호박 스파이스 라떼` 매핑 |
| `good-003` | 30s reddit 의료 질문 | `doctor's appointment → 병원 예약` **Retriever-backed** (`retrieved_doc_id` 존재), formal 존댓말 |
| `good-004` | 30s 추석 시댁 걱정 | `thanksgiving → 추석`, `in-laws → 시댁` 문화 이중 매핑 |
| `good-005` | 20s 운전면허 실패 공감요청 | ㅠㅠ는 laughter 아닌 emphasis(ellipsis) |
| `good-006` | 30s 회의 빈정거림 | `sarcasm_marker=true`, public register |

### Bad samples (15개, stage별 실패 기대)

| Stage | ID | 실패 이유 |
|---|---|---|
| **S1 Schema** | `bad-schema-001` | `emotion_intensity=7` (1-5 범위 초과) |
|  | `bad-schema-002` | `ko_text` 필드 누락 |
| **S2 Rules** | `bad-rule-laughter-001` | laughter=none인데 ko_text에 ㅋㅋㅋㅋ |
|  | `bad-rule-laughter-002` | laughter=lol(1-3자)인데 ㅋ 10자 연속 |
|  | `bad-rule-register-001` | intimate register인데 존댓말 "습니다" 남발 |
|  | `bad-rule-register-002` | formal register인데 반말 "야/봐봐" |
|  | `bad-rule-ascii-001` | "Starbucks에서 coffee 한잔 at 3pm" 영어 잔여 |
|  | `bad-rule-cultural-001` | `prom→졸업파티` 매핑 주장하나 ko_text에 "prom" 그대로 |
| **S3 Safety/PII** | `bad-safety-pii-rrn` | 주민등록번호 패턴 노출 |
|  | `bad-safety-pii-phone` | 010-XXXX-XXXX 패턴 노출 |
|  | `bad-safety-toxic-001` | 지역 차별 발언 |
| **S4 Semantic** | `bad-semantic-meaning` | WiFi 질문 → 점심 고민 (완전 drift) |
|  | `bad-semantic-property` | 강한 분노(5) → "조금 기분이 별로" (1-2) |
|  | `bad-semantic-naturalness` | "lit" → "뜨겁다" 직역 |
|  | `bad-semantic-cultural` | `prom → 수학여행` 부적절 매핑 |

## 활용 예시

### 1. Stage별 unit test

```python
import json
from pathlib import Path
from schema import Sample

# 입력 로드
candidates = [json.loads(line) for line in
              Path("mockup-data/candidates.jsonl").read_text().splitlines()]
expected = json.loads(Path("mockup-data/expected-outcomes.json").read_text())

# S1 Schema 검증
def test_s1():
    for c in candidates:
        sid = c["id"]
        exp = expected["samples"][sid]
        try:
            Sample.model_validate(c)
            passed = True
        except Exception:
            passed = False

        if exp.get("expected_fail_stage") == "S1":
            assert not passed, f"{sid} should fail S1"
        else:
            assert passed, f"{sid} should pass S1"
```

### 2. 룰 validator 단독 테스트

```python
from stages.s2_rules import rule_checks

for c in candidates:
    sid = c["id"]
    if expected["samples"][sid].get("expected_fail_stage") != "S2":
        continue
    errs = rule_checks(c)
    assert len(errs) > 0, f"{sid}: rule should have triggered"
    print(sid, "→", errs[0]["rule"])
```

### 3. 파이프라인 dry-run

전체 Curator 파이프라인에 `candidates.jsonl`을 입력으로 주고,
accepted/rejected 결과를 `expected-outcomes.json`과 대조.

```python
accepted_ids = {json.loads(l)["id"] for l in open("val_out/accepted.jsonl")}
rejected_ids = {json.loads(l)["id"] for l in open("val_out/rejected.jsonl")}

# Good 샘플은 전부 accepted
for sid, info in expected["samples"].items():
    if info.get("expected") == "pass":
        assert sid in accepted_ids, f"{sid} should be accepted"
    else:
        assert sid in rejected_ids, f"{sid} should be rejected"
```

