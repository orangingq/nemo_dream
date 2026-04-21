from __future__ import annotations
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
EXPECTED = REPO / "mockup-data" / "expected-outcomes.json"
ACCEPTED = REPO / "val_out" / "accepted.jsonl"
REJECTED = REPO / "val_out" / "rejected.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["samples"]
    accepted = {row["id"]: row for row in _load_jsonl(ACCEPTED)}
    rejected = {row["id"]: row for row in _load_jsonl(REJECTED)}

    failures: list[str] = []
    for sid, exp in expected.items():
        if exp.get("expected") == "pass":
            if sid not in accepted:
                first = (rejected.get(sid, {}).get("reject_reasons") or [{}])[0]
                failures.append(
                    f"{sid}: expected accepted but rejected at "
                    f"{first.get('stage')} ({first.get('rule')}: {first.get('detail')})"
                )
            continue

        want_stage = exp["expected_fail_stage"].split("_")[0]
        if sid not in rejected:
            failures.append(f"{sid}: expected reject at {want_stage} but accepted")
            continue
        reasons = rejected[sid].get("reject_reasons") or []
        if not reasons:
            failures.append(f"{sid}: rejected with no reasons")
            continue
        got_stage = reasons[0]["stage"]
        if got_stage != want_stage:
            failures.append(
                f"{sid}: expected {want_stage} but rejected at {got_stage} "
                f"({reasons[0].get('rule')}: {reasons[0].get('detail')})"
            )

    _check_ascii_001(rejected, failures)

    total = len(expected)
    if failures:
        print(f"\nFAIL: {len(failures)}/{total} mismatches")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nOK: all {total} samples match expected outcomes")
    return 0


def _check_ascii_001(rejected: dict, failures: list[str]) -> None:
    sid = "bad-rule-ascii-001"
    row = rejected.get(sid)
    if row is None:
        return
    rules = {r.get("rule") for r in row.get("reject_reasons") or []}
    expected_rules = {"R4_ascii", "R5_cultural_missing"}
    missing = expected_rules - rules
    if missing:
        failures.append(f"{sid}: expected reject_rules {expected_rules}, missing {missing}")


if __name__ == "__main__":
    sys.exit(main())
