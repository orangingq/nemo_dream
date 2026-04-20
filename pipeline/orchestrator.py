from __future__ import annotations
import json
from pathlib import Path
from pipeline.schema import Record
from pipeline.stages.base import Stage


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, records: list[Record]) -> list[Record]:
        for stage in self.stages:
            print(f"[{stage.name}] start: {sum(r.valid for r in records)} valid")
            records = stage.run(records)
            print(f"[{stage.name}] done:  {sum(r.valid for r in records)} valid")
        return records


def load_candidates(path: str | Path) -> list[Record]:
    out: list[Record] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(Record(raw=json.loads(line)))
    return out


def write_outputs(records: list[Record], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted = out_dir / "accepted.jsonl"
    rejected = out_dir / "rejected.jsonl"
    with accepted.open("w", encoding="utf-8") as fa, rejected.open("w", encoding="utf-8") as fr:
        for r in records:
            line = json.dumps(r.to_output(), ensure_ascii=False)
            (fa if r.valid else fr).write(line + "\n")
    print(f"wrote {accepted} and {rejected}")
