from __future__ import annotations

"""Stage 0: seed dataset normalization.

Input JSONL shape:
{"id": "...", "text": "...", "metadata": {...}}

Accepted aliases for the source text field:
- text
- source_text
- en_text
- context

Output JSONL shape (`stage0.jsonl`):
{"id": "...", "text": "...", "metadata": {...}}
"""

from pathlib import Path
from typing import Any

from pipeline.designer_curator import load_seed_rows, write_jsonl


class SeedStage:
    name = "stage0_seed"

    def __init__(self, input_path: str | Path, output_path: str | Path, limit: int | None = None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.limit = limit

    def run(self) -> list[dict[str, Any]]:
        rows = load_seed_rows(self.input_path, self.limit)
        write_jsonl(rows, self.output_path)
        return rows
