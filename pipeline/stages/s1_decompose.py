from __future__ import annotations

"""Stage 1: sociolinguistic decomposition.

Input row shape:
{"id": "...", "text": "...", "metadata": {...}}

Output row shape (`stage1.jsonl`):
{
  "id": "...",
  "text": "...",
  "metadata": {...},
  "decomposed": {
    "source_text": "...",
    "speech_act": "...",
    "register": "...",
    "emotion": {"type": "...", "intensity": 1},
    "cultural_refs": [{"type": "...", "term": "..."}],
    "internet_markers": {...},
    "estimated_age_group": "...",
    "platform_fit": ["..."]
  }
}
"""

from pathlib import Path
from typing import Any

from pipeline.designer_curator import DesignerCuratorConfig, nim_decompose, write_jsonl


class DecomposeStage:
    name = "stage1_decompose"

    def __init__(self, cfg: DesignerCuratorConfig, output_path: str | Path):
        self.cfg = cfg
        self.output_path = Path(output_path)

    def run(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stage1 = []
        for row in rows:
            stage1.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": row.get("metadata", {}),
                    "decomposed": nim_decompose(row["text"], self.cfg),
                }
            )
        write_jsonl(stage1, self.output_path)
        return stage1
