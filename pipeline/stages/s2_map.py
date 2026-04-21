from __future__ import annotations

"""Stage 2: cultural reference mapping.

Input row shape:
{
  "id": "...",
  "text": "...",
  "metadata": {...},
  "decomposed": {...}
}

Output row shape (`stage2.jsonl`):
{
  "id": "...",
  "source_text": "...",
  "metadata": {...},
  "decomposed": {...},
  "mapped_refs": [
    {
      "term": "...",
      "ko": "...",
      "type": "...",
      "source": "dict|retriever|web+llm",
      "retrieved": false,
      "notes": "..."
    }
  ]
}
"""

from pathlib import Path
from typing import Any

from pipeline.designer_curator import DesignerCuratorConfig, nim_map_refs, write_jsonl


class MapStage:
    name = "stage2_map"

    def __init__(self, cfg: DesignerCuratorConfig, output_path: str | Path):
        self.cfg = cfg
        self.output_path = Path(output_path)

    def run(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stage2 = []
        for row in rows:
            stage2_row = {
                "id": row["id"],
                "source_text": row["text"],
                "metadata": row.get("metadata", {}),
                "decomposed": row["decomposed"],
                "mapped_refs": [],
            }
            stage2_row["mapped_refs"] = nim_map_refs(stage2_row, self.cfg)
            stage2.append(stage2_row)
        write_jsonl(stage2, self.output_path)
        return stage2
