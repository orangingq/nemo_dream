from __future__ import annotations

"""Stage 3: Korean rewrite generation.

Input row shape:
{
  "id": "...",
  "source_text": "...",
  "metadata": {...},
  "decomposed": {...},
  "mapped_refs": [...]
}

Output row shape (`stage3.jsonl`):
{
  "id": "...",
  "source_text": "...",
  "metadata": {...},
  "decomposed": {...},
  "mapped_refs": [...],
  "generation": {
    "target_platform": "...",
    "target_age_group": "...",
    "target_community": "...",
    "target_gender_style": "neutral",
    "rewritten_text": "...",
    "generation_notes": "..."
  }
}
"""

from pathlib import Path
from typing import Any

from pipeline.designer_curator import DesignerCuratorConfig, nim_rewrite, write_jsonl


class RewriteStage:
    name = "stage3_rewrite"

    def __init__(self, cfg: DesignerCuratorConfig, output_path: str | Path):
        self.cfg = cfg
        self.output_path = Path(output_path)

    def run(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stage3 = []
        for row in rows:
            stage3.append({**row, "generation": nim_rewrite(row, self.cfg)})
        write_jsonl(stage3, self.output_path)
        return stage3
