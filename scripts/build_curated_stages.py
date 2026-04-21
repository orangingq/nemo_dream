from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline.designer_curator import DesignerCuratorConfig
from pipeline.stages.s0_seed import SeedStage
from pipeline.stages.s1_decompose import DecomposeStage
from pipeline.stages.s2_map import MapStage
from pipeline.stages.s3_rewrite import RewriteStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build curated Stage1~3 files from seed text.",
    )
    parser.add_argument("--input", default="data/curated/stage0.jsonl")
    parser.add_argument("--out-dir", default="data/curated")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-30b-a3b"))
    parser.add_argument("--base-url", default=os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"))
    parser.add_argument("--cultural-map", default="conf/cultural_map.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DesignerCuratorConfig(
        model=args.model,
        base_url=args.base_url,
        cultural_map=args.cultural_map,
    )
    out_dir = Path(args.out_dir)
    seed_rows = SeedStage(args.input, out_dir / "stage0.jsonl", args.limit).run()

    print(f"[curated] loaded {len(seed_rows)} rows from {args.input}")
    stage1_path = out_dir / "stage1.jsonl"
    stage2_path = out_dir / "stage2.jsonl"
    stage3_path = out_dir / "stage3.jsonl"
    stage1 = DecomposeStage(cfg, stage1_path).run(seed_rows)
    print(f"[curated] wrote {stage1_path} ({len(stage1)} rows)")
    stage2 = MapStage(cfg, stage2_path).run(stage1)
    print(f"[curated] wrote {stage2_path} ({len(stage2)} rows)")
    stage3 = RewriteStage(cfg, stage3_path).run(stage2)
    print(f"[curated] wrote {stage3_path} ({len(stage3)} rows)")


if __name__ == "__main__":
    main()
