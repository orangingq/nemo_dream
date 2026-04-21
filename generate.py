from __future__ import annotations

import argparse
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf

from dashboard import serve_dashboard
from pipeline.designer_curator import DesignerCuratorConfig, write_jsonl
from pipeline.orchestrator import Pipeline, write_outputs
from pipeline.schema import Record
from pipeline.stages.s0_seed import SeedStage
from pipeline.stages.s1_decompose import DecomposeStage
from pipeline.stages.s2_map import MapStage
from pipeline.stages.s3_rewrite import RewriteStage
from pipeline.stages.s4_normalize import NormalizeStage
from pipeline.stages.s5_schema import SchemaStage
from pipeline.stages.s6_rules import RulesStage
from pipeline.stages.s7_safety import SafetyValidationStage
from pipeline.stages.s8_semantic import SemanticStage
from pipeline.stages.s9_quality import QualityStage
from pipeline.stages.s10_dedup import DedupStage
from pipeline.stages.s11_reward import RewardScoringStage
from pipeline.stages.s12_report import ReportWriterStage


CANONICAL_STAGE_FILES = {
    "seed": "stage0.jsonl",
    "stage4": "stage4.jsonl",
    "stage5": "stage5.jsonl",
    "stage6": "stage6.jsonl",
    "stage7": "stage7.jsonl",
    "stage8": "stage8.jsonl",
    "stage9": "stage9.jsonl",
    "stage10": "stage10.jsonl",
    "stage11": "stage11.jsonl",
    "stage12": "stage12.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the unified data designer -> curator pipeline end-to-end.",
    )
    parser.add_argument("--seed_dataset", default="data/curated/stage0.jsonl")
    parser.add_argument("--output_dir", default="data/curated")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="nvidia/nemotron-3-nano-30b-a3b")
    parser.add_argument("--base_url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--cultural_map", default="conf/cultural_map.json")
    parser.add_argument("--judge", choices=["mock", "nim"], default="mock")
    parser.add_argument("--chart", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


def load_runtime_config(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.judge = OmegaConf.load(f"conf/judge/{args.judge}.yaml")

    output_dir = Path(args.output_dir)
    cfg.generate.enabled = False
    cfg.generate.source_jsonl = str(output_dir / "stage3.jsonl")
    cfg.generate.out_jsonl = str(output_dir / "raw" / "normalized.jsonl")
    cfg.paths.out_dir = str(output_dir / "validation")
    cfg.report.chart = bool(args.chart)
    return cfg


def record_to_dict(record: Record) -> dict[str, Any]:
    payload = record.to_output()
    payload["valid"] = record.valid
    return payload


def write_record_snapshot(records: list[Record], path: Path) -> None:
    write_jsonl([record_to_dict(record) for record in records], path)


def write_stage7_summary(records: list[Record], path: Path) -> None:
    accepted = [record for record in records if record.valid]
    rejected = [record for record in records if not record.valid]
    summary = {
        "totals": {
            "input": len(records),
            "accepted": len(accepted),
            "rejected": len(rejected),
        },
        "reject_by_stage": dict(Counter(record.reject_reasons[0].stage for record in rejected if record.reject_reasons)),
        "reject_by_rule": dict(
            Counter((record.reject_reasons[0].rule or record.reject_reasons[0].stage) for record in rejected if record.reject_reasons)
        ),
        "accepted_ids": [record.id for record in accepted],
    }
    write_jsonl([summary], path)


def build_curator_stages(cfg: Any, judge: Any) -> list[Any]:
    return [
        NormalizeStage(cfg.generate, cfg.generate.out_jsonl),
        SchemaStage(),
        RulesStage(cfg.thresholds),
        SafetyValidationStage(judge),
        SemanticStage(judge, cfg.thresholds),
        QualityStage(cfg.thresholds, cfg.quality_weights),
        DedupStage(cfg.dedup),
        RewardScoringStage(judge),
        ReportWriterStage(cfg.paths.out_dir, chart=cfg.report.chart),
    ]


def maybe_start_dashboard(args: argparse.Namespace) -> None:
    if not args.visualize:
        return
    thread = threading.Thread(
        target=serve_dashboard,
        args=(args.host, args.port, Path(args.output_dir)),
        daemon=True,
    )
    thread.start()
    print(f"[visualize] live dashboard: http://{args.host}:{args.port}/")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation").mkdir(parents=True, exist_ok=True)

    maybe_start_dashboard(args)

    seed_stage = SeedStage(args.seed_dataset, output_dir / CANONICAL_STAGE_FILES["seed"], args.limit)
    seed_rows = seed_stage.run()
    seed_copy_path = seed_stage.output_path
    print(f"[generate] loaded {len(seed_rows)} seed rows from {args.seed_dataset}")
    print(f"[generate] wrote seed snapshot -> {seed_copy_path}")

    designer_cfg = DesignerCuratorConfig(
        model=args.model,
        base_url=args.base_url,
        cultural_map=args.cultural_map,
    )

    stage1 = DecomposeStage(designer_cfg, output_dir / "stage1.jsonl").run(seed_rows)
    print(f"[stage1] wrote {len(stage1)} rows -> {output_dir / 'stage1.jsonl'}")
    stage2 = MapStage(designer_cfg, output_dir / "stage2.jsonl").run(stage1)
    print(f"[stage2] wrote {len(stage2)} rows -> {output_dir / 'stage2.jsonl'}")
    stage3 = RewriteStage(designer_cfg, output_dir / "stage3.jsonl").run(stage2)
    print(f"[stage3] wrote {len(stage3)} rows -> {output_dir / 'stage3.jsonl'}")

    cfg = load_runtime_config(args)
    judge = instantiate(cfg.judge)

    snapshot_paths = {
        "S4_normalize": output_dir / CANONICAL_STAGE_FILES["stage4"],
        "S5_schema": output_dir / CANONICAL_STAGE_FILES["stage5"],
        "S6_rules": output_dir / CANONICAL_STAGE_FILES["stage6"],
        "S7_safety": output_dir / CANONICAL_STAGE_FILES["stage7"],
        "S8_semantic": output_dir / CANONICAL_STAGE_FILES["stage8"],
        "S9_quality": output_dir / CANONICAL_STAGE_FILES["stage9"],
        "S10_dedup": output_dir / CANONICAL_STAGE_FILES["stage10"],
        "S11_reward": output_dir / CANONICAL_STAGE_FILES["stage11"],
        "S12_report": output_dir / CANONICAL_STAGE_FILES["stage12"],
    }

    def after_stage(stage: Any, records: list[Record]) -> None:
        snapshot_path = snapshot_paths.get(stage.name)
        if snapshot_path and stage.name != "S12_report":
            write_record_snapshot(records, snapshot_path)
            print(f"[{stage.name}] snapshot -> {snapshot_path}")
        elif stage.name == "S12_report":
            write_outputs(records, cfg.paths.out_dir)
            write_stage7_summary(records, snapshot_path)
            print(f"[{stage.name}] snapshot -> {snapshot_path}")

    records: list[Record] = Pipeline(build_curator_stages(cfg, judge)).run([], after_stage=after_stage)

    print("[generate] full pipeline complete")
    print(f"[generate] curated stages -> {output_dir}")
    print(f"[generate] validation outputs -> {cfg.paths.out_dir}")


if __name__ == "__main__":
    main()
