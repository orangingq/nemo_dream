from __future__ import annotations
import json
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader

from pipeline.curator_bridge import RECORD_COL, StageAdapter
from pipeline.schema import Record
from pipeline.stages.s0_generate import GenerateStage
from pipeline.stages.s1_schema import SchemaValidationStage
from pipeline.stages.s2_rules import RuleValidatorStage
from pipeline.stages.s3_safety import SafetyStage
from pipeline.stages.s4_semantic import SemanticJudgeStage
from pipeline.stages.s5_filter import FuzzyDedupStage, QualityFilterStage
from pipeline.stages.s6_reward import RewardStage
from pipeline.stages.s7_report import ReportStage


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    judge = instantiate(cfg.judge)

    if cfg.generate.enabled:
        GenerateStage(cfg.generate, cfg.generate.out_jsonl).run([])
        candidates_path = str(cfg.generate.out_jsonl)
    else:
        candidates_path = str(cfg.paths.candidates)

    stages = [
        StageAdapter(SchemaValidationStage()),
        StageAdapter(RuleValidatorStage(cfg.thresholds)),
        StageAdapter(SafetyStage(judge)),
        StageAdapter(SemanticJudgeStage(judge, cfg.thresholds)),
        StageAdapter(QualityFilterStage(cfg.thresholds, cfg.quality_weights)),
        StageAdapter(FuzzyDedupStage(cfg.dedup)),
        StageAdapter(RewardStage(judge)),
        StageAdapter(ReportStage(cfg.paths.out_dir, chart=cfg.report.chart)),
    ]

    pipe = Pipeline(
        name="nemotron_k_sovereign_validation",
        description="EN->KO SDG validation pipeline (NeMo Curator driven)",
        stages=[JsonlReader(file_paths=candidates_path), *stages],
    )

    results = pipe.run(executor=RayActorPoolExecutor()) or []
    _write_outputs(results, cfg.paths.out_dir)


def _write_outputs(tasks, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted_path = out / "accepted.jsonl"
    rejected_path = out / "rejected.jsonl"
    with accepted_path.open("w", encoding="utf-8") as fa, rejected_path.open("w", encoding="utf-8") as fr:
        for task in tasks:
            df = task.to_pandas()
            if RECORD_COL not in df.columns:
                continue
            for serialized in df[RECORD_COL]:
                r = Record.model_validate_json(serialized)
                line = json.dumps(r.to_output(), ensure_ascii=False)
                (fa if r.valid else fr).write(line + "\n")
    print(f"wrote {accepted_path} and {rejected_path}")


if __name__ == "__main__":
    main()
