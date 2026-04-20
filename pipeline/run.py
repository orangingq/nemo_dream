from __future__ import annotations
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pipeline.orchestrator import Pipeline, load_candidates, write_outputs
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

    validation_stages = [
        SchemaValidationStage(),
        RuleValidatorStage(cfg.thresholds),
        SafetyStage(judge),
        SemanticJudgeStage(judge, cfg.thresholds),
        QualityFilterStage(cfg.thresholds, cfg.quality_weights),
        FuzzyDedupStage(cfg.dedup),
        RewardStage(judge),
        ReportStage(cfg.paths.out_dir, chart=cfg.report.chart),
    ]

    if cfg.generate.enabled:
        pipe = Pipeline([GenerateStage(cfg.generate, cfg.generate.out_jsonl), *validation_stages])
        records = pipe.run([])
    else:
        pipe = Pipeline(validation_stages)
        records = pipe.run(load_candidates(cfg.paths.candidates))

    write_outputs(records, cfg.paths.out_dir)


if __name__ == "__main__":
    main()
