from __future__ import annotations
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pipeline.orchestrator import Pipeline, load_candidates, write_outputs
from pipeline.stages.s4_normalize import NormalizeStage
from pipeline.stages.s5_schema import SchemaStage
from pipeline.stages.s6_rules import RulesStage
from pipeline.stages.s7_safety import SafetyValidationStage
from pipeline.stages.s8_semantic import SemanticStage
from pipeline.stages.s9_quality import QualityStage
from pipeline.stages.s10_dedup import DedupStage
from pipeline.stages.s11_reward import RewardScoringStage
from pipeline.stages.s12_report import ReportWriterStage


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    judge = instantiate(cfg.judge)

    validation_stages = [
        SchemaStage(),
        RulesStage(cfg.thresholds),
        SafetyValidationStage(judge),
        SemanticStage(judge, cfg.thresholds),
        QualityStage(cfg.thresholds, cfg.quality_weights),
        DedupStage(cfg.dedup),
        RewardScoringStage(judge),
        ReportWriterStage(cfg.paths.out_dir, chart=cfg.report.chart),
    ]

    if cfg.generate.enabled or cfg.generate.get("source_jsonl"):
        pipe = Pipeline([NormalizeStage(cfg.generate, cfg.generate.out_jsonl), *validation_stages])
        records = pipe.run([])
    else:
        pipe = Pipeline(validation_stages)
        records = pipe.run(load_candidates(cfg.paths.candidates))

    write_outputs(records, cfg.paths.out_dir)


if __name__ == "__main__":
    main()
