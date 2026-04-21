from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from pipeline.schema import Record
from pipeline.stages.base import BatchStage


AXIS_METRICS = (
    "semantic_cosine",
    "property_preservation",
    "naturalness",
    "cultural_appropriateness",
    "register_consistency",
    "aggregate",
)


class ReportStage(BatchStage):
    name = "S7_report"

    def __init__(self, out_dir: str | Path, chart: bool = True):
        self.out_dir = Path(out_dir)
        self.chart = chart

    def run(self, records: list[Record]) -> list[Record]:
        accepted = [r for r in records if r.valid]
        rejected = [r for r in records if not r.valid]

        report = {
            "totals": {
                "input": len(records),
                "accepted": len(accepted),
                "rejected": len(rejected),
            },
            "reject_by_stage": _reject_by_stage(rejected),
            "reject_by_rule": _reject_by_rule(rejected),
            "accepted_distribution": _accepted_distribution(accepted),
            "quality_summary": _quality_summary(accepted),
        }

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        _print_report(report)

        if self.chart and accepted:
            _write_chart(accepted, self.out_dir / "score_distribution.png")
        return records


def _reject_by_stage(rejected: list[Record]) -> dict:
    return dict(Counter(r.reject_reasons[0].stage for r in rejected))


def _reject_by_rule(rejected: list[Record]) -> dict:
    return dict(Counter(
        (r.reject_reasons[0].rule or r.reject_reasons[0].stage)
        for r in rejected
    ))


def _accepted_distribution(accepted: list[Record]) -> dict:
    if not accepted:
        return {}
    rows = [r.sample.metadata.model_dump() for r in accepted if r.sample]
    df = pd.DataFrame(rows)
    out = {}
    for col in ("speech_act", "register", "estimated_age_group", "target_platform"):
        if col in df:
            out[col] = df[col].value_counts().to_dict()
    return out


def _quality_summary(accepted: list[Record]) -> dict:
    if not accepted:
        return {}
    rows = [r.quality.model_dump(exclude_none=True) for r in accepted]
    df = pd.DataFrame(rows)
    cols = [c for c in AXIS_METRICS if c in df]
    return {c: df[c].describe().to_dict() for c in cols}


def _print_report(report: dict) -> None:
    print("\n=== Validation Report ===")
    print(json.dumps(report["totals"], ensure_ascii=False))
    print("Reject by stage:", report["reject_by_stage"])
    print("Reject by rule:", report["reject_by_rule"])


def _write_chart(accepted: list[Record], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r.quality.model_dump(exclude_none=True) for r in accepted]
    df = pd.DataFrame(rows)
    cols = [c for c in AXIS_METRICS if c in df]
    if not cols:
        return

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_axes([0.08, 0.15, 0.38, 0.72])
    if "aggregate" in df:
        ax1.hist(df["aggregate"].dropna(), bins=10, edgecolor="black")
        ax1.set_title("Aggregate Score Distribution")
        ax1.set_xlabel("aggregate")
    ax1.set_ylabel("Count")

    ax2 = fig.add_axes([0.58, 0.15, 0.34, 0.72])
    means = {c: float(df[c].mean()) for c in cols}
    ax2.bar(list(means.keys()), list(means.values()), edgecolor="black")
    ax2.set_title("Average Score by Axis")
    ax2.set_xlabel("Metric")
    ax2.set_ylabel("Mean")
    for label in ax2.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    fig.suptitle("Pipeline Evaluation Overview")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Nemo / NeMo Curator equivalent (parity reference, not wired)
# ---------------------------------------------------------------------------
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


class NemoReportStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native report writer.

    Operates directly on a DocumentBatch (post-filter) and writes a JSON
    summary of accepted-row distribution and quality stats. Pure
    ProcessingStage so it composes into a Curator Pipeline without StageAdapter.
    """

    name = "S7_report_nemo"

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        report: dict = {"totals": {"accepted": int(len(df))}}
        for col in ("speech_act", "register", "estimated_age_group", "target_platform"):
            if col in df:
                report.setdefault("accepted_distribution", {})[col] = (
                    df[col].value_counts().to_dict()
                )
        quality_cols = [c for c in (
            "semantic_cosine", "_semantic_cosine", "fineweb_score",
            "property_preservation", "naturalness",
            "cultural_appropriateness", "register_consistency", "aggregate",
        ) if c in df]
        if quality_cols:
            report["quality_summary"] = {
                c: df[c].describe().to_dict() for c in quality_cols
            }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "report_nemo.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str)
        )
        return batch
