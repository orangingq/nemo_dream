from __future__ import annotations
try:
    from datasketch import MinHash, MinHashLSH
except ModuleNotFoundError:
    class MinHash:
        def __init__(self, num_perm: int):
            self.tokens: set[str] = set()

        def update(self, token: bytes) -> None:
            self.tokens.add(token.decode("utf-8"))

    class MinHashLSH:
        def __init__(self, threshold: float, num_perm: int):
            self.threshold = threshold
            self.items: dict[str, MinHash] = {}

        def query(self, mh: MinHash) -> list[str]:
            return [
                key for key, other in self.items.items()
                if _jaccard(mh.tokens, other.tokens) >= self.threshold
            ]

        def insert(self, key: str, mh: MinHash) -> None:
            self.items[key] = mh
from omegaconf import DictConfig
from pipeline.schema import Record
from pipeline.stages.base import BatchStage


class QualityFilterStage(BatchStage):
    name = "S5_filter"

    def __init__(self, thresholds: DictConfig, weights: DictConfig):
        self.t = thresholds
        self.w = weights

    def run(self, records: list[Record]) -> list[Record]:
        for r in records:
            if not r.valid:
                continue
            agg = _aggregate(r, self.w)
            r.quality.aggregate = agg
            if agg < self.t.aggregate:
                r.reject(
                    "S5",
                    detail=f"aggregate={agg:.2f} below {self.t.aggregate}",
                    rule="quality_threshold",
                )
        return records


class FuzzyDedupStage(BatchStage):
    name = "S5_dedup"

    def __init__(self, dedup_cfg: DictConfig):
        self.threshold = dedup_cfg.jaccard_threshold
        self.num_perm = dedup_cfg.num_perm

    def run(self, records: list[Record]) -> list[Record]:
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for r in records:
            if not r.valid or r.sample is None:
                continue
            mh = _make_minhash(r.sample.ko_text, self.num_perm)
            matches = lsh.query(mh)
            if matches:
                r.reject(
                    "S5",
                    detail=f"near-duplicate of {matches[0]}",
                    rule="dedup",
                )
                continue
            lsh.insert(r.sample.id, mh)
        return records


def _aggregate(r: Record, w: DictConfig) -> float:
    q = r.quality
    return (
        w.semantic * (q.semantic_cosine or 0.0) * 5.0
        + w.property * (q.property_preservation or 0)
        + w.naturalness * (q.naturalness or 0)
        + w.cultural * (q.cultural_appropriateness or 0)
        + w.register * (q.register_consistency or 0)
    )


def _make_minhash(text: str, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for token in _shingles(text, k=3):
        mh.update(token.encode("utf-8"))
    return mh


def _shingles(text: str, k: int) -> list[str]:
    s = text.replace(" ", "")
    if len(s) < k:
        return [s]
    return [s[i:i + k] for i in range(len(s) - k + 1)]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


# ---------------------------------------------------------------------------
# Nemo / NeMo Curator equivalent (parity reference, not wired)
# ---------------------------------------------------------------------------
try:
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.text.classifiers import FineWebNemotronEduClassifier
    from nemo_curator.tasks import DocumentBatch
except Exception:
    class ProcessingStage:
        def __class_getitem__(cls, item):
            return cls

    class DocumentBatch:
        pass

    FineWebNemotronEduClassifier = None


class NemoFineWebQualityStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator-native quality filter via FineWeb-Nemotron Edu classifier.

    Loads `nvidia/nemocurator-fineweb-nemotron-4-edu-classifier` and writes a
    `fineweb_score` column (0-5). A downstream Curator `Filter` keeps rows
    whose `fineweb_score >= threshold` (default 3.0).
    """

    name = "S5_quality_nemo_fineweb"

    def __init__(self, text_field: str = "ko_text", threshold: float = 3.0):
        if FineWebNemotronEduClassifier is None:
            raise ImportError("FineWebNemotronEduClassifier requires compatible nemo-curator/transformers deps.")
        self.threshold = threshold
        self.classifier = FineWebNemotronEduClassifier(
            text_field=text_field,
            label_field="fineweb_score",
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.classifier.inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.classifier.outputs()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        return self.classifier.process(batch)


def nemo_fuzzy_dedup_workflow(
    input_path: str,
    output_path: str,
    text_field: str = "ko_text",
    char_ngrams: int = 24,
    num_bands: int = 20,
    minhashes_per_band: int = 13,
    jaccard_threshold: float = 0.8,
):
    """Curator-native fuzzy dedup workflow (cuDF/cuPy required, GPU-only).

    Imported lazily so this module remains CPU-importable. Composes Curator's
    MinHash → LSH → buckets-to-edges → connected-components → identify-duplicates
    workflow on a Ray cluster with cuDF backend.
    """
    from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

    workflow = FuzzyDeduplicationWorkflow(
        input_path=input_path,
        output_path=output_path,
        text_field=text_field,
        char_ngrams=char_ngrams,
        num_bands=num_bands,
        minhashes_per_band=minhashes_per_band,
        jaccard_threshold=jaccard_threshold,
    )
    workflow.run()


def nemo_exact_dedup_workflow(
    input_path: str,
    output_path: str,
    text_field: str = "ko_text",
):
    """Curator-native exact dedup workflow (cuDF required, GPU-only)."""
    from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

    workflow = ExactDeduplicationWorkflow(
        input_path=input_path,
        output_path=output_path,
        text_field=text_field,
    )
    workflow.run()


def nemo_semantic_dedup_workflow(
    input_path: str,
    output_path: str,
    text_field: str = "ko_text",
    eps: float = 0.07,
):
    """Curator-native semantic dedup workflow (cuPy required, GPU-only).

    Decomposes to embeddings → KMeans → pairwise-cosine within clusters →
    identify-duplicates over an `eps` threshold.
    """
    from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow

    workflow = SemanticDeduplicationWorkflow(
        input_path=input_path,
        output_path=output_path,
        text_field=text_field,
        eps=eps,
    )
    workflow.run()
