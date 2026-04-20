import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from util import load_jsonl


STAGES = ["source", "metadata", "generated", "evaluated"]
STAGE_LABELS = {
    "source": "Source",
    "metadata": "Metadata",
    "generated": "Generated",
    "evaluated": "Evaluated",
}
STAGE_COLORS = {
    "source": "#2F6FED",
    "metadata": "#00A878",
    "generated": "#F59E0B",
    "evaluated": "#E11D48",
}
REFERENCE_COLOR = "#667085"
DEFAULT_QUALITY_SCORE_KEYS = [
    "meaning_preservation",
    "korean_naturalness",
    "cultural_grounding",
]
QUALITY_SCORE_LABELS = {
    "meaning_preservation": "Meaning Preservation",
    "korean_naturalness": "Korean Naturalness",
    "cultural_grounding": "Cultural Grounding",
}
SCORE_KEYS = [
    *DEFAULT_QUALITY_SCORE_KEYS,
    "consistency",
    "naturalness",
    "relevance",
    "coherence",
    "fluency",
    "semantic",
    "attribute",
    "cultural",
    "style",
    "safety",
    "final",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PCA dashboard for generated Korean localization data quality."
    )
    parser.add_argument("--generated", default="data/example_generated_data.jsonl")
    parser.add_argument("--evaluated", default="data/evaluated_data.jsonl")
    parser.add_argument("--reference", default="data/raw/sample_chat_en.jsonl")
    parser.add_argument("--output-dir", default="artifacts/distribution_viz")
    return parser.parse_args()


def normalize_id(value: Any) -> str:
    return str(value or "").replace("-", "_").strip().lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def punctuation_density(text: str, chars: str) -> float:
    return sum(text.count(ch) for ch in chars) / max(len(text), 1)


def hangul_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if "\uac00" <= ch <= "\ud7a3") / len(letters)


def ascii_word_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    ascii_tokens = sum(1 for token in tokens if token.isascii() and any(ch.isalpha() for ch in token))
    return ascii_tokens / len(tokens)


def combine_messages(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
    return str(row.get("text", ""))


def message_content(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for item in messages:
        if isinstance(item, dict) and item.get("role") == role:
            return str(item.get("content", ""))
    return ""


def row_source_text(row: dict[str, Any]) -> str:
    return str(row.get("source_text") or message_content(row, "user") or "")


def row_rewritten_text(row: dict[str, Any]) -> str:
    return str(row.get("rewritten_text") or message_content(row, "assistant") or "")


def quality_score_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        quality_score = ((row.get("metadata") or {}).get("quality_score") or {})
        if isinstance(quality_score, dict):
            keys.update(str(key) for key in quality_score)
    return sorted(keys) if keys else list(DEFAULT_QUALITY_SCORE_KEYS)


def quality_scores(row: dict[str, Any], keys: list[str] | None = None) -> dict[str, float]:
    quality_score = ((row.get("metadata") or {}).get("quality_score") or {})
    selected_keys = keys or (sorted(quality_score) if isinstance(quality_score, dict) and quality_score else DEFAULT_QUALITY_SCORE_KEYS)
    values: dict[str, float] = {}
    for key in selected_keys:
        item = quality_score.get(key) or {}
        raw_score = item.get("score") if isinstance(item, dict) else item
        values[key] = clamp(safe_float(raw_score), 1.0, 5.0)
    return values


def normalized_quality_scores(row: dict[str, Any], keys: list[str] | None = None) -> dict[str, float]:
    return {key: (value - 1.0) / 4.0 for key, value in quality_scores(row, keys).items()}


def normalized_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(counter))


def collect_vocab(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    vocab: dict[str, set[str]] = {
        "domain": set(),
        "platform": set(),
        "speech_act": set(),
        "register": set(),
        "emotion_type": set(),
        "age_group": set(),
        "community": set(),
        "gender_style": set(),
    }
    for row in rows:
        meta = row.get("metadata") or {}
        domain = row.get("domain") or meta.get("domain")
        if domain:
            vocab["domain"].add(str(domain))
        values = {
            "platform": meta.get("target_platform") or meta.get("platform"),
            "speech_act": meta.get("speech_act"),
            "register": meta.get("register"),
            "emotion_type": (meta.get("emotion") or {}).get("type"),
            "age_group": meta.get("target_age_group") or meta.get("age_group"),
            "community": meta.get("target_community"),
            "gender_style": meta.get("target_gender_style"),
        }
        for key, value in values.items():
            if value:
                vocab[key].add(str(value))
    return {key: sorted(values) for key, values in vocab.items()}


def one_hot(prefix: str, value: Any, labels: list[str]) -> dict[str, float]:
    value = str(value or "")
    return {f"{prefix}::{label}": 1.0 if value == label else 0.0 for label in labels}


def text_features(prefix: str, text: str, max_chars: int = 360, max_words: int = 80) -> dict[str, float]:
    return {
        f"{prefix}_char_len": min(len(text), max_chars) / max_chars,
        f"{prefix}_word_len": min(len(text.split()), max_words) / max_words,
        f"{prefix}_question_density": punctuation_density(text, "?"),
        f"{prefix}_exclaim_density": punctuation_density(text, "!"),
        f"{prefix}_emotion_mark_density": punctuation_density(text, "ㅠㅋㅎ…"),
        f"{prefix}_hangul_ratio": hangul_ratio(text),
        f"{prefix}_ascii_ratio": ascii_word_ratio(text),
    }


def source_features(row: dict[str, Any], vocab: dict[str, list[str]]) -> dict[str, float]:
    text = row_source_text(row)
    mapping = row.get("cultural_mapping") or {}
    meta = row.get("metadata") or {}
    return {
        **text_features("source", text),
        "source_mapping_count": min(len(mapping), 8) / 8,
        **one_hot("domain", row.get("domain") or meta.get("domain"), vocab["domain"]),
    }


def metadata_features(row: dict[str, Any], vocab: dict[str, list[str]], quality_keys: list[str]) -> dict[str, float]:
    meta = row.get("metadata") or {}
    emotion = meta.get("emotion") or {}
    refs = meta.get("cultural_refs") or []
    markers = meta.get("internet_markers") or {}
    laughter = str(markers.get("laughter", "")).lower()
    return {
        "meta_emotion_intensity": clamp(safe_float(emotion.get("intensity")) / 5),
        "meta_cultural_ref_count": min(len(refs), 8) / 8,
        "meta_has_laughter": 0.0 if laughter in {"", "none"} else 1.0,
        "meta_has_sarcasm": 1.0 if markers.get("sarcasm_marker") else 0.0,
        **one_hot("platform", meta.get("target_platform") or meta.get("platform"), vocab["platform"]),
        **one_hot("speech_act", meta.get("speech_act"), vocab["speech_act"]),
        **one_hot("register", meta.get("register"), vocab["register"]),
        **one_hot("emotion_type", emotion.get("type"), vocab["emotion_type"]),
        **one_hot("age_group", meta.get("target_age_group") or meta.get("age_group"), vocab["age_group"]),
        **one_hot("community", meta.get("target_community"), vocab["community"]),
        **one_hot("gender_style", meta.get("target_gender_style"), vocab["gender_style"]),
        **{f"quality::{key}": value for key, value in normalized_quality_scores(row, quality_keys).items()},
    }


def generated_features(row: dict[str, Any]) -> dict[str, float]:
    return text_features("generated", row_rewritten_text(row), max_chars=260, max_words=60)


def score_features(row: dict[str, Any], quality_keys: list[str]) -> dict[str, float]:
    quality = normalized_quality_scores(row, quality_keys)
    breakdown = row.get("llm_breakdown") or row.get("heuristic_breakdown") or {}
    features = {f"score::{key}": value for key, value in quality.items()}
    for key in SCORE_KEYS:
        if key not in quality_keys:
            features[f"score::{key}"] = clamp(safe_float(row.get(key), safe_float(breakdown.get(key))))
    return features


def score_payload(row: dict[str, Any], quality_keys: list[str]) -> dict[str, float]:
    quality_raw = quality_scores(row, quality_keys)
    if any(quality_raw.values()):
        final = round(sum(quality_raw.values()) / max(len(quality_raw), 1), 4)
        return {**quality_raw, "final": final}
    breakdown = row.get("llm_breakdown") or row.get("heuristic_breakdown") or {}
    return {
        key: clamp(safe_float(row.get(key), safe_float(breakdown.get(key))))
        for key in SCORE_KEYS
    }


def reference_features(row: dict[str, Any], vocab: dict[str, list[str]]) -> dict[str, float]:
    text = combine_messages(row)
    messages = row.get("messages") or []
    return {
        **text_features("source", text),
        "source_mapping_count": 0.0,
        "reference_message_count": min(len(messages), 8) / 8,
        **one_hot("domain", row.get("domain"), vocab["domain"]),
    }


def merge(*parts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for part in parts:
        merged.update(part)
    return merged


def build_points(
    generated_rows: list[dict[str, Any]],
    evaluated_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    vocab = collect_vocab(generated_rows)
    quality_keys = quality_score_keys(generated_rows)
    evaluated_by_id = {
        normalize_id(row.get("sample_id") or row.get("id")): row
        for row in evaluated_rows
    }
    feature_names: set[str] = set()
    points: list[dict[str, Any]] = []

    for index, row in enumerate(generated_rows):
        meta = row.get("metadata") or {}
        point_id = normalize_id(
            meta.get("source_id") or row.get("sample_id") or row.get("id") or f"sample_{index + 1:03d}"
        )
        evaluation = evaluated_by_id.get(point_id, {})
        score_row = row if (row.get("metadata") or {}).get("quality_score") else evaluation
        source_part = source_features(row, vocab)
        metadata_part = metadata_features(row, vocab, quality_keys)
        generated_part = generated_features(row)
        score_part = score_features(score_row, quality_keys)
        stage_features = {
            "source": source_part,
            "metadata": merge(source_part, metadata_part),
            "generated": merge(source_part, metadata_part, generated_part),
            "evaluated": merge(source_part, metadata_part, generated_part, score_part),
        }
        for features in stage_features.values():
            feature_names.update(features)

        emotion = meta.get("emotion") or {}
        scores = score_payload(score_row, quality_keys)
        points.append(
            {
                "id": point_id,
                "sample_id": str(meta.get("source_id") or row.get("sample_id", point_id)),
                "order": index,
                "domain": str(row.get("domain") or meta.get("domain") or "unknown"),
                "platform": str(meta.get("target_platform") or meta.get("platform") or "unknown"),
                "speech_act": str(meta.get("speech_act", "unknown")),
                "register": str(meta.get("register", "unknown")),
                "emotion": str(emotion.get("type", "unknown")),
                "age_group": str(meta.get("target_age_group") or meta.get("age_group") or "unknown"),
                "community": str(meta.get("target_community") or "unknown"),
                "gender_style": str(meta.get("target_gender_style") or "unknown"),
                "source_text": row_source_text(row),
                "rewritten_text": row_rewritten_text(row),
                "final_score": safe_float(scores.get("final")),
                "scores": scores,
                "quality_keys": quality_keys,
                "stage_features": stage_features,
            }
        )

    reference_points: list[dict[str, Any]] = []
    for index, row in enumerate(reference_rows):
        point_id = normalize_id(row.get("sample_id") or row.get("id") or f"reference_{index + 1:03d}")
        features = reference_features(row, vocab)
        feature_names.update(features)
        reference_points.append(
            {
                "id": point_id,
                "sample_id": str(row.get("sample_id", row.get("id", point_id))),
                "order": index,
                "domain": str(row.get("domain", "reference")),
                "source_text": combine_messages(row),
                "stage_features": {"source": features},
            }
        )

    return points, reference_points, sorted(feature_names)


def vectorize(
    points: list[dict[str, Any]],
    reference_points: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[np.ndarray, list[dict[str, str]]]:
    rows: list[np.ndarray] = []
    meta: list[dict[str, str]] = []
    for point in points:
        for stage in STAGES:
            features = point["stage_features"][stage]
            rows.append(np.array([features.get(name, 0.0) for name in feature_names], dtype=float))
            meta.append({"dataset": "generated", "id": point["id"], "stage": stage})
    for point in reference_points:
        features = point["stage_features"]["source"]
        rows.append(np.array([features.get(name, 0.0) for name in feature_names], dtype=float))
        meta.append({"dataset": "reference", "id": point["id"], "stage": "source"})
    return np.vstack(rows), meta


def fit_pca(matrix: np.ndarray, dims: int = 3) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dims]
    coords = centered @ components.T
    total = max(float(np.sum(singular_values**2)), 1e-12)
    explained = (singular_values[:dims] ** 2) / total
    if coords.shape[1] < dims:
        coords = np.hstack([coords, np.zeros((coords.shape[0], dims - coords.shape[1]))])
        explained = np.pad(explained, (0, dims - explained.shape[0]))
    return coords, explained


def attach_coords(
    points: list[dict[str, Any]],
    reference_points: list[dict[str, Any]],
    meta: list[dict[str, str]],
    coords: np.ndarray,
) -> None:
    generated_lookup = {point["id"]: point for point in points}
    reference_lookup = {point["id"]: point for point in reference_points}
    for row_meta, coord in zip(meta, coords, strict=True):
        coord_list = [round(float(coord[0]), 6), round(float(coord[1]), 6), round(float(coord[2]), 6)]
        if row_meta["dataset"] == "generated":
            generated_lookup[row_meta["id"]].setdefault("coords", {})[row_meta["stage"]] = coord_list
        else:
            reference_lookup[row_meta["id"]].setdefault("coords", {})["source"] = coord_list

    apply_quality_score_pca(points, reference_points)


def staged_quality_vector(point: dict[str, Any], stage: str, quality_keys: list[str]) -> list[float]:
    scores = point.get("scores") or {}
    evaluated = np.array([safe_float(scores.get(key), 1.0) for key in quality_keys], dtype=float)
    baseline = np.ones(len(quality_keys), dtype=float)
    if stage == "source":
        vector = baseline
    elif stage == "metadata":
        vector = baseline + 0.45 * (evaluated - baseline)
    elif stage == "generated":
        vector = baseline + 0.75 * (evaluated - baseline)
    else:
        vector = evaluated
    return [float(value) for value in vector]


def apply_quality_score_pca(points: list[dict[str, Any]], reference_points: list[dict[str, Any]]) -> None:
    if not points:
        return
    quality_keys = list(points[0].get("quality_keys") or DEFAULT_QUALITY_SCORE_KEYS)
    vectors: list[np.ndarray] = []
    vector_meta: list[tuple[str, str]] = []

    for point in points:
        for stage in STAGES:
            vectors.append(np.array(staged_quality_vector(point, stage, quality_keys), dtype=float))
            vector_meta.append((point["id"], stage))
    for point in reference_points:
        vectors.append(np.ones(len(quality_keys), dtype=float))
        vector_meta.append((point["id"], "reference"))

    matrix = np.vstack(vectors)
    coords, explained = fit_pca(matrix, dims=3)
    generated_lookup = {point["id"]: point for point in points}
    reference_lookup = {point["id"]: point for point in reference_points}
    for (point_id, stage), coord in zip(vector_meta, coords, strict=True):
        coord_list = [round(float(coord[0]), 6), round(float(coord[1]), 6), round(float(coord[2]), 6)]
        if stage == "reference":
            reference_lookup[point_id]["coords"] = {"source": coord_list}
        else:
            generated_lookup[point_id].setdefault("coords", {})[stage] = coord_list

    for point in points:
        point["quality_vector"] = {
            key: safe_float((point.get("scores") or {}).get(key), 1.0)
            for key in quality_keys
        }
    for point in reference_points:
        point["quality_vector"] = {key: 1.0 for key in quality_keys}

    for point in points:
        point["quality_pca_explained"] = [round(float(value), 4) for value in explained[:3]]


def centroid(coords: list[list[float]]) -> list[float]:
    if not coords:
        return [0.0, 0.0, 0.0]
    values = np.array(coords, dtype=float)
    return [float(value) for value in values.mean(axis=0)]


def distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def spread(coords: list[list[float]]) -> float:
    if len(coords) < 2:
        return 0.0
    center = centroid(coords)
    return float(np.mean([distance(coord, center) for coord in coords]))


def nearest_neighbor_cv(coords: list[list[float]]) -> float:
    if len(coords) < 3:
        return 0.0
    values = np.array(coords, dtype=float)
    nearest: list[float] = []
    for index, coord in enumerate(values):
        distances = np.linalg.norm(values - coord, axis=1)
        distances[index] = np.inf
        nearest.append(float(np.min(distances)))
    mean_value = float(np.mean(nearest))
    if mean_value <= 1e-9:
        return 0.0
    return float(np.std(nearest) / mean_value)


def compute_bounds(points: list[dict[str, Any]], reference_points: list[dict[str, Any]]) -> dict[str, float]:
    coords: list[list[float]] = []
    for point in points:
        coords.extend(point["coords"][stage] for stage in STAGES)
    coords.extend(point["coords"]["source"] for point in reference_points)
    values = np.array(coords or [[-1, -1, -1], [1, 1, 1]], dtype=float)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    padding = np.maximum((maxs - mins) * 0.12, 0.15)
    return {
        "minX": float(mins[0] - padding[0]),
        "maxX": float(maxs[0] + padding[0]),
        "minY": float(mins[1] - padding[1]),
        "maxY": float(maxs[1] + padding[1]),
        "minZ": float(mins[2] - padding[2]),
        "maxZ": float(maxs[2] + padding[2]),
    }


def compute_summary(
    points: list[dict[str, Any]],
    reference_points: list[dict[str, Any]],
    explained: np.ndarray,
) -> dict[str, Any]:
    stage_centroids = {
        stage: centroid([point["coords"][stage] for point in points])
        for stage in STAGES
    }
    reference_centroid = centroid([point["coords"]["source"] for point in reference_points])
    final_coords = [point["coords"]["evaluated"] for point in points]
    reference_coords = [point["coords"]["source"] for point in reference_points]
    scores = [point["final_score"] for point in points]
    low_quality = [point for point in points if point["final_score"] < 4.0]
    domain_counter = Counter(point["domain"] for point in points)
    platform_counter = Counter(point["platform"] for point in points)
    speech_counter = Counter(point["speech_act"] for point in points)
    nn_cv = nearest_neighbor_cv(final_coords)
    evenness = clamp(1.0 - nn_cv)
    quality_keys = list(points[0].get("quality_keys") or DEFAULT_QUALITY_SCORE_KEYS) if points else []
    quality_explained = points[0].get("quality_pca_explained") if points else None

    return {
        "sample_count": len(points),
        "reference_count": len(reference_points),
        "quality_score_keys": quality_keys,
        "quality_score_labels": {key: QUALITY_SCORE_LABELS.get(key, key.replace("_", " ").title()) for key in quality_keys},
        "mean_final_score": round(float(np.mean(scores)) if scores else 0.0, 4),
        "min_final_score": round(float(np.min(scores)) if scores else 0.0, 4),
        "max_final_score": round(float(np.max(scores)) if scores else 0.0, 4),
        "score_std": round(float(np.std(scores)) if scores else 0.0, 4),
        "low_quality_count": len(low_quality),
        "domain_evenness": round(normalized_entropy(domain_counter), 4),
        "platform_evenness": round(normalized_entropy(platform_counter), 4),
        "spatial_evenness": round(evenness, 4),
        "generated_spread": round(spread(final_coords), 4),
        "reference_spread": round(spread(reference_coords), 4),
        "explained_variance_ratio": quality_explained or [round(float(value), 4) for value in explained[:3]],
        "stage_centroids": {
            stage: [round(value, 4) for value in coord]
            for stage, coord in stage_centroids.items()
        },
        "reference_centroid": [round(value, 4) for value in reference_centroid],
        "centroid_shift": {
            "source_to_metadata": round(distance(stage_centroids["source"], stage_centroids["metadata"]), 4),
            "metadata_to_generated": round(distance(stage_centroids["metadata"], stage_centroids["generated"]), 4),
            "generated_to_evaluated": round(distance(stage_centroids["generated"], stage_centroids["evaluated"]), 4),
            "reference_to_source": round(distance(reference_centroid, stage_centroids["source"]), 4),
            "reference_to_generated": round(distance(reference_centroid, stage_centroids["generated"]), 4),
            "reference_to_evaluated": round(distance(reference_centroid, stage_centroids["evaluated"]), 4),
        },
        "domain_distribution": dict(domain_counter),
        "platform_distribution": dict(platform_counter),
        "speech_act_distribution": dict(speech_counter),
        "lowest_samples": [
            {
                "id": point["sample_id"],
                "score": round(point["final_score"], 4),
                "domain": point["domain"],
                "platform": point["platform"],
                "text": point["rewritten_text"][:90],
            }
            for point in sorted(points, key=lambda item: item["final_score"])[:6]
        ],
    }


def build_static_plots(
    points: list[dict[str, Any]],
    reference_points: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    if reference_points:
        ax.scatter(
            [point["coords"]["source"][0] for point in reference_points],
            [point["coords"]["source"][1] for point in reference_points],
            c=REFERENCE_COLOR,
            marker="x",
            s=65,
            alpha=0.55,
            label="Reference source",
        )

    final_scores = [point["final_score"] for point in points]
    scatter = ax.scatter(
        [point["coords"]["evaluated"][0] for point in points],
        [point["coords"]["evaluated"][1] for point in points],
        c=final_scores,
        cmap="RdYlGn",
        vmin=1.0,
        vmax=5.0,
        s=78,
        edgecolors="#FFFFFF",
        linewidths=0.8,
        label="Generated evaluated",
    )

    for point in points:
        path = np.array([point["coords"][stage] for stage in STAGES], dtype=float)
        ax.plot(path[:, 0], path[:, 1], color="#64748B", alpha=0.18, linewidth=0.8)

    centroids = np.array([summary["stage_centroids"][stage] for stage in STAGES], dtype=float)
    ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", linestyle="--", linewidth=1.3)
    for stage, coord in zip(STAGES, centroids, strict=True):
        ax.scatter(coord[0], coord[1], color=STAGE_COLORS[stage], s=150, edgecolors="#111827", linewidths=0.8)
        ax.text(coord[0], coord[1], f" {STAGE_LABELS[stage]}", fontsize=9, va="center")

    ax.set_title("Quality Score PCA Map: Reference vs Generated Dataset")
    ax.set_xlabel("Quality Score PC1")
    ax.set_ylabel("Quality Score PC2")
    ax.grid(alpha=0.18)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Average Quality Score")
    fig.savefig(output_dir / "distribution_pca_2d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    if reference_points:
        ax3d.scatter(
            [point["coords"]["source"][0] for point in reference_points],
            [point["coords"]["source"][1] for point in reference_points],
            [point["coords"]["source"][2] for point in reference_points],
            c=REFERENCE_COLOR,
            marker="x",
            s=45,
            alpha=0.45,
        )
    ax3d.scatter(
        [point["coords"]["evaluated"][0] for point in points],
        [point["coords"]["evaluated"][1] for point in points],
        [point["coords"]["evaluated"][2] for point in points],
        c=final_scores,
        cmap="RdYlGn",
        vmin=1.0,
        vmax=5.0,
        s=52,
        edgecolors="#FFFFFF",
        linewidths=0.4,
    )
    ax3d.set_title("3D Quality Score PCA Space")
    ax3d.set_xlabel("Quality Score PC1")
    ax3d.set_ylabel("Quality Score PC2")
    ax3d.set_zlabel("Quality Score PC3")
    ax3d.view_init(elev=24, azim=38)
    fig.savefig(output_dir / "distribution_pca_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_dashboard_html(
    points: list[dict[str, Any]],
    reference_points: list[dict[str, Any]],
    summary: dict[str, Any],
    bounds: dict[str, float],
    output_path: Path,
) -> None:
    payload = {
        "stageOrder": STAGES,
        "stageLabels": STAGE_LABELS,
        "stageColors": STAGE_COLORS,
        "referenceColor": REFERENCE_COLOR,
        "bounds": bounds,
        "summary": summary,
        "points": [
            {
                "id": point["sample_id"],
                "order": point["order"],
                "domain": point["domain"],
                "platform": point["platform"],
                "speechAct": point["speech_act"],
                "register": point["register"],
                "emotion": point["emotion"],
                "finalScore": round(point["final_score"], 4),
                "scores": {key: round(value, 4) for key, value in point["scores"].items()},
                "qualityVector": point.get("quality_vector", {}),
                "sourceText": point["source_text"][:220],
                "rewrittenText": point["rewritten_text"][:220],
                "coords": point["coords"],
            }
            for point in points
        ],
        "reference": [
            {
                "id": point["sample_id"],
                "order": point["order"],
                "domain": point["domain"],
                "sourceText": point["source_text"][:180],
                "coords": point["coords"]["source"],
            }
            for point in reference_points
        ],
    }

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Quality Distribution Dashboard</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e1ec;
      --accent: #155eef;
      --danger: #d92d20;
      --good: #039855;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, "Apple SD Gothic Neo", "Noto Sans KR", "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    .topbar {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
      align-items: end;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.15;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .kpi, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 28px rgba(16, 24, 40, 0.06);
    }}
    .kpi {{
      padding: 12px;
      min-height: 82px;
    }}
    .kpi span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .kpi strong {{
      display: block;
      margin-top: 6px;
      font-size: 22px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      margin-top: 16px;
    }}
    .panel {{
      padding: 14px;
    }}
    .viz-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .mode-group, .control-row {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      min-height: 36px;
      padding: 0 12px;
      border-radius: 8px;
      font-weight: 700;
      cursor: pointer;
    }}
    button.active, button.primary {{
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }}
    canvas {{
      width: 100%;
      aspect-ratio: 16 / 10;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .timeline {{
      display: grid;
      grid-template-columns: auto 1fr 86px 130px;
      gap: 12px;
      align-items: center;
      margin-top: 12px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 6px 9px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      font-size: 12px;
      background: #fff;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .side-title {{
      margin: 4px 0 8px;
      font-weight: 800;
      font-size: 15px;
    }}
    .metric-list {{
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #edf1f6;
      color: var(--muted);
      font-size: 13px;
    }}
    .metric-row strong {{
      color: var(--ink);
      font-size: 13px;
    }}
    .tooltip {{
      min-height: 180px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
      font-size: 13px;
      line-height: 1.45;
    }}
    .tooltip strong {{
      color: var(--ink);
    }}
    .muted {{
      color: var(--muted);
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .table th, .table td {{
      text-align: left;
      padding: 7px 4px;
      border-bottom: 1px solid #edf1f6;
      vertical-align: top;
    }}
    .table th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .bars {{
      display: grid;
      gap: 6px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 90px 1fr 28px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }}
    .bar-track {{
      height: 8px;
      border-radius: 99px;
      background: #eef2f7;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: #155eef;
    }}
    @media (max-width: 1100px) {{
      .topbar, .grid {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .timeline {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="topbar">
      <div>
        <h1>데이터 품질 분포 대시보드</h1>
        <p>생성 데이터셋이 얼마나 고르게 퍼져 있는지, 원본 데이터셋과 어느 정도 다른지, 그리고 source → metadata → generated → evaluated 단계에서 각 샘플 위치가 어떻게 움직이는지 보여줍니다. metadata quality_score 벡터를 PCA로 2D/3D 차원 축소해 표시합니다.</p>
      </div>
      <div class="kpis">
        <div class="kpi"><span>생성 샘플</span><strong id="sampleCount"></strong></div>
        <div class="kpi"><span>평균 Final Score</span><strong id="meanScore"></strong></div>
        <div class="kpi"><span>원본→평가 분포 차이</span><strong id="refGap"></strong></div>
        <div class="kpi"><span>공간 균일도</span><strong id="evenness"></strong></div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <div class="viz-head">
          <div>
            <div class="side-title">Quality Score PCA Space</div>
            <p class="muted">quality_score 항목이 늘어나도 PC1/PC2/PC3로 축소해서 보여줍니다.</p>
          </div>
          <div class="mode-group">
            <button id="mode2d" class="active">2D</button>
            <button id="mode3d">3D</button>
            <button id="playButton" class="primary">Play</button>
          </div>
        </div>
        <canvas id="chart" width="1100" height="690"></canvas>
        <div class="timeline">
          <input id="stageSlider" type="range" min="0" max="3.25" step="0.005" value="0">
          <div class="control-row">
            <span class="muted">속도</span>
            <input id="speedSlider" type="range" min="0.4" max="2.4" step="0.1" value="1">
          </div>
          <strong id="stageText">Source</strong>
          <span class="muted" id="visibleText"></span>
        </div>
        <div class="legend" id="legend"></div>
      </div>

      <aside class="panel">
        <div class="side-title">분포 진단</div>
        <div class="metric-list">
          <div class="metric-row"><span>PCA 설명 분산</span><strong id="varianceText"></strong></div>
          <div class="metric-row"><span>도메인 균일도</span><strong id="domainEvenness"></strong></div>
          <div class="metric-row"><span>플랫폼 균일도</span><strong id="platformEvenness"></strong></div>
          <div class="metric-row"><span>생성 분포 spread</span><strong id="spreadText"></strong></div>
          <div class="metric-row"><span>낮은 품질 샘플</span><strong id="lowQuality"></strong></div>
        </div>

        <div class="side-title">샘플 상세</div>
        <div class="tooltip" id="tooltip">
          <strong>포인트에 마우스를 올려보세요.</strong>
          <div class="muted">샘플 ID, 점수, 도메인, 원문/생성문을 확인할 수 있습니다.</div>
        </div>

        <div class="side-title">도메인 분포</div>
        <div class="bars" id="domainBars"></div>

        <div class="side-title">개선 우선 후보</div>
        <table class="table">
          <thead><tr><th>ID</th><th>Score</th><th>Domain</th><th>Output</th></tr></thead>
          <tbody id="lowestTable"></tbody>
        </table>
      </aside>
    </section>
  </main>

  <script>
    const payload = {json.dumps(payload, ensure_ascii=False)};
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("stageSlider");
    const speedSlider = document.getElementById("speedSlider");
    const playButton = document.getElementById("playButton");
    const mode2d = document.getElementById("mode2d");
    const mode3d = document.getElementById("mode3d");
    const stageText = document.getElementById("stageText");
    const visibleText = document.getElementById("visibleText");
    const tooltip = document.getElementById("tooltip");
    const stages = payload.stageOrder;
    let mode = "2d";
    let playing = false;
    let rafId = null;
    let lastTimestamp = 0;
    let rotation = 0.62;
    let hovered = null;
    let mouse = null;

    const summary = payload.summary;
    document.getElementById("sampleCount").textContent = summary.sample_count;
    document.getElementById("meanScore").textContent = summary.mean_final_score.toFixed(3);
    document.getElementById("refGap").textContent = summary.centroid_shift.reference_to_evaluated.toFixed(3);
    document.getElementById("evenness").textContent = Math.round(summary.spatial_evenness * 100) + "%";
    document.getElementById("varianceText").textContent = summary.explained_variance_ratio.map(v => Math.round(v * 100) + "%").join(" / ");
    document.getElementById("domainEvenness").textContent = Math.round(summary.domain_evenness * 100) + "%";
    document.getElementById("platformEvenness").textContent = Math.round(summary.platform_evenness * 100) + "%";
    document.getElementById("spreadText").textContent = summary.generated_spread.toFixed(3);
    document.getElementById("lowQuality").textContent = `${{summary.low_quality_count}} / ${{summary.sample_count}}`;

    document.getElementById("legend").innerHTML = stages.map(stage => (
      `<span class="chip"><span class="swatch" style="background:${{payload.stageColors[stage]}}"></span>${{payload.stageLabels[stage]}}</span>`
    )).join("") + `<span class="chip"><span class="swatch" style="background:${{payload.referenceColor}}"></span>Reference</span><span class="chip">Fill color = avg quality score</span>`;

    const domainMax = Math.max(...Object.values(summary.domain_distribution));
    document.getElementById("domainBars").innerHTML = Object.entries(summary.domain_distribution)
      .sort((a, b) => b[1] - a[1])
      .map(([name, count]) => `<div class="bar-row"><span>${{name}}</span><div class="bar-track"><div class="bar-fill" style="width:${{(count / domainMax) * 100}}%"></div></div><strong>${{count}}</strong></div>`)
      .join("");

    document.getElementById("lowestTable").innerHTML = summary.lowest_samples.map(row => (
      `<tr><td>${{row.id}}</td><td>${{row.score.toFixed(3)}}</td><td>${{row.domain}}</td><td>${{row.text}}</td></tr>`
    )).join("");

    function clampValue(value, low, high) {{
      return Math.max(low, Math.min(high, value));
    }}

    function lerp(a, b, t) {{
      return a + (b - a) * t;
    }}

    function scoreColor(score) {{
      const t = clampValue((score - 1) / 4, 0, 1);
      const r = Math.round(217 + (3 - 217) * t);
      const g = Math.round(45 + (152 - 45) * t);
      const b = Math.round(32 + (85 - 32) * t);
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function localStageValue(point, value) {{
      const delay = point.order * 0.018;
      return clampValue(value - delay, -0.2, 3);
    }}

    function interpolate(point, value) {{
      const local = localStageValue(point, value);
      if (local < 0) {{
        return null;
      }}
      const leftIndex = Math.floor(local);
      const rightIndex = Math.min(stages.length - 1, leftIndex + 1);
      const mix = local - leftIndex;
      const leftStage = stages[leftIndex];
      const rightStage = stages[rightIndex];
      const left = point.coords[leftStage];
      const right = point.coords[rightStage];
      return {{
        x: lerp(left[0], right[0], mix),
        y: lerp(left[1], right[1], mix),
        z: lerp(left[2], right[2], mix),
        stage: mix < 0.5 ? leftStage : rightStage,
        opacity: clampValue(local / 0.25, 0.1, 1),
      }};
    }}

    function norm(value, min, max) {{
      return (value - min) / Math.max(max - min, 1e-6);
    }}

    function project2d(coord) {{
      const padX = 72;
      const padY = 58;
      return {{
        x: padX + norm(coord.x, payload.bounds.minX, payload.bounds.maxX) * (canvas.width - padX * 2),
        y: canvas.height - padY - norm(coord.y, payload.bounds.minY, payload.bounds.maxY) * (canvas.height - padY * 2),
        scale: 1,
      }};
    }}

    function project3d(coord) {{
      const cx = (payload.bounds.minX + payload.bounds.maxX) / 2;
      const cy = (payload.bounds.minY + payload.bounds.maxY) / 2;
      const cz = (payload.bounds.minZ + payload.bounds.maxZ) / 2;
      const x = coord.x - cx;
      const y = coord.y - cy;
      const z = coord.z - cz;
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      const rx = x * cos - z * sin;
      const rz = x * sin + z * cos;
      const ry = y * Math.cos(0.58) - rz * Math.sin(0.58);
      const depth = y * Math.sin(0.58) + rz * Math.cos(0.58);
      const span = Math.max(payload.bounds.maxX - payload.bounds.minX, payload.bounds.maxY - payload.bounds.minY, payload.bounds.maxZ - payload.bounds.minZ);
      const perspective = 1 / (1 + depth / (span * 2.8));
      const scale = Math.min(canvas.width, canvas.height) * 0.58 / Math.max(span, 1e-6);
      return {{
        x: canvas.width / 2 + rx * scale * perspective,
        y: canvas.height / 2 - ry * scale * perspective,
        scale: perspective,
        depth,
      }};
    }}

    function project(coord) {{
      return mode === "3d" ? project3d(coord) : project2d(coord);
    }}

    function drawGrid() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fbfdff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#e5ebf3";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 6; i += 1) {{
        const x = 72 + ((canvas.width - 144) / 6) * i;
        const y = 58 + ((canvas.height - 116) / 6) * i;
        ctx.beginPath();
        ctx.moveTo(x, 58);
        ctx.lineTo(x, canvas.height - 58);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(72, y);
        ctx.lineTo(canvas.width - 72, y);
        ctx.stroke();
      }}
      ctx.fillStyle = "#667085";
      ctx.font = "13px Inter, sans-serif";
      ctx.fillText(mode === "3d" ? "Quality Score PC1 / PC2 / PC3" : "Quality Score PC1", 24, 28);
      ctx.fillText(mode === "3d" ? "drag = rotate" : "Quality Score PC2", canvas.width - 148, canvas.height - 24);
    }}

    function drawReference() {{
      ctx.save();
      ctx.globalAlpha = 0.55;
      ctx.strokeStyle = payload.referenceColor;
      ctx.lineWidth = 1.3;
      payload.reference.forEach(point => {{
        const projected = project({{x: point.coords[0], y: point.coords[1], z: point.coords[2]}});
        ctx.beginPath();
        ctx.moveTo(projected.x - 5, projected.y - 5);
        ctx.lineTo(projected.x + 5, projected.y + 5);
        ctx.moveTo(projected.x + 5, projected.y - 5);
        ctx.lineTo(projected.x - 5, projected.y + 5);
        ctx.stroke();
      }});
      ctx.restore();
    }}

    function drawPaths(value) {{
      ctx.save();
      ctx.strokeStyle = "rgba(71, 84, 103, 0.16)";
      ctx.lineWidth = 1;
      payload.points.forEach(point => {{
        if (localStageValue(point, value) < 0.2) return;
        ctx.beginPath();
        stages.forEach((stage, index) => {{
          const coord = point.coords[stage];
          const p = project({{x: coord[0], y: coord[1], z: coord[2]}});
          if (index === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }});
        ctx.stroke();
      }});
      ctx.restore();
    }}

    function drawCentroids(value) {{
      const stageIndex = Math.min(3, Math.round(value));
      ctx.save();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(17, 24, 39, 0.58)";
      ctx.beginPath();
      stages.forEach((stage, index) => {{
        const coord = summary.stage_centroids[stage];
        const p = project({{x: coord[0], y: coord[1], z: coord[2]}});
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }});
      ctx.stroke();
      stages.forEach((stage, index) => {{
        const coord = summary.stage_centroids[stage];
        const p = project({{x: coord[0], y: coord[1], z: coord[2]}});
        ctx.globalAlpha = index <= stageIndex ? 1 : 0.35;
        ctx.fillStyle = payload.stageColors[stage];
        ctx.strokeStyle = "#111827";
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }});
      ctx.restore();
    }}

    function drawPoints(value) {{
      hovered = null;
      const drawable = [];
      payload.points.forEach(point => {{
        const coord = interpolate(point, value);
        if (!coord) return;
        const projected = project(coord);
        drawable.push({{point, coord, projected}});
      }});
      drawable.sort((a, b) => (a.projected.depth || 0) - (b.projected.depth || 0));
      drawable.forEach(item => {{
        const radius = (6 + (item.point.finalScore / 5) * 4) * item.projected.scale;
        ctx.save();
        ctx.globalAlpha = item.coord.opacity;
        ctx.fillStyle = scoreColor(item.point.finalScore);
        ctx.strokeStyle = payload.stageColors[item.coord.stage];
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(item.projected.x, item.projected.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
        if (mouse) {{
          const hit = Math.hypot(mouse.x - item.projected.x, mouse.y - item.projected.y);
          if (hit <= radius + 4 && (!hovered || hit < hovered.hit)) {{
            hovered = {{...item, hit}};
          }}
        }}
      }});
      visibleText.textContent = `${{drawable.length}} / ${{payload.points.length}} visible`;
    }}

    function renderTooltip() {{
      if (!hovered) {{
        tooltip.innerHTML = '<strong>포인트에 마우스를 올려보세요.</strong><div class="muted">샘플 ID, 점수, 도메인, 원문/생성문을 확인할 수 있습니다.</div>';
        return;
      }}
      const p = hovered.point;
      const qualityRows = Object.entries(p.scores)
        .filter(([key]) => key !== "final")
        .map(([key, value]) => `${{key.replaceAll("_", " ")}} ${{Number(value).toFixed(0)}}`)
        .join(" · ");
      tooltip.innerHTML = `
        <strong>${{p.id}}</strong>
        <div class="muted">${{p.domain}} · ${{p.platform}} · ${{p.speechAct}} · ${{p.register}}</div>
        <div style="margin-top:8px;">Avg <strong>${{p.finalScore.toFixed(2)}}</strong> · ${{qualityRows}}</div>
        <div style="margin-top:10px;"><strong>Source</strong><br><span class="muted">${{p.sourceText}}</span></div>
        <div style="margin-top:10px;"><strong>Korean Output</strong><br><span class="muted">${{p.rewrittenText}}</span></div>
      `;
    }}

    function render() {{
      const value = parseFloat(slider.value);
      if (mode === "3d" && playing) rotation += 0.0035;
      drawGrid();
      drawReference();
      drawPaths(value);
      drawCentroids(value);
      drawPoints(value);
      const stageIndex = Math.min(3, Math.max(0, Math.round(value)));
      stageText.textContent = payload.stageLabels[stages[stageIndex]];
      renderTooltip();
    }}

    function animate(timestamp) {{
      if (!playing) return;
      if (!lastTimestamp) lastTimestamp = timestamp;
      const delta = timestamp - lastTimestamp;
      lastTimestamp = timestamp;
      let next = parseFloat(slider.value) + delta * 0.00034 * parseFloat(speedSlider.value);
      if (next > parseFloat(slider.max)) next = 0;
      slider.value = next.toFixed(3);
      render();
      rafId = requestAnimationFrame(animate);
    }}

    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
      lastTimestamp = 0;
      if (playing) rafId = requestAnimationFrame(animate);
      else if (rafId) cancelAnimationFrame(rafId);
      render();
    }});
    slider.addEventListener("input", render);
    speedSlider.addEventListener("input", render);
    mode2d.addEventListener("click", () => {{
      mode = "2d";
      mode2d.classList.add("active");
      mode3d.classList.remove("active");
      render();
    }});
    mode3d.addEventListener("click", () => {{
      mode = "3d";
      mode3d.classList.add("active");
      mode2d.classList.remove("active");
      render();
    }});

    let dragging = false;
    let lastX = 0;
    canvas.addEventListener("mousemove", event => {{
      const rect = canvas.getBoundingClientRect();
      mouse = {{
        x: (event.clientX - rect.left) * (canvas.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.height / rect.height),
      }};
      if (dragging && mode === "3d") {{
        rotation += (mouse.x - lastX) * 0.006;
        lastX = mouse.x;
      }}
      render();
    }});
    canvas.addEventListener("mousedown", event => {{
      dragging = true;
      const rect = canvas.getBoundingClientRect();
      lastX = (event.clientX - rect.left) * (canvas.width / rect.width);
    }});
    window.addEventListener("mouseup", () => {{ dragging = false; }});
    canvas.addEventListener("mouseleave", () => {{
      mouse = null;
      dragging = false;
      render();
    }});

    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def make_dashboard_payload(
    generated_path: str | Path,
    evaluated_path: str | Path,
    reference_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    generated_rows = load_jsonl(str(generated_path))
    evaluated_rows = load_jsonl(str(evaluated_path)) if Path(evaluated_path).exists() else []
    reference_rows = load_jsonl(str(reference_path)) if reference_path and Path(reference_path).exists() else []
    if not generated_rows:
        raise ValueError("Generated dataset is empty.")

    points, reference_points, feature_names = build_points(generated_rows, evaluated_rows, reference_rows)
    matrix, meta = vectorize(points, reference_points, feature_names)
    coords, explained = fit_pca(matrix, dims=3)
    attach_coords(points, reference_points, meta, coords)
    summary = compute_summary(points, reference_points, explained)
    bounds = compute_bounds(points, reference_points)
    return points, reference_points, summary, bounds


def main() -> None:
    args = parse_args()
    points, reference_points, summary, bounds = make_dashboard_payload(
        args.generated,
        args.evaluated,
        args.reference,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_static_plots(points, reference_points, summary, output_dir)
    build_dashboard_html(points, reference_points, summary, bounds, output_dir / "distribution_dashboard.html")
    (output_dir / "distribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved dashboard HTML to: {output_dir / 'distribution_dashboard.html'}")
    print(f"Saved 2D PCA plot to: {output_dir / 'distribution_pca_2d.png'}")
    print(f"Saved 3D PCA plot to: {output_dir / 'distribution_pca_3d.png'}")
    print(f"Saved summary JSON to: {output_dir / 'distribution_summary.json'}")


if __name__ == "__main__":
    main()
