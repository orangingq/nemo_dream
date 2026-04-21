import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from util import load_jsonl


STAGE_FILES = [
    ("input", "stage0.jsonl", "Raw Input", "Text-only seed samples"),
    ("stage1", "stage1.jsonl", "Stage 1: Decomposition", "Intent, emotion, cultural references"),
    ("stage2", "stage2.jsonl", "Stage 2: Cultural Mapping", "Mapped references and localization candidates"),
    ("stage3", "stage3.jsonl", "Stage 3: Generation", "Final Korean rewrite candidates"),
    ("stage4", "stage4.jsonl", "Stage 4: Normalize", "Stage3 rows normalized into curator Sample records"),
    ("stage5", "stage5.jsonl", "Stage 5: Schema", "Pydantic schema validation"),
    ("stage6", "stage6.jsonl", "Stage 6: Rules", "Rule-based validation and rejection"),
    ("stage7", "stage7.jsonl", "Stage 7: Safety", "PII and content safety checks"),
    ("stage8", "stage8.jsonl", "Stage 8: Semantic", "Semantic, property, naturalness, cultural, register axes"),
    ("stage9", "stage9.jsonl", "Stage 9: Quality", "Aggregate quality thresholding"),
    ("stage10", "stage10.jsonl", "Stage 10: Dedup", "Near-duplicate filtering"),
    ("stage11", "stage11.jsonl", "Stage 11: Reward", "Reward-model axes for accepted rows"),
    ("stage12", "stage12.jsonl", "Stage 12: Report", "Accepted/rejected totals and distribution summary"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a full curated-pipeline dashboard.")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--output", default="artifacts/dashboard.html")
    parser.add_argument("--serve", action="store_true", help="Serve the dashboard and live-refresh payload from JSONL files.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


def safe_score(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_message(row: dict[str, Any], role: str) -> str:
    for item in row.get("messages", []):
        if isinstance(item, dict) and item.get("role") == role:
            return str(item.get("content", ""))
    return ""


def score_vector(score_obj: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    def add(prefix: str, item: Any) -> None:
        if isinstance(item, bool):
            values[prefix] = 5.0 if item else 1.0
            return
        if isinstance(item, dict):
            if "score" in item:
                values[prefix] = safe_score(item.get("score"))
                return
            for child_key, child_value in sorted(item.items()):
                add(f"{prefix}_{child_key}", child_value)
            return
        if isinstance(item, (int, float)):
            values[prefix] = safe_score(item)

    for key, item in sorted(score_obj.items()):
        add(key, item)
    return values


def score_source(row: dict[str, Any], score_field: str) -> dict[str, Any]:
    meta = row.get("metadata") or {}
    if score_field == "quality_score":
        return meta.get("quality_score", {})
    return row.get(score_field, {})


def fit_pca(vectors: list[list[float]], dims: int = 3) -> tuple[np.ndarray, list[float]]:
    matrix = np.array(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros((0, dims)), [0.0] * dims
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dims]
    coords = centered @ components.T
    total = max(float(np.sum(singular_values**2)), 1e-12)
    explained = [float(value) for value in ((singular_values[:dims] ** 2) / total)]
    if coords.shape[1] < dims:
        coords = np.hstack([coords, np.zeros((coords.shape[0], dims - coords.shape[1]))])
        explained.extend([0.0] * (dims - len(explained)))
    return coords, explained[:dims]


def fit_tsne(vectors: list[list[float]], dims: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.array(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros((0, dims)), {"perplexity": 0.0, "fallback": "empty"}
    if matrix.shape[0] < 3:
        coords, _ = fit_pca(vectors, dims=dims)
        return coords[:, :dims], {"perplexity": 0.0, "fallback": "pca_small_sample"}

    from sklearn.manifold import TSNE

    perplexity = min(30.0, max(1.0, (matrix.shape[0] - 1) / 3))
    init = "pca" if matrix.shape[1] > 1 else "random"
    coords = TSNE(
        n_components=dims,
        perplexity=perplexity,
        init=init,
        learning_rate="auto",
        random_state=42,
        max_iter=350,
    ).fit_transform(matrix)
    return coords, {"perplexity": round(float(perplexity), 2), "fallback": None}


def build_pca_points(rows: list[dict[str, Any]], score_field: str, text_field: str) -> dict[str, Any]:
    vectors: list[dict[str, float]] = []
    for row in rows:
        vectors.append(score_vector(score_source(row, score_field)))
    keys = sorted({key for vector in vectors for key in vector})
    raw_vectors = [[vector.get(key, 1.0) for key in keys] for vector in vectors]
    coords, explained = fit_pca(raw_vectors, dims=3)
    tsne_coords, tsne_meta = fit_tsne(raw_vectors, dims=2)
    points = []
    for row, vector, coord, tsne_coord in zip(rows, vectors, coords, tsne_coords, strict=True):
        meta = row.get("metadata") or {}
        source = get_message(row, "user") or row.get("source_text") or row.get("en_text") or row.get("id", "")
        output = get_message(row, "assistant") or row.get(text_field) or row.get("ko_text") or row.get("final_text", "")
        score_values = list(vector.values())
        points.append(
            {
                "id": meta.get("source_id") or row.get("source_id") or row.get("id"),
                "domain": meta.get("domain", "unknown"),
                "platform": meta.get("target_platform", "unknown"),
                "source": source,
                "output": output,
                "scores": vector,
                "avg": round(sum(score_values) / max(len(score_values), 1), 4),
                "coords": [round(float(coord[0]), 6), round(float(coord[1]), 6), round(float(coord[2]), 6)],
                "tsne_coords": [round(float(tsne_coord[0]), 6), round(float(tsne_coord[1]), 6)],
            }
        )
    return {
        "keys": keys,
        "explained": [round(value, 4) for value in explained],
        "tsne": tsne_meta,
        "points": points,
    }


def build_postprocess_pca(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return build_pca_points(rows, "quality", "ko_text")


def normalize(values: list[float], fallback_step_count: int) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if abs(max_value - min_value) < 1e-9:
        return [(idx + 1) / (fallback_step_count + 1) for idx in range(len(values))]
    return [0.12 + ((value - min_value) / (max_value - min_value)) * 0.76 for value in values]


def row_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("source_id") or row.get("source_id") or row.get("id"))


def row_status(row: dict[str, Any]) -> str:
    if not row:
        return "missing"
    if row.get("valid") is False:
        reasons = row.get("reject_reasons") or []
        if reasons:
            reason = reasons[0]
            return f"rejected: {reason.get('rule') or reason.get('stage')}"
        return "rejected"
    filter_obj = row.get("filter") or {}
    if filter_obj.get("status"):
        return str(filter_obj.get("status"))
    return "accepted" if row.get("valid") is True else "unknown"


def build_path_point(
    stage_idx: int,
    stage_count: int,
    sample_id: str,
    lane_y: float,
    label: str,
    output: str,
    axis: str,
    score: float | None = None,
    local_x: float = 0.5,
    local_y: float | None = None,
) -> dict[str, Any]:
    stage_x = 0.08 + (stage_idx / max(stage_count - 1, 1)) * 0.84
    x = stage_x + (local_x - 0.5) * 0.08
    y = local_y if local_y is not None else lane_y
    return {
        "id": sample_id,
        "x": round(max(0.03, min(0.97, x)), 6),
        "y": round(max(0.05, min(0.95, y)), 6),
        "label": label,
        "output": output,
        "axis": axis,
        "score": None if score is None else round(score, 3),
    }


def build_stage_path(
    rows_by_stage: dict[str, list[dict[str, Any]]],
    quality: dict[str, Any],
    postprocess_pca: dict[str, Any],
    llm: dict[str, Any],
) -> dict[str, Any]:
    stage_defs = [
        ("input", "Input", "stable sample lane"),
        ("stage1", "Stage 1", "decomposition lane"),
        ("stage2", "Stage 2", "mapping lane"),
        ("stage3", "Stage 3", "generation lane"),
        ("stage4", "Stage 4", "normalization lane"),
        ("stage5", "Stage 5", "schema validation lane"),
        ("stage6", "Stage 6", "rule validation lane"),
        ("stage7", "Stage 7", "safety validation lane"),
        ("stage8", "Stage 8", "semantic judge PCA local axis"),
        ("stage9", "Stage 9", "quality threshold lane"),
        ("stage10", "Stage 10", "filter/dedup PCA local axis"),
        ("stage11", "Stage 11", "reward PCA local axis"),
        ("stage12", "Stage 12", "report summary lane"),
    ]
    ids = [row_id(row) for row in rows_by_stage.get("stage8", [])] or [row_id(row) for row in rows_by_stage.get("input", [])]
    if not ids:
        ids = sorted({row_id(row) for rows in rows_by_stage.values() for row in rows})
    lanes = {sample_id: (idx + 1) / (len(ids) + 1) for idx, sample_id in enumerate(ids)}

    quality_points = {str(point["id"]): point for point in quality.get("points", [])}
    postprocess_points = {str(point["id"]): point for point in postprocess_pca.get("points", [])}
    llm_points = {str(point["id"]): point for point in llm.get("points", [])}
    quality_x = normalize([point["coords"][0] for point in quality_points.values()], len(quality_points))
    quality_y = normalize([point["coords"][1] for point in quality_points.values()], len(quality_points))
    postprocess_x = normalize([point["coords"][0] for point in postprocess_points.values()], len(postprocess_points))
    postprocess_y = normalize([point["coords"][1] for point in postprocess_points.values()], len(postprocess_points))
    llm_x = normalize([point["coords"][0] for point in llm_points.values()], len(llm_points))
    llm_y = normalize([point["coords"][1] for point in llm_points.values()], len(llm_points))
    quality_local = {
        sample_id: {"x": quality_x[idx], "y": quality_y[idx]}
        for idx, sample_id in enumerate(quality_points)
    }
    postprocess_local = {
        sample_id: {"x": postprocess_x[idx], "y": postprocess_y[idx]}
        for idx, sample_id in enumerate(postprocess_points)
    }
    llm_local = {sample_id: {"x": llm_x[idx], "y": llm_y[idx]} for idx, sample_id in enumerate(llm_points)}

    row_maps = {
        stage_id: {row_id(row): row for row in rows}
        for stage_id, rows in rows_by_stage.items()
    }
    frames = []
    for stage_idx, (stage_id, label, axis) in enumerate(stage_defs):
        points = []
        for sample_id in ids:
            lane_y = lanes.get(sample_id, 0.5)
            row = row_maps.get(stage_id, {}).get(sample_id, {})
            fallback_row = row_maps.get("stage8", {}).get(sample_id, {})
            effective_row = row or fallback_row
            score = None
            local_x = 0.5
            local_y: float | None = None
            output = ""
            point_label = sample_id
            if stage_id == "input":
                output = str(effective_row.get("text") or effective_row.get("en_text", ""))
            elif stage_id == "stage1":
                decomposed = row.get("decomposed") or {}
                meta = effective_row.get("metadata") or {}
                emotion = decomposed.get("emotion") or {}
                speech_act = decomposed.get("speech_act") or meta.get("speech_act", "unknown")
                register = decomposed.get("register") or meta.get("register", "unknown")
                emotion_type = emotion.get("type") or meta.get("emotion_type", "unknown")
                cultural_refs = decomposed.get("cultural_refs") or meta.get("cultural_refs", [])
                point_label = f"{sample_id}: {speech_act}"
                output = f"{register} · {emotion_type} · refs {len(cultural_refs)}"
            elif stage_id == "stage2":
                mapped = row.get("mapped_refs") or (effective_row.get("metadata") or {}).get("cultural_refs", [])
                point_label = f"{sample_id}: {len(mapped)} refs"
                output = ", ".join(str(item.get("ko") or item.get("ko_term") or item.get("term")) for item in mapped[:4])
            elif stage_id == "stage3":
                generation = row.get("generation") or {}
                output = str(generation.get("rewritten_text") or effective_row.get("ko_text", ""))
            elif stage_id == "stage4":
                output = str(row.get("ko_text") or "")
                point_label = f"{sample_id}: normalized"
            elif stage_id == "stage5":
                output = row_status(row)
                point_label = f"{sample_id}: schema"
            elif stage_id == "stage6":
                output = row_status(row)
                point_label = f"{sample_id}: rules"
            elif stage_id == "stage7":
                output = row_status(row)
                point_label = f"{sample_id}: safety"
            elif stage_id == "stage8":
                pca_point = quality_points.get(sample_id, {})
                local = quality_local.get(sample_id, {})
                local_x = safe_score(local.get("x"), 0.5)
                local_y = safe_score(local.get("y"), lane_y)
                score = pca_point.get("avg")
                output = f"{row_status(row)} · {pca_point.get('output', '')}"
            elif stage_id == "stage9":
                output = f"{row_status(row)} · aggregate {safe_score((row.get('quality') or {}).get('aggregate'), 0):.2f}"
                point_label = f"{sample_id}: quality"
            elif stage_id == "stage10":
                pca_point = postprocess_points.get(sample_id, {})
                local = postprocess_local.get(sample_id, {})
                local_x = safe_score(local.get("x"), 0.5)
                local_y = safe_score(local.get("y"), lane_y)
                output = f"{row_status(row)} · {row.get('ko_text', '')}"
                score = pca_point.get("avg")
                point_label = f"{sample_id}: {row_status(row)}"
            elif stage_id == "stage11":
                pca_point = llm_points.get(sample_id, {})
                local = llm_local.get(sample_id, {})
                local_x = safe_score(local.get("x"), 0.5)
                local_y = safe_score(local.get("y"), lane_y)
                score = pca_point.get("avg")
                output = str(pca_point.get("output", ""))
            elif stage_id == "stage12":
                report = rows_by_stage.get("stage12", [{}])[0] if rows_by_stage.get("stage12") else {}
                totals = report.get("totals") or {}
                output = f"accepted {totals.get('accepted', 0)} / rejected {totals.get('rejected', 0)}"
                point_label = f"{sample_id}: report"
            points.append(
                build_path_point(stage_idx, len(stage_defs), sample_id, lane_y, point_label, output, axis, score, local_x, local_y)
            )
        frames.append({"id": stage_id, "label": label, "axis": axis, "points": points})
    return {"stages": [{"id": stage_id, "label": label, "axis": axis} for stage_id, label, axis in stage_defs], "frames": frames}


def stage_summary(stage_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if stage_id == "input":
        return {"count": len(rows), "metrics": {"text rows": len(rows)}}
    if stage_id == "stage1":
        refs = sum(len((row.get("decomposed") or {}).get("cultural_refs", [])) for row in rows)
        return {"count": len(rows), "metrics": {"cultural refs": refs, "avg refs": round(refs / max(len(rows), 1), 2)}}
    if stage_id == "stage2":
        mapped = sum(len(row.get("mapped_refs", [])) for row in rows)
        retrieved = sum(1 for row in rows for ref in row.get("mapped_refs", []) if ref.get("retrieved"))
        return {"count": len(rows), "metrics": {"mapped refs": mapped, "retrieved": retrieved}}
    if stage_id == "stage3":
        return {"count": len(rows), "metrics": {"generated rewrites": len(rows)}}
    if stage_id == "stage4":
        return {"count": len(rows), "metrics": {"normalized": len(rows)}}
    if stage_id == "stage5":
        rejected = sum(1 for row in rows if row.get("valid") is False)
        return {"count": len(rows), "metrics": {"schema checked": len(rows), "rejected": rejected}}
    if stage_id == "stage6":
        rejected = sum(1 for row in rows if row.get("valid") is False)
        return {"count": len(rows), "metrics": {"rule checked": len(rows), "rejected": rejected}}
    if stage_id == "stage7":
        rejected = sum(1 for row in rows if row.get("valid") is False)
        return {"count": len(rows), "metrics": {"safety checked": len(rows), "rejected": rejected}}
    if stage_id == "stage8":
        rejected = sum(1 for row in rows if row.get("valid") is False)
        return {"count": len(rows), "metrics": {"semantic judged": len(rows), "rejected": rejected}}
    if stage_id == "stage9":
        accepted = sum(1 for row in rows if row.get("valid") is True)
        rejected = sum(1 for row in rows if row.get("valid") is False)
        return {"count": len(rows), "metrics": {"quality passed": accepted, "rejected": rejected}}
    if stage_id == "stage10":
        accepted = sum(1 for row in rows if row.get("valid") is True)
        rejected = sum(1 for row in rows if row.get("valid") is False)
        return {"count": len(rows), "metrics": {"accepted": accepted, "rejected": rejected}}
    if stage_id == "stage11":
        aggregates = [safe_score((row.get("quality") or {}).get("aggregate")) for row in rows]
        return {"count": len(rows), "metrics": {"reward scored": len(rows), "avg aggregate": round(sum(aggregates) / max(len(aggregates), 1), 3)}}
    if stage_id == "stage12":
        report = rows[0] if rows else {}
        totals = report.get("totals") or {}
        return {"count": len(rows), "metrics": {"accepted": totals.get("accepted", 0), "rejected": totals.get("rejected", 0)}}
    return {"count": len(rows), "metrics": {"rows": len(rows)}}


def load_pipeline(curated_dir: Path) -> dict[str, Any]:
    stages = []
    rows_by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage_id, filename, label, description in STAGE_FILES:
        path = curated_dir / filename
        rows = load_jsonl(str(path)) if path.exists() else []
        rows_by_stage[stage_id] = rows
        stages.append(
            {
                "id": stage_id,
                "label": label,
                "description": description,
                "file": str(path),
                **stage_summary(stage_id, rows),
            }
        )
    quality = build_pca_points(rows_by_stage["stage8"], "quality", "ko_text")
    llm = build_pca_points(rows_by_stage["stage11"], "quality", "ko_text")
    postprocess = build_postprocess_pca(rows_by_stage["stage10"])
    stage_path = build_stage_path(rows_by_stage, quality, postprocess, llm)
    return {"stages": stages, "qualityPca": quality, "postprocessPca": postprocess, "llmPca": llm, "stagePath": stage_path}


def pipeline_signature(curated_dir: Path) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for _, filename, _, _ in STAGE_FILES:
        path = curated_dir / filename
        if not path.exists():
            signature.append((filename, 0, 0))
            continue
        stat = path.stat()
        signature.append((filename, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def dashboard_html(payload: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Curated Pipeline Dashboard</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; background: #f6f8fb; color: #172033; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 29px; }}
    p {{ margin: 0; color: #667085; line-height: 1.45; }}
    .layout {{ display: grid; grid-template-columns: 380px 1fr; gap: 16px; margin-top: 18px; }}
    .panel, .card {{ background: #fff; border: 1px solid #d9e1ec; border-radius: 8px; box-shadow: 0 10px 28px rgba(16, 24, 40, 0.06); }}
    .panel {{ padding: 14px; }}
    .timeline {{ display: grid; gap: 10px; }}
    .card {{ padding: 12px; }}
    .card strong {{ display: block; font-size: 14px; margin-bottom: 4px; }}
    .metric {{ display: inline-flex; margin: 8px 8px 0 0; padding: 5px 8px; border-radius: 8px; background: #eef4ff; color: #155eef; font-size: 12px; }}
    .tabs {{ display: flex; gap: 8px; margin-bottom: 12px; }}
    .toolbar {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
    .tabs {{ margin-bottom: 0; }}
    .projection-tabs {{ display: flex; gap: 8px; }}
    button {{ min-height: 36px; border: 1px solid #d9e1ec; border-radius: 8px; background: #fff; padding: 0 12px; font-weight: 800; cursor: pointer; }}
    button.active {{ background: #155eef; border-color: #155eef; color: #fff; }}
    .path-note {{ min-height: 20px; margin: -4px 0 10px; font-size: 12px; color: #667085; }}
    canvas {{ width: 100%; aspect-ratio: 16 / 10; border: 1px solid #d9e1ec; border-radius: 8px; background: #fbfdff; display: block; }}
    .side {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }}
    .kpi {{ border: 1px solid #d9e1ec; border-radius: 8px; padding: 10px; }}
    .kpi span {{ color: #667085; font-size: 12px; }}
    .kpi strong {{ display: block; margin-top: 5px; }}
    .detail {{ min-height: 120px; margin-top: 12px; padding: 12px; border: 1px solid #d9e1ec; border-radius: 8px; background: #fbfdff; font-size: 13px; line-height: 1.45; }}
    .muted {{ color: #667085; }}
    @media (max-width: 1050px) {{ .layout {{ grid-template-columns: 1fr; }} .side {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Full Curator Pipeline Dashboard</h1>
  <p>Stage1~7은 pipeline coverage로 보여주고, Stage8 semantic, Stage10 dedup, Stage11 reward는 PCA와 t-SNE projection으로 보여줍니다.</p>
  <section class="layout">
    <aside class="panel">
      <div class="timeline" id="timeline"></div>
    </aside>
    <section class="panel">
      <div class="toolbar">
        <div class="tabs">
          <button id="qualityBtn" class="active">Stage8 Semantic</button>
          <button id="postprocessBtn">Stage10 Dedup</button>
          <button id="llmBtn">Stage11 Reward</button>
          <button id="pathBtn">Live Stage Path</button>
        </div>
        <div class="projection-tabs">
          <button id="pcaBtn" class="active">PCA</button>
          <button id="tsneBtn">t-SNE</button>
        </div>
      </div>
      <div class="path-note" id="pathNote"></div>
      <canvas id="chart" width="1000" height="620"></canvas>
      <div class="side">
        <div class="kpi"><span>Score keys</span><strong id="keysText"></strong></div>
        <div class="kpi"><span>Projection</span><strong id="explainedText"></strong></div>
        <div class="kpi"><span>Samples</span><strong id="countText"></strong></div>
      </div>
      <div class="detail" id="detail"><strong>Hover a point</strong><div class="muted">샘플 점수와 출력이 여기에 표시됩니다.</div></div>
    </section>
  </section>
</main>
<script>
let payload = {json.dumps(payload, ensure_ascii=False)};
const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
let mode = "qualityPca";
let projection = "pca";
let mouse = null;
let hovered = null;
let pathStartedAt = performance.now();
let animationHandle = null;
let stageClickTargets = [];

function setMode(next) {{
  mode = next;
  document.getElementById("qualityBtn").classList.toggle("active", mode === "qualityPca");
  document.getElementById("postprocessBtn").classList.toggle("active", mode === "postprocessPca");
  document.getElementById("llmBtn").classList.toggle("active", mode === "llmPca");
  document.getElementById("pathBtn").classList.toggle("active", mode === "stagePath");
  document.getElementById("pcaBtn").disabled = mode === "stagePath";
  document.getElementById("tsneBtn").disabled = mode === "stagePath";
  document.getElementById("pathNote").textContent = mode === "stagePath"
    ? "각 stage는 local coordinate입니다. Stage8/10/11 column을 클릭하면 해당 projection 화면으로 전환됩니다."
    : "";
  if (animationHandle && mode !== "stagePath") {{
    cancelAnimationFrame(animationHandle);
    animationHandle = null;
  }}
  if (mode === "stagePath") pathStartedAt = performance.now();
  render();
}}
function setProjection(next) {{
  projection = next;
  document.getElementById("pcaBtn").classList.toggle("active", projection === "pca");
  document.getElementById("tsneBtn").classList.toggle("active", projection === "tsne");
  render();
}}
function projectionCoords(point) {{
  return projection === "tsne" ? point.tsne_coords : point.coords;
}}
function bounds(points) {{
  const xs = points.map(p => projectionCoords(p)[0]);
  const ys = points.map(p => projectionCoords(p)[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  return {{minX, maxX, minY, maxY, padX: Math.max((maxX - minX) * 0.15, 0.2), padY: Math.max((maxY - minY) * 0.15, 0.2)}};
}}
function sx(x, b) {{ return 60 + ((x - (b.minX - b.padX)) / Math.max((b.maxX + b.padX) - (b.minX - b.padX), 1e-6)) * (canvas.width - 120); }}
function sy(y, b) {{ return canvas.height - 56 - ((y - (b.minY - b.padY)) / Math.max((b.maxY + b.padY) - (b.minY - b.padY), 1e-6)) * (canvas.height - 112); }}
function color(avg) {{
  const t = Math.max(0, Math.min(1, (avg - 3.5) / 1.5));
  const r = Math.round(217 + (3 - 217) * t);
  const g = Math.round(45 + (152 - 45) * t);
  const b = Math.round(32 + (85 - 32) * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function renderTimeline() {{
  document.getElementById("timeline").innerHTML = payload.stages.map(stage => {{
    const metrics = Object.entries(stage.metrics).map(([k, v]) => `<span class="metric">${{k}}: ${{v}}</span>`).join("");
    return `<div class="card"><strong>${{stage.label}}</strong><div class="muted">${{stage.description}}</div><div class="muted">${{stage.file}}</div>${{metrics}}</div>`;
  }}).join("");
}}
function render() {{
  if (mode === "stagePath") {{
    renderPath();
    return;
  }}
  const data = payload[mode];
  const points = data.points;
  const b = bounds(points);
  hovered = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fbfdff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#e5ebf3"; ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {{
    const x = 60 + ((canvas.width - 120) / 6) * i;
    const y = 56 + ((canvas.height - 112) / 6) * i;
    ctx.beginPath(); ctx.moveTo(x, 56); ctx.lineTo(x, canvas.height - 56); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(60, y); ctx.lineTo(canvas.width - 60, y); ctx.stroke();
  }}
  ctx.fillStyle = "#667085"; ctx.font = "13px sans-serif";
  const axisX = projection === "tsne" ? "t-SNE 1" : "PC1";
  const axisY = projection === "tsne" ? "t-SNE 2" : "PC2";
  ctx.fillText(axisY, 24, 28); ctx.fillText(axisX, canvas.width - 98, canvas.height - 22);
  points.forEach(point => {{
    const coords = projectionCoords(point);
    const x = sx(coords[0], b), y = sy(coords[1], b);
    ctx.beginPath(); ctx.fillStyle = color(point.avg); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    if (mouse && Math.hypot(mouse.x - x, mouse.y - y) < 12) hovered = point;
  }});
  document.getElementById("keysText").textContent = data.keys.join(", ");
  document.getElementById("explainedText").textContent = projection === "tsne"
    ? `t-SNE · perplexity ${{data.tsne.perplexity}}`
    : `PCA · ${{data.explained.map(v => Math.round(v * 100) + "%").join(" / ")}}`;
  document.getElementById("countText").textContent = points.length;
  renderDetail();
}}
function pointColor(score, idx) {{
  if (score !== null && score !== undefined) return color(score);
  const palette = ["#155eef", "#099250", "#dc6803", "#7a5af8", "#c11574", "#0e7090"];
  return palette[idx % palette.length];
}}
function pathPixel(point) {{
  return {{
    x: 64 + point.x * (canvas.width - 128),
    y: 54 + point.y * (canvas.height - 108)
  }};
}}
function interpPoint(a, b, t) {{
  return {{
    ...b,
    id: a.id,
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    label: b.label,
    output: b.output,
    axis: b.axis,
    score: b.score
  }};
}}
function currentPathPoints() {{
  const frames = payload.stagePath.frames;
  const total = frames.length - 1;
  const cycle = 7800;
  const elapsed = (performance.now() - pathStartedAt) % cycle;
  const raw = (elapsed / cycle) * total;
  const idx = Math.floor(raw);
  const t = raw - idx;
  const from = frames[Math.min(idx, total)];
  const to = frames[Math.min(idx + 1, total)];
  const fromById = Object.fromEntries(from.points.map(p => [p.id, p]));
  const points = to.points.map(point => interpPoint(fromById[point.id] || point, point, t));
  return {{points, from, to, progress: t}};
}}
function renderPath() {{
  const frames = payload.stagePath.frames;
  const live = currentPathPoints();
  hovered = null;
  stageClickTargets = [];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fbfdff"; ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#d9e1ec"; ctx.lineWidth = 1;
  ctx.fillStyle = "#667085"; ctx.font = "12px sans-serif";
  frames.forEach(frame => {{
    const stageX = 0.08 + (frames.indexOf(frame) / Math.max(frames.length - 1, 1)) * 0.84;
    const x = pathPixel({{x: stageX, y: 0.5}}).x;
    const targetMode = frame.id === "stage8" ? "qualityPca" : frame.id === "stage10" ? "postprocessPca" : frame.id === "stage11" ? "llmPca" : null;
    if (targetMode) {{
      stageClickTargets.push({{mode: targetMode, x1: x - 46, x2: x + 46, y1: 0, y2: canvas.height}});
      ctx.fillStyle = "rgba(21,94,239,0.06)";
      ctx.fillRect(x - 46, 54, 92, canvas.height - 108);
    }}
    ctx.beginPath(); ctx.moveTo(x, 54); ctx.lineTo(x, canvas.height - 54); ctx.stroke();
    ctx.fillStyle = targetMode ? "#155eef" : "#667085";
    ctx.fillText(frame.label, x - 24, 28);
  }});

  const ids = live.points.map(point => point.id);
  ids.forEach((id, idx) => {{
    ctx.beginPath();
    frames.forEach((frame, frameIdx) => {{
      const point = frame.points.find(item => item.id === id);
      if (!point) return;
      const px = pathPixel(point);
      if (frameIdx === 0) ctx.moveTo(px.x, px.y);
      else ctx.lineTo(px.x, px.y);
    }});
    ctx.strokeStyle = "rgba(102,112,133,0.18)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }});

  live.points.forEach((point, idx) => {{
    const px = pathPixel(point);
    ctx.beginPath();
    ctx.fillStyle = pointColor(point.score, idx);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.arc(px.x, px.y, 9, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#172033";
    ctx.font = "11px sans-serif";
    ctx.fillText(point.id, px.x + 12, px.y + 4);
    if (mouse && Math.hypot(mouse.x - px.x, mouse.y - px.y) < 14) hovered = point;
  }});

  document.getElementById("keysText").textContent = live.to.axis;
  document.getElementById("explainedText").textContent = `${{live.from.label}} → ${{live.to.label}}`;
  document.getElementById("countText").textContent = live.points.length;
  renderDetail();
  animationHandle = requestAnimationFrame(() => {{
    if (mode === "stagePath") renderPath();
  }});
}}
function renderDetail() {{
  const detail = document.getElementById("detail");
  if (!hovered) {{
    detail.innerHTML = "<strong>Hover a point</strong><div class='muted'>샘플 점수와 출력이 여기에 표시됩니다.</div>";
    return;
  }}
  if (mode === "stagePath") {{
    const scoreText = hovered.score === null || hovered.score === undefined ? "" : `<div style="margin-top:8px;">Score-like signal <strong>${{hovered.score.toFixed(2)}}</strong></div>`;
    detail.innerHTML = `<strong>${{hovered.label}}</strong><div class="muted">${{hovered.axis}}</div>${{scoreText}}<div style="margin-top:10px;"><strong>Stage output</strong><br><span class="muted">${{hovered.output}}</span></div>`;
    return;
  }}
  const scores = Object.entries(hovered.scores).map(([k, v]) => `${{k}} ${{v}}`).join(" · ");
  detail.innerHTML = `<strong>${{hovered.id}}</strong><div class="muted">${{hovered.platform}} · ${{hovered.domain}}</div><div style="margin-top:8px;">Avg <strong>${{hovered.avg.toFixed(2)}}</strong> · ${{scores}}</div><div style="margin-top:10px;"><strong>Output</strong><br><span class="muted">${{hovered.output}}</span></div>`;
}}
document.getElementById("qualityBtn").addEventListener("click", () => setMode("qualityPca"));
document.getElementById("postprocessBtn").addEventListener("click", () => setMode("postprocessPca"));
document.getElementById("llmBtn").addEventListener("click", () => setMode("llmPca"));
document.getElementById("pathBtn").addEventListener("click", () => setMode("stagePath"));
document.getElementById("pcaBtn").addEventListener("click", () => setProjection("pca"));
document.getElementById("tsneBtn").addEventListener("click", () => setProjection("tsne"));
canvas.addEventListener("mousemove", event => {{
  const rect = canvas.getBoundingClientRect();
  mouse = {{x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height}};
  render();
}});
canvas.addEventListener("mouseleave", () => {{ mouse = null; render(); }});
canvas.addEventListener("click", event => {{
  if (mode !== "stagePath") return;
  const rect = canvas.getBoundingClientRect();
  const click = {{x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height}};
  const target = stageClickTargets.find(item => click.x >= item.x1 && click.x <= item.x2 && click.y >= item.y1 && click.y <= item.y2);
  if (target) setMode(target.mode);
}});
renderTimeline();
render();
async function refreshPayload() {{
  if (!location.protocol.startsWith("http")) return;
  try {{
    const response = await fetch("/api/payload", {{cache: "no-store"}});
    if (!response.ok) return;
    payload = await response.json();
    renderTimeline();
    if (mode !== "stagePath") render();
  }} catch (error) {{
    // Static-file mode has no API endpoint; the embedded payload still works.
  }}
}}
setInterval(refreshPayload, 2000);
</script>
</body>
</html>
"""


def serve_dashboard(host: str, port: int, curated_dir: Path) -> None:
    cached_signature: tuple[tuple[str, int, int], ...] | None = None
    cached_payload: dict[str, Any] | None = None

    def current_payload() -> dict[str, Any]:
        nonlocal cached_signature, cached_payload
        signature = pipeline_signature(curated_dir)
        if cached_payload is None or signature != cached_signature:
            cached_payload = load_pipeline(curated_dir)
            cached_signature = signature
        return cached_payload

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_text(self, content: str, content_type: str) -> None:
            encoded = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path in {"/", "/dashboard", "/index.html"}:
                self.send_text(dashboard_html(current_payload()), "text/html")
                return
            if self.path == "/api/payload":
                self.send_text(json.dumps(current_payload(), ensure_ascii=False), "application/json")
                return
            self.send_response(404)
            self.end_headers()

    server = HTTPServer((host, port), Handler)
    print(f"Serving full pipeline dashboard at http://{host}:{port}/")
    print(f"Polling curated files from: {curated_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nStopped dashboard server.")


def main() -> None:
    args = parse_args()
    if args.serve:
        serve_dashboard(args.host, args.port, Path(args.curated_dir))
        return
    payload = load_pipeline(Path(args.curated_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dashboard_html(payload), encoding="utf-8")
    print(f"Saved full pipeline dashboard to: {output}")
    print(f"Stage8 semantic PCA keys: {', '.join(payload['qualityPca']['keys'])}")
    print(f"Stage10 dedup PCA keys: {', '.join(payload['postprocessPca']['keys'])}")
    print(f"Stage11 reward PCA keys: {', '.join(payload['llmPca']['keys'])}")


if __name__ == "__main__":
    main()
