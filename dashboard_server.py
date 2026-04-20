import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from visualize_distribution import (
    REFERENCE_COLOR,
    STAGE_COLORS,
    STAGE_LABELS,
    STAGES,
    make_dashboard_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live quality-score data dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--generated", default="data/example_generated_data.jsonl")
    parser.add_argument("--evaluated", default="data/evaluated_data.jsonl")
    parser.add_argument("--reference", default="data/raw/sample_chat_en.jsonl")
    parser.add_argument("--poll-ms", type=int, default=2000)
    return parser.parse_args()


def file_signature(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            parts.append(f"{path}:missing")
    return "|".join(parts)


def serialize_payload(
    points: list[dict[str, Any]],
    reference_points: list[dict[str, Any]],
    summary: dict[str, Any],
    bounds: dict[str, float],
    *,
    version: str,
    updated_at: float,
) -> dict[str, Any]:
    return {
        "version": version,
        "updatedAt": updated_at,
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
                "sourceText": point["source_text"][:240],
                "rewrittenText": point["rewritten_text"][:240],
                "coords": point["coords"],
            }
            for point in points
        ],
        "reference": [
            {
                "id": point["sample_id"],
                "order": point["order"],
                "domain": point["domain"],
                "sourceText": point["source_text"][:200],
                "coords": point["coords"]["source"],
            }
            for point in reference_points
        ],
    }


class DashboardState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.generated = Path(args.generated)
        self.evaluated = Path(args.evaluated)
        self.reference = Path(args.reference)
        self.poll_ms = args.poll_ms
        self._signature = ""
        self._payload: dict[str, Any] | None = None
        self._error: str | None = None

    @property
    def paths(self) -> list[Path]:
        return [self.generated, self.evaluated, self.reference]

    def payload(self, force: bool = False) -> dict[str, Any]:
        signature = file_signature(self.paths)
        if not force and self._payload is not None and signature == self._signature:
            return self._payload

        try:
            points, reference_points, summary, bounds = make_dashboard_payload(
                self.generated,
                self.evaluated,
                self.reference,
            )
            self._payload = serialize_payload(
                points,
                reference_points,
                summary,
                bounds,
                version=str(abs(hash(signature))),
                updated_at=time.time(),
            )
            self._signature = signature
            self._error = None
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            if self._payload is None:
                raise
        return self._payload

    def status(self) -> dict[str, Any]:
        payload = self.payload()
        return {
            "version": payload["version"],
            "updatedAt": payload["updatedAt"],
            "pollMs": self.poll_ms,
            "paths": {
                "generated": str(self.generated),
                "evaluated": str(self.evaluated),
                "reference": str(self.reference),
            },
            "error": self._error,
        }


def dashboard_html(poll_ms: int) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Dataset Quality Dashboard</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e1ec;
      --accent: #155eef;
      --ok: #039855;
      --warn: #dc6803;
      --bad: #d92d20;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Apple SD Gothic Neo", "Noto Sans KR", "Segoe UI", sans-serif;
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 22px;
    }}
    .topbar {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      line-height: 1.45;
      max-width: 860px;
    }}
    .status {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .dot {{
      width: 9px;
      height: 9px;
      border-radius: 99px;
      background: var(--ok);
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 370px;
      gap: 14px;
    }}
    .panel, .kpi {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 28px rgba(16, 24, 40, 0.06);
    }}
    .panel {{ padding: 14px; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .kpi {{
      padding: 12px;
      min-height: 80px;
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
    .viz-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .title {{
      font-weight: 800;
      font-size: 15px;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button {{
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      background: #fff;
      color: var(--ink);
      font-weight: 800;
      cursor: pointer;
    }}
    button.active, button.primary {{
      background: var(--accent);
      border-color: var(--accent);
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
    .timeline {{
      display: grid;
      grid-template-columns: 1fr 86px 120px;
      gap: 12px;
      align-items: center;
      margin-top: 12px;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 9px;
      color: var(--muted);
      font-size: 12px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 99px;
      display: inline-block;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 0;
      border-bottom: 1px solid #edf1f6;
      color: var(--muted);
      font-size: 13px;
    }}
    .metric strong {{ color: var(--ink); }}
    .detail {{
      min-height: 180px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
      font-size: 13px;
      line-height: 1.45;
    }}
    .muted {{ color: var(--muted); }}
    .bars {{
      display: grid;
      gap: 6px;
      margin-top: 8px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 96px 1fr 26px;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }}
    .bar-track {{
      height: 8px;
      border-radius: 99px;
      background: #eef2f7;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 12px;
    }}
    th, td {{
      text-align: left;
      padding: 7px 4px;
      border-bottom: 1px solid #edf1f6;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 800;
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
        <h1>Live Data Quality Distribution</h1>
        <div class="subtitle">Curator pipeline 산출 JSONL을 계속 읽어서 metadata quality_score 벡터를 PCA로 2D/3D 차원 축소해 갱신합니다.</div>
      </div>
      <div class="status"><span class="dot" id="statusDot"></span><span id="statusText">connecting</span></div>
    </section>

    <section class="kpis">
      <div class="kpi"><span>생성 샘플</span><strong id="sampleCount">-</strong></div>
      <div class="kpi"><span>평균 Final Score</span><strong id="meanScore">-</strong></div>
      <div class="kpi"><span>원본→평가 분포 차이</span><strong id="refGap">-</strong></div>
      <div class="kpi"><span>공간 균일도</span><strong id="evenness">-</strong></div>
    </section>

    <section class="grid">
      <div class="panel">
        <div class="viz-head">
          <div>
            <div class="title">Quality Score PCA Space</div>
            <div class="muted" id="versionText">-</div>
          </div>
          <div class="controls">
            <button id="mode2d" class="active">2D</button>
            <button id="mode3d">3D</button>
            <button id="playButton" class="primary">Play</button>
            <button id="reloadButton">Refresh</button>
          </div>
        </div>
        <canvas id="chart" width="1100" height="690"></canvas>
        <div class="timeline">
          <input id="stageSlider" type="range" min="0" max="3.25" step="0.005" value="0">
          <strong id="stageText">Source</strong>
          <span class="muted" id="visibleText"></span>
        </div>
        <div class="chips" id="legend"></div>
      </div>

      <aside class="panel">
        <div class="title">분포 진단</div>
        <div class="metric"><span>PCA 설명 분산</span><strong id="varianceText">-</strong></div>
        <div class="metric"><span>도메인 균일도</span><strong id="domainEvenness">-</strong></div>
        <div class="metric"><span>플랫폼 균일도</span><strong id="platformEvenness">-</strong></div>
        <div class="metric"><span>생성 분포 spread</span><strong id="spreadText">-</strong></div>
        <div class="metric"><span>낮은 품질 샘플</span><strong id="lowQuality">-</strong></div>

        <div class="title" style="margin-top:14px;">샘플 상세</div>
        <div class="detail" id="detail"><strong>Point detail</strong><div class="muted">-</div></div>

        <div class="title" style="margin-top:14px;">도메인 분포</div>
        <div class="bars" id="domainBars"></div>

        <div class="title" style="margin-top:14px;">개선 우선 후보</div>
        <table>
          <thead><tr><th>ID</th><th>Score</th><th>Domain</th><th>Output</th></tr></thead>
          <tbody id="lowestTable"></tbody>
        </table>
      </aside>
    </section>
  </main>

  <script>
    const pollMs = {poll_ms};
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("stageSlider");
    const playButton = document.getElementById("playButton");
    const reloadButton = document.getElementById("reloadButton");
    const mode2d = document.getElementById("mode2d");
    const mode3d = document.getElementById("mode3d");
    let payload = null;
    let previousVersion = "";
    let mode = "2d";
    let playing = true;
    let lastTimestamp = 0;
    let rotation = 0.62;
    let mouse = null;
    let hovered = null;

    function fmt(value, digits = 3) {{
      return Number.isFinite(value) ? value.toFixed(digits) : "-";
    }}
    function setStatus(text, ok = true) {{
      document.getElementById("statusText").textContent = text;
      document.getElementById("statusDot").style.background = ok ? "var(--ok)" : "var(--bad)";
    }}
    function clamp(value, low, high) {{ return Math.max(low, Math.min(high, value)); }}
    function lerp(a, b, t) {{ return a + (b - a) * t; }}
    function scoreColor(score) {{
      const t = clamp((score - 1) / 4, 0, 1);
      const r = Math.round(217 + (3 - 217) * t);
      const g = Math.round(45 + (152 - 45) * t);
      const b = Math.round(32 + (85 - 32) * t);
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
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
        depth: 0,
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
      return {{ x: canvas.width / 2 + rx * scale * perspective, y: canvas.height / 2 - ry * scale * perspective, scale: perspective, depth }};
    }}
    function project(coord) {{ return mode === "3d" ? project3d(coord) : project2d(coord); }}
    function localStage(point, value) {{
      return clamp(value - point.order * 0.018, -0.2, 3);
    }}
    function interpolate(point, value) {{
      const local = localStage(point, value);
      if (local < 0) return null;
      const stages = payload.stageOrder;
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
        opacity: clamp(local / 0.25, 0.12, 1),
      }};
    }}
    function renderShell() {{
      if (!payload) return;
      const summary = payload.summary;
      document.getElementById("sampleCount").textContent = summary.sample_count;
      document.getElementById("meanScore").textContent = fmt(summary.mean_final_score);
      document.getElementById("refGap").textContent = fmt(summary.centroid_shift.reference_to_evaluated);
      document.getElementById("evenness").textContent = Math.round(summary.spatial_evenness * 100) + "%";
      document.getElementById("varianceText").textContent = summary.explained_variance_ratio.map(v => Math.round(v * 100) + "%").join(" / ");
      document.getElementById("domainEvenness").textContent = Math.round(summary.domain_evenness * 100) + "%";
      document.getElementById("platformEvenness").textContent = Math.round(summary.platform_evenness * 100) + "%";
      document.getElementById("spreadText").textContent = fmt(summary.generated_spread);
      document.getElementById("lowQuality").textContent = `${{summary.low_quality_count}} / ${{summary.sample_count}}`;
      document.getElementById("versionText").textContent = `updated ${{new Date(payload.updatedAt * 1000).toLocaleTimeString()}}`;
      document.getElementById("legend").innerHTML = payload.stageOrder.map(stage => (
        `<span class="chip"><span class="swatch" style="background:${{payload.stageColors[stage]}}"></span>${{payload.stageLabels[stage]}}</span>`
      )).join("") + `<span class="chip"><span class="swatch" style="background:${{payload.referenceColor}}"></span>Reference</span><span class="chip">Fill = score</span>`;
      const domainMax = Math.max(1, ...Object.values(summary.domain_distribution));
      document.getElementById("domainBars").innerHTML = Object.entries(summary.domain_distribution)
        .sort((a, b) => b[1] - a[1])
        .map(([name, count]) => `<div class="bar-row"><span>${{name}}</span><div class="bar-track"><div class="bar-fill" style="width:${{(count / domainMax) * 100}}%"></div></div><strong>${{count}}</strong></div>`)
        .join("");
      document.getElementById("lowestTable").innerHTML = summary.lowest_samples.map(row => (
        `<tr><td>${{row.id}}</td><td>${{fmt(row.score)}}</td><td>${{row.domain}}</td><td>${{row.text}}</td></tr>`
      )).join("");
    }}
    async function loadPayload(force = false) {{
      try {{
        const response = await fetch(`/api/payload${{force ? "?force=1" : ""}}`, {{cache: "no-store"}});
        if (!response.ok) throw new Error(await response.text());
        const next = await response.json();
        const changed = next.version !== previousVersion;
        payload = next;
        if (changed) {{
          previousVersion = next.version;
          renderShell();
          slider.value = "0";
        }}
        setStatus(`live · ${{payload.points.length}} samples`, true);
        render();
      }} catch (error) {{
        setStatus(`error · ${{error.message}}`, false);
      }}
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
        ctx.beginPath(); ctx.moveTo(x, 58); ctx.lineTo(x, canvas.height - 58); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(72, y); ctx.lineTo(canvas.width - 72, y); ctx.stroke();
      }}
      ctx.fillStyle = "#667085";
      ctx.font = "13px Inter, sans-serif";
      ctx.fillText(mode === "3d" ? "Quality Score PC1 / PC2 / PC3" : "Quality Score PC1", 24, 28);
      ctx.fillText(mode === "3d" ? "drag = rotate" : "Quality Score PC2", canvas.width - 148, canvas.height - 24);
    }}
    function drawReference() {{
      ctx.save();
      ctx.globalAlpha = 0.54;
      ctx.strokeStyle = payload.referenceColor;
      ctx.lineWidth = 1.3;
      payload.reference.forEach(point => {{
        const p = project({{x: point.coords[0], y: point.coords[1], z: point.coords[2]}});
        ctx.beginPath();
        ctx.moveTo(p.x - 5, p.y - 5); ctx.lineTo(p.x + 5, p.y + 5);
        ctx.moveTo(p.x + 5, p.y - 5); ctx.lineTo(p.x - 5, p.y + 5);
        ctx.stroke();
      }});
      ctx.restore();
    }}
    function drawPaths(value) {{
      ctx.save();
      ctx.strokeStyle = "rgba(71, 84, 103, 0.15)";
      ctx.lineWidth = 1;
      payload.points.forEach(point => {{
        if (localStage(point, value) < 0.2) return;
        ctx.beginPath();
        payload.stageOrder.forEach((stage, index) => {{
          const c = point.coords[stage];
          const p = project({{x: c[0], y: c[1], z: c[2]}});
          if (index === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }});
        ctx.stroke();
      }});
      ctx.restore();
    }}
    function drawCentroids(value) {{
      const stageIndex = Math.min(3, Math.max(0, Math.round(value)));
      ctx.save();
      ctx.strokeStyle = "rgba(17, 24, 39, 0.55)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      payload.stageOrder.forEach((stage, index) => {{
        const c = payload.summary.stage_centroids[stage];
        const p = project({{x: c[0], y: c[1], z: c[2]}});
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }});
      ctx.stroke();
      payload.stageOrder.forEach((stage, index) => {{
        const c = payload.summary.stage_centroids[stage];
        const p = project({{x: c[0], y: c[1], z: c[2]}});
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
        const c = interpolate(point, value);
        if (!c) return;
        drawable.push({{point, coord: c, projected: project(c)}});
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
          if (hit <= radius + 4 && (!hovered || hit < hovered.hit)) hovered = {{...item, hit}};
        }}
      }});
      document.getElementById("visibleText").textContent = `${{drawable.length}} / ${{payload.points.length}} visible`;
    }}
    function renderDetail() {{
      const detail = document.getElementById("detail");
      if (!hovered) {{
        detail.innerHTML = "<strong>Point detail</strong><div class='muted'>-</div>";
        return;
      }}
      const p = hovered.point;
      const qualityRows = Object.entries(p.scores)
        .filter(([key]) => key !== "final")
        .map(([key, value]) => `${{key.replaceAll("_", " ")}} ${{fmt(Number(value), 0)}}`)
        .join(" · ");
      detail.innerHTML = `
        <strong>${{p.id}}</strong>
        <div class="muted">${{p.domain}} · ${{p.platform}} · ${{p.speechAct}} · ${{p.register}}</div>
        <div style="margin-top:8px;">Avg <strong>${{fmt(p.finalScore, 2)}}</strong> · ${{qualityRows}}</div>
        <div style="margin-top:10px;"><strong>Source</strong><br><span class="muted">${{p.sourceText}}</span></div>
        <div style="margin-top:10px;"><strong>Korean Output</strong><br><span class="muted">${{p.rewrittenText || "-"}}</span></div>
      `;
    }}
    function render() {{
      if (!payload) return;
      const value = parseFloat(slider.value);
      drawGrid();
      drawReference();
      drawPaths(value);
      drawCentroids(value);
      drawPoints(value);
      const stageIndex = Math.min(3, Math.max(0, Math.round(value)));
      document.getElementById("stageText").textContent = payload.stageLabels[payload.stageOrder[stageIndex]];
      renderDetail();
    }}
    function animate(timestamp) {{
      if (!lastTimestamp) lastTimestamp = timestamp;
      const delta = timestamp - lastTimestamp;
      lastTimestamp = timestamp;
      if (playing && payload) {{
        let next = parseFloat(slider.value) + delta * 0.00034;
        if (next > parseFloat(slider.max)) next = 0;
        slider.value = next.toFixed(3);
        if (mode === "3d") rotation += 0.0035;
        render();
      }}
      requestAnimationFrame(animate);
    }}
    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
      render();
    }});
    reloadButton.addEventListener("click", () => loadPayload(true));
    slider.addEventListener("input", render);
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
    window.addEventListener("mouseup", () => dragging = false);
    canvas.addEventListener("mouseleave", () => {{
      mouse = null;
      dragging = false;
      render();
    }});
    loadPayload(true);
    setInterval(() => loadPayload(false), pollMs);
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_text(self, body: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_text(json.dumps(payload, ensure_ascii=False), "application/json", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/dashboard"}:
                self.send_text(dashboard_html(self.state.poll_ms), "text/html")
            elif parsed.path == "/api/payload":
                force = "force=1" in parsed.query
                self.send_json(self.state.payload(force=force))
            elif parsed.path == "/api/status":
                self.send_json(self.state.status())
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    args = parse_args()
    state = DashboardState(args)
    DashboardHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving live dashboard at http://{args.host}:{args.port}")
    print(f"Watching generated={state.generated}, evaluated={state.evaluated}, reference={state.reference}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
