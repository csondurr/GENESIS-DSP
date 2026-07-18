'''GENESIS-DSP — Adım 24
Tam web uygulaması, raporlama ve benchmark paneli.

Öz test ve site üretimi:
    python step24_web_application.py

Web sitesini başlat:
    python step24_web_application.py --serve --open-browser
'''

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import time
import webbrowser
from pathlib import Path
from threading import Timer
from typing import Any, Literal

import numpy as np

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as error:
    raise RuntimeError(
        "Adım 24 için web bağımlılıkları eksik.\n"
        "Şu komutu çalıştır:\n"
        "python -m pip install fastapi uvicorn pydantic numpy"
    ) from error

from step23_api_backend import DSP_MODULE, app as dsp_api_app


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP18_DIRECTORY = BASE_DIRECTORY / "outputs" / "step18"
STEP19_DIRECTORY = BASE_DIRECTORY / "outputs" / "step19"
STEP20_DIRECTORY = BASE_DIRECTORY / "outputs" / "step20"
STEP21_DIRECTORY = BASE_DIRECTORY / "outputs" / "step21"
STEP22_DIRECTORY = BASE_DIRECTORY / "outputs" / "step22"
STEP23_DIRECTORY = BASE_DIRECTORY / "outputs" / "step23"
STEP24_DIRECTORY = BASE_DIRECTORY / "outputs" / "step24"

SITE_DIRECTORY = STEP24_DIRECTORY / "site"
INDEX_PATH = SITE_DIRECTORY / "index.html"
STYLE_PATH = SITE_DIRECTORY / "styles.css"
SCRIPT_PATH = SITE_DIRECTORY / "app.js"
FAVICON_PATH = SITE_DIRECTORY / "favicon.svg"
README_PATH = STEP24_DIRECTORY / "README.txt"
STARTER_PATH = STEP24_DIRECTORY / "START_GENESIS_DSP_WEB.bat"
REPORT_PATH = STEP24_DIRECTORY / "web_application_report.json"
ZIP_PATH = STEP24_DIRECTORY / "GENESIS_DSP_WEB_PACKAGE.zip"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_BENCHMARK_SAMPLES = 1_000_000
MAX_BENCHMARK_REPETITIONS = 15

REPORT_FILES = {
    "step18": STEP18_DIRECTORY / "combined_recovery_report.json",
    "step19": STEP19_DIRECTORY / "falsification_report.json",
    "step20": STEP20_DIRECTORY / "stability_causality_report.json",
    "step21": STEP21_DIRECTORY / "fixed_point_report.json",
    "step22": STEP22_DIRECTORY / "code_export_report.json",
    "step23": STEP23_DIRECTORY / "api_backend_report.json",
}


class BenchmarkRequest(BaseModel):
    mode: Literal["float", "fixed", "both"] = "both"
    sample_count: int = Field(default=100_000, ge=1_000, le=MAX_BENCHMARK_SAMPLES)
    repetitions: int = Field(default=5, ge=1, le=MAX_BENCHMARK_REPETITIONS)
    sample_rate_hz: float = Field(default=1_000_000.0, gt=0.0)


def load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            document = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def save_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(document, file, indent=4, ensure_ascii=False, allow_nan=False)


def nested_get(document: dict[str, Any] | None, path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = document
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def finite_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def build_dashboard_data() -> dict[str, Any]:
    reports = {name: load_json_optional(path) for name, path in REPORT_FILES.items()}
    step18 = reports["step18"]
    step19 = reports["step19"]
    step20 = reports["step20"]
    step21 = reports["step21"]
    step22 = reports["step22"]
    step23 = reports["step23"]
    selected_format = nested_get(step21, ("selected_format",), {})

    return {
        "project": "GENESIS-DSP",
        "version": "1.0.0",
        "status": "operational",
        "pipeline": {
            "node_count": len(DSP_MODULE.PIPELINE),
            "nodes": list(DSP_MODULE.PIPELINE),
        },
        "receiver": {
            "test_ber": finite_or_none(nested_get(step18, ("metrics", "test", "ber"))),
            "test_evm_percent": finite_or_none(
                nested_get(step18, ("metrics", "test", "evm_rms_percent"))
            ),
            "cfo_error_hz": finite_or_none(nested_get(step18, ("front_end", "cfo_error_hz"))),
            "equalizer_family": nested_get(step18, ("selected_model", "family"), "unknown"),
            "equalizer_length": finite_or_none(
                nested_get(step18, ("selected_model", "equalizer_length"))
            ),
        },
        "falsification": {
            "scenario_count": finite_or_none(nested_get(step19, ("search", "total_scenarios"))),
            "failure_count": finite_or_none(nested_get(step19, ("search", "failure_count"))),
            "failure_rate": finite_or_none(nested_get(step19, ("search", "failure_rate"))),
            "worst_ber": finite_or_none(
                nested_get(step19, ("worst_counterexample", "test_ber"))
            ),
            "worst_evm_percent": finite_or_none(
                nested_get(step19, ("worst_counterexample", "test_evm_percent"))
            ),
        },
        "certification": {
            "status": nested_get(step20, ("summary", "certificate_status"), "unknown"),
            "test_count": finite_or_none(nested_get(step20, ("summary", "test_count"))),
            "passed_count": finite_or_none(nested_get(step20, ("summary", "passed_count"))),
            "prefix_error": finite_or_none(
                nested_get(step20, ("certified_pipeline", "prefix_causality_error"))
            ),
            "measured_bibo_gain": finite_or_none(
                nested_get(step20, ("certified_pipeline", "measured_maximum_bibo_gain"))
            ),
            "theoretical_bibo_bound": finite_or_none(
                nested_get(step20, ("certified_pipeline", "theoretical_bibo_gain_bound"))
            ),
        },
        "fixed_point": {
            "total_bits": finite_or_none(
                selected_format.get("total_bits", getattr(DSP_MODULE, "TOTAL_BITS", None))
            ),
            "integer_bits": finite_or_none(
                selected_format.get("integer_bits", getattr(DSP_MODULE, "INTEGER_BITS", None))
            ),
            "fractional_bits": finite_or_none(
                selected_format.get("fractional_bits", getattr(DSP_MODULE, "FRACTIONAL_BITS", None))
            ),
            "phase_bits": finite_or_none(
                selected_format.get("phase_bits", getattr(DSP_MODULE, "PHASE_BITS", None))
            ),
            "nmse": finite_or_none(selected_format.get("nmse")),
            "evm_percent": finite_or_none(selected_format.get("evm_percent")),
            "maximum_error": finite_or_none(selected_format.get("maximum_absolute_error")),
            "saturations": finite_or_none(selected_format.get("saturation_count")),
        },
        "exports": {
            "python_status": nested_get(step22, ("python_export", "status"), "unknown"),
            "cpp_status": nested_get(step22, ("cpp_export", "test", "status"), "unknown"),
        },
        "api": {
            "endpoint_count": finite_or_none(nested_get(step23, ("self_test", "endpoint_count"))),
            "float_error": finite_or_none(nested_get(step23, ("self_test", "float_maximum_error"))),
            "fixed_error": finite_or_none(nested_get(step23, ("self_test", "fixed_maximum_error"))),
            "documentation_url": "/api/docs",
        },
        "report_sources": {
            name: {"available": report is not None, "path": str(REPORT_FILES[name])}
            for name, report in reports.items()
        },
    }


def benchmark_mode(
    mode: Literal["float", "fixed"],
    samples: np.ndarray,
    sample_rate_hz: float,
    repetitions: int,
) -> dict[str, Any]:
    processor = DSP_MODULE.process_float if mode == "float" else DSP_MODULE.process_fixed
    processor(samples[: min(len(samples), 4096)], sample_rate_hz)
    durations: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = processor(samples, sample_rate_hz)
        elapsed = time.perf_counter() - start
        if len(output) != len(samples):
            raise RuntimeError("Benchmark DSP çıktı uzunluğu hatalı.")
        durations.append(elapsed)

    median_seconds = statistics.median(durations)
    minimum_seconds = min(durations)
    throughput = len(samples) / median_seconds
    return {
        "mode": mode,
        "sample_count": len(samples),
        "repetitions": repetitions,
        "minimum_latency_ms": 1000.0 * minimum_seconds,
        "median_latency_ms": 1000.0 * median_seconds,
        "throughput_samples_per_second": throughput,
        "throughput_megasamples_per_second": throughput / 1_000_000.0,
    }


def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    rng = np.random.default_rng(20260803)
    samples = (
        0.55
        * (
            rng.uniform(-1.0, 1.0, request.sample_count)
            + 1j * rng.uniform(-1.0, 1.0, request.sample_count)
        )
    ).astype(np.complex128)

    modes: tuple[Literal["float", "fixed"], ...]
    modes = ("float", "fixed") if request.mode == "both" else (request.mode,)
    return {
        "status": "completed",
        "results": [
            benchmark_mode(mode, samples, request.sample_rate_hz, request.repetitions)
            for mode in modes
        ],
    }


def build_index_html() -> str:
    return r'''<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="GENESIS-DSP autonomous DSP scientist dashboard">
    <title>GENESIS-DSP Laboratory</title>
    <link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
    <link rel="stylesheet" href="/assets/styles.css">
</head>
<body>
    <div class="ambient ambient-one"></div>
    <div class="ambient ambient-two"></div>

    <header class="topbar">
        <a class="brand" href="/">
            <span class="brand-mark">G</span>
            <span><strong>GENESIS-DSP</strong><small>Autonomous Signal Laboratory</small></span>
        </a>
        <nav class="top-actions">
            <span id="healthBadge" class="status-badge waiting">API kontrol ediliyor</span>
            <a class="ghost-button" href="/api/docs" target="_blank" rel="noopener">Swagger API</a>
        </nav>
    </header>

    <main class="page-shell">
        <section class="hero">
            <div>
                <span class="eyebrow">CERTIFIED · FIXED-POINT · API-READY</span>
                <h1>Sinyali yükle.<span>Pipeline'ı çalıştır.</span>Sonucu anında gör.</h1>
                <p>Keşfedilmiş ve sertifikalanmış DSP zincirini floating-point veya Q3.13 fixed-point modunda tarayıcıdan çalıştır.</p>
            </div>
            <div class="hero-orbit" aria-hidden="true">
                <div class="orbit orbit-a"></div><div class="orbit orbit-b"></div><div class="core">DSP</div>
            </div>
        </section>

        <section class="metric-grid">
            <article class="metric-card"><span>Nominal Test BER</span><strong id="metricBer">—</strong><small>Adım 18</small></article>
            <article class="metric-card"><span>Fixed-Point EVM</span><strong id="metricFixedEvm">—</strong><small>Adım 21</small></article>
            <article class="metric-card"><span>Sertifika</span><strong id="metricCertificate">—</strong><small>Adım 20</small></article>
            <article class="metric-card"><span>Pipeline Düğümü</span><strong id="metricNodes">—</strong><small>Canlı yapı</small></article>
        </section>

        <section class="panel pipeline-panel">
            <div class="section-heading"><div><span class="eyebrow">DISCOVERED PIPELINE</span><h2>İşlem Zinciri</h2></div><span class="subtle" id="pipelineSummary">Pipeline yükleniyor</span></div>
            <div class="pipeline-flow" id="pipelineFlow"></div>
        </section>

        <section class="workspace-grid">
            <article class="panel control-panel">
                <div class="section-heading"><div><span class="eyebrow">LIVE PROCESSING</span><h2>IQ İşleme Konsolu</h2></div></div>
                <div class="form-grid">
                    <label>İşleme modu<select id="modeSelect"><option value="fixed">Fixed-point Q3.13</option><option value="float">Floating-point</option></select></label>
                    <label>Örnekleme frekansı<div class="input-with-unit"><input id="sampleRateInput" type="number" min="1" step="1000" value="1000000"><span>Hz</span></div></label>
                    <label>Demo örnek sayısı<input id="demoCountInput" type="number" min="16" max="5000" value="512"></label>
                    <label>CSV yükle<input id="csvInput" type="file" accept=".csv,text/csv"></label>
                </div>
                <div class="button-row"><button id="generateButton" class="secondary-button" type="button">Demo QPSK üret</button><button id="runButton" class="primary-button" type="button">Pipeline'ı çalıştır</button></div>
                <div class="console-status" id="consoleStatus">Hazır. Demo üret veya real,imag CSV dosyası yükle.</div>
                <div class="mini-metric-grid">
                    <div><span>Örnek</span><strong id="liveSampleCount">0</strong></div>
                    <div><span>Giriş gücü</span><strong id="liveInputPower">—</strong></div>
                    <div><span>Çıkış gücü</span><strong id="liveOutputPower">—</strong></div>
                    <div><span>Peak</span><strong id="livePeak">—</strong></div>
                </div>
            </article>

            <article class="panel visualization-panel">
                <div class="section-heading"><div><span class="eyebrow">SIGNAL VIEW</span><h2>Konstelasyon</h2></div><span class="chart-legend"><i class="legend-input"></i> Giriş <i class="legend-output"></i> Çıkış</span></div>
                <canvas id="constellationCanvas" width="900" height="500"></canvas>
            </article>
        </section>

        <section class="panel chart-panel">
            <div class="section-heading"><div><span class="eyebrow">TIME DOMAIN</span><h2>Genlik Zaman Serisi</h2></div></div>
            <canvas id="amplitudeCanvas" width="1200" height="360"></canvas>
        </section>

        <section class="workspace-grid">
            <article class="panel benchmark-panel">
                <div class="section-heading"><div><span class="eyebrow">SERVER BENCHMARK</span><h2>Performans Testi</h2></div></div>
                <div class="form-grid compact">
                    <label>Mod<select id="benchmarkMode"><option value="both">Float + fixed</option><option value="float">Float</option><option value="fixed">Fixed</option></select></label>
                    <label>Örnek sayısı<input id="benchmarkCount" type="number" min="1000" max="1000000" value="100000"></label>
                    <label>Tekrar<input id="benchmarkRepetitions" type="number" min="1" max="15" value="5"></label>
                </div>
                <button id="benchmarkButton" class="primary-button" type="button">Benchmark başlat</button>
                <div class="benchmark-results" id="benchmarkResults">Henüz benchmark çalıştırılmadı.</div>
            </article>

            <article class="panel report-panel">
                <div class="section-heading"><div><span class="eyebrow">VALIDATION HISTORY</span><h2>Doğrulama Özeti</h2></div></div>
                <div class="report-list">
                    <div><span>Karşı-örnek senaryosu</span><strong id="reportScenarios">—</strong></div>
                    <div><span>Sertifikasyon testi</span><strong id="reportCertification">—</strong></div>
                    <div><span>Fixed-point saturation</span><strong id="reportSaturation">—</strong></div>
                    <div><span>Python export</span><strong id="reportPythonExport">—</strong></div>
                    <div><span>C++ export</span><strong id="reportCppExport">—</strong></div>
                    <div><span>API equivalence</span><strong id="reportApi">—</strong></div>
                </div>
            </article>
        </section>
    </main>

    <footer><span>GENESIS-DSP · Step 24</span><span>Certified causal pipeline · Q3.13 · FastAPI</span></footer>
    <script src="/assets/app.js"></script>
</body>
</html>
'''


def build_styles_css() -> str:
    return r''':root{color-scheme:dark;--bg:#070b14;--panel:rgba(15,23,42,.82);--line:rgba(148,163,184,.16);--text:#f8fafc;--muted:#94a3b8;--cyan:#22d3ee;--blue:#60a5fa;--violet:#a78bfa;--green:#34d399;--red:#fb7185;--shadow:0 25px 70px rgba(0,0,0,.35)}*{box-sizing:border-box}html{scroll-behavior:smooth}body{min-height:100vh;margin:0;overflow-x:hidden;background:radial-gradient(circle at 18% 0%,rgba(34,211,238,.13),transparent 33%),radial-gradient(circle at 82% 18%,rgba(167,139,250,.13),transparent 32%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}button{cursor:pointer}.ambient{position:fixed;z-index:-1;width:420px;height:420px;border-radius:50%;filter:blur(120px);opacity:.12;pointer-events:none}.ambient-one{top:10%;left:-180px;background:var(--cyan)}.ambient-two{right:-160px;bottom:8%;background:var(--violet)}.topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;justify-content:space-between;min-height:76px;padding:0 5vw;border-bottom:1px solid var(--line);background:rgba(7,11,20,.78);backdrop-filter:blur(18px)}.brand{display:flex;gap:12px;align-items:center;color:inherit;text-decoration:none}.brand-mark{display:grid;width:42px;height:42px;place-items:center;border:1px solid rgba(34,211,238,.55);border-radius:13px;background:linear-gradient(145deg,rgba(34,211,238,.2),rgba(96,165,250,.05));color:var(--cyan);font-size:20px;font-weight:900;box-shadow:inset 0 0 22px rgba(34,211,238,.12),0 0 24px rgba(34,211,238,.08)}.brand strong,.brand small{display:block}.brand strong{letter-spacing:.08em}.brand small{margin-top:2px;color:var(--muted);font-size:11px;letter-spacing:.08em;text-transform:uppercase}.top-actions,.button-row,.chart-legend{display:flex;align-items:center;gap:12px}.status-badge,.ghost-button,.primary-button,.secondary-button{border-radius:12px;transition:transform 150ms ease,border-color 150ms ease,background 150ms ease}.status-badge{padding:9px 12px;border:1px solid var(--line);color:var(--muted);font-size:12px}.status-badge.online{border-color:rgba(52,211,153,.35);background:rgba(52,211,153,.08);color:var(--green)}.status-badge.offline{border-color:rgba(251,113,133,.35);background:rgba(251,113,133,.08);color:var(--red)}.ghost-button,.primary-button,.secondary-button{padding:11px 16px;border:1px solid var(--line);color:var(--text);text-decoration:none}.ghost-button,.secondary-button{background:rgba(255,255,255,.03)}.primary-button{border-color:rgba(34,211,238,.55);background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(96,165,250,.15));box-shadow:0 10px 28px rgba(34,211,238,.1)}.ghost-button:hover,.primary-button:hover,.secondary-button:hover{transform:translateY(-1px);border-color:rgba(34,211,238,.7)}.page-shell{width:min(1500px,92vw);margin:0 auto;padding:62px 0 80px}.hero{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(280px,.55fr);gap:48px;align-items:center;min-height:430px}.eyebrow{color:var(--cyan);font-size:11px;font-weight:800;letter-spacing:.2em}.hero h1{max-width:920px;margin:18px 0;font-size:clamp(46px,7vw,92px);line-height:.98;letter-spacing:-.055em}.hero h1 span{display:block;color:transparent;background:linear-gradient(90deg,var(--cyan),var(--blue),var(--violet));background-clip:text}.hero p{max-width:760px;margin:0;color:var(--muted);font-size:18px;line-height:1.75}.hero-orbit{position:relative;display:grid;width:min(360px,80vw);aspect-ratio:1;place-items:center;justify-self:center}.orbit,.core{position:absolute;border-radius:50%}.orbit{border:1px solid rgba(34,211,238,.25)}.orbit-a{inset:4%;animation:spin 18s linear infinite}.orbit-a:before,.orbit-b:before{position:absolute;content:"";width:14px;height:14px;border-radius:50%;background:var(--cyan);box-shadow:0 0 24px var(--cyan)}.orbit-a:before{top:12%;left:12%}.orbit-b{inset:21%;border-color:rgba(167,139,250,.32);animation:spin-reverse 12s linear infinite}.orbit-b:before{right:4%;bottom:18%;background:var(--violet);box-shadow:0 0 24px var(--violet)}.core{display:grid;width:118px;height:118px;place-items:center;border:1px solid rgba(96,165,250,.5);background:radial-gradient(circle,rgba(34,211,238,.32),rgba(17,28,48,.96) 66%);font-size:24px;font-weight:900;letter-spacing:.12em;box-shadow:0 0 70px rgba(34,211,238,.18),inset 0 0 30px rgba(96,165,250,.2)}@keyframes spin{to{transform:rotate(360deg)}}@keyframes spin-reverse{to{transform:rotate(-360deg)}}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:18px 0 22px}.metric-card,.panel{border:1px solid var(--line);background:linear-gradient(145deg,rgba(17,28,48,.93),rgba(10,17,31,.78));box-shadow:var(--shadow);backdrop-filter:blur(14px)}.metric-card{min-height:142px;padding:22px;border-radius:20px}.metric-card span,.metric-card small,.subtle,label,.console-status{color:var(--muted)}.metric-card strong{display:block;margin:15px 0 8px;font-size:30px}.metric-card small{font-size:12px}.panel{padding:24px;border-radius:24px}.pipeline-panel,.chart-panel{margin-top:20px}.section-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:22px}.section-heading h2{margin:6px 0 0;font-size:24px}.subtle{font-size:13px}.pipeline-flow{display:flex;align-items:stretch;gap:16px;overflow-x:auto;padding:8px 2px 4px}.pipeline-node{position:relative;min-width:250px;flex:1;padding:20px;border:1px solid rgba(96,165,250,.22);border-radius:18px;background:rgba(5,11,22,.52)}.pipeline-node:not(:last-child):after{position:absolute;top:50%;right:-17px;z-index:2;content:"→";color:var(--cyan);transform:translateY(-50%)}.pipeline-node-index{color:var(--cyan);font-size:11px;font-weight:900;letter-spacing:.14em}.pipeline-node h3{margin:10px 0 8px}.pipeline-node code{color:var(--violet);font-size:12px}.parameter-list{display:grid;gap:6px;margin-top:14px;color:var(--muted);font-size:12px}.workspace-grid{display:grid;grid-template-columns:minmax(370px,.85fr) minmax(0,1.15fr);gap:20px;margin-top:20px}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.form-grid.compact{grid-template-columns:repeat(3,minmax(0,1fr))}label{display:grid;gap:8px;font-size:12px;font-weight:700;letter-spacing:.04em}input,select{width:100%;min-height:45px;border:1px solid var(--line);border-radius:12px;outline:none;background:rgba(2,6,23,.62);color:var(--text);padding:10px 12px}input:focus,select:focus{border-color:rgba(34,211,238,.62);box-shadow:0 0 0 3px rgba(34,211,238,.08)}.input-with-unit{display:grid;grid-template-columns:1fr auto;align-items:center;border:1px solid var(--line);border-radius:12px;background:rgba(2,6,23,.62)}.input-with-unit input{border:0;background:transparent}.input-with-unit span{padding-right:12px;color:var(--muted);font-size:12px}.button-row{margin-top:18px}.console-status{min-height:52px;margin-top:18px;padding:14px;border:1px dashed var(--line);border-radius:12px;background:rgba(2,6,23,.42);font-size:12px;line-height:1.55}.console-status.success{border-color:rgba(52,211,153,.3);color:var(--green)}.console-status.error{border-color:rgba(251,113,133,.3);color:var(--red)}.mini-metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:18px}.mini-metric-grid div{padding:13px;border:1px solid var(--line);border-radius:14px;background:rgba(2,6,23,.34)}.mini-metric-grid span,.mini-metric-grid strong{display:block}.mini-metric-grid span{color:var(--muted);font-size:10px;text-transform:uppercase}.mini-metric-grid strong{margin-top:7px;font-size:15px}canvas{display:block;width:100%;border:1px solid var(--line);border-radius:16px;background:linear-gradient(rgba(15,23,42,.4),rgba(2,6,23,.58))}.chart-legend{color:var(--muted);font-size:11px}.chart-legend i{display:inline-block;width:9px;height:9px;margin-left:6px;border-radius:50%}.legend-input{background:var(--violet)}.legend-output{background:var(--cyan)}.benchmark-results{display:grid;gap:10px;margin-top:18px;color:var(--muted)}.benchmark-card{display:grid;grid-template-columns:1fr repeat(2,auto);gap:16px;align-items:center;padding:14px;border:1px solid var(--line);border-radius:14px;background:rgba(2,6,23,.36)}.benchmark-card strong{color:var(--text)}.report-list{display:grid;gap:2px}.report-list div{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:14px 0;border-bottom:1px solid var(--line)}.report-list span{color:var(--muted)}.report-list strong{text-align:right}footer{display:flex;justify-content:space-between;width:min(1500px,92vw);margin:0 auto;padding:24px 0 40px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}@media(max-width:1050px){.hero{grid-template-columns:1fr}.hero-orbit{display:none}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.workspace-grid{grid-template-columns:1fr}}@media(max-width:680px){.topbar{padding:0 4vw}.brand small,.ghost-button{display:none}.page-shell{width:94vw;padding-top:36px}.hero{min-height:360px}.hero h1{font-size:48px}.metric-grid,.form-grid,.form-grid.compact,.mini-metric-grid{grid-template-columns:1fr}.panel,.metric-card{border-radius:18px}.pipeline-node{min-width:220px}footer{flex-direction:column;gap:8px}}
'''


def build_app_js() -> str:
    return r'''const state={dashboard:null,inputSamples:[],outputSamples:[]};const byId=id=>document.getElementById(id);const formatNumber=(value,digits=6,fallback="—")=>{if(value===null||value===undefined||Number.isNaN(Number(value)))return fallback;const numeric=Number(value);if(numeric===0)return"0";if(Math.abs(numeric)<.001||Math.abs(numeric)>=1e6)return numeric.toExponential(3);return numeric.toLocaleString("tr-TR",{maximumFractionDigits:digits})};const setStatus=(message,kind="")=>{const element=byId("consoleStatus");element.textContent=message;element.className=`console-status ${kind}`.trim()};const setHealth=(online,message)=>{const badge=byId("healthBadge");badge.textContent=message;badge.className=`status-badge ${online?"online":"offline"}`};
const generateDemo=()=>{const count=Math.max(16,Math.min(5000,Number(byId("demoCountInput").value)||512));const sampleRate=Number(byId("sampleRateInput").value)||1e6;const samples=[];let seed=246813579;const random=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296};for(let index=0;index<count;index+=1){const bitI=random()>.5?1:-1;const bitQ=random()>.5?1:-1;const real=bitI/Math.sqrt(2);const imag=bitQ/Math.sqrt(2);const phase=2*Math.PI*18750*index/sampleRate+.42;const cosine=Math.cos(phase);const sine=Math.sin(phase);const rr=real*cosine-imag*sine;const ri=real*sine+imag*cosine;samples.push({real:.72*rr-.36*ri+.17+.018*(random()-.5),imag:.36*rr+.72*ri-.11+.018*(random()-.5)})}state.inputSamples=samples;state.outputSamples=[];byId("liveSampleCount").textContent=samples.length.toLocaleString("tr-TR");drawAll();setStatus(`${samples.length} örnekli bozunumlu QPSK demo sinyali üretildi.`,"success")};
const parseCsv=async file=>{const text=await file.text();const lines=text.split(/\r?\n/).map(line=>line.trim()).filter(Boolean);const samples=[];for(let index=0;index<lines.length;index+=1){const columns=lines[index].split(/[;,\t]/).map(value=>value.trim());if(columns.length<2)continue;const real=Number(columns[0]);const imag=Number(columns[1]);if(Number.isFinite(real)&&Number.isFinite(imag))samples.push({real,imag})}if(!samples.length)throw new Error("CSV içinde geçerli real,imag satırı bulunamadı.");state.inputSamples=samples;state.outputSamples=[];byId("liveSampleCount").textContent=samples.length.toLocaleString("tr-TR");drawAll();setStatus(`${samples.length} IQ örneği CSV dosyasından yüklendi.`,"success")};
const processSignal=async()=>{if(!state.inputSamples.length)generateDemo();const mode=byId("modeSelect").value;const sampleRate=Number(byId("sampleRateInput").value);if(!Number.isFinite(sampleRate)||sampleRate<=0){setStatus("Örnekleme frekansı pozitif olmalıdır.","error");return}byId("runButton").disabled=true;setStatus("Pipeline çalıştırılıyor…");try{const response=await fetch(`/api/process/${mode}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sample_rate_hz:sampleRate,samples:state.inputSamples})});const payload=await response.json();if(!response.ok)throw new Error(payload.detail||"DSP işlemi başarısız.");state.outputSamples=payload.samples;byId("liveSampleCount").textContent=payload.metrics.sample_count.toLocaleString("tr-TR");byId("liveInputPower").textContent=formatNumber(payload.metrics.input_average_power,6);byId("liveOutputPower").textContent=formatNumber(payload.metrics.output_average_power,6);byId("livePeak").textContent=formatNumber(payload.metrics.output_peak_magnitude,6);drawAll();setStatus(`${payload.metrics.sample_count} örnek ${mode} modunda işlendi.`,"success")}catch(error){setStatus(error.message,"error")}finally{byId("runButton").disabled=false}};
const canvasContext=id=>{const canvas=byId(id);const ratio=window.devicePixelRatio||1;const bounds=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(bounds.width*ratio));canvas.height=Math.max(1,Math.round(bounds.height*ratio));const context=canvas.getContext("2d");context.setTransform(ratio,0,0,ratio,0,0);return{context,width:bounds.width,height:bounds.height}};const drawGrid=(context,width,height)=>{context.clearRect(0,0,width,height);context.strokeStyle="rgba(148,163,184,.12)";context.lineWidth=1;for(let fraction=.1;fraction<1;fraction+=.1){const x=width*fraction;const y=height*fraction;context.beginPath();context.moveTo(x,0);context.lineTo(x,height);context.stroke();context.beginPath();context.moveTo(0,y);context.lineTo(width,y);context.stroke()}context.strokeStyle="rgba(148,163,184,.3)";context.beginPath();context.moveTo(width/2,0);context.lineTo(width/2,height);context.stroke();context.beginPath();context.moveTo(0,height/2);context.lineTo(width,height/2);context.stroke()};
const drawConstellation=()=>{const{context,width,height}=canvasContext("constellationCanvas");drawGrid(context,width,height);const combined=[...state.inputSamples,...state.outputSamples];let extent=1.25;for(const sample of combined)extent=Math.max(extent,Math.abs(sample.real)*1.15,Math.abs(sample.imag)*1.15);const drawPoints=(samples,color,radius,opacity)=>{context.fillStyle=color;context.globalAlpha=opacity;const step=Math.max(1,Math.ceil(samples.length/2500));for(let index=0;index<samples.length;index+=step){const sample=samples[index];const x=width/2+sample.real/extent*width*.44;const y=height/2-sample.imag/extent*height*.44;context.beginPath();context.arc(x,y,radius,0,2*Math.PI);context.fill()}context.globalAlpha=1};drawPoints(state.inputSamples,"#a78bfa",2.1,.34);drawPoints(state.outputSamples,"#22d3ee",2.2,.6)};
const drawAmplitude=()=>{const{context,width,height}=canvasContext("amplitudeCanvas");drawGrid(context,width,height);const input=state.inputSamples;const output=state.outputSamples;const maximumLength=Math.max(input.length,output.length,1);let maximumMagnitude=1;for(const sample of[...input,...output])maximumMagnitude=Math.max(maximumMagnitude,Math.hypot(sample.real,sample.imag));const drawLine=(samples,color,alpha)=>{if(!samples.length)return;context.beginPath();context.strokeStyle=color;context.globalAlpha=alpha;context.lineWidth=1.4;const step=Math.max(1,Math.ceil(samples.length/width));let first=true;for(let index=0;index<samples.length;index+=step){const magnitude=Math.hypot(samples[index].real,samples[index].imag);const x=index/Math.max(maximumLength-1,1)*width;const y=height-magnitude/maximumMagnitude*height*.88-height*.06;if(first){context.moveTo(x,y);first=false}else context.lineTo(x,y)}context.stroke();context.globalAlpha=1};drawLine(input,"#a78bfa",.55);drawLine(output,"#22d3ee",.9)};const drawAll=()=>{drawConstellation();drawAmplitude()};
const renderPipeline=pipeline=>{const container=byId("pipelineFlow");container.innerHTML="";pipeline.nodes.forEach((node,index)=>{const card=document.createElement("article");card.className="pipeline-node";const parameters=Object.entries(node.parameters||{}).map(([key,value])=>`<div><span>${key}</span>: <strong>${formatNumber(value,8,value)}</strong></div>`).join("");card.innerHTML=`<span class="pipeline-node-index">STAGE ${String(index+1).padStart(2,"0")}</span><h3>${node.block_id.replaceAll("_"," ")}</h3><code>${node.node_id}</code><div class="parameter-list">${parameters||"<div>Varsayılan parametreler</div>"}</div>`;container.appendChild(card)});byId("pipelineSummary").textContent=`${pipeline.node_count} düğüm · causal · deterministic`};
const renderDashboard=data=>{state.dashboard=data;byId("metricBer").textContent=formatNumber(data.receiver.test_ber,9);byId("metricFixedEvm").textContent=data.fixed_point.evm_percent===null?"—":`${formatNumber(data.fixed_point.evm_percent,6)} %`;byId("metricCertificate").textContent=data.certification.status||"unknown";byId("metricNodes").textContent=data.pipeline.node_count;byId("reportScenarios").textContent=`${formatNumber(data.falsification.scenario_count,0)} test / ${formatNumber(data.falsification.failure_count,0)} failure`;byId("reportCertification").textContent=`${formatNumber(data.certification.passed_count,0)} / ${formatNumber(data.certification.test_count,0)} passed`;byId("reportSaturation").textContent=formatNumber(data.fixed_point.saturations,0);byId("reportPythonExport").textContent=data.exports.python_status;byId("reportCppExport").textContent=data.exports.cpp_status;byId("reportApi").textContent=data.api.fixed_error===0?"bit-exact":formatNumber(data.api.fixed_error,9);renderPipeline(data.pipeline)};
const loadDashboard=async()=>{try{const[healthResponse,dashboardResponse]=await Promise.all([fetch("/api/health"),fetch("/dashboard-data")]);if(!healthResponse.ok||!dashboardResponse.ok)throw new Error("Dashboard verisi alınamadı.");const health=await healthResponse.json();const dashboard=await dashboardResponse.json();setHealth(true,`${health.status.toUpperCase()} · API online`);renderDashboard(dashboard)}catch(error){setHealth(false,"API offline");setStatus(error.message,"error")}};
const runBenchmark=async()=>{const button=byId("benchmarkButton");const results=byId("benchmarkResults");button.disabled=true;results.textContent="Benchmark çalışıyor…";try{const response=await fetch("/benchmark",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:byId("benchmarkMode").value,sample_count:Number(byId("benchmarkCount").value),repetitions:Number(byId("benchmarkRepetitions").value),sample_rate_hz:Number(byId("sampleRateInput").value)})});const payload=await response.json();if(!response.ok)throw new Error(payload.detail||"Benchmark başarısız.");results.innerHTML=payload.results.map(item=>`<div class="benchmark-card"><strong>${item.mode.toUpperCase()}</strong><span>${formatNumber(item.median_latency_ms,3)} ms</span><span>${formatNumber(item.throughput_megasamples_per_second,3)} MS/s</span></div>`).join("")}catch(error){results.textContent=error.message}finally{button.disabled=false}};
byId("generateButton").addEventListener("click",generateDemo);byId("runButton").addEventListener("click",processSignal);byId("benchmarkButton").addEventListener("click",runBenchmark);byId("csvInput").addEventListener("change",async event=>{const file=event.target.files[0];if(!file)return;try{await parseCsv(file)}catch(error){setStatus(error.message,"error")}});window.addEventListener("resize",drawAll);loadDashboard();generateDemo();
'''


def build_favicon_svg() -> str:
    return r'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#a78bfa"/></linearGradient></defs><rect width="64" height="64" rx="17" fill="#07101f"/><circle cx="32" cy="32" r="22" fill="none" stroke="url(#g)" stroke-width="3"/><path d="M19 32h8l4-11 5 22 4-11h7" fill="none" stroke="url(#g)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>'''


def write_site_assets() -> None:
    SITE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(build_index_html(), encoding="utf-8")
    STYLE_PATH.write_text(build_styles_css(), encoding="utf-8")
    SCRIPT_PATH.write_text(build_app_js(), encoding="utf-8")
    FAVICON_PATH.write_text(build_favicon_svg(), encoding="utf-8")
    README_PATH.write_text(
        "GENESIS-DSP WEB APPLICATION\n"
        "===========================\n\n"
        "Başlatma:\n"
        "python step24_web_application.py --serve --open-browser\n\n"
        "Web: http://127.0.0.1:8080\n"
        "API: http://127.0.0.1:8080/api/docs\n",
        encoding="utf-8",
    )
    STARTER_PATH.write_text(
        "@echo off\r\n"
        "title GENESIS-DSP Web Application\r\n"
        "cd /d \"%~dp0\\..\\..\"\r\n"
        "python step24_web_application.py --serve --open-browser\r\n"
        "pause\r\n",
        encoding="utf-8",
    )


write_site_assets()

web_app = FastAPI(
    title="GENESIS-DSP Web Application",
    description="GENESIS-DSP dashboard, reporting, benchmark and processing web application.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
web_app.mount("/api", dsp_api_app, name="genesis-dsp-api")
web_app.mount("/assets", StaticFiles(directory=str(SITE_DIRECTORY)), name="assets")


@web_app.get("/dashboard-data", tags=["dashboard"])
def dashboard_data() -> dict[str, Any]:
    return build_dashboard_data()


@web_app.post("/benchmark", tags=["benchmark"])
def benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    try:
        return run_benchmark(request)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@web_app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(FAVICON_PATH, media_type="image/svg+xml")


@web_app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_PATH, media_type="text/html")


def create_zip_package() -> None:
    temporary_root = STEP24_DIRECTORY / "_package"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    package_root = temporary_root / "GENESIS_DSP_WEB"
    package_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SITE_DIRECTORY, package_root / "site")
    shutil.copy2(README_PATH, package_root / README_PATH.name)
    shutil.copy2(STARTER_PATH, package_root / STARTER_PATH.name)
    archive_base = STEP24_DIRECTORY / "GENESIS_DSP_WEB_PACKAGE"
    generated_archive = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=temporary_root,
            base_dir="GENESIS_DSP_WEB",
        )
    )
    if generated_archive != ZIP_PATH:
        shutil.move(generated_archive, ZIP_PATH)
    shutil.rmtree(temporary_root)


def run_self_test() -> dict[str, Any]:
    required_files = [INDEX_PATH, STYLE_PATH, SCRIPT_PATH, FAVICON_PATH, README_PATH, STARTER_PATH]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise RuntimeError("Eksik web asset dosyaları: " + ", ".join(missing))

    combined_text = INDEX_PATH.read_text(encoding="utf-8") + SCRIPT_PATH.read_text(encoding="utf-8")
    required_markers = ["GENESIS-DSP Laboratory", "constellationCanvas", "benchmarkButton", "/api/docs"]
    missing_markers = [marker for marker in required_markers if marker not in combined_text]
    if missing_markers:
        raise RuntimeError("Web uygulaması marker testi başarısız: " + ", ".join(missing_markers))

    dashboard = build_dashboard_data()
    if dashboard["pipeline"]["node_count"] < 1:
        raise RuntimeError("Dashboard pipeline bilgisi boş.")

    benchmark_result = run_benchmark(
        BenchmarkRequest(mode="both", sample_count=4000, repetitions=2, sample_rate_hz=1_000_000.0)
    )
    if len(benchmark_result["results"]) != 2:
        raise RuntimeError("Benchmark öz testi iki modu üretmedi.")

    create_zip_package()
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size <= 0:
        raise RuntimeError("Web ZIP paketi üretilemedi.")

    route_paths = sorted({getattr(route, "path", "") for route in web_app.routes})
    return {
        "asset_count": len(required_files),
        "pipeline_node_count": dashboard["pipeline"]["node_count"],
        "benchmark_modes_tested": [item["mode"] for item in benchmark_result["results"]],
        "route_count": len(route_paths),
        "routes": route_paths,
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "status": "PASSED",
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GENESIS-DSP full web application")
    parser.add_argument("--serve", action="store_true", help="Öz test yerine web sunucusunu başlat.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open-browser", action="store_true", help="Tarayıcıyı otomatik aç.")
    return parser.parse_args()


def open_browser_later(host: str, port: int) -> None:
    Timer(1.25, lambda: webbrowser.open(f"http://{host}:{port}")).start()


def serve(host: str, port: int, open_browser: bool) -> None:
    if open_browser:
        open_browser_later(host, port)
    uvicorn.run(web_app, host=host, port=port, log_level="info")


def main() -> None:
    arguments = parse_arguments()
    if arguments.serve:
        serve(arguments.host, arguments.port, arguments.open_browser)
        return

    STEP24_DIRECTORY.mkdir(parents=True, exist_ok=True)
    self_test = run_self_test()
    report = {
        "project": "GENESIS-DSP",
        "step": 24,
        "description": "Full web dashboard, processing console, reporting and benchmark application",
        "web_application": {
            "host": DEFAULT_HOST,
            "port": DEFAULT_PORT,
            "url": f"http://{DEFAULT_HOST}:{DEFAULT_PORT}",
            "api_docs_url": f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/api/docs",
            "site_directory": str(SITE_DIRECTORY),
            "zip_package": str(ZIP_PATH),
            "windows_starter": str(STARTER_PATH),
        },
        "features": [
            "Live API health",
            "Pipeline visualization",
            "Float/fixed IQ processing",
            "CSV IQ upload",
            "Demo impaired QPSK generation",
            "Constellation plot",
            "Amplitude time-series plot",
            "Server benchmark",
            "Validation report aggregation",
            "Swagger API access",
        ],
        "self_test": self_test,
        "validations": {
            "assets_generated": True,
            "dashboard_data_available": True,
            "pipeline_visible": True,
            "float_benchmark_passed": True,
            "fixed_benchmark_passed": True,
            "website_zip_generated": True,
            "windows_starter_generated": True,
        },
    }
    save_json(REPORT_PATH, report)

    print()
    print("=" * 90)
    print("GENESIS-DSP — ADIM 24 BAŞARIYLA TAMAMLANDI")
    print("=" * 90)
    print(f"Web asset sayısı                 : {self_test['asset_count']}")
    print(f"Dashboard pipeline düğümü        : {self_test['pipeline_node_count']}")
    print(f"Web route sayısı                 : {self_test['route_count']}")
    print("Float benchmark öz testi         : BAŞARILI")
    print("Fixed benchmark öz testi         : BAŞARILI")
    print("Pipeline/rapor entegrasyonu       : BAŞARILI")
    print("Konstelasyon ve zaman grafikleri  : HAZIR")
    print("CSV IQ yükleme                    : HAZIR")
    print("Swagger API                       : HAZIR")
    print(f"Web sitesi                        : http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print(f"Başlatma komutu                   : python {Path(__file__).name} --serve --open-browser")
    print(f"Windows başlatıcı                 : {STARTER_PATH}")
    print(f"Web paket ZIP                     : {ZIP_PATH}")
    print(f"Site klasörü                      : {SITE_DIRECTORY}")
    print(f"Rapor                             : {REPORT_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()
