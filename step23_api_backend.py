"""
GENESIS-DSP — Adım 23
FastAPI tabanlı DSP servis katmanı.

Bu program:
1. Adım 22 bağımsız Python DSP export'unu dinamik olarak yükler.
2. Floating-point ve fixed-point işleme için FastAPI endpoint'leri oluşturur.
3. Health, pipeline bilgisi ve OpenAPI şeması sunar.
4. JSON IQ örneklerini doğrular ve işler.
5. Varsayılan çalıştırmada API öz testini yapıp rapor üretir.
6. --serve seçeneğiyle gerçek HTTP sunucusunu başlatır.

Kurulum:
    pip install fastapi uvicorn pydantic

Öz test:
    python step23_api_backend.py

Sunucuyu başlat:
    python step23_api_backend.py --serve

Tarayıcı:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as error:
    raise RuntimeError(
        "Adım 23 için FastAPI bağımlılıkları eksik.\n"
        "Şu komutu çalıştır:\n"
        "pip install fastapi uvicorn pydantic"
    ) from error


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP21_DIRECTORY = BASE_DIRECTORY / "outputs" / "step21"
STEP22_DIRECTORY = BASE_DIRECTORY / "outputs" / "step22"
STEP23_DIRECTORY = BASE_DIRECTORY / "outputs" / "step23"

GENERATED_DSP_PATH = (
    STEP22_DIRECTORY / "genesis_dsp_export.py"
)
MANIFEST_PATH = (
    STEP22_DIRECTORY / "export_manifest.json"
)
REFERENCE_SIGNALS_PATH = (
    STEP21_DIRECTORY / "fixed_point_signals.npz"
)

OPENAPI_PATH = (
    STEP23_DIRECTORY / "openapi_schema.json"
)
REPORT_PATH = (
    STEP23_DIRECTORY / "api_backend_report.json"
)
EXAMPLE_REQUEST_PATH = (
    STEP23_DIRECTORY / "example_request.json"
)
EXAMPLE_RESPONSE_PATH = (
    STEP23_DIRECTORY / "example_response.json"
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_SAMPLES_PER_REQUEST = 200_000
SELF_TEST_SAMPLE_COUNT = 512
FLOAT_TOLERANCE = 1e-12
FIXED_TOLERANCE = 0.0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON dosyası bulunamadı: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise TypeError(
            f"JSON kökü dict olmalıdır: {path}"
        )

    return document


def save_json(
    path: Path,
    document: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            document,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_generated_dsp() -> Any:
    if not GENERATED_DSP_PATH.exists():
        raise FileNotFoundError(
            f"Adım 22 Python export bulunamadı: "
            f"{GENERATED_DSP_PATH}\n"
            "Önce şu komutu çalıştır:\n"
            "python step22_code_export.py"
        )

    specification = importlib.util.spec_from_file_location(
        "genesis_dsp_step22_export",
        GENERATED_DSP_PATH,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            "Adım 22 DSP modülü yüklenemedi."
        )

    module = importlib.util.module_from_spec(
        specification
    )
    sys.modules[specification.name] = module
    specification.loader.exec_module(
        module
    )

    for required_name in (
        "PIPELINE",
        "process_float",
        "process_fixed",
    ):
        if not hasattr(
            module,
            required_name,
        ):
            raise AttributeError(
                f"Generated DSP modülünde eksik alan: "
                f"{required_name}"
            )

    return module


DSP_MODULE = load_generated_dsp()


class IQSample(BaseModel):
    real: float = Field(
        ...,
        description="IQ örneğinin gerçek bileşeni.",
    )
    imag: float = Field(
        ...,
        description="IQ örneğinin sanal bileşeni.",
    )


class ProcessRequest(BaseModel):
    sample_rate_hz: float = Field(
        ...,
        gt=0.0,
        description="Örnekleme frekansı.",
    )
    samples: list[IQSample] = Field(
        ...,
        min_length=1,
        max_length=MAX_SAMPLES_PER_REQUEST,
        description="İşlenecek kompleks IQ örnekleri.",
    )


class ProcessMetrics(BaseModel):
    sample_count: int
    input_average_power: float
    output_average_power: float
    output_peak_magnitude: float


class ProcessResponse(BaseModel):
    mode: Literal["float", "fixed"]
    sample_rate_hz: float
    samples: list[IQSample]
    metrics: ProcessMetrics


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    dsp_export_sha256: str


class PipelineResponse(BaseModel):
    node_count: int
    nodes: list[dict[str, Any]]
    fixed_point: dict[str, int]


app = FastAPI(
    title="GENESIS-DSP API",
    description=(
        "Sertifikalı GENESIS-DSP pipeline'ı için "
        "floating-point ve fixed-point işleme servisi."
    ),
    version="1.0.0",
)


def request_to_numpy(
    request: ProcessRequest,
) -> np.ndarray:
    samples = np.asarray(
        [
            complex(
                item.real,
                item.imag,
            )
            for item in request.samples
        ],
        dtype=np.complex128,
    )

    if samples.ndim != 1:
        raise ValueError(
            "IQ örnekleri tek boyutlu olmalıdır."
        )

    if not np.all(
        np.isfinite(samples.real)
    ):
        raise ValueError(
            "Gerçek bileşenler sonlu olmalıdır."
        )

    if not np.all(
        np.isfinite(samples.imag)
    ):
        raise ValueError(
            "Sanal bileşenler sonlu olmalıdır."
        )

    return samples


def numpy_to_iq(
    samples: np.ndarray,
) -> list[IQSample]:
    return [
        IQSample(
            real=float(value.real),
            imag=float(value.imag),
        )
        for value in samples
    ]


def calculate_metrics(
    input_samples: np.ndarray,
    output_samples: np.ndarray,
) -> ProcessMetrics:
    return ProcessMetrics(
        sample_count=int(
            len(output_samples)
        ),
        input_average_power=float(
            np.mean(
                np.abs(input_samples) ** 2
            )
        ),
        output_average_power=float(
            np.mean(
                np.abs(output_samples) ** 2
            )
        ),
        output_peak_magnitude=float(
            np.max(
                np.abs(output_samples)
            )
        ),
    )


def process_request(
    request: ProcessRequest,
    mode: Literal["float", "fixed"],
) -> ProcessResponse:
    try:
        input_samples = request_to_numpy(
            request
        )

        if mode == "float":
            output_samples = (
                DSP_MODULE.process_float(
                    input_samples,
                    request.sample_rate_hz,
                )
            )
        elif mode == "fixed":
            output_samples = (
                DSP_MODULE.process_fixed(
                    input_samples,
                    request.sample_rate_hz,
                )
            )
        else:
            raise ValueError(
                f"Desteklenmeyen mod: {mode}"
            )

        output_samples = np.asarray(
            output_samples,
            dtype=np.complex128,
        )

        if output_samples.shape != input_samples.shape:
            raise RuntimeError(
                "DSP çıkış boyutu giriş boyutuyla eşleşmiyor."
            )

        if not np.all(
            np.isfinite(output_samples.real)
        ) or not np.all(
            np.isfinite(output_samples.imag)
        ):
            raise RuntimeError(
                "DSP çıkışında NaN veya Inf bulundu."
            )

        return ProcessResponse(
            mode=mode,
            sample_rate_hz=float(
                request.sample_rate_hz
            ),
            samples=numpy_to_iq(
                output_samples
            ),
            metrics=calculate_metrics(
                input_samples,
                output_samples,
            ),
        )

    except (
        ValueError,
        TypeError,
        RuntimeError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="GENESIS-DSP API",
        version="1.0.0",
        dsp_export_sha256=sha256_file(
            GENERATED_DSP_PATH
        ),
    )


@app.get(
    "/pipeline",
    response_model=PipelineResponse,
    tags=["system"],
)
def pipeline_info() -> PipelineResponse:
    return PipelineResponse(
        node_count=len(
            DSP_MODULE.PIPELINE
        ),
        nodes=list(
            DSP_MODULE.PIPELINE
        ),
        fixed_point={
            "total_bits": int(
                DSP_MODULE.TOTAL_BITS
            ),
            "integer_bits": int(
                DSP_MODULE.INTEGER_BITS
            ),
            "fractional_bits": int(
                DSP_MODULE.FRACTIONAL_BITS
            ),
            "phase_bits": int(
                DSP_MODULE.PHASE_BITS
            ),
        },
    )


@app.post(
    "/process/float",
    response_model=ProcessResponse,
    tags=["processing"],
)
def process_float(
    request: ProcessRequest,
) -> ProcessResponse:
    return process_request(
        request=request,
        mode="float",
    )


@app.post(
    "/process/fixed",
    response_model=ProcessResponse,
    tags=["processing"],
)
def process_fixed(
    request: ProcessRequest,
) -> ProcessResponse:
    return process_request(
        request=request,
        mode="fixed",
    )


def maximum_error(
    reference: np.ndarray,
    measured: np.ndarray,
) -> float:
    if reference.shape != measured.shape:
        raise ValueError(
            "Karşılaştırma boyutları eşleşmiyor."
        )

    return float(
        np.max(
            np.abs(
                measured - reference
            )
        )
    )


def response_to_numpy(
    response: ProcessResponse,
) -> np.ndarray:
    return np.asarray(
        [
            complex(
                item.real,
                item.imag,
            )
            for item in response.samples
        ],
        dtype=np.complex128,
    )


def run_self_test() -> dict[str, Any]:
    if not REFERENCE_SIGNALS_PATH.exists():
        raise FileNotFoundError(
            f"Adım 21 referans sinyalleri bulunamadı: "
            f"{REFERENCE_SIGNALS_PATH}"
        )

    with np.load(
        REFERENCE_SIGNALS_PATH,
        allow_pickle=False,
    ) as package:
        input_samples = package[
            "input_samples"
        ].astype(np.complex128)[
            :SELF_TEST_SAMPLE_COUNT
        ]
        sample_rate_hz = float(
            package["sample_rate_hz"]
        )

    request = ProcessRequest(
        sample_rate_hz=sample_rate_hz,
        samples=[
            IQSample(
                real=float(value.real),
                imag=float(value.imag),
            )
            for value in input_samples
        ],
    )

    float_response = process_float(
        request
    )
    fixed_response = process_fixed(
        request
    )

    float_output = response_to_numpy(
        float_response
    )
    fixed_output = response_to_numpy(
        fixed_response
    )

    expected_float = (
        DSP_MODULE.process_float(
            input_samples,
            sample_rate_hz,
        )
    )
    expected_fixed = (
        DSP_MODULE.process_fixed(
            input_samples,
            sample_rate_hz,
        )
    )

    float_error = maximum_error(
        expected_float,
        float_output,
    )
    fixed_error = maximum_error(
        expected_fixed,
        fixed_output,
    )

    repeated_fixed = response_to_numpy(
        process_fixed(request)
    )
    determinism_error = maximum_error(
        fixed_output,
        repeated_fixed,
    )

    health_response = health()
    pipeline_response = pipeline_info()

    if health_response.status != "ok":
        raise RuntimeError(
            "Health endpoint öz testi başarısız."
        )

    if (
        pipeline_response.node_count
        != len(DSP_MODULE.PIPELINE)
    ):
        raise RuntimeError(
            "Pipeline endpoint node sayısı hatalı."
        )

    if float_error > FLOAT_TOLERANCE:
        raise RuntimeError(
            "Float endpoint referansla eşleşmedi."
        )

    if fixed_error > FIXED_TOLERANCE:
        raise RuntimeError(
            "Fixed endpoint referansla bit-exact eşleşmedi."
        )

    if determinism_error != 0.0:
        raise RuntimeError(
            "API fixed endpoint deterministik değil."
        )

    invalid_request_rejected = False

    try:
        ProcessRequest(
            sample_rate_hz=-1.0,
            samples=[
                IQSample(
                    real=1.0,
                    imag=0.0,
                )
            ],
        )
    except Exception:
        invalid_request_rejected = True

    if not invalid_request_rejected:
        raise RuntimeError(
            "Geçersiz sample rate reddedilmedi."
        )

    example_request = request.model_dump()
    example_response = fixed_response.model_dump()

    save_json(
        EXAMPLE_REQUEST_PATH,
        example_request,
    )
    save_json(
        EXAMPLE_RESPONSE_PATH,
        example_response,
    )
    save_json(
        OPENAPI_PATH,
        app.openapi(),
    )

    return {
        "sample_count": len(
            input_samples
        ),
        "float_maximum_error": (
            float_error
        ),
        "fixed_maximum_error": (
            fixed_error
        ),
        "determinism_maximum_error": (
            determinism_error
        ),
        "invalid_request_rejected": (
            invalid_request_rejected
        ),
        "health_status": (
            health_response.status
        ),
        "pipeline_node_count": (
            pipeline_response.node_count
        ),
        "endpoint_count": len(
            [
                route
                for route in app.routes
                if hasattr(
                    route,
                    "methods",
                )
            ]
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GENESIS-DSP FastAPI backend"
        )
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Öz test yerine uvicorn sunucusunu başlat."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
    )

    return parser.parse_args()


def run_server(
    host: str,
    port: int,
) -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "Uvicorn eksik.\n"
            "Şu komutu çalıştır:\n"
            "pip install uvicorn"
        ) from error

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


def main() -> None:
    arguments = parse_arguments()

    if arguments.serve:
        run_server(
            host=arguments.host,
            port=arguments.port,
        )
        return

    STEP23_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    self_test = run_self_test()
    manifest = (
        load_json(MANIFEST_PATH)
        if MANIFEST_PATH.exists()
        else None
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 23,
        "description": (
            "FastAPI backend for floating-point "
            "and fixed-point DSP processing"
        ),
        "service": {
            "title": app.title,
            "version": app.version,
            "default_host": DEFAULT_HOST,
            "default_port": DEFAULT_PORT,
            "documentation_url": (
                f"http://{DEFAULT_HOST}:"
                f"{DEFAULT_PORT}/docs"
            ),
            "maximum_samples_per_request": (
                MAX_SAMPLES_PER_REQUEST
            ),
        },
        "source_export": {
            "path": str(
                GENERATED_DSP_PATH
            ),
            "sha256": sha256_file(
                GENERATED_DSP_PATH
            ),
            "manifest_available": (
                manifest is not None
            ),
        },
        "self_test": self_test,
        "endpoints": [
            {
                "method": "GET",
                "path": "/health",
            },
            {
                "method": "GET",
                "path": "/pipeline",
            },
            {
                "method": "POST",
                "path": "/process/float",
            },
            {
                "method": "POST",
                "path": "/process/fixed",
            },
        ],
        "validations": {
            "health_endpoint": True,
            "pipeline_endpoint": True,
            "float_endpoint_equivalence": True,
            "fixed_endpoint_bit_exact_equivalence": True,
            "deterministic_fixed_endpoint": True,
            "invalid_request_rejection": True,
            "openapi_schema_generated": True,
        },
    }

    save_json(
        REPORT_PATH,
        report,
    )

    print()
    print("=" * 88)
    print(
        "GENESIS-DSP — ADIM 23 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 88)
    print(
        f"API endpoint sayısı              : "
        f"{self_test['endpoint_count']}"
    )
    print(
        f"Öz test örnek sayısı             : "
        f"{self_test['sample_count']}"
    )
    print(
        f"Float endpoint maksimum hata     : "
        f"{self_test['float_maximum_error']:.3e}"
    )
    print(
        f"Fixed endpoint maksimum hata     : "
        f"{self_test['fixed_maximum_error']:.3e}"
    )
    print(
        f"Determinism maksimum hatası      : "
        f"{self_test['determinism_maximum_error']:.3e}"
    )
    print(
        "Geçersiz istek koruması          : BAŞARILI"
    )
    print(
        "OpenAPI şeması                   : BAŞARILI"
    )
    print(
        f"Sunucu komutu                    : "
        f"python {Path(__file__).name} --serve"
    )
    print(
        f"Swagger arayüzü                  : "
        f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/docs"
    )
    print(
        f"OpenAPI dosyası                  : "
        f"{OPENAPI_PATH}"
    )
    print(
        f"Örnek istek                      : "
        f"{EXAMPLE_REQUEST_PATH}"
    )
    print(
        f"Örnek yanıt                      : "
        f"{EXAMPLE_RESPONSE_PATH}"
    )
    print(
        f"Rapor                            : "
        f"{REPORT_PATH}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
