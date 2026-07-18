"""
GENESIS-DSP — Adım 12
Pipeline JSON kayıt, yükleme ve bütünlük doğrulama sistemi.

Bu program:
1. Adım 11 pipeline graph yapılandırmasını yükler.
2. Graph'ı sürümlü bir GENESIS-DSP pipeline paketine dönüştürür.
3. Canonical JSON üzerinden SHA-256 bütünlük özeti üretir.
4. Paketi JSON dosyasına kaydeder ve yeniden yükler.
5. Orijinal ve yeniden yüklenen graph çıktılarını karşılaştırır.
6. Değiştirilmiş paketleri hash kontrolüyle reddeder.
7. Round-trip ve execution raporları üretir.

Çalıştırma:
    python step12_pipeline_serialization.py
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from step09_dsp_block_interface import SignalFrame
from step10_block_registry import build_default_registry
from step11_pipeline_graph import PipelineGraph


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP11_DIRECTORY = BASE_DIRECTORY / "outputs" / "step11"
STEP12_DIRECTORY = BASE_DIRECTORY / "outputs" / "step12"

INPUT_GRAPH_CONFIG = (
    STEP11_DIRECTORY / "pipeline_graph_config.json"
)

PACKAGE_PATH = (
    STEP12_DIRECTORY / "pipeline_package.json"
)
ROUNDTRIP_GRAPH_PATH = (
    STEP12_DIRECTORY / "pipeline_graph_roundtrip.json"
)
REPORT_PATH = (
    STEP12_DIRECTORY / "pipeline_serialization_report.json"
)

PACKAGE_SCHEMA_NAME = "GENESIS-DSP PipelinePackage"
PACKAGE_SCHEMA_VERSION = "1.0.0"


def canonical_json_bytes(
    document: dict[str, Any],
) -> bytes:
    """
    Aynı içerik için her zaman aynı byte dizisini üretir.

    Hash hesabı girinti, key sırası ve platform farklarından etkilenmez.
    """

    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return encoded.encode("utf-8")


def sha256_document(
    document: dict[str, Any],
) -> str:
    """Bir JSON belgesinin canonical SHA-256 özetini üretir."""

    return hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    """JSON dosyasını dict olarak yükler."""

    if not path.exists():
        raise FileNotFoundError(
            f"JSON dosyası bulunamadı: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
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
    """JSON belgesini okunabilir biçimde kaydeder."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            document,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )


class PipelinePackage:
    """Pipeline graph için sürümlü ve hash korumalı JSON paketleyici."""

    @staticmethod
    def create_document(
        graph: PipelineGraph,
        pipeline_id: str,
        description: str,
    ) -> dict[str, Any]:
        if not pipeline_id.strip():
            raise ValueError(
                "pipeline_id boş olamaz."
            )

        graph_document = graph.to_config()
        graph_sha256 = sha256_document(
            graph_document
        )

        return {
            "schema_name": PACKAGE_SCHEMA_NAME,
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "pipeline_id": pipeline_id,
            "description": description,
            "integrity": {
                "algorithm": "SHA-256",
                "canonicalization": (
                    "JSON sort_keys compact UTF-8"
                ),
                "graph_sha256": graph_sha256,
            },
            "graph": graph_document,
        }

    @staticmethod
    def validate_document(
        document: dict[str, Any],
    ) -> None:
        allowed_keys = {
            "schema_name",
            "schema_version",
            "pipeline_id",
            "description",
            "integrity",
            "graph",
        }

        unknown_keys = (
            set(document) - allowed_keys
        )

        if unknown_keys:
            raise ValueError(
                "Bilinmeyen package alanları: "
                + ", ".join(
                    sorted(unknown_keys)
                )
            )

        required_keys = allowed_keys

        missing_keys = (
            required_keys - set(document)
        )

        if missing_keys:
            raise ValueError(
                "Eksik package alanları: "
                + ", ".join(
                    sorted(missing_keys)
                )
            )

        if (
            document["schema_name"]
            != PACKAGE_SCHEMA_NAME
        ):
            raise ValueError(
                "Desteklenmeyen package schema_name."
            )

        if (
            document["schema_version"]
            != PACKAGE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Desteklenmeyen package schema_version."
            )

        if not isinstance(
            document["pipeline_id"],
            str,
        ) or not document["pipeline_id"].strip():
            raise ValueError(
                "pipeline_id boş olmayan string olmalıdır."
            )

        if not isinstance(
            document["description"],
            str,
        ):
            raise TypeError(
                "description string olmalıdır."
            )

        integrity = document["integrity"]

        if not isinstance(integrity, dict):
            raise TypeError(
                "integrity dict olmalıdır."
            )

        required_integrity_keys = {
            "algorithm",
            "canonicalization",
            "graph_sha256",
        }

        if (
            set(integrity)
            != required_integrity_keys
        ):
            raise ValueError(
                "integrity alanları geçersiz."
            )

        if integrity["algorithm"] != "SHA-256":
            raise ValueError(
                "Yalnızca SHA-256 desteklenir."
            )

        graph_document = document["graph"]

        if not isinstance(
            graph_document,
            dict,
        ):
            raise TypeError(
                "graph alanı dict olmalıdır."
            )

        stored_hash = integrity["graph_sha256"]

        if not isinstance(
            stored_hash,
            str,
        ) or len(stored_hash) != 64:
            raise ValueError(
                "graph_sha256 geçersiz."
            )

        calculated_hash = sha256_document(
            graph_document
        )

        if calculated_hash != stored_hash:
            raise ValueError(
                "Pipeline package bütünlük kontrolü başarısız: "
                "graph SHA-256 eşleşmiyor."
            )

        PipelineGraph.from_config(
            graph_document
        )

    @classmethod
    def save(
        cls,
        path: Path,
        graph: PipelineGraph,
        pipeline_id: str,
        description: str,
    ) -> dict[str, Any]:
        document = cls.create_document(
            graph=graph,
            pipeline_id=pipeline_id,
            description=description,
        )

        cls.validate_document(
            document
        )
        save_json(
            path,
            document,
        )

        return document

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> tuple[PipelineGraph, dict[str, Any]]:
        document = load_json(
            path
        )
        cls.validate_document(
            document
        )

        graph = PipelineGraph.from_config(
            document["graph"]
        )

        return graph, document


def build_test_signal() -> SignalFrame:
    """Round-trip execution karşılaştırması için deterministik test sinyali."""

    sample_rate_hz = 1_000_000.0
    number_of_samples = 16_384
    tone_frequency_hz = 62_500.0
    injected_dc = complex(
        0.18,
        -0.09,
    )

    indices = np.arange(
        number_of_samples,
        dtype=np.float64,
    )

    tone = np.exp(
        1j
        * 2.0
        * np.pi
        * tone_frequency_hz
        * indices
        / sample_rate_hz
    )

    frame = SignalFrame(
        samples=(
            tone + injected_dc
        ).astype(np.complex128),
        sample_rate_hz=sample_rate_hz,
        metadata={
            "source": (
                "step12_roundtrip_test"
            ),
            "tone_frequency_hz": (
                tone_frequency_hz
            ),
        },
    )
    frame.validate()

    return frame


def compare_graph_executions(
    original_graph: PipelineGraph,
    loaded_graph: PipelineGraph,
) -> dict[str, Any]:
    """İki graph'ın aynı girişte aynı çıktıları verdiğini doğrular."""

    registry = build_default_registry()
    input_frame = build_test_signal()

    original_result = original_graph.execute(
        registry=registry,
        input_frame=input_frame,
    )

    loaded_result = loaded_graph.execute(
        registry=registry,
        input_frame=input_frame,
    )

    if (
        original_result.topological_order
        != loaded_result.topological_order
    ):
        raise RuntimeError(
            "Round-trip sonrasında topological order değişti."
        )

    if (
        original_result.root_nodes
        != loaded_result.root_nodes
    ):
        raise RuntimeError(
            "Round-trip sonrasında root nodes değişti."
        )

    if (
        original_result.leaf_nodes
        != loaded_result.leaf_nodes
    ):
        raise RuntimeError(
            "Round-trip sonrasında leaf nodes değişti."
        )

    node_errors: dict[str, float] = {}

    for node_id in (
        original_result.topological_order
    ):
        original_samples = (
            original_result.output(
                node_id
            ).samples
        )
        loaded_samples = (
            loaded_result.output(
                node_id
            ).samples
        )

        if (
            original_samples.shape
            != loaded_samples.shape
        ):
            raise RuntimeError(
                f"'{node_id}' çıktı boyutu değişti."
            )

        maximum_error = float(
            np.max(
                np.abs(
                    original_samples
                    - loaded_samples
                )
            )
        )

        node_errors[node_id] = (
            maximum_error
        )

        if maximum_error > 1e-12:
            raise RuntimeError(
                f"'{node_id}' round-trip çıktısı değişti."
            )

    return {
        "status": "PASSED",
        "topological_order": (
            original_result.topological_order
        ),
        "root_nodes": (
            original_result.root_nodes
        ),
        "leaf_nodes": (
            original_result.leaf_nodes
        ),
        "node_maximum_errors": (
            node_errors
        ),
        "global_maximum_error": max(
            node_errors.values(),
            default=0.0,
        ),
    }


def run_tamper_detection_test(
    valid_document: dict[str, Any],
) -> dict[str, Any]:
    """Graph değiştirilip hash güncellenmezse paketin reddedildiğini doğrular."""

    tampered = copy.deepcopy(
        valid_document
    )

    first_node = tampered["graph"]["nodes"][0]

    parameters = first_node.setdefault(
        "parameters",
        {},
    )
    parameters["tampered_parameter"] = 1

    try:
        PipelinePackage.validate_document(
            tampered
        )
    except ValueError as error:
        if "sha-256" not in str(error).lower():
            raise

        return {
            "status": "PASSED",
            "rejected_reason": str(error),
        }

    raise RuntimeError(
        "Değiştirilmiş pipeline package reddedilmedi."
    )


def compare_documents(
    original_graph: PipelineGraph,
    loaded_graph: PipelineGraph,
    package_document: dict[str, Any],
) -> dict[str, Any]:
    """Graph config ve hash round-trip eşitliğini doğrular."""

    original_config = (
        original_graph.to_config()
    )
    loaded_config = (
        loaded_graph.to_config()
    )

    if original_config != loaded_config:
        raise RuntimeError(
            "Graph config round-trip sonrasında değişti."
        )

    original_hash = sha256_document(
        original_config
    )
    loaded_hash = sha256_document(
        loaded_config
    )
    stored_hash = package_document[
        "integrity"
    ]["graph_sha256"]

    if not (
        original_hash
        == loaded_hash
        == stored_hash
    ):
        raise RuntimeError(
            "Graph hash round-trip sonrasında değişti."
        )

    return {
        "status": "PASSED",
        "original_graph_sha256": (
            original_hash
        ),
        "loaded_graph_sha256": (
            loaded_hash
        ),
        "stored_graph_sha256": (
            stored_hash
        ),
        "exact_config_equality": True,
    }


def main() -> None:
    STEP12_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_GRAPH_CONFIG.exists():
        raise FileNotFoundError(
            f"Adım 11 graph config bulunamadı: "
            f"{INPUT_GRAPH_CONFIG}\n"
            "Önce şu komutu çalıştır:\n"
            "python step11_pipeline_graph.py"
        )

    source_graph_document = load_json(
        INPUT_GRAPH_CONFIG
    )

    original_graph = (
        PipelineGraph.from_config(
            source_graph_document
        )
    )

    package_document = PipelinePackage.save(
        path=PACKAGE_PATH,
        graph=original_graph,
        pipeline_id="step11-self-test-pipeline",
        description=(
            "Adım 11 graph motorundan alınan "
            "round-trip doğrulama pipeline'ı."
        ),
    )

    loaded_graph, loaded_document = (
        PipelinePackage.load(
            PACKAGE_PATH
        )
    )

    document_test = compare_documents(
        original_graph=original_graph,
        loaded_graph=loaded_graph,
        package_document=loaded_document,
    )

    execution_test = compare_graph_executions(
        original_graph=original_graph,
        loaded_graph=loaded_graph,
    )

    tamper_test = run_tamper_detection_test(
        valid_document=package_document
    )

    save_json(
        ROUNDTRIP_GRAPH_PATH,
        loaded_graph.to_config(),
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 12,
        "description": (
            "Versioned pipeline JSON serialization, "
            "loading and SHA-256 integrity verification"
        ),
        "package": {
            "path": str(
                PACKAGE_PATH
            ),
            "schema_name": (
                loaded_document[
                    "schema_name"
                ]
            ),
            "schema_version": (
                loaded_document[
                    "schema_version"
                ]
            ),
            "pipeline_id": (
                loaded_document[
                    "pipeline_id"
                ]
            ),
            "graph_sha256": (
                loaded_document[
                    "integrity"
                ]["graph_sha256"]
            ),
            "node_count": len(
                loaded_graph.nodes
            ),
            "edge_count": len(
                loaded_graph.edges
            ),
        },
        "document_roundtrip_test": (
            document_test
        ),
        "execution_roundtrip_test": (
            execution_test
        ),
        "tamper_detection_test": (
            tamper_test
        ),
        "validations": {
            "schema_validation": "PASSED",
            "hash_validation": "PASSED",
            "graph_validation": "PASSED",
            "config_roundtrip": "PASSED",
            "execution_roundtrip": "PASSED",
            "tamper_detection": "PASSED",
        },
    }

    save_json(
        REPORT_PATH,
        report,
    )

    print()
    print("=" * 78)
    print(
        "GENESIS-DSP — ADIM 12 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 78)
    print(
        f"Pipeline node sayısı        : "
        f"{len(loaded_graph.nodes)}"
    )
    print(
        f"Pipeline edge sayısı        : "
        f"{len(loaded_graph.edges)}"
    )
    print(
        "JSON schema doğrulaması     : BAŞARILI"
    )
    print(
        "SHA-256 bütünlük kontrolü   : BAŞARILI"
    )
    print(
        "Graph config round-trip     : BAŞARILI"
    )
    print(
        "Graph execution round-trip  : BAŞARILI"
    )
    print(
        "Değişiklik tespiti          : BAŞARILI"
    )
    print(
        f"Maksimum çıktı farkı        : "
        f"{execution_test['global_maximum_error']:.3e}"
    )
    print(
        f"Graph SHA-256               : "
        f"{document_test['stored_graph_sha256']}"
    )
    print(
        f"Pipeline package            : "
        f"{PACKAGE_PATH}"
    )
    print(
        f"Round-trip graph            : "
        f"{ROUNDTRIP_GRAPH_PATH}"
    )
    print(
        f"Test raporu                 : "
        f"{REPORT_PATH}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
