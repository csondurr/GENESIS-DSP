"""
GENESIS-DSP — Adım 10
DSP block registry ve yapılandırmadan blok oluşturma sistemi.

Bu program:
1. Adım 09'daki DSPBlock altyapısını kullanır.
2. DSP bloklarını benzersiz block_id ile kaydeder.
3. Blokları isim, kategori ve kimliğe göre listeler.
4. JSON-benzeri yapılandırmadan blok nesnesi üretir.
5. Registry şemasını JSON olarak dışa aktarır.
6. Duplicate registration, bilinmeyen blok ve hatalı parametre testleri yapar.
7. Registry üzerinden örnek bir DSP zinciri çalıştırır.

Çalıştırma:
    python step10_block_registry.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Type

import numpy as np

from step09_dsp_block_interface import (
    BlockExecutionRecord,
    ComplexGainBlock,
    DCRemovalBlock,
    DSPBlock,
    FrequencyShiftBlock,
    SignalFrame,
    execute_block,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = BASE_DIRECTORY / "outputs" / "step10"


class BlockRegistry:
    """DSPBlock sınıflarını yöneten merkezi kayıt sistemi."""

    def __init__(self) -> None:
        self._blocks: dict[str, Type[DSPBlock]] = {}

    def register(
        self,
        block_class: Type[DSPBlock],
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(block_class, type):
            raise TypeError("block_class bir sınıf olmalıdır.")

        if not issubclass(block_class, DSPBlock):
            raise TypeError(
                "Kaydedilen sınıf DSPBlock sınıfından türemelidir."
            )

        block_id = str(block_class.block_id).strip()

        if not block_id:
            raise ValueError("block_id boş olamaz.")

        if block_id == "abstract":
            raise ValueError(
                "Soyut veya varsayılan block_id kaydedilemez."
            )

        if block_id in self._blocks and not replace:
            raise KeyError(
                f"'{block_id}' kimlikli blok zaten kayıtlı."
            )

        self._blocks[block_id] = block_class

    def register_many(
        self,
        block_classes: Iterable[Type[DSPBlock]],
        *,
        replace: bool = False,
    ) -> None:
        for block_class in block_classes:
            self.register(
                block_class,
                replace=replace,
            )

    def unregister(self, block_id: str) -> Type[DSPBlock]:
        normalized_id = str(block_id).strip()

        if normalized_id not in self._blocks:
            raise KeyError(
                f"'{normalized_id}' kimlikli blok kayıtlı değil."
            )

        return self._blocks.pop(normalized_id)

    def contains(self, block_id: str) -> bool:
        return str(block_id).strip() in self._blocks

    def get_class(self, block_id: str) -> Type[DSPBlock]:
        normalized_id = str(block_id).strip()

        if normalized_id not in self._blocks:
            available = ", ".join(self.block_ids())
            raise KeyError(
                f"'{normalized_id}' kimlikli blok bulunamadı. "
                f"Kayıtlı bloklar: {available}"
            )

        return self._blocks[normalized_id]

    def create(
        self,
        block_id: str,
        **parameters: Any,
    ) -> DSPBlock:
        block_class = self.get_class(block_id)
        return block_class(**parameters)

    def create_from_config(
        self,
        configuration: dict[str, Any],
    ) -> DSPBlock:
        if not isinstance(configuration, dict):
            raise TypeError(
                "Blok yapılandırması dict olmalıdır."
            )

        allowed_keys = {
            "block_id",
            "parameters",
        }
        unknown_keys = set(configuration) - allowed_keys

        if unknown_keys:
            raise ValueError(
                "Bilinmeyen yapılandırma alanları: "
                + ", ".join(sorted(unknown_keys))
            )

        if "block_id" not in configuration:
            raise ValueError(
                "Yapılandırmada 'block_id' alanı bulunmalıdır."
            )

        parameters = configuration.get(
            "parameters",
            {},
        )

        if not isinstance(parameters, dict):
            raise TypeError(
                "'parameters' alanı dict olmalıdır."
            )

        return self.create(
            str(configuration["block_id"]),
            **parameters,
        )

    def block_ids(self) -> list[str]:
        return sorted(self._blocks)

    def list_blocks(
        self,
        *,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        descriptions: list[dict[str, Any]] = []

        for block_id in self.block_ids():
            block_class = self._blocks[block_id]

            if (
                category is not None
                and block_class.category != category
            ):
                continue

            instance = block_class()
            descriptions.append(
                instance.describe()
            )

        return descriptions

    def categories(self) -> list[str]:
        return sorted(
            {
                block_class.category
                for block_class in self._blocks.values()
            }
        )

    def export_schema(self) -> dict[str, Any]:
        return {
            "schema_name": "GENESIS-DSP BlockRegistry",
            "schema_version": "1.0.0",
            "block_count": len(self._blocks),
            "categories": self.categories(),
            "blocks": self.list_blocks(),
        }


def build_default_registry() -> BlockRegistry:
    registry = BlockRegistry()
    registry.register_many(
        (
            DCRemovalBlock,
            ComplexGainBlock,
            FrequencyShiftBlock,
        )
    )
    return registry


def execute_configured_chain(
    registry: BlockRegistry,
    frame: SignalFrame,
    chain_configuration: list[dict[str, Any]],
) -> tuple[SignalFrame, list[BlockExecutionRecord]]:
    if not isinstance(chain_configuration, list):
        raise TypeError(
            "chain_configuration liste olmalıdır."
        )

    if not chain_configuration:
        raise ValueError(
            "chain_configuration boş olamaz."
        )

    current_frame = frame
    records: list[BlockExecutionRecord] = []

    for index, block_configuration in enumerate(
        chain_configuration
    ):
        try:
            block = registry.create_from_config(
                block_configuration
            )
            current_frame, record = execute_block(
                block,
                current_frame,
            )
        except Exception as error:
            raise RuntimeError(
                f"Zincirin {index}. bloğu çalıştırılamadı: "
                f"{block_configuration}"
            ) from error

        records.append(record)

    return current_frame, records


def run_registry_self_tests(
    registry: BlockRegistry,
) -> dict[str, Any]:
    tests: dict[str, str] = {}

    expected_ids = [
        "complex_gain",
        "dc_removal",
        "frequency_shift",
    ]

    if registry.block_ids() != expected_ids:
        raise RuntimeError(
            "Varsayılan registry blok listesi hatalı."
        )

    tests["default_registration"] = "PASSED"

    try:
        registry.register(DCRemovalBlock)
    except KeyError:
        tests["duplicate_rejection"] = "PASSED"
    else:
        raise RuntimeError(
            "Duplicate registration reddedilmedi."
        )

    try:
        registry.create("not_existing_block")
    except KeyError:
        tests["unknown_block_rejection"] = "PASSED"
    else:
        raise RuntimeError(
            "Bilinmeyen blok kimliği reddedilmedi."
        )

    try:
        registry.create(
            "complex_gain",
            unknown_parameter=123,
        )
    except ValueError:
        tests["unknown_parameter_rejection"] = "PASSED"
    else:
        raise RuntimeError(
            "Bilinmeyen parametre reddedilmedi."
        )

    frequency_blocks = registry.list_blocks(
        category="frequency"
    )

    if (
        len(frequency_blocks) != 1
        or frequency_blocks[0]["block_id"]
        != "frequency_shift"
    ):
        raise RuntimeError(
            "Kategori filtresi hatalı çalıştı."
        )

    tests["category_filter"] = "PASSED"

    temporary_registry = build_default_registry()
    removed_class = temporary_registry.unregister(
        "complex_gain"
    )

    if removed_class is not ComplexGainBlock:
        raise RuntimeError(
            "Unregister yanlış sınıf döndürdü."
        )

    if temporary_registry.contains("complex_gain"):
        raise RuntimeError(
            "Unregister sonrası blok hâlâ kayıtlı."
        )

    tests["unregister"] = "PASSED"

    return tests


def run_chain_self_test(
    registry: BlockRegistry,
) -> dict[str, Any]:
    sample_rate_hz = 1_000_000.0
    number_of_samples = 16_384
    tone_frequency_hz = 62_500.0
    injected_dc = complex(0.18, -0.09)

    sample_indices = np.arange(
        number_of_samples,
        dtype=np.float64,
    )

    tone = np.exp(
        1j
        * 2.0
        * np.pi
        * tone_frequency_hz
        * sample_indices
        / sample_rate_hz
    )

    input_frame = SignalFrame(
        samples=(
            tone + injected_dc
        ).astype(np.complex128),
        sample_rate_hz=sample_rate_hz,
        metadata={
            "source": "step10_self_test",
            "tone_frequency_hz": tone_frequency_hz,
        },
    )
    input_frame.validate()

    chain_configuration = [
        {
            "block_id": "dc_removal",
            "parameters": {},
        },
        {
            "block_id": "frequency_shift",
            "parameters": {
                "frequency_hz": -tone_frequency_hz,
                "initial_phase_degrees": 0.0,
            },
        },
        {
            "block_id": "complex_gain",
            "parameters": {
                "gain_real": 0.75,
                "gain_imag": -0.25,
            },
        },
    ]

    output_frame, execution_records = (
        execute_configured_chain(
            registry=registry,
            frame=input_frame,
            chain_configuration=chain_configuration,
        )
    )

    expected_output = np.full(
        number_of_samples,
        complex(0.75, -0.25),
        dtype=np.complex128,
    )

    maximum_error = float(
        np.max(
            np.abs(
                output_frame.samples
                - expected_output
            )
        )
    )

    if maximum_error > 1e-10:
        raise RuntimeError(
            "Registry zincir öz testi başarısız."
        )

    if len(execution_records) != len(
        chain_configuration
    ):
        raise RuntimeError(
            "Yürütme kaydı sayısı hatalı."
        )

    return {
        "status": "PASSED",
        "chain_configuration": chain_configuration,
        "input_power": input_frame.average_power,
        "output_power": output_frame.average_power,
        "output_mean_real": float(
            np.mean(output_frame.samples).real
        ),
        "output_mean_imag": float(
            np.mean(output_frame.samples).imag
        ),
        "maximum_absolute_error": maximum_error,
        "execution_records": [
            asdict(record)
            for record in execution_records
        ],
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = build_default_registry()

    registry_tests = run_registry_self_tests(
        registry
    )
    chain_test = run_chain_self_test(
        registry
    )

    schema = registry.export_schema()

    schema_path = (
        OUTPUT_DIRECTORY
        / "block_registry_schema.json"
    )
    report_path = (
        OUTPUT_DIRECTORY
        / "block_registry_report.json"
    )

    with schema_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            schema,
            file,
            indent=4,
            ensure_ascii=False,
        )

    report = {
        "project": "GENESIS-DSP",
        "step": 10,
        "description": (
            "Central DSP block registry and configuration factory"
        ),
        "registry": {
            "block_count": schema["block_count"],
            "block_ids": registry.block_ids(),
            "categories": registry.categories(),
        },
        "registry_self_tests": registry_tests,
        "configured_chain_self_test": chain_test,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print(
        "GENESIS-DSP — ADIM 10 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 72)
    print(
        f"Kayıtlı blok sayısı          : "
        f"{schema['block_count']}"
    )
    print(
        f"Kayıtlı block_id değerleri   : "
        f"{', '.join(registry.block_ids())}"
    )
    print(
        f"Kategori sayısı              : "
        f"{len(registry.categories())}"
    )
    print(
        "Duplicate koruması           : BAŞARILI"
    )
    print(
        "Config üzerinden blok üretimi: BAŞARILI"
    )
    print(
        "Registry zincir öz testi     : BAŞARILI"
    )
    print(
        f"Maksimum zincir hatası       : "
        f"{chain_test['maximum_absolute_error']:.3e}"
    )
    print(
        f"Registry şeması              : {schema_path}"
    )
    print(
        f"Test raporu                  : {report_path}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
