"""
GENESIS-DSP — Adım 21
Sabit nokta (fixed-point) dönüşüm ve doğrulama sistemi.

Bu program:
1. Adım 20'de sertifikalanmış causal pipeline'ı yükler.
2. Adım 17 test sinyalini kullanarak floating-point referans üretir.
3. Sample/coefficient word length ve NCO phase word length kombinasyonlarını tarar.
4. Causal DC blocker, complex gain ve frequency shift bloklarını
   fixed-point olarak emüle eder.
5. NMSE, EVM, maksimum hata, saturation ve determinism ölçer.
6. En düşük maliyetli başarılı fixed-point yapılandırmayı seçer.
7. JSON config, arama CSV'si, NPZ kayıt, grafik ve rapor üretir.

Çalıştırma:
    python step21_fixed_point_conversion.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from step09_dsp_block_interface import SignalFrame
from step11_pipeline_graph import PipelineGraph
from step12_pipeline_serialization import PipelinePackage
from step20_stability_causality_constraints import (
    build_certified_registry,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP17_DIRECTORY = BASE_DIRECTORY / "outputs" / "step17"
STEP20_DIRECTORY = BASE_DIRECTORY / "outputs" / "step20"
STEP21_DIRECTORY = BASE_DIRECTORY / "outputs" / "step21"

INPUT_PIPELINE_PATH = (
    STEP20_DIRECTORY / "certified_causal_pipeline.json"
)
INPUT_SIGNAL_PATH = (
    STEP17_DIRECTORY / "rediscovery_signals.npz"
)

CONFIG_PATH = (
    STEP21_DIRECTORY / "fixed_point_config.json"
)
SEARCH_PATH = (
    STEP21_DIRECTORY / "fixed_point_search.csv"
)
REPORT_PATH = (
    STEP21_DIRECTORY / "fixed_point_report.json"
)
SIGNALS_PATH = (
    STEP21_DIRECTORY / "fixed_point_signals.npz"
)
PLOT_PATH = (
    STEP21_DIRECTORY / "fixed_point_error.png"
)

INTEGER_BITS = 3
TOTAL_BIT_CANDIDATES = (12, 14, 16, 18)
PHASE_BIT_CANDIDATES = (16, 20, 24)

NMSE_LIMIT = 1e-6
EVM_LIMIT_PERCENT = 0.10
MAXIMUM_ERROR_LIMIT = 0.01
DETERMINISM_TOLERANCE = 0.0

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FixedPointFormat:
    total_bits: int
    integer_bits: int

    @property
    def fractional_bits(self) -> int:
        return self.total_bits - self.integer_bits

    @property
    def scale(self) -> float:
        return float(2 ** self.fractional_bits)

    @property
    def minimum_value(self) -> float:
        return -float(2 ** (self.integer_bits - 1))

    @property
    def maximum_value(self) -> float:
        return (
            float(2 ** (self.integer_bits - 1))
            - 1.0 / self.scale
        )

    @property
    def quantum(self) -> float:
        return 1.0 / self.scale

    def validate(self) -> None:
        if self.total_bits < 4:
            raise ValueError(
                "total_bits en az 4 olmalıdır."
            )

        if self.integer_bits < 2:
            raise ValueError(
                "integer_bits en az 2 olmalıdır."
            )

        if self.integer_bits >= self.total_bits:
            raise ValueError(
                "integer_bits, total_bits değerinden küçük olmalıdır."
            )


@dataclass
class QuantizationStatistics:
    real_saturations: int = 0
    imaginary_saturations: int = 0
    quantization_operations: int = 0

    @property
    def total_saturations(self) -> int:
        return (
            self.real_saturations
            + self.imaginary_saturations
        )


@dataclass(frozen=True)
class CandidateResult:
    rank: int
    total_bits: int
    integer_bits: int
    fractional_bits: int
    phase_bits: int
    quantum: float
    nmse: float
    evm_percent: float
    maximum_absolute_error: float
    rms_absolute_error: float
    saturation_count: int
    quantization_operations: int
    estimated_cost: int
    passed: bool


@dataclass(frozen=True)
class FixedExecutionResult:
    samples: ComplexArray
    statistics: QuantizationStatistics


class FixedPointQuantizer:
    def __init__(
        self,
        number_format: FixedPointFormat,
        statistics: QuantizationStatistics,
    ) -> None:
        number_format.validate()
        self.number_format = number_format
        self.statistics = statistics

    def quantize_real(
        self,
        values: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        below = (
            array
            < self.number_format.minimum_value
        )
        above = (
            array
            > self.number_format.maximum_value
        )

        self.statistics.real_saturations += int(
            np.count_nonzero(
                below | above
            )
        )
        self.statistics.quantization_operations += int(
            array.size
        )

        clipped = np.clip(
            array,
            self.number_format.minimum_value,
            self.number_format.maximum_value,
        )

        return (
            np.rint(
                clipped
                * self.number_format.scale
            )
            / self.number_format.scale
        ).astype(np.float64)

    def quantize_complex(
        self,
        values: ComplexArray | complex,
    ) -> ComplexArray:
        array = np.asarray(
            values,
            dtype=np.complex128,
        )

        real_array = array.real
        imaginary_array = array.imag

        below_real = (
            real_array
            < self.number_format.minimum_value
        )
        above_real = (
            real_array
            > self.number_format.maximum_value
        )
        below_imaginary = (
            imaginary_array
            < self.number_format.minimum_value
        )
        above_imaginary = (
            imaginary_array
            > self.number_format.maximum_value
        )

        self.statistics.real_saturations += int(
            np.count_nonzero(
                below_real | above_real
            )
        )
        self.statistics.imaginary_saturations += int(
            np.count_nonzero(
                below_imaginary
                | above_imaginary
            )
        )
        self.statistics.quantization_operations += int(
            2 * array.size
        )

        real_quantized = (
            np.rint(
                np.clip(
                    real_array,
                    self.number_format.minimum_value,
                    self.number_format.maximum_value,
                )
                * self.number_format.scale
            )
            / self.number_format.scale
        )

        imaginary_quantized = (
            np.rint(
                np.clip(
                    imaginary_array,
                    self.number_format.minimum_value,
                    self.number_format.maximum_value,
                )
                * self.number_format.scale
            )
            / self.number_format.scale
        )

        return (
            real_quantized
            + 1j * imaginary_quantized
        ).astype(np.complex128)


def load_inputs() -> tuple[
    PipelineGraph,
    dict[str, Any],
    ComplexArray,
    float,
]:
    if not INPUT_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Adım 20 sertifikalı pipeline bulunamadı: "
            f"{INPUT_PIPELINE_PATH}\n"
            "Önce şu komutu çalıştır:\n"
            "python step20_stability_causality_constraints.py"
        )

    if not INPUT_SIGNAL_PATH.exists():
        raise FileNotFoundError(
            f"Adım 17 test sinyali bulunamadı: "
            f"{INPUT_SIGNAL_PATH}\n"
            "Önce şu komutu çalıştır:\n"
            "python step17_algorithm_rediscovery.py"
        )

    graph, package_document = PipelinePackage.load(
        INPUT_PIPELINE_PATH
    )

    with np.load(
        INPUT_SIGNAL_PATH,
        allow_pickle=False,
    ) as package:
        received_samples = package[
            "received_samples"
        ].astype(np.complex128)
        sample_rate_hz = float(
            package["sample_rate_hz"]
        )

    return (
        graph,
        package_document,
        received_samples,
        sample_rate_hz,
    )


def execute_float_reference(
    graph: PipelineGraph,
    samples: ComplexArray,
    sample_rate_hz: float,
) -> ComplexArray:
    registry = build_certified_registry()

    frame = SignalFrame(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        metadata={
            "source": "step21_float_reference",
        },
    )
    frame.validate()

    execution = graph.execute(
        registry=registry,
        input_frame=frame,
    )

    if len(execution.leaf_nodes) != 1:
        raise RuntimeError(
            "Fixed-point dönüşüm yalnızca tek leaf node'lu pipeline destekler."
        )

    return execution.output(
        execution.leaf_nodes[0]
    ).samples


def quantized_complex_gain(
    samples: ComplexArray,
    parameters: dict[str, Any],
    quantizer: FixedPointQuantizer,
) -> ComplexArray:
    gain = complex(
        float(
            parameters.get(
                "gain_real",
                1.0,
            )
        ),
        float(
            parameters.get(
                "gain_imag",
                0.0,
            )
        ),
    )

    quantized_gain = complex(
        quantizer.quantize_complex(
            np.asarray(
                [gain],
                dtype=np.complex128,
            )
        )[0]
    )

    return quantizer.quantize_complex(
        samples * quantized_gain
    )


def quantized_causal_dc_blocker(
    samples: ComplexArray,
    parameters: dict[str, Any],
    quantizer: FixedPointQuantizer,
) -> ComplexArray:
    pole_radius = float(
        parameters.get(
            "pole_radius",
            0.995,
        )
    )

    quantized_pole = float(
        quantizer.quantize_real(
            np.asarray(
                [pole_radius],
                dtype=np.float64,
            )
        )[0]
    )

    output = np.empty_like(
        samples,
        dtype=np.complex128,
    )

    previous_input = complex(
        0.0,
        0.0,
    )
    previous_output = complex(
        0.0,
        0.0,
    )

    for index, current_input in enumerate(
        samples
    ):
        current_input_quantized = complex(
            quantizer.quantize_complex(
                np.asarray(
                    [current_input],
                    dtype=np.complex128,
                )
            )[0]
        )

        current_output = (
            current_input_quantized
            - previous_input
            + quantized_pole
            * previous_output
        )

        current_output_quantized = complex(
            quantizer.quantize_complex(
                np.asarray(
                    [current_output],
                    dtype=np.complex128,
                )
            )[0]
        )

        output[index] = (
            current_output_quantized
        )
        previous_input = (
            current_input_quantized
        )
        previous_output = (
            current_output_quantized
        )

    return output


def quantized_frequency_shift(
    samples: ComplexArray,
    parameters: dict[str, Any],
    sample_rate_hz: float,
    phase_bits: int,
    quantizer: FixedPointQuantizer,
) -> ComplexArray:
    frequency_hz = float(
        parameters.get(
            "frequency_hz",
            0.0,
        )
    )
    initial_phase_degrees = float(
        parameters.get(
            "initial_phase_degrees",
            0.0,
        )
    )

    phase_modulus = 1 << phase_bits

    increment_word = int(
        np.rint(
            frequency_hz
            / sample_rate_hz
            * phase_modulus
        )
    )

    initial_phase_word = int(
        np.rint(
            (
                initial_phase_degrees
                / 360.0
            )
            * phase_modulus
        )
    )

    phase_words = (
        initial_phase_word
        + increment_word
        * np.arange(
            len(samples),
            dtype=np.int64,
        )
    ) % phase_modulus

    phase_radians = (
        2.0
        * np.pi
        * phase_words.astype(
            np.float64
        )
        / phase_modulus
    )

    oscillator = quantizer.quantize_complex(
        np.exp(
            1j * phase_radians
        ).astype(np.complex128)
    )

    return quantizer.quantize_complex(
        samples * oscillator
    )


def execute_fixed_point(
    graph: PipelineGraph,
    samples: ComplexArray,
    sample_rate_hz: float,
    number_format: FixedPointFormat,
    phase_bits: int,
) -> FixedExecutionResult:
    statistics = QuantizationStatistics()
    quantizer = FixedPointQuantizer(
        number_format=number_format,
        statistics=statistics,
    )

    graph_config = graph.to_config()
    node_map = {
        str(node["node_id"]): node
        for node in graph_config["nodes"]
    }

    current = quantizer.quantize_complex(
        samples
    )

    for node_id in graph.topological_order:
        node = node_map[node_id]
        block_id = str(
            node["block_id"]
        )
        parameters = dict(
            node.get(
                "parameters",
                {},
            )
        )

        if block_id == "complex_gain":
            current = quantized_complex_gain(
                samples=current,
                parameters=parameters,
                quantizer=quantizer,
            )

        elif block_id == "causal_dc_blocker":
            current = (
                quantized_causal_dc_blocker(
                    samples=current,
                    parameters=parameters,
                    quantizer=quantizer,
                )
            )

        elif block_id == "frequency_shift":
            current = (
                quantized_frequency_shift(
                    samples=current,
                    parameters=parameters,
                    sample_rate_hz=sample_rate_hz,
                    phase_bits=phase_bits,
                    quantizer=quantizer,
                )
            )

        else:
            raise ValueError(
                f"Fixed-point emülasyonu tanımlanmamış blok: {block_id}"
            )

    return FixedExecutionResult(
        samples=current,
        statistics=statistics,
    )


def calculate_nmse(
    reference: ComplexArray,
    measured: ComplexArray,
) -> float:
    denominator = float(
        np.sum(
            np.abs(reference) ** 2
        )
    )

    if denominator <= 0.0:
        raise ValueError(
            "Referans enerjisi pozitif olmalıdır."
        )

    numerator = float(
        np.sum(
            np.abs(
                measured - reference
            ) ** 2
        )
    )

    return numerator / denominator


def evaluate_candidate(
    graph: PipelineGraph,
    input_samples: ComplexArray,
    float_reference: ComplexArray,
    sample_rate_hz: float,
    total_bits: int,
    phase_bits: int,
) -> tuple[
    CandidateResult,
    FixedExecutionResult,
]:
    number_format = FixedPointFormat(
        total_bits=total_bits,
        integer_bits=INTEGER_BITS,
    )

    fixed_result = execute_fixed_point(
        graph=graph,
        samples=input_samples,
        sample_rate_hz=sample_rate_hz,
        number_format=number_format,
        phase_bits=phase_bits,
    )

    difference = (
        fixed_result.samples
        - float_reference
    )

    nmse = calculate_nmse(
        float_reference,
        fixed_result.samples,
    )
    evm_percent = float(
        100.0 * np.sqrt(nmse)
    )
    maximum_error = float(
        np.max(
            np.abs(difference)
        )
    )
    rms_error = float(
        np.sqrt(
            np.mean(
                np.abs(difference) ** 2
            )
        )
    )

    saturation_count = (
        fixed_result.statistics.total_saturations
    )

    passed = bool(
        nmse <= NMSE_LIMIT
        and evm_percent
        <= EVM_LIMIT_PERCENT
        and maximum_error
        <= MAXIMUM_ERROR_LIMIT
        and saturation_count == 0
    )

    estimated_cost = int(
        total_bits * total_bits
        + 2 * total_bits
        + phase_bits
    )

    result = CandidateResult(
        rank=0,
        total_bits=total_bits,
        integer_bits=INTEGER_BITS,
        fractional_bits=(
            number_format.fractional_bits
        ),
        phase_bits=phase_bits,
        quantum=number_format.quantum,
        nmse=nmse,
        evm_percent=evm_percent,
        maximum_absolute_error=(
            maximum_error
        ),
        rms_absolute_error=rms_error,
        saturation_count=(
            saturation_count
        ),
        quantization_operations=(
            fixed_result.statistics.quantization_operations
        ),
        estimated_cost=estimated_cost,
        passed=passed,
    )

    return result, fixed_result


def save_search(
    results: list[CandidateResult],
) -> None:
    with SEARCH_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "total_bits",
                "integer_bits",
                "fractional_bits",
                "phase_bits",
                "quantum",
                "nmse",
                "evm_percent",
                "maximum_absolute_error",
                "rms_absolute_error",
                "saturation_count",
                "quantization_operations",
                "estimated_cost",
                "passed",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                asdict(result)
            )


def create_plot(
    results: list[CandidateResult],
    selected: CandidateResult,
) -> None:
    labels = [
        (
            f"Q{item.total_bits - item.integer_bits} / "
            f"P{item.phase_bits}"
        )
        for item in results
    ]

    values = [
        max(
            item.evm_percent,
            1e-12,
        )
        for item in results
    ]

    positions = np.arange(
        len(results),
        dtype=np.int64,
    )

    figure = plt.figure(
        figsize=(13, 7)
    )
    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.bar(
        positions,
        values,
    )
    axis.axhline(
        EVM_LIMIT_PERCENT,
        linestyle="--",
        label="EVM başarı eşiği",
    )

    selected_index = next(
        index
        for index, item in enumerate(
            results
        )
        if (
            item.total_bits
            == selected.total_bits
            and item.phase_bits
            == selected.phase_bits
        )
    )

    axis.scatter(
        [selected_index],
        [selected.evm_percent],
        marker="*",
        s=240,
        label="Seçilen format",
    )

    axis.set_yscale("log")
    axis.set_xticks(
        positions
    )
    axis.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )
    axis.set_title(
        "GENESIS-DSP Fixed-Point Format Araması"
    )
    axis.set_xlabel(
        "Fractional bit / phase accumulator bit"
    )
    axis.set_ylabel(
        "Floating-point referansa göre RMS EVM (%)"
    )
    axis.grid(
        True,
        axis="y",
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        PLOT_PATH,
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    STEP21_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        graph,
        package_document,
        input_samples,
        sample_rate_hz,
    ) = load_inputs()

    float_reference = (
        execute_float_reference(
            graph=graph,
            samples=input_samples,
            sample_rate_hz=sample_rate_hz,
        )
    )

    raw_results: list[
        CandidateResult
    ] = []
    output_map: dict[
        tuple[int, int],
        FixedExecutionResult,
    ] = {}

    for total_bits in TOTAL_BIT_CANDIDATES:
        for phase_bits in PHASE_BIT_CANDIDATES:
            result, fixed_output = (
                evaluate_candidate(
                    graph=graph,
                    input_samples=input_samples,
                    float_reference=float_reference,
                    sample_rate_hz=sample_rate_hz,
                    total_bits=total_bits,
                    phase_bits=phase_bits,
                )
            )

            raw_results.append(result)
            output_map[
                (
                    total_bits,
                    phase_bits,
                )
            ] = fixed_output

    raw_results.sort(
        key=lambda item: (
            not item.passed,
            item.estimated_cost,
            item.nmse,
            item.total_bits,
            item.phase_bits,
        )
    )

    results = [
        CandidateResult(
            **{
                **asdict(item),
                "rank": rank,
            }
        )
        for rank, item in enumerate(
            raw_results,
            start=1,
        )
    ]

    passing = [
        item
        for item in results
        if item.passed
    ]

    if not passing:
        raise RuntimeError(
            "Hiçbir fixed-point format başarı eşiklerini geçemedi."
        )

    selected = min(
        passing,
        key=lambda item: (
            item.estimated_cost,
            item.nmse,
            item.total_bits,
            item.phase_bits,
        ),
    )

    selected_output = output_map[
        (
            selected.total_bits,
            selected.phase_bits,
        )
    ]

    repeated_output = execute_fixed_point(
        graph=graph,
        samples=input_samples,
        sample_rate_hz=sample_rate_hz,
        number_format=FixedPointFormat(
            total_bits=selected.total_bits,
            integer_bits=selected.integer_bits,
        ),
        phase_bits=selected.phase_bits,
    )

    determinism_error = float(
        np.max(
            np.abs(
                repeated_output.samples
                - selected_output.samples
            )
        )
    )

    if (
        determinism_error
        > DETERMINISM_TOLERANCE
    ):
        raise RuntimeError(
            "Fixed-point yürütme determinism testini geçemedi."
        )

    input_peak = float(
        np.max(
            np.abs(input_samples)
        )
    )
    output_peak = float(
        np.max(
            np.abs(
                selected_output.samples
            )
        )
    )

    selected_format = FixedPointFormat(
        total_bits=selected.total_bits,
        integer_bits=selected.integer_bits,
    )

    if (
        input_peak
        >= selected_format.maximum_value
        or output_peak
        >= selected_format.maximum_value
    ):
        raise RuntimeError(
            "Seçilen formatta headroom yetersiz."
        )

    fixed_point_config = {
        "schema_name": (
            "GENESIS-DSP FixedPointConfiguration"
        ),
        "schema_version": "1.0.0",
        "source_pipeline_id": (
            package_document["pipeline_id"]
        ),
        "arithmetic": {
            "representation": (
                "signed two's-complement emulation"
            ),
            "rounding": "round_to_nearest",
            "overflow": "saturate",
            "sample_and_coefficient_format": {
                "total_bits": (
                    selected.total_bits
                ),
                "integer_bits": (
                    selected.integer_bits
                ),
                "fractional_bits": (
                    selected.fractional_bits
                ),
                "quantum": (
                    selected.quantum
                ),
                "minimum_value": (
                    selected_format.minimum_value
                ),
                "maximum_value": (
                    selected_format.maximum_value
                ),
            },
            "nco_phase_accumulator_bits": (
                selected.phase_bits
            ),
        },
        "acceptance_limits": {
            "nmse": NMSE_LIMIT,
            "evm_percent": (
                EVM_LIMIT_PERCENT
            ),
            "maximum_absolute_error": (
                MAXIMUM_ERROR_LIMIT
            ),
            "saturation_count": 0,
        },
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            fixed_point_config,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    save_search(
        results
    )

    np.savez_compressed(
        SIGNALS_PATH,
        input_samples=input_samples,
        floating_point_output=(
            float_reference
        ),
        fixed_point_output=(
            selected_output.samples
        ),
        fixed_point_error=(
            selected_output.samples
            - float_reference
        ),
        sample_rate_hz=np.float64(
            sample_rate_hz
        ),
        total_bits=np.int64(
            selected.total_bits
        ),
        integer_bits=np.int64(
            selected.integer_bits
        ),
        fractional_bits=np.int64(
            selected.fractional_bits
        ),
        phase_bits=np.int64(
            selected.phase_bits
        ),
    )

    create_plot(
        results=results,
        selected=selected,
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 21,
        "description": (
            "Fixed-point conversion, word-length search "
            "and floating-point equivalence verification"
        ),
        "source_pipeline_id": (
            package_document["pipeline_id"]
        ),
        "search_space": {
            "total_bit_candidates": list(
                TOTAL_BIT_CANDIDATES
            ),
            "integer_bits": INTEGER_BITS,
            "phase_bit_candidates": list(
                PHASE_BIT_CANDIDATES
            ),
            "candidate_count": len(
                results
            ),
            "passing_candidate_count": len(
                passing
            ),
        },
        "selected_format": asdict(
            selected
        ),
        "dynamic_range": {
            "input_peak_magnitude": (
                input_peak
            ),
            "output_peak_magnitude": (
                output_peak
            ),
            "maximum_representable_component": (
                selected_format.maximum_value
            ),
        },
        "determinism": {
            "maximum_error": (
                determinism_error
            ),
            "status": "PASSED",
        },
        "top_five": [
            asdict(item)
            for item in results[:5]
        ],
        "validations": {
            "nmse_below_limit": True,
            "evm_below_limit": True,
            "maximum_error_below_limit": True,
            "zero_saturation": True,
            "dynamic_range_headroom": True,
            "deterministic_reexecution": True,
            "configuration_saved": True,
        },
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    print()
    print("=" * 86)
    print(
        "GENESIS-DSP — ADIM 21 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 86)
    print(
        f"Değerlendirilen fixed format    : "
        f"{len(results)}"
    )
    print(
        f"Başarılı format                 : "
        f"{len(passing)}"
    )
    print(
        f"Seçilen toplam bit              : "
        f"{selected.total_bits}"
    )
    print(
        f"Seçilen integer/fractional bit  : "
        f"{selected.integer_bits} / "
        f"{selected.fractional_bits}"
    )
    print(
        f"NCO phase accumulator bit       : "
        f"{selected.phase_bits}"
    )
    print(
        f"Fixed-point quantum             : "
        f"{selected.quantum:.12e}"
    )
    print(
        f"Floating referansa göre NMSE    : "
        f"{selected.nmse:.12e}"
    )
    print(
        f"Floating referansa göre EVM     : "
        f"{selected.evm_percent:.9f} %"
    )
    print(
        f"Maksimum mutlak hata            : "
        f"{selected.maximum_absolute_error:.12e}"
    )
    print(
        f"Saturation sayısı               : "
        f"{selected.saturation_count}"
    )
    print(
        f"Determinism maksimum hatası     : "
        f"{determinism_error:.3e}"
    )
    print(
        "Fixed-point doğrulaması         : BAŞARILI"
    )
    print(
        f"Fixed-point config              : "
        f"{CONFIG_PATH}"
    )
    print(
        f"Arama tablosu                   : "
        f"{SEARCH_PATH}"
    )
    print(
        f"Grafik                          : "
        f"{PLOT_PATH}"
    )
    print(
        f"Rapor                           : "
        f"{REPORT_PATH}"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
