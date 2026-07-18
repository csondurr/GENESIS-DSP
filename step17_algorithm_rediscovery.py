"""
GENESIS-DSP — Adım 17
Bilinen DSP algoritmalarını yeniden keşfetme testi.

Bu program:
1. Bilinen bir kompleks sinyale DC offset, CFO ve kompleks gain bozunumu ekler.
2. DSP bloklarının tüm alt-küme ve sıralamalarını aday topoloji olarak üretir.
3. Her topoloji için frekans ve kompleks gain parametrelerini optimize eder.
4. DC removal + frequency correction + complex gain recovery zincirini
   herhangi bir doğru sıralamayla yeniden keşfeder.
5. Bulunan pipeline'ı gerçek PipelineGraph motoruyla yeniden çalıştırır.
6. Sonucu sürümlü JSON pipeline paketi, CSV, grafik ve rapor olarak kaydeder.

Çalıştırma:
    python step17_algorithm_rediscovery.py
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from step09_dsp_block_interface import SignalFrame
from step10_block_registry import build_default_registry
from step11_pipeline_graph import PipelineGraph
from step12_pipeline_serialization import PipelinePackage


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP13_DIRECTORY = BASE_DIRECTORY / "outputs" / "step13"
STEP17_DIRECTORY = BASE_DIRECTORY / "outputs" / "step17"

INPUT_PACKAGE = (
    STEP13_DIRECTORY / "optimized_pipeline_package.json"
)
DISCOVERED_PACKAGE_PATH = (
    STEP17_DIRECTORY / "rediscovered_pipeline.json"
)
RANKING_PATH = (
    STEP17_DIRECTORY / "rediscovery_ranking.csv"
)
REPORT_PATH = (
    STEP17_DIRECTORY / "algorithm_rediscovery_report.json"
)
PLOT_PATH = (
    STEP17_DIRECTORY / "algorithm_rediscovery_overview.png"
)
DATA_PATH = (
    STEP17_DIRECTORY / "rediscovery_signals.npz"
)

RANDOM_SEED = 20260731

SAMPLE_RATE_HZ = 1_000_000.0
NUMBER_OF_SAMPLES = 1024

TRUE_CFO_HZ = 18_750.0
TRUE_CHANNEL_GAIN = complex(0.72, 0.36)
TRUE_DC_OFFSET = complex(0.17, -0.11)
SNR_DB = 40.0

FREQUENCY_SEARCH_MIN_HZ = -35_000.0
FREQUENCY_SEARCH_MAX_HZ = 0.0
COARSE_GRID_POINTS = 281
REFINEMENT_GRID_POINTS = 101
REFINEMENT_ROUNDS = 4

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BlockTemplate:
    node_id: str
    block_id: str


@dataclass(frozen=True)
class CandidateResult:
    rank: int
    signature: str
    node_order: tuple[str, ...]
    block_order: tuple[str, ...]
    block_count: int
    estimated_frequency_hz: float
    estimated_gain_real: float
    estimated_gain_imag: float
    nmse: float
    evm_percent: float
    evaluations: int


def load_source_templates() -> tuple[
    list[BlockTemplate],
    dict[str, Any],
]:
    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 13 pipeline paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python step13_continuous_optimizer.py"
        )

    graph, package_document = PipelinePackage.load(
        INPUT_PACKAGE
    )
    config = graph.to_config()

    required_nodes = {
        "remove_dc": "dc_removal",
        "downconvert": "frequency_shift",
        "constant_output": "complex_gain",
    }

    node_map = {
        str(node["node_id"]): str(node["block_id"])
        for node in config["nodes"]
    }

    for node_id, block_id in required_nodes.items():
        if node_map.get(node_id) != block_id:
            raise ValueError(
                f"Gerekli blok bulunamadı: {node_id} / {block_id}"
            )

    templates = [
        BlockTemplate(
            node_id=node_id,
            block_id=block_id,
        )
        for node_id, block_id
        in required_nodes.items()
    ]

    return templates, package_document


def generate_clean_signal(
    rng: np.random.Generator,
) -> ComplexArray:
    """
    Rastgele QPSK örneklerinden oluşan bir kompleks referans sinyali üretir.

    Rastgele yapı; DC, CFO ve gain bloklarının yanlış sıralamalarının
    kolayca ayırt edilmesini sağlar.
    """

    bits = rng.integers(
        0,
        2,
        size=(NUMBER_OF_SAMPLES, 2),
        dtype=np.int8,
    )

    in_phase = 1.0 - 2.0 * bits[:, 0].astype(np.float64)
    quadrature = 1.0 - 2.0 * bits[:, 1].astype(np.float64)

    clean = (
        in_phase + 1j * quadrature
    ) / np.sqrt(2.0)

    return clean.astype(np.complex128)


def create_impaired_signal(
    clean: ComplexArray,
    rng: np.random.Generator,
) -> tuple[ComplexArray, ComplexArray]:
    indices = np.arange(
        len(clean),
        dtype=np.float64,
    )

    carrier = np.exp(
        1j
        * 2.0
        * np.pi
        * TRUE_CFO_HZ
        * indices
        / SAMPLE_RATE_HZ
    )

    noiseless = (
        TRUE_CHANNEL_GAIN
        * clean
        * carrier
        + TRUE_DC_OFFSET
    ).astype(np.complex128)

    signal_power = float(
        np.mean(np.abs(noiseless) ** 2)
    )
    noise_power = (
        signal_power
        / (10.0 ** (SNR_DB / 10.0))
    )
    component_std = np.sqrt(
        noise_power / 2.0
    )

    noise = component_std * (
        rng.standard_normal(len(clean))
        + 1j * rng.standard_normal(len(clean))
    )
    noise = noise.astype(np.complex128)

    received = (
        noiseless + noise
    ).astype(np.complex128)

    return received, noise


def generate_topologies(
    templates: list[BlockTemplate],
) -> list[tuple[BlockTemplate, ...]]:
    topologies: list[
        tuple[BlockTemplate, ...]
    ] = []

    for length in range(
        1,
        len(templates) + 1,
    ):
        topologies.extend(
            itertools.permutations(
                templates,
                length,
            )
        )

    return topologies


def apply_topology(
    samples: ComplexArray,
    ordered_templates: tuple[
        BlockTemplate,
        ...,
    ],
    frequency_hz: float,
    complex_gain: complex,
) -> ComplexArray:
    output = np.asarray(
        samples,
        dtype=np.complex128,
    ).copy()

    indices = np.arange(
        len(output),
        dtype=np.float64,
    )

    for template in ordered_templates:
        if template.block_id == "dc_removal":
            output = (
                output - np.mean(output)
            ).astype(np.complex128)

        elif template.block_id == "frequency_shift":
            output = (
                output
                * np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * frequency_hz
                    * indices
                    / SAMPLE_RATE_HZ
                )
            ).astype(np.complex128)

        elif template.block_id == "complex_gain":
            output = (
                output * complex_gain
            ).astype(np.complex128)

        else:
            raise ValueError(
                f"Desteklenmeyen blok: {template.block_id}"
            )

    return output


def optimal_complex_gain(
    base_output: ComplexArray,
    target: ComplexArray,
) -> complex:
    denominator = np.vdot(
        base_output,
        base_output,
    )

    if abs(denominator) <= 1e-15:
        return complex(0.0, 0.0)

    gain = (
        np.vdot(base_output, target)
        / denominator
    )

    return complex(gain)


def calculate_nmse(
    reference: ComplexArray,
    measured: ComplexArray,
) -> float:
    denominator = float(
        np.sum(np.abs(reference) ** 2)
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


def evaluate_frequency(
    received: ComplexArray,
    target: ComplexArray,
    ordered_templates: tuple[
        BlockTemplate,
        ...,
    ],
    frequency_hz: float,
) -> tuple[float, complex, ComplexArray]:
    has_gain = any(
        template.block_id == "complex_gain"
        for template in ordered_templates
    )

    base_output = apply_topology(
        samples=received,
        ordered_templates=ordered_templates,
        frequency_hz=frequency_hz,
        complex_gain=complex(1.0, 0.0),
    )

    if has_gain:
        gain = optimal_complex_gain(
            base_output=base_output,
            target=target,
        )
        output = (
            base_output * gain
        ).astype(np.complex128)
    else:
        gain = complex(1.0, 0.0)
        output = base_output

    nmse = calculate_nmse(
        reference=target,
        measured=output,
    )

    return nmse, gain, output


def optimize_candidate(
    received: ComplexArray,
    target: ComplexArray,
    ordered_templates: tuple[
        BlockTemplate,
        ...,
    ],
) -> tuple[
    float,
    complex,
    float,
    ComplexArray,
    int,
]:
    has_frequency = any(
        template.block_id == "frequency_shift"
        for template in ordered_templates
    )

    evaluations = 0

    if not has_frequency:
        nmse, gain, output = evaluate_frequency(
            received=received,
            target=target,
            ordered_templates=ordered_templates,
            frequency_hz=0.0,
        )

        return (
            0.0,
            gain,
            nmse,
            output,
            1,
        )

    lower = FREQUENCY_SEARCH_MIN_HZ
    upper = FREQUENCY_SEARCH_MAX_HZ

    best_frequency = 0.0
    best_gain = complex(1.0, 0.0)
    best_nmse = float("inf")
    best_output = np.empty(
        0,
        dtype=np.complex128,
    )

    grids = [
        np.linspace(
            lower,
            upper,
            COARSE_GRID_POINTS,
            dtype=np.float64,
        )
    ]

    for refinement_index in range(
        REFINEMENT_ROUNDS + 1
    ):
        if refinement_index == 0:
            grid = grids[0]
        else:
            previous_step = (
                upper - lower
            ) / (
                (
                    COARSE_GRID_POINTS - 1
                    if refinement_index == 1
                    else REFINEMENT_GRID_POINTS - 1
                )
            )

            half_width = (
                2.0 * previous_step
            )

            lower = max(
                FREQUENCY_SEARCH_MIN_HZ,
                best_frequency - half_width,
            )
            upper = min(
                FREQUENCY_SEARCH_MAX_HZ,
                best_frequency + half_width,
            )

            grid = np.linspace(
                lower,
                upper,
                REFINEMENT_GRID_POINTS,
                dtype=np.float64,
            )

        for frequency_hz in grid:
            nmse, gain, output = evaluate_frequency(
                received=received,
                target=target,
                ordered_templates=ordered_templates,
                frequency_hz=float(frequency_hz),
            )
            evaluations += 1

            if nmse < best_nmse:
                best_nmse = nmse
                best_frequency = float(
                    frequency_hz
                )
                best_gain = gain
                best_output = output

    return (
        best_frequency,
        best_gain,
        best_nmse,
        best_output,
        evaluations,
    )


def candidate_signature(
    ordered_templates: tuple[
        BlockTemplate,
        ...,
    ],
) -> str:
    return " -> ".join(
        template.node_id
        for template in ordered_templates
    )


def build_graph(
    ordered_templates: tuple[
        BlockTemplate,
        ...,
    ],
    frequency_hz: float,
    gain: complex,
) -> PipelineGraph:
    nodes: list[dict[str, Any]] = []

    for template in ordered_templates:
        parameters: dict[str, Any]

        if template.block_id == "frequency_shift":
            parameters = {
                "frequency_hz": float(
                    frequency_hz
                ),
                "initial_phase_degrees": 0.0,
            }

        elif template.block_id == "complex_gain":
            parameters = {
                "gain_real": float(
                    gain.real
                ),
                "gain_imag": float(
                    gain.imag
                ),
            }

        else:
            parameters = {}

        nodes.append(
            {
                "node_id": template.node_id,
                "block_id": template.block_id,
                "parameters": parameters,
            }
        )

    edges = [
        {
            "source": ordered_templates[index].node_id,
            "target": ordered_templates[index + 1].node_id,
        }
        for index in range(
            len(ordered_templates) - 1
        )
    ]

    return PipelineGraph.from_config(
        {
            "schema_name": (
                "GENESIS-DSP PipelineGraph"
            ),
            "schema_version": "1.0.0",
            "nodes": nodes,
            "edges": edges,
        }
    )


def save_ranking(
    ranking: list[CandidateResult],
) -> None:
    with RANKING_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "signature",
                "node_order",
                "block_order",
                "block_count",
                "estimated_frequency_hz",
                "estimated_gain_real",
                "estimated_gain_imag",
                "nmse",
                "evm_percent",
                "evaluations",
            ],
        )
        writer.writeheader()

        for result in ranking:
            row = asdict(result)
            row["node_order"] = " | ".join(
                result.node_order
            )
            row["block_order"] = " | ".join(
                result.block_order
            )
            writer.writerow(row)


def create_plot(
    clean: ComplexArray,
    received: ComplexArray,
    recovered: ComplexArray,
    ranking: list[CandidateResult],
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 14),
    )

    plot_count = min(
        1024,
        len(clean),
    )

    axes[0].scatter(
        received[:plot_count].real,
        received[:plot_count].imag,
        s=8,
        alpha=0.30,
        label="Bozulmuş",
    )
    axes[0].scatter(
        recovered[:plot_count].real,
        recovered[:plot_count].imag,
        s=8,
        alpha=0.35,
        label="Yeniden keşfedilen pipeline çıkışı",
    )
    axes[0].set_title(
        "Kompleks Düzlem — Bozulmuş ve Kurtarılmış"
    )
    axes[0].set_xlabel("I")
    axes[0].set_ylabel("Q")
    axes[0].set_aspect(
        "equal",
        adjustable="box",
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].scatter(
        clean[:plot_count].real,
        clean[:plot_count].imag,
        s=18,
        alpha=0.40,
        label="Referans",
    )
    axes[1].scatter(
        recovered[:plot_count].real,
        recovered[:plot_count].imag,
        s=8,
        alpha=0.35,
        label="Kurtarılmış",
    )
    axes[1].set_title(
        "Referans ve Kurtarılmış QPSK"
    )
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].set_aspect(
        "equal",
        adjustable="box",
    )
    axes[1].grid(True)
    axes[1].legend()

    top_count = min(
        10,
        len(ranking),
    )
    labels = [
        result.signature
        for result in ranking[:top_count]
    ]
    values = [
        max(
            result.nmse,
            1e-18,
        )
        for result in ranking[:top_count]
    ]

    axes[2].bar(
        np.arange(top_count),
        values,
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks(
        np.arange(top_count)
    )
    axes[2].set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )
    axes[2].set_title(
        "En İyi Topoloji Adayları"
    )
    axes[2].set_ylabel("NMSE")
    axes[2].grid(
        True,
        axis="y",
    )

    figure.tight_layout()
    figure.savefig(
        PLOT_PATH,
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    STEP17_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    templates, source_package = (
        load_source_templates()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )
    clean = generate_clean_signal(
        rng
    )
    received, noise = create_impaired_signal(
        clean=clean,
        rng=rng,
    )

    raw_results: list[
        tuple[
            tuple[BlockTemplate, ...],
            float,
            complex,
            float,
            ComplexArray,
            int,
        ]
    ] = []

    for topology in generate_topologies(
        templates
    ):
        (
            frequency_hz,
            gain,
            nmse,
            output,
            evaluations,
        ) = optimize_candidate(
            received=received,
            target=clean,
            ordered_templates=topology,
        )

        raw_results.append(
            (
                topology,
                frequency_hz,
                gain,
                nmse,
                output,
                evaluations,
            )
        )

    raw_results.sort(
        key=lambda item: (
            item[3],
            len(item[0]),
            candidate_signature(
                item[0]
            ),
        )
    )

    ranking: list[
        CandidateResult
    ] = []

    for rank, raw in enumerate(
        raw_results,
        start=1,
    ):
        topology = raw[0]
        nmse = raw[3]

        ranking.append(
            CandidateResult(
                rank=rank,
                signature=candidate_signature(
                    topology
                ),
                node_order=tuple(
                    template.node_id
                    for template in topology
                ),
                block_order=tuple(
                    template.block_id
                    for template in topology
                ),
                block_count=len(
                    topology
                ),
                estimated_frequency_hz=(
                    raw[1]
                ),
                estimated_gain_real=float(
                    raw[2].real
                ),
                estimated_gain_imag=float(
                    raw[2].imag
                ),
                nmse=nmse,
                evm_percent=float(
                    100.0 * np.sqrt(nmse)
                ),
                evaluations=raw[5],
            )
        )

    best_raw = raw_results[0]
    best_topology = best_raw[0]
    best_frequency = best_raw[1]
    best_gain = best_raw[2]
    best_nmse = best_raw[3]
    best_output = best_raw[4]
    best = ranking[0]

    discovered_graph = build_graph(
        ordered_templates=best_topology,
        frequency_hz=best_frequency,
        gain=best_gain,
    )

    registry = build_default_registry()
    input_frame = SignalFrame(
        samples=received,
        sample_rate_hz=SAMPLE_RATE_HZ,
        metadata={
            "source": (
                "step17_algorithm_rediscovery"
            ),
        },
    )

    graph_execution = discovered_graph.execute(
        registry=registry,
        input_frame=input_frame,
    )

    if len(graph_execution.leaf_nodes) != 1:
        raise RuntimeError(
            "Keşfedilen graph tek leaf node içermelidir."
        )

    graph_output = graph_execution.output(
        graph_execution.leaf_nodes[0]
    ).samples

    graph_vs_search_error = float(
        np.max(
            np.abs(
                graph_output - best_output
            )
        )
    )

    required_blocks = {
        "dc_removal",
        "frequency_shift",
        "complex_gain",
    }

    if set(best.block_order) != required_blocks:
        raise RuntimeError(
            "Yeniden keşfedilen pipeline gerekli üç algoritmayı içermiyor."
        )

    dc_index = best.block_order.index(
        "dc_removal"
    )
    frequency_index = best.block_order.index(
        "frequency_shift"
    )

    if dc_index > frequency_index:
        raise RuntimeError(
            "DC removal, frequency correction işleminden önce keşfedilmedi."
        )

    expected_frequency = -TRUE_CFO_HZ
    frequency_error_hz = (
        best_frequency
        - expected_frequency
    )

    expected_gain = (
        1.0 / TRUE_CHANNEL_GAIN
    )
    gain_error = abs(
        best_gain - expected_gain
    )

    if abs(frequency_error_hz) > 2.0:
        raise RuntimeError(
            "Yeniden keşfedilen frekans parametresi tolerans dışında."
        )

    if gain_error > 0.02:
        raise RuntimeError(
            "Yeniden keşfedilen kompleks gain tolerans dışında."
        )

    if best_nmse > 5e-4:
        raise RuntimeError(
            "Yeniden keşfedilen pipeline NMSE eşiğine ulaşamadı."
        )

    incomplete_candidates = [
        item
        for item in ranking
        if item.block_count < 3
    ]
    best_incomplete = min(
        incomplete_candidates,
        key=lambda item: item.nmse,
    )

    improvement_ratio = (
        best_incomplete.nmse
        / best_nmse
    )

    if improvement_ratio < 20.0:
        raise RuntimeError(
            "Tam pipeline eksik topolojilere karşı yeterli üstünlük sağlamadı."
        )

    if graph_vs_search_error > 1e-12:
        raise RuntimeError(
            "Graph motoru ile arama motoru çıktıları eşleşmedi."
        )

    PipelinePackage.save(
        path=DISCOVERED_PACKAGE_PATH,
        graph=discovered_graph,
        pipeline_id=(
            "step17-rediscovered-dsp-chain"
        ),
        description=(
            "DC removal, CFO correction ve kompleks gain recovery "
            "algoritmalarını otomatik yeniden keşfeden pipeline."
        ),
    )

    save_ranking(
        ranking
    )

    np.savez_compressed(
        DATA_PATH,
        clean_samples=clean,
        received_samples=received,
        recovered_samples=graph_output,
        noise_samples=noise,
        sample_rate_hz=np.float64(
            SAMPLE_RATE_HZ
        ),
        true_cfo_hz=np.float64(
            TRUE_CFO_HZ
        ),
        estimated_correction_hz=np.float64(
            best_frequency
        ),
        true_channel_gain=np.complex128(
            TRUE_CHANNEL_GAIN
        ),
        estimated_inverse_gain=np.complex128(
            best_gain
        ),
        true_dc_offset=np.complex128(
            TRUE_DC_OFFSET
        ),
    )

    create_plot(
        clean=clean,
        received=received,
        recovered=graph_output,
        ranking=ranking,
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 17,
        "description": (
            "Automatic rediscovery of DC removal, carrier correction "
            "and complex gain recovery algorithms"
        ),
        "source_pipeline_id": (
            source_package["pipeline_id"]
        ),
        "experiment": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "number_of_samples": NUMBER_OF_SAMPLES,
            "snr_db": SNR_DB,
            "true_cfo_hz": TRUE_CFO_HZ,
            "true_channel_gain_real": float(
                TRUE_CHANNEL_GAIN.real
            ),
            "true_channel_gain_imag": float(
                TRUE_CHANNEL_GAIN.imag
            ),
            "true_dc_offset_real": float(
                TRUE_DC_OFFSET.real
            ),
            "true_dc_offset_imag": float(
                TRUE_DC_OFFSET.imag
            ),
            "candidate_topology_count": (
                len(ranking)
            ),
        },
        "rediscovered_solution": {
            **asdict(best),
            "expected_frequency_correction_hz": (
                expected_frequency
            ),
            "frequency_error_hz": (
                frequency_error_hz
            ),
            "expected_inverse_gain_real": float(
                expected_gain.real
            ),
            "expected_inverse_gain_imag": float(
                expected_gain.imag
            ),
            "complex_gain_error_magnitude": (
                gain_error
            ),
            "graph_vs_search_maximum_error": (
                graph_vs_search_error
            ),
        },
        "best_incomplete_solution": (
            asdict(best_incomplete)
        ),
        "improvement_ratio_over_best_incomplete": (
            improvement_ratio
        ),
        "top_five": [
            asdict(item)
            for item in ranking[:5]
        ],
        "validations": {
            "all_15_topologies_evaluated": True,
            "all_required_algorithms_rediscovered": True,
            "dc_removal_before_frequency_correction": True,
            "frequency_error_below_2_hz": True,
            "complex_gain_error_below_0_02": True,
            "nmse_below_5e_4": True,
            "beats_incomplete_topologies_by_20x": True,
            "graph_execution_matches_search": True,
            "pipeline_package_saved": True,
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
    print("=" * 84)
    print(
        "GENESIS-DSP — ADIM 17 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 84)
    print(
        f"Değerlendirilen topoloji        : "
        f"{len(ranking)}"
    )
    print(
        f"Yeniden keşfedilen pipeline     : "
        f"{best.signature}"
    )
    print(
        f"Gerçek CFO                      : "
        f"{TRUE_CFO_HZ:,.6f} Hz"
    )
    print(
        f"Keşfedilen frekans düzeltmesi   : "
        f"{best_frequency:,.6f} Hz"
    )
    print(
        f"Frekans parametre hatası        : "
        f"{frequency_error_hz:+.6f} Hz"
    )
    print(
        f"Beklenen inverse gain           : "
        f"{expected_gain.real:+.6f} "
        f"{expected_gain.imag:+.6f}j"
    )
    print(
        f"Keşfedilen inverse gain         : "
        f"{best_gain.real:+.6f} "
        f"{best_gain.imag:+.6f}j"
    )
    print(
        f"Kompleks gain hatası            : "
        f"{gain_error:.6e}"
    )
    print(
        f"Yeniden keşif NMSE              : "
        f"{best_nmse:.12e}"
    )
    print(
        f"Yeniden keşif RMS EVM           : "
        f"{best.evm_percent:.6f} %"
    )
    print(
        f"Eksik topolojiye üstünlük       : "
        f"{improvement_ratio:.3f}x"
    )
    print(
        f"Pipeline paketi                 : "
        f"{DISCOVERED_PACKAGE_PATH}"
    )
    print(
        f"Sıralama                        : "
        f"{RANKING_PATH}"
    )
    print(
        f"Grafik                          : "
        f"{PLOT_PATH}"
    )
    print(
        f"Rapor                           : "
        f"{REPORT_PATH}"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
