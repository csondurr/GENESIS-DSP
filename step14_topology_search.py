

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
STEP14_DIRECTORY = BASE_DIRECTORY / "outputs" / "step14"

INPUT_PACKAGE = (
    STEP13_DIRECTORY / "optimized_pipeline_package.json"
)
BEST_PACKAGE_PATH = (
    STEP14_DIRECTORY / "best_topology_pipeline.json"
)
REPORT_PATH = (
    STEP14_DIRECTORY / "topology_search_report.json"
)
RANKING_PATH = (
    STEP14_DIRECTORY / "topology_ranking.csv"
)
PLOT_PATH = (
    STEP14_DIRECTORY / "topology_search_overview.png"
)

COMPLEXITY_PENALTY_PER_BLOCK = 1e-6
TRUE_TONE_FREQUENCY_HZ = 62_500.0
TRUE_TARGET_GAIN = complex(0.50, 0.25)
TEST_SAMPLE_RATE_HZ = 1_000_000.0
TEST_NUMBER_OF_SAMPLES = 2048
TEST_DC_OFFSET = complex(0.18, -0.09)

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class BlockTemplate:
    node_id: str
    block_id: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class TopologyScore:
    rank: int
    signature: str
    block_count: int
    node_order: tuple[str, ...]
    block_order: tuple[str, ...]
    data_nmse: float
    complexity_penalty: float
    total_objective: float
    output_power: float


def load_source_templates(
    graph: PipelineGraph,
) -> list[BlockTemplate]:
    """Adım 13 graph'ından topoloji aramasında kullanılacak blokları alır."""

    source = graph.to_config()

    required_node_ids = (
        "remove_dc",
        "downconvert",
        "constant_output",
    )

    node_map = {
        str(node["node_id"]): node
        for node in source["nodes"]
    }

    missing = [
        node_id
        for node_id in required_node_ids
        if node_id not in node_map
    ]

    if missing:
        raise ValueError(
            "Optimize pipeline içinde gerekli düğümler eksik: "
            + ", ".join(missing)
        )

    templates: list[BlockTemplate] = []

    for node_id in required_node_ids:
        node = node_map[node_id]

        templates.append(
            BlockTemplate(
                node_id=node_id,
                block_id=str(node["block_id"]),
                parameters=dict(
                    node.get("parameters", {})
                ),
            )
        )

    return templates


def build_test_frame() -> SignalFrame:
    """Bilinen hedefe sahip deterministik kompleks test sinyali üretir."""

    indices = np.arange(
        TEST_NUMBER_OF_SAMPLES,
        dtype=np.float64,
    )

    tone = np.exp(
        1j
        * 2.0
        * np.pi
        * TRUE_TONE_FREQUENCY_HZ
        * indices
        / TEST_SAMPLE_RATE_HZ
    )

    frame = SignalFrame(
        samples=(
            tone + TEST_DC_OFFSET
        ).astype(np.complex128),
        sample_rate_hz=TEST_SAMPLE_RATE_HZ,
        metadata={
            "source": "step14_topology_search",
            "tone_frequency_hz": (
                TRUE_TONE_FREQUENCY_HZ
            ),
        },
    )
    frame.validate()

    return frame


def build_target_samples() -> ComplexArray:
    return np.full(
        TEST_NUMBER_OF_SAMPLES,
        TRUE_TARGET_GAIN,
        dtype=np.complex128,
    )


def generate_candidate_orders(
    templates: list[BlockTemplate],
) -> list[tuple[BlockTemplate, ...]]:
    """
    Bütün non-empty alt kümelerin bütün sıralamalarını üretir.

    3 blok için:
        3 tekli + 6 ikili + 6 üçlü = 15 aday
    """

    candidates: list[
        tuple[BlockTemplate, ...]
    ] = []

    for length in range(
        1,
        len(templates) + 1,
    ):
        candidates.extend(
            itertools.permutations(
                templates,
                length,
            )
        )

    return candidates


def candidate_to_graph(
    ordered_templates: tuple[
        BlockTemplate,
        ...,
    ],
) -> PipelineGraph:
    """Bir blok sırasını zincir PipelineGraph'a dönüştürür."""

    nodes = [
        {
            "node_id": template.node_id,
            "block_id": template.block_id,
            "parameters": dict(
                template.parameters
            ),
        }
        for template in ordered_templates
    ]

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


def calculate_nmse(
    reference: ComplexArray,
    measured: ComplexArray,
) -> float:
    if reference.shape != measured.shape:
        raise ValueError(
            "reference ve measured boyutları eşleşmiyor."
        )

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


def evaluate_topology(
    graph: PipelineGraph,
    input_frame: SignalFrame,
    target_samples: ComplexArray,
) -> tuple[
    float,
    float,
    float,
    ComplexArray,
]:
    """Graph leaf çıktısını hedefle karşılaştırır."""

    registry = build_default_registry()

    execution = graph.execute(
        registry=registry,
        input_frame=input_frame,
    )

    if len(execution.leaf_nodes) != 1:
        raise RuntimeError(
            "Aday zincirin tam olarak bir leaf node'u olmalıdır."
        )

    leaf_node = execution.leaf_nodes[0]
    output = execution.output(
        leaf_node
    ).samples

    data_nmse = calculate_nmse(
        target_samples,
        output,
    )

    complexity_penalty = (
        COMPLEXITY_PENALTY_PER_BLOCK
        * len(graph.nodes)
    )

    total_objective = (
        data_nmse
        + complexity_penalty
    )

    output_power = float(
        np.mean(np.abs(output) ** 2)
    )

    return (
        data_nmse,
        complexity_penalty,
        total_objective,
        output,
    )


def topology_signature(
    ordered_templates: tuple[
        BlockTemplate,
        ...,
    ],
) -> str:
    return " -> ".join(
        template.node_id
        for template in ordered_templates
    )


def search_topologies(
    templates: list[BlockTemplate],
    input_frame: SignalFrame,
    target_samples: ComplexArray,
) -> tuple[
    list[TopologyScore],
    PipelineGraph,
    ComplexArray,
]:
    candidates = generate_candidate_orders(
        templates
    )

    raw_results: list[
        tuple[
            str,
            tuple[str, ...],
            tuple[str, ...],
            float,
            float,
            float,
            float,
            PipelineGraph,
            ComplexArray,
        ]
    ] = []

    for ordered_templates in candidates:
        graph = candidate_to_graph(
            ordered_templates
        )

        (
            data_nmse,
            complexity_penalty,
            total_objective,
            output,
        ) = evaluate_topology(
            graph=graph,
            input_frame=input_frame,
            target_samples=target_samples,
        )

        raw_results.append(
            (
                topology_signature(
                    ordered_templates
                ),
                tuple(
                    template.node_id
                    for template
                    in ordered_templates
                ),
                tuple(
                    template.block_id
                    for template
                    in ordered_templates
                ),
                data_nmse,
                complexity_penalty,
                total_objective,
                float(
                    np.mean(
                        np.abs(output) ** 2
                    )
                ),
                graph,
                output,
            )
        )

    raw_results.sort(
        key=lambda item: (
            item[5],
            item[0],
        )
    )

    ranking: list[TopologyScore] = []

    for rank, item in enumerate(
        raw_results,
        start=1,
    ):
        ranking.append(
            TopologyScore(
                rank=rank,
                signature=item[0],
                block_count=len(item[1]),
                node_order=item[1],
                block_order=item[2],
                data_nmse=item[3],
                complexity_penalty=item[4],
                total_objective=item[5],
                output_power=item[6],
            )
        )

    best_graph = raw_results[0][7]
    best_output = raw_results[0][8]

    return ranking, best_graph, best_output


def save_ranking(
    ranking: list[TopologyScore],
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
                "block_count",
                "node_order",
                "block_order",
                "data_nmse",
                "complexity_penalty",
                "total_objective",
                "output_power",
            ],
        )
        writer.writeheader()

        for score in ranking:
            row = asdict(score)
            row["node_order"] = " | ".join(
                score.node_order
            )
            row["block_order"] = " | ".join(
                score.block_order
            )
            writer.writerow(row)


def create_plot(
    ranking: list[TopologyScore],
) -> None:
    ranks = np.asarray(
        [
            score.rank
            for score in ranking
        ],
        dtype=np.int64,
    )

    objectives = np.asarray(
        [
            max(
                score.total_objective,
                1e-18,
            )
            for score in ranking
        ],
        dtype=np.float64,
    )

    block_counts = np.asarray(
        [
            score.block_count
            for score in ranking
        ],
        dtype=np.int64,
    )

    figure = plt.figure(
        figsize=(11, 7)
    )
    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    scatter = axis.scatter(
        ranks,
        objectives,
        c=block_counts,
        s=80,
    )
    axis.set_yscale("log")
    axis.set_title(
        "Kesikli DSP Topoloji Arama Sonuçları"
    )
    axis.set_xlabel("Sıralama")
    axis.set_ylabel(
        "Toplam objective (NMSE + karmaşıklık)"
    )
    axis.grid(True)

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
    )
    colorbar.set_label(
        "Blok sayısı"
    )

    figure.tight_layout()
    figure.savefig(
        PLOT_PATH,
        dpi=180,
    )
    plt.close(figure)


def validate_search(
    ranking: list[TopologyScore],
) -> dict[str, Any]:
    if len(ranking) != 15:
        raise RuntimeError(
            "Beklenen topoloji aday sayısı 15 değil."
        )

    best = ranking[0]

    required_nodes = {
        "remove_dc",
        "downconvert",
        "constant_output",
    }

    if set(best.node_order) != required_nodes:
        raise RuntimeError(
            "En iyi topoloji gerekli üç bloğun tamamını içermiyor."
        )

    if best.data_nmse > 1e-5:
        raise RuntimeError(
            "En iyi topoloji veri NMSE eşiğine ulaşamadı."
        )

    two_block_scores = [
        score
        for score in ranking
        if score.block_count == 2
    ]

    best_two_block = min(
        two_block_scores,
        key=lambda score: score.total_objective,
    )

    if best.total_objective >= best_two_block.total_objective:
        raise RuntimeError(
            "Üç bloklu çözüm iki bloklu çözümlerden daha iyi çıkmadı."
        )

    return {
        "candidate_count": len(ranking),
        "best_signature": best.signature,
        "best_data_nmse": best.data_nmse,
        "best_total_objective": (
            best.total_objective
        ),
        "best_two_block_signature": (
            best_two_block.signature
        ),
        "best_two_block_objective": (
            best_two_block.total_objective
        ),
        "improvement_ratio_over_best_two_block": (
            best_two_block.total_objective
            / best.total_objective
        ),
    }


def main() -> None:
    STEP14_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 13 optimize pipeline bulunamadı: "
            f"{INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python step13_continuous_optimizer.py"
        )

    source_graph, source_package = (
        PipelinePackage.load(
            INPUT_PACKAGE
        )
    )

    templates = load_source_templates(
        source_graph
    )
    input_frame = build_test_frame()
    target_samples = build_target_samples()

    ranking, best_graph, best_output = (
        search_topologies(
            templates=templates,
            input_frame=input_frame,
            target_samples=target_samples,
        )
    )

    validation = validate_search(
        ranking
    )

    PipelinePackage.save(
        path=BEST_PACKAGE_PATH,
        graph=best_graph,
        pipeline_id=(
            "step14-best-discrete-topology"
        ),
        description=(
            "Adım 14 kesikli topoloji aramasıyla "
            "seçilen en iyi DSP pipeline."
        ),
    )

    save_ranking(
        ranking
    )
    create_plot(
        ranking
    )

    np.savez_compressed(
        STEP14_DIRECTORY
        / "best_topology_output.npz",
        input_samples=input_frame.samples,
        target_samples=target_samples,
        best_output_samples=best_output,
        sample_rate_hz=np.float64(
            input_frame.sample_rate_hz
        ),
    )

    best = ranking[0]

    report = {
        "project": "GENESIS-DSP",
        "step": 14,
        "description": (
            "Discrete exhaustive DSP topology search "
            "over block subsets and permutations"
        ),
        "source_pipeline_id": (
            source_package["pipeline_id"]
        ),
        "search_space": {
            "available_nodes": [
                asdict(template)
                for template in templates
            ],
            "candidate_count": len(ranking),
            "complexity_penalty_per_block": (
                COMPLEXITY_PENALTY_PER_BLOCK
            ),
            "search_method": (
                "Exhaustive subset-permutation enumeration"
            ),
        },
        "best_topology": asdict(best),
        "top_five": [
            asdict(score)
            for score in ranking[:5]
        ],
        "validation": validation,
        "validations": {
            "all_15_candidates_evaluated": True,
            "best_contains_required_blocks": True,
            "best_data_nmse_below_1e_5": True,
            "best_beats_all_two_block_candidates": True,
            "best_pipeline_saved": True,
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
    print("=" * 80)
    print(
        "GENESIS-DSP — ADIM 14 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 80)
    print(
        f"Değerlendirilen topoloji      : "
        f"{len(ranking)}"
    )
    print(
        f"En iyi topoloji              : "
        f"{best.signature}"
    )
    print(
        f"En iyi blok sayısı           : "
        f"{best.block_count}"
    )
    print(
        f"En iyi veri NMSE             : "
        f"{best.data_nmse:.12e}"
    )
    print(
        f"En iyi toplam objective      : "
        f"{best.total_objective:.12e}"
    )
    print(
        f"En iyi 2-blok objective      : "
        f"{validation['best_two_block_objective']:.12e}"
    )
    print(
        f"İyileştirme oranı            : "
        f"{validation['improvement_ratio_over_best_two_block']:.3f}x"
    )
    print(
        f"En iyi pipeline              : "
        f"{BEST_PACKAGE_PATH}"
    )
    print(
        f"Topoloji sıralaması          : "
        f"{RANKING_PATH}"
    )
    print(
        f"Grafik                       : "
        f"{PLOT_PATH}"
    )
    print(
        f"Rapor                        : "
        f"{REPORT_PATH}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
