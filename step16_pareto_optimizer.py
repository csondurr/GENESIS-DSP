

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from step10_block_registry import build_default_registry
from step12_pipeline_serialization import PipelinePackage
from step14_topology_search import (
    build_target_samples,
    build_test_frame,
    candidate_to_graph,
    generate_candidate_orders,
    load_source_templates,
    topology_signature,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP13_DIRECTORY = BASE_DIRECTORY / "outputs" / "step13"
STEP16_DIRECTORY = BASE_DIRECTORY / "outputs" / "step16"

INPUT_PACKAGE = (
    STEP13_DIRECTORY / "optimized_pipeline_package.json"
)
SELECTED_PACKAGE_PATH = (
    STEP16_DIRECTORY / "pareto_selected_pipeline.json"
)
ALL_CANDIDATES_PATH = (
    STEP16_DIRECTORY / "pareto_all_candidates.csv"
)
PARETO_FRONT_PATH = (
    STEP16_DIRECTORY / "pareto_front.csv"
)
REPORT_PATH = (
    STEP16_DIRECTORY / "pareto_optimizer_report.json"
)
PLOT_PATH = (
    STEP16_DIRECTORY / "pareto_front.png"
)

# Yaklaşık gerçek-aritmetik işlem maliyetleri / kompleks örnek.
BLOCK_OPERATION_COST = {
    "dc_removal": 4,
    "frequency_shift": 10,
    "complex_gain": 6,
}


@dataclass(frozen=True)
class CandidateMetrics:
    signature: str
    node_order: tuple[str, ...]
    block_order: tuple[str, ...]
    block_count: int
    data_nmse: float
    operation_cost_per_sample: int
    total_operation_cost: int
    output_power: float
    pareto_optimal: bool = False
    utopia_distance: float | None = None


def calculate_nmse(
    reference: np.ndarray,
    measured: np.ndarray,
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
        np.sum(np.abs(measured - reference) ** 2)
    )

    return numerator / denominator


def evaluate_candidates() -> tuple[
    list[CandidateMetrics],
    dict[str, Any],
]:
    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 13 optimize pipeline bulunamadı: "
            f"{INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python step13_continuous_optimizer.py"
        )

    source_graph, source_package = PipelinePackage.load(
        INPUT_PACKAGE
    )

    templates = load_source_templates(
        source_graph
    )
    candidate_orders = generate_candidate_orders(
        templates
    )

    input_frame = build_test_frame()
    target_samples = build_target_samples()
    registry = build_default_registry()

    results: list[CandidateMetrics] = []

    for ordered_templates in candidate_orders:
        graph = candidate_to_graph(
            ordered_templates
        )

        execution = graph.execute(
            registry=registry,
            input_frame=input_frame,
        )

        if len(execution.leaf_nodes) != 1:
            raise RuntimeError(
                "Aday graph tam olarak bir leaf node içermelidir."
            )

        output = execution.output(
            execution.leaf_nodes[0]
        ).samples

        block_order = tuple(
            template.block_id
            for template in ordered_templates
        )

        operation_cost_per_sample = int(
            sum(
                BLOCK_OPERATION_COST[
                    block_id
                ]
                for block_id in block_order
            )
        )

        results.append(
            CandidateMetrics(
                signature=topology_signature(
                    ordered_templates
                ),
                node_order=tuple(
                    template.node_id
                    for template in ordered_templates
                ),
                block_order=block_order,
                block_count=len(
                    ordered_templates
                ),
                data_nmse=calculate_nmse(
                    target_samples,
                    output,
                ),
                operation_cost_per_sample=(
                    operation_cost_per_sample
                ),
                total_operation_cost=(
                    operation_cost_per_sample
                    * len(input_frame.samples)
                ),
                output_power=float(
                    np.mean(
                        np.abs(output) ** 2
                    )
                ),
            )
        )

    return results, source_package


def dominates(
    first: CandidateMetrics,
    second: CandidateMetrics,
) -> bool:
    """
    first, second adayını tüm amaçlarda kötü değil ve en az birinde
    kesin daha iyi ise domine eder.
    """

    first_values = (
        first.data_nmse,
        first.operation_cost_per_sample,
        first.block_count,
    )
    second_values = (
        second.data_nmse,
        second.operation_cost_per_sample,
        second.block_count,
    )

    no_worse = all(
        first_value <= second_value
        for first_value, second_value
        in zip(
            first_values,
            second_values,
            strict=True,
        )
    )

    strictly_better = any(
        first_value < second_value
        for first_value, second_value
        in zip(
            first_values,
            second_values,
            strict=True,
        )
    )

    return no_worse and strictly_better


def extract_pareto_front(
    candidates: list[CandidateMetrics],
) -> list[CandidateMetrics]:
    pareto: list[CandidateMetrics] = []

    for candidate in candidates:
        is_dominated = any(
            dominates(other, candidate)
            for other in candidates
            if other is not candidate
        )

        if not is_dominated:
            pareto.append(candidate)

    pareto.sort(
        key=lambda item: (
            item.operation_cost_per_sample,
            item.data_nmse,
            item.signature,
        )
    )

    return pareto


def normalize_values(
    values: np.ndarray,
) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if np.isclose(
        maximum,
        minimum,
        atol=1e-15,
    ):
        return np.zeros_like(
            values,
            dtype=np.float64,
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def assign_utopia_distances(
    pareto: list[CandidateMetrics],
) -> list[CandidateMetrics]:
    """
    NMSE log uzayında normalize edilir. Maliyet ve blok sayısı lineer
    normalize edilir. Her üç hedef eşit ağırlıklıdır.
    """

    if not pareto:
        raise RuntimeError(
            "Pareto cephesi boş olamaz."
        )

    log_nmse = np.log10(
        np.maximum(
            np.asarray(
                [
                    item.data_nmse
                    for item in pareto
                ],
                dtype=np.float64,
            ),
            1e-18,
        )
    )

    operation_cost = np.asarray(
        [
            item.operation_cost_per_sample
            for item in pareto
        ],
        dtype=np.float64,
    )

    block_count = np.asarray(
        [
            item.block_count
            for item in pareto
        ],
        dtype=np.float64,
    )

    normalized_error = normalize_values(
        log_nmse
    )
    normalized_operations = normalize_values(
        operation_cost
    )
    normalized_blocks = normalize_values(
        block_count
    )

    distances = np.sqrt(
        normalized_error ** 2
        + normalized_operations ** 2
        + normalized_blocks ** 2
    )

    updated: list[CandidateMetrics] = []

    for item, distance in zip(
        pareto,
        distances,
        strict=True,
    ):
        updated.append(
            CandidateMetrics(
                **{
                    **asdict(item),
                    "pareto_optimal": True,
                    "utopia_distance": float(
                        distance
                    ),
                }
            )
        )

    updated.sort(
        key=lambda item: (
            float(item.utopia_distance),
            item.data_nmse,
            item.operation_cost_per_sample,
            item.signature,
        )
    )

    return updated


def save_csv(
    path: Path,
    candidates: list[CandidateMetrics],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "signature",
                "node_order",
                "block_order",
                "block_count",
                "data_nmse",
                "operation_cost_per_sample",
                "total_operation_cost",
                "output_power",
                "pareto_optimal",
                "utopia_distance",
            ],
        )
        writer.writeheader()

        for candidate in candidates:
            row = asdict(candidate)
            row["node_order"] = " | ".join(
                candidate.node_order
            )
            row["block_order"] = " | ".join(
                candidate.block_order
            )
            writer.writerow(row)


def create_plot(
    candidates: list[CandidateMetrics],
    pareto: list[CandidateMetrics],
    selected: CandidateMetrics,
) -> None:
    figure = plt.figure(
        figsize=(12, 8)
    )
    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    candidate_cost = np.asarray(
        [
            item.operation_cost_per_sample
            for item in candidates
        ],
        dtype=np.float64,
    )
    candidate_nmse = np.asarray(
        [
            item.data_nmse
            for item in candidates
        ],
        dtype=np.float64,
    )

    axis.scatter(
        candidate_cost,
        candidate_nmse,
        s=60,
        alpha=0.55,
        label="Tüm adaylar",
    )

    ordered_pareto = sorted(
        pareto,
        key=lambda item: (
            item.operation_cost_per_sample,
            item.data_nmse,
        ),
    )

    pareto_cost = np.asarray(
        [
            item.operation_cost_per_sample
            for item in ordered_pareto
        ],
        dtype=np.float64,
    )
    pareto_nmse = np.asarray(
        [
            item.data_nmse
            for item in ordered_pareto
        ],
        dtype=np.float64,
    )

    axis.plot(
        pareto_cost,
        pareto_nmse,
        marker="o",
        linewidth=2.0,
        label="Pareto cephesi",
    )

    axis.scatter(
        [selected.operation_cost_per_sample],
        [selected.data_nmse],
        marker="*",
        s=260,
        label="Seçilen dengeli çözüm",
    )

    for candidate in ordered_pareto:
        axis.annotate(
            candidate.signature,
            (
                candidate.operation_cost_per_sample,
                candidate.data_nmse,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_yscale("log")
    axis.set_title(
        "GENESIS-DSP Çok Amaçlı Pareto Cephesi"
    )
    axis.set_xlabel(
        "Tahmini işlem maliyeti / kompleks örnek"
    )
    axis.set_ylabel(
        "Veri NMSE"
    )
    axis.grid(True)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        PLOT_PATH,
        dpi=180,
    )
    plt.close(figure)


def validate_results(
    candidates: list[CandidateMetrics],
    pareto: list[CandidateMetrics],
    selected: CandidateMetrics,
) -> dict[str, Any]:
    if len(candidates) != 15:
        raise RuntimeError(
            "Beklenen aday sayısı 15 değil."
        )

    if not pareto:
        raise RuntimeError(
            "Pareto cephesi boş."
        )

    for pareto_candidate in pareto:
        if any(
            dominates(
                other,
                pareto_candidate,
            )
            for other in candidates
            if other.signature
            != pareto_candidate.signature
        ):
            raise RuntimeError(
                "Pareto cephesinde domine edilmiş aday bulundu."
            )

    best_accuracy = min(
        candidates,
        key=lambda item: (
            item.data_nmse,
            item.operation_cost_per_sample,
        ),
    )

    pareto_signatures = {
        item.signature
        for item in pareto
    }

    if (
        best_accuracy.signature
        not in pareto_signatures
    ):
        raise RuntimeError(
            "En düşük NMSE çözümü Pareto cephesinde değil."
        )

    if selected.signature not in pareto_signatures:
        raise RuntimeError(
            "Seçilen çözüm Pareto cephesinde değil."
        )

    minimum_distance = min(
        float(item.utopia_distance)
        for item in pareto
    )

    if not np.isclose(
        float(selected.utopia_distance),
        minimum_distance,
        atol=1e-15,
    ):
        raise RuntimeError(
            "Seçilen çözüm minimum utopia-distance değerine sahip değil."
        )

    return {
        "candidate_count": len(candidates),
        "pareto_count": len(pareto),
        "best_accuracy_signature": (
            best_accuracy.signature
        ),
        "best_accuracy_nmse": (
            best_accuracy.data_nmse
        ),
        "selected_signature": (
            selected.signature
        ),
        "selected_utopia_distance": (
            selected.utopia_distance
        ),
    }


def main() -> None:
    STEP16_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates, source_package = (
        evaluate_candidates()
    )

    raw_pareto = extract_pareto_front(
        candidates
    )
    pareto = assign_utopia_distances(
        raw_pareto
    )
    selected = pareto[0]

    validation = validate_results(
        candidates=candidates,
        pareto=pareto,
        selected=selected,
    )

    source_graph, _ = PipelinePackage.load(
        INPUT_PACKAGE
    )
    templates = load_source_templates(
        source_graph
    )
    template_map = {
        template.node_id: template
        for template in templates
    }

    selected_order = tuple(
        template_map[node_id]
        for node_id in selected.node_order
    )
    selected_graph = candidate_to_graph(
        selected_order
    )

    PipelinePackage.save(
        path=SELECTED_PACKAGE_PATH,
        graph=selected_graph,
        pipeline_id=(
            "step16-pareto-selected-pipeline"
        ),
        description=(
            "Adım 16 Pareto optimizasyonunda "
            "utopia-distance ile seçilen dengeli pipeline."
        ),
    )

    pareto_signature_set = {
        item.signature
        for item in pareto
    }

    all_candidates_marked = [
        CandidateMetrics(
            **{
                **asdict(item),
                "pareto_optimal": (
                    item.signature
                    in pareto_signature_set
                ),
                "utopia_distance": next(
                    (
                        pareto_item.utopia_distance
                        for pareto_item in pareto
                        if pareto_item.signature
                        == item.signature
                    ),
                    None,
                ),
            }
        )
        for item in candidates
    ]

    all_candidates_marked.sort(
        key=lambda item: (
            not item.pareto_optimal,
            (
                float(item.utopia_distance)
                if item.utopia_distance is not None
                else float("inf")
            ),
            item.data_nmse,
            item.operation_cost_per_sample,
        )
    )

    save_csv(
        ALL_CANDIDATES_PATH,
        all_candidates_marked,
    )
    save_csv(
        PARETO_FRONT_PATH,
        pareto,
    )

    create_plot(
        candidates=candidates,
        pareto=pareto,
        selected=selected,
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 16,
        "description": (
            "Multi-objective Pareto optimization over "
            "accuracy, operation cost and block count"
        ),
        "source_pipeline_id": (
            source_package["pipeline_id"]
        ),
        "objectives": [
            {
                "name": "data_nmse",
                "direction": "minimize",
                "scale": "log10 for knee selection",
            },
            {
                "name": (
                    "operation_cost_per_sample"
                ),
                "direction": "minimize",
                "block_costs": (
                    BLOCK_OPERATION_COST
                ),
            },
            {
                "name": "block_count",
                "direction": "minimize",
            },
        ],
        "selection_method": (
            "Equal-weight normalized Euclidean "
            "distance to the utopia point"
        ),
        "selected_solution": (
            asdict(selected)
        ),
        "pareto_front": [
            asdict(item)
            for item in pareto
        ],
        "validation": validation,
        "validations": {
            "all_15_candidates_evaluated": True,
            "pareto_front_nonempty": True,
            "no_dominated_member_in_front": True,
            "minimum_nmse_solution_in_front": True,
            "selected_solution_in_front": True,
            "minimum_utopia_distance_selected": True,
            "selected_pipeline_saved": True,
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
    print("=" * 82)
    print(
        "GENESIS-DSP — ADIM 16 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 82)
    print(
        f"Değerlendirilen aday          : "
        f"{len(candidates)}"
    )
    print(
        f"Pareto çözüm sayısı          : "
        f"{len(pareto)}"
    )
    print(
        f"Seçilen dengeli topoloji     : "
        f"{selected.signature}"
    )
    print(
        f"Seçilen NMSE                 : "
        f"{selected.data_nmse:.12e}"
    )
    print(
        f"İşlem maliyeti / örnek       : "
        f"{selected.operation_cost_per_sample}"
    )
    print(
        f"Seçilen blok sayısı          : "
        f"{selected.block_count}"
    )
    print(
        f"Utopia distance              : "
        f"{float(selected.utopia_distance):.12e}"
    )
    print(
        f"Seçilen pipeline             : "
        f"{SELECTED_PACKAGE_PATH}"
    )
    print(
        f"Pareto CSV                   : "
        f"{PARETO_FRONT_PATH}"
    )
    print(
        f"Grafik                       : "
        f"{PLOT_PATH}"
    )
    print(
        f"Rapor                        : "
        f"{REPORT_PATH}"
    )
    print("=" * 82)


if __name__ == "__main__":
    main()
