

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from step09_dsp_block_interface import SignalFrame
from step10_block_registry import build_default_registry
from step11_pipeline_graph import PipelineGraph
from step12_pipeline_serialization import PipelinePackage


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP13_DIRECTORY = BASE_DIRECTORY / "outputs" / "step13"
STEP14_DIRECTORY = BASE_DIRECTORY / "outputs" / "step14"
STEP15_DIRECTORY = BASE_DIRECTORY / "outputs" / "step15"

INPUT_OPTIMIZED_PACKAGE = (
    STEP13_DIRECTORY / "optimized_pipeline_package.json"
)
INPUT_EXHAUSTIVE_REPORT = (
    STEP14_DIRECTORY / "topology_search_report.json"
)

BEST_PACKAGE_PATH = (
    STEP15_DIRECTORY / "evolutionary_best_pipeline.json"
)
REPORT_PATH = (
    STEP15_DIRECTORY / "evolutionary_discovery_report.json"
)
HISTORY_PATH = (
    STEP15_DIRECTORY / "evolution_history.csv"
)
FINAL_POPULATION_PATH = (
    STEP15_DIRECTORY / "final_population.json"
)
PLOT_PATH = (
    STEP15_DIRECTORY / "evolution_overview.png"
)

RANDOM_SEED = 20260730
POPULATION_SIZE = 10
MAX_GENERATIONS = 30
ELITE_COUNT = 2
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.90
MUTATION_PROBABILITY = 0.65
STALL_LIMIT = 10

COMPLEXITY_PENALTY_PER_BLOCK = 1e-6

TEST_SAMPLE_RATE_HZ = 1_000_000.0
TEST_NUMBER_OF_SAMPLES = 2048
TRUE_TONE_FREQUENCY_HZ = 62_500.0
TEST_DC_OFFSET = complex(0.18, -0.09)
TRUE_TARGET_GAIN = complex(0.50, 0.25)


@dataclass(frozen=True)
class BlockTemplate:
    node_id: str
    block_id: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class EvaluatedGenome:
    genome: tuple[str, ...]
    data_nmse: float
    complexity_penalty: float
    total_objective: float

    @property
    def signature(self) -> str:
        return " -> ".join(self.genome)


@dataclass(frozen=True)
class EvolutionResult:
    best: EvaluatedGenome
    generations_completed: int
    unique_evaluations: int
    history: list[dict[str, float]]
    final_population: list[EvaluatedGenome]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON dosyası bulunamadı: {path}")

    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise TypeError("JSON kökü dict olmalıdır.")

    return document


def save_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            document,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )


def load_templates(
    graph: PipelineGraph,
) -> list[BlockTemplate]:
    required_node_ids = (
        "remove_dc",
        "downconvert",
        "constant_output",
    )

    graph_config = graph.to_config()
    node_map = {
        str(node["node_id"]): node
        for node in graph_config["nodes"]
    }

    missing = [
        node_id
        for node_id in required_node_ids
        if node_id not in node_map
    ]

    if missing:
        raise ValueError(
            "Gerekli bloklar optimize pipeline içinde bulunamadı: "
            + ", ".join(missing)
        )

    return [
        BlockTemplate(
            node_id=node_id,
            block_id=str(node_map[node_id]["block_id"]),
            parameters=dict(
                node_map[node_id].get("parameters", {})
            ),
        )
        for node_id in required_node_ids
    ]


def build_test_frame() -> SignalFrame:
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
        samples=(tone + TEST_DC_OFFSET).astype(np.complex128),
        sample_rate_hz=TEST_SAMPLE_RATE_HZ,
        metadata={
            "source": "step15_evolutionary_discovery",
            "tone_frequency_hz": TRUE_TONE_FREQUENCY_HZ,
        },
    )
    frame.validate()

    return frame


def build_target_samples() -> np.ndarray:
    return np.full(
        TEST_NUMBER_OF_SAMPLES,
        TRUE_TARGET_GAIN,
        dtype=np.complex128,
    )


def genome_to_graph(
    genome: tuple[str, ...],
    template_map: dict[str, BlockTemplate],
) -> PipelineGraph:
    if not genome:
        raise ValueError("Genom boş olamaz.")

    if len(set(genome)) != len(genome):
        raise ValueError("Genom duplicate node içeremez.")

    unknown = set(genome) - set(template_map)

    if unknown:
        raise ValueError(
            "Genom bilinmeyen node içeriyor: "
            + ", ".join(sorted(unknown))
        )

    nodes = [
        {
            "node_id": template_map[node_id].node_id,
            "block_id": template_map[node_id].block_id,
            "parameters": dict(
                template_map[node_id].parameters
            ),
        }
        for node_id in genome
    ]

    edges = [
        {
            "source": genome[index],
            "target": genome[index + 1],
        }
        for index in range(len(genome) - 1)
    ]

    return PipelineGraph.from_config(
        {
            "schema_name": "GENESIS-DSP PipelineGraph",
            "schema_version": "1.0.0",
            "nodes": nodes,
            "edges": edges,
        }
    )


def calculate_nmse(
    reference: np.ndarray,
    measured: np.ndarray,
) -> float:
    if reference.shape != measured.shape:
        raise ValueError("reference ve measured boyutları eşleşmiyor.")

    denominator = float(np.sum(np.abs(reference) ** 2))

    if denominator <= 0.0:
        raise ValueError("Referans enerjisi pozitif olmalıdır.")

    numerator = float(
        np.sum(np.abs(measured - reference) ** 2)
    )

    return numerator / denominator


class TopologyObjective:
    def __init__(
        self,
        templates: list[BlockTemplate],
        input_frame: SignalFrame,
        target_samples: np.ndarray,
    ) -> None:
        self.template_map = {
            template.node_id: template
            for template in templates
        }
        self.available_nodes = tuple(
            self.template_map
        )
        self.input_frame = input_frame
        self.target_samples = np.asarray(
            target_samples,
            dtype=np.complex128,
        )
        self.registry = build_default_registry()
        self.cache: dict[
            tuple[str, ...],
            EvaluatedGenome,
        ] = {}

    def evaluate(
        self,
        genome: tuple[str, ...],
    ) -> EvaluatedGenome:
        if genome in self.cache:
            return self.cache[genome]

        graph = genome_to_graph(
            genome,
            self.template_map,
        )

        result = graph.execute(
            registry=self.registry,
            input_frame=self.input_frame,
        )

        if len(result.leaf_nodes) != 1:
            raise RuntimeError(
                "Evrimsel adayın tek leaf node'u olmalıdır."
            )

        output = result.output(
            result.leaf_nodes[0]
        ).samples

        data_nmse = calculate_nmse(
            self.target_samples,
            output,
        )

        complexity_penalty = (
            COMPLEXITY_PENALTY_PER_BLOCK
            * len(genome)
        )

        evaluated = EvaluatedGenome(
            genome=genome,
            data_nmse=data_nmse,
            complexity_penalty=complexity_penalty,
            total_objective=(
                data_nmse
                + complexity_penalty
            ),
        )

        self.cache[genome] = evaluated
        return evaluated


class EvolutionaryTopologySearch:
    def __init__(
        self,
        available_nodes: tuple[str, ...],
        random_seed: int,
    ) -> None:
        if len(available_nodes) < 2:
            raise ValueError(
                "En az iki kullanılabilir node bulunmalıdır."
            )

        if len(set(available_nodes)) != len(available_nodes):
            raise ValueError(
                "available_nodes duplicate içeremez."
            )

        if POPULATION_SIZE <= ELITE_COUNT:
            raise ValueError(
                "POPULATION_SIZE, ELITE_COUNT değerinden büyük olmalıdır."
            )

        self.available_nodes = available_nodes
        self.rng = np.random.default_rng(random_seed)

    def random_genome(self) -> tuple[str, ...]:
        length = int(
            self.rng.integers(
                1,
                len(self.available_nodes) + 1,
            )
        )

        selected = self.rng.choice(
            np.asarray(self.available_nodes),
            size=length,
            replace=False,
        )

        return tuple(str(item) for item in selected)

    def initialize_population(
        self,
    ) -> list[tuple[str, ...]]:
        population: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()

        maximum_unique_attempts = 1000
        attempts = 0

        while (
            len(population) < POPULATION_SIZE
            and attempts < maximum_unique_attempts
        ):
            attempts += 1
            genome = self.random_genome()

            if genome in seen:
                continue

            seen.add(genome)
            population.append(genome)

        while len(population) < POPULATION_SIZE:
            population.append(
                self.random_genome()
            )

        return population

    def tournament_select(
        self,
        evaluated_population: list[EvaluatedGenome],
    ) -> EvaluatedGenome:
        indices = self.rng.choice(
            len(evaluated_population),
            size=min(
                TOURNAMENT_SIZE,
                len(evaluated_population),
            ),
            replace=False,
        )

        competitors = [
            evaluated_population[int(index)]
            for index in indices
        ]

        return min(
            competitors,
            key=lambda item: (
                item.total_objective,
                item.signature,
            ),
        )

    def crossover(
        self,
        parent_a: tuple[str, ...],
        parent_b: tuple[str, ...],
    ) -> tuple[str, ...]:
        if self.rng.random() > CROSSOVER_PROBABILITY:
            return parent_a

        active_nodes: list[str] = []

        for node_id in self.available_nodes:
            in_a = node_id in parent_a
            in_b = node_id in parent_b

            if in_a and in_b:
                include = True
            elif in_a or in_b:
                include = bool(
                    self.rng.random() < 0.5
                )
            else:
                include = False

            if include:
                active_nodes.append(node_id)

        if not active_nodes:
            active_nodes.append(
                str(
                    self.rng.choice(
                        np.asarray(self.available_nodes)
                    )
                )
            )

        rank_a = {
            node_id: index
            for index, node_id in enumerate(parent_a)
        }
        rank_b = {
            node_id: index
            for index, node_id in enumerate(parent_b)
        }

        large_rank = len(self.available_nodes) + 1

        decorated: list[
            tuple[float, float, str]
        ] = []

        for node_id in active_nodes:
            a_rank = rank_a.get(
                node_id,
                large_rank,
            )
            b_rank = rank_b.get(
                node_id,
                large_rank,
            )

            combined_rank = (
                0.5 * float(a_rank)
                + 0.5 * float(b_rank)
            )
            jitter = float(
                self.rng.uniform(
                    -0.20,
                    0.20,
                )
            )

            decorated.append(
                (
                    combined_rank + jitter,
                    float(self.rng.random()),
                    node_id,
                )
            )

        decorated.sort()

        return tuple(
            item[2]
            for item in decorated
        )

    def mutate(
        self,
        genome: tuple[str, ...],
    ) -> tuple[str, ...]:
        if self.rng.random() > MUTATION_PROBABILITY:
            return genome

        mutable = list(genome)
        operation = str(
            self.rng.choice(
                np.asarray(
                    [
                        "swap",
                        "insert",
                        "remove",
                        "move",
                    ]
                )
            )
        )

        if operation == "swap" and len(mutable) >= 2:
            first, second = self.rng.choice(
                len(mutable),
                size=2,
                replace=False,
            )
            first_index = int(first)
            second_index = int(second)
            mutable[first_index], mutable[second_index] = (
                mutable[second_index],
                mutable[first_index],
            )

        elif operation == "insert":
            inactive = [
                node_id
                for node_id in self.available_nodes
                if node_id not in mutable
            ]

            if inactive:
                node_id = str(
                    self.rng.choice(
                        np.asarray(inactive)
                    )
                )
                position = int(
                    self.rng.integers(
                        0,
                        len(mutable) + 1,
                    )
                )
                mutable.insert(
                    position,
                    node_id,
                )

        elif operation == "remove" and len(mutable) > 1:
            position = int(
                self.rng.integers(
                    0,
                    len(mutable),
                )
            )
            mutable.pop(position)

        elif operation == "move" and len(mutable) >= 2:
            source = int(
                self.rng.integers(
                    0,
                    len(mutable),
                )
            )
            node_id = mutable.pop(source)
            target = int(
                self.rng.integers(
                    0,
                    len(mutable) + 1,
                )
            )
            mutable.insert(
                target,
                node_id,
            )

        return tuple(mutable)

    def run(
        self,
        objective: TopologyObjective,
    ) -> EvolutionResult:
        population = self.initialize_population()
        history: list[dict[str, float]] = []

        global_best: EvaluatedGenome | None = None
        stall_generations = 0
        generations_completed = 0

        for generation in range(MAX_GENERATIONS + 1):
            evaluated = [
                objective.evaluate(genome)
                for genome in population
            ]

            evaluated.sort(
                key=lambda item: (
                    item.total_objective,
                    item.signature,
                )
            )

            generation_best = evaluated[0]

            if (
                global_best is None
                or generation_best.total_objective
                < global_best.total_objective - 1e-18
            ):
                global_best = generation_best
                stall_generations = 0
            else:
                stall_generations += 1

            unique_genomes = len(
                set(population)
            )

            history.append(
                {
                    "generation": float(generation),
                    "best_objective": float(
                        generation_best.total_objective
                    ),
                    "median_objective": float(
                        np.median(
                            [
                                item.total_objective
                                for item in evaluated
                            ]
                        )
                    ),
                    "best_block_count": float(
                        len(generation_best.genome)
                    ),
                    "population_unique_genomes": float(
                        unique_genomes
                    ),
                    "cumulative_unique_evaluations": float(
                        len(objective.cache)
                    ),
                }
            )

            generations_completed = generation

            if generation == MAX_GENERATIONS:
                population = [
                    item.genome
                    for item in evaluated
                ]
                break

            if stall_generations >= STALL_LIMIT:
                population = [
                    item.genome
                    for item in evaluated
                ]
                break

            next_population: list[
                tuple[str, ...]
            ] = [
                item.genome
                for item in evaluated[:ELITE_COUNT]
            ]

            while len(next_population) < POPULATION_SIZE:
                parent_a = self.tournament_select(
                    evaluated
                )
                parent_b = self.tournament_select(
                    evaluated
                )

                child = self.crossover(
                    parent_a.genome,
                    parent_b.genome,
                )
                child = self.mutate(
                    child
                )

                next_population.append(
                    child
                )

            population = next_population

        if global_best is None:
            raise RuntimeError(
                "Evolutionary search bir sonuç üretemedi."
            )

        final_evaluated = [
            objective.evaluate(genome)
            for genome in population
        ]
        final_evaluated.sort(
            key=lambda item: (
                item.total_objective,
                item.signature,
            )
        )

        return EvolutionResult(
            best=global_best,
            generations_completed=generations_completed,
            unique_evaluations=len(objective.cache),
            history=history,
            final_population=final_evaluated,
        )


def save_history(
    history: list[dict[str, float]],
) -> None:
    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "generation",
                "best_objective",
                "median_objective",
                "best_block_count",
                "population_unique_genomes",
                "cumulative_unique_evaluations",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def create_plot(
    history: list[dict[str, float]],
) -> None:
    generations = np.asarray(
        [
            item["generation"]
            for item in history
        ],
        dtype=np.float64,
    )

    best_values = np.asarray(
        [
            max(
                item["best_objective"],
                1e-18,
            )
            for item in history
        ],
        dtype=np.float64,
    )

    median_values = np.asarray(
        [
            max(
                item["median_objective"],
                1e-18,
            )
            for item in history
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 10),
    )

    axes[0].semilogy(
        generations,
        best_values,
        label="En iyi objective",
    )
    axes[0].semilogy(
        generations,
        median_values,
        label="Medyan objective",
    )
    axes[0].set_title(
        "Evrimsel Topoloji Keşfi Yakınsaması"
    )
    axes[0].set_xlabel("Nesil")
    axes[0].set_ylabel("Objective")
    axes[0].grid(True)
    axes[0].legend()

    unique_evaluations = np.asarray(
        [
            item["cumulative_unique_evaluations"]
            for item in history
        ],
        dtype=np.float64,
    )

    population_diversity = np.asarray(
        [
            item["population_unique_genomes"]
            for item in history
        ],
        dtype=np.float64,
    )

    axes[1].plot(
        generations,
        unique_evaluations,
        label="Kümülatif benzersiz değerlendirme",
    )
    axes[1].plot(
        generations,
        population_diversity,
        label="Popülasyon çeşitliliği",
    )
    axes[1].set_title(
        "Arama Kapsamı ve Popülasyon Çeşitliliği"
    )
    axes[1].set_xlabel("Nesil")
    axes[1].set_ylabel("Adet")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        PLOT_PATH,
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    STEP15_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_OPTIMIZED_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 13 optimize pipeline bulunamadı: "
            f"{INPUT_OPTIMIZED_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python step13_continuous_optimizer.py"
        )

    if not INPUT_EXHAUSTIVE_REPORT.exists():
        raise FileNotFoundError(
            f"Adım 14 raporu bulunamadı: "
            f"{INPUT_EXHAUSTIVE_REPORT}\n"
            "Önce şu komutu çalıştır:\n"
            "python step14_topology_search.py"
        )

    source_graph, source_package = (
        PipelinePackage.load(
            INPUT_OPTIMIZED_PACKAGE
        )
    )

    exhaustive_report = load_json(
        INPUT_EXHAUSTIVE_REPORT
    )

    exhaustive_best_objective = float(
        exhaustive_report[
            "best_topology"
        ]["total_objective"]
    )

    exhaustive_best_data_nmse = float(
        exhaustive_report[
            "best_topology"
        ]["data_nmse"]
    )

    templates = load_templates(
        source_graph
    )
    input_frame = build_test_frame()
    target_samples = build_target_samples()

    objective = TopologyObjective(
        templates=templates,
        input_frame=input_frame,
        target_samples=target_samples,
    )

    engine = EvolutionaryTopologySearch(
        available_nodes=objective.available_nodes,
        random_seed=RANDOM_SEED,
    )

    result = engine.run(
        objective
    )

    template_map = {
        template.node_id: template
        for template in templates
    }

    best_graph = genome_to_graph(
        result.best.genome,
        template_map,
    )

    PipelinePackage.save(
        path=BEST_PACKAGE_PATH,
        graph=best_graph,
        pipeline_id="step15-evolutionary-best",
        description=(
            "Evrimsel topoloji keşif motorunun "
            "bulduğu en iyi DSP pipeline."
        ),
    )

    relative_objective_error = abs(
        result.best.total_objective
        - exhaustive_best_objective
    ) / max(
        exhaustive_best_objective,
        1e-18,
    )

    if relative_objective_error > 1e-9:
        raise RuntimeError(
            "Evrimsel arama exhaustive optimuma ulaşamadı."
        )

    if result.best.data_nmse > exhaustive_best_data_nmse + 1e-12:
        raise RuntimeError(
            "Evrimsel çözümün NMSE değeri exhaustive optimumdan kötü."
        )

    if set(result.best.genome) != set(
        objective.available_nodes
    ):
        raise RuntimeError(
            "Evrimsel optimum gerekli tüm blokları içermiyor."
        )

    save_history(
        result.history
    )
    create_plot(
        result.history
    )

    final_population_document = {
        "schema_name": (
            "GENESIS-DSP EvolutionaryPopulation"
        ),
        "schema_version": "1.0.0",
        "population": [
            asdict(item)
            for item in result.final_population
        ],
    }

    save_json(
        FINAL_POPULATION_PATH,
        final_population_document,
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 15,
        "description": (
            "Evolutionary DSP topology discovery using "
            "tournament selection, crossover, mutation and elitism"
        ),
        "source_pipeline_id": (
            source_package["pipeline_id"]
        ),
        "engine": {
            "random_seed": RANDOM_SEED,
            "population_size": POPULATION_SIZE,
            "maximum_generations": MAX_GENERATIONS,
            "elite_count": ELITE_COUNT,
            "tournament_size": TOURNAMENT_SIZE,
            "crossover_probability": CROSSOVER_PROBABILITY,
            "mutation_probability": MUTATION_PROBABILITY,
            "stall_limit": STALL_LIMIT,
            "complexity_penalty_per_block": (
                COMPLEXITY_PENALTY_PER_BLOCK
            ),
        },
        "result": {
            "best": asdict(result.best),
            "best_signature": result.best.signature,
            "generations_completed": (
                result.generations_completed
            ),
            "unique_topologies_evaluated": (
                result.unique_evaluations
            ),
        },
        "exhaustive_reference": {
            "best_objective": exhaustive_best_objective,
            "best_data_nmse": exhaustive_best_data_nmse,
            "relative_objective_error": (
                relative_objective_error
            ),
        },
        "validations": {
            "exhaustive_optimum_reached": True,
            "all_required_blocks_discovered": True,
            "best_pipeline_saved": True,
            "deterministic_seed_used": True,
        },
    }

    save_json(
        REPORT_PATH,
        report,
    )

    print()
    print("=" * 82)
    print(
        "GENESIS-DSP — ADIM 15 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 82)
    print(
        f"Tamamlanan nesil              : "
        f"{result.generations_completed}"
    )
    print(
        f"Benzersiz topoloji ölçümü     : "
        f"{result.unique_evaluations}"
    )
    print(
        f"Keşfedilen en iyi topoloji    : "
        f"{result.best.signature}"
    )
    print(
        f"En iyi veri NMSE              : "
        f"{result.best.data_nmse:.12e}"
    )
    print(
        f"En iyi toplam objective       : "
        f"{result.best.total_objective:.12e}"
    )
    print(
        f"Exhaustive optimum            : "
        f"{exhaustive_best_objective:.12e}"
    )
    print(
        f"Optimum göreli farkı          : "
        f"{relative_objective_error:.3e}"
    )
    print(
        "Exhaustive optimum doğrulaması: BAŞARILI"
    )
    print(
        f"En iyi pipeline               : "
        f"{BEST_PACKAGE_PATH}"
    )
    print(
        f"Nesil geçmişi                 : "
        f"{HISTORY_PATH}"
    )
    print(
        f"Final popülasyon              : "
        f"{FINAL_POPULATION_PATH}"
    )
    print(
        f"Grafik                        : "
        f"{PLOT_PATH}"
    )
    print(
        f"Rapor                         : "
        f"{REPORT_PATH}"
    )
    print("=" * 82)


if __name__ == "__main__":
    main()
