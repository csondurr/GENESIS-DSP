"""
GENESIS-DSP — Adım 13
Sürekli parametre optimizasyon motoru.

Bu program:
1. Adım 12 pipeline paketini yükler.
2. Graph içindeki sayısal parametreleri bounds içinde optimize eder.
3. Bounded Differential Evolution kullanır.
4. Sonucu coordinate-search ile hassaslaştırır.
5. Optimize edilmiş pipeline paketini kaydeder.
6. Yakınsama geçmişi, JSON raporu ve grafik üretir.

Bu öz testte optimize edilen parametreler:
- downconvert.frequency_hz
- constant_output.gain_real
- constant_output.gain_imag

Çalıştırma:
    python step13_continuous_optimizer.py
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from step09_dsp_block_interface import SignalFrame
from step10_block_registry import build_default_registry
from step11_pipeline_graph import PipelineGraph
from step12_pipeline_serialization import PipelinePackage


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP12_DIRECTORY = BASE_DIRECTORY / "outputs" / "step12"
STEP13_DIRECTORY = BASE_DIRECTORY / "outputs" / "step13"

INPUT_PACKAGE = STEP12_DIRECTORY / "pipeline_package.json"
OPTIMIZED_PACKAGE = STEP13_DIRECTORY / "optimized_pipeline_package.json"
REPORT_PATH = STEP13_DIRECTORY / "continuous_optimizer_report.json"
HISTORY_PATH = STEP13_DIRECTORY / "optimization_history.csv"
PLOT_PATH = STEP13_DIRECTORY / "optimization_convergence.png"

RANDOM_SEED = 20260729
POPULATION_SIZE = 24
MAX_GENERATIONS = 40
MUTATION_FACTOR = 0.75
CROSSOVER_PROBABILITY = 0.90
OBJECTIVE_STOP = 1e-10
COORDINATE_MAX_ITERATIONS = 100

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParameterBinding:
    node_id: str
    parameter_name: str
    minimum: float
    maximum: float

    def validate(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id boş olamaz.")

        if not self.parameter_name.strip():
            raise ValueError("parameter_name boş olamaz.")

        if not np.isfinite(self.minimum):
            raise ValueError("minimum sonlu olmalıdır.")

        if not np.isfinite(self.maximum):
            raise ValueError("maximum sonlu olmalıdır.")

        if self.minimum >= self.maximum:
            raise ValueError("minimum, maximum değerinden küçük olmalıdır.")


@dataclass(frozen=True)
class OptimizationResult:
    best_vector: FloatArray
    best_objective: float
    evaluations: int
    generations: int
    converged: bool
    history: list[dict[str, float]]


class GraphParameterObjective:
    """Graph parametre vektörünü PipelineGraph üzerinde değerlendiren objective."""

    def __init__(
        self,
        base_graph_config: dict[str, Any],
        bindings: list[ParameterBinding],
        input_frame: SignalFrame,
        target_node_id: str,
        target_samples: NDArray[np.complex128],
    ) -> None:
        self.base_graph_config = copy.deepcopy(base_graph_config)
        self.bindings = list(bindings)
        self.input_frame = input_frame
        self.target_node_id = target_node_id
        self.target_samples = np.asarray(
            target_samples,
            dtype=np.complex128,
        )
        self.registry = build_default_registry()
        self.evaluations = 0
        self.cache: dict[tuple[float, ...], float] = {}

        self.input_frame.validate()

        if self.target_samples.ndim != 1:
            raise ValueError("target_samples tek boyutlu olmalıdır.")

        if len(self.target_samples) == 0:
            raise ValueError("target_samples boş olamaz.")

        for binding in self.bindings:
            binding.validate()

        self._validate_bindings_exist()

    def _validate_bindings_exist(self) -> None:
        node_map = {
            str(node["node_id"]): node
            for node in self.base_graph_config["nodes"]
        }

        for binding in self.bindings:
            if binding.node_id not in node_map:
                raise ValueError(
                    f"Graph içinde node bulunamadı: {binding.node_id}"
                )

            parameters = node_map[binding.node_id].setdefault(
                "parameters",
                {},
            )

            if binding.parameter_name not in parameters:
                raise ValueError(
                    f"Parametre bulunamadı: "
                    f"{binding.node_id}.{binding.parameter_name}"
                )

    def bounds(self) -> tuple[FloatArray, FloatArray]:
        lower = np.asarray(
            [binding.minimum for binding in self.bindings],
            dtype=np.float64,
        )
        upper = np.asarray(
            [binding.maximum for binding in self.bindings],
            dtype=np.float64,
        )
        return lower, upper

    def vector_to_graph_config(
        self,
        vector: FloatArray,
    ) -> dict[str, Any]:
        vector = np.asarray(vector, dtype=np.float64)

        if vector.shape != (len(self.bindings),):
            raise ValueError("Parametre vektörü boyutu hatalı.")

        graph_config = copy.deepcopy(self.base_graph_config)
        node_map = {
            str(node["node_id"]): node
            for node in graph_config["nodes"]
        }

        for value, binding in zip(
            vector,
            self.bindings,
            strict=True,
        ):
            node_map[binding.node_id]["parameters"][
                binding.parameter_name
            ] = float(value)

        return graph_config

    def __call__(
        self,
        vector: FloatArray,
    ) -> float:
        vector = np.asarray(vector, dtype=np.float64)
        lower, upper = self.bounds()

        if vector.shape != lower.shape:
            raise ValueError("Objective vektör boyutu hatalı.")

        if np.any(vector < lower) or np.any(vector > upper):
            return float("inf")

        cache_key = tuple(
            float(round(value, 12))
            for value in vector
        )

        if cache_key in self.cache:
            return self.cache[cache_key]

        graph_config = self.vector_to_graph_config(vector)
        graph = PipelineGraph.from_config(graph_config)

        execution = graph.execute(
            registry=self.registry,
            input_frame=self.input_frame,
        )

        measured = execution.output(
            self.target_node_id
        ).samples

        if measured.shape != self.target_samples.shape:
            raise RuntimeError(
                "Objective çıkışı ile target boyutu eşleşmiyor."
            )

        target_energy = float(
            np.sum(np.abs(self.target_samples) ** 2)
        )

        if target_energy <= 0.0:
            raise RuntimeError("Target enerjisi pozitif olmalıdır.")

        error_energy = float(
            np.sum(
                np.abs(
                    measured - self.target_samples
                ) ** 2
            )
        )

        objective = error_energy / target_energy

        if not np.isfinite(objective):
            objective = float("inf")

        self.evaluations += 1
        self.cache[cache_key] = float(objective)

        return float(objective)


class BoundedDifferentialEvolution:
    """Bounds destekli deterministik Differential Evolution optimizasyonu."""

    def __init__(
        self,
        population_size: int,
        max_generations: int,
        mutation_factor: float,
        crossover_probability: float,
        random_seed: int,
        objective_stop: float,
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size en az 4 olmalıdır.")

        if max_generations <= 0:
            raise ValueError("max_generations pozitif olmalıdır.")

        if not 0.0 < mutation_factor <= 2.0:
            raise ValueError("mutation_factor 0 ile 2 arasında olmalıdır.")

        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError(
                "crossover_probability 0 ile 1 arasında olmalıdır."
            )

        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_factor = mutation_factor
        self.crossover_probability = crossover_probability
        self.random_seed = random_seed
        self.objective_stop = objective_stop

    def optimize(
        self,
        objective: Callable[[FloatArray], float],
        lower_bounds: FloatArray,
        upper_bounds: FloatArray,
    ) -> OptimizationResult:
        lower = np.asarray(lower_bounds, dtype=np.float64)
        upper = np.asarray(upper_bounds, dtype=np.float64)

        if lower.shape != upper.shape:
            raise ValueError("Bounds boyutları eşleşmiyor.")

        if lower.ndim != 1 or len(lower) == 0:
            raise ValueError("Bounds tek boyutlu ve boş olmayan olmalıdır.")

        if np.any(lower >= upper):
            raise ValueError("Her lower bound, upper bound'dan küçük olmalıdır.")

        rng = np.random.default_rng(self.random_seed)
        dimension = len(lower)

        population = rng.uniform(
            low=lower,
            high=upper,
            size=(self.population_size, dimension),
        )

        population[0] = (lower + upper) / 2.0

        scores = np.asarray(
            [objective(candidate) for candidate in population],
            dtype=np.float64,
        )

        history: list[dict[str, float]] = []
        converged = False
        completed_generations = 0

        for generation in range(self.max_generations + 1):
            best_index = int(np.argmin(scores))
            best_score = float(scores[best_index])

            history.append(
                {
                    "stage": 0.0,
                    "iteration": float(generation),
                    "best_objective": best_score,
                    "median_objective": float(np.median(scores)),
                }
            )

            completed_generations = generation

            if best_score <= self.objective_stop:
                converged = True
                break

            if generation == self.max_generations:
                break

            for candidate_index in range(self.population_size):
                available = np.delete(
                    np.arange(self.population_size),
                    candidate_index,
                )

                a_index, b_index, c_index = rng.choice(
                    available,
                    size=3,
                    replace=False,
                )

                mutant = (
                    population[a_index]
                    + self.mutation_factor
                    * (
                        population[b_index]
                        - population[c_index]
                    )
                )
                mutant = np.clip(mutant, lower, upper)

                crossover_mask = (
                    rng.random(dimension)
                    < self.crossover_probability
                )
                crossover_mask[
                    rng.integers(0, dimension)
                ] = True

                trial = np.where(
                    crossover_mask,
                    mutant,
                    population[candidate_index],
                )

                trial_score = objective(trial)

                if trial_score <= scores[candidate_index]:
                    population[candidate_index] = trial
                    scores[candidate_index] = trial_score

        best_index = int(np.argmin(scores))

        return OptimizationResult(
            best_vector=population[best_index].copy(),
            best_objective=float(scores[best_index]),
            evaluations=0,
            generations=completed_generations,
            converged=converged,
            history=history,
        )


def coordinate_polish(
    objective: Callable[[FloatArray], float],
    initial_vector: FloatArray,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    max_iterations: int,
) -> tuple[FloatArray, float, list[dict[str, float]]]:
    """DE sonucunu bounded coordinate search ile hassaslaştırır."""

    current = np.asarray(
        initial_vector,
        dtype=np.float64,
    ).copy()
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)

    current = np.clip(current, lower, upper)
    current_score = float(objective(current))

    steps = 0.02 * (upper - lower)
    minimum_steps = np.maximum(
        1e-9,
        1e-8 * (upper - lower),
    )

    history: list[dict[str, float]] = []

    for iteration in range(max_iterations):
        improved = False

        for dimension in range(len(current)):
            for direction in (-1.0, 1.0):
                candidate = current.copy()
                candidate[dimension] += (
                    direction * steps[dimension]
                )
                candidate = np.clip(
                    candidate,
                    lower,
                    upper,
                )

                candidate_score = float(
                    objective(candidate)
                )

                if candidate_score < current_score:
                    current = candidate
                    current_score = candidate_score
                    improved = True

        history.append(
            {
                "stage": 1.0,
                "iteration": float(iteration),
                "best_objective": current_score,
                "median_objective": current_score,
            }
        )

        if current_score <= OBJECTIVE_STOP:
            break

        if not improved:
            steps *= 0.5

        if np.all(steps <= minimum_steps):
            break

    return current, current_score, history


def build_test_problem(
    graph_config: dict[str, Any],
) -> tuple[
    GraphParameterObjective,
    list[ParameterBinding],
    dict[str, float],
]:
    """Bilinen optimuma sahip deterministik graph optimizasyon problemi kurar."""

    sample_rate_hz = 1_000_000.0
    number_of_samples = 2048
    tone_frequency_hz = 62_500.0
    injected_dc = complex(0.18, -0.09)
    target_gain = complex(0.50, 0.25)

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

    input_frame = SignalFrame(
        samples=(tone + injected_dc).astype(np.complex128),
        sample_rate_hz=sample_rate_hz,
        metadata={
            "source": "step13_optimizer_self_test",
            "tone_frequency_hz": tone_frequency_hz,
        },
    )

    target_samples = np.full(
        number_of_samples,
        target_gain,
        dtype=np.complex128,
    )

    bindings = [
        ParameterBinding(
            node_id="downconvert",
            parameter_name="frequency_hz",
            minimum=-70_000.0,
            maximum=-55_000.0,
        ),
        ParameterBinding(
            node_id="constant_output",
            parameter_name="gain_real",
            minimum=0.0,
            maximum=1.0,
        ),
        ParameterBinding(
            node_id="constant_output",
            parameter_name="gain_imag",
            minimum=-0.50,
            maximum=0.75,
        ),
    ]

    objective = GraphParameterObjective(
        base_graph_config=graph_config,
        bindings=bindings,
        input_frame=input_frame,
        target_node_id="constant_output",
        target_samples=target_samples,
    )

    truth = {
        "frequency_hz": -tone_frequency_hz,
        "gain_real": float(target_gain.real),
        "gain_imag": float(target_gain.imag),
    }

    return objective, bindings, truth


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
                "stage",
                "iteration",
                "best_objective",
                "median_objective",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def create_plot(
    history: list[dict[str, float]],
) -> None:
    objective_values = np.asarray(
        [
            max(item["best_objective"], 1e-18)
            for item in history
        ],
        dtype=np.float64,
    )

    x_axis = np.arange(
        len(objective_values),
        dtype=np.int64,
    )

    figure = plt.figure(figsize=(11, 6))
    axis = figure.add_subplot(1, 1, 1)
    axis.semilogy(
        x_axis,
        objective_values,
    )
    axis.set_title("Sürekli Parametre Optimizasyonu Yakınsaması")
    axis.set_xlabel("Kayıtlı iterasyon")
    axis.set_ylabel("En iyi objective (NMSE)")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    STEP13_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 12 pipeline paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python step12_pipeline_serialization.py"
        )

    base_graph, package_document = PipelinePackage.load(
        INPUT_PACKAGE
    )
    base_graph_config = base_graph.to_config()

    objective, bindings, truth = build_test_problem(
        base_graph_config
    )
    lower_bounds, upper_bounds = objective.bounds()

    optimizer = BoundedDifferentialEvolution(
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        mutation_factor=MUTATION_FACTOR,
        crossover_probability=CROSSOVER_PROBABILITY,
        random_seed=RANDOM_SEED,
        objective_stop=OBJECTIVE_STOP,
    )

    de_result = optimizer.optimize(
        objective=objective,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    polished_vector, polished_objective, polish_history = (
        coordinate_polish(
            objective=objective,
            initial_vector=de_result.best_vector,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            max_iterations=COORDINATE_MAX_ITERATIONS,
        )
    )

    complete_history = (
        de_result.history + polish_history
    )

    optimized_config = objective.vector_to_graph_config(
        polished_vector
    )
    optimized_graph = PipelineGraph.from_config(
        optimized_config
    )

    PipelinePackage.save(
        path=OPTIMIZED_PACKAGE,
        graph=optimized_graph,
        pipeline_id="step13-optimized-pipeline",
        description=(
            "Adım 13 sürekli parametre optimizasyonuyla "
            "oluşturulan pipeline."
        ),
    )

    estimated_values = {
        binding.parameter_name: float(value)
        for binding, value in zip(
            bindings,
            polished_vector,
            strict=True,
        )
    }

    frequency_error_hz = (
        estimated_values["frequency_hz"]
        - truth["frequency_hz"]
    )
    gain_real_error = (
        estimated_values["gain_real"]
        - truth["gain_real"]
    )
    gain_imag_error = (
        estimated_values["gain_imag"]
        - truth["gain_imag"]
    )

    if abs(frequency_error_hz) > 5.0:
        raise RuntimeError(
            "Optimize edilen frekans hatası 5 Hz sınırını aştı."
        )

    if abs(gain_real_error) > 0.01:
        raise RuntimeError(
            "Optimize edilen gain_real hatası sınırı aştı."
        )

    if abs(gain_imag_error) > 0.01:
        raise RuntimeError(
            "Optimize edilen gain_imag hatası sınırı aştı."
        )

    if polished_objective > 1e-5:
        raise RuntimeError(
            "Sürekli optimizer objective başarı eşiğine ulaşamadı."
        )

    save_history(
        complete_history
    )
    create_plot(
        complete_history
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 13,
        "description": (
            "Bounded continuous graph parameter optimization "
            "using Differential Evolution and coordinate polishing"
        ),
        "source_pipeline_id": package_document["pipeline_id"],
        "optimizer": {
            "algorithm": "Differential Evolution + Coordinate Search",
            "random_seed": RANDOM_SEED,
            "population_size": POPULATION_SIZE,
            "maximum_generations": MAX_GENERATIONS,
            "mutation_factor": MUTATION_FACTOR,
            "crossover_probability": CROSSOVER_PROBABILITY,
            "objective_stop": OBJECTIVE_STOP,
            "objective_evaluations": objective.evaluations,
        },
        "bindings": [
            asdict(binding)
            for binding in bindings
        ],
        "truth": truth,
        "estimated": estimated_values,
        "errors": {
            "frequency_error_hz": frequency_error_hz,
            "gain_real_error": gain_real_error,
            "gain_imag_error": gain_imag_error,
        },
        "result": {
            "de_best_objective": de_result.best_objective,
            "final_objective": polished_objective,
            "de_generations": de_result.generations,
            "de_converged": de_result.converged,
            "history_records": len(complete_history),
        },
        "validations": {
            "frequency_error_below_5_hz": True,
            "gain_real_error_below_0_01": True,
            "gain_imag_error_below_0_01": True,
            "objective_below_1e_5": True,
            "optimized_pipeline_saved": True,
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
    print("=" * 78)
    print("GENESIS-DSP — ADIM 13 BAŞARIYLA TAMAMLANDI")
    print("=" * 78)
    print(
        f"Objective değerlendirmesi    : "
        f"{objective.evaluations}"
    )
    print(
        f"Final objective              : "
        f"{polished_objective:.12e}"
    )
    print(
        f"Gerçek frekans parametresi   : "
        f"{truth['frequency_hz']:,.6f} Hz"
    )
    print(
        f"Optimize edilen frekans      : "
        f"{estimated_values['frequency_hz']:,.6f} Hz"
    )
    print(
        f"Frekans hatası               : "
        f"{frequency_error_hz:+.6f} Hz"
    )
    print(
        f"Gerçek kompleks gain         : "
        f"{truth['gain_real']:+.6f} "
        f"{truth['gain_imag']:+.6f}j"
    )
    print(
        f"Optimize edilen gain         : "
        f"{estimated_values['gain_real']:+.6f} "
        f"{estimated_values['gain_imag']:+.6f}j"
    )
    print(
        f"Optimize pipeline            : "
        f"{OPTIMIZED_PACKAGE}"
    )
    print(
        f"Yakınsama geçmişi            : "
        f"{HISTORY_PATH}"
    )
    print(
        f"Grafik                       : "
        f"{PLOT_PATH}"
    )
    print(
        f"Rapor                        : "
        f"{REPORT_PATH}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
