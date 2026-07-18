"""
GENESIS-DSP — Adım 20
Kararlılık, nedensellik ve fiziksel/yapısal kısıt doğrulama sistemi.

Bu program:
1. Adım 17'de yeniden keşfedilen pipeline'ı yükler.
2. Mevcut global-ortalama DC removal bloğunun çevrimdışı ve
   non-causal olduğunu prefix testiyle gösterir.
3. Causal ve BIBO-stable bir DC blocker bloğu tanımlar.
4. Pipeline içindeki legacy DC removal bloğunu causal sürümle değiştirir.
5. Blok ve graph seviyesinde:
   - prefix causality
   - BIBO stability
   - determinism
   - finite-output
   - parameter constraints
   - Nyquist constraint
   - cycle ve multi-parent rejection
   testlerini uygular.
6. Sertifikalı pipeline paketi, test matrisi, rapor ve grafik üretir.

Çalıştırma:
    python step20_stability_causality_constraints.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from step09_dsp_block_interface import (
    ComplexGainBlock,
    DCRemovalBlock,
    DSPBlock,
    FrequencyShiftBlock,
    ParameterSpec,
    SignalFrame,
    execute_block,
)
from step10_block_registry import (
    BlockRegistry,
    build_default_registry,
)
from step11_pipeline_graph import PipelineGraph
from step12_pipeline_serialization import PipelinePackage


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP17_DIRECTORY = BASE_DIRECTORY / "outputs" / "step17"
STEP20_DIRECTORY = BASE_DIRECTORY / "outputs" / "step20"

INPUT_PIPELINE_PATH = (
    STEP17_DIRECTORY / "rediscovered_pipeline.json"
)

CERTIFIED_PIPELINE_PATH = (
    STEP20_DIRECTORY / "certified_causal_pipeline.json"
)
REPORT_PATH = (
    STEP20_DIRECTORY / "stability_causality_report.json"
)
CERTIFICATE_PATH = (
    STEP20_DIRECTORY / "constraint_certificate.json"
)
TEST_MATRIX_PATH = (
    STEP20_DIRECTORY / "constraint_test_matrix.csv"
)
PLOT_PATH = (
    STEP20_DIRECTORY / "stability_causality_overview.png"
)

RANDOM_SEED = 20260802
SAMPLE_RATE_HZ = 1_000_000.0
PREFIX_LENGTH = 2048
FULL_LENGTH = 4096
CAUSALITY_TOLERANCE = 1e-12
DETERMINISM_TOLERANCE = 1e-12
STABILITY_TOLERANCE = 1e-9

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class TestRecord:
    test_id: str
    scope: str
    subject: str
    expected: str
    observed_value: float | None
    threshold: float | None
    passed: bool
    note: str


class CausalDCBlockerBlock(DSPBlock):
    """
    Birinci dereceden causal DC blocker.

    Transfer fonksiyonu:
        H(z) = (1 - z^-1) / (1 - r z^-1)

    0 <= r < 1 için:
    - causal
    - BIBO stable
    - impulse-response L1 normu 2
    """

    block_id = "causal_dc_blocker"
    block_name = "Causal DC Blocker"
    category = "correction"

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="pole_radius",
                parameter_type="float",
                default=0.995,
                minimum=0.0,
                maximum=0.999999,
                description=(
                    "DC blocker kutup yarıçapı. "
                    "1 değerinden kesinlikle küçük olmalıdır."
                ),
            ),
        )

    def process(self, frame: SignalFrame) -> SignalFrame:
        pole_radius = float(
            self.parameters["pole_radius"]
        )

        input_samples = frame.samples
        output_samples = np.empty_like(
            input_samples,
            dtype=np.complex128,
        )

        previous_input = complex(0.0, 0.0)
        previous_output = complex(0.0, 0.0)

        for index, current_input in enumerate(
            input_samples
        ):
            current_output = (
                current_input
                - previous_input
                + pole_radius
                * previous_output
            )

            output_samples[index] = (
                current_output
            )
            previous_input = complex(
                current_input
            )
            previous_output = complex(
                current_output
            )

        return frame.with_samples(
            output_samples,
            {
                "dc_blocker_type": (
                    "causal_first_order_iir"
                ),
                "dc_blocker_pole_radius": (
                    pole_radius
                ),
                "theoretical_bibo_linf_bound": (
                    2.0
                ),
            },
        )

    def estimated_mac_count(
        self,
        input_samples: int,
    ) -> int:
        return int(
            3 * input_samples
        )


def build_certified_registry() -> BlockRegistry:
    registry = build_default_registry()
    registry.register(
        CausalDCBlockerBlock
    )
    return registry


def make_frame(
    samples: ComplexArray,
) -> SignalFrame:
    frame = SignalFrame(
        samples=np.asarray(
            samples,
            dtype=np.complex128,
        ),
        sample_rate_hz=SAMPLE_RATE_HZ,
        metadata={
            "source": "step20_certification",
        },
    )
    frame.validate()
    return frame


def build_prefix_test_signals() -> tuple[
    ComplexArray,
    ComplexArray,
]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = np.arange(
        FULL_LENGTH,
        dtype=np.float64,
    )

    tone = 0.45 * np.exp(
        1j
        * 2.0
        * np.pi
        * 37_500.0
        * indices
        / SAMPLE_RATE_HZ
    )

    random_component = 0.15 * (
        rng.uniform(
            -1.0,
            1.0,
            FULL_LENGTH,
        )
        + 1j
        * rng.uniform(
            -1.0,
            1.0,
            FULL_LENGTH,
        )
    )

    full_signal = (
        tone + random_component
    ).astype(np.complex128)

    full_signal[:PREFIX_LENGTH] += complex(
        0.10,
        -0.06,
    )
    full_signal[PREFIX_LENGTH:] += complex(
        1.25,
        0.85,
    )

    prefix_signal = full_signal[
        :PREFIX_LENGTH
    ].copy()

    return prefix_signal, full_signal


def execute_block_factory(
    block_factory: Callable[[], DSPBlock],
    samples: ComplexArray,
) -> ComplexArray:
    output, _ = execute_block(
        block_factory(),
        make_frame(samples),
    )
    return output.samples


def block_prefix_error(
    block_factory: Callable[[], DSPBlock],
    prefix_samples: ComplexArray,
    full_samples: ComplexArray,
) -> float:
    prefix_output = execute_block_factory(
        block_factory,
        prefix_samples,
    )
    full_output = execute_block_factory(
        block_factory,
        full_samples,
    )

    return float(
        np.max(
            np.abs(
                prefix_output
                - full_output[
                    :len(prefix_output)
                ]
            )
        )
    )


def execute_graph_samples(
    graph: PipelineGraph,
    registry: BlockRegistry,
    samples: ComplexArray,
) -> ComplexArray:
    result = graph.execute(
        registry=registry,
        input_frame=make_frame(samples),
    )

    if len(result.leaf_nodes) != 1:
        raise RuntimeError(
            "Sertifikasyon graph'ı tam olarak bir leaf node içermelidir."
        )

    return result.output(
        result.leaf_nodes[0]
    ).samples


def graph_prefix_error(
    graph: PipelineGraph,
    registry: BlockRegistry,
    prefix_samples: ComplexArray,
    full_samples: ComplexArray,
) -> float:
    prefix_output = execute_graph_samples(
        graph,
        registry,
        prefix_samples,
    )
    full_output = execute_graph_samples(
        graph,
        registry,
        full_samples,
    )

    return float(
        np.max(
            np.abs(
                prefix_output
                - full_output[
                    :len(prefix_output)
                ]
            )
        )
    )


def load_legacy_pipeline() -> tuple[
    PipelineGraph,
    dict[str, Any],
]:
    if not INPUT_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Adım 17 pipeline paketi bulunamadı: "
            f"{INPUT_PIPELINE_PATH}\n"
            "Önce şu komutu çalıştır:\n"
            "python step17_algorithm_rediscovery.py"
        )

    return PipelinePackage.load(
        INPUT_PIPELINE_PATH
    )


def repair_pipeline(
    legacy_graph: PipelineGraph,
) -> PipelineGraph:
    configuration = legacy_graph.to_config()
    replacement_count = 0

    for node in configuration["nodes"]:
        if node["block_id"] == "dc_removal":
            node["block_id"] = (
                "causal_dc_blocker"
            )
            node["parameters"] = {
                "pole_radius": 0.995,
            }
            replacement_count += 1

    if replacement_count != 1:
        raise RuntimeError(
            "Legacy pipeline içinde tam olarak bir "
            "dc_removal bloğu bekleniyordu."
        )

    return PipelineGraph.from_config(
        configuration
    )


def block_theoretical_bound(
    block_id: str,
    parameters: dict[str, Any],
) -> float:
    if block_id == "complex_gain":
        return abs(
            complex(
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
        )

    if block_id == "frequency_shift":
        return 1.0

    if block_id == "causal_dc_blocker":
        return 2.0

    raise ValueError(
        f"Teorik bound tanımlanmamış blok: {block_id}"
    )


def graph_theoretical_bound(
    graph: PipelineGraph,
) -> float:
    bound = 1.0

    for node in graph.to_config()["nodes"]:
        bound *= block_theoretical_bound(
            str(node["block_id"]),
            dict(
                node.get(
                    "parameters",
                    {},
                )
            ),
        )

    return float(bound)


def bounded_test_signals() -> dict[str, ComplexArray]:
    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )
    length = 4096
    indices = np.arange(
        length,
        dtype=np.float64,
    )

    impulse = np.zeros(
        length,
        dtype=np.complex128,
    )
    impulse[0] = complex(
        1.0,
        -0.5,
    )

    return {
        "impulse": impulse,
        "step": np.full(
            length,
            complex(0.80, -0.60),
            dtype=np.complex128,
        ),
        "alternating": (
            (0.75 + 0.25j)
            * ((-1.0) ** indices)
        ).astype(np.complex128),
        "bounded_random": (
            0.70
            * (
                rng.uniform(
                    -1.0,
                    1.0,
                    length,
                )
                + 1j
                * rng.uniform(
                    -1.0,
                    1.0,
                    length,
                )
            )
        ).astype(np.complex128),
        "tone_plus_dc": (
            0.65
            * np.exp(
                1j
                * 2.0
                * np.pi
                * 91_000.0
                * indices
                / SAMPLE_RATE_HZ
            )
            + complex(
                0.25,
                -0.15,
            )
        ).astype(np.complex128),
        "large_bounded": np.full(
            length,
            complex(
                1_000_000.0,
                -750_000.0,
            ),
            dtype=np.complex128,
        ),
    }


def maximum_graph_gain(
    graph: PipelineGraph,
    registry: BlockRegistry,
) -> tuple[float, dict[str, float]]:
    gains: dict[str, float] = {}

    for signal_name, samples in (
        bounded_test_signals().items()
    ):
        input_maximum = float(
            np.max(
                np.abs(samples)
            )
        )

        output = execute_graph_samples(
            graph,
            registry,
            samples,
        )

        if not np.all(
            np.isfinite(output.real)
        ) or not np.all(
            np.isfinite(output.imag)
        ):
            raise RuntimeError(
                f"{signal_name} testinde non-finite graph çıktısı bulundu."
            )

        output_maximum = float(
            np.max(
                np.abs(output)
            )
        )

        gain = (
            output_maximum
            / max(
                input_maximum,
                1e-30,
            )
        )
        gains[signal_name] = gain

    return (
        max(
            gains.values()
        ),
        gains,
    )


def expect_exception(
    test_id: str,
    callback: Callable[[], Any],
    exception_type: type[BaseException],
    records: list[TestRecord],
    note: str,
) -> None:
    try:
        callback()
    except exception_type:
        records.append(
            TestRecord(
                test_id=test_id,
                scope="constraint",
                subject=test_id,
                expected=(
                    f"{exception_type.__name__} rejection"
                ),
                observed_value=None,
                threshold=None,
                passed=True,
                note=note,
            )
        )
    else:
        raise RuntimeError(
            f"Kısıt testi başarısız: {test_id}"
        )


def run_constraint_tests(
    records: list[TestRecord],
) -> None:
    reference_frame = make_frame(
        np.ones(
            128,
            dtype=np.complex128,
        )
    )

    expect_exception(
        test_id="nyquist_frequency_rejection",
        callback=lambda: execute_block(
            FrequencyShiftBlock(
                frequency_hz=(
                    SAMPLE_RATE_HZ / 2.0
                ),
            ),
            reference_frame,
        ),
        exception_type=ValueError,
        records=records,
        note=(
            "Nyquist frekansına eşit shift reddedildi."
        ),
    )

    expect_exception(
        test_id="complex_gain_upper_bound",
        callback=lambda: ComplexGainBlock(
            gain_real=1000.01,
            gain_imag=0.0,
        ),
        exception_type=ValueError,
        records=records,
        note=(
            "Tanımlı gain üst sınırı uygulandı."
        ),
    )

    expect_exception(
        test_id="unstable_pole_rejection",
        callback=lambda: CausalDCBlockerBlock(
            pole_radius=1.0,
        ),
        exception_type=ValueError,
        records=records,
        note=(
            "BIBO-stable bölgenin dışındaki kutup reddedildi."
        ),
    )

    expect_exception(
        test_id="nonfinite_input_rejection",
        callback=lambda: make_frame(
            np.asarray(
                [
                    complex(1.0, 0.0),
                    complex(
                        float("nan"),
                        0.0,
                    ),
                ],
                dtype=np.complex128,
            )
        ),
        exception_type=ValueError,
        records=records,
        note=(
            "NaN içeren SignalFrame reddedildi."
        ),
    )

    cycle_config = {
        "nodes": [
            {
                "node_id": "a",
                "block_id": "complex_gain",
                "parameters": {},
            },
            {
                "node_id": "b",
                "block_id": "frequency_shift",
                "parameters": {},
            },
        ],
        "edges": [
            {
                "source": "a",
                "target": "b",
            },
            {
                "source": "b",
                "target": "a",
            },
        ],
    }

    expect_exception(
        test_id="cycle_rejection",
        callback=lambda: PipelineGraph.from_config(
            cycle_config
        ),
        exception_type=ValueError,
        records=records,
        note=(
            "Cyclic DSP graph reddedildi."
        ),
    )

    multi_parent_config = {
        "nodes": [
            {
                "node_id": "a",
                "block_id": "complex_gain",
                "parameters": {},
            },
            {
                "node_id": "b",
                "block_id": "frequency_shift",
                "parameters": {},
            },
            {
                "node_id": "c",
                "block_id": "causal_dc_blocker",
                "parameters": {},
            },
        ],
        "edges": [
            {
                "source": "a",
                "target": "c",
            },
            {
                "source": "b",
                "target": "c",
            },
        ],
    }

    expect_exception(
        test_id="multiple_parent_rejection",
        callback=lambda: PipelineGraph.from_config(
            multi_parent_config
        ),
        exception_type=ValueError,
        records=records,
        note=(
            "Desteklenmeyen multi-parent düğüm reddedildi."
        ),
    )


def save_test_matrix(
    records: list[TestRecord],
) -> None:
    with TEST_MATRIX_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test_id",
                "scope",
                "subject",
                "expected",
                "observed_value",
                "threshold",
                "passed",
                "note",
            ],
        )
        writer.writeheader()

        for record in records:
            writer.writerow(
                asdict(record)
            )


def create_plot(
    legacy_prefix_error: float,
    certified_prefix_error: float,
    measured_gain: float,
    theoretical_gain_bound: float,
    determinism_error: float,
) -> None:
    labels = [
        "Legacy causality error",
        "Certified causality error",
        "Measured BIBO gain",
        "Theoretical BIBO bound",
        "Determinism error",
    ]

    values = [
        max(
            legacy_prefix_error,
            1e-18,
        ),
        max(
            certified_prefix_error,
            1e-18,
        ),
        max(
            measured_gain,
            1e-18,
        ),
        max(
            theoretical_gain_bound,
            1e-18,
        ),
        max(
            determinism_error,
            1e-18,
        ),
    ]

    figure = plt.figure(
        figsize=(12, 7)
    )
    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.bar(
        labels,
        values,
    )
    axis.set_yscale("log")
    axis.set_title(
        "GENESIS-DSP Kararlılık ve Nedensellik Sertifikasyonu"
    )
    axis.set_ylabel(
        "Logaritmik test değeri"
    )
    axis.tick_params(
        axis="x",
        rotation=25,
    )
    axis.grid(
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
    STEP20_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    legacy_graph, legacy_package = (
        load_legacy_pipeline()
    )
    certified_graph = repair_pipeline(
        legacy_graph
    )

    legacy_registry = build_default_registry()
    certified_registry = (
        build_certified_registry()
    )

    (
        prefix_samples,
        full_samples,
    ) = build_prefix_test_signals()

    records: list[TestRecord] = []

    legacy_dc_prefix_error = (
        block_prefix_error(
            block_factory=DCRemovalBlock,
            prefix_samples=prefix_samples,
            full_samples=full_samples,
        )
    )

    if legacy_dc_prefix_error <= 1e-6:
        raise RuntimeError(
            "Legacy DC removal için beklenen non-causal davranış tespit edilmedi."
        )

    records.append(
        TestRecord(
            test_id="legacy_dc_prefix_causality",
            scope="block",
            subject="dc_removal",
            expected=(
                "Violation must be detected"
            ),
            observed_value=(
                legacy_dc_prefix_error
            ),
            threshold=1e-6,
            passed=True,
            note=(
                "Global frame ortalaması gelecekteki örneklere bağlıdır."
            ),
        )
    )

    causal_dc_prefix_error = (
        block_prefix_error(
            block_factory=lambda: (
                CausalDCBlockerBlock(
                    pole_radius=0.995
                )
            ),
            prefix_samples=prefix_samples,
            full_samples=full_samples,
        )
    )

    if (
        causal_dc_prefix_error
        > CAUSALITY_TOLERANCE
    ):
        raise RuntimeError(
            "Causal DC blocker prefix testini geçemedi."
        )

    records.append(
        TestRecord(
            test_id="causal_dc_prefix_causality",
            scope="block",
            subject="causal_dc_blocker",
            expected="error <= tolerance",
            observed_value=(
                causal_dc_prefix_error
            ),
            threshold=CAUSALITY_TOLERANCE,
            passed=True,
            note=(
                "Prefix çıktısı gelecekteki örneklerden bağımsızdır."
            ),
        )
    )

    legacy_graph_prefix_error = (
        graph_prefix_error(
            graph=legacy_graph,
            registry=legacy_registry,
            prefix_samples=prefix_samples,
            full_samples=full_samples,
        )
    )

    if legacy_graph_prefix_error <= 1e-6:
        raise RuntimeError(
            "Legacy graph içindeki non-causal davranış tespit edilmedi."
        )

    records.append(
        TestRecord(
            test_id="legacy_graph_prefix_causality",
            scope="graph",
            subject=(
                legacy_package["pipeline_id"]
            ),
            expected=(
                "Violation must be detected"
            ),
            observed_value=(
                legacy_graph_prefix_error
            ),
            threshold=1e-6,
            passed=True,
            note=(
                "Legacy pipeline çevrimdışı/global DC removal içeriyor."
            ),
        )
    )

    certified_graph_prefix_error = (
        graph_prefix_error(
            graph=certified_graph,
            registry=certified_registry,
            prefix_samples=prefix_samples,
            full_samples=full_samples,
        )
    )

    if (
        certified_graph_prefix_error
        > CAUSALITY_TOLERANCE
    ):
        raise RuntimeError(
            "Sertifikalı graph prefix causality testini geçemedi."
        )

    records.append(
        TestRecord(
            test_id="certified_graph_prefix_causality",
            scope="graph",
            subject="certified_causal_pipeline",
            expected="error <= tolerance",
            observed_value=(
                certified_graph_prefix_error
            ),
            threshold=CAUSALITY_TOLERANCE,
            passed=True,
            note=(
                "Pipeline gelecekteki örneklere bağlı değildir."
            ),
        )
    )

    theoretical_gain_bound = (
        graph_theoretical_bound(
            certified_graph
        )
    )

    (
        measured_maximum_gain,
        gain_by_signal,
    ) = maximum_graph_gain(
        graph=certified_graph,
        registry=certified_registry,
    )

    if (
        measured_maximum_gain
        > theoretical_gain_bound
        + STABILITY_TOLERANCE
    ):
        raise RuntimeError(
            "Sertifikalı pipeline teorik BIBO bound değerini aştı."
        )

    records.append(
        TestRecord(
            test_id="certified_graph_bibo_stability",
            scope="graph",
            subject="certified_causal_pipeline",
            expected="gain <= theoretical bound",
            observed_value=(
                measured_maximum_gain
            ),
            threshold=(
                theoretical_gain_bound
            ),
            passed=True,
            note=(
                "Bütün bounded girişlerde bounded çıkış elde edildi."
            ),
        )
    )

    first_execution = (
        execute_graph_samples(
            certified_graph,
            certified_registry,
            full_samples,
        )
    )
    second_execution = (
        execute_graph_samples(
            certified_graph,
            certified_registry,
            full_samples,
        )
    )

    determinism_error = float(
        np.max(
            np.abs(
                first_execution
                - second_execution
            )
        )
    )

    if (
        determinism_error
        > DETERMINISM_TOLERANCE
    ):
        raise RuntimeError(
            "Sertifikalı pipeline determinism testini geçemedi."
        )

    records.append(
        TestRecord(
            test_id="certified_graph_determinism",
            scope="graph",
            subject="certified_causal_pipeline",
            expected="error <= tolerance",
            observed_value=determinism_error,
            threshold=DETERMINISM_TOLERANCE,
            passed=True,
            note=(
                "Aynı giriş iki çalıştırmada aynı çıktıyı üretti."
            ),
        )
    )

    run_constraint_tests(
        records
    )

    PipelinePackage.save(
        path=CERTIFIED_PIPELINE_PATH,
        graph=certified_graph,
        pipeline_id=(
            "step20-certified-causal-pipeline"
        ),
        description=(
            "Causal DC blocker ile onarılmış; "
            "nedensellik, BIBO kararlılık, determinism "
            "ve yapısal kısıt testlerini geçen pipeline."
        ),
    )

    save_test_matrix(
        records
    )

    all_tests_passed = all(
        record.passed
        for record in records
    )

    if not all_tests_passed:
        raise RuntimeError(
            "En az bir sertifikasyon testi başarısız."
        )

    certificate = {
        "schema_name": (
            "GENESIS-DSP ConstraintCertificate"
        ),
        "schema_version": "1.0.0",
        "pipeline_id": (
            "step20-certified-causal-pipeline"
        ),
        "status": "CERTIFIED",
        "properties": {
            "causal": True,
            "bibo_stable": True,
            "deterministic": True,
            "finite_output": True,
            "nyquist_constrained": True,
            "acyclic_graph": True,
            "single_parent_nodes": True,
            "parameter_bounds_enforced": True,
        },
        "quantitative_bounds": {
            "prefix_causality_tolerance": (
                CAUSALITY_TOLERANCE
            ),
            "observed_prefix_error": (
                certified_graph_prefix_error
            ),
            "theoretical_linf_gain_bound": (
                theoretical_gain_bound
            ),
            "observed_maximum_linf_gain": (
                measured_maximum_gain
            ),
            "determinism_tolerance": (
                DETERMINISM_TOLERANCE
            ),
            "observed_determinism_error": (
                determinism_error
            ),
        },
        "legacy_issue": {
            "block_id": "dc_removal",
            "issue": (
                "Uses full-frame mean and is non-causal "
                "for streaming execution."
            ),
            "legacy_block_prefix_error": (
                legacy_dc_prefix_error
            ),
            "legacy_graph_prefix_error": (
                legacy_graph_prefix_error
            ),
            "repair": (
                "Replaced by causal_dc_blocker "
                "with pole_radius=0.995."
            ),
        },
    }

    with CERTIFICATE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            certificate,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    report = {
        "project": "GENESIS-DSP",
        "step": 20,
        "description": (
            "Stability, causality, determinism and "
            "constraint certification"
        ),
        "source_pipeline_id": (
            legacy_package["pipeline_id"]
        ),
        "legacy_analysis": {
            "dc_block_prefix_error": (
                legacy_dc_prefix_error
            ),
            "graph_prefix_error": (
                legacy_graph_prefix_error
            ),
            "noncausal_behavior_detected": True,
        },
        "repair": {
            "old_block_id": "dc_removal",
            "new_block_id": (
                "causal_dc_blocker"
            ),
            "pole_radius": 0.995,
        },
        "certified_pipeline": {
            "node_count": len(
                certified_graph.nodes
            ),
            "edge_count": len(
                certified_graph.edges
            ),
            "topological_order": (
                certified_graph.topological_order
            ),
            "prefix_causality_error": (
                certified_graph_prefix_error
            ),
            "theoretical_bibo_gain_bound": (
                theoretical_gain_bound
            ),
            "measured_maximum_bibo_gain": (
                measured_maximum_gain
            ),
            "measured_gain_by_signal": (
                gain_by_signal
            ),
            "determinism_error": (
                determinism_error
            ),
        },
        "tests": [
            asdict(record)
            for record in records
        ],
        "summary": {
            "test_count": len(records),
            "passed_count": sum(
                int(record.passed)
                for record in records
            ),
            "all_tests_passed": (
                all_tests_passed
            ),
            "certificate_status": (
                "CERTIFIED"
            ),
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

    create_plot(
        legacy_prefix_error=(
            legacy_graph_prefix_error
        ),
        certified_prefix_error=(
            certified_graph_prefix_error
        ),
        measured_gain=(
            measured_maximum_gain
        ),
        theoretical_gain_bound=(
            theoretical_gain_bound
        ),
        determinism_error=(
            determinism_error
        ),
    )

    print()
    print("=" * 86)
    print(
        "GENESIS-DSP — ADIM 20 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 86)
    print(
        f"Uygulanan sertifikasyon testi   : "
        f"{len(records)}"
    )
    print(
        f"Başarılı test                   : "
        f"{sum(int(record.passed) for record in records)}"
    )
    print(
        "Legacy DC causality ihlali      : TESPİT EDİLDİ"
    )
    print(
        f"Legacy graph prefix hatası      : "
        f"{legacy_graph_prefix_error:.6e}"
    )
    print(
        "Causal DC blocker onarımı       : BAŞARILI"
    )
    print(
        f"Sertifikalı prefix hatası       : "
        f"{certified_graph_prefix_error:.3e}"
    )
    print(
        f"Ölçülen maksimum BIBO gain      : "
        f"{measured_maximum_gain:.9f}"
    )
    print(
        f"Teorik BIBO gain sınırı         : "
        f"{theoretical_gain_bound:.9f}"
    )
    print(
        f"Determinism maksimum hatası     : "
        f"{determinism_error:.3e}"
    )
    print(
        "Nyquist/parametre kısıtları     : BAŞARILI"
    )
    print(
        "Cycle/multi-parent koruması     : BAŞARILI"
    )
    print(
        "Sertifika durumu                : CERTIFIED"
    )
    print(
        f"Sertifikalı pipeline            : "
        f"{CERTIFIED_PIPELINE_PATH}"
    )
    print(
        f"Kısıt sertifikası               : "
        f"{CERTIFICATE_PATH}"
    )
    print(
        f"Test matrisi                    : "
        f"{TEST_MATRIX_PATH}"
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
