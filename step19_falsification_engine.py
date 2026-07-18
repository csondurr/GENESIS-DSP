"""
GENESIS-DSP — Adım 19
Falsification ve karşı-örnek arama motoru.

Bu program:
1. Adım 18'de keşfedilen sabit alıcı reçetesini yükler.
2. Adım 06 nominal bozunum konfigürasyonunu başlangıç noktası kabul eder.
3. Yapılandırılmış zor durumlar ve rastgele bozunum senaryoları üretir.
4. Alıcıyı yeniden eğitmeden her senaryoda test eder.
5. BER, EVM, NMSE ve CFO hatasına göre başarısızlık arar.
6. En kötü senaryonun çevresinde ikinci bir yerel arama yapar.
7. Tekrarlanabilir karşı-örnek, CSV, JSON, NPZ ve grafik üretir.

Çalıştırma:
    python step19_falsification_engine.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from step18_combined_recovery import (
    build_features,
    calculate_ber,
    calculate_evm_percent,
    calculate_nmse,
    correct_cfo,
    estimate_and_remove_dc,
    estimate_qpsk_cfo_fourth_power,
    matched_filter,
    qpsk_hard_demodulate,
    sample_symbols,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP02_DIRECTORY = BASE_DIRECTORY / "outputs" / "step02"
STEP06_DIRECTORY = BASE_DIRECTORY / "outputs" / "step06"
STEP18_DIRECTORY = BASE_DIRECTORY / "outputs" / "step18"
STEP19_DIRECTORY = BASE_DIRECTORY / "outputs" / "step19"

CLEAN_PACKAGE_PATH = STEP02_DIRECTORY / "signal_record.npz"
NOMINAL_CONFIG_PATH = STEP06_DIRECTORY / "pipeline_config.json"
RECEIVER_RECORD_PATH = (
    STEP18_DIRECTORY / "combined_recovery_record.npz"
)
RECEIVER_REPORT_PATH = (
    STEP18_DIRECTORY / "combined_recovery_report.json"
)

REPORT_PATH = (
    STEP19_DIRECTORY / "falsification_report.json"
)
SCENARIO_TABLE_PATH = (
    STEP19_DIRECTORY / "falsification_scenarios.csv"
)
COUNTEREXAMPLE_CONFIG_PATH = (
    STEP19_DIRECTORY / "counterexample_config.json"
)
COUNTEREXAMPLE_DATA_PATH = (
    STEP19_DIRECTORY / "counterexample_signals.npz"
)
PLOT_PATH = (
    STEP19_DIRECTORY / "falsification_overview.png"
)

STEP06_MODULE_PATH = (
    BASE_DIRECTORY / "06_impairment_pipeline.py"
)


def load_step06_module() -> Any:
    """Sayıyla başlayan Adım 06 dosyasını güvenli biçimde yükler."""

    if not STEP06_MODULE_PATH.exists():
        raise FileNotFoundError(
            f"Adım 06 Python dosyası bulunamadı: {STEP06_MODULE_PATH}\n"
            "Dosyanın Proje klasöründe olduğundan emin ol."
        )

    specification = importlib.util.spec_from_file_location(
        "genesis_step06_impairment_pipeline",
        STEP06_MODULE_PATH,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            "Adım 06 modülü için import specification oluşturulamadı."
        )

    module = importlib.util.module_from_spec(
        specification
    )
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)

    return module


_STEP06_MODULE = load_step06_module()
ImpairmentConfig = _STEP06_MODULE.ImpairmentConfig
run_pipeline = _STEP06_MODULE.run_pipeline

RANDOM_SEED = 20260801
RANDOM_SCENARIO_COUNT = 22
LOCAL_MUTATION_COUNT = 14

FAILURE_BER_THRESHOLD = 0.02
FAILURE_EVM_THRESHOLD_PERCENT = 30.0
FAILURE_CFO_ERROR_THRESHOLD_HZ = 50.0

TRAIN_SYMBOLS = 1200
VALIDATION_SYMBOLS = 600
TEST_START_SYMBOL = TRAIN_SYMBOLS + VALIDATION_SYMBOLS

ComplexArray = NDArray[np.complex128]
BitArray = NDArray[np.int8]


@dataclass(frozen=True)
class ReceiverRecipe:
    timing_start: int
    equalizer_family: str
    equalizer_length: int
    ridge: float
    equalizer_weights: ComplexArray
    rrc_taps: NDArray[np.float64]


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    scenario_group: str
    test_ber: float
    test_bit_errors: int
    test_total_bits: int
    test_nmse: float
    test_evm_percent: float
    estimated_cfo_hz: float
    true_cfo_hz: float
    cfo_error_hz: float
    failure: bool
    failure_score: float
    config: ImpairmentConfig


@dataclass(frozen=True)
class RecoveryOutput:
    predictions: ComplexArray
    reference_symbols: ComplexArray
    reference_bits: BitArray
    estimated_cfo_hz: float
    received_samples: ComplexArray


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON dosyası bulunamadı: {path}")

    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise TypeError(f"JSON kökü dict olmalıdır: {path}")

    return document


def load_clean_signal() -> tuple[
    ComplexArray,
    BitArray,
    ComplexArray,
    float,
]:
    if not CLEAN_PACKAGE_PATH.exists():
        raise FileNotFoundError(
            f"Adım 02 paketi bulunamadı: {CLEAN_PACKAGE_PATH}\n"
            "Önce şu komutu çalıştır:\n"
            "python 02_signal_record.py"
        )

    with np.load(
        CLEAN_PACKAGE_PATH,
        allow_pickle=False,
    ) as package:
        samples = package["samples"].astype(np.complex128)
        bits = package["bits"].astype(np.int8)
        symbols = package["symbols"].astype(np.complex128)
        sample_rate_hz = float(package["sample_rate_hz"])

    return samples, bits, symbols, sample_rate_hz


def load_nominal_config() -> ImpairmentConfig:
    document = load_json(NOMINAL_CONFIG_PATH)
    raw = document["config"]

    return ImpairmentConfig(
        random_seed=int(raw["random_seed"]),
        cfo_hz=float(raw["cfo_hz"]),
        phase_offset_degrees=float(
            raw["phase_offset_degrees"]
        ),
        path_delays_samples=tuple(
            int(value)
            for value in raw["path_delays_samples"]
        ),
        path_magnitudes=tuple(
            float(value)
            for value in raw["path_magnitudes"]
        ),
        path_phases_degrees=tuple(
            float(value)
            for value in raw["path_phases_degrees"]
        ),
        fractional_delay_samples=float(
            raw["fractional_delay_samples"]
        ),
        fractional_delay_filter_taps=int(
            raw["fractional_delay_filter_taps"]
        ),
        iq_gain_imbalance_db=float(
            raw["iq_gain_imbalance_db"]
        ),
        iq_phase_imbalance_degrees=float(
            raw["iq_phase_imbalance_degrees"]
        ),
        dc_offset_i_rms_ratio=float(
            raw["dc_offset_i_rms_ratio"]
        ),
        dc_offset_q_rms_ratio=float(
            raw["dc_offset_q_rms_ratio"]
        ),
        clipping_level_rms_multiplier=float(
            raw["clipping_level_rms_multiplier"]
        ),
        target_snr_db=float(
            raw["target_snr_db"]
        ),
    )


def load_receiver_recipe() -> ReceiverRecipe:
    if not RECEIVER_RECORD_PATH.exists():
        raise FileNotFoundError(
            f"Adım 18 receiver kaydı bulunamadı: "
            f"{RECEIVER_RECORD_PATH}\n"
            "Önce şu komutu çalıştır:\n"
            "python step18_combined_recovery.py"
        )

    with np.load(
        RECEIVER_RECORD_PATH,
        allow_pickle=False,
    ) as package:
        timing_start = int(package["timing_start"])
        equalizer_length = int(
            package["equalizer_length"]
        )
        ridge = float(package["ridge"])
        equalizer_family = str(
            package["equalizer_family"].item()
        )
        equalizer_weights = package[
            "equalizer_weights"
        ].astype(np.complex128)
        rrc_taps = package["rrc_taps"].astype(np.float64)

    return ReceiverRecipe(
        timing_start=timing_start,
        equalizer_family=equalizer_family,
        equalizer_length=equalizer_length,
        ridge=ridge,
        equalizer_weights=equalizer_weights,
        rrc_taps=rrc_taps,
    )


def run_fixed_receiver(
    received_samples: ComplexArray,
    bits: BitArray,
    symbols: ComplexArray,
    sample_rate_hz: float,
    recipe: ReceiverRecipe,
) -> RecoveryOutput:
    dc_corrected, _ = estimate_and_remove_dc(
        received_samples
    )

    estimated_cfo_hz, _, _ = (
        estimate_qpsk_cfo_fourth_power(
            samples=dc_corrected,
            sample_rate_hz=sample_rate_hz,
        )
    )

    cfo_corrected = correct_cfo(
        samples=dc_corrected,
        sample_rate_hz=sample_rate_hz,
        estimated_cfo_hz=estimated_cfo_hz,
    )

    matched = matched_filter(
        samples=cfo_corrected,
        rrc_taps=recipe.rrc_taps,
    )

    sampled = sample_symbols(
        matched_samples=matched,
        timing_start=recipe.timing_start,
        samples_per_symbol=4,
        maximum_symbols=len(symbols),
    )

    features, half_length = build_features(
        sampled_symbols=sampled,
        equalizer_length=recipe.equalizer_length,
        family=recipe.equalizer_family,
    )

    if features.shape[1] != len(
        recipe.equalizer_weights
    ):
        raise RuntimeError(
            "Receiver feature sayısı ile equalizer weight sayısı eşleşmiyor."
        )

    predictions = (
        features
        @ recipe.equalizer_weights
    ).astype(np.complex128)

    reference_symbols = symbols[
        half_length:
        half_length + len(predictions)
    ]
    reference_bits = bits[
        half_length:
        half_length + len(predictions)
    ]

    usable = min(
        len(predictions),
        len(reference_symbols),
        len(reference_bits),
    )

    return RecoveryOutput(
        predictions=predictions[:usable],
        reference_symbols=reference_symbols[:usable],
        reference_bits=reference_bits[:usable],
        estimated_cfo_hz=estimated_cfo_hz,
        received_samples=received_samples,
    )


def evaluate_scenario(
    scenario_id: str,
    scenario_group: str,
    config: ImpairmentConfig,
    clean_samples: ComplexArray,
    bits: BitArray,
    symbols: ComplexArray,
    sample_rate_hz: float,
    recipe: ReceiverRecipe,
) -> tuple[ScenarioResult, RecoveryOutput]:
    pipeline_result = run_pipeline(
        clean_samples=clean_samples,
        sample_rate_hz=sample_rate_hz,
        config=config,
    )

    recovery = run_fixed_receiver(
        received_samples=(
            pipeline_result.received_samples
        ),
        bits=bits,
        symbols=symbols,
        sample_rate_hz=sample_rate_hz,
        recipe=recipe,
    )

    test_stop = len(recovery.predictions)

    if test_stop <= TEST_START_SYMBOL:
        raise RuntimeError(
            "Falsification testi için yeterli test sembolü yok."
        )

    predictions = recovery.predictions[
        TEST_START_SYMBOL:test_stop
    ]
    references = recovery.reference_symbols[
        TEST_START_SYMBOL:test_stop
    ]
    reference_bits = recovery.reference_bits[
        TEST_START_SYMBOL:test_stop
    ]

    estimated_bits = qpsk_hard_demodulate(
        predictions
    )

    ber, errors, total = calculate_ber(
        reference_bits,
        estimated_bits,
    )
    nmse = calculate_nmse(
        references,
        predictions,
    )
    evm = calculate_evm_percent(
        references,
        predictions,
    )
    cfo_error = (
        recovery.estimated_cfo_hz
        - config.cfo_hz
    )

    failure = bool(
        ber > FAILURE_BER_THRESHOLD
        or evm > FAILURE_EVM_THRESHOLD_PERCENT
        or abs(cfo_error)
        > FAILURE_CFO_ERROR_THRESHOLD_HZ
    )

    failure_score = float(
        6.0 * ber
        + evm / 100.0
        + min(
            abs(cfo_error) / 500.0,
            4.0,
        )
    )

    result = ScenarioResult(
        scenario_id=scenario_id,
        scenario_group=scenario_group,
        test_ber=ber,
        test_bit_errors=errors,
        test_total_bits=total,
        test_nmse=nmse,
        test_evm_percent=evm,
        estimated_cfo_hz=(
            recovery.estimated_cfo_hz
        ),
        true_cfo_hz=config.cfo_hz,
        cfo_error_hz=cfo_error,
        failure=failure,
        failure_score=failure_score,
        config=config,
    )

    return result, recovery


def structured_scenarios(
    nominal: ImpairmentConfig,
) -> list[tuple[str, ImpairmentConfig]]:
    return [
        (
            "nominal",
            nominal,
        ),
        (
            "phase_rotation_90deg",
            replace(
                nominal,
                random_seed=RANDOM_SEED + 1,
                phase_offset_degrees=122.0,
            ),
        ),
        (
            "low_snr",
            replace(
                nominal,
                random_seed=RANDOM_SEED + 2,
                target_snr_db=3.0,
            ),
        ),
        (
            "heavy_clipping",
            replace(
                nominal,
                random_seed=RANDOM_SEED + 3,
                clipping_level_rms_multiplier=0.72,
            ),
        ),
        (
            "severe_iq",
            replace(
                nominal,
                random_seed=RANDOM_SEED + 4,
                iq_gain_imbalance_db=4.0,
                iq_phase_imbalance_degrees=17.0,
            ),
        ),
        (
            "strong_multipath",
            replace(
                nominal,
                random_seed=RANDOM_SEED + 5,
                path_magnitudes=(0.65, 1.0, 0.85),
                path_phases_degrees=(
                    35.0,
                    -110.0,
                    145.0,
                ),
            ),
        ),
        (
            "combined_extreme",
            replace(
                nominal,
                random_seed=RANDOM_SEED + 6,
                cfo_hz=92_000.0,
                phase_offset_degrees=-135.0,
                path_magnitudes=(0.55, 1.0, 0.90),
                path_phases_degrees=(
                    70.0,
                    -125.0,
                    160.0,
                ),
                fractional_delay_samples=-0.46,
                iq_gain_imbalance_db=4.5,
                iq_phase_imbalance_degrees=19.0,
                dc_offset_i_rms_ratio=0.28,
                dc_offset_q_rms_ratio=-0.24,
                clipping_level_rms_multiplier=0.68,
                target_snr_db=1.5,
            ),
        ),
    ]


def random_config(
    nominal: ImpairmentConfig,
    rng: np.random.Generator,
    seed: int,
) -> ImpairmentConfig:
    magnitudes = (
        float(rng.uniform(0.45, 1.20)),
        float(rng.uniform(0.20, 1.20)),
        float(rng.uniform(0.10, 1.00)),
    )

    phases = tuple(
        float(value)
        for value in rng.uniform(
            -180.0,
            180.0,
            size=3,
        )
    )

    return replace(
        nominal,
        random_seed=seed,
        cfo_hz=float(
            rng.uniform(
                -100_000.0,
                100_000.0,
            )
        ),
        phase_offset_degrees=float(
            rng.uniform(
                -180.0,
                180.0,
            )
        ),
        path_magnitudes=magnitudes,
        path_phases_degrees=phases,
        fractional_delay_samples=float(
            rng.uniform(
                -0.48,
                0.48,
            )
        ),
        iq_gain_imbalance_db=float(
            rng.uniform(
                -4.5,
                4.5,
            )
        ),
        iq_phase_imbalance_degrees=float(
            rng.uniform(
                -20.0,
                20.0,
            )
        ),
        dc_offset_i_rms_ratio=float(
            rng.uniform(
                -0.30,
                0.30,
            )
        ),
        dc_offset_q_rms_ratio=float(
            rng.uniform(
                -0.30,
                0.30,
            )
        ),
        clipping_level_rms_multiplier=float(
            rng.uniform(
                0.65,
                1.60,
            )
        ),
        target_snr_db=float(
            rng.uniform(
                1.0,
                18.0,
            )
        ),
    )


def mutate_around_worst(
    worst: ImpairmentConfig,
    nominal: ImpairmentConfig,
    rng: np.random.Generator,
    seed: int,
) -> ImpairmentConfig:
    def clip(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return float(
            np.clip(
                value,
                minimum,
                maximum,
            )
        )

    magnitudes = tuple(
        clip(
            value
            * float(
                rng.uniform(
                    0.75,
                    1.25,
                )
            ),
            0.05,
            1.50,
        )
        for value in worst.path_magnitudes
    )

    phases = tuple(
        clip(
            value
            + float(
                rng.normal(
                    0.0,
                    25.0,
                )
            ),
            -180.0,
            180.0,
        )
        for value in worst.path_phases_degrees
    )

    return replace(
        nominal,
        random_seed=seed,
        cfo_hz=clip(
            worst.cfo_hz
            + float(
                rng.normal(
                    0.0,
                    12_000.0,
                )
            ),
            -110_000.0,
            110_000.0,
        ),
        phase_offset_degrees=clip(
            worst.phase_offset_degrees
            + float(
                rng.normal(
                    0.0,
                    25.0,
                )
            ),
            -180.0,
            180.0,
        ),
        path_magnitudes=magnitudes,
        path_phases_degrees=phases,
        fractional_delay_samples=clip(
            worst.fractional_delay_samples
            + float(
                rng.normal(
                    0.0,
                    0.08,
                )
            ),
            -0.49,
            0.49,
        ),
        iq_gain_imbalance_db=clip(
            worst.iq_gain_imbalance_db
            + float(
                rng.normal(
                    0.0,
                    0.8,
                )
            ),
            -5.0,
            5.0,
        ),
        iq_phase_imbalance_degrees=clip(
            worst.iq_phase_imbalance_degrees
            + float(
                rng.normal(
                    0.0,
                    3.0,
                )
            ),
            -22.0,
            22.0,
        ),
        dc_offset_i_rms_ratio=clip(
            worst.dc_offset_i_rms_ratio
            + float(
                rng.normal(
                    0.0,
                    0.05,
                )
            ),
            -0.35,
            0.35,
        ),
        dc_offset_q_rms_ratio=clip(
            worst.dc_offset_q_rms_ratio
            + float(
                rng.normal(
                    0.0,
                    0.05,
                )
            ),
            -0.35,
            0.35,
        ),
        clipping_level_rms_multiplier=clip(
            worst.clipping_level_rms_multiplier
            + float(
                rng.normal(
                    0.0,
                    0.10,
                )
            ),
            0.55,
            1.80,
        ),
        target_snr_db=clip(
            worst.target_snr_db
            + float(
                rng.normal(
                    0.0,
                    2.0,
                )
            ),
            0.0,
            20.0,
        ),
    )


def result_sort_key(
    result: ScenarioResult,
) -> tuple[float, float, float]:
    return (
        result.failure_score,
        result.test_ber,
        result.test_evm_percent,
    )


def config_to_json(
    config: ImpairmentConfig,
) -> dict[str, Any]:
    return asdict(config)


def save_scenario_table(
    results: list[ScenarioResult],
) -> None:
    with SCENARIO_TABLE_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "scenario_id",
            "scenario_group",
            "failure",
            "failure_score",
            "test_ber",
            "test_bit_errors",
            "test_total_bits",
            "test_nmse",
            "test_evm_percent",
            "estimated_cfo_hz",
            "true_cfo_hz",
            "cfo_error_hz",
            "phase_offset_degrees",
            "fractional_delay_samples",
            "iq_gain_imbalance_db",
            "iq_phase_imbalance_degrees",
            "dc_offset_i_rms_ratio",
            "dc_offset_q_rms_ratio",
            "clipping_level_rms_multiplier",
            "target_snr_db",
            "path_magnitudes",
            "path_phases_degrees",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for result in results:
            config = result.config

            writer.writerow(
                {
                    "scenario_id": result.scenario_id,
                    "scenario_group": result.scenario_group,
                    "failure": result.failure,
                    "failure_score": result.failure_score,
                    "test_ber": result.test_ber,
                    "test_bit_errors": result.test_bit_errors,
                    "test_total_bits": result.test_total_bits,
                    "test_nmse": result.test_nmse,
                    "test_evm_percent": result.test_evm_percent,
                    "estimated_cfo_hz": result.estimated_cfo_hz,
                    "true_cfo_hz": result.true_cfo_hz,
                    "cfo_error_hz": result.cfo_error_hz,
                    "phase_offset_degrees": (
                        config.phase_offset_degrees
                    ),
                    "fractional_delay_samples": (
                        config.fractional_delay_samples
                    ),
                    "iq_gain_imbalance_db": (
                        config.iq_gain_imbalance_db
                    ),
                    "iq_phase_imbalance_degrees": (
                        config.iq_phase_imbalance_degrees
                    ),
                    "dc_offset_i_rms_ratio": (
                        config.dc_offset_i_rms_ratio
                    ),
                    "dc_offset_q_rms_ratio": (
                        config.dc_offset_q_rms_ratio
                    ),
                    "clipping_level_rms_multiplier": (
                        config.clipping_level_rms_multiplier
                    ),
                    "target_snr_db": (
                        config.target_snr_db
                    ),
                    "path_magnitudes": " | ".join(
                        f"{value:.8f}"
                        for value in config.path_magnitudes
                    ),
                    "path_phases_degrees": " | ".join(
                        f"{value:.8f}"
                        for value in config.path_phases_degrees
                    ),
                }
            )


def create_plot(
    results: list[ScenarioResult],
    nominal: ScenarioResult,
    worst: ScenarioResult,
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 15),
    )

    indices = np.arange(
        len(results),
        dtype=np.int64,
    )

    ber_values = np.asarray(
        [
            item.test_ber
            for item in results
        ],
        dtype=np.float64,
    )
    evm_values = np.asarray(
        [
            item.test_evm_percent
            for item in results
        ],
        dtype=np.float64,
    )
    scores = np.asarray(
        [
            item.failure_score
            for item in results
        ],
        dtype=np.float64,
    )

    axes[0].scatter(
        indices,
        ber_values,
        c=[
            int(item.failure)
            for item in results
        ],
        s=55,
    )
    axes[0].axhline(
        FAILURE_BER_THRESHOLD,
        linestyle="--",
        label="BER failure eşiği",
    )
    axes[0].set_title(
        "Falsification Senaryolarında Test BER"
    )
    axes[0].set_xlabel(
        "Senaryo indeksi"
    )
    axes[0].set_ylabel(
        "Test BER"
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].scatter(
        evm_values,
        ber_values,
        c=scores,
        s=70,
    )
    axes[1].axvline(
        FAILURE_EVM_THRESHOLD_PERCENT,
        linestyle="--",
        label="EVM failure eşiği",
    )
    axes[1].axhline(
        FAILURE_BER_THRESHOLD,
        linestyle="--",
        label="BER failure eşiği",
    )
    axes[1].set_title(
        "BER–EVM Karşı-Örnek Haritası"
    )
    axes[1].set_xlabel(
        "Test RMS EVM (%)"
    )
    axes[1].set_ylabel(
        "Test BER"
    )
    axes[1].grid(True)
    axes[1].legend()

    labels = [
        "Nominal BER",
        "Worst BER",
        "Nominal EVM / 100",
        "Worst EVM / 100",
    ]
    values = [
        nominal.test_ber,
        worst.test_ber,
        nominal.test_evm_percent / 100.0,
        worst.test_evm_percent / 100.0,
    ]

    axes[2].bar(
        labels,
        values,
    )
    axes[2].set_title(
        "Nominal Durum ve En Kötü Karşı-Örnek"
    )
    axes[2].set_ylabel(
        "Normalize metrik"
    )
    axes[2].tick_params(
        axis="x",
        rotation=20,
    )
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
    STEP19_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        clean_samples,
        bits,
        symbols,
        sample_rate_hz,
    ) = load_clean_signal()

    nominal_config = load_nominal_config()
    recipe = load_receiver_recipe()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results: list[
        ScenarioResult
    ] = []
    recovery_map: dict[
        str,
        RecoveryOutput
    ] = {}

    for scenario_name, config in structured_scenarios(
        nominal_config
    ):
        result, recovery = evaluate_scenario(
            scenario_id=scenario_name,
            scenario_group="structured",
            config=config,
            clean_samples=clean_samples,
            bits=bits,
            symbols=symbols,
            sample_rate_hz=sample_rate_hz,
            recipe=recipe,
        )
        results.append(result)
        recovery_map[
            result.scenario_id
        ] = recovery

    for index in range(
        RANDOM_SCENARIO_COUNT
    ):
        scenario_id = (
            f"random_{index:03d}"
        )
        config = random_config(
            nominal=nominal_config,
            rng=rng,
            seed=(
                RANDOM_SEED
                + 100
                + index
            ),
        )

        result, recovery = evaluate_scenario(
            scenario_id=scenario_id,
            scenario_group="random",
            config=config,
            clean_samples=clean_samples,
            bits=bits,
            symbols=symbols,
            sample_rate_hz=sample_rate_hz,
            recipe=recipe,
        )
        results.append(result)
        recovery_map[
            result.scenario_id
        ] = recovery

    current_worst = max(
        results,
        key=result_sort_key,
    )

    for index in range(
        LOCAL_MUTATION_COUNT
    ):
        scenario_id = (
            f"local_{index:03d}"
        )
        config = mutate_around_worst(
            worst=current_worst.config,
            nominal=nominal_config,
            rng=rng,
            seed=(
                RANDOM_SEED
                + 1000
                + index
            ),
        )

        result, recovery = evaluate_scenario(
            scenario_id=scenario_id,
            scenario_group="local_search",
            config=config,
            clean_samples=clean_samples,
            bits=bits,
            symbols=symbols,
            sample_rate_hz=sample_rate_hz,
            recipe=recipe,
        )
        results.append(result)
        recovery_map[
            result.scenario_id
        ] = recovery

        if result_sort_key(result) > result_sort_key(
            current_worst
        ):
            current_worst = result

    results.sort(
        key=result_sort_key,
        reverse=True,
    )

    worst = results[0]

    nominal_matches = [
        item
        for item in results
        if item.scenario_id == "nominal"
    ]

    if len(nominal_matches) != 1:
        raise RuntimeError(
            "Nominal senaryo sonucu bulunamadı."
        )

    nominal_result = nominal_matches[0]

    if nominal_result.test_ber > FAILURE_BER_THRESHOLD:
        raise RuntimeError(
            "Nominal alıcı testi failure eşiğini geçti."
        )

    failures = [
        item
        for item in results
        if item.failure
    ]

    if not failures:
        raise RuntimeError(
            "Falsification motoru karşı-örnek bulamadı."
        )

    repeated_result, repeated_recovery = evaluate_scenario(
        scenario_id=worst.scenario_id,
        scenario_group=worst.scenario_group,
        config=worst.config,
        clean_samples=clean_samples,
        bits=bits,
        symbols=symbols,
        sample_rate_hz=sample_rate_hz,
        recipe=recipe,
    )

    prediction_reproducibility_error = float(
        np.max(
            np.abs(
                repeated_recovery.predictions
                - recovery_map[
                    worst.scenario_id
                ].predictions
            )
        )
    )

    if prediction_reproducibility_error > 1e-12:
        raise RuntimeError(
            "Karşı-örnek tekrarlanabilirlik testini geçemedi."
        )

    if not np.isclose(
        repeated_result.failure_score,
        worst.failure_score,
        atol=1e-15,
    ):
        raise RuntimeError(
            "Tekrarlanan karşı-örnek skoru değişti."
        )

    counterexample_document = {
        "schema_name": (
            "GENESIS-DSP Counterexample"
        ),
        "schema_version": "1.0.0",
        "scenario_id": worst.scenario_id,
        "scenario_group": worst.scenario_group,
        "failure_thresholds": {
            "ber": FAILURE_BER_THRESHOLD,
            "evm_percent": (
                FAILURE_EVM_THRESHOLD_PERCENT
            ),
            "absolute_cfo_error_hz": (
                FAILURE_CFO_ERROR_THRESHOLD_HZ
            ),
        },
        "metrics": {
            "test_ber": worst.test_ber,
            "test_bit_errors": (
                worst.test_bit_errors
            ),
            "test_total_bits": (
                worst.test_total_bits
            ),
            "test_nmse": worst.test_nmse,
            "test_evm_percent": (
                worst.test_evm_percent
            ),
            "estimated_cfo_hz": (
                worst.estimated_cfo_hz
            ),
            "true_cfo_hz": (
                worst.true_cfo_hz
            ),
            "cfo_error_hz": (
                worst.cfo_error_hz
            ),
            "failure_score": (
                worst.failure_score
            ),
        },
        "impairment_config": config_to_json(
            worst.config
        ),
    }

    with COUNTEREXAMPLE_CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            counterexample_document,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    worst_recovery = recovery_map[
        worst.scenario_id
    ]

    np.savez_compressed(
        COUNTEREXAMPLE_DATA_PATH,
        clean_samples=clean_samples,
        received_samples=(
            worst_recovery.received_samples
        ),
        recovered_symbols=(
            worst_recovery.predictions
        ),
        reference_symbols=(
            worst_recovery.reference_symbols
        ),
        reference_bits=(
            worst_recovery.reference_bits
        ),
        sample_rate_hz=np.float64(
            sample_rate_hz
        ),
        estimated_cfo_hz=np.float64(
            worst.estimated_cfo_hz
        ),
        true_cfo_hz=np.float64(
            worst.true_cfo_hz
        ),
    )

    save_scenario_table(
        results
    )

    create_plot(
        results=results,
        nominal=nominal_result,
        worst=worst,
    )

    receiver_report = load_json(
        RECEIVER_REPORT_PATH
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 19,
        "description": (
            "Falsification and deterministic counterexample "
            "search for the fixed Step 18 receiver"
        ),
        "receiver_under_test": {
            "timing_start": (
                recipe.timing_start
            ),
            "equalizer_family": (
                recipe.equalizer_family
            ),
            "equalizer_length": (
                recipe.equalizer_length
            ),
            "ridge": recipe.ridge,
            "nominal_test_ber": (
                receiver_report[
                    "metrics"
                ]["test"]["ber"]
            ),
        },
        "search": {
            "random_seed": RANDOM_SEED,
            "structured_scenarios": len(
                structured_scenarios(
                    nominal_config
                )
            ),
            "random_scenarios": (
                RANDOM_SCENARIO_COUNT
            ),
            "local_mutations": (
                LOCAL_MUTATION_COUNT
            ),
            "total_scenarios": len(
                results
            ),
            "failure_count": len(
                failures
            ),
            "failure_rate": float(
                len(failures)
                / len(results)
            ),
        },
        "nominal_result": {
            **asdict(
                nominal_result
            ),
            "config": config_to_json(
                nominal_result.config
            ),
        },
        "worst_counterexample": {
            **asdict(
                worst
            ),
            "config": config_to_json(
                worst.config
            ),
        },
        "reproducibility": {
            "maximum_prediction_error": (
                prediction_reproducibility_error
            ),
            "status": "PASSED",
        },
        "validations": {
            "nominal_case_passed": True,
            "counterexample_found": True,
            "failure_threshold_exceeded": True,
            "counterexample_reproducible": True,
            "counterexample_saved": True,
        },
    }

    # Dataclass içindeki config nesnesini JSON-safe dict ile değiştir.
    report["nominal_result"].pop(
        "config",
        None,
    )
    report["nominal_result"][
        "impairment_config"
    ] = config_to_json(
        nominal_result.config
    )

    report["worst_counterexample"].pop(
        "config",
        None,
    )
    report["worst_counterexample"][
        "impairment_config"
    ] = config_to_json(
        worst.config
    )

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
        "GENESIS-DSP — ADIM 19 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 86)
    print(
        f"Test edilen senaryo             : "
        f"{len(results)}"
    )
    print(
        f"Bulunan başarısız senaryo       : "
        f"{len(failures)}"
    )
    print(
        f"Failure oranı                   : "
        f"{100.0 * len(failures) / len(results):.3f} %"
    )
    print(
        f"Nominal test BER                : "
        f"{nominal_result.test_ber:.9f}"
    )
    print(
        f"En kötü karşı-örnek             : "
        f"{worst.scenario_id}"
    )
    print(
        f"Karşı-örnek test BER            : "
        f"{worst.test_ber:.9f}"
    )
    print(
        f"Karşı-örnek bit hatası          : "
        f"{worst.test_bit_errors} / "
        f"{worst.test_total_bits}"
    )
    print(
        f"Karşı-örnek RMS EVM             : "
        f"{worst.test_evm_percent:.6f} %"
    )
    print(
        f"Karşı-örnek CFO hatası          : "
        f"{worst.cfo_error_hz:+.6f} Hz"
    )
    print(
        f"Failure skoru                   : "
        f"{worst.failure_score:.9f}"
    )
    print(
        f"Reproducibility max hata        : "
        f"{prediction_reproducibility_error:.3e}"
    )
    print(
        f"Karşı-örnek config              : "
        f"{COUNTEREXAMPLE_CONFIG_PATH}"
    )
    print(
        f"Senaryo tablosu                 : "
        f"{SCENARIO_TABLE_PATH}"
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
