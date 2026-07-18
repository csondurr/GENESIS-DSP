
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP01_DIRECTORY = BASE_DIRECTORY / "outputs" / "step01"
STEP06_DIRECTORY = BASE_DIRECTORY / "outputs" / "step06"
STEP08_DIRECTORY = BASE_DIRECTORY / "outputs" / "step08"
STEP18_DIRECTORY = BASE_DIRECTORY / "outputs" / "step18"

INPUT_PACKAGE = STEP06_DIRECTORY / "dataset_case_0001.npz"
INPUT_CONFIG = STEP06_DIRECTORY / "pipeline_config.json"
RRC_TAPS_PATH = STEP01_DIRECTORY / "rrc_filter_taps.npy"
BASELINE_REPORT_PATH = STEP08_DIRECTORY / "baseline_report.json"

REPORT_PATH = STEP18_DIRECTORY / "combined_recovery_report.json"
RECIPE_PATH = STEP18_DIRECTORY / "combined_recovery_recipe.json"
SEARCH_PATH = STEP18_DIRECTORY / "model_search.csv"
RECORD_PATH = STEP18_DIRECTORY / "combined_recovery_record.npz"
PLOT_PATH = STEP18_DIRECTORY / "combined_recovery_overview.png"

TRAIN_SYMBOLS = 1200
VALIDATION_SYMBOLS = 600

TIMING_SEARCH_START = 0
TIMING_SEARCH_STOP = 128
COARSE_TIMING_STEP = 2
TIMING_REFINEMENT_RADIUS = 3
FINAL_TIMING_RADIUS = 1

EQUALIZER_LENGTHS = (5, 7, 9, 11, 13)
RIDGE_VALUES = (1e-6, 1e-4, 1e-2)
EQUALIZER_FAMILIES = ("linear", "widely_linear")

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.int8]


@dataclass(frozen=True)
class Dataset:
    received_samples: ComplexArray
    bits: BitArray
    symbols: ComplexArray
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: int
    true_cfo_hz: float
    true_phase_offset_degrees: float
    target_snr_db: float
    achieved_snr_db: float
    rrc_taps: FloatArray


@dataclass(frozen=True)
class ReceiverModel:
    timing_start: int
    family: str
    equalizer_length: int
    ridge: float


@dataclass(frozen=True)
class CandidateEvaluation:
    timing_start: int
    family: str
    equalizer_length: int
    ridge: float
    feature_count: int
    validation_ber: float
    validation_bit_errors: int
    validation_nmse: float
    validation_evm_percent: float
    selection_score: float


@dataclass(frozen=True)
class ModelOutput:
    model: ReceiverModel
    weights: ComplexArray
    sampled_symbols: ComplexArray
    predictions: ComplexArray
    reference_symbols: ComplexArray
    reference_bits: BitArray
    feature_half_length: int
    feature_count: int


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON dosyası bulunamadı: {path}")

    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise TypeError(f"JSON kökü dict olmalıdır: {path}")

    return document


def load_dataset() -> Dataset:
    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 06 veri paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python 06_impairment_pipeline.py"
        )

    if not INPUT_CONFIG.exists():
        raise FileNotFoundError(
            f"Adım 06 config dosyası bulunamadı: {INPUT_CONFIG}"
        )

    if not RRC_TAPS_PATH.exists():
        raise FileNotFoundError(
            f"RRC filtresi bulunamadı: {RRC_TAPS_PATH}\n"
            "Önce şu komutu çalıştır:\n"
            "python 01_qpsk_generator.py"
        )

    configuration = load_json(INPUT_CONFIG)["config"]

    with np.load(INPUT_PACKAGE, allow_pickle=False) as package:
        received_samples = package["received_samples"].astype(
            np.complex128
        )
        bits = package["bits"].astype(np.int8)
        symbols = package["symbols"].astype(np.complex128)
        sample_rate_hz = float(package["sample_rate_hz"])
        symbol_rate_hz = float(package["symbol_rate_hz"])
        samples_per_symbol = int(package["samples_per_symbol"])
        achieved_snr_db = float(package["achieved_snr_db"])

    rrc_taps = np.load(
        RRC_TAPS_PATH,
        allow_pickle=False,
    ).astype(np.float64)

    if received_samples.ndim != 1 or len(received_samples) == 0:
        raise ValueError(
            "received_samples tek boyutlu ve boş olmayan olmalıdır."
        )

    if not np.all(np.isfinite(received_samples.real)):
        raise ValueError(
            "received_samples gerçek kısmında NaN veya Inf bulundu."
        )

    if not np.all(np.isfinite(received_samples.imag)):
        raise ValueError(
            "received_samples sanal kısmında NaN veya Inf bulundu."
        )

    if bits.ndim != 2 or bits.shape[1] != 2:
        raise ValueError(
            "bits dizisi (sembol_sayısı, 2) biçiminde olmalıdır."
        )

    if symbols.ndim != 1 or len(symbols) != len(bits):
        raise ValueError(
            "symbols ile bits uzunlukları eşleşmiyor."
        )

    if rrc_taps.ndim != 1 or len(rrc_taps) == 0:
        raise ValueError(
            "RRC tap dizisi tek boyutlu ve boş olmayan olmalıdır."
        )

    return Dataset(
        received_samples=received_samples,
        bits=bits,
        symbols=symbols,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
        true_cfo_hz=float(configuration["cfo_hz"]),
        true_phase_offset_degrees=float(
            configuration["phase_offset_degrees"]
        ),
        target_snr_db=float(configuration["target_snr_db"]),
        achieved_snr_db=achieved_snr_db,
        rrc_taps=rrc_taps,
    )


def estimate_and_remove_dc(
    samples: ComplexArray,
) -> tuple[ComplexArray, complex]:
    estimated_dc = complex(np.mean(samples))
    corrected = (samples - estimated_dc).astype(np.complex128)
    return corrected, estimated_dc


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value pozitif olmalıdır.")

    return 1 << (value - 1).bit_length()


def parabolic_peak_offset(
    left_value: float,
    center_value: float,
    right_value: float,
) -> float:
    denominator = (
        left_value
        - 2.0 * center_value
        + right_value
    )

    if abs(denominator) <= 1e-30:
        return 0.0

    offset = (
        0.5
        * (left_value - right_value)
        / denominator
    )

    return float(np.clip(offset, -0.5, 0.5))


def estimate_qpsk_cfo_fourth_power(
    samples: ComplexArray,
    sample_rate_hz: float,
) -> tuple[float, FloatArray, FloatArray]:
    fourth_power = samples ** 4
    window = np.hanning(len(fourth_power))

    fft_size = next_power_of_two(
        max(65_536, 4 * len(fourth_power))
    )

    spectrum = np.fft.fftshift(
        np.fft.fft(
            fourth_power * window,
            n=fft_size,
        )
    )

    magnitude_squared = np.abs(spectrum) ** 2
    frequencies_hz = np.fft.fftshift(
        np.fft.fftfreq(
            fft_size,
            d=1.0 / sample_rate_hz,
        )
    )

    peak_index = int(
        np.argmax(magnitude_squared)
    )
    refined_peak_index = float(peak_index)

    if 0 < peak_index < len(magnitude_squared) - 1:
        refined_peak_index += parabolic_peak_offset(
            float(magnitude_squared[peak_index - 1]),
            float(magnitude_squared[peak_index]),
            float(magnitude_squared[peak_index + 1]),
        )

    frequency_step_hz = (
        sample_rate_hz / fft_size
    )
    fourth_power_frequency_hz = (
        refined_peak_index
        - fft_size / 2.0
    ) * frequency_step_hz

    estimated_cfo_hz = (
        fourth_power_frequency_hz / 4.0
    )

    normalized_power = (
        magnitude_squared
        / float(np.max(magnitude_squared))
    )
    spectrum_db = 10.0 * np.log10(
        np.maximum(
            normalized_power,
            1e-15,
        )
    )

    return (
        float(estimated_cfo_hz),
        frequencies_hz.astype(np.float64),
        spectrum_db.astype(np.float64),
    )


def correct_cfo(
    samples: ComplexArray,
    sample_rate_hz: float,
    estimated_cfo_hz: float,
) -> ComplexArray:
    indices = np.arange(
        len(samples),
        dtype=np.float64,
    )

    correction = np.exp(
        -1j
        * 2.0
        * np.pi
        * estimated_cfo_hz
        * indices
        / sample_rate_hz
    )

    return (samples * correction).astype(np.complex128)


def matched_filter(
    samples: ComplexArray,
    rrc_taps: FloatArray,
) -> ComplexArray:
    return np.convolve(
        samples,
        rrc_taps,
        mode="full",
    ).astype(np.complex128)


def sample_symbols(
    matched_samples: ComplexArray,
    timing_start: int,
    samples_per_symbol: int,
    maximum_symbols: int,
) -> ComplexArray:
    indices = (
        timing_start
        + np.arange(
            maximum_symbols,
            dtype=np.int64,
        )
        * samples_per_symbol
    )

    valid_indices = indices[
        indices < len(matched_samples)
    ]

    return matched_samples[
        valid_indices
    ].astype(np.complex128)


def build_features(
    sampled_symbols: ComplexArray,
    equalizer_length: int,
    family: str,
) -> tuple[ComplexArray, int]:
    if equalizer_length < 3 or equalizer_length % 2 == 0:
        raise ValueError(
            "equalizer_length en az 3 ve tek sayı olmalıdır."
        )

    if family not in EQUALIZER_FAMILIES:
        raise ValueError(
            f"Desteklenmeyen equalizer family: {family}"
        )

    if len(sampled_symbols) < equalizer_length:
        raise ValueError(
            "Equalizer için yeterli sembol örneği yok."
        )

    row_count = (
        len(sampled_symbols)
        - equalizer_length
        + 1
    )

    windows = np.stack(
        [
            sampled_symbols[
                tap_index:
                tap_index + row_count
            ]
            for tap_index in range(
                equalizer_length
            )
        ],
        axis=1,
    )

    feature_parts = [windows]

    if family == "widely_linear":
        feature_parts.append(
            np.conj(windows)
        )

    feature_parts.append(
        np.ones(
            (row_count, 1),
            dtype=np.complex128,
        )
    )

    features = np.concatenate(
        feature_parts,
        axis=1,
    )

    return (
        features.astype(np.complex128),
        equalizer_length // 2,
    )


def fit_complex_ridge(
    features: ComplexArray,
    targets: ComplexArray,
    ridge: float,
) -> ComplexArray:
    if ridge < 0.0:
        raise ValueError("ridge negatif olamaz.")

    gram = (
        features.conj().T
        @ features
    )
    right_side = (
        features.conj().T
        @ targets
    )

    regularizer = (
        ridge
        * np.eye(
            gram.shape[0],
            dtype=np.complex128,
        )
    )

    try:
        weights = np.linalg.solve(
            gram + regularizer,
            right_side,
        )
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(
            gram + regularizer,
            right_side,
            rcond=None,
        )[0]

    return weights.astype(np.complex128)


def qpsk_hard_demodulate(
    symbols: ComplexArray,
) -> BitArray:
    bits = np.empty(
        (len(symbols), 2),
        dtype=np.int8,
    )

    bits[:, 0] = (
        symbols.real < 0.0
    ).astype(np.int8)
    bits[:, 1] = (
        symbols.imag < 0.0
    ).astype(np.int8)

    return bits


def calculate_ber(
    reference_bits: BitArray,
    estimated_bits: BitArray,
) -> tuple[float, int, int]:
    if reference_bits.shape != estimated_bits.shape:
        raise ValueError(
            "reference_bits ve estimated_bits boyutları eşleşmiyor."
        )

    errors = int(
        np.count_nonzero(
            reference_bits
            != estimated_bits
        )
    )
    total = int(reference_bits.size)

    return (
        float(errors / total),
        errors,
        total,
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


def calculate_evm_percent(
    reference: ComplexArray,
    measured: ComplexArray,
) -> float:
    return float(
        100.0
        * np.sqrt(
            calculate_nmse(
                reference,
                measured,
            )
        )
    )


def train_model(
    dataset: Dataset,
    matched_samples: ComplexArray,
    model: ReceiverModel,
) -> ModelOutput | None:
    sampled = sample_symbols(
        matched_samples=matched_samples,
        timing_start=model.timing_start,
        samples_per_symbol=dataset.samples_per_symbol,
        maximum_symbols=len(dataset.symbols),
    )

    if len(sampled) <= (
        model.equalizer_length
        + TRAIN_SYMBOLS
        + VALIDATION_SYMBOLS
    ):
        return None

    features, half_length = build_features(
        sampled_symbols=sampled,
        equalizer_length=model.equalizer_length,
        family=model.family,
    )

    reference_symbols = dataset.symbols[
        half_length:
        half_length + len(features)
    ]
    reference_bits = dataset.bits[
        half_length:
        half_length + len(features)
    ]

    if (
        len(reference_symbols) != len(features)
        or len(reference_bits) != len(features)
    ):
        return None

    weights = fit_complex_ridge(
        features=features[:TRAIN_SYMBOLS],
        targets=reference_symbols[:TRAIN_SYMBOLS],
        ridge=model.ridge,
    )

    predictions = (
        features @ weights
    ).astype(np.complex128)

    return ModelOutput(
        model=model,
        weights=weights,
        sampled_symbols=sampled,
        predictions=predictions,
        reference_symbols=reference_symbols,
        reference_bits=reference_bits,
        feature_half_length=half_length,
        feature_count=int(features.shape[1]),
    )


def validation_evaluation(
    output: ModelOutput,
) -> CandidateEvaluation:
    start = TRAIN_SYMBOLS
    stop = min(
        TRAIN_SYMBOLS + VALIDATION_SYMBOLS,
        len(output.predictions),
    )

    if stop <= start:
        raise RuntimeError(
            "Validation bölümü için yeterli örnek yok."
        )

    predictions = output.predictions[
        start:stop
    ]
    reference = output.reference_symbols[
        start:stop
    ]
    reference_bits = output.reference_bits[
        start:stop
    ]

    estimated_bits = qpsk_hard_demodulate(
        predictions
    )

    ber, errors, _ = calculate_ber(
        reference_bits,
        estimated_bits,
    )
    nmse = calculate_nmse(
        reference,
        predictions,
    )
    evm = calculate_evm_percent(
        reference,
        predictions,
    )

    # Önce BER, sonra hata enerjisi, en son model karmaşıklığı önemlidir.
    selection_score = (
        ber
        + nmse
        + 1e-8
        * output.feature_count
    )

    return CandidateEvaluation(
        timing_start=output.model.timing_start,
        family=output.model.family,
        equalizer_length=output.model.equalizer_length,
        ridge=output.model.ridge,
        feature_count=output.feature_count,
        validation_ber=ber,
        validation_bit_errors=errors,
        validation_nmse=nmse,
        validation_evm_percent=evm,
        selection_score=selection_score,
    )


def evaluate_model(
    dataset: Dataset,
    matched_samples: ComplexArray,
    model: ReceiverModel,
) -> tuple[
    CandidateEvaluation,
    ModelOutput,
] | None:
    output = train_model(
        dataset=dataset,
        matched_samples=matched_samples,
        model=model,
    )

    if output is None:
        return None

    return (
        validation_evaluation(output),
        output,
    )


def candidate_sort_key(
    evaluation: CandidateEvaluation,
) -> tuple[
    float,
    float,
    int,
    int,
    float,
    str,
]:
    return (
        evaluation.validation_ber,
        evaluation.validation_nmse,
        evaluation.feature_count,
        evaluation.equalizer_length,
        evaluation.ridge,
        evaluation.family,
    )


def search_timing(
    dataset: Dataset,
    matched_samples: ComplexArray,
) -> tuple[int, list[CandidateEvaluation]]:
    timing_records: list[
        CandidateEvaluation
    ] = []

    coarse_starts = range(
        TIMING_SEARCH_START,
        TIMING_SEARCH_STOP + 1,
        COARSE_TIMING_STEP,
    )

    for timing_start in coarse_starts:
        result = evaluate_model(
            dataset=dataset,
            matched_samples=matched_samples,
            model=ReceiverModel(
                timing_start=timing_start,
                family="widely_linear",
                equalizer_length=11,
                ridge=1e-4,
            ),
        )

        if result is not None:
            timing_records.append(
                result[0]
            )

    if not timing_records:
        raise RuntimeError(
            "Coarse timing aramasında geçerli aday bulunamadı."
        )

    coarse_best = min(
        timing_records,
        key=candidate_sort_key,
    )

    refined_start = max(
        TIMING_SEARCH_START,
        coarse_best.timing_start
        - TIMING_REFINEMENT_RADIUS,
    )
    refined_stop = min(
        TIMING_SEARCH_STOP,
        coarse_best.timing_start
        + TIMING_REFINEMENT_RADIUS,
    )

    tested_starts = {
        item.timing_start
        for item in timing_records
    }

    for timing_start in range(
        refined_start,
        refined_stop + 1,
    ):
        if timing_start in tested_starts:
            continue

        result = evaluate_model(
            dataset=dataset,
            matched_samples=matched_samples,
            model=ReceiverModel(
                timing_start=timing_start,
                family="widely_linear",
                equalizer_length=11,
                ridge=1e-4,
            ),
        )

        if result is not None:
            timing_records.append(
                result[0]
            )

    best_timing = min(
        timing_records,
        key=candidate_sort_key,
    ).timing_start

    return best_timing, timing_records


def search_receiver_models(
    dataset: Dataset,
    matched_samples: ComplexArray,
    best_timing: int,
) -> tuple[
    CandidateEvaluation,
    ModelOutput,
    list[CandidateEvaluation],
]:
    evaluations: list[
        CandidateEvaluation
    ] = []
    output_map: dict[
        tuple[int, str, int, float],
        ModelOutput,
    ] = {}

    timing_candidates = range(
        max(
            TIMING_SEARCH_START,
            best_timing - FINAL_TIMING_RADIUS,
        ),
        min(
            TIMING_SEARCH_STOP,
            best_timing + FINAL_TIMING_RADIUS,
        ) + 1,
    )

    for timing_start in timing_candidates:
        for family in EQUALIZER_FAMILIES:
            for equalizer_length in EQUALIZER_LENGTHS:
                for ridge in RIDGE_VALUES:
                    model = ReceiverModel(
                        timing_start=timing_start,
                        family=family,
                        equalizer_length=equalizer_length,
                        ridge=ridge,
                    )

                    result = evaluate_model(
                        dataset=dataset,
                        matched_samples=matched_samples,
                        model=model,
                    )

                    if result is None:
                        continue

                    evaluation, output = result
                    evaluations.append(
                        evaluation
                    )
                    output_map[
                        (
                            timing_start,
                            family,
                            equalizer_length,
                            ridge,
                        )
                    ] = output

    if not evaluations:
        raise RuntimeError(
            "Receiver model aramasında geçerli aday bulunamadı."
        )

    best_evaluation = min(
        evaluations,
        key=candidate_sort_key,
    )

    best_key = (
        best_evaluation.timing_start,
        best_evaluation.family,
        best_evaluation.equalizer_length,
        best_evaluation.ridge,
    )

    return (
        best_evaluation,
        output_map[best_key],
        evaluations,
    )


def split_metrics(
    name: str,
    output: ModelOutput,
    start: int,
    stop: int,
) -> dict[str, Any]:
    actual_stop = min(
        stop,
        len(output.predictions),
    )

    if actual_stop <= start:
        raise RuntimeError(
            f"{name} bölümü için geçerli sembol yok."
        )

    predictions = output.predictions[
        start:actual_stop
    ]
    references = output.reference_symbols[
        start:actual_stop
    ]
    reference_bits = output.reference_bits[
        start:actual_stop
    ]

    estimated_bits = qpsk_hard_demodulate(
        predictions
    )

    ber, errors, total = calculate_ber(
        reference_bits,
        estimated_bits,
    )

    return {
        "name": name,
        "start_symbol": start,
        "stop_symbol_exclusive": actual_stop,
        "symbol_count": int(
            actual_stop - start
        ),
        "ber": ber,
        "bit_errors": errors,
        "total_bits": total,
        "nmse": calculate_nmse(
            references,
            predictions,
        ),
        "evm_rms_percent": calculate_evm_percent(
            references,
            predictions,
        ),
    }


def raw_symbol_metrics(
    output: ModelOutput,
) -> dict[str, Any]:
    half = output.feature_half_length
    usable_count = min(
        len(output.predictions),
        len(output.sampled_symbols) - 2 * half,
    )

    raw = output.sampled_symbols[
        half:
        half + usable_count
    ]
    reference = output.reference_symbols[
        :usable_count
    ]
    reference_bits = output.reference_bits[
        :usable_count
    ]

    estimated_bits = qpsk_hard_demodulate(
        raw
    )

    ber, errors, total = calculate_ber(
        reference_bits,
        estimated_bits,
    )

    return {
        "symbol_count": usable_count,
        "ber": ber,
        "bit_errors": errors,
        "total_bits": total,
        "nmse": calculate_nmse(
            reference,
            raw,
        ),
        "evm_rms_percent": calculate_evm_percent(
            reference,
            raw,
        ),
    }


def save_search_results(
    timing_records: list[CandidateEvaluation],
    model_records: list[CandidateEvaluation],
) -> None:
    with SEARCH_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "search_stage",
            "timing_start",
            "family",
            "equalizer_length",
            "ridge",
            "feature_count",
            "validation_ber",
            "validation_bit_errors",
            "validation_nmse",
            "validation_evm_percent",
            "selection_score",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in timing_records:
            writer.writerow(
                {
                    "search_stage": "timing",
                    **asdict(record),
                }
            )

        for record in model_records:
            writer.writerow(
                {
                    "search_stage": "model",
                    **asdict(record),
                }
            )


def load_baseline_comparison() -> dict[str, Any] | None:
    if not BASELINE_REPORT_PATH.exists():
        return None

    baseline = load_json(
        BASELINE_REPORT_PATH
    )

    try:
        test_metrics = baseline[
            "metrics"
        ]["test"]

        return {
            "available": True,
            "test_ber": float(
                test_metrics["ber"]
            ),
            "test_evm_rms_percent": float(
                test_metrics[
                    "evm_rms_percent"
                ]
            ),
            "selected_timing_start": int(
                baseline[
                    "receiver"
                ][
                    "selected_timing_start"
                ]
            ),
            "equalizer_length": int(
                baseline[
                    "receiver"
                ][
                    "equalizer_length"
                ]
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return {
            "available": False,
            "reason": (
                "Adım 08 raporu beklenen yapıda değil."
            ),
        }


def create_plot(
    cfo_frequencies_hz: FloatArray,
    cfo_spectrum_db: FloatArray,
    estimated_cfo_hz: float,
    timing_records: list[CandidateEvaluation],
    output: ModelOutput,
    raw_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(12, 18),
    )

    display_mask = (
        np.abs(cfo_frequencies_hz)
        <= 100_000.0
    )

    axes[0].plot(
        cfo_frequencies_hz[
            display_mask
        ] / 1_000.0,
        cfo_spectrum_db[
            display_mask
        ],
    )
    axes[0].axvline(
        4.0
        * estimated_cfo_hz
        / 1_000.0,
        linestyle="--",
        label="Tahmin edilen 4×CFO",
    )
    axes[0].set_title(
        "Dördüncü-Kuvvet CFO Spektrumu"
    )
    axes[0].set_xlabel(
        "Frekans (kHz)"
    )
    axes[0].set_ylabel(
        "Normalize güç (dB)"
    )
    axes[0].set_ylim(
        -100.0,
        5.0,
    )
    axes[0].grid(True)
    axes[0].legend()

    ordered_timing = sorted(
        timing_records,
        key=lambda item: item.timing_start,
    )

    axes[1].plot(
        [
            item.timing_start
            for item in ordered_timing
        ],
        [
            item.validation_nmse
            for item in ordered_timing
        ],
        marker="o",
        markersize=3,
    )
    axes[1].axvline(
        output.model.timing_start,
        linestyle="--",
        label="Seçilen timing",
    )
    axes[1].set_title(
        "Otomatik Timing Araması"
    )
    axes[1].set_xlabel(
        "İlk örnek indeksi"
    )
    axes[1].set_ylabel(
        "Validation NMSE"
    )
    axes[1].grid(True)
    axes[1].legend()

    half = output.feature_half_length
    plot_count = min(
        2200,
        len(output.predictions),
        len(output.sampled_symbols) - half,
    )

    raw_symbols = output.sampled_symbols[
        half:
        half + plot_count
    ]

    axes[2].scatter(
        raw_symbols.real,
        raw_symbols.imag,
        s=6,
        alpha=0.25,
        label="Equalizer öncesi",
    )
    axes[2].scatter(
        output.predictions[
            :plot_count
        ].real,
        output.predictions[
            :plot_count
        ].imag,
        s=7,
        alpha=0.35,
        label="Birleşik kurtarma sonrası",
    )
    axes[2].set_title(
        "Karma Bozunum Kurtarma Konstelasyonu"
    )
    axes[2].set_xlabel("I")
    axes[2].set_ylabel("Q")
    axes[2].set_aspect(
        "equal",
        adjustable="box",
    )
    axes[2].grid(True)
    axes[2].legend()

    metric_names = [
        "Raw BER",
        "Recovered BER",
        "Raw EVM / 100",
        "Recovered EVM / 100",
    ]
    metric_values = [
        raw_metrics["ber"],
        test_metrics["ber"],
        raw_metrics[
            "evm_rms_percent"
        ] / 100.0,
        test_metrics[
            "evm_rms_percent"
        ] / 100.0,
    ]

    axes[3].bar(
        metric_names,
        metric_values,
    )
    axes[3].set_title(
        "Kurtarma Öncesi ve Sonrası"
    )
    axes[3].set_ylabel(
        "Normalize metrik"
    )
    axes[3].tick_params(
        axis="x",
        rotation=20,
    )
    axes[3].grid(
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
    STEP18_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset()

    dc_corrected_samples, estimated_dc = (
        estimate_and_remove_dc(
            dataset.received_samples
        )
    )

    (
        estimated_cfo_hz,
        cfo_frequencies_hz,
        cfo_spectrum_db,
    ) = estimate_qpsk_cfo_fourth_power(
        samples=dc_corrected_samples,
        sample_rate_hz=dataset.sample_rate_hz,
    )

    cfo_corrected_samples = correct_cfo(
        samples=dc_corrected_samples,
        sample_rate_hz=dataset.sample_rate_hz,
        estimated_cfo_hz=estimated_cfo_hz,
    )

    matched_samples = matched_filter(
        samples=cfo_corrected_samples,
        rrc_taps=dataset.rrc_taps,
    )

    best_timing, timing_records = (
        search_timing(
            dataset=dataset,
            matched_samples=matched_samples,
        )
    )

    (
        best_evaluation,
        best_output,
        model_records,
    ) = search_receiver_models(
        dataset=dataset,
        matched_samples=matched_samples,
        best_timing=best_timing,
    )

    train_metrics = split_metrics(
        name="train",
        output=best_output,
        start=0,
        stop=TRAIN_SYMBOLS,
    )

    validation_metrics = split_metrics(
        name="validation",
        output=best_output,
        start=TRAIN_SYMBOLS,
        stop=(
            TRAIN_SYMBOLS
            + VALIDATION_SYMBOLS
        ),
    )

    test_metrics = split_metrics(
        name="test",
        output=best_output,
        start=(
            TRAIN_SYMBOLS
            + VALIDATION_SYMBOLS
        ),
        stop=len(
            best_output.predictions
        ),
    )

    raw_metrics = raw_symbol_metrics(
        best_output
    )

    repeated_output = train_model(
        dataset=dataset,
        matched_samples=matched_samples,
        model=best_output.model,
    )

    if repeated_output is None:
        raise RuntimeError(
            "Reproducibility çalıştırması model üretemedi."
        )

    reproducibility_error = float(
        np.max(
            np.abs(
                repeated_output.predictions
                - best_output.predictions
            )
        )
    )

    cfo_error_hz = (
        estimated_cfo_hz
        - dataset.true_cfo_hz
    )

    if abs(cfo_error_hz) > 50.0:
        raise RuntimeError(
            "CFO tahmin hatası 50 Hz sınırını aştı."
        )

    if validation_metrics["ber"] > 0.02:
        raise RuntimeError(
            "Validation BER başarı eşiğinin üzerinde."
        )

    if test_metrics["ber"] > 0.02:
        raise RuntimeError(
            "Test BER başarı eşiğinin üzerinde."
        )

    if (
        test_metrics[
            "evm_rms_percent"
        ]
        > 30.0
    ):
        raise RuntimeError(
            "Test EVM başarı eşiğinin üzerinde."
        )

    if raw_metrics["ber"] - test_metrics["ber"] < 0.20:
        raise RuntimeError(
            "Birleşik kurtarma BER iyileştirmesi yetersiz."
        )

    if reproducibility_error > 1e-12:
        raise RuntimeError(
            "Birleşik alıcı tekrarlanabilirlik testini geçemedi."
        )

    save_search_results(
        timing_records=timing_records,
        model_records=model_records,
    )

    np.savez_compressed(
        RECORD_PATH,
        received_samples=dataset.received_samples,
        dc_corrected_samples=dc_corrected_samples,
        cfo_corrected_samples=cfo_corrected_samples,
        matched_filter_samples=matched_samples,
        sampled_symbols=best_output.sampled_symbols,
        equalized_symbols=best_output.predictions,
        reference_symbols=best_output.reference_symbols,
        reference_bits=best_output.reference_bits,
        equalizer_weights=best_output.weights,
        rrc_taps=dataset.rrc_taps,
        estimated_dc=np.complex128(
            estimated_dc
        ),
        estimated_cfo_hz=np.float64(
            estimated_cfo_hz
        ),
        true_cfo_hz=np.float64(
            dataset.true_cfo_hz
        ),
        timing_start=np.int64(
            best_output.model.timing_start
        ),
        equalizer_length=np.int64(
            best_output.model.equalizer_length
        ),
        ridge=np.float64(
            best_output.model.ridge
        ),
        equalizer_family=np.asarray(
            best_output.model.family
        ),
        train_symbols=np.int64(
            TRAIN_SYMBOLS
        ),
        validation_symbols=np.int64(
            VALIDATION_SYMBOLS
        ),
    )

    baseline_comparison = (
        load_baseline_comparison()
    )

    recipe = {
        "schema_name": (
            "GENESIS-DSP CombinedRecoveryRecipe"
        ),
        "schema_version": "1.0.0",
        "input": {
            "sample_rate_hz": (
                dataset.sample_rate_hz
            ),
            "symbol_rate_hz": (
                dataset.symbol_rate_hz
            ),
            "samples_per_symbol": (
                dataset.samples_per_symbol
            ),
        },
        "ordered_steps": [
            {
                "operation": "dc_removal",
                "method": "complex_sample_mean",
                "estimated_dc_real": float(
                    estimated_dc.real
                ),
                "estimated_dc_imag": float(
                    estimated_dc.imag
                ),
            },
            {
                "operation": "cfo_estimation",
                "method": (
                    "qpsk_fourth_power_fft"
                ),
                "estimated_cfo_hz": (
                    estimated_cfo_hz
                ),
            },
            {
                "operation": "cfo_correction",
                "frequency_hz": (
                    -estimated_cfo_hz
                ),
            },
            {
                "operation": "matched_filter",
                "filter": "root_raised_cosine",
                "number_of_taps": int(
                    len(dataset.rrc_taps)
                ),
            },
            {
                "operation": "symbol_sampling",
                "timing_start": (
                    best_output.model.timing_start
                ),
                "samples_per_symbol": (
                    dataset.samples_per_symbol
                ),
            },
            {
                "operation": "adaptive_equalization",
                "family": (
                    best_output.model.family
                ),
                "equalizer_length": (
                    best_output.model.equalizer_length
                ),
                "ridge": (
                    best_output.model.ridge
                ),
                "feature_count": (
                    best_output.feature_count
                ),
                "training_symbols": (
                    TRAIN_SYMBOLS
                ),
            },
            {
                "operation": "qpsk_hard_decision",
            },
        ],
    }

    with RECIPE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            recipe,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    report = {
        "project": "GENESIS-DSP",
        "step": 18,
        "description": (
            "End-to-end mixed impairment recovery with automatic "
            "timing and equalizer model selection"
        ),
        "front_end": {
            "estimated_dc_real": float(
                estimated_dc.real
            ),
            "estimated_dc_imag": float(
                estimated_dc.imag
            ),
            "true_cfo_hz": (
                dataset.true_cfo_hz
            ),
            "estimated_cfo_hz": (
                estimated_cfo_hz
            ),
            "cfo_error_hz": cfo_error_hz,
        },
        "selected_model": {
            **asdict(
                best_output.model
            ),
            "feature_count": (
                best_output.feature_count
            ),
            "validation_selection": (
                asdict(
                    best_evaluation
                )
            ),
        },
        "metrics": {
            "raw": raw_metrics,
            "train": train_metrics,
            "validation": (
                validation_metrics
            ),
            "test": test_metrics,
        },
        "search": {
            "timing_candidates_evaluated": (
                len(timing_records)
            ),
            "receiver_models_evaluated": (
                len(model_records)
            ),
        },
        "baseline_comparison": (
            baseline_comparison
        ),
        "reproducibility": {
            "maximum_prediction_error": (
                reproducibility_error
            ),
            "status": "PASSED",
        },
        "validations": {
            "cfo_error_below_50_hz": True,
            "validation_ber_below_0_02": True,
            "test_ber_below_0_02": True,
            "test_evm_below_30_percent": True,
            "raw_to_recovered_ber_improvement_above_0_20": True,
            "deterministic_reexecution": True,
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
        cfo_frequencies_hz=cfo_frequencies_hz,
        cfo_spectrum_db=cfo_spectrum_db,
        estimated_cfo_hz=estimated_cfo_hz,
        timing_records=timing_records,
        output=best_output,
        raw_metrics=raw_metrics,
        test_metrics=test_metrics,
    )

    print()
    print("=" * 84)
    print(
        "GENESIS-DSP — ADIM 18 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 84)
    print(
        f"Gerçek CFO                     : "
        f"{dataset.true_cfo_hz:,.6f} Hz"
    )
    print(
        f"Tahmini CFO                    : "
        f"{estimated_cfo_hz:,.6f} Hz"
    )
    print(
        f"CFO tahmin hatası              : "
        f"{cfo_error_hz:+.6f} Hz"
    )
    print(
        f"Seçilen timing                 : "
        f"{best_output.model.timing_start}"
    )
    print(
        f"Equalizer ailesi               : "
        f"{best_output.model.family}"
    )
    print(
        f"Equalizer uzunluğu             : "
        f"{best_output.model.equalizer_length}"
    )
    print(
        f"Ridge                          : "
        f"{best_output.model.ridge:.3e}"
    )
    print(
        f"Kurtarma öncesi BER            : "
        f"{raw_metrics['ber']:.9f}"
    )
    print(
        f"Validation BER                 : "
        f"{validation_metrics['ber']:.9f}"
    )
    print(
        f"Test BER                       : "
        f"{test_metrics['ber']:.9f}"
    )
    print(
        f"Test bit hatası                : "
        f"{test_metrics['bit_errors']} / "
        f"{test_metrics['total_bits']}"
    )
    print(
        f"Test RMS EVM                   : "
        f"{test_metrics['evm_rms_percent']:.6f} %"
    )
    print(
        f"Değerlendirilen receiver model : "
        f"{len(model_records)}"
    )
    print(
        f"Reproducibility max hata       : "
        f"{reproducibility_error:.3e}"
    )
    print(
        f"Kurtarma reçetesi              : "
        f"{RECIPE_PATH}"
    )
    print(
        f"Model arama tablosu            : "
        f"{SEARCH_PATH}"
    )
    print(
        f"Grafik                         : "
        f"{PLOT_PATH}"
    )
    print(
        f"Rapor                          : "
        f"{REPORT_PATH}"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
